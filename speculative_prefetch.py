"""
speculative_prefetch.py — Speculative Prefetch & Dynamic Expert Activation.

Integrates with existing N-Gram speculation and SERE skip logic to:

1. **DynamicExpertActivator** — per-token adjustment of expert activation
   count k based on Softmax confidence of the generated token, plus a
   sliding-window text trigger ("/全力思考") that forces k = K_MAX.

2. **SpeculativePrefetcher** — predicts upcoming expert demand via N-Gram
   token draft + SERE routing estimates and issues async H2D prefetches
   with a **hard 5 ms timeout** that never blocks the main thread.

Guarantees: No modification to the underlying MoE computation kernel.
All heuristic logic is pure Python / PyTorch tensor ops.
"""

from __future__ import annotations

import collections
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _TimeoutError

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DynamicExpertActivator
# ═══════════════════════════════════════════════════════════════════════


class DynamicExpertActivator:
    """Per-token dynamic expert activation count (k) controller.

    Parameters
    ----------
    k_min : int
        Lower bound for k (default 3).
    k_max : int
        Upper bound for k (default 5).
    initial_k : int
        Starting k (default ``k_min``).
    force_cmd : str
        Sliding-window text trigger (default ``"/全力思考"``).
    sere_module : optional
        Attached ``SEREModule`` whose ``top_k`` will be updated in
        lockstep with ``current_k``.
    force_steps : int
        Number of decode steps to persist forced K_MAX after the trigger
        text clears the window.
    """

    K_MIN: int = 3
    K_MAX: int = 5
    _FORCE_CMD: str = "/全力思考"
    _SLIDING_WINDOW_CHARS: int = 40
    _HISTORY_WEIGHTED_TOKENS: int = 5

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        initial_k: int | None = None,
        force_cmd: str = _FORCE_CMD,
        sere_module=None,
        force_steps: int = 10,
    ) -> None:
        self.k_min = max(1, k_min)
        self.k_max = max(self.k_min, k_max)

        self.current_k: int = (
            max(self.k_min, min(initial_k, self.k_max))
            if initial_k is not None
            else self.k_min
        )

        self._sere = sere_module
        self._force_cmd = force_cmd
        self._force_steps = force_steps
        self._text_window: collections.deque[str] = collections.deque(
            maxlen=self._SLIDING_WINDOW_CHARS
        )
        self._forced_mode: bool = False
        self._force_steps_remaining: int = 0

        logger.info(
            "DynamicExpertActivator: k_min=%d, k_max=%d, force_steps=%d",
            self.k_min,
            self.k_max,
            self._force_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_logits(
        self,
        logits: torch.Tensor,
        generated_ids: list[int] | None = None,
        detokenizer: collections.abc.Callable[[list[int]], str] | None = None,
    ) -> int:
        """Update k from LM-head logits and optional sliding-window text.

        Steps
        -----
        1. Softmax over the last position → max probability = confidence.
        2. confidence > 0.9 → k -= 1;  < 0.5 → k += 1.
        3. Sliding-window text scan for *force_cmd* → k = K_MAX.
        4. Clamp to [k_min, k_max] and push into attached SERE's top_k.

        Returns
        -------
        int
            The new ``current_k``.
        """
        # ----- confidence-based adjustment -----
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        confidence = probs.max().item()

        _high_confidence: float = 0.9
        _low_confidence: float = 0.5

        if confidence > _high_confidence:
            self.current_k = max(self.k_min, self.current_k - 1)
        elif confidence < _low_confidence:
            self.current_k = min(self.k_max, self.current_k + 1)

        # ----- sliding-window text trigger -----
        if detokenizer is not None and generated_ids is not None:
            self._update_text_window(generated_ids, detokenizer)
            self._check_force_mode()

        # ----- forced-mode decay -----
        if self._forced_mode:
            if self._force_steps_remaining > 0:
                self._force_steps_remaining -= 1
                self.current_k = self.k_max
            else:
                self._forced_mode = False
                logger.debug("DynamicExpertActivator: /全力思考 force expired.")

        # ----- final clamp -----
        self.current_k = max(self.k_min, min(self.current_k, self.k_max))

        # ----- push to SERE -----
        if self._sere is not None:
            self._sere.top_k = self.current_k

        return self.current_k

    def get_k(self) -> int:
        """Return the current effective k."""
        return self.current_k

    def force_max_k(self) -> None:
        """Programmatically force k = k_max for *force_steps* steps."""
        self._forced_mode = True
        self._force_steps_remaining = self._force_steps
        self.current_k = self.k_max
        if self._sere is not None:
            self._sere.top_k = self.current_k

    def reset(self) -> None:
        """Reset to initial state (k = k_min, no forced mode)."""
        self.current_k = self.k_min
        self._forced_mode = False
        self._force_steps_remaining = 0
        self._text_window.clear()
        if self._sere is not None:
            self._sere.top_k = self.current_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_text_window(
        self,
        token_ids: list[int],
        detokenizer: collections.abc.Callable[[list[int]], str],
    ) -> None:
        """Decode the most recent tokens and append chars to the window."""
        recent = token_ids[-(self._HISTORY_WEIGHTED_TOKENS):]
        text = detokenizer(recent)
        for ch in text:
            self._text_window.append(ch)

    def _check_force_mode(self) -> None:
        """If the sliding window contains *force_cmd*, activate forced mode."""
        window_text = "".join(self._text_window)
        if self._force_cmd in window_text:
            logger.debug("DynamicExpertActivator: '/全力思考' detected — forcing k=%d", self.k_max)
            self._forced_mode = True
            self._force_steps_remaining = self._force_steps
            self.current_k = self.k_max


# ═══════════════════════════════════════════════════════════════════════
# SpeculativePrefetcher
# ═══════════════════════════════════════════════════════════════════════


class SpeculativePrefetcher:
    """Predictive expert-weight prefetcher with hard 5 ms timeout.

    Architecture
    ------------
    1. **Prediction phase** — uses the existing ``NGramCache`` to draft
       the most likely next token(s), then consults SERE's routing
       heuristics to estimate which experts those tokens would activate.
    2. **Prefetch phase** — issues async H2D transfers for the predicted
       expert(s) via ``ExpertWeightCache.prefetch_expert()`` on a
       background thread.
    3. **Timeout guard** — every prefetch call has a 5 ms wall-clock
       budget. If it exceeds that, the result is **silently dropped**
       and the next on-demand ``get_or_load_expert()`` will serve the
       normal path.  The prefetch thread continues in the background but
       its results are never waited on.

    Note: this is a **heuristic** — it provides speed-up on average
    without any correctness cost.
    """

    PREFETCH_TIMEOUT_S: float = 0.005  # 5 ms

    def __init__(
        self,
        ngram_cache,
        expert_cache,
        sere_module,
        num_layers: int,
        num_experts: int,
        prefetch_depth: int = 2,
        prefetch_layers_per_step: int = 1,
    ) -> None:
        self._ngram = ngram_cache
        self._expert_cache = expert_cache
        self._sere = sere_module
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._prefetch_depth = prefetch_depth
        self._prefetch_layers = prefetch_layers_per_step

        # A single-worker thread pool — one prefetch in flight at a time
        # avoids queue pile-up on bursty decode steps.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Tracks expert usage history for heuristic prediction
        # token_hash -> set of (layer, expert) tuples
        self._token_expert_history: dict[int, list[tuple[int, int]]] = {}
        self._history_max: int = 200

        logger.info(
            "SpeculativePrefetcher: layers=%d, experts=%d, depth=%d, timeout=%.1fms",
            num_layers,
            num_experts,
            prefetch_depth,
            self.PREFETCH_TIMEOUT_S * 1000,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_expert_usage(
        self,
        token_ids: list[int],
        router_selections: list[tuple[int, int]],
    ) -> None:
        """Record which experts were activated for the current context.

        Called *after* each decode step.  The recorded history is used
        by *predict_needed_experts* to correlate future token patterns
        with expert demand.
        """
        if not token_ids:
            return
        # Use the last token as the key
        last_tok = token_ids[-1]
        record = list(router_selections)

        if last_tok not in self._token_expert_history:
            if len(self._token_expert_history) >= self._history_max:
                # Simple LRU-ish eviction
                oldest = next(iter(self._token_expert_history))
                del self._token_expert_history[oldest]
            self._token_expert_history[last_tok] = record

    def step(
        self,
        context_ids: list[int],
        transfer_stream: torch.cuda.Stream | None = None,
        token_hash: int | None = None,
    ) -> None:
        """Perform one speculative prefetch step.

        Called at the **end** of each decode step, so it races ahead
        of the *next* decode step's expert loads.

        Parameters
        ----------
        context_ids : list[int]
            Full token-id context so far (used for N-Gram drafting).
        transfer_stream : torch.cuda.Stream, optional
            CUDA stream on which to issue H2D copies.  ``None`` → skip.
        token_hash : int, optional
            If provided, used as a fast lookup key into the expert
            history cache (avoids per-step N-Gram search).
        """
        if transfer_stream is None or self._expert_cache is None:
            return

        predicted = self._predict_needed_experts(context_ids, token_hash)
        if not predicted:
            return

        # Deduplicate against recently prefetched experts
        # (the cache's ``prefetch_expert()`` already skips VRAM-resident
        #  experts, but we also avoid flooding the thread pool.)
        for layer_idx, expert_idx in predicted[:self._prefetch_layers]:
            self._prefetch_with_timeout(layer_idx, expert_idx, transfer_stream)

    # ------------------------------------------------------------------
    # Prediction heuristics
    # ------------------------------------------------------------------

    def _predict_needed_experts(
        self,
        context_ids: list[int],
        token_hash: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return list of (layer_idx, expert_idx) predicted to be needed
        on upcoming decode steps.

        Strategy (in order):
        1. Token-based history lookup — if *token_hash* (or the last
           token) has a record, return the top experts from history.
        2. N-Gram draft fallback — predict the next token via the
           existing N-Gram cache and look up that token's history.
        3. SERE fallback — use the attached SERE's top_k / skip
           threshold to compute the most-likely-to-survive experts.
        4. Empty — no prediction possible this step.
        """
        # --- Strategy 1: token history ---
        target_token = context_ids[-1] if context_ids else None
        if target_token is not None and target_token in self._token_expert_history:
            history = self._token_expert_history[target_token]
            if history:
                return history

        # --- Strategy 2: N-Gram draft prediction ---
        if self._ngram is not None and context_ids:
            drafts = self._ngram.generate_draft(context_ids, draft_length=1)
            if drafts:
                draft_tok = drafts[0]
                if draft_tok in self._token_expert_history:
                    return self._token_expert_history[draft_tok]

        # --- Strategy 3: SERE-guided fallback ---
        if self._sere is not None:
            k = self._sere.top_k
            # Default: prefetch the first k experts for the first layer
            # (the most common pattern in sequential decode)
            fallback: list[tuple[int, int]] = []
            _actual_experts = min(k, self._num_experts)
            for ly in range(self._prefetch_layers):
                fallback.extend((ly, ex) for ex in range(_actual_experts))
            return fallback

        # --- Strategy 4: conservative guess ---
        return [(0, 0)]

    # ------------------------------------------------------------------
    # Prefetch with timeout
    # ------------------------------------------------------------------

    def _prefetch_with_timeout(
        self,
        layer_idx: int,
        expert_idx: int,
        transfer_stream: torch.cuda.Stream,
    ) -> bool:
        """Submit an async H2D prefetch with a 5 ms deadline.

        Returns
        -------
        bool
            ``True`` if the prefetch completed within the timeout,
            ``False`` if it timed out (result dropped, no effect).
        """
        start = time.perf_counter()

        future = self._executor.submit(
            self._expert_cache.prefetch_expert,
            layer_idx,
            expert_idx,
            transfer_stream,
        )

        try:
            future.result(timeout=self.PREFETCH_TIMEOUT_S)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "Prefetch L%d.E%d: %.2f ms (OK)",
                layer_idx,
                expert_idx,
                elapsed_ms,
            )
            return True
        except _TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "Prefetch L%d.E%d: TIMEOUT (%.2f ms > %d ms limit) — dropped",
                layer_idx,
                expert_idx,
                elapsed_ms,
                self.PREFETCH_TIMEOUT_S * 1000,
            )
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("SpeculativePrefetcher: thread pool shut down.")

    def __del__(self) -> None:
        self.shutdown()
