"""
UnifiedScheduler — Three-Layer End-to-End MoE Inference Pipeline.

Fixes
-----
- KV cache is now **actually stored** in HybridCache blocks after prefill
  and **retrieved** on each decode step, giving O(n) attention per token.
- Batch prefill stores per-request KV into individually allocated blocks.
- Decode passes ``past_key_values`` from cache, avoids full-sequence recompute.
- Finished requests free their cache blocks for recycling.
"""

from __future__ import annotations

import collections.abc
import logging
import time
from dataclasses import dataclass, field

import torch

import attention_kernel  # noqa: F401
from cache_manager import HybridCache

# [Goose] Lazy import
_GOOSE_AVAILABLE = False
try:
    import goose_core
    _GOOSE_AVAILABLE = True
except ImportError:
    pass

# [AFCE] Anchored Forward Cache Extension
_AFCE_AVAILABLE = False
try:
    import afce
    _AFCE_AVAILABLE = True
except ImportError:
    pass

# [OEF] Opportunistic Entropy Freeze
_OEF_AVAILABLE = False
try:
    import oef
    _OEF_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _extract_logits(model_output) -> torch.Tensor:
    """Extract logits tensor from both raw tensor and HF CausalLMOutput."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def _extract_past_key_values(model_output):
    """Extract past_key_values from HF output or GGUF-side-effect."""
    if hasattr(model_output, "past_key_values"):
        return model_output.past_key_values
    return None


@dataclass
class Request:
    prompt_tokens: list[int]
    request_id: str
    max_new_tokens: int = 256
    cached_heads: list[int] = field(default_factory=list)


@dataclass
class DecodeRequest:
    tokens: list[int]
    generated_tokens: list[int]
    request_id: str
    max_new_tokens: int
    _step_count: int = 0
    # Block ID in HybridCache that holds this request's KV cache
    cache_block_id: int | None = None

    def step(self) -> None:
        self._step_count += 1

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.max_new_tokens


# ===================================================================
# UnifiedScheduler — corrected KV cache integration
# ===================================================================


class UnifiedScheduler:
    CHUNK_SIZE = 512
    _PREFILL_BATCH_TIMEOUT = 0.5
    _PREFILL_BATCH_MAX = 8
    _EXPERT_CAPACITY_FACTOR = 1.2
    _KV_CACHE_ENABLED = True  # set False to fall back to full recompute

    def __init__(
        self,
        model: object,
        cache: HybridCache,
        detokenizer: collections.abc.Callable[[list[int]], str] | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self._detokenizer = detokenizer

        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.transfer_stream = torch.cuda.Stream()

        self.pending_requests: list[Request] = []
        self.active_decode_pool: list[DecodeRequest] = []
        self._decode_bs = 0
        self._last_prefill_time = time.monotonic()
        self._running = True

        self._sere = None
        self._expert_cache = None
        self._batch_plan = None

        # [Context truncation / position encoding reset]
        self.step_since_reset = 0
        self.trigger_reset = False
        self.max_len = getattr(model, 'max_seq_len', None) or getattr(model, 'config', None) and getattr(model.config, 'max_position_embeddings', 4096) or 4096
        self._compression_threshold = int(self.max_len * 0.95)

        # [Speculative Prefetch & Dynamic Expert Activation]
        self._dynamic_activator = None
        self._spec_prefetcher = None

        # [Goose] Speculative decoding engine (Phase 0/1/2)
        self._goose_enabled: bool = False
        self._goose_engine: object | None = None
        self._spec_handled: set[str] = set()  # request_ids handled this step

        # [AFCE] Anchored Forward Cache Extension
        self._afce_manager: afce.AnchorManager | None = None
        if _AFCE_AVAILABLE:
            self._afce_manager = afce.AnchorManager()
            logger.info("AFCE: AnchorManager initialized")

        # [OEF] Opportunistic Entropy Freeze
        self._oef_controller: oef.OEFController | None = None
        self._init_oef_lazy: bool = False

        logger.info(
            "UnifiedScheduler: chunk=%d, batch_max=%d, cap_factor=%.2f, kv_cache=%s",
            self.CHUNK_SIZE,
            self._PREFILL_BATCH_MAX,
            self._EXPERT_CAPACITY_FACTOR,
            self._KV_CACHE_ENABLED,
        )

    # ==================================================================
    # Layer init (lazy)
    # ==================================================================

    def _init_speculative_prefetch(self):
        from speculative_prefetch import (  # noqa: PLC0415
            DynamicExpertActivator,
            SpeculativePrefetcher,
        )

        if self._dynamic_activator is None:
            self._dynamic_activator = DynamicExpertActivator(
                sere_module=self._sere,
            )

        layers = self._get_decoder_layers()
        num_layers = getattr(self.model, "num_layers", 0)
        num_experts = getattr(self.model, "num_experts", 8)

        if self._spec_prefetcher is None and self._expert_cache is not None and layers is not None:
            self._spec_prefetcher = SpeculativePrefetcher(
                expert_cache=self._expert_cache,
                decoder_layers=list(layers),
                num_layers=num_layers,
                num_experts=num_experts,
                num_reserved_experts=64,
            )

    def _get_decoder_layers(self):
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return None

    def _init_sere(self):
        from sere import SEREModule  # noqa: PLC0415

        num_experts = getattr(self.model, "num_experts", 8)
        top_k = getattr(self.model, "num_experts_per_tok", 2)
        self._sere = SEREModule(
            num_experts=num_experts, top_k=top_k, skip_threshold=0.15, min_experts=1
        )

        layers = self._get_decoder_layers()
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "sere") and layer.sere is not None:
                    layer.sere = self._sere

    def _init_expert_cache(self, vram_capacity=64):
        from expert_cache import ExpertWeightCache  # noqa: PLC0415

        model = self.model
        if not getattr(model, "is_moe", False):
            return

        layers = self._get_decoder_layers()
        if layers is None:
            return

        hidden_size = model.hidden_size
        num_layers = model.num_layers
        num_experts = getattr(model, "num_experts", 8)

        intermediate_size = hidden_size * 8 // 3
        for layer in layers:
            if hasattr(layer, "intermediate_size"):
                intermediate_size = layer.intermediate_size
                break

        block_bytes = (intermediate_size * hidden_size * 2 * 3) * 2

        self._expert_cache = ExpertWeightCache(
            vram_capacity=vram_capacity,
            block_bytes=block_bytes,
            num_layers=num_layers,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            dtype=torch.float16,
        )

        for ly in range(num_layers):
            layer = layers[ly]
            if not hasattr(layer, "get_cpu_expert_weights"):
                continue
            for ex in range(num_experts):
                try:
                    gate_w, up_w, down_w = layer.get_cpu_expert_weights(ex)
                    self._expert_cache.register_expert(ly, ex, gate_w, up_w, down_w)
                except Exception as exc:
                    logger.warning("Expert L%d.E%d skip: %s", ly, ex, exc)

        for layer in layers:
            if hasattr(layer, "expert_cache"):
                layer.expert_cache = self._expert_cache
            # Seed _prefetch_stream (set by SpeculativePrefetcher if created)
            if hasattr(layer, '_prefetch_stream') and layer._prefetch_stream is None:
                layer._prefetch_stream = self.transfer_stream

        logger.info(
            "Expert cache: %d x %d experts, VRAM=%d blocks",
            num_layers,
            num_experts,
            vram_capacity,
        )

    # ==================================================================
    # Batch planning
    # ==================================================================

    def _plan_batch(self, batch_requests: list[Request]) -> dict:
        if not self._expert_cache:
            return {"needed_experts": set(), "capacity_check": True}

        total_tokens = sum(len(r.prompt_tokens) for r in batch_requests)
        capacity_limit = int(
            total_tokens * self._EXPERT_CAPACITY_FACTOR / max(self.model.num_experts, 1)
        )
        min_capacity = 4
        capacity_ok = capacity_limit >= min_capacity

        return {"needed_experts": set(), "capacity_check": capacity_ok}

    # ==================================================================
    # Goose speculative decoding init
    # ==================================================================

    def _init_goose(self, tree_enabled: bool = False, max_draft: int = 5) -> None:
        """Initialize the Goose speculative decoding engine.

        Safe to call multiple times — skips if already initialized.
        Falls back gracefully when ``goose_core`` module is unavailable.
        """
        if self._goose_engine is not None:
            return
        if not _GOOSE_AVAILABLE:
            logger.warning("goose_core module not available; speculation disabled.")
            return

        vocab_size = getattr(self.model, "vocab_size", 32000)
        self._goose_engine = goose_core.SpeculativeEngine(
            vocab_size=vocab_size,
            max_draft=max_draft,
            tree_enabled=tree_enabled,
        )
        self._goose_enabled = True
        self._goose_engine.enable()  # skip warm-up for immediate testing
        logger.info(
            "Goose engine initialized: tree=%s, max_draft=%d, vocab=%d",
            tree_enabled, max_draft, vocab_size,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def submit(self, request: Request) -> None:
        self.pending_requests.append(request)

    def shutdown(self) -> None:
        self._running = False
        if self._spec_prefetcher is not None:
            self._spec_prefetcher.shutdown()

    # ==================================================================
    # Prefill with KV cache storage
    # ==================================================================

    def _batch_prefill(self):
        if not self.pending_requests:
            return

        now = time.monotonic()
        elapsed = now - self._last_prefill_time
        if (
            elapsed < self._PREFILL_BATCH_TIMEOUT
            and len(self.pending_requests) < self._PREFILL_BATCH_MAX
        ):
            return

        batch_requests = list(self.pending_requests)
        self.pending_requests.clear()
        self._last_prefill_time = now

        # Process each request individually so we can capture per-request KV
        for req in batch_requests:
            tokens = req.prompt_tokens
            inp = torch.tensor([tokens], dtype=torch.long, device="cuda")

            with torch.cuda.stream(self.prefill_stream), torch.no_grad():
                out = self.model.forward(
                    input_ids=inp,
                    use_cache=True,
                )

            # Retrieve KV cache (GGUF side-effect or HF output)
            kv_cache = getattr(self.model, "_last_kv_cache", None)
            if kv_cache is None:
                kv_cache = _extract_past_key_values(out)
            if kv_cache is not None:
                # Allocate a cache block for this request
                cache_block = self.cache.allocate(tokens)
                # Store KV into block (list of (k, v) per layer)
                self.cache.store_kv(cache_block.block_id, kv_cache)

                # [AFCE] Extract anchors from this prefill if applicable
                if self._afce_manager is not None and len(tokens) >= afce.CLUSTER_SIZE:
                    try:
                        from afce import extract_anchors_after_prefill
                        extract_anchors_after_prefill(
                            self._afce_manager,
                            self.cache,
                            tokens,
                            kv_cache,
                        )
                    except Exception as exc:
                        logger.debug("AFCE prefill extract skipped: %s", exc)

                self.active_decode_pool.append(
                    DecodeRequest(
                        tokens=req.prompt_tokens,
                        generated_tokens=[],
                        request_id=req.request_id,
                        max_new_tokens=req.max_new_tokens,
                        cache_block_id=cache_block.block_id,
                    )
                )
                logger.debug(
                    "Prefill %s: %d tokens -> block %d",
                    req.request_id,
                    len(tokens),
                    cache_block.block_id,
                )
            else:
                # No KV cache (model doesn't support use_cache) — fallback
                logger.warning("Model did not produce _last_kv_cache — KV caching disabled.")
                self.active_decode_pool.append(
                    DecodeRequest(
                        tokens=req.prompt_tokens,
                        generated_tokens=[],
                        request_id=req.request_id,
                        max_new_tokens=req.max_new_tokens,
                        cache_block_id=None,
                    )
                )

        logger.info("Prefill: %d requests processed", len(batch_requests))

    # ==================================================================
    # Decode with KV cache
    # ==================================================================

    def _decode_step(self):
        if not self.active_decode_pool:
            self._decode_bs = 0
            return

        # [Plan 4] Single boolean check: trigger_reset → context compression
        if self.trigger_reset:
            for req in self.active_decode_pool:
                self._compress_kv(req)
            self.step_since_reset = 0
            self.trigger_reset = False
            logger.info("Context compression triggered for %d requests", len(self.active_decode_pool))

        with torch.cuda.stream(self.decode_stream):
            for i, req in enumerate(self.active_decode_pool):
                # [Goose] Skip requests already handled by speculative decode
                if req.request_id in self._spec_handled:
                    continue

                last_tok = req.tokens[-1] if req.tokens else 0
                inp = torch.tensor([[last_tok]], dtype=torch.long, device="cuda")

                # Load cached KV if available
                past_kv = None
                if req.cache_block_id is not None:
                    past_kv = self.cache.load_kv(req.cache_block_id)

                # [AFCE] Extend KV with anchors + build attention mask
                _afce_hash_key: str | None = None
                _afce_num_anchors: int = 0
                _afce_mask: torch.Tensor | None = None
                _afce_past_kv: list | tuple | None = None

                if (
                    past_kv is not None
                    and self._KV_CACHE_ENABLED
                    and self._afce_manager is not None
                    and req.cache_block_id is not None
                ):
                    _afce_hash_key = self.cache.compute_hash(req.tokens)
                    if _afce_hash_key and self._afce_manager.has_sidecar(_afce_hash_key):
                        extended: list[tuple[torch.Tensor, torch.Tensor]] = []
                        _na = 0
                        for _k, _v in past_kv:
                            _ek, _ev, _m, _n = self._afce_manager.extend_for_decode(
                                _afce_hash_key, _k, _v, len(req.tokens) - 1,
                            )
                            extended.append((_ek, _ev))
                            _na = _n
                        if _na > 0:
                            _afce_num_anchors = _na
                            _afce_mask = _m
                            _afce_past_kv = extended

                # [OEF] Feed OEF skip suggestions into SERE before forward
                if self._oef_controller is not None and self._sere is not None:
                    suggestions = self._oef_controller.get_skip_suggestions()
                    self._sere._oef_skip_suggestions = suggestions if suggestions else None

                with torch.no_grad():
                    if past_kv is not None and self._KV_CACHE_ENABLED:
                        model_out = self.model.forward(
                            input_ids=inp,
                            past_key_values=_afce_past_kv if _afce_past_kv is not None else past_kv,
                            use_cache=True,
                            attention_mask=_afce_mask if _afce_num_anchors > 0 else None,
                        )
                    else:
                        # Full recompute path (fallback if no KV cache)
                        full_input = torch.tensor(
                            [req.tokens], dtype=torch.long, device="cuda"
                        )
                        model_out = self.model.forward(
                            input_ids=full_input, use_cache=False
                        )

                # Extract logits (works for both GGUF tensor and HF output)
                logits_t = _extract_logits(model_out)

                # Extract next token
                _min_logit_dims = 2
                if logits_t.dim() >= _min_logit_dims:
                    next_tok = int(logits_t[0, -1, :].argmax().item())
                else:
                    next_tok = 0

                req.step()
                req.generated_tokens.append(next_tok)
                req.tokens.append(next_tok)

                # Update KV cache after decode (both GGUF and HF paths)
                if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                    new_kv = getattr(self.model, "_last_kv_cache", None)
                    if new_kv is None:
                        new_kv = _extract_past_key_values(model_out)
                    if new_kv is not None:
                        # [AFCE] Strip anchor positions before storing back
                        if _afce_num_anchors > 0 and self._afce_manager is not None:
                            new_kv = self._afce_manager.strip_anchors_from_kv(
                                new_kv, _afce_num_anchors,
                            )
                        self.cache.store_kv(req.cache_block_id, new_kv)

                # [AFCE] Async prefetch next cluster anchors（红线 #2）
                if (
                    _afce_hash_key is not None
                    and _afce_num_anchors > 0
                    and self._afce_manager is not None
                ):
                    self._afce_manager.prefetch(
                        _afce_hash_key, len(req.tokens) - 1,
                        stream=self.transfer_stream,
                    )

                # [OEF] Observe router probs for entropy tracking
                if (
                    self._oef_controller is not None
                    and hasattr(self.model, "_last_router_probs")
                    and self.model._last_router_probs is not None
                ):
                    with torch.no_grad():
                        _rp = self.model._last_router_probs
                        _, _ti = torch.topk(_rp, _rp.shape[-1], dim=-1)
                        self._oef_controller.observe(_rp, _ti)

                # Confidence-based k adjustment on request 0
                if self._dynamic_activator is not None and i == 0:
                    self._dynamic_activator.update_from_logits(
                            logits_t[None, :, :],
                            generated_ids=req.tokens,
                            detokenizer=self._detokenizer,
                        )

            # Speculative prefetch
            if self._spec_prefetcher is not None and len(self.active_decode_pool) > 0:
                self._spec_prefetcher.step(
                    context_ids=self.active_decode_pool[0].tokens,
                    transfer_stream=self.transfer_stream,
                )

        # [Plan 2] Increment step counter; set trigger flag at threshold
        self.step_since_reset += 1
        if self.step_since_reset >= self._compression_threshold:
            self.trigger_reset = True

    # ==================================================================
    # Main step
    # ==================================================================

    # ==================================================================
    # Context compression (position encoding reset)
    # ==================================================================

    def _compress_kv(self, req: DecodeRequest) -> None:
        """Compress one request's KV cache: keep first 4 + last 2048 tokens.

        1. Truncate token list to tokens[:4] + tokens[-2048:]
        2. Slice KV tensors to match (keep first 4 + last 2048 positions)
        3. Free any intermediate KV blocks via existing pop interface
        4. [AFCE] Clear orphan sidecar anchors before compression
        """
        if req.cache_block_id is None or not req.tokens:
            return

        old_len = len(req.tokens)
        if old_len <= 2052:
            # Too short to need compression
            return

        # [AFCE] Clear orphan sidecar for old hash — token sequence will
        # change after compression, leaving stale anchors unreachable.
        if self._afce_manager is not None:
            old_hash = self.cache.compute_hash(req.tokens)
            if old_hash and self._afce_manager.has_sidecar(old_hash):
                self._afce_manager.remove_sidecar(old_hash)

        # Build new token sequence
        new_tokens = req.tokens[:4] + req.tokens[-2048:]
        req.tokens = new_tokens

        # Compress KV tensors in place
        kv = self.cache.load_kv(req.cache_block_id)
        if kv is not None:
            compressed_kv = []
            for k, v in kv:
                # k, v shape: [1, num_heads, seq_len, head_dim]
                first_k = k[:, :, :4, :]
                last_k = k[:, :, -2048:, :]
                first_v = v[:, :, :4, :]
                last_v = v[:, :, -2048:, :]
                compressed_k = torch.cat([first_k, last_k], dim=2)
                compressed_v = torch.cat([first_v, last_v], dim=2)
                compressed_kv.append((compressed_k, compressed_v))
            self.cache.store_kv(req.cache_block_id, compressed_kv)

        logger.info(
            "KV compressed for %s: %d -> %d tokens",
            req.request_id, old_len, len(new_tokens),
        )

    # ==================================================================
    # Goose speculative decode (Phase 0/1 → chain; Phase 2 → tree)
    # ==================================================================

    def _decode_speculative(self) -> None:
        """KV-cache-aware speculative decode using Goose engine.

        Phase 0/1: linear chain verification (simple forward + argmax).
        Phase 2: tree attention verification (single forward with mask).

        Handles: draft generation, prefix KV load, model forward,
        token verification, KV cache update, logit harvesting.
        Falls through to normal ``_decode_step`` for any request where
        speculation is not applicable.
        """
        if not self._goose_enabled or self._goose_engine is None:
            return

        engine = self._goose_engine
        spec_stream = self.decode_stream

        for req in list(self.active_decode_pool):
            if req.is_done:
                continue

            context = req.tokens

            # ---- Step 1: Check if speculation is viable ----
            if not engine.can_speculate(context):
                continue

            # ---- Step 2: Generate draft tokens ----
            drafts, _bypass = engine.generate_draft(context)
            if not drafts:
                continue

            # ---- Step 3: Load prefix KV cache ----
            past_kv = None
            prefix_len = 0
            _goose_afce_anchors: int = 0
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                past_kv = self.cache.load_kv(req.cache_block_id)
                if past_kv is not None and len(past_kv) > 0 and past_kv[0][0] is not None:
                    prefix_len = past_kv[0][0].shape[2]

                # [AFCE] Extend KV with anchors for long-context speculative
                # quality.  The forward inside verify uses is_causal=True which
                # naturally allows attending to prepended anchors (whose abs
                # positions are all < query position).
                if self._afce_manager is not None:
                    _gk = self.cache.compute_hash(context)
                    if _gk and self._afce_manager.has_sidecar(_gk):
                        _ext: list[tuple[torch.Tensor, torch.Tensor]] = []
                        _an = 0
                        for _k, _v in past_kv:
                            _ek, _ev, _m, _n = self._afce_manager.extend_for_decode(
                                _gk, _k, _v, len(context) - 1,
                            )
                            _ext.append((_ek, _ev))
                            _an = _n
                        if _an > 0:
                            _goose_afce_anchors = _an
                            # Adjust prefix_len: anchors are prepended
                            prefix_len += _an
                            past_kv = _ext

            # ---- Step 4: Verify ----
            with torch.cuda.stream(spec_stream), torch.no_grad():
                if engine.tree_enabled:
                    # Phase 2: Build spine tree and use tree attention
                    tree = engine.build_spine_tree(context[-1], drafts)
                    accepted, next_token, new_kv = engine.verify_tree(
                        self.model, past_kv, tree, prefix_len,
                    )
                else:
                    # Phase 0/1: Linear chain verification
                    accepted, next_token, new_kv = engine.verify_linear(
                        self.model, past_kv, drafts, context,
                    )

            # ---- Step 5: Update request state ----
            if not accepted and past_kv is None:
                # No speculation benefit (warm-up or non-KV path)
                continue

            # Mark as handled by speculation
            self._spec_handled.add(req.request_id)

            # Append accepted tokens
            for tok in accepted:
                req.step()
                req.generated_tokens.append(tok)
                req.tokens.append(tok)

            # Append bonus token
            req.step()
            req.generated_tokens.append(next_token)
            req.tokens.append(next_token)

            # Update KV cache (sliced to accepted prefix)
            if new_kv is not None and req.cache_block_id is not None:
                # [AFCE] Strip anchor positions from speculative output
                if _goose_afce_anchors > 0 and self._afce_manager is not None:
                    new_kv = self._afce_manager.strip_anchors_from_kv(
                        new_kv, _goose_afce_anchors,
                    )
                self.cache.store_kv(req.cache_block_id, new_kv)

            # Update static KV (for non-cache forward tracking)
            if hasattr(self.model, "_last_kv_cache"):
                self.model._last_kv_cache = new_kv

            # ---- Step 6: Harvest logits into transition table ----
            if past_kv is not None:
                # Re-run a small forward on the generated window to get
                # logits for harvest. Use the same token window as context.
                harvest_window = req.tokens[-(len(accepted) + 1):]
                if harvest_window:
                    last_input = torch.tensor([harvest_window], dtype=torch.long, device="cuda")
                    with torch.cuda.stream(spec_stream), torch.no_grad():
                        lb_out = self.model.forward(
                            input_ids=last_input,
                            use_cache=False,
                        )
                    lb_logits = _extract_logits(lb_out)
                    engine.harvest_logits(lb_logits, list(harvest_window))

        logger.debug(
            "Speculative decode: %d/%d requests handled",
            len(self._spec_handled),
            len(self.active_decode_pool),
        )

    # ==================================================================
    # Main step
    # ==================================================================

    async def step(self):
        # Init layers (lazy)
        if self._expert_cache is None and getattr(self.model, "is_moe", False):
            self._init_expert_cache(vram_capacity=64)

        self._batch_prefill()

        if self._sere is None and getattr(self.model, "is_moe", False):
            self._init_sere()

        # [OEF] Lazy init: depends on model's expert count
        if _OEF_AVAILABLE and self._oef_controller is None and getattr(self.model, "is_moe", False):
            num_experts = getattr(self.model, "num_experts", 8)
            self._oef_controller = oef.OEFController(num_experts=num_experts)
            self._init_oef_lazy = True
            logger.info("OEF: OEFController initialized (%d experts)", num_experts)

        if self._dynamic_activator is None or self._spec_prefetcher is None:
            self._init_speculative_prefetch()

        # [Goose] Try speculative decode first
        self._decode_speculative()

        self._decode_step()

        # Clear speculation tracking
        self._spec_handled.clear()

        # Establish cross-stream ordering for next step.
        # wait_stream sets a GPU-side dependency (CPU returns immediately).
        # No full synchronize() here — it would stall the async pipeline
        # by idling the GPU between decode steps.  The targeted sync lives
        # inside _garbage_collect where actual memory freeing occurs.
        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        self._garbage_collect()

    # ==================================================================
    # Garbage collection
    # ==================================================================

    def _garbage_collect(self):
        finished = [d for d in self.active_decode_pool if d.is_done]
        if not finished:
            # Nothing to free — skip expensive sync
            self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
            self.cache.gc()
            return

        # Targeted sync: ensure no GPU kernel references blocks we're freeing
        torch.cuda.synchronize()

        for d in finished:
            logger.info(
                "Request %s complete (%d tokens)",
                d.request_id, len(d.generated_tokens),
            )
            if d.cache_block_id is not None:
                self.cache.free_block(d.cache_block_id)
            # [AFCE] Clean up per-request offset table
            if self._afce_manager is not None:
                self._afce_manager.remove_offset_table(d.request_id)
        self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
        self.cache.gc()
