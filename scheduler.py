"""
UnifiedScheduler — Chunked Prefill + Decode with dual CUDA streams.

Architecture
------------
- Prefill runs on a dedicated CUDA stream (``prefill_stream``).
- Decode runs on a dedicated CUDA stream (``decode_stream``).
- Both streams are synchronised from the **main** CUDA stream **only** — never
  inside the stream work itself — to avoid deadlock.
- New requests are chunked at ``CHUNK_SIZE`` (512) tokens. Every other prefill
  chunk triggers a decode step from the running queue, keeping both pipelines
  occupied.
- Prefix cache hits (via ``HybridCache.match_prefix``) reduce redundant
  computation.
- Completed request cache blocks are released via ``cache.gc()``.

⚠️  **Integration note**: This scheduler is a **reference design**. The chunked
prefill logic correctly demonstrates the dual-stream pipeline pattern, but a
production deployment would need to:
  1. Wire ``past_key_values`` from ``HybridCache`` physical blocks into the
     model's attention layers (via a custom ``Cache`` implementation).
  2. Perform actual ``model.forward()`` for decode steps (not just the
     lifecycle hook).
  3. Handle KV-cache page tables for true PagedAttention.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import attention_kernel  # noqa: F401 — ensures the module is importable

from cache_manager import HybridCache, Block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """A new incoming request awaiting its first prefill."""
    prompt_tokens: List[int]
    request_id: str
    cached_heads: List[int] = field(default_factory=list)


@dataclass
class DecodeRequest:
    """A request that has been prefilled and is now being decoded
    auto-regressively."""

    tokens: List[int]
    generated_tokens: List[int]
    request_id: str
    max_new_tokens: int
    _step_count: int = 0

    def step(self) -> None:
        """Lifecycle hook — advance internal step counter."""
        self._step_count += 1

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.max_new_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_kv_to_gpu(physical_address: int, kv_state: object) -> None:
    """Record that KV cache is written to the given physical address.

    Production path: perform an asynchronous H2D copy of the incremental KV
    entries into the pre-allocated page-pool at *physical_address*.  For this
    reference design we only log the event so the GC can trace liveness.
    """
    logger.debug("KV written to physical block 0x%x (simulated)", physical_address)


# ---------------------------------------------------------------------------
# UnifiedScheduler
# ---------------------------------------------------------------------------


class UnifiedScheduler:
    """Orchestrate chunked prefill and decode across two CUDA streams.

    Parameters
    ----------
    model:
        A HuggingFace ``PreTrainedModel`` (or compatible) instance whose
        ``forward`` method accepts keyword arguments such as
        ``input_ids``, ``past_key_values``, ``use_cache``, etc.

    cache: HybridCache
        Block-level KV cache manager (see ``cache_manager``).
    """

    CHUNK_SIZE: int = 512

    def __init__(self, model: object, cache: HybridCache) -> None:
        self.model = model
        self.cache = cache

        # Dedicated CUDA streams — created lazily so that they bind to
        # the device that is current at construction time.
        self.prefill_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.decode_stream: torch.cuda.Stream = torch.cuda.Stream()

        # Request queues
        self.waiting_queue: List[Request] = []
        self.running_requests: List[DecodeRequest] = []

        # Scheduler flag
        self._running: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: Request) -> None:
        """Enqueue a new request for the next :meth:`step`."""
        self.waiting_queue.append(request)

    def shutdown(self) -> None:
        """Signal the scheduler to stop after the current step."""
        self._running = False

    async def step(self) -> None:
        """Execute one scheduling step.

        Stages
        ------
        1. **Chunked prefill** — drain the waiting queue on ``prefill_stream``;
           every other chunk also pops a decode step on ``decode_stream``.

        2. **Regular decode** — one iteration per running request on
           ``decode_stream`` (only when ``prefill_stream`` is idle).

        3. **Stream sync** — synchronise both streams back to the main stream
           (⚠️  never inside the per-stream blocks).

        4. **GC** — release cache blocks belonging to completed requests.
        """
        # ---- 1. Chunked prefill injection ---------------------------------
        pending = list(self.waiting_queue)
        self.waiting_queue.clear()

        for req in pending:
            tokens = req.prompt_tokens
            chunks = [
                tokens[i : i + self.CHUNK_SIZE]
                for i in range(0, len(tokens), self.CHUNK_SIZE)
            ]

            for chunk_index, chunk in enumerate(chunks):
                # --- Prefix cache lookup ---
                hit_block_id, remaining = self.cache.match_prefix(chunk)
                if hit_block_id is not None:
                    req.cached_heads.append(hit_block_id)

                if remaining:
                    new_block: Block = self.cache.allocate(remaining)
                    # Prefill on the prefill stream
                    with torch.cuda.stream(self.prefill_stream):
                        # ── Production note ──
                        # ``past_key_values`` should come from the cache
                        # manager's physical block pool.  A custom
                        # ``DynamicCache`` subclass that reads/writes
                        # from pre-allocated GPU buffers is required.
                        # See the project README for integration guidance.
                        output = self.model.forward(
                            input_ids=torch.tensor([remaining], device="cuda"),
                            past_key_values=None,
                            use_cache=True,
                        )
                    write_kv_to_gpu(new_block.physical_address, output.past_key_values)

                # Every other chunk: steal a decode step
                if chunk_index % 2 == 1 and self.running_requests:
                    decode_req = self.running_requests.pop(0)
                    with torch.cuda.stream(self.decode_stream):
                        # ── Production note ──
                        # In a real deployment the model call would go here.
                        # The ``step()`` hook advances the internal counter
                        # so GC can still free completed requests.
                        # next_token_id = model.forward(...)
                        # decode_req.tokens.append(next_token_id)
                        # decode_req.generated_tokens.append(next_token_id)
                        decode_req.step()

                    if not decode_req.is_done:
                        self.running_requests.append(decode_req)

        # ---- 2. Regular decode for running requests -----------------------
        for decode_req in list(self.running_requests):
            # Non-blocking check: only decode when prefill stream is idle
            if not self.prefill_stream.query():
                with torch.cuda.stream(self.decode_stream):
                    decode_req.step()

        # ---- 3. Stream synchronisation (MAIN thread only) -----------------
        # ⚠️  Never call synchronize inside the per-stream blocks above.
        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.synchronize()

        # ---- 4. Garbage collection ----------------------------------------
        self._garbage_collect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cache_from_block(block: Block) -> Optional[object]:
        """Return the KV cache stored at *block*.

        This accessor allows production code to override the lookup strategy
        (e.g. indirect page tables) without modifying the hot loop in
        :meth:`step`.  Currently returns the physical address as a placeholder.
        """
        return block.physical_address if hasattr(block, "physical_address") else None

    def _garbage_collect(self) -> None:
        """Release cache blocks belonging to completed requests."""
        finished: List[DecodeRequest] = []
        still_running: List[DecodeRequest] = []

        for dreq in self.running_requests:
            if dreq.is_done:
                finished.append(dreq)
            else:
                still_running.append(dreq)

        self.running_requests = still_running

        for dreq in finished:
            logger.info(
                "Request %s complete, %d generated tokens",
                dreq.request_id,
                len(dreq.generated_tokens),
            )

        self.cache.gc()
