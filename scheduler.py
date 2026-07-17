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


CUDA_GRAPH_BATCH_SIZES = (1, 4, 8)


class CUDAGraphManager:
    """CUDA Graph pre-recording for static-batch single-token decode.

    Note: when KV cache is active, CUDA Graphs are **disabled** because
    the model needs to return per-layer KV tensors (non-static output).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.head_dim = 128
        self.num_heads = 32
        self.num_kv_heads = 4

        if hasattr(model, "lm_head") and model.lm_head is not None:
            self.vocab_size = model.lm_head.out_features
            self.hidden_size = model.lm_head.in_features
        elif hasattr(model, "embed_tokens") and model.embed_tokens is not None:
            self.vocab_size = model.embed_tokens.num_embeddings
            self.hidden_size = model.embed_tokens.embedding_dim

        if hasattr(model, "layers") and len(model.layers) > 0:
            layer = model.layers[0]
            if hasattr(layer, "head_dim"):
                self.head_dim = layer.head_dim
            if hasattr(layer, "num_heads"):
                self.num_heads = layer.num_heads
            if hasattr(layer, "num_kv_heads"):
                self.num_kv_heads = layer.num_kv_heads

        self._pool = torch.cuda.graph_pool_handle()
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._static_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._static_outputs: dict[int, dict[str, torch.Tensor]] = {}

    def record_all(self) -> None:
        device = next(self.model.parameters()).device
        for bs in CUDA_GRAPH_BATCH_SIZES:
            logger.info("Recording CUDA Graph for BS=%d ...", bs)
            static_input_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
            _ = self._model_forward(static_input_ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                static_output = self._model_forward(static_input_ids)
            self._graphs[bs] = graph
            self._static_inputs[bs] = {"input_ids": static_input_ids}
            self._static_outputs[bs] = {"logits": static_output}

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False)
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "logits"):
            return out.logits
        return out

    def replay(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        graph = self._graphs.get(batch_size)
        if graph is None:
            raise RuntimeError(f"No CUDA Graph for BS={batch_size}.")
        self._static_inputs[batch_size]["input_ids"].copy_(input_ids)
        graph.replay()
        return self._static_outputs[batch_size]["logits"]

    @property
    def available_batch_sizes(self) -> list[int]:
        return sorted(self._graphs.keys())

    def has_graph_for(self, bs: int) -> bool:
        return bs in self._graphs


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

        self._cuda_graph_mgr: CUDAGraphManager | None = None
        self._ngram_cache = None
        self._spec_gen = None
        self._sere = None
        self._expert_cache = None
        self._batch_plan = None

        # [Speculative Prefetch & Dynamic Expert Activation]
        self._dynamic_activator = None
        self._spec_prefetcher = None

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

    def _init_ngram(self):
        from ngram_speculation import NGramCache, SpeculativeGenerator  # noqa: PLC0415

        if self._ngram_cache is None:
            self._ngram_cache = NGramCache(max_n=5, max_nodes=100000)
            self._spec_gen = SpeculativeGenerator(
                model=self.model, ngram_cache=self._ngram_cache, max_draft=5, enabled=True
            )

    def _init_speculative_prefetch(self):
        from speculative_prefetch import (  # noqa: PLC0415
            DynamicExpertActivator,
            SpeculativePrefetcher,
        )

        if self._dynamic_activator is None:
            self._dynamic_activator = DynamicExpertActivator(
                sere_module=self._sere,
            )

        if self._spec_prefetcher is None and self._ngram_cache is not None:
            num_layers = getattr(self.model, "num_layers", 0)
            num_experts = getattr(self.model, "num_experts", 8)
            self._spec_prefetcher = SpeculativePrefetcher(
                ngram_cache=self._ngram_cache,
                expert_cache=self._expert_cache,
                sere_module=self._sere,
                num_layers=num_layers,
                num_experts=num_experts,
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
    # CUDA Graph init
    # ==================================================================

    def init_cuda_graphs(self):
        """Record CUDA Graphs only when KV cache is disabled (static output)."""
        if self._KV_CACHE_ENABLED:
            logger.info("KV cache active — CUDA Graphs disabled (dynamic KV output).")
            return
        if self._expert_cache is not None:
            logger.info("Expert cache active — CUDA Graphs disabled.")
            return
        logger.info("Recording CUDA Graphs...")
        mgr = CUDAGraphManager(model=self.model)
        mgr.record_all()
        self._cuda_graph_mgr = mgr
        logger.info("CUDA Graphs ready: BS=%s", mgr.available_batch_sizes)

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

        # -- Speculative decode (request 0 only, only when CUDA Graphs active) --
        spec_advanced = 0
        _spec_eligible = (
            self._spec_gen
            and len(self.active_decode_pool) >= 1
            and self._cuda_graph_mgr is not None
            and not self._KV_CACHE_ENABLED
            and self._expert_cache is None
        )
        if _spec_eligible:
            main_req = self.active_decode_pool[0]
            ctx_list = main_req.tokens
            if len(ctx_list) > 0:
                ctx_tensor = torch.tensor([ctx_list], dtype=torch.long, device="cuda")
                new_ids, num_accepted, _draft_used = self._spec_gen.decode(ctx_tensor)
                if num_accepted > 0:
                    for tok in new_ids[0, len(main_req.tokens):].tolist():
                        main_req.tokens.append(tok)
                        main_req.generated_tokens.append(tok)
                        main_req.step()
                    spec_advanced = num_accepted + 1

        with torch.cuda.stream(self.decode_stream):
            for i, req in enumerate(self.active_decode_pool):
                if i == 0 and spec_advanced > 0:
                    continue

                last_tok = req.tokens[-1] if req.tokens else 0
                inp = torch.tensor([[last_tok]], dtype=torch.long, device="cuda")

                # Load cached KV if available
                past_kv = None
                if req.cache_block_id is not None:
                    past_kv = self.cache.load_kv(req.cache_block_id)

                with torch.no_grad():
                    if past_kv is not None and self._KV_CACHE_ENABLED:
                        model_out = self.model.forward(
                            input_ids=inp,
                            past_key_values=past_kv,
                            use_cache=True,
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
                        self.cache.store_kv(req.cache_block_id, new_kv)

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

    # ==================================================================
    # Main step
    # ==================================================================

    async def step(self):
        # Init layers (lazy)
        if self._expert_cache is None and getattr(self.model, "is_moe", False):
            self._init_expert_cache(vram_capacity=64)

        if self._cuda_graph_mgr is None and hasattr(self.model, "parameters"):
            self.init_cuda_graphs()

        self._batch_prefill()

        if self._spec_gen is None:
            self._init_ngram()
        if self._sere is None and getattr(self.model, "is_moe", False):
            self._init_sere()

        if self._dynamic_activator is None or self._spec_prefetcher is None:
            self._init_speculative_prefetch()

        self._decode_step()

        # Sync streams
        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        torch.cuda.synchronize()

        self._garbage_collect()

    # ==================================================================
    # Garbage collection
    # ==================================================================

    def _garbage_collect(self):
        finished = [d for d in self.active_decode_pool if d.is_done]
        for d in finished:
            logger.info("Request %s complete (%d tokens)", d.request_id, len(d.generated_tokens))
            if d.cache_block_id is not None:
                self.cache.free_block(d.cache_block_id)
        self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
        self.cache.gc()
