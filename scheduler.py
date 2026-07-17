"""
UnifiedScheduler — Three-Layer End-to-End MoE Inference Pipeline.
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

    def step(self) -> None:
        self._step_count += 1

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.max_new_tokens


CUDA_GRAPH_BATCH_SIZES = (1, 4, 8)


class CUDAGraphManager:
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
# UnifiedScheduler
# ===================================================================


class UnifiedScheduler:
    CHUNK_SIZE = 512
    _PREFILL_BATCH_TIMEOUT = 0.5
    _PREFILL_BATCH_MAX = 8
    _EXPERT_CAPACITY_FACTOR = 1.2

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
            "UnifiedScheduler: chunk=%d, batch_max=%d, cap_factor=%.2f",
            self.CHUNK_SIZE,
            self._PREFILL_BATCH_MAX,
            self._EXPERT_CAPACITY_FACTOR,
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
        """Lazy init DynamicExpertActivator + SpeculativePrefetcher."""
        from speculative_prefetch import (  # noqa: PLC0415
            DynamicExpertActivator,
            SpeculativePrefetcher,
        )

        if self._dynamic_activator is None:
            # All tunable parameters use the class defaults — no manual overrides.
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
        """[Fix 3] Get decoder layers from both HF (model.model.layers) and GGUF (model.layers)."""
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return None

    def _init_sere(self):
        """[Fix 3] Attach SERE to all MoE layers regardless of model nesting."""
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
                elif hasattr(layer, "self_attn"):
                    # Also check the HF sub-layer's self_attn
                    pass  # HF attention injected via _inject_attention_kernel, not SERE

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
            "Expert cache: %d x %d experts, VRAM=%d blocks", num_layers, num_experts, vram_capacity
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
    # CUDA Graph init [Fix 2]
    # ==================================================================

    def init_cuda_graphs(self):
        """[Fix 2] Only record if expert cache is NOT active (H2D breaks graph)."""
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
    # Prefill
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

        all_ids = [
            torch.tensor(r.prompt_tokens, dtype=torch.long, device="cpu") for r in batch_requests
        ]
        reversed_ids = [ids.flip(0) for ids in all_ids]
        padded_rev = torch.nn.utils.rnn.pad_sequence(
            reversed_ids, batch_first=True, padding_value=0
        )
        padded_ids = padded_rev.flip(1)
        attention_mask = (padded_ids != 0).to(torch.long)

        with torch.cuda.stream(self.prefill_stream):
            padded_ids_gpu = padded_ids.to(device="cuda", non_blocking=True)
            attention_mask_gpu = attention_mask.to(device="cuda", non_blocking=True)
            _ = self.model.forward(
                input_ids=padded_ids_gpu, attention_mask=attention_mask_gpu, use_cache=True
            )

        for req in batch_requests:
            self.active_decode_pool.append(
                DecodeRequest(
                    tokens=req.prompt_tokens,
                    generated_tokens=[],
                    request_id=req.request_id,
                    max_new_tokens=req.max_new_tokens,
                )
            )

        logger.info(
            "Batch prefill: %d requests, padded=%s", len(batch_requests), list(padded_ids.shape)
        )

    # ==================================================================
    # Decode [Fix 2] [Fix 5]
    # ==================================================================

    def _decode_step(self):
        if not self.active_decode_pool:
            self._decode_bs = 0
            return

        current_count = len(self.active_decode_pool)
        mgr = self._cuda_graph_mgr

        if self._decode_bs == 0 or current_count > self._decode_bs:
            for bs in sorted(CUDA_GRAPH_BATCH_SIZES):
                if bs >= current_count:
                    target_bs = bs
                    break
            else:
                target_bs = CUDA_GRAPH_BATCH_SIZES[-1]
            self._decode_bs = target_bs

        # [Fix 2] If expert cache is active, mgr may exist but is a no-op.
        use_cuda_graph = mgr is not None and self._expert_cache is None

        # [Fix 7] Speculative decode on request 0 only.
        spec_advanced = 0
        if self._spec_gen and current_count >= 1 and use_cuda_graph:
            # [Fix 5] Build context list on CPU side to avoid CUDA sync.
            main_req = self.active_decode_pool[0]
            ctx_list = main_req.tokens  # already CPU list
            ctx_tensor = torch.tensor([ctx_list], dtype=torch.long, device="cuda")
            new_ids, num_accepted, draft_used = self._spec_gen.decode(ctx_tensor)
            if num_accepted > 0:
                for tok in new_ids[0, len(main_req.tokens) :].tolist():
                    main_req.tokens.append(tok)
                    main_req.generated_tokens.append(tok)
                    main_req.step()
                spec_advanced = num_accepted + 1
                logger.debug(
                    "Spec: draft=%d, accepted=%d, advanced=%d",
                    draft_used,
                    num_accepted,
                    spec_advanced,
                )

        with torch.cuda.stream(self.decode_stream):
            if use_cuda_graph:
                batch_tokens = []
                for i in range(self._decode_bs):
                    if i < current_count:
                        req = self.active_decode_pool[i]
                        batch_tokens.append(req.tokens[-1] if req.tokens else 0)
                    else:
                        batch_tokens.append(0)

                input_ids = torch.tensor(batch_tokens, dtype=torch.long, device="cuda").view(
                    self._decode_bs, 1
                )
                logits = mgr.replay(batch_size=self._decode_bs, input_ids=input_ids)
                next_token_ids = logits[:, -1, :].argmax(dim=-1)

                # [Dynamic Expert Activation] confidence-based k adjustment
                if self._dynamic_activator is not None:
                    # Use the first active request's logit as the confidence signal
                    self._dynamic_activator.update_from_logits(
                        logits[:1, :, :],
                        generated_ids=self.active_decode_pool[0].tokens if current_count > 0 else None,
                        detokenizer=self._detokenizer,
                    )

                for i in range(current_count):
                    req = self.active_decode_pool[i]
                    if i == 0 and spec_advanced > 0:
                        continue
                    req.step()
                    tok = int(next_token_ids[i].item())
                    req.generated_tokens.append(tok)
                    req.tokens.append(tok)

                # [Speculative Prefetch] trigger next-step prediction
                if self._spec_prefetcher is not None and current_count > 0:
                    self._spec_prefetcher.step(
                        context_ids=self.active_decode_pool[0].tokens,
                        transfer_stream=self.transfer_stream,
                    )
            else:
                # [Fix 2] Direct forward (no CUDA Graph or expert cache active).
                for i, req in enumerate(self.active_decode_pool):
                    if i == 0 and spec_advanced > 0:
                        continue
                    inp = torch.tensor([[req.tokens[-1]]], dtype=torch.long, device="cuda")
                    with torch.no_grad():
                        logits = self.model(inp)
                    next_tok = logits[0, -1, :].argmax().item()
                    req.step()
                    req.generated_tokens.append(next_tok)
                    req.tokens.append(next_tok)

                    # [Dynamic Expert Activation] per-token confidence adjustment
                    if self._dynamic_activator is not None and i == 0:
                        self._dynamic_activator.update_from_logits(
                            logits[None, :, :],  # [1, seq, vocab]
                            generated_ids=req.tokens,
                            detokenizer=self._detokenizer,
                        )

                # [Speculative Prefetch] trigger after all requests processed
                if self._spec_prefetcher is not None and current_count > 0:
                    self._spec_prefetcher.step(
                        context_ids=self.active_decode_pool[0].tokens,
                        transfer_stream=self.transfer_stream,
                    )

    # ==================================================================
    # Main step [Fix 2] expert cache before CUDA graphs
    # ==================================================================

    async def step(self):
        # [Fix 2] Init expert cache BEFORE CUDA graphs so the guard is correct.
        if self._expert_cache is None and getattr(self.model, "is_moe", False):
            self._init_expert_cache(vram_capacity=64)

        if self._cuda_graph_mgr is None and hasattr(self.model, "parameters"):
            self.init_cuda_graphs()

        self._batch_prefill()

        if self._spec_gen is None:
            self._init_ngram()
        if self._sere is None and getattr(self.model, "is_moe", False):
            self._init_sere()

        # [Speculative Prefetch] init after ngram + sere + expert_cache ready
        if self._dynamic_activator is None or self._spec_prefetcher is None:
            self._init_speculative_prefetch()

        self._decode_step()

        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        torch.cuda.synchronize()

        self._garbage_collect()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _garbage_collect(self):
        finished = [d for d in self.active_decode_pool if d.is_done]
        self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
        for d in finished:
            logger.info("Request %s complete", d.request_id)
        self.cache.gc()
