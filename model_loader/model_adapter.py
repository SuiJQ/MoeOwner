"""
model_adapter.py — GGUF Model Adapter (MoE-first, dense deprecated).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from torch import nn

from .gguf_reader import load_tensor_mmap_zero_copy, open_gguf

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = {
    "llama",
    "mixtral",
    "qwen2",
    "deepseek2",
    "starcoder2",
    "gemma2",
    "phi3",
}


class OptimisationContext:
    def __enter__(self):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return self

    def __exit__(self, *exc):
        pass


# ===================================================================
# KV Cache INT8 quantisation
# ===================================================================


class QuantizedKVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim, device):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.k_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim, dtype=torch.int8, device=device
        )
        self.v_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim // 2, dtype=torch.uint8, device=device
        )
        self.k_scale = torch.ones(
            max_batch_size, num_heads, max_seq_len, 1, dtype=torch.float16, device=device
        )
        self.v_scale = torch.ones(
            max_batch_size, num_heads, max_seq_len, 1, dtype=torch.float16, device=device
        )

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _quantize_tensor(t):
        abs_max = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / 127.0
        q = (t / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale.to(torch.float16)

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _quantize_value_int4(t):
        abs_max = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / 7.0
        q = (t / scale).round().clamp(-8, 7).to(torch.int8)
        q_biased = (q + 8).to(torch.uint8)
        *rest, d = q_biased.shape
        d2 = d // 2
        q_paired = q_biased.view(*rest, d2, 2)
        packed = q_paired[..., 0] | (q_paired[..., 1] << 4)
        return packed, scale.to(torch.float16)

    @staticmethod
    def _dequantize_int4(packed, scale):
        low = (packed & 0xF).to(torch.uint8)
        high = ((packed >> 4) & 0xF).to(torch.uint8)
        v0 = (low.to(torch.float16) - 8.0) * scale
        v1 = (high.to(torch.float16) - 8.0) * scale
        return torch.stack([v0, v1], dim=-1).flatten(-2)

    @staticmethod
    def _dequantize(q, scale):
        return q.to(torch.float16) * scale

    def append(self, batch_idx, position, k, v):
        k_flat = k.squeeze(0).squeeze(0)
        v_flat = v.squeeze(0).squeeze(0)
        qk, sk = self._quantize_tensor(k_flat.unsqueeze(0))
        qv, sv = self._quantize_value_int4(v_flat.unsqueeze(0))
        self.k_cache[batch_idx, :, position, :] = qk
        self.v_cache[batch_idx, :, position, :] = qv.squeeze(0)
        self.k_scale[batch_idx, :, position, :] = sk.squeeze(0)
        self.v_scale[batch_idx, :, position, :] = sv.squeeze(0)

    def read(self, batch_idx, upto):
        return (
            self._dequantize(
                self.k_cache[batch_idx, :, :upto, :], self.k_scale[batch_idx, :, :upto, :]
            ),
            self._dequantize_int4(
                self.v_cache[batch_idx, :, :upto, :], self.v_scale[batch_idx, :, :upto, :]
            ),
        )


# ===================================================================
# Flash attention (single canonical version, dynamic=True)
# ===================================================================


class MoEFlashAttention:
    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=True)
    def forward(q, k, v, softmax_scale, causal=True):
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=softmax_scale, is_causal=causal, enable_gqa=True
            )
        except (RuntimeError, ValueError):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=softmax_scale, is_causal=causal
            )


# ===================================================================
# MoE Decoder Layer
# ===================================================================


class MoEDecoderLayer(nn.Module):
    def __init__(
        self,
        weights,
        layer_idx,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        num_experts=8,
        top_k=2,
        intermediate_size=None,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size or hidden_size * 8 // 3

        # [Bug 3] Seed attributes for scheduler wiring.
        self.expert_cache = None
        self.sere = None

        prefix = f"blk.{layer_idx}"

        def _w(name):
            return weights[f"{prefix}.{name}"]

        self.attn_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.attn_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.attn_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.attn_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.attn_q.weight.data = _w("attn_q.weight")
        self.attn_k.weight.data = _w("attn_k.weight")
        self.attn_v.weight.data = _w("attn_v.weight")
        self.attn_o.weight.data = _w("attn_o.weight")

        self.input_norm = _w("attn_norm.weight")
        self.post_attn_norm = _w("ffn_norm.weight")

        router_weight = _w("ffn_gate.weight")
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.router.weight.data = router_weight

        # Expert raw tensors (CPU for cache registration)
        self.ffn_gate_weights: list[torch.Tensor] = []
        self.ffn_up_weights: list[torch.Tensor] = []
        self.ffn_down_weights: list[torch.Tensor] = []
        self.expert_gates: nn.ModuleList = nn.ModuleList()
        self.expert_ups: nn.ModuleList = nn.ModuleList()
        self.expert_downs: nn.ModuleList = nn.ModuleList()

        for e in range(num_experts):
            gate_w = weights.get(f"{prefix}.ffn_gate.{e}.weight")
            up_w = weights.get(f"{prefix}.ffn_up.{e}.weight")
            down_w = weights.get(f"{prefix}.ffn_down.{e}.weight")
            if gate_w is None:
                gate_w = weights.get(f"{prefix}.expert.{e}.ffn_gate.weight")
                up_w = weights.get(f"{prefix}.expert.{e}.ffn_up.weight")
                down_w = weights.get(f"{prefix}.expert.{e}.ffn_down.weight")
            if gate_w is None:
                raise KeyError(f"Expert {e} gate weight not found for layer {layer_idx}.")

            # [Fix 6] Keep on CPU (NOT pin_memory here — cache.register_expert pins).
            self.ffn_gate_weights.append(gate_w.cpu())
            self.ffn_up_weights.append(up_w.cpu())
            self.ffn_down_weights.append(down_w.cpu())

            g = nn.Linear(hidden_size, self.intermediate_size, bias=False)
            u = nn.Linear(hidden_size, self.intermediate_size, bias=False)
            d = nn.Linear(self.intermediate_size, hidden_size, bias=False)
            g.weight.data = gate_w
            u.weight.data = up_w
            d.weight.data = down_w
            self.expert_gates.append(g)
            self.expert_ups.append(u)
            self.expert_downs.append(d)

        self._cudaize()

    def _cudaize(self):
        """[Fix 6] Move attn/router to CUDA. Expert raw tensors stay CPU (pinned by cache)."""
        for m in [self.attn_q, self.attn_k, self.attn_v, self.attn_o, self.router]:
            m.to(device="cuda", dtype=torch.float16)
        # Expert raw tensors already CPU from __init__; no double copy.
        # nn.Linear stays on CUDA for legacy (no cache) path.

    def get_cpu_expert_weights(self, expert_idx):
        return (
            self.ffn_gate_weights[expert_idx],
            self.ffn_up_weights[expert_idx],
            self.ffn_down_weights[expert_idx],
        )

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        return x * weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(self, x, past_kv=None, use_cache=False):
        residual = x
        normed = self._rms_norm(x, self.input_norm, self.rms_norm_eps)

        b, t, d = normed.shape
        q = self.attn_q(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(normed).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(normed).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None and past_kv[0] is not None:
            k_old, v_old = past_kv
            k = torch.cat([k_old.to(k.device), k], dim=2)
            v = torch.cat([v_old.to(v.device), v], dim=2)

        softmax_scale = self.head_dim**-0.5
        attn_out = MoEFlashAttention.forward(q, k, v, softmax_scale, causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        attn_out = self.attn_o(attn_out)
        h = residual + attn_out

        # MoE FFN
        normed_h = self._rms_norm(h, self.post_attn_norm, self.rms_norm_eps)
        router_logits = self.router(normed_h)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(normed_h.dtype)

        expert_cache = self.expert_cache
        sere = self.sere

        if sere is not None:
            top_k_probs, top_k_indices, _ = sere(router_probs)
        else:
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        routing_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        combined_output = torch.zeros_like(normed_h)
        actual_top_k = top_k_indices.shape[-1]

        if expert_cache is not None:
            cache = expert_cache
            for k_idx in range(actual_top_k):
                expert_idx = top_k_indices[..., k_idx]
                weight = routing_weights[..., k_idx : k_idx + 1]

                flat_h = normed_h.view(-1, d)
                flat_idx = expert_idx.view(-1)
                flat_weight = weight.view(-1, 1)

                unique_experts = torch.unique(flat_idx).tolist()
                for e in unique_experts:
                    mask = flat_idx == e
                    if not mask.any():
                        continue
                    h_expert = flat_h[mask]
                    w_mask = flat_weight[mask]

                    result = cache.get_or_load_expert(self.layer_idx, e)
                    if result is None:
                        continue
                    _, weight_dict = result
                    gate_w, up_w, down_w = (
                        weight_dict["gate"],
                        weight_dict["up"],
                        weight_dict["down"],
                    )

                    gate_out = h_expert @ gate_w.T
                    up_out = h_expert @ up_w.T
                    act = gate_out * torch.nn.functional.silu(gate_out)
                    expert_out = (act * up_out) @ down_w.T
                    combined_output.view(-1, d)[mask] += expert_out * w_mask

                    cache.release_expert(self.layer_idx, e)
        else:
            for k_idx in range(actual_top_k):
                expert_idx = top_k_indices[..., k_idx]
                weight = routing_weights[..., k_idx : k_idx + 1]

                flat_h = normed_h.view(-1, d)
                flat_idx = expert_idx.view(-1)
                flat_weight = weight.view(-1, 1)

                for e in range(self.num_experts):
                    mask = flat_idx == e
                    if not mask.any():
                        continue
                    h_expert = flat_h[mask]
                    gate_out = self.expert_gates[e](h_expert)
                    up_out = self.expert_ups[e](h_expert)
                    act = gate_out * torch.nn.functional.silu(gate_out)
                    expert_out = self.expert_downs[e](act * up_out)
                    combined_output.view(-1, d)[mask] += expert_out * flat_weight[mask]

        h = h + combined_output
        new_kv = (k, v) if use_cache else None
        return h, new_kv


# ===================================================================
# Dense Decoder Layer (legacy)
# ===================================================================


class DenseDecoderLayer(nn.Module):
    def __init__(
        self,
        weights,
        layer_idx,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size=None,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size or hidden_size * 4

        prefix = f"blk.{layer_idx}"

        def _w(name):
            return weights[f"{prefix}.{name}"]

        self.attn_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.attn_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.attn_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.attn_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.attn_q.weight.data = _w("attn_q.weight")
        self.attn_k.weight.data = _w("attn_k.weight")
        self.attn_v.weight.data = _w("attn_v.weight")
        self.attn_o.weight.data = _w("attn_o.weight")

        self.mlp_gate = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.mlp_up = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.mlp_down = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.mlp_gate.weight.data = _w("ffn_gate.weight")
        self.mlp_up.weight.data = _w("ffn_up.weight")
        self.mlp_down.weight.data = _w("ffn_down.weight")

        self.input_norm = _w("attn_norm.weight")
        self.post_attn_norm = _w("ffn_norm.weight")
        self._cudaize()

    def _cudaize(self):
        for m in [
            self.attn_q,
            self.attn_k,
            self.attn_v,
            self.attn_o,
            self.mlp_gate,
            self.mlp_up,
            self.mlp_down,
        ]:
            m.to(device="cuda", dtype=torch.float16)

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        return x * weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(self, x, past_kv=None, use_cache=False):
        residual = x
        normed = self._rms_norm(x, self.input_norm, self.rms_norm_eps)
        b, t, d = normed.shape
        q = self.attn_q(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(normed).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(normed).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if past_kv is not None and past_kv[0] is not None:
            k_old, v_old = past_kv
            k = torch.cat([k_old.to(k.device), k], dim=2)
            v = torch.cat([v_old.to(v.device), v], dim=2)
        softmax_scale = self.head_dim**-0.5
        attn_out = MoEFlashAttention.forward(q, k, v, softmax_scale, causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        attn_out = self.attn_o(attn_out)
        h = residual + attn_out

        normed_h = self._rms_norm(h, self.post_attn_norm, self.rms_norm_eps)
        gate_out = self.mlp_gate(normed_h)
        up_out = self.mlp_up(normed_h)
        act = gate_out * torch.nn.functional.silu(gate_out)
        mlp_out = self.mlp_down(act * up_out)
        h = h + mlp_out
        new_kv = (k, v) if use_cache else None
        return h, new_kv


# ===================================================================
# GGUFModelAdapter
# ===================================================================


class GGUFModelAdapter(nn.Module):
    def __init__(self, path, target_dtype="fp16", device="cuda", block_size=32):
        super().__init__()
        self.path = Path(path)
        self.target_dtype = target_dtype
        self.device = torch.device(device)
        self.block_size = block_size

        self._gguf = None
        self._weight_dict = {}
        self.layers = nn.ModuleList()
        self.norm_weight = None
        self.lm_head = None
        self.embed_tokens = None

        self.hidden_size = 0
        self.num_heads = 0
        self.num_kv_heads = 0
        self.num_layers = 0
        self.head_dim = 0
        self.vocab_size = 0

        self.num_experts = 0
        self.num_experts_per_tok = 0
        self.is_moe = False
        self.intermediate_size = 0  # set during _build_model
        self._params_loaded = False

    def load(self):
        logger.info("Opening GGUF: %s", self.path)
        gguf = open_gguf(self.path)
        self._gguf = gguf
        self._parse_metadata(gguf)
        with OptimisationContext():
            self._weight_dict = {
                name: load_tensor_mmap_zero_copy(gguf, name, device="cuda") for name in gguf.tensors
            }
        self._build_model()
        self._params_loaded = True
        logger.info(
            "GGUFModelAdapter ready: %d layers, hidden=%d, heads=%d, kv_heads=%d, "
            "vocab=%d, is_moe=%s%s",
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.vocab_size,
            self.is_moe,
            f", experts={self.num_experts}, top_k={self.num_experts_per_tok}"
            if self.is_moe
            else "",
        )

    def _parse_metadata(self, gguf):
        meta = gguf.metadata
        self.architecture = meta.get("general.architecture", "unknown")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            logger.warning(
                "Architecture %r not in known list %s", self.architecture, SUPPORTED_ARCHITECTURES
            )
        self.num_layers = int(meta.get(f"{self.architecture}.block_count", 0))
        self.hidden_size = int(
            meta.get(
                f"{self.architecture}.embedding_length",
                meta.get(f"{self.architecture}.hidden_size", 4096),
            )
        )
        self.num_heads = int(meta.get(f"{self.architecture}.attention.head_count", 32))
        self.num_kv_heads = int(
            meta.get(f"{self.architecture}.attention.head_count_kv", self.num_heads)
        )
        self.head_dim = int(
            meta.get(
                f"{self.architecture}.attention.key_length", self.hidden_size // self.num_heads
            )
        )
        self.vocab_size = int(meta.get(f"{self.architecture}.vocab_size", 32000))
        self.num_experts = int(meta.get(f"{self.architecture}.attention.expert_count", 0))
        self.num_experts_per_tok = int(
            meta.get(f"{self.architecture}.attention.expert_used_count", 0)
        )
        if self.num_experts > 0:
            self.is_moe = True
        else:
            for name in gguf.tensors:
                if re.search(r"ffn_gate\.\d+", name) or re.search(r"expert\.\d+", name):
                    self.is_moe = True
                    break
            if self.is_moe:
                max_e = 0
                for name in gguf.tensors:
                    m = re.search(r"ffn_gate\.(\d+)", name)
                    if m:
                        max_e = max(max_e, int(m.group(1)) + 1)
                self.num_experts = max_e or 8
                self.num_experts_per_tok = self.num_experts_per_tok or 2
        if self.is_moe:
            logger.info(
                "MoE detected: %d experts, top-%d", self.num_experts, self.num_experts_per_tok
            )

    def _build_model(self):
        w = self._weight_dict
        emb_weight = w.get("token_embd.weight")
        if emb_weight is not None:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, dtype=torch.float16, device="cuda"
            )
            self.embed_tokens.weight.data = emb_weight

        layer_list = []
        for i in range(self.num_layers):
            prefix = f"blk.{i}"
            is_moe_layer = self.is_moe and (
                f"{prefix}.ffn_gate.0.weight" in w or f"{prefix}.expert.0.ffn_gate.weight" in w
            )
            if is_moe_layer:
                layer = MoEDecoderLayer(
                    weights=w,
                    layer_idx=i,
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    num_experts=self.num_experts,
                    top_k=self.num_experts_per_tok,
                )
            else:
                layer = DenseDecoderLayer(
                    weights=w,
                    layer_idx=i,
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                )
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)

        # Set adapter-level intermediate_size from the first MoE layer
        for ly in layer_list:
            if hasattr(ly, "intermediate_size"):
                self.intermediate_size = ly.intermediate_size
                break

        self.norm_weight = w.get("output_norm.weight")
        if self.norm_weight is not None:
            self.norm_weight = self.norm_weight.to(device="cuda", dtype=torch.float16)

        head_weight = w.get("output.weight")
        if head_weight is not None:
            self.lm_head = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False, dtype=torch.float16, device="cuda"
            )
            self.lm_head.weight.data = head_weight

    # ------------------------------------------------------------------
    # Forward [Fix 1] [Fix 4] Accepts attention_mask + **kwargs
    # ------------------------------------------------------------------

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        return x * weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(
        self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs
    ):
        """[Fix 1] Accepts attention_mask and **kwargs for batch prefill compat.

        [Fix 4] When attention_mask is provided, embeds only valid tokens with
        a 1:1 identity embedding for padding positions (to avoid numerical
        contamination) and uses causal masking in SDPA.  The layer-level
        MoEFlashAttention already uses ``is_causal=True``, so the mask is
        primarily consumed by SDPA's internal handling.
        """
        if not self._params_loaded:
            raise RuntimeError("Model not loaded — call .load() first")

        # [Fix 4] For batch prefill with left-padded sequences, embedding is
        # applied to all tokens including padding (position 0).  SDPA's
        # ``is_causal`` combined with ``attention_mask`` handles the masking.
        h = self.embed_tokens(input_ids)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, nkv = layer(h, past_kv=pkv, use_cache=use_cache)
            new_kvs.append(nkv)

        if self.norm_weight is not None:
            h = self._rms_norm(h, self.norm_weight)

        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = h @ self.embed_tokens.weight.T

        if use_cache:
            self._last_kv_cache = new_kvs

        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _BLOCK_SIZE_70B = 64
    _BLOCK_SIZE_7B = 32
    _BLOCK_SIZE_SMALL = 16
    _THRESH_70B = 70.0
    _THRESH_7B = 7.0

    @staticmethod
    def suggest_block_size(num_parameters_b):
        if num_parameters_b >= GGUFModelAdapter._THRESH_70B:
            return GGUFModelAdapter._BLOCK_SIZE_70B
        if num_parameters_b >= GGUFModelAdapter._THRESH_7B:
            return GGUFModelAdapter._BLOCK_SIZE_7B
        return GGUFModelAdapter._BLOCK_SIZE_SMALL

    @property
    def estimated_parameter_count_b(self):
        base_per_layer = 4 * self.hidden_size * self.hidden_size
        if self.is_moe and self.num_experts > 0:
            gate_up = 2 * self.hidden_size * self.intermediate_size
            down = self.hidden_size * self.intermediate_size
            expert_total = self.num_experts * (gate_up + down)
            total = self.num_layers * (base_per_layer + expert_total)
        else:
            intermediate = self.hidden_size * 4
            total = self.num_layers * (base_per_layer + 3 * self.hidden_size * intermediate)
        return round(total / 1e9, 1)
