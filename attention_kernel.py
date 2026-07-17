"""
FlashAttentionKernel — optimized scaled dot-product attention via torch.compile.

Uses PyTorch's native SDPA with ``torch.compile`` graph capture and
GQA support.  Falls back gracefully when ``enable_gqa`` is not
available on older PyTorch builds.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class FlashAttentionKernel:
    """Compiled flash attention wrapper using PyTorch's native SDPA.

    Uses ``dynamic=True`` so that variable-length inputs do not trigger
    recompilation — critical for real-world inference with batching.
    """

    @torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=True)
    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute flash attention with optional causal masking and GQA.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``(B, H_q, T, D)``.
        k : torch.Tensor
            Key tensor, shape ``(B, H_k, T, D)``.
        v : torch.Tensor
            Value tensor, shape ``(B, H_v, T, D)``.
        softmax_scale : float
            Scale factor applied before softmax (typically ``1 / sqrt(D)``).
        causal : bool, optional
            Whether to apply causal masking (default ``True``).

        Returns
        -------
        torch.Tensor
            Attention output tensor, shape ``(B, H_q, T, D)``.
        """
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=softmax_scale,
                is_causal=causal,
                enable_gqa=True,
            )
        except (RuntimeError, ValueError) as _exc:
            # PyTorch < 2.6 or non-GQA head counts may reject enable_gqa.
            logger.warning(
                "SDPA with enable_gqa failed (%s: %s), falling back to standard SDPA",
                type(_exc).__name__,
                _exc,
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=softmax_scale,
                is_causal=causal,
            )
