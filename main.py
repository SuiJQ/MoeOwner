#!/usr/bin/env python3
"""
Pure Python Inference Engine — Main Entry Point

Usage:
    python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Stages (see project README for full design):
    1. Dependency locking
    2. Global torch performance configuration
    3. Module loading (attention_kernel, cache_manager, scheduler)
    4. Model loading & kernel injection
    5. Async event loop
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

import torch
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("engine")


# ---------------------------------------------------------------------------
# Stage 2: Global torch performance configuration
# Must execute before any model loading.
# ---------------------------------------------------------------------------

def configure_global_torch() -> None:
    """Enable all available CUDA performance knobs for inference.

    ⚠️  Call immediately after ``import torch``, before model loading.
    """
    # Flash SDP backend (cuDNN flash-attention)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # TF32 matmul / cuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.cudnn.allow_tf32 = True

    # Float32 matmul precision – 'high' uses TF32 where possible
    torch.set_float32_matmul_precision("high")

    # Dedicate a separate default CUDA stream for the engine
    torch.cuda.set_stream(torch.cuda.Stream())

    logger.info("Global torch performance configuration applied")


configure_global_torch()


# ---------------------------------------------------------------------------
# Stage 3: Import real modules
# ---------------------------------------------------------------------------

try:
    from cache_manager import Block, HybridCache  # noqa: F401
    from scheduler import DecodeRequest, Request, UnifiedScheduler  # noqa: F401
    logger.info("Imported cache_manager & scheduler modules")
except ImportError as exc:
    logger.error("Failed to import engine modules: %s", exc)
    logger.error("Ensure cache_manager.py and scheduler.py are in PYTHONPATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 3b: Import GGUF model loader (optional)
# ---------------------------------------------------------------------------

_HAS_GGUF = False
try:
    from model_loader import load_model as load_gguf_model
    from model_loader.gguf_reader import GGUFFile as _GGUFFile  # noqa: F401
    _HAS_GGUF = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Stage 4: Model loading & attention kernel injection
# ---------------------------------------------------------------------------

def _inject_attention_kernel(layer: torch.nn.Module) -> None:
    """Replace ``self_attn.forward`` with a SDPA-based compiled flash kernel.

    The actual attention computation is handled by
    ``attention_kernel.FlashAttentionKernel.forward``; this function
    wires it into an existing HuggingFace decoder layer by monkey-patching
    the layer's ``self_attn.forward`` method.
    """
    from attention_kernel import FlashAttentionKernel  # noqa: PLC0415

    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return

    # Capture the original forward for any model that needs it internally
    # (e.g. cross-attention layers); our patched forward primarily handles
    # the causal self-attention path.
    original_forward = attn.forward  # noqa: F841

    def _patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple:
        """Replacement forward that uses FlashAttentionKernel.

        This is a lean wrapper: it unpacks Q/K/V, calls the compiled
        SDPA kernel, then packs the output back into the format
        expected by the HuggingFace decoder layer.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q/K/V using the layer's own projection weights
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape to multi-head format
        num_heads = attn.num_heads
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Compute softmax scale
        softmax_scale = head_dim ** -0.5

        # Run compiled SDPA
        attn_output = FlashAttentionKernel.forward(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=True,
        )

        # Restore shape: (B, H, T, D) -> (B, T, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Final projection
        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = attn.o_proj(attn_output)

        # Return in the format HuggingFace decoder layers expect
        return (attn_output, None)  # (output, past_key_value)

    attn.forward = _patched_forward
    attn.is_causal = True
    logger.debug("Injected FlashAttentionKernel into layer %s", type(layer).__name__)


def load_and_inject_model(model_name: str) -> torch.nn.Module:
    """Load a HuggingFace model, move to GPU, inject attention kernels.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local path.

    Returns
    -------
    torch.nn.Module
        Loaded model in eval mode with custom attention kernels.
    """
    logger.info("Loading HuggingFace model: %s ...", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device="cuda")
    model.eval()

    # Inject custom flash-attention kernel into every decoder layer
    layers = getattr(model.model, "layers", None)
    if layers is not None:
        for layer in layers:
            _inject_attention_kernel(layer)
    else:
        logger.warning(
            "Could not locate model.model.layers; attention injection skipped. "
            "The model will run with its original attention implementation."
        )

    # Warmup forward pass to trigger JIT compilation / CUDA graph capture
    dummy_input = torch.randint(0, 1000, (1, 64), device="cuda")
    with torch.no_grad():
        model(dummy_input)

    logger.info("HuggingFace model loaded, injected, and warmed up.")
    return model


def load_and_inject_gguf_model(
    gguf_path: str,
    device: str = "cuda",
    block_size: int | None = None,
) -> torch.nn.Module:
    """Load a model from a GGUF file using the pure-Python GGUF reader.

    Parameters
    ----------
    gguf_path : str
        Path to a ``.gguf`` file.
    device : str
        Target device (default: cuda).
    block_size : int or None
        KV cache block size. If None, dynamically suggested from model size.

    Returns
    -------
    GGUFModelAdapter
        Model adapter ready for the engine scheduler.
    """
    if not _HAS_GGUF:
        raise ImportError(
            "model_loader not available. Run from pure-python-engine/ dir."
        )

    # Determine block size dynamically if not specified (optimisation #4)
    if block_size is None:
        # Default to 32 for GGUF models (better for larger models)
        block_size = 32

    logger.info("Loading GGUF model: %s (block_size=%d)", gguf_path, block_size)
    model = load_gguf_model(
        path=gguf_path,
        dtype="fp16",
        device=device,
        block_size=block_size,
    )

    # Log dynamic block-size suggestion
    est_params = model.estimated_parameter_count_b
    suggested_bs = model.suggest_block_size(est_params)
    if suggested_bs != block_size:
        logger.info(
            "💡 Suggested block_size=%d for ~%.1fB model "
            "(current=%d)",
            suggested_bs, est_params, block_size,
        )

    logger.info("GGUF model loaded: %d layers, %.1fB params",
                model.num_layers, est_params)
    return model


# ---------------------------------------------------------------------------
# Stage 5: Main async loop
# ---------------------------------------------------------------------------

def _setup_signal_handlers(scheduler: UnifiedScheduler) -> None:
    """Register signal handlers for graceful shutdown."""

    def _signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down gracefully...", signum)
        scheduler.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


async def main() -> None:
    """Main entry point: configure, load model, run event loop."""
    parser = argparse.ArgumentParser(
        description="Pure Python Inference Engine — Hybrid Paged+Radix KV Cache"
    )
    parser = argparse.ArgumentParser(
        description="Pure Python Inference Engine — Hybrid Paged+Radix KV Cache"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", ""),
        help="HuggingFace model name or local path to GGUF file",
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Path to .gguf file (overrides --model if set)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Number of tokens per KV cache block "
             "(default: 16 for HF, 32 for GGUF — auto-tuned for model size)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden dimension of the model (default: 4096)",
    )
    parser.add_argument(
        "--total-blocks",
        type=int,
        default=None,
        help="Override automatic GPU-memory-based block count",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Stage 4: Load model (GGUF or HuggingFace)
    if args.gguf or (args.model and args.model.endswith(".gguf")):
        gguf_path = args.gguf or args.model
        model = load_and_inject_gguf_model(
            gguf_path=gguf_path,
            device="cuda",
            block_size=args.block_size,
        )
        # Extract model dimensions from the GGUF adapter
        model_hidden_size = model.hidden_size
        block_size = args.block_size or model.block_size
    else:
        model = load_and_inject_model(args.model)
        model_hidden_size = args.hidden_size
        block_size = args.block_size or 16

    # Stage 3: Create cache and scheduler with the real implementations
    cache = HybridCache(
        block_size=block_size,
        hidden_size=model_hidden_size,
        total_blocks=args.total_blocks,
    )
    scheduler = UnifiedScheduler(model, cache)
    _setup_signal_handlers(scheduler)

    logger.info("Engine running. Press Ctrl+C to stop.")
    try:
        while scheduler._running:
            await scheduler.step()
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Engine stopped.")


if __name__ == "__main__":
    asyncio.run(main())
