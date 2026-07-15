"""
model_loader — Pure-Python GGUF reader + PyTorch weight adapter.

Entry point:
    load_model(model_path, dtype="fp16", device="cuda")

See gguf_reader.py for the low-level GGUF spec parser and per-format
dequantisation kernels.
"""

from .gguf_reader import GGUFFile, load_model

__all__ = ["GGUFFile", "load_model"]
