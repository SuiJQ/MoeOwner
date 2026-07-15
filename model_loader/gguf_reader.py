"""
gguf_reader.py — Pure-Python GGUF spec v3 parser with PyTorch-native
dequantisation (no llama-cpp-python required).

┌──────────────────────────────────────────────────────────────────────────┐
│ GGUF v3 File Layout                                                      │
│                                                                          │
│  +──────────────────────────────────────────────────────────────────+    │
│  │ Header (magic=GGUF, version=3, tensor_count, metadata_kv_size)   │    │
│  ├──────────────────────────────────────────────────────────────────┤    │
│  │ Metadata KV pairs (metadata_kv_size bytes)                       │    │
│  ├──────────────────────────────────────────────────────────────────┤    │
│  │ Tensor Info Table (tensor_count x name+n_dims+dims+type+offset)  │    │
│  ├──────────────────────────────────────────────────────────────────┤    │
│  │ Padding to GGML_DEFAULT_ALIGNMENT (32 bytes)                    │    │
│  ├──────────────────────────────────────────────────────────────────┤    │
│  │ Tensor Data Section (each tensor at its declared offset)         │    │
│  +──────────────────────────────────────────────────────────────────+    │
│                                                                          │
│ Supported GGML type ⇨ PyTorch dtype mapping:                            │
│   F32 (0)   → torch.float32  (zero-copy from mmap)                      │
│   F16 (1)   → torch.float16  (zero-copy from mmap)                      │
│   Q4_0 (2)  → torch.float16  (PyTorch bitwise dequant)                  │
│   Q8_0 (8)  → torch.float16  (PyTorch bitwise dequant)                  │
└──────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model_adapter import GGUFModelAdapter

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GGML type constants
# ---------------------------------------------------------------------------

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
# K-quants (can be dequantised via Q4_0-like path when no native support)
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_Q4_K_M = 18

GGML_DEFAULT_ALIGNMENT = 32
GGUF_VERSION_MAX = 3

# Metadata value types (GGUF metadata spec)
GGUF_META_TYPE_UINT8 = 0
GGUF_META_TYPE_INT8 = 1
GGUF_META_TYPE_UINT16 = 2
GGUF_META_TYPE_INT16 = 3
GGUF_META_TYPE_UINT32 = 4
GGUF_META_TYPE_INT32 = 5
GGUF_META_TYPE_FLOAT32 = 6
GGUF_META_TYPE_BOOL = 7
GGUF_META_TYPE_STRING = 8
GGUF_META_TYPE_ARRAY = 9
GGUF_META_TYPE_UINT64 = 10
GGUF_META_TYPE_INT64 = 11
GGUF_META_TYPE_FLOAT64 = 12

GGML_BLOCK_SIZES = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_Q4_0: 32,   # 32 elements per block, 18 bytes each
    GGML_TYPE_Q8_0: 32,   # 32 elements per block, 34 bytes each
}

# Per-block byte size: (scale_dtype_size + quantized_data_size)
GGML_BLOCK_BYTES = {
    GGML_TYPE_Q4_0: 18,   # 2 (fp16 scale) + 16 (packed 4-bit)
    GGML_TYPE_Q8_0: 34,   # 2 (fp16 scale) + 32 (int8 values)
}

# Human-readable names for logging
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    18: "Q4_K_M",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TensorInfo:
    """Metadata for one tensor in the GGUF file."""
    name: str
    dims: list[int]
    ggml_type: int
    offset: int          # byte offset from start of tensor data section
    n_bytes: int = 0     # computed on init


@dataclass
class GGUFFile:
    """Parsed GGUF file in memory (backed by mmap)."""

    path: Path
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    tensor_data_offset: int = 0  # byte offset where tensor data begins

    _mmap: mmap.mmap | None = None
    _file_size: int = 0

    def close(self) -> None:
        """Release the mmap mapping."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# GGUF file parser (pure struct + mmap — no dependencies)
# ---------------------------------------------------------------------------


def _read_uint8(buf: memoryview, offset: int) -> int:
    return buf[offset]


_INT8_BIAS = 128


def _read_int8(buf: memoryview, offset: int) -> int:
    val = buf[offset]
    return val - 256 if val >= _INT8_BIAS else val


def _read_uint16_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<H", buf, offset)[0]


def _read_int16_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<h", buf, offset)[0]


def _read_uint32_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _read_int32_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<i", buf, offset)[0]


def _read_uint64_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<Q", buf, offset)[0]


def _read_int64_le(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<q", buf, offset)[0]


def _read_float32_le(buf: memoryview, offset: int) -> float:
    return struct.unpack_from("<f", buf, offset)[0]


def _read_float64_le(buf: memoryview, offset: int) -> float:
    return struct.unpack_from("<d", buf, offset)[0]


def _read_string(buf: memoryview, offset: int) -> tuple[str, int]:
    """Read a GGUF-length-prefixed UTF-8 string.

    Returns (string, bytes_consumed).
    """
    length = _read_uint64_le(buf, offset)
    start = offset + 8
    raw = bytes(buf[start:start + length])
    string = raw.decode("utf-8", errors="replace")
    return string, 8 + length


def _read_metadata_value(buf: memoryview, offset: int, value_type: int
                         ) -> tuple[Any, int]:
    """Read a single metadata value.

    Returns (parsed_value, bytes_consumed).
    """
    if value_type == GGUF_META_TYPE_UINT8:
        return _read_uint8(buf, offset), 1
    elif value_type == GGUF_META_TYPE_INT8:
        return _read_int8(buf, offset), 1
    elif value_type == GGUF_META_TYPE_UINT16:
        return _read_uint16_le(buf, offset), 2
    elif value_type == GGUF_META_TYPE_INT16:
        return _read_int16_le(buf, offset), 2
    elif value_type == GGUF_META_TYPE_UINT32:
        return _read_uint32_le(buf, offset), 4
    elif value_type == GGUF_META_TYPE_INT32:
        return _read_int32_le(buf, offset), 4
    elif value_type == GGUF_META_TYPE_FLOAT32:
        return _read_float32_le(buf, offset), 4
    elif value_type == GGUF_META_TYPE_BOOL:
        return bool(_read_uint8(buf, offset)), 1
    elif value_type == GGUF_META_TYPE_STRING:
        return _read_string(buf, offset)
    elif value_type == GGUF_META_TYPE_ARRAY:
        elem_type = _read_uint32_le(buf, offset)
        length = _read_uint64_le(buf, offset + 4)
        items: list[Any] = []
        consumed = 4 + 8
        for _ in range(length):
            val, n = _read_metadata_value(buf, offset + consumed, elem_type)
            items.append(val)
            consumed += n
        return items, consumed
    elif value_type == GGUF_META_TYPE_UINT64:
        return _read_uint64_le(buf, offset), 8
    elif value_type == GGUF_META_TYPE_INT64:
        return _read_int64_le(buf, offset), 8
    elif value_type == GGUF_META_TYPE_FLOAT64:
        return _read_float64_le(buf, offset), 8
    else:
        logger.warning("Unknown GGUF metadata value type %d, skipping 4 bytes",
                       value_type)
        return None, 4


def _compute_tensor_n_bytes(ggml_type: int, dims: list[int]) -> int:
    """Compute the byte size of a tensor in the GGUF file.

    For unquantized types (F32, F16), this is straight product x element size.
    For quantized types, blocks have a fixed byte-size per N elements.
    """
    n_elements = 1
    for d in dims:
        n_elements *= d

    if ggml_type == GGML_TYPE_F32:
        return n_elements * 4
    elif ggml_type == GGML_TYPE_F16:
        return n_elements * 2
    elif ggml_type in GGML_BLOCK_BYTES:
        block_size = GGML_BLOCK_SIZES[ggml_type]  # 32 for Q4_0/Q8_0
        block_bytes = GGML_BLOCK_BYTES[ggml_type]
        n_blocks = math.ceil(n_elements / block_size)
        return n_blocks * block_bytes
    else:
        logger.warning(
            "GGML type %d (%s) byte-size estimation not available — "
            "using raw n_bytes=0 (will fail at load time)",
            ggml_type, GGML_TYPE_NAMES.get(ggml_type, "?"),
        )
        return 0


def open_gguf(path: str | Path) -> GGUFFile:
    """Open and parse a GGUF file header + metadata (tensor info only
    — tensor data is lazily read via load_tensor()).

    Parameters
    ----------
    path : str | Path
        Path to a ``.gguf`` file.

    Returns
    -------
    GGUFFile
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    file_size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)

    try:
        # --- Parse file header ---
        magic = bytes(buf[:4])
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic.hex()!r})")

        version = _read_uint32_le(buf, 4)
        tensor_count = _read_uint64_le(buf, 8)
        metadata_kv_size = _read_uint64_le(buf, 16)

        logger.info(
            "GGUF header: version=%d, tensors=%d, metadata_kv_size=%d",
            version, tensor_count, metadata_kv_size,
        )

        if version < 1 or version > GGUF_VERSION_MAX:
            raise NotImplementedError(
                f"GGUF version {version} — this parser handles v1-v3"
            )

        if version == 1:
            # GGUF v1: tensors appear before metadata
            raise NotImplementedError(
                "GGUF v1 tensor-before-metadata not supported; "
                "use v2 or v3 files"
            )

        # --- Parse metadata KV pairs (offset 24 for v2/v3) ---
        metadata_offset = 24
        metadata_end = metadata_offset + metadata_kv_size
        offset = metadata_offset
        meta_dict: dict[str, Any] = {}

        while offset < metadata_end:
            key, n_key = _read_string(buf, offset)
            offset += n_key
            value_type = _read_uint32_le(buf, offset)
            offset += 4
            value, n_value = _read_metadata_value(buf, offset, value_type)
            offset += n_value
            meta_dict[key] = value

        # --- Parse tensor info table ---
        tensors: dict[str, TensorInfo] = {}
        for _ in range(tensor_count):
            name, n_name = _read_string(buf, offset)
            offset += n_name
            n_dims = _read_uint32_le(buf, offset)
            offset += 4
            dims = [_read_uint64_le(buf, offset + 8 * i) for i in range(n_dims)]
            offset += 8 * n_dims
            ggml_type = _read_uint32_le(buf, offset)
            offset += 4
            tensor_offset = _read_uint64_le(buf, offset)
            offset += 8

            info = TensorInfo(
                name=name,
                dims=dims,
                ggml_type=ggml_type,
                offset=tensor_offset,
                n_bytes=_compute_tensor_n_bytes(ggml_type, dims),
            )
            tensors[name] = info

        # --- Compute tensor data start (aligned to GGML_DEFAULT_ALIGNMENT) ---
        # GGUF v3 spec: tensor data starts at the next GGML_DEFAULT_ALIGNMENT
        # boundary after the end of the tensor info table.
        tensor_data_offset = (
            (offset + GGML_DEFAULT_ALIGNMENT - 1) // GGML_DEFAULT_ALIGNMENT
        ) * GGML_DEFAULT_ALIGNMENT

        result = GGUFFile(
            path=path,
            version=version,
            metadata=meta_dict,
            tensors=tensors,
            tensor_data_offset=tensor_data_offset,
            _mmap=buf,
            _file_size=file_size,
        )

        # Log summary
        logger.info("Parsed %d tensors (data offset=%d)",
                    len(tensors), tensor_data_offset)
        quant_counts: dict[str, int] = {}
        for info in tensors.values():
            tname = GGML_TYPE_NAMES.get(info.ggml_type, f"type_{info.ggml_type}")
            quant_counts[tname] = quant_counts.get(tname, 0) + 1
        for tname, count in sorted(quant_counts.items()):
            logger.info("  %s: %d tensors", tname, count)

        return result

    except Exception:
        # Ensure mmap is cleaned up on error
        buf.close()
        raise


def get_tensor_bytes(gguf: GGUFFile, tensor_name: str) -> memoryview:
    """Return a raw memoryview into the tensor data (no copy).

    Parameters
    ----------
    gguf : GGUFFile
    tensor_name : str
        Name of the tensor (e.g. ``"blk.0.attn_q.weight"``).

    Returns
    -------
    memoryview
        Read-only view into the mmap'd tensor data.
    """
    info = gguf.tensors.get(tensor_name)
    if info is None:
        available = "\n  ".join(sorted(gguf.tensors.keys()))
        raise KeyError(
            f"Tensor {tensor_name!r} not found in GGUF file. "
            f"Available tensors:\n  {available}"
        )

    start = gguf.tensor_data_offset + info.offset
    return memoryview(gguf._mmap)[start:start + info.n_bytes]


# ---------------------------------------------------------------------------
# Dequantisation kernels (pure PyTorch — no llama-cpp, no C extensions)
# ---------------------------------------------------------------------------


def dequantize_q4_0(data: memoryview) -> torch.Tensor:
    """Dequantise a Q4_0 block into FP16.

    Q4_0 layout (per 32-element block, 18 bytes):
      [0:2]   float16 scale (d)
      [2:18]  uint8[16] — each byte holds two 4-bit values (low nibble first)

    Returns torch.float16 tensor of shape (n_elements,).
    """
    raw = torch.frombuffer(data, dtype=torch.uint8)
    n_blocks = (raw.shape[0] + 17) // 18   # ceil division for safety

    # Slice into blocks: each block is 18 bytes
    # Handle the raw data as (n_blocks, 18)
    block_data = raw[:n_blocks * 18].view(-1, 18)

    # Scales: bytes 0-1 of each block, reinterpreted as float16
    # Use struct for the scale since torch cannot reinterpret uint8→float16 on CPU
    scales_list = []
    for i in range(n_blocks):
        scale_bytes = bytes(block_data[i, :2].tolist())
        scale = struct.unpack("<e", scale_bytes)[0]
        scales_list.append(scale)
    scales = torch.tensor(scales_list, dtype=torch.float16, device="cpu")  # (B,)

    # Quantised nibbles: bytes 2-17 of each block
    nibbles_packed = block_data[:, 2:]  # (B, 16) uint8
    low = nibbles_packed & 0x0F        # (B, 16) — low nibbles (Q4[0], Q4[2], ...)
    high = (nibbles_packed >> 4) & 0x0F  # (B, 16) — high nibbles (Q4[1], Q4[3], ...)
    values = torch.stack([low, high], dim=2).reshape(n_blocks, 32)  # (B, 32)
    values = values.to(torch.float16) - 8.0  # center around 8
    values = values * scales.unsqueeze(1)      # scale each block
    return values.flatten()


def dequantize_q8_0(data: memoryview) -> torch.Tensor:
    """Dequantise a Q8_0 block into FP16.

    Q8_0 layout (per 32-element block, 34 bytes):
      [0:2]   float16 scale (d)
      [2:34]  int8[32]

    Returns torch.float16 tensor of shape (n_elements,).
    """
    raw = torch.frombuffer(data, dtype=torch.uint8)
    n_blocks = (raw.shape[0] + 33) // 34

    block_data = raw[:n_blocks * 34].view(-1, 34)

    # Scales
    scales_list = []
    for i in range(n_blocks):
        scale_bytes = bytes(block_data[i, :2].tolist())
        scale = struct.unpack("<e", scale_bytes)[0]
        scales_list.append(scale)
    scales = torch.tensor(scales_list, dtype=torch.float16, device="cpu")

    # Quantised int8 values
    qs = block_data[:, 2:].to(torch.float16)  # (B, 32) uint8 → fp16
    # Reinterpret: uint8 0..255 stored at int8 0..127,-128..-1
    qs = torch.where(qs >= _INT8_BIAS, qs - 256, qs)  # convert to signed
    qs = qs * scales.unsqueeze(1)
    return qs.flatten()


# ---------------------------------------------------------------------------
# Load single tensor as torch.Tensor
# ---------------------------------------------------------------------------


def load_tensor(
    gguf: GGUFFile,
    tensor_name: str,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load a single GGUF tensor into a PyTorch tensor.

    Supports F32 (zero-copy), F16 (zero-copy), Q4_0, Q8_0.
    Quantized types are dequantised using pure-PyTorch kernels.

    Parameters
    ----------
    gguf : GGUFFile
        Opened GGUF file.
    tensor_name : str
        Tensor name (e.g. ``"blk.0.attn_q.weight"``).
    device : str | torch.device
        Target device (default: ``"cpu"``).

    Returns
    -------
    torch.Tensor
        Dequantised tensor in FP16 on the target device.
    """
    info = gguf.tensors.get(tensor_name)
    if info is None:
        raise KeyError(f"Tensor {tensor_name!r} not found")

    data = get_tensor_bytes(gguf, tensor_name)

    # --- Unquantized: zero-copy via torch.frombuffer ---
    # Note: for CPU, clone() after reshape to decouple from the mmap.
    # Without clone(), the tensor shares storage with the mmap and
    # prevents mmap.close().  On CUDA, .to(device) implicitly copies.
    _need_clone = device in ("cpu",) or torch.device(device).type == "cpu"

    if info.ggml_type == GGML_TYPE_F32:
        tensor = torch.frombuffer(
            data, dtype=torch.float32
        ).reshape(info.dims).to(device=device, dtype=torch.float16)

    elif info.ggml_type == GGML_TYPE_F16:
        tensor = torch.frombuffer(
            data, dtype=torch.float16
        ).reshape(info.dims).to(device=device)

    # --- Quantized: pure-PyTorch dequant ---
    elif info.ggml_type == GGML_TYPE_Q4_0:
        flat = dequantize_q4_0(data)
        tensor = flat.reshape(info.dims).to(device=device)
        _need_clone = False  # dequant already created new tensor

    elif info.ggml_type == GGML_TYPE_Q8_0:
        flat = dequantize_q8_0(data)
        tensor = flat.reshape(info.dims).to(device=device)
        _need_clone = False  # dequant already created new tensor

    else:
        raise NotImplementedError(
            f"GGML type {info.ggml_type} "
            f"({GGML_TYPE_NAMES.get(info.ggml_type, '?')}) not yet supported"
        )

    if _need_clone:
        tensor = tensor.clone()  # detach from mmap storage

    logger.debug(
        "Loaded tensor %s: dtype=%s, shape=%s, device=%s",
        tensor_name, tensor.dtype, tuple(tensor.shape), device,
    )
    return tensor


# ---------------------------------------------------------------------------
# Batch load model weights into a flat dict
# ---------------------------------------------------------------------------


def load_all_tensors(
    gguf: GGUFFile,
    device: str | torch.device = "cuda",
    tensor_filter: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load all tensors from a GGUF file into a dictionary.

    Parameters
    ----------
    gguf : GGUFFile
    device : str | torch.device
    tensor_filter : str | None
        If given, only load tensors whose name contains this substring
        (e.g. ``"attn_q"``).

    Returns
    -------
    Dict[str, torch.Tensor]
        Tensor name → FP16 tensor on target device.
    """
    result: dict[str, torch.Tensor] = {}
    for name in gguf.tensors:
        if tensor_filter and tensor_filter not in name:
            continue
        logger.info("Loading tensor: %s ...", name)
        result[name] = load_tensor(gguf, name, device=device)
    return result


# ---------------------------------------------------------------------------
# High-level model loading API (replaces HuggingFace AutoModelForCausalLM)
# ---------------------------------------------------------------------------


def load_model(
    path: str | Path,
    dtype: str = "fp16",
    device: str = "cuda",
    block_size: int = 32,
) -> GGUFModelAdapter:
    """Load a GGUF model as a PyTorch-compatible model adapter.

    This is the recommended entry point for the pure-python inference engine.

    Parameters
    ----------
    path : str | Path
        Path to the ``.gguf`` model file.
    dtype : str
        Target internal dtype: ``"fp16"`` (default), ``"bf16"``, or ``"fp32"``.
        All dequantised weights are stored in this dtype.
    device : str
        Target device (default: ``"cuda"``).
    block_size : int
        KV cache block size (recommended: 32 for GGUF models >7B, 16 otherwise).

    Returns
    -------
    GGUFModelAdapter
        A model-like object with ``forward()`` interface compatible with
        the engine's ``UnifiedScheduler``.
    """
    from .model_adapter import GGUFModelAdapter as _Adapter  # noqa: PLC0415

    adapter = _Adapter(
        path=path,
        target_dtype=dtype,
        device=device,
        block_size=block_size,
    )
    adapter.load()
    return adapter
