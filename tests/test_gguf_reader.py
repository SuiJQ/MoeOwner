#!/usr/bin/env python3
"""
tests/test_gguf_reader.py — Unit tests for the pure-Python GGUF reader.

Tests are designed to run without a GPU (CPU-only).  The GGUF dequantisation
tests construct synthetic Q4_0/Q8_0 block payloads in-memory and verify that
the dequantised values match the expected FP16 result.

Run with:
    cd /workspace/pure-python-engine
    python -m pytest tests/test_gguf_reader.py -v
"""

from __future__ import annotations

import array
import os
import struct
import sys
import tempfile

# Ensure the module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from model_loader.gguf_reader import (
    GGML_DEFAULT_ALIGNMENT,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    _compute_tensor_n_bytes,
    _read_float32_le,
    _read_int16_le,
    _read_metadata_value,
    _read_string,
    _read_uint8,
    _read_uint16_le,
    _read_uint32_le,
    _read_uint64_le,
    dequantize_q4_0,
    dequantize_q8_0,
    load_tensor,
    open_gguf,
)

# ====================================================================
# Binary reader helpers
# ====================================================================


class TestBinaryReaders:
    """Test the low-level struct.unpack wrappers."""

    @pytest.fixture
    def buf(self):
        """Create a memoryview of known little-endian bytes."""
        data = array.array("B", [
            0x00,  # uint8: 0
            0xFF,  # uint8: 255
            0x10, 0x00,  # uint16 LE: 16
            0xFF, 0x7F,  # int16 LE: 32767
            0x78, 0x56, 0x34, 0x12,  # uint32 LE: 0x12345678
            0x38, 0xB4, 0x96, 0x49,  # float32 LE: 1234567.0
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,  # uint64 LE: 0x0123456789ABCDEF
        ])
        return memoryview(bytes(data))

    def test_read_uint8(self, buf):
        assert _read_uint8(buf, 0) == 0
        assert _read_uint8(buf, 1) == 255

    def test_read_uint16_le(self, buf):
        assert _read_uint16_le(buf, 2) == 16

    def test_read_int16_le(self, buf):
        assert _read_int16_le(buf, 4) == 32767

    def test_read_uint32_le(self, buf):
        assert _read_uint32_le(buf, 6) == 0x12345678

    def test_read_float32_le(self, buf):
        val = _read_float32_le(buf, 10)
        assert abs(val - 1234567.0) < 0.1

    def test_read_uint64_le(self, buf):
        assert _read_uint64_le(buf, 14) == 0x0123456789ABCDEF


class TestStringReader:
    """Test the GGUF length-prefixed UTF-8 string reader."""

    def test_read_string(self):
        test_str = "hello"
        raw = struct.pack("<Q", len(test_str)) + test_str.encode("utf-8")
        buf = memoryview(raw)
        s, n = _read_string(buf, 0)
        assert s == "hello"
        assert n == 8 + 5

    def test_read_string_empty(self):
        raw = struct.pack("<Q", 0)
        buf = memoryview(raw)
        s, n = _read_string(buf, 0)
        assert s == ""
        assert n == 8

    def test_read_unicode(self):
        test_str = "你好🌍"
        encoded = test_str.encode("utf-8")
        raw = struct.pack("<Q", len(encoded)) + encoded
        buf = memoryview(raw)
        s, n = _read_string(buf, 0)
        assert s == test_str
        assert n == 8 + len(encoded)


class TestMetadataReader:
    """Test metadata value parsing for all GGUF metadata types."""

    def _build_value(self, typ: int, payload: bytes) -> memoryview:
        return memoryview(payload)

    def test_uint8(self):
        val, n = _read_metadata_value(memoryview(b"\x2A"), 0, 0)
        assert val == 42
        assert n == 1

    def test_int8(self):
        val, n = _read_metadata_value(memoryview(b"\xFE"), 0, 1)
        assert val == -2
        assert n == 1

    def test_uint32(self):
        payload = struct.pack("<I", 1000000)
        val, n = _read_metadata_value(memoryview(payload), 0, 4)
        assert val == 1000000
        assert n == 4

    def test_float32(self):
        payload = struct.pack("<f", 3.14159)
        val, n = _read_metadata_value(memoryview(payload), 0, 6)
        assert abs(val - 3.14159) < 1e-5
        assert n == 4

    def test_bool_true(self):
        val, n = _read_metadata_value(memoryview(b"\x01"), 0, 7)
        assert val is True
        assert n == 1

    def test_bool_false(self):
        val, n = _read_metadata_value(memoryview(b"\x00"), 0, 7)
        assert val is False
        assert n == 1

    def test_string(self):
        test_str = "llama"
        raw = struct.pack("<Q", len(test_str)) + test_str.encode("utf-8")
        val, n = _read_metadata_value(memoryview(raw), 0, 8)
        assert val == "llama"
        assert n == 8 + 5

    def test_array(self):
        """Test array of uint32: [1, 2, 3]"""
        raw = struct.pack("<I", 4)  # value type: uint32
        raw += struct.pack("<Q", 3)  # length: 3
        raw += struct.pack("<III", 1, 2, 3)
        val, _n = _read_metadata_value(memoryview(raw), 0, 9)
        assert val == [1, 2, 3]


class TestTensorNBytes:
    """Test tensor byte size computation."""

    def test_f32(self):
        assert _compute_tensor_n_bytes(GGML_TYPE_F32, [64, 4096]) == 64 * 4096 * 4

    def test_f16(self):
        assert _compute_tensor_n_bytes(GGML_TYPE_F16, [64, 4096]) == 64 * 4096 * 2

    def test_q4_0(self):
        # 32 elements → 1 block × 18 bytes
        assert _compute_tensor_n_bytes(GGML_TYPE_Q4_0, [32]) == 18
        # 64 elements → 2 blocks × 18 bytes = 36
        assert _compute_tensor_n_bytes(GGML_TYPE_Q4_0, [64]) == 36
        # 33 elements → 2 blocks × 18 bytes = 36 (ceil)
        assert _compute_tensor_n_bytes(GGML_TYPE_Q4_0, [33]) == 36

    def test_q8_0(self):
        assert _compute_tensor_n_bytes(GGML_TYPE_Q8_0, [32]) == 34
        assert _compute_tensor_n_bytes(GGML_TYPE_Q8_0, [64]) == 68

    def test_3d_tensor_q4_0(self):
        # [2, 4, 32] = 256 elements → 8 blocks × 18 = 144
        assert _compute_tensor_n_bytes(GGML_TYPE_Q4_0, [2, 4, 32]) == 144


# ====================================================================
# Dequantisation kernel tests
# ====================================================================


class TestDequantQ40:
    """Test Q4_0 dequantisation against known golden values."""

    def _make_q4_0_block(self, values_float: list) -> bytes:
        """Create a Q4_0 block from 32 FP32 values."""
        assert len(values_float) == 32
        arr = torch.tensor(values_float, dtype=torch.float32)

        # Compute scale: max abs
        abs_max = arr.abs().max().item()
        d = abs_max / 7.0 if abs_max > 0 else 1.0
        # Scale values to 0..15 centered at 8
        quantized = (arr.numpy() / d + 8.0).round().clip(0, 15).astype(int)

        # Pack nibbles: low nibble first
        packed = []
        for i in range(0, 32, 2):
            low = quantized[i]
            high = quantized[i + 1]
            packed.append(low | (high << 4))

        return struct.pack("<e", d) + bytes(packed)

    def test_single_block_constant(self):
        """All values = 1.0 → all nibbles = 9 → dequant ≈ 1.0"""
        values = [1.0] * 32
        data = self._make_q4_0_block(values)
        result = dequantize_q4_0(memoryview(data))
        assert result.shape == (32,)
        assert result.dtype == torch.float16
        # Should approximately match
        assert torch.allclose(result, torch.tensor(values, dtype=torch.float16),
                              atol=0.3)

    def test_single_block_linear(self):
        """Linearly increasing values."""
        values = [float(i) for i in range(32)]
        data = self._make_q4_0_block(values)
        result = dequantize_q4_0(memoryview(data))
        assert result.shape == (32,)
        # Check rough correctness
        expected = torch.tensor(values, dtype=torch.float16)
        # Q4_0 has limited precision, expect ~10% relative error at most
        rel_error = (result - expected).abs().mean().item()
        assert rel_error < 2.0, f"Mean absolute error too high: {rel_error}"

    def test_two_blocks(self):
        """64 elements = 2 blocks."""
        values = [float(i % 32) for i in range(64)]
        data1 = self._make_q4_0_block(values[:32])
        data2 = self._make_q4_0_block(values[32:])
        combined = data1 + data2
        result = dequantize_q4_0(memoryview(combined))
        assert result.shape == (64,)

    def test_non_multiples(self):
        """Data shorter than one block — should still work."""
        # Just 18 bytes (one block)
        values = [1.0] * 32
        data = self._make_q4_0_block(values)
        result = dequantize_q4_0(memoryview(data[:18]))
        assert result.shape == (32,)


class TestDequantQ80:
    """Test Q8_0 dequantisation."""

    def _make_q8_0_block(self, values_float: list) -> bytes:
        """Create a Q8_0 block from 32 FP32 values."""
        assert len(values_float) == 32
        arr = torch.tensor(values_float, dtype=torch.float32)
        abs_max = arr.abs().max().item()
        abs_max = max(abs_max, 1e-8)
        d = abs_max / 127.0
        quantized = (arr.numpy() / d).round().clip(-128, 127).astype(np.int8)
        return struct.pack("<e", d) + bytes(quantized)

    def test_single_block_ones(self):
        values = [1.0] * 32
        data = self._make_q8_0_block(values)
        result = dequantize_q8_0(memoryview(data))
        assert result.shape == (32,)
        assert result.dtype == torch.float16
        assert torch.allclose(result, torch.tensor(values, dtype=torch.float16),
                              atol=0.1)

    def test_single_block_negative(self):
        values = [float(i - 16) for i in range(32)]
        data = self._make_q8_0_block(values)
        result = dequantize_q8_0(memoryview(data))
        expected = torch.tensor(values, dtype=torch.float16)
        rel_error = (result - expected).abs().mean().item()
        assert rel_error < 1.0, f"Q8_0 error too high: {rel_error}"


# ====================================================================
# GGUF file creation helpers & integration tests
# ====================================================================


def _create_minimal_gguf_bytes(
    metadata: dict,
    tensors_data: list,
    version: int = 3,
) -> bytes:
    """Build a valid (minimal) GGUF v3 file in-memory from Python objects.

    Parameters
    ----------
    metadata : dict
        String-key → (value_type_int, value) pairs.
    tensors_data : list
        List of (name, dims, ggml_type, raw_bytes) tuples.
    version : int
        GGUF version (2 or 3).

    Returns
    -------
    bytes
        Ready-to-save GGUF file.
    """
    header = bytearray()
    # Magic
    header.extend(b"GGUF")
    # Version
    header.extend(struct.pack("<I", version))

    # We need to know metadata_kv_size first — build metadata section
    meta_section = bytearray()
    for key, (vtype, value) in metadata.items():
        # Key
        key_encoded = key.encode("utf-8")
        meta_section.extend(struct.pack("<Q", len(key_encoded)))
        meta_section.extend(key_encoded)
        # Value type
        meta_section.extend(struct.pack("<I", vtype))
        # Value payload
        if vtype == 4:  # uint32
            meta_section.extend(struct.pack("<I", value))
        elif vtype == 6:  # float32
            meta_section.extend(struct.pack("<f", value))
        elif vtype == 8:  # string
            value_encoded = value.encode("utf-8")
            meta_section.extend(struct.pack("<Q", len(value_encoded)))
            meta_section.extend(value_encoded)
        elif vtype == 7:  # bool
            meta_section.extend(b"\x01" if value else b"\x00")
        elif vtype == 10:  # uint64
            meta_section.extend(struct.pack("<Q", value))
        elif vtype == 0:  # uint8
            meta_section.extend(struct.pack("<B", value))
        elif vtype == 5:  # int32
            meta_section.extend(struct.pack("<i", value))
        elif vtype == 9:  # array — simplified: array of uint32
            elem_type, items = value[0], value[1]
            meta_section.extend(struct.pack("<I", elem_type))
            meta_section.extend(struct.pack("<Q", len(items)))
            for item in items:
                meta_section.extend(struct.pack("<I", item))
        else:
            raise ValueError(f"Unsupported test meta type {vtype}")

    # Tensor count
    header.extend(struct.pack("<Q", len(tensors_data)))
    # Metadata KV size
    header.extend(struct.pack("<Q", len(meta_section)))

    # Build tensor info section
    tensor_info = bytearray()
    tensor_body = bytearray()
    current_tensor_offset = 0

    for name, dims, ggml_type, raw_bytes in tensors_data:
        name_encoded = name.encode("utf-8")
        tensor_info.extend(struct.pack("<Q", len(name_encoded)))
        tensor_info.extend(name_encoded)
        tensor_info.extend(struct.pack("<I", len(dims)))
        for d in dims:
            tensor_info.extend(struct.pack("<Q", d))
        tensor_info.extend(struct.pack("<I", ggml_type))
        tensor_info.extend(struct.pack("<Q", current_tensor_offset))

        tensor_body.extend(raw_bytes)
        current_tensor_offset += len(raw_bytes)

    # Align tensor data to GGML_DEFAULT_ALIGNMENT
    pre_tensor_size = len(header) + len(meta_section) + len(tensor_info)
    aligned_start = (
        (pre_tensor_size + GGML_DEFAULT_ALIGNMENT - 1) // GGML_DEFAULT_ALIGNMENT
    ) * GGML_DEFAULT_ALIGNMENT
    padding_needed = aligned_start - pre_tensor_size

    return bytes(header + meta_section + tensor_info) + b"\x00" * padding_needed + bytes(tensor_body)


class TestSmallGgufFile:
    """Test parsing a tiny synthetic GGUF file."""

    @pytest.fixture
    def gguf_path(self):
        """Create a minimal valid GGUF file for testing."""
        # A 1D F16 tensor with 4 elements: [1.0, 2.0, 3.0, 4.0]
        f16_data = struct.pack("<eeee", 1.0, 2.0, 3.0, 4.0)
        metadata = {
            "general.architecture": (8, "test"),
            "test.block_count": (10, 1),
            "test.embedding_length": (10, 64),
        }
        tensors = [
            ("test_tensor", [4], GGML_TYPE_F16, f16_data),
        ]
        content = _create_minimal_gguf_bytes(metadata, tensors, version=3)

        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(content)
        tmp.close()
        yield tmp.name
        os.unlink(tmp.name)

    def test_open_and_parse(self, gguf_path):
        with open_gguf(gguf_path) as gguf:
            assert gguf.version == 3
            assert gguf.metadata["general.architecture"] == "test"
            assert gguf.metadata["test.block_count"] == 1
            assert "test_tensor" in gguf.tensors
            info = gguf.tensors["test_tensor"]
            assert info.dims == [4]
            assert info.ggml_type == GGML_TYPE_F16

    def test_load_tensor(self, gguf_path):
        with open_gguf(gguf_path) as gguf:
            t = load_tensor(gguf, "test_tensor", device="cpu")
            assert t.dtype == torch.float16
            assert t.shape == (4,)
            expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
            assert torch.allclose(t, expected)


class TestGgufQ40File:
    """Test parsing and dequantising a Q4_0 GGUF file."""

    @pytest.fixture
    def gguf_path(self):
        """Create a GGUF file with a Q4_0 tensor (one block = 32 elements)."""
        # Make known Q4_0 data: scale=1.0, alternating nibbles
        d = 1.0
        scale_bytes = struct.pack("<e", d)
        # 16 bytes of packed nibbles: [low=9, high=9] repeated → value=1.0
        packed = b"\x99" * 16
        q4_data = scale_bytes + packed  # 18 bytes

        metadata = {
            "general.architecture": (8, "test_q4"),
            "test.block_count": (10, 1),
        }
        tensors = [
            ("tensor_q4", [32], GGML_TYPE_Q4_0, q4_data),
        ]
        content = _create_minimal_gguf_bytes(metadata, tensors, version=3)

        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(content)
        tmp.close()
        yield tmp.name
        os.unlink(tmp.name)

    def test_load_and_dequant(self, gguf_path):
        with open_gguf(gguf_path) as gguf:
            assert gguf.tensors["tensor_q4"].ggml_type == GGML_TYPE_Q4_0
            t = load_tensor(gguf, "tensor_q4", device="cpu")
            assert t.shape == (32,)
            assert t.dtype == torch.float16
            # nibble=9 ⇒ value=9→ (9-8)=1.0 × scale=1.0 ⇒ 1.0
            expected = torch.ones(32, dtype=torch.float16)
            assert torch.allclose(t, expected, atol=0.1)


# ====================================================================
# Edge cases
# ====================================================================


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            open_gguf("/nonexistent/path.gguf")

    def test_not_a_gguf(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(b"This is not a GGUF file")
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Not a GGUF file"):
                open_gguf(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_missing_tensor(self):
        """load_tensor with a non-existent name should raise KeyError."""
        # Create minimal valid GGUF
        metadata = {"general.architecture": (8, "test")}
        f16_data = struct.pack("<ee", 1.0, 2.0)
        tensors = [("a", [2], GGML_TYPE_F16, f16_data)]
        content = _create_minimal_gguf_bytes(metadata, tensors)
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(content)
        tmp.close()
        try:
            with open_gguf(tmp.name) as gguf, pytest.raises(KeyError):
                load_tensor(gguf, "nonexistent_tensor")
        finally:
            os.unlink(tmp.name)

    def test_unsupported_ggml_type(self):
        """Unsupported GGML type should raise NotImplementedError."""
        q3_k = 11
        metadata = {"general.architecture": (8, "test")}
        # 1 block of Q3_K: 32 elements
        data = b"\x00" * 32  # dummy
        tensors = [("x", [32], q3_k, data)]
        content = _create_minimal_gguf_bytes(metadata, tensors)
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(content)
        tmp.close()
        try:
            with open_gguf(tmp.name) as gguf, pytest.raises(NotImplementedError):
                load_tensor(gguf, "x", device="cpu")
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
