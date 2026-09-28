"""Reader and writer for the Nawa binary formats (see docs/model_format.md).

This module depends only on numpy (no torch), so it can be used by the verification script
and in CI without installing PyTorch.

Tensors are numpy arrays of dtype float32. Every read is validated; malformed input raises
NawaFormatError.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

MODEL_MAGIC = b"NAWA"
TENSOR_MAGIC = b"NTSR"
MODEL_VERSION = 1          # float32 models (layer types 1-4)
MODEL_VERSION_INT8 = 2     # adds layer type 5 (LinearInt8)
SUPPORTED_MODEL_VERSIONS = (MODEL_VERSION, MODEL_VERSION_INT8)
TENSOR_VERSION = 1
MAX_NDIM = 8

LAYER_LINEAR = 1
LAYER_RELU = 2
LAYER_SIGMOID = 3
LAYER_SOFTMAX = 4
LAYER_LINEAR_INT8 = 5
INT8_MAX_IN_FEATURES = 2147483647 // (127 * 127)  # int32 accumulation must not overflow

# "<" = little-endian, no padding. These match the type table in the spec.
_U8 = struct.Struct("<B")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")
_F32_DTYPE = np.dtype("<f4")


class NawaFormatError(ValueError):
    """Raised when a file or object does not follow docs/model_format.md."""


# ---------------------------------------------------------------------------
# Model description
# ---------------------------------------------------------------------------


@dataclass
class Normalization:
    """x = (raw * pixel_scale - mean[c]) / std[c]"""

    pixel_scale: float
    mean: list[float]
    std: list[float]


@dataclass
class Linear:
    weight: np.ndarray  # {in_features, out_features} -- transposed from PyTorch
    bias: np.ndarray | None = None  # {out_features}


@dataclass
class LinearInt8:
    """Per-output-channel symmetric int8 weights (format version 2)."""

    weight: np.ndarray  # int8 {out_features, in_features}, values in -127..127
    scales: np.ndarray  # float32 {out_features}, > 0: weight[j, k] * scales[j] ~ original
    bias: np.ndarray | None = None  # float32 {out_features}


@dataclass
class ReLU:
    pass


@dataclass
class Sigmoid:
    pass


@dataclass
class Softmax:
    axis: int = -1


Layer = Union[Linear, LinearInt8, ReLU, Sigmoid, Softmax]


@dataclass
class Model:
    input_shape: tuple[int, ...]  # one sample, without the batch dimension
    normalization: Normalization
    layers: list[Layer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level reading
# ---------------------------------------------------------------------------


class _Reader:
    """Reads fields from a bytes object, failing cleanly on truncated input."""

    def __init__(self, data: bytes):
        self._data = memoryview(data)
        self.pos = 0

    def take(self, n: int, what: str) -> memoryview:
        if n > len(self._data) - self.pos:
            raise NawaFormatError(
                f"truncated file: need {n} bytes for {what} at offset {self.pos}, "
                f"but only {len(self._data) - self.pos} remain"
            )
        chunk = self._data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def _unpack(self, s: struct.Struct, what: str):
        return s.unpack(self.take(s.size, what))[0]

    def u8(self, what: str) -> int:
        return self._unpack(_U8, what)

    def u32(self, what: str) -> int:
        return self._unpack(_U32, what)

    def u64(self, what: str) -> int:
        return self._unpack(_U64, what)

    def i32(self, what: str) -> int:
        return self._unpack(_I32, what)

    def f32(self, what: str) -> float:
        return self._unpack(_F32, what)

    def expect_end(self) -> None:
        extra = len(self._data) - self.pos
        if extra:
            raise NawaFormatError(f"{extra} unexpected trailing bytes at offset {self.pos}")


# ---------------------------------------------------------------------------
# Tensor blocks
# ---------------------------------------------------------------------------


def _check_shape(shape: tuple[int, ...], what: str) -> None:
    if len(shape) > MAX_NDIM:
        raise NawaFormatError(f"{what}: ndim {len(shape)} exceeds the maximum of {MAX_NDIM}")
    if any(d < 1 for d in shape):
        raise NawaFormatError(f"{what}: every dimension must be >= 1, got shape {list(shape)}")


def encode_tensor_block(array: np.ndarray) -> bytes:
    """Serializes an array as a tensor block (converted to contiguous little-endian f32)."""
    # Not np.ascontiguousarray: it turns a 0-d array (scalar, shape ()) into shape (1,).
    arr = np.asarray(array, dtype=_F32_DTYPE, order="C")
    _check_shape(arr.shape, "tensor")
    header = _U32.pack(arr.ndim) + b"".join(_U64.pack(d) for d in arr.shape)
    return header + arr.tobytes(order="C")


def _read_tensor_block(r: _Reader, what: str) -> np.ndarray:
    ndim = r.u32(f"{what} ndim")
    if ndim > MAX_NDIM:
        raise NawaFormatError(f"{what}: ndim {ndim} exceeds the maximum of {MAX_NDIM}")
    shape = tuple(r.u64(f"{what} dims[{i}]") for i in range(ndim))
    _check_shape(shape, what)
    numel = math.prod(shape)  # Python ints don't overflow; the byte count is checked by take()
    raw = r.take(4 * numel, f"{what} data ({numel} floats)")
    # .copy() gives an independent, writable, native-endian array.
    return np.frombuffer(raw, dtype=_F32_DTYPE).reshape(shape).astype(np.float32, copy=True)


# ---------------------------------------------------------------------------
# Tensor files (.ntsr)
# ---------------------------------------------------------------------------


def encode_tensor_file(array: np.ndarray) -> bytes:
    return TENSOR_MAGIC + _U32.pack(TENSOR_VERSION) + encode_tensor_block(array)


def decode_tensor_file(data: bytes) -> np.ndarray:
    r = _Reader(data)
    magic = bytes(r.take(4, "magic"))
    if magic != TENSOR_MAGIC:
        raise NawaFormatError(f"bad magic {magic!r}, expected {TENSOR_MAGIC!r}")
    version = r.u32("version")
    if version != TENSOR_VERSION:
        raise NawaFormatError(f"unsupported tensor file version {version}")
    tensor = _read_tensor_block(r, "tensor")
    r.expect_end()
    return tensor


def save_tensor(path: str | Path, array: np.ndarray) -> None:
    Path(path).write_bytes(encode_tensor_file(array))


def load_tensor(path: str | Path) -> np.ndarray:
    return decode_tensor_file(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Model files (.nawa)
# ---------------------------------------------------------------------------


def _validate_normalization(n: Normalization) -> None:
    if len(n.mean) < 1 or len(n.mean) != len(n.std):
        raise NawaFormatError(
            f"normalization needs >= 1 mean/std pairs of equal length, "
            f"got {len(n.mean)} means and {len(n.std)} stds"
        )
    if not (math.isfinite(n.pixel_scale) and n.pixel_scale > 0):
        raise NawaFormatError(f"pixel_scale must be finite and > 0, got {n.pixel_scale}")
    if not all(math.isfinite(m) for m in n.mean):
        raise NawaFormatError(f"mean values must be finite, got {n.mean}")
    if not all(math.isfinite(s) and s > 0 for s in n.std):
        raise NawaFormatError(f"std values must be finite and > 0, got {n.std}")


def _validate_linear(weight: np.ndarray, bias: np.ndarray | None, index: int) -> None:
    if weight.ndim != 2:
        raise NawaFormatError(
            f"layer {index} (Linear): weight must be 2-D {{in, out}}, got shape {list(weight.shape)}"
        )
    if bias is not None and bias.shape != (weight.shape[1],):
        raise NawaFormatError(
            f"layer {index} (Linear): bias shape {list(bias.shape)} does not match "
            f"weight {list(weight.shape)}; expected [{weight.shape[1]}]"
        )


def _validate_linear_int8(layer: LinearInt8, index: int) -> None:
    w, scales = layer.weight, layer.scales
    if w.ndim != 2 or w.dtype != np.int8:
        raise NawaFormatError(f"layer {index} (LinearInt8): weight must be a 2-D int8 array")
    out_features, in_features = w.shape
    if not 1 <= in_features <= INT8_MAX_IN_FEATURES or out_features < 1:
        raise NawaFormatError(f"layer {index} (LinearInt8): bad shape {list(w.shape)}")
    if (w == -128).any():
        raise NawaFormatError(f"layer {index} (LinearInt8): weights must be in -127..127")
    if scales.shape != (out_features,) or not (np.isfinite(scales).all() and (scales > 0).all()):
        raise NawaFormatError(f"layer {index} (LinearInt8): need {out_features} finite scales > 0")
    if layer.bias is not None and layer.bias.shape != (out_features,):
        raise NawaFormatError(f"layer {index} (LinearInt8): bias shape {list(layer.bias.shape)} "
                              f"does not match out_features {out_features}")


def encode_model(model: Model) -> bytes:
    shape = tuple(int(d) for d in model.input_shape)
    if len(shape) < 1:
        raise NawaFormatError("input_shape must have at least one dimension")
    _check_shape(shape, "input_shape")
    norm = model.normalization
    _validate_normalization(norm)
    if not model.layers:
        raise NawaFormatError("a model needs at least one layer")

    # Version 2 only when needed, so float32 models stay byte-identical to version 1 files.
    version = MODEL_VERSION_INT8 if any(isinstance(l, LinearInt8) for l in model.layers) else MODEL_VERSION
    parts = [MODEL_MAGIC, _U32.pack(version), _U32.pack(len(shape))]
    parts += [_U64.pack(d) for d in shape]
    parts.append(_F32.pack(norm.pixel_scale))
    parts.append(_U32.pack(len(norm.mean)))
    parts += [_F32.pack(m) for m in norm.mean]
    parts += [_F32.pack(s) for s in norm.std]
    parts.append(_U32.pack(len(model.layers)))

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            weight = np.asarray(layer.weight, dtype=np.float32)
            bias = None if layer.bias is None else np.asarray(layer.bias, dtype=np.float32)
            _validate_linear(weight, bias, i)
            parts += [_U32.pack(LAYER_LINEAR), _U8.pack(0 if bias is None else 1)]
            parts.append(encode_tensor_block(weight))
            if bias is not None:
                parts.append(encode_tensor_block(bias))
        elif isinstance(layer, LinearInt8):
            weight = np.asarray(layer.weight)
            scales = np.asarray(layer.scales, dtype=np.float32)
            bias = None if layer.bias is None else np.asarray(layer.bias, dtype=np.float32)
            _validate_linear_int8(LinearInt8(weight, scales, bias), i)
            out_features, in_features = weight.shape
            parts += [_U32.pack(LAYER_LINEAR_INT8), _U8.pack(0 if bias is None else 1),
                      _U32.pack(in_features), _U32.pack(out_features),
                      np.ascontiguousarray(scales, dtype=_F32_DTYPE).tobytes(),
                      np.ascontiguousarray(weight, dtype=np.int8).tobytes()]
            if bias is not None:
                parts.append(encode_tensor_block(bias))
        elif isinstance(layer, ReLU):
            parts.append(_U32.pack(LAYER_RELU))
        elif isinstance(layer, Sigmoid):
            parts.append(_U32.pack(LAYER_SIGMOID))
        elif isinstance(layer, Softmax):
            parts += [_U32.pack(LAYER_SOFTMAX), _I32.pack(layer.axis)]
        else:
            raise NawaFormatError(f"layer {i}: unsupported layer type {type(layer).__name__}")
    return b"".join(parts)


def decode_model(data: bytes) -> Model:
    r = _Reader(data)
    magic = bytes(r.take(4, "magic"))
    if magic != MODEL_MAGIC:
        raise NawaFormatError(f"bad magic {magic!r}, expected {MODEL_MAGIC!r}")
    version = r.u32("version")
    if version not in SUPPORTED_MODEL_VERSIONS:
        raise NawaFormatError(f"unsupported model format version {version}")

    input_ndim = r.u32("input_ndim")
    if not 1 <= input_ndim <= MAX_NDIM:
        raise NawaFormatError(f"input_ndim must be in 1..{MAX_NDIM}, got {input_ndim}")
    input_shape = tuple(r.u64(f"input_dims[{i}]") for i in range(input_ndim))
    _check_shape(input_shape, "input_shape")

    pixel_scale = r.f32("pixel_scale")
    norm_count = r.u32("norm_count")
    if norm_count < 1:
        raise NawaFormatError("norm_count must be >= 1")
    # Each value is 4 bytes; checking up front avoids looping over a huge bogus count.
    if 8 * norm_count > len(data) - r.pos:
        raise NawaFormatError(f"truncated file: norm_count {norm_count} exceeds the file size")
    mean = [r.f32(f"mean[{i}]") for i in range(norm_count)]
    std = [r.f32(f"std[{i}]") for i in range(norm_count)]
    normalization = Normalization(pixel_scale, mean, std)
    _validate_normalization(normalization)

    num_layers = r.u32("num_layers")
    if num_layers < 1:
        raise NawaFormatError("num_layers must be >= 1")

    layers: list[Layer] = []
    for i in range(num_layers):
        type_id = r.u32(f"layer {i} type id")
        if type_id == LAYER_LINEAR:
            has_bias = r.u8(f"layer {i} has_bias")
            if has_bias not in (0, 1):
                raise NawaFormatError(f"layer {i} (Linear): has_bias must be 0 or 1, got {has_bias}")
            weight = _read_tensor_block(r, f"layer {i} weight")
            bias = _read_tensor_block(r, f"layer {i} bias") if has_bias else None
            _validate_linear(weight, bias, i)
            layers.append(Linear(weight, bias))
        elif type_id == LAYER_LINEAR_INT8:
            if version < MODEL_VERSION_INT8:
                raise NawaFormatError(f"layer {i}: layer type 5 (LinearInt8) requires format "
                                      f"version 2, found it in a version {version} file")
            has_bias = r.u8(f"layer {i} has_bias")
            if has_bias not in (0, 1):
                raise NawaFormatError(f"layer {i} (LinearInt8): has_bias must be 0 or 1, got {has_bias}")
            in_features = r.u32(f"layer {i} in_features")
            out_features = r.u32(f"layer {i} out_features")
            if not 1 <= in_features <= INT8_MAX_IN_FEATURES or out_features < 1:
                raise NawaFormatError(f"layer {i} (LinearInt8): bad shape {out_features} x {in_features}")
            scales = np.frombuffer(r.take(4 * out_features, f"layer {i} scales"), dtype=_F32_DTYPE)
            weight = np.frombuffer(r.take(in_features * out_features, f"layer {i} int8 weights"),
                                   dtype=np.int8).reshape(out_features, in_features)
            bias = _read_tensor_block(r, f"layer {i} bias") if has_bias else None
            layer = LinearInt8(weight.copy(), scales.astype(np.float32), bias)
            _validate_linear_int8(layer, i)
            layers.append(layer)
        elif type_id == LAYER_RELU:
            layers.append(ReLU())
        elif type_id == LAYER_SIGMOID:
            layers.append(Sigmoid())
        elif type_id == LAYER_SOFTMAX:
            layers.append(Softmax(r.i32(f"layer {i} axis")))
        else:
            raise NawaFormatError(f"layer {i}: unknown layer type id {type_id}")
    r.expect_end()
    return Model(input_shape, normalization, layers)


def save_model(path: str | Path, model: Model) -> None:
    Path(path).write_bytes(encode_model(model))


def load_model(path: str | Path) -> Model:
    return decode_model(Path(path).read_bytes())
