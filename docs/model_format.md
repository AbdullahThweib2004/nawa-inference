# Nawa binary formats

This document is the contract between the Python exporter (`python/nawa_format.py`) and the
C++ loader. Both sides must follow it exactly. Any change to the layout requires a new format
version.

There are two file types:

| File         | Extension | Magic  | Purpose                                        |
|--------------|-----------|--------|------------------------------------------------|
| Model file   | `.nawa`   | `NAWA` | A complete network: metadata + layers          |
| Tensor file  | `.ntsr`   | `NTSR` | A single tensor (test fixtures, debugging)     |

## General rules

- **Byte order:** every integer and float is **little-endian**.
- **No padding, no alignment.** Fields follow each other directly. The offsets below are
  exact byte positions.
- **Types:**

  | Name  | Size | Meaning                            |
  |-------|------|------------------------------------|
  | `u8`  | 1    | unsigned 8-bit integer             |
  | `u32` | 4    | unsigned 32-bit integer            |
  | `u64` | 8    | unsigned 64-bit integer            |
  | `i32` | 4    | signed 32-bit integer (two's complement) |
  | `f32` | 4    | IEEE-754 binary32 float            |

- **Magic** values are 4 ASCII bytes with no terminator (for example, `NAWA` is the bytes
  `4E 41 57 41`).
- A file must end exactly where its last field ends. **Trailing bytes are an error.**

## Tensor block

This is the building block for every tensor in both file types. It matches the C++
`Tensor`: float32, contiguous, row-major.

| Offset         | Type          | Field  | Constraints                                   |
|----------------|---------------|--------|-----------------------------------------------|
| 0              | `u32`         | `ndim` | 0 ≤ ndim ≤ 8 (0 means a scalar)               |
| 4              | `u64` × ndim  | `dims` | every dim ≥ 1 (zero-sized dims are rejected)  |
| 4 + 8·ndim     | `f32` × numel | `data` | row-major (C order)                           |

`numel` is the product of `dims`, or 1 when `ndim` is 0. The total block size is
`4 + 8·ndim + 4·numel` bytes.

Example: the 2×3 tensor `[[1, 2, 3], [4, 5, 6]]` takes 4 + 16 + 24 = 44 bytes:

```
02 00 00 00                              ndim = 2
02 00 00 00 00 00 00 00                  dims[0] = 2
03 00 00 00 00 00 00 00                  dims[1] = 3
00 00 80 3F  00 00 00 40  00 00 40 40    1.0, 2.0, 3.0
00 00 80 40  00 00 A0 40  00 00 C0 40    4.0, 5.0, 6.0
```

## Tensor file (`.ntsr`)

| Offset | Type         | Field     | Value / constraint |
|--------|--------------|-----------|--------------------|
| 0      | 4 bytes      | `magic`   | `NTSR`             |
| 4      | `u32`        | `version` | 1                  |
| 8      | tensor block | `tensor`  |                    |

The file ends exactly after the tensor block.

## Model file (`.nawa`)

### Header and metadata

| Offset   | Type                     | Field         | Value / constraint                               |
|----------|--------------------------|---------------|--------------------------------------------------|
| 0        | 4 bytes                  | `magic`       | `NAWA`                                           |
| 4        | `u32`                    | `version`     | 1                                                |
| 8        | `u32`                    | `input_ndim`  | 1 ≤ input_ndim ≤ 8                               |
| 12       | `u64` × input_ndim       | `input_dims`  | shape of ONE sample, without the batch dimension; every dim ≥ 1 |
| …        | `f32`                    | `pixel_scale` | finite and > 0                                   |
| …        | `u32`                    | `norm_count`  | ≥ 1: number of mean/std pairs (channels)         |
| …        | `f32` × norm_count       | `mean`        | finite                                           |
| …        | `f32` × norm_count       | `std`         | finite and > 0                                   |
| …        | `u32`                    | `num_layers`  | ≥ 1                                              |
| …        | layer × num_layers       | `layers`      | see below                                        |

The file ends exactly after the last layer.

**Normalization.** Input is raw pixel values (for MNIST, 0..255). The engine must
normalize every input value before the first layer:

```
x = (raw * pixel_scale - mean[c]) / std[c]
```

`c` is the channel of the value. When `norm_count` is 1 (MNIST, a single grayscale
channel), the same mean and std apply to every value. This matches PyTorch's
`ToTensor()` (divide by 255) followed by `Normalize(mean, std)`.

The MNIST model stores `input_dims = {784}` (a flattened 28×28 image),
`pixel_scale = 1/255`, `mean = {0.1307}` and `std = {0.3081}`.

### Layers

Each layer is a `u32` type id followed by that layer's payload:

| Type id | Layer   | Payload                                   |
|---------|---------|-------------------------------------------|
| 1       | Linear  | see below                                 |
| 2       | ReLU    | none                                      |
| 3       | Sigmoid | none                                      |
| 4       | Softmax | `i32 axis` (negative counts from the end) |

Any other type id is an error.

**Linear payload:**

| Type         | Field      | Constraint                               |
|--------------|------------|------------------------------------------|
| `u8`         | `has_bias` | 0 or 1                                   |
| tensor block | `weight`   | 2-D, shape `{in_features, out_features}` |
| tensor block | `bias`     | only if `has_bias` = 1; shape `{out_features}` |

> **The weight is the transpose of PyTorch's `nn.Linear.weight`**, which has shape
> `{out_features, in_features}`. The exporter writes `weight.T`, so the engine computes
> `y = x · W + b` with a plain matmul.

### Example layout: the MNIST model

```
"NAWA"  u32 1                               header                     8 bytes
u32 1  u64 784                              input_dims = {784}        12 bytes
f32 1/255  u32 1  f32 0.1307  f32 0.3081    normalization             16 bytes
u32 4                                       num_layers                 4 bytes
u32 1  u8 1  block{784,128}  block{128}     Linear(784 -> 128)
u32 2                                       ReLU
u32 1  u8 1  block{128,10}   block{10}      Linear(128 -> 10)
u32 4  i32 -1                               Softmax(axis=-1)
```

## Validation rules for readers

A reader must reject a file (Python: `NawaFormatError`; C++: an exception) when any of these
hold:

1. The magic is wrong, or the version is not a supported version.
2. The file ends before a field is complete (truncated file).
3. A tensor block has `ndim` > 8, a dim of 0, or a `numel` / byte count that overflows.
4. A metadata constraint above is violated (for example, `std` ≤ 0 or `num_layers` = 0).
5. A layer type id is unknown, or `has_bias` is not 0 or 1.
6. A Linear weight is not 2-D, or its bias shape is not `{out_features}`.
7. There are bytes left over after the last field.

Whether the layers chain together (each layer's input size matching the previous layer's
output size, and the first one matching `input_dims`) is checked by the model runtime, not
by the file reader.

## Version history

| Version | Changes         |
|---------|-----------------|
| 1       | Initial format. |
