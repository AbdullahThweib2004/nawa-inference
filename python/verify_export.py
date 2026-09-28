"""Checks the exported model with a pure numpy forward pass (no torch needed).

    python python/verify_export.py    # from the repository root

Reads models/mnist_mlp.nawa and tests/fixtures/ with nawa_format.py only, runs the model
exactly as the C++ engine will (normalize, then each layer in order), and compares against
PyTorch's saved outputs. Exits with status 1 on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import nawa_format as nf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mnist_mlp.nawa"
FIXTURES = ROOT / "tests" / "fixtures"
TOLERANCE = 1e-5
# Per-layer names, matching the order of the layers in the model and the fixture names.
LAYER_FIXTURES = ["linear1", "relu", "linear2", "softmax"]


def normalize(raw: np.ndarray, norm: nf.Normalization) -> np.ndarray:
    # Single-channel model: one mean/std for every value.
    scale = np.float32(norm.pixel_scale)
    return (raw * scale - np.float32(norm.mean[0])) / np.float32(norm.std[0])


def forward_layer(layer: nf.Layer, x: np.ndarray) -> np.ndarray:
    if isinstance(layer, nf.Linear):
        y = x @ layer.weight  # {batch, in} @ {in, out}
        return y + layer.bias if layer.bias is not None else y
    if isinstance(layer, nf.ReLU):
        return np.maximum(x, np.float32(0))
    if isinstance(layer, nf.Sigmoid):
        return 1.0 / (1.0 + np.exp(-x))
    if isinstance(layer, nf.Softmax):
        e = np.exp(x - x.max(axis=layer.axis, keepdims=True))  # stable softmax
        return e / e.sum(axis=layer.axis, keepdims=True)
    raise TypeError(f"unknown layer {layer!r}")


def main() -> int:
    model = nf.load_model(MODEL_PATH)
    if len(model.normalization.mean) != 1:
        print("FAIL: this script only supports single-channel normalization")
        return 1
    images = nf.load_tensor(FIXTURES / "mnist_test100_images.ntsr")
    labels = nf.load_tensor(FIXTURES / "mnist_test100_labels.ntsr").astype(np.int64)
    expected = nf.load_tensor(FIXTURES / "mnist_test100_probs.ntsr")

    print(f"model: input {list(model.input_shape)}, {len(model.layers)} layers")
    for layer in model.layers:
        extra = f" {list(layer.weight.shape)}" if isinstance(layer, nf.Linear) else ""
        print(f"  {type(layer).__name__}{extra}")

    ok = True

    # Layer by layer on the first 3 images: pinpoints WHERE a mismatch starts.
    #
    # Intermediate values are not bounded (linear1 reaches ~20), and float32 has a fixed
    # number of significant digits, so its absolute rounding error grows with the value:
    # at 20 one float32 step is already ~2e-6, and a 784-term sum accumulated in a different
    # order than PyTorch's differs by a few steps. The per-stage check therefore scales
    # the error by the layer's magnitude: max_abs_err / max(1, max|expected|).
    print("first 3 images, per stage:  max abs error  (scaled)")
    x = normalize(images[:3], model.normalization)
    stages = [("normalize", None)] + list(zip(LAYER_FIXTURES, model.layers, strict=True))
    for name, layer in stages:
        if layer is not None:
            x = forward_layer(layer, x)
        fixture = "input_normalized" if layer is None else name
        ref = nf.load_tensor(FIXTURES / f"mnist_test3_{fixture}.ntsr")
        err = float(np.abs(x - ref).max())
        scaled = err / max(1.0, float(np.abs(ref).max()))
        print(f"  {name:<10}                {err:9.3g}  ({scaled:.3g})")
        ok &= scaled < TOLERANCE

    # All 100 images end to end.
    x = normalize(images, model.normalization)
    for layer in model.layers:
        x = forward_layer(layer, x)
    probs = x
    max_error = float(np.abs(probs - expected).max())
    same_argmax = bool((probs.argmax(axis=1) == expected.argmax(axis=1)).all())
    accuracy = float((probs.argmax(axis=1) == labels).mean())

    print(f"100 images: max abs error {max_error:.3g} (tolerance {TOLERANCE:g})")
    print(f"100 images: argmax matches PyTorch: {same_argmax}")
    print(f"100 images: accuracy vs labels: {accuracy * 100:.0f}%")

    ok &= max_error < TOLERANCE and same_argmax
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
