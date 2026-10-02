"""Exports the trained PyTorch model to the Nawa format and writes C++ test fixtures.

    python python/export.py           # from the repository root, after train.py

Outputs:
  models/mnist_mlp.nawa                       the model (docs/model_format.md)
  tests/fixtures/mnist_test100_images.ntsr    {100, 784} RAW pixels 0..255 as float32
  tests/fixtures/mnist_test100_labels.ntsr    {100} labels as float32
  tests/fixtures/mnist_test100_probs.ntsr     {100, 10} PyTorch probabilities
  tests/fixtures/mnist_test3_*.ntsr           per-layer outputs for the first 3 images
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

import nawa_format as nf
from train import DATA_DIR, MNIST_MEAN, MNIST_STD, MODEL_PATH, ROOT, MnistMLP

NAWA_PATH = ROOT / "models" / "mnist_mlp.nawa"
FIXTURES = ROOT / "tests" / "fixtures"
NUM_IMAGES = 100
NUM_LAYER_DEBUG = 3
PIXEL_SCALE = np.float32(1.0 / 255.0)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def build_nawa_model(model: MnistMLP) -> nf.Model:
    def linear(layer: torch.nn.Linear) -> nf.Linear:
        # PyTorch stores {out, in}; our engine expects {in, out}, so transpose.sss
        return nf.Linear(weight=to_numpy(layer.weight.T), bias=to_numpy(layer.bias))

    return nf.Model(
        input_shape=(784,),
        normalization=nf.Normalization(float(PIXEL_SCALE), [MNIST_MEAN], [MNIST_STD]),
        layers=[
            linear(model.linear1),
            nf.ReLU(),
            linear(model.linear2),
            # Training used CrossEntropyLoss on raw logits; inference wants probabilities.
            nf.Softmax(axis=-1),
        ],
    )


def normalize(raw: np.ndarray) -> np.ndarray:
    # Exactly what the spec says the engine must do, in float32:
    # x = (raw * pixel_scale - mean) / std
    return (raw * PIXEL_SCALE - np.float32(MNIST_MEAN)) / np.float32(MNIST_STD)


def main() -> None:
    model = MnistMLP()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    nawa_model = build_nawa_model(model)
    nf.save_model(NAWA_PATH, nawa_model)
    print(f"wrote {NAWA_PATH.relative_to(ROOT)} ({NAWA_PATH.stat().st_size:,} bytes)")

    # Raw test images, no transform: uint8 {10000, 28, 28} -> float32 {100, 784}.
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
    raw = test_set.data[:NUM_IMAGES].reshape(NUM_IMAGES, 784).numpy().astype(np.float32)
    labels = test_set.targets[:NUM_IMAGES].numpy().astype(np.float32)

    x = torch.from_numpy(normalize(raw))
    with torch.no_grad():
        linear1 = model.linear1(x)
        relu = model.relu(linear1)
        linear2 = model.linear2(relu)
        probs = torch.softmax(linear2, dim=-1)

    fixtures = {
        "mnist_test100_images": raw,
        "mnist_test100_labels": labels,
        "mnist_test100_probs": to_numpy(probs),
        "mnist_test3_input_normalized": to_numpy(x[:NUM_LAYER_DEBUG]),
        "mnist_test3_linear1": to_numpy(linear1[:NUM_LAYER_DEBUG]),
        "mnist_test3_relu": to_numpy(relu[:NUM_LAYER_DEBUG]),
        "mnist_test3_linear2": to_numpy(linear2[:NUM_LAYER_DEBUG]),
        "mnist_test3_softmax": to_numpy(probs[:NUM_LAYER_DEBUG]),
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, array in fixtures.items():
        path = FIXTURES / f"{name}.ntsr"
        nf.save_tensor(path, array)
        print(f"wrote {path.relative_to(ROOT)} {list(array.shape)} ({path.stat().st_size:,} bytes)")

    correct = int((probs.argmax(dim=1).numpy() == labels).sum())
    print(f"PyTorch accuracy on these {NUM_IMAGES} images: {correct}/{NUM_IMAGES}")


if __name__ == "__main__":
    main()
