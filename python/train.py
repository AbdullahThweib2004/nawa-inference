"""Trains a small MLP on MNIST and saves its PyTorch state_dict.

    python python/train.py            # from the repository root

Model: 784 -> Linear(128) -> ReLU -> Linear(10). The network outputs raw scores (logits);
CrossEntropyLoss applies log-softmax internally, so there is no Softmax layer here. The
exporter appends one for inference.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "mnist_mlp.pt"

# Standard MNIST statistics: mean and std of all training pixels after scaling to [0, 1].
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
SEED = 42


class MnistMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)  # {batch, 1, 28, 28} -> {batch, 784}
        return self.linear2(self.relu(self.linear1(x)))


def mnist_transform() -> transforms.Compose:
    # ToTensor: uint8 0..255 -> float 0..1. Normalize: (x - mean) / std.
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seeds(SEED)
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform())
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=mnist_transform())
    # A seeded generator makes the shuffling order reproducible.
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(test_set, batch_size=1000)

    model = MnistMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"epoch {epoch}/{args.epochs}  loss {total_loss / len(train_set):.4f}")

    accuracy = evaluate(model, test_loader)
    print(f"test accuracy: {accuracy * 100:.2f}% ({len(test_set)} images)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
