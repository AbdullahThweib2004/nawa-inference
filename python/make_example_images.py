"""Creates the demo images in examples/images/ from MNIST test samples.

    python python/make_example_images.py     # needs data/MNIST/raw (run train.py once)

Three versions show what the C++ preprocessing (include/inference/data/digit_preprocess.hpp)
has to handle:
  digit7_mnist.png      test image 0 exactly as stored in MNIST (28x28, white on black)
  digit2_inverted.png   test image 1, enlarged to 112x112 and inverted to black on white,
                        like a digit drawn with a dark pen on paper
  digit4_offcenter.png  test image 4, small and near the top-left corner of a 120x120 canvas
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MNIST" / "raw"
OUT = ROOT / "examples" / "images"


def load_test_images() -> np.ndarray:
    data = (RAW / "t10k-images-idx3-ubyte").read_bytes()
    # IDX header: 4-byte magic, then big-endian u32 count, rows, cols.
    count, rows, cols = np.frombuffer(data, dtype=">u4", count=3, offset=4)
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def main() -> None:
    images = load_test_images()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. As-is.
    Image.fromarray(images[0]).save(OUT / "digit7_mnist.png")

    # 2. Enlarged and inverted: dark digit on a light background.
    big = Image.fromarray(images[1]).resize((112, 112), Image.Resampling.BILINEAR)
    Image.fromarray(255 - np.asarray(big)).save(OUT / "digit2_inverted.png")

    # 3. Small and off-center: crop the digit to its bounding box, shrink it to 18 px tall
    #    (keeping the aspect ratio), and paste it near the top-left of a 120x120 canvas.
    digit = images[4]
    ys, xs = np.nonzero(digit)
    crop = Image.fromarray(digit[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    height = 18
    width = max(1, round(crop.width * height / crop.height))
    small = crop.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (120, 120), 0)
    canvas.paste(small, (10, 14))
    canvas.save(OUT / "digit4_offcenter.png")

    for path in sorted(OUT.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
