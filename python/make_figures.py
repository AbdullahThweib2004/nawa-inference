"""Figures for the README and posts, made from the project's real data.

    python python/make_figures.py        # from the repo root, with .venv active

Needs matplotlib (python/requirements-dev.txt) and, for confident_wrong.png, build/bin/nawa.
Every figure is 1600 px wide and saved twice in docs/images/: NAME.png (light) and NAME_dark.png.

    speedup_chart     % of CPU peak per optimization step   <- benchmarks/results/*.json
    neuron_maps       weights of 16 hidden neurons          <- models/mnist_mlp.nawa
    architecture      Python training -> .nawa -> C++ engine -> web demo
    confident_wrong   an inverted "2" the model calls a 7   <- nawa predict --json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nawa_format as nf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "images"
NAWA = ROOT / "build" / "bin" / "nawa"
MODEL = ROOT / "models" / "mnist_mlp.nawa"
INVERTED = ROOT / "examples" / "images" / "digit2_inverted.png"

WIDTH_IN = 16  # at DPI 100 -> 1600 px
DPI = 100
PEAK_FLOP_PER_CYCLE = 32  # per core: 2 FMA units x 8 floats x 2 FLOP (AVX2, i7-11370H)

THEMES = {
    "light": {
        "bg": "#ffffff", "fg": "#1b1f24", "muted": "#5b6470", "grid": "#e3e6ea",
        "panel": "#f4f6f8", "edge": "#c9ced6", "accent": "#2f6fdf", "accent_soft": "#a9c3f0",
        "bad": "#d6453d", "diverging": ["#1f5fae", "#6fa8dc", "#ffffff", "#f08a75", "#b2182b"],
    },
    "dark": {
        "bg": "#0f1216", "fg": "#e8ebef", "muted": "#9aa3ae", "grid": "#262b33",
        "panel": "#181c22", "edge": "#39404b", "accent": "#5b9bff", "accent_soft": "#2e4a78",
        "bad": "#ff6b61", "diverging": ["#4f9dff", "#28517f", "#12151a", "#8a3a33", "#ff6b5b"],
    },
}


def style(theme: dict) -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"],
        "font.size": 22,
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"],
        "text.color": theme["fg"],
        "axes.labelcolor": theme["fg"],
        "axes.edgecolor": theme["edge"],
        "xtick.color": theme["fg"],
        "ytick.color": theme["muted"],
        "axes.titleweight": "bold",
    })


def save(fig, name: str, scheme: str) -> Path:
    path = OUT / (f"{name}.png" if scheme == "light" else f"{name}_dark.png")
    fig.savefig(path, dpi=DPI)  # no bbox_inches="tight": it would change the 1600 px width
    plt.close(fig)
    return path


def from_top(fig, pixels: float) -> float:
    """Figure y coordinate `pixels` below the top edge, so spacing is the same in every figure."""
    return 1 - pixels / (fig.get_figheight() * DPI)


def title(fig, text: str, subtitle: str | None, theme: dict) -> None:
    fig.text(0.04, from_top(fig, 45), text, fontsize=40, fontweight="bold", va="top",
             color=theme["fg"])
    if subtitle:
        fig.text(0.04, from_top(fig, 118), subtitle, fontsize=22, va="top", color=theme["muted"])


def text_run(fig, x: float, y: float, parts: list[tuple[str, dict]]) -> None:
    """Draws text segments one after another on a line, each with its own style."""
    renderer = fig.canvas.get_renderer()
    for part, kwargs in parts:
        t = fig.text(x, y, part, va="top", **kwargs)
        x += t.get_window_extent(renderer).width / (fig.get_figwidth() * DPI)

# ---------------------------------------------------------------------------
# 1. speedup_chart: % of CPU peak per step
# ---------------------------------------------------------------------------


def medians(name: str) -> dict[str, dict]:
    with open(RESULTS / f"{name}.json") as f:
        return {b["run_name"]: b for b in json.load(f)["benchmarks"]
                if b.get("aggregate_name") == "median"}


def speedup_data() -> tuple[list[tuple[str, str, float]], float]:
    """(label, detail, FLOP/cycle) for the 512x512x512 matmul, plus the 9.5 thread speedup.

    Steps 9.3 and 9.4 use BM_MatmulPacked: pre-packing the weights is how the engine runs a model.
    """
    square = "M:512/K:512/N:512"
    steps = [
        ("Baseline", "naive loops", "baseline", "BM_Matmul"),
        ("9.1", "loop order", "step9_1_loop_order", "BM_Matmul"),
        ("9.2", "-march=native", "step9_2_native", "BM_Matmul"),
        ("9.3", "AVX2 GEMM", "step9_3_blocked_gemm", "BM_MatmulPacked"),
        ("9.4", "memory + fusion", "step9_4_memory_fusion", "BM_MatmulPacked"),
    ]
    rows = []
    for label, detail, file, bench in steps:
        rows.append((label, detail, medians(file)[f"{bench}/{square}"]["FLOP_per_cycle"]))
    threads = medians("step9_5_threads")
    one = threads["BM_GemmThreads/threads:1/M:1024/K:1024/N:1024/real_time"]["real_time"]
    four = threads["BM_GemmThreads/threads:4/M:1024/K:1024/N:1024/real_time"]["real_time"]
    return rows, one / four


def speedup_chart(scheme: str) -> Path:
    theme = THEMES[scheme]
    style(theme)
    rows, thread_speedup = speedup_data()
    pct = [100 * r[2] / PEAK_FLOP_PER_CYCLE for r in rows]

    fig = plt.figure(figsize=(WIDTH_IN, 9.5))
    title(fig, f"From {pct[0]:.1f}% to {pct[-1]:.0f}% of CPU peak",
          "512 × 512 matrix multiply, one core of an Intel i7-11370H (median of the saved runs)",
          theme)
    ax = fig.add_axes([0.07, 0.26, 0.90, 0.52])
    x = np.arange(len(rows))
    colors = [theme["accent_soft"]] * (len(rows) - 1) + [theme["accent"]]
    bars = ax.bar(x, pct, width=0.62, color=colors, zorder=3)
    for i, (bar, value) in enumerate(zip(bars, pct)):
        cx = bar.get_x() + bar.get_width() / 2
        if value > 60:  # tall bars: the value goes inside, clear of the 100% line
            on_soft = i < len(rows) - 1 and scheme == "light"
            ax.text(cx, value - 3, f"{value:.1f}%", ha="center", va="top", fontsize=28,
                    fontweight="bold", color=theme["fg"] if on_soft else "#ffffff")
        else:
            ax.text(cx, value + 2.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=28,
                    fontweight="bold", color=theme["fg"])
    ax.axhline(100, color=theme["muted"], lw=2, ls=(0, (6, 4)), zorder=2)
    ax.text(-0.4, 102, "CPU peak: 32 FLOP/cycle per core", ha="left", va="bottom", fontsize=18,
            color=theme["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=24, fontweight="bold")
    for xi, r in zip(x, rows):
        ax.text(xi, -0.17, r[1], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=21, color=theme["muted"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=18)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=theme["grid"], lw=1.2, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.text(0.04, 0.035,
             f"+ step 9.5, a thread pool: {thread_speedup:.1f}× faster again on 4 cores "
             f"(1024³ GEMM).   Data: benchmarks/results/*.json",
             fontsize=18, color=theme["muted"])
    return save(fig, "speedup_chart", scheme)

# ---------------------------------------------------------------------------
# 2. neuron_maps: what the hidden neurons look for
# ---------------------------------------------------------------------------


def neuron_maps(scheme: str) -> Path:
    theme = THEMES[scheme]
    style(theme)
    model = nf.load_model(MODEL)
    w1 = model.layers[0].weight  # {784, 128}: column j = the weights of hidden neuron j
    variance = w1.var(axis=0)
    chosen = np.argsort(variance)[::-1][:16]
    cmap = LinearSegmentedColormap.from_list("div", theme["diverging"])

    fig = plt.figure(figsize=(WIDTH_IN, 19))
    title(fig, "What the hidden neurons look for",
          "First-layer weights of the 16 neurons with the largest weight variance, as 28 × 28 "
          "images", theme)
    muted = {"fontsize": 22, "color": theme["muted"]}
    text_run(fig, 0.04, from_top(fig, 160), [
        ("Red", {"fontsize": 22, "fontweight": "bold", "color": theme["diverging"][-1]}),
        (": ink here excites the neuron.   ", muted),
        ("Blue", {"fontsize": 22, "fontweight": "bold", "color": theme["diverging"][0]}),
        (": ink here inhibits it.", muted)])

    left, right, top, bottom = 0.05, 0.95, from_top(fig, 230), 0.02
    gap_x, label_h = 0.03, 0.028
    cell_w = (right - left - 3 * gap_x) / 4
    cell_h = cell_w * WIDTH_IN / 19  # square in inches
    gap_y = (top - bottom - 4 * (cell_h + label_h)) / 3
    for i, neuron in enumerate(chosen):
        r, c = divmod(i, 4)
        x0 = left + c * (cell_w + gap_x)
        y0 = top - (r + 1) * (cell_h + label_h) - r * gap_y
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        image = w1[:, neuron].reshape(28, 28)
        limit = np.abs(image).max()  # each map on its own scale, centered at zero
        ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(theme["edge"])
        fig.text(x0 + cell_w / 2, y0 + cell_h + 0.006, f"Neuron {neuron}", ha="center",
                 va="bottom", fontsize=24, fontweight="bold", color=theme["fg"])
    return save(fig, "neuron_maps", scheme)

# ---------------------------------------------------------------------------
# 3. architecture
# ---------------------------------------------------------------------------


def architecture(scheme: str) -> Path:
    theme = THEMES[scheme]
    style(theme)
    fig = plt.figure(figsize=(WIDTH_IN, 6.4))
    title(fig, "How Nawa fits together", "Train in Python, run in C++ built from scratch", theme)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # (x, width, heading, tag, body lines, highlighted)
    boxes = [
        (0.3, 3.0, "Training", "Python · PyTorch", ["MLP on MNIST", "784 → 128 → 10"], False),
        (4.1, 3.3, "mnist_mlp.nawa", "binary file", ["weights +", "normalization"], False),
        (8.2, 4.0, "Nawa engine", "C++20 · no math libs",
         ["tensors", "AVX2 GEMM", "thread pool", "INT8"], True),
        (13.0, 2.8, "Web demo", "HTML · JS", ["draw a digit,", "see the network"], False),
    ]
    y, h = 0.45, 3.9
    for x, w, heading, tag, lines, hot in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", lw=3 if hot else 2,
            facecolor=theme["panel"], edgecolor=theme["accent"] if hot else theme["edge"]))
        cx = x + w / 2
        ax.text(cx, y + h - 0.4, heading, ha="center", va="top", fontsize=26, fontweight="bold",
                color=theme["accent"] if hot else theme["fg"])
        ax.text(cx, y + h - 1.1, tag, ha="center", va="top", fontsize=19, color=theme["muted"])
        if hot:  # two columns of short features
            for i, line in enumerate(lines):
                col, row = divmod(i, 2)
                ax.text(x + w * (0.27 + 0.47 * col), y + 1.6 - row * 0.7, line, ha="center",
                        va="center", fontsize=21, color=theme["fg"])
        else:
            for i, line in enumerate(lines):
                ax.text(cx, y + 1.6 - i * 0.7, line, ha="center", va="center", fontsize=21,
                        color=theme["fg"])
    for (x, w, *_), (nx, *_) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x + w + 0.08, y + h / 2), (nx - 0.08, y + h / 2),
                                     arrowstyle="-|>,head_length=0.35,head_width=0.2",
                                     mutation_scale=40, lw=3, color=theme["muted"]))
    return save(fig, "architecture", scheme)

# ---------------------------------------------------------------------------
# 4. confident_wrong: an inverted 2 predicted as a 7
# ---------------------------------------------------------------------------


def run_predict(preprocess: bool) -> dict:
    if not NAWA.exists():
        sys.exit(f"{NAWA} not found: build first (cmake --build build -j)")
    cmd = [str(NAWA), "predict", str(MODEL), str(INVERTED), "--json"]
    if not preprocess:
        cmd.append("--no-preprocess")
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def confident_wrong(scheme: str, raw: dict, fixed: dict) -> Path:
    theme = THEMES[scheme]
    style(theme)
    probs = np.array(raw["probabilities"]) * 100
    pred = raw["prediction"]
    seen = np.array(raw["input28"]).reshape(28, 28)
    drawn = plt.imread(INVERTED)
    if drawn.ndim == 3:
        drawn = drawn[..., :3].mean(axis=2)

    fig = plt.figure(figsize=(WIDTH_IN, 7.8))
    title(fig, "100% confident. Completely wrong.",
          "A dark “2” on a white background. The network only ever saw white digits on black.",
          theme)
    panels = [(0.04, drawn, "The image: a 2"), (0.275, seen, "What the model saw")]
    for x0, image, label in panels:
        ax = fig.add_axes([x0, 0.17, 0.205, 0.205 * WIDTH_IN / 7.8])
        ax.imshow(image, cmap="gray", vmin=0, vmax=image.max(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(theme["edge"])
            spine.set_linewidth(2)
        ax.set_title(label, fontsize=22, fontweight="bold", color=theme["fg"], pad=12)

    ax = fig.add_axes([0.56, 0.17, 0.41, 0.47])
    colors = [theme["bad"] if d == pred else theme["accent_soft"] for d in range(10)]
    ax.bar(np.arange(10), np.maximum(probs, 0.6), width=0.7, color=colors, zorder=3)
    ax.text(pred, probs[pred] + 3, f"{probs[pred]:.2f}%", ha="center", va="bottom",
            fontsize=26, fontweight="bold", color=theme["bad"])
    ax.set_xticks(np.arange(10), [str(d) for d in range(10)], fontsize=24)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"], fontsize=18)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=theme["grid"], lw=1.2, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Prediction: {pred}", fontsize=26, fontweight="bold", color=theme["bad"],
                 pad=14)
    fig.text(0.04, 0.04,
             f"With Nawa's preprocessing (invert, crop, center) the same image is a "
             f"{fixed['prediction']} at {max(fixed['probabilities']) * 100:.1f}%.",
             fontsize=19, color=theme["muted"])
    return save(fig, "confident_wrong", scheme)

# ---------------------------------------------------------------------------


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, fixed = run_predict(preprocess=False), run_predict(preprocess=True)
    for scheme in THEMES:
        for path in (speedup_chart(scheme), neuron_maps(scheme), architecture(scheme),
                     confident_wrong(scheme, raw, fixed)):
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
