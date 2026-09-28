"""Reproducible screenshots of the web demo with a hand-drawn "5".

    python tools/screenshot.py        # from the repo root, with .venv active

Needs build/bin/nawa (it builds nothing) and Playwright with Chromium:
    pip install -r python/requirements-dev.txt && python -m playwright install chromium

Starts `nawa serve` (fp32 + int8) on a free port, draws a "5" with real pointer events, clicks
Predict, hovers the hidden neuron that pushes hardest toward the answer, and saves
docs/images/demo_light.png, demo_dark.png and demo_network.png. The server is always stopped.
"""

from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
NAWA = ROOT / "build" / "bin" / "nawa"
OUT = ROOT / "docs" / "images"
VIEWPORT = {"width": 1600, "height": 1000}
TARGET_DIGIT = 5

# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def start_server() -> tuple[subprocess.Popen, str]:
    if not NAWA.exists():
        sys.exit(f"{NAWA} not found: build first (cmake --build build -j)")
    server = subprocess.Popen(
        [str(NAWA), "serve", str(ROOT / "models" / "mnist_mlp.nawa"),
         "--int8", str(ROOT / "models" / "mnist_mlp_int8.nawa"), "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = server.stdout.readline()  # "Nawa web demo: http://127.0.0.1:<port>/ ..."
    if "http://" not in line:
        server.kill()
        sys.exit(f"nawa serve did not start: {line.strip()}")
    url = "http://" + line.split("http://", 1)[1].split()[0]
    # Wait until the API answers.
    for _ in range(50):
        try:
            with urllib.request.urlopen(url + "api/model", timeout=1) as r:
                if r.status == 200:
                    return server, url
        except OSError:
            time.sleep(0.1)
    server.kill()
    sys.exit("nawa serve did not answer /api/model")


def stop_server(server: subprocess.Popen) -> None:
    server.send_signal(signal.SIGTERM)  # nawa serve shuts down cleanly on SIGTERM
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()

# ---------------------------------------------------------------------------
# Drawing a "5"
# ---------------------------------------------------------------------------


def bezier(points: list[tuple[float, float]], samples: int) -> list[tuple[float, float]]:
    """Samples a Bezier curve through control points (de Casteljau), in unit coordinates."""
    out = []
    for i in range(samples + 1):
        t = i / samples
        pts = list(points)
        while len(pts) > 1:
            pts = [((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(pts, pts[1:])]
        out.append(pts[0])
    return out


def five_strokes(variant: int) -> list[list[tuple[float, float]]]:
    """A handwritten 5 in canvas-relative coordinates (0..1), about 70% of the canvas tall.

    Stroke 1: the top bar. Stroke 2: down the left side, then the round belly that ends at the
    bottom-left. `variant` changes the belly a little for retries.
    """
    belly = [0.70, 0.74, 0.66][variant % 3]    # how far right the belly bulges
    drop = [0.45, 0.47, 0.43][variant % 3]     # where the vertical stroke ends
    top = [(0.34, 0.17), (0.50, 0.155), (0.69, 0.15)]
    left = [(0.345, 0.175), (0.33, 0.31), (0.315, drop)]
    curve = bezier([(0.315, drop), (0.47, drop - 0.08), (belly + 0.06, drop + 0.02),
                    (belly + 0.04, 0.80), (0.46, 0.88), (0.27, 0.80)], 40)
    return [bezier(top, 16), left + curve[1:]]


def draw(page: Page, strokes: list[list[tuple[float, float]]]) -> None:
    box = page.locator("#pad").bounding_box()
    to_px = lambda p: (box["x"] + p[0] * box["width"], box["y"] + p[1] * box["height"])
    for stroke in strokes:
        x, y = to_px(stroke[0])
        page.mouse.move(x, y)
        page.mouse.down()
        for point in stroke[1:]:
            x, y = to_px(point)
            page.mouse.move(x, y, steps=3)  # many small pointer moves, like a real hand
        page.mouse.up()


def predict(page: Page) -> dict:
    """Clicks Predict and returns the server's JSON response once the page has rendered it."""
    with page.expect_response(lambda r: r.url.endswith("/api/predict")) as info:
        page.click("#predict-btn")
    result = info.value.json()
    page.wait_for_function(
        "expected => document.getElementById('digit').textContent === String(expected)",
        arg=result["prediction"])
    page.wait_for_timeout(300)  # a few frames for the SVG links (set via requestAnimationFrame)
    return result


def draw_until_five(page: Page) -> dict:
    result = {}
    for attempt in range(3):
        page.click("#clear-btn")
        draw(page, five_strokes(attempt))
        result = predict(page)
        print(f"attempt {attempt + 1}: predicted {result['prediction']} "
              f"({result['confidence'] * 100:.1f}%)")
        if result["prediction"] == TARGET_DIGIT:
            return result
    print(f"WARNING: the model predicted {result['prediction']}, not {TARGET_DIGIT}, after 3 "
          f"attempts; saving the screenshots anyway")
    return result


def hover_best_neuron(page: Page, result: dict) -> int:
    """Hovers the hidden neuron with the largest positive contribution to the winner."""
    positive = [c for c in result["contributions"] if c["contribution"] > 0]
    best = max(positive or result["contributions"], key=lambda c: c["contribution"])
    neuron = best["neuron"]
    page.hover(f'#hidden circle.hidden-node[data-id="{neuron}"]')
    page.wait_for_function(
        "id => !document.getElementById('neuron-card').hidden && "
        "document.getElementById('neuron-title').textContent === 'Hidden neuron ' + id",
        arg=neuron)
    page.wait_for_timeout(200)
    return neuron

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def capture(browser, url: str, scheme: str) -> tuple[dict, int, Page]:
    context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2, color_scheme=scheme,
                                  reduced_motion="reduce")  # no transitions mid-screenshot
    page = context.new_page()
    page.goto(url)
    page.wait_for_function("document.getElementById('status').classList.contains('ok')")
    result = draw_until_five(page)
    neuron = hover_best_neuron(page, result)
    return result, neuron, page


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    server, url = start_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            outputs = {}
            for scheme in ("light", "dark"):
                result, neuron, page = capture(browser, url, scheme)
                path = OUT / f"demo_{scheme}.png"
                page.screenshot(path=str(path), full_page=True)
                outputs[scheme] = (result, neuron, path)
                if scheme == "light":
                    close_up = OUT / "demo_network.png"
                    page.locator(".net-panel").screenshot(path=str(close_up))
                    outputs["network"] = (result, neuron, close_up)
                page.context.close()
            browser.close()
    finally:
        stop_server(server)

    for name, (result, neuron, path) in outputs.items():
        print(f"{name:8s} predicted {result['prediction']} "
              f"({result['confidence'] * 100:.2f}%), hovered neuron {neuron} -> "
              f"{path.relative_to(ROOT)}")
    ok = all(r["prediction"] == TARGET_DIGIT for r, _, _ in outputs.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
