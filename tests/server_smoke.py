"""Smoke test for `nawa serve`: start the real binary, call the API, stop it.

    python3 tests/server_smoke.py <path to nawa> <source dir>

Standard library only. Registered in CTest (tests/CMakeLists.txt).
"""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import urllib.error
import urllib.request


def load_first_fixture_image(path: str) -> list[int]:
    """Reads image 0 of a .ntsr tensor file (docs/model_format.md) without numpy."""
    data = open(path, "rb").read()
    assert data[:4] == b"NTSR", "not a tensor file"
    ndim = struct.unpack_from("<I", data, 8)[0]
    offset = 12 + 8 * ndim  # magic + version + ndim + dims
    values = struct.unpack_from("<784f", data, offset)
    return [int(v) for v in values]


def request(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> int:
    nawa, source = sys.argv[1], sys.argv[2]
    server = subprocess.Popen(
        [nawa, "serve", os.path.join(source, "models", "mnist_mlp.nawa"),
         "--int8", os.path.join(source, "models", "mnist_mlp_int8.nawa"), "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # The first line is "Nawa web demo: http://127.0.0.1:<port>/ ..."
        line = server.stdout.readline()
        print(line.strip())
        url = line.split("http://", 1)[1].split()[0].rstrip("/")
        base = "http://" + url

        status, info = request(base + "/api/model")
        assert status == 200 and info["variants"] == ["fp32", "int8"], (status, info)

        pixels = load_first_fixture_image(
            os.path.join(source, "tests", "fixtures", "mnist_test100_images.ntsr"))
        for variant in ("fp32", "int8"):
            status, result = request(base + "/api/predict", {
                "width": 28, "height": 28, "pixels": pixels, "variant": variant})
            assert status == 200, (status, result)
            assert result["prediction"] == 7, result["prediction"]
            print(f"{variant}: predicted {result['prediction']} "
                  f"({result['confidence'] * 100:.2f}%), "
                  f"forward {result['timing_us']['forward']:.1f} us")

        status, error = request(base + "/api/predict", {"width": 28})
        assert status == 400 and "error" in error, (status, error)
        print("PASS")
        return 0
    finally:
        server.send_signal(signal.SIGTERM)  # nawa serve stops cleanly on SIGINT/SIGTERM
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            raise


if __name__ == "__main__":
    sys.exit(main())
