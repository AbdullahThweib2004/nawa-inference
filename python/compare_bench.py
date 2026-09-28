"""Compares two nawa_bench result files (Google Benchmark JSON). Standard library only.

    python python/compare_bench.py benchmarks/results/baseline.json new.json
    python python/compare_bench.py old.json new.json --metric cycles_per_iter
    python python/compare_bench.py old.json new.json --filter Matmul

For each benchmark present in both files, prints old, new and speedup (> 1 means the new
result is better). With aggregated results (--benchmark_repetitions), the median is used.

Which metric to trust: on a laptop the clock changes with turbo budget and temperature, so
`real_time` (the default) can move 10-25% between runs of identical code. Cycle-based
counters (cycles_per_iter, cycles_per_image, FLOP_per_cycle) are much steadier; use
--metric cycles_per_iter to see changes smaller than the wall-clock noise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Metrics where a larger value is better. Everything else (times, cycles, allocations,
# latency percentiles) is better when smaller.
HIGHER_IS_BETTER = {"GFLOPS", "FLOP_per_cycle", "GBps", "items_per_second", "bytes_per_second"}
TIME_METRICS = {"real_time", "cpu_time"}
NS_PER_UNIT = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def load_results(path: Path, stat: str) -> dict[str, dict]:
    """Returns {benchmark name: result entry}, choosing the `stat` aggregate if present."""
    entries = json.loads(path.read_text())["benchmarks"]
    aggregated = {e["run_name"]: e for e in entries if e.get("aggregate_name") == stat}
    if aggregated:
        return aggregated
    # No aggregates (a run without repetitions): use the plain iteration entries.
    return {e["run_name"]: e for e in entries if e.get("run_type", "iteration") == "iteration"}


def metric_value(entry: dict, metric: str) -> float | None:
    if metric in TIME_METRICS:
        return entry[metric] * NS_PER_UNIT[entry.get("time_unit", "ns")]  # always in ns
    value = entry.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def format_value(value: float, metric: str) -> str:
    if metric in TIME_METRICS:
        for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
            if value >= scale:
                return f"{value / scale:.3g} {unit}"
        return f"{value:.3g} ns"
    for suffix, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path, help="baseline result file")
    parser.add_argument("new", type=Path, help="new result file")
    parser.add_argument("--metric", default="real_time",
                        help="real_time (default), cpu_time, or any counter, e.g. cycles_per_iter")
    parser.add_argument("--stat", default="median", help="aggregate to use (default: median)")
    parser.add_argument("--filter", default="", help="only benchmarks matching this regex")
    args = parser.parse_args()

    old = load_results(args.old, args.stat)
    new = load_results(args.new, args.stat)
    pattern = re.compile(args.filter)
    higher_better = args.metric in HIGHER_IS_BETTER

    rows = []
    for name in old:
        if name not in new or not pattern.search(name):
            continue
        a, b = metric_value(old[name], args.metric), metric_value(new[name], args.metric)
        if a is None or b is None or a == 0 or b == 0:
            continue  # this benchmark doesn't report the metric
        speedup = b / a if higher_better else a / b
        rows.append((name, format_value(a, args.metric), format_value(b, args.metric), speedup))

    if not rows:
        print(f"No benchmarks with metric '{args.metric}' in both files.", file=sys.stderr)
        return 1

    width = max(len("benchmark"), *(len(r[0]) for r in rows))
    direction = "higher is better" if higher_better else "lower is better"
    print(f"metric: {args.metric} ({direction}), {args.stat} of repetitions")
    print(f"old: {args.old}\nnew: {args.new}\n")
    print(f"{'benchmark':<{width}}  {'old':>12}  {'new':>12}  {'speedup':>8}")
    print("-" * (width + 38))
    for name, a, b, speedup in rows:
        print(f"{name:<{width}}  {a:>12}  {b:>12}  {speedup:>7.2f}x")

    only_old = sorted(n for n in set(old) - set(new) if pattern.search(n))
    only_new = sorted(n for n in set(new) - set(old) if pattern.search(n))
    if only_old:
        print(f"\nonly in old: {', '.join(only_old)}")
    if only_new:
        print(f"\nonly in new: {', '.join(only_new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
