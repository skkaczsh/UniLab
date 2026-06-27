#!/usr/bin/env python3
"""Summarize OmniCar PPO text logs into machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
METRIC_RE = re.compile(r"^\s*([A-Za-z0-9_./ -]+):\s+([-+0-9.eE]+)\s*$")

DEFAULT_BEST_KEYS = (
    "omni_car/tracking_error",
    "omni_car/collision_rate",
    "omni_car/static_collision_rate",
    "omni_car/agent_collision_rate",
    "omni_car/mean_clearance",
    "Mean reward",
    "Mean episode length",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Training log produced by remote_omni_car_tmux.")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of latest iterations used for rolling means.",
    )
    parser.add_argument(
        "--best-key",
        action="append",
        default=[],
        help="Metric key to include in best-min/best-max summaries. May be repeated.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON to this path.")
    return parser.parse_args(argv)


def _coerce_number(raw: str) -> int | float:
    value = float(raw)
    if value.is_integer() and "." not in raw and "e" not in raw.lower():
        return int(value)
    return value


def parse_training_log(text: str) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            if current is not None:
                iterations.append(current)
            current = {
                "iteration": int(iter_match.group(1)),
                "max_iteration": int(iter_match.group(2)),
                "metrics": {},
            }
            continue
        if current is None:
            continue
        metric_match = METRIC_RE.match(line)
        if metric_match:
            key = metric_match.group(1).strip()
            current["metrics"][key] = _coerce_number(metric_match.group(2))
    if current is not None:
        iterations.append(current)
    return iterations


def _metric_value(iteration: dict[str, Any], key: str) -> float | None:
    value = iteration["metrics"].get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _best_summary(iterations: list[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for key in keys:
        values = [(iteration, _metric_value(iteration, key)) for iteration in iterations]
        values = [(iteration, value) for iteration, value in values if value is not None]
        if not values:
            continue
        min_iter, min_value = min(values, key=lambda pair: pair[1])
        max_iter, max_value = max(values, key=lambda pair: pair[1])
        best[key] = {
            "min": {"iteration": min_iter["iteration"], "value": min_value},
            "max": {"iteration": max_iter["iteration"], "value": max_value},
        }
    return best


def _window_summary(iterations: list[dict[str, Any]], window: int) -> dict[str, Any]:
    if not iterations:
        return {"size": 0, "mean": {}}
    selected = iterations[-max(1, int(window)) :]
    metric_keys = sorted({key for iteration in selected for key in iteration["metrics"]})
    means: dict[str, float] = {}
    for key in metric_keys:
        values = [_metric_value(iteration, key) for iteration in selected]
        numeric = [value for value in values if value is not None]
        if numeric:
            means[key] = float(sum(numeric) / len(numeric))
    return {"size": len(selected), "mean": means}


def summarize_training_log(
    log_path: Path, *, window: int = 20, best_keys: Sequence[str] = DEFAULT_BEST_KEYS
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iterations = parse_training_log(text)
    latest = iterations[-1] if iterations else None
    summary: dict[str, Any] = {
        "log_path": str(log_path),
        "iteration_count": len(iterations),
        "latest": latest,
        "best": _best_summary(iterations, best_keys),
        "window": _window_summary(iterations, window),
    }
    if latest is not None:
        max_iteration = int(latest["max_iteration"])
        current_iteration = int(latest["iteration"])
        summary["progress_fraction"] = (
            current_iteration / max_iteration if max_iteration > 0 else None
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    best_keys = tuple(dict.fromkeys((*DEFAULT_BEST_KEYS, *args.best_key)))
    summary = summarize_training_log(args.log_path, window=args.window, best_keys=best_keys)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
