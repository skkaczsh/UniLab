#!/usr/bin/env python3
"""Search OmniCar repair candidates and rank them with the broad robustness gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_omni_car_sgd_bc_curriculum import (  # noqa: E402
    DEFAULT_UV_BIN,
    _run_checked,
    _run_gate,
    build_bc_command,
    build_gate_command,
)

DEFAULT_MANIFEST = Path("logs/omni_car_repair_search/manifest.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts/omni_car/checkpoints")


@dataclass(frozen=True)
class RepairProfile:
    name: str
    scenario_groups: tuple[str, ...]
    scenarios: tuple[str, ...] = ()


PROFILES: dict[str, RepairProfile] = {
    "front": RepairProfile(
        name="front",
        scenario_groups=("base", "directional_clear", "directional_front"),
    ),
    "front_wall": RepairProfile(
        name="front_wall",
        scenario_groups=("base", "directional_clear", "directional_front", "directional_wall"),
    ),
    "max_stick": RepairProfile(
        name="max_stick",
        scenario_groups=("base", "max_stick_clear", "max_stick_front", "max_stick_wall"),
    ),
    "max_stick_failures": RepairProfile(
        name="max_stick_failures",
        scenario_groups=("base",),
        scenarios=(
            "max_stick_front_blocked_dir_00_circle",
            "max_stick_front_blocked_dir_04_box",
            "max_stick_left_wall_dir_04",
        ),
    ),
    "full": RepairProfile(name="full", scenario_groups=()),
}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    unknown = [value for value in values if value not in PROFILES]
    if unknown:
        raise ValueError(f"unknown repair profile(s): {', '.join(unknown)}")
    return values


def _lr_slug(value: float) -> str:
    text = f"{float(value):.0e}" if value < 1.0e-3 else f"{float(value):g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def build_candidates(
    *,
    profiles: Sequence[str],
    seeds: Sequence[int],
    learning_rates: Sequence[float],
    iterations: Sequence[int],
    max_candidates: int | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    index = 0
    for profile_name in profiles:
        profile = PROFILES[profile_name]
        for iteration_count in iterations:
            for learning_rate in learning_rates:
                for seed in seeds:
                    if index >= int(start_index):
                        candidates.append(
                            {
                                "index": index,
                                "profile": profile.name,
                                "scenario_groups": list(profile.scenario_groups),
                                "scenarios": list(profile.scenarios),
                                "iterations": int(iteration_count),
                                "learning_rate": float(learning_rate),
                                "seed": int(seed),
                            }
                        )
                    index += 1
                    if max_candidates is not None and len(candidates) >= int(max_candidates):
                        return candidates
    return candidates


def gate_score(gate: dict[str, Any]) -> tuple[float, ...]:
    categories = gate.get("category_summary", {})

    def _passed(name: str) -> float:
        item = categories.get(name, {})
        return float(item.get("passed", 0))

    def _count(name: str) -> float:
        item = categories.get(name, {})
        return float(item.get("count", 0))

    def _collision(name: str) -> float:
        item = categories.get(name, {})
        return float(item.get("collision_fraction_max", 0.0))

    def _projection_min(name: str) -> float:
        item = categories.get(name, {})
        value = float(item.get("projection_mean_min", -1.0e9))
        return value if math.isfinite(value) else -1.0e9

    clear_names = ("clear", "max_stick_clear")
    front_names = ("front_blocked", "max_stick_front_blocked")
    side_names = ("side_wall", "max_stick_side_wall")
    other_names = ("yaw", "zero")
    all_names = clear_names + front_names + side_names + other_names
    total_passed = sum(_passed(name) for name in all_names)
    total_count = sum(_count(name) for name in all_names)
    collision_cost = sum(_collision(name) for name in all_names)
    failed_count = float(gate.get("failed_count", max(total_count - total_passed, 0.0)))
    return (
        1.0 if bool(gate.get("strict_passed")) else 0.0,
        -failed_count,
        total_passed,
        sum(_passed(name) for name in front_names),
        sum(_passed(name) for name in side_names),
        sum(_passed(name) for name in clear_names),
        _passed("zero"),
        _passed("yaw"),
        -collision_cost,
        max(_projection_min(name) for name in front_names),
        max(_projection_min(name) for name in side_names),
        max(_projection_min(name) for name in clear_names),
    )


def _gate_record(
    *,
    command: Sequence[str],
    output: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    failed = [item for item in summary.get("scenarios", []) if not bool(item.get("passed"))]
    return {
        "command": list(command),
        "output": str(output),
        "strict_passed": bool(summary.get("strict_passed")),
        "scenario_count": int(summary.get("scenario_count", 0)),
        "failed_count": len(failed),
        "failed_scenarios": [str(item.get("scenario", "")) for item in failed],
        "category_summary": summary.get("category_summary", {}),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_name(prefix: str, candidate: dict[str, Any]) -> str:
    return (
        f"{prefix}_c{int(candidate['index']):02d}_"
        f"{candidate['profile']}_it{int(candidate['iterations'])}_"
        f"lr{_lr_slug(float(candidate['learning_rate']))}_"
        f"s{int(candidate['seed'])}"
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _parse_csv_strings(str(args.profiles))
    seeds = _parse_csv_ints(str(args.seeds))
    learning_rates = _parse_csv_floats(str(args.learning_rates))
    iterations = _parse_csv_ints(str(args.iterations))
    candidates = build_candidates(
        profiles=profiles,
        seeds=seeds,
        learning_rates=learning_rates,
        iterations=iterations,
        max_candidates=args.max_candidates,
        start_index=int(args.start_index),
    )
    manifest_path = Path(args.output)
    artifact_dir = Path(args.artifact_dir)
    manifest: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_run": str(args.load_run),
        "candidate_count": len(candidates),
        "candidates": [],
        "best": None,
        "completed": False,
    }
    _write_manifest(manifest_path, manifest)

    for candidate in candidates:
        name = _candidate_name(str(args.name_prefix), candidate)
        checkpoint = artifact_dir / f"{name}.pt"
        gate_output = manifest_path.parent / f"{name}_gate.json"
        train_command = build_bc_command(
            uv_bin=str(args.uv_bin),
            load_run=args.load_run,
            output=checkpoint,
            num_envs=int(args.num_envs),
            iterations=int(candidate["iterations"]),
            rollout_steps=int(args.rollout_steps),
            rollout_actions=str(args.rollout_actions),
            learning_rate=float(candidate["learning_rate"]),
            seed=int(candidate["seed"]),
            device=args.device,
            scenario_groups=candidate["scenario_groups"],
            scenarios=candidate["scenarios"],
            progress_interval=int(args.progress_interval),
        )
        _run_checked(train_command, dry_run=bool(args.dry_run))
        gate_command = build_gate_command(
            uv_bin=str(args.uv_bin),
            load_run=checkpoint,
            num_envs=int(args.gate_num_envs),
            num_steps=int(args.gate_num_steps),
            directions=int(args.gate_directions),
            seed=int(args.gate_seed),
            suite=str(args.gate_suite),
            device=args.gate_device or args.device,
        )
        summary = _run_gate(gate_command, output=gate_output, dry_run=bool(args.dry_run))
        gate = _gate_record(command=gate_command, output=gate_output, summary=summary)
        score = gate_score(gate)
        record = {
            **candidate,
            "name": name,
            "checkpoint": str(checkpoint),
            "train_command": train_command,
            "gate": gate,
            "score": list(score),
        }
        manifest["candidates"].append(record)
        best = manifest.get("best")
        if best is None or score > tuple(best["score"]):
            manifest["best"] = {
                "name": name,
                "checkpoint": str(checkpoint),
                "score": list(score),
                "gate": gate,
            }
        manifest["completed"] = bool((manifest.get("best") or {}).get("gate", {}).get("strict_passed"))
        _write_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "candidate": name,
                    "strict_passed": gate["strict_passed"],
                    "failed_count": gate["failed_count"],
                    "failed_scenarios": gate["failed_scenarios"],
                    "best": (manifest.get("best") or {}).get("name"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if gate["strict_passed"] and bool(args.stop_on_pass):
            break

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(manifest_path, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Base checkpoint to repair.")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST), help="Search manifest JSON path.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--name-prefix", default="omni_car_repair_search")
    parser.add_argument("--profiles", default="front,front_wall")
    parser.add_argument("--seeds", default="361,362,363")
    parser.add_argument("--learning-rates", default="5e-5,3e-5")
    parser.add_argument("--iterations", default="900,1400")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-actions", choices=("policy", "target"), default="target")
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-interval", type=int, default=300)
    parser.add_argument("--gate-num-envs", type=int, default=16)
    parser.add_argument("--gate-num-steps", type=int, default=128)
    parser.add_argument("--gate-directions", type=int, default=16)
    parser.add_argument("--gate-suite", choices=("broad", "max_stick"), default="broad")
    parser.add_argument("--gate-seed", type=int, default=101)
    parser.add_argument("--gate-device", default=None)
    parser.add_argument("--stop-on-pass", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--uv-bin", default=DEFAULT_UV_BIN)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_search(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if bool(args.dry_run) or not bool(args.require_pass):
        return 0
    return 0 if bool(manifest.get("completed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
