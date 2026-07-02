#!/usr/bin/env python3
"""Run the staged OmniCar SGD behavior-correction curriculum."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_UV_BIN = "uv"
DEFAULT_ARTIFACT_DIR = Path("artifacts/omni_car/checkpoints")
DEFAULT_MANIFEST = Path("logs/omni_car_sgd_bc_curriculum/manifest.json")
DEFAULT_REPAIR_GROUPS = ("base", "directional_clear", "directional_front")


def build_bc_command(
    *,
    uv_bin: str,
    load_run: str | Path,
    output: str | Path,
    num_envs: int,
    iterations: int,
    rollout_steps: int,
    rollout_actions: str,
    learning_rate: float,
    seed: int,
    device: str | None = None,
    checkpoint: str | None = None,
    scenarios: Sequence[str] = (),
    scenario_groups: Sequence[str] = (),
    progress_interval: int = 0,
) -> list[str]:
    command = [
        str(uv_bin),
        "run",
        "scripts/train_omni_car_wall_slide_bc.py",
        "--load-run",
        str(load_run),
        "--output",
        str(output),
        "--num-envs",
        str(int(num_envs)),
        "--iterations",
        str(int(iterations)),
        "--rollout-steps",
        str(int(rollout_steps)),
        "--rollout-actions",
        str(rollout_actions),
        "--learning-rate",
        f"{float(learning_rate):g}",
        "--seed",
        str(int(seed)),
    ]
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if device:
        command.extend(["--device", str(device)])
    if progress_interval > 0:
        command.extend(["--progress-interval", str(int(progress_interval))])
    for scenario in scenarios:
        command.extend(["--scenario", str(scenario)])
    for group in scenario_groups:
        command.extend(["--scenario-group", str(group)])
    return command


def build_gate_command(
    *,
    uv_bin: str,
    load_run: str | Path,
    num_envs: int,
    num_steps: int,
    directions: int,
    seed: int,
    suite: str = "broad",
    device: str | None = None,
) -> list[str]:
    command = [
        str(uv_bin),
        "run",
        "scripts/evaluate_omni_car_robustness.py",
        "--load-run",
        str(load_run),
        "--num-envs",
        str(int(num_envs)),
        "--num-steps",
        str(int(num_steps)),
        "--directions",
        str(int(directions)),
        "--suite",
        str(suite),
        "--seed",
        str(int(seed)),
        "--json",
    ]
    if device:
        command.extend(["--device", str(device)])
    return command


def _run_checked(command: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def _run_gate(command: Sequence[str], *, output: Path, dry_run: bool) -> dict[str, Any]:
    print("+ " + " ".join(str(part) for part in command) + f" > {output}", flush=True)
    if dry_run:
        summary: dict[str, Any] = {
            "checkpoint_path": str(command[command.index("--load-run") + 1]),
            "strict_passed": False,
            "scenario_count": 0,
            "category_summary": {},
            "scenarios": [],
            "dry_run": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _gate_record(
    name: str, command: Sequence[str], output: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    failed = [item for item in summary.get("scenarios", []) if not bool(item.get("passed"))]
    return {
        "name": name,
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


def run_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.output)
    artifact_dir = Path(args.artifact_dir)
    prefix = str(args.name_prefix)
    init_checkpoint = artifact_dir / f"{prefix}_init_sgd_bc.pt"
    repair_checkpoint = artifact_dir / f"{prefix}_front_repair.pt"
    init_gate_output = manifest_path.parent / f"{prefix}_init_gate.json"
    repair_gate_output = manifest_path.parent / f"{prefix}_front_repair_gate.json"

    manifest: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "initial_load_run": str(args.load_run),
        "initial_checkpoint": args.checkpoint,
        "latest_checkpoint": str(args.load_run),
        "stages": [],
        "gates": [],
    }
    _write_manifest(manifest_path, manifest)

    init_command = build_bc_command(
        uv_bin=str(args.uv_bin),
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        output=init_checkpoint,
        num_envs=int(args.num_envs),
        iterations=int(args.init_iterations),
        rollout_steps=int(args.rollout_steps),
        rollout_actions=str(args.rollout_actions),
        learning_rate=float(args.init_learning_rate),
        seed=int(args.init_seed),
        device=args.device,
        progress_interval=int(args.progress_interval),
    )
    _run_checked(init_command, dry_run=bool(args.dry_run))
    manifest["stages"].append(
        {"name": "init_sgd_bc", "command": init_command, "checkpoint": str(init_checkpoint)}
    )
    manifest["latest_checkpoint"] = str(init_checkpoint)
    _write_manifest(manifest_path, manifest)

    init_gate_command = build_gate_command(
        uv_bin=str(args.uv_bin),
        load_run=init_checkpoint,
        num_envs=int(args.gate_num_envs),
        num_steps=int(args.gate_num_steps),
        directions=int(args.gate_directions),
        seed=int(args.gate_seed),
        suite=str(args.gate_suite),
        device=args.gate_device or args.device,
    )
    init_summary = _run_gate(init_gate_command, output=init_gate_output, dry_run=bool(args.dry_run))
    init_gate = _gate_record("init_sgd_bc", init_gate_command, init_gate_output, init_summary)
    manifest["gates"].append(init_gate)
    _write_manifest(manifest_path, manifest)
    if init_gate["strict_passed"] and not bool(args.force_repair):
        manifest["completed_early"] = True
        manifest["completed"] = True
        manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_manifest(manifest_path, manifest)
        return manifest

    repair_command = build_bc_command(
        uv_bin=str(args.uv_bin),
        load_run=init_checkpoint,
        output=repair_checkpoint,
        num_envs=int(args.num_envs),
        iterations=int(args.repair_iterations),
        rollout_steps=int(args.rollout_steps),
        rollout_actions=str(args.rollout_actions),
        learning_rate=float(args.repair_learning_rate),
        seed=int(args.repair_seed),
        device=args.device,
        scenario_groups=args.repair_scenario_group or DEFAULT_REPAIR_GROUPS,
        progress_interval=int(args.progress_interval),
    )
    _run_checked(repair_command, dry_run=bool(args.dry_run))
    manifest["stages"].append(
        {"name": "front_repair", "command": repair_command, "checkpoint": str(repair_checkpoint)}
    )
    manifest["latest_checkpoint"] = str(repair_checkpoint)
    _write_manifest(manifest_path, manifest)

    repair_gate_command = build_gate_command(
        uv_bin=str(args.uv_bin),
        load_run=repair_checkpoint,
        num_envs=int(args.gate_num_envs),
        num_steps=int(args.gate_num_steps),
        directions=int(args.gate_directions),
        seed=int(args.gate_seed),
        suite=str(args.gate_suite),
        device=args.gate_device or args.device,
    )
    repair_summary = _run_gate(
        repair_gate_command,
        output=repair_gate_output,
        dry_run=bool(args.dry_run),
    )
    repair_gate = _gate_record("front_repair", repair_gate_command, repair_gate_output, repair_summary)
    manifest["gates"].append(repair_gate)
    manifest["completed_early"] = False
    manifest["completed"] = bool(repair_gate["strict_passed"])
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(manifest_path, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Initial checkpoint path, usually model_0.pt.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name for load-run.")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST), help="Manifest JSON path.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--name-prefix", default="silu_capacity_sgd_bc")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--init-iterations", type=int, default=2500)
    parser.add_argument("--repair-iterations", type=int, default=1400)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-actions", choices=("policy", "target"), default="target")
    parser.add_argument("--init-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--repair-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--init-seed", type=int, default=351)
    parser.add_argument("--repair-seed", type=int, default=361)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--gate-num-envs", type=int, default=16)
    parser.add_argument("--gate-num-steps", type=int, default=128)
    parser.add_argument("--gate-directions", type=int, default=16)
    parser.add_argument("--gate-suite", choices=("broad", "max_stick"), default="broad")
    parser.add_argument("--gate-seed", type=int, default=101)
    parser.add_argument("--gate-device", default=None)
    parser.add_argument(
        "--repair-scenario-group",
        action="append",
        default=None,
        choices=(
            "base",
            "directional",
            "directional_clear",
            "directional_front",
            "directional_wall",
            "max_stick",
            "max_stick_clear",
            "max_stick_front",
            "max_stick_wall",
        ),
        help="Scenario group for the repair stage. Defaults to base, directional_clear, directional_front.",
    )
    parser.add_argument("--force-repair", action="store_true")
    parser.add_argument("--uv-bin", default=DEFAULT_UV_BIN)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_curriculum(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if bool(args.dry_run):
        return 0
    return 0 if bool(manifest.get("completed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
