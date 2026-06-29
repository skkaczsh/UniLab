#!/usr/bin/env python3
"""Run alternating OmniCar PPO curriculum phases and select checkpoints by behavior probes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.scan_omni_car_checkpoints import discover_checkpoint_ids, thin_checkpoint_ids

DEFAULT_UV_BIN = "uv"
DEFAULT_TASK_DIR = "OmniCarGridAvoidance"
REQUIRED_SCENARIOS = (
    "zero_input_hold",
    "clear_forward_follow",
    "clear_diagonal_follow",
    "front_blocked_stop",
    "right_wall_forward",
)


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    overrides: tuple[str, ...]
    scenario_weights: dict[str, float]


PHASES: dict[str, CurriculumPhase] = {
    "open_follow": CurriculumPhase(
        name="open_follow",
        overrides=(
            "env.command.zero_fraction=0.18",
            "env.command.long_hold_fraction=0.65",
            "env.obstacles.count=6",
            "env.obstacles.clear_path_fraction=0.95",
            "env.obstacles.front_blocker_fraction=0.00",
            "env.obstacles.side_wall_fraction=0.00",
            "env.large_scene.static_obstacle_count=80",
            "env.large_scene.max_dynamic_agents=4",
            "env.large_scene.dense_region_fraction=0.25",
            "env.reward.intent=260.0",
            "env.reward.intent_projection=200.0",
            "env.reward.response=140.0",
            "env.reward.vx_track=95.0",
            "env.reward.vy_track=85.0",
            "env.reward.vyaw_track=70.0",
            "env.reward.target_collision=30.0",
            "env.reward.clearance_motion=20.0",
            "env.reward.clearance_target_motion=35.0",
            "env.reward.clearance_opening=20.0",
            "env.reward.clearance_target_opening=20.0",
            "env.reward.blocked_projection=120.0",
            "env.reward.blocked_speed=10.0",
            "env.reward.blocked_stop=4.0",
            "env.reward.blocked_lateral_escape=8.0",
            "env.reward.idle_stop=45.0",
            "env.reward.yaw_idle_stop=40.0",
            "env.reward.off_axis=35.0",
            "env.reward.reverse=60.0",
        ),
        scenario_weights={
            "zero_input_hold": 5.0,
            "clear_forward_follow": 14.0,
            "clear_diagonal_follow": 14.0,
            "front_blocked_stop": 0.5,
            "right_wall_forward": 3.0,
        },
    ),
    "clear_explore": CurriculumPhase(
        name="clear_explore",
        overrides=(
            "env.command.zero_fraction=0.10",
            "env.command.long_hold_fraction=0.55",
            "env.obstacles.clear_path_fraction=0.85",
            "env.obstacles.front_blocker_fraction=0.05",
            "env.obstacles.side_wall_fraction=0.05",
            "env.reward.intent=220.0",
            "env.reward.intent_projection=160.0",
            "env.reward.response=120.0",
            "env.reward.vx_track=80.0",
            "env.reward.vy_track=70.0",
            "env.reward.vyaw_track=55.0",
            "env.reward.target_collision=120.0",
            "env.reward.clearance_target_motion=120.0",
            "env.reward.blocked_projection=700.0",
            "env.reward.clearance_target_opening=80.0",
            "env.reward.blocked_speed=50.0",
            "env.reward.blocked_stop=18.0",
            "env.reward.blocked_lateral_escape=30.0",
            "env.reward.idle_stop=24.0",
            "env.reward.yaw_idle_stop=18.0",
        ),
        scenario_weights={
            "zero_input_hold": 2.0,
            "clear_forward_follow": 10.0,
            "clear_diagonal_follow": 10.0,
            "front_blocked_stop": 1.5,
            "right_wall_forward": 5.0,
        },
    ),
    "side_follow": CurriculumPhase(
        name="side_follow",
        overrides=(
            "env.command.zero_fraction=0.22",
            "env.command.long_hold_fraction=0.55",
            "env.obstacles.clear_path_fraction=0.25",
            "env.obstacles.front_blocker_fraction=0.10",
            "env.obstacles.side_wall_fraction=0.60",
            "env.reward.intent=150.0",
            "env.reward.intent_projection=110.0",
            "env.reward.response=80.0",
            "env.reward.vx_track=52.0",
            "env.reward.vy_track=46.0",
            "env.reward.vyaw_track=34.0",
            "env.reward.target_collision=110.0",
            "env.reward.clearance_motion=45.0",
            "env.reward.clearance_target_motion=90.0",
            "env.reward.clearance_opening=55.0",
            "env.reward.clearance_target_opening=140.0",
            "env.reward.blocked_projection=520.0",
            "env.reward.blocked_speed=45.0",
            "env.reward.blocked_stop=16.0",
            "env.reward.blocked_lateral_escape=35.0",
            "env.reward.idle_stop=48.0",
            "env.reward.yaw_idle_stop=42.0",
            "env.reward.off_axis=30.0",
            "env.reward.reverse=50.0",
        ),
        scenario_weights={
            "zero_input_hold": 5.0,
            "clear_forward_follow": 5.0,
            "clear_diagonal_follow": 5.0,
            "front_blocked_stop": 4.0,
            "right_wall_forward": 14.0,
        },
    ),
    "safety": CurriculumPhase(
        name="safety",
        overrides=(
            "env.obstacles.clear_path_fraction=0.25",
            "env.obstacles.front_blocker_fraction=0.55",
            "env.obstacles.side_wall_fraction=0.10",
            "env.reward.intent=105.0",
            "env.reward.intent_projection=65.0",
            "env.reward.response=45.0",
            "env.reward.vx_track=32.0",
            "env.reward.vy_track=30.0",
            "env.reward.target_collision=240.0",
            "env.reward.blocked_projection=1300.0",
            "env.reward.clearance_target_opening=50.0",
            "env.reward.blocked_speed=260.0",
            "env.reward.blocked_stop=42.0",
            "env.reward.blocked_lateral_escape=95.0",
        ),
        scenario_weights={
            "zero_input_hold": 3.0,
            "clear_forward_follow": 1.0,
            "clear_diagonal_follow": 1.0,
            "front_blocked_stop": 8.0,
            "right_wall_forward": 2.0,
        },
    ),
    "clear": CurriculumPhase(
        name="clear",
        overrides=(
            "env.obstacles.clear_path_fraction=0.60",
            "env.obstacles.front_blocker_fraction=0.25",
            "env.obstacles.side_wall_fraction=0.10",
            "env.reward.intent=125.0",
            "env.reward.intent_projection=75.0",
            "env.reward.response=50.0",
            "env.reward.vx_track=34.0",
            "env.reward.vy_track=32.0",
            "env.reward.target_collision=180.0",
            "env.reward.blocked_projection=900.0",
            "env.reward.clearance_target_opening=70.0",
            "env.reward.blocked_speed=90.0",
            "env.reward.blocked_stop=28.0",
            "env.reward.blocked_lateral_escape=45.0",
        ),
        scenario_weights={
            "zero_input_hold": 3.0,
            "clear_forward_follow": 6.0,
            "clear_diagonal_follow": 6.0,
            "front_blocked_stop": 2.5,
            "right_wall_forward": 4.0,
        },
    ),
    "balanced": CurriculumPhase(
        name="balanced",
        overrides=(
            "env.obstacles.clear_path_fraction=0.40",
            "env.obstacles.front_blocker_fraction=0.40",
            "env.obstacles.side_wall_fraction=0.15",
            "env.reward.intent=95.0",
            "env.reward.intent_projection=55.0",
            "env.reward.response=40.0",
            "env.reward.vx_track=28.0",
            "env.reward.vy_track=26.0",
            "env.reward.target_collision=220.0",
            "env.reward.blocked_projection=1200.0",
            "env.reward.clearance_target_opening=60.0",
            "env.reward.blocked_speed=180.0",
            "env.reward.blocked_stop=36.0",
            "env.reward.blocked_lateral_escape=75.0",
        ),
        scenario_weights={
            "zero_input_hold": 4.0,
            "clear_forward_follow": 4.0,
            "clear_diagonal_follow": 4.0,
            "front_blocked_stop": 5.0,
            "right_wall_forward": 4.0,
        },
    ),
}


def _normalize_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _scenario_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gate = row.get("behavior_gate")
    if not isinstance(gate, dict):
        return {}
    scenarios = gate.get("scenarios", [])
    return {
        str(item.get("scenario")): item
        for item in scenarios
        if isinstance(item, dict) and item.get("scenario") is not None
    }


def scenario_cost(name: str, item: dict[str, Any]) -> float:
    projection = float(item.get("projection_mean") or 0.0)
    planar = float(item.get("planar_speed_mean") or 0.0)
    yaw = float(item.get("yaw_abs_mean") or 0.0)
    collision = float(item.get("collision_fraction") or 0.0)
    cost = 80.0 * collision
    if name == "zero_input_hold":
        cost += 8.0 * max(planar - 0.08, 0.0)
        cost += 4.0 * max(yaw - 0.08, 0.0)
    elif name == "clear_forward_follow":
        cost += 3.0 * max(0.45 - projection, 0.0)
    elif name == "clear_diagonal_follow":
        cost += 3.0 * max(0.40 - projection, 0.0)
    elif name == "front_blocked_stop":
        cost += 4.0 * max(projection - 0.20, 0.0)
        cost += 2.0 * max(planar - 0.35, 0.0)
    elif name == "right_wall_forward":
        cost += 3.0 * max(0.20 - projection, 0.0)
        cost += 80.0 * max(collision - 0.02, 0.0)
    return float(cost)


def behavior_cost(row: dict[str, Any], phase: CurriculumPhase) -> float:
    scenarios = _scenario_map(row)
    if not scenarios:
        return float("inf")
    total = 0.0
    for name, weight in phase.scenario_weights.items():
        item = scenarios.get(name)
        if item is None:
            total += 1_000.0 * weight
            continue
        total += float(weight) * scenario_cost(name, item)
    return total


def behavior_pass_score(row: dict[str, Any], phase: CurriculumPhase) -> float:
    scenarios = _scenario_map(row)
    score = 0.0
    for name, weight in phase.scenario_weights.items():
        item = scenarios.get(name)
        if item is not None and item.get("passed") is True:
            score += float(weight)
    return score


def behavior_pass_count(row: dict[str, Any]) -> int:
    scenarios = _scenario_map(row)
    return sum(
        1
        for name in REQUIRED_SCENARIOS
        if scenarios.get(name, {}).get("passed") is True
    )


def select_checkpoint(scan: dict[str, Any], phase: CurriculumPhase) -> dict[str, Any]:
    rows = [row for row in scan.get("evaluations", []) if isinstance(row, dict)]
    if not rows:
        raise ValueError("scan result contains no evaluations")

    def all_gate_key(row: dict[str, Any]) -> tuple[float, float]:
        generic = float(row.get("selection_score") or 0.0)
        return behavior_cost(row, phase), generic

    def partial_gate_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        generic = float(row.get("selection_score") or 0.0)
        return (
            -float(behavior_pass_count(row)),
            -behavior_pass_score(row, phase),
            behavior_cost(row, phase),
            generic,
        )

    all_gates = [
        row
        for row in rows
        if isinstance(row.get("behavior_gate"), dict)
        and row["behavior_gate"].get("passed") is True
    ]
    selected = min(all_gates, key=all_gate_key) if all_gates else min(rows, key=partial_gate_key)
    return {
        "checkpoint": int(selected["checkpoint"]),
        "checkpoint_path": selected.get("checkpoint_path"),
        "behavior_pass_count": behavior_pass_count(selected),
        "behavior_pass_score": behavior_pass_score(selected, phase),
        "behavior_cost": behavior_cost(selected, phase),
        "selection_score": float(selected.get("selection_score") or 0.0),
        "passed_all_behavior_gates": bool(
            isinstance(selected.get("behavior_gate"), dict)
            and selected["behavior_gate"].get("passed") is True
        ),
        "row": selected,
    }


def archive_key(selection: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        int(bool(selection.get("passed_all_behavior_gates"))),
        int(selection.get("behavior_pass_count") or 0),
        -float(selection.get("behavior_cost") or 0.0),
        -float(selection.get("selection_score") or 0.0),
    )


def choose_continuation_checkpoint(
    *,
    selection: dict[str, Any],
    archive_best: dict[str, Any] | None,
    allow_coverage_regression: bool,
) -> tuple[dict[str, Any], bool]:
    if archive_best is None or allow_coverage_regression:
        return selection, False
    selected_count = int(selection.get("behavior_pass_count") or 0)
    archive_count = int(archive_best.get("behavior_pass_count") or 0)
    if selected_count < archive_count:
        return archive_best, True
    return selection, False


def build_train_command(
    *,
    uv_bin: str,
    run_name: str,
    load_run: str,
    max_iterations: int,
    num_envs: int,
    num_steps_per_env: int,
    save_interval: int,
    phase: CurriculumPhase,
    resume_action_std: float | None = None,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    command = [
        uv_bin,
        "run",
        "train",
        "--algo",
        "ppo",
        "--task",
        "omni_car_grid_avoidance",
        "--sim",
        "mujoco",
        "training.logger=tensorboard",
        f"training.log_root=logs/{run_name}",
        f"algo.load_run={load_run}",
        f"algo.run_name={run_name}",
        f"algo.num_envs={int(num_envs)}",
        f"algo.num_steps_per_env={int(num_steps_per_env)}",
        f"algo.max_iterations={int(max_iterations)}",
        f"algo.save_interval={int(save_interval)}",
        *phase.overrides,
    ]
    if resume_action_std is not None:
        command.append(f"algo.resume_action_std={float(resume_action_std)}")
    command.extend(extra_overrides)
    return command


def build_scan_command(
    *,
    uv_bin: str,
    run_dir: Path,
    checkpoints: Sequence[int],
    output: Path,
    num_envs: int,
    num_steps: int,
    seed: int,
    device: str | None,
    behavior_num_envs: int,
    behavior_num_steps: int,
    behavior_seed: int,
) -> list[str]:
    command = [
        uv_bin,
        "run",
        "scripts/scan_omni_car_checkpoints.py",
        "--load-run",
        str(run_dir),
        "--checkpoints",
        *(str(item) for item in checkpoints),
        "--num-envs",
        str(int(num_envs)),
        "--num-steps",
        str(int(num_steps)),
        "--seed",
        str(int(seed)),
        "--behavior-gate",
        "--behavior-num-envs",
        str(int(behavior_num_envs)),
        "--behavior-num-steps",
        str(int(behavior_num_steps)),
        "--behavior-seed",
        str(int(behavior_seed)),
        "--output",
        str(output),
    ]
    if device:
        command.extend(["--device", str(device)])
    return command


def latest_run_dir(log_root: Path, task_dir: str = DEFAULT_TASK_DIR) -> Path:
    root = log_root / task_dir
    candidates = [path for path in root.glob("*_mujoco") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No *_mujoco run directories found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def phase_checkpoint_ids(run_dir: Path, *, every: int) -> list[int]:
    return thin_checkpoint_ids(discover_checkpoint_ids(run_dir), every=every, include_last=True)


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def run_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    phase_names = _normalize_csv(args.phases)
    if not phase_names:
        raise ValueError("--phases must name at least one phase")
    unknown = [name for name in phase_names if name not in PHASES]
    if unknown:
        raise ValueError(f"Unknown phase(s): {', '.join(unknown)}")

    manifest: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "initial_load_run": str(args.load_run),
        "run_prefix": str(args.run_prefix),
        "rounds": int(args.rounds),
        "phases": [],
    }
    current_load_run = str(args.load_run)
    manifest_path = Path(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_best: dict[str, Any] | None = None

    for round_idx in range(int(args.rounds)):
        for phase_name in phase_names:
            phase = PHASES[phase_name]
            phase_index = len(manifest["phases"])
            run_name = f"{args.run_prefix}_r{round_idx:02d}_{phase.name}"
            train_command = build_train_command(
                uv_bin=str(args.uv_bin),
                run_name=run_name,
                load_run=current_load_run,
                max_iterations=int(args.phase_iterations),
                num_envs=int(args.num_envs),
                num_steps_per_env=int(args.num_steps_per_env),
                save_interval=int(args.save_interval),
                phase=phase,
                resume_action_std=args.resume_action_std,
                extra_overrides=args.overrides,
            )
            _run(train_command, dry_run=bool(args.dry_run))
            if bool(args.dry_run):
                run_dir = Path("logs") / run_name / DEFAULT_TASK_DIR / "DRY_RUN_mujoco"
                checkpoints = [0]
                scenarios = [
                    {
                        "scenario": "zero_input_hold",
                        "projection_mean": 0.0,
                        "planar_speed_mean": 0.0,
                        "yaw_abs_mean": 0.0,
                        "collision_fraction": 0.0,
                    },
                    {
                        "scenario": "clear_forward_follow",
                        "projection_mean": 0.0,
                        "planar_speed_mean": 0.0,
                        "yaw_abs_mean": 0.0,
                        "collision_fraction": 0.0,
                    },
                    {
                        "scenario": "clear_diagonal_follow",
                        "projection_mean": 0.0,
                        "planar_speed_mean": 0.0,
                        "yaw_abs_mean": 0.0,
                        "collision_fraction": 0.0,
                    },
                    {
                        "scenario": "front_blocked_stop",
                        "projection_mean": 0.0,
                        "planar_speed_mean": 0.0,
                        "yaw_abs_mean": 0.0,
                        "collision_fraction": 0.0,
                    },
                    {
                        "scenario": "right_wall_forward",
                        "projection_mean": 0.0,
                        "planar_speed_mean": 0.0,
                        "yaw_abs_mean": 0.0,
                        "collision_fraction": 0.0,
                    },
                ]
                scan_result: dict[str, Any] = {
                    "evaluations": [
                        {
                            "checkpoint": 0,
                            "checkpoint_path": str(run_dir / "model_0.pt"),
                            "selection_score": 0.0,
                            "behavior_gate": {
                                "passed": False,
                                "scenarios": scenarios,
                            },
                        }
                    ]
                }
            else:
                run_dir = latest_run_dir(Path("logs") / run_name)
                checkpoints = phase_checkpoint_ids(run_dir, every=int(args.scan_every))
                scan_output = manifest_path.parent / f"{run_name}_scan.json"
                scan_command = build_scan_command(
                    uv_bin=str(args.uv_bin),
                    run_dir=run_dir,
                    checkpoints=checkpoints,
                    output=scan_output,
                    num_envs=int(args.scan_num_envs),
                    num_steps=int(args.scan_num_steps),
                    seed=int(args.scan_seed),
                    device=args.scan_device,
                    behavior_num_envs=int(args.behavior_num_envs),
                    behavior_num_steps=int(args.behavior_num_steps),
                    behavior_seed=int(args.behavior_seed),
                )
                _run(scan_command, dry_run=False)
                scan_result = json.loads(scan_output.read_text(encoding="utf-8"))
            selection = select_checkpoint(scan_result, phase)
            if archive_best is None or archive_key(selection) > archive_key(archive_best):
                archive_best = selection
            continuation, regression_guarded = choose_continuation_checkpoint(
                selection=selection,
                archive_best=archive_best,
                allow_coverage_regression=bool(args.allow_coverage_regression),
            )
            current_load_run = str(continuation["checkpoint_path"])
            phase_record = {
                "index": phase_index,
                "round": round_idx,
                "phase": phase.name,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "checkpoints": checkpoints,
                "continued_checkpoint_path": current_load_run,
                "coverage_regression_guarded": regression_guarded,
                "selected": {
                    key: value
                    for key, value in selection.items()
                    if key != "row"
                },
                "archive_best": {
                    key: value
                    for key, value in (archive_best or {}).items()
                    if key != "row"
                },
            }
            manifest["phases"].append(phase_record)
            manifest["latest_checkpoint"] = current_load_run
            manifest["archive_best_checkpoint"] = str(
                (archive_best or {}).get("checkpoint_path", "")
            )
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(
                f"[curriculum] selected {selection['checkpoint_path']} "
                f"continue={current_load_run} "
                f"pass_count={selection['behavior_pass_count']} "
                f"cost={selection['behavior_cost']:.4f} "
                f"passed={selection['passed_all_behavior_gates']} "
                f"guarded={regression_guarded}",
                flush=True,
            )
            if selection["passed_all_behavior_gates"]:
                manifest["completed_early"] = True
                manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                return manifest
    manifest["completed_early"] = False
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Initial checkpoint path.")
    parser.add_argument("--run-prefix", default="remote_auto_curriculum")
    parser.add_argument("--output", default="logs/remote_auto_curriculum/manifest.json")
    parser.add_argument("--phases", default="safety,clear,balanced")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--phase-iterations", type=int, default=75)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps-per-env", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--scan-every", type=int, default=25)
    parser.add_argument("--scan-num-envs", type=int, default=8)
    parser.add_argument("--scan-num-steps", type=int, default=96)
    parser.add_argument("--scan-seed", type=int, default=17)
    parser.add_argument("--scan-device", default="cpu")
    parser.add_argument("--behavior-num-envs", type=int, default=16)
    parser.add_argument("--behavior-num-steps", type=int, default=96)
    parser.add_argument("--behavior-seed", type=int, default=17)
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=None,
        help="Reset Gaussian action std after loading each phase checkpoint.",
    )
    parser.add_argument("--uv-bin", default=DEFAULT_UV_BIN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-coverage-regression",
        action="store_true",
        help="Continue from the phase-selected checkpoint even if it passes fewer behavior gates.",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    manifest = run_curriculum(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
