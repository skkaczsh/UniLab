#!/usr/bin/env python3
"""Evaluate multiple OmniCar checkpoints and rank them against reference gates."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.evaluate_omni_car_behaviors as evaluate_omni_car_behaviors
import scripts.evaluate_omni_car_checkpoint as evaluate_omni_car_checkpoint

_CHECKPOINT_RE = re.compile(r"^(?:model_)?(?P<id>\d+)(?:\.pt)?$")


def checkpoint_id_from_token(token: str) -> int:
    match = _CHECKPOINT_RE.match(str(token).strip())
    if not match:
        raise ValueError(f"Invalid checkpoint token: {token!r}")
    return int(match.group("id"))


def expand_checkpoint_specs(specs: Sequence[str]) -> list[int]:
    checkpoint_ids: list[int] = []
    for raw_spec in specs:
        spec = str(raw_spec).strip()
        if not spec:
            continue
        if ":" not in spec:
            checkpoint_ids.append(checkpoint_id_from_token(spec))
            continue
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid checkpoint range: {spec!r}")
        start = checkpoint_id_from_token(parts[0])
        stop = checkpoint_id_from_token(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step <= 0:
            raise ValueError(f"Checkpoint range step must be positive: {spec!r}")
        if stop < start:
            raise ValueError(f"Checkpoint range stop must be >= start: {spec!r}")
        checkpoint_ids.extend(range(start, stop + 1, step))
    return sorted(set(checkpoint_ids))


def discover_checkpoint_ids(run_dir: Path) -> list[int]:
    if not run_dir.is_dir():
        raise ValueError(f"Checkpoint discovery requires a run directory: {run_dir}")
    ids: list[int] = []
    for path in run_dir.glob("model_*.pt"):
        try:
            ids.append(checkpoint_id_from_token(path.name))
        except ValueError:
            continue
    if not ids:
        raise FileNotFoundError(f"No model_*.pt checkpoints found under {run_dir}")
    return sorted(set(ids))


def thin_checkpoint_ids(ids: Sequence[int], *, every: int | None, include_last: bool = True) -> list[int]:
    unique_ids = sorted(set(int(item) for item in ids))
    if every is None or every <= 1:
        return unique_ids
    selected = [item for item in unique_ids if item % every == 0]
    if include_last and unique_ids[-1] not in selected:
        selected.append(unique_ids[-1])
    return sorted(set(selected))


def load_reference_manifest(manifest_path: Path) -> dict[str, float | str]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Reference manifest must be a JSON object: {manifest_path}")
    evaluation = manifest.get("evaluation")
    aggregate = evaluation.get("aggregate") if isinstance(evaluation, dict) else None
    if not isinstance(aggregate, dict):
        raise ValueError(f"Reference manifest missing evaluation.aggregate: {manifest_path}")
    collision = aggregate.get("collision_fraction_mean")
    tracking = aggregate.get("tracking_error_mean")
    if not isinstance(collision, int | float):
        raise ValueError(
            f"Reference manifest missing numeric aggregate.collision_fraction_mean: {manifest_path}"
        )
    if not isinstance(tracking, int | float):
        raise ValueError(
            f"Reference manifest missing numeric aggregate.tracking_error_mean: {manifest_path}"
        )
    return {
        "source": str(manifest_path),
        "collision": float(collision),
        "tracking": float(tracking),
    }


def _float_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Evaluation summary missing numeric metric {key!r}")
    return float(value)


def candidate_score(
    summary: dict[str, Any],
    *,
    collision_weight: float,
    tracking_weight: float,
    jerk_weight: float,
) -> float:
    jerk_sum = (
        _float_metric(summary, "omni_car/vx_jerk_cost")
        + _float_metric(summary, "omni_car/vy_jerk_cost")
        + _float_metric(summary, "omni_car/vyaw_jerk_cost")
    )
    return (
        collision_weight * _float_metric(summary, "collision_fraction")
        + tracking_weight * _float_metric(summary, "omni_car/tracking_error")
        + jerk_weight * jerk_sum
    )


def reference_gate(
    summary: dict[str, Any],
    *,
    reference_collision: float | None,
    reference_tracking: float | None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {"enabled": reference_collision is not None or reference_tracking is not None}
    if reference_collision is not None:
        collision = _float_metric(summary, "collision_fraction")
        gate["reference_collision"] = float(reference_collision)
        gate["collision"] = collision
        gate["collision_delta"] = collision - float(reference_collision)
        gate["collision_passed"] = collision <= float(reference_collision)
    if reference_tracking is not None:
        tracking = _float_metric(summary, "omni_car/tracking_error")
        gate["reference_tracking"] = float(reference_tracking)
        gate["tracking"] = tracking
        gate["tracking_delta"] = tracking - float(reference_tracking)
        gate["tracking_passed"] = tracking <= float(reference_tracking)
    checked = [value for key, value in gate.items() if key.endswith("_passed")]
    gate["passed"] = bool(checked) and all(bool(value) for value in checked)
    return gate


def compact_summary(checkpoint: int, summary: dict[str, Any], score: float) -> dict[str, Any]:
    keys = [
        "checkpoint_path",
        "collision_fraction",
        "mean_episode_return",
        "mean_episode_length",
        "mean_step_reward",
        "omni_car/tracking_error",
        "omni_car/response_progress",
        "omni_car/command_clearance",
        "omni_car/command_safety_gate",
        "omni_car/mean_clearance",
        "vx_tracking_mae",
        "vy_tracking_mae",
        "vyaw_tracking_mae",
        "omni_car/vx_jerk_cost",
        "omni_car/vy_jerk_cost",
        "omni_car/vyaw_jerk_cost",
        "omni_car/reward/blocked_stop",
        "omni_car/reward/blocked_motion",
        "omni_car/reward/idle_stop",
        "omni_car/reward/off_axis",
        "omni_car/reward/reverse",
        "omni_car/reward/total",
    ]
    row: dict[str, Any] = {"checkpoint": int(checkpoint), "selection_score": float(score)}
    for key in keys:
        if key in summary:
            row[key] = summary[key]
    return row


Evaluator = Callable[[argparse.Namespace], dict[str, Any]]


def _evaluate_checkpoint(
    evaluator: Evaluator,
    args: argparse.Namespace,
    *,
    verbose: bool,
) -> dict[str, Any]:
    if verbose:
        return evaluator(args)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return evaluator(args)


def compact_behavior_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(summary.get("strict_passed")),
        "scenarios": [
            {
                "scenario": item.get("scenario"),
                "passed": bool(item.get("passed")),
                "failures": list(item.get("failures", [])),
                "planar_speed_mean": item.get("planar_speed_mean"),
                "yaw_abs_mean": item.get("yaw_abs_mean"),
                "projection_mean": item.get("projection_mean"),
                "off_axis_abs_mean": item.get("off_axis_abs_mean"),
                "collision_fraction": item.get("collision_fraction"),
                "command_safety_gate_mean": item.get("command_safety_gate_mean"),
            }
            for item in summary.get("scenarios", [])
            if isinstance(item, dict)
        ],
    }


def _row_passes_all_gates(row: dict[str, Any], *, behavior_gate_enabled: bool) -> bool:
    reference = row.get("reference_gate")
    if isinstance(reference, dict) and bool(reference.get("enabled")):
        if reference.get("passed") is not True:
            return False
    if behavior_gate_enabled:
        behavior = row.get("behavior_gate")
        if not isinstance(behavior, dict) or behavior.get("passed") is not True:
            return False
    return True


def scan_checkpoints(
    *,
    load_run: str,
    checkpoints: Sequence[int],
    num_envs: int,
    num_steps: int,
    seed: int,
    device: str | None,
    collision_weight: float,
    tracking_weight: float,
    jerk_weight: float,
    reference_collision: float | None,
    reference_tracking: float | None,
    reference_source: str | None = None,
    evaluator: Evaluator = evaluate_omni_car_checkpoint.evaluate_checkpoint,
    behavior_gate: bool = False,
    behavior_num_envs: int = 16,
    behavior_num_steps: int = 96,
    behavior_seed: int | None = None,
    behavior_evaluator: Evaluator = evaluate_omni_car_behaviors.evaluate_behaviors,
    verbose: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        args = argparse.Namespace(
            load_run=load_run,
            checkpoint=str(checkpoint),
            num_envs=int(num_envs),
            num_steps=int(num_steps),
            seed=int(seed),
            device=device,
            json=True,
        )
        summary = _evaluate_checkpoint(evaluator, args, verbose=verbose)
        score = candidate_score(
            summary,
            collision_weight=collision_weight,
            tracking_weight=tracking_weight,
            jerk_weight=jerk_weight,
        )
        row = compact_summary(int(checkpoint), summary, score)
        row["reference_gate"] = reference_gate(
            summary,
            reference_collision=reference_collision,
            reference_tracking=reference_tracking,
        )
        if behavior_gate:
            behavior_args = argparse.Namespace(
                load_run=load_run,
                checkpoint=str(checkpoint),
                num_envs=int(behavior_num_envs),
                num_steps=int(behavior_num_steps),
                seed=int(seed if behavior_seed is None else behavior_seed),
                device=device,
                json=True,
                strict=True,
            )
            behavior_summary = _evaluate_checkpoint(
                behavior_evaluator,
                behavior_args,
                verbose=verbose,
            )
            row["behavior_gate"] = compact_behavior_summary(behavior_summary)
        rows.append(row)
    if not rows:
        raise ValueError("No checkpoints selected for scanning.")
    best = min(rows, key=lambda row: float(row["selection_score"]))
    gated = [row for row in rows if row["reference_gate"].get("passed") is True]
    all_gated = [
        row for row in rows if _row_passes_all_gates(row, behavior_gate_enabled=behavior_gate)
    ]
    return {
        "load_run": load_run,
        "checkpoints": [int(item) for item in checkpoints],
        "num_envs": int(num_envs),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "device": device,
        "selection_weights": {
            "collision": float(collision_weight),
            "tracking": float(tracking_weight),
            "jerk": float(jerk_weight),
        },
        "reference": {
            "source": reference_source,
            "collision": reference_collision,
            "tracking": reference_tracking,
        },
        "behavior_gate": {
            "enabled": bool(behavior_gate),
            "num_envs": int(behavior_num_envs),
            "num_steps": int(behavior_num_steps),
            "seed": int(seed if behavior_seed is None else behavior_seed),
        },
        "best_by_score": best,
        "best_passing_reference_gate": min(gated, key=lambda row: float(row["selection_score"]))
        if gated
        else None,
        "best_passing_all_gates": min(all_gated, key=lambda row: float(row["selection_score"]))
        if all_gated
        else None,
        "evaluations": rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Run directory or checkpoint path.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Checkpoint ids, filenames, or inclusive ranges like 500:1500:250.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover model_*.pt checkpoints from --load-run when --checkpoints is omitted.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=None,
        help="When discovering, keep checkpoints divisible by this value and always include the last.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--collision-weight", type=float, default=10.0)
    parser.add_argument("--tracking-weight", type=float, default=1.0)
    parser.add_argument("--jerk-weight", type=float, default=0.25)
    parser.add_argument(
        "--behavior-gate",
        action="store_true",
        help="Also run controlled zero-input/obstacle behavior probes for each checkpoint.",
    )
    parser.add_argument("--behavior-num-envs", type=int, default=16)
    parser.add_argument("--behavior-num-steps", type=int, default=96)
    parser.add_argument(
        "--behavior-seed",
        type=int,
        default=None,
        help="Seed for behavior probes. Defaults to --seed.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help=(
            "Best-checkpoint manifest whose evaluation.aggregate values define "
            "reference collision/tracking gates."
        ),
    )
    parser.add_argument("--reference-collision", type=float, default=None)
    parser.add_argument("--reference-tracking", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Do not suppress evaluator model/debug logs while scanning.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected checkpoints and exit.")
    return parser.parse_args(argv)


def _selected_checkpoints(args: argparse.Namespace) -> list[int]:
    if args.checkpoints:
        return expand_checkpoint_specs(args.checkpoints)
    if args.discover:
        return thin_checkpoint_ids(
            discover_checkpoint_ids(Path(str(args.load_run))),
            every=args.every,
            include_last=True,
        )
    raise ValueError("Provide --checkpoints or use --discover with a run directory.")


def _resolve_references(args: argparse.Namespace) -> tuple[float | None, float | None, str | None]:
    reference_collision = args.reference_collision
    reference_tracking = args.reference_tracking
    reference_source = None
    if args.reference_manifest is not None:
        manifest_reference = load_reference_manifest(args.reference_manifest)
        reference_source = str(manifest_reference["source"])
        if reference_collision is None:
            reference_collision = float(manifest_reference["collision"])
        if reference_tracking is None:
            reference_tracking = float(manifest_reference["tracking"])
    return reference_collision, reference_tracking, reference_source


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoints = _selected_checkpoints(args)
    if args.dry_run:
        print(json.dumps({"load_run": args.load_run, "checkpoints": checkpoints}, indent=2))
        return 0
    reference_collision, reference_tracking, reference_source = _resolve_references(args)
    result = scan_checkpoints(
        load_run=str(args.load_run),
        checkpoints=checkpoints,
        num_envs=int(args.num_envs),
        num_steps=int(args.num_steps),
        seed=int(args.seed),
        device=args.device,
        collision_weight=float(args.collision_weight),
        tracking_weight=float(args.tracking_weight),
        jerk_weight=float(args.jerk_weight),
        reference_collision=reference_collision,
        reference_tracking=reference_tracking,
        reference_source=reference_source,
        behavior_gate=bool(args.behavior_gate),
        behavior_num_envs=int(args.behavior_num_envs),
        behavior_num_steps=int(args.behavior_num_steps),
        behavior_seed=args.behavior_seed,
        verbose=bool(args.verbose),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
