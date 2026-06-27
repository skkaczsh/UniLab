from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "scan_omni_car_checkpoints.py"
    spec = importlib.util.spec_from_file_location("scan_omni_car_checkpoints", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(*, collision: float, tracking: float, jerk: float, ret: float = 1.0) -> dict[str, object]:
    return {
        "checkpoint_path": "/tmp/model.pt",
        "collision_fraction": collision,
        "mean_episode_return": ret,
        "mean_episode_length": 10.0,
        "mean_step_reward": 1.0,
        "omni_car/tracking_error": tracking,
        "omni_car/response_progress": 0.02,
        "omni_car/mean_clearance": 1.0,
        "vx_tracking_mae": tracking,
        "vy_tracking_mae": tracking,
        "vyaw_tracking_mae": tracking,
        "omni_car/vx_jerk_cost": jerk,
        "omni_car/vy_jerk_cost": jerk,
        "omni_car/vyaw_jerk_cost": jerk,
    }


def test_expand_checkpoint_specs_accepts_ids_filenames_and_ranges() -> None:
    module = _load_module()

    assert module.expand_checkpoint_specs(["model_500.pt", "900", "1200:1800:300"]) == [
        500,
        900,
        1200,
        1500,
        1800,
    ]


def test_discover_checkpoint_ids_and_thinning_include_last(tmp_path: Path) -> None:
    module = _load_module()
    for name in ["model_0.pt", "model_100.pt", "model_299.pt", "notes.txt"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    ids = module.discover_checkpoint_ids(tmp_path)

    assert ids == [0, 100, 299]
    assert module.thin_checkpoint_ids(ids, every=100) == [0, 100, 299]


def test_reference_gate_requires_all_provided_metrics_to_pass() -> None:
    module = _load_module()
    summary = _summary(collision=0.03, tracking=0.4, jerk=0.1)

    gate = module.reference_gate(
        summary,
        reference_collision=0.04,
        reference_tracking=0.35,
    )

    assert gate["collision_passed"] is True
    assert gate["tracking_passed"] is False
    assert gate["passed"] is False


def test_scan_checkpoints_ranks_and_reports_reference_gate() -> None:
    module = _load_module()
    summaries = {
        "100": _summary(collision=0.05, tracking=0.50, jerk=0.20, ret=10.0),
        "200": _summary(collision=0.02, tracking=0.35, jerk=0.10, ret=20.0),
    }

    def _fake_evaluator(args: Namespace) -> dict[str, object]:
        return summaries[str(args.checkpoint)]

    result = module.scan_checkpoints(
        load_run="/tmp/run",
        checkpoints=[100, 200],
        num_envs=4,
        num_steps=8,
        seed=7,
        device="cpu",
        collision_weight=10.0,
        tracking_weight=1.0,
        jerk_weight=0.25,
        reference_collision=0.03,
        reference_tracking=0.40,
        evaluator=_fake_evaluator,
    )

    assert result["best_by_score"]["checkpoint"] == 200
    assert result["best_passing_reference_gate"]["checkpoint"] == 200
    assert result["evaluations"][0]["reference_gate"]["passed"] is False
    assert result["evaluations"][1]["reference_gate"]["passed"] is True


def test_scan_checkpoints_suppresses_noisy_evaluator_output_by_default(capsys) -> None:
    module = _load_module()

    def _noisy_evaluator(args: Namespace) -> dict[str, object]:
        print(f"loading model for checkpoint {args.checkpoint}")
        return _summary(collision=0.02, tracking=0.30, jerk=0.10)

    module.scan_checkpoints(
        load_run="/tmp/run",
        checkpoints=[100],
        num_envs=4,
        num_steps=8,
        seed=7,
        device="cpu",
        collision_weight=10.0,
        tracking_weight=1.0,
        jerk_weight=0.25,
        reference_collision=0.03,
        reference_tracking=0.40,
        evaluator=_noisy_evaluator,
    )

    assert capsys.readouterr().out == ""
