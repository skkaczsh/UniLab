from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_omni_car_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_omni_car_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_accumulator_tracks_axis_metrics_and_episode_stats() -> None:
    module = _load_module()
    acc = module.OmniCarEvalAccumulator(num_envs=2)

    acc.update(
        rewards=np.asarray([1.0, 2.0], dtype=np.float32),
        dones=np.asarray([False, True]),
        commands=np.asarray([[1.0, 0.0, 0.5], [0.0, -1.0, -0.5]], dtype=np.float32),
        actions=np.asarray([[0.5, 0.0, 0.0], [0.0, -0.5, -1.0]], dtype=np.float32),
        collisions=np.asarray([0.0, 1.0], dtype=np.float32),
        nearest_clearance=np.asarray([0.3, -0.1], dtype=np.float32),
        step_logs={
            "omni_car/tracking_error": 0.8,
            "omni_car/response_progress": 0.05,
            "omni_car/vx_diff_cost": 0.2,
        },
    )
    acc.update(
        rewards=np.asarray([3.0, 4.0], dtype=np.float32),
        dones=np.asarray([True, False]),
        commands=np.asarray([[1.0, 1.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32),
        actions=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        collisions=np.asarray([0.0, 0.0], dtype=np.float32),
        nearest_clearance=np.asarray([0.2, 0.4], dtype=np.float32),
        step_logs={
            "omni_car/tracking_error": 0.4,
            "omni_car/response_progress": 0.10,
            "omni_car/vx_diff_cost": 0.1,
        },
    )

    summary = acc.finalize()
    assert summary["num_steps"] == 2
    assert summary["env_steps"] == 4
    assert summary["episodes_completed"] == 2
    assert summary["mean_step_reward"] == 2.5
    assert summary["collision_fraction"] == 0.25
    assert summary["min_clearance_observed"] == pytest.approx(-0.1)
    assert summary["mean_episode_return"] == 3.0
    assert summary["mean_episode_length"] == 1.5
    assert summary["vx_tracking_mae"] == 0.125
    assert summary["vy_tracking_mae"] == 0.375
    assert summary["vyaw_tracking_mae"] == 0.375
    assert summary["omni_car/tracking_error"] == pytest.approx(0.6)
    assert summary["omni_car/response_progress"] == pytest.approx(0.075)
    assert summary["omni_car/vx_diff_cost"] == pytest.approx(0.15)
