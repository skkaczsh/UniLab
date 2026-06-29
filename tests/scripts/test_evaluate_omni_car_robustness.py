from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_omni_car_robustness.py"
    spec = importlib.util.spec_from_file_location("evaluate_omni_car_robustness", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_robustness_scenarios_cover_zero_yaw_clear_front_and_walls() -> None:
    module = _load_module()

    scenarios = module.build_robustness_scenarios(directions=8)
    names = [scenario.name for scenario in scenarios]

    assert "zero_input_hold_broad" in names
    assert "yaw_positive_hold_broad" in names
    assert "yaw_negative_hold_broad" in names
    assert sum(name.startswith("clear_dir_") for name in names) == 8 * 3
    assert sum(name.startswith("front_blocked_dir_") for name in names) == 8
    assert sum("_wall_dir_" in name for name in names) == 16
    assert any(scenario.obstacle_type == ("box",) for scenario in scenarios)
    assert any(scenario.obstacle_type == ("wall",) for scenario in scenarios)


def test_directional_command_respects_axis_velocity_limits() -> None:
    module = _load_module()

    for index in range(16):
        command = module._directional_command(2.0 * 3.141592653589793 * index / 16, 1.0)
        assert abs(command[0]) <= module.MAX_X_SPEED + 1e-6
        assert abs(command[1]) <= module.MAX_Y_SPEED + 1e-6


def test_robustness_category_summary_counts_passes() -> None:
    module = _load_module()

    summary = module._summarize_categories(
        [
            {
                "scenario": "clear_dir_00_speed_0.25",
                "passed": True,
                "projection_mean": 0.4,
                "projection_ratio_mean": 0.6,
                "planar_speed_mean": 0.5,
                "collision_fraction": 0.0,
                "off_axis_abs_mean": 0.1,
            },
            {
                "scenario": "front_blocked_dir_00_circle",
                "passed": False,
                "projection_mean": 0.3,
                "projection_ratio_mean": 0.3,
                "planar_speed_mean": 0.4,
                "collision_fraction": 0.1,
                "off_axis_abs_mean": 0.2,
            },
        ]
    )

    assert summary["clear"]["count"] == 1
    assert summary["clear"]["passed"] == 1
    assert summary["front_blocked"]["count"] == 1
    assert summary["front_blocked"]["passed"] == 0
    assert summary["front_blocked"]["collision_fraction_max"] == 0.1
