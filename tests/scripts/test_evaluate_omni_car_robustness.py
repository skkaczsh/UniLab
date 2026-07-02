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


def test_max_stick_scenarios_cover_clear_blocked_and_side_walls() -> None:
    module = _load_module()

    scenarios = module.build_max_stick_scenarios(directions=8)
    names = [scenario.name for scenario in scenarios]

    assert sum(name.startswith("max_stick_clear_dir_") for name in names) == 8
    assert sum(name.startswith("max_stick_front_blocked_dir_") for name in names) == 8
    assert sum("_wall_dir_" in name for name in names) == 16
    for scenario in scenarios:
        assert abs(scenario.command[0]) <= module.MAX_X_SPEED + 1e-6
        assert abs(scenario.command[1]) <= module.MAX_Y_SPEED + 1e-6


def test_build_scenarios_selects_max_stick_suite() -> None:
    module = _load_module()

    scenarios = module.build_scenarios("max_stick", directions=4)

    assert scenarios
    assert all(scenario.name.startswith("max_stick_") for scenario in scenarios)


def test_filter_scenarios_selects_exact_names() -> None:
    module = _load_module()

    scenarios = module.build_max_stick_scenarios(directions=8)
    selected = module._filter_scenarios(scenarios, ["max_stick_right_wall_dir_07"])

    assert [scenario.name for scenario in selected] == ["max_stick_right_wall_dir_07"]


def test_parse_constant_action_validates_limits() -> None:
    module = _load_module()

    assert module._parse_action("0.4,-0.08,0") == (0.4, -0.08, 0.0)

    try:
        module._parse_action("3.0,0,0")
    except ValueError as exc:
        assert "physical limits" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


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


def test_robustness_category_summary_counts_max_stick_categories() -> None:
    module = _load_module()

    summary = module._summarize_categories(
        [
            {
                "scenario": "max_stick_clear_dir_00",
                "passed": True,
                "projection_mean": 0.8,
                "projection_ratio_mean": 0.4,
                "planar_speed_mean": 0.9,
                "collision_fraction": 0.0,
                "off_axis_abs_mean": 0.2,
            },
            {
                "scenario": "max_stick_front_blocked_dir_00_circle",
                "passed": False,
                "projection_mean": 0.4,
                "projection_ratio_mean": 0.2,
                "planar_speed_mean": 0.6,
                "collision_fraction": 0.05,
                "off_axis_abs_mean": 0.1,
            },
            {
                "scenario": "max_stick_left_wall_dir_00",
                "passed": True,
                "projection_mean": 0.4,
                "projection_ratio_mean": 0.2,
                "planar_speed_mean": 0.6,
                "collision_fraction": 0.01,
                "off_axis_abs_mean": 0.1,
            },
        ]
    )

    assert summary["max_stick_clear"]["count"] == 1
    assert summary["max_stick_front_blocked"]["passed"] == 0
    assert summary["max_stick_side_wall"]["collision_fraction_max"] == 0.01
