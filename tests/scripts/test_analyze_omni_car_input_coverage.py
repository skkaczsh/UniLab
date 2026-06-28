from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "analyze_omni_car_input_coverage.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_omni_car_input_coverage", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_commands_reports_modes_and_bands() -> None:
    module = _load_module()
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, -0.4, 0.0],
            [0.8, 0.4, 1.2],
        ],
        dtype=np.float32,
    )

    summary = module.summarize_commands(
        commands,
        limits=np.asarray([2.0, 1.0, 2.0], dtype=np.float32),
        deadband=0.1,
    )

    assert summary["sample_count"] == 4
    assert summary["zero_fraction"] == pytest.approx(0.25)
    assert summary["mode_fraction"]["zero"] == pytest.approx(0.25)
    assert summary["mode_fraction"]["vx"] == pytest.approx(0.25)
    assert summary["mode_fraction"]["vy"] == pytest.approx(0.25)
    assert summary["mode_fraction"]["vx_vy_vyaw"] == pytest.approx(0.25)
    assert summary["planar_active_fraction"] == pytest.approx(0.75)


def test_summarize_hold_steps_reports_long_fraction() -> None:
    module = _load_module()

    summary = module.summarize_hold_steps(
        np.asarray([20, 40, 160, 400], dtype=np.int32),
        ctrl_dt=0.05,
    )

    assert summary["mean_s"] == pytest.approx(7.75)
    assert summary["long_ge_8s_fraction"] == pytest.approx(0.5)
    assert summary["very_long_ge_20s_fraction"] == pytest.approx(0.25)
