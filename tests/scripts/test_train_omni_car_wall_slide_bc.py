from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_omni_car_wall_slide_bc.py"
    spec = importlib.util.spec_from_file_location("train_omni_car_wall_slide_bc", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wall_slide_oracle_scenarios_include_bidirectional_slide_and_guards() -> None:
    module = _load_module()

    scenarios = {scenario.name: scenario for scenario in module.ORACLE_SCENARIOS}

    assert scenarios["zero_input_hold"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["front_blocked_stop"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["right_wall_slide"].target_action[0] > 0.0
    assert scenarios["right_wall_slide"].target_action[1] > 0.0
    assert scenarios["left_wall_slide"].target_action[0] > 0.0
    assert scenarios["left_wall_slide"].target_action[1] < 0.0


def test_wall_slide_scenario_probabilities_are_normalized() -> None:
    module = _load_module()

    probabilities = module._scenario_probabilities()

    assert probabilities.shape == (len(module.ORACLE_SCENARIOS),)
    assert np.isclose(float(np.sum(probabilities)), 1.0)
    assert np.all(probabilities > 0.0)


def test_wall_slide_target_tensor_repeats_action() -> None:
    module = _load_module()
    scenario = module.ORACLE_SCENARIOS[-1]

    target = module._target_tensor(scenario, num_envs=3, device="cpu")

    assert target.shape == (3, 3)
    torch.testing.assert_close(
        target,
        torch.tensor([scenario.target_action] * 3, dtype=torch.float32),
    )
