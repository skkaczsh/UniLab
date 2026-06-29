from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_oracle_scenarios_include_directional_follow_slide_and_guards() -> None:
    module = _load_module()

    scenarios = {scenario.name: scenario for scenario in module.ORACLE_SCENARIOS}

    assert scenarios["zero_input_hold"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["front_blocked_stop"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["front_blocked_stop"].weight > scenarios["clear_forward_follow"].weight
    assert scenarios["clear_forward_follow"].target_action == (1.0, 0.0, 0.0)
    assert scenarios["clear_left_follow"].target_action == (0.0, 0.7, 0.0)
    assert scenarios["clear_right_follow"].target_action == (0.0, -0.7, 0.0)
    assert scenarios["clear_backward_follow"].target_action == (-0.8, 0.0, 0.0)
    assert scenarios["clear_slow_diagonal_follow"].target_action == (0.35, -0.25, 0.0)
    assert scenarios["yaw_only_follow"].target_action == (0.0, 0.0, 1.0)
    assert scenarios["right_wall_slide"].target_action[0] > 0.0
    assert scenarios["right_wall_slide"].target_action[1] > 0.0
    assert scenarios["right_wall_slide"].target_action[1] >= scenarios["right_wall_slide"].target_action[0]
    assert scenarios["right_wall_slide"].weight > scenarios["clear_forward_follow"].weight
    assert scenarios["left_wall_slide"].target_action[0] > 0.0
    assert scenarios["left_wall_slide"].target_action[1] < 0.0
    assert abs(scenarios["left_wall_slide"].target_action[1]) >= scenarios["left_wall_slide"].target_action[0]
    assert scenarios["left_wall_slide"].weight > scenarios["clear_forward_follow"].weight


def test_wall_slide_scenario_probabilities_are_normalized() -> None:
    module = _load_module()

    probabilities = module._scenario_probabilities(module.ORACLE_SCENARIOS)

    assert probabilities.shape == (len(module.ORACLE_SCENARIOS),)
    assert np.isclose(float(np.sum(probabilities)), 1.0)
    assert np.all(probabilities > 0.0)


def test_wall_slide_bc_can_filter_oracle_scenarios() -> None:
    module = _load_module()

    selected = module._selected_scenarios(["front_blocked_stop", "right_wall_slide"])
    probabilities = module._scenario_probabilities(selected)

    assert [scenario.name for scenario in selected] == [
        "front_blocked_stop",
        "right_wall_slide",
    ]
    assert probabilities.shape == (2,)
    assert np.isclose(float(np.sum(probabilities)), 1.0)


def test_behavior_correction_rolls_out_policy_actions_by_default() -> None:
    module = _load_module()

    args = module._parse_args(["--load-run", "model.pt", "--output", "corrected.pt"])

    assert args.rollout_actions == "policy"


def test_wall_slide_target_tensor_repeats_action() -> None:
    module = _load_module()
    scenario = module.ORACLE_SCENARIOS[-1]

    target = module._target_tensor(scenario, num_envs=3, device="cpu")

    assert target.shape == (3, 3)
    torch.testing.assert_close(
        target,
        torch.tensor([scenario.target_action] * 3, dtype=torch.float32),
    )


def test_balanced_batch_training_visits_every_oracle_scenario(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    policy = torch.nn.Linear(3, 3)
    env = SimpleNamespace(num_envs=2, close=lambda: None)

    class _WrappedEnv:
        def reset(self):  # type: ignore[no-untyped-def]
            return None

        def step(self, action):  # type: ignore[no-untyped-def]
            return (
                torch.zeros((2, 3), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.bool),
                {},
            )

    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))

    def _fake_make_runner(_args):  # type: ignore[no-untyped-def]
        return runner, env, _WrappedEnv(), tmp_path / "load.pt", "cpu"

    monkeypatch.setattr(module, "_make_runner", _fake_make_runner)
    monkeypatch.setattr(
        module,
        "_apply_scenario",
        lambda _env, _wrapped, _scenario: torch.zeros((2, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(module, "_save_corrected_checkpoint", lambda _runner, output: output.touch())

    summary = module.train_wall_slide_bc(
        SimpleNamespace(
            seed=1,
            learning_rate=1.0e-3,
            num_envs=2,
            iterations=1,
            rollout_steps=1,
            rollout_actions="policy",
            balanced_batch=True,
            scenario=None,
            progress_interval=0,
            dry_run=False,
            output=str(tmp_path / "corrected.pt"),
        )
    )

    assert summary["status"] == "completed"
    assert summary["balanced_batch"] is True
    assert summary["loss_initial"] is not None
    assert all(count == 1 for count in summary["scenario_counts"].values())
