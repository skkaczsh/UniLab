from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
    assert scenarios["yaw_negative_follow"].target_action == (0.0, 0.0, -1.0)
    assert scenarios["zero_input_hold"].weight > scenarios["clear_forward_follow"].weight
    assert scenarios["yaw_only_follow"].weight > scenarios["clear_forward_follow"].weight
    assert scenarios["yaw_negative_follow"].weight == scenarios["yaw_only_follow"].weight
    assert scenarios["right_wall_slide"].target_action[0] > 0.0
    assert scenarios["right_wall_slide"].target_action[1] > 0.0
    assert scenarios["right_wall_slide"].target_action[0] > scenarios["right_wall_slide"].target_action[1]
    assert scenarios["right_wall_slide"].weight > scenarios["clear_forward_follow"].weight
    assert scenarios["left_wall_slide"].target_action[0] > 0.0
    assert scenarios["left_wall_slide"].target_action[1] < 0.0
    assert scenarios["left_wall_slide"].target_action[0] > abs(
        scenarios["left_wall_slide"].target_action[1]
    )
    assert scenarios["left_wall_slide"].weight > scenarios["clear_forward_follow"].weight
    assert "directional_clear_00_slow" in scenarios
    assert "directional_front_stop_00_circle" in scenarios
    assert "directional_right_wall_00" in scenarios
    assert scenarios["directional_left_wall_04"].target_action[1] > abs(
        scenarios["directional_left_wall_04"].target_action[0]
    )
    assert scenarios["directional_left_wall_04"].weight == scenarios["left_wall_slide"].weight
    assert "directional32_clear_03_mid" in scenarios
    assert "directional32_front_stop_03_circle" in scenarios
    assert "directional32_left_wall_04" in scenarios
    assert "max_stick_front_blocked_dir_00_circle" in scenarios
    assert "max_stick_front_blocked_dir_04_box" in scenarios
    assert "max_stick_left_wall_dir_04" in scenarios
    assert scenarios["max_stick_front_blocked_dir_00_circle"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["max_stick_front_blocked_dir_04_box"].target_action == (0.0, 0.0, 0.0)
    assert scenarios["max_stick_left_wall_dir_04"].target_action[0] < 0.0
    assert scenarios["max_stick_left_wall_dir_04"].target_action[1] > 0.0
    assert len(scenarios) == len(module.ORACLE_SCENARIOS)


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


def test_wall_slide_bc_can_filter_oracle_scenario_groups() -> None:
    module = _load_module()

    selected = module._selected_scenarios(None, ["directional_front"])
    names = [scenario.name for scenario in selected]

    assert names
    assert all(name.startswith("directional_front_stop_") for name in names)
    assert any(name.endswith("_box") for name in names)
    assert any(name.endswith("_wall") for name in names)


def test_wall_slide_bc_can_filter_dense_directional_scenario_groups() -> None:
    module = _load_module()

    selected = module._selected_scenarios(None, ["directional32_front"])
    names = [scenario.name for scenario in selected]

    assert len(names) == 32
    assert all(name.startswith("directional32_front_stop_") for name in names)
    assert "directional32_front_stop_03_circle" in names


def test_wall_slide_bc_can_filter_max_stick_scenario_groups() -> None:
    module = _load_module()

    selected = module._selected_scenarios(None, ["max_stick_front", "max_stick_wall"])
    names = [scenario.name for scenario in selected]

    assert "max_stick_front_blocked_dir_00_circle" in names
    assert "max_stick_front_blocked_dir_04_box" in names
    assert "max_stick_left_wall_dir_04" in names
    assert all(
        name.startswith("max_stick_front_blocked_") or "_wall_dir_" in name
        for name in names
    )


def test_wall_slide_bc_can_weight_selected_repair_scenario() -> None:
    module = _load_module()

    selected = module._selected_scenarios(
        ["front_blocked_stop", "directional_left_wall_04"],
        scenario_weight=["directional_left_wall_04=5"],
    )
    by_name = {scenario.name: scenario for scenario in selected}

    assert by_name["directional_left_wall_04"].weight == module.WALL_SLIDE_WEIGHT * 5.0
    assert by_name["front_blocked_stop"].weight == 12.0


def test_wall_slide_bc_can_override_selected_target_action() -> None:
    module = _load_module()

    selected = module._selected_scenarios(
        ["max_stick_right_wall_dir_07"],
        scenario_target=["max_stick_right_wall_dir_07=0.42,0.07,0.0"],
    )

    assert selected[0].target_action == (0.42, 0.07, 0.0)


def test_wall_slide_bc_rejects_unselected_target_override() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="not selected"):
        module._selected_scenarios(
            ["max_stick_right_wall_dir_07"],
            scenario_target=["max_stick_front_blocked_dir_00_circle=0.0,0.0,0.0"],
        )


def test_behavior_correction_rolls_out_policy_actions_by_default() -> None:
    module = _load_module()

    args = module._parse_args(["--load-run", "model.pt", "--output", "corrected.pt"])

    assert args.rollout_actions == "policy"


def test_dry_run_reports_filtered_scenarios(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    policy = torch.nn.Linear(3, 3)
    env = SimpleNamespace(num_envs=2, close=lambda: None)

    class _WrappedEnv:
        def reset(self):  # type: ignore[no-untyped-def]
            return None

    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))

    def _fake_make_runner(_args):  # type: ignore[no-untyped-def]
        return runner, env, _WrappedEnv(), tmp_path / "load.pt", "cpu"

    monkeypatch.setattr(module, "_make_runner", _fake_make_runner)
    monkeypatch.setattr(
        module,
        "_apply_scenario",
        lambda _env, _wrapped, _scenario: torch.zeros((2, 3), dtype=torch.float32),
    )

    summary = module.train_wall_slide_bc(
        SimpleNamespace(
            seed=1,
            learning_rate=1.0e-3,
            num_envs=2,
            iterations=1,
            rollout_steps=1,
            rollout_actions="policy",
            balanced_batch=False,
            scenario=["max_stick_front_blocked_dir_00_circle"],
            scenario_group=None,
            scenario_weight=None,
            progress_interval=0,
            dry_run=True,
            output=str(tmp_path / "corrected.pt"),
        )
    )

    assert summary["status"] == "dry_run"
    assert summary["scenarios"] == ["max_stick_front_blocked_dir_00_circle"]


def test_wall_slide_target_tensor_repeats_action() -> None:
    module = _load_module()
    scenario = module.ORACLE_SCENARIOS[-1]

    target = module._target_tensor(scenario, num_envs=3, device="cpu")

    assert target.shape == (3, 3)
    torch.testing.assert_close(
        target,
        torch.tensor([scenario.target_action] * 3, dtype=torch.float32),
    )


def test_branch_supervision_targets_front_stop_and_wall_escape() -> None:
    module = _load_module()

    class _Policy(torch.nn.Module):
        def branch_action_outputs(self, obs):  # type: ignore[no-untyped-def]
            batch = obs.shape[0]
            return (
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.ones((batch, 3), dtype=torch.float32),
                torch.full((batch, 1), 0.5, dtype=torch.float32),
            )

    obs = torch.zeros((2, 4), dtype=torch.float32)
    stop_target = torch.zeros((2, 3), dtype=torch.float32)
    escape_target = torch.ones((2, 3), dtype=torch.float32)

    front_loss = module._branch_supervision_loss(
        policy=_Policy(),
        obs=obs,
        target=stop_target,
        scenario_name="max_stick_front_blocked_dir_00_circle",
        action_weight=1.0,
        gate_weight=1.0,
    )
    wall_loss = module._branch_supervision_loss(
        policy=_Policy(),
        obs=obs,
        target=escape_target,
        scenario_name="max_stick_right_wall_dir_07",
        action_weight=1.0,
        gate_weight=1.0,
    )
    clear_loss = module._branch_supervision_loss(
        policy=_Policy(),
        obs=obs,
        target=escape_target,
        scenario_name="max_stick_clear_dir_07",
        action_weight=1.0,
        gate_weight=1.0,
    )

    assert front_loss is not None
    assert wall_loss is not None
    assert clear_loss is None
    assert float(front_loss) > 0.0
    assert float(wall_loss) > 0.0


def test_branch_supervision_can_keep_clear_behavior_on_both_heads() -> None:
    module = _load_module()

    class _Policy(torch.nn.Module):
        def branch_action_outputs(self, obs):  # type: ignore[no-untyped-def]
            batch = obs.shape[0]
            return (
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.ones((batch, 3), dtype=torch.float32),
                torch.full((batch, 1), 0.5, dtype=torch.float32),
            )

    obs = torch.zeros((2, 4), dtype=torch.float32)
    target = torch.full((2, 3), 0.5, dtype=torch.float32)

    default_loss = module._branch_supervision_loss(
        policy=_Policy(),
        obs=obs,
        target=target,
        scenario_name="max_stick_clear_dir_07",
        action_weight=1.0,
        gate_weight=1.0,
    )
    clear_loss = module._branch_supervision_loss(
        policy=_Policy(),
        obs=obs,
        target=target,
        scenario_name="max_stick_clear_dir_07",
        action_weight=1.0,
        gate_weight=1.0,
        supervise_clear_heads=True,
    )

    assert default_loss is None
    assert clear_loss is not None
    assert float(clear_loss) > 0.0


def test_branch_heads_only_freezes_shared_policy_body() -> None:
    module = _load_module()

    class _Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = torch.nn.Linear(3, 4)
            self.stop_action_head = torch.nn.Linear(4, 3)
            self.escape_action_head = torch.nn.Linear(4, 3)
            self.action_gate_head = torch.nn.Linear(4, 1)

    policy = _Policy()

    trainable = module._set_branch_heads_only_trainable(policy)

    assert trainable == sum(
        param.numel()
        for module_part in (
            policy.stop_action_head,
            policy.escape_action_head,
            policy.action_gate_head,
        )
        for param in module_part.parameters()
    )
    assert not any(param.requires_grad for param in policy.shared.parameters())
    assert all(param.requires_grad for param in policy.stop_action_head.parameters())
    assert all(param.requires_grad for param in policy.escape_action_head.parameters())
    assert all(param.requires_grad for param in policy.action_gate_head.parameters())


def test_dagger_replay_buffer_caps_and_samples() -> None:
    module = _load_module()
    rng = np.random.default_rng(3)
    replay = module.DaggerReplayBuffer(max_samples=3)

    replay.append(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.ones((4, 3), dtype=torch.float32),
        weight=2.0,
        rng=rng,
        samples_per_step=4,
    )

    assert replay.size == 3
    obs, target, weight = replay.sample(rng=rng, batch_size=2, device="cpu")
    assert obs.shape == (2, 3)
    assert target.shape == (2, 3)
    assert weight.shape == (2,)
    torch.testing.assert_close(weight, torch.full((2,), 2.0))


def test_wall_slide_bc_can_jitter_obstacle_context_without_changing_command() -> None:
    module = _load_module()
    rng = np.random.default_rng(4)
    scenario = {item.name: item for item in module.ORACLE_SCENARIOS}["max_stick_right_wall_dir_07"]
    target = scenario.target_action

    behavior = module._training_behavior(
        scenario,
        rng=rng,
        xy_std=0.05,
        radius_std=0.01,
        yaw_std=0.02,
    )

    assert behavior.command == scenario.behavior.command
    assert behavior.obstacle_xy != scenario.behavior.obstacle_xy
    assert behavior.obstacle_radius != scenario.behavior.obstacle_radius
    assert behavior.obstacle_yaw != scenario.behavior.obstacle_yaw
    assert scenario.target_action == target


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
            scenario_group=None,
            scenario_weight=None,
            progress_interval=0,
            dry_run=False,
            output=str(tmp_path / "corrected.pt"),
        )
    )

    assert summary["status"] == "completed"
    assert summary["balanced_batch"] is True
    assert summary["loss_initial"] is not None
    assert all(count == 1 for count in summary["scenario_counts"].values())


def test_dagger_replay_training_reports_replay_updates(monkeypatch, tmp_path: Path) -> None:
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
            dagger_replay_epochs=1,
            dagger_replay_batch_size=4,
            dagger_replay_max_samples=8,
            dagger_samples_per_step=1,
            scenario=["zero_input_hold", "front_blocked_stop"],
            scenario_group=None,
            scenario_weight=None,
            scenario_target=None,
            progress_interval=0,
            dry_run=False,
            output=str(tmp_path / "corrected.pt"),
        )
    )

    assert summary["status"] == "completed"
    assert summary["replay_size"] == 2
    assert summary["replay_updates"] == 1
    assert summary["replay_loss_final"] is not None
