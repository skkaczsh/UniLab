from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from unilab.algos.torch.omni_car import OmniCarGridCNNGRUModel, OmniCarGridCNNModel
from unilab.base import registry
from unilab.envs.navigation.omni_car import OmniCarGridAvoidanceCfg


def test_omni_car_grid_contract() -> None:
    registry.ensure_registries()
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={"seed": 7, "max_episode_seconds": 1.0},
    )

    state = env.init_state()
    assert state.obs["obs"].shape == (4, env.obs_groups_spec["obs"])
    assert state.obs["critic"].shape == (4, env.obs_groups_spec["critic"])
    assert env.obs_groups_spec["obs"] == 10 * 80 * 80 + 3 + 3 + 3 + 24 * 9
    assert env.obs_groups_spec["critic"] == env.obs_groups_spec["obs"] + 5
    assert env.action_space.shape == (3,)

    next_state = env.step(np.zeros((4, 3), dtype=np.float32))
    assert next_state.obs["obs"].shape == (4, env.obs_groups_spec["obs"])
    assert next_state.reward.shape == (4,)
    assert next_state.terminated.shape == (4,)
    assert next_state.truncated.shape == (4,)
    assert "commands" in next_state.info
    assert "command_clearance" in next_state.info
    assert "command_safety_gate" in next_state.info
    assert next_state.info["command_clearance"].shape == (4,)
    assert next_state.info["command_safety_gate"].shape == (4,)
    assert "reward_components" in next_state.info
    assert "total" in next_state.info["reward_components"]
    assert next_state.info["reward_components"]["total"].shape == (4,)
    assert "omni_car/tracking_error" in next_state.info["log"]
    assert "omni_car/reward/total" in next_state.info["log"]
    assert "omni_car/response_progress" in next_state.info["log"]
    assert "omni_car/reward/yaw_idle_stop" in next_state.info["log"]
    assert "omni_car/vx_track_cost" in next_state.info["log"]
    assert "omni_car/vy_track_cost" in next_state.info["log"]
    assert "omni_car/vyaw_track_cost" in next_state.info["log"]
    assert "omni_car/vx_diff_cost" in next_state.info["log"]
    assert "omni_car/vy_diff_cost" in next_state.info["log"]
    assert "omni_car/vyaw_diff_cost" in next_state.info["log"]
    assert "omni_car/vx_jerk_cost" in next_state.info["log"]
    assert "omni_car/vy_jerk_cost" in next_state.info["log"]
    assert "omni_car/vyaw_jerk_cost" in next_state.info["log"]
    assert env.play_capabilities.supports_native_interactive_renderer is True
    env.close()


def test_omni_car_observation_layout_matches_reference_concatenate() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=3,
        env_cfg_override={"seed": 31, "obstacles": {"count": 2}},
    )
    env.init_state()
    env.step(
        np.asarray(
            [
                [0.2, -0.1, 0.05],
                [0.0, 0.3, -0.2],
                [-0.4, 0.1, 0.15],
            ],
            dtype=np.float32,
        )
    )
    env_indices = np.asarray([2, 0], dtype=np.int32)

    actual = env._build_obs(env_indices)
    grid = env._grid_history[env_indices].reshape(env_indices.size, -1)
    command_hist = env._command_history[env_indices].reshape(env_indices.size, -1)
    velocity_hist = env._velocity_history[env_indices].reshape(env_indices.size, -1)
    action_hist = env._action_history[env_indices].reshape(env_indices.size, -1)
    clearance = env._nearest_clearance[env_indices, None]
    collision = env._collision[env_indices, None].astype(env._dtype)
    expected_obs = np.concatenate(
        [
            grid,
            env._commands[env_indices],
            env._velocity[env_indices],
            env._last_action[env_indices],
            command_hist,
            velocity_hist,
            action_hist,
        ],
        axis=1,
        dtype=env._dtype,
    )
    expected_critic = np.concatenate(
        [expected_obs, clearance, collision, env._pose[env_indices]],
        axis=1,
        dtype=env._dtype,
    )

    np.testing.assert_array_equal(actual["obs"], expected_obs)
    np.testing.assert_array_equal(actual["critic"], expected_critic)
    env.close()


def test_omni_car_grid_history_excludes_privileged_actor_inputs() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 43,
            "grid_history_len": 3,
            "command": {"zero_fraction": 0.0},
            "obstacles": {"count": 1},
        },
    )
    state = env.init_state()
    grid_stack_dim = env._grid_history_len * env._grid_dim

    assert state.obs["obs"].shape[1] == grid_stack_dim + 3 + 3 + 3 + env._history_dim
    assert state.obs["critic"].shape[1] == state.obs["obs"].shape[1] + 5
    np.testing.assert_array_equal(
        state.obs["critic"][:, : state.obs["obs"].shape[1]],
        state.obs["obs"],
    )

    first_frame = env._grid_history[0, 0].copy()
    for frame_id in range(1, env._grid_history_len):
        np.testing.assert_array_equal(env._grid_history[0, frame_id], first_frame)

    env._obstacle_xy[0, 0] = np.asarray([0.50, 0.0], dtype=np.float32)
    env._obstacle_radius[0, 0] = 0.22
    env._obstacle_type[0, 0] = env._OBSTACLE_CIRCLE
    env._grid_history_initialized[0] = False
    env._build_obs(np.asarray([0], dtype=np.int32))
    first_frame = env._grid_history[0, 0].copy()

    env._obstacle_xy[0, 0] = np.asarray([1.25, 0.0], dtype=np.float32)
    state = env.step(np.zeros((1, 3), dtype=np.float32))
    assert state.obs["critic"][0, -5] == pytest.approx(env._nearest_clearance[0])
    assert state.obs["critic"][0, -4] == pytest.approx(float(env._collision[0]))
    assert np.count_nonzero(env._grid_history[0, 0] != env._grid_history[0, -1]) > 0
    env.close()


def test_omni_car_balanced_command_sampler_covers_modes_and_limits() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={"seed": 47},
    )
    samples = env._sample_commands(700)
    active = np.abs(samples) > env._cfg.command.deadband
    zero_rows = np.linalg.norm(samples, axis=1) == 0.0

    np.testing.assert_array_less(np.abs(samples[:, 0]), 2.0 + 1e-6)
    np.testing.assert_array_less(np.abs(samples[:, 1]), 1.0 + 1e-6)
    np.testing.assert_array_less(np.abs(samples[:, 2]), 2.0 + 1e-6)
    assert np.mean(zero_rows) == pytest.approx(env._cfg.command.zero_fraction, abs=0.06)
    for mode in env._COMMAND_MODE_MASKS:
        assert np.any(np.all(active == mode, axis=1))
    planar_arbitrary = np.mean(active[:, 0] & active[:, 1])
    assert planar_arbitrary > 0.25

    normalized = np.abs(samples) / np.asarray([2.0, 1.0, 2.0], dtype=np.float32)
    nonzero = normalized[normalized > 0.0]
    assert np.any((0.15 <= nonzero) & (nonzero < 0.35))
    assert np.any((0.35 <= nonzero) & (nonzero < 0.65))
    assert np.any(nonzero >= 0.65)
    env.close()


def test_omni_car_command_sampler_respects_mode_weights() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 48,
            "command": {
                "zero_fraction": 0.0,
                "mode_weights": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "deadband": 0.0,
            },
        },
    )
    mode_ids = env._sample_command_mode_ids(128)
    samples = env._sample_commands(128)

    assert np.all(mode_ids == 3)
    assert np.all(np.linalg.norm(samples[:, :2], axis=1) > 0.0)
    assert np.all(samples[:, 2] == 0.0)
    env.close()


def test_omni_car_command_hold_sampler_includes_long_segments() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 49,
            "ctrl_dt": 0.05,
            "command": {
                "hold_min_s": 1.0,
                "hold_max_s": 2.0,
                "long_hold_fraction": 0.5,
                "long_hold_min_s": 10.0,
                "long_hold_max_s": 12.0,
            },
        },
    )
    hold_steps = env._sample_command_hold_steps(200)

    assert np.any(hold_steps >= int(round(10.0 / env._cfg.ctrl_dt)))
    assert np.any(hold_steps <= int(round(2.0 / env._cfg.ctrl_dt)))
    env.close()


def test_omni_car_human_command_overrides_selected_agent() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={
            "seed": 53,
            "human_command": {
                "enabled": True,
                "env_index": "random",
                "replay_fanout": 1,
                "backend": "zero",
                "smoothing_tau_s": 0.0,
            },
        },
    )
    assert env._human_command_env_ids.shape == (2,)
    assert 0 <= env._human_command_env_id < env.num_envs

    command = np.asarray([1.2, -0.4, 0.5], dtype=np.float32)
    env._read_human_command = lambda: command.astype(env._dtype)
    state = env.init_state()

    expected_command_rows = np.broadcast_to(command, (env._human_command_env_ids.size, 3))
    np.testing.assert_allclose(
        env._raw_commands[env._human_command_env_ids], expected_command_rows
    )
    np.testing.assert_allclose(
        env._commands[env._human_command_env_ids], expected_command_rows
    )
    for env_id in env._human_command_env_ids:
        np.testing.assert_allclose(env._command_history[env_id], np.broadcast_to(command, (24, 3)))

    next_command = np.asarray([-0.8, 0.3, -1.1], dtype=np.float32)
    env._read_human_command = lambda: next_command.astype(env._dtype)
    state = env.step(np.zeros((4, 3), dtype=np.float32))

    expected_next_command_rows = np.broadcast_to(
        next_command, (env._human_command_env_ids.size, 3)
    )
    np.testing.assert_allclose(
        env._raw_commands[env._human_command_env_ids], expected_next_command_rows
    )
    np.testing.assert_allclose(
        env._commands[env._human_command_env_ids], expected_next_command_rows
    )
    assert state.info["human_command_enabled"] is True
    assert state.info["human_command_env_id"] == env._human_command_env_id
    np.testing.assert_array_equal(state.info["human_command_env_ids"], env._human_command_env_ids)
    np.testing.assert_allclose(state.info["human_command"], next_command)
    assert state.info["log"]["omni_car/human_command_norm"] == pytest.approx(
        float(np.linalg.norm(next_command))
    )
    env.close()


def test_omni_car_human_command_axis_mapping_uses_xbox_sticks() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 59,
            "human_command": {
                "enabled": True,
                "backend": "zero",
                "deadzone": 0.10,
                "axis_vx": 1,
                "axis_vy": 0,
                "axis_vyaw": 2,
                "invert_vx": True,
                "invert_vy": False,
                "invert_vyaw": False,
            },
        },
    )
    axes = np.asarray([0.55, -0.55, 0.55], dtype=np.float32)
    command = env._map_human_axes_to_command(axes)
    expected_axis = (0.55 - 0.10) / (1.0 - 0.10)
    np.testing.assert_allclose(
        command,
        [2.0 * expected_axis, 1.0 * expected_axis, 2.0 * expected_axis],
    )

    deadzone_command = env._map_human_axes_to_command(np.asarray([0.05, -0.05, 0.05]))
    np.testing.assert_array_equal(deadzone_command, np.zeros((3,), dtype=env._dtype))

    drift_command = env._map_human_axes_to_command(np.asarray([0.12, -0.12, 0.12]))
    np.testing.assert_array_equal(drift_command, np.zeros((3,), dtype=env._dtype))
    env.close()


def test_omni_car_human_zero_input_holds_executed_action() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "seed": 60,
            "obstacles": {"count": 0},
            "human_command": {
                "enabled": True,
                "env_index": 0,
                "backend": "zero",
                "smoothing_tau_s": 0.0,
                "idle_action_hold": True,
            },
        },
    )
    env.init_state()

    action = np.asarray([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]], dtype=np.float32)
    state = env.step(action)

    np.testing.assert_allclose(state.info["policy_action"][0], action[0])
    np.testing.assert_allclose(state.info["executed_action"][0], np.zeros(3), atol=1e-6)
    assert bool(state.info["human_idle_hold"][0]) is True
    assert state.info["log"]["omni_car/focus_human_idle_hold"] == pytest.approx(1.0)
    assert state.info["log"]["omni_car/focus_executed_action_norm"] == pytest.approx(0.0)
    env.close()


def test_omni_car_human_command_live_viewer_focuses_human_agent() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=3,
        env_cfg_override={
            "seed": 61,
            "human_command": {
                "enabled": True,
                "env_index": 2,
                "backend": "zero",
                "render_enabled": True,
            },
        },
    )
    assert env._viewer_focus_env_id() == 2
    env._human_live_render_enabled = False
    env.close()


def test_omni_car_physical_limits_apply_before_integration() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 11,
            "physical_limits": {
                "max_x_speed": 1.0,
                "max_y_speed": 1.0,
                "max_yaw_rate": 1.0,
                "max_x_accel": 2.0,
                "max_y_accel": 2.0,
                "max_yaw_accel": 2.0,
            },
            "ctrl_dt": 0.1,
        },
    )
    env.init_state()

    env.step(np.asarray([[10.0, -10.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(env._velocity, [[0.2, -0.2, 0.2]], atol=1e-6)

    for _ in range(10):
        env.step(np.asarray([[10.0, -10.0, 10.0]], dtype=np.float32))
    assert np.all(env._velocity <= 1.0 + 1e-6)
    assert np.all(env._velocity >= -1.0 - 1e-6)
    env.close()


def test_omni_car_step_rewards_command_seen_by_policy_before_resample() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 12,
            "command": {"smoothing_tau_s": 0.0},
            "obstacles": {"count": 0},
            "reward": {
                "intent": 1.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "yaw_idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32)
    env._raw_commands[:] = env._commands
    env._command_steps_remaining[:] = 1
    env._sample_commands = lambda count: np.zeros((count, 3), dtype=env._dtype)

    state = env.step(np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32))

    assert state.reward[0] > 0.0
    assert state.info["reward_components"]["intent"][0] > 0.0
    np.testing.assert_allclose(env._commands, np.zeros((1, 3), dtype=env._dtype))
    env.close()


def test_omni_car_response_diff_and_jerk_rewards_are_measured() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 13,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 1.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    env._last_action[:] = 0.0
    env._nearest_clearance[:] = 2.0

    response_reward = env._compute_reward(np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32))
    assert env._response_progress[0] > 0.0
    stalled_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))

    assert response_reward[0] > 0.0
    assert stalled_reward[0] == pytest.approx(0.0)

    env._cfg.reward.response = 0.0
    env._cfg.reward.vx_diff = 1.0
    env._cfg.reward.vy_diff = 2.0
    env._cfg.reward.vyaw_diff = 3.0
    env._last_action[:] = 0.0
    env._last_action_delta[:] = 0.0
    diff_reward = env._compute_reward(np.asarray([[0.15, 0.30, 0.20]], dtype=np.float32))

    np.testing.assert_allclose(env._diff_cost, [[1.0, 4.0, 1.0]], atol=1e-6)
    assert diff_reward[0] == pytest.approx(-(1.0 + 8.0 + 3.0))

    env._cfg.reward.vx_diff = 0.0
    env._cfg.reward.vy_diff = 0.0
    env._cfg.reward.vyaw_diff = 0.0
    env._cfg.reward.vx_jerk = 1.0
    env._cfg.reward.vy_jerk = 2.0
    env._cfg.reward.vyaw_jerk = 3.0
    env._last_action[:] = np.asarray([[0.15, 0.15, 0.20]], dtype=np.float32)
    env._last_action_delta[:] = np.asarray([[0.15, 0.15, 0.20]], dtype=np.float32)
    jerk_reward = env._compute_reward(np.asarray([[0.15, -0.15, 0.0]], dtype=np.float32))

    np.testing.assert_allclose(env._jerk_cost, [[1.0, 9.0, 4.0]], atol=1e-6)
    assert jerk_reward[0] == pytest.approx(-(1.0 + 18.0 + 12.0))
    env.close()


def test_omni_car_projection_reward_penalizes_off_axis_and_reverse_motion() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 16,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 4.0,
                "intent_projection": 4.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 6.0,
                "reverse": 8.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    forward = env._compute_reward(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    off_axis = env._compute_reward(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32))[0]
    off_axis_component = float(env._reward_components["off_axis"][0])
    off_axis_intent = float(env._reward_components["intent"][0])
    reverse = env._compute_reward(np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32))[0]

    assert forward > off_axis
    assert forward > reverse
    assert off_axis_intent == pytest.approx(0.0)
    assert off_axis_component < 0.0
    assert env._reward_components["reverse"][0] < 0.0
    env.close()


def test_omni_car_axis_tracking_penalty_is_command_direction_gated() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 14,
            "obstacles": {
                "count": 1,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 1.0,
                "vy_track": 2.0,
                "vyaw_track": 3.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, -1.0, 1.0]], dtype=np.float32)
    env._obstacle_xy[0, 0] = np.asarray([-1.0, 1.0], dtype=np.float32)
    env._obstacle_radius[0, 0] = 0.20

    safe_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(env._track_cost, [[0.25, 1.0, 0.25]], atol=1e-6)
    assert env._command_safety_gate[0] == pytest.approx(1.0)
    assert safe_reward[0] == pytest.approx(-(0.25 + 2.0 + 0.75))

    env._obstacle_xy[0, 0] = np.asarray([0.40, -0.40], dtype=np.float32)
    blocked_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert env._command_safety_gate[0] == pytest.approx(0.0)
    assert blocked_reward[0] == pytest.approx(0.0)
    env.close()


def test_omni_car_front_obstacle_rewards_stop_over_forward_push() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 15,
            "obstacles": {
                "count": 1,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
            "reward": {
                "intent": 8.0,
                "intent_projection": 4.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 10.0,
                "blocked_motion": 8.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    env._obstacle_xy[0, 0] = np.asarray([0.55, 0.0], dtype=np.float32)
    env._obstacle_radius[0, 0] = 0.22
    env._nearest_clearance[:] = env._compute_clearance(np.asarray([0], dtype=np.int32))

    stop_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    stop_blocked_reward = float(env._reward_components["blocked_stop"][0])
    push_reward = env._compute_reward(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    push_blocked_motion = float(env._reward_components["blocked_motion"][0])

    assert env._command_safety_gate[0] == pytest.approx(0.0)
    assert stop_reward[0] > push_reward[0]
    assert stop_blocked_reward > 0.0
    assert push_blocked_motion < 0.0
    env.close()


def test_omni_car_side_wall_keeps_forward_intent_gate_open() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 19,
            "obstacles": {
                "count": 3,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    env._obstacle_xy[0] = np.asarray(
        [[0.60, -0.34], [1.05, -0.34], [1.50, -0.34]], dtype=np.float32
    )
    env._obstacle_radius[0] = 0.22
    env._nearest_clearance[:] = env._compute_clearance(np.asarray([0], dtype=np.int32))

    env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))

    assert env._command_safety_gate[0] == pytest.approx(1.0)
    env.close()


def test_omni_car_obstacle_curriculum_samples_front_blockers_and_side_walls() -> None:
    front_env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 20,
            "obstacles": {
                "count": 3,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
                "front_blocker_fraction": 1.0,
                "side_wall_fraction": 0.0,
                "front_blocker_box_fraction": 0.0,
                "front_blocker_wall_fraction": 1.0,
            },
        },
    )
    front_env.init_state()
    front_env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    front_env._sample_obstacles(np.asarray([0], dtype=np.int32))
    front_env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert front_env._obstacle_type[0, 0] == front_env._OBSTACLE_WALL
    assert front_env._command_safety_gate[0] == pytest.approx(0.0)
    front_env.close()

    box_env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 22,
            "obstacles": {
                "count": 1,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
                "front_blocker_fraction": 1.0,
                "side_wall_fraction": 0.0,
                "front_blocker_box_fraction": 1.0,
                "front_blocker_wall_fraction": 0.0,
            },
        },
    )
    box_env.init_state()
    box_env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    box_env._sample_obstacles(np.asarray([0], dtype=np.int32))
    box_env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert box_env._obstacle_type[0, 0] == box_env._OBSTACLE_BOX
    assert box_env._command_safety_gate[0] == pytest.approx(0.0)
    box_env.close()

    side_env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 21,
            "obstacles": {
                "count": 3,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
                "front_blocker_fraction": 0.0,
                "side_wall_fraction": 1.0,
            },
        },
    )
    side_env.init_state()
    side_env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    side_env._sample_obstacles(np.asarray([0], dtype=np.int32))
    side_env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert side_env._command_safety_gate[0] == pytest.approx(1.0)
    side_env.close()


def test_omni_car_zero_command_rewards_idle_action() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 18,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 1.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 4.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = 0.0

    idle_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    drift_reward = env._compute_reward(np.asarray([[0.08, 0.0, 0.12]], dtype=np.float32))

    assert idle_reward[0] > drift_reward[0]
    assert env._reward_components["idle_stop"][0] < 0.0
    env.close()


def test_omni_car_yaw_intent_only_rewards_active_yaw_commands() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 22,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 3.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()

    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    no_yaw_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert no_yaw_reward[0] == pytest.approx(0.0)
    assert env._reward_components["yaw_intent"][0] == pytest.approx(0.0)

    env._commands[:] = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    active_yaw_reward = env._compute_reward(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
    assert active_yaw_reward[0] == pytest.approx(3.0)
    assert env._reward_components["yaw_intent"][0] == pytest.approx(3.0)
    env.close()


def test_omni_car_yaw_idle_stop_penalizes_uncommanded_yaw() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 23,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "yaw_idle_stop": 5.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_track": 0.0,
                "vy_track": 0.0,
                "vyaw_track": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()
    env._commands[:] = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    no_yaw_command_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.24]], dtype=np.float32))
    assert no_yaw_command_reward[0] == pytest.approx(-20.0)
    assert env._reward_components["yaw_idle_stop"][0] == pytest.approx(-20.0)

    env._commands[:] = np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32)
    yaw_command_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.24]], dtype=np.float32))
    assert yaw_command_reward[0] == pytest.approx(0.0)
    assert env._reward_components["yaw_idle_stop"][0] == pytest.approx(0.0)
    env.close()


def test_omni_car_samples_circle_box_and_wall_obstacles() -> None:
    cases = [
        ("circle_fraction", 0),
        ("box_fraction", 1),
        ("wall_fraction", 2),
    ]
    for fraction_key, expected_type in cases:
        fractions = {"circle_fraction": 0.0, "box_fraction": 0.0, "wall_fraction": 0.0}
        fractions[fraction_key] = 1.0
        env = registry.make(
            "OmniCarGridAvoidance",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override={
                "seed": 17,
                "obstacles": {
                    "count": 4,
                    "front_blocker_fraction": 0.0,
                    "side_wall_fraction": 0.0,
                    **fractions,
                },
            },
        )
        env.init_state()
        assert set(env._obstacle_type[0].tolist()) == {expected_type}
        if expected_type == 0:
            assert np.all(env._obstacle_radius[0] > 0.0)
        else:
            assert np.all(env._obstacle_half_extents[0] > 0.0)
        assert np.count_nonzero(env._occupancy_grid(np.asarray([0], dtype=np.int32))) > 0
        env.close()


def test_omni_car_grid_and_clearance_match_dense_reference() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 19,
            "obstacles": {
                "count": 3,
            },
        },
    )
    env.init_state()
    env._pose[0] = np.asarray([0.15, -0.10, 0.42], dtype=np.float32)
    env._obstacle_xy[0] = np.asarray(
        [
            [0.55, 0.20],
            [-0.45, 0.65],
            [1.10, -0.55],
        ],
        dtype=np.float32,
    )
    env._obstacle_radius[0] = np.asarray([0.22, 0.18, 0.20], dtype=np.float32)
    env._obstacle_half_extents[0] = np.asarray(
        [
            [0.0, 0.0],
            [0.28, 0.14],
            [0.72, 0.08],
        ],
        dtype=np.float32,
    )
    env._obstacle_yaw[0] = np.asarray([0.0, 0.30, -0.65], dtype=np.float32)
    env._obstacle_type[0] = np.asarray(
        [env._OBSTACLE_CIRCLE, env._OBSTACLE_BOX, env._OBSTACLE_WALL], dtype=np.int8
    )

    local_xy = env._world_to_body_points(0, env._obstacle_xy[0])
    axis = (
        (np.arange(env._cfg.grid.size, dtype=np.float32) + 0.5 - env._cfg.grid.size / 2.0)
        * env._cfg.grid.cell_size
    )
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    pad = env._cfg.grid.cell_size * 0.5
    occupied = np.zeros((env._cfg.grid.size, env._cfg.grid.size), dtype=bool)
    signed = np.empty((env._cfg.obstacles.count,), dtype=np.float32)

    for obstacle_id in range(env._cfg.obstacles.count):
        center_x, center_y = local_xy[obstacle_id]
        if env._obstacle_type[0, obstacle_id] == env._OBSTACLE_CIRCLE:
            radius = env._obstacle_radius[0, obstacle_id] + pad
            occupied |= (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2 <= radius**2
            signed[obstacle_id] = np.linalg.norm(local_xy[obstacle_id]) - env._obstacle_radius[
                0, obstacle_id
            ]
            continue

        half_extent_x, half_extent_y = env._obstacle_half_extents[0, obstacle_id]
        rel_yaw = env._obstacle_yaw[0, obstacle_id] - env._pose[0, 2]
        cos_yaw = np.cos(rel_yaw)
        sin_yaw = np.sin(rel_yaw)
        delta_x = grid_x - center_x
        delta_y = grid_y - center_y
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y
        occupied |= (np.abs(local_x) <= half_extent_x + pad) & (
            np.abs(local_y) <= half_extent_y + pad
        )

        point_local_x = cos_yaw * center_x + sin_yaw * center_y
        point_local_y = -sin_yaw * center_x + cos_yaw * center_y
        qx = abs(point_local_x) - half_extent_x
        qy = abs(point_local_y) - half_extent_y
        outside_x = max(qx, 0.0)
        outside_y = max(qy, 0.0)
        outside_distance = float(np.hypot(outside_x, outside_y))
        inside_distance = min(max(qx, qy), 0.0)
        signed[obstacle_id] = outside_distance + inside_distance

    expected_grid = occupied.reshape(1, -1).astype(np.float32)
    actual_grid = env._occupancy_grid(np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(actual_grid, expected_grid)

    safety_radius = 0.5 * float(np.hypot(env._cfg.body.length_m, env._cfg.body.width_m))
    safety_radius += env._cfg.grid.safety_margin_m
    expected_clearance = np.min(signed - safety_radius)
    actual_clearance = env._compute_clearance(np.asarray([0], dtype=np.int32))
    np.testing.assert_allclose(actual_clearance, [expected_clearance], atol=1e-6)
    env.close()


def test_omni_car_large_batch_grid_matches_scalar_path() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=260,
        env_cfg_override={
            "seed": 29,
            "obstacles": {
                "count": 4,
            },
        },
    )
    env.init_state()
    env_indices = np.arange(env.num_envs, dtype=np.int32)
    batched = np.zeros((env.num_envs, env._cfg.grid.size, env._cfg.grid.size), dtype=env._dtype)
    scalar = np.zeros_like(batched)

    env._fill_occupancy_grid(env_indices, batched)
    env._fill_occupancy_grid_scalar(env_indices, scalar)

    np.testing.assert_array_equal(batched, scalar)
    env.close()


def test_omni_car_logs_pre_reset_collision_metrics() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 23,
            "max_episode_seconds": 0.1,
            "obstacles": {
                "count": 1,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
        },
    )
    env.init_state()
    env._obstacle_xy[0, 0] = env._pose[0, :2]
    env._obstacle_radius[0, 0] = 0.4
    env._obstacle_type[0, 0] = 0

    state = env.step(np.zeros((1, 3), dtype=np.float32))

    assert bool(state.info["collision"][0]) is True
    assert state.info["log"]["omni_car/collision_rate"] == pytest.approx(1.0)
    assert state.info["log"]["omni_car/mean_clearance"] <= 0.0
    env.close()


def test_omni_car_large_scene_agent_collision_is_dynamic_obstacle() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "seed": 37,
            "max_episode_seconds": 0.05,
            "obstacles": {"count": 0},
            "large_scene": {
                "enabled": True,
                "world_size_m": 10.0,
                "static_obstacle_count": 0,
                "border_wall_segments_per_side": 1,
                "agent_collision_radius_m": 0.35,
                "reset_on_timeout": False,
                "stagnation_warmup_steps": 1000,
            },
        },
    )
    env.init_state()
    env._pose[0] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    env._pose[1] = np.asarray([0.30, 0.0, 0.0], dtype=np.float32)

    grid = env._occupancy_grid(np.asarray([0], dtype=np.int32))
    assert np.count_nonzero(grid) > 0

    state = env.step(np.zeros((2, 3), dtype=np.float32))

    assert bool(state.info["agent_collision"][0]) is True
    assert bool(state.info["collision"][0]) is True
    assert state.info["log"]["omni_car/agent_collision_rate"] == pytest.approx(1.0)
    assert bool(state.truncated[0]) is False
    env.close()


def test_omni_car_large_scene_border_collision_and_stagnation_reset() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 41,
            "max_episode_seconds": 0.05,
            "obstacles": {"count": 0},
            "large_scene": {
                "enabled": True,
                "world_size_m": 8.0,
                "static_obstacle_count": 0,
                "border_wall_segments_per_side": 1,
                "agent_collision_radius_m": 0.35,
                "reset_on_timeout": False,
                "stagnation_warmup_steps": 1,
                "stagnation_window_steps": 1,
                "stagnation_min_return_delta": 1.0e9,
            },
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
                "blocked_stop": 0.0,
                "blocked_motion": 0.0,
                "idle_stop": 0.0,
                "off_axis": 0.0,
                "reverse": 0.0,
                "vx_diff": 0.0,
                "vy_diff": 0.0,
                "vyaw_diff": 0.0,
                "vx_jerk": 0.0,
                "vy_jerk": 0.0,
                "vyaw_jerk": 0.0,
                "clearance": 0.0,
                "collision": 0.0,
            },
        },
    )
    env.init_state()

    state = env.step(np.zeros((1, 3), dtype=np.float32))
    assert bool(state.info["stagnated"][0]) is True
    assert bool(state.truncated[0]) is False

    env._pose[0] = np.asarray([4.2, 0.0, 0.0], dtype=np.float32)
    clearance = env._compute_clearance(np.asarray([0], dtype=np.int32))
    assert clearance[0] <= 0.0
    assert bool(env._border_collision[0]) is True
    env.close()


def test_omni_car_cfg_validates_grid_shape() -> None:
    cfg = OmniCarGridAvoidanceCfg()
    cfg.grid.size = 79
    try:
        cfg.validate()
    except ValueError as exc:
        assert "grid.size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected odd grid size to fail validation")


def test_omni_car_interactive_play_plan() -> None:
    registry.ensure_registries()
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={"seed": 3},
    )

    auto_plan = env.resolve_play_render_plan(
        play_render_mode="auto",
        play_steps=12,
        output_video=None,
    )
    assert auto_plan.mode == "interactive"
    assert auto_plan.headless is False
    assert auto_plan.record_video is False
    assert auto_plan.num_steps == 12

    none_plan = env.resolve_play_render_plan(
        play_render_mode="none",
        play_steps=12,
        output_video=None,
    )
    assert none_plan.mode == "none"

    with pytest.raises(NotImplementedError, match="video recording"):
        env.resolve_play_render_plan(
            play_render_mode="record",
            play_steps=12,
            output_video="out.mp4",
        )


def test_omni_car_viewer_xml_loads() -> None:
    import mujoco

    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={"seed": 5},
    )
    model = mujoco.MjModel.from_xml_string(env._viewer_xml())
    assert model.ncam == 1
    assert model.ngeom >= 1


def test_omni_car_cnn_model_forward_actor_and_critic() -> None:
    cfg = OmniCarGridAvoidanceCfg()
    obs_dim = (
        cfg.grid.size * cfg.grid.size
        + 3
        + 3
        + 3
        + cfg.obs_history_len * 9
        + 2
    )
    critic_obs_dim = obs_dim + 3

    actor_obs = torch.zeros((2, obs_dim), dtype=torch.float32)
    critic_obs = torch.zeros((2, critic_obs_dim), dtype=torch.float32)
    actor = OmniCarGridCNNModel(
        TensorDict({"actor": actor_obs}, batch_size=2),
        {"actor": ["actor"]},
        "actor",
        3,
        hidden_dims=[16],
        cnn_feature_dim=8,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    )
    critic = OmniCarGridCNNModel(
        TensorDict({"critic": critic_obs}, batch_size=2),
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[16],
        cnn_feature_dim=8,
    )

    actor_out = actor(TensorDict({"actor": actor_obs}, batch_size=2))
    critic_out = critic(TensorDict({"critic": critic_obs}, batch_size=2))
    assert actor_out.shape == (2, 3)
    assert critic_out.shape == (2, 1)


def test_omni_car_cnn_gru_model_forward_actor_and_critic() -> None:
    cfg = OmniCarGridAvoidanceCfg()
    actor_obs_dim = (
        cfg.grid_history_len * cfg.grid.size * cfg.grid.size
        + 3
        + 3
        + 3
        + cfg.obs_history_len * 9
    )
    critic_obs_dim = actor_obs_dim + 5

    actor_obs = torch.zeros((2, actor_obs_dim), dtype=torch.float32)
    critic_obs = torch.zeros((2, critic_obs_dim), dtype=torch.float32)
    actor = OmniCarGridCNNGRUModel(
        TensorDict({"actor": actor_obs}, batch_size=2),
        {"actor": ["actor"]},
        "actor",
        3,
        hidden_dims=[16],
        grid_history_len=cfg.grid_history_len,
        cnn_feature_dim=8,
        gru_hidden_dim=8,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    )
    critic = OmniCarGridCNNGRUModel(
        TensorDict({"critic": critic_obs}, batch_size=2),
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[16],
        grid_history_len=cfg.grid_history_len,
        cnn_feature_dim=8,
        gru_hidden_dim=8,
    )

    actor_out = actor(TensorDict({"actor": actor_obs}, batch_size=2))
    critic_out = critic(TensorDict({"critic": critic_obs}, batch_size=2))
    assert actor_out.shape == (2, 3)
    assert critic_out.shape == (2, 1)
