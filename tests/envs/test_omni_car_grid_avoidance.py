from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from unilab.algos.torch.omni_car import OmniCarGridCNNModel
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
    assert env.action_space.shape == (3,)

    next_state = env.step(np.zeros((4, 3), dtype=np.float32))
    assert next_state.obs["obs"].shape == (4, env.obs_groups_spec["obs"])
    assert next_state.reward.shape == (4,)
    assert next_state.terminated.shape == (4,)
    assert next_state.truncated.shape == (4,)
    assert "commands" in next_state.info
    assert "omni_car/tracking_error" in next_state.info["log"]
    assert "omni_car/response_progress" in next_state.info["log"]
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
    grid = env._occupancy_grid(env_indices)
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
            clearance,
            collision,
        ],
        axis=1,
        dtype=env._dtype,
    )
    expected_critic = np.concatenate(
        [expected_obs, env._pose[env_indices]],
        axis=1,
        dtype=env._dtype,
    )

    np.testing.assert_array_equal(actual["obs"], expected_obs)
    np.testing.assert_array_equal(actual["critic"], expected_critic)
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


def test_omni_car_axis_tracking_penalty_is_clearance_gated() -> None:
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": 14,
            "obstacles": {"count": 0},
            "reward": {
                "intent": 0.0,
                "intent_projection": 0.0,
                "yaw_intent": 0.0,
                "response": 0.0,
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

    safe_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(env._track_cost, [[0.25, 0.25, 0.25]], atol=1e-6)
    assert safe_reward[0] == pytest.approx(-(0.25 + 0.5 + 0.75))

    env._nearest_clearance[:] = 0.0
    blocked_reward = env._compute_reward(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert blocked_reward[0] == pytest.approx(0.0)
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
