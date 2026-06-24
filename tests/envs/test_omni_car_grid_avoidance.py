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
    assert "omni_car/vx_diff_cost" in next_state.info["log"]
    assert "omni_car/vy_diff_cost" in next_state.info["log"]
    assert "omni_car/vyaw_diff_cost" in next_state.info["log"]
    assert "omni_car/vx_jerk_cost" in next_state.info["log"]
    assert "omni_car/vy_jerk_cost" in next_state.info["log"]
    assert "omni_car/vyaw_jerk_cost" in next_state.info["log"]
    assert env.play_capabilities.supports_native_interactive_renderer is True
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
