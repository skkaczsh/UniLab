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
    assert state.obs["obs"].shape == (4, 6411)
    assert state.obs["critic"].shape == (4, 6414)
    assert env.action_space.shape == (3,)

    next_state = env.step(np.zeros((4, 3), dtype=np.float32))
    assert next_state.obs["obs"].shape == (4, 6411)
    assert next_state.reward.shape == (4,)
    assert next_state.terminated.shape == (4,)
    assert next_state.truncated.shape == (4,)
    assert "commands" in next_state.info
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
    actor_obs = torch.zeros((2, 6411), dtype=torch.float32)
    critic_obs = torch.zeros((2, 6414), dtype=torch.float32)
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
