from __future__ import annotations

import numpy as np
import pytest

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
