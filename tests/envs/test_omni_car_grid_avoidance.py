from __future__ import annotations

import numpy as np

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
