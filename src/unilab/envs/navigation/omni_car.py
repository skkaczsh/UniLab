from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.base import ABEnv, EnvCfg
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype


@dataclass
class OmniCarCommandCfg:
    max_x_speed: float = 2.0
    max_y_speed: float = 2.0
    max_yaw_rate: float = 2.0
    resample_interval_s: float = 2.0
    deadband: float = 0.15


@dataclass
class OmniCarGridCfg:
    size: int = 80
    cell_size: float = 0.05
    safety_margin_m: float = 0.05


@dataclass
class OmniCarBodyCfg:
    length_m: float = 0.48
    width_m: float = 0.32


@dataclass
class OmniCarObstacleCfg:
    count: int = 14
    radius_min_m: float = 0.10
    radius_max_m: float = 0.28
    spawn_radius_m: float = 3.2
    keepout_radius_m: float = 0.75


@dataclass
class OmniCarRewardCfg:
    intent: float = 2.0
    intent_projection: float = 0.6
    yaw_intent: float = 0.4
    clearance: float = 0.8
    collision: float = -8.0
    action_rate: float = -0.03
    speed_limit: float = -0.4


@registry.envcfg("OmniCarGridAvoidance")
@dataclass
class OmniCarGridAvoidanceCfg(EnvCfg):
    """Config for a rectangular omnidirectional car in a local occupancy grid."""

    sim_dt: float = 0.05
    ctrl_dt: float = 0.05
    max_episode_seconds: float = 12.0
    command: OmniCarCommandCfg = field(default_factory=OmniCarCommandCfg)
    grid: OmniCarGridCfg = field(default_factory=OmniCarGridCfg)
    body: OmniCarBodyCfg = field(default_factory=OmniCarBodyCfg)
    obstacles: OmniCarObstacleCfg = field(default_factory=OmniCarObstacleCfg)
    reward: OmniCarRewardCfg = field(default_factory=OmniCarRewardCfg)
    reward_config: dict[str, Any] | None = None
    seed: int | None = None

    def validate(self) -> None:
        super().validate()
        if self.grid.size <= 0 or self.grid.size % 2 != 0:
            raise ValueError("grid.size must be a positive even integer")
        if self.grid.cell_size <= 0.0:
            raise ValueError("grid.cell_size must be positive")
        if self.body.length_m <= 0.0 or self.body.width_m <= 0.0:
            raise ValueError("body dimensions must be positive")
        if self.obstacles.count < 0:
            raise ValueError("obstacles.count must be non-negative")
        if self.obstacles.radius_min_m <= 0.0:
            raise ValueError("obstacles.radius_min_m must be positive")
        if self.obstacles.radius_max_m < self.obstacles.radius_min_m:
            raise ValueError("obstacles.radius_max_m must be >= radius_min_m")


@registry.env("OmniCarGridAvoidance", sim_backend="mujoco")
class OmniCarGridAvoidanceEnv(ABEnv):
    """Vectorized 2D grid avoidance environment for a rectangular omnidirectional car.

    The policy action is the safe velocity command executed by the vehicle:
    body-frame ``x`` velocity, body-frame ``y`` velocity, and yaw rate. The user
    command is sampled independently and included in the observation. Rewarding
    the policy for matching that command while penalizing low clearance creates
    the desired arbitration behavior: preserve intent when safe, deviate or slow
    down when obstacles enter the local grid.
    """

    _cfg: OmniCarGridAvoidanceCfg

    def __init__(
        self,
        cfg: OmniCarGridAvoidanceCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        del backend_type
        self._cfg = cfg
        self._apply_reward_config()
        self._num_envs = int(num_envs)
        self._state: NpEnvState | None = None
        self._rng = np.random.default_rng(cfg.seed)
        self._dtype = get_global_dtype()
        self._pose = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._velocity = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._last_action = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._commands = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._obstacle_xy = np.zeros((self._num_envs, cfg.obstacles.count, 2), dtype=self._dtype)
        self._obstacle_radius = np.zeros((self._num_envs, cfg.obstacles.count), dtype=self._dtype)
        self._nearest_clearance = np.zeros((self._num_envs,), dtype=self._dtype)
        self._collision = np.zeros((self._num_envs,), dtype=bool)
        self._truncated = np.zeros((self._num_envs,), dtype=bool)
        self._obs_buffer = np.zeros(
            (self._num_envs, self.obs_groups_spec["obs"]), dtype=self._dtype
        )
        self._critic_buffer = np.zeros(
            (self._num_envs, self.obs_groups_spec["critic"]), dtype=self._dtype
        )

        xs = np.arange(cfg.grid.size, dtype=np.float32) + 0.5 - cfg.grid.size / 2.0
        ys = np.arange(cfg.grid.size, dtype=np.float32) + 0.5 - cfg.grid.size / 2.0
        gx, gy = np.meshgrid(xs * cfg.grid.cell_size, ys * cfg.grid.cell_size, indexing="ij")
        self._grid_points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(self._dtype)
        self._grid_extent = cfg.grid.size * cfg.grid.cell_size * 0.5

    def _apply_reward_config(self) -> None:
        if not self._cfg.reward_config:
            return
        for key, value in self._cfg.reward_config.items():
            if not hasattr(self._cfg.reward, key):
                raise ValueError(f"Unknown OmniCar reward field: {key}")
            setattr(self._cfg.reward, key, float(value))

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def cfg(self) -> OmniCarGridAvoidanceCfg:
        return self._cfg

    @property
    def state(self) -> NpEnvState | None:
        return self._state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # occupancy grid + command + current velocity + last action + clearance + collision flag
        grid_dim = self._cfg.grid.size * self._cfg.grid.size
        return {"obs": grid_dim + 3 + 3 + 3 + 2, "critic": grid_dim + 3 + 3 + 3 + 2 + 3}

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            -np.inf, np.inf, shape=(self.obs_groups_spec["obs"],), dtype=np.float32
        )

    @property
    def action_space(self) -> gym.Space:
        cmd = self._cfg.command
        high = np.asarray([cmd.max_x_speed, cmd.max_y_speed, cmd.max_yaw_rate], dtype=np.float32)
        return gym.spaces.Box(-high, high, dtype=np.float32)

    def init_state(self) -> NpEnvState:
        obs, info = self.reset(np.arange(self._num_envs, dtype=np.int32))
        reward = np.zeros((self._num_envs,), dtype=self._dtype)
        terminated = np.zeros((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        self._state = NpEnvState(obs, reward, terminated, truncated, info)
        return self._state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        if env_indices.size == 0:
            return self._build_obs(env_indices), self._info(env_indices)
        self._pose[env_indices] = 0.0
        self._velocity[env_indices] = 0.0
        self._last_action[env_indices] = 0.0
        self._commands[env_indices] = self._sample_commands(env_indices.size)
        self._sample_obstacles(env_indices)
        self._collision[env_indices] = False
        self._nearest_clearance[env_indices] = self._compute_clearance(env_indices)
        info = self._info(env_indices)
        info["steps"] = np.zeros((env_indices.size,), dtype=np.uint32)
        return self._build_obs(env_indices), info

    def step(self, actions: np.ndarray) -> NpEnvState:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        actions = np.asarray(actions, dtype=self._dtype)
        if actions.shape != (self._num_envs, 3):
            raise ValueError(f"Expected action shape {(self._num_envs, 3)}, got {actions.shape}")
        self._state.info["_final_observation"] = np.zeros((self._num_envs,), dtype=bool)

        clipped = self._clip_action(actions)
        self._integrate(clipped)
        self._state.info["steps"] += 1

        resample_steps = max(
            1, int(round(self._cfg.command.resample_interval_s / self._cfg.ctrl_dt))
        )
        resample_mask = self._state.info["steps"] % resample_steps == 0
        if np.any(resample_mask):
            self._commands[resample_mask] = self._sample_commands(
                int(np.count_nonzero(resample_mask))
            )

        self._nearest_clearance = self._compute_clearance(np.arange(self._num_envs, dtype=np.int32))
        self._collision = self._nearest_clearance <= 0.0
        reward = self._compute_reward(clipped)
        terminated = self._collision.copy()
        self._truncated.fill(False)
        if self._cfg.max_episode_steps is not None:
            np.greater_equal(
                self._state.info["steps"], self._cfg.max_episode_steps, out=self._truncated
            )
        done = terminated | self._truncated

        obs = self._build_obs(np.arange(self._num_envs, dtype=np.int32))
        self._state.info["commands"] = self._commands.copy()
        self._state.info["nearest_clearance"] = self._nearest_clearance.copy()
        self._state.info["collision"] = self._collision.copy()
        final_observation = None
        if np.any(done):
            final_observation = {key: value.copy() for key, value in obs.items()}

        self._last_action = clipped.copy()
        self._state = self._state.replace(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=self._truncated.copy(),
            final_observation=final_observation,
            info=self._state.info,
        )

        if np.any(done):
            done_ids = np.flatnonzero(done).astype(np.int32)
            reset_obs, reset_info = self.reset(done_ids)
            for key, value in reset_obs.items():
                self._state.obs[key][done_ids] = value
            self._state.info["steps"][done_ids] = reset_info["steps"]
            self._state.info["final_observation"] = final_observation
            terminal_mask = np.zeros((self._num_envs,), dtype=bool)
            terminal_mask[done_ids] = True
            self._state.info["_final_observation"] = terminal_mask

        self._state.info["log"] = {
            "omni_car/mean_clearance": float(np.mean(self._nearest_clearance)),
            "omni_car/collision_rate": float(np.mean(self._collision.astype(np.float32))),
            "omni_car/command_norm": float(np.mean(np.linalg.norm(self._commands[:, :2], axis=1))),
        }
        return self._state

    def close(self) -> None:
        return None

    def set_nan_guard(self, guard: Any) -> None:
        del guard

    def _sample_commands(self, count: int) -> np.ndarray:
        cmd = self._cfg.command
        sampled = self._rng.uniform(
            low=[-cmd.max_x_speed, -cmd.max_y_speed, -cmd.max_yaw_rate],
            high=[cmd.max_x_speed, cmd.max_y_speed, cmd.max_yaw_rate],
            size=(count, 3),
        ).astype(self._dtype)
        small = np.linalg.norm(sampled[:, :2], axis=1) < cmd.deadband
        sampled[small, 0] = cmd.max_x_speed * 0.5
        return sampled

    def _sample_obstacles(self, env_indices: np.ndarray) -> None:
        cfg = self._cfg.obstacles
        if cfg.count == 0:
            return
        for env_id in env_indices:
            radii = self._rng.uniform(cfg.radius_min_m, cfg.radius_max_m, size=(cfg.count,))
            angles = self._rng.uniform(-np.pi, np.pi, size=(cfg.count,))
            distances = self._rng.uniform(
                cfg.keepout_radius_m, cfg.spawn_radius_m, size=(cfg.count,)
            )
            xy = np.stack([np.cos(angles) * distances, np.sin(angles) * distances], axis=1)
            # Bias one obstacle into the commanded path so avoidance matters during short runs.
            cmd = self._commands[env_id]
            planar_norm = float(np.linalg.norm(cmd[:2]))
            if planar_norm > 1e-6:
                direction = cmd[:2] / planar_norm
                lateral = np.asarray([-direction[1], direction[0]], dtype=self._dtype)
                xy[0] = direction * self._rng.uniform(0.9, 1.6) + lateral * self._rng.uniform(
                    -0.25, 0.25
                )
                radii[0] = max(float(radii[0]), 0.22)
            self._obstacle_xy[env_id] = xy.astype(self._dtype)
            self._obstacle_radius[env_id] = radii.astype(self._dtype)

    def _clip_action(self, actions: np.ndarray) -> np.ndarray:
        high = self.action_space.high.astype(self._dtype)
        return np.clip(actions, -high, high).astype(self._dtype)

    def _integrate(self, action: np.ndarray) -> None:
        yaw = self._pose[:, 2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vx_body = action[:, 0]
        vy_body = action[:, 1]
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
        self._pose[:, 0] += vx_world * self._cfg.ctrl_dt
        self._pose[:, 1] += vy_world * self._cfg.ctrl_dt
        self._pose[:, 2] = self._wrap_angle(self._pose[:, 2] + action[:, 2] * self._cfg.ctrl_dt)
        self._velocity = action.astype(self._dtype)

    def _build_obs(self, env_indices: np.ndarray) -> dict[str, np.ndarray]:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        grid = self._occupancy_grid(env_indices)
        obs = self._obs_buffer[: env_indices.size]
        critic = self._critic_buffer[: env_indices.size]
        clearance = self._nearest_clearance[env_indices, None]
        collision = self._collision[env_indices, None].astype(self._dtype)
        obs[:, :] = np.concatenate(
            [
                grid,
                self._commands[env_indices],
                self._velocity[env_indices],
                self._last_action[env_indices],
                clearance,
                collision,
            ],
            axis=1,
            dtype=self._dtype,
        )
        critic[:, :] = np.concatenate(
            [
                grid,
                self._commands[env_indices],
                self._velocity[env_indices],
                self._last_action[env_indices],
                clearance,
                collision,
                self._pose[env_indices],
            ],
            axis=1,
            dtype=self._dtype,
        )
        return {"obs": obs.copy(), "critic": critic.copy()}

    def _occupancy_grid(self, env_indices: np.ndarray) -> np.ndarray:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        grid = np.zeros(
            (env_indices.size, self._cfg.grid.size * self._cfg.grid.size), dtype=self._dtype
        )
        if self._cfg.obstacles.count == 0:
            return grid
        for row, env_id in enumerate(env_indices):
            local_xy = self._world_to_body_points(env_id, self._obstacle_xy[env_id])
            in_range = (np.abs(local_xy[:, 0]) <= self._grid_extent) & (
                np.abs(local_xy[:, 1]) <= self._grid_extent
            )
            for center, radius in zip(local_xy[in_range], self._obstacle_radius[env_id][in_range]):
                delta = self._grid_points - center
                occupied = (
                    np.einsum("ij,ij->i", delta, delta)
                    <= float(radius + self._cfg.grid.cell_size * 0.5) ** 2
                )
                grid[row, occupied] = 1.0
        return grid

    def _world_to_body_points(self, env_id: int, points_world: np.ndarray) -> np.ndarray:
        delta = points_world - self._pose[env_id, :2]
        yaw = -float(self._pose[env_id, 2])
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = c * delta[:, 0] - s * delta[:, 1]
        y = s * delta[:, 0] + c * delta[:, 1]
        return np.stack([x, y], axis=1).astype(self._dtype)

    def _compute_clearance(self, env_indices: np.ndarray) -> np.ndarray:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        clearance = np.full((env_indices.size,), self._grid_extent, dtype=self._dtype)
        half_diag = 0.5 * float(np.hypot(self._cfg.body.length_m, self._cfg.body.width_m))
        safety_radius = half_diag + self._cfg.grid.safety_margin_m
        if self._cfg.obstacles.count == 0:
            return clearance
        for row, env_id in enumerate(env_indices):
            distances = np.linalg.norm(self._obstacle_xy[env_id] - self._pose[env_id, :2], axis=1)
            signed = distances - self._obstacle_radius[env_id] - safety_radius
            clearance[row] = np.min(signed).astype(self._dtype)
        return clearance

    def _compute_reward(self, action: np.ndarray) -> np.ndarray:
        cfg = self._cfg.reward
        cmd = self._commands
        planar_scale = np.maximum(np.linalg.norm(cmd[:, :2], axis=1), 0.25)
        planar_error = np.linalg.norm(action[:, :2] - cmd[:, :2], axis=1) / planar_scale
        intent_reward = np.exp(-planar_error * planar_error)
        projection = np.sum(action[:, :2] * cmd[:, :2], axis=1) / (planar_scale * planar_scale)
        yaw_error = np.abs(action[:, 2] - cmd[:, 2]) / max(self._cfg.command.max_yaw_rate, 1e-6)
        yaw_reward = np.exp(-yaw_error * yaw_error)
        clearance_penalty = np.exp(-np.maximum(self._nearest_clearance, 0.0) / 0.35)
        action_rate = np.linalg.norm(action - self._last_action, axis=1)
        high = self.action_space.high.astype(self._dtype)
        speed_violation = np.maximum(np.abs(action) - high, 0.0).sum(axis=1)
        reward = (
            cfg.intent * intent_reward
            + cfg.intent_projection * projection
            + cfg.yaw_intent * yaw_reward
            - cfg.clearance * clearance_penalty
            + cfg.action_rate * action_rate
            + cfg.speed_limit * speed_violation
            + cfg.collision * self._collision.astype(self._dtype)
        )
        return reward.astype(self._dtype)

    def _info(self, env_indices: np.ndarray) -> dict[str, Any]:
        return {
            "commands": self._commands[env_indices].copy(),
            "nearest_clearance": self._nearest_clearance[env_indices].copy(),
            "collision": self._collision[env_indices].copy(),
        }

    @staticmethod
    def _wrap_angle(angle: np.ndarray) -> np.ndarray:
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(get_global_dtype())
