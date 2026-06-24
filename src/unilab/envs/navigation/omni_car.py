from __future__ import annotations

import time
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend.base import BackendPlayRenderPlan, normalize_play_render_mode
from unilab.base.base import ABEnv, EnvCfg, EnvPlayCapabilities
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


@dataclass
class OmniCarPhysicalLimitCfg:
    max_x_speed: float = 2.0
    max_y_speed: float = 2.0
    max_yaw_rate: float = 2.0
    max_x_accel: float = 3.0
    max_y_accel: float = 3.0
    max_yaw_accel: float = 4.0


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
    physical_limits: OmniCarPhysicalLimitCfg = field(default_factory=OmniCarPhysicalLimitCfg)
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
        limits = self.physical_limits
        if min(limits.max_x_speed, limits.max_y_speed, limits.max_yaw_rate) <= 0.0:
            raise ValueError("physical velocity limits must be positive")
        if min(limits.max_x_accel, limits.max_y_accel, limits.max_yaw_accel) <= 0.0:
            raise ValueError("physical acceleration limits must be positive")


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
        limits = self._cfg.physical_limits
        high = np.asarray(
            [limits.max_x_speed, limits.max_y_speed, limits.max_yaw_rate], dtype=np.float32
        )
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

        limited = self._apply_physical_limits(actions)
        self._integrate(limited)
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
        reward = self._compute_reward(limited)
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

        self._last_action = limited.copy()
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

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        return EnvPlayCapabilities(supports_native_interactive_renderer=True)

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "interactive" if mode == "auto" else mode
        if effective_mode == "none":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if effective_mode == "record":
            raise NotImplementedError(
                "OmniCarGridAvoidance currently supports native interactive MuJoCo "
                "playback, not headless video recording."
            )
        return BackendPlayRenderPlan(
            mode="interactive",
            headless=False,
            record_video=False,
            num_steps=int(play_steps) if play_steps is not None else None,
            output_video=None,
        )

    def run_playback(
        self,
        *,
        initialize: Any,
        step: Any,
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Any = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Any = None,
    ) -> str | None:
        del output_video, render_spacing, render_offset_mode, frame_state_getter, extra_data_getter
        if headless or record_video:
            raise NotImplementedError("OmniCarGridAvoidance playback is interactive-only.")

        obs = initialize()
        print("[omni_car] Opening native OpenGL viewer. Close the window or press Esc to quit.")
        self._run_opengl_playback(obs, step, num_steps=num_steps, camera_kwargs=camera_kwargs)
        return None

    def _run_opengl_playback(
        self,
        obs: Any,
        step: Any,
        *,
        num_steps: int | None,
        camera_kwargs: dict[str, Any] | None,
    ) -> None:
        import glfw
        from OpenGL import GL, GLU

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW for OmniCarGridAvoidance viewer.")
        window = None
        try:
            glfw.window_hint(glfw.SAMPLES, 4)
            window = glfw.create_window(1280, 800, "UniLab OmniCar Grid Avoidance", None, None)
            if window is None:
                raise RuntimeError("Failed to create GLFW window for OmniCarGridAvoidance viewer.")
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glClearColor(0.06, 0.07, 0.08, 1.0)

            step_count = 0
            while not glfw.window_should_close(window) and (
                num_steps is None or step_count < num_steps
            ):
                start = time.perf_counter()
                self._draw_opengl_frame(GL, GLU, glfw, window, camera_kwargs)
                glfw.swap_buffers(window)
                glfw.poll_events()
                obs = step(obs)
                step_count += 1
                sleep_s = self._cfg.ctrl_dt - (time.perf_counter() - start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        finally:
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()

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

    def _apply_physical_limits(self, actions: np.ndarray) -> np.ndarray:
        limits = self._cfg.physical_limits
        high = np.asarray(
            [limits.max_x_speed, limits.max_y_speed, limits.max_yaw_rate], dtype=self._dtype
        )
        accel = np.asarray(
            [limits.max_x_accel, limits.max_y_accel, limits.max_yaw_accel], dtype=self._dtype
        )
        target = np.clip(actions, -high, high).astype(self._dtype)
        delta_limit = accel * self._cfg.ctrl_dt
        delta = np.clip(target - self._velocity, -delta_limit, delta_limit)
        return np.clip(self._velocity + delta, -high, high).astype(self._dtype)

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
        reward = (
            cfg.intent * intent_reward
            + cfg.intent_projection * projection
            + cfg.yaw_intent * yaw_reward
            - cfg.clearance * clearance_penalty
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

    def _viewer_xml(self) -> str:
        return f"""
<mujoco model="omni_car_grid_avoidance_viewer">
  <option timestep="{self._cfg.ctrl_dt}" gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="50"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".18 .20 .22" rgb2=".24 .27 .30"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 -4 6" dir="0 1 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="overview" pos="0 -5 4" xyaxes="1 0 0 0 0.65 0.76"/>
    <geom name="floor" type="plane" size="8 8 0.01" material="grid"/>
  </worldbody>
</mujoco>
"""

    def _draw_opengl_frame(
        self,
        GL: Any,
        GLU: Any,
        glfw: Any,
        window: Any,
        camera_kwargs: dict[str, Any] | None,
    ) -> None:
        width, height = glfw.get_framebuffer_size(window)
        width = max(1, int(width))
        height = max(1, int(height))
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, width / height, 0.05, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        kwargs = camera_kwargs or {}
        distance = float(kwargs.get("cam_distance", 5.0) or 5.0)
        pose = self._pose[0]
        eye = np.array([pose[0] - distance * 0.55, pose[1] - distance * 0.85, distance * 0.62])
        center = np.array([pose[0], pose[1], 0.0])
        GLU.gluLookAt(*eye, *center, 0.0, 0.0, 1.0)

        self._gl_draw_floor(GL)
        self._gl_draw_grid_footprint(GL, pose)
        self._gl_draw_obstacles(GL)
        self._gl_draw_car(GL, pose)
        self._gl_draw_arrows(GL, pose)

    def _gl_draw_floor(self, GL: Any) -> None:
        extent = 6
        GL.glLineWidth(1.0)
        GL.glColor4f(0.28, 0.31, 0.34, 1.0)
        GL.glBegin(GL.GL_LINES)
        for i in range(-extent, extent + 1):
            GL.glVertex3f(float(i), -float(extent), 0.0)
            GL.glVertex3f(float(i), float(extent), 0.0)
            GL.glVertex3f(-float(extent), float(i), 0.0)
            GL.glVertex3f(float(extent), float(i), 0.0)
        GL.glEnd()

    def _gl_draw_grid_footprint(self, GL: Any, pose: np.ndarray) -> None:
        GL.glPushMatrix()
        GL.glTranslatef(float(pose[0]), float(pose[1]), 0.01)
        GL.glRotatef(float(np.degrees(pose[2])), 0.0, 0.0, 1.0)
        half = float(self._grid_extent)
        GL.glColor4f(0.1, 0.35, 1.0, 0.12)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-half, -half, 0.0)
        GL.glVertex3f(half, -half, 0.0)
        GL.glVertex3f(half, half, 0.0)
        GL.glVertex3f(-half, half, 0.0)
        GL.glEnd()
        GL.glLineWidth(2.0)
        GL.glColor4f(0.2, 0.55, 1.0, 0.65)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(-half, -half, 0.012)
        GL.glVertex3f(half, -half, 0.012)
        GL.glVertex3f(half, half, 0.012)
        GL.glVertex3f(-half, half, 0.012)
        GL.glEnd()
        GL.glPopMatrix()

    def _gl_draw_obstacles(self, GL: Any) -> None:
        GL.glColor4f(0.95, 0.24, 0.12, 0.95)
        for center, radius in zip(self._obstacle_xy[0], self._obstacle_radius[0]):
            self._gl_cylinder(
                GL,
                x=float(center[0]),
                y=float(center[1]),
                radius=float(radius),
                height=0.32,
            )

    def _gl_draw_car(self, GL: Any, pose: np.ndarray) -> None:
        GL.glPushMatrix()
        GL.glTranslatef(float(pose[0]), float(pose[1]), 0.10)
        GL.glRotatef(float(np.degrees(pose[2])), 0.0, 0.0, 1.0)
        GL.glColor4f(0.15, 0.45, 1.0, 1.0)
        self._gl_box(
            GL,
            half_x=self._cfg.body.length_m * 0.5,
            half_y=self._cfg.body.width_m * 0.5,
            half_z=0.10,
        )
        GL.glColor4f(0.95, 0.95, 0.95, 1.0)
        GL.glBegin(GL.GL_TRIANGLES)
        GL.glVertex3f(self._cfg.body.length_m * 0.30, 0.0, 0.14)
        GL.glVertex3f(self._cfg.body.length_m * 0.04, self._cfg.body.width_m * 0.22, 0.14)
        GL.glVertex3f(self._cfg.body.length_m * 0.04, -self._cfg.body.width_m * 0.22, 0.14)
        GL.glEnd()
        GL.glPopMatrix()

    def _gl_draw_arrows(self, GL: Any, pose: np.ndarray) -> None:
        origin = np.array([pose[0], pose[1], 0.34], dtype=np.float64)
        command = self._body_velocity_to_world(0, self._commands[0, :2])
        velocity = self._body_velocity_to_world(0, self._velocity[0, :2])
        self._gl_arrow(
            GL,
            origin + np.array([0.0, 0.0, 0.08]),
            origin + np.array([command[0], command[1], 0.08]) * 0.45,
            rgba=(0.1, 0.95, 0.2, 1.0),
        )
        self._gl_arrow(
            GL,
            origin - np.array([0.0, 0.0, 0.08]),
            origin - np.array([0.0, 0.0, 0.08]) + np.array([velocity[0], velocity[1], 0.0]) * 0.45,
            rgba=(0.2, 0.55, 1.0, 1.0),
        )

    @staticmethod
    def _gl_box(GL: Any, *, half_x: float, half_y: float, half_z: float) -> None:
        x, y, z = float(half_x), float(half_y), float(half_z)
        vertices = [
            (-x, -y, -z),
            (x, -y, -z),
            (x, y, -z),
            (-x, y, -z),
            (-x, -y, z),
            (x, -y, z),
            (x, y, z),
            (-x, y, z),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]
        GL.glBegin(GL.GL_QUADS)
        for face in faces:
            for idx in face:
                GL.glVertex3f(*vertices[idx])
        GL.glEnd()

    @staticmethod
    def _gl_cylinder(
        GL: Any,
        *,
        x: float,
        y: float,
        radius: float,
        height: float,
        segments: int = 32,
    ) -> None:
        z0 = 0.0
        z1 = float(height)
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            GL.glVertex3f(float(px), float(py), z0)
            GL.glVertex3f(float(px), float(py), z1)
        GL.glEnd()
        for z in (z0, z1):
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3f(x, y, z)
            for i in range(segments + 1):
                angle = 2.0 * np.pi * i / segments
                GL.glVertex3f(
                    float(x + radius * np.cos(angle)),
                    float(y + radius * np.sin(angle)),
                    z,
                )
            GL.glEnd()

    @staticmethod
    def _gl_arrow(GL: Any, start: np.ndarray, end: np.ndarray, *, rgba: tuple[float, ...]) -> None:
        GL.glColor4f(*rgba)
        GL.glLineWidth(4.0)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(float(start[0]), float(start[1]), float(start[2]))
        GL.glVertex3f(float(end[0]), float(end[1]), float(end[2]))
        GL.glEnd()

    def _configure_viewer_camera(self, viewer: Any, camera_kwargs: dict[str, Any] | None) -> None:
        if not hasattr(viewer, "cam"):
            return
        kwargs = camera_kwargs or {}
        viewer.cam.distance = float(kwargs.get("cam_distance", 5.0))
        viewer.cam.elevation = float(kwargs.get("cam_elevation", -55.0))
        viewer.cam.azimuth = float(kwargs.get("cam_azimuth", 90.0))
        viewer.cam.lookat[0] = float(self._pose[0, 0])
        viewer.cam.lookat[1] = float(self._pose[0, 1])
        viewer.cam.lookat[2] = 0.0

    def _draw_viewer_scene(self, viewer: Any, mujoco: Any) -> None:
        scene = viewer.user_scn
        scene.ngeom = 0
        env_id = 0
        pose = self._pose[env_id]
        yaw = float(pose[2])
        rot = self._yaw_matrix(yaw)
        car_pos = np.array([pose[0], pose[1], 0.08], dtype=np.float64)

        if hasattr(viewer, "cam"):
            viewer.cam.lookat[0] = float(pose[0])
            viewer.cam.lookat[1] = float(pose[1])

        self._add_box(
            scene,
            mujoco,
            pos=np.array([pose[0], pose[1], 0.012], dtype=np.float64),
            size=np.array([self._grid_extent, self._grid_extent, 0.004], dtype=np.float64),
            mat=rot,
            rgba=np.array([0.15, 0.35, 1.0, 0.12], dtype=np.float32),
        )
        self._add_box(
            scene,
            mujoco,
            pos=car_pos,
            size=np.array(
                [self._cfg.body.length_m * 0.5, self._cfg.body.width_m * 0.5, 0.08],
                dtype=np.float64,
            ),
            mat=rot,
            rgba=np.array([0.15, 0.45, 1.0, 1.0], dtype=np.float32),
        )

        for center, radius in zip(self._obstacle_xy[env_id], self._obstacle_radius[env_id]):
            self._add_cylinder(
                scene,
                mujoco,
                pos=np.array([center[0], center[1], 0.16], dtype=np.float64),
                radius=float(radius),
                half_height=0.16,
                rgba=np.array([0.95, 0.25, 0.12, 0.9], dtype=np.float32),
            )

        command_world = self._body_velocity_to_world(env_id, self._commands[env_id, :2])
        velocity_world = self._body_velocity_to_world(env_id, self._velocity[env_id, :2])
        arrow_origin = np.array([pose[0], pose[1], 0.28], dtype=np.float64)
        self._add_arrow(
            scene,
            mujoco,
            arrow_origin + np.array([0.0, 0.0, 0.08]),
            arrow_origin + np.array([command_world[0], command_world[1], 0.08]) * 0.4,
            width=0.035,
            rgba=np.array([0.1, 0.95, 0.2, 1.0], dtype=np.float32),
        )
        self._add_arrow(
            scene,
            mujoco,
            arrow_origin - np.array([0.0, 0.0, 0.08]),
            arrow_origin
            - np.array([0.0, 0.0, 0.08])
            + np.array([velocity_world[0], velocity_world[1], 0.0]) * 0.4,
            width=0.025,
            rgba=np.array([0.2, 0.55, 1.0, 1.0], dtype=np.float32),
        )

    def _body_velocity_to_world(self, env_id: int, velocity_xy: np.ndarray) -> np.ndarray:
        yaw = float(self._pose[env_id, 2])
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array(
            [c * velocity_xy[0] - s * velocity_xy[1], s * velocity_xy[0] + c * velocity_xy[1]],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_matrix(yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _add_box(
        scene: Any,
        mujoco: Any,
        *,
        pos: np.ndarray,
        size: np.ndarray,
        mat: np.ndarray,
        rgba: np.ndarray,
    ) -> bool:
        if scene.ngeom >= scene.maxgeom:
            return False
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            pos,
            mat.reshape(-1),
            rgba,
        )
        scene.ngeom += 1
        return True

    @staticmethod
    def _add_cylinder(
        scene: Any,
        mujoco: Any,
        *,
        pos: np.ndarray,
        radius: float,
        half_height: float,
        rgba: np.ndarray,
    ) -> bool:
        if scene.ngeom >= scene.maxgeom:
            return False
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([radius, half_height, 0.0], dtype=np.float64),
            pos,
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        scene.ngeom += 1
        return True

    @staticmethod
    def _add_arrow(
        scene: Any,
        mujoco: Any,
        start: np.ndarray,
        end: np.ndarray,
        *,
        width: float,
        rgba: np.ndarray,
    ) -> bool:
        if scene.ngeom >= scene.maxgeom:
            return False
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1
        return True
