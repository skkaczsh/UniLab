from __future__ import annotations

import math
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
    max_y_speed: float = 1.0
    max_yaw_rate: float = 2.0
    resample_interval_s: float = 2.0
    deadband: float = 0.15
    smoothing_tau_s: float = 0.40


@dataclass
class OmniCarGridCfg:
    size: int = 80
    cell_size: float = 0.05
    safety_margin_m: float = 0.05


@dataclass
class OmniCarBodyCfg:
    length_m: float = 0.56
    width_m: float = 0.32


@dataclass
class OmniCarObstacleCfg:
    count: int = 14
    radius_min_m: float = 0.10
    radius_max_m: float = 0.28
    box_length_min_m: float = 0.22
    box_length_max_m: float = 0.75
    box_width_min_m: float = 0.12
    box_width_max_m: float = 0.45
    wall_length_min_m: float = 1.0
    wall_length_max_m: float = 2.4
    wall_width_min_m: float = 0.08
    wall_width_max_m: float = 0.18
    circle_fraction: float = 0.50
    box_fraction: float = 0.35
    wall_fraction: float = 0.15
    spawn_radius_m: float = 3.2
    keepout_radius_m: float = 0.75


@dataclass
class OmniCarLargeSceneCfg:
    enabled: bool = False
    world_size_m: float = 32.0
    static_obstacle_count: int = 220
    max_local_static_obstacles: int = 72
    max_dynamic_agents: int = 24
    dense_region_count: int = 5
    dense_region_fraction: float = 0.55
    dense_region_radius_min_m: float = 2.0
    dense_region_radius_max_m: float = 5.0
    border_wall_segments_per_side: int = 24
    border_wall_thickness_m: float = 0.35
    agent_collision_radius_m: float = 0.34
    agent_spawn_keepout_m: float = 0.55
    reset_on_timeout: bool = False
    stagnation_warmup_steps: int = 240
    stagnation_window_steps: int = 400
    stagnation_min_return_delta: float = 1.0
    resample_scene_on_full_reset: bool = False


@dataclass
class OmniCarRewardCfg:
    intent: float = 14.0
    intent_projection: float = 5.0
    yaw_intent: float = 4.2
    response: float = 7.0
    vx_track: float = 0.0
    vy_track: float = 0.0
    vyaw_track: float = 0.0
    vx_diff: float = 8.0
    vy_diff: float = 8.0
    vyaw_diff: float = 10.0
    vx_jerk: float = 3.2
    vy_jerk: float = 3.2
    vyaw_jerk: float = 4.8
    clearance: float = 1.2
    collision: float = -12.0


@dataclass
class OmniCarPhysicalLimitCfg:
    max_x_speed: float = 2.0
    max_y_speed: float = 1.0
    max_yaw_rate: float = 2.0
    max_x_accel: float = 3.0
    max_y_accel: float = 3.0
    max_yaw_accel: float = 4.0


@dataclass
class OmniCarHumanCommandCfg:
    enabled: bool = False
    env_index: int | str = "random"
    replay_fanout: int = 0
    backend: str = "pygame"
    joystick_index: int = 0
    require_joystick: bool = True
    deadzone: float = 0.08
    smoothing_tau_s: float = 0.10
    axis_vx: int = 1
    axis_vy: int = 0
    axis_vyaw: int = 2
    invert_vx: bool = True
    invert_vy: bool = False
    invert_vyaw: bool = False


@registry.envcfg("OmniCarGridAvoidance")
@dataclass
class OmniCarGridAvoidanceCfg(EnvCfg):
    """Config for a rectangular omnidirectional car in a local occupancy grid."""

    sim_dt: float = 0.05
    ctrl_dt: float = 0.05
    max_episode_seconds: float = 60.0
    grid_history_len: int = 10
    obs_history_len: int = 24
    command: OmniCarCommandCfg = field(default_factory=OmniCarCommandCfg)
    grid: OmniCarGridCfg = field(default_factory=OmniCarGridCfg)
    body: OmniCarBodyCfg = field(default_factory=OmniCarBodyCfg)
    obstacles: OmniCarObstacleCfg = field(default_factory=OmniCarObstacleCfg)
    large_scene: OmniCarLargeSceneCfg = field(default_factory=OmniCarLargeSceneCfg)
    reward: OmniCarRewardCfg = field(default_factory=OmniCarRewardCfg)
    physical_limits: OmniCarPhysicalLimitCfg = field(default_factory=OmniCarPhysicalLimitCfg)
    human_command: OmniCarHumanCommandCfg = field(default_factory=OmniCarHumanCommandCfg)
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
        if self.obstacles.box_length_max_m < self.obstacles.box_length_min_m:
            raise ValueError("obstacles.box_length_max_m must be >= box_length_min_m")
        if self.obstacles.box_width_max_m < self.obstacles.box_width_min_m:
            raise ValueError("obstacles.box_width_max_m must be >= box_width_min_m")
        if self.obstacles.wall_length_max_m < self.obstacles.wall_length_min_m:
            raise ValueError("obstacles.wall_length_max_m must be >= wall_length_min_m")
        if self.obstacles.wall_width_max_m < self.obstacles.wall_width_min_m:
            raise ValueError("obstacles.wall_width_max_m must be >= wall_width_min_m")
        fractions = (
            self.obstacles.circle_fraction,
            self.obstacles.box_fraction,
            self.obstacles.wall_fraction,
        )
        if min(fractions) < 0.0 or sum(fractions) <= 0.0:
            raise ValueError("obstacle type fractions must be non-negative with positive sum")
        if self.obs_history_len <= 0:
            raise ValueError("obs_history_len must be a positive integer")
        if self.grid_history_len <= 0:
            raise ValueError("grid_history_len must be a positive integer")
        scene = self.large_scene
        if scene.world_size_m <= 2.0:
            raise ValueError("large_scene.world_size_m must be greater than 2m")
        if scene.static_obstacle_count < 0:
            raise ValueError("large_scene.static_obstacle_count must be non-negative")
        if scene.max_local_static_obstacles <= 0:
            raise ValueError("large_scene.max_local_static_obstacles must be positive")
        if scene.max_dynamic_agents <= 0:
            raise ValueError("large_scene.max_dynamic_agents must be positive")
        if scene.dense_region_count <= 0:
            raise ValueError("large_scene.dense_region_count must be positive")
        if not 0.0 <= scene.dense_region_fraction <= 1.0:
            raise ValueError("large_scene.dense_region_fraction must be in [0, 1]")
        if scene.dense_region_radius_max_m < scene.dense_region_radius_min_m:
            raise ValueError("large_scene dense radius max must be >= min")
        if scene.border_wall_segments_per_side <= 0:
            raise ValueError("large_scene.border_wall_segments_per_side must be positive")
        if min(scene.border_wall_thickness_m, scene.agent_collision_radius_m) <= 0.0:
            raise ValueError("large_scene border thickness and agent radius must be positive")
        if scene.stagnation_window_steps <= 0:
            raise ValueError("large_scene.stagnation_window_steps must be positive")
        limits = self.physical_limits
        if min(limits.max_x_speed, limits.max_y_speed, limits.max_yaw_rate) <= 0.0:
            raise ValueError("physical velocity limits must be positive")
        if min(limits.max_x_accel, limits.max_y_accel, limits.max_yaw_accel) <= 0.0:
            raise ValueError("physical acceleration limits must be positive")
        human = self.human_command
        if human.replay_fanout < 0:
            raise ValueError("human_command.replay_fanout must be non-negative")
        if not 0.0 <= human.deadzone < 1.0:
            raise ValueError("human_command.deadzone must be in [0, 1)")
        if human.smoothing_tau_s < 0.0:
            raise ValueError("human_command.smoothing_tau_s must be non-negative")
        if min(human.axis_vx, human.axis_vy, human.axis_vyaw) < 0:
            raise ValueError("human_command axis indices must be non-negative")


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
    _OBSTACLE_CIRCLE = 0
    _OBSTACLE_BOX = 1
    _OBSTACLE_WALL = 2
    _COMMAND_MODE_MASKS = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    _COMMAND_AMPLITUDE_BANDS = (
        (0.15, 0.35),
        (0.35, 0.65),
        (0.65, 1.00),
    )

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
        self._grid_history_len = int(max(cfg.grid_history_len, 1))
        self._obs_history_len = int(max(cfg.obs_history_len, 1))
        self._grid_dim = cfg.grid.size * cfg.grid.size
        self._grid_stack_dim = self._grid_history_len * self._grid_dim
        self._history_block_dim = self._obs_history_len * 3
        self._history_dim = self._history_block_dim * 3
        self._obs_dim = self._grid_stack_dim + 3 + 3 + 3 + self._history_dim
        self._critic_dim = self._obs_dim + 1 + 1 + 3
        self._rng = np.random.default_rng(cfg.seed)
        self._dtype = get_global_dtype()
        self._all_env_indices = np.arange(self._num_envs, dtype=np.int32)
        self._pose = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._velocity = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._last_action = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._last_action_delta = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._velocity_limit = self._make_velocity_limit()
        self._accel_limit = self._make_accel_limit()
        self._accel_delta_limit = self._accel_limit * self._cfg.ctrl_dt
        self._raw_commands = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._commands = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._human_command_enabled = bool(cfg.human_command.enabled)
        self._human_command_env_id = self._select_human_command_env_id()
        self._human_command_env_ids = self._select_human_command_env_ids()
        self._human_command = np.zeros((3,), dtype=self._dtype)
        self._human_command_connected = False
        self._human_controller_name = ""
        self._human_pygame: Any | None = None
        self._human_joystick: Any | None = None
        if self._human_command_enabled:
            self._ensure_human_command_backend()
        self._command_history = np.zeros(
            (self._num_envs, self._obs_history_len, 3), dtype=self._dtype
        )
        self._velocity_history = np.zeros(
            (self._num_envs, self._obs_history_len, 3), dtype=self._dtype
        )
        self._action_history = np.zeros(
            (self._num_envs, self._obs_history_len, 3), dtype=self._dtype
        )
        self._obstacle_xy = np.zeros((self._num_envs, cfg.obstacles.count, 2), dtype=self._dtype)
        self._obstacle_radius = np.zeros((self._num_envs, cfg.obstacles.count), dtype=self._dtype)
        self._obstacle_half_extents = np.zeros(
            (self._num_envs, cfg.obstacles.count, 2), dtype=self._dtype
        )
        self._obstacle_yaw = np.zeros((self._num_envs, cfg.obstacles.count), dtype=self._dtype)
        self._obstacle_type = np.zeros((self._num_envs, cfg.obstacles.count), dtype=np.int8)
        self._large_scene_enabled = bool(cfg.large_scene.enabled)
        border_obstacle_count = (
            4 * int(cfg.large_scene.border_wall_segments_per_side)
            if self._large_scene_enabled
            else 0
        )
        self._scene_obstacle_count = (
            int(cfg.large_scene.static_obstacle_count) + border_obstacle_count
            if self._large_scene_enabled
            else 0
        )
        self._scene_initialized = False
        self._scene_obstacle_xy = np.zeros((self._scene_obstacle_count, 2), dtype=self._dtype)
        self._scene_obstacle_radius = np.zeros((self._scene_obstacle_count,), dtype=self._dtype)
        self._scene_obstacle_half_extents = np.zeros(
            (self._scene_obstacle_count, 2), dtype=self._dtype
        )
        self._scene_obstacle_yaw = np.zeros((self._scene_obstacle_count,), dtype=self._dtype)
        self._scene_obstacle_type = np.zeros((self._scene_obstacle_count,), dtype=np.int8)
        self._nearest_clearance = np.zeros((self._num_envs,), dtype=self._dtype)
        self._collision = np.zeros((self._num_envs,), dtype=bool)
        self._static_collision = np.zeros((self._num_envs,), dtype=bool)
        self._agent_collision = np.zeros((self._num_envs,), dtype=bool)
        self._border_collision = np.zeros((self._num_envs,), dtype=bool)
        self._stagnated = np.zeros((self._num_envs,), dtype=bool)
        self._episode_return = np.zeros((self._num_envs,), dtype=self._dtype)
        self._best_episode_return = np.zeros((self._num_envs,), dtype=self._dtype)
        self._steps_since_reward_improvement = np.zeros((self._num_envs,), dtype=np.uint32)
        self._truncated = np.zeros((self._num_envs,), dtype=bool)
        self._tracking_error = np.zeros((self._num_envs,), dtype=self._dtype)
        self._response_progress = np.zeros((self._num_envs,), dtype=self._dtype)
        self._track_cost = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._diff_cost = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._jerk_cost = np.zeros((self._num_envs, 3), dtype=self._dtype)
        self._grid_buffer = np.zeros(
            (self._num_envs, cfg.grid.size, cfg.grid.size), dtype=self._dtype
        )
        self._grid_history = np.zeros(
            (self._num_envs, self._grid_history_len, cfg.grid.size, cfg.grid.size),
            dtype=self._dtype,
        )
        self._grid_history_initialized = np.zeros((self._num_envs,), dtype=bool)
        self._obs_buffer = np.zeros((self._num_envs, self._obs_dim), dtype=self._dtype)
        self._critic_buffer = np.zeros((self._num_envs, self._critic_dim), dtype=self._dtype)

        xs = np.arange(cfg.grid.size, dtype=np.float32) + 0.5 - cfg.grid.size / 2.0
        ys = np.arange(cfg.grid.size, dtype=np.float32) + 0.5 - cfg.grid.size / 2.0
        gx, gy = np.meshgrid(xs * cfg.grid.cell_size, ys * cfg.grid.cell_size, indexing="ij")
        self._grid_axis = (xs * cfg.grid.cell_size).astype(self._dtype)
        self._grid_points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(self._dtype)
        self._grid_x = self._grid_points[:, 0]
        self._grid_y = self._grid_points[:, 1]
        self._grid_size = int(cfg.grid.size)
        self._grid_cell_size = float(cfg.grid.cell_size)
        self._grid_half_index = self._grid_size / 2.0 - 0.5
        self._grid_pad = self._grid_cell_size * 0.5
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
        # Actor uses deployable signals only. Critic appends privileged train-time state.
        return {
            "obs": self._obs_dim,
            "critic": self._critic_dim,
        }

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

    def _select_human_command_env_id(self) -> int:
        if not self._human_command_enabled:
            return -1
        requested = self._cfg.human_command.env_index
        if isinstance(requested, str):
            if requested != "random":
                raise ValueError("human_command.env_index must be an integer or 'random'")
            return int(self._rng.integers(0, self._num_envs))
        env_id = int(requested)
        if not 0 <= env_id < self._num_envs:
            raise ValueError(
                f"human_command.env_index must be in [0, {self._num_envs}), got {env_id}"
            )
        return env_id

    def _select_human_command_env_ids(self) -> np.ndarray:
        if self._human_command_env_id < 0:
            return np.zeros((0,), dtype=np.int32)
        ids = [self._human_command_env_id]
        fanout = min(int(self._cfg.human_command.replay_fanout), max(self._num_envs - 1, 0))
        if fanout > 0:
            candidates = np.setdiff1d(self._all_env_indices, np.asarray(ids, dtype=np.int32))
            replay_ids = self._rng.choice(candidates, size=fanout, replace=False)
            ids.extend(int(env_id) for env_id in replay_ids)
        return np.asarray(ids, dtype=np.int32)

    def _ensure_human_command_backend(self) -> None:
        backend = self._cfg.human_command.backend
        if backend == "zero":
            self._human_command_connected = True
            self._human_controller_name = "zero"
            return
        if backend != "pygame":
            raise ValueError(f"Unsupported human_command.backend: {backend}")
        if self._human_joystick is not None:
            return
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - exercised when optional dep missing
            raise RuntimeError(
                "human_command.backend=pygame requires pygame. Install project dependencies "
                "with `uv sync` or disable env.human_command.enabled."
            ) from exc
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            self._human_command_connected = False
            if self._cfg.human_command.require_joystick:
                raise RuntimeError("human_command is enabled but pygame found no joystick")
            return
        joystick_index = int(self._cfg.human_command.joystick_index)
        if not 0 <= joystick_index < joystick_count:
            raise RuntimeError(
                f"human_command.joystick_index={joystick_index} but pygame found "
                f"{joystick_count} joystick(s)"
            )
        joystick = pygame.joystick.Joystick(joystick_index)
        joystick.init()
        self._human_pygame = pygame
        self._human_joystick = joystick
        self._human_command_connected = True
        self._human_controller_name = joystick.get_name()

    def _read_human_command(self) -> np.ndarray:
        backend = self._cfg.human_command.backend
        if backend == "zero":
            return np.zeros((3,), dtype=self._dtype)
        self._ensure_human_command_backend()
        if self._human_joystick is None:
            return np.zeros((3,), dtype=self._dtype)
        self._human_pygame.event.pump()
        joystick = self._human_joystick
        max_axis = max(
            self._cfg.human_command.axis_vx,
            self._cfg.human_command.axis_vy,
            self._cfg.human_command.axis_vyaw,
        )
        if joystick.get_numaxes() <= max_axis:
            raise RuntimeError(
                f"Joystick '{self._human_controller_name}' exposes {joystick.get_numaxes()} "
                f"axes, but human_command needs axis {max_axis}"
            )
        axes = np.asarray([joystick.get_axis(axis_id) for axis_id in range(max_axis + 1)])
        return self._map_human_axes_to_command(axes)

    def _map_human_axes_to_command(self, axes: np.ndarray) -> np.ndarray:
        human = self._cfg.human_command

        def shaped(axis_id: int, invert: bool) -> float:
            value = float(axes[axis_id])
            if invert:
                value = -value
            magnitude = abs(value)
            if magnitude < human.deadzone:
                return 0.0
            scaled = (magnitude - human.deadzone) / (1.0 - human.deadzone)
            return math.copysign(scaled, value)

        normalized = np.asarray(
            [
                shaped(human.axis_vx, human.invert_vx),
                shaped(human.axis_vy, human.invert_vy),
                shaped(human.axis_vyaw, human.invert_vyaw),
            ],
            dtype=self._dtype,
        )
        return (normalized * self._velocity_limit).astype(self._dtype)

    def _refresh_human_commands(self) -> None:
        if not self._human_command_enabled or self._human_command_env_ids.size == 0:
            return
        self._human_command = self._read_human_command()
        self._raw_commands[self._human_command_env_ids] = self._human_command

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
        if self._large_scene_enabled:
            full_reset = env_indices.size == self._num_envs
            if (
                not self._scene_initialized
                or (full_reset and self._cfg.large_scene.resample_scene_on_full_reset)
            ):
                self._sample_large_scene()
            self._reset_large_scene_agents(env_indices)
        else:
            self._pose[env_indices] = 0.0
        self._velocity[env_indices] = 0.0
        self._last_action[env_indices] = 0.0
        self._last_action_delta[env_indices] = 0.0
        sampled_commands = self._sample_commands(env_indices.size)
        self._raw_commands[env_indices] = sampled_commands
        self._commands[env_indices] = sampled_commands.copy()
        self._refresh_human_commands()
        human_reset = np.intersect1d(env_indices, self._human_command_env_ids, assume_unique=False)
        if human_reset.size > 0:
            self._commands[human_reset] = self._raw_commands[human_reset]
        self._seed_history(env_indices)
        if not self._large_scene_enabled:
            self._sample_obstacles(env_indices)
        self._collision[env_indices] = False
        self._static_collision[env_indices] = False
        self._agent_collision[env_indices] = False
        self._border_collision[env_indices] = False
        self._stagnated[env_indices] = False
        self._episode_return[env_indices] = 0.0
        self._best_episode_return[env_indices] = 0.0
        self._steps_since_reward_improvement[env_indices] = 0
        self._tracking_error[env_indices] = 0.0
        self._response_progress[env_indices] = 0.0
        self._track_cost[env_indices] = 0.0
        self._diff_cost[env_indices] = 0.0
        self._jerk_cost[env_indices] = 0.0
        self._nearest_clearance[env_indices] = self._compute_clearance(env_indices)
        self._grid_history_initialized[env_indices] = False
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
        action_delta = limited - self._last_action
        self._integrate(limited)
        self._state.info["steps"] += 1

        resample_steps = max(
            1, int(round(self._cfg.command.resample_interval_s / self._cfg.ctrl_dt))
        )
        resample_mask = self._state.info["steps"] % resample_steps == 0
        if np.any(resample_mask):
            self._raw_commands[resample_mask] = self._sample_commands(
                int(np.count_nonzero(resample_mask))
            )

        self._refresh_human_commands()
        self._update_commands()

        self._nearest_clearance = self._compute_clearance(self._all_env_indices)
        self._collision = self._nearest_clearance <= 0.0
        reward = self._compute_reward(limited)
        self._update_reward_progress(reward)
        log_snapshot = {
            "commands": self._commands.copy(),
            "nearest_clearance": self._nearest_clearance.copy(),
            "collision": self._collision.copy(),
            "static_collision": self._static_collision.copy(),
            "agent_collision": self._agent_collision.copy(),
            "border_collision": self._border_collision.copy(),
            "stagnated": self._stagnated.copy(),
            "tracking_error": self._tracking_error.copy(),
            "response_progress": self._response_progress.copy(),
            "track_cost": self._track_cost.copy(),
            "diff_cost": self._diff_cost.copy(),
            "jerk_cost": self._jerk_cost.copy(),
            "human_command": self._human_command.copy(),
            "human_command_env_id": self._human_command_env_id,
            "human_command_agent_count": self._human_command_env_ids.size,
        }
        terminated = self._collision.copy()
        self._truncated.fill(False)
        if self._cfg.max_episode_steps is not None and (
            not self._large_scene_enabled or self._cfg.large_scene.reset_on_timeout
        ):
            np.greater_equal(
                self._state.info["steps"], self._cfg.max_episode_steps, out=self._truncated
            )
        done = terminated | self._truncated | self._stagnated

        self._append_history()

        obs = self._build_obs(self._all_env_indices)
        self._state.info["commands"] = self._commands.copy()
        self._state.info["nearest_clearance"] = self._nearest_clearance.copy()
        self._state.info["collision"] = self._collision.copy()
        self._state.info["static_collision"] = self._static_collision.copy()
        self._state.info["agent_collision"] = self._agent_collision.copy()
        self._state.info["border_collision"] = self._border_collision.copy()
        self._state.info["stagnated"] = self._stagnated.copy()
        self._state.info["human_command_enabled"] = self._human_command_enabled
        self._state.info["human_command_env_id"] = self._human_command_env_id
        self._state.info["human_command_env_ids"] = self._human_command_env_ids.copy()
        self._state.info["human_command"] = self._human_command.copy()
        self._state.info["human_command_connected"] = self._human_command_connected
        self._state.info["human_controller_name"] = self._human_controller_name
        final_observation = None
        if np.any(done):
            final_observation = {key: value.copy() for key, value in obs.items()}

        self._last_action = limited.copy()
        self._last_action_delta = action_delta.copy()
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
            "omni_car/mean_clearance": float(np.mean(log_snapshot["nearest_clearance"])),
            "omni_car/collision_rate": float(np.mean(log_snapshot["collision"].astype(np.float32))),
            "omni_car/static_collision_rate": float(
                np.mean(log_snapshot["static_collision"].astype(np.float32))
            ),
            "omni_car/agent_collision_rate": float(
                np.mean(log_snapshot["agent_collision"].astype(np.float32))
            ),
            "omni_car/border_collision_rate": float(
                np.mean(log_snapshot["border_collision"].astype(np.float32))
            ),
            "omni_car/stagnation_reset_rate": float(
                np.mean(log_snapshot["stagnated"].astype(np.float32))
            ),
            "omni_car/command_norm": float(
                np.mean(np.linalg.norm(log_snapshot["commands"][:, :2], axis=1))
            ),
            "omni_car/tracking_error": float(np.mean(log_snapshot["tracking_error"])),
            "omni_car/response_progress": float(np.mean(log_snapshot["response_progress"])),
            "omni_car/vx_track_cost": float(np.mean(log_snapshot["track_cost"][:, 0])),
            "omni_car/vy_track_cost": float(np.mean(log_snapshot["track_cost"][:, 1])),
            "omni_car/vyaw_track_cost": float(np.mean(log_snapshot["track_cost"][:, 2])),
            "omni_car/vx_diff_cost": float(np.mean(log_snapshot["diff_cost"][:, 0])),
            "omni_car/vy_diff_cost": float(np.mean(log_snapshot["diff_cost"][:, 1])),
            "omni_car/vyaw_diff_cost": float(np.mean(log_snapshot["diff_cost"][:, 2])),
            "omni_car/vx_jerk_cost": float(np.mean(log_snapshot["jerk_cost"][:, 0])),
            "omni_car/vy_jerk_cost": float(np.mean(log_snapshot["jerk_cost"][:, 1])),
            "omni_car/vyaw_jerk_cost": float(np.mean(log_snapshot["jerk_cost"][:, 2])),
            "omni_car/human_command_env_id": float(log_snapshot["human_command_env_id"]),
            "omni_car/human_command_agent_count": float(
                log_snapshot["human_command_agent_count"]
            ),
            "omni_car/human_command_norm": float(np.linalg.norm(log_snapshot["human_command"])),
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
        if count == 0:
            return np.zeros((0, 3), dtype=self._dtype)
        cmd = self._cfg.command
        limits = np.asarray([cmd.max_x_speed, cmd.max_y_speed, cmd.max_yaw_rate], dtype=self._dtype)
        mode_count = self._COMMAND_MODE_MASKS.shape[0]
        band_count = len(self._COMMAND_AMPLITUDE_BANDS)
        mode_ids = (np.arange(count) + int(self._rng.integers(mode_count))) % mode_count
        band_ids = (np.arange(count) + int(self._rng.integers(band_count))) % band_count
        self._rng.shuffle(mode_ids)
        self._rng.shuffle(band_ids)

        sampled = np.zeros((count, 3), dtype=self._dtype)
        for row, (mode_id, band_id) in enumerate(zip(mode_ids, band_ids, strict=True)):
            active = self._COMMAND_MODE_MASKS[mode_id]
            low, high = self._COMMAND_AMPLITUDE_BANDS[band_id]
            magnitude = self._rng.uniform(low, high, size=3).astype(self._dtype) * limits
            signs = self._rng.choice(np.asarray([-1.0, 1.0], dtype=self._dtype), size=3)
            sampled[row, active] = magnitude[active] * signs[active]

        small = np.linalg.norm(sampled[:, :2], axis=1) < cmd.deadband
        sampled[small, :2] = 0.0
        return sampled

    def _seed_history(self, env_indices: np.ndarray) -> None:
        if env_indices.size == 0:
            return
        self._command_history[env_indices] = np.broadcast_to(
            self._commands[env_indices, None, :],
            (env_indices.size, self._obs_history_len, 3),
        )
        self._velocity_history[env_indices] = np.broadcast_to(
            self._velocity[env_indices, None, :],
            (env_indices.size, self._obs_history_len, 3),
        )
        self._action_history[env_indices] = np.broadcast_to(
            self._last_action[env_indices, None, :],
            (env_indices.size, self._obs_history_len, 3),
        )

    def _update_commands(self) -> None:
        if self._cfg.command.smoothing_tau_s <= 0.0:
            self._commands = self._raw_commands.copy()
            self._update_human_smoothed_commands()
            return

        alpha = float(self._cfg.ctrl_dt / (self._cfg.command.smoothing_tau_s + self._cfg.ctrl_dt))
        self._commands = ((1.0 - alpha) * self._commands + alpha * self._raw_commands).astype(
            self._dtype
        )
        self._update_human_smoothed_commands()

    def _update_human_smoothed_commands(self) -> None:
        if not self._human_command_enabled or self._human_command_env_ids.size == 0:
            return
        tau = float(self._cfg.human_command.smoothing_tau_s)
        if tau <= 0.0:
            self._commands[self._human_command_env_ids] = self._raw_commands[
                self._human_command_env_ids
            ]
            return
        alpha = float(self._cfg.ctrl_dt / (tau + self._cfg.ctrl_dt))
        ids = self._human_command_env_ids
        self._commands[ids] = (
            (1.0 - alpha) * self._commands[ids] + alpha * self._raw_commands[ids]
        ).astype(self._dtype)

    def _append_history(self) -> None:
        if self._obs_history_len == 1:
            self._command_history[:, 0] = self._commands
            self._velocity_history[:, 0] = self._velocity
            self._action_history[:, 0] = self._last_action
            return
        self._command_history[:, :-1] = self._command_history[:, 1:]
        self._velocity_history[:, :-1] = self._velocity_history[:, 1:]
        self._action_history[:, :-1] = self._action_history[:, 1:]
        self._command_history[:, -1] = self._commands
        self._velocity_history[:, -1] = self._velocity
        self._action_history[:, -1] = self._last_action

    def _append_grid_history(self, env_indices: np.ndarray, current_grid: np.ndarray) -> None:
        if env_indices.size == 0:
            return
        uninitialized = ~self._grid_history_initialized[env_indices]
        if np.any(uninitialized):
            cold_ids = env_indices[uninitialized]
            self._grid_history[cold_ids] = current_grid[uninitialized, None, :, :]
            self._grid_history_initialized[cold_ids] = True
        if np.any(~uninitialized):
            warm_ids = env_indices[~uninitialized]
            if self._grid_history_len > 1:
                self._grid_history[warm_ids, :-1] = self._grid_history[warm_ids, 1:]
            self._grid_history[warm_ids, -1] = current_grid[~uninitialized]

    def _sample_obstacles(self, env_indices: np.ndarray) -> None:
        cfg = self._cfg.obstacles
        if cfg.count == 0:
            return
        type_weights = np.asarray(
            [cfg.circle_fraction, cfg.box_fraction, cfg.wall_fraction], dtype=np.float64
        )
        type_weights = type_weights / np.sum(type_weights)
        for env_id in env_indices:
            obstacle_types = self._rng.choice(3, size=(cfg.count,), p=type_weights).astype(np.int8)
            radii = self._rng.uniform(cfg.radius_min_m, cfg.radius_max_m, size=(cfg.count,))
            half_extents = np.zeros((cfg.count, 2), dtype=self._dtype)
            yaw = self._rng.uniform(-np.pi, np.pi, size=(cfg.count,)).astype(self._dtype)
            angles = self._rng.uniform(-np.pi, np.pi, size=(cfg.count,))
            distances = self._rng.uniform(
                cfg.keepout_radius_m, cfg.spawn_radius_m, size=(cfg.count,)
            )
            xy = np.stack([np.cos(angles) * distances, np.sin(angles) * distances], axis=1)
            box_mask = obstacle_types == self._OBSTACLE_BOX
            wall_mask = obstacle_types == self._OBSTACLE_WALL
            half_extents[box_mask, 0] = 0.5 * self._rng.uniform(
                cfg.box_length_min_m, cfg.box_length_max_m, size=int(np.count_nonzero(box_mask))
            )
            half_extents[box_mask, 1] = 0.5 * self._rng.uniform(
                cfg.box_width_min_m, cfg.box_width_max_m, size=int(np.count_nonzero(box_mask))
            )
            half_extents[wall_mask, 0] = 0.5 * self._rng.uniform(
                cfg.wall_length_min_m, cfg.wall_length_max_m, size=int(np.count_nonzero(wall_mask))
            )
            half_extents[wall_mask, 1] = 0.5 * self._rng.uniform(
                cfg.wall_width_min_m, cfg.wall_width_max_m, size=int(np.count_nonzero(wall_mask))
            )
            # Bias one obstacle into the commanded path so avoidance matters during short runs.
            cmd = self._commands[env_id]
            planar_norm = float(np.linalg.norm(cmd[:2]))
            if planar_norm > 1e-6:
                direction = cmd[:2] / planar_norm
                lateral = np.asarray([-direction[1], direction[0]], dtype=self._dtype)
                xy[0] = direction * self._rng.uniform(0.9, 1.6) + lateral * self._rng.uniform(
                    -0.25, 0.25
                )
                obstacle_types[0] = self._rng.choice(3, p=type_weights)
                radii[0] = max(float(radii[0]), 0.22)
                yaw[0] = float(np.arctan2(direction[1], direction[0])) + self._rng.uniform(
                    -0.6, 0.6
                )
                if obstacle_types[0] == self._OBSTACLE_BOX:
                    half_extents[0] = [
                        0.5 * self._rng.uniform(cfg.box_length_min_m, cfg.box_length_max_m),
                        0.5 * self._rng.uniform(cfg.box_width_min_m, cfg.box_width_max_m),
                    ]
                elif obstacle_types[0] == self._OBSTACLE_WALL:
                    half_extents[0] = [
                        0.5 * self._rng.uniform(cfg.wall_length_min_m, cfg.wall_length_max_m),
                        0.5 * self._rng.uniform(cfg.wall_width_min_m, cfg.wall_width_max_m),
                    ]
            self._obstacle_xy[env_id] = xy.astype(self._dtype)
            self._obstacle_radius[env_id] = radii.astype(self._dtype)
            self._obstacle_half_extents[env_id] = half_extents
            self._obstacle_yaw[env_id] = yaw.astype(self._dtype)
            self._obstacle_type[env_id] = obstacle_types

    def _sample_large_scene(self) -> None:
        scene = self._cfg.large_scene
        obstacle_cfg = self._cfg.obstacles
        if self._scene_obstacle_count == 0:
            self._scene_initialized = True
            return

        type_weights = np.asarray(
            [
                obstacle_cfg.circle_fraction,
                obstacle_cfg.box_fraction,
                obstacle_cfg.wall_fraction,
            ],
            dtype=np.float64,
        )
        type_weights = type_weights / np.sum(type_weights)
        static_count = int(scene.static_obstacle_count)
        border_count = self._scene_obstacle_count - static_count
        half_world = 0.5 * float(scene.world_size_m)
        spawn_limit = max(
            0.5,
            half_world - max(
                scene.border_wall_thickness_m,
                obstacle_cfg.wall_length_max_m * 0.5,
                obstacle_cfg.radius_max_m,
            ),
        )

        if static_count > 0:
            obstacle_types = self._rng.choice(3, size=(static_count,), p=type_weights).astype(
                np.int8
            )
            radii = self._rng.uniform(
                obstacle_cfg.radius_min_m, obstacle_cfg.radius_max_m, size=(static_count,)
            )
            half_extents = np.zeros((static_count, 2), dtype=self._dtype)
            yaw = self._rng.uniform(-np.pi, np.pi, size=(static_count,)).astype(self._dtype)

            dense_count = int(round(static_count * scene.dense_region_fraction))
            uniform_count = static_count - dense_count
            region_centers = self._rng.uniform(
                -spawn_limit * 0.70,
                spawn_limit * 0.70,
                size=(scene.dense_region_count, 2),
            )
            region_radii = self._rng.uniform(
                scene.dense_region_radius_min_m,
                scene.dense_region_radius_max_m,
                size=(scene.dense_region_count,),
            )
            dense_xy = np.empty((dense_count, 2), dtype=self._dtype)
            if dense_count > 0:
                region_ids = self._rng.integers(0, scene.dense_region_count, size=(dense_count,))
                angles = self._rng.uniform(-np.pi, np.pi, size=(dense_count,))
                radii_scale = np.sqrt(self._rng.uniform(0.0, 1.0, size=(dense_count,)))
                offsets = np.stack(
                    [
                        np.cos(angles) * region_radii[region_ids] * radii_scale,
                        np.sin(angles) * region_radii[region_ids] * radii_scale,
                    ],
                    axis=1,
                )
                dense_xy = region_centers[region_ids] + offsets
            uniform_xy = self._rng.uniform(
                -spawn_limit, spawn_limit, size=(uniform_count, 2)
            ).astype(self._dtype)
            xy = np.concatenate([dense_xy, uniform_xy], axis=0).astype(self._dtype, copy=False)
            np.clip(xy, -spawn_limit, spawn_limit, out=xy)

            box_mask = obstacle_types == self._OBSTACLE_BOX
            wall_mask = obstacle_types == self._OBSTACLE_WALL
            half_extents[box_mask, 0] = 0.5 * self._rng.uniform(
                obstacle_cfg.box_length_min_m,
                obstacle_cfg.box_length_max_m,
                size=int(np.count_nonzero(box_mask)),
            )
            half_extents[box_mask, 1] = 0.5 * self._rng.uniform(
                obstacle_cfg.box_width_min_m,
                obstacle_cfg.box_width_max_m,
                size=int(np.count_nonzero(box_mask)),
            )
            half_extents[wall_mask, 0] = 0.5 * self._rng.uniform(
                obstacle_cfg.wall_length_min_m,
                obstacle_cfg.wall_length_max_m,
                size=int(np.count_nonzero(wall_mask)),
            )
            half_extents[wall_mask, 1] = 0.5 * self._rng.uniform(
                obstacle_cfg.wall_width_min_m,
                obstacle_cfg.wall_width_max_m,
                size=int(np.count_nonzero(wall_mask)),
            )

            self._scene_obstacle_xy[:static_count] = xy
            self._scene_obstacle_radius[:static_count] = radii.astype(self._dtype)
            self._scene_obstacle_half_extents[:static_count] = half_extents
            self._scene_obstacle_yaw[:static_count] = yaw
            self._scene_obstacle_type[:static_count] = obstacle_types

        if border_count > 0:
            start = static_count
            segments = int(scene.border_wall_segments_per_side)
            segment_len = scene.world_size_m / segments
            half_thickness = 0.5 * scene.border_wall_thickness_m
            centers = np.linspace(
                -half_world + 0.5 * segment_len,
                half_world - 0.5 * segment_len,
                segments,
                dtype=np.float64,
            )
            obstacle_id = start
            for side in (-1.0, 1.0):
                y = side * (half_world - half_thickness)
                for x in centers:
                    self._scene_obstacle_xy[obstacle_id] = [x, y]
                    self._scene_obstacle_half_extents[obstacle_id] = [
                        0.5 * segment_len,
                        half_thickness,
                    ]
                    self._scene_obstacle_yaw[obstacle_id] = 0.0
                    self._scene_obstacle_type[obstacle_id] = self._OBSTACLE_WALL
                    obstacle_id += 1
            for side in (-1.0, 1.0):
                x = side * (half_world - half_thickness)
                for y in centers:
                    self._scene_obstacle_xy[obstacle_id] = [x, y]
                    self._scene_obstacle_half_extents[obstacle_id] = [
                        0.5 * segment_len,
                        half_thickness,
                    ]
                    self._scene_obstacle_yaw[obstacle_id] = np.pi * 0.5
                    self._scene_obstacle_type[obstacle_id] = self._OBSTACLE_WALL
                    obstacle_id += 1

        self._scene_initialized = True

    def _reset_large_scene_agents(self, env_indices: np.ndarray) -> None:
        scene = self._cfg.large_scene
        half_world = 0.5 * float(scene.world_size_m)
        agent_radius = float(scene.agent_collision_radius_m)
        spawn_limit = half_world - max(
            scene.border_wall_thickness_m + agent_radius + scene.agent_spawn_keepout_m,
            agent_radius,
        )
        spawn_limit = max(spawn_limit, 0.25)
        env_set = set(int(env_id) for env_id in env_indices)
        active_ids = [int(env_id) for env_id in range(self._num_envs) if env_id not in env_set]
        accepted: list[int] = []

        for env_id in env_indices:
            env_int = int(env_id)
            pose = None
            for _ in range(128):
                candidate_xy = self._rng.uniform(-spawn_limit, spawn_limit, size=(2,)).astype(
                    self._dtype
                )
                candidate_yaw = float(self._rng.uniform(-np.pi, np.pi))
                if not self._candidate_spawn_is_clear(
                    candidate_xy,
                    active_ids + accepted,
                    agent_radius,
                    scene.agent_spawn_keepout_m,
                ):
                    continue
                pose = [candidate_xy[0], candidate_xy[1], candidate_yaw]
                break
            if pose is None:
                angle = 2.0 * np.pi * (env_int + 0.5) / max(self._num_envs, 1)
                radius = spawn_limit * 0.5
                pose = [radius * np.cos(angle), radius * np.sin(angle), angle + np.pi]
            self._pose[env_int] = np.asarray(pose, dtype=self._dtype)
            accepted.append(env_int)

    def _candidate_spawn_is_clear(
        self,
        candidate_xy: np.ndarray,
        other_ids: list[int],
        agent_radius: float,
        keepout: float,
    ) -> bool:
        if other_ids:
            other_xy = self._pose[np.asarray(other_ids, dtype=np.int32), :2]
            min_distance = float(np.min(np.linalg.norm(other_xy - candidate_xy[None, :], axis=1)))
            if min_distance < 2.0 * agent_radius + keepout:
                return False
        if self._scene_obstacle_count == 0:
            return True
        signed = self._static_signed_distances(candidate_xy[None, :], np.asarray([0.0]))
        return float(np.min(signed)) > agent_radius + keepout

    def _apply_physical_limits(self, actions: np.ndarray) -> np.ndarray:
        target = np.clip(actions, -self._velocity_limit, self._velocity_limit).astype(self._dtype)
        delta = np.clip(
            target - self._velocity,
            -self._accel_delta_limit,
            self._accel_delta_limit,
        )
        return np.clip(
            self._velocity + delta,
            -self._velocity_limit,
            self._velocity_limit,
        ).astype(self._dtype)

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
        count = env_indices.size
        grid_view = self._grid_buffer[:count]
        self._fill_occupancy_grid(env_indices, grid_view)
        self._append_grid_history(env_indices, grid_view)
        grid = self._grid_history[env_indices].reshape(count, -1)
        command_hist = self._command_history[env_indices].reshape(env_indices.size, -1)
        velocity_hist = self._velocity_history[env_indices].reshape(env_indices.size, -1)
        action_hist = self._action_history[env_indices].reshape(env_indices.size, -1)
        obs = self._obs_buffer[:count]
        critic = self._critic_buffer[:count]
        clearance = self._nearest_clearance[env_indices, None]
        collision = self._collision[env_indices, None].astype(self._dtype)
        col = 0
        obs[:, col : col + self._grid_stack_dim] = grid
        col += self._grid_stack_dim
        obs[:, col : col + 3] = self._commands[env_indices]
        col += 3
        obs[:, col : col + 3] = self._velocity[env_indices]
        col += 3
        obs[:, col : col + 3] = self._last_action[env_indices]
        col += 3
        obs[:, col : col + self._history_block_dim] = command_hist
        col += self._history_block_dim
        obs[:, col : col + self._history_block_dim] = velocity_hist
        col += self._history_block_dim
        obs[:, col : col + self._history_block_dim] = action_hist
        critic[:, : self._obs_dim] = obs
        critic_col = self._obs_dim
        critic[:, critic_col : critic_col + 1] = clearance
        critic_col += 1
        critic[:, critic_col : critic_col + 1] = collision
        critic_col += 1
        critic[:, critic_col : critic_col + 3] = self._pose[env_indices]
        return {"obs": obs.copy(), "critic": critic.copy()}

    def _occupancy_grid(self, env_indices: np.ndarray) -> np.ndarray:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        grid_size = self._cfg.grid.size
        grid = np.zeros((env_indices.size, grid_size, grid_size), dtype=self._dtype)
        self._fill_occupancy_grid(env_indices, grid)
        return grid.reshape(env_indices.size, -1)

    def _fill_occupancy_grid(self, env_indices: np.ndarray, grid: np.ndarray) -> None:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        grid.fill(0.0)
        if self._large_scene_enabled:
            self._fill_large_scene_occupancy_grid(env_indices, grid)
            return
        if env_indices.size == 0 or self._cfg.obstacles.count == 0:
            return
        if env_indices.size < 256:
            self._fill_occupancy_grid_scalar(env_indices, grid)
            return
        local_xy_batch = self._world_to_body_obstacles(env_indices)
        center_x = local_xy_batch[:, :, 0]
        center_y = local_xy_batch[:, :, 1]
        in_range = (np.abs(center_x) <= self._grid_extent) & (
            np.abs(center_y) <= self._grid_extent
        )
        obstacle_types = self._obstacle_type[env_indices]
        circle_mask = obstacle_types == self._OBSTACLE_CIRCLE
        radii = self._obstacle_radius[env_indices] + self._grid_pad
        half_extents = self._obstacle_half_extents[env_indices]
        rel_yaw = self._obstacle_yaw[env_indices] - self._pose[env_indices, 2][:, None]
        cos_yaw = np.cos(rel_yaw)
        sin_yaw = np.sin(rel_yaw)
        rect_extent_x = (
            np.abs(cos_yaw) * half_extents[:, :, 0]
            + np.abs(sin_yaw) * half_extents[:, :, 1]
            + self._grid_pad
        )
        rect_extent_y = (
            np.abs(sin_yaw) * half_extents[:, :, 0]
            + np.abs(cos_yaw) * half_extents[:, :, 1]
            + self._grid_pad
        )
        extent_x = np.where(circle_mask, radii, rect_extent_x)
        extent_y = np.where(circle_mask, radii, rect_extent_y)
        ix0, ix1, iy0, iy1 = self._grid_bounds_batch(center_x, center_y, extent_x, extent_y)

        for row in range(env_indices.size):
            obstacle_ids = np.flatnonzero(in_range[row])
            if obstacle_ids.size == 0:
                continue
            grid_view = grid[row]

            circle_ids = obstacle_ids[circle_mask[row, obstacle_ids]]
            for obstacle_id in circle_ids:
                center_x_i = float(center_x[row, obstacle_id])
                center_y_i = float(center_y[row, obstacle_id])
                radius = float(radii[row, obstacle_id])
                ix0_i = int(ix0[row, obstacle_id])
                ix1_i = int(ix1[row, obstacle_id])
                iy0_i = int(iy0[row, obstacle_id])
                iy1_i = int(iy1[row, obstacle_id])
                delta_x = self._grid_axis[ix0_i:ix1_i, None] - center_x_i
                delta_y = self._grid_axis[None, iy0_i:iy1_i] - center_y_i
                hits = delta_x * delta_x + delta_y * delta_y <= radius * radius
                np.maximum(
                    grid_view[ix0_i:ix1_i, iy0_i:iy1_i],
                    hits,
                    out=grid_view[ix0_i:ix1_i, iy0_i:iy1_i],
                )

            rect_ids = obstacle_ids[~circle_mask[row, obstacle_ids]]
            for obstacle_id in rect_ids:
                center_x_i = float(center_x[row, obstacle_id])
                center_y_i = float(center_y[row, obstacle_id])
                half_extent_x = float(half_extents[row, obstacle_id, 0])
                half_extent_y = float(half_extents[row, obstacle_id, 1])
                cos_yaw_i = float(cos_yaw[row, obstacle_id])
                sin_yaw_i = float(sin_yaw[row, obstacle_id])
                ix0_i = int(ix0[row, obstacle_id])
                ix1_i = int(ix1[row, obstacle_id])
                iy0_i = int(iy0[row, obstacle_id])
                iy1_i = int(iy1[row, obstacle_id])
                delta_x = self._grid_axis[ix0_i:ix1_i, None] - center_x_i
                delta_y = self._grid_axis[None, iy0_i:iy1_i] - center_y_i
                local_x = cos_yaw_i * delta_x + sin_yaw_i * delta_y
                local_y = -sin_yaw_i * delta_x + cos_yaw_i * delta_y
                hits = (np.abs(local_x) <= half_extent_x + self._grid_pad) & (
                    np.abs(local_y) <= half_extent_y + self._grid_pad
                )
                np.maximum(
                    grid_view[ix0_i:ix1_i, iy0_i:iy1_i],
                    hits,
                    out=grid_view[ix0_i:ix1_i, iy0_i:iy1_i],
                )

    def _fill_occupancy_grid_scalar(self, env_indices: np.ndarray, grid: np.ndarray) -> None:
        local_xy_batch = self._world_to_body_obstacles(env_indices)
        for row, env_id in enumerate(env_indices):
            local_xy = local_xy_batch[row]
            in_range = (np.abs(local_xy[:, 0]) <= self._grid_extent) & (
                np.abs(local_xy[:, 1]) <= self._grid_extent
            )
            obstacle_ids = np.flatnonzero(in_range)
            if obstacle_ids.size == 0:
                continue
            grid_view = grid[row]
            obstacle_types = self._obstacle_type[env_id, obstacle_ids]

            circle_ids = obstacle_ids[obstacle_types == self._OBSTACLE_CIRCLE]
            for obstacle_id in circle_ids:
                center_x = float(local_xy[obstacle_id, 0])
                center_y = float(local_xy[obstacle_id, 1])
                radius = float(self._obstacle_radius[env_id, obstacle_id] + self._grid_pad)
                ix0, ix1, iy0, iy1 = self._grid_bounds(center_x, center_y, radius, radius)
                delta_x = self._grid_axis[ix0:ix1, None] - center_x
                delta_y = self._grid_axis[None, iy0:iy1] - center_y
                hits = delta_x * delta_x + delta_y * delta_y <= radius * radius
                np.maximum(grid_view[ix0:ix1, iy0:iy1], hits, out=grid_view[ix0:ix1, iy0:iy1])

            rect_ids = obstacle_ids[obstacle_types != self._OBSTACLE_CIRCLE]
            for obstacle_id in rect_ids:
                center_x = float(local_xy[obstacle_id, 0])
                center_y = float(local_xy[obstacle_id, 1])
                half_extent_x = float(self._obstacle_half_extents[env_id, obstacle_id, 0])
                half_extent_y = float(self._obstacle_half_extents[env_id, obstacle_id, 1])
                rel_yaw = float(self._obstacle_yaw[env_id, obstacle_id] - self._pose[env_id, 2])
                cos_yaw = float(np.cos(rel_yaw))
                sin_yaw = float(np.sin(rel_yaw))
                aabb_x = (
                    abs(cos_yaw) * half_extent_x
                    + abs(sin_yaw) * half_extent_y
                    + self._grid_pad
                )
                aabb_y = (
                    abs(sin_yaw) * half_extent_x
                    + abs(cos_yaw) * half_extent_y
                    + self._grid_pad
                )
                ix0, ix1, iy0, iy1 = self._grid_bounds(center_x, center_y, aabb_x, aabb_y)
                delta_x = self._grid_axis[ix0:ix1, None] - center_x
                delta_y = self._grid_axis[None, iy0:iy1] - center_y
                local_x = cos_yaw * delta_x + sin_yaw * delta_y
                local_y = -sin_yaw * delta_x + cos_yaw * delta_y
                hits = (np.abs(local_x) <= half_extent_x + self._grid_pad) & (
                    np.abs(local_y) <= half_extent_y + self._grid_pad
                )
                np.maximum(grid_view[ix0:ix1, iy0:iy1], hits, out=grid_view[ix0:ix1, iy0:iy1])

    def _fill_large_scene_occupancy_grid(self, env_indices: np.ndarray, grid: np.ndarray) -> None:
        if env_indices.size == 0:
            return
        scene = self._cfg.large_scene
        agent_radius = float(scene.agent_collision_radius_m + self._grid_pad)
        for row, env_id in enumerate(env_indices):
            env_int = int(env_id)
            grid_view = grid[row]
            if self._scene_obstacle_count > 0:
                local_xy = self._world_to_body_points(env_int, self._scene_obstacle_xy)
                obstacle_ids = self._select_local_obstacles(
                    local_xy,
                    max_count=scene.max_local_static_obstacles,
                    extra_extent=max(
                        self._cfg.obstacles.radius_max_m,
                        self._cfg.obstacles.wall_length_max_m * 0.5,
                        scene.border_wall_thickness_m,
                    ),
                )
                if obstacle_ids.size > 0:
                    rel_yaw = self._scene_obstacle_yaw[obstacle_ids] - self._pose[env_int, 2]
                    self._rasterize_local_obstacles(
                        grid_view,
                        local_xy[obstacle_ids],
                        self._scene_obstacle_type[obstacle_ids],
                        self._scene_obstacle_radius[obstacle_ids],
                        self._scene_obstacle_half_extents[obstacle_ids],
                        rel_yaw,
                    )

            if self._num_envs <= 1:
                continue
            other_ids = self._all_env_indices[self._all_env_indices != env_int]
            local_agents = self._world_to_body_points(env_int, self._pose[other_ids, :2])
            agent_ids = self._select_local_obstacles(
                local_agents,
                max_count=scene.max_dynamic_agents,
                extra_extent=agent_radius,
            )
            for agent_id in agent_ids:
                center_x = float(local_agents[agent_id, 0])
                center_y = float(local_agents[agent_id, 1])
                ix0, ix1, iy0, iy1 = self._grid_bounds(
                    center_x, center_y, agent_radius, agent_radius
                )
                delta_x = self._grid_axis[ix0:ix1, None] - center_x
                delta_y = self._grid_axis[None, iy0:iy1] - center_y
                hits = delta_x * delta_x + delta_y * delta_y <= agent_radius * agent_radius
                np.maximum(grid_view[ix0:ix1, iy0:iy1], hits, out=grid_view[ix0:ix1, iy0:iy1])

    def _select_local_obstacles(
        self, local_xy: np.ndarray, *, max_count: int, extra_extent: float
    ) -> np.ndarray:
        if local_xy.size == 0:
            return np.zeros((0,), dtype=np.int32)
        extent = self._grid_extent + float(extra_extent)
        in_range = (np.abs(local_xy[:, 0]) <= extent) & (np.abs(local_xy[:, 1]) <= extent)
        obstacle_ids = np.flatnonzero(in_range).astype(np.int32)
        if obstacle_ids.size <= max_count:
            return obstacle_ids
        distances = np.sum(local_xy[obstacle_ids] * local_xy[obstacle_ids], axis=1)
        keep = np.argpartition(distances, max_count - 1)[:max_count]
        return obstacle_ids[keep]

    def _rasterize_local_obstacles(
        self,
        grid_view: np.ndarray,
        local_xy: np.ndarray,
        obstacle_types: np.ndarray,
        radii: np.ndarray,
        half_extents: np.ndarray,
        rel_yaw: np.ndarray,
    ) -> None:
        for obstacle_id, center in enumerate(local_xy):
            center_x = float(center[0])
            center_y = float(center[1])
            obstacle_type = int(obstacle_types[obstacle_id])
            if obstacle_type == self._OBSTACLE_CIRCLE:
                radius = float(radii[obstacle_id] + self._grid_pad)
                ix0, ix1, iy0, iy1 = self._grid_bounds(center_x, center_y, radius, radius)
                delta_x = self._grid_axis[ix0:ix1, None] - center_x
                delta_y = self._grid_axis[None, iy0:iy1] - center_y
                hits = delta_x * delta_x + delta_y * delta_y <= radius * radius
            else:
                half_extent_x = float(half_extents[obstacle_id, 0])
                half_extent_y = float(half_extents[obstacle_id, 1])
                cos_yaw = float(np.cos(rel_yaw[obstacle_id]))
                sin_yaw = float(np.sin(rel_yaw[obstacle_id]))
                aabb_x = (
                    abs(cos_yaw) * half_extent_x
                    + abs(sin_yaw) * half_extent_y
                    + self._grid_pad
                )
                aabb_y = (
                    abs(sin_yaw) * half_extent_x
                    + abs(cos_yaw) * half_extent_y
                    + self._grid_pad
                )
                ix0, ix1, iy0, iy1 = self._grid_bounds(center_x, center_y, aabb_x, aabb_y)
                delta_x = self._grid_axis[ix0:ix1, None] - center_x
                delta_y = self._grid_axis[None, iy0:iy1] - center_y
                rect_x = cos_yaw * delta_x + sin_yaw * delta_y
                rect_y = -sin_yaw * delta_x + cos_yaw * delta_y
                hits = (np.abs(rect_x) <= half_extent_x + self._grid_pad) & (
                    np.abs(rect_y) <= half_extent_y + self._grid_pad
                )
            np.maximum(grid_view[ix0:ix1, iy0:iy1], hits, out=grid_view[ix0:ix1, iy0:iy1])

    def _world_to_body_points(self, env_id: int, points_world: np.ndarray) -> np.ndarray:
        delta = points_world - self._pose[env_id, :2]
        yaw = -float(self._pose[env_id, 2])
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = c * delta[:, 0] - s * delta[:, 1]
        y = s * delta[:, 0] + c * delta[:, 1]
        return np.stack([x, y], axis=1).astype(self._dtype)

    def _world_to_body_obstacles(self, env_indices: np.ndarray) -> np.ndarray:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        pose = self._pose[env_indices]
        delta = self._obstacle_xy[env_indices] - pose[:, None, :2]
        cos_yaw = np.cos(pose[:, 2])[:, None]
        sin_yaw = np.sin(pose[:, 2])[:, None]
        x = cos_yaw * delta[:, :, 0] + sin_yaw * delta[:, :, 1]
        y = -sin_yaw * delta[:, :, 0] + cos_yaw * delta[:, :, 1]
        return np.stack([x, y], axis=2).astype(self._dtype, copy=False)

    def _static_signed_distances(self, points_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        points_world = np.asarray(points_world, dtype=self._dtype)
        yaw = np.asarray(yaw, dtype=self._dtype)
        if self._scene_obstacle_count == 0:
            return np.full((points_world.shape[0], 1), self._grid_extent, dtype=self._dtype)
        delta = self._scene_obstacle_xy[None, :, :] - points_world[:, None, :]
        local_xy = np.empty_like(delta)
        cos_yaw_point = np.cos(yaw)[:, None]
        sin_yaw_point = np.sin(yaw)[:, None]
        local_xy[:, :, 0] = cos_yaw_point * delta[:, :, 0] + sin_yaw_point * delta[:, :, 1]
        local_xy[:, :, 1] = -sin_yaw_point * delta[:, :, 0] + cos_yaw_point * delta[:, :, 1]
        signed = np.empty((points_world.shape[0], self._scene_obstacle_count), dtype=self._dtype)
        circle_mask = self._scene_obstacle_type == self._OBSTACLE_CIRCLE
        if np.any(circle_mask):
            circle_dist = np.linalg.norm(local_xy[:, circle_mask], axis=2)
            signed[:, circle_mask] = circle_dist - self._scene_obstacle_radius[circle_mask]

        rect_mask = ~circle_mask
        if np.any(rect_mask):
            rel_yaw = self._scene_obstacle_yaw[rect_mask][None, :] - yaw[:, None]
            cos_yaw = np.cos(rel_yaw)
            sin_yaw = np.sin(rel_yaw)
            rect_xy = local_xy[:, rect_mask]
            rect_x = cos_yaw * rect_xy[:, :, 0] + sin_yaw * rect_xy[:, :, 1]
            rect_y = -sin_yaw * rect_xy[:, :, 0] + cos_yaw * rect_xy[:, :, 1]
            half_extents = self._scene_obstacle_half_extents[rect_mask][None, :, :]
            qx = np.abs(rect_x) - half_extents[:, :, 0]
            qy = np.abs(rect_y) - half_extents[:, :, 1]
            outside_x = np.maximum(qx, 0.0)
            outside_y = np.maximum(qy, 0.0)
            outside_distance = np.sqrt(outside_x * outside_x + outside_y * outside_y)
            inside_distance = np.minimum(np.maximum(qx, qy), 0.0)
            signed[:, rect_mask] = outside_distance + inside_distance
        return signed.astype(self._dtype, copy=False)

    def _grid_bounds(
        self, center_x: float, center_y: float, extent_x: float, extent_y: float
    ) -> tuple[int, int, int, int]:
        ix0 = max(
            int(math.floor((center_x - extent_x) / self._grid_cell_size + self._grid_half_index))
            - 1,
            0,
        )
        ix1 = min(
            int(math.ceil((center_x + extent_x) / self._grid_cell_size + self._grid_half_index))
            + 2,
            self._grid_size,
        )
        iy0 = max(
            int(math.floor((center_y - extent_y) / self._grid_cell_size + self._grid_half_index))
            - 1,
            0,
        )
        iy1 = min(
            int(math.ceil((center_y + extent_y) / self._grid_cell_size + self._grid_half_index))
            + 2,
            self._grid_size,
        )
        return ix0, ix1, iy0, iy1

    def _grid_bounds_batch(
        self,
        center_x: np.ndarray,
        center_y: np.ndarray,
        extent_x: np.ndarray,
        extent_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ix0 = (
            np.floor((center_x - extent_x) / self._grid_cell_size + self._grid_half_index).astype(
                np.int32
            )
            - 1
        )
        ix1 = (
            np.ceil((center_x + extent_x) / self._grid_cell_size + self._grid_half_index).astype(
                np.int32
            )
            + 2
        )
        iy0 = (
            np.floor((center_y - extent_y) / self._grid_cell_size + self._grid_half_index).astype(
                np.int32
            )
            - 1
        )
        iy1 = (
            np.ceil((center_y + extent_y) / self._grid_cell_size + self._grid_half_index).astype(
                np.int32
            )
            + 2
        )
        return (
            np.clip(ix0, 0, self._grid_size),
            np.clip(ix1, 0, self._grid_size),
            np.clip(iy0, 0, self._grid_size),
            np.clip(iy1, 0, self._grid_size),
        )

    def _compute_clearance(self, env_indices: np.ndarray) -> np.ndarray:
        env_indices = np.asarray(env_indices, dtype=np.int32)
        clearance = np.full((env_indices.size,), self._grid_extent, dtype=self._dtype)
        half_diag = 0.5 * float(np.hypot(self._cfg.body.length_m, self._cfg.body.width_m))
        safety_radius = half_diag + self._cfg.grid.safety_margin_m
        if self._large_scene_enabled:
            if env_indices.size == 0:
                return clearance
            scene = self._cfg.large_scene
            static_signed = self._static_signed_distances(
                self._pose[env_indices, :2], self._pose[env_indices, 2]
            )
            static_clearance = np.min(static_signed - safety_radius, axis=1)
            if self._num_envs > 1:
                delta = self._pose[env_indices, None, :2] - self._pose[None, :, :2]
                distances = np.linalg.norm(delta, axis=2)
                self_mask = env_indices[:, None] == self._all_env_indices[None, :]
                distances[self_mask] = np.inf
                agent_clearance = np.min(
                    distances - 2.0 * float(scene.agent_collision_radius_m), axis=1
                )
            else:
                agent_clearance = np.full((env_indices.size,), np.inf, dtype=self._dtype)
            half_world = 0.5 * float(scene.world_size_m)
            border_clearance = (
                half_world
                - np.max(np.abs(self._pose[env_indices, :2]), axis=1)
                - float(scene.agent_collision_radius_m)
            )
            clearance[:] = np.minimum(
                np.minimum(static_clearance, agent_clearance), border_clearance
            ).astype(self._dtype, copy=False)
            self._static_collision[env_indices] = static_clearance <= 0.0
            self._agent_collision[env_indices] = agent_clearance <= 0.0
            self._border_collision[env_indices] = border_clearance <= 0.0
            return clearance
        self._static_collision[env_indices] = False
        self._agent_collision[env_indices] = False
        self._border_collision[env_indices] = False
        if self._cfg.obstacles.count == 0:
            return clearance
        local_xy = self._world_to_body_obstacles(env_indices)
        obstacle_types = self._obstacle_type[env_indices]
        signed = np.empty((env_indices.size, self._cfg.obstacles.count), dtype=self._dtype)

        circle_mask = obstacle_types == self._OBSTACLE_CIRCLE
        if np.any(circle_mask):
            circle_signed = np.linalg.norm(local_xy, axis=2) - self._obstacle_radius[env_indices]
            signed[circle_mask] = circle_signed[circle_mask]

        rect_mask = ~circle_mask
        if np.any(rect_mask):
            rel_yaw = self._obstacle_yaw[env_indices] - self._pose[env_indices, 2][:, None]
            cos_yaw = np.cos(rel_yaw)
            sin_yaw = np.sin(rel_yaw)
            local_x = cos_yaw * local_xy[:, :, 0] + sin_yaw * local_xy[:, :, 1]
            local_y = -sin_yaw * local_xy[:, :, 0] + cos_yaw * local_xy[:, :, 1]
            half_extents = self._obstacle_half_extents[env_indices]
            qx = np.abs(local_x) - half_extents[:, :, 0]
            qy = np.abs(local_y) - half_extents[:, :, 1]
            outside_x = np.maximum(qx, 0.0)
            outside_y = np.maximum(qy, 0.0)
            outside_distance = np.sqrt(outside_x * outside_x + outside_y * outside_y)
            inside_distance = np.minimum(np.maximum(qx, qy), 0.0)
            rect_signed = outside_distance + inside_distance
            signed[rect_mask] = rect_signed[rect_mask]

        clearance[:] = np.min(signed - safety_radius, axis=1).astype(self._dtype, copy=False)
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
        high = self._velocity_limit
        accel_delta = self._accel_delta_limit
        prev_error = np.linalg.norm((cmd - self._last_action) / high, axis=1)
        new_error = np.linalg.norm((cmd - action) / high, axis=1)
        safety_gate = np.clip((self._nearest_clearance - 0.05) / 0.45, 0.0, 1.0)
        response_progress = safety_gate * np.maximum(prev_error - new_error, 0.0)
        action_delta = action - self._last_action
        action_jerk = action_delta - self._last_action_delta
        track_cost = ((action - cmd) / high) ** 2
        diff_cost = (action_delta / accel_delta) ** 2
        jerk_cost = (action_jerk / accel_delta) ** 2
        track_weights = np.asarray(
            [cfg.vx_track, cfg.vy_track, cfg.vyaw_track], dtype=self._dtype
        )
        diff_weights = np.asarray(
            [cfg.vx_diff, cfg.vy_diff, cfg.vyaw_diff], dtype=self._dtype
        )
        jerk_weights = np.asarray(
            [cfg.vx_jerk, cfg.vy_jerk, cfg.vyaw_jerk], dtype=self._dtype
        )
        clearance_penalty = np.exp(-np.maximum(self._nearest_clearance, 0.0) / 0.35)
        self._tracking_error = new_error.astype(self._dtype)
        self._response_progress = response_progress.astype(self._dtype)
        self._track_cost = track_cost.astype(self._dtype)
        self._diff_cost = diff_cost.astype(self._dtype)
        self._jerk_cost = jerk_cost.astype(self._dtype)
        reward = (
            cfg.intent * intent_reward
            + cfg.intent_projection * projection
            + cfg.yaw_intent * yaw_reward
            + cfg.response * response_progress
            - safety_gate * np.sum(track_cost * track_weights, axis=1)
            - np.sum(diff_cost * diff_weights, axis=1)
            - np.sum(jerk_cost * jerk_weights, axis=1)
            - cfg.clearance * clearance_penalty
            + cfg.collision * self._collision.astype(self._dtype)
        )
        return reward.astype(self._dtype)

    def _update_reward_progress(self, reward: np.ndarray) -> None:
        if not self._large_scene_enabled:
            self._stagnated.fill(False)
            return
        scene = self._cfg.large_scene
        self._episode_return += reward.astype(self._dtype, copy=False)
        improved = self._episode_return > (
            self._best_episode_return + float(scene.stagnation_min_return_delta)
        )
        self._best_episode_return[improved] = self._episode_return[improved]
        self._steps_since_reward_improvement[improved] = 0
        self._steps_since_reward_improvement[~improved] += 1
        warm = self._state.info["steps"] >= int(scene.stagnation_warmup_steps)
        stale = self._steps_since_reward_improvement >= int(scene.stagnation_window_steps)
        self._stagnated = (warm & stale).astype(bool)

    def _make_velocity_limit(self) -> np.ndarray:
        limits = self._cfg.physical_limits
        return np.asarray(
            [limits.max_x_speed, limits.max_y_speed, limits.max_yaw_rate], dtype=self._dtype
        )

    def _make_accel_limit(self) -> np.ndarray:
        limits = self._cfg.physical_limits
        return np.asarray(
            [limits.max_x_accel, limits.max_y_accel, limits.max_yaw_accel], dtype=self._dtype
        )

    @staticmethod
    def _rotate_points(points: np.ndarray, yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = c * points[:, 0] - s * points[:, 1]
        y = s * points[:, 0] + c * points[:, 1]
        return np.stack([x, y], axis=1).astype(get_global_dtype())

    @staticmethod
    def _rectangle_signed_distance(point: np.ndarray, half_extents: np.ndarray) -> np.float32:
        q = np.abs(point) - half_extents
        outside = np.maximum(q, 0.0)
        outside_distance = np.linalg.norm(outside)
        inside_distance = min(max(float(q[0]), float(q[1])), 0.0)
        return np.asarray(outside_distance + inside_distance, dtype=get_global_dtype())

    def _info(self, env_indices: np.ndarray) -> dict[str, Any]:
        return {
            "commands": self._commands[env_indices].copy(),
            "nearest_clearance": self._nearest_clearance[env_indices].copy(),
            "collision": self._collision[env_indices].copy(),
            "static_collision": self._static_collision[env_indices].copy(),
            "agent_collision": self._agent_collision[env_indices].copy(),
            "border_collision": self._border_collision[env_indices].copy(),
            "stagnated": self._stagnated[env_indices].copy(),
            "human_command_enabled": self._human_command_enabled,
            "human_command_env_id": self._human_command_env_id,
            "human_command_env_ids": self._human_command_env_ids.copy(),
            "human_command": self._human_command.copy(),
            "human_command_connected": self._human_command_connected,
            "human_controller_name": self._human_controller_name,
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
        if self._large_scene_enabled:
            self._gl_draw_other_cars(GL)
        self._gl_draw_car(GL, pose)
        self._gl_draw_arrows(GL, pose)

    def _gl_draw_floor(self, GL: Any) -> None:
        if self._large_scene_enabled:
            extent = int(math.ceil(self._cfg.large_scene.world_size_m * 0.5))
        else:
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
        if self._large_scene_enabled:
            obstacle_xy = self._scene_obstacle_xy
            obstacle_type_array = self._scene_obstacle_type
            obstacle_radius = self._scene_obstacle_radius
            obstacle_half_extents = self._scene_obstacle_half_extents
            obstacle_yaw = self._scene_obstacle_yaw
        else:
            obstacle_xy = self._obstacle_xy[0]
            obstacle_type_array = self._obstacle_type[0]
            obstacle_radius = self._obstacle_radius[0]
            obstacle_half_extents = self._obstacle_half_extents[0]
            obstacle_yaw = self._obstacle_yaw[0]
        for obstacle_id, center in enumerate(obstacle_xy):
            obstacle_type = int(obstacle_type_array[obstacle_id])
            if obstacle_type == self._OBSTACLE_CIRCLE:
                GL.glColor4f(0.95, 0.24, 0.12, 0.95)
                self._gl_cylinder(
                    GL,
                    x=float(center[0]),
                    y=float(center[1]),
                    radius=float(obstacle_radius[obstacle_id]),
                    height=0.32,
                )
            else:
                half_extents = obstacle_half_extents[obstacle_id]
                if obstacle_type == self._OBSTACLE_WALL:
                    GL.glColor4f(0.68, 0.20, 0.95, 0.92)
                    half_z = 0.22
                else:
                    GL.glColor4f(0.95, 0.48, 0.12, 0.92)
                    half_z = 0.16
                GL.glPushMatrix()
                GL.glTranslatef(float(center[0]), float(center[1]), half_z)
                GL.glRotatef(float(np.degrees(obstacle_yaw[obstacle_id])), 0.0, 0.0, 1.0)
                self._gl_box(
                    GL,
                    half_x=float(half_extents[0]),
                    half_y=float(half_extents[1]),
                    half_z=half_z,
                )
                GL.glPopMatrix()

    def _gl_draw_other_cars(self, GL: Any) -> None:
        for env_id in range(1, self._num_envs):
            pose = self._pose[env_id]
            GL.glPushMatrix()
            GL.glTranslatef(float(pose[0]), float(pose[1]), 0.08)
            GL.glRotatef(float(np.degrees(pose[2])), 0.0, 0.0, 1.0)
            if self._collision[env_id]:
                GL.glColor4f(1.0, 0.15, 0.08, 0.92)
            else:
                GL.glColor4f(0.25, 0.78, 0.65, 0.72)
            self._gl_box(
                GL,
                half_x=self._cfg.body.length_m * 0.5,
                half_y=self._cfg.body.width_m * 0.5,
                half_z=0.08,
            )
            GL.glPopMatrix()

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
