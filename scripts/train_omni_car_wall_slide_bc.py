#!/usr/bin/env python3
"""Supervised behavior correction for OmniCar PPO actor checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.evaluate_omni_car_checkpoint as checkpoint_eval
import scripts.train_rsl_rl as train_rsl_rl
from scripts.evaluate_omni_car_behaviors import BehaviorScenario, _apply_scenario
from unilab.training.experiment import patch_rsl_rl_resume_state

MAX_X_SPEED = 2.0
MAX_Y_SPEED = 1.0
WALL_SLIDE_PARALLEL_SPEED = 0.65
WALL_SLIDE_LATERAL_AWAY_SPEED = 0.25
WALL_SLIDE_WEIGHT = 12.0


@dataclass(frozen=True)
class OracleScenario:
    name: str
    behavior: BehaviorScenario
    target_action: tuple[float, float, float]
    weight: float = 1.0


@dataclass(frozen=True)
class TeacherScoreConfig:
    profile: str = "default"
    front_collision_penalty: float = 220.0
    front_projection_penalty: float = 35.0
    front_speed_penalty: float = 10.0
    front_closing_penalty: float = 5.0
    front_opening_reward: float = 1.5
    front_clearance_floor: float = 0.05
    front_clearance_floor_penalty: float = 0.0
    front_lateral_penalty: float = 0.0
    wall_collision_penalty: float = 260.0
    wall_closing_penalty: float = 14.0
    wall_projection_reward: float = 3.0
    wall_opening_reward: float = 2.0
    wall_lateral_penalty: float = 0.7
    wall_speed_penalty: float = 0.10
    wall_clearance_floor: float = 0.05
    wall_clearance_floor_penalty: float = 0.0
    wall_gate_projection_on_closing: bool = False


@dataclass
class DaggerReplayBuffer:
    max_samples: int
    obs: Any | None = None
    target: torch.Tensor | None = None
    weight: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return 0 if self.obs is None else int(self.obs.shape[0])

    def append(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: float,
        rng: np.random.Generator,
        samples_per_step: int,
    ) -> None:
        if self.max_samples <= 0 or samples_per_step <= 0:
            return
        count = min(int(samples_per_step), int(obs.shape[0]))
        if count <= 0:
            return
        indices = rng.choice(int(obs.shape[0]), size=count, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=obs.device)
        obs_cpu = _index_batch(obs, index_tensor).detach().cpu()
        target_cpu = target.index_select(0, index_tensor).detach().cpu()
        weight_cpu = torch.full((count,), float(weight), dtype=torch.float32)
        if self.obs is None:
            self.obs = obs_cpu
            self.target = target_cpu
            self.weight = weight_cpu
        else:
            assert self.target is not None and self.weight is not None
            self.obs = torch.cat((self.obs, obs_cpu), dim=0)
            self.target = torch.cat((self.target, target_cpu), dim=0)
            self.weight = torch.cat((self.weight, weight_cpu), dim=0)
        if self.size > self.max_samples:
            start = self.size - self.max_samples
            assert self.target is not None and self.weight is not None
            self.obs = self.obs[start:]
            self.target = self.target[start:]
            self.weight = self.weight[start:]

    def sample(
        self,
        *,
        rng: np.random.Generator,
        batch_size: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.obs is None or self.target is None or self.weight is None:
            raise ValueError("Cannot sample an empty DAgger replay buffer")
        count = min(int(batch_size), self.size)
        indices = rng.choice(self.size, size=count, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return (
            _index_batch(self.obs, index_tensor).to(device),
            self.target.index_select(0, index_tensor).to(device),
            self.weight.index_select(0, index_tensor).to(device),
        )


def _index_batch(batch: Any, index: torch.Tensor) -> Any:
    if hasattr(batch, "index_select"):
        return batch.index_select(0, index)
    return batch[index]


def _unit(angle: float) -> np.ndarray:
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)


def _lateral(direction: np.ndarray) -> np.ndarray:
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)


def _directional_command(angle: float, fraction: float) -> tuple[float, float, float]:
    direction = _unit(angle)
    axis_scale = min(
        MAX_X_SPEED / max(abs(float(direction[0])), 1e-6),
        MAX_Y_SPEED / max(abs(float(direction[1])), 1e-6),
    )
    xy = direction * axis_scale * float(fraction)
    return (float(xy[0]), float(xy[1]), 0.0)


def _point(direction: np.ndarray, forward: float, lateral: float) -> tuple[float, float]:
    xy = direction * float(forward) + _lateral(direction) * float(lateral)
    return (float(xy[0]), float(xy[1]))


def _yaw_from_direction(direction: np.ndarray, lateral_axis: bool = False) -> float:
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    if lateral_axis:
        yaw += math.pi * 0.5
    return float(yaw)


def _clip_target_xy(xy: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(
        np.asarray(xy, dtype=np.float64),
        np.asarray([-MAX_X_SPEED, -MAX_Y_SPEED], dtype=np.float64),
        np.asarray([MAX_X_SPEED, MAX_Y_SPEED], dtype=np.float64),
    )
    return (float(clipped[0]), float(clipped[1]), 0.0)


def _jitter_behavior(
    behavior: BehaviorScenario,
    *,
    rng: np.random.Generator,
    xy_std: float,
    radius_std: float,
    yaw_std: float,
) -> BehaviorScenario:
    if not behavior.obstacle_xy or (xy_std <= 0.0 and radius_std <= 0.0 and yaw_std <= 0.0):
        return behavior
    xy = np.asarray(behavior.obstacle_xy, dtype=np.float64)
    if xy_std > 0.0:
        xy = xy + rng.normal(0.0, float(xy_std), size=xy.shape)
    radius = tuple(
        max(0.04, float(value) + float(rng.normal(0.0, radius_std)))
        for value in behavior.obstacle_radius
    )
    half_extents = tuple(
        (
            max(0.04, float(extent[0]) + float(rng.normal(0.0, radius_std))),
            max(0.04, float(extent[1]) + float(rng.normal(0.0, radius_std))),
        )
        for extent in behavior.obstacle_half_extents
    )
    yaw = tuple(float(value) + float(rng.normal(0.0, yaw_std)) for value in behavior.obstacle_yaw)
    return replace(
        behavior,
        obstacle_xy=tuple((float(item[0]), float(item[1])) for item in xy),
        obstacle_radius=radius,
        obstacle_half_extents=half_extents,
        obstacle_yaw=yaw,
    )


def _training_behavior(
    scenario: OracleScenario,
    *,
    rng: np.random.Generator,
    xy_std: float,
    radius_std: float,
    yaw_std: float,
) -> BehaviorScenario:
    return _jitter_behavior(
        scenario.behavior,
        rng=rng,
        xy_std=xy_std,
        radius_std=radius_std,
        yaw_std=yaw_std,
    )


BASE_ORACLE_SCENARIOS: tuple[OracleScenario, ...] = (
    OracleScenario(
        name="zero_input_hold",
        behavior=BehaviorScenario(name="zero_input_hold", command=(0.0, 0.0, 0.0)),
        target_action=(0.0, 0.0, 0.0),
        weight=8.0,
    ),
    OracleScenario(
        name="clear_forward_follow",
        behavior=BehaviorScenario(name="clear_forward_follow", command=(1.0, 0.0, 0.0)),
        target_action=(1.0, 0.0, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_diagonal_follow",
        behavior=BehaviorScenario(name="clear_diagonal_follow", command=(0.8, 0.4, 0.0)),
        target_action=(0.8, 0.4, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_left_follow",
        behavior=BehaviorScenario(name="clear_left_follow", command=(0.0, 0.7, 0.0)),
        target_action=(0.0, 0.7, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_right_follow",
        behavior=BehaviorScenario(name="clear_right_follow", command=(0.0, -0.7, 0.0)),
        target_action=(0.0, -0.7, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_backward_follow",
        behavior=BehaviorScenario(name="clear_backward_follow", command=(-0.8, 0.0, 0.0)),
        target_action=(-0.8, 0.0, 0.0),
        weight=2.5,
    ),
    OracleScenario(
        name="clear_slow_diagonal_follow",
        behavior=BehaviorScenario(
            name="clear_slow_diagonal_follow",
            command=(0.35, -0.25, 0.0),
        ),
        target_action=(0.35, -0.25, 0.0),
        weight=2.0,
    ),
    OracleScenario(
        name="yaw_only_follow",
        behavior=BehaviorScenario(name="yaw_only_follow", command=(0.0, 0.0, 1.0)),
        target_action=(0.0, 0.0, 1.0),
        weight=5.0,
    ),
    OracleScenario(
        name="yaw_negative_follow",
        behavior=BehaviorScenario(name="yaw_negative_follow", command=(0.0, 0.0, -1.0)),
        target_action=(0.0, 0.0, -1.0),
        weight=5.0,
    ),
    OracleScenario(
        name="front_blocked_stop",
        behavior=BehaviorScenario(
            name="front_blocked_stop",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.70, 0.0),),
            obstacle_radius=(0.22,),
        ),
        target_action=(0.0, 0.0, 0.0),
        weight=12.0,
    ),
    OracleScenario(
        name="right_wall_slide",
        behavior=BehaviorScenario(
            name="right_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, -0.34), (1.05, -0.34), (1.50, -0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(WALL_SLIDE_PARALLEL_SPEED, WALL_SLIDE_LATERAL_AWAY_SPEED, 0.0),
        weight=WALL_SLIDE_WEIGHT,
    ),
    OracleScenario(
        name="left_wall_slide",
        behavior=BehaviorScenario(
            name="left_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, 0.34), (1.05, 0.34), (1.50, 0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(WALL_SLIDE_PARALLEL_SPEED, -WALL_SLIDE_LATERAL_AWAY_SPEED, 0.0),
        weight=WALL_SLIDE_WEIGHT,
    ),
)


def _make_directional_oracle_scenarios(
    directions: int = 16,
    *,
    name_prefix: str = "directional",
) -> tuple[OracleScenario, ...]:
    scenarios: list[OracleScenario] = []
    angles = [2.0 * math.pi * i / int(directions) for i in range(int(directions))]
    for angle_index, angle in enumerate(angles):
        direction = _unit(angle)
        for speed_name, fraction, weight in (
            ("slow", 0.30, 1.6),
            ("mid", 0.60, 2.0),
            ("fast", 0.85, 2.2),
        ):
            command = _directional_command(angle, fraction)
            scenarios.append(
                OracleScenario(
                    name=f"{name_prefix}_clear_{angle_index:02d}_{speed_name}",
                    behavior=BehaviorScenario(
                        name=f"{name_prefix}_clear_{angle_index:02d}_{speed_name}",
                        command=command,
                    ),
                    target_action=command,
                    weight=weight,
                )
            )
        shape = ("circle", "box", "wall")[angle_index % 3]
        scenarios.append(
            OracleScenario(
                name=f"{name_prefix}_front_stop_{angle_index:02d}_{shape}",
                behavior=BehaviorScenario(
                    name=f"{name_prefix}_front_stop_{angle_index:02d}_{shape}",
                    command=_directional_command(angle, 0.65),
                    obstacle_xy=(_point(direction, 0.70, 0.0),),
                    obstacle_radius=(0.22,),
                    obstacle_type=(shape,),
                    obstacle_half_extents=((0.22, 0.18),),
                    obstacle_yaw=(_yaw_from_direction(direction, lateral_axis=True),),
                ),
                target_action=(0.0, 0.0, 0.0),
                weight=10.0,
            )
        )
        if angle_index % 2 == 0:
            for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
                lateral_offset = 0.34 * side_sign
                away = -side_sign
                target_xy = direction * WALL_SLIDE_PARALLEL_SPEED + _lateral(direction) * (
                    WALL_SLIDE_LATERAL_AWAY_SPEED * away
                )
                scenarios.append(
                    OracleScenario(
                        name=f"{name_prefix}_{side_name}_wall_{angle_index:02d}",
                        behavior=BehaviorScenario(
                            name=f"{name_prefix}_{side_name}_wall_{angle_index:02d}",
                            command=_directional_command(angle, 0.65),
                            obstacle_xy=tuple(
                                _point(direction, forward, lateral_offset)
                                for forward in (0.60, 1.05, 1.50)
                            ),
                            obstacle_radius=(0.22, 0.22, 0.22),
                            obstacle_type=("circle", "wall", "circle"),
                            obstacle_half_extents=((0.22, 0.22), (0.36, 0.08), (0.22, 0.22)),
                            obstacle_yaw=(0.0, _yaw_from_direction(direction), 0.0),
                        ),
                        target_action=_clip_target_xy(target_xy),
                        weight=WALL_SLIDE_WEIGHT,
                    )
                )
    return tuple(scenarios)


DIRECTIONAL_ORACLE_SCENARIOS = _make_directional_oracle_scenarios()
DIRECTIONAL32_ORACLE_SCENARIOS = _make_directional_oracle_scenarios(
    32,
    name_prefix="directional32",
)


def _make_max_stick_oracle_scenarios(directions: int = 8) -> tuple[OracleScenario, ...]:
    scenarios: list[OracleScenario] = []
    angles = [2.0 * math.pi * i / int(directions) for i in range(int(directions))]
    for angle_index, angle in enumerate(angles):
        direction = _unit(angle)
        command = _directional_command(angle, 1.0)
        scenarios.append(
            OracleScenario(
                name=f"max_stick_clear_dir_{angle_index:02d}",
                behavior=BehaviorScenario(
                    name=f"max_stick_clear_dir_{angle_index:02d}",
                    command=command,
                ),
                target_action=command,
                weight=3.0,
            )
        )

        shape = ("circle", "box", "wall")[angle_index % 3]
        scenarios.append(
            OracleScenario(
                name=f"max_stick_front_blocked_dir_{angle_index:02d}_{shape}",
                behavior=BehaviorScenario(
                    name=f"max_stick_front_blocked_dir_{angle_index:02d}_{shape}",
                    command=command,
                    obstacle_xy=(_point(direction, 0.72, 0.0),),
                    obstacle_radius=(0.24,),
                    obstacle_type=(shape,),
                    obstacle_half_extents=((0.24, 0.20),),
                    obstacle_yaw=(_yaw_from_direction(direction, lateral_axis=True),),
                ),
                target_action=(0.0, 0.0, 0.0),
                weight=18.0,
            )
        )

        for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
            lateral_offset = 0.34 * side_sign
            away = -side_sign
            target_xy = direction * WALL_SLIDE_PARALLEL_SPEED + _lateral(direction) * (
                WALL_SLIDE_LATERAL_AWAY_SPEED * away
            )
            scenarios.append(
                OracleScenario(
                    name=f"max_stick_{side_name}_wall_dir_{angle_index:02d}",
                    behavior=BehaviorScenario(
                        name=f"max_stick_{side_name}_wall_dir_{angle_index:02d}",
                        command=command,
                        obstacle_xy=tuple(
                            _point(direction, forward, lateral_offset)
                            for forward in (0.55, 0.95, 1.35)
                        ),
                        obstacle_radius=(0.22, 0.22, 0.22),
                        obstacle_type=("circle", "wall", "circle"),
                        obstacle_half_extents=((0.22, 0.22), (0.36, 0.08), (0.22, 0.22)),
                        obstacle_yaw=(0.0, _yaw_from_direction(direction), 0.0),
                    ),
                    target_action=_clip_target_xy(target_xy),
                    weight=WALL_SLIDE_WEIGHT,
                )
            )
    return tuple(scenarios)


MAX_STICK_ORACLE_SCENARIOS = _make_max_stick_oracle_scenarios()
ORACLE_SCENARIOS: tuple[OracleScenario, ...] = (
    BASE_ORACLE_SCENARIOS
    + DIRECTIONAL_ORACLE_SCENARIOS
    + DIRECTIONAL32_ORACLE_SCENARIOS
    + MAX_STICK_ORACLE_SCENARIOS
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Checkpoint path to correct.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument(
        "--rollout-actions",
        choices=("policy", "target"),
        default="policy",
        help=(
            "Action source used to advance the supervised rollout. "
            "The default trains on the policy's own closed-loop state distribution."
        ),
    )
    parser.add_argument(
        "--balanced-batch",
        action="store_true",
        help=(
            "Accumulate one optimizer update across every oracle scenario each iteration. "
            "Useful for diagnostics, but per-scenario SGD is usually better for the "
            "clear/front/wall oracle mix because full-batch gradients can cancel."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--actor-action-head-mode",
        choices=("single", "gated_two_head"),
        default=None,
        help="Optional actor head override, e.g. gated_two_head for branch separation tests.",
    )
    parser.add_argument(
        "--actor-branch-hidden-dims",
        default=None,
        help="Comma-separated branch hidden dimensions for gated actor checkpoints.",
    )
    parser.add_argument("--actor-action-gate-init-bias", type=float, default=None)
    parser.add_argument(
        "--partial-actor-load",
        action="store_true",
        help=(
            "Load only matching actor parameters with strict=False. Use when migrating "
            "a single-head checkpoint into a branched actor architecture."
        ),
    )
    parser.add_argument(
        "--branch-heads-only",
        action="store_true",
        help=(
            "Freeze the shared policy body and train only stop / escape / gate heads. "
            "This is useful for proving branch separation before low-LR full-policy tuning."
        ),
    )
    parser.add_argument(
        "--action-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the mixed actor output MSE. Set to 0 with branch supervision "
            "to train the gated heads without the combined output loss cancelling roles."
        ),
    )
    parser.add_argument(
        "--branch-supervision-weight",
        type=float,
        default=0.0,
        help="Extra branch-head action loss for front-stop vs side-wall specialization.",
    )
    parser.add_argument(
        "--branch-gate-weight",
        type=float,
        default=0.0,
        help="Binary gate supervision weight: front-blocked -> stop head, wall -> escape head.",
    )
    parser.add_argument(
        "--branch-supervise-clear",
        action="store_true",
        help=(
            "For clear, yaw, and zero-input scenarios, train both branch heads to match "
            "the target action without applying a gate label."
        ),
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("static", "rollout_clearance"),
        default="static",
        help=(
            "Target generator. rollout_clearance keeps clear/yaw/zero targets static but "
            "selects front/wall targets from candidate actions scored by predicted "
            "multi-step clearance, closing speed, and command projection."
        ),
    )
    parser.add_argument(
        "--teacher-horizon-steps",
        type=int,
        default=8,
        help="Prediction horizon in control steps for rollout_clearance teacher scoring.",
    )
    parser.add_argument(
        "--teacher-score-profile",
        choices=("default", "aggressive_safety"),
        default="default",
        help=(
            "Scoring weights for rollout_clearance. default preserves the previous "
            "teacher; aggressive_safety makes front-blocked labels prefer stopping "
            "earlier and only rewards wall-parallel progress when clearance is not "
            "being consumed."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in ORACLE_SCENARIOS),
        help="Limit BC to one or more exact oracle scenarios. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=(
            "base",
            "directional",
            "directional_clear",
            "directional_front",
            "directional_wall",
            "directional32",
            "directional32_clear",
            "directional32_front",
            "directional32_wall",
            "max_stick",
            "max_stick_clear",
            "max_stick_front",
            "max_stick_wall",
        ),
        help="Add a named oracle scenario group to the selected BC set.",
    )
    parser.add_argument(
        "--scenario-weight",
        action="append",
        default=None,
        metavar="NAME=MULTIPLIER",
        help=(
            "Multiply an oracle scenario's sampling/loss weight after filtering. "
            "Useful for focused repair while keeping rehearsal scenarios active."
        ),
    )
    parser.add_argument(
        "--scenario-target",
        action="append",
        default=None,
        metavar="NAME=VX,VY,VYAW",
        help=(
            "Override an oracle target action after filtering. "
            "Useful when a named scenario needs a slower or more conservative target."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Print progress every N optimizer iterations. Disabled by default.",
    )
    parser.add_argument(
        "--dagger-replay-epochs",
        type=int,
        default=0,
        help=(
            "Extra supervised epochs over a bounded replay buffer of policy-induced "
            "states after each collection iteration. Disabled by default."
        ),
    )
    parser.add_argument(
        "--dagger-replay-batch-size",
        type=int,
        default=512,
        help="Mini-batch size for DAgger replay updates.",
    )
    parser.add_argument(
        "--dagger-replay-max-samples",
        type=int,
        default=8192,
        help="Maximum CPU samples kept in the DAgger replay buffer.",
    )
    parser.add_argument(
        "--dagger-samples-per-step",
        type=int,
        default=2,
        help="Number of env rows sampled into replay from each rollout step.",
    )
    parser.add_argument(
        "--scenario-jitter-xy-std",
        type=float,
        default=0.0,
        help="Gaussian std in meters for obstacle XY jitter in oracle scenarios.",
    )
    parser.add_argument(
        "--scenario-jitter-radius-std",
        type=float,
        default=0.0,
        help="Gaussian std in meters for obstacle radius / half-extent jitter.",
    )
    parser.add_argument(
        "--scenario-jitter-yaw-std",
        type=float,
        default=0.0,
        help="Gaussian std in radians for obstacle yaw jitter.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _scenario_group_names(groups: Sequence[str] | None) -> set[str]:
    if not groups:
        return set()
    selected: set[str] = set()
    for group in groups:
        if group == "base":
            selected.update(scenario.name for scenario in BASE_ORACLE_SCENARIOS)
        elif group == "directional":
            selected.update(scenario.name for scenario in DIRECTIONAL_ORACLE_SCENARIOS)
        elif group == "directional_clear":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if scenario.name.startswith("directional_clear_")
            )
        elif group == "directional_front":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if scenario.name.startswith("directional_front_stop_")
            )
        elif group == "directional_wall":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if "_wall_" in scenario.name
            )
        elif group == "directional32":
            selected.update(scenario.name for scenario in DIRECTIONAL32_ORACLE_SCENARIOS)
        elif group == "directional32_clear":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL32_ORACLE_SCENARIOS
                if scenario.name.startswith("directional32_clear_")
            )
        elif group == "directional32_front":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL32_ORACLE_SCENARIOS
                if scenario.name.startswith("directional32_front_stop_")
            )
        elif group == "directional32_wall":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL32_ORACLE_SCENARIOS
                if "_wall_" in scenario.name
            )
        elif group == "max_stick":
            selected.update(scenario.name for scenario in MAX_STICK_ORACLE_SCENARIOS)
        elif group == "max_stick_clear":
            selected.update(
                scenario.name
                for scenario in MAX_STICK_ORACLE_SCENARIOS
                if scenario.name.startswith("max_stick_clear_")
            )
        elif group == "max_stick_front":
            selected.update(
                scenario.name
                for scenario in MAX_STICK_ORACLE_SCENARIOS
                if scenario.name.startswith("max_stick_front_blocked_")
            )
        elif group == "max_stick_wall":
            selected.update(
                scenario.name
                for scenario in MAX_STICK_ORACLE_SCENARIOS
                if "_wall_dir_" in scenario.name
            )
        else:
            raise ValueError(f"Unknown scenario group: {group}")
    return selected


def _scenario_weight_multipliers(raw: Sequence[str] | None) -> dict[str, float]:
    multipliers: dict[str, float] = {}
    known = {scenario.name for scenario in ORACLE_SCENARIOS}
    for item in raw or ():
        if "=" not in item:
            raise ValueError(f"scenario weight must be NAME=MULTIPLIER, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in known:
            raise ValueError(f"Unknown oracle scenario for weight multiplier: {name}")
        multiplier = float(value)
        if multiplier <= 0.0:
            raise ValueError("scenario weight multipliers must be positive")
        multipliers[name] = multiplier
    return multipliers


def _scenario_target_overrides(raw: Sequence[str] | None) -> dict[str, tuple[float, float, float]]:
    overrides: dict[str, tuple[float, float, float]] = {}
    known = {scenario.name for scenario in ORACLE_SCENARIOS}
    for item in raw or ():
        if "=" not in item:
            raise ValueError(f"scenario target must be NAME=VX,VY,VYAW, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in known:
            raise ValueError(f"Unknown oracle scenario for target override: {name}")
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"scenario target must have three comma-separated values: {item!r}")
        target = tuple(float(part) for part in parts)
        limits = (MAX_X_SPEED, MAX_Y_SPEED, 2.0)
        if any(abs(target[index]) > limits[index] + 1e-6 for index in range(3)):
            raise ValueError(f"scenario target exceeds physical limits: {item!r}")
        overrides[name] = target
    return overrides


def _selected_scenarios(
    names: Sequence[str] | None,
    groups: Sequence[str] | None = None,
    scenario_weight: Sequence[str] | None = None,
    scenario_target: Sequence[str] | None = None,
) -> tuple[OracleScenario, ...]:
    selected_names = set(names or ())
    selected_names.update(_scenario_group_names(groups))
    if not selected_names:
        selected = ORACLE_SCENARIOS
    else:
        selected = tuple(scenario for scenario in ORACLE_SCENARIOS if scenario.name in selected_names)
        if len(selected) != len(selected_names):
            known = {scenario.name for scenario in ORACLE_SCENARIOS}
            missing = sorted(selected_names - known)
            raise ValueError(f"Unknown oracle scenario(s): {missing}")
    multipliers = _scenario_weight_multipliers(scenario_weight)
    target_overrides = _scenario_target_overrides(scenario_target)
    selected_lookup = {scenario.name for scenario in selected}
    unused_targets = sorted(set(target_overrides) - selected_lookup)
    if unused_targets:
        raise ValueError(f"Scenario target overrides were not selected: {unused_targets}")
    if not multipliers and not target_overrides:
        return selected
    unused = sorted(set(multipliers) - selected_lookup)
    if unused:
        raise ValueError(f"Scenario weight multipliers were not selected: {unused}")
    return tuple(
        replace(
            scenario,
            weight=scenario.weight * multipliers.get(scenario.name, 1.0),
            target_action=target_overrides.get(scenario.name, scenario.target_action),
        )
        for scenario in selected
    )


def _scenario_probabilities(scenarios: Sequence[OracleScenario]) -> np.ndarray:
    weights = np.asarray([scenario.weight for scenario in scenarios], dtype=np.float64)
    return weights / np.sum(weights)


def _shuffled_scenarios(
    rng: np.random.Generator, scenarios: Sequence[OracleScenario]
) -> list[OracleScenario]:
    order = rng.permutation(len(scenarios))
    return [scenarios[int(index)] for index in order]


def _target_tensor(
    scenario: OracleScenario, *, num_envs: int, device: str | torch.device
) -> torch.Tensor:
    return torch.tensor(scenario.target_action, dtype=torch.float32, device=device).repeat(
        int(num_envs), 1
    )


def _dedupe_actions(actions: list[np.ndarray]) -> np.ndarray:
    if not actions:
        return np.zeros((0, 3), dtype=np.float32)
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    low = np.asarray([-MAX_X_SPEED, -MAX_Y_SPEED, -2.0], dtype=np.float64)
    high = np.asarray([MAX_X_SPEED, MAX_Y_SPEED, 2.0], dtype=np.float64)
    for action in actions:
        clipped = np.clip(np.asarray(action, dtype=np.float64), low, high)
        key = tuple(float(round(value, 4)) for value in clipped)
        unique[key] = clipped
    return np.stack(list(unique.values()), axis=0).astype(np.float32)


def _teacher_candidate_actions(scenario: OracleScenario) -> np.ndarray:
    command = np.asarray(scenario.behavior.command, dtype=np.float64)
    target = np.asarray(scenario.target_action, dtype=np.float64)
    actions: list[np.ndarray] = [target, np.zeros(3, dtype=np.float64), command]
    planar = command[:2]
    norm = float(np.linalg.norm(planar))
    if norm <= 1.0e-6:
        actions.extend(
            [
                np.asarray([0.0, 0.0, target[2]], dtype=np.float64),
                np.asarray([0.0, 0.0, command[2]], dtype=np.float64),
            ]
        )
        return _dedupe_actions(actions)
    direction = planar / norm
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    for scale in (0.10, 0.20, 0.35, 0.50, 0.65, 0.85, 1.0):
        actions.append(np.asarray([command[0] * scale, command[1] * scale, target[2]]))
    for parallel in (0.0, 0.12, 0.25, 0.40, 0.60):
        for lateral_speed in (0.12, 0.22, 0.35, 0.50, 0.70):
            for sign in (-1.0, 1.0):
                xy = direction * parallel + lateral * lateral_speed * sign
                actions.append(np.asarray([xy[0], xy[1], target[2]], dtype=np.float64))
    for parallel in (0.15, 0.30, 0.45, 0.65):
        for sign in (-1.0, 1.0):
            xy = direction * parallel + lateral * 0.25 * sign
            actions.append(np.asarray([xy[0], xy[1], target[2]], dtype=np.float64))
    return _dedupe_actions(actions)


def _teacher_score_config(profile: str = "default") -> TeacherScoreConfig:
    if profile == "default":
        return TeacherScoreConfig(profile="default")
    if profile == "aggressive_safety":
        return TeacherScoreConfig(
            profile="aggressive_safety",
            front_collision_penalty=420.0,
            front_projection_penalty=80.0,
            front_speed_penalty=22.0,
            front_closing_penalty=24.0,
            front_opening_reward=2.0,
            front_clearance_floor=0.14,
            front_clearance_floor_penalty=55.0,
            front_lateral_penalty=1.2,
            wall_collision_penalty=520.0,
            wall_closing_penalty=42.0,
            wall_projection_reward=2.2,
            wall_opening_reward=4.0,
            wall_lateral_penalty=1.2,
            wall_speed_penalty=0.16,
            wall_clearance_floor=0.12,
            wall_clearance_floor_penalty=85.0,
            wall_gate_projection_on_closing=True,
        )
    raise ValueError(f"Unknown teacher score profile: {profile!r}")


def _predict_repeated_action_metrics(
    env: Any, candidates: np.ndarray, horizon_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env_ids = np.arange(env.num_envs, dtype=np.int32)
    horizon = max(int(horizon_steps), 1)
    candidate_count = int(candidates.shape[0])
    min_clearance = np.empty((env.num_envs, candidate_count), dtype=np.float64)
    final_clearance = np.empty((env.num_envs, candidate_count), dtype=np.float64)
    mean_executed = np.empty((env.num_envs, candidate_count, 3), dtype=np.float64)
    velocity_limit = np.asarray(env._velocity_limit, dtype=np.float64)
    accel_delta = np.asarray(env._accel_delta_limit, dtype=np.float64)
    start_pose = np.asarray(env._pose, dtype=np.float64)
    start_velocity = np.asarray(env._velocity, dtype=np.float64)
    for candidate_id, candidate in enumerate(np.asarray(candidates, dtype=np.float64)):
        pose = start_pose.copy()
        velocity = start_velocity.copy()
        clearance_values: list[np.ndarray] = []
        executed_sum = np.zeros_like(velocity)
        target = np.clip(candidate[None, :], -velocity_limit[None, :], velocity_limit[None, :])
        for _step in range(horizon):
            delta = np.clip(target - velocity, -accel_delta[None, :], accel_delta[None, :])
            velocity = np.clip(
                velocity + delta,
                -velocity_limit[None, :],
                velocity_limit[None, :],
            )
            pose = env._predict_pose_from_action(pose, velocity, dt=env._cfg.ctrl_dt)
            clearance_values.append(
                np.asarray(env._compute_clearance_at_pose(env_ids, pose), dtype=np.float64)
            )
            executed_sum += velocity
        clearance_stack = np.stack(clearance_values, axis=0)
        min_clearance[:, candidate_id] = np.min(clearance_stack, axis=0)
        final_clearance[:, candidate_id] = clearance_stack[-1]
        mean_executed[:, candidate_id, :] = executed_sum / float(horizon)
    return min_clearance, final_clearance, mean_executed


def _rollout_clearance_teacher_actions(
    env: Any,
    scenario: OracleScenario,
    *,
    horizon_steps: int,
    score_config: TeacherScoreConfig | None = None,
) -> np.ndarray:
    role = _branch_role(scenario.name)
    if role is None:
        return np.asarray([scenario.target_action] * env.num_envs, dtype=np.float32)
    cfg = score_config or _teacher_score_config("default")
    candidates = _teacher_candidate_actions(scenario)
    if candidates.shape[0] == 0:
        return np.asarray([scenario.target_action] * env.num_envs, dtype=np.float32)
    min_clearance, final_clearance, mean_executed = _predict_repeated_action_metrics(
        env, candidates, horizon_steps
    )
    command = np.asarray(env._commands, dtype=np.float64)
    planar = command[:, :2]
    planar_norm = np.linalg.norm(planar, axis=1)
    direction = np.zeros_like(planar)
    active = planar_norm > float(env._cfg.command.deadband)
    direction[active] = planar[active] / np.maximum(planar_norm[active, None], 1.0e-6)
    projection = np.sum(candidates[None, :, :2] * direction[:, None, :], axis=2)
    lateral = np.abs(
        candidates[None, :, 0] * direction[:, None, 1]
        - candidates[None, :, 1] * direction[:, None, 0]
    )
    speed = np.linalg.norm(candidates[None, :, :2], axis=2)
    prev_clearance = np.asarray(env._nearest_clearance, dtype=np.float64)[:, None]
    closing = np.maximum(prev_clearance - min_clearance, 0.0)
    opening = np.maximum(final_clearance - prev_clearance, 0.0)
    collision_depth = np.maximum(0.05 - min_clearance, 0.0)
    if role == 0.0:
        floor_depth = np.maximum(float(cfg.front_clearance_floor) - min_clearance, 0.0)
        score = (
            -float(cfg.front_collision_penalty) * collision_depth
            - float(cfg.front_clearance_floor_penalty) * floor_depth
            - float(cfg.front_projection_penalty) * np.maximum(projection, 0.0) ** 2
            - float(cfg.front_speed_penalty) * speed**2
            - float(cfg.front_closing_penalty) * closing
            - float(cfg.front_lateral_penalty) * lateral
            + float(cfg.front_opening_reward) * opening
        )
    else:
        normalized_projection = projection / np.maximum(planar_norm[:, None], 1.0e-6)
        projection_credit = np.clip(normalized_projection, 0.0, 1.0)
        if bool(cfg.wall_gate_projection_on_closing):
            projection_credit *= np.clip((0.015 - closing) / 0.015, 0.0, 1.0)
        floor_depth = np.maximum(float(cfg.wall_clearance_floor) - min_clearance, 0.0)
        score = (
            -float(cfg.wall_collision_penalty) * collision_depth
            - float(cfg.wall_clearance_floor_penalty) * floor_depth
            - float(cfg.wall_closing_penalty) * closing
            + float(cfg.wall_projection_reward) * projection_credit
            + float(cfg.wall_opening_reward) * opening
            - float(cfg.wall_lateral_penalty) * lateral
            - float(cfg.wall_speed_penalty) * speed**2
        )
    chosen = np.argmax(score, axis=1)
    return candidates[chosen].astype(np.float32, copy=False)


def _teacher_target_tensor(
    env: Any,
    scenario: OracleScenario,
    *,
    device: str | torch.device,
    mode: str,
    horizon_steps: int,
    score_config: TeacherScoreConfig | None = None,
) -> torch.Tensor:
    if str(mode) == "static":
        return _target_tensor(scenario, num_envs=env.num_envs, device=device)
    if str(mode) != "rollout_clearance":
        raise ValueError(f"Unknown teacher mode: {mode!r}")
    actions = _rollout_clearance_teacher_actions(
        env,
        scenario,
        horizon_steps=int(horizon_steps),
        score_config=score_config,
    )
    return torch.as_tensor(actions, dtype=torch.float32, device=device)


def _make_runner(args: argparse.Namespace) -> tuple[Any, Any, Any, Path, str]:
    train_rsl_rl.ensure_registries()
    cfg = checkpoint_eval._compose_cfg(args)
    device = checkpoint_eval._resolve_device(args.device)
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(cfg)
    env_cfg_override.update(
        {
            "seed": int(args.seed),
            "large_scene": {"enabled": False},
            "obstacles": {
                "count": 4,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
            "human_command": {"enabled": False, "render_enabled": False},
        }
    )
    env = train_rsl_rl.create_env(cfg, num_envs=int(args.num_envs), env_cfg_override=env_cfg_override)
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["logger"] = "none"
    train_cfg["logger"] = "none"
    patch_rsl_rl_resume_state()
    runner = train_rsl_rl.OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    load_path, _ = train_rsl_rl.parse_checkpoint_path(cfg, root_dir=train_rsl_rl.ROOT_DIR)
    if load_path is None or not load_path.exists():
        raise FileNotFoundError(
            f"Could not resolve checkpoint from load_run={args.load_run!r} checkpoint={args.checkpoint!r}"
        )
    with train_rsl_rl.policy_load_dim_guard(
        env_obs_dim=getattr(wrapped_env, "num_obs", None),
        env_action_dim=getattr(wrapped_env, "num_actions", None),
        algo_name="ppo",
    ):
        if bool(getattr(args, "partial_actor_load", False)):
            runner.load(
                str(load_path),
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
                strict=False,
                map_location=device,
            )
            policy = runner.alg.get_policy()
            initializer = getattr(policy, "initialize_branches_from_shared_head", None)
            if callable(initializer):
                initializer()
        else:
            runner.load(str(load_path), map_location=device)
    return runner, env, wrapped_env, load_path, device


def _save_corrected_checkpoint(runner: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = getattr(runner.alg, "optimizer", None)
    if optimizer is not None:
        optimizer.state.clear()
    payload = runner.alg.save()
    payload["iter"] = int(getattr(runner, "current_learning_iteration", 0))
    payload["infos"] = None
    logger = getattr(runner, "logger", None)
    payload["unilab_logger_state"] = {
        "tot_time": float(getattr(logger, "tot_time", 0.0)),
        "tot_timesteps": int(getattr(logger, "tot_timesteps", 0)),
    }
    torch.save(payload, output)


def _weighted_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    per_sample = torch.mean((prediction - target) ** 2, dim=1)
    return torch.sum(per_sample * weight) / torch.clamp(torch.sum(weight), min=1.0e-6)


def _set_branch_heads_only_trainable(policy: torch.nn.Module) -> int:
    branch_modules = (
        getattr(policy, "stop_action_head", None),
        getattr(policy, "escape_action_head", None),
        getattr(policy, "action_gate_head", None),
    )
    if any(module is None for module in branch_modules):
        raise ValueError("--branch-heads-only requires a gated_two_head actor")
    for param in policy.parameters():
        param.requires_grad_(False)
    trainable = 0
    for module in branch_modules:
        assert isinstance(module, torch.nn.Module)
        for param in module.parameters():
            param.requires_grad_(True)
            trainable += int(param.numel())
    if trainable <= 0:
        raise ValueError("No branch-head parameters were made trainable")
    return trainable


def _trainable_parameters(policy: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in policy.parameters() if param.requires_grad]


def _branch_role(scenario_name: str) -> float | None:
    if "front_blocked" in scenario_name or scenario_name == "front_blocked_stop":
        return 0.0
    if "_wall_" in scenario_name or scenario_name in {"right_wall_slide", "left_wall_slide"}:
        return 1.0
    return None


def _branch_supervision_loss(
    *,
    policy: torch.nn.Module,
    obs: Any,
    target: torch.Tensor,
    scenario_name: str,
    action_weight: float,
    gate_weight: float,
    supervise_clear_heads: bool = False,
) -> torch.Tensor | None:
    role = _branch_role(scenario_name)
    if role is None and (not supervise_clear_heads or action_weight <= 0.0):
        return None
    if role is not None and action_weight <= 0.0 and gate_weight <= 0.0:
        return None
    branch_outputs = getattr(policy, "branch_action_outputs", None)
    if not callable(branch_outputs):
        return None
    outputs = branch_outputs(obs)
    if outputs is None:
        return None
    stop_action, escape_action, gate = outputs
    loss = target.new_zeros(())
    if role is None:
        loss = loss + float(action_weight) * 0.5 * (
            torch.nn.functional.mse_loss(stop_action, target)
            + torch.nn.functional.mse_loss(escape_action, target)
        )
        return loss
    if action_weight > 0.0:
        selected = stop_action if role == 0.0 else escape_action
        loss = loss + float(action_weight) * torch.nn.functional.mse_loss(selected, target)
    if gate_weight > 0.0:
        gate_target = torch.full_like(gate, float(role))
        gate_clamped = torch.clamp(gate, min=1.0e-6, max=1.0 - 1.0e-6)
        loss = loss + float(gate_weight) * torch.nn.functional.binary_cross_entropy(
            gate_clamped,
            gate_target,
        )
    return loss


def _supervised_loss(
    *,
    policy: torch.nn.Module,
    obs: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    scenario_name: str,
    action_loss_weight: float,
    branch_supervision_weight: float,
    branch_gate_weight: float,
    branch_supervise_clear: bool,
) -> torch.Tensor:
    loss = target.new_zeros(())
    if action_loss_weight > 0.0:
        loss = loss + float(action_loss_weight) * torch.nn.functional.mse_loss(
            prediction, target
        )
    branch_loss = _branch_supervision_loss(
        policy=policy,
        obs=obs,
        target=target,
        scenario_name=scenario_name,
        action_weight=branch_supervision_weight,
        gate_weight=branch_gate_weight,
        supervise_clear_heads=branch_supervise_clear,
    )
    if branch_loss is not None:
        loss = loss + branch_loss
    return loss


def _train_dagger_replay(
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: DaggerReplayBuffer,
    rng: np.random.Generator,
    device: str | torch.device,
    epochs: int,
    batch_size: int,
    action_loss_weight: float = 1.0,
) -> list[float]:
    losses: list[float] = []
    if epochs <= 0 or replay.size <= 0 or action_loss_weight <= 0.0:
        return losses
    updates_per_epoch = max(1, math.ceil(replay.size / max(int(batch_size), 1)))
    for _epoch in range(int(epochs)):
        for _update in range(updates_per_epoch):
            obs_batch, target_batch, weight_batch = replay.sample(
                rng=rng,
                batch_size=int(batch_size),
                device=device,
            )
            prediction = policy(obs_batch)
            loss = float(action_loss_weight) * _weighted_action_mse(
                prediction, target_batch, weight_batch
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_trainable_parameters(policy), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return losses


def train_wall_slide_bc(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    runner, env, wrapped_env, load_path, device = _make_runner(args)
    policy = runner.alg.get_policy()
    policy.train()
    branch_heads_only = bool(getattr(args, "branch_heads_only", False))
    trainable_param_count = (
        _set_branch_heads_only_trainable(policy) if branch_heads_only else None
    )
    trainable_params = _trainable_parameters(policy)
    if not trainable_params:
        raise ValueError("No trainable policy parameters are available")
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))
    scenarios = _selected_scenarios(
        args.scenario,
        args.scenario_group,
        args.scenario_weight,
        getattr(args, "scenario_target", None),
    )
    probabilities = _scenario_probabilities(scenarios)
    loss_history: list[float] = []
    replay_loss_history: list[float] = []
    scenario_counts = {scenario.name: 0 for scenario in scenarios}
    dagger_replay_epochs = int(getattr(args, "dagger_replay_epochs", 0))
    dagger_replay_batch_size = int(getattr(args, "dagger_replay_batch_size", 512))
    dagger_replay_samples_per_step = int(getattr(args, "dagger_samples_per_step", 2))
    scenario_jitter_xy_std = float(getattr(args, "scenario_jitter_xy_std", 0.0))
    scenario_jitter_radius_std = float(getattr(args, "scenario_jitter_radius_std", 0.0))
    scenario_jitter_yaw_std = float(getattr(args, "scenario_jitter_yaw_std", 0.0))
    branch_supervision_weight = float(getattr(args, "branch_supervision_weight", 0.0))
    branch_gate_weight = float(getattr(args, "branch_gate_weight", 0.0))
    branch_supervise_clear = bool(getattr(args, "branch_supervise_clear", False))
    action_loss_weight = float(getattr(args, "action_loss_weight", 1.0))
    if action_loss_weight < 0.0:
        raise ValueError("--action-loss-weight must be non-negative")
    teacher_mode = str(getattr(args, "teacher_mode", "static"))
    teacher_horizon_steps = int(getattr(args, "teacher_horizon_steps", 8))
    if teacher_horizon_steps <= 0:
        raise ValueError("--teacher-horizon-steps must be positive")
    teacher_score_profile = str(getattr(args, "teacher_score_profile", "default"))
    teacher_score_config = _teacher_score_config(teacher_score_profile)
    replay = DaggerReplayBuffer(
        max_samples=int(getattr(args, "dagger_replay_max_samples", 8192))
    )
    started_at = time.time()
    try:
        wrapped_env.reset()
        if bool(args.dry_run):
            selected = scenarios[0]
            behavior = _training_behavior(
                selected,
                rng=rng,
                xy_std=scenario_jitter_xy_std,
                radius_std=scenario_jitter_radius_std,
                yaw_std=scenario_jitter_yaw_std,
            )
            obs = _apply_scenario(env, wrapped_env, behavior)
            output = policy(obs)
            target = _teacher_target_tensor(
                env,
                selected,
                device=device,
                mode=teacher_mode,
                horizon_steps=teacher_horizon_steps,
                score_config=teacher_score_config,
            )
            loss = torch.nn.functional.mse_loss(output, target)
            return {
                "status": "dry_run",
                "load_path": str(load_path),
                "output": str(args.output),
                "loss": float(loss.detach().cpu().item()),
                "policy_output_shape": list(output.shape),
                "rollout_actions": str(args.rollout_actions),
                "teacher_mode": teacher_mode,
                "teacher_score_profile": teacher_score_profile,
                "scenarios": [scenario.name for scenario in scenarios],
            }
        for iteration in range(int(args.iterations)):
            if bool(args.balanced_batch):
                optimizer.zero_grad(set_to_none=True)
                iteration_losses: list[float] = []
                normalizer = float(sum(scenario.weight for scenario in scenarios))
                normalizer *= float(max(int(args.rollout_steps), 1))
                for scenario in _shuffled_scenarios(rng, scenarios):
                    scenario_counts[scenario.name] += 1
                    behavior = _training_behavior(
                        scenario,
                        rng=rng,
                        xy_std=scenario_jitter_xy_std,
                        radius_std=scenario_jitter_radius_std,
                        yaw_std=scenario_jitter_yaw_std,
                    )
                    obs = _apply_scenario(env, wrapped_env, behavior)
                    for _step in range(int(args.rollout_steps)):
                        prediction = policy(obs)
                        target = _teacher_target_tensor(
                            env,
                            scenario,
                            device=device,
                            mode=teacher_mode,
                            horizon_steps=teacher_horizon_steps,
                            score_config=teacher_score_config,
                        )
                        raw_loss = _supervised_loss(
                            policy=policy,
                            obs=obs,
                            prediction=prediction,
                            target=target,
                            scenario_name=scenario.name,
                            action_loss_weight=action_loss_weight,
                            branch_supervision_weight=branch_supervision_weight,
                            branch_gate_weight=branch_gate_weight,
                            branch_supervise_clear=branch_supervise_clear,
                        )
                        loss = raw_loss * float(scenario.weight) / normalizer
                        loss.backward()
                        replay.append(
                            obs,
                            target,
                            weight=float(scenario.weight),
                            rng=rng,
                            samples_per_step=(
                                dagger_replay_samples_per_step
                                if dagger_replay_epochs > 0
                                else 0
                            ),
                        )
                        raw_loss_value = float(raw_loss.detach().cpu().item())
                        iteration_losses.append(raw_loss_value)
                        loss_history.append(raw_loss_value)
                        with torch.no_grad():
                            step_action = (
                                prediction.detach()
                                if args.rollout_actions == "policy"
                                else target
                            )
                            obs, _rewards, dones, _infos = wrapped_env.step(step_action)
                            if bool(torch.any(dones).item()):
                                behavior = _training_behavior(
                                    scenario,
                                    rng=rng,
                                    xy_std=scenario_jitter_xy_std,
                                    radius_std=scenario_jitter_radius_std,
                                    yaw_std=scenario_jitter_yaw_std,
                                )
                                obs = _apply_scenario(env, wrapped_env, behavior)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                replay_losses = _train_dagger_replay(
                    policy=policy,
                    optimizer=optimizer,
                    replay=replay,
                    rng=rng,
                    device=device,
                    epochs=dagger_replay_epochs,
                    batch_size=dagger_replay_batch_size,
                    action_loss_weight=action_loss_weight,
                )
                replay_loss_history.extend(replay_losses)
                if args.progress_interval > 0 and (
                    (iteration + 1) % int(args.progress_interval) == 0
                ):
                    mean_loss = float(np.mean(iteration_losses)) if iteration_losses else 0.0
                    max_loss = float(np.max(iteration_losses)) if iteration_losses else 0.0
                    print(
                        json.dumps(
                            {
                                "iteration": iteration + 1,
                                "iterations": int(args.iterations),
                                "loss_iteration_mean": mean_loss,
                                "loss_iteration_max": max_loss,
                                "replay_loss_recent": (
                                    replay_loss_history[-1] if replay_loss_history else None
                                ),
                                "replay_size": replay.size,
                                "replay_updates": len(replay_loss_history),
                                "scenario_counts": scenario_counts,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                continue

            scenario = scenarios[int(rng.choice(len(scenarios), p=probabilities))]
            scenario_counts[scenario.name] += 1
            behavior = _training_behavior(
                scenario,
                rng=rng,
                xy_std=scenario_jitter_xy_std,
                radius_std=scenario_jitter_radius_std,
                yaw_std=scenario_jitter_yaw_std,
            )
            obs = _apply_scenario(env, wrapped_env, behavior)
            for _step in range(int(args.rollout_steps)):
                prediction = policy(obs)
                target = _teacher_target_tensor(
                    env,
                    scenario,
                    device=device,
                    mode=teacher_mode,
                    horizon_steps=teacher_horizon_steps,
                    score_config=teacher_score_config,
                )
                loss = _supervised_loss(
                    policy=policy,
                    obs=obs,
                    prediction=prediction,
                    target=target,
                    scenario_name=scenario.name,
                    action_loss_weight=action_loss_weight,
                    branch_supervision_weight=branch_supervision_weight,
                    branch_gate_weight=branch_gate_weight,
                    branch_supervise_clear=branch_supervise_clear,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                replay.append(
                    obs,
                    target,
                    weight=float(scenario.weight),
                    rng=rng,
                    samples_per_step=(
                        dagger_replay_samples_per_step if dagger_replay_epochs > 0 else 0
                    ),
                )
                loss_history.append(float(loss.detach().cpu().item()))
                with torch.no_grad():
                    step_action = prediction.detach() if args.rollout_actions == "policy" else target
                    obs, _rewards, dones, _infos = wrapped_env.step(step_action)
                    if bool(torch.any(dones).item()):
                        behavior = _training_behavior(
                            scenario,
                            rng=rng,
                            xy_std=scenario_jitter_xy_std,
                            radius_std=scenario_jitter_radius_std,
                            yaw_std=scenario_jitter_yaw_std,
                        )
                        obs = _apply_scenario(env, wrapped_env, behavior)
            replay_losses = _train_dagger_replay(
                policy=policy,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
                device=device,
                epochs=dagger_replay_epochs,
                batch_size=dagger_replay_batch_size,
                action_loss_weight=action_loss_weight,
            )
            replay_loss_history.extend(replay_losses)
            if args.progress_interval > 0 and (
                (iteration + 1) % int(args.progress_interval) == 0
            ):
                print(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "iterations": int(args.iterations),
                            "loss_recent": float(loss_history[-1]),
                            "replay_loss_recent": (
                                replay_loss_history[-1] if replay_loss_history else None
                            ),
                            "replay_size": replay.size,
                            "replay_updates": len(replay_loss_history),
                            "scenario_counts": scenario_counts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        output = Path(args.output)
        _save_corrected_checkpoint(runner, output)
        return {
            "status": "completed",
            "load_path": str(load_path),
            "output": str(output),
            "iterations": int(args.iterations),
            "rollout_steps": int(args.rollout_steps),
            "num_envs": int(args.num_envs),
            "learning_rate": float(args.learning_rate),
            "rollout_actions": str(args.rollout_actions),
            "balanced_batch": bool(args.balanced_batch),
            "dagger_replay_epochs": dagger_replay_epochs,
            "dagger_replay_batch_size": dagger_replay_batch_size,
            "dagger_replay_max_samples": int(getattr(args, "dagger_replay_max_samples", 8192)),
            "dagger_samples_per_step": dagger_replay_samples_per_step,
            "scenario_jitter_xy_std": scenario_jitter_xy_std,
            "scenario_jitter_radius_std": scenario_jitter_radius_std,
            "scenario_jitter_yaw_std": scenario_jitter_yaw_std,
            "branch_supervision_weight": branch_supervision_weight,
            "branch_gate_weight": branch_gate_weight,
            "branch_supervise_clear": branch_supervise_clear,
            "branch_heads_only": branch_heads_only,
            "action_loss_weight": action_loss_weight,
            "teacher_mode": teacher_mode,
            "teacher_horizon_steps": teacher_horizon_steps,
            "teacher_score_profile": teacher_score_profile,
            "trainable_param_count": (
                int(trainable_param_count)
                if trainable_param_count is not None
                else int(sum(param.numel() for param in trainable_params))
            ),
            "replay_size": replay.size,
            "replay_updates": len(replay_loss_history),
            "replay_loss_final": replay_loss_history[-1] if replay_loss_history else None,
            "scenarios": [scenario.name for scenario in scenarios],
            "scenario_counts": scenario_counts,
            "loss_initial": loss_history[0] if loss_history else None,
            "loss_final": loss_history[-1] if loss_history else None,
            "loss_mean_last_20": (
                float(np.mean(loss_history[-20:])) if len(loss_history) >= 20 else None
            ),
            "wall_time_sec": time.time() - started_at,
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = train_wall_slide_bc(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
