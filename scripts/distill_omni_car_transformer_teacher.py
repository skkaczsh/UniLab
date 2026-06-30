#!/usr/bin/env python3
"""Distill an OmniCar Transformer teacher into the deployable CNN-GRU student."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
from scripts.train_omni_car_wall_slide_bc import _save_corrected_checkpoint
from unilab.training.experiment import patch_rsl_rl_resume_state


@dataclass
class DistillContext:
    student_runner: Any
    teacher_runner: Any
    env: Any
    wrapped_env: Any
    student_load_path: Path
    teacher_load_path: Path
    device: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-load-run", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--student-load-run", required=True)
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--rollout-source", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--action-loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--vx-weight", type=float, default=1.0)
    parser.add_argument("--vy-weight", type=float, default=1.0)
    parser.add_argument("--vyaw-weight", type=float, default=1.0)
    parser.add_argument("--diff-weight", type=float, default=0.15)
    parser.add_argument("--jerk-weight", type=float, default=0.05)
    parser.add_argument("--zero-command-weight", type=float, default=0.5)
    parser.add_argument("--zero-command-norm", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _checkpoint_args(load_run: str, checkpoint: str | None, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        load_run=load_run,
        checkpoint=checkpoint,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
    )


def _runner_from_cfg(cfg: Any, wrapped_env: Any, *, device: str, load_run: str, checkpoint: str | None):
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["logger"] = "none"
    train_cfg["logger"] = "none"
    runner = train_rsl_rl.OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    load_path, _ = train_rsl_rl.parse_checkpoint_path(cfg, root_dir=train_rsl_rl.ROOT_DIR)
    if load_path is None or not load_path.exists():
        raise FileNotFoundError(
            f"Could not resolve checkpoint from load_run={load_run!r} checkpoint={checkpoint!r}"
        )
    with train_rsl_rl.policy_load_dim_guard(
        env_obs_dim=getattr(wrapped_env, "num_obs", None),
        env_action_dim=getattr(wrapped_env, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(load_path), map_location=device)
    return runner, load_path


def _make_context(args: argparse.Namespace) -> DistillContext:
    train_rsl_rl.ensure_registries()
    student_args = _checkpoint_args(str(args.student_load_run), args.student_checkpoint, args)
    teacher_args = _checkpoint_args(str(args.teacher_load_run), args.teacher_checkpoint, args)
    student_cfg = checkpoint_eval._compose_cfg(student_args)
    teacher_cfg = checkpoint_eval._compose_cfg(teacher_args)
    device = checkpoint_eval._resolve_device(args.device)
    rl_cfg = train_rsl_rl._algo_config_dict(student_cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(student_cfg)
    env_cfg_override.update(
        {
            "seed": int(args.seed),
            "human_command": {"enabled": False, "render_enabled": False},
        }
    )
    env = train_rsl_rl.create_env(
        student_cfg,
        num_envs=int(args.num_envs),
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = wrapper_cls(env, device=device)
    patch_rsl_rl_resume_state()
    student_runner, student_path = _runner_from_cfg(
        student_cfg,
        wrapped_env,
        device=device,
        load_run=str(args.student_load_run),
        checkpoint=args.student_checkpoint,
    )
    teacher_runner, teacher_path = _runner_from_cfg(
        teacher_cfg,
        wrapped_env,
        device=device,
        load_run=str(args.teacher_load_run),
        checkpoint=args.teacher_checkpoint,
    )
    return DistillContext(
        student_runner=student_runner,
        teacher_runner=teacher_runner,
        env=env,
        wrapped_env=wrapped_env,
        student_load_path=student_path,
        teacher_load_path=teacher_path,
        device=device,
    )


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    axis_weights: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    if kind == "mse":
        raw = torch.square(prediction - target)
    else:
        raw = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return torch.mean(raw * axis_weights)


def _current_command(obs: Any, grid_stack_dim: int) -> torch.Tensor:
    if "actor" in obs.keys():
        flat = obs["actor"]
    else:
        flat = torch.cat([obs[key] for key in sorted(obs.keys())], dim=-1)
    return flat[..., grid_stack_dim : grid_stack_dim + 3]


def distill(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    context = _make_context(args)
    student_policy = context.student_runner.alg.get_policy()
    teacher_module = context.teacher_runner.alg.get_policy()
    teacher_policy = context.teacher_runner.get_inference_policy(device=context.device)
    student_policy.train()
    teacher_module.eval()
    optimizer = torch.optim.Adam(student_policy.parameters(), lr=float(args.learning_rate))
    axis_weights = torch.tensor(
        [float(args.vx_weight), float(args.vy_weight), float(args.vyaw_weight)],
        dtype=torch.float32,
        device=context.device,
    ).reshape(1, 3)
    loss_history: list[float] = []
    started_at = time.time()
    prev_student: torch.Tensor | None = None
    prev_teacher: torch.Tensor | None = None
    prev_student_diff: torch.Tensor | None = None
    prev_teacher_diff: torch.Tensor | None = None
    try:
        obs, _ = context.wrapped_env.reset()
        if bool(args.dry_run):
            with torch.inference_mode():
                teacher_action = teacher_policy(obs)
                student_action = student_policy(obs)
            return {
                "status": "dry_run",
                "teacher_load_path": str(context.teacher_load_path),
                "student_load_path": str(context.student_load_path),
                "teacher_action_shape": list(teacher_action.shape),
                "student_action_shape": list(student_action.shape),
                "output": str(args.output),
            }
        for iteration in range(int(args.iterations)):
            with torch.no_grad():
                teacher_action = teacher_policy(obs).detach()
            student_action = student_policy(obs)
            loss = _weighted_loss(
                student_action,
                teacher_action,
                axis_weights=axis_weights,
                kind=str(args.action_loss),
            )
            if prev_student is not None and prev_teacher is not None and args.diff_weight > 0.0:
                student_diff = student_action - prev_student
                teacher_diff = teacher_action - prev_teacher
                loss = loss + float(args.diff_weight) * _weighted_loss(
                    student_diff,
                    teacher_diff,
                    axis_weights=axis_weights,
                    kind=str(args.action_loss),
                )
                if (
                    prev_student_diff is not None
                    and prev_teacher_diff is not None
                    and args.jerk_weight > 0.0
                ):
                    loss = loss + float(args.jerk_weight) * _weighted_loss(
                        student_diff - prev_student_diff,
                        teacher_diff - prev_teacher_diff,
                        axis_weights=axis_weights,
                        kind=str(args.action_loss),
                    )
                prev_student_diff = student_diff.detach()
                prev_teacher_diff = teacher_diff.detach()
            command = _current_command(obs, context.env._grid_history_len * context.env._grid_dim)
            zero_mask = (
                torch.linalg.vector_norm(command, dim=-1, keepdim=True)
                < float(args.zero_command_norm)
            ).to(dtype=student_action.dtype)
            if args.zero_command_weight > 0.0 and bool(torch.any(zero_mask > 0.0).item()):
                zero_loss = torch.mean(torch.square(student_action) * zero_mask)
                loss = loss + float(args.zero_command_weight) * zero_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_policy.parameters(), 1.0)
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))
            prev_student = student_action.detach()
            prev_teacher = teacher_action.detach()
            with torch.no_grad():
                step_action = teacher_action if args.rollout_source == "teacher" else student_action.detach()
                obs, _rewards, dones, _infos = context.wrapped_env.step(step_action)
                if bool(torch.any(dones).item()):
                    prev_student = None
                    prev_teacher = None
                    prev_student_diff = None
                    prev_teacher_diff = None
            if args.progress_interval > 0 and (
                (iteration + 1) % int(args.progress_interval) == 0
            ):
                print(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "iterations": int(args.iterations),
                            "loss_recent": loss_history[-1],
                            "loss_mean_last_20": float(np.mean(loss_history[-20:])),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        output = Path(args.output)
        _save_corrected_checkpoint(context.student_runner, output)
        return {
            "status": "completed",
            "teacher_load_path": str(context.teacher_load_path),
            "student_load_path": str(context.student_load_path),
            "output": str(output),
            "iterations": int(args.iterations),
            "num_envs": int(args.num_envs),
            "learning_rate": float(args.learning_rate),
            "rollout_source": str(args.rollout_source),
            "loss_initial": loss_history[0] if loss_history else None,
            "loss_final": loss_history[-1] if loss_history else None,
            "loss_mean_last_20": (
                float(np.mean(loss_history[-20:])) if len(loss_history) >= 20 else None
            ),
            "wall_time_sec": time.time() - started_at,
        }
    finally:
        context.env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = distill(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
