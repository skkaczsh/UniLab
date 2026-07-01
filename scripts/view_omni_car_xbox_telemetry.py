#!/usr/bin/env python3
"""Open an Xbox-controlled OmniCar checkpoint viewer with per-step telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.evaluate_omni_car_checkpoint as checkpoint_eval
import scripts.train_rsl_rl as train_rsl_rl


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-run", required=True, help="Run dir name, run dir path, or checkpoint path."
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint id or filename, e.g. 93 or model_93.pt."
    )
    parser.add_argument("--steps", type=int, default=20_000, help="Viewer steps to run.")
    parser.add_argument("--seed", type=int, default=7, help="Environment seed.")
    parser.add_argument("--device", default="cpu", help="Torch inference device.")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--axis-vx", type=int, default=1)
    parser.add_argument("--axis-vy", type=int, default=0)
    parser.add_argument(
        "--axis-vyaw",
        type=int,
        default=2,
        help="Yaw axis. On the local Xbox One S controller, right-stick X is axis 2.",
    )
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--zero-snap-norm", type=float, default=0.10)
    parser.add_argument("--idle-action-hold-norm", type=float, default=0.12)
    parser.add_argument(
        "--disable-idle-action-hold",
        action="store_true",
        help="Let the policy output actions even when the human command is near zero.",
    )
    parser.add_argument(
        "--extra-human-window",
        action="store_true",
        help="Also open the secondary human-command HUD window from the env.",
    )
    parser.add_argument(
        "--enable-stagnation-reset",
        action="store_true",
        help="Keep large-scene stagnation resets enabled. Disabled by default for manual debugging.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT_DIR / "logs" / "omni_car_viewer" / "xbox_telemetry.jsonl",
        help="JSONL telemetry output path.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Write every N steps. Collision/reset steps are always written.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print every N steps. Collision/reset steps are always printed.",
    )
    parser.add_argument("--cam-distance", type=float, default=5.0)
    return parser.parse_args(argv)


def _to_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _to_float(value: Any, env_id: int = 0) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[min(env_id, arr.size - 1)]) if arr.size else 0.0


def _to_bool(value: Any, env_id: int = 0) -> bool:
    arr = np.asarray(value, dtype=bool).reshape(-1)
    return bool(arr[min(env_id, arr.size - 1)]) if arr.size else False


def _component_row(components: Mapping[str, Any], env_id: int) -> dict[str, float]:
    return {name: _to_float(values, env_id) for name, values in sorted(components.items())}


def _telemetry_row(env: Any, step_index: int, reward: torch.Tensor, done: torch.Tensor) -> dict[str, Any]:
    if env.state is None:
        raise RuntimeError("Environment state is unavailable after step.")
    info = env.state.info
    env_id = int(info.get("human_command_env_id", 0))
    if env_id < 0:
        env_id = 0
    reward_components = info.get("reward_components", {})
    row = {
        "wall_time": time.time(),
        "step": int(step_index),
        "env_id": env_id,
        "episode_step": int(np.asarray(info["steps"]).reshape(-1)[env_id]),
        "human_command": _to_list(info.get("human_command", np.zeros(3))),
        "command": _to_list(np.asarray(info["commands"])[env_id]),
        "policy_action": _to_list(np.asarray(info["policy_action"])[env_id]),
        "executed_action": _to_list(np.asarray(info["executed_action"])[env_id]),
        "velocity": _to_list(env._velocity[env_id]),
        "pose": _to_list(env._pose[env_id]),
        "nearest_clearance": _to_float(info["nearest_clearance"], env_id),
        "command_clearance": _to_float(info.get("command_clearance", 0.0), env_id),
        "clearance_risk": _to_float(info.get("clearance_risk", 0.0), env_id),
        "reward": _to_float(reward.detach().cpu().numpy(), env_id),
        "done": _to_bool(done.detach().cpu().numpy(), env_id),
        "final_observation": _to_bool(info.get("_final_observation", False), env_id),
        "collision": _to_bool(info["collision"], env_id),
        "static_collision": _to_bool(info["static_collision"], env_id),
        "agent_collision": _to_bool(info["agent_collision"], env_id),
        "border_collision": _to_bool(info["border_collision"], env_id),
        "stagnated": _to_bool(info["stagnated"], env_id),
        "human_idle_hold": _to_bool(info.get("human_idle_hold", False), env_id),
        "reward_components": _component_row(reward_components, env_id),
    }
    return row


def _format_line(row: Mapping[str, Any]) -> str:
    def vec(value: Any) -> str:
        items = np.asarray(value, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(f"{item:+.2f}" for item in items[:3]) + "]"

    flags = []
    for key in ("collision", "static_collision", "agent_collision", "border_collision", "stagnated"):
        if row.get(key):
            flags.append(key)
    flag_text = ",".join(flags) if flags else "ok"
    total = float(row["reward_components"].get("total", row["reward"]))
    return (
        f"step={row['step']:06d} ep={row['episode_step']:04d} {flag_text} "
        f"clear={row['nearest_clearance']:+.3f} risk={row['clearance_risk']:.2f} "
        f"cmd={vec(row['command'])} policy={vec(row['policy_action'])} "
        f"exec={vec(row['executed_action'])} reward={total:+.3f}"
    )


def _make_policy_and_env(args: argparse.Namespace) -> tuple[Any, Any, Any, Path]:
    train_rsl_rl.ensure_registries()
    cfg = checkpoint_eval._compose_cfg(args)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "training.play_only", True, merge=False)
    OmegaConf.update(cfg, "training.play_render_mode", "interactive", merge=False)
    OmegaConf.update(cfg, "training.play_env_num", 1, merge=False)
    OmegaConf.update(cfg, "training.device", str(args.device), merge=False)

    device = checkpoint_eval._resolve_device(str(args.device))
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(cfg)
    env_cfg_override.update(
        {
            "seed": int(args.seed),
            "large_scene": {
                "stagnation_warmup_steps": (
                    240 if bool(args.enable_stagnation_reset) else 1_000_000_000
                ),
                "stagnation_window_steps": (
                    420 if bool(args.enable_stagnation_reset) else 1_000_000_000
                ),
            },
            "human_command": {
                "enabled": True,
                "backend": "pygame",
                "joystick_index": int(args.joystick_index),
                "env_index": 0,
                "axis_vx": int(args.axis_vx),
                "axis_vy": int(args.axis_vy),
                "axis_vyaw": int(args.axis_vyaw),
                "deadzone": float(args.deadzone),
                "zero_snap_norm": float(args.zero_snap_norm),
                "idle_action_hold": not bool(args.disable_idle_action_hold),
                "idle_action_hold_norm": float(args.idle_action_hold_norm),
                "replay_fanout": 0,
                "require_joystick": True,
                "render_enabled": bool(args.extra_human_window),
                "render_every_steps": 1,
            },
        }
    )

    env = train_rsl_rl.create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["logger"] = "none"
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
        runner.load(str(load_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
    return env, wrapped_env, policy, load_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    env, wrapped_env, policy, load_path = _make_policy_and_env(args)
    print(f"Loaded checkpoint: {load_path}")
    print(f"Telemetry log: {args.log_path}")
    print("Close the native OpenGL window or press Esc to quit.")

    step_index = 0
    log_every = max(1, int(args.log_every))
    print_every = max(1, int(args.print_every))

    with args.log_path.open("w", encoding="utf-8") as stream:

        def initialize() -> Any:
            return wrapped_env.reset()[0]

        def step(obs: Any) -> Any:
            nonlocal step_index
            with torch.inference_mode():
                next_obs, reward, done, _info = wrapped_env.step(policy(obs))
            row = _telemetry_row(env, step_index, reward, done)
            is_event = bool(
                row["done"]
                or row["final_observation"]
                or row["collision"]
                or row["static_collision"]
                or row["agent_collision"]
                or row["border_collision"]
                or row["stagnated"]
            )
            if step_index % log_every == 0 or is_event:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
            if step_index % print_every == 0 or is_event:
                print(_format_line(row), flush=True)
            step_index += 1
            return next_obs

        try:
            env.run_playback(
                initialize=initialize,
                step=step,
                num_steps=max(1, int(args.steps)),
                camera_kwargs={"cam_distance": float(args.cam_distance)},
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
