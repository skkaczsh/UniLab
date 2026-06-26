from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_viewer_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "view_omni_car_checkpoint.py"
    spec = importlib.util.spec_from_file_location("view_omni_car_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viewer_script_builds_playback_command(monkeypatch) -> None:
    viewer = _load_viewer_module()
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(command, check: bool = False):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["check"] = check
        return _Result()

    monkeypatch.setattr(viewer.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        viewer.cli,
        "find_spec",
        lambda name: object() if name == "mujoco" else None,
    )
    rc = viewer.main(
        [
            "--load-run",
            "/tmp/omni/model_93.pt",
            "--device",
            "cpu",
            "--",
            "interactive.keyboard=true",
        ]
    )

    assert rc == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[1] == str(Path(viewer.ROOT_DIR) / "scripts" / "train_rsl_rl.py")
    assert "task=omni_car_grid_avoidance/mujoco" in command
    assert "algo.load_run=/tmp/omni/model_93.pt" in command
    assert "training.play_only=true" in command
    assert "training.play_render_mode=interactive" in command
    assert "training.device=cpu" in command
    assert "interactive.keyboard=true" in command
