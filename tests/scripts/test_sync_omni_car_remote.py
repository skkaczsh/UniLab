from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "sync_omni_car_remote.py"
    spec = importlib.util.spec_from_file_location("sync_omni_car_remote", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_sync_args_uses_omni_car_remote_defaults() -> None:
    module = _load_module()
    args = module._parse_args(["--dry-run"])

    sync_args = module.build_sync_args(args)

    assert sync_args[:2] == ["--remote", "zsh@skkac.top"]
    assert "--ssh-port" in sync_args
    assert "6010" in sync_args
    assert "https://github.com/skkaczsh/UniLab.git" in sync_args
    assert "git@github.com:skkaczsh/UniLab.git" in sync_args
    assert "/home/zsh/develop/worktrees/UniLab-omni-car-git" in sync_args
    assert "--remote-worktree-branch" in sync_args
    assert "codex/omni-car-grid-ppo-wt-remote" in sync_args
    assert "--clone-proxy" in sync_args
    assert "http://127.0.0.1:7890" in sync_args
    assert "--dry-run" in sync_args


def test_remote_worktree_branch_does_not_follow_local_branch_name() -> None:
    module = _load_module()
    args = module._parse_args([])

    sync_args = module.build_sync_args(args)

    assert "codex/omni-car-grid-ppo-wt-remote" in sync_args
    assert "codex/omni-car-grid-ppo-wt-remote-remote" not in sync_args


def test_no_clone_proxy_omits_proxy_args() -> None:
    module = _load_module()
    args = module._parse_args(["--no-clone-proxy"])

    sync_args = module.build_sync_args(args)

    assert "--clone-proxy" not in sync_args
    assert "http://127.0.0.1:7890" not in sync_args


def test_clone_proxy_env_override() -> None:
    module = _load_module()

    assert module.default_clone_proxy({"UNILAB_SYNC_CLONE_PROXY": "http://127.0.0.1:7899"}) == (
        "http://127.0.0.1:7899"
    )


def test_main_delegates_to_bundle_sync(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    def _fake_main(argv):  # type: ignore[no-untyped-def]
        captured["argv"] = list(argv)
        return 7

    monkeypatch.setattr(module.sync_remote_bundle, "main", _fake_main)

    rc = module.main(["--local-host", "192.168.0.3", "--dry-run"])

    assert rc == 7
    assert captured["argv"] is not None
    assert "--local-host" in captured["argv"]
    assert "192.168.0.3" in captured["argv"]
