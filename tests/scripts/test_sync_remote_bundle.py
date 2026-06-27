from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sync_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "sync_remote_bundle.py"
    spec = importlib.util.spec_from_file_location("sync_remote_bundle", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_remote_bundle"] = module
    spec.loader.exec_module(module)
    return module


def test_remote_script_updates_existing_worktree_without_deleting_it() -> None:
    sync = _load_sync_module()
    plan = sync.SyncPlan(
        repo_root=Path("/local/UniLab"),
        branch="codex/omni-car-grid-ppo-wt",
        head="a" * 40,
        remote="zsh@skkac.top",
        ssh_port=6010,
        local_host="192.168.0.3",
        http_port=8766,
        bundle_name="unilab.bundle",
        bundle_dir=Path("/tmp"),
        remote_bundle_path="/home/zsh/develop/repos/UniLab.gitbundle",
        remote_repo_path="/home/zsh/develop/repos/UniLab",
        remote_worktree_path="/home/zsh/develop/worktrees/UniLab-omni-car-git",
        remote_worktree_branch="codex/omni-car-grid-ppo-wt-remote",
        origin_url="git@github.com:skkaczsh/UniLab.git",
        venv_source="/home/zsh/develop/worktrees/UniLab-omni-car/.venv",
    )

    remote_script = sync._build_remote_script(plan)

    assert "rm -rf" not in remote_script
    assert "rev-parse --is-inside-work-tree" in remote_script
    assert "git clone /home/zsh/develop/repos/UniLab.gitbundle" in remote_script
    assert "git -C /home/zsh/develop/repos/UniLab fetch --force" in remote_script
    assert "refs/remotes/bundle/codex/omni-car-grid-ppo-wt" in remote_script
    assert "git -C /home/zsh/develop/worktrees/UniLab-omni-car-git reset --hard" in remote_script
    assert "git@github.com:skkaczsh/UniLab.git" in remote_script
    assert "ln -sfn /home/zsh/develop/worktrees/UniLab-omni-car/.venv" in remote_script
