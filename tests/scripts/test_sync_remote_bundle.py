from __future__ import annotations

import importlib.util
import subprocess
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
        bundle_source_url=None,
        clone_proxy=None,
        incremental_base=None,
        transport="http",
    )

    remote_script = sync._build_remote_script(plan)

    assert "rm -rf" not in remote_script
    assert "rev-parse --is-inside-work-tree" in remote_script
    assert "ls-files --others --exclude-standard -z" in remote_script
    assert "rm -f -- /home/zsh/develop/worktrees/UniLab-omni-car-git/" in remote_script
    assert "git clone /home/zsh/develop/repos/UniLab.gitbundle" in remote_script
    assert "git -C /home/zsh/develop/repos/UniLab fetch --force" in remote_script
    assert "refs/remotes/bundle/codex/omni-car-grid-ppo-wt" in remote_script
    assert "git -C /home/zsh/develop/worktrees/UniLab-omni-car-git reset --hard" in remote_script
    assert "git@github.com:skkaczsh/UniLab.git" in remote_script
    assert "ln -sfn /home/zsh/develop/worktrees/UniLab-omni-car/.venv" in remote_script


def test_bundle_env_sets_proxy_for_source_clone() -> None:
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
        origin_url=None,
        venv_source=None,
        bundle_source_url="https://github.com/skkaczsh/UniLab.git",
        clone_proxy="http://127.0.0.1:7890",
        incremental_base=None,
        transport="http",
    )

    env = sync._bundle_env(plan)

    assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_bundle_source_fetches_local_head_after_bare_clone(monkeypatch) -> None:
    sync = _load_sync_module()
    plan = sync.SyncPlan(
        repo_root=Path("/local/UniLab"),
        branch="codex/omni-car-grid-ppo-wt",
        head="b" * 40,
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
        origin_url=None,
        venv_source=None,
        bundle_source_url="https://github.com/skkaczsh/UniLab.git",
        clone_proxy="http://127.0.0.1:7890",
        incremental_base=None,
        transport="http",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    monkeypatch.setattr(sync, "_run", lambda cmd, cwd=None: plan.head)

    sync._clone_bundle_source(plan, Path("/tmp/source.git"))
    sync._fetch_local_head_into_source(plan, Path("/tmp/source.git"))

    assert calls[0][:4] == ["git", "clone", "--quiet", "--bare"]
    assert "--single-branch" in calls[0]
    assert calls[1] == [
        "git",
        "-C",
        "/tmp/source.git",
        "fetch",
        "--force",
        "/local/UniLab",
        "HEAD:refs/heads/codex/omni-car-grid-ppo-wt",
    ]


def test_incremental_bundle_uses_prerequisite_refspec_and_verify(monkeypatch) -> None:
    sync = _load_sync_module()
    plan = sync.SyncPlan(
        repo_root=Path("/local/UniLab"),
        branch="codex/omni-car-grid-ppo-wt",
        head="b" * 40,
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
        origin_url=None,
        venv_source=None,
        bundle_source_url="https://github.com/skkaczsh/UniLab.git",
        clone_proxy="http://127.0.0.1:7890",
        incremental_base="a" * 40,
        transport="http",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    sync._create_bundle(plan)

    assert calls[0] == [
        "git",
        "-C",
        "/local/UniLab",
        "bundle",
        "create",
        "/tmp/unilab.bundle",
        "codex/omni-car-grid-ppo-wt",
        "^" + "a" * 40,
    ]
    assert calls[1] == [
        "git",
        "-C",
        "/local/UniLab",
        "bundle",
        "verify",
        "/tmp/unilab.bundle",
    ]


def test_incremental_remote_script_requires_existing_repo() -> None:
    sync = _load_sync_module()
    plan = sync.SyncPlan(
        repo_root=Path("/local/UniLab"),
        branch="codex/omni-car-grid-ppo-wt",
        head="b" * 40,
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
        origin_url=None,
        venv_source=None,
        bundle_source_url=None,
        clone_proxy=None,
        incremental_base="a" * 40,
        transport="http",
    )

    remote_script = sync._build_remote_script(plan)

    assert "Incremental bundle requires an existing remote repo" in remote_script
    assert "git clone /home/zsh/develop/repos/UniLab.gitbundle" not in remote_script
    assert "fetch --force /home/zsh/develop/repos/UniLab.gitbundle" in remote_script


def test_scp_remote_script_uses_pre_copied_bundle() -> None:
    sync = _load_sync_module()
    plan = sync.SyncPlan(
        repo_root=Path("/local/UniLab"),
        branch="codex/omni-car-grid-ppo-wt",
        head="b" * 40,
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
        origin_url=None,
        venv_source=None,
        bundle_source_url=None,
        clone_proxy=None,
        incremental_base=None,
        transport="scp",
    )

    remote_script = sync._build_remote_script(plan)

    assert "curl --fail" not in remote_script
    assert "test -s /home/zsh/develop/repos/UniLab.gitbundle" in remote_script
