#!/usr/bin/env python3
"""Sync this branch to the OmniCar training host with project defaults."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.sync_remote_bundle as sync_remote_bundle

DEFAULT_REMOTE = "zsh@skkac.top"
DEFAULT_SSH_PORT = 6010
DEFAULT_LOCAL_CLASH_PROXY = "http://127.0.0.1:7890"
DEFAULT_BUNDLE_SOURCE_URL = "https://github.com/skkaczsh/UniLab.git"
DEFAULT_ORIGIN_URL = "git@github.com:skkaczsh/UniLab.git"
DEFAULT_REMOTE_BUNDLE_PATH = "/home/zsh/develop/repos/UniLab.gitbundle"
DEFAULT_REMOTE_REPO_PATH = "/home/zsh/develop/repos/UniLab"
DEFAULT_REMOTE_WORKTREE_PATH = "/home/zsh/develop/worktrees/UniLab-omni-car-git"
DEFAULT_REMOTE_WORKTREE_BRANCH = "codex/omni-car-grid-ppo-wt-remote"
DEFAULT_VENV_SOURCE = "/home/zsh/develop/worktrees/UniLab-omni-car/.venv"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def default_clone_proxy(env: dict[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get("UNILAB_SYNC_CLONE_PROXY") or DEFAULT_LOCAL_CLASH_PROXY


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="SSH target.")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT, help="SSH port.")
    parser.add_argument(
        "--local-host",
        default=None,
        help="LAN address exposed to the remote host. Defaults to sync_remote_bundle auto-detection.",
    )
    parser.add_argument("--http-port", type=int, default=0, help="Temporary local HTTP port.")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR, help="Local repo root.")
    parser.add_argument(
        "--bundle-source-url",
        default=DEFAULT_BUNDLE_SOURCE_URL,
        help="Full clone source used to avoid partial/promisor bundle corruption.",
    )
    parser.add_argument(
        "--clone-proxy",
        default=None,
        help=(
            "Local proxy for --bundle-source-url cloning. Defaults to "
            "UNILAB_SYNC_CLONE_PROXY or http://127.0.0.1:7890."
        ),
    )
    parser.add_argument("--no-clone-proxy", action="store_true", help="Disable clone proxy.")
    parser.add_argument("--origin-url", default=DEFAULT_ORIGIN_URL, help="Remote origin URL.")
    parser.add_argument("--remote-bundle-path", default=DEFAULT_REMOTE_BUNDLE_PATH)
    parser.add_argument("--remote-repo-path", default=DEFAULT_REMOTE_REPO_PATH)
    parser.add_argument("--remote-worktree-path", default=DEFAULT_REMOTE_WORKTREE_PATH)
    parser.add_argument(
        "--remote-worktree-branch",
        default=DEFAULT_REMOTE_WORKTREE_BRANCH,
        help="Stable branch name used for the remote training worktree.",
    )
    parser.add_argument("--venv-source", default=DEFAULT_VENV_SOURCE)
    parser.add_argument(
        "--incremental-base",
        default=None,
        help="Force an incremental bundle from this prerequisite commit.",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable automatic incremental bundle selection.",
    )
    parser.add_argument(
        "--transport",
        choices=("scp", "http"),
        default="scp",
        help="Bundle transfer method. scp avoids opening a local HTTP port.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print sync plan and exit.")
    return parser.parse_args(argv)


def _local_head(repo_root: Path) -> str:
    return sync_remote_bundle._run(["git", "rev-parse", "HEAD"], cwd=repo_root.resolve())


def _looks_like_commit(value: str) -> bool:
    return bool(COMMIT_RE.fullmatch(value.strip()))


def _remote_worktree_head(args: argparse.Namespace) -> str | None:
    remote_script = f"git -C {shlex.quote(str(args.remote_worktree_path))} rev-parse HEAD"
    try:
        result = subprocess.run(
            ["ssh", "-p", str(args.ssh_port), str(args.remote), remote_script],
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        return None
    head = stdout[-1].strip()
    if not _looks_like_commit(head):
        return None
    return head


def _is_ancestor(repo_root: Path, base: str, head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, head],
            cwd=str(repo_root.resolve()),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _auto_incremental_base(args: argparse.Namespace) -> tuple[str | None, bool]:
    local_head = _local_head(Path(args.repo_root))
    remote_head = _remote_worktree_head(args)
    if remote_head is None:
        return None, False
    if remote_head == local_head:
        return remote_head, True
    if _is_ancestor(Path(args.repo_root), remote_head, local_head):
        return remote_head, False
    return None, False


def build_sync_args(args: argparse.Namespace, *, incremental_base: str | None = None) -> list[str]:
    effective_incremental_base = incremental_base or args.incremental_base
    sync_args = [
        "--remote",
        str(args.remote),
        "--ssh-port",
        str(args.ssh_port),
        "--repo-root",
        str(Path(args.repo_root)),
        "--http-port",
        str(args.http_port),
        "--transport",
        str(args.transport),
        "--origin-url",
        str(args.origin_url),
        "--remote-bundle-path",
        str(args.remote_bundle_path),
        "--remote-repo-path",
        str(args.remote_repo_path),
        "--remote-worktree-path",
        str(args.remote_worktree_path),
        "--remote-worktree-branch",
        str(args.remote_worktree_branch),
        "--venv-source",
        str(args.venv_source),
    ]
    if effective_incremental_base:
        sync_args.extend(["--incremental-base", str(effective_incremental_base)])
    else:
        sync_args.extend(["--bundle-source-url", str(args.bundle_source_url)])
    if args.local_host:
        sync_args.extend(["--local-host", str(args.local_host)])
    if not effective_incremental_base and not bool(args.no_clone_proxy):
        sync_args.extend(["--clone-proxy", str(args.clone_proxy or default_clone_proxy())])
    if bool(args.dry_run):
        sync_args.append("--dry-run")
    return sync_args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.no_incremental and args.incremental_base:
        raise SystemExit("--no-incremental cannot be combined with --incremental-base")

    incremental_base = args.incremental_base
    if not args.dry_run and not args.no_incremental and incremental_base is None:
        incremental_base, up_to_date = _auto_incremental_base(args)
        if up_to_date:
            print(f"remote worktree already at {incremental_base}")
            return 0
        if incremental_base:
            print(f"using incremental bundle from remote base {incremental_base}")
    return sync_remote_bundle.main(build_sync_args(args, incremental_base=incremental_base))


if __name__ == "__main__":
    raise SystemExit(main())
