#!/usr/bin/env python3
"""Sync this branch to the OmniCar training host with project defaults."""

from __future__ import annotations

import argparse
import os
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
DEFAULT_VENV_SOURCE = "/home/zsh/develop/worktrees/UniLab-omni-car/.venv"


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
    parser.add_argument("--venv-source", default=DEFAULT_VENV_SOURCE)
    parser.add_argument("--dry-run", action="store_true", help="Print sync plan and exit.")
    return parser.parse_args(argv)


def build_sync_args(args: argparse.Namespace) -> list[str]:
    sync_args = [
        "--remote",
        str(args.remote),
        "--ssh-port",
        str(args.ssh_port),
        "--repo-root",
        str(Path(args.repo_root)),
        "--http-port",
        str(args.http_port),
        "--bundle-source-url",
        str(args.bundle_source_url),
        "--origin-url",
        str(args.origin_url),
        "--remote-bundle-path",
        str(args.remote_bundle_path),
        "--remote-repo-path",
        str(args.remote_repo_path),
        "--remote-worktree-path",
        str(args.remote_worktree_path),
        "--venv-source",
        str(args.venv_source),
    ]
    if args.local_host:
        sync_args.extend(["--local-host", str(args.local_host)])
    if not bool(args.no_clone_proxy):
        sync_args.extend(["--clone-proxy", str(args.clone_proxy or default_clone_proxy())])
    if bool(args.dry_run):
        sync_args.append("--dry-run")
    return sync_args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return sync_remote_bundle.main(build_sync_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
