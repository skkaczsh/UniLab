#!/usr/bin/env python3
"""Sync the current git branch into a remote repo/worktree via a local git bundle."""

from __future__ import annotations

import argparse
import http.client
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SyncPlan:
    repo_root: Path
    branch: str
    head: str
    remote: str
    ssh_port: int
    local_host: str
    http_port: int
    bundle_name: str
    bundle_dir: Path
    remote_bundle_path: str
    remote_repo_path: str
    remote_worktree_path: str
    remote_worktree_branch: str
    origin_url: str | None
    venv_source: str | None
    bundle_source_url: str | None
    clone_proxy: str | None
    incremental_base: str | None

    @property
    def bundle_path(self) -> Path:
        return self.bundle_dir / self.bundle_name

    @property
    def bundle_url(self) -> str:
        return f"http://{self.local_host}:{self.http_port}/{self.bundle_name}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="SSH target, e.g. zsh@skkac.top")
    parser.add_argument("--ssh-port", type=int, default=22, help="Remote SSH port.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT_DIR,
        help="Local repo root to bundle. Defaults to this repo.",
    )
    parser.add_argument(
        "--local-host",
        default=None,
        help="LAN host/IP for the temporary local HTTP server. Auto-detected by default.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=0,
        help="Temporary local HTTP port. Defaults to an automatically selected free port.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Directory used for the temporary bundle file.",
    )
    parser.add_argument(
        "--remote-bundle-path",
        default="/home/zsh/develop/repos/UniLab.gitbundle",
        help="Remote path where the bundle file is downloaded.",
    )
    parser.add_argument(
        "--remote-repo-path",
        default="/home/zsh/develop/repos/UniLab",
        help="Remote git repo path.",
    )
    parser.add_argument(
        "--remote-worktree-path",
        default="/home/zsh/develop/worktrees/UniLab-omni-car-git",
        help="Remote worktree path.",
    )
    parser.add_argument(
        "--remote-worktree-branch",
        default=None,
        help="Remote worktree branch name. Defaults to <local-branch>-remote.",
    )
    parser.add_argument(
        "--origin-url",
        default=None,
        help="Optional git origin URL to set on the remote repo after bundle sync.",
    )
    parser.add_argument(
        "--venv-source",
        default=None,
        help="Optional remote venv path to symlink into the remote worktree as .venv.",
    )
    parser.add_argument(
        "--bundle-source-url",
        default=None,
        help=(
            "Optional git URL to clone into a temporary full source before creating the "
            "bundle. Use this when the local repo is a partial/promisor clone."
        ),
    )
    parser.add_argument(
        "--clone-proxy",
        default=None,
        help="Optional HTTP(S) proxy used only for --bundle-source-url cloning.",
    )
    parser.add_argument(
        "--incremental-base",
        default=None,
        help=(
            "Optional prerequisite commit. When set, create an incremental bundle "
            "containing <branch> excluding this base commit; the remote repo must "
            "already contain the base."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    return parser.parse_args(argv)


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _resolve_branch(repo_root: Path) -> tuple[str, str]:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if branch == "HEAD":
        raise RuntimeError("Detached HEAD is not supported for remote bundle sync.")
    return branch, head


def _resolve_local_host(explicit_host: str | None, remote: str) -> str:
    if explicit_host:
        return explicit_host
    host = remote.rsplit("@", 1)[-1]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, 1))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _resolve_http_port(explicit_port: int) -> int:
    if explicit_port > 0:
        return explicit_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _create_plan(args: argparse.Namespace) -> SyncPlan:
    repo_root = args.repo_root.resolve()
    branch, head = _resolve_branch(repo_root)
    bundle_name = f"{repo_root.name}-{branch.replace('/', '_')}.bundle"
    local_host = _resolve_local_host(args.local_host, args.remote)
    http_port = _resolve_http_port(int(args.http_port))
    remote_worktree_branch = args.remote_worktree_branch or f"{branch}-remote"
    return SyncPlan(
        repo_root=repo_root,
        branch=branch,
        head=head,
        remote=args.remote,
        ssh_port=int(args.ssh_port),
        local_host=local_host,
        http_port=http_port,
        bundle_name=bundle_name,
        bundle_dir=args.bundle_dir.resolve(),
        remote_bundle_path=str(args.remote_bundle_path),
        remote_repo_path=str(args.remote_repo_path),
        remote_worktree_path=str(args.remote_worktree_path),
        remote_worktree_branch=remote_worktree_branch,
        origin_url=args.origin_url,
        venv_source=args.venv_source,
        bundle_source_url=args.bundle_source_url,
        clone_proxy=args.clone_proxy,
        incremental_base=args.incremental_base,
    )


def _bundle_env(plan: SyncPlan) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if plan.clone_proxy:
        env["HTTP_PROXY"] = plan.clone_proxy
        env["HTTPS_PROXY"] = plan.clone_proxy
        env["http_proxy"] = plan.clone_proxy
        env["https_proxy"] = plan.clone_proxy
    return env


def _create_bundle_from_repo(plan: SyncPlan, source_repo: Path) -> None:
    refspec = [plan.branch]
    if plan.incremental_base:
        refspec.append(f"^{plan.incremental_base}")
    subprocess.run(
        ["git", "-C", str(source_repo), "bundle", "create", str(plan.bundle_path), *refspec],
        check=True,
    )


def _clone_bundle_source(plan: SyncPlan, source_repo: Path) -> None:
    if plan.bundle_source_url is None:
        raise ValueError("bundle_source_url is required to clone a bundle source")
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--bare",
            "--single-branch",
            "--branch",
            plan.branch,
            plan.bundle_source_url,
            str(source_repo),
        ],
        check=True,
        env=_bundle_env(plan),
    )


def _fetch_local_head_into_source(plan: SyncPlan, source_repo: Path) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "fetch",
            "--force",
            str(plan.repo_root),
            f"HEAD:refs/heads/{plan.branch}",
        ],
        check=True,
    )
    source_head = _run(["git", "-C", str(source_repo), "rev-parse", plan.branch])
    if source_head != plan.head:
        raise RuntimeError(
            f"Bundle source did not resolve local HEAD: expected {plan.head}, got {source_head}"
        )


def _verify_complete_bundle_clones(plan: SyncPlan) -> None:
    with tempfile.TemporaryDirectory(prefix="unilab-bundle-check-") as tmp:
        checkout = Path(tmp) / "checkout"
        subprocess.run(
            ["git", "clone", "--quiet", "--branch", plan.branch, str(plan.bundle_path), str(checkout)],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "fsck", "--full"], check=True)


def _verify_incremental_bundle(plan: SyncPlan) -> None:
    subprocess.run(
        ["git", "-C", str(plan.repo_root), "bundle", "verify", str(plan.bundle_path)],
        check=True,
    )


def _verify_bundle(plan: SyncPlan) -> None:
    if plan.incremental_base:
        _verify_incremental_bundle(plan)
    else:
        _verify_complete_bundle_clones(plan)


def _create_bundle(plan: SyncPlan) -> None:
    plan.bundle_dir.mkdir(parents=True, exist_ok=True)
    if plan.bundle_path.exists():
        plan.bundle_path.unlink()
    if plan.incremental_base:
        _create_bundle_from_repo(plan, plan.repo_root)
    elif plan.bundle_source_url:
        with tempfile.TemporaryDirectory(prefix="unilab-bundle-source-") as tmp:
            source_repo = Path(tmp) / "repo"
            _clone_bundle_source(plan, source_repo)
            _fetch_local_head_into_source(plan, source_repo)
            _create_bundle_from_repo(plan, source_repo)
    else:
        _create_bundle_from_repo(plan, plan.repo_root)
    _verify_bundle(plan)


def _wait_for_http_server(*, host: str, port: int, bundle_name: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1.0)
            conn.request("HEAD", f"/{bundle_name}")
            response = conn.getresponse()
            if response.status == 200:
                return
            last_error = f"HTTP {response.status} for /{bundle_name}"
        except OSError:
            last_error = "connection failed"
            time.sleep(0.1)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
    raise TimeoutError(
        f"Timed out waiting for local bundle server on http://{host}:{port}; {last_error}"
    )


def _start_http_server(plan: SyncPlan) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(plan.http_port),
            "--bind",
            "0.0.0.0",
        ],
        cwd=str(plan.bundle_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_http_server(host="127.0.0.1", port=plan.http_port, bundle_name=plan.bundle_name)
    except Exception:
        server.terminate()
        raise
    return server


def _build_remote_script(plan: SyncPlan) -> str:
    bundle_parent = str(Path(plan.remote_bundle_path).parent)
    repo_parent = str(Path(plan.remote_repo_path).parent)
    worktree_parent = str(Path(plan.remote_worktree_path).parent)
    sync_ref = f"refs/remotes/bundle/{plan.branch}"
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(bundle_parent)} {shlex.quote(repo_parent)} {shlex.quote(worktree_parent)}",
        f"curl --fail --location {shlex.quote(plan.bundle_url)} -o {shlex.quote(plan.remote_bundle_path)}",
        f"if ! git -C {shlex.quote(plan.remote_repo_path)} rev-parse --is-inside-work-tree >/dev/null 2>&1; then",
    ]
    if plan.incremental_base:
        lines.extend(
            [
                "  echo 'Incremental bundle requires an existing remote repo with the base commit.' >&2",
                "  exit 2",
            ]
        )
    else:
        lines.append(
            f"  git clone {shlex.quote(plan.remote_bundle_path)} {shlex.quote(plan.remote_repo_path)}"
        )
    lines.extend(
        [
            "else",
            (
                f"  git -C {shlex.quote(plan.remote_repo_path)} fetch --force "
                f"{shlex.quote(plan.remote_bundle_path)} "
                f"{shlex.quote('refs/heads/' + plan.branch + ':' + sync_ref)}"
            ),
            "fi",
            f"git -C {shlex.quote(plan.remote_repo_path)} cat-file -e {shlex.quote(plan.head + '^{commit}')}",
        ]
    )
    if plan.origin_url:
        lines.extend(
            [
                f"git -C {shlex.quote(plan.remote_repo_path)} remote remove origin 2>/dev/null || true",
                f"git -C {shlex.quote(plan.remote_repo_path)} remote add origin {shlex.quote(plan.origin_url)}",
            ]
        )
    lines.extend(
        [
            f"if ! git -C {shlex.quote(plan.remote_worktree_path)} rev-parse --is-inside-work-tree >/dev/null 2>&1; then",
            (
                f"  git -C {shlex.quote(plan.remote_repo_path)} worktree add "
                f"{shlex.quote(plan.remote_worktree_path)} "
                f"-B {shlex.quote(plan.remote_worktree_branch)} {shlex.quote(plan.head)}"
            ),
            "else",
            (
                f"  git -C {shlex.quote(plan.remote_worktree_path)} checkout "
                f"-B {shlex.quote(plan.remote_worktree_branch)} {shlex.quote(plan.head)}"
            ),
            f"  git -C {shlex.quote(plan.remote_worktree_path)} reset --hard {shlex.quote(plan.head)}",
            "fi",
        ]
    )
    if plan.venv_source:
        lines.extend(
            [
                f"ln -sfn {shlex.quote(plan.venv_source)} {shlex.quote(plan.remote_worktree_path + '/.venv')}",
                f"printf '.venv\\n' >> {shlex.quote(plan.remote_repo_path + '/.git/info/exclude')}",
            ]
        )
    lines.extend(
        [
            f"cd {shlex.quote(plan.remote_worktree_path)}",
            "git status --short --branch",
            "git rev-parse --short HEAD",
        ]
    )
    return "\n".join(lines)


def _run_remote_script(plan: SyncPlan) -> None:
    remote_script = _build_remote_script(plan)
    subprocess.run(
        ["ssh", "-p", str(plan.ssh_port), plan.remote, remote_script],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = _create_plan(args)
    if args.dry_run:
        print(f"repo_root={plan.repo_root}")
        print(f"branch={plan.branch}")
        print(f"head={plan.head}")
        print(f"bundle_path={plan.bundle_path}")
        print(f"bundle_url={plan.bundle_url}")
        print(f"bundle_source_url={plan.bundle_source_url}")
        print(f"clone_proxy={plan.clone_proxy}")
        print(f"incremental_base={plan.incremental_base}")
        print("--- remote script ---")
        print(_build_remote_script(plan))
        return 0

    _create_bundle(plan)
    server = _start_http_server(plan)
    try:
        _run_remote_script(plan)
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
