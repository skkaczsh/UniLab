#!/usr/bin/env python3
"""Fetch and verify the curated OmniCar checkpoint from its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "omni_car" / "axis_track_2999_checkpoint_manifest.json"
DEFAULT_CACHE_DIR = ROOT_DIR / "artifacts" / "omni_car" / "checkpoints"


@dataclass(frozen=True)
class CheckpointRef:
    task: str
    remote_host: str
    ssh_port: int
    remote_path: str
    checkpoint_name: str
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class LocalCheckpointStatus:
    path: Path
    exists: bool
    actual_bytes: int | None
    actual_sha256: str | None
    expected_bytes: int
    expected_sha256: str

    @property
    def byte_count_ok(self) -> bool:
        return self.actual_bytes == self.expected_bytes

    @property
    def sha256_ok(self) -> bool:
        return self.actual_sha256 == self.expected_sha256

    @property
    def ok(self) -> bool:
        return self.exists and self.byte_count_ok and self.sha256_ok


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer")
    return value


def load_checkpoint_ref(manifest_path: Path) -> CheckpointRef:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")

    remote = manifest.get("remote")
    run = manifest.get("run")
    if not isinstance(remote, dict):
        raise ValueError("checkpoint manifest must contain a remote object")
    if not isinstance(run, dict):
        raise ValueError("checkpoint manifest must contain a run object")

    return CheckpointRef(
        task=_require_str(manifest, "task"),
        remote_host=_require_str(remote, "host"),
        ssh_port=_require_int(remote, "ssh_port"),
        remote_path=_require_str(run, "checkpoint_path"),
        checkpoint_name=_require_str(run, "checkpoint"),
        expected_bytes=_require_int(run, "checkpoint_bytes"),
        expected_sha256=_require_str(run, "checkpoint_sha256"),
    )


def default_output_path(ref: CheckpointRef) -> Path:
    return DEFAULT_CACHE_DIR / ref.checkpoint_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_checkpoint_status(ref: CheckpointRef, path: Path) -> LocalCheckpointStatus:
    if not path.exists():
        return LocalCheckpointStatus(
            path=path,
            exists=False,
            actual_bytes=None,
            actual_sha256=None,
            expected_bytes=ref.expected_bytes,
            expected_sha256=ref.expected_sha256,
        )
    return LocalCheckpointStatus(
        path=path,
        exists=True,
        actual_bytes=path.stat().st_size,
        actual_sha256=sha256_file(path),
        expected_bytes=ref.expected_bytes,
        expected_sha256=ref.expected_sha256,
    )


def build_scp_command(ref: CheckpointRef, output_path: Path) -> list[str]:
    return [
        "scp",
        "-P",
        str(ref.ssh_port),
        f"{ref.remote_host}:{ref.remote_path}",
        str(output_path),
    ]


def format_status(status: LocalCheckpointStatus) -> str:
    lines = [f"path: {status.path}"]
    if not status.exists:
        lines.append("exists: false")
        lines.append(f"expected_bytes: {status.expected_bytes}")
        lines.append(f"expected_sha256: {status.expected_sha256}")
        return "\n".join(lines)
    lines.extend(
        [
            "exists: true",
            f"bytes: {status.actual_bytes}",
            f"bytes_ok: {str(status.byte_count_ok).lower()}",
            f"sha256: {status.actual_sha256}",
            f"sha256_ok: {str(status.sha256_ok).lower()}",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Checkpoint manifest JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Local checkpoint cache path. Defaults to artifacts/omni_car/checkpoints/<checkpoint>.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify the local cache.")
    parser.add_argument("--force", action="store_true", help="Refetch even when the local cache is valid.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned scp command and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.resolve()
    ref = load_checkpoint_ref(manifest_path)
    output_path = args.output.resolve() if args.output is not None else default_output_path(ref)
    command = build_scp_command(ref, output_path)

    if args.dry_run:
        print(f"manifest: {manifest_path}")
        print(f"output: {output_path}")
        print(f"scp: {shlex.join(command)}")
        return 0

    status = local_checkpoint_status(ref, output_path)
    if args.verify_only:
        print(format_status(status))
        return 0 if status.ok else 1

    if status.ok and not args.force:
        print(format_status(status))
        print("checkpoint cache is valid")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    status = local_checkpoint_status(ref, output_path)
    print(format_status(status))
    if not status.ok:
        return 1

    rel_output = output_path.relative_to(ROOT_DIR) if output_path.is_relative_to(ROOT_DIR) else output_path
    print("checkpoint cache is valid")
    print(f"viewer: uv run scripts/view_omni_car_checkpoint.py --load-run {rel_output} --device cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
