from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "fetch_omni_car_checkpoint.py"
    spec = importlib.util.spec_from_file_location("fetch_omni_car_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, payload: bytes) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    manifest: dict[str, object] = {
        "task": "OmniCarGridAvoidance",
        "remote": {
            "host": "zsh@skkac.top",
            "ssh_port": 6010,
            "worktree": "/home/zsh/develop/worktrees/UniLab-omni-car-git",
        },
        "run": {
            "checkpoint": "model_2999.pt",
            "checkpoint_path": "/remote/run/model_2999.pt",
            "checkpoint_bytes": len(payload),
            "checkpoint_sha256": digest,
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_manifest_builds_checkpoint_ref_and_scp_command(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, b"checkpoint")

    ref = module.load_checkpoint_ref(manifest_path)
    command = module.build_scp_command(ref, tmp_path / "model_2999.pt")

    assert ref.task == "OmniCarGridAvoidance"
    assert ref.remote_host == "zsh@skkac.top"
    assert ref.ssh_port == 6010
    assert ref.remote_path == "/remote/run/model_2999.pt"
    assert command == [
        "scp",
        "-P",
        "6010",
        "zsh@skkac.top:/remote/run/model_2999.pt",
        str(tmp_path / "model_2999.pt"),
    ]


def test_default_manifest_points_to_current_large_scene_checkpoint() -> None:
    module = _load_module()

    assert module.DEFAULT_MANIFEST.name == "remote_large_scene_c22_2900_checkpoint_manifest.json"


def test_tracked_checkpoint_manifests_use_standard_commands() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    manifest_paths = sorted((root / "artifacts" / "omni_car").glob("*checkpoint_manifest.json"))

    assert manifest_paths
    seen_cache_names: set[str] = set()
    for manifest_path in manifest_paths:
        ref = module.load_checkpoint_ref(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        commands = manifest["commands"]
        cache_name = ref.local_cache_name or ref.checkpoint_name

        assert cache_name not in seen_cache_names
        seen_cache_names.add(cache_name)
        assert ref.task == "OmniCarGridAvoidance"
        assert ref.remote_host == "zsh@skkac.top"
        assert ref.ssh_port == 6010
        assert ref.remote_path.endswith(f"/{ref.checkpoint_name}")
        assert re.fullmatch(r"[0-9a-f]{64}", ref.expected_sha256)
        assert ref.expected_bytes > 0
        assert commands["local_fetch_checkpoint"].startswith(
            "uv run scripts/fetch_omni_car_checkpoint.py --manifest "
        )
        assert commands["local_verify_checkpoint"].startswith(
            "uv run scripts/fetch_omni_car_checkpoint.py --manifest "
        )
        assert " --verify-only" in commands["local_verify_checkpoint"]
        assert "uv run scripts/evaluate_omni_car_checkpoint.py" in commands["remote_eval_seed_7"]
        for command in commands.values():
            assert "uv run python scripts/" not in command
            assert "shasum" not in command
            assert "mkdir -p artifacts/omni_car/checkpoints" not in command


def test_local_checkpoint_status_validates_size_and_hash(tmp_path: Path) -> None:
    module = _load_module()
    payload = b"checkpoint"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, payload)
    checkpoint_path = tmp_path / "model_2999.pt"
    checkpoint_path.write_bytes(payload)

    status = module.local_checkpoint_status(
        module.load_checkpoint_ref(manifest_path), checkpoint_path
    )

    assert status.ok
    assert status.byte_count_ok
    assert status.sha256_ok
    assert "sha256_ok: true" in module.format_status(status)


def test_manifest_can_override_default_cache_name(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "manifest.json"
    manifest = _write_manifest(manifest_path, b"checkpoint")
    manifest["run"]["local_cache_name"] = "large_scene_model_2900.pt"  # type: ignore[index]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ref = module.load_checkpoint_ref(manifest_path)

    assert ref.local_cache_name == "large_scene_model_2900.pt"
    assert module.default_output_path(ref).name == "large_scene_model_2900.pt"


def test_verify_only_returns_nonzero_for_missing_cache(tmp_path: Path, capsys) -> None:
    module = _load_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, b"checkpoint")

    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "missing.pt"),
            "--verify-only",
        ]
    )

    assert rc == 1
    assert "exists: false" in capsys.readouterr().out


def test_main_skips_fetch_when_cache_is_valid(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    payload = b"checkpoint"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, payload)
    checkpoint_path = tmp_path / "model_2999.pt"
    checkpoint_path.write_bytes(payload)

    def _fail_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("scp should not run for a valid cache")

    monkeypatch.setattr(module.subprocess, "run", _fail_run)

    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(checkpoint_path),
        ]
    )

    assert rc == 0
    assert "checkpoint cache is valid" in capsys.readouterr().out
