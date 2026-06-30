from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "distill_omni_car_transformer_teacher.py"
    )
    spec = importlib.util.spec_from_file_location("distill_omni_car_transformer_teacher", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ConstantPolicy(torch.nn.Module):
    def __init__(self, value: tuple[float, float, float]) -> None:
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, obs):  # type: ignore[no-untyped-def]
        batch = obs["actor"].shape[0]
        return self.action.reshape(1, 3).expand(batch, 3)


class _WrappedEnv:
    def __init__(self) -> None:
        self.obs = TensorDict({"actor": torch.zeros((2, 8), dtype=torch.float32)}, batch_size=2)

    def reset(self):  # type: ignore[no-untyped-def]
        return self.obs, {}

    def step(self, _action):  # type: ignore[no-untyped-def]
        return (
            self.obs,
            torch.zeros((2,), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.bool),
            {},
        )


def _fake_context(module, *, student_value=(0.0, 0.0, 0.0), teacher_value=(0.2, -0.1, 0.05)):
    student_policy = _ConstantPolicy(student_value)
    teacher_policy = _ConstantPolicy(teacher_value)
    student_runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: student_policy))
    teacher_runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: teacher_policy),
        get_inference_policy=lambda device=None: teacher_policy,
    )
    env = SimpleNamespace(_grid_history_len=1, _grid_dim=1, close=lambda: None)
    return module.DistillContext(
        student_runner=student_runner,
        teacher_runner=teacher_runner,
        env=env,
        wrapped_env=_WrappedEnv(),
        student_load_path=Path("/tmp/student.pt"),
        teacher_load_path=Path("/tmp/teacher.pt"),
        device="cpu",
    )


def test_distill_weighted_loss_applies_axis_weights() -> None:
    module = _load_module()
    prediction = torch.tensor([[1.0, 1.0, 1.0]])
    target = torch.zeros((1, 3))
    weights = torch.tensor([[1.0, 2.0, 3.0]])

    loss = module._weighted_loss(prediction, target, axis_weights=weights, kind="mse")

    assert loss.item() == 2.0


def test_distill_dry_run_reports_teacher_and_student_shapes(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_make_context", lambda _args: _fake_context(module))

    summary = module.distill(
        module._parse_args(
            [
                "--teacher-load-run",
                "/tmp/teacher.pt",
                "--student-load-run",
                "/tmp/student.pt",
                "--output",
                str(tmp_path / "student_distilled.pt"),
                "--dry-run",
            ]
        )
    )

    assert summary["status"] == "dry_run"
    assert summary["teacher_action_shape"] == [2, 3]
    assert summary["student_action_shape"] == [2, 3]


def test_distill_trains_student_and_saves_checkpoint(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    context = _fake_context(module)
    output = tmp_path / "distilled.pt"
    saved: dict[str, Path] = {}
    monkeypatch.setattr(module, "_make_context", lambda _args: context)
    monkeypatch.setattr(
        module,
        "_save_corrected_checkpoint",
        lambda _runner, path: (Path(path).touch(), saved.setdefault("path", Path(path))),
    )

    summary = module.distill(
        module._parse_args(
            [
                "--teacher-load-run",
                "/tmp/teacher.pt",
                "--student-load-run",
                "/tmp/student.pt",
                "--output",
                str(output),
                "--iterations",
                "3",
                "--learning-rate",
                "0.01",
                "--progress-interval",
                "0",
            ]
        )
    )

    assert summary["status"] == "completed"
    assert summary["loss_final"] is not None
    assert saved["path"] == output
    assert output.exists()
