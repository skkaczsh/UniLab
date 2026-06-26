from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_omni_car_env.py"
    spec = importlib.util.spec_from_file_location("benchmark_omni_car_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_reports_summary() -> None:
    bench = _load_benchmark_module()
    args = bench._parse_args(
        [
            "--num-envs",
            "4",
            "--steps",
            "4",
            "--warmup-steps",
            "1",
            "--obstacles",
            "3",
            "--action-mode",
            "command",
            "--profile-top",
            "5",
        ]
    )

    summary, profile_text = bench.run_benchmark(args)

    assert summary["num_envs"] == 4
    assert summary["steps"] == 4
    assert summary["obstacles"] == 3
    assert summary["action_mode"] == "command"
    assert float(summary["wall_time_s"]) >= 0.0
    assert float(summary["per_step_ms"]) >= 0.0
    assert float(summary["env_steps_per_second"]) >= 0.0
    assert profile_text is not None
    assert "omni_car.py" in profile_text
