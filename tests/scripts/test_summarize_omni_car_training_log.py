from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "summarize_omni_car_training_log.py"
    spec = importlib.util.spec_from_file_location("summarize_omni_car_training_log", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_training_log_extracts_iterations_and_metrics() -> None:
    module = load_module()
    text = """
################################################################################
                            Learning iteration 7/3000

                            Total steps: 32768
                       Steps per second: 5700
                omni_car/collision_rate: 0.0027
                omni_car/tracking_error: 0.8774
--------------------------------------------------------------------------------
################################################################################
                          Learning iteration 8/3000

                            Total steps: 36864
                       Steps per second: 5861
                omni_car/collision_rate: 0.0015
                omni_car/tracking_error: 0.8280
--------------------------------------------------------------------------------
"""

    iterations = module.parse_training_log(text)

    assert [item["iteration"] for item in iterations] == [7, 8]
    assert iterations[0]["max_iteration"] == 3000
    assert iterations[1]["metrics"]["Total steps"] == 36864
    assert iterations[1]["metrics"]["omni_car/tracking_error"] == 0.8280


def test_summarize_training_log_reports_latest_best_and_window(tmp_path: Path) -> None:
    module = load_module()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        """
Learning iteration 1/4
omni_car/tracking_error: 0.9
omni_car/collision_rate: 0.10
Mean reward: -10.0
Learning iteration 2/4
omni_car/tracking_error: 0.7
omni_car/collision_rate: 0.20
Mean reward: 5.0
Learning iteration 3/4
omni_car/tracking_error: 0.8
omni_car/collision_rate: 0.05
Mean reward: 2.0
""",
        encoding="utf-8",
    )

    summary = module.summarize_training_log(log_path, window=2)

    assert summary["iteration_count"] == 3
    assert summary["latest"]["iteration"] == 3
    assert summary["progress_fraction"] == 0.75
    assert summary["best"]["omni_car/tracking_error"]["min"] == {
        "iteration": 2,
        "value": 0.7,
    }
    assert summary["best"]["Mean reward"]["max"] == {"iteration": 2, "value": 5.0}
    assert summary["window"]["size"] == 2
    assert summary["window"]["mean"]["omni_car/collision_rate"] == 0.125
