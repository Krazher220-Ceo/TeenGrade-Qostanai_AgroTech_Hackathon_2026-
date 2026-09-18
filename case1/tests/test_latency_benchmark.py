"""
Тесты бенчмарка задержки конвейера и кинематики опрыскивателя
согласно шпаргалке ментора.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.ml.latency_benchmark import (
    get_hardware_environment,
    calculate_travelled_distance,
    run_latency_benchmark,
)


def test_hardware_environment():
    env = get_hardware_environment()
    assert "platform" in env
    assert "python_version" in env
    assert "torch_version" in env
    assert env["device_used"] in {"cpu", "mps", "cuda"}


def test_calculate_travelled_distance():
    # 50 мс задержка
    res_50ms = calculate_travelled_distance(0.050)
    assert res_50ms["displacement_18kmh_m"] == 0.25  # 5.0 * 0.05 = 0.25 м
    assert res_50ms["displacement_20kmh_m"] == 0.278  # 5.56 * 0.05 = 0.278 м

    # 100 мс задержка
    res_100ms = calculate_travelled_distance(0.100)
    assert res_100ms["displacement_18kmh_m"] == 0.50
    assert res_100ms["displacement_20kmh_m"] == 0.556


def test_run_latency_benchmark_smoke(tmp_path):
    report_file = tmp_path / "benchmark_report.json"
    report = run_latency_benchmark(iterations=10, output_file=report_file)

    assert report_file.exists()
    assert "latency_breakdown" in report
    assert "total_per_crop" in report["latency_breakdown"]
    assert "p50_ms" in report["latency_breakdown"]["total_per_crop"]
    assert "sprayer_kinematics" in report
    assert report["operational_mode"] in {
        "Real-time Edge Spraying Capable",
        "Post-flight Advisory Mapping",
    }
