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
    MODEL_PATH,
)


def _is_lfs_pointer(path: Path) -> bool:
    """True when `path` is a small Git LFS pointer text file rather than the
    real binary (e.g. a checkout with `git lfs pull` skipped / `lfs: false`
    in CI). Used to skip weight-dependent tests instead of failing them."""
    if not path.exists():
        return True
    try:
        if path.stat().st_size >= 1024 * 1024:
            return False
        return path.read_bytes()[:200].startswith(b"version https://git-lfs")
    except OSError:
        return True


requires_classifier_weights = pytest.mark.skipif(
    _is_lfs_pointer(MODEL_PATH),
    reason=(
        "Веса классификатора не загружены (обнаружен Git LFS pointer вместо бинарного файла). "
        "Выполните `git lfs pull` (или `make lfs-pull`), чтобы запустить этот тест."
    ),
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


@requires_classifier_weights
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
