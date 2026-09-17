"""
Бенчмарк задержки конвейера (Latency Benchmark) и расчет кинематики опрыскивателя
согласно агрономической шпаргалке ментора (Qostanai AgroTech Hackathon 2026).

Шпаргалка ментора:
- Скорость движения опрыскивателя: 18–20 км/ч (5,0–5,56 м/с).
- Важен не просто FPS, а полная задержка от захвата кадра до привязанного к координате решения
  и расстояние, пройденное штангой опрыскивателя за время этой задержки.
- При задержке > 50–100 мс система честно маркируется как Post-flight Advisory Mapping
  (предварительное составление карты дифференцированного внесения перед выездом техники).
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.ml.multitask_model import WeedMultiTaskModel, infer_num_species_from_state_dict
from case1.fleet.agronomy_rules import AgronomyRuleEngine

OUTPUT_REPORT = ROOT_DIR / "case1" / "output" / "latency_benchmark_report.json"
MODEL_PATH = ROOT_DIR / "case1" / "models" / "multitask_weeds_best.pt"


def get_hardware_environment() -> Dict[str, Any]:
    """Сбор честной информации об оборудовании и программном стеке."""
    env_info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
    }
    if torch.backends.mps.is_available():
        env_info["device_used"] = "mps"
    elif torch.cuda.is_available():
        env_info["device_used"] = "cuda"
    return env_info


def calculate_travelled_distance(latency_s: float) -> Dict[str, float]:
    """
    Расчет расстояния, пройденного опрыскивателем за время задержки:
    - 18 км/ч = 5.0 м/с
    - 20 км/ч = 5.56 м/с
    """
    v18_m_s = 5.0
    v20_m_s = 5.56
    return {
        "latency_s": round(latency_s, 5),
        "speed_18kmh_m_s": v18_m_s,
        "displacement_18kmh_m": round(v18_m_s * latency_s, 4),
        "speed_20kmh_m_s": v20_m_s,
        "displacement_20kmh_m": round(v20_m_s * latency_s, 4),
    }


def run_latency_benchmark(
    iterations: int = 40,
    output_file: Optional[Path] = None,
    device_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Выполнение замера задержек компонентов конвейера:
    1. Preprocessing (Crop resize + normalisation)
    2. Classifier inference (EfficientNet-B0 forward pass)
    3. Agronomy rule evaluation (EPV thresholding + dosage recommendation)
    4. End-to-end per-crop decision latency
    """
    env = get_hardware_environment()
    target_device = torch.device(device_override or env["device_used"])

    # Загрузка классификатора
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена для бенчмарка")

    state_dict = torch.load(MODEL_PATH, map_location=target_device)
    num_species = infer_num_species_from_state_dict(state_dict)
    model = WeedMultiTaskModel(num_species=num_species, num_stages=2, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rule_engine = AgronomyRuleEngine()

    # Создание синтетического тестового изображения для воспроизводимого бенчмарка
    dummy_img = Image.new("RGB", (320, 320), color=(80, 140, 60))

    # Прогрев (Warm-up)
    for _ in range(5):
        t = transform(dummy_img).unsqueeze(0).to(target_device)
        _ = model.predict_crop(t)
        _ = rule_engine.evaluate_weed_patch(annual_density_per_m2=8.0)

    preprocess_times = []
    inference_times = []
    rules_times = []
    total_times = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        tensor_crop = transform(dummy_img).unsqueeze(0).to(target_device)
        t1 = time.perf_counter()

        pred = model.predict_crop(tensor_crop)
        t2 = time.perf_counter()

        _ = rule_engine.evaluate_weed_patch(
            annual_density_per_m2=7.5,
            perennial_density_per_m2=1.0,
            growth_stage="cotyledon_to_2_leaves",
        )
        t3 = time.perf_counter()

        preprocess_times.append(t1 - t0)
        inference_times.append(t2 - t1)
        rules_times.append(t3 - t2)
        total_times.append(t3 - t0)

    def calc_percentiles(arr: List[float]) -> Dict[str, float]:
        np_arr = np.array(arr) * 1000.0  # в миллисекундах
        return {
            "p50_ms": round(float(np.percentile(np_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(np_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(np_arr, 99)), 2),
            "mean_ms": round(float(np.mean(np_arr)), 2),
        }

    prep_stats = calc_percentiles(preprocess_times)
    inf_stats = calc_percentiles(inference_times)
    rule_stats = calc_percentiles(rules_times)
    total_stats = calc_percentiles(total_times)

    median_total_s = total_stats["p50_ms"] / 1000.0
    p95_total_s = total_stats["p95_ms"] / 1000.0

    kinematics_p50 = calculate_travelled_distance(median_total_s)
    kinematics_p95 = calculate_travelled_distance(p95_total_s)

    # Инженерный вывод
    target_inline_latency_ms = 50.0  # порог для точечного распылителя на ходу
    if total_stats["p95_ms"] <= target_inline_latency_ms:
        operational_mode = "Real-time Edge Spraying Capable"
        operational_mode_ru = "Пригодно для управления форсункой на ходу"
    else:
        operational_mode = "Post-flight Advisory Mapping"
        operational_mode_ru = "Режим предварительного картирования поля (Post-flight Advisory Mapping)"

    recommendation_text = (
        f"При скорости штанги опрыскивателя 18–20 км/ч (5,0–5,56 м/с) задержка конвейера "
        f"p50={total_stats['p50_ms']} мс и p95={total_stats['p95_ms']} мс приводит к смещению "
        f"точки срабатывания на {kinematics_p50['displacement_18kmh_m']*100:.1f}–"
        f"{kinematics_p95['displacement_20kmh_m']*100:.1f} см. "
        f"Поэтому надёжный сценарий внедрения MVP — предварительный облет поля БПЛА и загрузка "
        f"карты дифференцированного опрыскивания в бортовой терминал опрыскивателя перед обработкой."
    )

    report = {
        "benchmark_title": "AgroVision AI — Latency & Kinematics Benchmark",
        "cheatsheet_reference": "Шпаргалка ментора: скорость опрыскивателя 18–20 км/ч, задержка пайплайна",
        "iterations": iterations,
        "environment": env,
        "latency_breakdown": {
            "preprocessing": prep_stats,
            "classifier_inference": inf_stats,
            "agronomy_rules": rule_stats,
            "total_per_crop": total_stats,
        },
        "sprayer_kinematics": {
            "p50_median": kinematics_p50,
            "p95_high_load": kinematics_p95,
        },
        "operational_mode": operational_mode,
        "operational_mode_ru": operational_mode_ru,
        "engineering_recommendation": recommendation_text,
    }

    dest = output_file or OUTPUT_REPORT
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    rep = run_latency_benchmark()
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА ЗАДЕРЖКИ (18-20 КМ/Ч ОПРЫСКИВАТЕЛЬ)")
    print("=" * 80)
    print(f"Устройство: {rep['environment']['device_used']} ({rep['environment']['processor']})")
    print(f"Полная задержка (p50): {rep['latency_breakdown']['total_per_crop']['p50_ms']} мс")
    print(f"Полная задержка (p95): {rep['latency_breakdown']['total_per_crop']['p95_ms']} мс")
    print(f"Смещение при 18 км/ч:   {rep['sprayer_kinematics']['p50_median']['displacement_18kmh_m']} м")
    print(f"Смещение при 20 км/ч:   {rep['sprayer_kinematics']['p50_median']['displacement_20kmh_m']} м")
    print(f"Режим применения:       {rep['operational_mode_ru']}")
    print("=" * 80)
