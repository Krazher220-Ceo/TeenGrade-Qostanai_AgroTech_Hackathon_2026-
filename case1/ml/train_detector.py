"""
Скрипт дообучения детектора сорняков и культур (YOLOv8s) на объединенном датасете.
Qostanai AgroTech Hackathon 2026 — AgroVision AI.
"""

import argparse
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE1_DIR = ROOT_DIR / "case1"
DEFAULT_YAML = CASE1_DIR / "configs" / "combined_detector.yaml"
AERIAL_YAML = CASE1_DIR / "configs" / "weed_crop_aerial.yaml"


def train_detector(
    data_yaml: str = str(DEFAULT_YAML),
    base_weights: str = "yolov8s.pt",
    output_dir: str = "case1/models",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 0,
    device: str = "auto"
):
    yaml_path = Path(data_yaml)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Автоматическая сборка объединенного датасета, если конфиг не существует
    if not yaml_path.exists():
        if yaml_path == DEFAULT_YAML:
            print(f"[INFO] Конфиг {yaml_path} не найден. Запуск сборки объединенного датасета детекции...")
            from case1.ml.build_detector_dataset import build_combined_dataset
            build_combined_dataset()
        elif AERIAL_YAML.exists():
            yaml_path = AERIAL_YAML
        else:
            print(f"[INFO] Файл {yaml_path} не найден. Запуск конвертации COCO -> YOLO...")
            from case1.ml.prepare_yolo_aerial import convert_coco_to_yolo
            convert_coco_to_yolo(config_yaml_path=str(yaml_path))

    # Определение устройства
    if device == "auto":
        if torch.cuda.is_available():
            dev_str = "0"
        elif torch.backends.mps.is_available():
            dev_str = "mps"
            batch_size = min(batch_size, 8)
        else:
            dev_str = "cpu"
            batch_size = min(batch_size, 4)
    else:
        dev_str = device

    print("================================================================================")
    print(f"ДООБУЧЕНИЕ ДЕТЕКТОРА СОРНЯКОВ (YOLOv8s, {epochs} ЭПОХ, Device: {dev_str}, Batch: {batch_size}, Workers: {workers})")
    print("================================================================================")
    print(f"Датасет конфиг: {yaml_path}")
    print(f"Базовая модель: {base_weights}")

    # Инициализация весов
    weedblaster_weights = ROOT_DIR / "weedblaster-vision-yolov8s" / "best.pt"
    if Path(base_weights).exists():
        print(f"Инициализация от локальных весов: {base_weights}")
        model = YOLO(base_weights)
    elif weedblaster_weights.exists():
        print(f"Инициализация от агро-весов WeedBlaster: {weedblaster_weights}")
        model = YOLO(str(weedblaster_weights))
    else:
        print("Загрузка официального чекпоинта YOLOv8s...")
        model = YOLO("yolov8s.pt")

    # Дообучение детектора
    # val=False, cache=False, plots=False предотвращают пики потребления RAM и OOM в контейнерах
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=dev_str,
        freeze=5,
        workers=workers,
        deterministic=False,
        project=str(out_path / "detector_runs"),
        name="weed_crop_detector",
        exist_ok=True,
        save=True,
        save_period=5,
        verbose=True,
        plots=False,
        val=False,
        cache=False,
    )

    # Поиск сохраненных весов (best.pt или last.pt)
    candidate_checkpoints = [
        out_path / "detector_runs" / "weed_crop_detector" / "weights" / "best.pt",
        out_path / "detector_runs" / "weed_crop_detector" / "weights" / "last.pt",
        ROOT_DIR / "runs" / "detect" / "case1" / "models" / "detector_runs" / "weed_crop_detector" / "weights" / "best.pt",
        ROOT_DIR / "runs" / "detect" / "case1" / "models" / "detector_runs" / "weed_crop_detector" / "weights" / "last.pt",
    ] + list(ROOT_DIR.rglob("weed_crop_detector/weights/*.pt"))

    target_checkpoint = out_path / "weed_detector_finetuned.pt"
    found_ckpt = None
    for ckpt in candidate_checkpoints:
        if ckpt.exists():
            found_ckpt = ckpt
            break

    if found_ckpt:
        shutil.copy2(found_ckpt, target_checkpoint)
        print(f"\n★ Финальный чекпоинт ({found_ckpt.name}) скопирован в: {target_checkpoint}")
    else:
        print("\n[!] Чекпоинт не найден в стандартных путях, проверка runs/detect...")

    # Освобождаем память от графов обучения и оптимизатора перед валидацией
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Валидация на отложенном тесте чистой моделью
    print("\n--- Финальная валидация на тестовом сплите (Test split) ---")
    val_model_path = target_checkpoint if target_checkpoint.exists() else "yolov8s.pt"
    eval_model = YOLO(str(val_model_path))
    val_metrics = eval_model.val(
        data=str(yaml_path.resolve()),
        split="test",
        plots=False,
        save_json=False,
        half=(dev_str != "cpu")
    )
    map50 = float(val_metrics.box.map50)
    map_all = float(val_metrics.box.map)
    print(f"  [+] mAP@50:     {map50:.4f}")
    print(f"  [+] mAP@50-95:  {map_all:.4f}")

    # Сохраняем метрики в JSON для дашборда
    metrics_report = {
        "model": "YOLOv8s (Crop vs Weed)",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": dev_str,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map_all, 4),
        "classes": {0: "crop", 1: "weed"}
    }
    metrics_out = CASE1_DIR / "output" / "detector_training_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены в: {metrics_out}")

    return eval_model, val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение детектора сорняков и культур YOLOv8")
    parser.add_argument("--data", type=str, default=str(DEFAULT_YAML), help="Путь к YAML конфигу датасета")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Базовые веса (yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча (16 для стабильности RAM)")
    parser.add_argument("--workers", type=int, default=0, help="Количество воркеров DataLoader (0 для Docker shm безопасности)")
    parser.add_argument("--imgsz", type=int, default=640, help="Разрешение тайлов (640)")
    parser.add_argument("--device", type=str, default="auto", help="auto, 0, cpu, mps")
    args = parser.parse_args()

    train_detector(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        imgsz=args.imgsz,
        device=args.device
    )
