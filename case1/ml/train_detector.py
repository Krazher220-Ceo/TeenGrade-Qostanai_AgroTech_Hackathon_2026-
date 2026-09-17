"""
Скрипт дообучения детектора сорняков (YOLOv8s) на датасете аэрофотосъёмки weed-crop-aerial.
"""

import argparse
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

from case1.ml.prepare_yolo_aerial import convert_coco_to_yolo


def train_detector(
    data_yaml: str = "case1/configs/weed_crop_aerial.yaml",
    base_weights: str = "weedblaster-vision-yolov8s/best.pt",
    output_dir: str = "case1/models",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto"
):
    yaml_path = Path(data_yaml)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Проверка наличия данных, при необходимости автоматическая конвертация
    if not yaml_path.exists():
        print(f"[INFO] Файл {yaml_path} не найден. Запуск конвертации COCO -> YOLO...")
        convert_coco_to_yolo(config_yaml_path=str(yaml_path))

    if device == "auto":
        if torch.backends.mps.is_available():
            dev_str = "mps"
        elif torch.cuda.is_available():
            dev_str = "0"
        else:
            dev_str = "cpu"
    else:
        dev_str = device

    print(f"=== ДООБУЧЕНИЕ ДЕТЕКТОРА СОРНЯКОВ (YOLOv8s, {epochs} ЭПОХ, Device: {dev_str}) ===")

    # Базовая модель
    if Path(base_weights).exists():
        print(f"Инициализация от предобученных агро-весов: {base_weights}")
        model = YOLO(base_weights)
    else:
        print("Базовые веса не найдены, загрузка стандартной yolov8s.pt...")
        model = YOLO("yolov8s.pt")

    # Дообучение с заморозкой backbone (freeze=10) для быстрого трансферного обучения
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=dev_str,
        freeze=10,
        deterministic=False,
        project=str(out_path / "detector_runs"),
        name="weed_finetuned",
        exist_ok=True,
        save=True,
        verbose=True,
        plots=True
    )

    best_checkpoint = out_path / "detector_runs" / "weed_finetuned" / "weights" / "best.pt"
    target_checkpoint = out_path / "weed_detector_finetuned.pt"

    if best_checkpoint.exists():
        shutil.copy2(best_checkpoint, target_checkpoint)
        print(f"\n★ Лучшая дообученная модель сохранена в {target_checkpoint}")
    else:
        last_checkpoint = out_path / "detector_runs" / "weed_finetuned" / "weights" / "last.pt"
        if last_checkpoint.exists():
            shutil.copy2(last_checkpoint, target_checkpoint)
            print(f"\n★ Модель сохранена в {target_checkpoint}")

    # Валидация
    print("\nВалидация дообученной модели:")
    val_metrics = model.val(data=str(yaml_path.resolve()), split="test")
    print(f"mAP@50: {val_metrics.box.map50:.4f}")
    print(f"mAP@50-95: {val_metrics.box.map:.4f}")

    return model, val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detector on aerial weed imagery")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    train_detector(epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz, device=args.device)
