#!/usr/bin/env python3
"""
WeedBlaster Vision Model — Training Script
YOLOv8s on CropAndWeed Dataset (CropsOrWeed9 variant)

Trains a YOLOv8-s model to detect 8 crop species + weed superclass
for use in an autonomous laser weeding robot.

Requirements:
    pip install ultralytics torch

Dataset:
    https://github.com/cropandweed/cropandweed-dataset
    Expected structure: datasets/yolo_cropandweed/{train,val,test}/{images,labels}/
"""

import os
import sys
import shutil
from pathlib import Path


def setup_dataset(dataset_root: Path) -> Path:
    """Verify CropAndWeed YOLO dataset exists and return dataset.yaml path."""
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml.absolute()}\n"
            f"Download from: https://github.com/cropandweed/cropandweed-dataset"
        )

    for split in ["train", "val"]:
        images = list((dataset_root / split / "images").glob("*.jpg"))
        print(f"  {split}: {len(images)} images")

    return dataset_yaml


def train(dataset_yaml: Path, save_dir: Path):
    """Train YOLOv8s with CropsOrWeed9 configuration."""
    import torch
    from ultralytics import YOLO

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data=str(dataset_yaml.absolute()),
        epochs=100,
        imgsz=1280,
        batch=8,
        device="cuda:0",
        project=str(save_dir),
        name="cropweed_yolov8s_v1.9",
        save=True,
        verbose=True,
        plots=True,
        save_period=10,
        amp=True,

        # Augmentation — tuned for agricultural top-down imagery
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,       # No rotation (preserve crop row orientation)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,   # No perspective (top-down views)
        flipud=0.0,        # No vertical flip (plants have orientation)
        fliplr=0.5,
        mosaic=1.0,        # Key for small weed instances
        mixup=0.1,
        copy_paste=0.1,    # Synthetic weed density variation

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,          # Cosine anneal to lr0 * lrf
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    # Validation
    val_results = model.val()
    print(f"\nmAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    class_names = ["Maize", "Sugar Beet", "Soy", "Sunflower",
                   "Potato", "Pea", "Bean", "Pumpkin", "Weed"]
    if hasattr(val_results.box, "maps"):
        for i, ap in enumerate(val_results.box.maps):
            if i < len(class_names):
                print(f"  {class_names[i]}: {ap:.4f}")

    # Export
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nBest weights: {best_weights}")

    return results


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    dataset_root = script_dir / "datasets" / "yolo_cropandweed"

    print("WeedBlaster Vision — YOLOv8s Training")
    print("Classes: Maize, Sugar Beet, Soy, Sunflower, Potato, Pea, Bean, Pumpkin, Weed")
    print()

    dataset_yaml = setup_dataset(dataset_root)
    save_dir = script_dir / "trained_models"
    save_dir.mkdir(exist_ok=True)

    train(dataset_yaml, save_dir)
