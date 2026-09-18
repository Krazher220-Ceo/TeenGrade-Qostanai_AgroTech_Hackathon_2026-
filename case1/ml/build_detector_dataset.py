#!/usr/bin/env python3
"""
Генератор объединенного датасета детекции сорняков и культурных растений (YOLOv8).
Qostanai AgroTech Hackathon 2026 — AgroVision AI.

Классы детектора:
  0: crop  (Культурные растения: пшеница, подсолнечник, лен и др.)
  1: weed  (Сорняки всех 26 видов)

Источники данных:
  1. weed_crop_aerial (1 176 кадров БПЛА, классы crop и weed)
  2. crop_weed_detection (1 300 кадров с поля с боксами)
  3. Негативные фоновые тайлы (Hard Negatives): чистая почва и рядки пшеницы из /data/ФотоПолей (пустые метки .txt)
  4. Синтетические тайлы БПЛА (Copy-Paste): 26 видов сорняков из /data/Сорняки, наложенные на реальную почву Костаная
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE1_DIR = ROOT_DIR / "case1"
DATA_DIR = CASE1_DIR / "data"
CONFIGS_DIR = CASE1_DIR / "configs"

# Автодетект каталогов данных (локально или сервер /data)
if Path("/data").exists() and (Path("/data/Сорняки").exists() or Path("/data/ФотоПолей").exists()):
    SERVER_DATA = Path("/data")
    FIELD_DIR = SERVER_DATA / "ФотоПолей"
    WEEDS_DIR = SERVER_DATA / "Сорняки"
else:
    LOCAL_DATA = ROOT_DIR / "Dataset 1 кейс"
    FIELD_DIR = LOCAL_DATA / "ФотоПолей"
    WEEDS_DIR = LOCAL_DATA / "Сорняки"

YOLO_AERIAL_DIR = DATA_DIR / "yolo_aerial"
YOLO_FIELD_DIR = DATA_DIR / "yolo_field"
COMBINED_OUT_DIR = DATA_DIR / "yolo_combined"
COMBINED_YAML_PATH = CONFIGS_DIR / "combined_detector.yaml"


def extract_weed_cutout(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Извлечение зеленой растительной розетки/куста с помощью Excess Green Index (ExG):
    ExG = 2*G - R - B.
    Возвращает (RGBA вырезку растения, маску) или None.
    """
    h, w = img_bgr.shape[:2]
    if h < 40 or w < 40:
        return None

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b

    # Порог растительности
    thresh = np.percentile(exg, 65)
    thresh = max(10.0, float(thresh))
    mask = (exg > thresh).astype(np.uint8) * 255

    # Морфологическая фильтрация шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_cnt)
    if area < 400:
        return None

    x, y, cw, ch = cv2.boundingRect(max_cnt)
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    b_c, g_c, r_c = cv2.split(crop_bgr)
    crop_rgba = cv2.merge([b_c, g_c, r_c, crop_mask])

    return crop_rgba, crop_mask


def build_combined_dataset(
    output_dir: Path = COMBINED_OUT_DIR,
    yaml_path: Path = COMBINED_YAML_PATH,
    num_bg_tiles: int = 200,
    num_synth_tiles: int = 400,
    include_existing_yolo: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Генерация единого обучающего датасета YOLOv8 для детекции сорняков и культур."""
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    for s in splits:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "crop_boxes": 0,
        "weed_boxes": 0,
        "background_images": 0,
        "synth_tiles": 0,
        "sources": {}
    }

    print("================================================================================")
    print("ФОРМИРОВАНИЕ ОБЪЕДИНЕННОГО ДАТАСЕТА ДЕТЕКЦИИ (CROP vs WEED) ДЛЯ YOLOV8")
    print("================================================================================")
    print(f"Каталог назначения: {output_dir}")
    print(f"Каталог ФотоПолей:  {FIELD_DIR} (существует: {FIELD_DIR.exists()})")
    print(f"Каталог Сорняков:   {WEEDS_DIR} (существует: {WEEDS_DIR.exists()})")

    # 1. КОПИРОВАНИЕ / ИНТЕГРАЦИЯ СУЩЕСТВУЮЩИХ YOLO ВЫБОРОК (aerial + field)
    existing_yolo_dirs = []
    if include_existing_yolo:
        if YOLO_AERIAL_DIR.exists() and (YOLO_AERIAL_DIR / "images").exists():
            existing_yolo_dirs.append(("weed_crop_aerial", YOLO_AERIAL_DIR))
        if YOLO_FIELD_DIR.exists() and (YOLO_FIELD_DIR / "images").exists():
            existing_yolo_dirs.append(("crop_weed_field", YOLO_FIELD_DIR))

    for src_name, src_dir in existing_yolo_dirs:
        print(f"\n[+] Интеграция выборки {src_name} из {src_dir}...")
        src_img_count = 0
        src_weed_cnt = 0
        src_crop_cnt = 0

        for split in splits:
            img_split_dir = src_dir / "images" / split
            lbl_split_dir = src_dir / "labels" / split
            if not img_split_dir.exists():
                continue

            target_img_dir = output_dir / "images" / split
            target_lbl_dir = output_dir / "labels" / split

            for img_file in img_split_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                new_stem = f"{src_name}_{split}_{img_file.stem}"
                lbl_file = lbl_split_dir / f"{img_file.stem}.txt"

                target_img_file = target_img_dir / f"{new_stem}{img_file.suffix}"
                target_lbl_file = target_lbl_dir / f"{new_stem}.txt"

                if not target_img_file.exists():
                    shutil.copy2(img_file, target_img_file)

                lines_valid = []
                if lbl_file.exists():
                    with open(lbl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                if cls_id in (0, 1):
                                    lines_valid.append(f"{cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                                    if cls_id == 0:
                                        src_crop_cnt += 1
                                    else:
                                        src_weed_cnt += 1

                with open(target_lbl_file, "w", encoding="utf-8") as f:
                    for lv in lines_valid:
                        f.write(f"{lv}\n")

                src_img_count += 1
                stats[f"{split}_images"] += 1
                stats["total_images"] += 1

        stats["crop_boxes"] += src_crop_cnt
        stats["weed_boxes"] += src_weed_cnt
        stats["sources"][src_name] = {
            "images": src_img_count,
            "crop_boxes": src_crop_cnt,
            "weed_boxes": src_weed_cnt,
        }
        print(f"    Кадров: {src_img_count} | Crop: {src_crop_cnt} | Weed: {src_weed_cnt}")

    # 2. НАРЕЗКА НЕГАТИВНЫХ ТАЙЛОВ ФОНА ИЗ ФОТО ПОЛЕЙ (Hard Negatives)
    field_photos = []
    if FIELD_DIR.exists():
        field_photos = list(FIELD_DIR.rglob("*.[jJ][pP][gG]"))

    if field_photos and num_bg_tiles > 0:
        print(f"\n[+] Нарезка {num_bg_tiles} негативных тайлов почвы/рядков из {len(field_photos)} полевых кадров...")
        bg_created = 0
        tile_size = 640
        random.shuffle(field_photos)

        for p in field_photos:
            if bg_created >= num_bg_tiles:
                break
            try:
                img_cv = cv2.imread(str(p))
                if img_cv is None:
                    continue
                ih, iw = img_cv.shape[:2]
                if ih < tile_size + 100 or iw < tile_size + 100:
                    continue

                for _ in range(3):
                    if bg_created >= num_bg_tiles:
                        break
                    rx = random.randint(50, iw - tile_size - 50)
                    ry = random.randint(50, ih - tile_size - 50)
                    tile = img_cv[ry:ry+tile_size, rx:rx+tile_size]

                    mean_lum = float(np.mean(tile))
                    if mean_lum < 30 or mean_lum > 225:
                        continue

                    r_val = random.random()
                    split = "train" if r_val < 0.70 else ("val" if r_val < 0.85 else "test")

                    bg_stem = f"bg_soil_{p.stem}_{bg_created:04d}"
                    out_img_p = output_dir / "images" / split / f"{bg_stem}.jpg"
                    out_lbl_p = output_dir / "labels" / split / f"{bg_stem}.txt"

                    cv2.imwrite(str(out_img_p), tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    out_lbl_p.touch()

                    bg_created += 1
                    stats[f"{split}_images"] += 1
                    stats["total_images"] += 1
            except Exception:
                continue

        stats["background_images"] = bg_created
        stats["sources"]["hard_negatives_soil"] = {"images": bg_created, "boxes": 0}
        print(f"    Успешно создано {bg_created} негативных тайлов (пустые метки для подавления FP).")

    # 3. СИНТЕЗ ТАЙЛОВ БПЛА С 26 ВИДАМИ СОРНЯКОВ (Copy-Paste Augmentation)
    weed_photos = []
    if WEEDS_DIR.exists():
        weed_photos = list(WEEDS_DIR.rglob("*.[jJ][pP][gG]"))

    if weed_photos and field_photos and num_synth_tiles > 0:
        print(f"\n[+] Генерация {num_synth_tiles} синтетических тайлов БПЛА с 26 видами сорняков...")
        synth_created = 0
        tile_size = 640
        synth_weeds_cnt = 0

        weed_cutouts = []
        random.shuffle(weed_photos)
        print("    Извлечение растительных розеток сорняков через вегетационный индекс ExG...")
        for wp in weed_photos[:400]:
            try:
                w_bgr = cv2.imread(str(wp))
                if w_bgr is None:
                    continue
                extracted = extract_weed_cutout(w_bgr)
                if extracted is not None:
                    weed_cutouts.append((extracted[0], wp.parent.parent.name))
                    if len(weed_cutouts) >= 150:
                        break
            except Exception:
                continue

        print(f"    Подготовлено {len(weed_cutouts)} эталонных розеток сорняков разных видов.")

        if weed_cutouts:
            for i in range(num_synth_tiles):
                bg_photo = random.choice(field_photos)
                img_bg = cv2.imread(str(bg_photo))
                if img_bg is None:
                    continue
                ih, iw = img_bg.shape[:2]
                if ih < tile_size or iw < tile_size:
                    continue

                rx = random.randint(0, iw - tile_size)
                ry = random.randint(0, ih - tile_size)
                tile_bg = img_bg[ry:ry+tile_size, rx:rx+tile_size].copy()

                num_weeds_in_tile = random.randint(1, 4)
                boxes = []

                for _ in range(num_weeds_in_tile):
                    w_rgba, sp_name = random.choice(weed_cutouts)
                    wh, ww = w_rgba.shape[:2]

                    # Мультимасштабное распределение: от мелких всходов (45px) до крупных розеток (380px)
                    target_w = random.randint(45, 380)
                    scale = target_w / float(ww)
                    target_h = max(30, int(wh * scale))

                    resized_rgba = cv2.resize(w_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    angle = random.randint(0, 360)
                    center = (target_w // 2, target_h // 2)
                    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_rgba = cv2.warpAffine(resized_rgba, rot_mat, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                    px = random.randint(10, tile_size - target_w - 10)
                    py = random.randint(10, tile_size - target_h - 10)

                    alpha = rotated_rgba[:, :, 3].astype(float) / 255.0
                    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

                    roi = tile_bg[py:py+target_h, px:px+target_w]
                    weed_rgb = rotated_rgba[:, :, :3]

                    tile_bg[py:py+target_h, px:px+target_w] = (weed_rgb * alpha_3d + roi * (1.0 - alpha_3d)).astype(np.uint8)

                    x_center = (px + target_w / 2.0) / float(tile_size)
                    y_center = (py + target_h / 2.0) / float(tile_size)
                    norm_w = target_w / float(tile_size)
                    norm_h = target_h / float(tile_size)

                    boxes.append(f"1 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    synth_weeds_cnt += 1

                r_val = random.random()
                split = "train" if r_val < 0.70 else ("val" if r_val < 0.85 else "test")

                synth_stem = f"synth_weed_{i:04d}"
                out_img_p = output_dir / "images" / split / f"{synth_stem}.jpg"
                out_lbl_p = output_dir / "labels" / split / f"{synth_stem}.txt"

                cv2.imwrite(str(out_img_p), tile_bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(out_lbl_p, "w", encoding="utf-8") as f:
                    for b in boxes:
                        f.write(f"{b}\n")

                synth_created += 1
                stats[f"{split}_images"] += 1
                stats["total_images"] += 1

        stats["synth_tiles"] = synth_created
        stats["weed_boxes"] += synth_weeds_cnt
        stats["sources"]["synthetic_weeds_copy_paste"] = {
            "images": synth_created,
            "weed_boxes": synth_weeds_cnt,
        }
        print(f"    Создано {synth_created} синтетических тайлов БПЛА ({synth_weeds_cnt} сорняков).")

    # 4. СОЗДАНИЕ ДАТАСЕТ-КОНФИГА YOLOV8 (.YAML)
    yaml_content = f"""# Датасет детекции сорняков и культурных растений (Qostanai AgroTech 2026)
path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: crop
  1: weed
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[+] Сформирован конфиг детектора: {yaml_path}")
    print(f"=== ИТОГОВАЯ СТАТИСТИКА ДАТАСЕТА ===")
    print(f"  Всего изображений:     {stats['total_images']:,}")
    print(f"    - Train split:       {stats['train_images']:,} ({stats['train_images']/max(1, stats['total_images'])*100:.1f}%)")
    print(f"    - Val split:         {stats['val_images']:,} ({stats['val_images']/max(1, stats['total_images'])*100:.1f}%)")
    print(f"    - Test split:        {stats['test_images']:,} ({stats['test_images']/max(1, stats['total_images'])*100:.1f}%)")
    print(f"  Аннотаций Crop (0):    {stats['crop_boxes']:,}")
    print(f"  Аннотаций Weed (1):    {stats['weed_boxes']:,}")
    print(f"  Фоновых кадров (Neg):  {stats['background_images']:,}")
    print(f"  Синтетических тайлов:  {stats['synth_tiles']:,}")

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка объединенного YOLOv8 датасета сорняков и культур")
    parser.add_argument("--output-dir", type=str, default=str(COMBINED_OUT_DIR), help="Каталог датасета")
    parser.add_argument("--yaml-path", type=str, default=str(COMBINED_YAML_PATH), help="Путь к YAML конфигу")
    parser.add_argument("--bg-tiles", type=int, default=200, help="Количество негативных тайлов почвы")
    parser.add_argument("--synth-tiles", type=int, default=400, help="Количество синтетических тайлов БПЛА")
    parser.add_argument("--no-existing-yolo", action="store_true", help="Не включать existing yolo_aerial/yolo_field")
    args = parser.parse_args()

    build_combined_dataset(
        output_dir=Path(args.output_dir),
        yaml_path=Path(args.yaml_path),
        num_bg_tiles=args.bg_tiles,
        num_synth_tiles=args.synth_tiles,
        include_existing_yolo=not args.no_existing_yolo,
    )
