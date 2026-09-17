"""
Генератор кропов культурных растений (пшеница/культура) и сборка 4-классового манифеста:
0: field_thistle (Бодяк полевой)
1: field_bindweed (Вьюнок полевой)
2: couch_grass (Пырей ползучий)
3: crop_wheat (Пшеница / Культурное растение / Фон)
"""

import csv
import json
import random
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EXTERNAL_AERIAL = ROOT_DIR / "case1" / "data" / "external" / "weed_crop_aerial" / "rf100" / "weed-crop-aerial"
FIELD_PHOTOS_DIR = ROOT_DIR / "Dataset 1 кейс" / "ФотоПолей"
MANIFEST_V1 = ROOT_DIR / "case1" / "data" / "manifest.csv"
CROPS_OUT_DIR = ROOT_DIR / "case1" / "data" / "crops_wheat"
MANIFEST_V2 = ROOT_DIR / "case1" / "data" / "manifest_v2.csv"


def extract_aerial_crops(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    crop_samples = []

    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = EXTERNAL_AERIAL / split
        coco_path = split_dir / "_annotations.coco.json"
        if not coco_path.exists():
            continue

        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        crops_ann = [a for a in coco["annotations"] if a.get("category_id") == 1]  # 1 = crop

        # Выбираем качественные образцы
        for idx, ann in enumerate(crops_ann):
            img_info = images.get(ann["image_id"])
            if not img_info:
                continue

            img_path = split_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            x, y, w, h = ann["bbox"]
            if w < 20 or h < 20:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                # Добавляем 10% контекста вокруг растения
                pad_w = w * 0.1
                pad_h = h * 0.1
                x1 = max(0, int(x - pad_w))
                y1 = max(0, int(y - pad_h))
                x2 = min(img.width, int(x + w + pad_w))
                y2 = min(img.height, int(y + h + pad_h))

                crop = img.crop((x1, y1, x2, y2))
                if crop.width < 15 or crop.height < 15:
                    continue

                out_fname = f"crop_aerial_{split}_{idx:04d}.jpg"
                out_file = target_dir / out_fname
                crop.save(out_file, quality=92)

                # Распределение по сплитам: сохраняем сплит из датасета
                split_name = "val" if split == "valid" else split
                crop_samples.append({
                    "path": str(out_file),
                    "species": "crop_wheat",
                    "stage": "unknown",
                    "source": "aerial_crops",
                    "split": split_name,
                    "group_id": f"aerial_crop_{img_info['file_name'][:10]}",
                    "filename": out_fname
                })
            except Exception as e:
                continue

    return crop_samples


def extract_field_crop_patches(target_dir: Path, num_patches_per_image: int = 15):
    """Извлечение патчей рядков культурных растений из реальных 4K снимков DJI."""
    target_dir.mkdir(parents=True, exist_ok=True)
    field_samples = []

    field_images = list(FIELD_PHOTOS_DIR.glob("*.JPG")) + list(FIELD_PHOTOS_DIR.glob("*.jpg"))

    for img_p in field_images:
        try:
            img = Image.open(img_p).convert("RGB")
            w_total, h_total = img.size
            patch_sz = 320

            count = 0
            attempts = 0
            while count < num_patches_per_image and attempts < 60:
                attempts += 1
                rx = random.randint(100, w_total - patch_sz - 100)
                ry = random.randint(100, h_total - patch_sz - 100)

                patch = img.crop((rx, ry, rx + patch_sz, ry + patch_sz))
                # Простая эвристика: пропускаем абсолютно черные/белые края
                stat = patch.convert("L")
                stat_data = list(stat.getdata())
                mean_val = sum(stat_data) / len(stat_data)
                if mean_val < 35 or mean_val > 230:
                    continue

                out_fname = f"patch_dji_{img_p.stem}_{count:02d}.jpg"
                out_file = target_dir / out_fname
                patch.save(out_file, quality=90)

                r_split = random.random()
                if r_split < 0.70:
                    sp = "train"
                elif r_split < 0.85:
                    sp = "val"
                else:
                    sp = "test"

                field_samples.append({
                    "path": str(out_file),
                    "species": "crop_wheat",
                    "stage": "unknown",
                    "source": "dji_field_crops",
                    "split": sp,
                    "group_id": f"dji_patch_{img_p.stem}",
                    "filename": out_fname
                })
                count += 1
        except Exception as e:
            continue

    return field_samples


def build_manifest_v2():
    random.seed(42)
    print("=== ГЕНЕРАЦИЯ КРОПОВ КУЛЬТУРЫ / ПШЕНИЦЫ ДЛЯ 4-КЛАССОВОЙ МОДЕЛИ ===")

    aerial_crops = extract_aerial_crops(CROPS_OUT_DIR)
    print(f"Извлечено {len(aerial_crops)} кропов культуры из weed-crop-aerial.")

    field_patches = extract_field_crop_patches(CROPS_OUT_DIR, num_patches_per_image=20)
    print(f"Извлечено {len(field_patches)} патчей фона/культуры из снимков DJI.")

    all_crop_samples = aerial_crops + field_patches
    print(f"Всего образцов класса crop_wheat: {len(all_crop_samples)}")

    # Читаем существующий манифест (сорняки: 552 шт)
    existing_rows = []
    with open(MANIFEST_V1, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing_rows.append(r)

    print(f"Существующих эталонов сорняков: {len(existing_rows)}")

    # Объединяем сорняки и культуру
    total_manifest = existing_rows + all_crop_samples

    # Записываем manifest_v2.csv
    fieldnames = ["path", "species", "stage", "source", "group_id", "split", "filename"]
    with open(MANIFEST_V2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(total_manifest)

    # Статистика по сплитам
    splits_count = {"train": 0, "val": 0, "test": 0}
    species_count = {}
    for r in total_manifest:
        splits_count[r["split"]] = splits_count.get(r["split"], 0) + 1
        sp = r["species"]
        species_count[sp] = species_count.get(sp, 0) + 1

    print(f"\n★ Создан {MANIFEST_V2}")
    print(f"Сплиты: {splits_count}")
    print(f"Виды: {species_count}")

    return total_manifest


if __name__ == "__main__":
    build_manifest_v2()
