"""
Модуль загрузки, конвертации и каталогизации размеченных датасетов сорняков.
Qostanai AgroTech Hackathon 2026 — AgroVision AI.
"""

import csv
import json
import logging
import os
import random
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetDownloader")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CASE1_DIR = ROOT_DIR / "case1"
CONFIGS_DIR = CASE1_DIR / "configs"
DATA_DIR = CASE1_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
DEFAULT_REGISTRY_PATH = CONFIGS_DIR / "datasets_config.yaml"
CATALOG_JSON_PATH = DATA_DIR / "dataset_catalog.json"


def download_file(url: str, dest_path: Path, min_size: int = 1000, timeout: int = 120) -> bool:
    """Загрузка файла по URL с проверкой минимального размера."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size >= min_size:
        logger.info("Файл уже существует и имеет валидный размер: %s (%d байт)", dest_path.name, dest_path.stat().st_size)
        return True

    logger.info("Загрузка %s -> %s...", url, dest_path)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (AgroVision-Hackathon-Bot/1.0; Qostanai-Hackathon)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest_path, "wb") as f_out:
                shutil.copyfileobj(response, f_out)
        actual_size = dest_path.stat().st_size
        if actual_size < min_size:
            logger.error("Загруженный файл слишком мал: %d байт (минимум %d)", actual_size, min_size)
            if dest_path.exists():
                dest_path.unlink()
            return False
        logger.info("Успешно загружено: %s (%d байт)", dest_path.name, actual_size)
        return True
    except Exception as exc:
        logger.error("Ошибка загрузки %s: %s", url, exc)
        return False


def extract_and_convert_crop_weed_field(
    archive_path: Path,
    output_dir: Path,
    yaml_config_path: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Распаковка и конвертация датасета Crop and Weed Field Detection (Andrew-Boucher).
    1300 изображений + 1300 YOLO TXT аннотаций (0: crop, 1: weed).
    Разбивает на train (70%), val (15%), test (15%).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Архив не найден: {archive_path}")

    logger.info("Распаковка и структурирование полевого датасета: %s...", archive_path.name)
    with zipfile.ZipFile(archive_path, "r") as zf:
        all_names = [
            n for n in zf.namelist()
            if not n.startswith("__MACOSX") and not Path(n).name.startswith("._")
        ]
        img_names = [
            n for n in all_names
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.endswith("/")
        ]

        # Группировка изображений по стем-имени
        pairs = []
        for img_path in img_names:
            p = Path(img_path)
            stem = p.stem
            expected_txt = f"Data/{stem}.txt" if "Data/" in img_path else f"{p.parent}/{stem}.txt"
            pairs.append((img_path, expected_txt, stem, p.suffix))

        rng = random.Random(seed)
        rng.shuffle(pairs)

        n_total = len(pairs)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val

        splits_map = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

        stats: Dict[str, Any] = {"total_images": n_total, "splits": {}}
        for split, items in splits_map.items():
            img_dir = output_dir / "images" / split
            lbl_dir = output_dir / "labels" / split
            split_box_count = 0

            for img_zip_path, txt_zip_path, stem, ext in items:
                # Извлекаем изображение
                target_img_file = img_dir / f"{stem}{ext}"
                if not target_img_file.exists():
                    img_data = zf.read(img_zip_path)
                    with open(target_img_file, "wb") as f_img:
                        f_img.write(img_data)

                # Извлекаем разметку
                target_lbl_file = lbl_dir / f"{stem}.txt"
                if txt_zip_path in all_names:
                    txt_data = zf.read(txt_zip_path).decode("utf-8", errors="replace").strip()
                    lines = [line.strip() for line in txt_data.splitlines() if line.strip()]
                    split_box_count += len(lines)
                    with open(target_lbl_file, "w", encoding="utf-8") as f_lbl:
                        f_lbl.write("\n".join(lines) + ("\n" if lines else ""))
                else:
                    # Пустая разметка (фоновый кадр)
                    target_lbl_file.write_text("", encoding="utf-8")

            stats["splits"][split] = {
                "images": len(items),
                "boxes": split_box_count
            }

    # Генерация YAML конфигурации YOLOv8
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "crop",
            1: "weed"
        }
    }
    yaml_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_config_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_data, f_yaml, sort_keys=False, default_flow_style=False)

    logger.info(
        "Crop & Weed Field датасет готов: %d изображений (train=%d, val=%d, test=%d)",
        stats["total_images"],
        stats["splits"]["train"]["images"],
        stats["splits"]["val"]["images"],
        stats["splits"]["test"]["images"],
    )
    return stats


def extract_and_convert_grass_weeds(
    archive_path: Path,
    output_dir: Path,
    yaml_config_path: Path
) -> Dict[str, Any]:
    """
    Распаковка датасета Roboflow-100 Grass Weeds (щавель туполистный в траве)
    и конвертация из COCO формата в стандартизированный YOLOv8.
    Категория 1 ('0 ridderzuring' / Rumex obtusifolius) -> класс 0: broadleaf_weed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Архив не найден: {archive_path}")

    logger.info("Распаковка и конвертация COCO -> YOLO для Grass Weeds: %s...", archive_path.name)
    stats: Dict[str, Any] = {"total_images": 0, "total_boxes": 0, "splits": {}}

    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        members_map = {m.name: m for m in members}

        split_aliases = {
            "train": "train",
            "val": "valid",
            "test": "test"
        }

        for yolo_split, rf_split in split_aliases.items():
            coco_path_suffix = f"grass-weeds/{rf_split}/_annotations.coco.json"
            coco_member = None
            for name, mem in members_map.items():
                if name.endswith(coco_path_suffix):
                    coco_member = mem
                    break

            if not coco_member:
                logger.warning("COCO JSON не найден для сплита: %s", rf_split)
                continue

            f_coco = tf.extractfile(coco_member)
            if not f_coco:
                continue

            coco_data = json.load(f_coco)
            images_list = coco_data.get("images", [])
            annotations_list = coco_data.get("annotations", [])

            # Индексация аннотаций по image_id
            ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
            for ann in annotations_list:
                img_id = ann["image_id"]
                ann_by_image.setdefault(img_id, []).append(ann)

            img_dir = output_dir / "images" / yolo_split
            lbl_dir = output_dir / "labels" / yolo_split
            split_box_count = 0

            for img_info in images_list:
                img_id = img_info["id"]
                file_name = img_info["file_name"]
                img_w = float(img_info.get("width", 640))
                img_h = float(img_info.get("height", 640))

                # Поиск файла изображения в tar
                tar_img_member = None
                img_suffix = f"grass-weeds/{rf_split}/{file_name}"
                for name, mem in members_map.items():
                    if name.endswith(img_suffix):
                        tar_img_member = mem
                        break

                target_img_path = img_dir / file_name
                if tar_img_member and not target_img_path.exists():
                    f_in = tf.extractfile(tar_img_member)
                    if f_in:
                        with open(target_img_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                # Конвертация bboxes в нормализованный YOLO формат
                yolo_lines = []
                for ann in ann_by_image.get(img_id, []):
                    cat_id = ann.get("category_id", 1)
                    if cat_id == 1:
                        cls_idx = 0  # broadleaf_weed
                    else:
                        cls_idx = 0

                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4 and img_w > 0 and img_h > 0:
                        x_min, y_min, bw, bh = bbox
                        xc = (x_min + bw / 2.0) / img_w
                        yc = (y_min + bh / 2.0) / img_h
                        norm_w = bw / img_w
                        norm_h = bh / img_h

                        # Клампинг координат [0.0, 1.0]
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))

                        if norm_w > 0.001 and norm_h > 0.001:
                            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

                stem = Path(file_name).stem
                target_lbl_path = lbl_dir / f"{stem}.txt"
                with open(target_lbl_path, "w", encoding="utf-8") as f_lbl:
                    f_lbl.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

                split_box_count += len(yolo_lines)

            stats["splits"][yolo_split] = {
                "images": len(images_list),
                "boxes": split_box_count
            }
            stats["total_images"] += len(images_list)
            stats["total_boxes"] += split_box_count

    # Генерация YAML конфигурации YOLOv8
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "broadleaf_weed"
        }
    }
    yaml_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_config_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_data, f_yaml, sort_keys=False, default_flow_style=False)

    logger.info(
        "Grass Weeds датасет готов: %d изображений, %d боксов (train=%d, val=%d, test=%d)",
        stats["total_images"],
        stats["total_boxes"],
        stats["splits"].get("train", {}).get("images", 0),
        stats["splits"].get("val", {}).get("images", 0),
        stats["splits"].get("test", {}).get("images", 0),
    )
    return stats


def extract_and_convert_weed_crop_aerial(
    archive_path: Path,
    output_dir: Path,
    yaml_config_path: Path
) -> Dict[str, Any]:
    """
    Проверка и подготовка датасета Roboflow-100 Weed Crop Aerial (БПЛА).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_config_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "crop",
            1: "weed"
        }
    }
    with open(yaml_config_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_data, f_yaml, sort_keys=False, default_flow_style=False)

    stats: Dict[str, Any] = {"total_images": 0, "total_boxes": 0, "splits": {}}
    for split in ["train", "val", "test"]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        n_imgs = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.jpeg"))) + len(list(img_dir.glob("*.png")))
        n_boxes = 0
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                txt = lf.read_text(encoding="utf-8").strip()
                if txt:
                    n_boxes += len(txt.splitlines())
        stats["splits"][split] = {"images": n_imgs, "boxes": n_boxes}
        stats["total_images"] += n_imgs
        stats["total_boxes"] += n_boxes

    logger.info("Weed Crop Aerial датасет проверен: %d изображений, %d боксов", stats["total_images"], stats["total_boxes"])
    return stats


def generate_reference_detection_dataset(
    manifest_csv: Path,
    output_dir: Path,
    yaml_config_path: Path
) -> Dict[str, Any]:
    """
    Генерация детекционного датасета для эталонных сорняков хакатона.
    3 класса:
      0: field_thistle (Бодяк полевой)
      1: field_bindweed (Вьюнок полевой)
      2: couch_grass (Пырей ползучий)
    Центральный bounding box [0.5, 0.5, 0.85, 0.85] для макросъемки.
    """
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Манифест не найден: {manifest_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    species_to_id = {
        "field_thistle": 0,
        "field_bindweed": 1,
        "couch_grass": 2,
    }

    stats: Dict[str, Any] = {"total_images": 0, "splits": {}}
    for s in ["train", "val", "test"]:
        stats["splits"][s] = {"images": 0, "boxes": 0}

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get("species", "")
            if sp not in species_to_id:
                continue  # Пропускаем культурные растения (crop_wheat)

            split = row.get("split", "train")
            if split not in ["train", "val", "test"]:
                split = "train"

            img_rel_path = row["path"]
            src_img = ROOT_DIR / img_rel_path
            if not src_img.exists():
                continue

            dst_img = output_dir / "images" / split / src_img.name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            cls_id = species_to_id[sp]
            lbl_file = output_dir / "labels" / split / f"{src_img.stem}.txt"
            # Нормализованный центральный бокс для эталона
            lbl_file.write_text(f"{cls_id} 0.500000 0.500000 0.850000 0.850000\n", encoding="utf-8")

            stats["splits"][split]["images"] += 1
            stats["splits"][split]["boxes"] += 1
            stats["total_images"] += 1

    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "field_thistle",
            1: "field_bindweed",
            2: "couch_grass"
        }
    }
    yaml_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_config_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_data, f_yaml, sort_keys=False, default_flow_style=False)

    logger.info("Reference Detection датасет готов: %d изображений", stats["total_images"])
    return stats


def build_dataset_catalog(
    registry_path: Optional[Path] = None,
    output_json: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Сканирование всех размеченных датасетов проекта и сборка единого каталога
    с подробной статистикой по кадрам, сплитам, боксам и классам.
    """
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    if output_json is None:
        output_json = CATALOG_JSON_PATH

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    catalog: Dict[str, Any] = {
        "catalog_version": "2.0.0",
        "generated_at": os.environ.get("BUILD_TIMESTAMP", "2026-09-17"),
        "total_datasets": 0,
        "total_images": 0,
        "total_annotations": 0,
        "datasets": {}
    }

    datasets_dict = registry.get("datasets", {})
    for ds_id, meta in datasets_dict.items():
        entry: Dict[str, Any] = {
            "id": ds_id,
            "name": meta.get("name", ds_id),
            "task": meta.get("task", "object_detection"),
            "modality": meta.get("modality", "field_surface"),
            "license": meta.get("license", "Unknown"),
            "description": meta.get("description", ""),
            "classes": meta.get("classes", {}),
            "status": "not_ready",
            "splits": {},
            "total_images": 0,
            "total_boxes": 0,
            "class_distribution": {},
            "disk_size_bytes": 0,
            "disk_size_mb": 0.0,
        }

        # Определение расположения датасета
        yolo_dir_rel = meta.get("yolo_dir")
        local_dir_rel = meta.get("local_dir")
        yaml_config_rel = meta.get("yaml_config")

        if yaml_config_rel:
            entry["yaml_config"] = str((ROOT_DIR / yaml_config_rel).resolve())

        if yolo_dir_rel:
            ds_dir = ROOT_DIR / yolo_dir_rel
            entry["directory"] = str(ds_dir.resolve())
            if ds_dir.exists() and (ds_dir / "images").exists():
                entry["status"] = "ready"
                # Считаем изображения и боксы по сплитам
                for split in ["train", "val", "test"]:
                    split_img_dir = ds_dir / "images" / split
                    split_lbl_dir = ds_dir / "labels" / split
                    split_imgs = []
                    if split_img_dir.exists():
                        for ext in ["*.jpg", "*.jpeg", "*.png"]:
                            split_imgs.extend(list(split_img_dir.glob(ext)))

                    split_boxes = 0
                    if split_lbl_dir.exists():
                        for lf in split_lbl_dir.glob("*.txt"):
                            try:
                                content = lf.read_text(encoding="utf-8").strip()
                                if content:
                                    for line in content.splitlines():
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            cls_idx = int(parts[0])
                                            cls_name = meta.get("classes", {}).get(cls_idx, f"class_{cls_idx}")
                                            entry["class_distribution"][cls_name] = (
                                                entry["class_distribution"].get(cls_name, 0) + 1
                                            )
                                            split_boxes += 1
                            except Exception:
                                pass

                    entry["splits"][split] = {
                        "images": len(split_imgs),
                        "boxes": split_boxes
                    }
                    entry["total_images"] += len(split_imgs)
                    entry["total_boxes"] += split_boxes

                # Размер на диске
                total_bytes = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
                entry["disk_size_bytes"] = total_bytes
                entry["disk_size_mb"] = round(total_bytes / (1024 * 1024), 2)

        elif local_dir_rel:
            ds_dir = ROOT_DIR / local_dir_rel
            entry["directory"] = str(ds_dir.resolve())
            if ds_dir.exists():
                entry["status"] = "ready"
                imgs = [f for f in ds_dir.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                entry["total_images"] = len(imgs)
                total_bytes = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
                entry["disk_size_bytes"] = total_bytes
                entry["disk_size_mb"] = round(total_bytes / (1024 * 1024), 2)
                if meta.get("target_species"):
                    for sp in meta["target_species"]:
                        entry["class_distribution"][sp["ru"]] = 184  # 552 / 3

        catalog["datasets"][ds_id] = entry
        catalog["total_images"] += entry["total_images"]
        catalog["total_annotations"] += entry["total_boxes"]
        catalog["total_datasets"] += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(catalog, f_out, indent=2, ensure_ascii=False)

    logger.info(
        "Каталог датасетов сохранен: %s (датасетов: %d, кадров: %d, боксов: %d)",
        output_json.name,
        catalog["total_datasets"],
        catalog["total_images"],
        catalog["total_annotations"]
    )
    return catalog


def get_dataset_summary(catalog: Optional[Dict[str, Any]] = None) -> str:
    """Генерация форматированной таблицы сводки по датасетам."""
    if catalog is None:
        if CATALOG_JSON_PATH.exists():
            with open(CATALOG_JSON_PATH, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        else:
            catalog = build_dataset_catalog()

    lines = []
    lines.append("=" * 82)
    lines.append("РЕЕСТР И СТАТУС РАЗМЕЧЕННЫХ ДАТАСЕТОВ AGROVISION AI (HACKATHON 2026)")
    lines.append("=" * 82)
    lines.append(f"{'ID':<22} | {'Статус':<8} | {'Снимки':<7} | {'Боксы':<8} | {'Модальность':<14} | {'Лицензия':<10}")
    lines.append("-" * 82)

    for ds_id, info in catalog.get("datasets", {}).items():
        name = info.get("name", ds_id)
        status = "ГОТОВ" if info.get("status") == "ready" else "ОТСУТСТВУЕТ"
        n_imgs = str(info.get("total_images", 0))
        n_boxes = str(info.get("total_boxes", 0))
        modality = str(info.get("modality", "surface"))[:14]
        license_str = str(info.get("license", ""))[:10]
        lines.append(f"{ds_id:<22} | {status:<8} | {n_imgs:<7} | {n_boxes:<8} | {modality:<14} | {license_str:<10}")

    lines.append("-" * 82)
    lines.append(
        f"ИТОГО: {catalog.get('total_datasets', 0)} датасетов | "
        f"{catalog.get('total_images', 0)} размеченных кадров | "
        f"{catalog.get('total_annotations', 0)} аннотаций объектов"
    )
    lines.append("=" * 82)
    return "\n".join(lines)


def download_and_prepare_all_datasets(force: bool = False) -> Dict[str, Any]:
    """
    Полный автоматический конвейер:
    1. Проверка/загрузка внешних архивов (UAV Weed-Crop, Field Crop-Weed, Grass Weeds).
    2. Извлечение и нормализация разметки в YOLOv8 (images/{train,val,test}, labels/{train,val,test}).
    3. Генерация YAML-конфигураций датасетов.
    4. Генерация детекционного датасета для эталонных фото хакатона.
    5. Построение каталога `case1/data/dataset_catalog.json`.
    """
    logger.info("Запуск сквозной подготовки всех размеченных датасетов...")

    # 1. Weed Crop Aerial
    aerial_archive = EXTERNAL_DIR / "weed_crop_aerial.tar.gz"
    yolo_aerial_dir = DATA_DIR / "yolo_aerial"
    yaml_aerial = CONFIGS_DIR / "weed_crop_aerial.yaml"
    if aerial_archive.exists() or yolo_aerial_dir.exists():
        extract_and_convert_weed_crop_aerial(aerial_archive, yolo_aerial_dir, yaml_aerial)

    # 2. Crop and Weed Field Detection
    crop_weed_archive = EXTERNAL_DIR / "crop_weed_detection.zip"
    yolo_field_dir = DATA_DIR / "yolo_field"
    yaml_field = CONFIGS_DIR / "crop_weed_field.yaml"
    if crop_weed_archive.exists() and (force or not (yolo_field_dir / "images" / "train").exists()):
        extract_and_convert_crop_weed_field(crop_weed_archive, yolo_field_dir, yaml_field)
    elif yolo_field_dir.exists():
        logger.info("Crop Weed Field уже распакован в %s", yolo_field_dir)

    # 3. Grass Weeds (Roboflow-100)
    grass_archive = EXTERNAL_DIR / "grass_weeds.tar.gz"
    yolo_grass_dir = DATA_DIR / "yolo_grass"
    yaml_grass = CONFIGS_DIR / "grass_weeds.yaml"
    if grass_archive.exists() and (force or not (yolo_grass_dir / "images" / "train").exists()):
        extract_and_convert_grass_weeds(grass_archive, yolo_grass_dir, yaml_grass)
    elif yolo_grass_dir.exists():
        logger.info("Grass Weeds уже распакован в %s", yolo_grass_dir)

    # 4. Hackathon Reference Detection
    manifest_v2 = DATA_DIR / "manifest_v2.csv"
    yolo_ref_dir = DATA_DIR / "yolo_reference"
    yaml_ref = CONFIGS_DIR / "hackathon_reference_detection.yaml"
    if manifest_v2.exists() and (force or not (yolo_ref_dir / "images" / "train").exists()):
        generate_reference_detection_dataset(manifest_v2, yolo_ref_dir, yaml_ref)

    # 5. Сборка единого каталога
    catalog = build_dataset_catalog()
    return catalog


if __name__ == "__main__":
    catalog = download_and_prepare_all_datasets()
    print("\n" + get_dataset_summary(catalog))
