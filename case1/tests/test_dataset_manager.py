"""
Тесты целостности, формата разметки и каталогизации датасетов AgroVision AI.
"""

import json
from pathlib import Path
import pytest
import yaml

from case1.data.dataset_downloader import build_dataset_catalog, get_dataset_summary

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CASE1_DIR = ROOT_DIR / "case1"
CONFIGS_DIR = CASE1_DIR / "configs"
DATA_DIR = CASE1_DIR / "data"
CATALOG_PATH = DATA_DIR / "dataset_catalog.json"
REGISTRY_PATH = CONFIGS_DIR / "datasets_config.yaml"


def test_registry_config_structure():
    """Проверка структуры и обязательных полей в реестре datasets_config.yaml."""
    assert REGISTRY_PATH.exists(), f"Реестр не найден: {REGISTRY_PATH}"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    assert "datasets" in reg
    datasets = reg["datasets"]
    assert len(datasets) >= 5

    required_fields = ["id", "name", "task", "modality", "license"]
    for ds_id, meta in datasets.items():
        for field in required_fields:
            assert field in meta, f"Поле '{field}' отсутствует в датасете '{ds_id}'"


def test_generated_yolo_yaml_configs():
    """Проверка наличия и валидности сгенерированных YAML-конфигураций YOLOv8."""
    expected_configs = [
        "weed_crop_aerial.yaml",
        "crop_weed_field.yaml",
        "grass_weeds.yaml",
        "hackathon_reference_detection.yaml",
    ]
    for cfg_name in expected_configs:
        cfg_path = CONFIGS_DIR / cfg_name
        assert cfg_path.exists(), f"YAML конфиг отсутствует: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "train" in data
        assert "val" in data
        assert "names" in data
        assert len(data["names"]) >= 1


def test_dataset_catalog_json_structure():
    """Проверка целостности и агрегированной статистики в dataset_catalog.json."""
    assert CATALOG_PATH.exists(), f"Каталог не найден: {CATALOG_PATH}"
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert "catalog_version" in catalog
    assert "total_datasets" in catalog
    assert "total_images" in catalog
    assert "total_annotations" in catalog
    assert "datasets" in catalog

    assert catalog["total_datasets"] >= 5
    assert catalog["total_images"] >= 5000
    assert catalog["total_annotations"] >= 10000

    # Проверка обязательных датасетов
    expected_ids = ["weed_crop_aerial", "crop_weed_detection", "grass_weeds", "hackathon_reference"]
    for eid in expected_ids:
        assert eid in catalog["datasets"], f"Датасет {eid} отсутствует в каталоге"
        assert catalog["datasets"][eid]["status"] == "ready"


def test_bounding_box_coordinates_bounds():
    """Проверка того, что все координаты боксов в YOLO TXT лежат в диапазоне [0.0, 1.0]."""
    yolo_dirs = [
        DATA_DIR / "yolo_aerial",
        DATA_DIR / "yolo_field",
        DATA_DIR / "yolo_grass",
        DATA_DIR / "yolo_reference",
    ]

    for ydir in yolo_dirs:
        if not ydir.exists():
            continue

        label_files = list((ydir / "labels").rglob("*.txt"))
        assert len(label_files) > 0, f"Нет файлов разметки в {ydir}"

        # Проверяем выборку файлов разметки
        sample_files = label_files[:100]
        checked_boxes = 0
        for lf in sample_files:
            txt = lf.read_text(encoding="utf-8").strip()
            if not txt:
                continue
            for line in txt.splitlines():
                parts = line.strip().split()
                assert len(parts) >= 5, f"Невалидная строка разметки в {lf}: '{line}'"
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                assert cls_id >= 0, f"Отрицательный class_id в {lf}"
                assert 0.0 <= xc <= 1.0, f"xc вне [0, 1] ({xc}) в {lf}"
                assert 0.0 <= yc <= 1.0, f"yc вне [0, 1] ({yc}) в {lf}"
                assert 0.0 < w <= 1.0, f"w вне (0, 1] ({w}) в {lf}"
                assert 0.0 < h <= 1.0, f"h вне (0, 1] ({h}) в {lf}"
                checked_boxes += 1

        assert checked_boxes > 0, f"Не проверено ни одного бокса в {ydir}"


def test_splits_and_image_label_pairing():
    """Проверка наличия изображений для каждого сплита в датасетах."""
    datasets_to_check = ["yolo_aerial", "yolo_field", "yolo_grass"]
    for ds_name in datasets_to_check:
        ds_dir = DATA_DIR / ds_name
        for split in ["train", "val", "test"]:
            img_dir = ds_dir / "images" / split
            lbl_dir = ds_dir / "labels" / split
            assert img_dir.exists(), f"Папка изображений не найдена: {img_dir}"
            assert lbl_dir.exists(), f"Папка меток не найдена: {lbl_dir}"

            n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
            n_lbls = len(list(lbl_dir.glob("*.txt")))
            assert n_imgs > 0, f"Сплит {split} в {ds_name} пуст"
            assert n_lbls >= n_imgs, f"Метки отсутствуют для некоторых изображений в {ds_name}/{split}"


def test_get_dataset_summary_output():
    """Проверка генерации текстовой сводки каталога."""
    summary = get_dataset_summary()
    assert "РЕЕСТР И СТАТУС РАЗМЕЧЕННЫХ ДАТАСЕТОВ" in summary
    assert "ИТОГО" in summary
    assert "weed_crop_aerial" in summary
    assert "crop_weed_detection" in summary
    assert "grass_weeds" in summary
