#!/usr/bin/env python3
"""
================================================================================
Qostanai AgroTech Hackathon 2026 — КЕЙС №1 (MAIN): МОНИТОРИНГ СОРНЯКОВ С БПЛА
================================================================================

Главный управляющий файл для Кейса №1:
- Аудит и валидация данных (552 эталона сорняков, 5 снимков DJI 4096x3072).
- Загрузка и проверка официального чекпоинта WeedBlaster YOLOv8s.
- Моделирование бортового модуля дрона (Edge): извлечение EXIF/GPS, тайлинг, детекция, обрезка.
- Серверная классификация вида и фазы сорняка с контролируемым отказом (unknown / review_required).
- Экспорт результатов в форматы JSON, CSV и генерация размеченных кадров.

Использование:
    python3 case1_main.py status             # Проверка готовности окружения и данных
    python3 case1_main.py audit              # Подробный статистический аудит данных
    python3 case1_main.py download-weights   # Загрузка настоящих весов best.pt с HF
    python3 case1_main.py process            # Сквозная обработка полевых снимков
    python3 case1_main.py --help             # Справка по всем командам
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Базовые пути проекта
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "Dataset 1 кейс"
WEEDS_DIR = DATASET_DIR / "Сорняки"
FIELD_DIR = DATASET_DIR / "ФотоПолей"
WEEDBLASTER_DIR = ROOT_DIR / "weedblaster-vision-yolov8s"
WEIGHTS_PATH = WEEDBLASTER_DIR / "best.pt"
CASE1_DIR = ROOT_DIR / "case1"
OUTPUT_DIR = CASE1_DIR / "output"
FINETUNED_DETECTOR_PATH = CASE1_DIR / "models" / "weed_detector_finetuned.pt"

EXPECTED_SHA256 = "2a10f51e1d78db493b6c75662986748bebc01562fbce70f7f2b49f0176b4ecb6"
EXPECTED_SIZE = 22600106
OFFICIAL_HF_URL = "https://huggingface.co/NvMayMay/weedblaster-vision-yolov8s/resolve/main/best.pt"


def get_jpeg_size(file_path: Path) -> Optional[Tuple[int, int]]:
    """Быстрое чтение размеров JPEG без сторонних библиотек."""
    try:
        with open(file_path, "rb") as f:
            data = f.read(65536)
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF:
                    marker = data[i + 1]
                    if marker in [0xC0, 0xC1, 0xC2]:
                        h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                        return w, h
                    elif marker in [0xD8, 0xD9]:
                        i += 2
                    else:
                        if i + 4 > len(data):
                            break
                        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
                        i += 2 + length
                else:
                    i += 1
    except Exception:
        pass
    return None


def extract_dji_metadata(file_path: Path) -> Dict[str, Any]:
    """Извлечение EXIF и DJI XMP метаданных (координаты, высота, модель)."""
    meta: Dict[str, Any] = {
        "filename": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "lat": None,
        "lon": None,
        "abs_alt": None,
        "rel_alt": None,
        "model": None,
        "datetime": None,
    }

    # Попытка через macOS mdls
    import subprocess
    try:
        res = subprocess.run(
            ["mdls", "-name", "kMDItemLatitude", "-name", "kMDItemLongitude",
             "-name", "kMDItemAltitude", "-name", "kMDItemAcquisitionModel",
             "-name", "kMDItemContentCreationDate", str(file_path)],
            capture_output=True, text=True, timeout=3
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "kMDItemLatitude" in line and "=" in line:
                    val = line.split("=")[1].strip()
                    if val != "(null)": meta["lat"] = float(val)
                elif "kMDItemLongitude" in line and "=" in line:
                    val = line.split("=")[1].strip()
                    if val != "(null)": meta["lon"] = float(val)
                elif "kMDItemAltitude" in line and "=" in line:
                    val = line.split("=")[1].strip()
                    if val != "(null)": meta["abs_alt"] = float(val)
                elif "kMDItemAcquisitionModel" in line and "=" in line:
                    meta["model"] = line.split("=")[1].strip().strip('"')
                elif "kMDItemContentCreationDate" in line and "=" in line:
                    meta["datetime"] = line.split("=")[1].strip()
    except Exception:
        pass

    # Извлечение относительной высоты из DJI XMP
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(131072)
            match_rel = re.search(rb'drone-dji:RelativeAltitude=\"([^\"]+)\"', chunk)
            if match_rel:
                meta["rel_alt"] = float(match_rel.group(1).decode())
            match_abs = re.search(rb'drone-dji:AbsoluteAltitude=\"([^\"]+)\"', chunk)
            if match_abs and meta["abs_alt"] is None:
                meta["abs_alt"] = float(match_abs.group(1).decode())
    except Exception:
        pass

    return meta


def check_weights_status() -> Dict[str, Any]:
    """Проверка файла весов best.pt."""
    status: Dict[str, Any] = {
        "exists": WEIGHTS_PATH.exists(),
        "is_lfs_pointer": False,
        "is_valid_checkpoint": False,
        "size_bytes": 0,
        "sha256": None,
        "msg": "",
    }
    if not WEIGHTS_PATH.exists():
        status["msg"] = "Файл best.pt отсутствует на диске."
        return status

    size = WEIGHTS_PATH.stat().st_size
    status["size_bytes"] = size

    if size < 1024:
        content = WEIGHTS_PATH.read_text(errors="ignore")
        if "git-lfs" in content or "oid sha256" in content:
            status["is_lfs_pointer"] = True
            status["msg"] = (
                f"Файл best.pt является указателем Git LFS ({size} байт), "
                f"настоящие веса ({EXPECTED_SIZE} байт) не загружены."
            )
            return status

    # Если размер похож на веса, считаем SHA-256
    h = hashlib.sha256()
    with open(WEIGHTS_PATH, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    file_sha = h.hexdigest()
    status["sha256"] = file_sha

    if file_sha == EXPECTED_SHA256 and size == EXPECTED_SIZE:
        status["is_valid_checkpoint"] = True
        status["msg"] = f"Веса проверены: настоящий PyTorch checkpoint ({size / 1024 / 1024:.1f} МБ), SHA256 совпадает."
    else:
        status["msg"] = f"Файл весов найден ({size} байт), но хэш отличается от официального."
    return status


def cmd_status():
    """Команда проверки статуса проекта."""
    print("================================================================================")
    print("СТАТУС КЕЙСА №1: МОНИТОРИНГ СОРНЯКОВ (MAIN)")
    print("================================================================================")
    print(f"Рабочая директория: {ROOT_DIR}")
    print(f"Каталог кейса:      {CASE1_DIR}")
    print(f"Python версия:      {sys.version.split()[0]}")

    # Проверка библиотек
    for lib in ["torch", "torchvision", "ultralytics", "PIL", "fastapi", "cv2"]:
        try:
            __import__(lib)
            print(f"  [+] Библиотека {lib:<12}: доступна")
        except ImportError:
            print(f"  [-] Библиотека {lib:<12}: НЕ установлена")

    # Проверка данных
    print("\n--- Проверка локальных датасетов ---")
    if WEEDS_DIR.exists():
        weeds_count = len(list(WEEDS_DIR.rglob("*.[jJ][pP][gG]")))
        print(f"  [+] Справочные фото сорняков: {weeds_count} файлов в {WEEDS_DIR.name}")
    else:
        print(f"  [-] Папка {WEEDS_DIR} не найдена!")

    if FIELD_DIR.exists():
        field_count = len(list(FIELD_DIR.glob("*.[jJ][pP][gG]")))
        print(f"  [+] Полевые снимки DJI:       {field_count} файлов в {FIELD_DIR.name}")
    else:
        print(f"  [-] Папка {FIELD_DIR} не найдена!")

    # Проверка модели
    print("\n--- Проверка модели детектора (WeedBlaster) ---")
    w_stat = check_weights_status()
    print(f"  Статус best.pt: {w_stat['msg']}")
    if w_stat["is_lfs_pointer"]:
        print(f"  -> Для загрузки запустите: python3 case1_main.py download-weights")


def cmd_audit():
    """Команда подробного аудита данных."""
    print("================================================================================")
    print("ПОДРОБНЫЙ АУДИТ ДАННЫХ КЕЙСА №1")
    print("================================================================================")

    # 1. Эталоны сорняков
    print("\n1. СПРАВОЧНЫЕ ФОТОГРАФИИ СОРНЯКОВ:")
    if not WEEDS_DIR.exists():
        print("  Папка с сорняками не найдена!")
        return

    species_dict: Dict[str, Dict[str, int]] = {}
    unique_hashes = set()
    total_weeds = 0

    for p in WEEDS_DIR.rglob("*.[jJ][pP][gG]"):
        total_weeds += 1
        parts = p.relative_to(WEEDS_DIR).parts
        sp = parts[0] if len(parts) > 1 else "Unknown"
        st = parts[1] if len(parts) > 2 else "Unknown"
        species_dict.setdefault(sp, {}).setdefault(st, 0)
        species_dict[sp][st] += 1
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        unique_hashes.add(h)

    print(f"  Всего изображений: {total_weeds}")
    print(f"  Уникальных файлов по SHA-256: {len(unique_hashes)}")
    print("  Распределение по видам и фазам:")
    for sp, stages in sorted(species_dict.items()):
        print(f"    - {sp}:")
        for st, cnt in sorted(stages.items()):
            print(f"        * {st}: {cnt} фото")

    # Внимание на отсутствие розетки у пырея
    if "Пырей ползучий" in species_dict and "Розетка" not in species_dict["Пырей ползучий"]:
        print("\n  [ВАЖНО!] У пырея ползучего полностью отсутствует фаза «Розетка» (0 фото).")
        print("  Модель не должна синтезировать ложные предсказания для этой комбинации.")

    # 2. Полевые снимки
    print("\n2. ПОЛЕВЫЕ СНИМКИ С ДРОНА:")
    if not FIELD_DIR.exists():
        print("  Папка полевых снимков не найдена!")
        return

    field_files = sorted(FIELD_DIR.glob("*.[jJ][pP][gG]"))
    print(f"  Найдено снимков: {len(field_files)}")
    for f in field_files:
        dim = get_jpeg_size(f)
        meta = extract_dji_metadata(f)
        print(f"\n  Файл: {f.name}")
        print(f"    Размер: {f.stat().st_size / (1024*1024):.1f} МБ, Разрешение: {dim[0]}x{dim[1]} px" if dim else "")
        print(f"    Камера: {meta['model']}, Дата: {meta['datetime']}")
        print(f"    GPS: {meta['lat']}° N, {meta['lon']}° E")
        print(f"    Высота: абс={meta['abs_alt']} м, над землей (RelAlt)={meta['rel_alt']} м")


def cmd_download_weights():
    """Загрузка официальных весов с Hugging Face."""
    print("================================================================================")
    print("ЗАГРУЗКА ОФИЦИАЛЬНЫХ ВЕСОВ WEEDBLASTER YOLOV8S")
    print("================================================================================")
    print(f"Источник:         {OFFICIAL_HF_URL}")
    print(f"Целевой путь:     {WEIGHTS_PATH}")
    print(f"Ожидаемый размер: {EXPECTED_SIZE} байт (~22.6 МБ)")
    print(f"Ожидаемый SHA256: {EXPECTED_SHA256}")

    import urllib.request
    try:
        print("\nЗагрузка файла...")
        start = time.time()
        urllib.request.urlretrieve(OFFICIAL_HF_URL, WEIGHTS_PATH)
        elapsed = time.time() - start
        print(f"Загрузка завершена за {elapsed:.2f} с.")

        stat = check_weights_status()
        if stat["is_valid_checkpoint"]:
            print(f"[УСПЕХ] {stat['msg']}")
        else:
            print(f"[ОШИБКА] {stat['msg']}")
    except Exception as e:
        print(f"[ОШИБКА ЗАГРУЗКИ] {e}")


def cmd_process(
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    detector_path: Optional[str] = None,
):
    """Полноценная тайловая обработка полевых снимков с детекцией сорняков."""
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from PIL import Image

    out_base = Path(output_dir) if output_dir else OUTPUT_DIR
    annotated_dir = out_base / "annotated"
    crops_dir = out_base / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        print("[ОШИБКА] Веса best.pt отсутствуют! Сначала запустите: python3 case1_main.py download-weights")
        return None

    selected_detector = Path(detector_path) if detector_path else WEIGHTS_PATH
    if not selected_detector.exists():
        raise FileNotFoundError(f"Веса детектора не найдены: {selected_detector}")
    model = YOLO(str(selected_detector))
    weed_class_ids = {
        int(class_id)
        for class_id, class_name in model.names.items()
        if str(class_name).strip().lower() in {"weed", "weeds"}
    }
    if not weed_class_ids:
        raise RuntimeError(f"В детекторе {selected_detector} отсутствует класс Weed")
    print(f"Детектор: {selected_detector} | Weed class IDs: {sorted(weed_class_ids)}")

    if image_path:
        photos = [Path(image_path)]
    else:
        photos = sorted(FIELD_DIR.glob("*.[jJ][pP][gG]"))

    print(f"=== Запуск тайловой детекции сорняков ({len(photos)} снимков) ===")

    tile_size = 1280
    overlap = 0.20
    stride = int(tile_size * (1 - overlap))

    summary_stats = []
    all_csv_rows = []

    for p_idx, p in enumerate(photos):
        meta = extract_dji_metadata(p)
        print(f"\n[{p_idx+1}/{len(photos)}] Анализ {p.name} (высота над землей: {meta['rel_alt']} м)...")

        img = Image.open(p)
        W, H = img.size

        tiles = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                x_end = min(x + tile_size, W)
                y_end = min(y + tile_size, H)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                tiles.append((x_start, y_start, x_end, y_end))
        tiles = sorted(list(set(tiles)))

        t0 = time.time()
        raw_boxes = []
        raw_scores = []

        for (x1, y1, x2, y2) in tiles:
            tile_crop = img.crop((x1, y1, x2, y2))
            res = model.predict(tile_crop, imgsz=1280, conf=0.25, verbose=False, device="cpu")
            for b in res[0].boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                if cls_id in weed_class_ids:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    raw_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                    raw_scores.append(conf)

        t_inf = time.time() - t0

        # NMS слияние рамок
        keep = []
        if raw_boxes:
            boxes_arr = np.array(raw_boxes, dtype=np.float32)
            scores_arr = np.array(raw_scores, dtype=np.float32)
            x1_a, y1_a, x2_a, y2_a = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
            areas = (x2_a - x1_a) * (y2_a - y1_a)
            order = scores_arr.argsort()[::-1]
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1_a[i], x1_a[order[1:]])
                yy1 = np.maximum(y1_a[i], y1_a[order[1:]])
                xx2 = np.minimum(x2_a[i], x2_a[order[1:]])
                yy2 = np.minimum(y2_a[i], y2_a[order[1:]])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                inds = np.where(iou <= 0.45)[0]
                order = order[inds + 1]

        final_boxes = [raw_boxes[i] for i in keep]
        final_scores = [raw_scores[i] for i in keep]

        print(f"   -> Время детекции: {t_inf:.2f} с | Найдено сорняков: {len(final_boxes)}")

        # Загрузка классификатора видов и фаз
        classifier_path = CASE1_DIR / "models" / "multitask_weeds_best.pt"
        classifier = None
        if classifier_path.exists():
            import torch
            from torchvision import transforms
            from case1.ml.multitask_model import WeedMultiTaskModel, infer_num_species_from_state_dict
            dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            state_dict = torch.load(classifier_path, map_location=dev)
            num_species = infer_num_species_from_state_dict(state_dict)
            classifier = WeedMultiTaskModel(num_species=num_species, num_stages=2, pretrained=False)
            classifier.load_state_dict(state_dict)
            classifier.to(dev)
            classifier.eval()
            crop_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        cv_img = cv2.imread(str(p))
        photo_crop_dir = crops_dir / p.stem
        photo_crop_dir.mkdir(parents=True, exist_ok=True)

        detections_data = []
        species_counts = {}
        stage_counts = {}
        review_count = 0

        for d_idx, (box, score) in enumerate(zip(final_boxes, final_scores)):
            bx1, by1, bx2, by2 = [int(v) for v in box]
            bw = bx2 - bx1
            bh = by2 - by1

            crop_patch = cv_img[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)]
            crop_filename = f"crop_{d_idx:04d}_{score:.2f}.jpg"
            if crop_patch.size > 0:
                cv2.imwrite(str(photo_crop_dir / crop_filename), crop_patch)

            # Классификация вида и фазы
            if classifier is not None and crop_patch.size > 0:
                pil_crop = Image.fromarray(cv2.cvtColor(crop_patch, cv2.COLOR_BGR2RGB))
                tensor_crop = crop_transform(pil_crop).unsqueeze(0).to(dev)
                c_res = classifier.predict_crop(tensor_crop, species_thresh=0.55, stage_thresh=0.50)
            else:
                c_res = {
                    "species": "unknown", "species_ru": "Неизвестный сорняк",
                    "species_conf": 0.0, "stage": "unknown", "stage_ru": "Не определено",
                    "stage_conf": 0.0, "review_required": True,
                    "spray_action": "manual_review"
                }

            sp_ru = c_res["species_ru"]
            st_ru = c_res["stage_ru"]
            review = c_res["review_required"]

            species_counts[sp_ru] = species_counts.get(sp_ru, 0) + 1
            stage_counts[st_ru] = stage_counts.get(st_ru, 0) + 1
            if review:
                review_count += 1

            # Цветовое кодирование рамок
            if "Вьюнок" in sp_ru:
                color = (0, 220, 255)      # Желтый BGR
            elif "Бодяк" in sp_ru:
                color = (255, 0, 255)      # Пурпурный BGR
            elif "Пырей" in sp_ru:
                color = (255, 200, 0)      # Голубой BGR
            else:
                color = (0, 0, 255)        # Красный BGR для unknown / review

            cv2.rectangle(cv_img, (bx1, by1), (bx2, by2), color, 4)
            label_text = f"{sp_ru} [{st_ru}] {c_res['species_conf']:.2f}"
            if review:
                label_text += " (!)"
            cv2.putText(cv_img, label_text, (bx1, max(25, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            det_entry = {
                "object_id": d_idx + 1,
                "bbox_xyxy": [bx1, by1, bx2, by2],
                "bbox_wh": [bw, bh],
                "detector_conf": round(score, 4),
                "species": c_res["species"],
                "species_ru": sp_ru,
                "species_conf": c_res["species_conf"],
                "stage": c_res["stage"],
                "stage_ru": st_ru,
                "stage_conf": c_res["stage_conf"],
                "spray_action": c_res.get("spray_action", "manual_review"),
                "review_required": review,
                "crop_file": f"crops/{p.stem}/{crop_filename}"
            }
            detections_data.append(det_entry)

            all_csv_rows.append({
                "image_id": p.name,
                "object_id": d_idx + 1,
                "detector_conf": round(score, 4),
                "species": c_res["species"],
                "species_ru": sp_ru,
                "species_conf": c_res["species_conf"],
                "stage": c_res["stage"],
                "stage_ru": st_ru,
                "stage_conf": c_res["stage_conf"],
                "spray_action": c_res.get("spray_action", "manual_review"),
                "review_required": review,
                "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
                "bbox_width": bw, "bbox_height": bh,
                "drone_lat": meta["lat"], "drone_lon": meta["lon"],
                "drone_abs_alt": meta["abs_alt"], "drone_rel_alt": meta["rel_alt"],
                "timestamp": meta["datetime"],
                "crop_path": f"crops/{p.stem}/{crop_filename}"
            })

        annotated_file = annotated_dir / f"annotated_{p.name}"
        cv2.imwrite(str(annotated_file), cv_img)

        avg_conf = np.mean([d["species_conf"] for d in detections_data]) if detections_data else 0.0
        med_w = np.median([d["bbox_wh"][0] for d in detections_data]) if detections_data else 0
        med_h = np.median([d["bbox_wh"][1] for d in detections_data]) if detections_data else 0

        summary_stats.append({
            "filename": p.name,
            "rel_alt_m": meta["rel_alt"],
            "abs_alt_m": meta["abs_alt"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "time_s": round(t_inf, 2),
            "total_weeds": len(final_boxes),
            "review_required_count": review_count,
            "counts_by_species": species_counts,
            "counts_by_stage": stage_counts,
            "avg_species_conf": round(float(avg_conf), 3),
            "median_box_wh": [int(med_w), int(med_h)],
            "annotated_image": str(annotated_file.name),
            "detections": detections_data
        })

    import csv
    with open(out_base / "all_fields_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    csv_file = out_base / "all_fields_detections.csv"
    csv_columns = [
        "image_id", "object_id", "detector_conf", "species", "species_ru",
        "species_conf", "stage", "stage_ru", "stage_conf", "spray_action",
        "review_required", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_width", "bbox_height", "drone_lat", "drone_lon",
        "drone_abs_alt", "drone_rel_alt", "timestamp", "crop_path",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\n[ГОТОВО] Все снимки обработаны!")
    print(f"Отчет JSON сохранен в: {out_base / 'all_fields_report.json'}")
    print(f"Таблица CSV сохранена в: {csv_file}")
    print(f"Размеченные кадры:     {annotated_dir}")

    return {
        "output_dir": str(out_base),
        "report_json": str(out_base / "all_fields_report.json"),
        "detections_csv": str(csv_file),
        "annotated_dir": str(annotated_dir),
        "detector_path": str(selected_detector),
        "images_processed": len(summary_stats),
        "detections_total": len(all_csv_rows),
    }


def cmd_train_classifier(epochs: int = 35, use_focal: bool = True):
    from case1.ml.train import train_model
    train_model(epochs=epochs, use_focal=use_focal)


def cmd_train_detector(epochs: int = 15):
    from case1.ml.train_detector import train_detector
    train_detector(epochs=epochs)


def cmd_dashboard(port: int = 8501):
    import subprocess
    app_path = CASE1_DIR / "dashboard" / "app.py"
    streamlit_bin = ROOT_DIR / ".venv" / "bin" / "streamlit"
    cmd = [
        str(streamlit_bin) if streamlit_bin.exists() else "streamlit",
        "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["HOME"] = str(ROOT_DIR)
    env["STREAMLIT_CONFIG_DIR"] = str(ROOT_DIR / ".streamlit")
    print(f"Запуск Streamlit Dashboard на http://localhost:{port}...")
    subprocess.run(cmd, env=env)


def cmd_cluster_analysis():
    """Запуск кластерного анализа эмбеддингов t-SNE / Silhouette для защиты от галлюцинаций."""
    from case1.ml.cluster_analysis import run_cluster_analysis
    run_cluster_analysis()


def cmd_fleet_plan(num_drones: int = 3, altitude: float = 30.0, speed: float = 5.0, output_dir: Optional[str] = None):
    """Offline планирование покрытия поля 1–5 БПЛА."""
    from case1.fleet.demo_field import get_kostanay_demo_field, get_demo_landing_pads
    from case1.fleet.planner import plan_fleet_coverage
    from case1.fleet.exporters import export_fleet_plan_json, export_fleet_plan_geojson, OfflineFleetDatabase

    out = Path(output_dir) if output_dir else (OUTPUT_DIR / "fleet_plan")
    out.mkdir(parents=True, exist_ok=True)

    field = get_kostanay_demo_field()
    pads = get_demo_landing_pads(field)

    print("================================================================================")
    print("ПЛАНИРОВАНИЕ ОФЛАЙН-ФЛОТА БПЛА (1–5 ДРОНОВ) — AGROVISION AI")
    print("================================================================================")
    print(f"Поле:               {field.name}")
    print(f"Запрошено дронов:   {num_drones}")
    print(f"Высота / Скорость:  {altitude} м / {speed} м/с ({speed*3.6:.1f} км/ч)")

    plan = plan_fleet_coverage(field=field, num_drones=num_drones, altitude_m=altitude, ground_speed_m_s=speed, pads=pads)

    print("\n--- РЕЗУЛЬТАТЫ РАСЧЁТА ---")
    print(f"  [+] Оптимальный угол галсов: {plan.sweep_angle_deg}°")
    print(f"  [+] Footprint камеры:        {plan.footprint_width_m:.1f} x {plan.footprint_height_m:.1f} м")
    print(f"  [+] Разрешение на земле GSD: {plan.gsd_x_cm_px:.2f} см/px")
    print(f"  [+] Шаг между полосами:      {plan.line_spacing_m:.1f} м")
    print(f"  [+] Интервал триггера:       {plan.trigger_interval_s:.2f} с ({plan.trigger_distance_m:.1f} м)")
    print(f"  [+] Всего рабочих полос:     {plan.total_lanes_count}")
    print(f"  [+] Покрытие поля:           {plan.coverage_percentage:.1f}% ({plan.covered_area_ha:.1f} / {plan.field_area_ha:.1f} га)")
    print(f"  [+] Время операции флота:    {plan.makespan_s/60:.1f} мин ({plan.makespan_s:.0f} с)")
    print(f"  [+] Время одиночного дрона:  {plan.theoretical_single_drone_time_s/60:.1f} мин")
    print(f"  [+] Ускорение флота:         {plan.speedup_factor:.2f}x (назначено {plan.num_drones_assigned} БПЛА)")

    print("\n--- РАСПРЕДЕЛЕНИЕ ПО ДРОНАМ ---")
    for v in plan.vehicles:
        print(f"  - Дрон #{v.system_id} ({v.color_hex}): {len(v.lanes)} полос, {v.flight_length_m:.0f} м, {v.total_estimated_time_s/60:.1f} мин, батарея {v.battery_consumed_pct:.1f}% [площадка {v.pad.pad_id}]")

    json_path = export_fleet_plan_json(plan, out / "mission-pack.json")
    geojson_path = export_fleet_plan_geojson(plan, field, out / "fleet_mission.geojson")
    db_path = out / "fleet_offline.db"
    db = OfflineFleetDatabase(db_path)
    db.save_plan(plan)

    print(f"\n[ГОТОВО] Экспорт выполнен:")
    print(f"  - JSON mission pack: {json_path}")
    print(f"  - GeoJSON карта:     {geojson_path}")
    print(f"  - SQLite база:       {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Кейс №1: Система мониторинга и картирования сорняков (Qostanai AgroTech Hackathon 2026)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")

    subparsers.add_parser("status", help="Проверка окружения, весов и датасетов")
    subparsers.add_parser("audit", help="Детальный аудит данных сорняков и снимков DJI")
    subparsers.add_parser("download-weights", help="Скачать официальные веса с Hugging Face")
    subparsers.add_parser("cluster-analysis", help="Кластерный анализ эмбеддингов (t-SNE/Silhouette)")

    proc_parser = subparsers.add_parser("process", help="Запуск конвейера обработки полевых фото")
    proc_parser.add_argument("--image", type=str, default=None, help="Путь к конкретному снимку (по умолчанию вся папка)")
    proc_parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Каталог для результатов")
    proc_parser.add_argument(
        "--detector",
        choices=["baseline", "finetuned"],
        default="baseline",
        help="baseline — проверенный WeedBlaster; finetuned — экспериментальный aerial checkpoint",
    )

    tr_cls_parser = subparsers.add_parser("train-classifier", help="Обучение многозадачного классификатора сорняков")
    tr_cls_parser.add_argument("--epochs", type=int, default=35, help="Количество эпох")
    tr_cls_parser.add_argument("--no-focal", action="store_true", help="Отключить Focal Loss")

    tr_det_parser = subparsers.add_parser("train-detector", help="Дообучение детектора YOLOv8 на аэрофотосъемке")
    tr_det_parser.add_argument("--epochs", type=int, default=15, help="Количество эпох")

    dash_parser = subparsers.add_parser("dashboard", help="Запуск интерактивного веб-дашборда Streamlit")
    dash_parser.add_argument("--port", type=int, default=8501, help="Порт для веб-интерфейса")

    fleet_parser = subparsers.add_parser("fleet-plan", help="Офлайн-планирование миссии флота 1–5 БПЛА")
    fleet_parser.add_argument("--drones", type=int, default=3, help="Количество дронов (1..5)")
    fleet_parser.add_argument("--altitude", type=float, default=30.0, help="Высота полёта в метрах")
    fleet_parser.add_argument("--speed", type=float, default=5.0, help="Скорость полёта в м/с (5.0 м/с = 18 км/ч)")
    fleet_parser.add_argument("--output", type=str, default=None, help="Каталог для сохранения результатов")

    subparsers.add_parser("test", help="Запуск дымовых тестов системы")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "audit":
        cmd_audit()
    elif args.command == "download-weights":
        cmd_download_weights()
    elif args.command == "cluster-analysis":
        cmd_cluster_analysis()
    elif args.command == "process":
        detector_path = WEIGHTS_PATH if args.detector == "baseline" else FINETUNED_DETECTOR_PATH
        cmd_process(image_path=args.image, output_dir=args.output, detector_path=str(detector_path))
    elif args.command == "train-classifier":
        cmd_train_classifier(epochs=args.epochs, use_focal=not args.no_focal)
    elif args.command == "train-detector":
        cmd_train_detector(epochs=args.epochs)
    elif args.command == "dashboard":
        cmd_dashboard(port=args.port)
    elif args.command == "fleet-plan":
        cmd_fleet_plan(num_drones=args.drones, altitude=args.altitude, speed=args.speed, output_dir=args.output)
    elif args.command == "test":
        import subprocess
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT_DIR)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(CASE1_DIR / "tests")],
            env=env,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
