"""
Конвертация внешнего датасета weed-crop-aerial из COCO формата в YOLOv8 формат.
Категории:
  0: crop
  1: weed
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any


def convert_coco_to_yolo(
    source_dir: str = "case1/data/external/weed_crop_aerial/rf100/weed-crop-aerial",
    output_dir: str = "case1/data/yolo_aerial",
    config_yaml_path: str = "case1/configs/weed_crop_aerial.yaml"
):
    src_path = Path(source_dir).resolve()
    dst_path = Path(output_dir).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    # category_id in COCO: 1 -> crop (yolo 0), 2 -> weed (yolo 1)
    cat_mapping = {1: 0, 2: 1}
    class_names = ["crop", "weed"]

    stats = {}

    for split in splits:
        split_src = src_path / split
        coco_file = split_src / "_annotations.coco.json"
        if not coco_file.exists():
            print(f"[WARN] Файл не найден: {coco_file}")
            continue

        split_dst_name = "val" if split == "valid" else split
        img_dst_dir = dst_path / "images" / split_dst_name
        lbl_dst_dir = dst_path / "labels" / split_dst_name
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        with open(coco_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Индексируем картинки
        images_info = {img["id"]: img for img in coco_data.get("images", [])}
        
        # Группируем аннотации по image_id
        img_annotations: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

        converted_images = 0
        total_boxes = 0

        for img_id, img_info in images_info.items():
            fname = img_info["file_name"]
            img_src_file = split_src / fname
            if not img_src_file.exists():
                continue

            # Копируем или линкуем изображение
            img_target_file = img_dst_dir / fname
            if not img_target_file.exists():
                shutil.copy2(img_src_file, img_target_file)

            # Формируем YOLO метки
            label_target_file = lbl_dst_dir / f"{Path(fname).stem}.txt"
            w_img = float(img_info["width"])
            h_img = float(img_info["height"])

            lines = []
            anns = img_annotations.get(img_id, [])
            for ann in anns:
                cid = ann.get("category_id")
                if cid not in cat_mapping:
                    continue
                yolo_cls = cat_mapping[cid]
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                # Нормализация
                x_center = (x + w / 2.0) / w_img
                y_center = (y + h / 2.0) / h_img
                w_norm = w / w_img
                h_norm = h / h_img

                # Ограничение диапазона [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            with open(label_target_file, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(lines))

            converted_images += 1
            total_boxes += len(lines)

        stats[split_dst_name] = {"images": converted_images, "boxes": total_boxes}
        print(f"[{split_dst_name.upper()}] Конвертировано {converted_images} изображений, {total_boxes} боксов.")

    # Генерируем конфигурационный YAML файл для YOLOv8
    yaml_content = f"""# Автоматически сгенерированный датасет weed-crop-aerial для YOLOv8
path: {dst_path}
train: images/train
val: images/val
test: images/test

names:
  0: crop
  1: weed
"""
    yaml_file = Path(config_yaml_path)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nКонфигурация сохранена в {yaml_file}")
    return stats


if __name__ == "__main__":
    convert_coco_to_yolo()
