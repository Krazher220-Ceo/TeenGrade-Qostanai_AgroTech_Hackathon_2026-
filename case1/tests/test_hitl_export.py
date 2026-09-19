"""
Тесты HITL-экспорта верификаций агронома (case1/ml/hitl_export.py):
- решения агронома идут на дообучение, отменённые/незавершённые — исключаются;
- культура сохраняется как отрицательный класс crop_wheat;
- split назначается по кадру (image_id), без утечки между train/val/test;
- дедупликация по (image_id, object_id) и идемпотентность повторного экспорта;
- итоговый манифест читается case1.ml.dataset.WeedMultiTaskDataset.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from case1.ml.hitl_export import (
    is_decisive,
    load_verified_ledger,
    load_detection_index,
    build_hitl_records,
    assign_splits_by_frame,
    export_hitl_dataset,
    compute_hitl_stats,
    MANIFEST_FIELDNAMES,
)


# ------------------------------------------------------------------------------
# Фикстуры
# ------------------------------------------------------------------------------

def _make_crop(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (48, 48), color=(30, 140, 40))
    img.save(path, "JPEG")


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def hitl_env(tmp_path):
    """
    Небольшой синтетический реестр:
    - frameA:1 -> Бодяк полевой, вердикт spray_weed (решительный)
    - frameA:2 -> Пшеница/культура, вердикт do_not_spray (решительный, is_crop)
    - frameA:3 -> Бодяк полевой, вердикт manual_review (НЕ решительный, исключить)
    - frameB:1 -> Пырей ползучий, вердикт spray_weed, фаза "4–6 листьев"
    - frameB:2 -> Вьюнок полевой, вердикт cancelled (отменено, исключить)
    - frameC:1 -> Вьюнок полевой, вердикт spray_weed, НО файл вырезки отсутствует
    """
    output_dir = tmp_path / "output"
    crops_root = output_dir
    verified_path = output_dir / "verified_actions.json"
    csv_path = output_dir / "all_fields_detections.csv"

    verified_actions = {
        "frameA.JPG:1": {
            "image_id": "frameA.JPG", "object_id": 1,
            "verified_species": "field_thistle", "verified_species_ru": "Бодяк полевой",
            "verified_stage_ru": "Не определено", "is_crop": False, "action": "spray_weed",
            "verified_by": "Агроном А", "timestamp": "2026-09-18T08:00:00Z",
        },
        "frameA.JPG:2": {
            "image_id": "frameA.JPG", "object_id": 2,
            "verified_species": "crop_wheat", "verified_species_ru": "Пшеница (культура)",
            "verified_stage_ru": None, "is_crop": True, "action": "do_not_spray",
            "verified_by": "Агроном А", "timestamp": "2026-09-18T08:01:00Z",
        },
        "frameA.JPG:3": {
            "image_id": "frameA.JPG", "object_id": 3,
            "verified_species": "field_thistle", "verified_species_ru": "Бодяк полевой",
            "verified_stage_ru": "Не определено", "is_crop": False, "action": "manual_review",
            "verified_by": "Агроном А", "timestamp": "2026-09-18T08:02:00Z",
        },
        "frameB.JPG:1": {
            "image_id": "frameB.JPG", "object_id": 1,
            "verified_species": "couch_grass", "verified_species_ru": "Пырей ползучий",
            "verified_stage_ru": "4–6 листьев", "is_crop": False, "action": "spray_weed",
            "verified_by": "Агроном Б", "timestamp": "2026-09-18T09:00:00Z",
        },
        "frameB.JPG:2": {
            "image_id": "frameB.JPG", "object_id": 2,
            "verified_species": "field_bindweed", "verified_species_ru": "Вьюнок полевой",
            "verified_stage_ru": "Не определено", "is_crop": False, "action": "cancelled",
            "verified_by": "Агроном Б", "timestamp": "2026-09-18T09:01:00Z",
        },
        "frameC.JPG:1": {
            "image_id": "frameC.JPG", "object_id": 1,
            "verified_species": "field_bindweed", "verified_species_ru": "Вьюнок полевой",
            "verified_stage_ru": "Не определено", "is_crop": False, "action": "spray_weed",
            "verified_by": "Агроном В", "timestamp": "2026-09-18T10:00:00Z",
        },
    }
    _write_json(verified_path, verified_actions)

    csv_rows = [
        {"image_id": "frameA.JPG", "object_id": "1", "crop_path": "crops/frameA/crop_0001.jpg"},
        {"image_id": "frameA.JPG", "object_id": "2", "crop_path": "crops/frameA/crop_0002.jpg"},
        {"image_id": "frameA.JPG", "object_id": "3", "crop_path": "crops/frameA/crop_0003.jpg"},
        {"image_id": "frameB.JPG", "object_id": "1", "crop_path": "crops/frameB/crop_0001.jpg"},
        {"image_id": "frameB.JPG", "object_id": "2", "crop_path": "crops/frameB/crop_0002.jpg"},
        {"image_id": "frameC.JPG", "object_id": "1", "crop_path": "crops/frameC/crop_0001.jpg"},
    ]
    _write_csv(csv_path, csv_rows)

    # Реальные файлы вырезок создаём для всех, КРОМЕ frameC (имитация отсутствующей
    # вырезки — например, если crops/ ещё не сгенерированы на этом хосте).
    _make_crop(output_dir / "crops" / "frameA" / "crop_0001.jpg")
    _make_crop(output_dir / "crops" / "frameA" / "crop_0002.jpg")
    _make_crop(output_dir / "crops" / "frameA" / "crop_0003.jpg")
    _make_crop(output_dir / "crops" / "frameB" / "crop_0001.jpg")
    _make_crop(output_dir / "crops" / "frameB" / "crop_0002.jpg")
    # frameC/crop_0001.jpg намеренно не создаём.

    return {
        "output_dir": output_dir,
        "verified_path": verified_path,
        "csv_path": csv_path,
        "crops_root": crops_root,
    }


# ------------------------------------------------------------------------------
# is_decisive
# ------------------------------------------------------------------------------

def test_is_decisive_accepts_spray_and_do_not_spray():
    assert is_decisive("spray_weed") is True
    assert is_decisive("do_not_spray") is True


@pytest.mark.parametrize("action", [None, "", "manual_review", "cancelled", "cancel", "отменено"])
def test_is_decisive_rejects_non_verdicts(action):
    assert is_decisive(action) is False


# ------------------------------------------------------------------------------
# build_hitl_records
# ------------------------------------------------------------------------------

def test_build_hitl_records_excludes_non_decisive_and_missing_crops(hitl_env):
    verified = load_verified_ledger(hitl_env["verified_path"])
    det_index = load_detection_index(hitl_env["csv_path"])

    records, counters = build_hitl_records(verified, det_index, crops_root=hitl_env["crops_root"])

    keys = {(r["image_id"], str(r["object_id"])) for r in records}
    assert ("frameA.JPG", "1") in keys  # Бодяк, spray_weed -> included
    assert ("frameA.JPG", "2") in keys  # Культура, do_not_spray -> included (crop_wheat)
    assert ("frameA.JPG", "3") not in keys  # manual_review -> excluded
    assert ("frameB.JPG", "1") in keys  # Пырей, spray_weed -> included
    assert ("frameB.JPG", "2") not in keys  # cancelled -> excluded
    assert ("frameC.JPG", "1") not in keys  # crop file missing -> excluded

    assert counters["total"] == 6
    assert counters["excluded_not_decisive"] == 2  # manual_review + cancelled
    assert counters["excluded_missing_crop"] == 1  # frameC
    assert counters["included"] == 3


def test_build_hitl_records_maps_crop_decision_to_crop_wheat_class(hitl_env):
    verified = load_verified_ledger(hitl_env["verified_path"])
    det_index = load_detection_index(hitl_env["csv_path"])
    records, _ = build_hitl_records(verified, det_index, crops_root=hitl_env["crops_root"])

    crop_record = next(r for r in records if r["image_id"] == "frameA.JPG" and str(r["object_id"]) == "2")
    assert crop_record["species"] == "crop_wheat"


def test_build_hitl_records_resolves_stage_code_from_ru_text(hitl_env):
    verified = load_verified_ledger(hitl_env["verified_path"])
    det_index = load_detection_index(hitl_env["csv_path"])
    records, _ = build_hitl_records(verified, det_index, crops_root=hitl_env["crops_root"])

    couch_record = next(r for r in records if r["image_id"] == "frameB.JPG" and str(r["object_id"]) == "1")
    assert couch_record["species"] == "couch_grass"
    assert couch_record["stage"] == "4_to_6_leaves"


# ------------------------------------------------------------------------------
# assign_splits_by_frame
# ------------------------------------------------------------------------------

def test_assign_splits_keeps_whole_frame_together():
    records = [
        {"image_id": "f1.JPG", "object_id": 1},
        {"image_id": "f1.JPG", "object_id": 2},
        {"image_id": "f2.JPG", "object_id": 1},
        {"image_id": "f3.JPG", "object_id": 1},
        {"image_id": "f4.JPG", "object_id": 1},
        {"image_id": "f5.JPG", "object_id": 1},
    ]
    assign_splits_by_frame(records, val_frac=0.2, test_frac=0.2, seed=7)

    splits_by_frame = {}
    for r in records:
        splits_by_frame.setdefault(r["image_id"], set()).add(r["split"])

    # Один кадр -> ровно один split.
    for frame, splits in splits_by_frame.items():
        assert len(splits) == 1, f"{frame} split across multiple sets: {splits}"

    for r in records:
        assert r["split"] in {"train", "val", "test"}


def test_assign_splits_small_sample_goes_to_train():
    records = [
        {"image_id": "only_one.JPG", "object_id": 1},
        {"image_id": "only_one.JPG", "object_id": 2},
    ]
    assign_splits_by_frame(records)
    assert all(r["split"] == "train" for r in records)


# ------------------------------------------------------------------------------
# export_hitl_dataset — идемпотентность и отчёт
# ------------------------------------------------------------------------------

def test_export_hitl_dataset_writes_manifest_and_report(hitl_env):
    manifest_path = hitl_env["output_dir"].parent / "data" / "manifest_hitl.csv"

    report = export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )

    assert manifest_path.exists()
    assert report["included_records"] == 3
    assert report["new_records"] == 3
    assert report["updated_records"] == 0
    assert report["total_records_in_manifest"] == 3
    assert report["new_examples_by_species"].get("field_thistle") == 1
    assert report["new_examples_by_species"].get("crop_wheat") == 1
    assert report["new_examples_by_species"].get("couch_grass") == 1

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == MANIFEST_FIELDNAMES
        rows = list(reader)
    assert len(rows) == 3


def test_export_hitl_dataset_is_idempotent_on_rerun(hitl_env):
    manifest_path = hitl_env["output_dir"].parent / "data" / "manifest_hitl.csv"

    export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )
    second_report = export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )

    assert second_report["new_records"] == 0
    assert second_report["updated_records"] == 0
    assert second_report["total_records_in_manifest"] == 3


def test_export_hitl_dataset_picks_up_newly_verified_and_freshly_available_crops(hitl_env):
    manifest_path = hitl_env["output_dir"].parent / "data" / "manifest_hitl.csv"
    export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )

    # Теперь появляется вырезка для frameC (например, конвейер process её досчитал).
    _make_crop(hitl_env["crops_root"] / "crops" / "frameC" / "crop_0001.jpg")

    second_report = export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )
    assert second_report["new_records"] == 1
    assert second_report["total_records_in_manifest"] == 4


# ------------------------------------------------------------------------------
# compute_hitl_stats
# ------------------------------------------------------------------------------

def test_compute_hitl_stats_reports_pending_and_exported(hitl_env):
    manifest_path = hitl_env["output_dir"].parent / "data" / "manifest_hitl.csv"

    stats_before = compute_hitl_stats(
        verified_actions_path=hitl_env["verified_path"],
        output_manifest_path=manifest_path,
    )
    assert stats_before["total_verified_decisions"] == 6
    assert stats_before["decisive_decisions"] == 4  # 3 включены + 1 без вырезки (frameC)
    assert stats_before["excluded_decisions"] == 2
    assert stats_before["pending_export"] == 4  # манифест ещё не создан
    assert stats_before["manifest_exists"] is False

    export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )

    stats_after = compute_hitl_stats(
        verified_actions_path=hitl_env["verified_path"],
        output_manifest_path=manifest_path,
    )
    assert stats_after["exported_to_training"] == 3
    assert stats_after["pending_export"] == 1  # frameC всё ещё без вырезки
    assert stats_after["manifest_exists"] is True
    assert stats_after["decisions_by_species"].get("Бодяк полевой") == 1


# ------------------------------------------------------------------------------
# Интеграция с case1.ml.dataset.WeedMultiTaskDataset
# ------------------------------------------------------------------------------

def test_exported_manifest_is_readable_by_weed_multitask_dataset(hitl_env):
    from case1.ml.dataset import WeedMultiTaskDataset, SPECIES_TO_IDX

    manifest_path = hitl_env["output_dir"].parent / "data" / "manifest_hitl.csv"
    export_hitl_dataset(
        verified_actions_path=hitl_env["verified_path"],
        detections_csv_path=hitl_env["csv_path"],
        output_manifest_path=manifest_path,
        crops_root=hitl_env["crops_root"],
    )

    ds = WeedMultiTaskDataset(str(manifest_path), split="train", species_to_idx=SPECIES_TO_IDX)
    assert len(ds) >= 1
    img, sp_idx, st_idx, st_mask, meta = ds[0]
    assert img.mode == "RGB"
    assert int(sp_idx) in SPECIES_TO_IDX.values()
    assert meta["species"] in SPECIES_TO_IDX
