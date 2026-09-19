"""
Тесты дообучения на HITL-верификациях (case1/ml/finetune_hitl.py):
- decide_promotion никогда не разрешает заменить модель, если кандидат хуже;
- build_merged_manifest объединяет исходные и HITL-данные, отбрасывая
  неизвестные виды (которых нет в текущей species_mapping чекпойнта);
- короткий (1 эпоха, мало данных) end-to-end прогон дообучения сохраняет
  версионированного кандидата и НЕ трогает боевой чекпойнт, пока продвижение
  явно не запрошено и не разрешено сравнением метрик.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from case1.ml.multitask_model import WeedMultiTaskModel
from case1.ml.finetune_hitl import (
    combined_score,
    decide_promotion,
    build_merged_manifest,
    finetune_from_hitl,
    MANIFEST_MERGE_FIELDS,
)
import case1.ml.finetune_hitl as finetune_hitl_module


SPECIES_MAP = {"field_thistle": 0, "field_bindweed": 1, "couch_grass": 2, "crop_wheat": 3}


# ------------------------------------------------------------------------------
# combined_score / decide_promotion (детерминированные модульные тесты)
# ------------------------------------------------------------------------------

def test_combined_score_weights_species_more_than_stage():
    metrics = {"species_macro_f1": 1.0, "stage_macro_f1": 0.0}
    assert combined_score(metrics) == pytest.approx(0.6)


def test_decide_promotion_allows_equal_or_better_candidate():
    baseline = {"species_macro_f1": 0.5, "stage_macro_f1": 0.5}
    candidate_equal = {"species_macro_f1": 0.5, "stage_macro_f1": 0.5}
    candidate_better = {"species_macro_f1": 0.7, "stage_macro_f1": 0.6}

    promoted_eq, _, _, _ = decide_promotion(baseline, candidate_equal)
    promoted_better, _, _, _ = decide_promotion(baseline, candidate_better)

    assert promoted_eq is True
    assert promoted_better is True


def test_decide_promotion_refuses_worse_candidate():
    baseline = {"species_macro_f1": 0.8, "stage_macro_f1": 0.8}
    candidate_worse = {"species_macro_f1": 0.3, "stage_macro_f1": 0.3}

    promoted, reason, baseline_score, candidate_score = decide_promotion(baseline, candidate_worse)

    assert promoted is False
    assert candidate_score < baseline_score
    assert "НЕ заменяется" in reason


def test_decide_promotion_respects_min_improvement_threshold():
    baseline = {"species_macro_f1": 0.5, "stage_macro_f1": 0.5}
    candidate_slightly_better = {"species_macro_f1": 0.51, "stage_macro_f1": 0.51}

    promoted_no_threshold, *_ = decide_promotion(baseline, candidate_slightly_better, min_improvement=0.0)
    promoted_with_threshold, *_ = decide_promotion(baseline, candidate_slightly_better, min_improvement=0.05)

    assert promoted_no_threshold is True
    assert promoted_with_threshold is False


# ------------------------------------------------------------------------------
# build_merged_manifest
# ------------------------------------------------------------------------------

def _write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_MERGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_merged_manifest_combines_and_filters_unknown_species(tmp_path):
    original_path = tmp_path / "original.csv"
    hitl_path = tmp_path / "hitl.csv"
    merged_path = tmp_path / "merged.csv"

    _write_manifest(original_path, [
        {"path": "a.jpg", "species": "field_thistle", "stage": "cotyledon_to_2_leaves", "source": "ref", "group_id": "g1", "split": "train", "filename": "a.jpg"},
    ])
    _write_manifest(hitl_path, [
        {"path": "b.jpg", "species": "couch_grass", "stage": "4_to_6_leaves", "source": "hitl_verified", "group_id": "g2", "split": "train", "filename": "b.jpg"},
        {"path": "c.jpg", "species": "not_in_mapping_species", "stage": "", "source": "hitl_verified", "group_id": "g3", "split": "train", "filename": "c.jpg"},
    ])

    stats = build_merged_manifest(merged_path, original_path, hitl_path, SPECIES_MAP)

    assert stats["original_rows"] == 1
    assert stats["hitl_rows_used"] == 1
    assert stats["hitl_rows_skipped_unknown_species"] == 1
    assert stats["total_rows"] == 2

    with open(merged_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    species_in_merged = {r["species"] for r in rows}
    assert species_in_merged == {"field_thistle", "couch_grass"}


def test_build_merged_manifest_handles_missing_hitl_manifest(tmp_path):
    original_path = tmp_path / "original.csv"
    merged_path = tmp_path / "merged.csv"
    _write_manifest(original_path, [
        {"path": "a.jpg", "species": "field_thistle", "stage": "cotyledon_to_2_leaves", "source": "ref", "group_id": "g1", "split": "train", "filename": "a.jpg"},
    ])
    stats = build_merged_manifest(merged_path, original_path, tmp_path / "does_not_exist.csv", SPECIES_MAP)
    assert stats["hitl_rows_used"] == 0
    assert stats["total_rows"] == 1


# ------------------------------------------------------------------------------
# finetune_from_hitl — короткий end-to-end прогон
# ------------------------------------------------------------------------------

def _make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path, "JPEG")


@pytest.fixture
def finetune_env(tmp_path):
    images_dir = tmp_path / "images"
    colors = {
        "field_thistle": (200, 40, 40),
        "field_bindweed": (40, 200, 40),
        "couch_grass": (40, 40, 200),
        "crop_wheat": (200, 200, 40),
    }

    original_rows = []
    for split in ["train", "val", "test"]:
        for species, color in colors.items():
            img_path = images_dir / f"{species}_{split}.jpg"
            _make_image(img_path, color)
            original_rows.append({
                "path": str(img_path), "species": species,
                "stage": "cotyledon_to_2_leaves" if species != "crop_wheat" else "",
                "source": "reference_camera", "group_id": f"{species}_{split}",
                "split": split, "filename": img_path.name,
            })
    original_manifest = tmp_path / "manifest_v2.csv"
    _write_manifest(original_manifest, original_rows)

    hitl_img = images_dir / "hitl_field_thistle_train.jpg"
    _make_image(hitl_img, (180, 60, 60))
    hitl_manifest = tmp_path / "manifest_hitl.csv"
    _write_manifest(hitl_manifest, [
        {"path": str(hitl_img), "species": "field_thistle", "stage": "cotyledon_to_2_leaves",
         "source": "hitl_verified", "group_id": "hitl_frame1", "split": "train", "filename": hitl_img.name},
    ])

    species_mapping_path = tmp_path / "species_mapping.json"
    with open(species_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"species_to_idx": SPECIES_MAP, "num_classes": 4, "num_stages": 3}, f)

    base_model = WeedMultiTaskModel(num_species=4, num_stages=3, pretrained=False)
    base_checkpoint = tmp_path / "multitask_weeds_best.pt"
    torch.save(base_model.state_dict(), base_checkpoint)

    hitl_models_dir = tmp_path / "models_hitl"
    output_dir = tmp_path / "output"

    return {
        "base_checkpoint": base_checkpoint,
        "species_mapping_path": species_mapping_path,
        "original_manifest": original_manifest,
        "hitl_manifest": hitl_manifest,
        "hitl_models_dir": hitl_models_dir,
        "output_dir": output_dir,
    }


def test_finetune_from_hitl_produces_versioned_candidate_and_report(finetune_env):
    report = finetune_from_hitl(
        base_checkpoint=finetune_env["base_checkpoint"],
        species_mapping_path=finetune_env["species_mapping_path"],
        original_manifest_path=finetune_env["original_manifest"],
        hitl_manifest_path=finetune_env["hitl_manifest"],
        hitl_models_dir=finetune_env["hitl_models_dir"],
        output_dir=finetune_env["output_dir"],
        epochs=1,
        batch_size=4,
        promote_if_better=False,
    )

    candidate_path = Path(report["candidate_checkpoint"])
    assert candidate_path.exists()
    assert report["merge_stats"]["hitl_rows_used"] == 1
    assert report["training"]["epochs_run"] == 1
    assert report["training"]["train_samples"] >= 4
    assert "baseline_metrics" in report and "candidate_metrics" in report
    assert Path(report["report_path"]).exists()

    # Кандидат должен быть валидным state_dict, загружаемым в модель той же архитектуры.
    state = torch.load(candidate_path, map_location="cpu", weights_only=True)
    model = WeedMultiTaskModel(num_species=4, num_stages=3, pretrained=False)
    model.load_state_dict(state)  # не должно кидать исключение


def test_finetune_from_hitl_never_overwrites_base_checkpoint_by_default(finetune_env):
    base_bytes_before = finetune_env["base_checkpoint"].read_bytes()

    finetune_from_hitl(
        base_checkpoint=finetune_env["base_checkpoint"],
        species_mapping_path=finetune_env["species_mapping_path"],
        original_manifest_path=finetune_env["original_manifest"],
        hitl_manifest_path=finetune_env["hitl_manifest"],
        hitl_models_dir=finetune_env["hitl_models_dir"],
        output_dir=finetune_env["output_dir"],
        epochs=1,
        batch_size=4,
        promote_if_better=False,  # по умолчанию — не заменять
    )

    assert finetune_env["base_checkpoint"].read_bytes() == base_bytes_before


def test_finetune_from_hitl_promotes_only_when_not_worse(finetune_env, monkeypatch):
    """
    Дообучение на крошечных случайных данных недетерминировано по качеству,
    поэтому здесь детерминируем именно решение о продвижении (decide_promotion),
    а не полагаемся на реальные метрики — проверяем ИМЕННО механику копирования
    файла, которая обязана слушаться этого решения.
    """
    base_bytes_before = finetune_env["base_checkpoint"].read_bytes()

    # Сценарий 1: кандидат хуже -> НЕ заменяем, даже если promote_if_better=True.
    monkeypatch.setattr(
        finetune_hitl_module, "decide_promotion",
        lambda baseline, candidate, min_improvement=0.0: (False, "forced worse", 0.9, 0.1),
    )
    report_worse = finetune_from_hitl(
        base_checkpoint=finetune_env["base_checkpoint"],
        species_mapping_path=finetune_env["species_mapping_path"],
        original_manifest_path=finetune_env["original_manifest"],
        hitl_manifest_path=finetune_env["hitl_manifest"],
        hitl_models_dir=finetune_env["hitl_models_dir"],
        output_dir=finetune_env["output_dir"],
        epochs=1,
        batch_size=4,
        promote_if_better=True,
    )
    assert report_worse["promotion_allowed"] is False
    assert report_worse["promotion_applied"] is False
    assert finetune_env["base_checkpoint"].read_bytes() == base_bytes_before

    # Сценарий 2: кандидат не хуже -> заменяем, т.к. promote_if_better=True.
    monkeypatch.setattr(
        finetune_hitl_module, "decide_promotion",
        lambda baseline, candidate, min_improvement=0.0: (True, "forced better", 0.1, 0.9),
    )
    report_better = finetune_from_hitl(
        base_checkpoint=finetune_env["base_checkpoint"],
        species_mapping_path=finetune_env["species_mapping_path"],
        original_manifest_path=finetune_env["original_manifest"],
        hitl_manifest_path=finetune_env["hitl_manifest"],
        hitl_models_dir=finetune_env["hitl_models_dir"],
        output_dir=finetune_env["output_dir"],
        epochs=1,
        batch_size=4,
        promote_if_better=True,
    )
    assert report_better["promotion_allowed"] is True
    assert report_better["promotion_applied"] is True
    candidate_bytes = Path(report_better["candidate_checkpoint"]).read_bytes()
    assert finetune_env["base_checkpoint"].read_bytes() == candidate_bytes
    assert finetune_env["base_checkpoint"].read_bytes() != base_bytes_before


def test_finetune_from_hitl_without_promote_flag_ignores_decision(finetune_env, monkeypatch):
    """Даже если decide_promotion разрешает продвижение, без promote_if_better=True ничего не меняется."""
    base_bytes_before = finetune_env["base_checkpoint"].read_bytes()
    monkeypatch.setattr(
        finetune_hitl_module, "decide_promotion",
        lambda baseline, candidate, min_improvement=0.0: (True, "forced better", 0.1, 0.9),
    )
    report = finetune_from_hitl(
        base_checkpoint=finetune_env["base_checkpoint"],
        species_mapping_path=finetune_env["species_mapping_path"],
        original_manifest_path=finetune_env["original_manifest"],
        hitl_manifest_path=finetune_env["hitl_manifest"],
        hitl_models_dir=finetune_env["hitl_models_dir"],
        output_dir=finetune_env["output_dir"],
        epochs=1,
        batch_size=4,
        promote_if_better=False,
    )
    assert report["promotion_allowed"] is True
    assert report["promotion_applied"] is False
    assert finetune_env["base_checkpoint"].read_bytes() == base_bytes_before
