"""
Дообучение (fine-tuning) классификатора сорняков на HITL-верификациях агронома.

Реализует обещание питча: "каждое решение агронома сохраняется и идёт на
дообучение, с каждым выездом модель ошибается реже". Конвейер:

  1. case1.ml.hitl_export.export_hitl_dataset() превращает подтверждённые
     решения агронома (case1/output/verified_actions.json) в манифест
     case1/data/manifest_hitl.csv.
  2. finetune_from_hitl() стартует с ТЕКУЩЕГО чекпойнта (case1/models/
     multitask_weeds_best.pt), дообучает его на смеси исходного датасета
     (manifest_v2.csv) и HITL-примеров, недолго (по умолчанию 1 эпоха —
     полноценное переобучение с нуля не требуется).
  3. Кандидат оценивается на отложенной выборке (val/test split общего
     манифеста) и сравнивается с baseline-метриками ИСХОДНОГО чекпойнта на
     той же выборке.
  4. Кандидат ВСЕГДА сохраняется в отдельный версионированный файл
     (case1/models/hitl/multitask_weeds_hitl_<version>.pt). Рабочий чекпойнт
     (multitask_weeds_best.pt) НИКОГДА не подменяется автоматически: замена
     происходит только если вызывающий явно передал promote_if_better=True
     И комбинированная метрика кандидата не хуже baseline (decide_promotion).
     Если метрика хуже — модель не заменяется, причина фиксируется в отчёте.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from case1.ml.dataset import WeedMultiTaskDataset
from case1.ml.multitask_model import (
    WeedMultiTaskModel,
    infer_num_species_from_state_dict,
    infer_num_stages_from_state_dict,
)
from case1.ml.train import evaluate_loader

CASE1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CHECKPOINT = CASE1_DIR / "models" / "multitask_weeds_best.pt"
DEFAULT_SPECIES_MAPPING = CASE1_DIR / "models" / "species_mapping.json"
DEFAULT_ORIGINAL_MANIFEST = CASE1_DIR / "data" / "manifest_v2.csv"
DEFAULT_HITL_MANIFEST = CASE1_DIR / "data" / "manifest_hitl.csv"
DEFAULT_HITL_MODELS_DIR = CASE1_DIR / "models" / "hitl"
DEFAULT_OUTPUT_DIR = CASE1_DIR / "output"

MANIFEST_MERGE_FIELDS = ["path", "species", "stage", "source", "group_id", "split", "filename"]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Дообучение короткое и на небольшом добавке данных — аугментация мягче, чем в train.py,
# чтобы не "замыливать" немногочисленные новые примеры агронома.
FINETUNE_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_species_mapping(mapping_path: Path = DEFAULT_SPECIES_MAPPING) -> Dict[str, int]:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Species mapping не найден: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: int(v) for k, v in data["species_to_idx"].items()}


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> WeedMultiTaskModel:
    """Строит модель и загружает веса из существующего чекпойнта (без скачивания pretrained)."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    num_species = infer_num_species_from_state_dict(state)
    num_stages = infer_num_stages_from_state_dict(state)
    model = WeedMultiTaskModel(num_species=num_species, num_stages=num_stages, pretrained=False)
    model.load_state_dict(state)
    model.to(device)
    return model


def _read_manifest_rows(path: Optional[Path], allowed_species: Optional[set] = None) -> Tuple[List[Dict[str, str]], int]:
    """Читает манифест и оставляет только колонки MANIFEST_MERGE_FIELDS. Возвращает (rows, skipped_unknown_species)."""
    if path is None:
        return [], 0
    path = Path(path)
    if not path.exists():
        return [], 0
    rows: List[Dict[str, str]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row.get("species", "")
            if allowed_species is not None and species not in allowed_species:
                skipped += 1
                continue
            rows.append({field: row.get(field, "") for field in MANIFEST_MERGE_FIELDS})
    return rows, skipped


def build_merged_manifest(
    merged_path: Path,
    original_manifest_path: Optional[Path],
    hitl_manifest_path: Optional[Path],
    species_to_idx: Dict[str, int],
) -> Dict[str, int]:
    """
    Объединяет исходный манифест (manifest_v2.csv) и манифест HITL-примеров в один
    временный CSV в формате, читаемом WeedMultiTaskDataset. Строки HITL-манифеста
    с видом, отсутствующим в species_to_idx текущей модели, пропускаются (модель
    не может научиться классу, для которого нет выходного нейрона).
    """
    allowed = set(species_to_idx.keys())
    original_rows, _ = _read_manifest_rows(original_manifest_path, allowed_species=allowed)
    hitl_rows, skipped_hitl = _read_manifest_rows(hitl_manifest_path, allowed_species=allowed)

    merged_path = Path(merged_path)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_MERGE_FIELDS)
        writer.writeheader()
        for row in original_rows:
            writer.writerow(row)
        for row in hitl_rows:
            writer.writerow(row)

    return {
        "original_rows": len(original_rows),
        "hitl_rows_used": len(hitl_rows),
        "hitl_rows_skipped_unknown_species": skipped_hitl,
        "total_rows": len(original_rows) + len(hitl_rows),
    }


def combined_score(metrics: Dict[str, float]) -> float:
    """Единая шкала сравнения моделей: 0.6*species_F1 + 0.4*stage_F1 (как в train.py)."""
    return 0.6 * metrics.get("species_macro_f1", 0.0) + 0.4 * metrics.get("stage_macro_f1", 0.0)


def decide_promotion(
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    min_improvement: float = 0.0,
) -> Tuple[bool, str, float, float]:
    """
    Решает, можно ли продвинуть кандидата (baseline и candidate — метрики на одной
    и той же отложенной выборке). Возвращает (promoted, reason, baseline_score,
    candidate_score). Правило по ТЗ: модель НЕ заменяется автоматически, если
    метрика хуже -> продвижение разрешено только когда candidate_score >=
    baseline_score + min_improvement (равенство считается "не хуже").
    """
    baseline_score = combined_score(baseline_metrics)
    candidate_score = combined_score(candidate_metrics)
    delta = candidate_score - baseline_score
    if delta + 1e-9 >= min_improvement:
        reason = (
            f"Кандидат не хуже текущей модели (Δcombined={delta:+.4f} >= "
            f"min_improvement={min_improvement:.4f}): продвижение разрешено."
        )
        return True, reason, baseline_score, candidate_score
    reason = (
        f"Кандидат хуже текущей модели (Δcombined={delta:+.4f} < "
        f"min_improvement={min_improvement:.4f}): модель НЕ заменяется автоматически."
    )
    return False, reason, baseline_score, candidate_score


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    split: str,
    species_to_idx: Dict[str, int],
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Оценивает произвольный чекпойнт (species/stage F1) на указанном split манифеста."""
    model = build_model_from_checkpoint(checkpoint_path, device)
    ds = WeedMultiTaskDataset(str(manifest_path), split=split, transform=EVAL_TRANSFORM, species_to_idx=species_to_idx)
    if len(ds) == 0:
        return {"species_accuracy": 0.0, "species_macro_f1": 0.0, "stage_accuracy": 0.0, "stage_macro_f1": 0.0, "n_samples": 0}
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device)
    metrics["n_samples"] = len(ds)
    return metrics


def finetune_from_hitl(
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    species_mapping_path: Path = DEFAULT_SPECIES_MAPPING,
    original_manifest_path: Optional[Path] = DEFAULT_ORIGINAL_MANIFEST,
    hitl_manifest_path: Optional[Path] = DEFAULT_HITL_MANIFEST,
    hitl_models_dir: Path = DEFAULT_HITL_MODELS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-4,
    min_improvement: float = 0.0,
    promote_if_better: bool = False,
    device_str: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Дообучает base_checkpoint на смеси исходного датасета и HITL-примеров.

    Кандидат сохраняется всегда (версионированный файл). Продвижение в
    "боевой" чекпойнт (перезапись base_checkpoint) происходит только если
    promote_if_better=True И decide_promotion() разрешает (кандидат не хуже
    baseline). Возвращает подробный отчёт (также сохраняется в output_dir).
    """
    torch.manual_seed(seed)
    device = resolve_device(device_str)
    base_checkpoint = Path(base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Базовый чекпойнт не найден: {base_checkpoint}")

    species_to_idx = load_species_mapping(species_mapping_path)
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    hitl_models_dir = Path(hitl_models_dir)
    hitl_models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_manifest_path = hitl_models_dir / f"_merged_manifest_{version}.csv"
    merge_stats = build_merged_manifest(
        merged_manifest_path, original_manifest_path, hitl_manifest_path, species_to_idx
    )

    # --- Baseline: метрики ТЕКУЩЕГО чекпойнта на отложенной выборке (val+test) ---
    baseline_val = evaluate_checkpoint(base_checkpoint, merged_manifest_path, "val", species_to_idx, device, batch_size)
    baseline_test = evaluate_checkpoint(base_checkpoint, merged_manifest_path, "test", species_to_idx, device, batch_size)

    # --- Дообучение: старт с текущих весов, короткий прогон на train split ---
    model = build_model_from_checkpoint(base_checkpoint, device)
    train_ds = WeedMultiTaskDataset(
        str(merged_manifest_path), split="train", transform=FINETUNE_TRAIN_TRANSFORM, species_to_idx=species_to_idx
    )

    training_report: Dict[str, Any] = {"epochs_run": 0, "train_samples": len(train_ds)}

    # BatchNorm1d в головах модели требует batch size > 1 в train-режиме. При
    # малом числе примеров (типично для короткого HITL-дообучения) drop_last
    # защищает от финального "хвостового" батча из 1 элемента; при <2 образцах
    # обучать нечего — эпоха пропускается, это фиксируется в отчёте.
    if len(train_ds) >= 2:
        effective_batch_size = min(batch_size, len(train_ds))
        drop_last = len(train_ds) > effective_batch_size
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=0, drop_last=drop_last
        )
        species_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        stage_criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        start = time.time()
        epoch_losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for imgs, sp_targets, st_targets, st_masks, _ in train_loader:
                imgs = imgs.to(device)
                sp_targets = sp_targets.to(device)
                st_targets = st_targets.to(device)
                st_masks = st_masks.to(device)

                optimizer.zero_grad()
                sp_logits, st_logits = model(imgs)
                l_sp = species_criterion(sp_logits, sp_targets)
                l_st_raw = stage_criterion(st_logits, st_targets)
                if st_masks.sum() > 0:
                    l_st = (l_st_raw * st_masks).sum() / (st_masks.sum() + 1e-6)
                else:
                    l_st = torch.tensor(0.0, device=device)
                loss = l_sp + 0.85 * l_st
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(imgs)
                n_seen += len(imgs)
            epoch_losses.append(round(running_loss / max(n_seen, 1), 4))
        training_report["epochs_run"] = epochs
        training_report["training_time_s"] = round(time.time() - start, 2)
        training_report["epoch_losses"] = epoch_losses
    else:
        training_report["note"] = (
            "Недостаточно train-примеров (нужно минимум 2 для BatchNorm) — дообучение пропущено."
        )

    # --- Кандидат: сохраняем ВСЕГДА (даже если он окажется хуже) ---
    candidate_checkpoint_path = hitl_models_dir / f"multitask_weeds_hitl_{version}.pt"
    torch.save(model.state_dict(), candidate_checkpoint_path)

    candidate_val = evaluate_checkpoint(candidate_checkpoint_path, merged_manifest_path, "val", species_to_idx, device, batch_size)
    candidate_test = evaluate_checkpoint(candidate_checkpoint_path, merged_manifest_path, "test", species_to_idx, device, batch_size)

    # Продвижение оценивается на TEST (стабильнее val при малых выборках).
    promoted, reason, baseline_score, candidate_score = decide_promotion(
        baseline_test, candidate_test, min_improvement=min_improvement
    )

    applied_promotion = False
    if promote_if_better and promoted:
        shutil.copy2(candidate_checkpoint_path, base_checkpoint)
        applied_promotion = True

    report = {
        "version": version,
        "base_checkpoint": str(base_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint_path),
        "species_mapping_path": str(species_mapping_path),
        "merge_stats": merge_stats,
        "training": training_report,
        "baseline_metrics": {"val": baseline_val, "test": baseline_test},
        "candidate_metrics": {"val": candidate_val, "test": candidate_test},
        "baseline_combined_score_test": round(baseline_score, 4),
        "candidate_combined_score_test": round(candidate_score, 4),
        "promotion_allowed": promoted,
        "promotion_reason": reason,
        "promote_if_better_requested": promote_if_better,
        "promotion_applied": applied_promotion,
    }

    report_path = output_dir / f"hitl_finetune_report_{version}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)

    try:
        merged_manifest_path.unlink(missing_ok=True)
    except TypeError:
        # Python < 3.8 fallback (проект использует 3.11, но не рискуем).
        if merged_manifest_path.exists():
            merged_manifest_path.unlink()

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Дообучение классификатора на HITL-верификациях агронома")
    parser.add_argument("--epochs", type=int, default=1, help="Количество эпох дообучения (коротко, не с нуля)")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Скорость обучения (низкая, т.к. fine-tune)")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Минимальное улучшение combined score для продвижения")
    parser.add_argument("--promote", action="store_true", help="Заменить боевой чекпойнт, если кандидат не хуже baseline")
    args = parser.parse_args()

    report = finetune_from_hitl(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_improvement=args.min_improvement,
        promote_if_better=args.promote,
    )

    print("=" * 80)
    print("ДООБУЧЕНИЕ НА HITL-ВЕРИФИКАЦИЯХ АГРОНОМА")
    print("=" * 80)
    print(f"Версия кандидата:        {report['version']}")
    print(f"Кандидат сохранён в:     {report['candidate_checkpoint']}")
    print(f"Исходных строк:          {report['merge_stats']['original_rows']}")
    print(f"HITL строк использовано: {report['merge_stats']['hitl_rows_used']}")
    print(f"Baseline combined (test): {report['baseline_combined_score_test']}")
    print(f"Candidate combined (test): {report['candidate_combined_score_test']}")
    print(f"Продвижение разрешено:   {report['promotion_allowed']}")
    print(f"Причина:                 {report['promotion_reason']}")
    print(f"Продвижение применено:   {report['promotion_applied']}")
    print(f"Отчёт:                   {report['report_path']}")


if __name__ == "__main__":
    main()
