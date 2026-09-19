"""
Human-In-The-Loop (HITL) экспорт верификаций агронома в датасет дообучения.

Каждое решение, которое агроном подтвердил через FastAPI (POST /api/v1/verify-item
или /api/v1/sync), сохраняется сервером в реестре case1/output/verified_actions.json
("server_ledger"). Этот модуль превращает такой реестр в манифест CSV в формате,
который читает case1.ml.dataset.WeedMultiTaskDataset (колонки path,species,stage,
source,group_id,split,filename — как в case1/data/manifest_v2.csv), плюс несколько
служебных колонок (image_id, object_id, timestamp) для дедупликации и статистики.

Путь к вырезке (crop) в реестре верификаций не хранится напрямую — он ищется по
ключу "{image_id}:{object_id}" в case1/output/all_fields_detections.csv (колонка
crop_path), которую сервер обновляет той же верификацией (update_csv_with_verification
в case1/server/app.py).

Принципы:
- Решения "культура" (is_crop=True или verified_species == "crop_wheat") идут в
  датасет как отрицательный/фоновый класс crop_wheat — так же, как в исходных
  манифестах (там уже есть класс crop_wheat).
- Записи без подтверждённого вердикта — is_decisive(action) is False (пусто,
  "manual_review", "cancelled"/"отменено" и т.п.) — исключаются.
- Split назначается по кадру (image_id), а не по отдельному объекту: все объекты
  одного снимка попадают в один и тот же train/val/test, иначе один и тот же кадр
  "утекает" между обучением и отложенной выборкой.
- Дедупликация по ("image_id","object_id"): при повторной верификации побеждает
  запись с более поздним timestamp; повторный экспорт не плодит дубликаты строк.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CASE1_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = CASE1_DIR / "output"
DATA_DIR = CASE1_DIR / "data"

DEFAULT_VERIFIED_ACTIONS_PATH = OUTPUT_DIR / "verified_actions.json"
DEFAULT_DETECTIONS_CSV_PATH = OUTPUT_DIR / "all_fields_detections.csv"
DEFAULT_HITL_MANIFEST_PATH = DATA_DIR / "manifest_hitl.csv"

# Решения, которые считаются окончательным вердиктом агронома и годятся для
# дообучения. "manual_review" — объект оставлен на дальнейшую проверку (это не
# вердикт), а всё, что явно помечено отменой, вообще не должно попадать в реестр,
# но на случай появления такого статуса в будущем мы явно его исключаем здесь.
DECISIVE_ACTIONS = {"spray_weed", "do_not_spray"}
CANCELLED_ACTIONS = {"cancel", "cancelled", "canceled", "отмена", "отменено", "undo"}

# Соответствие русского названия фазы (как оно хранится в verified_stage_ru) коду
# фазы, который понимает case1.ml.dataset (STAGE_TO_IDX) и multitask_model.
STAGE_RU_TO_CODE = {
    "семядоли — 2 листа": "cotyledon_to_2_leaves",
    "семядоли - 2 листа": "cotyledon_to_2_leaves",
    "4–6 листьев": "4_to_6_leaves",
    "4-6 листьев": "4_to_6_leaves",
    "более 6 листьев / цветение": "over_6_leaves_or_flowering",
}

MANIFEST_FIELDNAMES = [
    "path", "species", "stage", "source", "group_id", "split", "filename",
    "image_id", "object_id", "verified_by", "timestamp",
]


def is_decisive(action: Optional[str]) -> bool:
    """True, если action — окончательный вердикт агронома (не отменён, не отложен)."""
    if not action:
        return False
    normalized = str(action).strip().lower()
    if normalized in CANCELLED_ACTIONS:
        return False
    return normalized in DECISIVE_ACTIONS


def load_verified_ledger(path: Path = DEFAULT_VERIFIED_ACTIONS_PATH) -> Dict[str, Dict[str, Any]]:
    """Загрузка реестра подтверждённых решений агронома (server_ledger)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_detection_index(path: Path = DEFAULT_DETECTIONS_CSV_PATH) -> Dict[str, Dict[str, str]]:
    """Индекс детекций "image_id:object_id" -> строка CSV (для поиска crop_path)."""
    path = Path(path)
    index: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('image_id', '')}:{row.get('object_id', '')}"
            index[key] = row
    return index


def _resolve_stage_code(verified_stage_ru: Optional[str]) -> str:
    if not verified_stage_ru:
        return ""
    return STAGE_RU_TO_CODE.get(str(verified_stage_ru).strip().lower(), "")


def _resolve_crop_path(det_row: Optional[Dict[str, str]], crops_root: Path) -> Optional[Path]:
    if not det_row:
        return None
    crop_rel = det_row.get("crop_path") or ""
    if not crop_rel:
        return None
    # crop_path хранится как "crops/<stem>/<file>.jpg" относительно case1/output.
    clean_rel = crop_rel[len("crops/"):] if crop_rel.startswith("crops/") else crop_rel
    candidate = crops_root / "crops" / clean_rel
    return candidate


def build_hitl_records(
    verified_actions: Dict[str, Dict[str, Any]],
    detection_index: Dict[str, Dict[str, str]],
    crops_root: Path = OUTPUT_DIR,
    require_crop_file: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Превращает реестр верификаций в список записей манифеста.

    Возвращает (records, report_counters), где report_counters содержит:
    total, excluded_not_decisive, excluded_missing_crop, included.
    """
    counters = Counter()
    counters["total"] = len(verified_actions)

    # Дедупликация: при нескольких верификациях одного объекта побеждает
    # запись с наибольшим timestamp (более свежее решение агронома).
    latest_by_key: Dict[str, Dict[str, Any]] = {}
    for key, action in verified_actions.items():
        if not is_decisive(action.get("action")):
            counters["excluded_not_decisive"] += 1
            continue
        existing = latest_by_key.get(key)
        if existing is None or str(action.get("timestamp", "")) >= str(existing.get("timestamp", "")):
            latest_by_key[key] = action

    records: List[Dict[str, Any]] = []
    for key, action in latest_by_key.items():
        image_id = action.get("image_id", "")
        object_id = action.get("object_id", "")
        det_row = detection_index.get(key) or detection_index.get(f"{image_id}:{object_id}")
        crop_path = _resolve_crop_path(det_row, crops_root)

        if require_crop_file and (crop_path is None or not crop_path.exists()):
            counters["excluded_missing_crop"] += 1
            continue

        is_crop = bool(action.get("is_crop")) or action.get("verified_species") == "crop_wheat"
        species = "crop_wheat" if is_crop else str(action.get("verified_species") or "")
        if not species:
            counters["excluded_missing_crop"] += 1
            continue

        stage_code = "" if is_crop else _resolve_stage_code(action.get("verified_stage_ru"))

        records.append({
            "path": str(crop_path) if crop_path is not None else "",
            "species": species,
            "stage": stage_code,
            "source": "hitl_verified",
            "group_id": f"hitl_{image_id}",
            "split": "",  # заполняется assign_splits_by_frame
            "filename": crop_path.name if crop_path is not None else "",
            "image_id": image_id,
            "object_id": object_id,
            "verified_by": action.get("verified_by", ""),
            "timestamp": action.get("timestamp", ""),
        })
        counters["included"] += 1

    return records, dict(counters)


def assign_splits_by_frame(
    records: List[Dict[str, Any]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Назначает split ("train"/"val"/"test") на основе image_id (кадра), а не
    отдельного объекта — так весь кадр целиком остаётся в одной выборке.
    Детерминировано (хэш image_id + seed), без утечки данных между сплитами.
    """
    import hashlib

    unique_frames = sorted({r["image_id"] for r in records})
    frame_split: Dict[str, str] = {}

    if len(unique_frames) <= 2:
        # Слишком мало кадров, чтобы честно разбить — всё идёт в train, чтобы
        # не оставить val/test пустыми "для галочки" на паре примеров.
        for frame in unique_frames:
            frame_split[frame] = "train"
    else:
        for frame in unique_frames:
            h = int(hashlib.sha256(f"{seed}:{frame}".encode("utf-8")).hexdigest(), 16)
            bucket = (h % 1000) / 1000.0
            if bucket < test_frac:
                frame_split[frame] = "test"
            elif bucket < test_frac + val_frac:
                frame_split[frame] = "val"
            else:
                frame_split[frame] = "train"

    for r in records:
        r["split"] = frame_split.get(r["image_id"], "train")


def _load_existing_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Загружает уже экспортированные записи манифеста, ключ "image_id:object_id"."""
    existing: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return existing
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('image_id', '')}:{row.get('object_id', '')}"
            existing[key] = row
    return existing


def export_hitl_dataset(
    verified_actions_path: Path = DEFAULT_VERIFIED_ACTIONS_PATH,
    detections_csv_path: Path = DEFAULT_DETECTIONS_CSV_PATH,
    output_manifest_path: Path = DEFAULT_HITL_MANIFEST_PATH,
    crops_root: Path = OUTPUT_DIR,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    require_crop_file: bool = True,
) -> Dict[str, Any]:
    """
    Полный конвейер: реестр верификаций -> манифест CSV дообучения.

    Идемпотентно: повторный запуск без новых верификаций не меняет файл и
    сообщает new_records=0. Более свежая верификация того же объекта заменяет
    старую запись (по timestamp).
    """
    output_manifest_path = Path(output_manifest_path)
    verified_actions = load_verified_ledger(verified_actions_path)
    detection_index = load_detection_index(detections_csv_path)

    records, counters = build_hitl_records(
        verified_actions, detection_index, crops_root=crops_root, require_crop_file=require_crop_file
    )
    assign_splits_by_frame(records, val_frac=val_frac, test_frac=test_frac)

    existing = _load_existing_manifest(output_manifest_path)
    new_count = 0
    updated_count = 0
    merged: Dict[str, Dict[str, Any]] = dict(existing)

    for rec in records:
        key = f"{rec['image_id']}:{rec['object_id']}"
        prev = existing.get(key)
        if prev is None:
            new_count += 1
        elif prev.get("timestamp", "") != rec.get("timestamp", "") or prev.get("species") != rec.get("species"):
            updated_count += 1
        merged[key] = rec

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for key in sorted(merged.keys()):
            row = {field: merged[key].get(field, "") for field in MANIFEST_FIELDNAMES}
            writer.writerow(row)

    species_counts = Counter(r["species"] for r in records)

    report = {
        "verified_actions_path": str(verified_actions_path),
        "output_manifest_path": str(output_manifest_path),
        "total_verified_decisions": counters.get("total", 0),
        "excluded_not_decisive": counters.get("excluded_not_decisive", 0),
        "excluded_missing_crop": counters.get("excluded_missing_crop", 0),
        "included_records": counters.get("included", 0),
        "new_records": new_count,
        "updated_records": updated_count,
        "total_records_in_manifest": len(merged),
        "new_examples_by_species": dict(species_counts),
    }
    return report


def compute_hitl_stats(
    verified_actions_path: Path = DEFAULT_VERIFIED_ACTIONS_PATH,
    output_manifest_path: Path = DEFAULT_HITL_MANIFEST_PATH,
) -> Dict[str, Any]:
    """
    Сводка для GET /api/v1/hitl/stats: сколько решений агронома накоплено,
    распределение по видам и сколько ещё не попало в манифест дообучения
    (т.е. не были экспортированы через export_hitl_dataset / hitl-export CLI).
    """
    verified_actions = load_verified_ledger(verified_actions_path)
    decisive = {k: v for k, v in verified_actions.items() if is_decisive(v.get("action"))}

    exported_keys = set()
    exported_manifest = _load_existing_manifest(output_manifest_path)
    for key in exported_manifest.keys():
        exported_keys.add(key)

    by_species: Counter = Counter()
    for v in decisive.values():
        sp_ru = v.get("verified_species_ru") or v.get("verified_species") or "Неизвестно"
        by_species[sp_ru] += 1

    pending_keys = [k for k in decisive.keys() if k not in exported_keys]

    return {
        "total_verified_decisions": len(verified_actions),
        "decisive_decisions": len(decisive),
        "excluded_decisions": len(verified_actions) - len(decisive),
        "exported_to_training": len([k for k in decisive.keys() if k in exported_keys]),
        "pending_export": len(pending_keys),
        "decisions_by_species": dict(by_species),
        "manifest_path": str(output_manifest_path),
        "manifest_exists": Path(output_manifest_path).exists(),
    }


def main() -> None:
    """CLI: python -m case1.ml.hitl_export (используется case1_main.py hitl-export)."""
    report = export_hitl_dataset()
    print("=" * 80)
    print("HITL-ЭКСПОРТ ВЕРИФИКАЦИЙ АГРОНОМА В ДАТАСЕТ ДООБУЧЕНИЯ")
    print("=" * 80)
    print(f"Реестр верификаций:        {report['verified_actions_path']}")
    print(f"Манифест дообучения:       {report['output_manifest_path']}")
    print(f"Всего решений в реестре:   {report['total_verified_decisions']}")
    print(f"Исключено (не вердикт):    {report['excluded_not_decisive']}")
    print(f"Исключено (нет вырезки):   {report['excluded_missing_crop']}")
    print(f"Готово к экспорту:         {report['included_records']}")
    print(f"Новых записей добавлено:   {report['new_records']}")
    print(f"Обновлено записей:         {report['updated_records']}")
    print(f"Итого записей в манифесте: {report['total_records_in_manifest']}")
    print("Новые примеры по видам:")
    for sp, cnt in sorted(report["new_examples_by_species"].items()):
        print(f"  - {sp}: {cnt}")


if __name__ == "__main__":
    main()
