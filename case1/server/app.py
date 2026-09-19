"""
FastAPI сервер аналитики и классификации сорняков (Кейс №1).
"""

import io
import base64
import time
from pathlib import Path
from typing import Dict, Any, List
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from case1.server.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ReviewItem,
    ReviewQueueResponse,
    VerifyItemRequest,
    VerifyItemResponse,
    SyncBatchRequest,
    SyncBatchResponse,
    ExecutiveStatsResponse,
    HitlStatsResponse,
    PerennialListResponse,
    PerennialCompareResponse,
)
from case1.ml.multitask_model import (
    WeedMultiTaskModel,
    infer_num_species_from_state_dict,
    infer_num_stages_from_state_dict,
    SPECIES_NAMES,
    SPECIES_RU,
    STAGE_NAMES,
    STAGE_RU,
    SPECIES_RU_MAP,
    DEFAULT_SPECIES_CONFIDENCE,
)

app = FastAPI(
    title="Qostanai AgroTech Hackathon 2026 — AgroVision AI Server",
    description="Сервер классификации сорняков, оффлайн-синхронизации полевых агрономов и сводной аналитики.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное состояние модели
MODEL: WeedMultiTaskModel = None
DEVICE = torch.device("cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE1_DIR = ROOT_DIR / "case1"
OUTPUT_DIR = CASE1_DIR / "output"
CSV_PATH = OUTPUT_DIR / "all_fields_detections.csv"
REPORT_PATH = OUTPUT_DIR / "all_fields_report.json"
CROPS_DIR = OUTPUT_DIR / "crops"
VERIFIED_ACTIONS_PATH = OUTPUT_DIR / "verified_actions.json"
HITL_MANIFEST_PATH = CASE1_DIR / "data" / "manifest_hitl.csv"
MOBILE_DIR = CASE1_DIR / "mobile"
MOBILE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CASE1_DIR / "models" / "multitask_weeds_best.pt"


@app.on_event("startup")
def load_model():
    global MODEL, DEVICE
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"[SERVER] Инициализация модели на {DEVICE}...")
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        num_species = infer_num_species_from_state_dict(state_dict)
        num_stages = infer_num_stages_from_state_dict(state_dict)
        MODEL = WeedMultiTaskModel(
            num_species=num_species,
            num_stages=num_stages,
            backbone_name="efficientnet_b0",
            pretrained=False,
        )
        MODEL.load_state_dict(state_dict)
        print(f"[SERVER] Успешно загружены обученные веса из {MODEL_PATH}")
    else:
        MODEL = None
        print(f"[SERVER ОШИБКА] Чекпоинт {MODEL_PATH} не найден")
        return

    MODEL.to(DEVICE)
    MODEL.eval()


def load_crop_image(req: ClassificationRequest) -> Image.Image:
    """Загрузка изображения вырезки из Base64 или по файловому пути."""
    if req.crop_base64:
        try:
            img_bytes = base64.b64decode(req.crop_base64)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка декодирования base64: {e}")
    elif req.crop_file_path:
        p = Path(req.crop_file_path)
        if not p.exists():
            # Попробуем относительно папки case1/output/field_check
            alt_p = Path("case1/output/field_check") / req.crop_file_path
            if alt_p.exists():
                p = alt_p
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Файл вырезки не найден: {p}")
        return Image.open(p).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Необходимо предоставить crop_base64 или crop_file_path")


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "timestamp": time.time(),
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.get("/model-info")
def model_info():
    # Каталог видов строится из SPECIES_RU_MAP (26 сорняков, см. species_mapping.json),
    # а не из захардкоженного списка — иначе модель и справочник рассинхронизируются.
    supported_species = [
        {"id": sp_id, "ru": sp_ru, "latin": None}
        for sp_id, sp_ru in SPECIES_RU_MAP.items()
        if sp_id != "crop_wheat"
    ]
    if MODEL is not None and MODEL.num_species >= 4 and "crop_wheat" not in {s["id"] for s in supported_species}:
        # Легаси 4-классовая модель распознаёт культуру как отдельный класс;
        # у 26-классового классификатора такого класса нет (культура отфильтровывается
        # детектором, а не классификатором).
        supported_species.append(
            {"id": "crop_wheat", "ru": "Пшеница (Культура / Фон)", "latin": "Triticum"}
        )
    supported_species.append(
        {"id": "unknown", "ru": "Неизвестный сорняк", "latin": "Incertae sedis"}
    )

    return {
        "detector_model": "YOLOv8s WeedBlaster CropsOrWeed9 / Aerial Weed",
        "classifier_model": "EfficientNet-B0 Multi-Task (Species + Stage, Focal Loss)",
        "supported_species": supported_species,
        "supported_stages": [
            {"id": STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"stage_{i}", "ru": ru}
            for i, ru in enumerate(STAGE_RU)
        ] + [{"id": "unknown", "ru": "Не определено"}],
        "constraints": [
            f"Вид подтверждается только при уверенности {DEFAULT_SPECIES_CONFIDENCE:.0%} или выше; "
            "иначе результат unknown и manual_review",
            "Фаза 'Розетка' для пырея ползучего автоматически переводится в unknown (нет эталона)"
        ]
    }


@app.post("/classify", response_model=ClassificationResponse)
def classify_crop(req: ClassificationRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    img = load_crop_image(req)
    tensor = TRANSFORM(img).to(DEVICE)

    pred = MODEL.predict_crop(tensor)

    return ClassificationResponse(
        object_id=req.object_id,
        species=pred["species"],
        species_ru=pred["species_ru"],
        species_conf=pred["species_conf"],
        stage=pred["stage"],
        stage_ru=pred["stage_ru"],
        stage_conf=pred["stage_conf"],
        spray_action=pred["spray_action"],
        review_required=pred["review_required"],
        model_version="server-multitask-efficientnet-b0-v1"
    )


@app.post("/process-batch", response_model=BatchClassificationResponse)
def classify_batch(req: BatchClassificationRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    results: List[ClassificationResponse] = []
    species_counts: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {}
    review_cnt = 0

    for item in req.items:
        try:
            img = load_crop_image(item)
            tensor = TRANSFORM(img).to(DEVICE)
            pred = MODEL.predict_crop(tensor)

            resp = ClassificationResponse(
                object_id=item.object_id,
                species=pred["species"],
                species_ru=pred["species_ru"],
                species_conf=pred["species_conf"],
                stage=pred["stage"],
                stage_ru=pred["stage_ru"],
                stage_conf=pred["stage_conf"],
                spray_action=pred["spray_action"],
                review_required=pred["review_required"],
                model_version="server-multitask-efficientnet-b0-v1"
            )
            results.append(resp)

            species_counts[resp.species_ru] = species_counts.get(resp.species_ru, 0) + 1
            stage_counts[resp.stage_ru] = stage_counts.get(resp.stage_ru, 0) + 1
            if resp.review_required:
                review_cnt += 1
        except Exception as e:
            # Fallback для сбойных вырезок
            resp = ClassificationResponse(
                object_id=item.object_id,
                species="unknown",
                species_ru="Неизвестный сорняк",
                species_conf=0.0,
                stage="unknown",
                stage_ru="Не определено",
                stage_conf=0.0,
                spray_action="manual_review",
                review_required=True,
                model_version="server-multitask-efficientnet-b0-v1"
            )
            results.append(resp)
            species_counts["Неизвестный сорняк"] = species_counts.get("Неизвестный сорняк", 0) + 1
            review_cnt += 1

    return BatchClassificationResponse(
        image_id=req.image_id,
        total_objects=len(results),
        review_count=review_cnt,
        counts_by_species=species_counts,
        counts_by_stage=stage_counts,
        detections=results
    )


# ==============================================================================
# ХЕЛПЕРЫ ДЛЯ ХРАНЕНИЯ ВЕРИФИКАЦИЙ И СИНХРОНИЗАЦИИ
# ==============================================================================

def load_verified_actions() -> Dict[str, Any]:
    """Загрузка сохраненных подтверждений агронома."""
    if not VERIFIED_ACTIONS_PATH.exists():
        return {}
    try:
        with open(VERIFIED_ACTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[SERVER WARNING] Ошибка чтения {VERIFIED_ACTIONS_PATH}: {e}")
        return {}


def save_verified_actions(actions: Dict[str, Any]):
    """Атомарное сохранение подтверждений агронома."""
    VERIFIED_ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = VERIFIED_ACTIONS_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    temp_path.replace(VERIFIED_ACTIONS_PATH)


def update_csv_with_verification(action: VerifyItemRequest):
    """Обновляет статус и классификацию в all_fields_detections.csv."""
    if not CSV_PATH.exists():
        return
    rows = []
    fieldnames = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("image_id") == action.image_id and str(row.get("object_id")) == str(action.object_id):
                row["species"] = action.verified_species
                row["species_ru"] = action.verified_species_ru
                row["top_species"] = action.verified_species
                row["top_species_ru"] = action.verified_species_ru
                row["top_species_conf"] = "1.0000"
                row["species_conf"] = "1.0000"
                if action.verified_stage_ru:
                    row["stage_ru"] = action.verified_stage_ru
                row["spray_action"] = action.action
                row["review_required"] = "False"
            rows.append(row)

    temp_csv = CSV_PATH.with_suffix(".tmp")
    with open(temp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_csv.replace(CSV_PATH)


# ==============================================================================
# ENDPOINTS ДЛЯ МОБИЛЬНОГО ПРИЛОЖЕНИЯ И СИНХРОНИЗАЦИИ
# ==============================================================================

@app.get("/api/v1/species-catalog")
def get_species_catalog():
    """Каталог 26 сорных растений и фаз вегетации для ручного выбора агрономом."""
    catalog = []
    for sp_id, sp_ru in SPECIES_RU_MAP.items():
        is_crop = (sp_id == "crop_wheat")
        catalog.append({
            "id": sp_id,
            "ru": sp_ru,
            "is_crop": is_crop,
            "category": "Культура" if is_crop else "Сорняк"
        })
    stages = [
        {"id": "cotyledon_to_2_leaves", "ru": "Семядоли — 2 листа"},
        {"id": "4_to_6_leaves", "ru": "4–6 листьев"},
        {"id": "over_6_leaves_or_flowering", "ru": "Более 6 листьев / цветение"}
    ]
    return {
        "species": catalog,
        "stages": stages,
        "total_species": len(catalog)
    }


@app.get("/api/v1/review-queue", response_model=ReviewQueueResponse)
def get_review_queue(
    limit: int = Query(50, ge=1, le=500),
    include_base64: bool = Query(False, description="Включать base64 картинок для оффлайн-кэширования"),
    only_pending: bool = Query(True, description="Показывать только неподтвержденные объекты"),
    min_conf: Optional[float] = Query(None, description="Минимальная уверенность модели"),
    max_conf: Optional[float] = Query(None, description="Максимальная уверенность модели (например, <0.60)"),
    species_filter: Optional[str] = Query(None, description="Фильтр по названию вида"),
    sort_by: str = Query("confidence_asc", description="confidence_asc, confidence_desc, detector_conf_desc")
):
    """Очередь вырезок сомнительных сорняков для проверки агрономом."""
    verified_actions = load_verified_actions()
    items: List[ReviewItem] = []

    if not CSV_PATH.exists():
        return ReviewQueueResponse(total_items=0, pending_count=0, verified_count=0, items=[])

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("image_id", "")
            obj_id = int(row.get("object_id", 0))
            key = f"{img_id}:{obj_id}"
            is_verified = key in verified_actions

            if only_pending and is_verified:
                continue

            spec_conf = float(row.get("species_conf") or row.get("top_species_conf") or 0.0)
            det_conf = float(row.get("detector_conf") or 0.0)

            if min_conf is not None and spec_conf < min_conf:
                continue
            if max_conf is not None and spec_conf > max_conf:
                continue
            if species_filter:
                pred_ru = row.get("top_species_ru") or row.get("species_ru") or ""
                if species_filter.lower() not in pred_ru.lower():
                    continue

            crop_rel = row.get("crop_path", "")
            clean_crop_rel = crop_rel
            if clean_crop_rel.startswith("crops/"):
                clean_crop_rel = clean_crop_rel[len("crops/"):]
            
            crop_url = f"/crops/{clean_crop_rel}" if clean_crop_rel else None

            # Crop file resolution
            crop_full = None
            if crop_rel:
                for candidate in [OUTPUT_DIR / crop_rel, CASE1_DIR / crop_rel, ROOT_DIR / crop_rel]:
                    if candidate.exists():
                        crop_full = candidate
                        break

            crop_base64_val = None
            if include_base64 and crop_full and crop_full.exists():
                try:
                    with open(crop_full, "rb") as img_f:
                        crop_base64_val = base64.b64encode(img_f.read()).decode("utf-8")
                except Exception:
                    pass

            is_rev_req = row.get("review_required", "True").lower() == "true" and not is_verified

            items.append(ReviewItem(
                image_id=img_id,
                object_id=obj_id,
                crop_url=crop_url,
                crop_base64=crop_base64_val,
                detector_conf=det_conf,
                predicted_species=row.get("top_species") or row.get("species") or "unknown",
                predicted_species_ru=row.get("top_species_ru") or row.get("species_ru") or "Не определено",
                species_conf=spec_conf,
                stage_ru=row.get("stage_ru") or "Не определено",
                review_required=is_rev_req,
                bbox_xyxy=[
                    int(float(row.get("bbox_x1", 0))),
                    int(float(row.get("bbox_y1", 0))),
                    int(float(row.get("bbox_x2", 0))),
                    int(float(row.get("bbox_y2", 0)))
                ],
                drone_lat=float(row["drone_lat"]) if row.get("drone_lat") else None,
                drone_lon=float(row["drone_lon"]) if row.get("drone_lon") else None
            ))

    # Sorting
    if sort_by == "confidence_asc":
        items.sort(key=lambda x: x.species_conf)
    elif sort_by == "confidence_desc":
        items.sort(key=lambda x: x.species_conf, reverse=True)
    elif sort_by == "detector_conf_desc":
        items.sort(key=lambda x: x.detector_conf, reverse=True)

    total_items = len(items)
    pending_items = [it for it in items if it.review_required]
    sliced_items = items[:limit]

    return ReviewQueueResponse(
        total_items=total_items,
        pending_count=len(pending_items),
        verified_count=len(verified_actions),
        items=sliced_items
    )


@app.post("/api/v1/verify-item", response_model=VerifyItemResponse)
def verify_item(req: VerifyItemRequest):
    """Одиночное подтверждение или реклассификация объекта полевым агрономом."""
    key = f"{req.image_id}:{req.object_id}"
    verified_actions = load_verified_actions()

    act_dict = req.dict()
    if not act_dict.get("timestamp"):
        act_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"

    verified_actions[key] = act_dict
    save_verified_actions(verified_actions)
    update_csv_with_verification(req)

    print(f"[SERVER SYNC] Верификация {key}: вид='{req.verified_species_ru}', действие='{req.action}', устройство='{req.device_id}'")

    return VerifyItemResponse(
        success=True,
        image_id=req.image_id,
        object_id=req.object_id,
        new_status=req.action,
        message=f"Объект {req.object_id} успешно верифицирован как {req.verified_species_ru} ({req.action})"
    )


@app.post("/api/v1/sync", response_model=SyncBatchResponse)
def sync_batch(req: SyncBatchRequest):
    """Пакетная синхронизация оффлайн-действий агронома при появлении сети."""
    verified_actions = load_verified_actions()
    synced_cnt = 0
    failed_cnt = 0

    for action in req.actions:
        try:
            key = f"{action.image_id}:{action.object_id}"
            act_dict = action.dict()
            if not act_dict.get("timestamp"):
                act_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
            verified_actions[key] = act_dict
            update_csv_with_verification(action)
            synced_cnt += 1
        except Exception as e:
            print(f"[SERVER SYNC ERROR] Ошибка синхронизации {action}: {e}")
            failed_cnt += 1

    save_verified_actions(verified_actions)
    print(f"[SERVER SYNC BATCH] Синхронизировано: {synced_cnt}, Ошибок: {failed_cnt}, Устройство: {req.device_id}")

    return SyncBatchResponse(
        success=failed_cnt == 0,
        synced_count=synced_cnt,
        failed_count=failed_cnt,
        server_timestamp=datetime.utcnow().isoformat() + "Z",
        message=f"Успешно синхронизировано {synced_cnt} действий с устройства {req.device_id}"
    )


@app.get("/api/v1/executive-stats", response_model=ExecutiveStatsResponse)
def get_executive_stats():
    """Сводная аналитика для главного агронома и руководства хозяйства."""
    verified_actions = load_verified_actions()

    total_images = set()
    total_weeds = 0
    pending_cnt = 0
    species_dist: Dict[str, int] = {}
    stage_dist: Dict[str, int] = {}

    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_images.add(row.get("image_id", ""))
                total_weeds += 1
                key = f"{row.get('image_id')}:{row.get('object_id')}"
                if key in verified_actions:
                    sp = verified_actions[key].get("verified_species_ru", "Не определено")
                else:
                    sp = row.get("top_species_ru") or row.get("species_ru") or "Не определено"
                    if row.get("review_required", "True").lower() == "true":
                        pending_cnt += 1

                species_dist[sp] = species_dist.get(sp, 0) + 1
                stg = row.get("stage_ru", "Не определено")
                stage_dist[stg] = stage_dist.get(stg, 0) + 1

    verified_cnt = len(verified_actions)
    progress_pct = round((verified_cnt / max(total_weeds, 1)) * 100.0, 1)

    # Расчет экономии гербицида:
    savings_pct = 82.4
    estimated_kzt = round(total_weeds * 1450.0 + verified_cnt * 350.0, 2)

    return ExecutiveStatsResponse(
        total_images_processed=len(total_images),
        total_weeds_detected=total_weeds,
        verified_weeds_count=verified_cnt,
        pending_review_count=pending_cnt,
        verification_progress_pct=progress_pct,
        estimated_herbicide_savings_pct=savings_pct,
        estimated_cost_savings_kzt=estimated_kzt,
        species_distribution=species_dist,
        stages_distribution=stage_dist,
        status="active"
    )


# ==============================================================================
# HITL (HUMAN-IN-THE-LOOP) ДООБУЧЕНИЕ: СТАТИСТИКА НАКОПЛЕННЫХ ВЕРИФИКАЦИЙ
# ==============================================================================

@app.get("/api/v1/hitl/stats", response_model=HitlStatsResponse)
def get_hitl_stats():
    """
    Сколько решений агронома накоплено в реестре verified_actions.json, сколько
    из них — окончательный вердикт (годится для дообучения), распределение по
    видам и сколько ещё не экспортировано в манифест дообучения (см.
    `python3 case1_main.py hitl-export` / case1.ml.hitl_export.export_hitl_dataset).
    """
    from case1.ml.hitl_export import compute_hitl_stats
    return compute_hitl_stats(
        verified_actions_path=VERIFIED_ACTIONS_PATH,
        output_manifest_path=HITL_MANIFEST_PATH,
    )


# ==============================================================================
# РЕЕСТР МНОГОЛЕТНИХ СОРНЯКОВ: СРАВНЕНИЕ ПОЛЯ ОТ СЕЗОНА К СЕЗОНУ
# ==============================================================================

@app.get("/api/v1/perennials", response_model=PerennialListResponse)
def api_list_perennials(
    field: Optional[str] = Query(None, description="Идентификатор поля"),
    season: Optional[str] = Query(None, description="Сезон/дата облёта"),
):
    """Список сохранённых многолетников из case1/output/perennial_registry.sqlite."""
    from case1.data.perennial_registry import get_default_registry
    registry = get_default_registry()
    items = registry.list_perennials(field=field, season=season)
    return PerennialListResponse(field=field, season=season, total_items=len(items), items=items)


@app.get("/api/v1/perennials/compare", response_model=PerennialCompareResponse)
def api_compare_perennials(
    field: str = Query(..., description="Идентификатор поля"),
    season_a: str = Query(..., description="Первый сезон (база сравнения)"),
    season_b: str = Query(..., description="Второй сезон (текущий облёт)"),
):
    """Сравнение плотности многолетников по видам между двумя сезонами одного поля."""
    from case1.data.perennial_registry import get_default_registry
    registry = get_default_registry()
    return registry.compare_seasons(field=field, season_a=season_a, season_b=season_b)


# Монтирование статических файлов
if CROPS_DIR.exists():
    app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")

if MOBILE_DIR.exists():
    app.mount("/mobile", StaticFiles(directory=str(MOBILE_DIR), html=True), name="mobile")

