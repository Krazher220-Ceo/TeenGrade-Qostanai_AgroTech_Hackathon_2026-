"""
FastAPI сервер аналитики и классификации сорняков (Кейс №1).
"""

import io
import base64
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, Request

from case1.server.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse
)
from case1.ml.multitask_model import (
    WeedMultiTaskModel,
    infer_num_species_from_state_dict,
    SPECIES_NAMES,
    SPECIES_RU,
    STAGE_NAMES,
    STAGE_RU
)

app = FastAPI(
    title="Qostanai AgroTech Hackathon 2026 — Weed Classification Server",
    description="Серверная модель детальной классификации видов и фаз сорняков с контролем неопределенности.",
    version="1.0.0"
)

# Глобальное состояние модели
MODEL: WeedMultiTaskModel = None
DEVICE = torch.device("cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "multitask_weeds_best.pt"


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
        MODEL = WeedMultiTaskModel(num_species=num_species, num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
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
    supported_species = [
        {"id": "field_thistle", "ru": "Бодяк полевой", "latin": "Cirsium arvense"},
        {"id": "field_bindweed", "ru": "Вьюнок полевой", "latin": "Convolvulus arvensis"},
        {"id": "couch_grass", "ru": "Пырей ползучий", "latin": "Elymus repens"},
    ]
    if MODEL is not None and MODEL.num_species >= 4:
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
            {"id": "rosette", "ru": "Розетка"},
            {"id": "stem_elongation", "ru": "Стеблевание"},
            {"id": "unknown", "ru": "Не определено"}
        ],
        "constraints": [
            "Вид подтверждается только при уверенности 70% или выше; иначе результат unknown и manual_review",
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
