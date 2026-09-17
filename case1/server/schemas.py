"""
Pydantic схемы контрактов API для сервера классификации сорняков.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    image_id: str = Field(..., description="Имя исходного снимка дрона")
    object_id: int = Field(..., description="Порядковый номер найденного сорняка")
    bbox_xyxy: List[int] = Field(..., description="Координаты бокса [x1, y1, x2, y2]")
    det_conf: float = Field(..., description="Уверенность бортового детектора")
    drone_lat: Optional[float] = Field(None, description="Широта съемки БПЛА")
    drone_lon: Optional[float] = Field(None, description="Долгота съемки БПЛА")
    drone_rel_alt: Optional[float] = Field(None, description="Высота БПЛА над землей (м)")
    timestamp: Optional[str] = Field(None, description="Время съемки кадра")
    edge_model_version: str = Field("yolov8s-weedblaster-v1.9", description="Версия бортовой модели")
    crop_base64: Optional[str] = Field(None, description="Base64-encoded изображение вырезки")
    crop_file_path: Optional[str] = Field(None, description="Локальный путь к сохраненной вырезке")


class ClassificationResponse(BaseModel):
    object_id: int
    species: str = Field(..., description="Код вида сорняка, культуры или unknown")
    species_ru: str = Field(..., description="Русское название вида")
    species_conf: float = Field(..., description="Уверенность определения вида (0.0..1.0)")
    stage: str = Field(..., description="Фаза вегетации (rosette / stem_elongation / unknown)")
    stage_ru: str = Field(..., description="Русское название фазы")
    stage_conf: float = Field(..., description="Уверенность определения фазы (0.0..1.0)")
    spray_action: str = Field(..., description="Действие: spray_weed / do_not_spray / manual_review")
    review_required: bool = Field(..., description="Требуется ли ручная проверка агрономом")
    model_version: str = Field("server-multitask-efficientnet-b0-v1", description="Версия серверной модели")


class BatchClassificationRequest(BaseModel):
    image_id: str
    drone_lat: Optional[float] = None
    drone_lon: Optional[float] = None
    drone_rel_alt: Optional[float] = None
    timestamp: Optional[str] = None
    items: List[ClassificationRequest]


class BatchClassificationResponse(BaseModel):
    image_id: str
    total_objects: int
    review_count: int
    counts_by_species: Dict[str, int]
    counts_by_stage: Dict[str, int]
    detections: List[ClassificationResponse]
