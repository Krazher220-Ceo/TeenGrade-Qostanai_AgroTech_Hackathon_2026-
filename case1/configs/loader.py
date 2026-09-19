"""
Централизованная загрузка настроек конвейера Кейса №1 (case1/configs/settings.yaml).

Единая точка чтения порога уверенности вида сорняка (review.species_confidence_threshold),
используемая классификатором, CLI (case1_main.py), Streamlit-дашбордом, FastAPI-сервером
и адаптером загрузки (viewer_pipeline.py), чтобы избежать рассинхронизации порогов
между компонентами.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"

# Значение по умолчанию, если ключ отсутствует в settings.yaml или файл недоступен.
# Всё ниже этого порога уверенности переводится в статус unknown / manual_review
# и уходит агроному на ручную проверку (см. питч: "всё ниже 75% уверенности уходит агроному").
DEFAULT_SPECIES_CONFIDENCE_THRESHOLD = 0.75


def load_settings(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Загружает case1/configs/settings.yaml целиком. Возвращает {} при ошибке чтения."""
    path = Path(settings_path) if settings_path else SETTINGS_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}


def get_species_confidence_threshold(settings_path: Optional[Path] = None) -> float:
    """
    Единый порог уверенности вида сорняка (review.species_confidence_threshold).

    Это единственная функция чтения порога — все компоненты конвейера
    (case1/ml/multitask_model.py, case1_main.py, case1/viewer_pipeline.py,
    case1/server/app.py, case1/dashboard/app.py) должны получать значение через неё,
    а не хранить собственную константу.
    """
    settings = load_settings(settings_path)
    review_cfg = settings.get("review", {}) if isinstance(settings, dict) else {}
    if not isinstance(review_cfg, dict):
        return DEFAULT_SPECIES_CONFIDENCE_THRESHOLD
    try:
        return float(review_cfg.get("species_confidence_threshold", DEFAULT_SPECIES_CONFIDENCE_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_SPECIES_CONFIDENCE_THRESHOLD
