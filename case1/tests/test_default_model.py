"""
Тесты дефолтной модели классификатора и единого порога уверенности вида (Кейс №1).

Контекст: в питче жюри заявлены "26 видов сорняков и 3 фазы вегетации,
98.4% / 98.7% точности; всё ниже 75% уверенности уходит агроному". Эти тесты
фиксируют, что именно эта (серверная, 26-классовая) модель загружается по
умолчанию во всех компонентах конвейера, а не старый 4-классовый чекпоинт, и
что порог уверенности 0.75 читается из единого конфига, а не разбросан по
модулям как отдельные константы.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.configs.loader import get_species_confidence_threshold
from case1.ml.multitask_model import (
    DEFAULT_SPECIES_CONFIDENCE,
    WeedMultiTaskModel,
    infer_num_species_from_state_dict,
    infer_num_stages_from_state_dict,
)

CASE1_DIR = ROOT_DIR / "case1"
MODELS_DIR = CASE1_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "multitask_weeds_best.pt"
LEGACY_MODEL_PATH = MODELS_DIR / "multitask_weeds_legacy_4cls.pt"
SPECIES_MAPPING_PATH = MODELS_DIR / "species_mapping.json"

EXPECTED_NUM_SPECIES = 26
EXPECTED_NUM_STAGES = 3


def test_default_model_path_is_the_26_species_checkpoint():
    """case1/models/multitask_weeds_best.pt (путь по умолчанию во всех модулях)
    обязан быть 26-видовым/3-фазным серверным чекпоинтом, а не старым
    4-видовым/2-фазным."""
    assert DEFAULT_MODEL_PATH.exists(), f"Дефолтный чекпоинт {DEFAULT_MODEL_PATH} не найден"
    state_dict = torch.load(DEFAULT_MODEL_PATH, map_location="cpu")

    num_species = infer_num_species_from_state_dict(state_dict)
    num_stages = infer_num_stages_from_state_dict(state_dict)

    assert num_species == EXPECTED_NUM_SPECIES
    assert num_stages == EXPECTED_NUM_STAGES

    # Модель обязана реально загружаться этой архитектурой без сюрпризов в форме тензоров.
    model = WeedMultiTaskModel(
        num_species=num_species,
        num_stages=num_stages,
        backbone_name="efficientnet_b0",
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    sp_logits, st_logits = model(dummy)
    assert sp_logits.shape == (1, EXPECTED_NUM_SPECIES)
    assert st_logits.shape == (1, EXPECTED_NUM_STAGES)


def test_legacy_4class_checkpoint_still_available_but_not_default():
    """Старый 4-классовый чекпоинт переименован, а не удалён, и не совпадает
    с дефолтным путём — чтобы явно нельзя было случайно загрузить его как
    основную модель."""
    assert LEGACY_MODEL_PATH.exists(), f"Легаси чекпоинт {LEGACY_MODEL_PATH} не найден"
    assert LEGACY_MODEL_PATH != DEFAULT_MODEL_PATH

    state_dict = torch.load(LEGACY_MODEL_PATH, map_location="cpu")
    assert infer_num_species_from_state_dict(state_dict) == 4
    assert infer_num_stages_from_state_dict(state_dict) == 2


def test_species_mapping_json_matches_26_species_3_stages():
    """species_mapping.json (используется predict_crop для RU-имён/индексов)
    должен лежать рядом с дефолтной моделью и описывать те же 26 видов и 3 фазы."""
    import json

    assert SPECIES_MAPPING_PATH.exists(), f"{SPECIES_MAPPING_PATH} не найден"
    with open(SPECIES_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert mapping["num_classes"] == EXPECTED_NUM_SPECIES
    assert mapping["num_stages"] == EXPECTED_NUM_STAGES
    assert len(mapping["species_to_idx"]) == EXPECTED_NUM_SPECIES
    assert len(mapping["idx_to_species"]) == EXPECTED_NUM_SPECIES
    assert len(mapping["stage_names"]) == EXPECTED_NUM_STAGES

    # Пшеница как культура НЕ входит в 26 классов сорняков — классификатор
    # видит только подтверждённые детекции сорняков (культура отфильтровывается
    # на уровне детектора, а не как один из классов классификатора).
    assert "crop_wheat" not in mapping["species_to_idx"]
    # "Падалица пшеницы" (самосев/сорная пшеница) — это отдельный от культуры
    # сорный вид и присутствует в списке.
    assert "volunteer_wheat" in mapping["species_to_idx"]


def test_species_confidence_threshold_is_075_from_config():
    """Единый порог (review.species_confidence_threshold в case1/configs/settings.yaml)
    равен 0.75, как заявлено в питче ('всё ниже 75% уверенности уходит агроному')."""
    assert get_species_confidence_threshold() == pytest.approx(0.75)


def test_multitask_model_default_species_confidence_matches_config():
    """DEFAULT_SPECIES_CONFIDENCE в case1.ml.multitask_model обязан браться из
    единого конфига, а не быть отдельной захардкоженной константой."""
    assert DEFAULT_SPECIES_CONFIDENCE == get_species_confidence_threshold()
    assert DEFAULT_SPECIES_CONFIDENCE == pytest.approx(0.75)


def test_species_confidence_threshold_missing_key_falls_back_to_075(tmp_path):
    """Если ключ review.species_confidence_threshold отсутствует в переданном
    файле настроек, функция обязана вернуть безопасный дефолт 0.75, а не упасть."""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("project:\n  name: test\n", encoding="utf-8")
    assert get_species_confidence_threshold(settings_file) == pytest.approx(0.75)


def test_species_confidence_threshold_reads_custom_value(tmp_path):
    """Функция обязана честно вернуть значение, заданное в конфиге, а не
    игнорировать его в пользу дефолта."""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        "review:\n  species_confidence_threshold: 0.82\n", encoding="utf-8"
    )
    assert get_species_confidence_threshold(settings_file) == pytest.approx(0.82)
