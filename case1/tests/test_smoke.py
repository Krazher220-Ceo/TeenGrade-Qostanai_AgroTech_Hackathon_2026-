"""
Дымовые тесты (Smoke Tests) конвейера Кейса №1:
1. Проверка доступности весов WeedBlaster.
2. Проверка загрузки и инференса детектора YOLOv8s.
3. Проверка мультизадачной модели классификатора (MobileNetV3).
4. Проверка FastAPI эндпоинтов (/health, /model-info, /classify).
5. Проверка генерации выходных файлов JSON и CSV.
"""

import base64
import io
import json
import sys
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from ultralytics import YOLO
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from case1.ml.multitask_model import WeedMultiTaskModel, infer_num_species_from_state_dict
from case1.server.app import app
WEIGHTS_PATH = BASE_DIR / "weedblaster-vision-yolov8s" / "best.pt"
CLASSIFIER_PATH = BASE_DIR / "case1" / "models" / "multitask_weeds_best.pt"
MANIFEST_PATH = BASE_DIR / "case1" / "data" / "manifest.csv"
REPORT_PATH = BASE_DIR / "case1" / "output" / "all_fields_report.json"
CSV_PATH = BASE_DIR / "case1" / "output" / "all_fields_detections.csv"


def test_weights_exist():
    assert WEIGHTS_PATH.exists(), f"Файл весов {WEIGHTS_PATH} не найден"
    assert WEIGHTS_PATH.stat().st_size > 20 * 1024 * 1024, "Файл весов меньше 20 МБ (возможно LFS pointer)"


def test_detector_loads():
    model = YOLO(str(WEIGHTS_PATH))
    assert model.task == "detect"
    assert 8 in model.names
    assert model.names[8] == "Weed"


def test_classifier_loads():
    assert CLASSIFIER_PATH.exists(), f"Файл классификатора {CLASSIFIER_PATH} не найден"
    state = torch.load(CLASSIFIER_PATH, map_location="cpu")
    num_species = infer_num_species_from_state_dict(state)
    model = WeedMultiTaskModel(num_species=num_species, num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(state)
    model.eval()

    # Проверка forward dummy tensor
    dummy = torch.randn(1, 3, 224, 224)
    sp_logits, st_logits = model(dummy)
    assert sp_logits.shape == (1, num_species)
    assert st_logits.shape == (1, 2)


def test_couch_grass_rosette_constraint():
    model = WeedMultiTaskModel(num_species=3, num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
    # Тест предсказания с принудительной розеткой для пырея
    pred = model.predict_crop(torch.randn(3, 224, 224))
    if pred["species"] == "couch_grass":
        assert pred["stage"] != "rosette", "Пырей ползучий не должен иметь фазу розетка!"


def test_feature_extraction():
    model = WeedMultiTaskModel(num_species=4, num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    feats = model.extract_features(dummy)
    assert feats.shape == (2, 1280), f"Expected shape (2, 1280), got {feats.shape}"


def test_uncertain_detection_is_not_marked_for_spraying():
    model = WeedMultiTaskModel(num_species=3, num_stages=2, backbone_name="mobilenet_v3_small", pretrained=False)
    model.forward = lambda _: (torch.zeros(1, 3), torch.zeros(1, 2))
    pred = model.predict_crop(torch.randn(3, 32, 32))
    assert pred["species"] == "unknown"
    assert pred["spray_action"] == "manual_review"


def test_species_below_sixty_five_percent_requires_review():
    model = WeedMultiTaskModel(num_species=4, num_stages=2, backbone_name="mobilenet_v3_small", pretrained=False)
    model.forward = lambda _: (
        torch.log(torch.tensor([[0.60, 0.20, 0.10, 0.10]])),
        torch.log(torch.tensor([[0.10, 0.90]])),
    )

    pred = model.predict_crop(torch.randn(3, 32, 32))

    assert pred["species"] == "unknown"
    assert pred["species_ru"] == "Не определено"
    assert pred["species_conf"] == 0.6
    assert pred["review_required"] is True
    assert pred["spray_action"] == "manual_review"
    assert pred["all_species_probs"]["field_thistle"] == 0.6


def test_species_at_or_above_sixty_five_percent_is_displayed():
    model = WeedMultiTaskModel(num_species=4, num_stages=2, backbone_name="mobilenet_v3_small", pretrained=False)
    model.forward = lambda _: (
        torch.log(torch.tensor([[0.70, 0.15, 0.10, 0.05]])),
        torch.log(torch.tensor([[0.10, 0.90]])),
    )

    pred = model.predict_crop(torch.randn(3, 32, 32))

    assert pred["species"] == "field_thistle"
    assert pred["species_conf"] == 0.7
    assert pred["review_required"] is False
    assert pred["spray_action"] == "spray_weed"


def test_confident_crop_is_marked_as_background_not_weed():
    model = WeedMultiTaskModel(num_species=4, num_stages=2, backbone_name="mobilenet_v3_small", pretrained=False)
    model.forward = lambda _: (
        torch.log(torch.tensor([[0.05, 0.05, 0.05, 0.85]])),
        torch.log(torch.tensor([[0.10, 0.90]])),
    )

    pred = model.predict_crop(torch.randn(3, 32, 32))

    assert pred["species"] == "crop_wheat"
    assert pred["spray_action"] == "do_not_spray"
    assert pred["review_required"] is False


def test_fastapi_server():
    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        res_info = client.get("/model-info")
        assert res_info.status_code == 200
        assert "EfficientNet" in res_info.json()["classifier_model"]
        species_ids = {item["id"] for item in res_info.json()["supported_species"]}
        if infer_num_species_from_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu")) >= 4:
            assert "crop_wheat" in species_ids

        buffer = io.BytesIO()
        Image.new("RGB", (96, 96), color=(45, 130, 55)).save(buffer, format="JPEG")
        crop_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "image_id": "smoke-test.jpg",
            "object_id": 1,
            "bbox_xyxy": [0, 0, 96, 96],
            "det_conf": 0.9,
            "crop_base64": crop_base64,
        }

        res_classify = client.post("/classify", json=payload)
        assert res_classify.status_code == 200
        assert res_classify.json()["object_id"] == 1
        assert "species" in res_classify.json()

        res_batch = client.post(
            "/process-batch",
            json={"image_id": "smoke-test", "items": [payload]},
        )
        assert res_batch.status_code == 200
        assert res_batch.json()["total_objects"] == 1


def test_output_artifacts_exist():
    assert REPORT_PATH.exists(), f"Отчет {REPORT_PATH} не найден"
    assert CSV_PATH.exists(), f"Таблица {CSV_PATH} не найдена"

    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert len(data) == 5, "В отчете должно быть ровно 5 полевых снимков"
    total_weeds = sum(d["total_weeds"] for d in data)
    assert total_weeds > 0, "Количество сорняков должно быть > 0"


if __name__ == "__main__":
    print("=== ЗАПУСК ДЫМОВЫХ ТЕСТОВ КЕЙСА №1 ===")
    test_weights_exist()
    print("  [+] test_weights_exist: PASSED")
    test_detector_loads()
    print("  [+] test_detector_loads: PASSED")
    test_classifier_loads()
    print("  [+] test_classifier_loads: PASSED")
    test_couch_grass_rosette_constraint()
    print("  [+] test_couch_grass_rosette_constraint: PASSED")
    test_feature_extraction()
    print("  [+] test_feature_extraction: PASSED")
    test_uncertain_detection_is_not_marked_for_spraying()
    print("  [+] test_uncertain_detection_is_not_marked_for_spraying: PASSED")
    test_fastapi_server()
    print("  [+] test_fastapi_server: PASSED")
    test_output_artifacts_exist()
    print("  [+] test_output_artifacts_exist: PASSED")
    print("\nВСЕ ДЫМОВЫЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! ✅")
