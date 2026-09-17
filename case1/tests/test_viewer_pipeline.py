import csv
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.viewer_pipeline import run_uploaded_field_image, validate_uploaded_image


def _image_bytes(size=(96, 80), image_format="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(40, 130, 70)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_validate_uploaded_image():
    metadata = validate_uploaded_image(_image_bytes(), "field.JPG")
    assert metadata["width"] == 96
    assert metadata["height"] == 80
    assert metadata["suffix"] == ".jpg"
    assert len(metadata["sha256"]) == 64


@pytest.mark.parametrize("filename,payload", [("field.txt", b"text"), ("field.jpg", b"not-an-image")])
def test_validate_uploaded_image_rejects_invalid_input(filename, payload):
    with pytest.raises(ValueError):
        validate_uploaded_image(payload, filename)


def test_run_uploaded_field_image_collects_artifacts(tmp_path):
    def fake_processor(image_path, output_dir, detector_conf, species_conf):
        assert detector_conf == 0.65
        assert species_conf == 0.65
        output = Path(output_dir)
        annotated = output / "annotated"
        detector_boxes = output / "detector_boxes"
        annotated.mkdir(parents=True)
        detector_boxes.mkdir(parents=True)
        Image.open(image_path).save(annotated / "annotated_input.jpg")
        Image.open(image_path).save(detector_boxes / "detector_boxes_input.jpg")
        report = [{
            "filename": Path(image_path).name,
            "total_weeds": 1,
            "review_required_count": 0,
            "time_s": 0.1,
            "avg_species_conf": 0.9,
        }]
        (output / "all_fields_report.json").write_text(json.dumps(report), encoding="utf-8")
        with (output / "all_fields_detections.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["object_id", "species_ru"])
            writer.writeheader()
            writer.writerow({"object_id": 1, "species_ru": "Бодяк полевой"})
        return {"images_processed": 1, "detections_total": 1}

    result = run_uploaded_field_image(
        _image_bytes(),
        "field.jpg",
        tmp_path,
        processor=fake_processor,
    )

    assert result["summary"]["total_weeds"] == 1
    assert result["detections"][0]["species_ru"] == "Бодяк полевой"
    assert Path(result["annotated_path"]).exists()
    assert Path(result["detector_boxes_path"]).exists()
