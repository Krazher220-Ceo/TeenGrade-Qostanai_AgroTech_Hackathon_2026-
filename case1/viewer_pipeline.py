"""Adapter for the judge-facing Streamlit upload workflow.

The module keeps upload validation, run-directory creation and artifact loading
outside Streamlit so the behavior can be unit-tested without starting a browser.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image, UnidentifiedImageError

from case1.configs.loader import get_species_confidence_threshold


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Единый порог уверенности вида (case1/configs/settings.yaml ->
# review.species_confidence_threshold), тот же, что использует CLI/сервер/дашборд.
DEFAULT_SPECIES_CONF = get_species_confidence_threshold()


def validate_uploaded_image(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Validate an uploaded field image and return safe metadata."""
    if not image_bytes:
        raise ValueError("Загружен пустой файл")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Размер изображения превышает 50 МБ")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Поддерживаются только JPG, JPEG и PNG")

    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise ValueError("Файл не является корректным JPG/PNG изображением")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = image.format
    except (UnidentifiedImageError, OSError, ModuleNotFoundError) as exc:
        raise ValueError("Файл не является корректным изображением") from exc

    if width < 64 or height < 64:
        raise ValueError("Минимальный размер изображения — 64 × 64 пикселя")

    return {
        "suffix": suffix,
        "width": width,
        "height": height,
        "format": image_format,
        "size_bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }


def run_uploaded_field_image(
    image_bytes: bytes,
    filename: str,
    output_root: Path,
    processor: Callable[..., Any] | None = None,
    detector_conf: float = 0.65,
    species_conf: float = DEFAULT_SPECIES_CONF,
    detector_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the existing detector/classifier pipeline for one uploaded image."""
    metadata = validate_uploaded_image(image_bytes, filename)
    run_id = metadata["sha256"][:12]
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / f"input{metadata['suffix']}"
    input_path.write_bytes(image_bytes)

    if processor is None:
        from case1_main import cmd_process

        processor = cmd_process

    process_result = processor(
        image_path=str(input_path),
        output_dir=str(run_dir),
        detector_path=detector_path,
        detector_conf=detector_conf,
        species_conf=species_conf,
    )
    if process_result is None:
        raise RuntimeError("Конвейер не создал результат; проверьте наличие весов моделей")

    report_path = run_dir / "all_fields_report.json"
    csv_path = run_dir / "all_fields_detections.csv"
    if not report_path.exists() or not csv_path.exists():
        raise RuntimeError("Конвейер не создал обязательные JSON/CSV артефакты")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        detections = list(csv.DictReader(handle))

    annotated_files = sorted((run_dir / "annotated").glob("annotated_*"))
    detector_box_files = sorted((run_dir / "detector_boxes").glob("detector_boxes_*"))
    if not annotated_files:
        raise RuntimeError("Конвейер не создал итоговое изображение с классификацией")
    if not detector_box_files:
        raise RuntimeError("Конвейер не создал промежуточное изображение с рамками детектора")

    summary = report[0] if report else {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_path": str(input_path),
        "detector_boxes_path": str(detector_box_files[0]),
        "annotated_path": str(annotated_files[0]),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "report": report,
        "detections": detections,
        "summary": summary,
        "agronomy_evaluation": summary.get("agronomy_evaluation", {}),
        "upload": metadata,
    }
