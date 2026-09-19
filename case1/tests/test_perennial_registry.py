"""
Тесты реестра многолетних сорняков (case1/data/perennial_registry.py):
- ингест отчёта детекций отбирает только многолетники (единый справочник
  PERENNIAL_SPECIES_MARKERS/CODES из case1.fleet.agronomy_rules), исключая
  культуру, малолетники и сомнительные (review_required) детекции;
- сравнение двух сезонов находит новые/исчезнувшие очаги и изменение плотности;
- демо-сезон нельзя замаскировать под реальный источник данных.
"""

import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from case1.data.perennial_registry import PerennialRegistry, is_perennial_species


def _write_report(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)


@pytest.fixture
def registry(tmp_path):
    db_path = tmp_path / "perennial_registry.sqlite"
    return PerennialRegistry(db_path=db_path)


@pytest.fixture
def sample_report(tmp_path):
    report = [
        {
            "filename": "frame1.JPG",
            "lat": 52.9, "lon": 63.3,
            "agronomy_evaluation": {"field_area_m2": 10.0},
            "detections": [
                {"species": "couch_grass", "review_required": False},   # perennial
                {"species": "couch_grass", "review_required": False},   # perennial
                {"species": "field_thistle", "review_required": False},  # perennial
                {"species": "crop_wheat", "review_required": False},    # culture -> excluded
                {"species": "unknown", "review_required": True},        # unknown -> excluded
            ],
        },
        {
            "filename": "frame2.JPG",
            "lat": 52.91, "lon": 63.31,
            "agronomy_evaluation": {"field_area_m2": 5.0},
            "detections": [
                {"species": "field_bindweed", "review_required": False},  # perennial
            ],
        },
    ]
    path = tmp_path / "all_fields_report.json"
    _write_report(path, report)
    return path


# ------------------------------------------------------------------------------
# is_perennial_species
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("species", ["couch_grass", "field_thistle", "field_bindweed", "Пырей ползучий"])
def test_is_perennial_species_true_for_known_perennials(species):
    assert is_perennial_species(species) is True


@pytest.mark.parametrize("species", ["crop_wheat", "wild_oat", "barnyard_grass", "unknown", "", None])
def test_is_perennial_species_false_for_annual_or_crop(species):
    assert is_perennial_species(species) is False


# ------------------------------------------------------------------------------
# ingest_field_report
# ------------------------------------------------------------------------------

def test_ingest_field_report_only_keeps_perennials(registry, sample_report):
    report = registry.ingest_field_report(
        report_path=sample_report, field="field_x", season="2026-06", source="model"
    )
    assert report["rows_inserted"] == 3  # couch_grass, field_thistle, field_bindweed
    assert report["counts_by_species"]["couch_grass"] == 2
    assert report["counts_by_species"]["field_thistle"] == 1
    assert report["counts_by_species"]["field_bindweed"] == 1
    assert "crop_wheat" not in report["counts_by_species"]
    assert "unknown" not in report["counts_by_species"]

    rows = registry.list_perennials(field="field_x", season="2026-06")
    assert len(rows) == 3
    couch_row = next(r for r in rows if r["species"] == "couch_grass")
    assert couch_row["occurrence_count"] == 2
    assert couch_row["density_per_m2"] == pytest.approx(2 / 10.0)
    assert couch_row["species_ru"] == "Пырей ползучий"
    assert couch_row["source"] == "model"


def test_ingest_field_report_replaces_existing_rows_on_rerun(registry, sample_report):
    registry.ingest_field_report(report_path=sample_report, field="field_x", season="2026-06", source="model")
    registry.ingest_field_report(report_path=sample_report, field="field_x", season="2026-06", source="model")
    rows = registry.list_perennials(field="field_x", season="2026-06")
    # Повторный ингест не должен дублировать строки.
    assert len(rows) == 3


def test_list_fields_and_seasons(registry, sample_report):
    registry.ingest_field_report(report_path=sample_report, field="field_x", season="2026-06", source="model")
    assert registry.list_fields() == ["field_x"]
    assert registry.list_seasons(field="field_x") == ["2026-06"]


# ------------------------------------------------------------------------------
# seed_demo_season
# ------------------------------------------------------------------------------

def test_seed_demo_season_rejects_real_source_labels(registry):
    with pytest.raises(ValueError):
        registry.seed_demo_season("field_x", "2025-06", {"couch_grass": 1.0}, source="model")
    with pytest.raises(ValueError):
        registry.seed_demo_season("field_x", "2025-06", {"couch_grass": 1.0}, source="agronomist")


def test_seed_demo_season_marks_rows_as_demo(registry):
    report = registry.seed_demo_season(
        "field_x", "2025-06-demo", {"couch_grass": 0.4, "field_thistle": 0.1},
        field_area_m2=10.0,
    )
    assert report["is_demo"] is True
    rows = registry.list_perennials(field="field_x", season="2025-06-demo")
    assert len(rows) == 2
    assert all(r["source"] == "demo_synthetic" for r in rows)


# ------------------------------------------------------------------------------
# compare_seasons
# ------------------------------------------------------------------------------

def test_compare_seasons_detects_new_and_disappeared_foci(registry):
    # Сезон A: только пырей.
    registry._insert_row(
        field="field_x", season="A", image_id="imgA", species="couch_grass",
        species_ru="Пырей ползучий", lat=52.9, lon=63.3,
        density_per_m2=1.0, occurrence_count=10, field_area_m2=10.0, source="model",
    )
    # Сезон B: пырея больше нет, зато появился бодяк.
    registry._insert_row(
        field="field_x", season="B", image_id="imgB", species="field_thistle",
        species_ru="Бодяк полевой", lat=52.9, lon=63.3,
        density_per_m2=2.0, occurrence_count=20, field_area_m2=10.0, source="model",
    )

    result = registry.compare_seasons("field_x", "A", "B")
    assert result["new_foci"] == ["field_thistle"]
    assert result["disappeared_foci"] == ["couch_grass"]

    by_species = {c["species"]: c for c in result["species_comparison"]}
    assert by_species["couch_grass"]["status"] == "disappeared"
    assert by_species["couch_grass"]["density_season_b"] == 0.0
    assert by_species["field_thistle"]["status"] == "new_focus"
    assert by_species["field_thistle"]["density_season_a"] == 0.0


def test_compare_seasons_computes_density_change_for_persistent_species(registry):
    registry._insert_row(
        field="field_x", season="A", image_id="imgA", species="couch_grass",
        species_ru="Пырей ползучий", lat=None, lon=None,
        density_per_m2=1.0, occurrence_count=10, field_area_m2=10.0, source="model",
    )
    registry._insert_row(
        field="field_x", season="B", image_id="imgB", species="couch_grass",
        species_ru="Пырей ползучий", lat=None, lon=None,
        density_per_m2=2.0, occurrence_count=20, field_area_m2=10.0, source="model",
    )
    result = registry.compare_seasons("field_x", "A", "B")
    entry = result["species_comparison"][0]
    assert entry["species"] == "couch_grass"
    assert entry["density_season_a"] == 1.0
    assert entry["density_season_b"] == 2.0
    assert entry["delta_density"] == 1.0
    assert entry["delta_pct"] == 100.0
    assert entry["status"] == "increased"


def test_compare_seasons_empty_when_no_data(registry):
    result = registry.compare_seasons("field_x", "A", "B")
    assert result["species_comparison"] == []
    assert result["new_foci"] == []
    assert result["disappeared_foci"] == []
