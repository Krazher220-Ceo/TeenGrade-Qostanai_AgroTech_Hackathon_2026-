"""
Тесты новых эндпоинтов FastAPI-сервера (Кейс №1):
- GET /api/v1/hitl/stats — сводка HITL-верификаций агронома;
- GET /api/v1/perennials, GET /api/v1/perennials/compare — реестр многолетников.

Обработчики вызываются напрямую (а не через TestClient(app) как контекстный
менеджер), потому что в этом окружении checkpoint классификатора —
плейсхолдер (не настоящие веса), и FastAPI startup-событие (@app.on_event
("startup")) падает при torch.load ещё до того, как можно проверить сами
маршруты; загрузку модели ведёт отдельный поток работ (см. системный
промпт задачи). Дополнительно проверяем, что маршруты зарегистрированы в
app.routes, чтобы не потерять реальную привязку path/response_model.
"""

import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import case1.server.app as server_app
import case1.data.perennial_registry as perennial_registry_module


@pytest.mark.parametrize("path", [
    "/api/v1/hitl/stats",
    "/api/v1/perennials",
    "/api/v1/perennials/compare",
])
def test_new_routes_are_registered(path):
    route_paths = {getattr(r, "path", None) for r in server_app.app.routes}
    assert path in route_paths


def test_get_hitl_stats_uses_verified_actions_ledger(tmp_path, monkeypatch):
    verified_path = tmp_path / "verified_actions.json"
    verified_path.write_text(json.dumps({
        "img.JPG:1": {
            "image_id": "img.JPG", "object_id": 1,
            "verified_species_ru": "Бодяк полевой", "action": "spray_weed",
        },
        "img.JPG:2": {
            "image_id": "img.JPG", "object_id": 2,
            "verified_species_ru": "Пырей ползучий", "action": "manual_review",
        },
    }), encoding="utf-8")

    monkeypatch.setattr(server_app, "VERIFIED_ACTIONS_PATH", verified_path)
    # Манифест дообучения намеренно не создаём -> pending_export == decisive_decisions.
    monkeypatch.setattr(server_app, "HITL_MANIFEST_PATH", tmp_path / "manifest_hitl_that_does_not_exist.csv")

    result = server_app.get_hitl_stats()

    assert result["total_verified_decisions"] == 2
    assert result["decisive_decisions"] == 1
    assert result["excluded_decisions"] == 1
    assert result["decisions_by_species"].get("Бодяк полевой") == 1
    assert result["manifest_exists"] is False


def test_api_list_perennials_uses_isolated_registry(tmp_path, monkeypatch):
    db_path = tmp_path / "perennial_registry.sqlite"
    monkeypatch.setattr(perennial_registry_module, "DEFAULT_DB_PATH", db_path)

    registry = perennial_registry_module.PerennialRegistry(db_path=db_path)
    registry._insert_row(
        field="field_x", season="2026-06", image_id="img1", species="couch_grass",
        species_ru="Пырей ползучий", lat=52.9, lon=63.3,
        density_per_m2=1.5, occurrence_count=15, field_area_m2=10.0, source="model",
    )

    result = server_app.api_list_perennials(field="field_x", season="2026-06")
    assert result.total_items == 1
    assert result.items[0].species == "couch_grass"
    assert result.items[0].density_per_m2 == 1.5

    empty_result = server_app.api_list_perennials(field="field_x", season="2099-01")
    assert empty_result.total_items == 0


def test_api_compare_perennials_uses_isolated_registry(tmp_path, monkeypatch):
    db_path = tmp_path / "perennial_registry.sqlite"
    monkeypatch.setattr(perennial_registry_module, "DEFAULT_DB_PATH", db_path)

    registry = perennial_registry_module.PerennialRegistry(db_path=db_path)
    registry._insert_row(
        field="field_x", season="A", image_id="img1", species="couch_grass",
        species_ru="Пырей ползучий", lat=None, lon=None,
        density_per_m2=1.0, occurrence_count=10, field_area_m2=10.0, source="model",
    )
    registry._insert_row(
        field="field_x", season="B", image_id="img2", species="couch_grass",
        species_ru="Пырей ползучий", lat=None, lon=None,
        density_per_m2=2.0, occurrence_count=20, field_area_m2=10.0, source="model",
    )

    result = server_app.api_compare_perennials(field="field_x", season_a="A", season_b="B")
    assert result["field"] == "field_x"
    assert result["species_comparison"][0]["status"] == "increased"
