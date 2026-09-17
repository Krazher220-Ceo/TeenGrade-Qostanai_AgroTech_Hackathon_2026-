import sys
import json
import tempfile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.demo_field import get_kostanay_demo_field
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.exporters import (
    export_fleet_plan_json,
    export_fleet_plan_geojson,
    OfflineFleetDatabase,
)


def test_export_json_and_geojson():
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        json_file = tmp_path / "mission-pack.json"
        geojson_file = tmp_path / "fleet_mission.geojson"

        # 1. JSON Export
        export_fleet_plan_json(plan, json_file)
        assert json_file.exists()
        loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded_json["plan_id"] == plan.plan_id
        assert len(loaded_json["vehicles"]) == len(plan.vehicles)
        assert loaded_json["human_confirmation_required"] is True

        # 2. GeoJSON Export
        export_fleet_plan_geojson(plan, field, geojson_file)
        assert geojson_file.exists()
        loaded_geojson = json.loads(geojson_file.read_text(encoding="utf-8"))
        assert loaded_geojson["type"] == "FeatureCollection"
        features = loaded_geojson["features"]
        assert len(features) > 0

        # Проверка типов слоев в GeoJSON
        feature_types = {f["properties"]["feature_type"] for f in features}
        assert "field_boundary" in feature_types
        assert "exclusion_zone" in feature_types
        assert "survey_lane" in feature_types
        assert "landing_pad" in feature_types
        assert "transit_route" in feature_types


def test_offline_sqlite_database():
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_fleet.db"
        db = OfflineFleetDatabase(db_path)

        db.save_plan(plan)
        loaded = db.get_plan(plan.plan_id)
        assert loaded is not None
        assert loaded["plan_id"] == plan.plan_id
        assert loaded["num_drones_assigned"] == 2
