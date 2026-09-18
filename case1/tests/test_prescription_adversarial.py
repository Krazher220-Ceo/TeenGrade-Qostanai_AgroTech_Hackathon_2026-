import json
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.verify_prescription_and_taskdata import (
    validate_geojson_file,
    validate_taskdata_file,
    run_adversarial_mutations,
    run_pipeline_regeneration,
)


def test_spot_spraying_geojson_and_taskdata_adversarial():
    geojson_path = ROOT_DIR / "case1/output/e2e_simulation/prescription.geojson"
    taskdata_path = ROOT_DIR / "case1/output/e2e_simulation/TASKDATA.XML"

    # 1. GeoJSON RFC 7946 & Domain Constraints
    res_gj = validate_geojson_file(geojson_path)
    assert res_gj["status"] == "PASS"
    assert res_gj["feature_count"] == 6

    # 2. ISO 11783-10 TaskData XML Conformance
    res_xml = validate_taskdata_file(taskdata_path, geojson_features=res_gj["features"])
    assert res_xml["status"] == "PASS"
    assert res_xml["point_count"] == 6
    assert res_xml["version_major"] == "4"

    # 3. Adversarial Mutations & Fuzzing Oracle
    raw_gj_data = json.loads(geojson_path.read_text(encoding="utf-8"))
    raw_xml_str = taskdata_path.read_text(encoding="utf-8")
    run_adversarial_mutations(raw_gj_data, raw_xml_str)


def test_pipeline_regeneration_reproducibility():
    simulation_script = ROOT_DIR / "scripts/e2e_pipeline_simulation.py"
    res = run_pipeline_regeneration(simulation_script)
    assert res["regeneration"] == "SUCCESS"
    assert res["feature_count"] == 6
    assert res["point_count"] == 6
