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

# scripts/e2e_pipeline_simulation.py is fully deterministic (fixed SEED,
# np.random.default_rng(SEED)): 6 synthetic truth objects are generated
# (field_thistle, field_bindweed, crop_wheat, couch_grass, wild_oat,
# redroot_pigweed), crop_wheat is excluded from spray treatment zones
# (case1/geo/zones.py only aggregates action in {"spray", "spray_weed"}),
# and none of the 5 remaining weed detections are close enough to merge
# into a shared zone -> exactly 5 non-overlapping treatment zones, and
# therefore exactly 5 <TZN> elements in the generated TASKDATA.XML. These
# exact counts (not a loose ">= 4") are the whole point of a deterministic
# reproducibility harness: a silent drop in emitted zones/points must fail
# the test, not slide under a loose lower bound.
EXPECTED_ZONE_COUNT = 5


def test_spot_spraying_geojson_and_taskdata_adversarial():
    geojson_path = ROOT_DIR / "case1/output/e2e_simulation/prescription.geojson"
    taskdata_path = ROOT_DIR / "case1/output/e2e_simulation/TASKDATA.XML"

    # 1. GeoJSON RFC 7946 & Domain Constraints (treatment-zone polygons)
    res_gj = validate_geojson_file(geojson_path)
    assert res_gj["status"] == "PASS"
    assert res_gj["feature_count"] == EXPECTED_ZONE_COUNT

    # 2. ISO 11783-10 TaskData XML Conformance (TZN boundaries + GRD grid)
    res_xml = validate_taskdata_file(taskdata_path, geojson_features=res_gj["features"])
    assert res_xml["status"] == "PASS"
    assert res_xml["zone_count"] == EXPECTED_ZONE_COUNT
    assert res_xml["version_major"] == "4"
    assert res_xml["grid"]["cols"] > 0 and res_xml["grid"]["rows"] > 0
    assert "raw" in res_xml["grid"], "GRD00001.bin must be resolvable next to TASKDATA.XML"

    # 3. Adversarial Mutations & Fuzzing Oracle (GeoJSON + XML + GRD binary)
    raw_gj_data = json.loads(geojson_path.read_text(encoding="utf-8"))
    raw_xml_str = taskdata_path.read_text(encoding="utf-8")
    mutations_tested = run_adversarial_mutations(raw_gj_data, raw_xml_str, taskdata_dir=taskdata_path.parent)
    # 13 mutations: 6 GeoJSON-level (crop-sprayed, zero-rate, oob lon/lat,
    # inverted coords, bad geometry type) + 3 XML-level (version, zone
    # count, PDV rate) + 1 GRD-missing + 3 GRD-binary-file-level (corrupted
    # size, missing file, invalid TreatmentZoneCode).
    assert mutations_tested == 13


def test_pipeline_regeneration_reproducibility():
    simulation_script = ROOT_DIR / "scripts/e2e_pipeline_simulation.py"
    res = run_pipeline_regeneration(simulation_script)
    assert res["regeneration"] == "SUCCESS"
    assert res["feature_count"] == EXPECTED_ZONE_COUNT
    assert res["zone_count"] == EXPECTED_ZONE_COUNT
