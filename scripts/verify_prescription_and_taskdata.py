#!/usr/bin/env python3
"""Adversarial stress-test and verification harness for Spot-Spraying artifacts.

Validates:
1. RFC 7946 GeoJSON compliance, Kostanay bounding box, Point geometry, agronomic rate invariants.
2. ISO 11783-10 TaskData XML schema, root version, point counts, coordinates, and PDV rates.
3. Cross-artifact consistency between GeoJSON and ISO-XML.
4. Mutation & Fuzzing Oracle: confirms that adversarial mutations are strictly rejected.
5. End-to-end pipeline reproducibility.

Usage:
    .venv/bin/python scripts/verify_prescription_and_taskdata.py
    .venv/bin/python scripts/verify_prescription_and_taskdata.py --geojson case1/output/e2e_simulation/prescription.geojson --taskdata case1/output/e2e_simulation/TASKDATA.XML
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Kostanay Oblast geographical bounding box
KOSTANAY_LON_MIN = 60.0
KOSTANAY_LON_MAX = 70.0
KOSTANAY_LAT_MIN = 50.0
KOSTANAY_LAT_MAX = 56.0

STANDARD_SPRAY_RATE_L_HA = 150.0
ZERO_SPRAY_RATE_L_HA = 0.0
ISOXML_STANDARD_RATE_ML_HA = 150000


class ValidationError(Exception):
    """Raised when an artifact fails adversarial validation."""


def validate_geojson_data(data: Dict[str, Any], context: str = "GeoJSON") -> Dict[str, Any]:
    """Validate GeoJSON data against RFC 7946 and domain-specific invariants."""
    if not isinstance(data, dict):
        raise ValidationError(f"[{context}] Top-level object must be a JSON dict, got {type(data).__name__}")

    if data.get("type") != "FeatureCollection":
        raise ValidationError(f"[{context}] RFC 7946 violation: root 'type' must be 'FeatureCollection', got {data.get('type')!r}")

    features = data.get("features")
    if not isinstance(features, list):
        raise ValidationError(f"[{context}] RFC 7946 violation: 'features' must be a list, got {type(features).__name__}")

    if len(features) == 0:
        raise ValidationError(f"[{context}] 'features' list is empty. Spot-spraying prescription must contain target points.")

    seen_ids = set()
    validated_features = []

    for idx, feat in enumerate(features):
        feat_ctx = f"{context} feature[{idx}]"
        if not isinstance(feat, dict):
            raise ValidationError(f"[{feat_ctx}] Must be a dict, got {type(feat).__name__}")

        if feat.get("type") != "Feature":
            raise ValidationError(f"[{feat_ctx}] RFC 7946 violation: 'type' must be 'Feature', got {feat.get('type')!r}")

        feat_id = feat.get("id")
        if not feat_id or not isinstance(feat_id, str):
            raise ValidationError(f"[{feat_ctx}] Feature 'id' must be a non-empty string, got {feat_id!r}")
        if feat_id in seen_ids:
            raise ValidationError(f"[{feat_ctx}] Duplicate feature id detected: {feat_id}")
        seen_ids.add(feat_id)

        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            raise ValidationError(f"[{feat_ctx}] 'geometry' must be a dict, got {type(geom).__name__}")

        geom_type = geom.get("type")
        if geom_type != "Point":
            raise ValidationError(f"[{feat_ctx}] Geometry type must be 'Point' for spot-spraying targets, got {geom_type!r}")

        coords = geom.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValidationError(f"[{feat_ctx}] 'coordinates' must be [lon, lat] of length 2, got {coords!r}")

        lon, lat = coords
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            raise ValidationError(f"[{feat_ctx}] Coordinates must be numeric floats, got {type(lon).__name__}, {type(lat).__name__}")

        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValidationError(f"[{feat_ctx}] Coordinates must be finite numbers, got lon={lon}, lat={lat}")

        # Bounding box check for Kostanay region (lon 60-70, lat 50-56)
        if not (KOSTANAY_LON_MIN <= lon <= KOSTANAY_LON_MAX):
            raise ValidationError(
                f"[{feat_ctx}] Longitude {lon:.7f} outside Kostanay region bounds [{KOSTANAY_LON_MIN}, {KOSTANAY_LON_MAX}]"
            )
        if not (KOSTANAY_LAT_MIN <= lat <= KOSTANAY_LAT_MAX):
            raise ValidationError(
                f"[{feat_ctx}] Latitude {lat:.7f} outside Kostanay region bounds [{KOSTANAY_LAT_MIN}, {KOSTANAY_LAT_MAX}]"
            )

        props = feat.get("properties")
        if not isinstance(props, dict):
            raise ValidationError(f"[{feat_ctx}] 'properties' must be a dict, got {type(props).__name__}")

        # Required properties
        for req_prop in ("species", "action", "rate_l_ha"):
            if req_prop not in props:
                raise ValidationError(f"[{feat_ctx}] Missing required property '{req_prop}' in properties")

        species = props["species"]
        action = props["action"]
        rate = props["rate_l_ha"]
        is_crop = props.get("is_crop", (species == "crop_wheat"))

        if not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValidationError(f"[{feat_ctx}] 'rate_l_ha' must be a finite float, got {rate!r}")

        # Invariant 1: do_not_spray or crop MUST have rate == 0.0 L/ha
        if action == "do_not_spray" or is_crop or species == "crop_wheat":
            if abs(rate - ZERO_SPRAY_RATE_L_HA) > 1e-6:
                raise ValidationError(
                    f"[{feat_ctx}] CROP SAFETY VIOLATION: action='{action}', species='{species}', is_crop={is_crop} "
                    f"has non-zero spray rate {rate} L/ha (expected 0.0 L/ha)!"
                )
            if species == "crop_wheat" and action != "do_not_spray":
                raise ValidationError(
                    f"[{feat_ctx}] CROP SAFETY VIOLATION: crop_wheat has action='{action}' (must be 'do_not_spray')!"
                )

        # Invariant 2: spray_weed MUST have rate == 150.0 L/ha and must NOT be crop
        elif action == "spray_weed":
            if abs(rate - STANDARD_SPRAY_RATE_L_HA) > 1e-6:
                raise ValidationError(
                    f"[{feat_ctx}] SPOT SPRAY RATE VIOLATION: action='spray_weed' has rate {rate} L/ha "
                    f"(expected {STANDARD_SPRAY_RATE_L_HA} L/ha)!"
                )
            if is_crop or species == "crop_wheat":
                raise ValidationError(
                    f"[{feat_ctx}] CROP SAFETY VIOLATION: Protected crop cannot have action='spray_weed'!"
                )
        else:
            raise ValidationError(f"[{feat_ctx}] Unrecognized action '{action}'")

        validated_features.append(feat)

    return {
        "status": "PASS",
        "feature_count": len(validated_features),
        "features": validated_features,
    }


def validate_geojson_file(path: Path) -> Dict[str, Any]:
    """Validate GeoJSON file on disk."""
    if not path.exists():
        raise ValidationError(f"GeoJSON file not found: {path}")
    raw_content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_content)
    except Exception as e:
        raise ValidationError(f"GeoJSON file {path} contains invalid JSON: {e}") from e
    return validate_geojson_data(data, context=str(path.name))


def validate_taskdata_xml(xml_content: str, geojson_features: List[Dict[str, Any]] | None = None, context: str = "TASKDATA.XML") -> Dict[str, Any]:
    """Validate ISO 11783-10 TaskData XML string."""
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        raise ValidationError(f"[{context}] Failed to parse XML: {e}") from e

    valid_root_tags = ("ISO11783_TaskData", "ISO11783_TaskFile")
    # Tag might contain namespace prefix
    tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_clean not in valid_root_tags:
        raise ValidationError(f"[{context}] Root tag must be one of {valid_root_tags}, got '{tag_clean}'")

    version_major = root.attrib.get("VersionMajor")
    if version_major != "4":
        raise ValidationError(f"[{context}] ISO 11783-10 requires VersionMajor='4', got '{version_major}'")

    task = root.find(".//TSK")
    if task is None:
        raise ValidationError(f"[{context}] Missing mandatory <TSK> (Task) element")

    zone = task.find(".//TZN")
    if zone is None:
        raise ValidationError(f"[{context}] Missing mandatory <TZN> (Treatment Zone) element")

    points = zone.findall("PNT")
    if len(points) == 0:
        raise ValidationError(f"[{context}] Zone <TZN> contains 0 <PNT> points")

    if geojson_features is not None:
        if len(points) != len(geojson_features):
            raise ValidationError(
                f"[{context}] Point count mismatch: XML has {len(points)} <PNT> elements, "
                f"but GeoJSON has {len(geojson_features)} features"
            )

    validated_points = []
    for idx, pnt in enumerate(points):
        pnt_ctx = f"{context} PNT[{idx+1}]"
        # B = North (Latitude), C = East (Longitude)
        lat_str = pnt.attrib.get("B")
        lon_str = pnt.attrib.get("C")
        if lat_str is None or lon_str is None:
            raise ValidationError(f"[{pnt_ctx}] Missing latitude (B) or longitude (C) attributes")

        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError as e:
            raise ValidationError(f"[{pnt_ctx}] Coordinates are not valid floats: B={lat_str!r}, C={lon_str!r}") from e

        if not (KOSTANAY_LAT_MIN <= lat <= KOSTANAY_LAT_MAX):
            raise ValidationError(f"[{pnt_ctx}] Latitude {lat:.7f} outside Kostanay bounds [{KOSTANAY_LAT_MIN}, {KOSTANAY_LAT_MAX}]")
        if not (KOSTANAY_LON_MIN <= lon <= KOSTANAY_LON_MAX):
            raise ValidationError(f"[{pnt_ctx}] Longitude {lon:.7f} outside Kostanay bounds [{KOSTANAY_LON_MIN}, {KOSTANAY_LON_MAX}]")

        # Check PDV (Process Data Variable)
        pdv = pnt.find("PDV")
        if pdv is None:
            raise ValidationError(f"[{pnt_ctx}] Missing child <PDV> (Process Data Variable) element")

        rate_str = pdv.attrib.get("B")
        if rate_str is None:
            raise ValidationError(f"[{pnt_ctx}] <PDV> missing rate attribute 'B'")

        try:
            rate_val = float(rate_str)
        except ValueError as e:
            raise ValidationError(f"[{pnt_ctx}] <PDV> rate B='{rate_str}' is not numeric") from e

        # Cross-validation with GeoJSON feature if provided
        if geojson_features is not None:
            feat = geojson_features[idx]
            feat_coords = feat["geometry"]["coordinates"]
            feat_lon, feat_lat = feat_coords[0], feat_coords[1]
            feat_action = feat["properties"]["action"]
            feat_rate = feat["properties"]["rate_l_ha"]

            if abs(lat - feat_lat) > 1e-5:
                raise ValidationError(f"[{pnt_ctx}] Latitude {lat} does not match GeoJSON latitude {feat_lat}")
            if abs(lon - feat_lon) > 1e-5:
                raise ValidationError(f"[{pnt_ctx}] Longitude {lon} does not match GeoJSON longitude {feat_lon}")

            if feat_action == "spray_weed":
                # Rate should represent 150 L/ha (in ml/ha: 150000, or in L/ha: 150.0)
                if rate_val not in (ISOXML_STANDARD_RATE_ML_HA, STANDARD_SPRAY_RATE_L_HA):
                    raise ValidationError(
                        f"[{pnt_ctx}] Weed target has invalid PDV rate {rate_val} "
                        f"(expected {ISOXML_STANDARD_RATE_ML_HA} ml/ha or {STANDARD_SPRAY_RATE_L_HA} L/ha)"
                    )
            elif feat_action == "do_not_spray":
                if abs(rate_val - 0.0) > 1e-6:
                    raise ValidationError(
                        f"[{pnt_ctx}] Non-spray target has non-zero PDV rate {rate_val} (expected 0)"
                    )

        validated_points.append({"lat": lat, "lon": lon, "rate": rate_val})

    return {
        "status": "PASS",
        "root_tag": tag_clean,
        "version_major": version_major,
        "point_count": len(validated_points),
        "points": validated_points,
    }


def validate_taskdata_file(path: Path, geojson_features: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Validate ISO-XML file on disk."""
    if not path.exists():
        raise ValidationError(f"TASKDATA.XML file not found: {path}")
    raw_content = path.read_text(encoding="utf-8")
    return validate_taskdata_xml(raw_content, geojson_features=geojson_features, context=str(path.name))


def run_adversarial_mutations(valid_geojson_data: Dict[str, Any], valid_xml_str: str) -> None:
    """Stress-test the validator itself by generating adversarial mutations and verifying rejection."""
    print("\n--- Running Adversarial Mutations & Fuzzing Oracle ---")
    mutations_tested = 0

    # 1. Mutation: Crop wheat with positive spray rate
    m_crop_sprayed = copy.deepcopy(valid_geojson_data)
    m_crop_sprayed["features"][0]["properties"]["species"] = "crop_wheat"
    m_crop_sprayed["features"][0]["properties"]["action"] = "spray_weed"
    m_crop_sprayed["features"][0]["properties"]["rate_l_ha"] = 150.0
    try:
        validate_geojson_data(m_crop_sprayed, context="mut_crop_sprayed")
        raise AssertionError("Oracle Failure: Protected crop with spray rate was NOT rejected!")
    except ValidationError as e:
        assert "CROP SAFETY VIOLATION" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Crop wheat with spray_weed action")

    # 2. Mutation: Weed with 0 L/ha while marked spray_weed
    m_weed_zero = copy.deepcopy(valid_geojson_data)
    m_weed_zero["features"][1]["properties"]["action"] = "spray_weed"
    m_weed_zero["features"][1]["properties"]["rate_l_ha"] = 0.0
    try:
        validate_geojson_data(m_weed_zero, context="mut_weed_zero")
        raise AssertionError("Oracle Failure: Weed with 0 rate was NOT rejected!")
    except ValidationError as e:
        assert "SPOT SPRAY RATE VIOLATION" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Weed with rate_l_ha=0.0")

    # 3. Mutation: Out of bounds longitude (e.g. 76.9 E, Almaty)
    m_oob_lon = copy.deepcopy(valid_geojson_data)
    m_oob_lon["features"][0]["geometry"]["coordinates"] = [76.9286, 53.2207]
    try:
        validate_geojson_data(m_oob_lon, context="mut_oob_lon")
        raise AssertionError("Oracle Failure: Out-of-bounds longitude was NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Coordinate outside Kostanay longitude bounds")

    # 4. Mutation: Out of bounds latitude (e.g. 43.2 N, Almaty)
    m_oob_lat = copy.deepcopy(valid_geojson_data)
    m_oob_lat["features"][0]["geometry"]["coordinates"] = [63.6254, 43.2389]
    try:
        validate_geojson_data(m_oob_lat, context="mut_oob_lat")
        raise AssertionError("Oracle Failure: Out-of-bounds latitude was NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Coordinate outside Kostanay latitude bounds")

    # 5. Mutation: Inverted coordinates [lat, lon]
    # In Kostanay lat is ~53, lon is ~63. If inverted: coords are [53, 63].
    # Then lon=53 (fails lon in 60-70) and lat=63 (fails lat in 50-56).
    m_inv_coords = copy.deepcopy(valid_geojson_data)
    m_inv_coords["features"][0]["geometry"]["coordinates"] = [53.2207, 63.6254]
    try:
        validate_geojson_data(m_inv_coords, context="mut_inv_coords")
        raise AssertionError("Oracle Failure: Inverted [lat, lon] coordinates were NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Inverted [lat, lon] order")

    # 6. Mutation: Invalid geometry type Polygon
    m_polygon = copy.deepcopy(valid_geojson_data)
    m_polygon["features"][0]["geometry"]["type"] = "Polygon"
    m_polygon["features"][0]["geometry"]["coordinates"] = [[[63.6, 53.2], [63.7, 53.2], [63.7, 53.3], [63.6, 53.2]]]
    try:
        validate_geojson_data(m_polygon, context="mut_polygon")
        raise AssertionError("Oracle Failure: Non-Point geometry was NOT rejected!")
    except ValidationError as e:
        assert "Geometry type must be 'Point'" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Non-Point geometry (Polygon)")

    # 7. Mutation: ISO-XML VersionMajor="3"
    m_xml_ver = valid_xml_str.replace('VersionMajor="4"', 'VersionMajor="3"')
    try:
        validate_taskdata_xml(m_xml_ver, context="mut_xml_ver")
        raise AssertionError("Oracle Failure: XML with VersionMajor='3' was NOT rejected!")
    except ValidationError as e:
        assert "VersionMajor='4'" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: ISO-XML VersionMajor='3'")

    # 8. Mutation: ISO-XML Point Count Mismatch
    try:
        # Pass geojson features with 1 extra fake feature
        faked_features = valid_geojson_data["features"] + [valid_geojson_data["features"][0]]
        validate_taskdata_xml(valid_xml_str, geojson_features=faked_features, context="mut_xml_count")
        raise AssertionError("Oracle Failure: XML point count mismatch was NOT rejected!")
    except ValidationError as e:
        assert "Point count mismatch" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Point count mismatch between GeoJSON and XML")

    # 9. Mutation: ISO-XML Rate Mismatch (e.g. weed point has PDV 0)
    m_xml_pdv_zero = valid_xml_str.replace('<PDV A="1" B="150000" C="0" />', '<PDV A="1" B="0" C="0" />', 1)
    try:
        validate_taskdata_xml(m_xml_pdv_zero, geojson_features=valid_geojson_data["features"], context="mut_xml_rate")
        raise AssertionError("Oracle Failure: XML PDV rate mismatch was NOT rejected!")
    except ValidationError as e:
        assert "Weed target has invalid PDV rate" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Weed target with 0 PDV application rate")

    print(f"--- All {mutations_tested} Adversarial Mutations Successfully Caught! ---\n")


def run_pipeline_regeneration(simulation_script: Path) -> Dict[str, Any]:
    """Test regeneration by executing e2e_pipeline_simulation.py in a clean temporary directory."""
    print("--- Testing Pipeline Regeneration in Sandbox Directory ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "e2e_sim_test"
        cmd = [sys.executable, str(simulation_script), "--output", str(tmp_out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValidationError(f"Pipeline regeneration failed with code {proc.returncode}:\n{proc.stderr}")

        # Check that all artifacts were created
        expected_files = [
            tmp_out / "prescription.geojson",
            tmp_out / "TASKDATA.XML",
            tmp_out / "server_ledger.json",
            tmp_out / "simulation_report.json",
            tmp_out / "offline_queue.sqlite",
            tmp_out / "plots" / "latency_distribution.png",
            tmp_out / "plots" / "confusion_matrix.png",
            tmp_out / "plots" / "roc_crop_safety.png",
            tmp_out / "plots" / "field_heatmap.png",
        ]
        for exp_f in expected_files:
            if not exp_f.exists():
                raise ValidationError(f"Expected generated artifact missing: {exp_f}")

        # Validate newly generated artifacts
        res_gj = validate_geojson_file(tmp_out / "prescription.geojson")
        res_xml = validate_taskdata_file(tmp_out / "TASKDATA.XML", geojson_features=res_gj["features"])

        print(f"  [PASS] Pipeline generated {res_gj['feature_count']} GeoJSON features and {res_xml['point_count']} XML points.")
        print(f"  [PASS] All 9 artifacts regenerated and verified in {tmp_out}")
        return {
            "regeneration": "SUCCESS",
            "feature_count": res_gj["feature_count"],
            "point_count": res_xml["point_count"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarially verify Spot-Spraying prescription GeoJSON and ISO-XML")
    parser.add_argument("--geojson", type=Path, default=Path("case1/output/e2e_simulation/prescription.geojson"))
    parser.add_argument("--taskdata", type=Path, default=Path("case1/output/e2e_simulation/TASKDATA.XML"))
    parser.add_argument("--pipeline-script", type=Path, default=Path("scripts/e2e_pipeline_simulation.py"))
    args = parser.parse_args()

    print("================================================================================")
    print("  EMPIRICAL CHALLENGER: Adversarial Spot-Spraying & ISO-XML Verification Suite  ")
    print("================================================================================")

    # 1. Validate Target GeoJSON
    print(f"\n1. Validating GeoJSON: {args.geojson}")
    res_gj = validate_geojson_file(args.geojson)
    print(f"   [PASS] RFC 7946 compliant FeatureCollection with {res_gj['feature_count']} features.")
    for idx, f in enumerate(res_gj["features"]):
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        print(f"          Point {idx+1}: [{coords[0]:.7f}, {coords[1]:.7f}] | {props['species']:<15} | action={props['action']:<12} | rate={props['rate_l_ha']} L/ha")

    # 2. Validate Target TASKDATA.XML
    print(f"\n2. Validating TASKDATA.XML: {args.taskdata}")
    res_xml = validate_taskdata_file(args.taskdata, geojson_features=res_gj["features"])
    print(f"   [PASS] ISO 11783-10 TaskData compliant root <{res_xml['root_tag']}> VersionMajor='{res_xml['version_major']}'.")
    print(f"   [PASS] Exactly {res_xml['point_count']} <PNT> elements cross-matched with GeoJSON.")
    for idx, p in enumerate(res_xml["points"]):
        print(f"          PNT {idx+1}: Lat={p['lat']:.7f}, Lon={p['lon']:.7f} | PDV={p['rate']} (DDI=1)")

    # 3. Adversarial Mutations & Fuzzing Oracle
    raw_gj_data = json.loads(args.geojson.read_text(encoding="utf-8"))
    raw_xml_str = args.taskdata.read_text(encoding="utf-8")
    run_adversarial_mutations(raw_gj_data, raw_xml_str)

    # 4. Pipeline Regeneration & Reproducibility Test
    run_pipeline_regeneration(args.pipeline_script)

    print("================================================================================")
    print("  VERDICT: ALL ADVERSARIAL VALIDATIONS PASSED CLEANLY (VERDICT: APPROVE)        ")
    print("================================================================================")


if __name__ == "__main__":
    main()
