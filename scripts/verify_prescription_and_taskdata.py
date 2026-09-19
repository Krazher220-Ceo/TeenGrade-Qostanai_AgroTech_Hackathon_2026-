#!/usr/bin/env python3
"""Adversarial stress-test and verification harness for Spot-Spraying artifacts.

Validates:
1. RFC 7946 GeoJSON compliance, Kostanay bounding box, Polygon/MultiPolygon
   treatment-zone geometry, agronomic rate invariants.
2. ISO 11783-10 TaskData XML schema, root version, zone (TZN) boundaries and
   PDV rates, and the GRD Grid Type 1 raster (dimensions, binary file size,
   TreatmentZoneCode values).
3. Cross-artifact consistency between GeoJSON zones and ISO-XML TZN/GRD.
4. Mutation & Fuzzing Oracle: confirms that adversarial mutations (including
   corrupted/missing/invalid GRD grid data) are strictly rejected.
5. End-to-end pipeline reproducibility.

This is the second generation of this harness: the original (point-per-
detection) model was superseded by buffered treatment-zone polygons plus a
GRD raster (see case1/geo/zones.py, case1/geo/isoxml.py). A prior revision of
this file (git history around commit 0d5cc45) quietly gutted most of the
checks below to an early ``return {"status": "PASS", ...}`` and loosened the
adversarial test assertions from exact counts to ``>= 4`` while adapting the
data model, instead of updating the checks to match the new schema. This
version restores every original invariant, adapts it to the zone+grid model,
and adds new GRD-specific adversarial mutations.

Usage:
    .venv/bin/python scripts/verify_prescription_and_taskdata.py
    .venv/bin/python scripts/verify_prescription_and_taskdata.py --geojson case1/output/e2e_simulation/prescription.geojson --taskdata case1/output/e2e_simulation/TASKDATA.XML
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Kostanay Oblast geographical bounding box
KOSTANAY_LON_MIN = 60.0
KOSTANAY_LON_MAX = 70.0
KOSTANAY_LAT_MIN = 50.0
KOSTANAY_LAT_MAX = 56.0

# DDI 0001 ("Setpoint Volume Per Area Application Rate", ISO 11783-11 Data
# Dictionary) has unit mm^3/m^2 and resolution 0.01. Since 1 L/ha ==
# 100 mm^3/m^2 (1 L / 10,000 m^2 = 1,000,000 mm^3 / 10,000 m^2), the raw PDV
# integer is rate_l_ha * 100 (mm^3/m^2) / 0.01 == rate_l_ha * 10000. Must
# match case1/geo/isoxml.py's export_isoxml() encoding exactly.
DDI_0001_RAW_PER_L_HA = 10000

_ROOT_DIR = Path(__file__).resolve().parents[1]
_AGRONOMY_CONFIG_PATH = _ROOT_DIR / "case1" / "configs" / "agronomy_rules.json"


def _load_treatment_zone_rates() -> Tuple[float, float]:
    """Read (base_rate_l_ha, perennial_rate_l_ha) from the same config file
    case1/geo/zones.py uses, so a rate change there doesn't silently break
    this validator (and vice versa)."""
    defaults = (200.0, 250.0)
    try:
        config = json.loads(_AGRONOMY_CONFIG_PATH.read_text(encoding="utf-8"))
        tz_cfg = config.get("treatment_zones", {})
        return (
            float(tz_cfg.get("base_rate_l_ha", defaults[0])),
            float(tz_cfg.get("perennial_rate_l_ha", defaults[1])),
        )
    except Exception:
        return defaults


BASE_RATE_L_HA, PERENNIAL_RATE_L_HA = _load_treatment_zone_rates()


class ValidationError(Exception):
    """Raised when an artifact fails adversarial validation."""


def _extract_geojson_rings(geom: Dict[str, Any]) -> Optional[List[List[Tuple[float, float]]]]:
    """Return a list of rings (each a list of (lon, lat) tuples) for a
    Polygon or MultiPolygon geometry, or None if malformed."""
    geom_type = geom.get("type")
    coords = geom.get("coordinates")
    if not isinstance(coords, list):
        return None
    try:
        if geom_type == "Polygon":
            rings = coords
        elif geom_type == "MultiPolygon":
            rings = [ring for poly in coords for ring in poly]
        else:
            return None
        out = []
        for ring in rings:
            out.append([(float(v[0]), float(v[1])) for v in ring])
        return out
    except (TypeError, ValueError, IndexError):
        return None


def validate_geojson_data(data: Dict[str, Any], context: str = "GeoJSON") -> Dict[str, Any]:
    """Validate treatment-zone GeoJSON against RFC 7946 and domain invariants."""
    if not isinstance(data, dict):
        raise ValidationError(f"[{context}] Top-level object must be a JSON dict, got {type(data).__name__}")

    if data.get("type") != "FeatureCollection":
        raise ValidationError(f"[{context}] RFC 7946 violation: root 'type' must be 'FeatureCollection', got {data.get('type')!r}")

    features = data.get("features")
    if not isinstance(features, list):
        raise ValidationError(f"[{context}] RFC 7946 violation: 'features' must be a list, got {type(features).__name__}")

    if len(features) == 0:
        raise ValidationError(f"[{context}] 'features' list is empty. Spot-spraying prescription must contain treatment zones.")

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
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValidationError(
                f"[{feat_ctx}] Geometry type must be 'Polygon' or 'MultiPolygon' for a spray treatment zone, got {geom_type!r}"
            )

        rings = _extract_geojson_rings(geom)
        if not rings:
            raise ValidationError(f"[{feat_ctx}] 'coordinates' must decode to at least one ring of [lon, lat] pairs, got {geom.get('coordinates')!r}")

        for ring_idx, ring in enumerate(rings):
            if len(ring) < 4:
                raise ValidationError(f"[{feat_ctx}] ring[{ring_idx}] has {len(ring)} vertices; a closed polygon ring needs >= 4")
            for v_idx, (lon, lat) in enumerate(ring):
                if not (math.isfinite(lon) and math.isfinite(lat)):
                    raise ValidationError(f"[{feat_ctx}] ring[{ring_idx}] vertex[{v_idx}] coordinates must be finite, got lon={lon}, lat={lat}")
                if not (KOSTANAY_LON_MIN <= lon <= KOSTANAY_LON_MAX):
                    raise ValidationError(
                        f"[{feat_ctx}] ring[{ring_idx}] vertex[{v_idx}] longitude {lon:.7f} outside Kostanay region bounds [{KOSTANAY_LON_MIN}, {KOSTANAY_LON_MAX}]"
                    )
                if not (KOSTANAY_LAT_MIN <= lat <= KOSTANAY_LAT_MAX):
                    raise ValidationError(
                        f"[{feat_ctx}] ring[{ring_idx}] vertex[{v_idx}] latitude {lat:.7f} outside Kostanay region bounds [{KOSTANAY_LAT_MIN}, {KOSTANAY_LAT_MAX}]"
                    )

        props = feat.get("properties")
        if not isinstance(props, dict):
            raise ValidationError(f"[{feat_ctx}] 'properties' must be a dict, got {type(props).__name__}")

        for req_prop in ("rate_l_ha", "dominant_species", "has_perennials", "weed_count"):
            if req_prop not in props:
                raise ValidationError(f"[{feat_ctx}] Missing required property '{req_prop}' in properties")

        rate = props["rate_l_ha"]
        dominant_species = props["dominant_species"]
        has_perennials = bool(props["has_perennials"])
        weed_count = props["weed_count"]
        action = props.get("action", "spray_weed")

        if not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValidationError(f"[{feat_ctx}] 'rate_l_ha' must be a finite float, got {rate!r}")

        if not isinstance(weed_count, (int, float)) or weed_count < 1:
            raise ValidationError(f"[{feat_ctx}] 'weed_count' must be >= 1 for a spray treatment zone, got {weed_count!r}")

        # Invariant 1 (crop safety): a spray treatment zone must never carry
        # the protected crop as its dominant species, and must never be
        # produced from a manual_review-only cluster.
        if dominant_species == "crop_wheat":
            raise ValidationError(
                f"[{feat_ctx}] CROP SAFETY VIOLATION: treatment zone dominant_species='crop_wheat' "
                f"(the protected crop must never form part of a spray zone)!"
            )
        if action == "manual_review":
            raise ValidationError(
                f"[{feat_ctx}] REVIEW SAFETY VIOLATION: action='manual_review' detections must never "
                f"be aggregated into a spray treatment zone (unresolved review must not auto-spray)!"
            )
        if action not in ("spray", "spray_weed"):
            raise ValidationError(f"[{feat_ctx}] Unrecognized action '{action}' for a spray treatment zone")

        # Invariant 2 (rate consistency): rate_l_ha must exactly match the
        # configured base/perennial rate for the zone's has_perennials flag
        # (case1/configs/agronomy_rules.json -> treatment_zones).
        expected_rate = PERENNIAL_RATE_L_HA if has_perennials else BASE_RATE_L_HA
        if abs(rate - expected_rate) > 1e-6:
            raise ValidationError(
                f"[{feat_ctx}] RATE VIOLATION: has_perennials={has_perennials} but rate_l_ha={rate} "
                f"(expected {expected_rate} L/ha)!"
            )
        if rate <= 0:
            raise ValidationError(f"[{feat_ctx}] SPOT SPRAY RATE VIOLATION: rate_l_ha must be > 0, got {rate}")

        validated_features.append(feat)

    return {
        "status": "PASS",
        "feature_count": len(validated_features),
        "zone_count": len(validated_features),
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


def _tzn_boundary_rings(tzn: ET.Element) -> List[List[Tuple[float, float]]]:
    """Extract boundary rings (list of (lat, lon) tuples) from a TZN's
    <PLN>/<LSG>/<PNT> children. PNT boundary-point attributes: C=North
    (latitude), D=East (longitude) — see case1/geo/isoxml.py."""
    rings: List[List[Tuple[float, float]]] = []
    for pln in tzn.findall("PLN"):
        for lsg in pln.findall("LSG"):
            ring = []
            for pnt in lsg.findall("PNT"):
                lat_str, lon_str = pnt.attrib.get("C"), pnt.attrib.get("D")
                if lat_str is None or lon_str is None:
                    continue
                ring.append((float(lat_str), float(lon_str)))
            if ring:
                rings.append(ring)
    return rings


def _validate_grd(
    grd: Optional[ET.Element],
    tzns: List[ET.Element],
    taskdata_dir: Optional[Path],
    context: str,
) -> Dict[str, Any]:
    if grd is None:
        raise ValidationError(f"[{context}] Missing mandatory <GRD> (Grid) element")

    required_attrs = ("A", "B", "C", "D", "E", "F", "G", "I")
    for attr in required_attrs:
        if attr not in grd.attrib:
            raise ValidationError(f"[{context}] <GRD> missing required attribute '{attr}'")

    try:
        min_lat = float(grd.attrib["A"])
        min_lon = float(grd.attrib["B"])
        cell_lat_size = float(grd.attrib["C"])
        cell_lon_size = float(grd.attrib["D"])
        cols = int(grd.attrib["E"])
        rows = int(grd.attrib["F"])
    except ValueError as e:
        raise ValidationError(f"[{context}] <GRD> has a non-numeric geometry attribute: {e}") from e

    if grd.attrib["I"] != "1":
        raise ValidationError(
            f"[{context}] <GRD> GridType (I) must be '1' (1 byte/cell TreatmentZoneCode raster "
            f"indexing <TZN A=...>), got {grd.attrib['I']!r}"
        )
    if cols <= 0 or rows <= 0:
        raise ValidationError(f"[{context}] <GRD> grid dimensions must be positive, got {cols}x{rows}")
    if not (KOSTANAY_LAT_MIN <= min_lat <= KOSTANAY_LAT_MAX) or not (KOSTANAY_LON_MIN <= min_lon <= KOSTANAY_LON_MAX):
        raise ValidationError(f"[{context}] <GRD> origin ({min_lat}, {min_lon}) is outside Kostanay region bounds")
    if cell_lat_size <= 0 or cell_lon_size <= 0:
        raise ValidationError(f"[{context}] <GRD> cell size must be positive, got C={cell_lat_size}, D={cell_lon_size}")

    valid_zone_codes = set()
    for tzn in tzns:
        a = tzn.attrib.get("A")
        if a is not None and a.isdigit():
            valid_zone_codes.add(int(a))

    result: Dict[str, Any] = {"cols": cols, "rows": rows, "min_lat": min_lat, "min_lon": min_lon}

    if taskdata_dir is None:
        return result

    filename = grd.attrib.get("G")
    if not filename:
        raise ValidationError(f"[{context}] <GRD> missing filename attribute 'G'")
    bin_path = Path(taskdata_dir) / f"{filename}.bin"
    if not bin_path.exists():
        raise ValidationError(f"[{context}] GRD binary file not found on disk: {bin_path.name} (referenced by <GRD G=...>)")

    actual_size = bin_path.stat().st_size
    expected_size = cols * rows * 1  # GridType 1 == 1 byte/cell
    if actual_size != expected_size:
        raise ValidationError(
            f"[{context}] GRD binary file size mismatch: {bin_path.name} is {actual_size} bytes on disk, "
            f"expected {expected_size} bytes for a {cols}x{rows} GridType 1 (1 byte/cell) grid"
        )

    declared_length = grd.attrib.get("H")
    if declared_length is not None:
        try:
            declared_length_int = int(declared_length)
        except ValueError as e:
            raise ValidationError(f"[{context}] <GRD H=...> is not an integer: {declared_length!r}") from e
        if declared_length_int != actual_size:
            raise ValidationError(
                f"[{context}] <GRD H={declared_length}> does not match the actual file size {actual_size} bytes"
            )

    raw = bin_path.read_bytes()
    bad_codes = sorted({b for b in raw if b != 0 and b not in valid_zone_codes})
    if bad_codes:
        raise ValidationError(
            f"[{context}] GRD binary file contains TreatmentZoneCode value(s) {bad_codes} with no "
            f"matching <TZN A=...> element (valid codes: {sorted(valid_zone_codes)})"
        )

    result["bin_path"] = bin_path
    result["raw"] = raw
    return result


def validate_taskdata_xml(
    xml_content: str,
    geojson_features: Optional[List[Dict[str, Any]]] = None,
    context: str = "TASKDATA.XML",
    taskdata_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate ISO 11783-10 TaskData XML string (zone + grid model).

    ``taskdata_dir`` is the directory containing TASKDATA.XML: when given,
    the referenced GRD binary file is opened and cross-checked (size,
    TreatmentZoneCode values); when omitted (pure in-memory mutations with
    no real files on disk), only the XML-structural GRD checks run.
    """
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        raise ValidationError(f"[{context}] Failed to parse XML: {e}") from e

    valid_root_tags = ("ISO11783_TaskData", "ISO11783_TaskFile")
    tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_clean not in valid_root_tags:
        raise ValidationError(f"[{context}] Root tag must be one of {valid_root_tags}, got '{tag_clean}'")

    version_major = root.attrib.get("VersionMajor")
    if version_major != "4":
        raise ValidationError(f"[{context}] ISO 11783-10 requires VersionMajor='4', got '{version_major}'")

    task = root.find(".//TSK")
    if task is None:
        raise ValidationError(f"[{context}] Missing mandatory <TSK> (Task) element")

    tzns = task.findall("TZN")
    if len(tzns) == 0:
        raise ValidationError(f"[{context}] Missing mandatory <TZN> (Treatment Zone) element")

    if geojson_features is not None and len(tzns) != len(geojson_features):
        raise ValidationError(
            f"[{context}] Zone count mismatch: XML has {len(tzns)} <TZN> elements, "
            f"but GeoJSON has {len(geojson_features)} features"
        )

    validated_zones = []
    for idx, tzn in enumerate(tzns):
        tzn_ctx = f"{context} TZN[{idx}] (A={tzn.attrib.get('A')!r})"

        pdv = tzn.find("PDV")
        if pdv is None:
            raise ValidationError(f"[{tzn_ctx}] Missing child <PDV> (Process Data Variable) element")
        if pdv.attrib.get("A") != "0001":
            raise ValidationError(
                f"[{tzn_ctx}] <PDV> DDI (A) must be '0001' (Setpoint Volume Per Area Application Rate), "
                f"got {pdv.attrib.get('A')!r}"
            )
        rate_raw_str = pdv.attrib.get("B")
        if rate_raw_str is None:
            raise ValidationError(f"[{tzn_ctx}] <PDV> missing rate attribute 'B'")
        try:
            rate_raw = float(rate_raw_str)
        except ValueError as e:
            raise ValidationError(f"[{tzn_ctx}] <PDV> rate B={rate_raw_str!r} is not numeric") from e

        rings = _tzn_boundary_rings(tzn)
        if not rings:
            raise ValidationError(f"[{tzn_ctx}] Treatment zone has no boundary polygon (<PLN>/<LSG>/<PNT>)")

        for ring_idx, ring in enumerate(rings):
            for v_idx, (lat, lon) in enumerate(ring):
                if not (KOSTANAY_LAT_MIN <= lat <= KOSTANAY_LAT_MAX):
                    raise ValidationError(f"[{tzn_ctx}] ring[{ring_idx}] vertex[{v_idx}] latitude {lat:.7f} outside Kostanay bounds")
                if not (KOSTANAY_LON_MIN <= lon <= KOSTANAY_LON_MAX):
                    raise ValidationError(f"[{tzn_ctx}] ring[{ring_idx}] vertex[{v_idx}] longitude {lon:.7f} outside Kostanay bounds")

        if geojson_features is not None:
            feat = geojson_features[idx]
            props = feat["properties"]
            expected_rate_l_ha = props["rate_l_ha"]
            expected_raw = round(expected_rate_l_ha * DDI_0001_RAW_PER_L_HA)
            if abs(rate_raw - expected_raw) > 1:
                raise ValidationError(
                    f"[{tzn_ctx}] PDV rate {rate_raw} does not match GeoJSON rate_l_ha={expected_rate_l_ha} "
                    f"(expected raw PDV value {expected_raw} at {DDI_0001_RAW_PER_L_HA} per L/ha)"
                )
            if expected_rate_l_ha > 0 and abs(rate_raw) < 1e-9:
                raise ValidationError(f"[{tzn_ctx}] Weed treatment zone has PDV rate 0 (expected > 0)")

            feat_rings = _extract_geojson_rings(feat["geometry"]) or []
            if len(rings) != len(feat_rings):
                raise ValidationError(
                    f"[{tzn_ctx}] boundary ring count {len(rings)} does not match GeoJSON ring count {len(feat_rings)}"
                )
            for r_idx, (xml_ring, gj_ring) in enumerate(zip(rings, feat_rings)):
                if len(xml_ring) != len(gj_ring):
                    raise ValidationError(
                        f"[{tzn_ctx}] ring[{r_idx}] vertex count mismatch: XML has {len(xml_ring)}, GeoJSON has {len(gj_ring)}"
                    )
                for v_idx, ((xlat, xlon), (glon, glat)) in enumerate(zip(xml_ring, gj_ring)):
                    if abs(xlat - glat) > 1e-5 or abs(xlon - glon) > 1e-5:
                        raise ValidationError(
                            f"[{tzn_ctx}] ring[{r_idx}] vertex[{v_idx}] mismatch: XML=({xlat:.7f},{xlon:.7f}) "
                            f"GeoJSON=({glat:.7f},{glon:.7f})"
                        )

        validated_zones.append({"zone_code": tzn.attrib.get("A"), "rate_raw": rate_raw, "rings": rings})

    grd = task.find("GRD")
    grid_result = _validate_grd(grd, tzns, taskdata_dir, context)

    return {
        "status": "PASS",
        "root_tag": tag_clean,
        "version_major": version_major,
        "zone_count": len(validated_zones),
        "point_count": len(validated_zones),  # kept for backward-compatible callers; == zone_count in this model
        "zones": validated_zones,
        "points": validated_zones,
        "grid": grid_result,
    }


def validate_taskdata_file(path: Path, geojson_features: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Validate ISO-XML file on disk, including its GRD binary sibling."""
    if not path.exists():
        raise ValidationError(f"TASKDATA.XML file not found: {path}")
    raw_content = path.read_text(encoding="utf-8")
    return validate_taskdata_xml(raw_content, geojson_features=geojson_features, context=str(path.name), taskdata_dir=path.parent)


def run_adversarial_mutations(
    valid_geojson_data: Dict[str, Any],
    valid_xml_str: str,
    taskdata_dir: Optional[Path] = None,
) -> int:
    """Stress-test the validator itself by generating adversarial mutations
    and verifying rejection. Returns the number of mutations exercised.

    ``taskdata_dir`` (a real TASKDATA/ directory containing GRD00001.bin),
    when given, additionally exercises GRD-file-level mutations that need a
    real binary file to corrupt (size, missing file, invalid zone codes).
    """
    print("\n--- Running Adversarial Mutations & Fuzzing Oracle ---")
    mutations_tested = 0

    # 1. Mutation: crop-tagged zone with positive spray rate
    m_crop_sprayed = copy.deepcopy(valid_geojson_data)
    m_crop_sprayed["features"][0]["properties"]["dominant_species"] = "crop_wheat"
    m_crop_sprayed["features"][0]["properties"]["rate_l_ha"] = PERENNIAL_RATE_L_HA
    try:
        validate_geojson_data(m_crop_sprayed, context="mut_crop_sprayed")
        raise AssertionError("Oracle Failure: Protected crop with spray rate was NOT rejected!")
    except ValidationError as e:
        assert "CROP SAFETY VIOLATION" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: treatment zone with dominant_species=crop_wheat")

    # 2. Mutation: zone with rate not matching its has_perennials flag (e.g. 0)
    m_weed_zero = copy.deepcopy(valid_geojson_data)
    m_weed_zero["features"][1]["properties"]["rate_l_ha"] = 0.0
    try:
        validate_geojson_data(m_weed_zero, context="mut_weed_zero")
        raise AssertionError("Oracle Failure: Zone with rate_l_ha=0 was NOT rejected!")
    except ValidationError as e:
        assert "RATE VIOLATION" in str(e) or "SPOT SPRAY RATE VIOLATION" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: treatment zone with rate_l_ha=0.0")

    # 3. Mutation: out-of-bounds longitude (e.g. 76.9 E, Almaty) on a ring vertex
    m_oob_lon = copy.deepcopy(valid_geojson_data)
    _mutate_first_vertex(m_oob_lon["features"][0]["geometry"], lon=76.9286, lat=53.2207)
    try:
        validate_geojson_data(m_oob_lon, context="mut_oob_lon")
        raise AssertionError("Oracle Failure: Out-of-bounds longitude was NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: zone vertex outside Kostanay longitude bounds")

    # 4. Mutation: out-of-bounds latitude (e.g. 43.2 N, Almaty)
    m_oob_lat = copy.deepcopy(valid_geojson_data)
    _mutate_first_vertex(m_oob_lat["features"][0]["geometry"], lon=63.6254, lat=43.2389)
    try:
        validate_geojson_data(m_oob_lat, context="mut_oob_lat")
        raise AssertionError("Oracle Failure: Out-of-bounds latitude was NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: zone vertex outside Kostanay latitude bounds")

    # 5. Mutation: inverted [lat, lon] vertex order
    m_inv_coords = copy.deepcopy(valid_geojson_data)
    _mutate_first_vertex(m_inv_coords["features"][0]["geometry"], lon=53.2207, lat=63.6254)
    try:
        validate_geojson_data(m_inv_coords, context="mut_inv_coords")
        raise AssertionError("Oracle Failure: Inverted [lat, lon] vertex order was NOT rejected!")
    except ValidationError as e:
        assert "outside Kostanay region bounds" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: inverted [lat, lon] vertex order")

    # 6. Mutation: invalid geometry type (LineString instead of Polygon/MultiPolygon)
    m_line = copy.deepcopy(valid_geojson_data)
    m_line["features"][0]["geometry"]["type"] = "LineString"
    try:
        validate_geojson_data(m_line, context="mut_line")
        raise AssertionError("Oracle Failure: Non-Polygon geometry was NOT rejected!")
    except ValidationError as e:
        assert "Geometry type must be" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: non-Polygon/MultiPolygon geometry (LineString)")

    # 7. Mutation: ISO-XML VersionMajor="3"
    m_xml_ver = valid_xml_str.replace('VersionMajor="4"', 'VersionMajor="3"')
    try:
        validate_taskdata_xml(m_xml_ver, context="mut_xml_ver")
        raise AssertionError("Oracle Failure: XML with VersionMajor='3' was NOT rejected!")
    except ValidationError as e:
        assert "VersionMajor='4'" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: ISO-XML VersionMajor='3'")

    # 8. Mutation: TZN/GeoJSON zone count mismatch
    try:
        faked_features = valid_geojson_data["features"] + [valid_geojson_data["features"][0]]
        validate_taskdata_xml(valid_xml_str, geojson_features=faked_features, context="mut_xml_count")
        raise AssertionError("Oracle Failure: XML zone count mismatch was NOT rejected!")
    except ValidationError as e:
        assert "Zone count mismatch" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: TZN/GeoJSON zone count mismatch")

    # 9. Mutation: ISO-XML PDV rate zeroed out for a zone that must be > 0
    m_xml_pdv_zero = re.sub(r'(<PDV[^>]*\bB=")[^"]*(")', r'\g<1>0\g<2>', valid_xml_str, count=1)
    try:
        validate_taskdata_xml(m_xml_pdv_zero, geojson_features=valid_geojson_data["features"], context="mut_xml_rate")
        raise AssertionError("Oracle Failure: XML PDV rate mismatch was NOT rejected!")
    except ValidationError as e:
        assert "PDV rate" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: treatment zone with zeroed-out PDV application rate")

    # 10. Mutation: GRD element removed entirely
    m_no_grd = valid_xml_str.replace("<GRD", "<NO_GRD")
    try:
        validate_taskdata_xml(m_no_grd, context="mut_no_grd")
        raise AssertionError("Oracle Failure: Missing <GRD> was NOT rejected!")
    except ValidationError as e:
        assert "Missing mandatory <GRD>" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: TASKDATA.XML with <GRD> element removed")

    if taskdata_dir is not None:
        mutations_tested += _run_grd_file_mutations(valid_geojson_data, valid_xml_str, Path(taskdata_dir))
    else:
        print("  [SKIP] GRD binary-file mutations (11-13) need a real TASKDATA/ directory; none was provided")

    print(f"--- All {mutations_tested} Adversarial Mutations Successfully Caught! ---\n")
    return mutations_tested


def _mutate_first_vertex(geometry: Dict[str, Any], lon: float, lat: float) -> None:
    """Overwrite the first ring's first vertex of a Polygon/MultiPolygon
    geometry in place, for adversarial GeoJSON mutations."""
    if geometry["type"] == "Polygon":
        geometry["coordinates"][0][0] = [lon, lat]
    elif geometry["type"] == "MultiPolygon":
        geometry["coordinates"][0][0][0] = [lon, lat]
    else:
        raise ValueError(f"cannot mutate vertex of geometry type {geometry['type']!r}")


def _run_grd_file_mutations(valid_geojson_data: Dict[str, Any], valid_xml_str: str, taskdata_dir: Path) -> int:
    """GRD-specific mutations that require a real GRD00001.bin on disk:
    corrupted file size, missing file, invalid TreatmentZoneCode value."""
    mutations_tested = 0
    geojson_features = valid_geojson_data["features"]

    # 11. Mutation: GRD binary file truncated (corrupted size)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_task_dir = Path(tmpdir) / "TASKDATA"
        shutil.copytree(taskdata_dir, tmp_task_dir)
        bin_files = list(tmp_task_dir.glob("*.bin"))
        assert bin_files, "expected at least one GRD*.bin file to mutate"
        bin_path = bin_files[0]
        original = bin_path.read_bytes()
        bin_path.write_bytes(original[: max(0, len(original) // 2)])
        xml_path = tmp_task_dir / "TASKDATA.XML"
        try:
            validate_taskdata_file(xml_path, geojson_features=geojson_features)
            raise AssertionError("Oracle Failure: Corrupted (truncated) GRD binary file size was NOT rejected!")
        except ValidationError as e:
            assert "size mismatch" in str(e)
            mutations_tested += 1
            print("  [PASS] Correctly rejected: GRD binary file truncated to half its declared size")

    # 12. Mutation: GRD binary file missing entirely
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_task_dir = Path(tmpdir) / "TASKDATA"
        shutil.copytree(taskdata_dir, tmp_task_dir)
        for bin_file in tmp_task_dir.glob("*.bin"):
            bin_file.unlink()
        xml_path = tmp_task_dir / "TASKDATA.XML"
        try:
            validate_taskdata_file(xml_path, geojson_features=geojson_features)
            raise AssertionError("Oracle Failure: Missing GRD binary file was NOT rejected!")
        except ValidationError as e:
            assert "not found on disk" in str(e)
            mutations_tested += 1
            print("  [PASS] Correctly rejected: GRD binary file referenced by <GRD G=...> missing on disk")

    # 13. Mutation: GRD binary file contains an invalid TreatmentZoneCode
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_task_dir = Path(tmpdir) / "TASKDATA"
        shutil.copytree(taskdata_dir, tmp_task_dir)
        bin_files = list(tmp_task_dir.glob("*.bin"))
        bin_path = bin_files[0]
        data = bytearray(bin_path.read_bytes())
        # 253 is outside the small set of real TreatmentZoneCodes (1..N
        # zones) used by this prescription; guaranteed to be invalid.
        data[0] = 253
        bin_path.write_bytes(bytes(data))
        xml_path = tmp_task_dir / "TASKDATA.XML"
        try:
            validate_taskdata_file(xml_path, geojson_features=geojson_features)
            raise AssertionError("Oracle Failure: Invalid TreatmentZoneCode in GRD binary was NOT rejected!")
        except ValidationError as e:
            assert "no matching <TZN A=...> element" in str(e)
            mutations_tested += 1
            print("  [PASS] Correctly rejected: GRD binary file with an unassigned TreatmentZoneCode byte")

    return mutations_tested


def run_pipeline_regeneration(simulation_script: Path) -> Dict[str, Any]:
    """Test regeneration by executing e2e_pipeline_simulation.py in a clean temporary directory."""
    print("--- Testing Pipeline Regeneration in Sandbox Directory ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "e2e_sim_test"
        cmd = [sys.executable, str(simulation_script), "--output", str(tmp_out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValidationError(f"Pipeline regeneration failed with code {proc.returncode}:\n{proc.stderr}")

        expected_files = [
            tmp_out / "prescription.geojson",
            tmp_out / "TASKDATA.XML",
            tmp_out / "TASKDATA" / "GRD00001.bin",
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

        res_gj = validate_geojson_file(tmp_out / "prescription.geojson")
        res_xml = validate_taskdata_file(tmp_out / "TASKDATA.XML", geojson_features=res_gj["features"])

        print(f"  [PASS] Pipeline generated {res_gj['feature_count']} GeoJSON zones and {res_xml['zone_count']} XML TZN elements.")
        print(f"  [PASS] All {len(expected_files)} artifacts regenerated and verified in {tmp_out}")
        return {
            "regeneration": "SUCCESS",
            "feature_count": res_gj["feature_count"],
            "zone_count": res_xml["zone_count"],
            "point_count": res_xml["zone_count"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarially verify Spot-Spraying treatment-zone GeoJSON and ISO-XML+GRD TaskData")
    parser.add_argument("--geojson", type=Path, default=Path("case1/output/e2e_simulation/prescription.geojson"))
    parser.add_argument("--taskdata", type=Path, default=Path("case1/output/e2e_simulation/TASKDATA.XML"))
    parser.add_argument("--pipeline-script", type=Path, default=Path("scripts/e2e_pipeline_simulation.py"))
    args = parser.parse_args()

    print("================================================================================")
    print("  EMPIRICAL CHALLENGER: Adversarial Spot-Spraying & ISO-XML+GRD Verification Suite  ")
    print("================================================================================")

    print(f"\n1. Validating GeoJSON: {args.geojson}")
    res_gj = validate_geojson_file(args.geojson)
    print(f"   [PASS] RFC 7946 compliant FeatureCollection with {res_gj['feature_count']} treatment zones.")
    for idx, f in enumerate(res_gj["features"]):
        props = f["properties"]
        print(
            f"          Zone {idx+1} ({f['id']}): {props['dominant_species']:<20} | "
            f"weeds={props['weed_count']:<3} | perennial={str(props['has_perennials']):<5} | "
            f"rate={props['rate_l_ha']} L/ha | area={props.get('area_m2', '?')} m^2"
        )

    print(f"\n2. Validating TASKDATA.XML (+ GRD): {args.taskdata}")
    res_xml = validate_taskdata_file(args.taskdata, geojson_features=res_gj["features"])
    print(f"   [PASS] ISO 11783-10 TaskData compliant root <{res_xml['root_tag']}> VersionMajor='{res_xml['version_major']}'.")
    print(f"   [PASS] Exactly {res_xml['zone_count']} <TZN> elements cross-matched with GeoJSON zones.")
    grid = res_xml["grid"]
    print(
        f"   [PASS] <GRD> GridType=1 raster: {grid['cols']}x{grid['rows']} cells, origin "
        f"({grid['min_lat']:.7f}, {grid['min_lon']:.7f})"
        + (f", {len(grid['raw'])} bytes verified against <TZN> codes" if "raw" in grid else "")
    )
    for idx, z in enumerate(res_xml["zones"]):
        print(f"          TZN {idx+1} (code={z['zone_code']}): PDV={z['rate_raw']:.0f} (DDI=0001), rings={len(z['rings'])}")

    raw_gj_data = json.loads(args.geojson.read_text(encoding="utf-8"))
    raw_xml_str = args.taskdata.read_text(encoding="utf-8")
    # GRD00001.bin is copied next to TASKDATA.XML itself (see
    # scripts/e2e_pipeline_simulation.py::write_taskdata / case1_main.py's
    # `prescribe` writes TASKDATA.XML directly under .../TASKDATA/).
    run_adversarial_mutations(raw_gj_data, raw_xml_str, taskdata_dir=args.taskdata.parent)

    run_pipeline_regeneration(args.pipeline_script)

    print("================================================================================")
    print("  VERDICT: ALL ADVERSARIAL VALIDATIONS PASSED CLEANLY (VERDICT: APPROVE)        ")
    print("================================================================================")


if __name__ == "__main__":
    main()
