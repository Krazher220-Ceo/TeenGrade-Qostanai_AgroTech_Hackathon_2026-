import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import shape

from case1.geo.isoxml import export_isoxml, read_grid
from case1.geo.zones import create_treatment_zones


def _sample_detections():
    return [
        # Perennial cluster -> one zone, rate 250 L/ha
        {"lat": 50.0, "lon": 60.0, "species": "field_thistle", "spray_action": "spray"},
        {"lat": 50.00001, "lon": 60.00001, "species": "couch_grass", "spray_action": "spray"},
        # Annual, far away -> separate zone, rate 200 L/ha
        {"lat": 50.001, "lon": 60.001, "species": "annual_weed", "spray_action": "spray"},
        # Crop and manual_review must never reach a zone (zones.py already
        # filters these; asserted again here as a regression guard).
        {"lat": 50.0, "lon": 60.0, "species": "crop_wheat", "spray_action": "ignore"},
        {"lat": 50.002, "lon": 60.002, "species": "unknown", "spray_action": "manual_review"},
    ]


def test_export_isoxml_grid_matches_zones(tmp_path: Path):
    zones_fc = create_treatment_zones(_sample_detections())
    assert len(zones_fc["features"]) == 2

    xml_path = export_isoxml(zones_fc, tmp_path, cell_size_m=1.0)
    assert xml_path.exists()
    bin_path = xml_path.parent / "GRD00001.bin"
    assert bin_path.exists()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    tzns = root.findall(".//TZN")
    assert len(tzns) == len(zones_fc["features"])

    grid_info = read_grid(xml_path)
    assert grid_info is not None
    grid = grid_info["grid"]
    assert grid.dtype == np.uint8
    # GridType 1 -> 1 byte/cell; bin file size must equal cols*rows exactly
    # (this is the invariant that was violated before the fix: a uint32
    # grid over the zones' full bounding box produced a multi-MB file for
    # a handful of square meters of treatment area).
    assert bin_path.stat().st_size == grid_info["cols"] * grid_info["rows"]
    assert grid.shape == (grid_info["rows"], grid_info["cols"])

    # Cross-check: for each TZN, its zone_code (A attribute) must appear at
    # the grid cell containing that zone's polygon centroid, and that cell
    # value must correspond to a rate (via PDV) matching the source
    # GeoJSON zone's rate_l_ha.
    zone_by_id = {f["id"]: f for f in zones_fc["features"]}
    matched_any_nonzero = False
    for idx, tzn in enumerate(tzns):
        zone_code = int(tzn.attrib["A"])
        pdv = tzn.find("PDV")
        assert pdv is not None
        assert pdv.attrib["A"] == "0001"
        raw_rate = float(pdv.attrib["B"])

        feat = zones_fc["features"][idx]
        expected_raw = round(feat["properties"]["rate_l_ha"] * 10000)
        assert raw_rate == expected_raw

        geom = shape(feat["geometry"])
        centroid = geom.centroid
        col = int((centroid.x - grid_info["min_lon"]) / grid_info["lon_size"])
        row = int((centroid.y - grid_info["min_lat"]) / grid_info["lat_size"])
        assert 0 <= row < grid_info["rows"]
        assert 0 <= col < grid_info["cols"]
        cell_value = int(grid[row, col])
        assert cell_value == zone_code, (
            f"grid cell at zone centroid holds TreatmentZoneCode {cell_value}, "
            f"expected {zone_code} (TZN A={tzn.attrib['A']})"
        )
        if cell_value != 0:
            matched_any_nonzero = True

    assert matched_any_nonzero, "expected at least one non-zero TreatmentZoneCode cell in the grid"

    # Every non-zero byte in the raw grid must be a real zone code (0..N),
    # never an out-of-range value.
    valid_codes = {int(t.attrib["A"]) for t in tzns}
    present_codes = {int(b) for b in grid.flatten() if b != 0}
    assert present_codes <= valid_codes


def test_export_isoxml_cell_size_m_changes_grid_dimensions(tmp_path: Path):
    """Regression test: the previous implementation accepted a
    ``cell_size_m`` parameter but never used it (lat/lon cell size was
    hardcoded to ~1m regardless), so this always produced the same grid
    resolution no matter what was requested."""
    zones_fc = create_treatment_zones(_sample_detections())

    xml_1m = export_isoxml(zones_fc, tmp_path / "cell_1m", cell_size_m=1.0)
    xml_5m = export_isoxml(zones_fc, tmp_path / "cell_5m", cell_size_m=5.0)

    grid_1m = read_grid(xml_1m)
    grid_5m = read_grid(xml_5m)

    assert grid_1m["cols"] > grid_5m["cols"]
    assert grid_1m["rows"] > grid_5m["rows"]
    # A 5x coarser cell should need roughly 5x fewer columns/rows (allow
    # slack for the +1 padding / ceil rounding in the rasterizer).
    assert grid_1m["cols"] / grid_5m["cols"] == pytest.approx(5.0, rel=0.5)


def test_export_isoxml_empty_zones_does_not_crash(tmp_path: Path):
    empty_fc = {"type": "FeatureCollection", "features": []}
    xml_path = export_isoxml(empty_fc, tmp_path)
    assert xml_path.exists()
    root = ET.parse(xml_path).getroot()
    assert root.tag == "ISO11783_TaskData"
    # No zones -> no TZN, no GRD (nothing to rasterize).
    assert root.find(".//TZN") is None
    assert root.find(".//GRD") is None
