import json
from pathlib import Path
import pytest
from shapely.geometry import Point, shape
from case1.geo.zones import create_treatment_zones

def test_create_treatment_zones():
    detections = [
        # Perennial
        {"lat": 50.0, "lon": 60.0, "species": "field_thistle", "spray_action": "spray"},
        # Another perennial close by
        {"lat": 50.00001, "lon": 60.00001, "species": "couch_grass", "spray_action": "spray"},
        # Annual, far away
        {"lat": 50.001, "lon": 60.001, "species": "annual_weed", "spray_action": "spray"},
        # Crop, should be ignored
        {"lat": 50.0, "lon": 60.0, "species": "crop_wheat", "spray_action": "ignore"},
        # Manual review, should be ignored
        {"lat": 50.002, "lon": 60.002, "species": "unknown", "spray_action": "manual_review"}
    ]
    
    fc = create_treatment_zones(detections)
    
    assert fc["type"] == "FeatureCollection"
    features = fc["features"]
    
    # We expect 2 zones: one for the 2 perennials merged, one for the annual
    assert len(features) == 2
    
    # Check properties
    perennial_zone = next(f for f in features if f["properties"]["has_perennials"])
    annual_zone = next(f for f in features if not f["properties"]["has_perennials"])
    
    assert perennial_zone["properties"]["weed_count"] == 2
    assert perennial_zone["properties"]["rate_l_ha"] == 250
    assert "field_thistle" in perennial_zone["properties"]["dominant_species"] or "couch_grass" in perennial_zone["properties"]["dominant_species"]
    
    assert annual_zone["properties"]["weed_count"] == 1
    assert annual_zone["properties"]["rate_l_ha"] == 200
