import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
import pandas as pd
from collections import Counter
import pyproj

def create_treatment_zones(detections, config_path="case1/configs/agronomy_rules.json"):
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
        
    tz_cfg = config.get("treatment_zones", {
        "annual_radius_m": 0.5,
        "perennial_radius_m": 1.5,
        "base_rate_l_ha": 200,
        "perennial_rate_l_ha": 250
    })
    
    species_cfg = config.get("species_classification", {})

    valid_points = []
    for d in detections:
        action = d.get("spray_action") or d.get("action")
        if action not in ["spray", "spray_weed"]:
            continue
            
        sp = d.get("species", "unknown")
        # determine if perennial
        sp_info = species_cfg.get(sp, {})
        is_perennial = sp_info.get("type") == "perennial"
        
        radius = tz_cfg["perennial_radius_m"] if is_perennial else tz_cfg["annual_radius_m"]
        
        valid_points.append({
            "geometry": Point(d["lon"], d["lat"]),
            "is_perennial": is_perennial,
            "species": sp,
            "radius": radius
        })
        
    if not valid_points:
        return {"type": "FeatureCollection", "features": []}
        
    gdf = gpd.GeoDataFrame(valid_points, crs="EPSG:4326")
    
    # Project to a local metric CRS (Azimuthal Equidistant centered on the first point)
    center_lon, center_lat = gdf.iloc[0].geometry.x, gdf.iloc[0].geometry.y
    aeqd_crs = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    gdf_m = gdf.to_crs(aeqd_crs)
    
    # Buffer each point
    gdf_m["geometry"] = gdf_m.apply(lambda row: row.geometry.buffer(row["radius"]), axis=1)
    
    # Union all intersecting
    unioned = gdf_m.unary_union
    
    if isinstance(unioned, Polygon):
        polygons = [unioned]
    elif isinstance(unioned, MultiPolygon):
        polygons = list(unioned.geoms)
    else:
        polygons = []
        
    # We create a new GeoDataFrame for zones
    zones_gdf = gpd.GeoDataFrame(geometry=polygons, crs=aeqd_crs)
    
    # Spatial join points with zones to aggregate properties
    points_m = gdf.to_crs(aeqd_crs)
    joined = gpd.sjoin(points_m, zones_gdf, how="inner", predicate="intersects")
    
    features = []
    
    # Convert zones back to WGS84 for GeoJSON
    zones_wgs84 = zones_gdf.to_crs("EPSG:4326")
    
    for idx, row in zones_wgs84.iterrows():
        # Get points in this zone
        pts_in_zone = joined[joined["index_right"] == idx]
        
        has_perennials = pts_in_zone["is_perennial"].any()
        weed_count = len(pts_in_zone)
        species_counts = Counter(pts_in_zone["species"])
        dominant_species = species_counts.most_common(1)[0][0] if species_counts else "unknown"
        
        rate = tz_cfg["perennial_rate_l_ha"] if has_perennials else tz_cfg["base_rate_l_ha"]
        
        # Calculate area in square meters (using the metric geometry)
        area_m2 = zones_gdf.loc[idx, "geometry"].area
        
        feature = {
            "type": "Feature",
            "id": f"zone_{idx}",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "id": f"zone_{idx}",
                "area_m2": round(area_m2, 2),
                "weed_count": weed_count,
                "dominant_species": dominant_species,
                "has_perennials": bool(has_perennials),
                "rate_l_ha": rate
            }
        }
        features.append(feature)
        
    return {
        "type": "FeatureCollection",
        "features": features
    }
