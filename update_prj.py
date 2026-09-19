with open("case1_main.py", "r") as f:
    text = f.read()

old_shp = """    # Export shapefile
    if zones_geojson["features"]:
        gdf = gpd.GeoDataFrame.from_features(zones_geojson["features"], crs="EPSG:4326")
        gdf.to_file(out_p / "zones.shp", driver="ESRI Shapefile")"""

new_shp = """    # Export shapefile
    if zones_geojson["features"]:
        gdf = gpd.GeoDataFrame.from_features(zones_geojson["features"], crs="EPSG:4326")
        gdf.to_file(out_p / "zones.shp", driver="ESRI Shapefile")
        # Ensure .prj is created
        prj_str = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
        with open(out_p / "zones.prj", "w") as prj:
            prj.write(prj_str)"""

text = text.replace(old_shp, new_shp)
with open("case1_main.py", "w") as f:
    f.write(text)
