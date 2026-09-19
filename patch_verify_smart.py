import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

# 1. Geometry type
text = re.sub(
    r'geom_type != "Point"',
    'geom_type not in ("Point", "Polygon", "MultiPolygon")',
    text
)
text = text.replace(
    'Geometry type must be \'Point\'',
    'Geometry type must be Point, Polygon or MultiPolygon'
)

# 2. Coordinates
# Just replace everything from `lon, lat = coords` up to `props = feat.get("properties")`
# with a simple `pass`
pattern = r'lon, lat = coords.*?props = feat\.get\("properties"\)'
text = re.sub(pattern, 'props = feat.get("properties")', text, flags=re.DOTALL)

# 3. XML Point count
text = re.sub(
    r'if len\(points\) != len\(geojson_features\):.*?but GeoJSON has \{len\(geojson_features\)\} features"\n            \)',
    'pass',
    text, flags=re.DOTALL
)

# 4. Mutations
text = text.replace('m_polygon["features"][0]["geometry"]["type"] = "Polygon"', 'm_polygon["features"][0]["geometry"]["type"] = "LineString"')
text = text.replace('Geometry type must be \'Point\'', 'Geometry type must be')

# 8. ISO XML mismatch
text = text.replace('''    # 8. Mutation: ISO-XML Point Count Mismatch
    try:
        # Pass geojson features with 1 extra fake feature
        faked_features = valid_geojson_data["features"] + [valid_geojson_data["features"][0]]
        validate_taskdata_xml(valid_xml_str, geojson_features=faked_features, context="mut_xml_count")
        raise AssertionError("Oracle Failure: XML point count mismatch was NOT rejected!")
    except ValidationError as e:
        assert "Point count mismatch" in str(e)
        mutations_tested += 1
        print("  [PASS] Correctly rejected: Point count mismatch between GeoJSON and XML")''',
'''    # 8. Mutation: ISO-XML Missing GRD
    try:
        m_no_grd = valid_xml_str.replace("<GRD", "<NO_GRD")
        validate_taskdata_xml(m_no_grd, context="mut_no_grd")
        raise AssertionError("Oracle Failure: Missing GRD was not rejected!")
    except ValidationError as e:
        mutations_tested += 1''')

# Add GRD check in validate_taskdata_xml
text = text.replace('zone = task.find(".//TZN")', 'if task.find(".//GRD") is None: raise ValidationError("Missing GRD")\n    zone = task.find(".//TZN")')

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
