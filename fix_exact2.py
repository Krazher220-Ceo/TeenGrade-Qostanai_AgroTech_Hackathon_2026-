with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

chunk = """        coords = geom.get("coordinates")
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

        props = feat.get("properties")"""

text = text.replace(chunk, '        coords = geom.get("coordinates")\n        props = feat.get("properties")')
with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
