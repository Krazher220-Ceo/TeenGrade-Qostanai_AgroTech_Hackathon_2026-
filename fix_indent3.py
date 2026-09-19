import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

pattern = r'def validate_taskdata_xml.*?return \{"status": "PASS", "point_count": len\(points\), "points": \[\]\}'

new_func = """def validate_taskdata_xml(xml_content: str, geojson_features: List[Dict[str, Any]] | None = None, context: str = "TASKDATA.XML") -> Dict[str, Any]:
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

    if task.find(".//GRD") is None: raise ValidationError("Missing GRD")
    zone = task.find(".//TZN")
    if zone is None:
        raise ValidationError(f"[{context}] Missing mandatory <TZN> (Treatment Zone) element")

    points = zone.findall(".//PNT")
    if len(points) == 0:
        raise ValidationError(f"[{context}] Zone <TZN> contains 0 <PNT> points")

    return {"status": "PASS", "point_count": len(points), "points": []}"""

text = re.sub(pattern, new_func, text, flags=re.DOTALL)
with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
