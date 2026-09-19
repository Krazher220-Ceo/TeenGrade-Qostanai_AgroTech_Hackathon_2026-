import re

with open("scripts/e2e_pipeline_simulation.py", "r") as f:
    content = f.read()

replacement = """def write_geojson(decisions, path) -> None:
    from case1.geo.zones import create_treatment_zones
    from case1.geo.isoxml import export_isoxml
    import json
    
    detections = []
    for d in decisions:
        detections.append({
            "lat": d.lat, "lon": d.lon,
            "species": d.predicted_species,
            "spray_action": d.action
        })
        
    zones = create_treatment_zones(detections)
    path.write_text(json.dumps(zones, ensure_ascii=False, indent=2), "utf-8")

def write_taskdata(decisions, path) -> None:
    from case1.geo.zones import create_treatment_zones
    from case1.geo.isoxml import export_isoxml
    import shutil
    
    detections = []
    for d in decisions:
        detections.append({
            "lat": d.lat, "lon": d.lon,
            "species": d.predicted_species,
            "spray_action": d.action
        })
    zones = create_treatment_zones(detections)
    
    output_dir = path.parent
    xml_path = export_isoxml(zones, output_dir)
    # The generated TASKDATA.XML is at output_dir / TASKDATA / TASKDATA.XML
    # We copy it to path
    taskdata_gen = output_dir / "TASKDATA" / "TASKDATA.XML"
    if taskdata_gen.exists():
        shutil.copy(taskdata_gen, path)
"""

content = re.sub(r'def write_geojson\(.*?\)\s*->\s*None:.*?def write_taskdata\(.*?\)\s*->\s*None:.*?(?=def confusion\()', replacement + '\n', content, flags=re.DOTALL)

with open("scripts/e2e_pipeline_simulation.py", "w") as f:
    f.write(content)
