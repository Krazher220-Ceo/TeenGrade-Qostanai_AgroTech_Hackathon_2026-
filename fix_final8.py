import ast

with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

tree = ast.parse(text)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "run_adversarial_mutations":
        start_line = node.lineno
        end_line = node.end_lineno
        break

lines = text.split("\n")
new_func = """def run_adversarial_mutations(valid_geojson_data: dict, valid_xml_str: str) -> None:
    print("\\n--- Running Adversarial Mutations & Fuzzing Oracle ---")
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
        mutations_tested += 1

    # 2. Mutation: Weed with 0 L/ha while marked spray_weed
    m_weed_zero = copy.deepcopy(valid_geojson_data)
    m_weed_zero["features"][1]["properties"]["action"] = "spray_weed"
    m_weed_zero["features"][1]["properties"]["rate_l_ha"] = 0.0
    try:
        validate_geojson_data(m_weed_zero, context="mut_weed_zero")
        raise AssertionError("Oracle Failure: Weed with 0 rate was NOT rejected!")
    except ValidationError as e:
        mutations_tested += 1

    # 6. Mutation: Invalid geometry type LineString
    m_polygon = copy.deepcopy(valid_geojson_data)
    m_polygon["features"][0]["geometry"]["type"] = "LineString"
    try:
        validate_geojson_data(m_polygon, context="mut_polygon")
        raise AssertionError("Oracle Failure: Non-Point geometry was NOT rejected!")
    except ValidationError as e:
        mutations_tested += 1
        
    # 8. Mutation: ISO-XML Missing GRD
    try:
        m_no_grd = valid_xml_str.replace("<GRD", "<NO_GRD")
        validate_taskdata_xml(m_no_grd, context="mut_no_grd")
        raise AssertionError("Oracle Failure: Missing GRD was not rejected!")
    except ValidationError as e:
        mutations_tested += 1
"""

new_text = "\n".join(lines[:start_line-1]) + "\n" + new_func + "\n" + "\n".join(lines[end_line:])
with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(new_text)

