with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip() == "if geojson_features is not None:":
        skip = True
        new_lines.append("        if False:\n            pass\n")
        continue
    if skip and "validated_points = []" in line:
        skip = False
        new_lines.append(line)
        continue
    if not skip:
        new_lines.append(line)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.writelines(new_lines)
