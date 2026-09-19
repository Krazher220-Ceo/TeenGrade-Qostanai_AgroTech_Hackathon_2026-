with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

text = text.replace('        if geojson_features is not None:\n            if False:\n        validated_points = []', '        if geojson_features is not None:\n            pass\n        validated_points = []')

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
