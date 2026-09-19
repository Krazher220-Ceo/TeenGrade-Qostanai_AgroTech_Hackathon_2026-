import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

text = re.sub(r'if False:\n\s+validated_points = \[\]', 'if False:\n            pass\n        validated_points = []', text)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
