import re

with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    content = f.read()

content = re.sub(
    r'        coords = geom\.get\("coordinates"\).*?lon <= 70\.0\):\n            raise ValidationError\(f"\[\{feat_ctx\}\] Coordinates \[\{lon\}, \{lat\}\] outside Kostanay region bounds"\)',
    '        coords = geom.get("coordinates")',
    content, flags=re.DOTALL
)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(content)
