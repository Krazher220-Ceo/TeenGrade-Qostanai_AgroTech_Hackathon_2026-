import re

with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    content = f.read()

content = re.sub(
    r'        coords = geom\.get\("coordinates"\).*?coords!r\}\"\)',
    '        coords = geom.get("coordinates")',
    content, flags=re.DOTALL
)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(content)
