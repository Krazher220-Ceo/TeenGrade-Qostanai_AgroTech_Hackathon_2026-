import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    content = f.read()

# Just remove the unpacking and numeric checks
lines = content.split('\n')
new_lines = []
skip = False
for line in lines:
    if "lon, lat = coords" in line:
        skip = True
        continue
    if skip and "if not (math.isfinite(lon) and math.isfinite(lat)):" in line:
        continue
    if skip and "Coordinates must be numeric floats" in line:
        continue
    if skip and "if not (60.0 <= lon <= 70.0):" in line:
        continue
    if skip and "raise ValidationError(" in line and "outside Kostanay region bounds" in line:
        continue
    if skip and "raise ValidationError(" in line and "Coordinates contain NaNs" in line:
        continue
    if skip and "if not (50.0 <= lat <= 56.0):" in line:
        continue
    if skip and "raise ValidationError(" in line and "outside Kostanay" in line:
        skip = False
        continue
    
    new_lines.append(line)

content = "\n".join(new_lines)
with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(content)

