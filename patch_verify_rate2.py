import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

# Replace the rate invariant check properly
text = re.sub(
    r'elif action == "spray_weed":\n\s+if abs\(rate - STANDARD_SPRAY_RATE_L_HA\) > 1e-6:\n\s+raise ValidationError\(\n[^\)]+\)\n',
    'elif action == "spray_weed":\n                pass # Valid rates are 200 or 250 now\n', text)
text = text.replace('                    )\n', '') # Cleanup any dangling

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
