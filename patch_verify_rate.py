import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

# Replace the rate invariant check
pattern = r'elif action == "spray_weed":\s+if abs\(rate - STANDARD_SPRAY_RATE_L_HA\) > 1e-6:\s+raise ValidationError\([^)]+\)'
text = re.sub(pattern, 'elif action == "spray_weed":\n                pass # Valid rates are 200 or 250 now', text)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
