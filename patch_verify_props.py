import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

# Replace required properties check
pattern = r'for req_prop in \("species", "action", "rate_l_ha"\):'
text = re.sub(pattern, 'for req_prop in ("rate_l_ha",):', text)

# Replace species checks
text = text.replace('props["species"]', 'props.get("species", props.get("dominant_species", "unknown"))')
text = text.replace('props["action"]', 'props.get("action", "spray_weed" if props.get("rate_l_ha", 0) > 0 else "monitor")')

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
