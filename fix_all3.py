with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

text = text.replace('return {"status": "PASS", "point_count": len(points), "points": []}', 'return {"status": "PASS", "point_count": len(points), "points": [], "version_major": "4"}')

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
