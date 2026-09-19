import re

with open("scripts/e2e_pipeline_simulation.py", "r") as f:
    content = f.read()

replacement = """import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
"""

if "sys.path.insert" not in content:
    content = content.replace("import argparse", replacement + "\nimport argparse")

with open("scripts/e2e_pipeline_simulation.py", "w") as f:
    f.write(content)
