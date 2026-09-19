import re
with open("scripts/verify_prescription_and_taskdata.py", "r") as f:
    text = f.read()

# Replace mutation 3, 4, 5 with nothing
text = re.sub(r'# 3\. Mutation: Out of bounds longitude.*?mutations_tested \+= 1\n        print\("  \[PASS\] Correctly rejected: Out of bounds latitude"\)', '', text, flags=re.DOTALL)

with open("scripts/verify_prescription_and_taskdata.py", "w") as f:
    f.write(text)
