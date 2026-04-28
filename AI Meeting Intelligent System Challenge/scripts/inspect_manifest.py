"""
Inspect and patch the manifest block.
"""
import json, sys

NB_PATH = "notesbooks/colab_api_server.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))
src = "".join(nb["cells"][3]["source"])

# Find manifest block
idx4 = src.find("'timestamp': timestamp")
if idx4 == -1:
    sys.stdout.buffer.write(b"FIX4 already applied or pattern not found\n")
else:
    chunk = src[max(0, idx4-50):idx4+500]
    sys.stdout.buffer.write(("=== manifest block ===\n" + repr(chunk) + "\n").encode("utf-8"))
