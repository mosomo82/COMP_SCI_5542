import json

nb = json.load(open("notesbooks/colab_api_server.ipynb"))
for i, c in enumerate(nb["cells"]):
    print("=== Cell %d [%s] ===" % (i, c["cell_type"]))
    print("".join(c["source"]))
    print()
