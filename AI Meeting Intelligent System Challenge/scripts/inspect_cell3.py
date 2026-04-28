"""
Dump Cell 3 source exactly so we can find patterns.
"""
import json

nb = json.load(open("notesbooks/colab_api_server.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][3]["source"])

# Find diarize fallback
idx = src.find("diarized_text")
print("=== Around diarized_text ===")
print(repr(src[max(0,idx-200):idx+300]))
print()

# Find TTS call
idx2 = src.find("synthesize_speech")
print("=== Around synthesize_speech ===")
print(repr(src[max(0,idx2-100):idx2+200]))
print()

# Find transcript save
idx3 = src.find("transcript.txt")
print("=== Around transcript.txt ===")
print(repr(src[max(0,idx3-50):idx3+300]))
print()

# Find manifest block
idx4 = src.find("'timestamp': timestamp")
print("=== Around manifest ===")
print(repr(src[max(0,idx4-50):idx4+500]))
