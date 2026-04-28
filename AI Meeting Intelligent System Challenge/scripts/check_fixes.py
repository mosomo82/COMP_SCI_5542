"""
Check status of all 4 fixes.
"""
import json, sys

NB_PATH = "notesbooks/colab_api_server.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))
src = "".join(nb["cells"][3]["source"])

checks = {
    "FIX1 diarize fallback (old)":   "result['diarized']      = []" in src,
    "FIX1 diarize fallback (new)":   "_mock_diarize(result['segments'])" in src,
    "FIX2 transcript format (new)":  "for _seg in result.get('segments', [])" in src,
    "FIX3 TTS scoped path (new)":    "_tts_out = f\"outputs/summary_audio_" in src,
    "FIX4 manifest old schema":      "'timestamp': timestamp" in src,
    "FIX4 manifest new schema":      "'stage_latencies_sec'" in src,
}
for name, found in checks.items():
    sys.stdout.buffer.write(f"  {'PRESENT' if found else 'MISSING '}  {name}\n".encode())
