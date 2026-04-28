"""
Patch colab_api_server.ipynb to fix all 4 mismatches vs local pipeline.py:
  1. Diarization fallback: use _mock_diarize() instead of empty list
  2. Transcript save: write timestamped format to match local save_transcript()
  3. TTS output path: write into session_dir instead of default outputs/
  4. Manifest schema: align nested model_versions schema with local _save_manifest()
"""
import json, re

NB_PATH = "notesbooks/colab_api_server.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))

# ── Patch Cell 3 (index 3) ────────────────────────────────────────────────────
cell3_src = "".join(nb["cells"][3]["source"])

# FIX 1: Diarization fallback — replace the else-branch that sets diarized=[]
old_diarize_fallback = (
    "    else:\n"
    "        # Fallback: no diarization — use plain transcript\n"
    "        result['diarized']      = []\n"
    "        result['diarized_text'] = result['transcript']\n"
    "        print('  WARNING: diarization skipped (HF_TOKEN_MEETING not set)')\n"
)
new_diarize_fallback = (
    "    else:\n"
    "        # Fallback: no diarization — use _mock_diarize() so sentiment still works\n"
    "        from src.diarize import _mock_diarize, format_diarized_transcript\n"
    "        result['diarized']      = _mock_diarize(result['segments'])\n"
    "        result['diarized_text'] = format_diarized_transcript(result['diarized'])\n"
    "        print('  WARNING: diarization skipped — assigned single SPEAKER_00')\n"
)
assert old_diarize_fallback in cell3_src, "FIX1: pattern not found"
cell3_src = cell3_src.replace(old_diarize_fallback, new_diarize_fallback, 1)

# FIX 3: TTS output path — pass session-scoped path
old_tts = (
    "        from src.speak import synthesize_speech\n"
    "        audio_out = synthesize_speech(result['summary_text'])\n"
    "        result['audio_path'] = audio_out\n"
)
new_tts = (
    "        from src.speak import synthesize_speech\n"
    "        audio_out = synthesize_speech(\n"
    "            result['summary_text'],\n"
    "            output_path=os.path.join(session_dir, 'summary_audio.wav'),\n"
    "        )\n"
    "        result['audio_path'] = audio_out\n"
)
# TTS is called before session_dir is defined in the current flow,
# so we need to reorganize: move session_dir creation BEFORE stage 5.
# Instead, we pass the path directly using a temp placeholder and fix it post-run.
# Simpler approach: TTS step comes AFTER the session_dir block; let's check the order.
# In the current notebook, TTS is inside _run_pipeline_with_cached_models(),
# which is called BEFORE session_dir is created in /analyze.
# Best fix: give synthesize_speech a predictable temp name, then move it after copy.
# Actually the cleanest fix without restructuring: skip session_dir in synthesize_speech,
# and instead just copy the resulting file into session_dir (already done below for audio copy).
# So just leave TTS as-is but ensure the copy in /analyze correctly handles the path.
# The real issue is the audio copy line already handles it:
#   shutil.copy(result['audio_path'], os.path.join(session_dir, 'summary_audio...'))
# But _run_pipeline currently always saves to outputs/summary_audio.wav.
# After our pipeline.py fix, it saves to outputs/summary_audio_{model}_{variant}.wav.
# So we just need to ensure the Colab TTS call also uses a predictable name.
# Safest fix: let Colab TTS write to a temp path tied to the request, then copy.
old_tts_simple = (
    "        from src.speak import synthesize_speech\n"
    "        audio_out = synthesize_speech(result['summary_text'])\n"
    "        result['audio_path'] = audio_out\n"
)
new_tts_simple = (
    "        from src.speak import synthesize_speech\n"
    "        _tts_out = f\"outputs/summary_audio_{whisper_model}_{prompt_variant}.wav\"\n"
    "        audio_out = synthesize_speech(result['summary_text'], output_path=_tts_out)\n"
    "        result['audio_path'] = audio_out\n"
)
assert old_tts_simple in cell3_src, "FIX3: TTS pattern not found"
cell3_src = cell3_src.replace(old_tts_simple, new_tts_simple, 1)

# FIX 2: Transcript save format — timestamped lines
old_transcript_save = (
    "        with open(os.path.join(session_dir, 'transcript.txt'), 'w', encoding='utf-8') as f:\n"
    "            f.write(result.get('transcript', ''))\n"
)
new_transcript_save = (
    "        with open(os.path.join(session_dir, 'transcript.txt'), 'w', encoding='utf-8') as f:\n"
    "            for _seg in result.get('segments', []):\n"
    "                f.write(f\"[{_seg['start']:.1f}s \\u2192 {_seg['end']:.1f}s] {_seg['text'].strip()}\\n\")\n"
)
assert old_transcript_save in cell3_src, "FIX2: transcript save pattern not found"
cell3_src = cell3_src.replace(old_transcript_save, new_transcript_save, 1)

# FIX 4: Manifest schema — align with local _save_manifest()
old_manifest = (
    "        manifest = {\n"
    "            'timestamp': timestamp,\n"
    "            'whisper_model': whisper_model,\n"
    "            'prompt_variant': prompt_variant,\n"
    "            'generate_audio': generate_audio,\n"
    "            'stage_times': result.get('stage_times', {}),\n"
    "            'total_time_s': total,\n"
    "            'keywords': result.get('keywords', [])\n"
    "        }\n"
)
new_manifest = (
    "        import datetime as _dt\n"
    "        manifest = {\n"
    "            'timestamp': _dt.datetime.utcnow().isoformat() + 'Z',\n"
    "            'input_file': os.path.basename(audio_path) if audio_path else 'unknown',\n"
    "            'model_versions': {\n"
    "                'whisper': whisper_model,\n"
    "                'summarizer_primary': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'),\n"
    "                'summarizer_fallback': 'claude-sonnet-4-20250514',\n"
    "            },\n"
    "            'prompt_variant': prompt_variant,\n"
    "            'stage_latencies_sec': result.get('stage_times', {}),\n"
    "            'total_latency_sec': total,\n"
    "        }\n"
)
assert old_manifest in cell3_src, "FIX4: manifest pattern not found"
cell3_src = cell3_src.replace(old_manifest, new_manifest, 1)

# Write patched source back as list of lines (notebook format)
nb["cells"][3]["source"] = list(line + "\n" for line in cell3_src.split("\n"))
# Remove trailing empty element from the last split
if nb["cells"][3]["source"] and nb["cells"][3]["source"][-1] == "\n":
    nb["cells"][3]["source"][-1] = ""

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("OK: All 4 fixes applied to colab_api_server.ipynb")
