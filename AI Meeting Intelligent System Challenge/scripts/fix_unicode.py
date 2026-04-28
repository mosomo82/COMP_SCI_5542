"""
Fix all Unicode arrow/symbol characters in src/*.py and scripts/run_eval.py
that break Windows cp1252 terminal encoding.
"""
import os, sys

replacements = {
    '\u25ba': '>>',   # ► black right-pointing pointer
    '\u2192': '->',   # → right arrow
    '\u2713': 'OK',   # ✓ check mark
    '\u2715': 'X',    # ✕ cross
    '\u2714': 'OK',   # ✔ heavy check
    '\u274c': 'X',    # ❌
    '\u2705': 'OK',   # ✅
}

src_dir = 'src'
targets = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.py')]
targets += ['scripts/run_eval.py']

for fpath in targets:
    txt = open(fpath, encoding='utf-8').read()
    changed = False
    for bad, good in replacements.items():
        if bad in txt:
            txt = txt.replace(bad, good)
            changed = True
    if changed:
        open(fpath, 'w', encoding='utf-8').write(txt)
        sys.stdout.buffer.write(("  Fixed: " + fpath + "\n").encode('utf-8'))

sys.stdout.buffer.write(b"Done.\n")
