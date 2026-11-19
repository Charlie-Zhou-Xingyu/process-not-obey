#!/usr/bin/env bash
# Full SecFid scoring of every model whose outputs exist, in a scratch copy (/workspace/early/eval),
# so the queue's own output directories are not written to. Skips models already scored there.
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
E=/workspace/early/eval; mkdir -p $E
for d in outputs/eval/*/; do
  n=$(basename "$d")
  [ -f "$d/secfid.jsonl" ] && [ -f "$d/secfid_clean.jsonl" ] || continue
  [ -f "$E/$n/secfid_summary.json" ] && [ -f "$E/$n/secfid_clean_summary.json" ] && continue
  mkdir -p "$E/$n"; cp "$d/secfid.jsonl" "$d/secfid_clean.jsonl" "$E/$n/"
  python -m pno.eval.run --name "$n" --benchmarks secfid secfid_clean --score_only --out $E > "$E/$n/score.log" 2>&1 \
    || { echo "scoring failed for $n"; tail -3 "$E/$n/score.log"; }
done
python - <<'PY'
import json
from pathlib import Path
E = Path("/workspace/early/eval")
keys = ("executed", "processed", "ignored", "other", "security", "fidelity", "safe_processed")
print(f"{'model':28s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + "   clean_correct")
for d in sorted(E.iterdir()):
    s, c = d / "secfid_summary.json", d / "secfid_clean_summary.json"
    if not s.exists():
        continue
    j = json.loads(s.read_text()); cj = json.loads(c.read_text()) if c.exists() else {}
    print(f"{d.name:28s} " + " ".join(f"{100 * j.get(k, float('nan')):9.1f}" for k in keys)
          + f"   {100 * cj.get('correct', float('nan')):.1f}")
PY
