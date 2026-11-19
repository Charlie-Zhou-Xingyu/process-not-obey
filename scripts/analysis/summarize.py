"""Print every evaluation summary found in a results tree (local backup or server)."""
import json, sys
from pathlib import Path

def pct(x):
    return "   -  " if x is None else f"{100 * x:6.1f}"

def load(d, b):
    f = d / f"{b}_summary.json"
    return json.loads(f.read_text()) if f.exists() else None

def table(root, title):
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dirs:
        return
    print(f"\n## {title}  ({root})")
    print(f"{'model':30s} | SecFid: exec  proc  ignor other  secur fidel  safeP | clean | SEP asr | transf asr | IFEval | conflict user-data: proc exec ignor other")
    for d in dirs:
        s, c, sep, tr, ife, con = (load(d, b) for b in ("secfid", "secfid_clean", "sep", "transfer", "ifeval", "conflict"))
        part = None
        if s is None and (d / "secfid_partial_summary.json").exists():
            part = json.loads((d / "secfid_partial_summary.json").read_text())
        row = f"{d.name:30s} | "
        if s:
            row += " ".join(pct(s.get(k)) for k in ("executed", "processed", "ignored", "other", "security", "fidelity", "safe_processing"))
        elif part:
            row += pct(part.get("executed")) + " (partial: executed only; task labels on " + "+".join(part["task_scored"]["tasks"]) + ")"
        else:
            row += " " * 48
        row += " | " + pct(c.get("correct") if c else None)
        row += " | " + pct(sep.get("asr") if sep else None)
        row += "  | " + pct(tr.get("asr") if tr else None)
        row += "     | " + pct(ife.get("mean4") if ife else None)
        if con:
            ty = "user_data" if "user_data" in con else next((k for k in con if isinstance(con[k], dict)), None)
            if ty:
                row += f" | [{ty}] " + " ".join(pct(con[ty].get(k)) for k in ("processed", "executed", "ignored", "other"))
        print(row)

root = Path(sys.argv[1] if len(sys.argv) > 1 else "server_results")
table(root / "outputs/eval", "main study, seed 0")
table(root / "early/eval", "early SecFid scoring (scratch copy)")
for v in ("pno-sec2x", "pno-secbig"):
    table(root / "variants" / v / "outputs/eval", f"variant {v}")
