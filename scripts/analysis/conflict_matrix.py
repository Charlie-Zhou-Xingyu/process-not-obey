"""Held-out conflict benchmark: share of exact 'processed' answers by trusted transform, one column per model."""
import json, sys, os, collections
E = "outputs/eval"
models = [m for m in sys.argv[1:] if os.path.exists(os.path.join(m if os.path.isdir(m) else os.path.join(E, m), "conflict_scores.jsonl"))]
def rows(m, f):
    return [json.loads(l) for l in open(os.path.join(m if os.path.isdir(m) else os.path.join(E, m), f))]
def tag(m):
    if os.path.isdir(m):
        parts = os.path.abspath(m).split(os.sep); m = parts[-4].replace("pno-", "") + ":" + parts[-1]
    return m.replace("-seed0", "").replace("_ordinary", "_ord").replace("_fidelity", "_fid")[:15]
data = {}
for m in models:
    outs, sc = rows(m, "conflict.jsonl"), rows(m, "conflict_scores.jsonl")
    d = collections.defaultdict(collections.Counter)
    for o, s in zip(outs, sc):
        d[(o["type"], o["t_user"])][s["label"]] += 1
        d[(o["type"], "ALL")][s["label"]] += 1
    data[m] = d
for ty in ("fidelity", "user_data", "system_user"):
    print(f"\n[{ty}] exact PROCESSED answers (%), then OTHER (%) in brackets; IGNORED shown as i=")
    print(f"{'trusted transform':26s} {'n':>4s} " + " ".join(f"{tag(m):>17s}" for m in models))
    keys = sorted({k[1] for m in models for k in data[m] if k[0] == ty}, key=lambda t: (t == "ALL", t))
    for t in keys:
        n = sum(data[models[0]][(ty, t)].values())
        cells = []
        for m in models:
            c = data[m][(ty, t)]; tot = max(1, sum(c.values()))
            ig = 100 * c.get("IGNORED", 0) / tot
            cells.append(f"{100*c.get('PROCESSED',0)/tot:5.1f} [{100*c.get('OTHER',0)/tot:4.0f}]" + (f" i={ig:.0f}" if ig >= 1 else "     "))
        print(f"{t:26s} {n:4d} " + " ".join(f"{c:>17s}" for c in cells))
