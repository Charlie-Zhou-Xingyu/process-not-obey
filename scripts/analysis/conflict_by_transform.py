"""Held-out conflict benchmark: labels by trusted transform, and for the counting transforms
how far the answer is from the processed and the ignored reference."""
import json, sys, os, re, collections
E = "outputs/eval"
models = sys.argv[1:]
def rows(m, f):
    p = os.path.join(m if os.path.isdir(m) else os.path.join(E, m), f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
first = rows(models[0], "conflict.jsonl")
print("output fields:", sorted(first[0].keys())); print("score fields:", sorted(rows(models[0], "conflict_scores.jsonl")[0].keys()))
ex = first[0]; print("example:", json.dumps({k: (v if not isinstance(v, str) else v[:160]) for k, v in ex.items() if k != "prompt"}, ensure_ascii=False)[:700])
for m in models:
    outs, sc = rows(m, "conflict.jsonl"), rows(m, "conflict_scores.jsonl")
    if outs is None or sc is None: continue
    print(f"\n== {m}")
    tab = collections.defaultdict(collections.Counter)
    cnt = collections.defaultdict(collections.Counter)
    for o, s in zip(outs, sc):
        ty = s.get("type") or o.get("type"); meta = o.get("meta") or {}
        tu = meta.get("t_user") or o.get("t_user") or "?"
        tab[(ty, tu)][s["label"]] += 1
        if tu.startswith("count_"):
            out = o["output"].strip()
            nums = re.findall(r"-?\d+", out)
            ref_p, ref_i = o.get("processed") or o.get("chosen"), o.get("ignored")
            try:
                d = int(nums[-1]) - int(str(ref_p).strip()) if nums else None
            except Exception:
                d = None
            cnt[ty]["no number" if d is None else ("== processed ref" if d == 0 else f"processed ref {d:+d}" if abs(d) <= 2 else "further off")] += 1
            cnt[ty]["bare number" if re.fullmatch(r"-?\d+\.?", out) else "number in text"] += 1
    for ty in sorted({k[0] for k in tab}):
        print(f"  [{ty}] label counts by trusted transform (processed/executed/ignored/other):")
        for (t, tu), c in sorted(tab.items()):
            if t != ty: continue
            n = sum(c.values()); print(f"     {tu:26s} n={n:3d}  " + "  ".join(f"{k[:4]} {100*c.get(k,0)/n:5.1f}" for k in ("PROCESSED", "EXECUTED", "IGNORED", "OTHER")) + ("" if set(c) <= {"PROCESSED","EXECUTED","IGNORED","OTHER"} else f"  labels={dict(c)}"))
        if cnt[ty]: print(f"     counting transforms, answer vs processed reference: {dict(cnt[ty])}")
