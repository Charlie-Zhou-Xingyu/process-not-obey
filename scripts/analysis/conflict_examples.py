"""Held-out fidelity items: what does a model write where its answer is not the reference?"""
import json, sys, os
E = "outputs/eval"
m, transforms = sys.argv[1], sys.argv[2:]
outs = [json.loads(l) for l in open(os.path.join(E, m, "conflict.jsonl"))]
sc = [json.loads(l) for l in open(os.path.join(E, m, "conflict_scores.jsonl"))]
for t in transforms:
    shown = 0
    for o, s in zip(outs, sc):
        if o["type"] != "fidelity" or o["t_user"] != t or s["label"] != "OTHER" or shown >= 2:
            continue
        shown += 1
        p = o["prompt"]
        print(f"--- {m} | {t} | {o['id']}")
        print("  instruction:", p["instruction"][:150])
        print("  data       :", repr((p.get("data") or "")[:330]))
        print("  processed  :", repr(o["processed"][:330]))
        print("  ignored    :", repr(o["ignored"][:200]))
        print("  output     :", repr(o["output"][:330]))
