"""SecFid counting and extraction: is a 'processed' label on attacked inputs explained by a general
tendency to give the larger answer? Compare each model's labels on the attacked input and on
the clean input of the same item (the clean input has no probe, so 'processed' there is an error)."""
import json, sys, os, re, collections
from pno.eval import secfid
from pno.io import load_config
cfg = load_config()
rows = {r["id"]: r for r in secfid.raw_rows(cfg)}
models = sys.argv[1:]
def outputs(m, b):
    p = os.path.join("outputs/eval", m, b + ".jsonl")
    return {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p)} if os.path.exists(p) else None
def label(r, resp):
    ev = r["eval"]; refs, norm = ev["references"], ev["normalization"]
    if secfid.reference_matches(resp, refs["processed"], norm): return "processed"
    if secfid.reference_matches(resp, refs["ignored"], norm): return "ignored"
    return "other"
ex = next(r for r in rows.values() if r["metadata"]["task"] == "counting")
print("example counting item:", json.dumps({"instruction": ex["instruction"], "base": ex["data"]["base"][:300], "input_tail": ex["data"]["input"][-260:],
      "probe": {k: (v if not isinstance(v, str) else v[:200]) for k, v in ex["probe"].items()}, "eval": ex["eval"]}, ensure_ascii=False)[:1900])
for task in ("counting", "extraction"):
    ids = [i for i, r in rows.items() if r["metadata"]["task"] == task and r["eval"]["method"] == "programmatic"]
    print(f"\n=== {task}: {len(ids)} items. label on attacked input / label on clean input of the same item (%)")
    print(f"{'model':28s} attacked: proc  ign  oth | clean: proc   ign   oth | proc on both | proc attacked only (net)")
    for m in models:
        a, c = outputs(m, "secfid"), outputs(m, "secfid_clean")
        if a is None or c is None: continue
        la = {i: label(rows[i], a[i]) for i in ids}; lc = {i: label(rows[i], c[i]) for i in ids}
        n = len(ids); f = lambda d, k: 100 * sum(v == k for v in d.values()) / n
        both = 100 * sum(la[i] == "processed" and lc[i] == "processed" for i in ids) / n
        net = 100 * sum(la[i] == "processed" and lc[i] != "processed" for i in ids) / n
        print(f"{m:28s}        {f(la,'processed'):5.1f} {f(la,'ignored'):5.1f} {f(la,'other'):5.1f} |      {f(lc,'processed'):5.1f} {f(lc,'ignored'):5.1f} {f(lc,'other'):5.1f} |    {both:5.1f}     |   {net:5.1f}")
# a few clean counting items where the last model is labelled 'processed' without any probe
m = models[-1]; c = outputs(m, "secfid_clean"); b = outputs(models[0], "secfid_clean")
shown = 0
for i, r in rows.items():
    if r["metadata"]["task"] != "counting" or shown >= 3: continue
    if label(r, c[i]) == "processed" and label(r, b[i]) == "ignored":
        shown += 1
        print(f"\n--- clean item {i}: instruction: {r['instruction'][:160]}")
        print("    refs: ignored =", r["eval"]["references"]["ignored"].get("values"), "| processed =", r["eval"]["references"]["processed"].get("values"))
        print(f"    {models[0]}: {b[i][:420]!r}")
        print(f"    {m}: {c[i][:420]!r}")
