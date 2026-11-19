"""SecFid editing and translation: did the model do the task, or copy the input (which would also be
labelled 'processed' on attacked inputs because the copy contains the probe sentence)?
For each item: similarity (difflib ratio on words) of the output to the input text and to the
gold references."""
import json, sys, os, difflib, statistics as st
from pno.eval import secfid
from pno.io import load_config
cfg = load_config()
rows = {r["id"]: r for r in secfid.raw_rows(cfg)}
models = sys.argv[1:]
def outputs(m, b):
    p = os.path.join("outputs/eval", m, b + ".jsonl")
    return {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p)} if os.path.exists(p) else None
def sim(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()
def vals_of(ref):
    v = ref.get("values", []) if isinstance(ref, dict) else ref
    return [v] if isinstance(v, str) else list(v)
def best(resp, ref):
    return max((sim(resp, v) for v in vals_of(ref)), default=float("nan"))
ex = next(r for r in rows.values() if r["metadata"]["task"] == "editing")
print("example editing item:", json.dumps({"instruction": ex["instruction"], "base": ex["data"]["base"][:260], "probe": ex["probe"].get("text"),
      "processed_ref": str(ex["eval"]["references"]["processed"])[:420], "ignored_ref": str(ex["eval"]["references"]["ignored"])[:300]}, ensure_ascii=False)[:1700])
import collections
kinds = collections.Counter(r["instruction"][:60] for r in rows.values() if r["metadata"]["task"] == "editing")
print("editing instructions (first 60 chars), most common:", kinds.most_common(6))
for task in ("editing", "translation"):
    ids = [i for i, r in rows.items() if r["metadata"]["task"] == task][:150]
    print(f"\n=== {task}: first {len(ids)} items; mean word-level similarity of the output to ...")
    print(f"{'model':28s} | attacked input: to input  to processed-ref  to ignored-ref  len(out)/len(in) | clean input: to input  to clean-ref  len ratio")
    for m in models:
        a, c = outputs(m, "secfid"), outputs(m, "secfid_clean")
        if a is None or c is None: continue
        A = [(sim(a[i], rows[i]["data"]["input"]), best(a[i], rows[i]["eval"]["references"]["processed"]),
              best(a[i], rows[i]["eval"]["references"]["ignored"]), len(a[i].split()) / max(1, len(rows[i]["data"]["input"].split()))) for i in ids]
        C = [(sim(c[i], rows[i]["data"]["base"]), best(c[i], rows[i]["eval"]["references"]["ignored"]),
              len(c[i].split()) / max(1, len(rows[i]["data"]["base"].split()))) for i in ids]
        mA = [st.mean(x[k] for x in A) for k in range(4)]; mC = [st.mean(x[k] for x in C) for k in range(3)]
        print(f"{m:28s} |                 {mA[0]:.3f}      {mA[1]:.3f}           {mA[2]:.3f}          {mA[3]:.2f}       |              {mC[0]:.3f}     {mC[1]:.3f}       {mC[2]:.2f}")
