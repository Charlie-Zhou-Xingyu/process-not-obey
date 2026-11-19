"""SecFid clean counting: is a wrong count the number of sentences (or lines) of the input rather
than the number of entities asked for?  Also prints a few outputs."""
import json, sys, os, re, collections
from pno.eval import secfid
from pno.io import load_config
rows = {r["id"]: r for r in secfid.raw_rows(load_config()) if r["metadata"]["task"] == "counting"}
def gold(r):
    for v in r["eval"]["references"]["ignored"]["values"]:
        if str(v).isdigit(): return int(v)
def stated(out):
    m = re.search(r"\*\*\s*(\d+)\s*\*\*", out) or re.search(r"(?i)(?:count|total)[^0-9\n]{0,40}?(\d+)", out)
    if m: return int(m.group(1))
    nums = re.findall(r"(?<![\w.])(\d+)(?![\w])", out)
    return int(nums[0]) if nums and re.fullmatch(r"\s*\d+\.?\s*", out) else (int(nums[-1]) if nums else None)
def n_sent(t): return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])
for m in sys.argv[1:]:
    p = os.path.join(m if os.path.isdir(m) else os.path.join("outputs/eval", m), "secfid_clean.jsonl")
    outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p)}
    c = collections.Counter(); shape = collections.Counter()
    for i, r in rows.items():
        o = outs[i].strip(); s, g, ns = stated(o), gold(r), n_sent(r["data"]["base"])
        shape["bare number" if re.fullmatch(r"\d+\.?", o) else "short (<=25 words)" if len(o.split()) <= 25 else "long"] += 1
        if s is None: c["no number"] += 1
        elif s == g: c["== entities asked for"] += 1
        elif s == ns: c["== sentences in the input"] += 1
        elif s == g + 1: c["entities + 1"] += 1
        else: c["something else"] += 1
    name = os.path.abspath(m).split(os.sep)[-4].replace("pno-", "") + ":" + os.path.basename(m.rstrip("/")) if os.path.isdir(m) else m
    n = len(rows)
    print(f"{name.replace('-seed0',''):30s} " + " | ".join(f"{k} {100*v/n:.1f}" for k, v in c.most_common()) + "   || shape: " + ", ".join(f"{k} {100*v/n:.0f}" for k, v in shape.most_common()))
m = sys.argv[-1]; p = os.path.join(m if os.path.isdir(m) else os.path.join("outputs/eval", m), "secfid_clean.jsonl")
outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p)}
for i in list(rows)[:5]:
    print(f"--- {i}: entities {gold(rows[i])}, sentences {n_sent(rows[i]['data']['base'])} | {outs[i][:200]!r}")
