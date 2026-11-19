"""SecFid counting on clean inputs: the stated count minus the gold count, per model.
The stated count is the first bold number, else the number after 'count:'/'total', else the last number."""
import json, sys, os, re, collections
from pno.eval import secfid
from pno.io import load_config
rows = {r["id"]: r for r in secfid.raw_rows(load_config()) if r["metadata"]["task"] == "counting"}
W = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"zero":0}
def gold(r):
    for v in r["eval"]["references"]["ignored"]["values"]:
        if str(v).isdigit(): return int(v)
def stated(out):
    m = re.search(r"\*\*\s*(\d+)\s*\*\*", out) or re.search(r"(?i)(?:count|total)[^0-9\n]{0,40}?(\d+)", out)
    if m: return int(m.group(1))
    nums = re.findall(r"(?<![\w.])(\d+)(?![\w])", out)
    if nums: return int(nums[-1])
    for w, n in W.items():
        if re.search(rf"(?i)\b{w}\b", out): return n
    return None
def listed(out):
    return len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S", out))
print(f"{'model':28s} n | stated - gold: <=-2   -1    0   +1   +2  >=+3  none | mean stated-gold | stated != items it lists (when it lists) | median words")
for m in sys.argv[1:]:
    p = os.path.join(m if os.path.isdir(m) else os.path.join("outputs/eval", m), "secfid_clean.jsonl")
    if not os.path.exists(p): continue
    parts = os.path.abspath(m).split(os.sep)
    name = ((parts[-4].replace("pno-", "") + ":" + parts[-1]) if os.path.isdir(m) else m).replace("-seed0", "")
    outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p)}
    d = collections.Counter(); diffs = []; mism = tot = 0; lens = []
    for i, r in rows.items():
        s, g = stated(outs[i]), gold(r); lens.append(len(outs[i].split()))
        if s is None or g is None: d["none"] += 1; continue
        k = s - g; diffs.append(k); d["<=-2" if k <= -2 else ">=+3" if k >= 3 else f"{k:+d}" if k else "0"] += 1
        L = listed(outs[i])
        if L >= 2: tot += 1; mism += (L != s)
    n = len(rows); f = lambda k: 100 * d[k] / n
    lens.sort()
    print(f"{name:28s} {n} |              {f('<=-2'):5.1f} {f('-1'):5.1f} {f('0'):5.1f} {f('+1'):5.1f} {f('+2'):5.1f} {f('>=+3'):5.1f} {f('none'):5.1f} | {sum(diffs)/max(1,len(diffs)):+.2f}            | {100*mism/max(1,tot):5.1f} (of {tot})                       | {lens[len(lens)//2]}")
m = sys.argv[-1]; outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(os.path.join(m if os.path.isdir(m) else os.path.join("outputs/eval", m), "secfid_clean.jsonl"))}
for i in list(rows)[:3]:
    print(f"--- {m} {i} gold {gold(rows[i])}: {outs[i][:330]!r}")
