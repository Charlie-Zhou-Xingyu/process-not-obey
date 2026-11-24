"""Held-out validation pairs for C3 and C5: 128 security + 128 ordinary pairs from the seed-1
data of the same condition, with every prompt that occurs in either seed-0 training file removed.
C3 and C5 get the same prompts (and the same rejected responses); only the preferred response differs.
    python mk_val.py <data/generated dir> <out dir>"""
import json, random, sys, collections
D, OUT = sys.argv[1], sys.argv[2]
rd = lambda p: [json.loads(l) for l in open(p)]
key = lambda r: json.dumps(r["prompt"], sort_keys=True)
seen = {key(r) for c in ("C3_process_ordinary", "C5_target_ordinary") for r in rd(f"{D}/seed0/{c}.jsonl")}
c3, c5 = rd(f"{D}/seed1/C3_process_ordinary.jsonl"), rd(f"{D}/seed1/C5_target_ordinary.jsonl")
by5 = collections.defaultdict(list)
for r in c5: by5[(key(r), r["kind"] if r["kind"] != "target" else "process")].append(r)
pairs = []
for r in c3:
    k = key(r)
    if k in seen: continue
    m = by5.get((k, r["kind"]))
    if not m: continue
    pairs.append((r, m[0]))
rng = random.Random(0)
sec = [p for p in pairs if p[0]["kind"] == "process"]; ordn = [p for p in pairs if p[0]["kind"] == "ordinary"]
pick = rng.sample(sec, 128) + rng.sample(ordn, 128)
for i, name in ((0, "C3_process_ordinary"), (1, "C5_target_ordinary")):
    with open(f"{OUT}/val_{name}.jsonl", "w") as f:
        for p in pick: f.write(json.dumps(p[i], ensure_ascii=False) + "\n")
print("candidates", len(sec), len(ordn), "| kinds C5:", collections.Counter(p[1]["kind"] for p in pick),
      "| rejected identical:", sum(p[0]["rejected"] == p[1]["rejected"] for p in pick), "/256",
      "| ordinary chosen identical:", sum(p[0]["chosen"] == p[1]["chosen"] for p in pick if p[0]["kind"] == "ordinary"))
