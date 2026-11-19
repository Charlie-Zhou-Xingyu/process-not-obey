"""Paired differences between two runs (A - B, points, 95% item bootstrap) on every benchmark both have.
A run is a name under outputs/eval or a directory (a variant's outputs/eval/<run>).
    python pair_diff.py A B [A2 B2 ...]"""
import json, sys, os
import numpy as np
E = "outputs/eval"
def d(m): return m if os.path.isdir(m) else os.path.join(E, m)
def tag(m):
    if os.path.isdir(m):
        parts = os.path.abspath(m).split(os.sep); m = parts[-4].replace("pno-", "") + ":" + parts[-1]
    return m.replace("-seed0", "")
def load(m, f):
    p = os.path.join(d(m), f)
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else None
rng = np.random.default_rng(182)
def boot(A, B, key, ids=None):
    ids = ids or sorted(set(A) & set(B))
    x = np.array([float(A[i][key]) - float(B[i][key]) for i in ids])
    bs = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return f"{100*x.mean():+6.1f} [{100*np.percentile(bs, 2.5):+.1f}, {100*np.percentile(bs, 97.5):+.1f}]"
def level(A, key, ids=None):
    ids = ids or sorted(A); return f"{100*np.mean([float(A[i][key]) for i in ids]):5.1f}"
args = sys.argv[1:]
for a, b in zip(args[0::2], args[1::2]):
    print(f"\n### {tag(a)}  minus  {tag(b)}")
    A, B = load(a, "secfid_scores.jsonl"), load(b, "secfid_scores.jsonl")
    if A and B:
        nc = [i for i in sorted(A) if A[i]["task"] != "counting"]
        for title, ids in (("SecFid, all items", None), ("SecFid, without counting", nc)):
            print(f"  {title:26s} " + " | ".join(f"{k} {level(A, k, ids)} vs {level(B, k, ids)}: {boot(A, B, k, ids)}" for k in ("safe_processed", "ignored", "executed")))
    A, B = load(a, "secfid_clean_scores.jsonl"), load(b, "secfid_clean_scores.jsonl")
    if A and B:
        task = {i: A[i]["task"] for i in A}
        cnt = [i for i in sorted(A) if task[i] == "counting"]; nc = [i for i in sorted(A) if task[i] != "counting"]
        print(f"  {'clean inputs, correct':26s} counting {level(A, 'correct', cnt)} vs {level(B, 'correct', cnt)}: {boot(A, B, 'correct', cnt)} | other tasks {level(A, 'correct', nc)} vs {level(B, 'correct', nc)}: {boot(A, B, 'correct', nc)}")
    for bench in ("sep", "transfer"):
        A, B = load(a, f"{bench}_scores.jsonl"), load(b, f"{bench}_scores.jsonl")
        if A and B:
            print(f"  {bench + ', attack success':26s} {level(A, 'attacked')} vs {level(B, 'attacked')}: {boot(A, B, 'attacked')}")
    A, B = load(a, "conflict_scores.jsonl"), load(b, "conflict_scores.jsonl")
    if A and B:
        for ty in ("fidelity", "user_data", "system_user"):
            ids = [i for i in sorted(A) if A[i]["type"] == ty and i in B]
            if ids:
                pa = {i: {"p": A[i]["label"] == "PROCESSED"} for i in ids}; pb = {i: {"p": B[i]["label"] == "PROCESSED"} for i in ids}
                print(f"  {'held-out ' + ty + ', exact':26s} {level(pa, 'p')} vs {level(pb, 'p')}: {boot(pa, pb, 'p', ids)}")
