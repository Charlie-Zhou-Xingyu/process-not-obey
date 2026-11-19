"""Seed-0 primary comparison on SecFid with a paired item bootstrap (10,000 resamples), on all
items and without the counting task (where C3's label is inflated by a count bias)."""
import json, sys, os
import numpy as np
E = "outputs/eval"
def scores(m):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(E, m, "secfid_scores.jsonl"))}
def clean(m):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(E, m, "secfid_clean_scores.jsonl"))}
names = [m for m in ["base_qwen35_2b", "C1_ignore_ordinary-seed0", "C2_ignore_fidelity-seed0", "C3_process_ordinary-seed0",
                     "C4_process_fidelity-seed0", "C5_target_ordinary-seed0", "C6_boundary_ordinary-seed0",
                     "llama31_8b_instr", "meta_secalign_8b"] if os.path.exists(os.path.join(E, m, "secfid_scores.jsonl"))]
S = {m: scores(m) for m in names}; C = {m: clean(m) for m in names}
ids_all = sorted(S[names[0]]); task = {i: S[names[0]][i]["task"] for i in ids_all}
subsets = {"all 1,168 items": ids_all, "without counting (861 items)": [i for i in ids_all if task[i] != "counting"]}
keys = ["executed", "processed", "ignored", "other", "safe_processed"]
for title, ids in subsets.items():
    print(f"\n### {title}")
    print(f"{'model':28s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + "   clean_correct")
    for m in names:
        print(f"{m:28s} " + " ".join(f"{100*np.mean([S[m][i][k] for i in ids]):9.1f}" for k in keys) + f"   {100*np.mean([C[m][i]['correct'] for i in ids]):9.1f}")
rng = np.random.default_rng(182)
def boot(a, b, ids, k, src=None):
    src = src or S
    d = np.array([src[a][i][k] - src[b][i][k] for i in ids])
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    bs = d[idx].mean(axis=1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
pairs = [("C3_process_ordinary-seed0", "C5_target_ordinary-seed0"), ("C3_process_ordinary-seed0", "C1_ignore_ordinary-seed0"),
         ("C5_target_ordinary-seed0", "base_qwen35_2b"), ("C1_ignore_ordinary-seed0", "base_qwen35_2b"), ("C3_process_ordinary-seed0", "base_qwen35_2b"),
         ("C4_process_fidelity-seed0", "C3_process_ordinary-seed0"), ("C4_process_fidelity-seed0", "C1_ignore_ordinary-seed0"), ("C2_ignore_fidelity-seed0", "C1_ignore_ordinary-seed0")]
for title, ids in subsets.items():
    print(f"\n### differences in points, 95% paired bootstrap interval over items ({title})")
    for a, b in pairs:
        if a not in S or b not in S: continue
        out = []
        for k in ("safe_processed", "ignored", "executed"):
            d, lo, hi = boot(a, b, ids, k); out.append(f"{k} {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        d, lo, hi = boot(a, b, ids, "correct", C); out.append(f"clean_correct {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        print(f"  {a.split('-')[0]:20s} - {b.split('-')[0]:20s}: " + " | ".join(out))
