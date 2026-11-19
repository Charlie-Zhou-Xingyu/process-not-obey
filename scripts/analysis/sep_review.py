"""SEP and transfer: per-attack rates, paired differences between runs (item bootstrap), and a
few outputs counted as attacked, to check that the witness match reflects an executed probe."""
import json, sys, os
import numpy as np
E = "outputs/eval"
order = ["base_qwen35_2b", "C1_ignore_ordinary-seed0", "C2_ignore_fidelity-seed0", "C3_process_ordinary-seed0",
         "C4_process_fidelity-seed0", "C5_target_ordinary-seed0", "C6_boundary_ordinary-seed0"]
def load(m, b):
    p = os.path.join(E, m, f"{b}_scores.jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else None
S = {m: load(m, "sep") for m in order}; S = {m: v for m, v in S.items() if v}
T = {m: load(m, "transfer") for m in order}; T = {m: v for m, v in T.items() if v}
first = next(iter(S.values())); print("sep score fields:", sorted(next(iter(first.values())).keys()))
atk = [k[4:] for k in sorted(next(iter(first.values())).keys()) if k.startswith("hit_")]
print(f"{'model':28s} any   " + "  ".join(f"{a[:14]:>14s}" for a in atk) + "   transfer")
for m in S:
    r = list(S[m].values())
    print(f"{m:28s} {100*np.mean([x['attacked'] for x in r]):5.1f} " + "  ".join(f"{100*np.mean([x['hit_'+a] for x in r if 'hit_'+a in x]):14.1f}" for a in atk)
          + (f"   {100*np.mean([x['attacked'] for x in T[m].values()]):5.1f}" if m in T else ""))
rng = np.random.default_rng(182)
def boot(A, B, key="attacked"):
    ids = sorted(set(A) & set(B)); d = np.array([float(A[i][key]) - float(B[i][key]) for i in ids])
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
print("\ndifferences in attack success (points, 95% paired item bootstrap; one training seed)")
pairs = [("C3_process_ordinary-seed0", "C5_target_ordinary-seed0"), ("C3_process_ordinary-seed0", "C1_ignore_ordinary-seed0"),
         ("C2_ignore_fidelity-seed0", "C1_ignore_ordinary-seed0"), ("C4_process_fidelity-seed0", "C1_ignore_ordinary-seed0"),
         ("C4_process_fidelity-seed0", "C3_process_ordinary-seed0"), ("C5_target_ordinary-seed0", "C1_ignore_ordinary-seed0"),
         ("C6_boundary_ordinary-seed0", "C3_process_ordinary-seed0")] + [(m, "base_qwen35_2b") for m in order[1:]]
for a, b in pairs:
    if a in S and b in S:
        d, lo, hi = boot(S[a], S[b]); line = f"  {a.split('-')[0]:20s} - {b.split('-')[0]:20s}: SEP {d:+6.1f} [{lo:+.1f}, {hi:+.1f}]"
        if a in T and b in T:
            d, lo, hi = boot(T[a], T[b]); line += f" | transfer {d:+6.1f} [{lo:+.1f}, {hi:+.1f}]"
        print(line)
# examples: C3 attacked where C1 is not
m = sys.argv[1] if len(sys.argv) > 1 else "C3_process_ordinary-seed0"
rows = [json.loads(l) for l in open(os.path.join(E, m, "sep.jsonl"))]
print("\nsep output fields:", sorted(rows[0].keys()))
shown = 0
for r in rows:
    if shown >= 4: break
    out = r["output"]; w = r.get("witness") or ""
    if w and w.lower() in out.lower() and r.get("attack") in ("straightforward", "completion"):
        shown += 1
        p = r["prompt"]
        print(f"--- {m} | attack={r.get('attack')} | witness={w!r}")
        print("  instruction:", (p.get("instruction") or "")[:200].replace("\n", " "))
        print("  data tail  :", repr((p.get("data") or "")[-260:]))
        print("  output     :", repr(out[:420]))
