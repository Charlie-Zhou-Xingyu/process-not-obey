"""SecFid by task and by persuasion for every scored model under <root>/outputs/eval."""
import json, sys, os, glob
root = sys.argv[1] if len(sys.argv) > 1 else "."
nan = float("nan")
order = ["base_qwen35_2b", "C1_ignore_ordinary-seed0", "C2_ignore_fidelity-seed0", "C3_process_ordinary-seed0",
         "C4_process_fidelity-seed0", "C5_target_ordinary-seed0", "C6_boundary_ordinary-seed0",
         "llama31_8b_instr", "meta_secalign_8b", "secpo_llama31_8b", "base_qwen35_2b_concat"]
dirs = sorted(glob.glob(os.path.join(root, "outputs/eval/*")), key=lambda d: (order.index(os.path.basename(d)) if os.path.basename(d) in order else 99, d))
print("SecFid with injection, by task: processed / ignored / executed / other (%)")
for d in dirs:
    f = os.path.join(d, "secfid_summary.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    print(f"  {os.path.basename(d):28s} " + " | ".join(
        f"{t[:6]:6s} {100*v.get('processed', nan):5.1f} {100*v.get('ignored', nan):5.1f} {100*v.get('executed', nan):5.1f} {100*v.get('other', nan):5.1f}"
        for t, v in j.get("by_task", {}).items()))
print("executed by persuasion (%)")
for d in dirs:
    f = os.path.join(d, "secfid_summary.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    print(f"  {os.path.basename(d):28s} " + "  ".join(f"{t} {100*v.get('executed', nan):.1f}" for t, v in j.get("by_persuasion", {}).items()))
print("clean inputs: correct by task (%), and how often the 'executed' check fires without any injection")
for d in dirs:
    f = os.path.join(d, "secfid_clean_summary.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    print(f"  {os.path.basename(d):28s} all {100*j.get('correct', nan):5.1f} | " + "  ".join(f"{t[:6]} {100*v.get('correct', nan):5.1f}" for t, v in j.get("by_task", {}).items())
          + f" | executed_false_positive {j.get('executed_false_positive')} | truncated {j.get('truncated_outputs')}")
