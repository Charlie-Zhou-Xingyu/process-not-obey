"""Rebuild every number, interval and figure in reports/report_v1 from the committed per-item scores.

    python scripts/report/make_figures.py            # writes reports/report_v1/figures/*.pdf
                                                     # and results/report_numbers.md

Needs numpy and matplotlib only; no GPU and no model outputs. Inputs:
results/scores/<setting>/<run>/<benchmark>_scores.jsonl (one line per test item) and
results/train_logs/<setting>/<run>.json (the trainer's log history). <setting> is "main" for plain
DPO and "sft" / "mixsft" for DPO + SFT (see results/README.md).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(ROOT, "results", "scores")
L = os.path.join(ROOT, "results", "train_logs")
FIG = os.path.join(ROOT, "reports", "report_v1", "figures")
B, SEED, BETA = 10_000, 182, 0.1

RUNS = {  # label in the report -> (setting, run)
    "untrained": ("main", "base_qwen35_2b"),
    "ops-drop": ("main", "C5_target_ordinary-seed0"),
    "ops-process": ("main", "C3_process_ordinary-seed0"),
    "standard": ("main", "C1_ignore_ordinary-seed0"),
    "standard+fid": ("main", "C2_ignore_fidelity-seed0"),
    "ops-drop SFT": ("sft", "C5_target_ordinary-seed0"),
    "ops-process SFT": ("sft", "C3_process_ordinary-seed0"),
    "standard SFT": ("sft", "C1_ignore_ordinary-seed0"),
    "standard+fid SFT": ("sft", "C2_ignore_fidelity-seed0"),
    "half-and-half SFT": ("mixsft", "C7_mix_ordinary-seed0"),
    "standard+process SFT": ("mixsft", "C9_ignorefull_mix-seed0"),
    "system-user": ("main", "C6_boundary_ordinary-seed0"),
    "ops-process+fid": ("main", "C4_process_fidelity-seed0"),
    "ops-process+fid SFT": ("sft", "C4_process_fidelity-seed0"),
    "half-and-half": ("mix", "C7_mix_ordinary-seed0"),
    "half-and-half+fid": ("mix", "C8_mix_fidelity-seed0"),
    "ops-process, no counting ops": ("nocount", "C3_process_ordinary-seed0"),
    "ops-process+fid, no counting ops": ("nocount", "C4_process_fidelity-seed0"),
    "standard 2x": ("sec2x", "C1_ignore_ordinary-seed0"),
    "standard 11x": ("secbig", "C1_ignore_ordinary-seed0"),
    "Meta-SecAlign-8B": ("main", "meta_secalign_8b"),
    "SecPO-8B": ("main", "secpo_llama31_8b"),
    "Llama-3.1-8B": ("main", "llama31_8b_instr"),
}


def load(label, bench):
    p = os.path.join(S, *RUNS[label], f"{bench}_scores.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), p
    return {r["id"]: r for r in rows}


def ifeval_mean(label):
    """Mean of prompt-level strict/loose and instruction-level strict/loose accuracy, as IFEval reports."""
    d = load(label, "ifeval")
    if d is None:
        return None
    r = list(d.values())
    parts = [np.mean([x["prompt_strict"] for x in r]), np.mean([x["prompt_loose"] for x in r]),
             np.mean([v for x in r for v in x["inst_strict"]]), np.mean([v for x in r for v in x["inst_loose"]])]
    return (100 * float(np.mean(parts)),)


def nocount(r):
    return r.get("task") != "counting"


def vec(d, key, ids):
    return np.array([float(d[i][key]) for i in ids])


def level(label, bench, key, filt=None):
    d = load(label, bench)
    if d is None:
        return None
    ids = sorted(i for i in d if filt is None or filt(d[i]))
    x = vec(d, key, ids)
    bs = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def diff(a, b, bench, key, filt=None):
    A, Bd = load(a, bench), load(b, bench)
    if A is None or Bd is None:
        return None
    assert set(A) == set(Bd), (a, b, bench)
    ids = sorted(i for i in A if filt is None or filt(A[i]))
    x = vec(A, key, ids) - vec(Bd, key, ids)
    bs = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def did(a, b, c, d, bench, key, filt=None):
    """(a - b) - (c - d), paired over items."""
    X = [load(m, bench) for m in (a, b, c, d)]
    ids = sorted(i for i in X[0] if filt is None or filt(X[0][i]))
    x = (vec(X[0], key, ids) - vec(X[1], key, ids)) - (vec(X[2], key, ids) - vec(X[3], key, ids))
    bs = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def fmt(t):
    return "n/a" if t is None else f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def numbers():
    out = ["# Numbers used in Report V1 (rebuilt by scripts/report/make_figures.py)", "",
           "Levels in %, with 95% item-bootstrap intervals in brackets (10,000 resamples of test items, seed 182);",
           "differences in percentage points from a paired bootstrap over the same items. IFEval has no interval.", "",
           "| model | dropped | obeyed | safe proc. | safe proc. (no counting) | dropped (no counting) | clean acc. | clean acc. (counting) | SEP attacked | transfer attacked | IFEval (mean of 4) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in RUNS:
        cells = [level(m, "secfid", "ignored"), level(m, "secfid", "executed"), level(m, "secfid", "safe_processed"),
                 level(m, "secfid", "safe_processed", nocount), level(m, "secfid", "ignored", nocount),
                 level(m, "secfid_clean", "correct"), level(m, "secfid_clean", "correct", lambda r: r.get("task") == "counting"),
                 level(m, "sep", "attacked"), level(m, "transfer", "attacked"),
                 ifeval_mean(m)]
        out.append(f"| {m} | " + " | ".join("" if c is None else (f"{c[0]:.1f}" if len(c) == 1 else fmt(c)) for c in cells) + " |")
    out += ["", "Held-out conflicts, share answered exactly as the processing target (%):", "",
            "| model | user-input conflicts | system-user conflicts | no conflict (fidelity) |", "|---|---|---|---|"]
    for m in RUNS:
        d = load(m, "conflict")
        if d is None:
            continue
        cells = []
        for ty in ("user_data", "system_user", "fidelity"):
            v = [r["label"] == "PROCESSED" for r in d.values() if r["type"] == ty]
            cells.append(f"{100 * np.mean(v):.1f}" if v else "")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", "| comparison | quantity | difference [95% CI] |", "|---|---|---|"]
    pairs = [("ops-process", "ops-drop"), ("ops-process SFT", "ops-drop SFT"), ("ops-drop", "standard"),
             ("ops-drop SFT", "standard SFT"), ("ops-drop", "untrained"), ("standard SFT", "untrained"),
             ("standard+fid SFT", "standard SFT"), ("half-and-half SFT", "standard SFT"),
             ("standard+process SFT", "standard SFT"), ("standard+process SFT", "half-and-half SFT")]
    for a, b in pairs:
        for name, bench, key, filt in [("safe processing", "secfid", "safe_processed", None),
                                       ("safe processing, no counting", "secfid", "safe_processed", nocount),
                                       ("dropped", "secfid", "ignored", None),
                                       ("dropped, no counting", "secfid", "ignored", nocount),
                                       ("SEP attacked", "sep", "attacked", None),
                                       ("transfer attacked", "transfer", "attacked", None)]:
            r = diff(a, b, bench, key, filt)
            if r is not None:
                out.append(f"| {a} - {b} | {name} | {fmt(r)} |")
    out.append(f"| (ops-process SFT - ops-drop SFT) - (ops-process - ops-drop) | SEP attacked | "
               f"{fmt(did('ops-process SFT', 'ops-drop SFT', 'ops-process', 'ops-drop', 'sep', 'attacked'))} |")
    open(os.path.join(ROOT, "results", "report_numbers.md"), "w").write("\n".join(out) + "\n")


def latex_tables():
    """Table bodies \\input by reports/report_v1/report_v1.tex, so the printed numbers come from the scores."""
    tdir = os.path.join(ROOT, "reports", "report_v1", "tables")
    os.makedirs(tdir, exist_ok=True)
    def row(m, cols):
        vals = []
        for bench, key, filt in cols:
            v = level(m, bench, key, filt)
            vals.append("--" if v is None else f"{v[0]:.1f}")
        return " & ".join(vals)
    main_cols = [("secfid", "ignored", None), ("secfid", "safe_processed", None), ("secfid", "safe_processed", nocount),
                 ("secfid_clean", "correct", None), ("sep", "attacked", None), ("transfer", "attacked", None)]
    groups = [("--", [("untrained", "Untrained model")]),
              ("DPO", [("ops-drop", "Text operations (drop)"), ("ops-process", "Text operations (process)"),
                       ("standard", "Alpaca (drop)"), ("standard+fid", "Alpaca (drop) + fidelity")]),
              ("DPO+SFT", [("ops-drop SFT", "Text operations (drop)"), ("ops-process SFT", "Text operations (process)"),
                           ("standard SFT", "Alpaca (drop)"), ("standard+fid SFT", "Alpaca (drop) + fidelity"),
                           ("half-and-half SFT", "1k Alpaca (drop) + 1k text op. (process)"),
                           ("standard+process SFT", "2k Alpaca (drop) + 1k text op. (process)")])]
    lines = []
    for gi, (loss, models) in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        for k, (m, name) in enumerate(models):
            lines.append(f"{loss if k == 0 else ''} & {name} & {row(m, main_cols)} \\\\")
    open(os.path.join(tdir, "main.tex"), "w").write("\n".join(lines) + "\n")
    other_cols = [("secfid", "ignored", None), ("secfid", "safe_processed", None), ("secfid", "safe_processed", nocount),
                  ("secfid_clean", "correct", None), ("sep", "attacked", None)]
    other = [("DPO", "ops-process+fid", "Text operations (process) + fidelity"),
             ("DPO", "system-user", "Text operations (process), system--user conflicts"),
             ("DPO", "half-and-half", "1k Alpaca (drop) + 1k text op. (process)"),
             ("DPO", "half-and-half+fid", "1k Alpaca (drop) + 1k text op. (process) + fidelity"),
             ("DPO", "standard 2x", "Alpaca (drop), 4,000 security pairs"),
             ("DPO", "standard 11x", "Alpaca (drop), 7,500 security pairs, 3 epochs"),
             ("DPO", "ops-process, no counting ops", "Text operations (process), no counting ops"),
             ("DPO", "ops-process+fid, no counting ops", "Text operations (process) + fidelity, no counting ops"),
             ("DPO+SFT", "ops-process+fid SFT", "Text operations (process) + fidelity"),
             (None, None, None),
             ("--", "Meta-SecAlign-8B", "Meta-SecAlign-8B (released)"),
             ("--", "SecPO-8B", "SecPO, Llama-3.1-8B (released)"),
             ("--", "Llama-3.1-8B", "Llama-3.1-8B-Instruct (undefended)")]
    lines = []
    for loss, m, name in other:
        if m is None:
            lines.append("\\midrule")
            continue
        lines.append(f"{loss} & {name} & {row(m, other_cols)} \\\\")
    open(os.path.join(tdir, "other.tex"), "w").write("\n".join(lines) + "\n")


def curve(setting, run, key):
    h = json.load(open(os.path.join(L, setting, run + ".json")))["log_history"]
    h = [e for e in h if key in e]
    return np.array([e["step"] for e in h]), np.array([float(e[key]) for e in h]) / BETA


def fig_curves(plt):
    """(a, b) training curves of six runs; (c, d) held-out losses of the two processing-target reruns."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    series = [("main", "C3_process_ordinary-seed0", "text operations (process), DPO", "#0072B2", "--"),
              ("main", "C5_target_ordinary-seed0", "text operations (drop), DPO", "#E69F00", "--"),
              ("main", "C1_ignore_ordinary-seed0", "Alpaca (drop), DPO", "#009E73", "--"),
              ("sft", "C3_process_ordinary-seed0", "text operations (process), DPO+SFT", "#0072B2", "-"),
              ("sft", "C5_target_ordinary-seed0", "text operations (drop), DPO+SFT", "#E69F00", "-"),
              ("sft", "C1_ignore_ordinary-seed0", "Alpaca (drop), DPO+SFT", "#009E73", "-")]
    for ax, key, title in zip(axes[0], ("rewards/chosen", "rewards/rejected"),
                              ("(a) training: preferred response", "(b) training: rejected response")):
        for setting, run, lab, col, ls in series:
            x, y = curve(setting, run, key)
            ax.plot(x, y, ls, color=col, lw=1.2, label=lab)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("log-prob. change (nats)", fontsize=8.5)
    runs = [("C3_dpo", "text operations (process), DPO", "--"), ("C3_sft", "text operations (process), DPO+SFT", "-")]
    have = all(os.path.exists(os.path.join(L, "valcurves", r + ".json")) for r, _, _ in runs)
    for ax, key, title, ylab in zip(axes[1], ("eval_dpo_loss", "eval_nll_loss"),
                                    ("(c) held-out DPO loss", "(d) held-out SFT loss"),
                                    ("DPO loss", "nats per token")):
        if have:
            for r, lab, ls in runs:
                h = json.load(open(os.path.join(L, "valcurves", r + ".json")))["log_history"]
                pts = [(e["step"], e[key]) for e in h if key in e]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", ms=3, color="#0072B2", lw=1.2, label=lab)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right" if key == "eval_dpo_loss" else "lower left")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab, fontsize=8.5)
    for ax in axes.flat:
        ax.set_xlabel("training step", fontsize=8.5)
        ax.tick_params(labelsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.2)
    fig.savefig(os.path.join(FIG, "curves.pdf"))


def fig_tradeoff(plt):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    kinds = [("untrained", "o", "0.4"), ("ops-drop", "s", "#E69F00"), ("ops-process", "D", "#0072B2"),
             ("standard", "^", "#009E73"), ("standard+fid", "v", "#56B4E9"), ("half-and-half", "P", "#D55E00"),
             ("standard+process", "X", "#CC79A7")]
    names = {"untrained": "untrained model", "ops-drop": "text operations (drop)",
             "ops-process": "text operations (process)", "standard": "Alpaca (drop)",
             "standard+fid": "Alpaca (drop) + fidelity", "half-and-half": "1k Alpaca (drop) + 1k text op. (process)",
             "standard+process": "2k Alpaca (drop) + 1k text op. (process)"}
    for k, mk, col in kinds:
        for m, filled in ((k, k == "untrained"), (k + " SFT", True)):
            if m not in RUNS or (m == k and k in ("half-and-half", "standard+process")):
                continue
            x, y = level(m, "sep", "attacked"), level(m, "secfid", "safe_processed")
            ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                        fmt=mk, ms=6, color=col, mfc=col if filled else "white", elinewidth=0.7, capsize=1.5,
                        label=names[k] if filled or k in ("ops-drop", "ops-process", "standard", "standard+fid") and False else None)
        ax.plot([], [], mk, color=col, ms=6, label=names[k])
    ax.plot([], [], "o", color="0.3", mfc="white", ms=6, label="hollow = plain DPO")
    ax.plot([], [], "o", color="0.3", ms=6, label="filled = DPO + SFT")
    h, lab = ax.get_legend_handles_labels()
    keep = [(a, b) for a, b in zip(h, lab) if not b.startswith("_")]
    seen, hh, ll = set(), [], []
    for a, b in keep:
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    ax.legend(hh, ll, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xlabel("SEP: tasks attacked (%), lower is more secure", fontsize=9)
    ax.set_ylabel("SecFid: safe processing (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.pdf"))


def fig_mixtures(plt):
    models = ["untrained", "standard SFT", "half-and-half SFT", "standard+process SFT"]
    names = ["untrained", "Alpaca\n(drop)", "1k Alpaca +\n1k text op.", "2k Alpaca +\n1k text op."]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, (bench, key, filt, title) in zip(axes, [("secfid", "ignored", nocount, "(a) SecFid: sentence dropped,\nno counting tasks (%)"),
                                                     ("sep", "attacked", None, "(b) SEP: tasks attacked (%)")]):
        v = [level(m, bench, key, filt) for m in models]
        ax.bar(range(4), [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]],
               color=["0.7", "#009E73", "#D55E00", "#CC79A7"], capsize=2, width=0.6)
        ax.set_xticks(range(4), names, fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixtures.pdf"))


def fig_validation(plt):
    """Held-out losses of the two processing-target reruns (plain DPO and DPO+SFT), from results/train_logs/valcurves/."""
    runs = [("C3_dpo", "plain DPO", "--"), ("C3_sft", "DPO+SFT", "-")]
    paths = {r: os.path.join(L, "valcurves", r + ".json") for r, _, _ in runs}
    if not all(os.path.exists(p) for p in paths.values()):
        print("validation logs missing; skipping validation figure")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for r, lab, ls in runs:
        h = json.load(open(paths[r]))["log_history"]
        for ax, key in zip(axes, ("eval_dpo_loss", "eval_nll_loss", "loss")):
            pts = [(e["step"], e[key]) for e in h if key in e]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o" if key != "loss" else None,
                    ms=3, color="#0072B2", lw=1.2, label=lab)
    for ax, title in zip(axes, ("(a) held-out DPO loss", "(b) held-out SFT loss\n(nats per token)",
                                "(c) training loss\n(the objective being trained)")):
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("training step", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "validation.pdf"))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    numbers()
    latex_tables()
    fig_curves(plt)
    fig_tradeoff(plt)
    fig_mixtures(plt)
    print("wrote", FIG, "and results/report_numbers.md")
