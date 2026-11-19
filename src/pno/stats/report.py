"""Collect per-item scores from every evaluated model into tables and bootstrap contrasts.

Model directories are named ``<condition>-seed<k>`` for trained runs (e.g.
``C3_process_ordinary-seed0``) and anything else for the base model and references.

    python -m pno.stats.report --eval_dir outputs/eval --out outputs/report

Writes ``metrics.csv`` (one row per model x benchmark metric), ``conditions.csv`` (mean
over seeds), ``contrasts.csv`` (config ``comparisons`` with 95% paired-bootstrap CIs),
``hypothesis.json`` (the pre-registered decision rule) and ``report.md``.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pno.io import load_config, read_jsonl, write_json
from pno.stats.bootstrap import contrast_bootstrap

# benchmark -> {metric name: per-item score field}. All are "higher is better" rates in [0, 1].
METRICS = {
    "secfid": {"safe_processing": "safe_processed", "security": "secure", "fidelity": "faithful",
               "processed": "processed", "other": "other"},
    "secfid_clean": {"correct": "correct"},
    "sep": {"security": "secure"},
    "transfer": {"security": "secure"},
    "ifeval": {"prompt_strict": "prompt_strict", "prompt_loose": "prompt_loose"},
    "conflict": {"safe_processed": "safe_processed", "executed": "executed", "ignored": "ignored"},
}
HEADLINE = [("secfid", "safe_processing"), ("secfid", "security"), ("secfid", "fidelity"),
            ("sep", "security"), ("transfer", "security"), ("ifeval", "prompt_strict"), ("secfid_clean", "correct")]
_RUN = re.compile(r"^(?P<cond>.+)-seed(?P<seed>\d+)$")


def load_scores(eval_dir: Path) -> dict:
    """scores[bench][metric][model] = {item_id: value}"""
    scores: dict = defaultdict(lambda: defaultdict(dict))
    for mdir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        for bench, metrics in METRICS.items():
            f = mdir / f"{bench}_scores.jsonl"
            if not f.exists():
                continue
            rows = read_jsonl(f)
            for metric, field in metrics.items():
                vals = {r["id"]: float(r[field]) for r in rows if field in r}
                if vals:
                    scores[bench][metric][mdir.name] = vals
    return scores


def group_runs(models: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for m in models:
        mt = _RUN.match(m)
        groups[mt.group("cond") if mt else m].append(m)
    return {k: sorted(v) for k, v in groups.items()}


def condition_matrix(per_model: dict[str, dict], runs: list[str], ids: list[str]) -> np.ndarray:
    return np.array([[per_model[r][i] for i in ids] for r in runs], dtype=float)


def build_report(eval_dir: Path, cfg: dict) -> dict:
    scores = load_scores(eval_dir)
    bcfg = cfg["eval"]["bootstrap"]
    metrics_rows, cond_rows, contrast_rows, warnings = [], [], [], []
    matrices: dict = {}
    for bench, by_metric in scores.items():
        for metric, per_model in by_metric.items():
            ids = sorted(set.intersection(*(set(v) for v in per_model.values())))
            full = max(len(v) for v in per_model.values())
            short = sorted(m for m, v in per_model.items() if len(v) < full)
            if short:
                warnings.append(f"{bench}:{metric} is computed on the {len(ids)} items common to all models; "
                                f"these have fewer than {full}: {', '.join(short)}")
            groups = group_runs(list(per_model))
            for model, vals in per_model.items():
                metrics_rows.append({"benchmark": bench, "metric": metric, "model": model,
                                     "value": float(np.mean([vals[i] for i in ids])), "n_items": len(ids)})
            for cond, runs in groups.items():
                m = condition_matrix(per_model, runs, ids)
                matrices[(bench, metric, cond)] = m
                seed_means = m.mean(axis=1)
                cond_rows.append({"benchmark": bench, "metric": metric, "condition": cond, "n_seeds": len(runs),
                                  "mean": float(seed_means.mean()),
                                  "seed_min": float(seed_means.min()), "seed_max": float(seed_means.max()),
                                  "n_items": len(ids)})
            for cname, weights in cfg.get("comparisons", {}).items():
                if not all((bench, metric, c) in matrices for c in weights):
                    continue
                r = contrast_bootstrap({c: matrices[(bench, metric, c)] for c in weights}, weights,
                                       n_boot=bcfg["n_boot"], alpha=bcfg["alpha"])
                contrast_rows.append({"comparison": cname, "benchmark": bench, "metric": metric,
                                      "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                      "n_items": r["n_items"],
                                      "n_seeds": min(matrices[(bench, metric, c)].shape[0] for c in weights)})
    for w in warnings:
        print("[report] WARNING: " + w)
    return {"metrics": metrics_rows, "conditions": cond_rows, "contrasts": contrast_rows, "warnings": warnings}


def hypothesis_check(contrasts: list[dict], margin: float) -> dict:
    """Proposal decision rule for the primary comparison (process-style vs target control)."""
    def find(bench, metric):
        for c in contrasts:
            if c["comparison"] == "primary_process_vs_target" and c["benchmark"] == bench and c["metric"] == metric:
                return c
        return None

    sp, sep, tr = find("secfid", "safe_processing"), find("sep", "security"), find("transfer", "security")
    out = {"margin": margin, "safe_processing": sp, "sep_security": sep, "transfer_security": tr}
    out["safe_processing_ci_above_zero"] = bool(sp and sp["ci_low"] > 0)
    # the proposal's rule is on the point estimate ("drops by at most 2 points"); we also report
    # the stricter CI version (the whole CI above -margin)
    for key, c in (("sep", sep), ("transfer", tr)):
        out[f"{key}_drop_within_margin"] = bool(c and c["estimate"] >= -margin)
        out[f"{key}_noninferior_by_ci"] = bool(c and c["ci_low"] > -margin)
    have_all = all([sp, sep, tr])
    out["supported"] = bool(have_all and out["safe_processing_ci_above_zero"]
                            and out["sep_drop_within_margin"] and out["transfer_drop_within_margin"])
    out["complete"] = have_all
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def to_markdown(rep: dict, hyp: dict) -> str:
    pct = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    conds = sorted({r["condition"] for r in rep["conditions"]})
    idx = {(r["benchmark"], r["metric"], r["condition"]): r for r in rep["conditions"]}
    lines = ["# Results\n", "All numbers are percentages; trained conditions are means over seeds "
             "(range over seeds in brackets).\n",
             "| model | " + " | ".join(f"{b}:{m}" for b, m in HEADLINE) + " |",
             "|---|" + "---|" * len(HEADLINE)]
    for c in conds:
        cells = []
        for b, m in HEADLINE:
            r = idx.get((b, m, c))
            if r is None:
                cells.append("-")
            elif r["n_seeds"] > 1:
                cells.append(f"{pct(r['mean'])} [{pct(r['seed_min'])}, {pct(r['seed_max'])}]")
            else:
                cells.append(pct(r["mean"]))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["\n## Contrasts (difference in points, 95% paired bootstrap over items)\n",
              "| comparison | benchmark:metric | diff | 95% CI | seeds |", "|---|---|---|---|---|"]
    for r in rep["contrasts"]:
        if (r["benchmark"], r["metric"]) in HEADLINE:
            lines.append(f"| {r['comparison']} | {r['benchmark']}:{r['metric']} | {100 * r['estimate']:+.1f} | "
                         f"[{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | {r['n_seeds']} |")
    lines += ["\n## Primary hypothesis (process-style vs target control)\n",
              f"- safe-processing CI above zero: **{hyp['safe_processing_ci_above_zero']}**",
              f"- SEP security drop within {100 * hyp['margin']:.0f} points: **{hyp['sep_drop_within_margin']}** "
              f"(CI version: {hyp['sep_noninferior_by_ci']})",
              f"- transfer-attack security drop within {100 * hyp['margin']:.0f} points: "
              f"**{hyp['transfer_drop_within_margin']}** (CI version: {hyp['transfer_noninferior_by_ci']})",
              f"- all three measured: {hyp['complete']}; hypothesis supported: **{hyp['supported']}**"]
    if rep.get("warnings"):
        lines += ["\n## Warnings\n"] + [f"- {w}" for w in rep["warnings"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="outputs/eval")
    ap.add_argument("--out", default="outputs/report")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rep = build_report(Path(a.eval_dir), cfg)
    hyp = hypothesis_check(rep["contrasts"], cfg["eval"]["noninferiority_margin"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "metrics.csv", rep["metrics"])
    _write_csv(out / "conditions.csv", rep["conditions"])
    _write_csv(out / "contrasts.csv", rep["contrasts"])
    write_json(out / "hypothesis.json", hyp)
    md = to_markdown(rep, hyp)
    (out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
