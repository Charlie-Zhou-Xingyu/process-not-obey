# Results needed to rebuild the report

Everything here is small, text-only and produced by our pipeline on the GPU server (seed 0).
No model outputs are included, only per-item scores; the raw outputs are too large for git.

- `scores/<setting>/<run>/<benchmark>_scores.jsonl`: one line per test item with its labels
  (SecFid: executed / processed / ignored / safe_processed; SEP and transfer: attacked;
  SecFid clean inputs: correct; IFEval: strict/loose accuracy; held-out conflicts: label).
  `<benchmark>_summary.json` holds the aggregate rates.
- `train_logs/<setting>/<run>.json`: the trainer's settings and per-step log history
  (loss, rewards/chosen, rewards/rejected, ...).
- `report_numbers.md`: every level, interval and difference used in Report V1, written by
  `python scripts/report/make_figures.py` (which also redraws the figures in
  `reports/report_v1/figures/`).

Settings:

| setting | loss | what differs |
|---|---|---|
| `main` | DPO | the six study conditions C1-C6, the untrained model and three 8B reference models |
| `sft` | DPO + SFT (weights 1:1) | C1-C5 retrained with the added SFT term |
| `mixsft` | DPO + SFT | C7 (half and half) and C9 (all Alpaca pairs + 1,000 processing pairs) |
| `mix` | DPO | C7 and C8 |
| `sec2x`, `secbig` | DPO | the standard recipe C1 with 2x security pairs and with about 11x training |
| `nocount` | DPO | C3 and C4 without the two counting operations |

Condition names: C1 standard recipe (Alpaca security pairs, dropping target), C2 C1 with
fidelity pairs instead of ordinary pairs, C3 text operations with the processing target, C4 C3
with fidelity pairs, C5 text operations with the dropping target (target control), C6 C3's
conflicts between the system and user messages, C7 1,000 Alpaca + 1,000 processing pairs,
C8 C7 with fidelity pairs, C9 2,000 Alpaca + 1,000 processing + 1,000 ordinary pairs.
