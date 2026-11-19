# What ran on the server on 2026-11-18 / 19

Copies of the driver scripts as they were run from `/workspace` on the rented machine, each in
its own tmux session and each waiting for the previous one's marker file.

## First order (started Nov 18 evening)

| script | tmux session | what it does | marker |
|---|---|---|---|
| `run_seed0.sh` | `seed0` | gate 1 with an output-format suffix (diagnostic), C1, backend check with the adapter, SecFid outputs of base and C1, the other five conditions, generation for every benchmark, scoring that needs no gated model | `.seed0_done` |
| `run_dl_gated.sh` | `dlgated` | downloads EmbeddingGemma and Llama-3.1-8B-Instruct once `hf auth login` was done | `.dl_gated_done` |
| `run_gate34.sh` | `gate34` | full SecFid scoring of the base model and C1 as soon as their outputs existed | `.gate34_done` |

`run_seed0.sh` was stopped at 05:12 UTC on Nov 19, after the six trainings and C1's SEP
outputs: in its order each model generated all six benchmarks before the next model started
(about 25 minutes per model, 15 of them for SEP's 12,000 prompts), so the headline benchmark
of the last condition would have come 2 hours later.

## Second order (`requeue.sh`, Nov 19 05:12 UTC)

| script | tmux session | what it does | marker |
|---|---|---|---|
| `run_seed0b.sh` | `seed0` | SecFid (generate + score) for C3 and C5, then the other runs; gate 2 (Meta-SecAlign-8B and undefended Llama-3.1-8B-Instruct on SecFid); the other benchmarks for the six runs and the base model; report | `.seed0_done` |
| `run_extras.sh` | `extras` | base model with instruction and data in one user message; then `run_c1_variant.sh sec2x 4000 1 1.25` (the proposal's gate 4 fallback) | `.extras_done` |
| `run_c1_variant.sh secbig 7500 3 1.2756` | `big` | C1 with 7,500 security pairs and 3 epochs: does content dropping appear with about 11 times the ignore-style training? | `.big_done` |
| `run_after.sh` | `after` | the three 8B reference models on the remaining benchmarks, anything still missing, report, gate summary; removes `.hold_stop` when it ends | `.after_done` |

In `run_seed0b.sh` and `run_after.sh` a failing step is reported (`STEP FAILED`) and the next
one still runs. `.hold_stop` tells the script on the laptop (wait, back up, verify by sha256,
`vastai stop instance`) that work is still queued.

`run_c1_variant.sh` works in a copy of the repo (`/workspace/pno-<name>`) with its own config,
data, run and outputs, so the main study is untouched. `early_score.sh` scores SecFid outputs
in a scratch copy (`/workspace/early`) without writing to the study's output directories; it
was used once to check that the scoring reproduces. The paths are those of that machine.
