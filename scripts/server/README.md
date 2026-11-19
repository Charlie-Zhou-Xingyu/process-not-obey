# What ran on the server on 2026-11-18 / 19

Copies of the driver scripts as they were run from `/workspace` on the rented machine, in this
order, each in its own tmux session and each waiting for the previous one's marker file:

| script | tmux session | what it does | marker |
|---|---|---|---|
| `run_seed0.sh` | `seed0` | gate 1 with an output-format suffix (diagnostic), C1, backend check with the adapter, SecFid outputs of base and C1, the other five conditions, generation for every benchmark, scoring that needs no gated model | `.seed0_done` |
| `run_dl_gated.sh` | `dlgated` | downloads EmbeddingGemma and Llama-3.1-8B-Instruct once `hf auth login` was done | `.dl_gated_done` |
| `run_gate34.sh` | `gate34` | full SecFid scoring of the base model and C1 as soon as their outputs existed | `.gate34_done` |
| `run_after.sh` | `after` | `40_eval.sh --seeds 0` (SecFid scoring for every model, the undefended Llama reference, gate 2), report | `.after_done` |
| `run_extras.sh` | `extras` | base model with instruction and data in one user message; then `run_c1_variant.sh sec2x 4000 1 1.25` (the proposal's gate 4 fallback) | `.extras_done` |
| `run_c1_variant.sh secbig 7500 3 1.2756` | `big` | C1 with 7,500 security pairs and 3 epochs: does content dropping appear with about 11 times the ignore-style training? | `.big_done` |

`run_c1_variant.sh` works in a copy of the repo (`/workspace/pno-<name>`) with its own config,
data, run and outputs, so the main study is untouched. The paths are those of that machine.
