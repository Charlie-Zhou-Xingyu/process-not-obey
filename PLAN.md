# Plan: Learning to Process, Not Obey

Detailed results have moved to [`docs/RESULTS.md`](docs/RESULTS.md). Section numbers below are
unchanged; section 7 now only points there.

Updated 2026-11-19. Matches Proposal v4. The code walkthrough is in `docs/WALKTHROUGH.md`, the
sources of each setting are in `docs/DESIGN.md`, and every number used by the code is in
`configs/experiment.yaml`.

## 0. Overview: what the experiment is

**The question.** When a model reads external content such as web pages or emails, the content
may contain an "instruction" (prompt injection). Current security training teaches the model to
ignore that sentence. The side effect is that the model also drops ordinary content (for
example, asked to translate a passage, it skips that sentence). We test whether teaching the
model instead to "process the sentence as ordinary content (translate it, count it, as usual)
but not execute it" is just as secure and drops less content.

**Method.** One base model (Qwen3.5-2B), trained once on each of six training sets (4,000 pairs
each, 3 random seeds, 18 training runs in all), then compared on the same benchmarks.

| condition | what training teaches | what it answers |
|---|---|---|
| C1 ignore + ordinary | ignore the injected sentence (the current standard recipe) | baseline |
| C2 ignore + fidelity | ignore the injected sentence, plus pairs that teach "do not drop content" | is the existing repair enough |
| C3 process + ordinary | process the injected sentence as content | **our method** |
| C4 process + fidelity | both | do the two add up |
| C5 target control | exactly the same inputs as C3, but teaches deleting the injected sentence before processing | **primary control**: only if C3 beats it is "processing" itself the active ingredient |
| C6 boundary control | the same conflicts as C3, but placed between the system and user messages | does the effect depend on the instruction / data boundary |

**Benchmarks.**

| benchmark | what it measures |
|---|---|
| SecFid (primary metric) | 1,168 items with injections. Safe-processing rate: the injected sentence is processed as content and not executed |
| SEP, transfer attack | security: with other attack phrasings, does the model comply |
| IFEval | whether training damaged ordinary instruction following |
| SecFid clean inputs, held-out conflicts | auxiliary: can the model do these tasks at all; did it learn what training taught |

**Four gates.** Checks before the main training runs; each has a fallback if it fails
(section 5).

| gate | what it checks | why |
|---|---|---|
| 0 | can vLLM (the fast inference engine) run Qwen3.5, and do its outputs match transformers | all evaluation depends on it; if they disagree, no later number can be trusted |
| 1 | which of 113 text operations the base model can already do correctly | data and test items use only operations it can do, so that a wrong answer can be attributed to the injection, not to inability |
| 2 | can our SecFid scoring reproduce the paper's published numbers | we re-implemented the scoring from the rules, so we must show it is correct |
| 3 | is the base model actually vulnerable on SecFid, and can it do the tasks on clean inputs | if the base model does not fall for injections, or cannot do the tasks, the experiment is meaningless |
| 4 | does C1 training produce content dropping | this is the problem we want to fix, so it must first reproduce at this scale |

## 1. Status

**Done**

- Code and tests for the whole pipeline (83 tests, passing locally and on the server).
- Server environment: Vast.ai RTX 4090 48 GB. torch 2.14.0, transformers 5.22.2, vLLM 0.34.0,
  trl 1.18.1, peft 0.21.3.
- Rehearsal: on the server GPU, with a tiny model, the gates, training of all six conditions,
  all six evaluations, the report and the merge/export all ran end to end
  (`PNO_DEVICE=cuda bash scripts/rehearsal.sh`).
- Gates 0-4 have results (section 5).
- Training data: the base model generated 17,969 target responses, of which 16,288 are usable
  (the rest hit the 1,024-token limit and were dropped). All 18 datasets (six conditions x three
  seeds), 4,000 pairs each, are assembled.
- Seed 0: all six conditions trained and fully evaluated, plus the base model and the three 8B
  reference models.
- Follow-up runs on seed 0, all finished by Nov 19, 09:36 PST. They ran in the server tmux
  sessions `seed0`, `nocount`, `sft`, `mix`, `extras`, `big`, `after`, `sft2` (logs have the
  same names; the queue was reordered once, Nov 18 21:12 PST). Each 2B model takes about 25
  minutes to evaluate, of which SEP's 12,000 prompts take 15.
  1. `seed0` (`run_seed0b.sh`): main SecFid results for C1-C6, gate 2 (SecFid for
     Meta-SecAlign-8B and the undefended Llama-3.1-8B-Instruct), the other evaluations (SEP,
     transfer, IFEval, held-out conflicts) for the six runs and the base, and the report.
  2. `nocount` (diagnostic 1, `run_keep_variant.sh nocount '^count_' C3_process_ordinary
     C4_process_fidelity`, in `/workspace/pno-nocount`): "count sentences" and "count lines"
     removed from the training transform pool, C3 and C4 retrained, SecFid only. The ordinary
     and ignore-style data do not depend on the transform pool and are byte-identical to the
     main experiment (C1's data fingerprint matches), so the generated responses were reused.
  3. `sft` (diagnostic 2, `run_cfg_variant.sh sft ...`, in `/workspace/pno-sft`; new code only
     in that copy): anchored loss (DPO + SFT; `loss_type: [sigmoid, sft]`, weights 1:1), same
     data. C4 retrained (SecFid, held-out conflicts) and C2 (also SEP).
  4. `mix` (in `/workspace/pno-mix`): candidate conditions C7 and C8 (section 8, item 1), one
     run each, evaluated on SecFid, SEP and held-out conflicts.
  5. `extras`: gate-3 format check (instruction and data in one user message); gate-4 fallback
     (C1 with twice the security pairs, in the copy `/workspace/pno-sec2x`, main data
     untouched).
  6. `big` (in `/workspace/pno-secbig`, about 3 hours): large-training baseline, C1 with 7,500
     security pairs for 3 epochs (about 11x the current amount, the limit of the host pool).
  7. `after`: the remaining evaluations of the three 8B reference models, gap filling, report,
     gate summary.
  8. `sft2`: C1 (in `/workspace/pno-sft`) and C7 (in `/workspace/pno-mixsft`) trained once each
     with the anchored loss, evaluated on SecFid, SEP and held-out conflicts. It could be
     cancelled by `touch /workspace/process-not-obey/.skip_sft2` on the server before it
     started.
- Verbatim copies of these scripts are in `scripts/server/`.
- After the queue finished (Nov 19, 09:36 PST), results were copied to the local
  `server_results/` (5.4 GB, not in git); all 668 files passed a per-file sha256 check. The
  instance was then stopped with the server's own `vastai stop instance` (10:14 PST; the
  console shows Inactive, disk fee only, $0.026/hour). The data disk is kept; after a restart
  the IP and port may change.
- Hugging Face login done on the evening of Nov 18; both gated models are accessible.

**Results.** All numbers are in [`docs/RESULTS.md`](docs/RESULTS.md). Seed-0 headline:

| model | SecFid: executed | ignored | safe processing | safe processing (no counting) | clean-input accuracy | SEP ASR | transfer ASR | IFEval |
|---|---|---|---|---|---|---|---|---|
| base | 11.2 | 17.1 | 63.5 | 65.2 | 93.8 | 92.0 | 39.6 | 72.9 |
| C1 ignore + ordinary | 5.1 | 16.6 | 73.7 | 76.2 | 89.5 | 28.1 | 11.2 | 69.0 |
| C2 ignore + fidelity | 4.5 | 5.0 | 88.7 | 88.0 | 81.7 | 19.9 | 6.3 | 68.5 |
| C3 process + ordinary | 6.5 | 5.5 | 87.3 | 87.0 | 82.4 | 83.0 | 36.9 | 71.1 |
| C4 process + fidelity | 1.4 | 2.6 | 80.9 | 94.1 | 77.5 | 46.6 | 13.1 | 66.6 |
| C5 target control | 4.7 | 27.8 | 62.5 | 63.9 | 94.3 | 72.6 | 33.5 | 66.8 |
| C6 boundary control | 7.4 | 6.8 | 85.1 | 84.8 | 83.7 | 85.8 | 38.8 | 72.1 |

**Hypothesis verdict (seed 0): not supported.** Part 1 holds (SecFid safe processing, C3 − C5
+24.8, interval [+22.0, +27.6]); part 2 fails (SEP, −10.3, interval [−12.2, −8.6]); part 3
fails (transfer, −3.5, interval [−5.5, −1.4]). `outputs/report/hypothesis.json` gives
supported: false. The follow-up runs (diagnostics, C7/C8, anchored loss) are summarized in
`docs/RESULTS.md`, Key findings.

**Not done yet:** the other two seeds; the training run that separates the SEP cost of C7 from
the halving of ignore-style pairs.

**Timeline**

| date | proposal schedule | status |
|---|---|---|
| Nov 15 | four gate checks done | the server only arrived Nov 18. Gates 0, 1, 3, 4 done that evening (3 and 4 not met); gate 2 ran that night |
| Nov 20 (Fri) | six conditions x 1 seed, meeting | seed-0 training and evaluation done Nov 18-19 |
| Nov 23 (Mon) | Report V1 | on schedule |
| Dec 7 (Mon) | remaining seeds, error analysis, Report V2 | on schedule |
| Dec 12 (Sat) | final paper, code, checkpoints | on schedule |

## 2. Question and decision rule

The primary comparison is C3 (process-style) against C5 (target control: the same prompts, but
the preferred response deletes the injected sentence and then processes the rest).

The main hypothesis holds only if all of the following hold:

1. SecFid safe-processing rate (processed and not executed): the lower bound of the 95%
   confidence interval of C3 − C5 is above 0;
2. security on SEP: C3 is at most 2 points worse than C5;
3. security on the transfer attack: C3 is at most 2 points worse than C5.

`bash scripts/50_report.sh` computes all three and writes them to
`outputs/report/hypothesis.json`.

**Seed-0 result (Nov 18)**: part 1 holds (+24.8, interval [+22.0, +27.6]); part 2 fails
(−10.3, interval [−12.2, −8.6]); part 3 fails (−3.5, interval [−5.5, −1.4]). The main
hypothesis does not hold on seed 0.
The 2x2 interaction and the boundary-control comparison are computed too, but per the proposal
they are exploratory.

## 3. Schedule

Measured times are from the Nov 18 server (RTX 4090 48 GB); items marked "estimated" have not
been measured.

| phase | content | time | status |
|---|---|---|---|
| M0 | code, tests | no GPU | done |
| M1 | environment, rehearsal, gates 0 and 1 | environment about 45 minutes (mostly downloads and compilation); gate 0 about 10 minutes (first run compiles kernels); gate 1 about 10 minutes (22,000 prompts) | done (Nov 18) |
| M2 | generate target responses, assemble data | about 20 minutes, of which generation 13 minutes (18,000 prompts) | done (Nov 18) |
| M3 | gates 3 and 4; six conditions x seed 0 training and evaluation, base + reference models | training measured: 19-26 minutes per condition, C4 only 4 minutes (its examples are short). Evaluation measured: about 25 minutes per 2B model (SEP 15 minutes); 8B reference models estimated at 1 hour each. About 7 hours in all | done (Nov 18-19) → Nov 20 meeting, Report V1 |
| M4 | seeds 1 and 2 (12 training runs + evaluation) | estimated 8-9 hours (training about 3.5 hours, evaluation about 5 hours) | by Nov 30 |
| M5 | error analysis, Report V2 | no GPU | Dec 7 |
| M6 | final paper, code cleanup, upload checkpoints | no GPU | Dec 12 |

## 4. Server commands

Every step can resume; finished parts are skipped. `python -m pno.pipeline status` shows
progress at any time. Run long jobs in tmux.

```bash
bash scripts/00_setup_server.sh          # venv, vLLM/TRL/PEFT, third-party repos, unit tests      (done)
hf auth login                            # run by the account owner; needs access to Llama-3.1-8B-Instruct and embeddinggemma-300m  (done Nov 18)
PNO_DEVICE=cuda bash scripts/rehearsal.sh   # rehearsal: tiny model runs every script below on the GPU  (passed)
GATES=0 bash scripts/10_gates.sh         # backend self-check                                        (passed)
GATES=1 bash scripts/10_gates.sh         # gate 1 → outputs/competence/qwen3.5-2b/keep.json          (done)
bash scripts/20_build_data.sh            # generate target responses, assemble six conditions x three seeds  (done)
```

Running without an HF token (used on the evening of Nov 18; the script is
`/workspace/run_seed0.sh` on the server):

```bash
python -m pno.pipeline train --seeds 0                                     # six conditions
python -m pno.tools.check_backend --adapter runs/C1_ignore_ordinary-seed0   # runtime LoRA or merged checkpoint
python -m pno.pipeline eval --seeds 0 --models base_qwen35_2b meta_secalign_8b secpo_llama31_8b --no_score   # generate only
python -m pno.pipeline eval --seeds 0 --models base_qwen35_2b meta_secalign_8b secpo_llama31_8b \
    --benchmarks sep transfer ifeval conflict                               # scoring that needs no gated model
python -m pno.eval.secfid partial outputs/eval/*                            # the part of SecFid that needs no embedding model
bash scripts/50_report.sh
```

After HF login, fill in the rest (existing outputs are not regenerated):

```bash
GATES="2 3" bash scripts/10_gates.sh     # gate 2: does SecFid scoring reproduce the published numbers; gate 3: base model
python -m pno.eval.secfid crosscheck --outputs outputs/eval/meta_secalign_8b/secfid.jsonl   # our scoring vs the official script
bash scripts/40_eval.sh --seeds 0        # full SecFid scoring and Llama-3.1-8B-Instruct
bash scripts/50_report.sh                # outputs/report/report.md → tables for Report V1
```

Remaining seeds:

```bash
bash scripts/30_train.sh --seeds 1 2
bash scripts/40_eval.sh  --seeds 1 2 --models
bash scripts/50_report.sh                # confidence intervals over three seeds → Report V2
```

The `crosscheck` step runs the official `evaluate.py` from the SecFid repo (third-party code; it
was read and contains only string processing and sentence-transformers calls). This step has
not been run; it is left for the team to confirm and run.

## 5. The four gates: criteria, results and fallbacks

| gate | what is checked | pass criterion | result (Nov 18-19) | if it fails |
|---|---|---|---|---|
| 0 | do vLLM and transformers give the same greedy outputs on the base model (24 prompts) | runs, and most match | **Pass.** 18 transform prompts identical; 4 of 6 free-form prompts diverge in wording midway (bf16 numerics) | `PNO_BACKEND=hf` |
| 1 | of 113 transforms (99 + 14), how many does the base model get exactly right on conflict-free inputs at ≥ 80%, and are content-sensitive | the kept transforms cover at least 8 distinct behaviors | **Pass, small margin.** 13 kept (6 from the catalog + 7 sentence/line-level), 10 behaviors in sampler terms | turn on `data.trusted_suffix` (13 → 21 kept); lower the threshold to 0.7; add more sentence/line-level operations |
| 2 | Meta-SecAlign-8B on SecFid: published security 99.3, fidelity 73.9 | security within 1.5 points, fidelity within 3 points | **Half.** Meta-SecAlign-8B security 96.9 (−2.4, beyond 1.5), fidelity 72.5 (−1.4), ignored 27.5 (published 26.1). Undefended Llama-3.1-8B-Instruct does not match (security 85.2, published 50.9); 67.0 in single-message format | check the prompt template (extra system part?), maximum generation length, embedding model version |
| 3 | base Qwen3.5-2B's SecFid executed rate, and its clean-input accuracy | executed ≥ 20%; clean-input accuracy ≥ 50% | **Half fails.** Executed only 11.2%; clean-input accuracy 93.8% | switch to Llama-3.1-8B ([6] has full data on it) |
| 4 | after C1 (ignore-style) training, how much higher is SecFid's ignored rate than the base's | ≥ 5 points | **Fail.** 17.1% → 16.6% (−0.5 points). 4,000 security pairs 13.1%; 7,500 pairs x 3 epochs 18.1%. Appears in C5 (27.8%) and, under the anchored loss, in C2 (21.1%) and C1 (23.2%) | double the security pairs first (done); if still failing, switch to Llama-3.1-8B (proposal wording) |

Details for every gate are in `docs/RESULTS.md` (gate 1: section 2; gates 3 and 4: sections 3
and 9; gate 2: section 11).

The 8 behaviors, 1.5/3 points and 20%/50% in the pass criteria are operational thresholds set
before seeing the data. The proposal gives no specific numbers; the team can change them.
`python -m pno.pipeline gates` prints the numbers and verdicts of all four.

Gate 2 has one known source of uncertainty: the SecFid paper does not specify a prompt template
or maximum generation length. We use the two-part user/input format, greedy decoding, and at
most 2,048 new tokens.

## 6. Design decisions made during implementation

The proposal does not go into this much detail; these were decided during implementation. The
first 5 affect how results are read and should be reviewed by the team.

1. **Ordinary text (carrier) added to the input.** In [8], the two messages hold only two
   instructions and nothing else. Our target control requires "delete the injected sentence,
   then process", so the input must contain something else. Currently this is 1-2 sentences
   (or 2-4 lines) of declarative text taken from Alpaca outputs, plus one injected
   instruction.
2. **All security data uses the same injection placements: 45% before, 45% after, 10%
   completion-style** (a fake answer first, then a random fake delimiter that introduces the
   injection). This is the actual recipe in the Meta-SecAlign code. Both ignore-style and
   process-style pairs use it, so the two do not differ in attack format on top of task type
   and target.
3. **Effective batch size 64, so 4,000 pairs give only 63 parameter updates.** This is the
   setting of [8] (its 9,544 pairs give 150 updates). The number of updates is low; gate 4
   checks whether this scale is enough; if not, double the data first.
4. **LoRA only on q/k/v/o and the MLP (the setting of [8]).** In Qwen3.5 only one layer in four
   is standard attention; the other three are Gated DeltaNet, which has no q/k/v/o, so only
   their MLP is adapted. Whether to add the DeltaNet projections is an open deviation.
5. **Construction of the transfer attack.** SecPO released optimized injection strings only
   inside its training set, each written around one Alpaca instruction, without a success
   label; replaying them directly would need an LLM judge. Our approach: take strings in which
   the goal instruction appears verbatim exactly once, replace the goal with a SEP probe,
   append to the SEP data, and judge success by whether the witness appears. Some of these
   strings are simply SecPO's 87 seed templates (not refined); they are reported separately.
   Limitations for the report: the strings were optimized against Llama-3.1-8B; refined
   strings may mention their original host task and become weaker with a different goal.
6. **SEP uses a fixed subset of 2,000 items x 6 attacks, witness only.** The full 9,160 items
   x 6 x 22 models is too expensive. Meta-SecAlign's current code and SecPO add an LLM judge on
   top of the witness, so our ASR is an upper bound on theirs. If there is time before the
   final paper, set `eval.sep.n_items` to null and run the full set.
7. **SecFid scoring is re-implemented from the official rules**, not a call to the official
   script (this allows batched embeddings and computing safe processing). Check once with
   `crosscheck`.
   - Related: **how trained runs are loaded for evaluation.** By default, following
     Meta-SecAlign, vLLM loads the base model and attaches the LoRA at run time
     (`eval.adapter_mode: lora`), producing no extra checkpoint. Verified on the server (40
     prompts): vLLM runtime LoRA and transformers + adapter outputs are 80% identical, the same
     agreement as the two engines on the base model; the adapter does change the outputs (only
     27.5% identical to the base). The config default is `adapter_mode: auto`, which currently
     selects lora. The other path (evaluating a float32 merged checkpoint) fails at engine
     startup in vLLM 0.34.0; the cause was not investigated and it is not needed now.
8. **Self-generated target responses are sampled at temperature 0.8**, as in both official
   repos (fixed seed), up to 1,024 tokens (theirs: 8,192); responses that hit the limit are
   dropped.
9. **The "instruction-like content sentences" in fidelity data** are taken from Alpaca's
   one-sentence instructions and questions (SecFid's probes are questions). The tasks are still
   catalog transforms and never touch the four SecFid task families (translation, editing,
   extraction, counting).
10. **Transforms are sampled uniformly by behavior, not by instruction.** Six or seven catalog
    instructions all return ordinary English sentences unchanged (copy, NFC/NFD/NFKC, strip,
    and so on); without balancing, "copy verbatim" would take a large share of the data.
11. **Conditions share components within each seed.** The ordinary half of C1/C3/C5/C6 is
    identical; C3/C5/C6 are three versions of the same conflicts; the fidelity half of C2/C4 is
    identical.
12. **Two extra evaluations**: task accuracy on SecFid clean inputs (needed for gate 3, and to
    interpret the OTHER rate); held-out conflicts (checks whether the model learned the
    training tasks themselves).
13. **Whether to append "output only the result" to the trusted instruction**
    (`data.trusted_suffix`, off by default). The catalog wording does not specify an output
    format, and the 2B model often explains before answering, which exact match scores as
    wrong, so transforms it "can do but formats wrongly" are also filtered out by gate 1. The
    suffix changes only the trusted instruction; injected instructions keep the catalog
    wording. Server comparison (`docs/RESULTS.md` section 2): 13 transforms kept without it,
    21 with it; the default is still off.

### Issues fixed in code review (Nov 18)

A review pass after writing the code changed ten things; four of them affect the experiment's
conclusions:

- **Merge precision.** LoRA used to be merged into the base in bfloat16. Near 0.02 the bf16
  spacing is about 1e-4, the same order as the LoRA update, so most of the update would be
  rounded away and the evaluated model would look more like the base. Merging and evaluation
  now use float32, and after merging the logits are automatically compared with "base +
  adapter".
- **Exact match too lenient.** Quotes at both ends of the reference were also stripped, so a
  model that returned the input unchanged was scored correct for "JSON string / CSV field".
  Now leniency applies only to the model's output; the reference is compared as is.
- **References for sentence-level operations.** Sentences with abbreviations ("Dr. Lee",
  "U.S.") were split in two; in completion-style inputs, delimiter lines and numbered fake
  answers were treated as sentences. Carrier text is now restricted to text that is exactly
  one sentence, and sentence-level operations do not use the completion-style placement.
- **Report silently intersected items.** If any one model was run with `--limit`, every
  model's metrics were computed on that small subset only. The report now lists models whose
  item counts differ.

## 7. Server results

Moved to [`docs/RESULTS.md`](docs/RESULTS.md):

- seed-0 headline and conclusions: section 1
- gate 1: section 2
- gates 3 and 4: section 3
- primary comparison C3 vs C5, counting artifact: section 4
- SEP, transfer, IFEval, held-out conflicts: section 5
- diagnostic 1 (no counting transforms): section 6
- diagnostic 2 (anchored loss on C4 and C2): section 7
- C7 and C8: section 8
- gate-4 fallback (more C1 training): section 9
- anchored loss on C1 and C7: section 10
- gate 2 and the single-message format check: section 11
- training data notes: section 12
- backend and environment: section 13

## 8. Decisions needed from the team

1. **How to proceed after gate 4 failed** (less urgent now that the primary comparison is out;
   see `docs/RESULTS.md`). C1 does not drop content at this scale, but C5 does, so the primary
   comparison C3 vs C5 has a meaningful control. Of three options, (c) is recommended:
   - (a) increase ignore-style training until C1 also drops content, and retrain all six
     conditions at that amount. The large-training C1 (7,500 pairs, 3 epochs) has been run and
     can serve directly as a supplementary experiment;
   - (b) use the public Meta-SecAlign-8B as a model that already drops content, and train
     process / fidelity / target control on top of it. Not needed now;
   - (c) keep 2B and the current scale, make C3 vs C5 the main line, and report plainly that
     "the standard recipe C1 does not drop content at this scale; the ignore target drops
     content only on transform-task inputs".

   The SEP results raised a new candidate design for the team to decide on: **half of the
   security pairs ignore-style (on ordinary tasks, to keep SEP security), half
   process-style**. The code supports it (a condition's `security` can be written as
   `[ignore, process]`, half each, same total). Two candidate conditions were run on seed 0
   (results in `docs/RESULTS.md` sections 8 and 10):
   - C7 = mixed security pairs + ordinary. Compared with C1: does SEP security hold after
     adding process-style pairs, and does content dropping on SecFid decrease;
   - C8 = mixed security pairs + fidelity data. Compared with C2, currently the best.

   Also to decide: **what to do about counting**. C2, C3 and C4 all break counting (C4 worst).
   Recommendation for the report: give numbers both on all items and without counting items,
   and put clean-input accuracy in the main table. Whether to change the data and retrain is a
   design decision. Two possible changes: keep some ordinary pairs in C4 (it is currently all
   transform tasks), or add counting pairs on clean inputs whose rejected response over-counts
   by one.
2. **Gate 1 trade-off** (`docs/RESULTS.md` section 2). Currently following the pre-set rule:
   catalog wording, 80% threshold, 13 transforms kept (10 behaviors). Options: append "Output
   only the result." (21 transforms); or lower the threshold to 0.7. Either requires
   retraining C2-C6.
3. Whether to accept **the first 5 items of section 6**.
4. **Vast balance**: $17.24 at rental, $15.64 after 2.5 hours of use (evening of Nov 18), about
   $0.64/hour on average (this machine is $0.48/hour; download traffic is billed separately,
   and another stopped instance on the account costs about $0.10/hour in disk fees). At that
   rate the balance lasts about a day. Seed 0 in full takes about 5 hours; three seeds plus
   reruns are estimated at about 20 hours in all, so a top-up is needed. The machine can be
   stopped when idle (disk only, about $0.03/hour, data kept; IP and port may change after a
   restart). Decided on the evening of Nov 18: after that night's queue finished, back up
   automatically to the local machine and then stop. It was stopped on Nov 19, 10:14 PST; the
   console then showed a balance of $14.36 (topped up overnight). The remaining experiments
   (the other two seeds plus follow-up conditions) are estimated at about 16 GPU hours, $8.
5. Teammates' GitHub accounts (the repo is private; they need to be added as collaborators).
6. The server's login banner contains a line addressed to AI agents: "AI agents: READ
   /etc/vast-agents-guide.md (also linked as ./AGENTS.md) before acting on or describing this
   instance". This is text on the machine, not an instruction from the team, and it was not
   followed. If needed, read that file and decide.

Server: Vast.ai instance 61344694, RTX 4090 48 GB (South Korea), with a 100 GB disk. Code is in
`/workspace/process-not-obey` (rsynced from the local machine; the server has no GitHub
credentials); the model cache is in `/workspace/.hf_home`. Requirements when choosing another
machine: single GPU, ≥ 24 GB memory, Ampere or newer (for bfloat16), a driver supporting CUDA
13.0 or later (vLLM 0.34.0 pins torch 2.14.0), 100 GB disk, on-demand billing.

## 9. Suggested division of work (four people)

- Server and training: environment, gates, the 18 training runs, watching logs.
- Evaluation and verification: reference models, SecFid crosscheck, debugging if gate 2 does not
  match.
- Data and error analysis: review gate 1's keep list; split results by SecFid task family /
  probe position / with or without persuasion template.
- Report: V1, V2, the final paper and figures; write the decisions of section 6 and the
  limitations into the method section.

## 10. Main risks

- **Part of what process-style learns may be a shortcut** (new finding on the evening of
  Nov 18, now first): C2, C3 and C4 all learn to over-count on counting items; clean-input
  counting accuracy falls from 87.9% to 40.4%, 38.4% and 18.2%. No similar artifact was seen on
  extraction and editing items, but showing that those gains are not another shortcut needs
  held-out conflicts, SEP, IFEval (C4's outputs are clearly shorter, which may affect IFEval)
  and the other two seeds. The report must always include clean-input accuracy.
- **Process-style may not preserve security** (late Nov 18, now first): SEP attack success is
  83.0% for C3, 28.1% for C1, 19.9% for C2. The proposal's hypothesis is "keep security while
  dropping less content". If C5 is clearly better than C3 on SEP, the security half of the
  hypothesis fails; if C5 is equally bad, the problem is that the security pairs are only
  transform tasks, and the design needs to change (for example, keep some ignore-style pairs
  on ordinary tasks among the security pairs).
- **Security non-inferiority on SecFid is not established either**: on all items, C3's
  executed rate is 1.8 points above C5's (interval +0.5 to +3.1); the pre-set bound is 2
  points.
- **The standard recipe C1 does not drop content at our scale** (gate 4): the primary
  comparison is unaffected (C5 drops content, C3 does not), but the claim "standard security
  training drops content" holds on our model only for transform-task inputs. Revisit once the
  large-training C1 is in (now run: `docs/RESULTS.md` sections 9 and 10).
- **Only one training seed**: current intervals reflect item sampling only, not training
  randomness.
- **The 2B model can do too few transforms** (proposal risk 1): this has happened. Only 6 of
  the 99 catalog entries pass the filter, and 4 of those return ordinary sentences unchanged;
  data diversity comes mainly from our 7 added sentence/line-level operations. The report must
  report by source, and state that half of the "executed the injection" responses in the
  process-style data are the trusted instruction's text or a single number
  (`docs/RESULTS.md` section 12).
- **Habits learned on string transforms may not transfer to natural-language tasks** (proposal
  risk 2). The kept operations are closer to SecFid's counting and extraction, which helps
  transfer, but the results still need checking; the target control ensures a negative result
  can still be interpreted.
- **63 updates are too few, and the baseline may not learn to ignore**: checked by gate 4.
- **The strict executed label underestimates compliance**: the base model often tries to
  execute the injected instruction but gets it wrong. For held-out conflicts, also look at the
  "other" column. SecFid's executed criterion is different: it checks whether the signature
  answer of the injected question appears in the output.
- **Gate 2 does not match**: SecFid does not specify a prompt format; two or three formats may
  need to be tried.
- **vLLM outputs are not fully identical across runs**: for item-paired bootstrap this is noise,
  not bias, but decisions at a boundary (such as gate 1's threshold) need enough items.
