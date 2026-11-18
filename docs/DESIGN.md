# Design record

What the code does, where each choice comes from, and where it departs from the cited work.
Reference numbers follow the proposal: [4] Meta-SecAlign (arXiv:2507.02735), [5] SecPO
(arXiv:2610.41998), [6] SecFid (arXiv:2606.30783), [8] goal-goal conflicts
(arXiv:2610.35093), [9] SEP (arXiv:2403.06833), [10] IFEval (arXiv:2311.07911).

Facts about [4], [5], [6] and [8] below were read from the papers and the released code or
data on 2026-11-18. Third-party repos are pinned by commit in `scripts/fetch_third_party.sh`.

## 1. Conditions

| name | security half (2,000 pairs) | benign half (2,000 pairs) | role |
|---|---|---|---|
| C1_ignore_ordinary | ignore-style (Alpaca) | ordinary | standard recipe, baseline |
| C2_ignore_fidelity | ignore-style | fidelity | repair of [6] with out-of-benchmark data |
| C3_process_ordinary | process-style (goal-goal at the user-data boundary) | ordinary | effect of processing alone |
| C4_process_fidelity | process-style | fidelity | full proposed recipe |
| C5_target_ordinary | target control | ordinary | same prompts as C3, ignore-style target |
| C6_boundary_ordinary | boundary control | ordinary | same conflicts between system and user, as in [8] |

Primary comparison: C3 against C5. Per seed, each component is sampled once and reused by
every condition that contains it (`pno/data/build.py`), so two conditions differ only in the
factor they vary.

## 2. Prompt format

Trusted instruction in the `user` turn, untrusted text in a following `input` turn, as in
[4] and [5]. For Qwen this is `<|im_start|>input\n...<|im_end|>\n`, the same line the released
SecPO Qwen template uses, followed by the non-thinking generation prompt
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. Prompts are rendered by
`pno/formatting.py` and passed to TRL and to the generators as strings or token ids, so
training and inference see identical text. Chat delimiters are deleted from `input` content
(as the SecPO templates do). The Llama references are rendered with their own released
template; the undefended Llama-3.1-8B-Instruct uses the Meta-SecAlign template (no default
system turn), which is how [4] evaluates undefended models.

## 3. Training data

### Ignore-style security pairs (after [4], `utils.generate_preference_dataset`)

| | Meta-SecAlign code | here |
|---|---|---|
| host | every alpaca-cleaned row with a non-empty input (19,157) | sampled from a host pool of such rows |
| injected task | a random row of the original Alpaca file: instruction + " " + input | a row of alpaca-cleaned from a separate pool, same text form |
| placement | 45% before the data, 45% after, 10% completion-style with random fake delimiters | same |
| prompt | [user: instruction, input: data with injection] | same |
| preferred | base model's reply to the clean task, sent as one user message (instruction, blank line, data) | same |
| rejected | base model's reply to the injected task alone | same |
| decoding of targets | temperature 0.8, up to 8,192 tokens, no seed | greedy, up to 1,024 tokens; replies that hit the limit are dropped |

Rows play one role only here (host, injected task, benign task, carrier source, content
source are disjoint pools); the original allows overlap.

### Process-style security pairs (after [8], moved to the user-data boundary)

In [8] the system message is one catalog instruction about "the user message", the user
message is another about "the system message", and nothing else; the preferred response
applies the system instruction to the user message and the rejected one applies the user
instruction to the system message. All 99x98 ordered pairs are built and 158 are dropped
(empty or identical responses), leaving 9,544.

Here the user message is a catalog instruction about "the input", and the input holds
carrier text plus one catalog instruction about "the user message":

- preferred = the user's transform applied to the whole input, injected sentence included;
- rejected = the injected transform applied to the user message.

Carrier text (1-2 sentences, or 2-4 lines for line-based transforms, taken from Alpaca
outputs) is our addition. The target control needs it: with the injected sentence removed
there must be something left to transform. Placement uses the same 45/45/10 scheme as the
ignore-style pairs, so the two kinds of security data do not differ in attack format; in
completion-style pairs the fake answer is the user's transform applied to the carrier.

Instructions are sampled uniformly over distinct behaviors, not over instructions: several
catalog entries (copy, NFC/NFD/NFKC, strip, CRLF to LF) return plain ASCII prose unchanged
and would otherwise make up a large share of the data.

Only transforms that pass the competence filter are used (clean accuracy at least 0.8 for
the base model, and output must change when a sentence is added). The catalog of [8] is in
`pno/catalog/goalgoal.py`; the authors released no implementations, so these are ours.
With each message set to its catalog sentence they reproduce the paper's counts (156 empty,
2 identical, 9,544 kept; `tests/test_catalog.py`). `pno/catalog/extra.py` adds 14 sentence-
and line-level operations, the fallback named in the proposal.

### Target control and boundary control

Target control: the process-style prompt and rejected response, with the preferred response
replaced by the user's transform applied to the carrier alone. Boundary control: the same
transform pair, carrier and placement, with the trusted instruction in the system message
and carrier plus injected instruction in the user message (no input turn).

### Benign halves

Ordinary: an uninjected Alpaca task; preferred = the base model's reply, rejected = its reply
to a different task. Fidelity: a catalog transform over carrier text that contains one
instruction-like sentence which is only content (a one-sentence Alpaca instruction or
question, placed at the start, middle or end); preferred = the transform of the whole input,
rejected = the transform with that sentence dropped. No fidelity or process-style example
uses translation, editing, extraction or counting, the four SecFid task families.

## 4. Training

| setting | value | source |
|---|---|---|
| method | LoRA + DPO (sigmoid loss), reference = frozen base model | [8] App. C |
| beta, learning rate, epochs | 0.1, 1.6e-4, 1 | [8]; same in [4] and [5] |
| LoRA | rank 64, alpha 8, dropout 0.1 | [8] gives rank 64, dropout 0.1 and "scaling coefficient 8"; [4] and [5] state r=64, alpha=8 |
| LoRA modules | q, k, v, o, gate, up, down projections | [8] |
| schedule | linear decay, 3% warmup, gradient clipping 1.0 | [8] "warmup then decay", 3%; [5] Qwen run |
| effective batch | 64 pairs, so 63 updates for 4,000 pairs | [8]: 1 x 8 devices x 8 accumulation |
| precision | bf16, gradient checkpointing | [8] |
| thinking | off | [8] |

In Qwen3.5 only every fourth layer is full attention and has q/k/v/o projections; the
Gated DeltaNet layers between them are adapted through their MLPs only under this module
list. Sixty-three updates is fewer than the 150 of [8] (9,544 pairs) but more than nothing to
check: gate 4 tests whether the baseline changes behavior at this scale.

Software checked locally: torch 2.15.0, transformers 5.23.0, trl 1.18.1, peft 0.21.3.
TRL receives prompts as plain strings and appends `<|im_end|>` to both responses.

Merging. For evaluation the adapter is merged into a copy of the base checkpoint. The merge
is done and stored in float32, and merged runs (and the base model) are evaluated in
float32: near a weight of 0.02 the bfloat16 grid has a spacing of about 1e-4, the same size
as the LoRA update at alpha/r = 0.125 after a few dozen steps, so a bfloat16 merge would
round much of the update away. `pno.train.merge` compares the merged model's next-token
logits with base + adapter and records the difference in `pno_merge_info.json`. Training
uses one GPU; the script refuses to start if several are visible.

## 5. Evaluation

| benchmark | items | criterion | source |
|---|---|---|---|
| SecFid | 1,168 | official label rules, re-implemented in `pno/eval/secfid.py`; safe processing = processed and not executed | [6], repo `mhermon/sec-fid` |
| SecFid, clean inputs | 1,168 | output matches the probe-free reference | ours, from the SecFid data |
| SEP | 2,000 examples x 6 attacks | witness substring under any attack | [9]; protocol of [4] `test.py` |
| transfer attack | 2,000 | SEP probe inside a released SecPO injection wrapper; witness substring | [5] trainset |
| IFEval | 541 | lm-eval reference checkers; prompt-level strict accuracy | [10] |
| held-out conflicts | 900 | exact match against Python references | ours |

Notes:

- SecFid scoring uses no LLM judge: a string check for the probe's signature answer, a
  containment check for counting and extraction, EmbeddingGemma cosine similarity for
  translation and editing. `python -m pno.eval.secfid crosscheck` compares our labels with
  the benchmark's own `evaluate.py` on one model's outputs.
- The paper counts an output as Other only if it has no label at all; `evaluate.py` also
  counts executed-only outputs. We report the paper's definition and keep the other as
  `task_other`. Security, fidelity and safe processing are unaffected.
- SecFid prescribes no prompt template or token limit. We use the user/input format above,
  greedy decoding and up to 2,048 new tokens (gold outputs reach about 4.4k characters).
- SEP: Meta-SecAlign v2/v3 counted the witness alone; its current code and SecPO add an LLM
  judge on top. Witness-only ASR is therefore an upper bound on theirs. The fake answer in
  completion attacks is the base model's reply to the clean prompt, generated once and
  reused, so every model sees the same attack prompts.
- Transfer attack: SecPO released optimized injections only inside its training sets, one
  per Alpaca host task, without a success label. We keep those in which the goal appears
  verbatim once, replace the goal with a SEP probe and append the result to the SEP data.
  Wrappers equal to one of SecPO's 87 seed templates are reported separately from refined
  ones. The injections were optimized against Llama-3.1-8B checkpoints, and refined wrappers
  may refer to their original host task, so this is a transfer attack only.
- Exact match (competence filter, held-out conflicts): the reference is compared as is,
  ignoring only surrounding blank space and line-end whitespace. A code fence, an "Output:"
  prefix or surrounding quotes are forgiven on the model's side only, never removed from
  the reference.
- Sentence operations: carrier and content sentences are restricted to text that the
  sentence splitter sees as exactly one sentence (no "Dr. Lee", no "U.S."), and
  completion-style placement is not used with sentence operations, whose references are
  not meaningful on delimiter lines and numbered fake answers.
- Statistics: per-item scores, averaged over seeds, contrast across conditions, items
  resampled 10,000 times (`pno/stats/bootstrap.py`). The report lists any model scored on
  fewer items than the others, since metrics use the items common to all models.

## 6. Known issues in the benchmarks

- SecFid: on 6 items a faithful "ignored" output contains the signature answer and is
  labeled executed; on 9 items a perfect "processed" output is. 25 of 93 editing items with
  the probe in the middle insert it inside a word. These affect every model alike.
- SecFid's Hugging Face copy (`mjhermon/SecFid`) is an older release with a different
  scoring spec; we use the GitHub data.
- SecPO does not evaluate fidelity. Its training labels drop injected content by
  construction (the preferred response is the reply to the clean task).
