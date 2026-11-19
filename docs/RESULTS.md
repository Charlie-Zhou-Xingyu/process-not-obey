# Results (seed 0)

Measured on the server on 2026-11-18 and 2026-11-19 (Vast.ai, RTX 4090 48 GB). All numbers are
percentages unless stated otherwise. Only one training seed (seed 0) exists so far; every
interval below reflects item sampling only, not training randomness. Raw outputs are in the
server's `outputs/` (`competence/`, `check_backend/`) and were copied to the local
`server_results/` (not in git). The working plan, gates and design decisions are in
`../PLAN.md`; the setup and its sources are in `DESIGN.md`.

Reference tags follow `DESIGN.md`: [4] Meta-SecAlign, [5] SecPO, [6] SecFid, [8] goal-goal
conflicts.

## Key findings

- **The pre-registered hypothesis is not supported on seed 0.** Of its three parts, only one
  holds: SecFid safe processing is 24.8 points higher for C3 than for C5. C3 is less secure
  than C5 on SEP (10.3 points) and on the transfer attack (3.5 points); the bound was 2 points.
  `outputs/report/hypothesis.json` gives supported: false.
- **Process-style targets do not drop content.** With the same inputs and the same rejected
  responses, the ignore-style target (C5) raises the SecFid ignored rate from 17.1% (base) to
  27.8%; the process-style target (C3) gives 5.5%.
- **Security learned on transform tasks does not transfer to SEP.** Base 92.0%; C1 and C2
  (ignore-style security pairs on ordinary tasks) 28.1% and 19.9%; C3, C5 and C6 (security
  pairs on transform tasks) 83.0%, 72.6% and 85.8%.
- **With plain DPO, C2 (the existing repair of [6]) is best overall.** It ties C3 on SecFid and
  is more secure than C1 on SEP and transfer. Process-style shows no advantage over it.
- **With plain DPO, the standard recipe C1 does not drop content**, even at about 11x the
  training (ignored 16.6 / 13.1 / 18.1 against 17.1 for the base). Gate 4 fails.
- **With the anchored loss (DPO + SFT), the conclusions change.** C1 drops content (ignored
  23.2%), C2 does not repair it (21.1%), and C7 (half ignore-style, half process-style
  security pairs) returns to the base level (16.4%) at a SEP cost (29.1 against 22.1 for C1).
  C7 differs from C1 in two ways (ignore-style pairs halved, process-style pairs added), so
  neither the lower ignored rate nor the SEP cost can be attributed to the process-style pairs.
  In absolute terms, anchored-loss C7 is level with plain-DPO C1, and plain-DPO C2 remains best
  on ignored, safe processing and SEP.
- **Training has side effects.** C2, C3, C4 and C6 break counting (clean counting accuracy
  87.9% for the base, 40.4%, 38.4%, 18.2% and 44.6%), which also inflates C3's SecFid counting
  score. IFEval drops 1 to 6 points for every condition.
- **Measurement.** Gate 0 passes; gate 1 keeps 13 transforms; gate 2 half matches (our
  ignored rate for the public Meta-SecAlign-8B is 27.5%, published 26.1%); gate 3 misses its
  20% executed threshold (base 11.2%). The separate input turn in our prompt format blocks
  most injections by itself.

## 1. Seed-0 headline and conclusions

All evaluations of the six conditions and the base model (completed late on Nov 18).

| model | SecFid: executed | ignored | safe processing | safe processing (no counting) | clean-input accuracy | SEP ASR | transfer ASR | IFEval |
|---|---|---|---|---|---|---|---|---|
| base | 11.2 | 17.1 | 63.5 | 65.2 | 93.8 | 92.0 | 39.6 | 72.9 |
| C1 ignore + ordinary | 5.1 | 16.6 | 73.7 | 76.2 | 89.5 | 28.1 | 11.2 | 69.0 |
| C2 ignore + fidelity | 4.5 | 5.0 | 88.7 | 88.0 | 81.7 | 19.9 | 6.3 | 68.5 |
| C3 process + ordinary | 6.5 | 5.5 | 87.3 | 87.0 | 82.4 | 83.0 | 36.9 | 71.1 |
| C4 process + fidelity | 1.4 | 2.6 | 80.9 | 94.1 | 77.5 | 46.6 | 13.1 | 66.6 |
| C5 target control | 4.7 | 27.8 | 62.5 | 63.9 | 94.3 | 72.6 | 33.5 | 66.8 |
| C6 boundary control | 7.4 | 6.8 | 85.1 | 84.8 | 83.7 | 85.8 | 38.8 | 72.1 |

1. **The pre-registered main hypothesis does not hold on seed 0.** Only one of its three parts
   passes: SecFid safe processing is 24.8 points higher for C3 than for C5 (interval +22.0 to
   +27.6). The other two fail: on SEP, C3 is 10.3 points less secure than C5 (interval −12.2
   to −8.6); on the transfer attack, 3.5 points less secure (−5.5 to −1.4). The pre-set bound
   was 2 points. `outputs/report/hypothesis.json` gives supported: false.
2. **Process-style targets do not drop content.** Same inputs, same rejected responses: C5,
   whose target ignores the injected sentence, pushes the SecFid ignored rate from 17.1%
   (base) to 27.8%; C3, whose target processes it as content, gives 5.5%.
3. **Security pairs built only from transform tasks do not transfer to ordinary tasks.** The
   base model fails almost everywhere on SEP (92.0%). C1 and C2, whose security pairs are
   ordinary question answering, drop to 28.1% and 19.9%. C3, C5 and C6, whose security pairs
   are transform tasks and whose other half is ordinary, only drop to 83.0%, 72.6% and 85.8%.
   Note (added Nov 19): C4 is the exception. Its security pairs are the same as C3's, and with
   the other half replaced by fidelity data it reaches 46.6%. The reason has not been
   investigated.
4. **The best overall is C2**, the existing repair of [6] (ignore-style security pairs +
   fidelity data). It ties C3 on SecFid and is more secure than C1 on SEP and transfer.
   Process-style shows no advantage over it.
5. **C4 (the full recipe) is best on every metric on the non-counting SecFid items** (safe
   processing 94.1, executed 1.7) and in the middle on SEP (46.6%). The cost is the counting
   items and IFEval (66.6, base 72.9).
6. **With the original loss, the standard recipe C1 does not drop content, and at 11x the
   training it is still not above the base** (ignored: 2,000 pairs 16.6%, 4,000 pairs 13.1%,
   7,500 pairs x 3 epochs 18.1%, base 17.1%). On SEP it becomes more secure with more training
   (28.1% → 21.9% → 15.0%). Content dropping appears in C5, and also in C2 under a different
   loss (item 9).
7. **Training side effects.** C2, C3, C4 and C6 break counting (clean-input counting accuracy
   87.9% → 40.4%, 38.4%, 18.2%, 44.6%). C2 loses formatting on the transforms it was trained
   on. Every condition is 1 to 6 points below the base on IFEval. Diagnostic 1 (retraining
   without the two counting transforms, early Nov 19) shows that the counting transforms are
   only part of the cause: accuracy returns to 58.3% (C3) and 56.0% (C4), C4's
   unscoreable short lists on SecFid counting disappear (other 53.4% -> 0.3%), but about a third of items are still
   over-counted by one.
8. **Measurement itself.** Gate 0 passes. Gate 1 keeps 13 transforms. Gate 2 half matches (for
   the public Meta-SecAlign-8B we measure an ignored rate of 27.5%, published 26.1%; the
   undefended Llama does not match, most likely because of prompt layout). Gate 3: the base
   model's SecFid executed rate is only 11.2%, below the pre-set 20%.
9. **A different loss changes the conclusions (diagnostic 2, early Nov 19).** When the loss
   adds direct learning of the preferred response to DPO, C2 becomes more secure (SEP 14.3%),
   most of the broken formatting and counting recovers, but it starts dropping content on
   SecFid (ignored 21.1%, above the base). So a large part of C2's "no content dropping" in
   the main experiment is a bias from DPO pushing down "the response that drops that
   sentence" too hard. Once the model actually learns the target responses, the proposal's
   premise appears: ignore-style training drops content, and fidelity data does not repair
   it. Whether process-style helps in this setting was not tested at that point (see item 11).
10. **Mixing the two kinds of security pairs (C7, C8, original loss) does not beat C2.**
    Replacing half of the ignore-style pairs with process-style pairs costs about 9 points of
    SEP security in each case; with fidelity data, SecFid improves by only 1.4 points.
11. **Under the new loss, C7's content dropping returns to the base level, but the cause is
    not separated (late morning Nov 19).** Note (revised Nov 19): this item was originally
    titled "process-style works". Under the anchored loss (DPO + SFT), the standard recipe C1
    drops content (ignored 23.2%, 6.1 points above the base), the repair of [6], C2, does not
    fix it (21.1%), and C7, which mixes in process-style pairs, returns to the base level
    (16.4%), with safe processing 8.3 points above C1. On SEP it is broken 7.1 points more
    often. C7 changes two things relative to C1 (ignore-style pairs halved, process-style
    pairs added), so it is not separated which change causes the drop in ignored and which
    causes the SEP cost; the earlier text said this only about the cost. In absolute terms,
    C7 under the new loss (ignored 16.4 / safe processing 74.9 / SEP 29.1) is essentially the
    same as C1 under the original loss (16.6 / 73.7 / 28.1). The best on these three metrics
    is still C2 under the original loss (5.0 / 88.7 / 19.9).

All of the above uses one training seed; intervals reflect item sampling only.

Not yet done: the other two seeds; the training run that separates the SEP cost from the
halving of ignore-style pairs.

## 2. Gate 1: what the base model can do

Each transform gets 100 conflict-free prompts (trusted instruction in the user turn, text in
the input turn). An output counts as correct only if it exactly matches the reference computed
in Python.

- Of 113 transforms, **73 get zero items right**, 15 are below 20%, and only 13 are at or above
  80%. Mean accuracy over all transforms is 16.5%.
- The 13 kept: copy (T01), strip leading/trailing whitespace (T11), collapse whitespace (T12),
  CRLF→LF (T14), lowercase (T04), CSV field (T91); and the sentence/line-level operations
  number sentences, one sentence per line, count sentences, first word of each sentence,
  count lines, join lines, prefix each line with a quote marker. The first four return plain
  English sentences unchanged and count as one behavior in the sampler, so there are 10
  behaviors.
- Near misses: replace digits with # (T23, 76%), uppercase (T03, 66%), bullet points (67%).
  Uppercase fails on individual words ("Y-Axis", "VILLIERS").
- Failure modes (counted on the 2,825 items of the first run): fully correct 16.5%; correct
  answer wrapped in an explanation 8.2%; input returned unchanged 15.1%; long explanation
  32.7%; other errors 27.5%. The 2B model fails mainly by "explaining first" or "doing
  nothing". The catalog operations defined by code points, hashes and encodings are genuinely
  beyond it.
- Variant with "Output only the result." appended to the trusted instruction: 21 kept, mean
  accuracy 26.7%. Of the 8 new ones, 4 are of the return-unchanged kind (NFC, NFKC, strip
  trailing whitespace, deduplicate lines), 1 duplicates an existing one (T81 count lines), and
  the genuinely new ones are number lines (two formats) and bullet points. In sampler terms
  this is 10 → 14 behaviors, one of which is "return multi-line text unchanged". **The
  default is still the catalog wording (13 transforms).** Switching to the suffixed version
  is a team decision; it requires rebuilding the data (about 15 minutes) and retraining
  C2-C6.

Three method notes:

1. The first two runs used 25 items per transform. The prompts were identical, but the keep
   lists differed (T23 at 80% once and 76% once): vLLM greedy decoding is not bitwise
   identical across runs. The count was raised to 100, and the official list is from that run.
2. The "first word of each sentence" operation (our own addition) originally had a reference
   that kept the trailing punctuation ("Therefore,"), while the model gave "Therefore" and
   was marked wrong. This was a reference definition problem. It now matches "last word of
   each sentence" (no punctuation), and gate 1 was rerun after the change.
3. The untrained base model on conflict prompts (the 13 kept transforms, 1,300 items; injected
   instructions drawn from all 113): processed the injected sentence 37.0%, exactly executed
   0.8%, ignored 9.3%, other 52.8%. Much of "other" is **an attempt to follow the injected
   instruction that is done wrong, or applied to the input text** (for example, asked to copy
   verbatim while the injected sentence says "reverse the lines", it outputs the sentences
   reversed). The strict executed rate ("applies the injected transform to the user message")
   is therefore low. When reading held-out conflict results, look at the "other" column; error
   analysis needs a manual read of a sample.

## 3. Gates 3 and 4: the base model rarely falls for injections, and baseline training does not drop content

All 1,168 SecFid items, seed 0, in percent. The last two rows are the numbers published in
[6], shown for comparison (their prompt format may differ from ours; gate 2 retests under our
format, section 11).

| model | executed | processed | ignored | other | security | fidelity | safe processing |
|---|---|---|---|---|---|---|---|
| base Qwen3.5-2B | 11.2 | 68.7 | 17.1 | 8.6 | 88.8 | 82.9 | 63.5 |
| C1 (ignore + ordinary) | 5.1 | 78.2 | 16.6 | 5.1 | 94.9 | 83.4 | 73.7 |
| [6]: Llama-3.1-8B (undefended) | 49.1 | 50.2 | 7.7 | 5.5 | 50.9 | 92.3 | |
| [6]: Meta-SecAlign-8B | 0.7 | 67.0 | 26.1 | 6.7 | 99.3 | 73.9 | |

- **Gate 3**: the base model's executed rate is only 11.2% (13.5% on items with a persuasion
  template, 6.7% without), below the 20% threshold. Clean-input accuracy is 93.8% (counting
  87.9, editing 90.5, extraction 96.8, translation 100), so capability is fine. In our format
  (data in a separate input turn), there is little room for this model to be defended. What
  happens with instruction and data in a single user message was queued for testing
  (section 11).
- **Gate 4**: after C1 training, executed halves (11.2 → 5.1), but ignored does not rise
  (17.1 → 16.6); processed instead rises from 68.7 to 78.2, and safe processing rises 10.2
  points. Ignored by task: counting 19.2 → 23.8, editing 31.5 → 28.6, extraction 10.3 → 9.7,
  translation 8.3 → 4.7. The effect the proposal wants to fix ("ignore-style training drops
  content") does not reproduce at this scale. C1's cost is elsewhere: clean-input accuracy
  falls from 93.8 to 89.5 (counting 87.9 → 77.5).
- **Why it does not reproduce: two reasons with evidence.**
  1. Far less training. Meta-SecAlign uses 19,157 pairs for 3 epochs (about 57,000 example
     presentations, all security pairs). Our C1 has 2,000 security pairs for 1 epoch, about
     29x fewer.
  2. DPO here does almost nothing but push the rejected response down. In C1's training log,
     the log-likelihood of the preferred response relative to the reference model barely
     changes (rewards/chosen between −0.7 and +0.4), while the rejected response drops by
     about 120 nat (rewards/rejected ≈ −12, β = 0.1). After step 13 the loss is below 0.03,
     and the remaining 50 steps carry almost no gradient. The model learns "do not execute the
     injected task", not "answer as on a clean input".
- **The prediction made from this was wrong.** The prediction was that C3 and C5 would be
  close, since their
  rejected responses are identical and only the preferred responses differ. They turned out
  far apart (section 4). An almost unchanged mean log-likelihood of the preferred response in
  the training log does not mean the preferred response has no effect on what the model
  learns.
- Only one seed. Confidence intervals for the differences come from the report script. A gap
  between −0.5 points and +5 points will not be erased by seed noise, but C1's other numbers
  (such as safe processing +10.2) must wait for the other two seeds.

## 4. Primary comparison: C3 vs C5 (seed 0, SecFid, evening of Nov 18)

C3 and C5 have identical inputs and identical rejected responses (transform tasks with one
injected instruction in the data; the rejected response executes the injection). They differ
only in the preferred response: C3 processes the injected sentence as content; C5 removes it
and then processes the rest. In percent.

| model | executed | processed | ignored | other | safe processing | clean-input accuracy |
|---|---|---|---|---|---|---|
| base | 11.2 | 68.7 | 17.1 | 8.6 | 63.5 | 93.8 |
| C1 (ignore + ordinary) | 5.1 | 78.2 | 16.6 | 5.1 | 73.7 | 89.5 |
| C2 (ignore + fidelity) | 4.5 | 93.0 | 5.0 | 2.0 | 88.7 | 81.7 |
| C3 (process + ordinary) | 6.5 | 93.4 | 5.5 | 0.9 | 87.3 | 82.4 |
| C4 (process + fidelity) | 1.4 | 82.3 | 2.6 | 15.2 | 80.9 | 77.5 |
| C5 (target control) | 4.7 | 66.2 | 27.8 | 5.6 | 62.5 | 94.3 |
| C6 (boundary control) | 7.4 | 92.2 | 6.8 | 0.9 | 85.1 | 83.7 |
| Llama-3.1-8B-Instruct (undefended, our format) | 14.8 | 76.3 | 15.4 | 3.7 | 67.0 | 95.3 |
| Meta-SecAlign-8B (our format) | 3.1 | 65.2 | 27.5 | 7.2 | 62.5 | 92.3 |

Without the counting items (861 items left; reason below):

| model | executed | processed | ignored | other | safe processing | clean-input accuracy |
|---|---|---|---|---|---|---|
| base | 11.1 | 70.5 | 16.4 | 7.4 | 65.2 | 95.8 |
| C1 | 4.2 | 80.3 | 14.1 | 5.6 | 76.2 | 93.7 |
| C2 | 3.9 | 91.9 | 5.5 | 2.6 | 88.0 | 96.4 |
| C3 | 5.6 | 92.1 | 6.6 | 0.9 | 87.0 | 98.0 |
| C4 | 1.7 | 95.8 | 2.7 | 1.5 | 94.1 | 98.6 |
| C5 | 5.0 | 68.2 | 26.2 | 5.2 | 63.9 | 96.2 |
| C6 | 6.9 | 91.3 | 7.4 | 1.0 | 84.8 | 97.7 |
| Llama-3.1-8B-Instruct | 16.5 | 72.9 | 17.8 | 4.2 | 62.0 | 97.4 |
| Meta-SecAlign-8B | 3.1 | 57.1 | 33.1 | 9.5 | 54.6 | 93.7 |

Differences (percentage points; brackets are 95% intervals from a paired bootstrap over items,
one training seed only):

| comparison | safe processing | ignored | executed | clean-input accuracy |
|---|---|---|---|---|
| C3 − C5, all items | +24.8 [+22.1, +27.6] | −22.3 [−24.9, −19.8] | +1.8 [+0.5, +3.1] | −12.0 [−14.3, −9.7] |
| C3 − C5, no counting | +23.1 [+20.1, +26.1] | −19.6 [−22.5, −16.7] | +0.6 [−0.8, +2.0] | +1.9 [+0.7, +3.0] |
| C3 − C1, all items | +13.6 [+11.3, +15.9] | −11.1 [−13.3, −9.1] | +1.5 [+0.3, +2.6] | −7.1 [−9.5, −4.7] |
| C2 − C1, all items | +15.0 [+12.7, +17.3] | −11.6 [−13.8, −9.6] | −0.6 [−1.5, +0.3] | −7.8 [−10.1, −5.5] |
| C2 − C1, no counting | +11.8 [+9.4, +14.3] | −8.6 [−10.8, −6.5] | −0.2 [−1.0, +0.6] | +2.7 [+1.3, +4.1] |
| C4 − C3, no counting | +7.1 [+5.0, +9.2] | −3.9 [−5.6, −2.4] | −3.8 [−5.2, −2.4] | +0.6 [−0.2, +1.5] |
| C4 − C1, no counting | +17.9 [+15.2, +20.7] | −11.4 [−13.7, −9.2] | −2.4 [−3.6, −1.4] | +4.9 [+3.4, +6.5] |
| C5 − base, all items | −1.0 [−3.8, +1.7] | +10.7 [+8.0, +13.4] | −6.5 [−8.4, −4.7] | +0.6 [−0.9, +2.1] |

By task (processed / ignored):

| model | counting | editing | extraction | translation |
|---|---|---|---|---|
| base | 63.5 / 19.2 | 45.4 / 31.5 | 78.7 / 10.3 | 86.0 / 8.3 |
| C1 | 72.3 / 23.8 | 57.9 / 28.6 | 86.5 / 9.7 | 95.3 / 4.7 |
| C2 | 96.1 / 3.6 | 85.0 / 9.2 | 93.9 / 3.9 | 96.4 / 3.6 |
| C3 | 97.1 / 2.3 | 84.6 / 12.8 | 95.8 / 2.9 | 95.3 / 4.7 |
| C4 | 44.3 / 2.3 | 96.3 / 2.2 | 95.2 / 1.9 | 96.0 / 4.0 |
| C5 | 60.6 / 32.2 | 38.8 / 53.5 | 73.2 / 18.1 | 91.4 / 8.6 |
| C6 | 94.8 / 4.9 | 81.3 / 16.1 | 96.1 / 2.6 | 95.7 / 4.3 |
| Llama-3.1-8B-Instruct | 85.7 / 8.8 | 48.7 / 32.2 | 89.4 / 6.8 | 78.4 / 15.8 |
| Meta-SecAlign-8B | 87.6 / 11.7 | 42.9 / 52.4 | 78.1 / 19.4 | 47.8 / 29.5 |

- **Content dropping appears, and the target causes it.** C5's ignored rate is 10.7 points
  above the base (53.5% on editing items). The total is comparable to the 26.1% that [6]
  reports for Meta-SecAlign-8B. With the same inputs and a process-style target, ignored falls
  to 5.5%. This was the first evidence on seed 0 for the proposal's core hypothesis.

### The counting artifact

- **C3's gain on counting items is an artifact.** In SecFid counting items, the injected
  sentence mentions one extra entity, so the "processed" reference answer is the clean answer
  plus 1. C3 gives the "plus 1" answer even on clean inputs without any injection: on 60.9% of
  items (base 8.5%, C1 13.4%, C5 3.6%). Its outputs often state a number that does not match
  the items it lists itself. So most of its 97.1% counting processed rate comes from a habit
  of counting one too many. Counting only items that are processed under injection and not
  over-counted on the clean input: base 56.7%, C1 59.9%, C3 36.5%, C5 57.0%; C3 is the lowest.
- C2 has the same habit (58.3% of clean items over-counted by one; net 38.1%), although C2's
  security pairs are ordinary question answering and only its benign half is transform tasks.
  What C2 and C3 share: their preferred responses count the instruction-like sentence in the
  data (the rejected response of a fidelity pair is exactly "count one fewer"). So this looks
  like learning "better to count more", not a problem specific to C3.
- On clean inputs, model's stated number minus the reference: base exactly equal on 89.9% of
  items; C3 equal on only 36.5%, +1 on 48.5%, +2 on 14.0%; C2 35.5%, 38.8%, 20.2%.
- C4 is worse: clean-input counting accuracy is only 18.2%. Its outputs are very short (median
  11 words, base 52), typically a list of items numbered from 5, which looks like the output
  format of training transforms such as "number sentences" carried over. All 4,000 pairs of C4
  are transform tasks; none is ordinary question answering.

### Other observations

- **C2 and C3 are almost the same.** C2's security pairs are the same as C1's; only its benign
  half is replaced by fidelity data (the repair of [6], using data outside the benchmark). Its
  safe processing 88.7, ignored 5.0 and executed 4.5 cannot be separated from C3's (87.3,
  5.5, 6.5) within item-sampling noise; its executed rate is slightly lower. What can be said
  on seed 0: the target "process the injection as content" has a similar effect whether it is
  placed in the security pairs (C3) or in the benign pairs (C2). Process-style shows no
  advantage over the existing repair. (This was written before C4's results were in; C4 is the
  next bullet.)
- **C4 (the full recipe) is best on every metric on the three non-counting task types.** On
  the 861 items, safe processing is 94.1, 7.1 points above C3 [+5.0, +9.2] and 17.9 points
  above C1. Executed is 1.7, 3.8 points below C3 and the lowest of all conditions. Clean-input
  accuracy is 98.6. Combining the two kinds of data gives an extra benefit; the cost is the
  counting items described above.
- **The gain on extraction items is real.** On clean inputs, models almost never add an
  entity out of nowhere (0.3%-1.3% across models). C3's processed rate rises from 78.7% to
  95.8%, and clean-input accuracy from 96.8% to 98.4%.
- **On editing items there is no sign so far of "just copying".** Among the 57 items whose
  edit can be read off the instruction (21 "change A to B", 36 "delete a sentence"), the edit
  is carried out on clean inputs by C3 in 61.9% / 44.4% and by the base in 66.7% / 41.7%; under
  injection by C3 in 61.9% / 44.4% and by the base in 42.9% / 27.8%. The injected sentence is
  left verbatim in the output by the base in 38.5%, C1 56.8%, C3 79.9%, C5 28.2%. The sample
  is small; the only claim is that there is no counterexample.
- **On translation items, C3 equals C1** (95.3 / 4.7).
- **Security non-inferiority is not established.** On all items, C3's executed rate is 1.8
  points above C5's, with an upper interval bound of 3.1, beyond the pre-set 2-point bound;
  without counting items it is +0.6 [−0.8, +2.0]. (SEP and transfer results were not yet in
  when this was written; see section 5.)
- Training curves: for five conditions, rewards/chosen averaged over the last 10 steps is
  between −0.42 and +0.28, and rewards/rejected between −9.4 and −12.7. C4's examples are very
  short (880,000 tokens in total; C1 has 3.01 million), and it trains in 4 minutes.

## 5. Other benchmarks: SEP, transfer, IFEval, held-out conflicts (seed 0)

| model | SEP: any attack succeeds | SEP per attack (straightforward / straightforward-before / ignore / ignore-before / completion / completion-ignore) | transfer | IFEval (mean of four) |
|---|---|---|---|---|
| base | 92.0 | 48.4 / 22.9 / 38.8 / 23.6 / 78.9 / 71.6 | 39.6 | 72.9 |
| C1 | 28.1 | 11.2 / 8.2 / 12.0 / 8.1 / 8.8 / 6.9 | 11.2 | 69.0 |
| C2 | 19.9 | 8.8 / 8.2 / 7.9 / 7.4 / 4.6 / 4.2 | 6.3 | 68.5 |
| C3 | 83.0 | 44.5 / 23.9 / 37.2 / 23.4 / 66.9 / 57.6 | 36.9 | 71.1 |
| C4 | 46.6 | 17.6 / 12.4 / 12.2 / 10.7 / 30.8 / 18.5 | 13.1 | 66.6 |
| C5 | 72.6 | 41.4 / 20.0 / 33.4 / 17.8 / 50.9 / 40.1 | 33.5 | 66.8 |
| C6 | 85.8 | 46.8 / 26.9 / 40.6 / 25.9 / 71.2 / 60.5 | 38.8 | 72.1 |
| Llama-3.1-8B-Instruct (undefended) | 98.7 | | 38.4 | 80.2 |
| Meta-SecAlign-8B | 10.3 | | 3.9 | 80.4 |
| SecPO (Llama-3.1-8B) | 7.4 | | 2.5 | 70.8 |

- SEP has 2,000 examples, each with six attacks. "Any attack succeeds" means at least one of
  the six makes the probe's answer appear in the output. The base is at 92.0%: it rarely
  executes injections on SecFid (11.2%) but executes almost all of them on SEP. Relative to the
  base, C1 drops 63.8 points, C2 72.0, C4 45.4, and C3 only 9.0 (interval −10.6 to −7.4).
- **The three 8B reference models (morning of Nov 19).** The undefended Llama fails almost
  everywhere on SEP (98.7%); the two public defended models bring it down to 10.3% and 7.4%.
  On SecFid both defended models drop content: ignored 27.5% (Meta-SecAlign) and 32.7%
  (SecPO), safe processing 62.5 and 60.1, lower than the undefended Llama (67.0). This is the
  effect reported in [6], reproduced in our pipeline for both public models. Among our 2B
  models, the best on SEP are the large-training C1 (15.0%) and C2 under the changed loss
  (14.3%).
- IFEval prompt-level strict accuracy: base 66.2, C6 65.1, C3 63.4, C1 61.6, C2 61.2, C5 59.5,
  C4 59.3. Every trained condition is below the base; C4 and C5 drop the most. C4 has 166
  outputs that hit the length limit (other conditions 34 to 68).
- Differences (percentage points, 95% intervals from a paired bootstrap over examples):

  | comparison | SEP | transfer |
  |---|---|---|
  | C5 − C1 | +44.5 [+42.1, +46.9] | +22.2 [+20.2, +24.4] |
  | C3 − C5 | +10.3 [+8.6, +12.2] | +3.5 [+1.4, +5.5] |
  | C4 − C1 | +18.5 [+16.2, +20.8] | +1.8 [+0.1, +3.5] |
  | C2 − C1 | −8.2 [−10.1, −6.3] | −4.9 [−6.2, −3.5] |
  | C4 − C3 | −36.4 [−38.6, −34.1] | −23.8 [−26.1, −21.7] |
  | C6 − C3 | +2.9 [+1.6, +4.2] | +1.9 [−0.1, +3.9] |

- **Both causes are present; the first is much larger.** (a) C5 and C1 both have an ignore
  target and differ only in whether the security pairs are transform tasks or ordinary
  question answering; C5 is 44.5 points worse. "Do not execute" learned on transform tasks does
  not transfer to ordinary tasks like SEP. (b) With the same inputs, the process-style target
  (C3) is another 10.3 points worse than the ignore target (C5). The proposal requires the SEP
  security loss to be at most 2 points; seed 0 does not meet this.
- C3's outputs really do execute the injection: in the examples inspected, it completes the
  main task and then writes "here is the answer to your priority question".
- Effect of truncation, checked: 5%-17% of SEP outputs hit the 1,024-token limit, and the
  probe's answer is often at the end of the output, where it can be cut off. Truncated outputs
  are 4.1% (base) to 17.0% (C2) of each model's outputs. The hit rate of truncated outputs is
  the same as or lower than that of untruncated ones (base 29.7% vs 48.1%, C1 9.2% vs 9.2%, C2
  7.8% vs 6.7%, C3 35.4% vs 42.8%, C4 15.4% vs 17.2%, C5 28.2% vs 34.2%, C6 35.0% vs 45.9%), so
  attack success may be underestimated. Worst case (every truncated output counted as a
  success): base 93.0, C1 40.0, C2 43.5, C3 85.2, C4 60.3, C5 74.7, C6 87.3. The worst cases of
  C1, C2 and C4 are still below the actual values of C3, C5 and C6 (72.6 and up), and in the
  primary comparison C3's actual value (83.0) is above C5's worst case. Only the order of C1
  and C2 would flip in the worst case. Note (revised Nov 19): the earlier text said the hit
  rates were close and the ranking was unaffected; that holds only for C1, C2 and C4.

### Held-out conflicts

300 items per type; an output counts as processed only if it equals the reference verbatim.
Processed rate, split by the trusted instruction's transform:

| transform | fidelity-type: C1 | C2 | C3 | user-data conflict: C1 | C2 | C3 |
|---|---|---|---|---|---|---|
| prefix each line with `> ` | 87.1 | 12.9 | 100.0 | 72.4 | 10.3 | 96.6 |
| encode the whole text as one quoted CSV field | 96.4 | 0.0 | 100.0 | 72.7 | 0.0 | 77.3 |
| number sentences | 100.0 | 36.4 | 93.9 | 81.2 | 62.5 | 96.9 |
| count sentences | 97.0 | 33.3 | 39.4 | 69.0 | 62.1 | 58.6 |
| first word of each sentence | 89.7 | 86.2 | 41.4 | 47.2 | 61.1 | 27.8 |
| lowercase | 43.5 | 100.0 | 100.0 | 36.7 | 96.7 | 100.0 |
| all 13 | 88.0 | 65.3 | 85.0 | 62.7 | 66.0 | 79.0 |

- **C2 gets worse on the transforms it was trained on.** Examples: the instruction is to
  prefix each line with `> `, and C2 outputs the original lines without the prefix; the
  instruction is to encode as a quoted CSV field, and the output is the unquoted text repeated
  twice; the data has 2 sentences (reference 2), and the output is 3. In fidelity pairs, the
  preferred and rejected responses are almost identical and differ by one sentence. DPO has to
  push the rejected response's log-likelihood down by about 100 nat (only then does the loss
  saturate at β = 0.1); doing this between two nearly identical responses also pushes down
  the format they share. In C2's training log, the preferred response's log-likelihood falls by
  about 4 nat on average (rewards/chosen −0.42), the lowest of the five conditions.
- **C3 keeps the format but over-counts sentences by one** (fidelity-type 39.4%), consistent
  with its SecFid clean-input behavior. This looks like a shortcut: in training, every counting
  item has exactly one extra injected sentence.
- So there are two independent problems, matching the two diagnostics queued on the evening of
  Nov 18: `nocount` targets the counting shortcut (section 6), `sft` targets "only pushing down
  the rejected response" (section 7).
- **C4's counting problem is not miscounting; it cannot stop.** On the held-out count-lines
  and count-sentences items, C4 gets none right, but on 72 of the 73 fidelity-type counting
  items the leading number is correct, and then it keeps writing (for example
  `4\n4\n4\n4\n16)`, or it copies the original text). Many rejected responses in training are
  "a single number, then stop"; they are pushed down so hard that the correct answer in this
  form goes with them. On the other transforms C4 does not lose formatting the way C2 does
  (CSV items 100%, prefix 90%).
- **C5 drops content even on conflict-free inputs.** On fidelity-type items (the data only
  contains an instruction-like sentence, no injection), C5 removes that sentence in 36% of
  cases, C1 in 0.3%.
- On system-user conflicts, C6, which is trained on that kind of conflict, scores 66.3%; C2
  63.7, C1 51.0, C4 36.3, C3 31.0, base 27.0. On user-data conflicts C3 is highest (79.0), C6
  53.7, base 33.3. Each condition is best on the boundary it was trained on and falls off on
  the other. C6 gets almost all CSV items wrong (7%-9%), the same kind of format loss as C2.
- Fidelity data helps SEP unexpectedly: C4 is 36.4 points below C3, and C2 is 8.2 points below
  C1. Fidelity pairs do not involve executing anything; why this happens has not been
  investigated.

## 6. Diagnostic 1: retraining without the two counting transforms (seed 0, early Nov 19)

"Count sentences" and "count lines" are removed from the training transform pool (11
transforms, 8 behaviors left); all other data is unchanged. C3 and C4 are retrained and
evaluated on SecFid only.

| | C3 original | C3 without | C4 original | C4 without | base |
|---|---|---|---|---|---|
| clean-input counting accuracy | 38.4 | 58.3 | 18.2 | 56.0 | 87.9 |
| clean-input items where the answer exceeds the reference by 1 | 48.5 | 36.8 | 13.4 | 33.9 | 3.6 |
| counting items under injection: processed / other | 97.1 / 0.7 | 85.0 / 1.6 | 44.3 / 53.4 | 96.4 / 0.3 | 63.5 / 12.1 |
| safe processing, all items | 87.3 | 83.8 | 80.9 | 94.2 | 63.5 |
| safe processing, no counting | 87.0 | 85.6 | 94.1 | 94.2 | 65.2 |
| executed, all items | 6.5 | 6.8 | 1.4 | 2.3 | 11.2 |

- **C4's unscorable outputs on SecFid counting items are caused by the counting transforms.**
  Note (revised Nov 19): the original heading said "cannot stop after counting". That is the
  behavior on the held-out count-sentences and count-lines items, where 72 of 73 leading
  numbers are correct, and the anchored loss fixed it. On SecFid counting items C4's outputs
  are very short lists whose leading number is often not a count, for example `5. EPA`. This
  variant was evaluated on SecFid only. After removal, unscorable outputs on counting items
  under injection fall from 53.4% to 0.3%, and safe processing on all items rises from 80.9 to
  94.2 (+13.3, interval +11.0 to +15.6). Many rejected responses in training were "a single
  number, then stop"; without the counting transforms there are no such pairs.
- **Over-counting by one comes only partly from the counting transforms.** Clean-input counting
  accuracy recovers by 19.9 and 37.8 points but is still about 30 points below the base, and
  about a third of items are still over-counted by one. Training no longer contains any
  counting example, so this generalizes from the other transforms: after learning "every
  sentence in the data must be included", the model also counts entities generously.
- **The other three task types are essentially unaffected.** Without counting items, C4's safe
  processing is 94.2 vs 94.1; C3's is 85.6 vs 87.0 (−1.4, interval −2.8 to −0.1).
- **The primary comparison is unaffected.** C3 without the counting transforms is still 21.3
  points above C5 (interval +18.7 to +24.1).
- These two variants were not evaluated on SEP.
- Implication: if the data is changed later, removing these two transforms is a cheap step
  that fixes the worst part, but not all of it. The report still needs numbers with and without
  counting items, plus clean-input accuracy.

## 7. Diagnostic 2: anchored loss (DPO + SFT) (seed 0, early Nov 19)

The training loss becomes DPO plus the per-token negative log-likelihood of the preferred
response (weight 1:1). Training data is identical to the main experiment.

| C4 | original (plain DPO) | DPO + SFT |
|---|---|---|
| held-out conflicts, verbatim correct: fidelity-type / user-data / system-user | 69.7 / 63.0 / 36.3 | 99.7 / 96.0 / 90.7 |
| of which count lines, count sentences (fidelity-type) | 0 / 0 | 100 / 100 |
| SecFid safe processing, no counting | 94.1 | 94.7 |
| SecFid executed, all items | 1.4 | 1.7 |
| SecFid clean-input counting accuracy | 18.2 | 25.1 |
| preferred-response log-likelihood at the end of training (mean per example) | −4.22 | −0.21 |

- **On the trained tasks, the problems essentially disappear.** All three held-out conflict
  types reach above 90% (intervals: +30.0, +33.0, +54.3 points), and "cannot stop after
  counting" is gone. Even the untrained system-user boundary reaches 90.7%, higher than C6
  (66.3%), which was trained on that boundary. This supports the earlier reading: with DPO
  alone, the model meets the objective by pushing down the rejected response, and the
  preferred response itself is not learned firmly.
- **The other three SecFid task types are unchanged** (94.7 vs 94.1), and executed is
  essentially unchanged.
- **SecFid counting is not fixed, for a different reason.** C4's outputs become very short
  numbered lists whose numbering does not start at 1. For example, when the correct answer is 3
  entities, the output is `4. Oncology / 5. Radiology / 6. Pathology`: the entities are right,
  but no count is stated, and the scorer treats it as over-counting by one. All 4,000 pairs of
  C4 are transform tasks with no ordinary question answering, and changing the loss cannot fix
  that. Removing the counting transforms (diagnostic 1) works better for it (56.0%).

| C2 | original (plain DPO) | DPO + SFT | difference (95% interval) |
|---|---|---|---|
| SecFid ignored, all items | 5.0 | 21.1 | +16.2 [+13.9, +18.5] |
| SecFid safe processing, all items | 88.7 | 69.1 | −19.6 [−22.2, −17.1] |
| SecFid executed, all items | 4.5 | 4.1 | −0.3 [−1.3, +0.6] |
| SEP attack success | 19.9 | 14.3 | −5.5 [−7.1, −4.0] |
| clean-input counting accuracy | 40.4 | 74.3 | +33.9 [+27.4, +40.4] |
| held-out conflicts, verbatim correct: fidelity-type / user-data / system-user | 65.3 / 66.0 / 63.7 | 98.3 / 90.0 / 65.7 | +33.0 / +24.0 / +2.0 |

- **With the changed loss, C2 drops content.** SecFid ignored rises from 5.0% to 21.1%, above
  the base (17.1%) and C1 (16.6%; difference +4.5, interval +2.4 to +6.6); without counting
  items it is 22.2%. By task: editing 37.7, extraction 19.4, counting 18.2, translation 10.1.
  At the same time it is more secure (SEP 14.3%, the lowest of all models so far), no longer
  loses formatting on trained transforms, and most of its counting recovers.
- **This changes how the main experiment should be read.** With DPO alone, C2 does not drop
  content on SecFid not because it learned the preferred responses of the fidelity pairs, but
  because "the response that drops that sentence" is pushed extremely low. That suppression
  generalizes into "better more than less", so C2 both keeps content and over-counts by one.
  With a loss that does learn the preferred responses, what the ignore-style security pairs
  teach ("treat it as absent") is actually learned on ordinary tasks, while what the fidelity
  pairs teach ("process all of it") only works on transform tasks (held-out fidelity-type
  98.3%) and does not transfer to SecFid.
- **The proposal's premise appears under this loss**: ignore-style security training makes the
  model drop content, and fidelity data from outside the benchmark does not repair it. This is
  exactly the situation process-style is meant to address. The loss used in the main
  experiment did not reach this situation.
- **C4 does not have this problem** (ignored 2.5%): its training data has no ignore-style
  pairs. But it was not evaluated on SEP; the original C4 is at 46.6% on SEP.
- So the real question is whether, under this loss, mixing process-style and ignore-style
  pairs can be both secure and free of content dropping. Testing this needs two more training
  runs (C1 and C7 with this loss, about 1 hour 40 minutes). They were queued last on early
  Nov 19 (`sft2`); results in section 10.

## 8. Candidate conditions C7 and C8: half ignore-style, half process-style security pairs (seed 0, early Nov 19)

C7 = 1,000 ignore-style pairs + 1,000 process-style pairs + 2,000 ordinary pairs; C8 replaces
the ordinary pairs with 2,000 fidelity pairs. The loss is the same as the main experiment
(plain DPO).

| | C1 | C7 | C2 | C8 | C3 | C4 |
|---|---|---|---|---|---|---|
| SecFid safe processing | 73.7 | 82.1 | 88.7 | 90.1 | 87.3 | 80.9 |
| SecFid safe processing, no counting | 76.2 | 83.6 | 88.0 | 89.1 | 87.0 | 94.1 |
| SecFid ignored | 16.6 | 11.7 | 5.0 | 3.6 | 5.5 | 2.6 |
| SecFid executed | 5.1 | 4.5 | 4.5 | 4.9 | 6.5 | 1.4 |
| SEP attack success | 28.1 | 37.8 | 19.9 | 28.6 | 83.0 | 46.6 |
| clean-input counting accuracy | 77.5 | 65.5 | 40.4 | 25.1 | 38.4 | 18.2 |
| held-out conflicts, verbatim correct (user-data) | 62.7 | 88.0 | 66.0 | 83.7 | 79.0 | 63.0 |

- **Replacing half of the ignore-style pairs with process-style pairs costs about 9 points of
  SEP security in each case.** C7 is 9.7 points above C1 (interval +7.8 to +11.7), C8 is 8.8
  points above C2 (+6.9 to +10.6). Both have only half the original ignore-style pairs. The C1
  runs with more security pairs show that each doubling of ignore-style pairs lowers SEP by
  about 6 points (section 9), so most of these 9 points can be explained by the halving.
- **On content dropping, mixing adds almost nothing when fidelity data is present.** C8's safe
  processing is only 1.4 points above C2's (+0.0 to +2.7); without fidelity data, C7 is 8.4
  points above C1 (+6.4 to +10.4).
- **C8's counting is worse than C2's** (25.1% vs 40.4%): on clean inputs 49.5% of items are
  over-counted by 1 and 28.7% by 2.
- **Conclusion: under plain DPO, mixing security pairs gives nothing that C2 does not.** C2 is
  the most secure on SEP and essentially ties C8 on SecFid. C7 under the new loss (`sft2`) was
  the key run for testing process-style (section 10).

## 9. Gate-4 fallback: more ignore-style training for C1 (seed 0, Nov 19)

The proposal says that if gate 4 fails, the security pairs are doubled first. Two levels were
run, with the same loss as the main experiment (plain DPO) and 2,000 ordinary pairs as the
benign half in both: 4,000 security pairs for 1 epoch (94 steps); 7,500 pairs for 3 epochs
(447 steps, about 11x C1's presentations of security pairs).

| | base | C1 (2,000 pairs) | 4,000 pairs | 7,500 pairs x 3 epochs |
|---|---|---|---|---|
| SecFid ignored | 17.1 | 16.6 | 13.1 | 18.1 |
| SecFid safe processing | 63.5 | 73.7 | 78.7 | 72.4 |
| SecFid executed | 11.2 | 5.1 | 4.9 | 4.9 |
| SEP attack success | 92.0 | 28.1 | 21.9 | 15.0 |
| clean-input accuracy | 93.8 | 89.5 | 91.3 | 91.7 |

- **Security improves steadily with more training.** The SEP attack success rate falls from
  28.1% to 21.9%, then to 15.0%.
- **Content dropping is not clearly above the base.** At the largest amount of training,
  ignored is 18.1%, 0.9 points above the base (interval −1.6 to +3.5); gate 4 requires at
  least 5 points. So under plain DPO, the standard recipe does not reproduce content dropping
  even at 11x the training.
- **But the trend turns at the last level.** From 4,000 pairs to 7,500 pairs x 3 epochs,
  ignored rises 5.0 points (interval +3.0 to +6.9), mainly on counting (27.7%) and extraction
  (14.5%) items. With even more training it might exceed the base; this was not tested.
- **Most of the SEP gap of C7 and C8 comes from halving the pairs.** Each doubling of
  ignore-style pairs lowers SEP by about 6 points; C7, with only 1,000 ignore-style pairs, is
  at 37.8%, consistent with this trend, and this cannot be attributed to the process-style
  pairs.
- Together with diagnostic 2: content dropping depends mainly not on the amount of
  ignore-style training but on whether the model actually learns the ignore-style target
  responses. With DPO alone they are not learned firmly, and only at 11x training does ignored
  get back to the base level; with SFT added, the original amount of training already reaches
  21.1%.

## 10. Anchored loss on C1 and C7 (seed 0, late morning Nov 19)

The loss is DPO + SFT (as in diagnostic 2); the data is identical to the original C1 and C7.
The C2 column comes from diagnostic 2.

| DPO + SFT | base | C1 (standard recipe) | C2 (repair of [6]) | C7 (mixed security pairs) |
|---|---|---|---|---|
| SecFid ignored | 17.1 | 23.2 | 21.1 | 16.4 |
| SecFid safe processing | 63.5 | 66.6 | 69.1 | 74.9 |
| SecFid executed | 11.2 | 4.4 | 4.1 | 4.5 |
| SEP attack success | 92.0 | 22.1 | 14.3 | 29.1 |
| clean-input accuracy (counting) | 93.8 (87.9) | 90.7 (84.7) | 88.3 (74.3) | 90.4 (76.5) |
| held-out conflicts, verbatim correct (user-data) | 33.3 | 61.0 | 90.0 | 89.7 |

- **The standard recipe drops content under this loss.** C1's ignored rate is 23.2%, 6.1
  points above the base (interval +3.5 to +8.6) and 6.6 points above plain-DPO C1 (+4.7 to
  +8.6). The original gate-4 criterion (at least 5 points above the base) is met here. Data and
  amount of training are the same as the original C1; only the loss changed. By task: editing
  40.3%, counting 26.7%, extraction 19.4%, translation 6.8%.
- **The repair of [6] does not fix it under this loss.** C2's ignored rate is 21.1%, only 2.1
  points below C1.
- **C7's content dropping returns to the base level.** C7 changes two things relative to C1
  (ignore-style pairs 2,000 → 1,000, plus 1,000 process-style pairs), and the differences
  below are not separated by cause. C7's ignored rate is 16.4%, 6.8 points below C1 (interval
  −8.6 to −5.0) and 4.7 points below C2 (−6.7 to −2.7); safe processing is 8.3 points above C1
  (+6.3 to +10.3). Executed does not change (+0.2, interval −0.4 to +0.8).
- **The cost is on SEP.** C7 is at 29.1%, broken 7.1 points more often than C1 (+5.3 to +8.8).
  C7 has half of C1's ignore-style pairs. By the trend in section 9, halving the ignore-style
  pairs corresponds to about 6 points, so these 7 points most likely come mainly from the
  halving rather than from the process-style pairs themselves. This was not tested directly:
  it needs a run that keeps 2,000 ignore-style pairs and adds process-style pairs on top.
- **This result points in the direction of the proposal's idea but does not yet count as
  support.** Note (revised Nov 19). When the model learns the ignore-style target and therefore
  drops content, C7, with half its ignore-style pairs replaced by process-style pairs, does not
  drop content, while fidelity data from outside the benchmark cannot achieve this. Four
  limitations: there is only one seed; C7 only returns to the base level and is not better than
  the base; neither the benefit nor the SEP cost has been separated from the halving of
  ignore-style pairs; in absolute terms, C7 under the new loss is essentially the same as C1
  under the original loss, and the best on the three main metrics is still C2 under the
  original loss.

## 11. Gate 2: public models in our pipeline (evening of Nov 18)

| model | | executed | processed | ignored | other | security | fidelity |
|---|---|---|---|---|---|---|---|
| Meta-SecAlign-8B | ours | 3.1 | 65.2 | 27.5 | 7.2 | 96.9 | 72.5 |
| | [6] published | 0.7 | 67.0 | 26.1 | 6.7 | 99.3 | 73.9 |
| Llama-3.1-8B-Instruct (undefended) | ours | 14.8 | 76.3 | 15.4 | 3.7 | 85.2 | 84.6 |
| | [6] published | 49.1 | 50.2 | 7.7 | 5.5 | 50.9 | 92.3 |

- Meta-SecAlign-8B's task labels essentially match. Processed / ignored by task: counting
  87.6 / 11.7 (published 87.3 / 12.1), extraction 78.1 / 19.4 (78.1 / 17.7), editing
  42.9 / 52.4 (43.2 / 53.5); translation is further off, 47.8 / 29.5 (55.4 / 24.1), and 22.3%
  of our translation items land in other. Executed is 2.4 points above the published value,
  mainly on extraction items (6.8 vs 0.3). By the 1.5-point criterion (PLAN.md section 5), the
  security part does not pass.
- The undefended Llama is far off, and in the direction of being much more secure in our
  pipeline. The most likely cause is a different layout in the two evaluations: we use the
  Meta-SecAlign template, with the data in a separate input turn; [6] does not describe the
  layout for undefended models. The base Qwen's executed rate of only 11.2% (gate 3) is most
  likely the same cause. A control was queued: base Qwen and Llama each run once with
  instruction and data in a single user message (Llama with its own chat template).
- **Single-message format check (early Nov 19)**: SecFid retested with instruction and data in
  one user message.

  | model | layout | executed | processed | ignored | other |
  |---|---|---|---|---|---|
  | Llama-3.1-8B-Instruct | separate input turn | 14.8 | 76.3 | 15.4 | 3.7 |
  | | single user message (its own template) | 33.0 | 75.5 | 7.4 | 2.5 |
  | | [6] published | 49.1 | 50.2 | 7.7 | 5.5 |
  | base Qwen3.5-2B | separate input turn | 11.2 | 68.7 | 17.1 | 8.6 |
  | | single user message | 15.2 | 60.0 | 15.8 | 15.2 |

  Layout explains part of Llama's gap. Note that this Llama run also switched to its own chat
  template, so layout and template are not separated; the base Qwen control changed only the
  layout. Executed more than doubles (14.8 → 33.0), ignored matches the published value (7.4
  vs 7.7), and executed on editing items also matches (59.0 vs 55.3). But executed on counting,
  extraction and translation items is still below the published values (18.6 vs 48.9, 18.4 vs
  32.6, 39.6 vs 61.9), so the prompt [6] used for undefended models differs from both layouts
  we tried. For base Qwen, layout matters much less (executed 11.2 → 15.2); it rarely executes
  SecFid injections under either layout, and the gate-3 conclusion stands.
- Two implications for our conclusions. First, the scoring is credible: for the same public
  model we measure a content-dropping rate of 27.5%, close to the published 26.1%. Second, the
  "separate input turn" layout by itself blocks most injections; the report must say so when
  comparing baselines.

## 12. Training data under this keep list

Of the 2,000 process-style pairs, 29% of the rejected responses ("executed the injected
instruction") are exactly the trusted instruction's text (because copy, join lines and one
sentence per line leave a single sentence unchanged), and 19% are a single number (count
sentences, count lines; recounted on the official seed-0 data; the 22% stated earlier came from
an older version of the data). This follows the construction rule of [8], but it means the
"do not execute" side of the signal is weak, and the security of process-style conditions has
to be tested mainly with SEP and the transfer attack.

## 13. Backend and environment

- vLLM 0.34.0 loads Qwen3.5-2B directly, at about 5,000-9,000 tokens/second.
- Problems found and fixed on the server: (a) with the acceleration packages
  (causal-conv1d, flash-linear-attention) installed, transformers also calls CUDA-only kernels
  on CPU, so the rehearsal fails on CPU, and the merge check on CPU hits the same path; (b) the
  container can use only about 26 cores, but the program opened threads for 192 cores, making
  the CPU parts about 10x slower; (c) the transformers backend did not stop at `<|im_end|>`
  and emitted an extra newline. Also, from reading the code, the evaluation pipeline would
  abort entirely if the scoring model were unavailable, so a "generate only, no scoring"
  option and partial SecFid scoring were added.
- One training run (4,000 pairs, 63 steps), measured: C1 26 minutes, C2 21 minutes, C3, C5
  and C6 about 19 minutes each, C4 4 minutes; about 13 GB of GPU memory. Each batch is padded
  to its longest example, so any condition that includes long responses takes close to 20
  minutes.
