# One-off analysis scripts (Nov 18, seed 0)

Written while reading the first SecFid results; run from the repo root on the machine that
holds `outputs/eval` (they import `pno`, except `summarize.py` and `bytask.py`).

| script | question |
|---|---|
| `summarize.py <root>` | one table of every summary found under `<root>/outputs/eval` |
| `bytask.py <root>` | SecFid by task and by persuasion; clean-input correctness by task |
| `primary.py` | condition differences with a paired item bootstrap, with and without the counting task |
| `count_bias.py <models>` | counting / extraction: is a `processed` label on the attacked input also given on the clean input of the same item (a general "one more" habit)? |
| `edit_success.py <models>` | editing: was the requested edit applied (items whose edit can be read off the instruction), and is the probe sentence kept verbatim? |
| `edit_check.py <models>` | editing / translation: word-level similarity of the output to the input and to the references |

They are not part of the pipeline or of the tests.
