import glob
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _find_qwen_snapshot() -> str | None:
    """A local Qwen3.5 snapshot (tokenizer + config), if one is cached on this machine."""
    env = os.environ.get("PNO_QWEN_LOCAL")
    if env and Path(env).exists():
        return env
    pats = [
        str(ROOT.parent / "hw09-hardware-check/hardware-cache/models--Qwen--Qwen3.5-2B/snapshots/*"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="session")
def qwen_snapshot():
    p = _find_qwen_snapshot()
    if p is None:
        pytest.skip("no local Qwen3.5-2B snapshot (set PNO_QWEN_LOCAL)")
    return p


@pytest.fixture(scope="session")
def toy_alpaca():
    """A small synthetic stand-in for alpaca-cleaned, enough to exercise every builder."""
    from pno.data.alpaca import AlpacaRow

    subjects = ["The river", "A quiet town", "The old library", "Her garden", "The committee", "A small cafe",
                "The research team", "The mountain trail", "The museum", "The night train", "The orchard",
                "The harbor", "The village school", "A local bakery", "The city council", "The lighthouse"]
    verbs = ["stays open during the long summer evenings", "attracts many visitors every single year",
             "was rebuilt after the storm ten years ago", "offers a clear view of the whole valley",
             "has changed very little since the last century", "serves fresh bread to the entire street",
             "published its findings early in the spring", "closes for two weeks in the winter",
             "was painted blue by the new owners", "stands at the end of a narrow road",
             "became famous after the film was released", "is older than most people believe",
             "lost its roof during the heavy snow", "hosts a small market on Sundays",
             "was mentioned twice in the morning paper", "keeps its lights on until midnight",
             "looks different under the autumn light", "received a grant from the county",
             "is closed to traffic on public holidays", "draws water from the same old well"]
    tasks = ["Write a short poem about autumn.", "Describe the process of photosynthesis.",
             "List three benefits of regular exercise.", "Explain how a bill becomes a law.",
             "Give two tips for learning a new language.", "Name a famous painter from Italy.",
             "Suggest a healthy breakfast idea.", "Compose a haiku about the sea.",
             "Explain why the sky appears blue.", "Recommend a classic novel to read."]
    rows, i = [], 0
    for a in subjects:
        for b in verbs:
            for t in tasks[:3]:
                out = f"{a} {b}. {subjects[(i + 3) % len(subjects)]} {verbs[(i + 5) % len(verbs)]}."
                rows.append(AlpacaRow(i, tasks[i % len(tasks)], "", out))
                i += 1
    host_tasks = ["Summarize the passage in one sentence.", "Translate the passage into French.",
                  "List the nouns in the passage.", "Rewrite the passage in the past tense.",
                  "Classify the tone of the passage."]
    for a in subjects:
        for b in verbs:
            for t in host_tasks:
                rows.append(AlpacaRow(i, t, f"{a} {b}. It is well known nearby.", f"{a} {b}."))
                i += 1
    return rows
