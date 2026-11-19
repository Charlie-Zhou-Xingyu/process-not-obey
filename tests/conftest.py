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
    from toy_data import toy_alpaca_rows

    return toy_alpaca_rows()
