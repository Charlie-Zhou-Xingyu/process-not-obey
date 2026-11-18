"""Build a tiny randomly initialised Qwen3.5 (same hybrid architecture and tokenizer as
Qwen3.5-2B, 4 layers, width 64) for CPU smoke tests of training, merging and generation."""
from __future__ import annotations

from pathlib import Path


def make_tiny_qwen(snapshot: str, out: str | Path, seed: int = 0) -> str:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = Path(out)
    if (out / "config.json").exists():
        return str(out)
    cfg = AutoConfig.from_pretrained(snapshot)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 4
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.linear_num_key_heads, t.linear_num_value_heads = 4, 4
    t.linear_key_head_dim, t.linear_value_head_dim = 16, 16
    t.mtp_num_hidden_layers = 0
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(snapshot).save_pretrained(out)
    return str(out)


if __name__ == "__main__":
    import sys

    print(make_tiny_qwen(sys.argv[1], sys.argv[2]))
