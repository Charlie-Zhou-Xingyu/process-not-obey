"""LoRA + DPO on one condition's preference pairs (TRL DPOTrainer).

Input rows (built by ``pno.data.build``)::

    {"id", "kind", "prompt": {"instruction", "data", "system"}, "chosen", "rejected", "meta"}

Prompts are rendered here (``--renderer chatml`` for Qwen3.5) and given to TRL as plain
strings, so TRL never applies a chat template; it appends the tokenizer's EOS
(``<|im_end|>`` for Qwen) to ``chosen`` and ``rejected``. All runs use the same number
of optimizer steps: ``max_steps = ceil(n_pairs / effective_batch)`` is fixed from the
nominal pair count in the config, not from what survives truncation.

Example::

    python -m pno.train.dpo --train data/generated/seed0/C3_process_ordinary.jsonl \
        --model Qwen/Qwen3.5-2B --out runs/C3/seed0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from pno.formatting import Prompt, render_prompt
from pno.io import load_config, read_jsonl, revision_for, write_json


def build_hf_dataset(rows: list[dict], renderer: str, tokenizer=None):
    from datasets import Dataset

    recs = []
    for r in rows:
        # Completions are stripped: one that starts with a newline would merge with the "\n\n"
        # ending the Qwen prompt, and the prompt would stop being a token prefix (see tests).
        chosen, rejected = r["chosen"].strip(), r["rejected"].strip()
        if not chosen or not rejected or chosen == rejected:
            raise ValueError(f"degenerate pair {r.get('id')}: chosen/rejected empty or identical")
        recs.append({
            "prompt": render_prompt(Prompt.from_dict(r["prompt"]), renderer=renderer, tokenizer=tokenizer),
            "chosen": chosen,
            "rejected": rejected,
        })
    return Dataset.from_list(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True, help="HF id or local path of the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="experiment.yaml (training section)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--renderer", default="chatml")
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    ap.add_argument("--max_steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=None, help="override per-device batch (smoke tests)")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from pno.gen.backends import pick_device

    full_cfg = load_config(args.config)
    cfg = full_cfg["training"]
    rev = revision_for(args.model, full_cfg)
    device = pick_device(args.device)
    on_cuda = device == "cuda"

    if on_cuda and torch.cuda.device_count() > 1:
        raise SystemExit(f"{torch.cuda.device_count()} GPUs are visible. Batch size and step count assume one; "
                         "set CUDA_VISIBLE_DEVICES to a single device.")
    rows = read_jsonl(args.train)
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_hf_dataset(rows, args.renderer, tok).shuffle(seed=args.seed)

    per_device = args.batch_size or cfg["per_device_batch_size"]
    grad_acc = cfg["effective_batch_size"] // per_device if not args.batch_size else 1
    eff = per_device * grad_acc
    n_nominal = len(rows) if args.limit else cfg["pairs_per_condition"]
    max_steps = args.max_steps or math.ceil(n_nominal / eff) * cfg["epochs"]

    dtype = torch.bfloat16 if on_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, revision=rev).to(device)

    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
    dpo_args = DPOConfig(
        output_dir=args.out,
        seed=args.seed,
        data_seed=args.seed,
        beta=cfg["beta"],
        loss_type=[cfg.get("loss_type", "sigmoid")],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg["warmup_ratio"],          # transformers >= 5: a float in [0, 1) is a ratio
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "adamw_torch"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_acc,
        max_length=cfg["max_length"],
        bf16=on_cuda,
        gradient_checkpointing=on_cuda and cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy="no",
        report_to=cfg.get("report_to") or "none",
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
    )
    trainer = DPOTrainer(model=model, args=dpo_args, train_dataset=ds, processing_class=tok,
                         peft_config=lora)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    out = Path(args.out)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    write_json(out / "train_log.json", {
        "args": vars(args), "training_cfg": cfg, "base_revision": rev, "max_steps": max_steps,
        "effective_batch": eff,
        "n_rows": len(rows), "n_after_prepare": len(trainer.train_dataset), "seconds": elapsed,
        "train_metrics": result.metrics, "log_history": trainer.state.log_history,
    })
    print(json.dumps({"out": str(out), "seconds": round(elapsed, 1), **result.metrics}, indent=1))


if __name__ == "__main__":
    main()
