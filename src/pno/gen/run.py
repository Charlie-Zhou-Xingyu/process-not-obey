"""Generate responses for a JSONL of structured prompts.

Input rows: ``{"key": ..., "prompt": {"instruction", "data", "system"}, "placement": optional}``.
Output rows: ``{"key": ..., "response": ..., "truncated": bool}`` (same order). Used for the self-generated
targets of ignore-style and ordinary benign pairs (``pno.data.build requests`` writes the
requests) and by the evaluation runners.

    python -m pno.gen.run --requests data/generated/requests.jsonl --model Qwen/Qwen3.5-2B \
        --out data/generated/responses.jsonl --backend vllm --max_new_tokens 512
"""
from __future__ import annotations

import argparse

from pno.formatting import Prompt, render_prompt
from pno.io import read_jsonl, write_jsonl


def render_rows(rows: list[dict], renderer: str, tokenizer=None, default_placement: str = "input_role") -> list[str]:
    return [render_prompt(Prompt.from_dict(r["prompt"]), renderer=renderer, tokenizer=tokenizer,
                          placement=r.get("placement", default_placement)) for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--renderer", default="chatml", choices=["chatml", "hf"])
    ap.add_argument("--placement", default="input_role", choices=["input_role", "user_concat"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8, help="hf backend only")
    ap.add_argument("--device", default=None, help="hf backend only")
    ap.add_argument("--max_model_len", type=int, default=4096, help="vllm backend only")
    ap.add_argument("--adapter", default=None, help="LoRA adapter applied on top of --model")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    args = ap.parse_args(argv)

    from pno.gen.backends import GenConfig, make_generator

    rows = read_jsonl(args.requests)
    kw = ({"batch_size": args.batch_size, "device": args.device, "adapter_path": args.adapter, "dtype": args.dtype}
          if args.backend == "hf" else
          {"max_model_len": args.max_model_len, "seed": args.seed, "lora_path": args.adapter,
           "dtype": args.dtype or "bfloat16"})
    gen = make_generator(args.backend, args.model, **kw)
    prompts = render_rows(rows, args.renderer, getattr(gen, "tok", None), args.placement)
    outs = gen.generate_full(prompts, GenConfig(max_new_tokens=args.max_new_tokens,
                                                temperature=args.temperature, seed=args.seed))
    n = write_jsonl(args.out, ({"key": r["key"], "response": o["text"], "truncated": o["truncated"]}
                               for r, o in zip(rows, outs)))
    print(f"wrote {n} responses to {args.out} ({sum(o['truncated'] for o in outs)} hit the token limit)")


if __name__ == "__main__":
    main()
