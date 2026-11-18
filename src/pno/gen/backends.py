"""Greedy text generation on pre-rendered prompt strings.

* ``vllm`` (server): fast batched decoding. Prompts are tokenized by us with
  ``add_special_tokens=False`` and passed as token ids, so a template that already writes
  ``<|begin_of_text|>`` never gets a second BOS.
* ``hf`` (laptop / CI): plain ``transformers`` batched generation with left padding.

``generate`` returns only the newly generated text, special tokens removed;
``generate_full`` also says whether a reply stopped at the token limit (``truncated``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


def pick_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFGenerator:
    def __init__(self, model_path: str, adapter_path: str | None = None, device: str | None = None,
                 dtype: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.tok = AutoTokenizer.from_pretrained(adapter_path or model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype))
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    def generate_full(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[dict]:
        import torch

        torch.manual_seed(cfg.seed)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        eos_ids.discard(None)
        # sort by length to waste less padding, then restore order
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res: dict[int, dict] = {}
        for b in range(math.ceil(len(order) / self.batch_size)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if cfg.temperature > 0:
                kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kwargs)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, row in zip(idx, new):
                ids = row.tolist()
                res[i] = {"text": self.tok.decode(row, skip_special_tokens=True),
                          "truncated": not any(t in eos_ids for t in ids)}
        return [res[i] for i in range(len(prompts))]

    def generate(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[str]:
        return [r["text"] for r in self.generate_full(prompts, cfg)]


class VLLMGenerator:
    def __init__(self, model_path: str, tokenizer_path: str | None = None, dtype: str = "bfloat16",
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                 tensor_parallel_size: int = 1, seed: int = 0, **llm_kwargs):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        kwargs = dict(model=model_path, tokenizer=tokenizer_path or model_path, dtype=dtype,
                      max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                      tensor_parallel_size=tensor_parallel_size, seed=seed, **llm_kwargs)
        try:
            self.llm = LLM(enable_prefix_caching=True, **kwargs)
        except TypeError:          # option renamed or removed in this vLLM version
            self.llm = LLM(**kwargs)
        self.max_model_len = max_model_len

    def generate_full(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[dict]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        ids = self.tok(prompts, add_special_tokens=False)["input_ids"]
        limit = self.max_model_len - cfg.max_new_tokens
        too_long = sum(len(x) > limit for x in ids)
        if too_long:
            print(f"[vllm] warning: {too_long} prompts exceed {limit} tokens; keeping their last {limit} tokens")
        ids = [x[-limit:] if len(x) > limit else x for x in ids]
        sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_new_tokens,
                            seed=cfg.seed, skip_special_tokens=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=True)
        return [{"text": o.outputs[0].text, "truncated": o.outputs[0].finish_reason == "length"} for o in outs]

    def generate(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[str]:
        return [r["text"] for r in self.generate_full(prompts, cfg)]


def make_generator(backend: str, model_path: str, **kw):
    if backend == "vllm":
        return VLLMGenerator(model_path, **kw)
    if backend == "hf":
        return HFGenerator(model_path, **kw)
    raise ValueError(f"unknown backend {backend!r}")
