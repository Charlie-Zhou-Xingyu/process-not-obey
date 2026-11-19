"""Greedy text generation on pre-rendered prompt strings.

* ``vllm`` (server): fast batched decoding. Prompts are tokenized by us with
  ``add_special_tokens=False`` and passed as token ids, so a template that already writes
  ``<|begin_of_text|>`` never gets a second BOS.
* ``hf`` (laptop / CI): plain ``transformers`` batched generation with left padding.

``generate`` returns only the newly generated text, special tokens removed;
``generate_full`` also says whether a reply stopped at the token limit (``truncated``).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pno.io import revision_for


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


GPU_ONLY_KERNELS = ("causal_conv1d", "fla")     # import names of causal-conv1d and flash-linear-attention
QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"


def use_reference_kernels() -> None:
    """Make transformers run Qwen3.5's Gated DeltaNet layers with its own PyTorch code.

    transformers binds the ``causal-conv1d`` and ``flash-linear-attention`` kernels when it
    imports the Qwen3.5 model code, if those packages are importable (``00_setup_server.sh``
    installs them for training speed), and then calls them on every device. The causal-conv1d
    kernel is CUDA-only ("Expected x.is_cuda() to be true"). Hiding both packages before the
    model code is imported selects the reference implementation, which runs anywhere.
    """
    bound = [n for n in GPU_ONLY_KERNELS if sys.modules.get(n) is not None]
    if bound and QWEN35_MODELING in sys.modules:
        raise RuntimeError(
            f"the Qwen3.5 model code was already imported with the CUDA-only kernels {bound}; "
            "choose the device (PNO_DEVICE=cpu) before any model is loaded in this process")
    for name in GPU_ONLY_KERNELS:
        sys.modules[name] = None      # a later ``import name`` raises ImportError


def cpu_quota(path: str = "/sys/fs/cgroup/cpu.max", visible: int | None = None) -> int | None:
    """Cores this container may use (cgroup v2), if that is fewer than the cores it can see."""
    try:
        quota, period = Path(path).read_text().split()[:2]
        n = max(1, int(quota) // int(period))      # "max" (no quota) raises ValueError
    except (OSError, ValueError):
        return None
    visible = visible or os.cpu_count() or n
    return n if n < visible else None


def limit_torch_threads() -> None:
    """Keep torch's CPU thread pool within the container's CPU quota.

    A rented container often sees every core of its host (192 on our server) while its quota
    is about 26. torch sizes its pool by the visible cores, the kernel throttles the process,
    and CPU work ran up to 10x slower there. Only our own transformers processes need this:
    vLLM sets the thread count of its engine itself, as long as OMP_NUM_THREADS is unset.
    """
    import torch

    n = cpu_quota()
    if n and n < torch.get_num_threads():
        torch.set_num_threads(n)


def pick_device(preferred: str | None = None) -> str:
    """The device for a model run with transformers; also prepares the process for it."""
    import torch

    device = preferred or os.environ.get("PNO_DEVICE")      # e.g. PNO_DEVICE=cpu for the rehearsal
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if not device.startswith("cuda"):
        use_reference_kernels()
    limit_torch_threads()
    return device


class HFGenerator:
    def __init__(self, model_path: str, adapter_path: str | None = None, device: str | None = None,
                 dtype: str | None = None, batch_size: int = 8, tokenizer_path: str | None = None,
                 revision: str | None = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        rev = revision_for(model_path) if revision == "auto" else revision
        tok_src = tokenizer_path or adapter_path or model_path
        self.tok = AutoTokenizer.from_pretrained(tok_src, revision=rev if tok_src == model_path else revision_for(tok_src))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), revision=rev)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    def generate_full(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[dict]:
        import torch

        torch.manual_seed(cfg.seed)
        # Stop on the tokenizer's end token as well as the model config's, as vLLM does. For
        # Qwen3.5 the config names <|endoftext|> and the tokenizer <|im_end|>; without this,
        # generation runs past the end of the turn ("<|im_end|>\n<|endoftext|>").
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
            kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                          eos_token_id=sorted(eos_ids))
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
    """vLLM engine. A trained run is either a merged checkpoint (``model_path``) or, as in
    Meta-SecAlign's evaluation code, the base model plus a LoRA adapter applied at run time
    (``lora_path``: ``LLM(enable_lora=True, max_lora_rank=r)`` and a ``LoRARequest``)."""

    def __init__(self, model_path: str, tokenizer_path: str | None = None, lora_path: str | None = None,
                 dtype: str = "bfloat16", max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                 tensor_parallel_size: int = 1, seed: int = 0, revision: str | None = "auto", **llm_kwargs):
        from transformers import AutoTokenizer
        from vllm import LLM

        rev = revision_for(model_path) if revision == "auto" else revision
        tok_src = tokenizer_path or model_path
        tok_rev = rev if tok_src == model_path else revision_for(tok_src)
        self.tok = AutoTokenizer.from_pretrained(tok_src, revision=tok_rev)
        kwargs = dict(model=model_path, tokenizer=tok_src, dtype=dtype,
                      max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                      tensor_parallel_size=tensor_parallel_size, seed=seed, **llm_kwargs)
        if rev:
            kwargs["revision"] = rev
        if tok_rev:
            kwargs["tokenizer_revision"] = tok_rev
        self.lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest

            rank = json.loads((Path(lora_path) / "adapter_config.json").read_text())["r"]
            kwargs.update(enable_lora=True, max_lora_rank=rank)
            self.lora_request = LoRARequest("pno_adapter", 1, str(lora_path))
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
            print(f"[vllm] warning: {too_long} prompts exceed {limit} tokens; cutting tokens out of their middle")
        head = limit // 2      # keeps the trusted instruction at the start and the generation prompt at the end
        ids = [x[:head] + x[len(x) - (limit - head):] if len(x) > limit else x for x in ids]
        sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_new_tokens,
                            seed=cfg.seed, skip_special_tokens=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=True,
                                 lora_request=self.lora_request)
        return [{"text": o.outputs[0].text, "truncated": o.outputs[0].finish_reason == "length"} for o in outs]

    def generate(self, prompts: list[str], cfg: GenConfig = GenConfig()) -> list[str]:
        return [r["text"] for r in self.generate_full(prompts, cfg)]


def make_generator(backend: str, model_path: str, **kw):
    if backend == "vllm":
        return VLLMGenerator(model_path, **kw)
    if backend == "hf":
        return HFGenerator(model_path, **kw)
    raise ValueError(f"unknown backend {backend!r}")
