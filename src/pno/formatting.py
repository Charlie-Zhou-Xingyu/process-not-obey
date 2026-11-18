"""Prompt rendering with a separate role for untrusted data.

The trusted instruction goes in the user message and the untrusted text in a following
message whose role is ``input`` (the Meta-SecAlign [4] / SecPO [5] convention).

We render prompts to strings ourselves, so training (TRL on pre-rendered strings) and
inference (vLLM / HF on token ids) see byte-identical prompts. Two renderers:

* ``chatml``  - Qwen3.5 ChatML in non-thinking mode. The stock Qwen3.5 template raises on
  unknown roles, so we render ``<|im_start|>input ... <|im_end|>`` ourselves.
  ``tests/test_formatting.py`` checks that for system/user messages this renderer matches
  the tokenizer's own template byte for byte.
* ``hf``      - the model's own ``tokenizer.apply_chat_template`` (used for the Llama-3.1
  references, whose released templates accept the ``input`` role).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DataPlacement = Literal["input_role", "user_concat"]

INPUT_ROLE = "input"
QWEN_NOTHINK_PREFIX = "<think>\n\n</think>\n\n"

# Strings that would be tokenized as chat structure if they appeared in untrusted text.
# The released SecPO Qwen template removes the same kind of delimiters from "input" content.
QWEN_UNTRUSTED_DELIMITERS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<tool_response>",
    "</tool_response>", "<think>", "</think>", "<tools>", "</tools>",
)


def sanitize_untrusted(text: str, delimiters: tuple[str, ...] = QWEN_UNTRUSTED_DELIMITERS) -> str:
    """Delete chat delimiters from untrusted text, repeating until nothing changes, so that
    data cannot open a new turn (``<|im_end|><|im_start|>user ...``)."""
    while True:
        new = text
        for d in delimiters:
            new = new.replace(d, "")
        if new == text:
            return text
        text = new


@dataclass(frozen=True)
class Prompt:
    """One structured prompt. ``data`` is untrusted; ``instruction`` and ``system`` are trusted."""

    instruction: str
    data: str | None = None
    system: str | None = None

    def messages(self, placement: DataPlacement = "input_role") -> list[dict]:
        msgs: list[dict] = []
        if self.system is not None:
            msgs.append({"role": "system", "content": self.system})
        if self.data is None:
            msgs.append({"role": "user", "content": self.instruction})
        elif placement == "input_role":
            msgs.append({"role": "user", "content": self.instruction})
            msgs.append({"role": INPUT_ROLE, "content": self.data})
        elif placement == "user_concat":
            msgs.append({"role": "user", "content": f"{self.instruction}\n\n{self.data}"})
        else:
            raise ValueError(f"unknown placement {placement!r}")
        return msgs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        return cls(instruction=d["instruction"], data=d.get("data"), system=d.get("system"))


def render_chatml(messages: list[dict], add_generation_prompt: bool = True,
                  enable_thinking: bool = False) -> str:
    """Qwen3.5 ChatML. Content is trimmed exactly like the stock template (Jinja ``trim``)."""
    parts = []
    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            raise ValueError("system message must come first")
        content = sanitize_untrusted(m["content"]) if m["role"] == INPUT_ROLE else m["content"]
        parts.append(f"<|im_start|>{m['role']}\n{content.strip()}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
        parts.append("<think>\n" if enable_thinking else QWEN_NOTHINK_PREFIX)
    return "".join(parts)


def render(messages: list[dict], renderer: str = "chatml", tokenizer=None,
           add_generation_prompt: bool = True) -> str:
    if renderer == "chatml":
        return render_chatml(messages, add_generation_prompt=add_generation_prompt)
    if renderer == "hf":
        if tokenizer is None:
            raise ValueError("renderer='hf' needs a tokenizer")
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=add_generation_prompt)
    raise ValueError(f"unknown renderer {renderer!r}")


def render_prompt(prompt: Prompt, renderer: str = "chatml", tokenizer=None,
                  placement: DataPlacement = "input_role") -> str:
    return render(prompt.messages(placement), renderer=renderer, tokenizer=tokenizer)


# Jinja template saved next to our Qwen3.5 checkpoints so that tools that call
# apply_chat_template (vLLM chat API, lm-eval) render the input role the same way. It does not
# strip delimiters from untrusted text (render_chatml does); our own pipeline never uses it.
def qwen_template_with_input_role(stock_template: str) -> str:
    old = '{%- elif message.role == "user" %}'
    if old not in stock_template:
        raise ValueError("unexpected Qwen chat template; cannot add the input role")
    return stock_template.replace(old, '{%- elif message.role == "user" or message.role == "input" %}', 1)
