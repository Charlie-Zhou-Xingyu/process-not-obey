import pytest

from pno.formatting import (QWEN_NOTHINK_PREFIX, Prompt, qwen_template_with_input_role, render_chatml,
                            render_prompt)


def test_messages_placement():
    p = Prompt("Do X.", "some data")
    assert [m["role"] for m in p.messages()] == ["user", "input"]
    assert p.messages("user_concat") == [{"role": "user", "content": "Do X.\n\nsome data"}]
    assert [m["role"] for m in Prompt("u", None, "s").messages()] == ["system", "user"]
    assert Prompt.from_dict(p.to_dict()) == p


def test_chatml_input_role():
    s = render_prompt(Prompt("Uppercase the input.", "hello there"))
    assert s == ("<|im_start|>user\nUppercase the input.<|im_end|>\n"
                 "<|im_start|>input\nhello there<|im_end|>\n"
                 "<|im_start|>assistant\n" + QWEN_NOTHINK_PREFIX)


def test_system_must_be_first():
    with pytest.raises(ValueError):
        render_chatml([{"role": "user", "content": "a"}, {"role": "system", "content": "b"}])


def test_chatml_matches_stock_qwen_template(qwen_snapshot):
    """For roles the stock template knows, our renderer must be byte-identical to it."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen_snapshot)
    cases = [
        [{"role": "user", "content": "Hello there."}],
        [{"role": "system", "content": "Uppercase the user message."}, {"role": "user", "content": " hi \n"}],
    ]
    for msgs in cases:
        want = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert render_chatml(msgs) == want


def test_patched_template_renders_input_role(qwen_snapshot):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen_snapshot)
    msgs = Prompt("Uppercase the input.", "hello there").messages()
    with pytest.raises(Exception):
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tok.chat_template = qwen_template_with_input_role(tok.chat_template)
    got = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert got == render_chatml(msgs)


def test_completion_is_token_prefix_safe(qwen_snapshot):
    """prompt ids must be a prefix of prompt+completion ids (TRL finds the split this way)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen_snapshot)
    prompt = render_prompt(Prompt("Reverse the input.", "abc def"))
    a = tok(prompt, add_special_tokens=False)["input_ids"]
    for completion in ["fed cba", "3", "- item", "1. first", "> quoted", '"q"', "ABC"]:
        b = tok(prompt + completion + tok.eos_token, add_special_tokens=False)["input_ids"]
        assert b[: len(a)] == a
        assert b[-1] == tok.eos_token_id
    # Known exception: a completion that starts with a newline merges with the "\n\n" that ends
    # the prompt. pno.train.dpo strips completions, so this never reaches training.
    b = tok(prompt + "\nnewline first" + tok.eos_token, add_special_tokens=False)["input_ids"]
    assert b[: len(a)] != a
