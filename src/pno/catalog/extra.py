"""Sentence- and line-level operations (proposal, risk 1 fallback).

These are easier for a 2B model than character-level string edits and closer to SecFid's
natural-language tasks. They are tagged ``source="extra"`` so that analyses can report
results with and without them.
"""
from __future__ import annotations

from pno.catalog.base import Registry, Transform, split_lines, split_sentences


def _number_sentences(t: str) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(split_sentences(t), 1))


def _sentences_per_line(t: str) -> str:
    return "\n".join(split_sentences(t))


def _reverse_sentence_order(t: str) -> str:
    return " ".join(reversed(split_sentences(t)))


def _sort_sentences(t: str) -> str:
    return " ".join(sorted(split_sentences(t), key=str.lower))


def _count_sentences(t: str) -> str:
    return str(len(split_sentences(t)))


def _bullet_sentences(t: str) -> str:
    return "\n".join(f"- {s}" for s in split_sentences(t))


def _number_lines(t: str) -> str:
    return "\n".join(f"{i}. {ln}" for i, ln in enumerate(split_lines(t), 1))


def _sort_lines(t: str) -> str:
    return "\n".join(sorted(split_lines(t), key=str.lower))


def _reverse_lines(t: str) -> str:
    return "\n".join(reversed(split_lines(t)))


def _count_lines(t: str) -> str:
    return str(len(split_lines(t)))


def _join_lines(t: str) -> str:
    return " ".join(ln.strip() for ln in split_lines(t))


def _quote_lines(t: str) -> str:
    return "\n".join(f"> {ln}" for ln in split_lines(t))


def _first_word_each_sentence(t: str) -> str:
    return " ".join(s.split()[0] for s in split_sentences(t) if s.split())


def _last_word_each_sentence(t: str) -> str:
    return " ".join(s.split()[-1].strip(".,!?;:") for s in split_sentences(t) if s.split())


def register(reg: Registry) -> None:
    add = reg.add
    add(Transform("number_sentences", "Put each sentence of the {target} on its own line, numbered 1., 2., 3., and so on.",
                  _number_sentences, "sentence", "text", "extra"))
    add(Transform("sentences_per_line", "Put each sentence of the {target} on its own line.",
                  _sentences_per_line, "sentence", "text", "extra"))
    add(Transform("reverse_sentence_order", "Reverse the order of the sentences in the {target}.",
                  _reverse_sentence_order, "sentence", "text", "extra"))
    add(Transform("sort_sentences", "Sort the sentences of the {target} alphabetically.",
                  _sort_sentences, "sentence", "text", "extra"))
    add(Transform("count_sentences", "Count the sentences in the {target}. Answer with the number only.",
                  _count_sentences, "sentence", "text", "extra"))
    add(Transform("bullet_sentences", "Turn the {target} into a bulleted list with one sentence per bullet, each starting with \"- \".",
                  _bullet_sentences, "sentence", "text", "extra"))
    add(Transform("first_word_each_sentence", "List the first word of each sentence in the {target}, separated by spaces.",
                  _first_word_each_sentence, "sentence", "text", "extra"))
    add(Transform("last_word_each_sentence", "List the last word of each sentence in the {target}, without punctuation, separated by spaces.",
                  _last_word_each_sentence, "sentence", "text", "extra"))
    add(Transform("number_lines", "Number the lines of the {target} as 1., 2., 3., and so on.",
                  _number_lines, "line", "lines", "extra"))
    add(Transform("sort_lines", "Sort the lines of the {target} alphabetically.",
                  _sort_lines, "line", "lines", "extra"))
    add(Transform("reverse_lines", "Reverse the order of the lines in the {target}.",
                  _reverse_lines, "line", "lines", "extra"))
    add(Transform("count_lines", "Count the lines in the {target}. Answer with the number only.",
                  _count_lines, "line", "lines", "extra"))
    add(Transform("join_lines", "Join the lines of the {target} into a single line, separated by single spaces.",
                  _join_lines, "line", "lines", "extra"))
    add(Transform("quote_lines", "Prefix every line of the {target} with \"> \".",
                  _quote_lines, "line", "lines", "extra"))
