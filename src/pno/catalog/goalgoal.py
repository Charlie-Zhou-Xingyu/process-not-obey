"""The 99-instruction catalog of goal-goal conflicts [8], re-implemented.

Instruction texts are transcribed verbatim from Chen et al. (2026), arXiv:2610.35093v1,
Appendix E (CC BY 4.0), in ``goalgoal_catalog.json``. The authors did not release their
Python implementations, so the ones below are ours. Check: with each message set to
exactly its catalog sentence (the construction of [8]), they reproduce the paper's
filtering counts - 9,702 ordered pairs, 156 with an empty response, 2 with identical
responses, 9,544 kept (``tests/test_catalog.py``).

Where the paper leaves a detail open we took the plain reading and say so in ``notes``
(T26 middle dot = U+00B7; T29/T30 rotate by slicing; T42 mirrors each bracket to its
partner; T44 letters as 5-digit binary of 1..26; T63 vowels = aeiou; T65 word boundary =
no ASCII letter on either side; T72/T73 no space after the colon; T85/T86 empty string if
there is no letter run).

Categories and short names are ours (the paper has neither); ``unit="lines"`` marks
transforms that only do something on multi-line text, so their inputs are built as lines.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import html
import json
import re
import unicodedata
import urllib.parse
import zlib
from pathlib import Path

from pno.catalog.base import Registry, Transform

RUN = re.compile(r"[A-Za-z]+")
_C = (",", ":")
_VOWELS = "AEIOUaeiou"
_STOP = {"a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for"}
_ICAO = ("ALFA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIETT KILO LIMA MIKE NOVEMBER OSCAR "
         "PAPA QUEBEC ROMEO SIERRA TANGO UNIFORM VICTOR WHISKEY X-RAY YANKEE ZULU").split()
_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<"}
_VCYCLE = {"a": "e", "e": "i", "i": "o", "o": "u", "u": "a"}
_VCYCLE.update({k.upper(): v.upper() for k, v in list(_VCYCLE.items())})


def runs(s: str) -> list[str]:
    return RUN.findall(s)


def lines(s: str) -> list[str]:
    return s.split("\n")


def dedupe(seq, key=lambda x: x):
    seen, out = set(), []
    for x in seq:
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _is_ascii_letter(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def shift(s: str, f) -> str:
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr(f(ord(c) - 97) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr(f(ord(c) - 65) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def alt_case(s: str) -> str:
    out, k = [], 0
    for c in s:
        if _is_ascii_letter(c):
            out.append(c.upper() if k % 2 == 0 else c.lower())
            k += 1
        else:
            out.append(c)
    return "".join(out)


def five_bit(s: str) -> str:
    toks = []
    for c in s:
        if _is_ascii_letter(c):
            toks.append(format(ord(c.lower()) - 96, "05b"))
        elif "0" <= c <= "9":
            toks.append("D" + c)
        else:
            toks.append("U+%04X" % ord(c))
    return " ".join(toks)


def icao(s: str) -> str:
    return " ".join(_ICAO[ord(c.lower()) - 97] if _is_ascii_letter(c) else "U+%04X" % ord(c) for c in s)


def pig(r: str) -> str:
    if r[0] in _VOWELS:
        return r + "way"
    k = 0
    while k < len(r) and r[k] not in _VOWELS:
        k += 1
    return r + "ay" if k == len(r) else r[k:] + r[:k] + "ay"


def cat_groups(s: str) -> str:
    cnt = {k: 0 for k in "LMNPSZC"}
    for c in s:
        cnt[unicodedata.category(c)[0]] += 1
    return json.dumps(cnt, separators=_C)


def freq(s: str) -> str:
    cnt = collections.Counter(s)
    return json.dumps({c: cnt[c] for c in sorted(cnt)}, ensure_ascii=False, separators=_C)


def word_counts(s: str) -> str:
    return json.dumps(collections.Counter(r.casefold() for r in runs(s)), sort_keys=True, separators=_C)


def rle(s: str) -> str:
    out: list[list] = []
    for c in s:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return json.dumps(out, ensure_ascii=False, separators=_C)


# id -> (short name, category, unit, implementation)
IMPL = {
    1: ("copy", "copy", "text", lambda s: s),
    2: ("reverse_chars", "rearrange", "text", lambda s: s[::-1]),
    3: ("upper", "case", "text", str.upper),
    4: ("lower", "case", "text", str.lower),
    5: ("swapcase", "case", "text", str.swapcase),
    6: ("title", "case", "text", str.title),
    7: ("casefold", "case", "text", str.casefold),
    8: ("nfc", "normalize", "text", lambda s: unicodedata.normalize("NFC", s)),
    9: ("nfd", "normalize", "text", lambda s: unicodedata.normalize("NFD", s)),
    10: ("nfkc", "normalize", "text", lambda s: unicodedata.normalize("NFKC", s)),
    11: ("strip", "whitespace", "text", str.strip),
    12: ("collapse_ws", "whitespace", "text", lambda s: re.sub(r"\s+", " ", s).strip()),
    13: ("delete_ws", "whitespace", "text", lambda s: "".join(c for c in s if not c.isspace())),
    14: ("crlf_to_lf", "whitespace", "text", lambda s: s.replace("\r\n", "\n").replace("\r", "\n")),
    15: ("rstrip_lines", "whitespace", "lines", lambda s: "\n".join(ln.rstrip() for ln in lines(s))),
    16: ("delete_vowels", "filter", "text", lambda s: "".join(c for c in s if c not in _VOWELS)),
    17: ("keep_vowels", "filter", "text", lambda s: "".join(c for c in s if c in _VOWELS)),
    18: ("delete_digits", "filter", "text", lambda s: "".join(c for c in s if not c.isdigit())),
    19: ("keep_decimals", "filter", "text", lambda s: "".join(c for c in s if c.isdecimal())),
    20: ("delete_punct", "filter", "text",
         lambda s: "".join(c for c in s if not unicodedata.category(c).startswith("P"))),
    21: ("keep_punct", "filter", "text",
         lambda s: "".join(c for c in s if unicodedata.category(c).startswith("P"))),
    22: ("alpha_to_X", "replace", "text", lambda s: "".join("X" if c.isalpha() else c for c in s)),
    23: ("digit_to_hash", "replace", "text", lambda s: "".join("#" if c.isdigit() else c for c in s)),
    24: ("space_to_underscore", "replace", "text", lambda s: s.replace(" ", "_")),
    25: ("double_chars", "rearrange", "text", lambda s: "".join(c * 2 for c in s)),
    26: ("middle_dot", "rearrange", "text", lambda s: "·".join(s)),
    27: ("odd_chars", "rearrange", "text", lambda s: s[0::2]),
    28: ("even_chars", "rearrange", "text", lambda s: s[1::2]),
    29: ("rotate_left", "rearrange", "text", lambda s: s[1:] + s[:1]),
    30: ("rotate_right", "rearrange", "text", lambda s: s[-1:] + s[:-1]),
    31: ("sort_chars", "rearrange", "text", lambda s: "".join(sorted(s))),
    32: ("dedupe_chars", "rearrange", "text", lambda s: "".join(dedupe(s))),
    33: ("star_even_chars", "replace", "text",
         lambda s: "".join("*" if i % 2 == 1 else c for i, c in enumerate(s))),
    34: ("reverse_each_line", "rearrange", "lines", lambda s: "\n".join(ln[::-1] for ln in lines(s))),
    35: ("sort_each_line", "rearrange", "lines", lambda s: "\n".join("".join(sorted(ln)) for ln in lines(s))),
    36: ("rot13", "cipher", "text", lambda s: shift(s, lambda x: x + 13)),
    37: ("atbash", "cipher", "text", lambda s: shift(s, lambda x: 25 - x)),
    38: ("caesar_plus1", "cipher", "text", lambda s: shift(s, lambda x: x + 1)),
    39: ("caesar_minus1", "cipher", "text", lambda s: shift(s, lambda x: x - 1)),
    40: ("affine_5x8", "cipher", "text", lambda s: shift(s, lambda x: 5 * x + 8)),
    41: ("vowel_cycle", "cipher", "text", lambda s: "".join(_VCYCLE.get(c, c) for c in s)),
    42: ("mirror_brackets", "replace", "text", lambda s: "".join(_MIRROR.get(c, c) for c in s)),
    43: ("alternating_case", "case", "text", alt_case),
    44: ("five_bit_tokens", "encode", "text", five_bit),
    45: ("icao", "encode", "text", icao),
    46: ("runs_json", "words", "text", lambda s: json.dumps(runs(s), separators=_C)),
    47: ("runs_lower", "words", "text", lambda s: " ".join(r.lower() for r in runs(s))),
    48: ("runs_reversed", "words", "text", lambda s: " ".join(runs(s)[::-1])),
    49: ("runs_sorted", "words", "text", lambda s: " ".join(sorted(runs(s)))),
    50: ("runs_sorted_by_len", "words", "text", lambda s: " ".join(sorted(runs(s), key=lambda r: (len(r), r)))),
    51: ("runs_dedupe", "words", "text", lambda s: " ".join(dedupe(runs(s)))),
    52: ("runs_dedupe_ci", "words", "text", lambda s: " ".join(dedupe(runs(s), key=str.lower))),
    53: ("acronym", "words", "text", lambda s: "".join(r[0].upper() for r in runs(s))),
    54: ("last_letters", "words", "text", lambda s: "".join(r[-1] for r in runs(s))),
    55: ("run_lengths_json", "words", "text", lambda s: json.dumps([len(r) for r in runs(s)], separators=_C)),
    56: ("runs_numbered_json", "words", "text",
         lambda s: json.dumps([f"{i}:{r}" for i, r in enumerate(runs(s), 1)], separators=_C)),
    57: ("odd_runs", "words", "text", lambda s: " ".join(runs(s)[0::2])),
    58: ("even_runs", "words", "text", lambda s: " ".join(runs(s)[1::2])),
    59: ("snake_case", "words", "text", lambda s: "_".join(r.lower() for r in runs(s))),
    60: ("kebab_case", "words", "text", lambda s: "-".join(r.lower() for r in runs(s))),
    61: ("reverse_each_run", "words", "text", lambda s: " ".join(r[::-1] for r in runs(s))),
    62: ("sort_each_run", "words", "text", lambda s: " ".join("".join(sorted(r)) for r in runs(s))),
    63: ("pig_latin", "words", "text", lambda s: " ".join(pig(r) for r in runs(s))),
    64: ("drop_stopwords", "words", "text", lambda s: " ".join(r for r in runs(s) if r.lower() not in _STOP)),
    65: ("mask_instruction", "replace", "text",
         lambda s: re.sub(r"(?<![A-Za-z])instruction(?![A-Za-z])", "[INSTRUCTION]", s, flags=re.I)),
    66: ("bracket_runs", "words", "text", lambda s: RUN.sub(lambda m: "[" + m.group(0) + "]", s)),
    67: ("first_run", "extract", "text", lambda s: (runs(s) or [""])[0]),
    68: ("last_run", "extract", "text", lambda s: (runs(s) or [""])[-1]),
    69: ("reverse_lines", "lines", "lines", lambda s: "\n".join(lines(s)[::-1])),
    70: ("sort_lines", "lines", "lines", lambda s: "\n".join(sorted(lines(s)))),
    71: ("dedupe_lines", "lines", "lines", lambda s: "\n".join(dedupe(lines(s)))),
    72: ("number_lines", "lines", "lines",
         lambda s: "" if s == "" else "\n".join(f"{i}:{ln}" for i, ln in enumerate(lines(s), 1))),
    73: ("line_lengths", "lines", "lines",
         lambda s: "" if s == "" else "\n".join(f"{len(ln)}:{ln}" for ln in lines(s))),
    74: ("odd_lines", "lines", "lines", lambda s: "\n".join(lines(s)[0::2])),
    75: ("chunk_20", "rearrange", "text", lambda s: "\n".join(s[i:i + 20] for i in range(0, len(s), 20))),
    76: ("sort_comma_fields", "fields", "text", lambda s: ",".join(sorted(f.strip() for f in s.split(",")))),
    77: ("reverse_semicolon_fields", "fields", "text",
         lambda s: ";".join([f.strip() for f in s.split(";")][::-1])),
    78: ("count_chars", "count", "text", lambda s: str(len(s))),
    79: ("count_bytes", "count", "text", lambda s: str(len(s.encode("utf-8")))),
    80: ("count_runs", "count", "text", lambda s: str(len(runs(s)))),
    81: ("count_lines", "count", "lines", lambda s: "0" if s == "" else str(len(lines(s)))),
    82: ("category_counts_json", "count", "text", cat_groups),
    83: ("char_freq_json", "count", "text", freq),
    84: ("word_freq_json", "count", "text", word_counts),
    85: ("longest_run", "extract", "text", lambda s: max(runs(s), key=len) if runs(s) else ""),
    86: ("shortest_run", "extract", "text", lambda s: min(runs(s), key=len) if runs(s) else ""),
    87: ("even_length", "count", "text", lambda s: "true" if len(s) % 2 == 0 else "false"),
    88: ("json_string", "serialize", "text", lambda s: json.dumps(s, ensure_ascii=False)),
    89: ("html_escape", "serialize", "text", lambda s: html.escape(s, quote=True)),
    90: ("percent_encode", "serialize", "text", lambda s: urllib.parse.quote(s, safe="")),
    91: ("csv_field", "serialize", "text", lambda s: '"' + s.replace('"', '""') + '"'),
    92: ("xml_text", "serialize", "text",
         lambda s: "<text>" + s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</text>"),
    93: ("utf8_hex", "encode", "text", lambda s: s.encode("utf-8").hex()),
    94: ("base64", "encode", "text", lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")),
    95: ("base32", "encode", "text", lambda s: base64.b32encode(s.encode("utf-8")).decode("ascii")),
    96: ("codepoints_json", "encode", "text", lambda s: json.dumps(["U+%04X" % ord(c) for c in s], separators=_C)),
    97: ("sha256", "hash", "text", lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()),
    98: ("crc32", "hash", "text", lambda s: "%08x" % (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)),
    99: ("rle_json", "encode", "text", rle),
}


def load_catalog() -> list[dict]:
    path = Path(__file__).with_name("goalgoal_catalog.json")
    return json.loads(path.read_text(encoding="utf-8"))["instructions"]


def register(reg: Registry) -> None:
    cat = load_catalog()
    assert len(cat) == 99 and sorted(IMPL) == list(range(1, 100))
    for e in cat:
        short, category, unit, fn = IMPL[e["id"]]
        text = e["instruction"]
        assert text.count("user message") == 1, e["name"]
        reg.add(Transform(name=f"{e['name']}_{short}", template=text.replace("user message", "{target}"),
                          fn=fn, category=category, unit=unit, source="goalgoal"))
