import random

import pytest

from pno.catalog import build_registry
from pno.catalog.base import sensitivity


@pytest.fixture(scope="module")
def reg():
    return build_registry()


def test_sizes(reg):
    assert len(reg.names(source="goalgoal")) == 99
    assert len(reg.names(source="extra")) >= 10
    for t in reg:
        assert "{target}" in t.template
        for target in ("input", "user", "system"):
            ins = t.instruction(target)
            assert "{" not in ins and ins[0].isupper()


def test_wording_matches_catalog(reg):
    t = reg["T02_reverse_chars"]
    assert t.instruction("user") == "Reverse the order of all Unicode code points in the user message."
    assert t.instruction("system") == "Reverse the order of all Unicode code points in the system message."
    assert t.instruction("input") == "Reverse the order of all Unicode code points in the input."


def test_reproduces_paper_pair_filtering(reg):
    """[8] builds every ordered pair of distinct instructions with each message equal to its
    catalog sentence, then drops pairs with an empty response (156) or identical responses (2),
    keeping 9,544. Our implementations must reproduce these counts."""
    gg = [reg[n] for n in reg.names(source="goalgoal")]
    empty = ident = kept = 0
    kept_pairs = set()
    for a in gg:
        for b in gg:
            if a is b:
                continue
            y_pos = a.apply(b.instruction("system"))   # system task applied to the user message
            y_neg = b.apply(a.instruction("user"))     # user task applied to the system message
            if y_pos == "" or y_neg == "":
                empty += 1
            elif y_pos == y_neg:
                ident += 1
            else:
                kept += 1
                kept_pairs.add((a.name, b.name))
    assert (empty, ident, kept) == (156, 2, 9544)
    assert all((b, a) in kept_pairs for a, b in kept_pairs)


def test_paper_example(reg):
    # Introduction example of [8], with the catalog's exact wording (T92 over T05)
    sys_t, usr_t = reg["T92_xml_text"], reg["T05_swapcase"]
    user_msg = usr_t.instruction("system")
    assert sys_t.apply(user_msg) == f"<text>{user_msg}</text>"
    assert usr_t.apply("Encode the user message.") == "eNCODE THE USER MESSAGE."


@pytest.mark.parametrize("name,inp,out", [
    ("T03_upper", "Hello, World.", "HELLO, WORLD."),
    ("T16_delete_vowels", "Hello, World.", "Hll, Wrld."),
    ("T24_space_to_underscore", "a b  c", "a_b__c"),
    ("T29_rotate_left", "abcd", "bcda"),
    ("T30_rotate_right", "abcd", "dabc"),
    ("T33_star_even_chars", "abcde", "a*c*e"),
    ("T36_rot13", "Hello", "Uryyb"),
    ("T37_atbash", "Abc xyz", "Zyx cba"),
    ("T40_affine_5x8", "ab", "in"),
    ("T43_alternating_case", "hello world", "HeLlO wOrLd"),
    ("T44_five_bit_tokens", "a1!", "00001 D1 U+0021"),
    ("T47_runs_lower", "It's 9 o'Clock", "it s o clock"),
    ("T53_acronym", "portable network graphics", "PNG"),
    ("T59_snake_case", "Hello big World", "hello_big_world"),
    ("T63_pig_latin", "apple string rhythm", "appleway ingstray rhythmay"),
    ("T64_drop_stopwords", "The cat and the hat", "cat hat"),
    ("T65_mask_instruction", "Instruction: no instructions", "[INSTRUCTION]: no instructions"),
    ("T66_bracket_runs", "a-b c", "[a]-[b] [c]"),
    ("T72_number_lines", "x\ny", "1:x\n2:y"),
    ("T73_line_lengths", "ab\nc", "2:ab\n1:c"),
    ("T75_chunk_20", "a" * 45, "a" * 20 + "\n" + "a" * 20 + "\n" + "a" * 5),
    ("T76_sort_comma_fields", "b , a,c", "a,b,c"),
    ("T78_count_chars", "héllo", "5"),
    ("T79_count_bytes", "héllo", "6"),
    ("T82_category_counts_json", "a1 .", '{"L":1,"M":0,"N":1,"P":1,"S":0,"Z":1,"C":0}'),
    ("T87_even_length", "ab", "true"),
    ("T88_json_string", 'a"b', '"a\\"b"'),
    ("T91_csv_field", 'a"b', '"a""b"'),
    ("T94_base64", "hi", "aGk="),
    ("T96_codepoints_json", "a ", '["U+0061","U+0020"]'),
    ("T98_crc32", "hello", "3610a686"),
    ("T99_rle_json", "aab", '[["a",2],["b",1]]'),
    ("number_sentences", "One is here. Two is there.", "1. One is here.\n2. Two is there."),
    ("sort_lines", "pear\nApple\nfig", "Apple\nfig\npear"),
    ("count_sentences", "A b. C d! E f?", "3"),
])
def test_reference_outputs(reg, name, inp, out):
    assert reg[name].apply(inp) == out


def test_deterministic_and_sensitive(reg):
    rng = random.Random(0)
    words = "the quick brown fox jumps over a lazy dog near river bank today".split()
    sents = [" ".join(rng.sample(words, 6)).capitalize() + "." for _ in range(40)]
    insensitive = []
    for t in reg:
        text = "\n".join(sents[:3]) if t.unit == "lines" else " ".join(sents[:2])
        assert t.apply(text) == t.apply(text)
        carriers = ["\n".join(rng.sample(sents, 3)) if t.unit == "lines" else " ".join(rng.sample(sents, 2))
                    for _ in range(30)]
        if sensitivity(t, carriers) < 0.95:
            insensitive.append(t.name)
    # transforms whose output does not depend on every sentence must be caught by the filter
    for must in ("T19_keep_decimals", "T67_first_run", "T87_even_length"):
        assert must in insensitive
    for must_not in ("T01_copy", "T03_upper", "T02_reverse_chars", "T88_json_string", "number_sentences"):
        assert must_not not in insensitive
