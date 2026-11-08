"""How much of an OCR text is real words: the same measure for both sides' text of a component.

A word is a run of three or more letters (shorter runs, "a", "of", "ss", can't be told from OCR
scraps: two-letter scraps are common abbreviations in any word list). It is real when it is in
wordfreq's English list in any case, or becomes so when some of its f's are read as s: type before
about 1800 prints the long s (ſ), which OCR reads as f ("neceffity"), and that is the period's
spelling, not an OCR error.
"""
from __future__ import annotations

import itertools
import re
from functools import lru_cache

from wordfreq import zipf_frequency

WORD = re.compile(r"[A-Za-z]{3,}")
LONG_S_MAX = 4  # words with more f's than this are tried as they are only


def words(text: str) -> list[str]:
    return WORD.findall(text)


def _listed(word: str) -> bool:
    return zipf_frequency(word, "en") > 0


@lru_cache(maxsize=65536)
def is_real_word(word: str) -> bool:
    word = word.lower()
    if _listed(word):
        return True
    fs = [k for k, c in enumerate(word) if c == "f"]
    if not fs or len(fs) > LONG_S_MAX:
        return False
    for n in range(1, len(fs) + 1):
        for chosen in itertools.combinations(fs, n):
            letters = list(word)
            for k in chosen:
                letters[k] = "s"
            if _listed("".join(letters)):
                return True
    return False


def real_word_rate(text: str) -> float | None:
    """The share of the text's words that are real, or None when it has none to judge."""
    found = words(text)
    return sum(map(is_real_word, found)) / len(found) if found else None


def word_counts(text: str) -> tuple[int, int]:
    """(real words, words) in the text, for summing over many texts."""
    found = words(text)
    return sum(map(is_real_word, found)), len(found)
