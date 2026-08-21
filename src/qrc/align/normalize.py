import re

from rapidfuzz.distance import Levenshtein

_RUN_RE = re.compile(r"(.)\1+")
# Short-vowel diacritics dropped at a pause (waqf) -- fatha, damma, kasra.
_SHORT_VOWELS = "َُِ"


def _collapse_runs(s: str) -> str:
    """Collapse tajweed-length rendering variation: any run of the same
    repeated symbol (madd length, gemination, ghunna -- e.g. "ۥۥۥۥ" vs
    "ۥۥ", 4 vs 2 harakat, both legitimate per the model card's own
    documented limitations) becomes a single instance. Safe to apply
    independently to each string: a run of one symbol can never collapse
    to look like a run of a *different* symbol, so this can't mask a
    genuine phoneme substitution.
    """
    return _RUN_RE.sub(r"\1", s)


def _normalize_pair(expected: str, actual: str) -> tuple[str, str]:
    """Collapse tajweed-only rendering variation, judged pairwise (never
    independently -- see phonemes_match).
    """
    e = _collapse_runs(expected)
    a = _collapse_runs(actual)

    longer, shorter = (e, a) if len(e) >= len(a) else (a, e)
    if len(longer) == len(shorter) + 1 and longer[-1] in _SHORT_VOWELS and longer.startswith(shorter):
        return shorter, shorter

    return e, a


def phonemes_match(expected: str, actual: str) -> bool:
    """Whether two phoneme strings are the same phoneme content, ignoring
    tajweed-only rendering variation.

    Requires exact equality after normalization, not a fuzzy similarity
    threshold: a length-proportional similarity cutoff systematically
    under-penalizes a single wrong character in a long word (one wrong
    vowel in a 19-character word only drops similarity to ~0.95, clearing
    a 0.90 threshold) -- caught from a live session where a single
    mid-word vowel substitution wasn't flagged. Tajweed is out of scope,
    but a real phoneme substitution should always be caught regardless of
    the word's length.

    Tajweed is out of scope, but a *wrong* short vowel is a real
    phoneme-identity mistake, not a tajweed rendering choice -- so unlike
    the length collapsing above, the waqf (pause-dropped trailing vowel)
    case is judged by comparing both strings against each other, not
    normalized independently. Independently stripping a trailing vowel
    from both sides would make "raa with fatha" and "raa with damma"
    collapse to the same thing, silently erasing a real mispronunciation
    -- also caught from a live session (see tests/test_normalize.py).
    """
    e, a = _normalize_pair(expected, actual)
    return e == a


def phoneme_similarity(expected: str, actual: str) -> float:
    """Diagnostic similarity score (not the match/mismatch gate -- see
    phonemes_match) after the same tajweed normalization.
    """
    e, a = _normalize_pair(expected, actual)
    return Levenshtein.normalized_similarity(e, a)
