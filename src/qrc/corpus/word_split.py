"""Character-level split of a merged corpus phoneme-word unit into its
constituent real written words.

Some corpus phoneme-word units (see scripts/build_word_text_map.py) cover
multiple written words at once, because Tajweed liaison rules merge them
acoustically before the training data was segmented. This module finds,
within such a merged unit's own phoneme string (the corpus's native
convention -- what the ASR was actually trained on), the split points
corresponding to each real written word, using quran-transcript's
per-written-word phoneme rendering (a similar but not identical
convention -- different madd/muqatta'at handling) as a guide.

Approach: collapse tajweed-length rendering variation (same rule as
align/normalize.py's run-collapsing, reused here via _RUN_RE so both
stay in sync) on both sides, since duration differences (e.g. corpus
madd rendered as "اااا" vs qt's "اا") would otherwise dominate a raw
char-level edit distance and obscure the underlying phoneme identity.
Then character-level DP-align (align/dp.py, same machinery the aligner
itself uses at runtime) the collapsed corpus string against the
collapsed concatenation of the qt per-word groups. Split points come
from where each qt group's own boundary lands in that alignment, mapped
back through the collapse step to real offsets in the *original*
(uncollapsed) corpus string -- verification must slice the real corpus
phoneme_text, not a collapsed stand-in for it.
"""

import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from qrc.align.dp import DELETE_A, INSERT_A, MATCH, edit_distance_with_backpointers, traceback
from qrc.align.normalize import _RUN_RE


@dataclass
class SplitResult:
    # None if the alignment wasn't confident enough to trust -- callers
    # should fall back to treating the unit as indivisible rather than
    # use a low-confidence guess. confidence is always populated (even
    # on failure) so callers can report/tune on the actual score
    # distribution instead of guessing a threshold upfront.
    spans: list[str] | None
    confidence: float


def _collapse_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse runs of identical repeated characters (same rule as
    align/normalize.py's _collapse_runs), also returning -- for each
    resulting character -- its start offset in the original string, so
    a split point found on the collapsed string can be mapped back.
    """
    collapsed_chars: list[str] = []
    orig_offsets: list[int] = []
    i = 0
    n = len(s)
    while i < n:
        collapsed_chars.append(s[i])
        orig_offsets.append(i)
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        i = j
    return "".join(collapsed_chars), orig_offsets


def split_merged_unit(corpus_phonemes: str, qt_groups: list[str]) -> SplitResult:
    """Split corpus_phonemes (one merged corpus phoneme-word unit) into
    len(qt_groups) sub-strings of corpus_phonemes, one per qt_groups
    entry in order. Confidence gating is the caller's responsibility --
    this always returns the achieved alignment confidence.
    """
    collapsed_corpus, corpus_offsets = _collapse_with_map(corpus_phonemes)

    # Collapsing each qt group independently (not the joined string) so a
    # run of repeated chars can never be merged across a word boundary.
    collapsed_qt_groups = [_RUN_RE.sub(r"\1", g) for g in qt_groups]
    collapsed_qt = "".join(collapsed_qt_groups)

    if not collapsed_corpus or not collapsed_qt:
        return SplitResult(spans=None, confidence=0.0)

    confidence = Levenshtein.normalized_similarity(collapsed_corpus, collapsed_qt)

    boundaries_collapsed: list[int] = []
    cursor = 0
    for g in collapsed_qt_groups:
        cursor += len(g)
        boundaries_collapsed.append(cursor)

    dp, bp = edit_distance_with_backpointers(collapsed_corpus, collapsed_qt)
    path = traceback(bp, len(collapsed_corpus), len(collapsed_qt))

    # For each cursor position in collapsed_qt, how many collapsed_corpus
    # characters the alignment had consumed by the time that position was
    # reached -- a non-decreasing step function of the collapsed_qt
    # cursor, since both cursors only ever advance along the path.
    consumed_corpus_at_qt = [0] * (len(collapsed_qt) + 1)
    i_consumed = j_consumed = 0
    for _i_idx, _j_idx, op in path:
        if op == MATCH:
            i_consumed += 1
            j_consumed += 1
        elif op == INSERT_A:
            i_consumed += 1
        elif op == DELETE_A:
            j_consumed += 1
        consumed_corpus_at_qt[j_consumed] = i_consumed

    def to_original_offset(collapsed_offset: int) -> int:
        if collapsed_offset >= len(corpus_offsets):
            return len(corpus_phonemes)
        return corpus_offsets[collapsed_offset]

    split_points = [0] + [to_original_offset(consumed_corpus_at_qt[b]) for b in boundaries_collapsed]

    spans = [corpus_phonemes[split_points[k] : split_points[k + 1]] for k in range(len(qt_groups))]
    if any(not s for s in spans):
        return SplitResult(spans=None, confidence=confidence)

    return SplitResult(spans=spans, confidence=confidence)


def _last_letter(word: str) -> str | None:
    """The word's actual final letter, ignoring trailing harakat/shadda/
    sukun/tanween/tatweel -- the diacritics carried on or after it, not a
    letter of their own.
    """
    for ch in reversed(word):
        if ch == "ـ" or unicodedata.category(ch) == "Mn":  # tatweel or a combining mark
            continue
        return ch
    return None


def rebalance_shared_gemination(written_words: list[str], spans: list[str]) -> list[str]:
    """When a written word's final letter is identical to the letter that
    opens the *next* word's phoneme span as a repeated run (idgham
    mutamathilain -- two identical adjacent letters merging into one
    geminated sound, e.g. "لَّهُم مَّغْفِرَةٌ", meem meeting meem), the
    corpus's own phoneme rendering (and quran-transcript's word-boundary
    mapping, which the split above is aligned against) attributes the
    *entire* run to the second word -- leaving the first word with none
    of its own final letter at all, even though it's plainly there in the
    script. Give the first word back exactly one instance of that letter
    -- enough that its own expected phonemes aren't silently missing the
    sound it visibly ends in -- while the rest of the run (the genuine
    cross-word gemination/duration) stays with the second word, where the
    emphasis actually lands.

    written_words and spans must be the same length and order (spans
    from a single split_merged_unit call, written_words the real words
    that call's qt_groups came from).
    """
    spans = list(spans)
    for k in range(len(spans) - 1):
        letter = _last_letter(written_words[k])
        if letter is None:
            continue
        nxt = spans[k + 1]
        if len(nxt) >= 2 and nxt[0] == letter and nxt[1] == letter:
            spans[k] += letter
            spans[k + 1] = nxt[1:]
    return spans
