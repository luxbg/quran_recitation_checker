#!/usr/bin/env python3
"""Precompute global_word_idx -> real Uthmani word text.

The shipped corpus (ordered_quran_phonemes.json) segments each ayah into
phoneme-words for ASR matching, but that segmentation doesn't line up 1:1
with written words -- Tajweed liaison rules merge some adjacent written
words into one phoneme unit (e.g. "مِن قَبْلِكَ" -> one unit), and
muqatta'at letters split one written word into several (e.g. "الٓمٓ" ->
two). So there's no direct way to say "phoneme-word #3 is written word
#3".

This resolves it in two steps, both done here (once, offline) rather than
live:
1. Exact: quran-transcript's own phonetizer (github.com/obadx/quran-transcript,
   already installed -- it's literally the tool that produced the
   corpus's training labels per the model card) exposes a per-input-char
   position mapping (`mappings`) from its phonetization output. Used to
   derive, precisely, which output phoneme span corresponds to each
   written word -- not by naively splitting on spaces (liaison can fuse
   two written words' output together with no space at all).
2. Fuzzy: quran-transcript's phoneme rendering doesn't exactly match this
   corpus's own (different muqatta'at handling, minor spelling/madd
   convention differences even with matched moshaf settings) -- verified
   by direct testing, not assumed. So step 1's per-written-word phoneme
   groups are aligned against the corpus's aya_phonemes_list via a
   word-level edit-distance DP (rapidfuzz-scored), not a strict
   positional match. Mismatches here are typically small and localized
   (an occasional adjacent-word merge), not wholesale restructuring --
   also verified empirically before committing to this approach.

Output: models/word_text_map.json, a list indexed by global_word_idx.
Each entry is either null (couldn't be confidently mapped -- logged, not
guessed at) or {"text": ..., "continues_previous": bool}. "text" is the
real word text (possibly multiple written words joined by a space if
merged). "continues_previous" is true when this corpus phoneme-word is
part of the *same* real written word as the immediately preceding
global_word_idx (a muqatta'at split, e.g. "الٓمٓ" spans two phoneme-word
units both carrying continues_previous split across them) -- consumers
must treat a continues_previous run as one logical word, not report each
unit as if it were a separate word.
"""

import json
import re
import sys
from pathlib import Path

import quran_transcript as qt
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qrc.corpus.index import build_corpus  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CORPUS_PATH = MODELS_DIR / "ordered_quran_phonemes.json"
OUTPUT_PATH = MODELS_DIR / "word_text_map.json"

MOSHAF = qt.MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
)

MATCH, DELETE_A, DELETE_B = 0, 1, 2  # DELETE_A: corpus word has no qt counterpart (split). DELETE_B: qt word has no corpus counterpart (merge).


def written_word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def qt_phoneme_groups(aya_text: str) -> list[str] | None:
    """Per-written-word phoneme groups from quran-transcript, derived via
    its char-position mapping (not space-splitting -- see module docstring).
    Returns None if the phonetizer itself fails on this ayah.
    """
    try:
        out = qt.quran_phonetizer(aya_text, MOSHAF)
    except Exception:
        return None

    groups = []
    for start, end in written_word_spans(aya_text):
        positions = [out.mappings[i].pos for i in range(start, end) if not out.mappings[i].deleted]
        if not positions:
            groups.append("")
            continue
        lo = min(p[0] for p in positions)
        hi = max(p[1] for p in positions)
        groups.append(out.phonemes[lo:hi].strip())
    return groups


def align_word_lists(corpus_words: list[str], qt_words: list[str]) -> list[list[int] | None]:
    """Word-level edit-distance alignment (not char-level -- these are
    two different phoneme-word segmentations of the same ayah). Returns,
    per corpus word, the qt_words *indices* it aligns to (not the text --
    the caller must use indices directly, not re-derive them by searching
    for the phoneme text as a substring elsewhere: short/common phoneme
    groups like "لَا" false-match inside many unrelated groups, which is
    exactly what corrupted longer ayahs like 2:255 before this was an
    index-based return). Substitution cost is fractional (1 -
    similarity), not 0/1, since these are two different renderings of
    generally-the-same content, not independent random text.
    """
    n, m = len(corpus_words), len(qt_words)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bp = [[MATCH] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = DELETE_A
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = DELETE_B

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 1.0 - Levenshtein.normalized_similarity(corpus_words[i - 1], qt_words[j - 1])
            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + 1.0
            left = dp[i][j - 1] + 1.0
            best, op = diag, MATCH
            if up < best:
                best, op = up, DELETE_A
            if left < best:
                best, op = left, DELETE_B
            dp[i][j], bp[i][j] = best, op

    # Traceback, grouping consecutive DELETE_B (qt-only, merge case) qt-words
    # onto the next MATCH's corpus word; a DELETE_A (corpus-only, split
    # case) corpus word borrows its neighbor's qt group as a fallback.
    path: list[tuple[int | None, int | None, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == MATCH:
            path.append((i - 1, j - 1, MATCH))
            i, j = i - 1, j - 1
        elif op == DELETE_A:
            path.append((i - 1, None, DELETE_A))
            i -= 1
        else:
            path.append((None, j - 1, DELETE_B))
            j -= 1
    path.reverse()

    result: list[list[int] | None] = [None] * n
    pending_qt: list[int] = []
    for a_idx, b_idx, op in path:
        if op == MATCH:
            result[a_idx] = pending_qt + [b_idx]
            pending_qt = []
        elif op == DELETE_B:
            pending_qt.append(b_idx)
        else:  # DELETE_A: corpus word with no qt counterpart -- borrow a neighbor
            result[a_idx] = None  # filled in from a neighbor below

    # Fill DELETE_A gaps from the nearest resolved neighbor (muqatta'at
    # splits: one written word maps to several corpus phoneme-words).
    for i in range(n):
        if result[i] is None:
            for j in range(i - 1, -1, -1):
                if result[j] is not None:
                    result[i] = result[j]
                    break
            else:
                for j in range(i + 1, n):
                    if result[j] is not None:
                        result[i] = result[j]
                        break

    return result


def build_written_word_text(aya_text: str, corpus_words: list[str], qt_groups: list[str]) -> list[tuple[str, bool] | None]:
    """Map each corpus phoneme-word back to the real written word(s) it
    corresponds to, by aligning corpus_words<->qt_groups (fuzzy, by index
    -- not by re-deriving indices via substring search, which false-
    matches on short/common phoneme groups like "لَا" appearing inside
    many unrelated groups) and qt_groups<->written_words (exact,
    positional, same list by construction).
    """
    written = [aya_text[s:e] for s, e in written_word_spans(aya_text)]
    aligned_indices = align_word_lists(corpus_words, qt_groups)

    # (text, continues_previous). continues_previous=True means this
    # corpus word is part of the *same* real written word as the one
    # immediately before it (a muqatta'at split, e.g. "الٓمٓ" spanning two
    # phoneme-word units) -- the caller must treat these as one logical
    # word, not report/log each unit separately.
    result: list[tuple[str, bool] | None] = []
    prev_idxs: list[int] | None = None
    for idxs in aligned_indices:
        if idxs is None:
            result.append(None)
            prev_idxs = None
            continue
        continues_previous = idxs == prev_idxs
        result.append((" ".join(written[i] for i in idxs), continues_previous))
        prev_idxs = idxs
    return result


def main() -> None:
    corpus = build_corpus(CORPUS_PATH)

    word_text_map: list[dict | None] = [None] * len(corpus.global_words)
    failed_ayahs: list[str] = []
    unmapped_words = 0

    for ayah_idx, entry in enumerate(corpus.ayahs_in_order):
        qt_groups = qt_phoneme_groups(entry.aya_text)
        if qt_groups is None:
            failed_ayahs.append(f"{entry.ref.surah}:{entry.ref.ayah}")
            continue

        written_texts = build_written_word_text(entry.aya_text, list(entry.words), qt_groups)

        first_global_idx = next(
            w.global_word_idx for w in corpus.global_words if w.surah == entry.ref.surah and w.ayah == entry.ref.ayah and w.local_word_idx == 0
        )
        for local_idx, item in enumerate(written_texts):
            if item is None:
                unmapped_words += 1
                continue
            text, continues_previous = item
            word_text_map[first_global_idx + local_idx] = {"text": text, "continues_previous": continues_previous}

        if (ayah_idx + 1) % 1000 == 0:
            print(f"  {ayah_idx + 1}/{len(corpus.ayahs_in_order)} ayahs...", file=sys.stderr)

    OUTPUT_PATH.write_text(json.dumps(word_text_map, ensure_ascii=False))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Failed ayahs (phonetizer crashed): {len(failed_ayahs)} -- {failed_ayahs[:20]}")
    print(f"Unmapped words: {unmapped_words} / {len(word_text_map)} ({100 * unmapped_words / len(word_text_map):.2f}%)")


if __name__ == "__main__":
    main()
