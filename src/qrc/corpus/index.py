import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

from qrc.corpus.loader import load_ayahs
from qrc.corpus.models import AyahEntry, GlobalWordEntry


@dataclass
class QuranCorpus:
    ayahs_in_order: list[AyahEntry]
    global_words: list[GlobalWordEntry]
    # Concatenation of every global word's phoneme text, NO separator --
    # the model's alphabet has no space/word-boundary token, so the raw
    # ASR stream is never space-delimited either. Keeping this unseparated
    # keeps char-level fuzzy matching (locator) and DP alignment (aligner)
    # consistent with what the ASR actually emits.
    corpus_text: str
    # char_offsets[i] = start char offset of global_words[i] in corpus_text.
    char_offsets: list[int] = field(default_factory=list)

    def word_at(self, global_word_idx: int) -> GlobalWordEntry | None:
        if 0 <= global_word_idx < len(self.global_words):
            return self.global_words[global_word_idx]
        return None

    def global_word_idx_for_char_offset(self, char_offset: int) -> int:
        """Map a char offset in corpus_text back to the nearest global_word_idx."""
        i = bisect.bisect_right(self.char_offsets, char_offset) - 1
        return max(0, min(i, len(self.global_words) - 1))


def _load_word_text_map(ordered_quran_phonemes_path: Path | str) -> list[dict | None] | None:
    map_path = Path(ordered_quran_phonemes_path).parent / "word_text_map.json"
    if not map_path.exists():
        return None
    with open(map_path, encoding="utf-8") as f:
        return json.load(f)


def build_corpus(ordered_quran_phonemes_path: Path | str) -> QuranCorpus:
    ayahs = load_ayahs(ordered_quran_phonemes_path)
    word_text_map = _load_word_text_map(ordered_quran_phonemes_path)

    global_words: list[GlobalWordEntry] = []
    corpus_parts: list[str] = []
    char_offsets: list[int] = []
    cursor = 0

    # scripts/build_word_text_map.py indexes its output by *original*
    # corpus phoneme-word position (one slot per ayah.words entry, counted
    # in the same ayahs-in-order / within-ayah traversal as here) -- not
    # by global_word_idx, which below is reassigned fresh per real written
    # word and shifts as soon as any earlier ayah in the Quran contains a
    # split. Track that original position with its own counter rather
    # than re-deriving it per word (which would be O(n) per lookup).
    original_gwi = 0

    for ayah in ayahs:
        local_idx = 0
        for word_phonemes in ayah.words:
            entry = word_text_map[original_gwi] if word_text_map is not None else None
            original_gwi += 1

            # Tajweed liaison merges some written words into one phoneme
            # unit (see build_word_text_map.py's module docstring); where
            # that offline precompute confidently split it back into its
            # real per-word phoneme sub-spans ("words" has more than one
            # entry), explode it into that many GlobalWordEntry rows here
            # instead of one -- so the aligner verifies each real written
            # word independently. Where it didn't split (single-element
            # "words", or no map at all), this is exactly today's
            # behavior: one row, using the corpus's own merged phoneme
            # text.
            sub_words = entry["words"] if entry is not None else [{"text": None, "phoneme_text": word_phonemes}]
            continues_previous = entry["continues_previous"] if entry is not None else False

            for sub_idx, sub_word in enumerate(sub_words):
                gwi = len(global_words)
                phoneme_text = sub_word["phoneme_text"]
                global_words.append(
                    GlobalWordEntry(
                        global_word_idx=gwi,
                        surah=ayah.ref.surah,
                        ayah=ayah.ref.ayah,
                        local_word_idx=local_idx,
                        phoneme_text=phoneme_text,
                        word_text=sub_word["text"],
                        # Only the *first* row of a split unit carries the
                        # original entry's own continues_previous (a
                        # muqatta'at split, unrelated to this merge-split);
                        # every row after the first, within the same
                        # original unit, is a fresh real word, not a
                        # continuation of the row before it.
                        word_text_continues_previous=continues_previous if sub_idx == 0 else False,
                    )
                )
                char_offsets.append(cursor)
                corpus_parts.append(phoneme_text)
                cursor += len(phoneme_text)
                local_idx += 1

    corpus_text = "".join(corpus_parts)

    return QuranCorpus(
        ayahs_in_order=ayahs,
        global_words=global_words,
        corpus_text=corpus_text,
        char_offsets=char_offsets,
    )
