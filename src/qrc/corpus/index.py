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

    for ayah in ayahs:
        for local_idx, word_phonemes in enumerate(ayah.words):
            gwi = len(global_words)
            entry = word_text_map[gwi] if word_text_map is not None else None
            global_words.append(
                GlobalWordEntry(
                    global_word_idx=gwi,
                    surah=ayah.ref.surah,
                    ayah=ayah.ref.ayah,
                    local_word_idx=local_idx,
                    phoneme_text=word_phonemes,
                    word_text=entry["text"] if entry is not None else None,
                    word_text_continues_previous=entry["continues_previous"] if entry is not None else False,
                )
            )

            char_offsets.append(cursor)
            corpus_parts.append(word_phonemes)
            cursor += len(word_phonemes)

    corpus_text = "".join(corpus_parts)

    return QuranCorpus(
        ayahs_in_order=ayahs,
        global_words=global_words,
        corpus_text=corpus_text,
        char_offsets=char_offsets,
    )
