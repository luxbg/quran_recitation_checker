from dataclasses import dataclass


@dataclass(frozen=True)
class AyahRef:
    surah: int
    ayah: int


@dataclass(frozen=True)
class AyahEntry:
    ref: AyahRef
    aya_text: str
    aya_phoneme: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class GlobalWordEntry:
    global_word_idx: int
    surah: int
    ayah: int
    local_word_idx: int
    phoneme_text: str
    # Real Uthmani script word(s), from scripts/build_word_text_map.py --
    # None if that precompute step wasn't run or couldn't confidently map
    # this word (see that script's docstring for why the mapping isn't 1:1).
    word_text: str | None = None
    # This word's own phoneme rendering when phonetized standalone (as if
    # paused/waqf right after it) -- differs from phoneme_text wherever
    # cross-word tajweed liaison (idgham/ikhfa/iqlab) or a pause-dropped
    # tanween/short-vowel affects the corpus's connected-recitation form.
    # From scripts/build_word_text_map.py; None if not precomputed. See
    # align/word_aligner.py's use of it for why this needs no context from
    # neighboring words -- it's the word's own pronunciation in isolation.
    isolated_phoneme_text: str | None = None
    # True when this corpus phoneme-word is part of the *same* real
    # written word as the immediately preceding global_word_idx (a
    # muqatta'at split, e.g. "الٓمٓ" spans two phoneme-word units) --
    # consumers must treat a continues_previous run as one logical word.
    word_text_continues_previous: bool = False
