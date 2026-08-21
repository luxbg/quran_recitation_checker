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
    # True when this corpus phoneme-word is part of the *same* real
    # written word as the immediately preceding global_word_idx (a
    # muqatta'at split, e.g. "الٓمٓ" spans two phoneme-word units) --
    # consumers must treat a continues_previous run as one logical word.
    word_text_continues_previous: bool = False
