from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


@dataclass
class Settings:
    sample_rate: int = 16000

    tokens_path: Path = MODELS_DIR / "tokens.txt"
    model_path: Path = MODELS_DIR / "zipformer_p_arabic_v3.int8.onnx"
    corpus_path: Path = MODELS_DIR / "ordered_quran_phonemes.json"
    provider: str = "cpu"

    # Localization (character-level: the model has no word-boundary token,
    # so the raw ASR stream can't be pre-segmented into words before a
    # location is known -- see IncrementalAyahLocator).
    min_trigger_chars: int = 12
    max_buffer_chars: int = 60
    confidence_threshold: float = 0.72
    margin_threshold: float = 0.08
    # max_l_dist for fuzzysearch, proportional to buffer length but capped --
    # benchmarked on the real corpus: capped at 8 keeps worst-case search
    # under ~30ms; uncapped tolerance on a long buffer measured over 1s.
    max_l_dist_ratio: float = 0.15
    max_l_dist_cap: int = 8

    # Alignment (Option B: character-level DP, no ASR word boundaries). A
    # match requires exact phoneme equality after tajweed normalization,
    # not a fuzzy similarity threshold -- see align/normalize.py for why.
    settle_lookahead_chars: int = 0  # how many extra actual chars must arrive
    # past a word boundary before we consider it "settled" and emit a verdict.

    # Relocalization: recent-character window used to reseed the locator
    # when the aligner's rolling confidence collapses.
    relocalize_seed_chars: int = 30
    relocalize_ema_alpha: float = 0.3
    relocalize_ema_threshold: float = 0.5

    # Backtracking: a reciter may restart from anywhere within the last N
    # pages of the furthest ayah actually reached, never earlier (and
    # never before the session's true start). Keeping the window this
    # narrow is also what makes the per-word backtrack-or-mistake probe in
    # RecitationChecker cheap enough to run on every non-matching word.
    # "Pages" are estimated from the corpus's average words/page (real
    # Mushaf pagination isn't in the corpus data), not exact boundaries.
    backtrack_window_pages: int = 2
    mushaf_total_pages: int = 604
    # How many words apart a candidate match must be from where the
    # aligner currently is before it's treated as a genuine backtrack
    # rather than an ordinary mismatch/ASR noise near the same spot.
    backtrack_min_word_gap: int = 5

    correct_words_log_path: Path = LOGS_DIR / "correct_words.jsonl"


DEFAULT_SETTINGS = Settings()
