from qrc.align.word_aligner import IncrementalWordAligner
from qrc.asr.streaming_recognizer import PhonemeToken
from qrc.config import Settings
from qrc.corpus.models import GlobalWordEntry


class FakeCorpus:
    """Minimal stand-in exposing only what IncrementalWordAligner needs."""

    def __init__(self, words: list[str]):
        self.entries = [
            GlobalWordEntry(global_word_idx=i, surah=1, ayah=1, local_word_idx=i, phoneme_text=w)
            for i, w in enumerate(words)
        ]

    def word_at(self, idx: int) -> GlobalWordEntry | None:
        if 0 <= idx < len(self.entries):
            return self.entries[idx]
        return None


def tokens_for(text: str) -> list[PhonemeToken]:
    return [PhonemeToken(symbol=ch, time_s=float(i)) for i, ch in enumerate(text)]


def make_aligner(words: list[str], results: list, settings: Settings | None = None) -> IncrementalWordAligner:
    corpus = FakeCorpus(words)
    settings = settings or Settings()
    settings.settle_lookahead_chars = 3  # small buffers in tests -> small margin
    aligner = IncrementalWordAligner(corpus=corpus, settings=settings, on_word_result=results.append)
    aligner.localize(0)
    return aligner


def test_clean_match():
    words = ["بسم", "الله", "الرحمن", "الرحيم", "قل", "هو"]
    results = []
    aligner = make_aligner(words, results)

    aligner.feed_tokens(tokens_for("".join(words)))
    aligner.flush()  # end of session: force-settle any still-pending tail words

    assert [r.status for r in results] == ["match"] * len(results)
    assert [r.expected_phonemes for r in results] == words


def test_substituted_word_flagged_mismatch():
    words = ["بسم", "الله", "الرحمن", "الرحيم", "قل", "هو"]
    results = []
    aligner = make_aligner(words, results)

    corrupted = list(words)
    corrupted[1] = "الشمس"  # wrong word recited instead of "الله"
    aligner.feed_tokens(tokens_for("".join(corrupted)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[0].status == "match"
    assert by_idx[1].status == "mismatch"
    assert by_idx[1].expected_phonemes == "الله"


def test_deleted_word_detected():
    words = ["بسم", "الله", "الرحمن", "الرحيم", "قل", "هو"]
    results = []
    aligner = make_aligner(words, results)

    corrupted = [words[0]] + words[2:]  # skip "الله" entirely
    aligner.feed_tokens(tokens_for("".join(corrupted)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[0].status == "match"
    assert by_idx[1].status == "deleted"
    assert by_idx[1].actual_phonemes is None
    # alignment re-syncs: subsequent words still recognized correctly
    assert by_idx[2].status == "match"


def test_resyncs_after_one_bad_word():
    words = ["بسم", "الله", "الرحمن", "الرحيم", "قل", "هو", "الله", "احد"]
    results = []
    aligner = make_aligner(words, results)

    corrupted = list(words)
    corrupted[2] = "شيء"  # one bad word in the middle
    aligner.feed_tokens(tokens_for("".join(corrupted)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[0].status == "match"
    assert by_idx[1].status == "match"
    assert by_idx[2].status == "mismatch"
    # everything after the bad word re-syncs correctly, no cascading failure
    for i in range(3, min(6, len(results))):
        assert by_idx[i].status == "match", f"word {i} should have re-synced"


def test_tajweed_madd_and_waqf_variation_not_flagged_as_mismatch():
    # Real cases pulled from a live session (Surah Ar-Rum 30:3-30:4): the
    # reciter's madd length / pause behavior differs from the reference's
    # canonical rendering, but it's the same phoneme identity -- tajweed is
    # out of scope, so these must normalize to "match", not "mismatch".
    words = ["سَيَغلِبُۥۥۥۥن", "فِۦۦ", "بِضعِ", "سِنِۦۦنَ"]
    recited = ["سَيَغلِبُۥۥن", "فِۦۦ", "بِضعِ", "سِنِۦۦن"]
    results = []
    aligner = make_aligner(words, results)

    aligner.feed_tokens(tokens_for("".join(recited)))
    aligner.flush()

    assert [r.status for r in results] == ["match"] * len(words)


def test_wrong_vowel_still_flagged_despite_tajweed_normalization():
    # Regression: reciting a word's final consonant with the wrong short
    # vowel (damma instead of fatha) is a genuine phoneme mistake and must
    # still be flagged -- it must not be forgiven just because it happens
    # to land on the trailing-vowel-at-a-pause normalization.
    words = ["بسم", "الله", "ءَكثَرَ", "الرحيم"]
    results = []
    aligner = make_aligner(words, results)

    corrupted = list(words)
    corrupted[2] = "ءَكثَرُ"  # wrong vowel: damma instead of fatha
    aligner.feed_tokens(tokens_for("".join(corrupted)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[2].status == "mismatch"


def test_single_mid_word_vowel_error_flagged_regardless_of_word_length():
    # Regression: a long word absorbing one wrong character used to still
    # clear the old proportional similarity threshold. Exact-match-after-
    # normalization must catch it regardless of word length.
    words = ["بسم", "الله", "غِشَااوَتُوووَلَهُم", "الرحيم"]
    results = []
    aligner = make_aligner(words, results)

    corrupted = list(words)
    corrupted[2] = "غِشَااوَتَوووَلَهُم"  # one wrong mid-word vowel
    aligner.feed_tokens(tokens_for("".join(corrupted)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[2].status == "mismatch"


def test_repeated_word_does_not_contaminate_next_word():
    # Real case from a live session (2:174), settle_lookahead_chars=0 (the
    # actual runtime setting): reciter said "وَيَشتَرُۥۥنَ", it settled
    # correctly on its own, then a trailing repeat of its ending arrived
    # before "بِهِۦۦ" -- the repeat's characters used to bleed into the
    # next word's actual_phonemes ("ۥۥنَبِهِۦۦ" instead of "بِهِۦۦ"),
    # causing a false mismatch on a word that was actually recited
    # correctly. Two feed_tokens calls, not one, to mirror how the repeat
    # arrives only *after* the word ahead of it has already settled.
    words = ["بسم", "الله", "وَيَشتَرُۥۥنَ", "بِهِۦۦ", "الرحيم"]
    results = []
    corpus = FakeCorpus(words)
    settings = Settings(settle_lookahead_chars=0)
    aligner = IncrementalWordAligner(corpus=corpus, settings=settings, on_word_result=results.append)
    aligner.localize(0)

    aligner.feed_tokens(tokens_for("".join(words[:3])))
    assert results[-1].word_index == 2
    assert results[-1].status == "match"

    trailing_repeat = "ۥۥنَ"  # a stutter/repeat of word 2's ending, not the whole word
    aligner.feed_tokens(tokens_for(trailing_repeat + words[3] + words[4]))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[3].status == "match"
    assert by_idx[3].actual_phonemes == "بِهِۦۦ"
    assert by_idx[4].status == "match"


def test_coincidental_word_boundary_overlap_not_treated_as_a_repeat():
    # Regression: the fix above was too aggressive. Real case from a live
    # session (32:5): "ٱلسَّمَآءِ" (actual 'سسَمَااااءِ') ends in "ءِ" and
    # the next word "إِلَى" (expected 'ءِلَ') starts with "ءِ" -- a
    # coincidental phonetic overlap (hamza+kasra is common at Arabic word
    # boundaries), not a repeat. Both words were recited correctly, but
    # stripping unconditionally cut the genuine leading "ءِ" off the
    # second word, turning a perfect match into a false mismatch
    # ('لَ' instead of 'ءِلَ'). Stripping must never fire on content that
    # already matches.
    words = ["مِنَ", "سسَمَااااءِ", "ءِلَ", "لءَرضِ"]
    results = []
    corpus = FakeCorpus(words)
    settings = Settings(settle_lookahead_chars=0)
    aligner = IncrementalWordAligner(corpus=corpus, settings=settings, on_word_result=results.append)
    aligner.localize(0)

    aligner.feed_tokens(tokens_for("".join(words)))
    aligner.flush()

    by_idx = {r.word_index: r for r in results}
    assert by_idx[2].status == "match"
    assert by_idx[2].actual_phonemes == "ءِلَ"


def test_flush_settles_trailing_skipped_word():
    words = ["بسم", "الله", "الرحمن", "الرحيم"]
    results = []
    aligner = make_aligner(words, results)

    # reciter stops after the second word -- never says the last two at all
    aligner.feed_tokens(tokens_for("".join(words[:2])))
    aligner.flush()

    assert len(results) == len(words)
    by_idx = {r.word_index: r for r in results}
    assert by_idx[0].status == "match"
    assert by_idx[1].status == "match"
    assert by_idx[2].status == "deleted"
    assert by_idx[2].actual_phonemes is None
    assert by_idx[3].status == "deleted"
    assert by_idx[3].actual_phonemes is None
