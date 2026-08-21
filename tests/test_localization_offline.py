from qrc.config import Settings
from qrc.corpus.models import AyahRef
from qrc.localize.ayah_locator import IncrementalAyahLocator


def _ayah_words(real_corpus, surah: int, ayah: int) -> list[str]:
    entry = next(a for a in real_corpus.ayahs_in_order if a.ref == AyahRef(surah, ayah))
    return list(entry.words)


def _first_global_idx(real_corpus, surah: int, ayah: int) -> int:
    return next(
        w.global_word_idx for w in real_corpus.global_words if w.surah == surah and w.ayah == ayah and w.local_word_idx == 0
    )


def _feed_chars_by_word(locator: IncrementalAyahLocator, words: list[str]):
    """Feed a raw (no-space) phoneme stream one word's worth of characters at
    a time, mirroring how the real ASR stream arrives -- no word boundaries,
    just a growing sequence of phoneme symbols."""
    result = None
    for w in words:
        result = locator.add_chars(w)
        if result is not None:
            break
    return result


def test_localizes_from_ayah_start(real_corpus):
    # Surah Al-Ikhlas (112:1) -- short and phonetically distinctive, unlike
    # the Basmalah (see test_ambiguous_repeated_phrase_defers_instead_of_guessing).
    words = _ayah_words(real_corpus, 112, 1)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings())

    result = _feed_chars_by_word(locator, words)

    assert result is not None
    expected_idx = _first_global_idx(real_corpus, 112, 1)
    assert result.global_word_idx == expected_idx


def test_ambiguous_repeated_phrase_defers_instead_of_guessing(real_corpus):
    # "بسم الله الرحمن الرحيم" (the Basmalah) is Al-Fatiha 1:1 verbatim, but
    # Surah An-Naml 27:30 also quotes it word-for-word mid-ayah (Sulayman's
    # letter). The phrase alone can't disambiguate which one is being
    # recited -- the locator must defer (return None) rather than guess,
    # per the margin-threshold design.
    words = _ayah_words(real_corpus, 1, 1)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings())

    result = _feed_chars_by_word(locator, words)
    assert result is None

    # The actual continuation (Al-Fatiha 1:2) resolves the ambiguity.
    next_word = _ayah_words(real_corpus, 1, 2)[0]
    result = locator.add_chars(next_word)
    assert result is not None
    assert result.global_word_idx == _first_global_idx(real_corpus, 1, 1)


def test_localizes_starting_mid_ayah(real_corpus):
    # Ayat al-Kursi (2:255) is long enough to start recitation mid-verse.
    words = _ayah_words(real_corpus, 2, 255)
    assert len(words) > 6

    start_local_idx = 3
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings())

    result = _feed_chars_by_word(locator, words[start_local_idx:])

    assert result is not None
    expected_entry = next(
        w for w in real_corpus.global_words if w.surah == 2 and w.ayah == 255 and w.local_word_idx == start_local_idx
    )
    assert result.global_word_idx == expected_entry.global_word_idx


def test_robust_to_one_word_asr_noise(real_corpus):
    words = _ayah_words(real_corpus, 112, 1)
    corrupted = list(words)
    # simulate an ASR substitution error on the 2nd buffered word: drop a char
    corrupted[1] = corrupted[1][:-1] if len(corrupted[1]) > 1 else corrupted[1]

    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings())
    result = _feed_chars_by_word(locator, corrupted)

    assert result is not None
    expected_idx = _first_global_idx(real_corpus, 112, 1)
    assert result.global_word_idx == expected_idx


def test_discards_buffer_after_max_chars_with_no_match(real_corpus):
    settings = Settings(min_trigger_chars=6, max_buffer_chars=20)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=settings)

    # Feed nonsense that shouldn't match anything in the Quran.
    garbage = ["زززز", "طططط", "جججج", "خخخخ", "ثثثث"]
    for w in garbage:
        locator.add_chars(w)

    assert len(locator.buffer) <= settings.max_buffer_chars
