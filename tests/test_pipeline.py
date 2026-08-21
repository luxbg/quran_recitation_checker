from qrc.asr.streaming_recognizer import PhonemeToken
from qrc.config import Settings
from qrc.corpus.models import AyahRef
from qrc.pipeline import RecitationChecker


def tokens_for(text: str) -> list[PhonemeToken]:
    return [PhonemeToken(symbol=ch, time_s=float(i)) for i, ch in enumerate(text)]


def _ayah_words(real_corpus, surah: int, ayah: int) -> list[str]:
    entry = next(a for a in real_corpus.ayahs_in_order if a.ref == AyahRef(surah, ayah))
    return list(entry.words)


def test_words_used_to_localize_are_not_reported_as_deleted(real_corpus):
    # Regression test: the words consumed while localizing must not be
    # silently dropped and reported as never-recited once localization
    # completes (real bug found from a live session transcript).
    words = _ayah_words(real_corpus, 112, 1)  # Al-Ikhlas: قُل هُوَ للَااهُ ءَحَدڇ
    settings = Settings()
    results = []

    checker = RecitationChecker(corpus=real_corpus, settings=settings, on_word_result=results.append)
    checker.feed_tokens(tokens_for("".join(words)))
    checker.finish()

    # The 4 recited words must all match with their full actual phonemes.
    # flush() also correctly reports the *un-recited* lookahead words from
    # 112:2 onward as "deleted" -- that's accurate, not part of this test.
    recited_results = [r for r in results if r.surah == 112 and r.ayah == 1]
    assert len(recited_results) == len(words)
    assert [r.status for r in recited_results] == ["match"] * len(words)
    assert [r.actual_phonemes for r in recited_results] == words
