from qrc.asr.streaming_recognizer import PhonemeToken
from qrc.config import Settings
from qrc.corpus.models import AyahRef
from qrc.localize.ayah_locator import IncrementalAyahLocator, RejectionReason
from qrc.pipeline import RecitationChecker


def _ayah_words(real_corpus, surah: int, ayah: int) -> list[str]:
    entry = next(a for a in real_corpus.ayahs_in_order if a.ref == AyahRef(surah, ayah))
    return list(entry.words)


def _first_global_idx(real_corpus, surah: int, ayah: int) -> int:
    return next(
        w.global_word_idx for w in real_corpus.global_words if w.surah == surah and w.ayah == ayah and w.local_word_idx == 0
    )


def tokens_for(text: str) -> list[PhonemeToken]:
    return [PhonemeToken(symbol=ch, time_s=0.0) for ch in text]


# Al-Ikhlas (112): 4 short, phonetically distinctive ayat -- good for
# exercising "started at ayah N, may return to N, may not go before N,
# may not skip ahead of the furthest ayah actually reached".


def test_locator_rejects_match_before_floor(real_corpus):
    floor_idx = _first_global_idx(real_corpus, 112, 2)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings(), min_global_word_idx=floor_idx)

    words = _ayah_words(real_corpus, 112, 1)  # ayah 1 -- before the floor
    result = None
    for w in words:
        result = locator.add_chars(w)

    assert result is None
    assert locator.last_rejection is RejectionReason.BEFORE_FLOOR


def test_locator_rejects_match_beyond_ceiling(real_corpus):
    floor_idx = _first_global_idx(real_corpus, 112, 1)
    ceiling_idx = _first_global_idx(real_corpus, 112, 2) - 1  # end of ayah 1
    locator = IncrementalAyahLocator(
        corpus=real_corpus, settings=Settings(), min_global_word_idx=floor_idx, max_global_word_idx=ceiling_idx
    )

    words = _ayah_words(real_corpus, 112, 3)  # ayah 3 -- past the ceiling, skips ahead
    result = None
    for w in words:
        result = locator.add_chars(w)

    assert result is None
    assert locator.last_rejection is RejectionReason.BEYOND_CEILING


def test_locator_accepts_match_exactly_at_floor(real_corpus):
    floor_idx = _first_global_idx(real_corpus, 112, 2)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings(), min_global_word_idx=floor_idx)

    words = _ayah_words(real_corpus, 112, 2)  # ayah 2 -- exactly the floor
    result = None
    for w in words:
        result = locator.add_chars(w)
        if result is not None:
            break

    assert result is not None
    assert result.global_word_idx == floor_idx


def test_locator_accepts_match_after_floor(real_corpus):
    floor_idx = _first_global_idx(real_corpus, 112, 2)
    locator = IncrementalAyahLocator(corpus=real_corpus, settings=Settings(), min_global_word_idx=floor_idx)

    words = _ayah_words(real_corpus, 112, 3)  # ayah 3 -- after the floor
    result = None
    for w in words:
        result = locator.add_chars(w)
        if result is not None:
            break

    assert result is not None
    assert result.global_word_idx == _first_global_idx(real_corpus, 112, 3)


def test_pipeline_end_to_end_backtrack_scenario(real_corpus):
    # Mirrors the requested scenario: session starts at ayah 2, progresses
    # to ayah 3, reciter goes back to ayah 2 (allowed) then tries ayah 1
    # (must be rejected, with a clear status explaining why).
    settings = Settings()
    word_results = []
    statuses = []

    checker = RecitationChecker(
        corpus=real_corpus,
        settings=settings,
        on_word_result=word_results.append,
        on_status=statuses.append,
    )

    # Session starts at 112:2.
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 2))))
    checker.finish()
    assert checker.session_start_global_word_idx == _first_global_idx(real_corpus, 112, 2)

    # Force relocalization so we can test backtracking without waiting on
    # the aligner's rolling confidence to organically collapse.
    checker.locator.state = checker.locator.state.__class__.RELOCALIZING
    checker.locator.buffer = ""

    # Try to go back to ayah 1 -- before the session's start -- must fail
    # and explain why via on_status.
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 1))))
    assert not any(s.startswith("localized: surah 112 ayah 1,") for s in statuses)
    assert any("can't go back before surah 112 ayah 2" in s for s in statuses)

    # Now recite ayah 2 again (exactly the session start) -- must succeed.
    checker.locator.buffer = ""
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 2))))
    assert any(s.startswith("localized: surah 112 ayah 2,") for s in statuses)


def test_pipeline_cannot_skip_ahead_of_furthest_reached(real_corpus):
    settings = Settings()
    statuses = []

    checker = RecitationChecker(
        corpus=real_corpus,
        settings=settings,
        on_word_result=lambda r: None,
        on_status=statuses.append,
    )

    # Recite ayah 1 then ayah 2 back-to-back, normal forward progress.
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 1) + _ayah_words(real_corpus, 112, 2))))
    checker.finish()
    assert checker.max_global_word_idx_reached is not None
    assert checker.max_global_word_idx_reached < _first_global_idx(real_corpus, 112, 3)

    # Force relocalization, then try to jump straight to ayah 4 -- skips
    # over ayah 3, which was never reached. Must be rejected.
    checker.locator.state = checker.locator.state.__class__.RELOCALIZING
    checker.locator.buffer = ""
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 4))))
    assert not any(s.startswith("localized: surah 112 ayah 4,") for s in statuses)
    assert any("can't skip ahead of" in s for s in statuses)

    # Going back to ayah 1 (already reached, at the session floor) still works.
    checker.locator.buffer = ""
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 1))))
    assert any(s.startswith("localized: surah 112 ayah 1,") for s in statuses)


def test_dynamic_floor_advances_past_two_pages(real_corpus):
    settings = Settings()
    checker = RecitationChecker(corpus=real_corpus, settings=settings, on_word_result=lambda r: None)

    session_start = _first_global_idx(real_corpus, 2, 1)
    # Surah 2, ayahs 1-25 is ~282 words -- more than the ~223-word 2-page
    # backtrack window, so the floor should move past ayah 1's start.
    words: list[str] = []
    for ayah in range(1, 26):
        words.extend(_ayah_words(real_corpus, 2, ayah))
    checker.feed_tokens(tokens_for("".join(words)))
    checker.finish()

    assert checker.session_start_global_word_idx == session_start
    assert checker.locator.min_global_word_idx > session_start
    # ... but never below the true session start, however far it advances.
    assert checker.locator.min_global_word_idx >= session_start


def test_backtrack_reroutes_without_reporting_a_false_mismatch(real_corpus):
    # The originally-reported bug: backtracking to an earlier ayah used to
    # register as a mistake, because the (still forward-positioned)
    # aligner tried to align the backtracked speech against words it was
    # still expecting. This must now be caught and rerouted instead.
    settings = Settings()
    word_results = []
    statuses = []

    checker = RecitationChecker(
        corpus=real_corpus,
        settings=settings,
        on_word_result=word_results.append,
        on_status=statuses.append,
    )

    # Recite 112:1, 112:2, 112:3 in sequence, normal forward progress.
    checker.feed_tokens(
        tokens_for("".join(_ayah_words(real_corpus, 112, 1) + _ayah_words(real_corpus, 112, 2) + _ayah_words(real_corpus, 112, 3)))
    )
    assert all(r.status == "match" for r in word_results), word_results

    # Now go back and recite 112:1 again, without any explicit relocalize
    # signal -- the aligner is still expecting 112:4 next, so this should
    # trigger the proactive backtrack check rather than reporting a
    # mismatch for whatever 112:4 word it lands on.
    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 1))))
    checker.finish()

    assert not any(r.status == "mismatch" and r.surah == 112 and r.ayah == 4 for r in word_results)
    assert any(s.startswith("localized: surah 112 ayah 1,") for s in statuses)
    # The backtracked recitation of 112:1 is correctly matched -- once
    # from the initial forward pass, once again from the backtrack.
    ayah1_results = [r for r in word_results if r.surah == 112 and r.ayah == 1]
    assert len(ayah1_results) == 8
    assert all(r.status == "match" for r in ayah1_results)


def test_flush_deleted_words_do_not_inflate_ceiling(real_corpus):
    # Regression: flush()'s tail loop force-emits "deleted" for
    # speculatively-refilled lookahead words that were never actually
    # attempted (e.g. the rest of the Quran after a short recitation ends).
    # Those must not count as "reached" for the skip-ahead ceiling.
    settings = Settings()
    checker = RecitationChecker(corpus=real_corpus, settings=settings, on_word_result=lambda r: None)

    checker.feed_tokens(tokens_for("".join(_ayah_words(real_corpus, 112, 1))))
    checker.finish()  # flush() pulls in and "deletes" several lookahead words

    last_real_word_idx = _first_global_idx(real_corpus, 112, 1) + len(_ayah_words(real_corpus, 112, 1)) - 1
    assert checker.max_global_word_idx_reached == last_real_word_idx
