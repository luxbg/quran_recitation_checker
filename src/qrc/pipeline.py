from dataclasses import dataclass, field
from typing import Callable

from qrc.align.word_aligner import IncrementalWordAligner, WordCheckResult
from qrc.asr.streaming_recognizer import PhonemeToken
from qrc.config import Settings
from qrc.corpus.index import QuranCorpus
from qrc.localize.ayah_locator import IncrementalAyahLocator, LocatorState, RejectionReason, search_window


def _noop(_msg: str) -> None:
    pass


@dataclass
class RecitationChecker:
    """Wires ASR token stream -> locator -> aligner -> WordCheckResult callback.

    Session lifecycle is continuous (one long-running stream, no resets):
    the checker rolls forward across ayah boundaries, and relocalizes in
    place if the aligner's rolling confidence collapses (e.g. the reciter
    jumped to an unexpected ayah), without discarding the whole session.
    """

    corpus: QuranCorpus
    settings: Settings
    on_word_result: Callable[[WordCheckResult], None]
    on_status: Callable[[str], None] = field(default=_noop)

    _RECENT_TEXT_CAP = 200

    def __post_init__(self) -> None:
        self.locator = IncrementalAyahLocator(corpus=self.corpus, settings=self.settings)
        self.aligner = IncrementalWordAligner(
            corpus=self.corpus, settings=self.settings, on_word_result=self._on_word_settled
        )
        self._recent_text = ""
        # Backtracking floor: the session's true start, set once on the
        # first successful localization and never updated again. The
        # *effective* floor used for matching (self.locator.min_global_word_idx)
        # is kept dynamic -- see _advance_progress -- but never below this.
        self.session_start_global_word_idx: int | None = None
        # Ceiling: the furthest position actually reached this session --
        # relocalization can go back to redo an earlier ayah, but can't
        # skip ahead into ayahs never yet recited. Advances as words settle
        # or as a (re)localization reaches a new position.
        self.max_global_word_idx_reached: int | None = None
        # Words/page is only an estimate (real Mushaf pagination isn't in
        # the corpus data) -- computed once from the corpus's own average.
        words_per_page = len(self.corpus.global_words) / self.settings.mushaf_total_pages
        self._backtrack_window_words = round(words_per_page * self.settings.backtrack_window_pages)
        self._flushing = False
        # Accumulates actual_phonemes from consecutive mismatches only,
        # for the backtrack-vs-mistake check -- see _reroute_if_backtrack
        # for why this can't just be a slice of _recent_text.
        self._suspect_buffer = ""

    def feed_tokens(self, tokens: list[PhonemeToken]) -> None:
        text = "".join(t.symbol for t in tokens if t.symbol != "<blank>")
        if not text:
            return
        self._recent_text = (self._recent_text + text)[-self._RECENT_TEXT_CAP :]

        if self.locator.state == LocatorState.LOCALIZED:
            self.aligner.feed_tokens(tokens)
            if self.aligner.confidence_collapsed():
                self._begin_relocalize()
            return

        result = self.locator.add_chars(text)
        if result is not None:
            self._on_localized(result.global_word_idx, result.matched_text)
        elif self.locator.last_rejection is not None:
            self._report_rejection(self.locator.last_rejection)

    def finish(self) -> None:
        """Call once at end of session/audio to force-settle any tail words."""
        if self.locator.state == LocatorState.LOCALIZED:
            self._flushing = True
            self.aligner.flush()
            self._flushing = False

    def _begin_relocalize(self) -> None:
        self.on_status("lost track of recitation -- relocalizing")
        self.locator.seed_for_relocalize(self._recent_text)

    def _on_word_settled(self, result: WordCheckResult) -> None:
        if result.status != "deleted":
            self._advance_progress(result.global_word_index)

        # A mismatch might not be a mistake at all -- the reciter may have
        # jumped to an earlier ayah within the backtrack window rather
        # than mispronounced this one. Check before reporting it as an
        # error: without this, every backtrack registers a spurious
        # mismatch for the word the (still forward-positioned) aligner was
        # expecting, because that verdict is only known to be wrong
        # *after* it's already been produced.
        #
        # Deliberately restricted to "mismatch" (real recited content that
        # didn't match) -- never "deleted" (nothing recited, so there's no
        # content to check against) -- and never while flush() is running:
        # calling aligner.localize() here would reset pending_words/
        # expected_buffer out from under flush()'s own in-progress loop.
        # This is exactly the bug that showed up as ~5x duplicated results
        # in tests/test_pipeline.py before this guard was added.
        if result.status == "match":
            self._suspect_buffer = ""
        elif result.status == "mismatch":
            if result.actual_phonemes:
                self._suspect_buffer += result.actual_phonemes
            if not self._flushing and self._reroute_if_backtrack(result):
                return

        self.on_word_result(result)

    def _advance_progress(self, global_word_idx: int) -> None:
        # "Deleted" words (nothing recited) never advance progress -- see
        # caller. Only genuine recited content (match/mismatch) counts,
        # otherwise flush()'s speculative lookahead tail would silently
        # inflate both the skip-ahead ceiling and the backtrack window.
        if self.max_global_word_idx_reached is None or global_word_idx > self.max_global_word_idx_reached:
            self.max_global_word_idx_reached = global_word_idx
        self.locator.max_global_word_idx = self.max_global_word_idx_reached

        floor = self.session_start_global_word_idx or 0
        self.locator.min_global_word_idx = max(floor, self.max_global_word_idx_reached - self._backtrack_window_words)

    def _reroute_if_backtrack(self, result: WordCheckResult) -> bool:
        # _suspect_buffer, not a slice of _recent_text: it accumulates
        # actual_phonemes from consecutive mismatches only (cleared on any
        # match), so it never mixes in old, already-correct content ahead
        # of the new backtracked speech -- that dilution blew the
        # edit-distance budget even at a 30-char window for short ayahs
        # (their whole recitation is barely longer than that). Builds up
        # across repeated mismatches too, in case one word's actual
        # content alone isn't distinctive enough to confidently place.
        query = self._suspect_buffer
        if len(query) < self.settings.min_trigger_chars:
            return False

        candidate = search_window(
            self.corpus,
            query,
            self.settings,
            min_global_word_idx=self.locator.min_global_word_idx,
            max_global_word_idx=self.max_global_word_idx_reached,
        )
        if candidate is None:
            return False

        # Only a jump to somewhere clearly different counts as a
        # backtrack -- a match that lands right around where we already
        # are is just this being an ordinary mismatch/ASR noise, not proof
        # the reciter went elsewhere.
        if abs(candidate.global_word_idx - result.global_word_index) < self.settings.backtrack_min_word_gap:
            return False

        # Any characters already fed to the old (wrong-position) aligner
        # but not yet attributed to a settled word would otherwise be
        # lost when aligner.localize() resets actual_buffer -- rescue
        # them, they're the same backtracked speech, just not yet
        # consumed into a WordCheckResult. Same "swallowed audio" failure
        # mode as the original localization handoff, recurring here.
        leftover = self.aligner.actual_buffer
        self._suspect_buffer = ""
        self._on_localized(candidate.global_word_idx, candidate.matched_text + leftover)
        return True

    def _on_localized(self, global_word_idx: int, matched_text: str) -> None:
        entry = self.corpus.word_at(global_word_idx)

        if self.session_start_global_word_idx is None:
            self.session_start_global_word_idx = global_word_idx

        self._advance_progress(global_word_idx)

        self.aligner.localize(global_word_idx)
        # The text that led to a successful localization was actually
        # recited -- replay it into the aligner rather than discarding it,
        # otherwise the words used to localize get silently misreported as
        # never recited.
        if matched_text:
            self.aligner.feed_text(matched_text)
        if entry is not None:
            self.on_status(f"localized: surah {entry.surah} ayah {entry.ayah}, word {entry.local_word_idx}")

    def _report_rejection(self, reason: RejectionReason) -> None:
        if reason is RejectionReason.BEFORE_FLOOR:
            bound_idx = self.locator.min_global_word_idx
            verb, note = "go back before", f"the last {self.settings.backtrack_window_pages} pages"
        else:
            bound_idx = self.max_global_word_idx_reached
            verb, note = "skip ahead of", "haven't recited that far yet"

        if bound_idx is None:
            return
        entry = self.corpus.word_at(bound_idx)
        if entry is not None:
            self.on_status(f"can't {verb} surah {entry.surah} ayah {entry.ayah} ({note})")
