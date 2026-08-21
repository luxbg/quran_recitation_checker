from dataclasses import dataclass, field
from enum import Enum, auto

from fuzzysearch import find_near_matches

from qrc.config import Settings
from qrc.corpus.index import QuranCorpus


class LocatorState(Enum):
    LOCALIZING = auto()
    LOCALIZED = auto()
    RELOCALIZING = auto()


class RejectionReason(Enum):
    BEFORE_FLOOR = auto()  # would skip backward before the session's start
    BEYOND_CEILING = auto()  # would skip ahead past the furthest ayah reached


@dataclass
class LocalizeResult:
    global_word_idx: int
    similarity: float
    # The raw buffered text that produced this match -- it was actually
    # recited, so the caller should replay it into the aligner instead of
    # discarding it (otherwise the words used to localize are silently
    # dropped and misreported as never-recited).
    matched_text: str


def _score_candidates(corpus: QuranCorpus, query: str, settings: Settings) -> list[tuple[int, float]]:
    """Fuzzy-substring-search query against the whole corpus, returning
    (char_offset, similarity) pairs sorted best-first. Shared by both the
    incremental locator and one-shot search_window.
    """
    max_l_dist = min(max(1, int(len(query) * settings.max_l_dist_ratio)), settings.max_l_dist_cap)
    matches = find_near_matches(query, corpus.corpus_text, max_l_dist=max_l_dist)
    scored = [(m.start, 1.0 - m.dist / max(len(query), m.end - m.start)) for m in matches]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def _accept(scored: list[tuple[int, float]], settings: Settings) -> bool:
    if not scored:
        return False
    top_sim = scored[0][1]
    second_sim = scored[1][1] if len(scored) > 1 else 0.0
    return top_sim >= settings.confidence_threshold and (top_sim - second_sim) >= settings.margin_threshold


def search_window(
    corpus: QuranCorpus,
    query: str,
    settings: Settings,
    min_global_word_idx: int,
    max_global_word_idx: int | None,
) -> LocalizeResult | None:
    """One-shot confident match of a complete (not growing) query string
    within [min_global_word_idx, max_global_word_idx]. Unlike
    IncrementalAyahLocator, there's no buffering -- the caller already has
    the full text to search for. Used to check whether a word that just
    failed to match the current forward position is actually a backtrack
    to somewhere else nearby, rather than a genuine mistake.
    """
    scored = _score_candidates(corpus, query, settings)
    in_range = [
        (offset, sim)
        for offset, sim in scored
        if min_global_word_idx <= corpus.global_word_idx_for_char_offset(offset)
        and (max_global_word_idx is None or corpus.global_word_idx_for_char_offset(offset) <= max_global_word_idx)
    ]
    if not _accept(in_range, settings):
        return None
    global_word_idx = corpus.global_word_idx_for_char_offset(in_range[0][0])
    return LocalizeResult(global_word_idx=global_word_idx, similarity=in_range[0][1], matched_text=query)


@dataclass
class IncrementalAyahLocator:
    """Localizes a recitation from a raw, growing phoneme character stream.

    The ASR alphabet has no word-boundary token, so there's no way to
    pre-segment the buffered audio into words before a location is known
    (unlike quran-muaalem's batch approach, which only ever searches after
    a complete utterance). Instead this does fuzzy substring search of the
    raw character buffer directly against the flat corpus text, which
    fuzzysearch handles natively -- benchmarked on the real corpus at
    single-digit milliseconds per attempt with a capped edit-distance
    tolerance (see Settings.max_l_dist_cap).
    """

    corpus: QuranCorpus
    settings: Settings

    state: LocatorState = LocatorState.LOCALIZING
    buffer: str = ""
    # Backtracking floor -- see RecitationChecker for how this is kept
    # dynamic (the last N pages, never below the session's true start).
    # 0 (the default) means unrestricted, correct before the session has
    # localized for the first time.
    min_global_word_idx: int = 0
    # Ceiling: no match past the furthest position actually reached this
    # session is ever accepted either -- a reciter may go back to redo an
    # earlier ayah, but relocalization can't skip ahead into ayahs never
    # yet recited. None (the default) means unrestricted, which is correct
    # before the session has localized for the first time (nothing has
    # been "reached" yet, so there's nothing to cap against).
    max_global_word_idx: int | None = None
    # Set by _attempt_match when the top candidate would otherwise have
    # been accepted but fell outside [min_global_word_idx,
    # max_global_word_idx] -- lets the caller distinguish "no match at
    # all" from "found the requested ayah, but it's out of the allowed
    # range" for a clearer status message. Cleared at the start of every
    # add_chars call.
    last_rejection: RejectionReason | None = field(default=None, init=False)

    def reset(self) -> None:
        self.state = LocatorState.LOCALIZING
        self.buffer = ""

    def seed_for_relocalize(self, recent_text: str) -> None:
        self.state = LocatorState.RELOCALIZING
        self.buffer = recent_text[-self.settings.relocalize_seed_chars :]

    def add_chars(self, new_text: str) -> LocalizeResult | None:
        """Feed more recognized phoneme characters. Returns a result once localized.

        On max_buffer_chars with no confident match, discards the stale
        buffer and starts fresh (per product decision -- avoids garbled
        early audio permanently poisoning later attempts).
        """
        self.last_rejection = None
        self.buffer += new_text

        if len(self.buffer) < self.settings.min_trigger_chars:
            return None

        result = self._attempt_match()
        if result is not None:
            self.state = LocatorState.LOCALIZED
            self.buffer = ""
            return result

        if len(self.buffer) >= self.settings.max_buffer_chars:
            self.buffer = ""

        return None

    def _in_range(self, global_word_idx: int) -> bool:
        if global_word_idx < self.min_global_word_idx:
            return False
        if self.max_global_word_idx is not None and global_word_idx > self.max_global_word_idx:
            return False
        return True

    def _attempt_match(self) -> LocalizeResult | None:
        scored = _score_candidates(self.corpus, self.buffer, self.settings)
        if not scored:
            return None

        in_range = [(offset, sim) for offset, sim in scored if self._in_range(self.corpus.global_word_idx_for_char_offset(offset))]

        if not in_range:
            # Nothing left after the floor/ceiling -- check whether we
            # *would* have matched something out of range, purely to
            # report a clearer status than silent buffering.
            if _accept(scored, self.settings):
                top_idx = self.corpus.global_word_idx_for_char_offset(scored[0][0])
                self.last_rejection = (
                    RejectionReason.BEFORE_FLOOR if top_idx < self.min_global_word_idx else RejectionReason.BEYOND_CEILING
                )
            return None

        if _accept(in_range, self.settings):
            global_word_idx = self.corpus.global_word_idx_for_char_offset(in_range[0][0])
            return LocalizeResult(global_word_idx=global_word_idx, similarity=in_range[0][1], matched_text=self.buffer)
        return None
