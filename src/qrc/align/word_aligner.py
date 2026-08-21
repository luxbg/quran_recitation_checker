import bisect
from dataclasses import dataclass, field
from typing import Literal

from qrc.align.dp import DELETE_A, INSERT_A, MATCH, edit_distance_with_backpointers, traceback
from qrc.align.normalize import phoneme_similarity, phonemes_match
from qrc.asr.streaming_recognizer import PhonemeToken
from qrc.config import Settings
from qrc.corpus.index import QuranCorpus
from qrc.corpus.models import GlobalWordEntry

WordStatus = Literal["match", "mismatch", "deleted"]


@dataclass
class WordCheckResult:
    surah: int
    ayah: int
    word_index: int  # 0-based, within the ayah (matches aya_phonemes_list)
    global_word_index: int
    expected_phonemes: str
    actual_phonemes: str | None  # None for a fully skipped/deleted word
    status: WordStatus
    similarity: float
    # Real Uthmani script word(s) -- None if scripts/build_word_text_map.py
    # wasn't run or couldn't confidently map this specific word.
    word_text: str | None = None
    # True when word_text is the *same* real written word as the previous
    # settled result (a muqatta'at split, e.g. "الٓمٓ" spans two phoneme-
    # word units) -- consumers must treat a continues_previous run as one
    # logical word, not report/log each unit separately.
    word_text_continues_previous: bool = False
    start_time_s: float | None = None
    end_time_s: float | None = None


@dataclass
class _PendingWord:
    entry: GlobalWordEntry
    start_offset: int  # start char offset in expected_buffer (inclusive)
    end_offset: int  # end char offset in expected_buffer (exclusive)


@dataclass
class IncrementalWordAligner:
    """Character-level (Option B) online alignment.

    The ASR alphabet has no space/word-boundary token, so we can't buffer
    ASR output between word emissions. Instead we DP-align the growing
    recognized-phoneme buffer against a known-in-advance concatenation of
    upcoming expected words' phonemes, and use the DP's own convergence
    (how far the alignment frontier has moved past a word's boundary) to
    decide when that word's verdict is safe to emit.

    Known limitation: because there are no ASR-side word boundaries, a
    whole extra recited word not in the expected text can't be reported as
    its own "inserted" entry -- it shows up as inflated actual_phonemes on
    the nearest expected word, surfaced as a low-similarity mismatch.
    """

    corpus: QuranCorpus
    settings: Settings
    on_word_result: object  # Callable[[WordCheckResult], None]

    actual_buffer: str = ""
    pending_words: list[_PendingWord] = field(default_factory=list)
    expected_buffer: str = ""
    next_global_word_idx: int | None = None
    rolling_similarity_ema: float = 1.0
    # The actual_phonemes of the last word settled with genuine content
    # (match/mismatch, not deleted) -- used to detect a repeated word
    # bleeding into the next one, see _strip_repeated_prefix.
    _last_settled_actual: str | None = None
    # The isolated_phoneme_text of the last word settled as a genuine
    # *match* (not mismatch/deleted) -- used to recover cross-word
    # tajweed-liaison bleed into the next word, see _strip_liaison_bleed.
    # None whenever the previous word wasn't a clean match: with no
    # confirmed idea what was actually said, there's nothing trustworthy
    # to attribute a leading bleed to.
    _last_settled_isolated: str | None = None

    LOOKAHEAD_WORDS = 8
    _MIN_REPEAT_OVERLAP = 2

    def localize(self, global_word_idx: int) -> None:
        self.next_global_word_idx = global_word_idx
        self.actual_buffer = ""
        self.pending_words = []
        self.expected_buffer = ""
        self.rolling_similarity_ema = 1.0
        self._last_settled_actual = None
        self._last_settled_isolated = None
        self._refill_expected()

    def _strip_repeated_prefix(self, actual: str) -> str:
        """A reciter repeating a word (self-correction, memorization
        practice -- reported from a live session) leaves the repeat's
        trailing characters attributed to the *next* word instead, since
        the DP has no way to know they're a repeat rather than new
        content: found via a real case where "...وَيَشتَرُۥۥنَ" (matched)
        was followed by an unrelated word whose actual_phonemes came back
        prefixed with "ۥۥنَ" -- the tail of the previous word, repeated.
        Strips the longest prefix of `actual` that's also a suffix of the
        previously settled word's actual content (min 2 chars, to avoid
        stripping a coincidental single-character overlap).
        """
        if not self._last_settled_actual or not actual:
            return actual
        max_overlap = min(len(self._last_settled_actual), len(actual))
        for length in range(max_overlap, self._MIN_REPEAT_OVERLAP - 1, -1):
            if self._last_settled_actual.endswith(actual[:length]):
                return actual[length:]
        return actual

    def _strip_liaison_bleed(self, actual: str) -> str:
        """A word recited in its own standalone (waqf) form -- rather than
        the corpus's connected-recitation phoneme_text, which already
        assumes obligatory tajweed liaison (idgham/ikhfa/iqlab) absorbed
        some of its trailing content into the next word -- leaves that
        extra trailing content attributed to the *next* word instead: the
        DP has no notion of "this belongs to the word before", it just
        sees unexplained leading characters and assigns them onward, same
        root cause as _strip_repeated_prefix. Real case: "مَن" recited
        with a clearly-pronounced ن (rather than the idgham-bighunna merge
        into the following "يَقُولُ" that the corpus's connected form for
        "مَن" assumes) left a leading ن attributed to "يَقُولُ"'s
        actual_phonemes.

        Strips the longest prefix of `actual` that's also a suffix of the
        *previous settled word's own* isolated-pronunciation phonemes
        (_last_settled_isolated -- already gated to only be set when that
        word settled as a clean match, see its docstring). Unlike
        _strip_repeated_prefix, a 1-char overlap is trusted here: this
        isn't a coincidental content match against arbitrary preceding
        text, it's a specific word's own known standalone pronunciation,
        and the caller still requires the stripped result to exactly
        match this word's own expected phonemes before adopting it.
        """
        if not self._last_settled_isolated or not actual:
            return actual
        max_overlap = min(len(self._last_settled_isolated), len(actual))
        for length in range(max_overlap, 0, -1):
            if self._last_settled_isolated.endswith(actual[:length]):
                return actual[length:]
        return actual

    def _refill_expected(self) -> None:
        assert self.next_global_word_idx is not None
        idx = self.next_global_word_idx + len(self.pending_words)
        while len(self.pending_words) < self.LOOKAHEAD_WORDS:
            entry = self.corpus.word_at(idx)
            if entry is None:
                break
            start = len(self.expected_buffer)
            self.expected_buffer += entry.phoneme_text
            self.pending_words.append(_PendingWord(entry=entry, start_offset=start, end_offset=len(self.expected_buffer)))
            idx += 1

    def feed_tokens(self, tokens: list[PhonemeToken]) -> None:
        blank = "<blank>"
        text = "".join(tok.symbol for tok in tokens if tok.symbol != blank)
        self.feed_text(text)

    # Large input is processed in small increments, not dumped into
    # actual_buffer all at once: expected_buffer only ever holds
    # LOOKAHEAD_WORDS words (a few dozen characters). If actual_buffer
    # were allowed to grow far past that before a settle attempt runs,
    # _try_settle's DP would have to explain a huge actual_buffer against
    # a tiny expected_buffer -- most of it gets treated as insertions
    # into whatever word is first pending, producing one garbage
    # wildly-long actual_phonemes instead of settling words normally, and
    # the O(n*m) DP itself becomes slow at that size. Caught via a
    # multi-ayah single-shot feed in testing; chunking keeps feed_text
    # correct for arbitrarily large input without that blowup.
    _FEED_CHUNK_CHARS = 20

    def feed_text(self, text: str) -> None:
        """Feed raw recognized phoneme characters directly (no PhonemeToken
        wrapping needed) -- used to replay text the locator already
        consumed while localizing, so those words aren't silently dropped.
        """
        for start in range(0, len(text), self._FEED_CHUNK_CHARS):
            self.actual_buffer += text[start : start + self._FEED_CHUNK_CHARS]
            self._refill_expected()
            self._try_settle()

    def _word_index_for_offset(self, offset: int) -> int:
        ends = [w.end_offset for w in self.pending_words]
        i = bisect.bisect_right(ends, offset)
        return min(i, len(self.pending_words) - 1)

    def _try_settle(self) -> None:
        while self.pending_words and self.actual_buffer:
            dp, bp = edit_distance_with_backpointers(self.actual_buffer, self.expected_buffer)
            last_row = dp[len(self.actual_buffer)]
            j_star = int(last_row.argmin())

            boundary = self.pending_words[0].end_offset
            if j_star - boundary < self.settings.settle_lookahead_chars:
                break

            path = traceback(bp, len(self.actual_buffer), j_star)
            self._emit_first_word(path)

    def _emit_first_word(self, path: list[tuple[int | None, int | None, int]]) -> None:
        first_word = self.pending_words[0]
        actual_chars: list[str] = []
        consumed_a_count = 0
        j_consumed = 0

        for i_idx, j_idx, op in path:
            ownership_j = j_idx if j_idx is not None else j_consumed
            owner_word = self._word_index_for_offset(ownership_j)
            if owner_word > 0:
                break

            if op == MATCH:
                actual_chars.append(self.actual_buffer[i_idx])
                consumed_a_count += 1
                j_consumed += 1
            elif op == INSERT_A:
                actual_chars.append(self.actual_buffer[i_idx])
                consumed_a_count += 1
            elif op == DELETE_A:
                j_consumed += 1

        actual_phonemes = "".join(actual_chars) or None
        expected_phonemes = first_word.entry.phoneme_text
        isolated_phonemes = first_word.entry.isolated_phoneme_text
        matched_isolated = False

        if actual_phonemes is not None and not phonemes_match(expected_phonemes, actual_phonemes):
            # A word recited with a brief pause right after it -- common in
            # word-by-word practice recitation, not just a formal end-of-ayah
            # waqf -- comes out in its own standalone pronunciation, which
            # can differ from the corpus's connected-recitation phoneme_text
            # wherever cross-word tajweed liaison (idgham/ikhfa/iqlab) or a
            # pause-dropped tanween/short-vowel applies. Check that first:
            # it's still the same word, correctly recited, just not in the
            # form that assumes it flows straight into the next one.
            if isolated_phonemes and phonemes_match(isolated_phonemes, actual_phonemes):
                matched_isolated = True
            else:
                # Repeat-prefix stripping is a *fallback for an otherwise-
                # mismatched word*, never applied to content that already
                # matches. Word boundaries in Arabic often coincidentally
                # share phonemes (e.g. "...ءِ" ending one word, "ءِ..."
                # starting the next -- hamza+kasra is common) -- stripping
                # unconditionally turned a correct recitation into a false
                # mismatch by cutting off genuine leading content that
                # happened to resemble the previous word's ending. Only
                # adopt the stripped version if it actually produces a
                # match; if it doesn't help, report the real (unstripped)
                # content instead of a partially-stripped guess.
                stripped = self._strip_repeated_prefix(actual_phonemes)
                if stripped != actual_phonemes and phonemes_match(expected_phonemes, stripped):
                    actual_phonemes = stripped or None
                elif self._last_settled_isolated:
                    bled = self._strip_liaison_bleed(actual_phonemes)
                    if bled != actual_phonemes and phonemes_match(expected_phonemes, bled):
                        actual_phonemes = bled or None

        similarity_reference = isolated_phonemes if matched_isolated else expected_phonemes
        similarity = 0.0 if actual_phonemes is None else phoneme_similarity(similarity_reference, actual_phonemes)

        if actual_phonemes is None:
            status: WordStatus = "deleted"
        elif matched_isolated or phonemes_match(expected_phonemes, actual_phonemes):
            status = "match"
        else:
            status = "mismatch"

        if actual_phonemes is not None:
            self._last_settled_actual = actual_phonemes
        self._last_settled_isolated = isolated_phonemes if status == "match" else None

        alpha = self.settings.relocalize_ema_alpha
        self.rolling_similarity_ema = alpha * similarity + (1 - alpha) * self.rolling_similarity_ema

        result = WordCheckResult(
            surah=first_word.entry.surah,
            ayah=first_word.entry.ayah,
            word_index=first_word.entry.local_word_idx,
            global_word_index=first_word.entry.global_word_idx,
            expected_phonemes=expected_phonemes,
            actual_phonemes=actual_phonemes,
            status=status,
            similarity=similarity,
            word_text=first_word.entry.word_text,
            word_text_continues_previous=first_word.entry.word_text_continues_previous,
        )

        # Trim: drop word 0 and shift remaining pending words' offsets down.
        shift = first_word.end_offset
        self.actual_buffer = self.actual_buffer[consumed_a_count:]
        self.expected_buffer = self.expected_buffer[shift:]
        self.pending_words = [
            _PendingWord(entry=w.entry, start_offset=w.start_offset - shift, end_offset=w.end_offset - shift)
            for w in self.pending_words[1:]
        ]
        self.next_global_word_idx = first_word.entry.global_word_idx + 1
        self._refill_expected()

        self.on_word_result(result)

    def confidence_collapsed(self) -> bool:
        return self.rolling_similarity_ema < self.settings.relocalize_ema_threshold

    def flush(self) -> None:
        """Force-settle every currently-pending word (end of session/ayah/Quran).

        Without this, the tail of a recitation -- or the very last ayah of
        the whole Quran, which has no further words to build settlement
        margin against -- would never emit a verdict.

        Deliberately not one big forced alignment against the whole
        pending window (targeting the very end of expected_buffer) --
        that let the DP "spread" genuinely-recited characters thin to
        help cover words with zero audio, truncating the last real word.
        Instead, each iteration either settles a word the *natural*
        (unforced) alignment already fully covers, or -- for the one word
        straddling where the actual audio trails off -- forces just up to
        that word's own boundary (not further), so it still gets whatever
        real content precedes it instead of being lumped in with the
        words that were never spoken at all.
        """
        while self.pending_words and self.actual_buffer:
            dp, bp = edit_distance_with_backpointers(self.actual_buffer, self.expected_buffer)
            last_row = dp[len(self.actual_buffer)]
            j_star = int(last_row.argmin())
            boundary = self.pending_words[0].end_offset
            j_target = j_star if j_star >= boundary else boundary
            path = traceback(bp, len(self.actual_buffer), j_target)
            self._emit_first_word(path)

        # Counted, not `while self.pending_words`: _emit_deleted_word calls
        # _refill_expected, which would otherwise keep pulling in new words
        # forever (there's always more Quran text) instead of just draining
        # what was actually pending when flush() was called.
        remaining = len(self.pending_words)
        for _ in range(remaining):
            self._emit_deleted_word()

    def _emit_deleted_word(self) -> None:
        first_word = self.pending_words[0]
        # A skipped word breaks the physical adjacency _strip_liaison_bleed
        # relies on (it strips content bled from the *immediately*
        # preceding word) -- nothing was actually recited here to bleed
        # from, and re-using an older reference across the gap would be a
        # guess, not a known-adjacent word's own pronunciation.
        self._last_settled_isolated = None
        result = WordCheckResult(
            surah=first_word.entry.surah,
            ayah=first_word.entry.ayah,
            word_index=first_word.entry.local_word_idx,
            global_word_index=first_word.entry.global_word_idx,
            expected_phonemes=first_word.entry.phoneme_text,
            actual_phonemes=None,
            status="deleted",
            similarity=0.0,
            word_text=first_word.entry.word_text,
            word_text_continues_previous=first_word.entry.word_text_continues_previous,
        )

        shift = first_word.end_offset
        self.expected_buffer = self.expected_buffer[shift:]
        self.pending_words = [
            _PendingWord(entry=w.entry, start_offset=w.start_offset - shift, end_offset=w.end_offset - shift)
            for w in self.pending_words[1:]
        ]
        self.next_global_word_idx = first_word.entry.global_word_idx + 1
        self._refill_expected()

        self.on_word_result(result)
