import json
import time
from dataclasses import dataclass
from pathlib import Path

from qrc.align.word_aligner import WordCheckResult


@dataclass
class _PendingGroup:
    surah: int
    ayah: int
    word_index: int  # first phoneme-word's index, for reference
    word_text: str
    all_match: bool


class CorrectWordLogger:
    """Appends the real Uthmani word text to a JSONL log each time a word
    is confirmed correctly recited -- once per real written word, not once
    per underlying phoneme-word unit.

    Some written words (muqatta'at letters, e.g. "الٓمٓ") span multiple
    corpus phoneme-word units, each producing its own WordCheckResult.
    Logging each unit separately would show the same word twice. Results
    carrying the same word_text with word_text_continues_previous=True are
    buffered into one pending group and flushed as a single log line once
    the group ends (a differently-texted result arrives, or the session
    does -- call flush() explicitly then). The group only gets logged if
    *every* unit in it matched -- a real word that's half-right shouldn't
    be logged as correctly recited just because its first phoneme unit was.

    Words without a resolved word_text (scripts/build_word_text_map.py
    wasn't run, or couldn't confidently map that specific word -- see that
    script's docstring) are silently skipped: logging a placeholder would
    be misleading, and that gap is inherent to the underlying alignment
    problem, not something a fallback string can paper over.
    """

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: _PendingGroup | None = None

    def __call__(self, result: WordCheckResult) -> None:
        continues = (
            result.word_text_continues_previous
            and self._pending is not None
            and self._pending.word_text == result.word_text
            and self._pending.surah == result.surah
            and self._pending.ayah == result.ayah
        )
        if continues:
            self._pending.all_match = self._pending.all_match and (result.status == "match")
            return

        self.flush()
        if result.word_text:
            self._pending = _PendingGroup(
                surah=result.surah,
                ayah=result.ayah,
                word_index=result.word_index,
                word_text=result.word_text,
                all_match=(result.status == "match"),
            )

    def flush(self) -> None:
        """Write out any pending group. Call at end of session -- the last
        group has no *following* result to trigger its own flush.
        """
        if self._pending is not None and self._pending.all_match:
            self._write(self._pending)
        self._pending = None

    def _write(self, group: _PendingGroup) -> None:
        entry = {
            "timestamp": time.time(),
            "surah": group.surah,
            "ayah": group.ayah,
            "word_index": group.word_index,
            "word_text": group.word_text,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
