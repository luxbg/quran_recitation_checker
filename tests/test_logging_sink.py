import json

from qrc.align.word_aligner import WordCheckResult
from qrc.logging_sink import CorrectWordLogger


def _result(**overrides) -> WordCheckResult:
    base = dict(
        surah=112,
        ayah=1,
        word_index=0,
        global_word_index=67212,
        expected_phonemes="قُل",
        actual_phonemes="قُل",
        status="match",
        similarity=1.0,
        word_text="قُلْ",
        word_text_continues_previous=False,
    )
    base.update(overrides)
    return WordCheckResult(**base)


def _read_lines(log_path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_logs_correctly_recited_word_with_text(tmp_path):
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result())
    logger.flush()

    entries = _read_lines(log_path)
    assert len(entries) == 1
    assert entries[0]["surah"] == 112
    assert entries[0]["ayah"] == 1
    assert entries[0]["word_text"] == "قُلْ"


def test_skips_mismatch(tmp_path):
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result(status="mismatch"))
    logger.flush()

    assert _read_lines(log_path) == []


def test_skips_match_with_no_resolved_word_text(tmp_path):
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result(word_text=None))
    logger.flush()

    assert _read_lines(log_path) == []


def test_appends_across_multiple_words(tmp_path):
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result(word_index=0, word_text="قُلْ"))
    logger(_result(word_index=1, word_text="هُوَ"))
    logger.flush()

    entries = _read_lines(log_path)
    assert len(entries) == 2
    assert entries[0]["word_text"] == "قُلْ"
    assert entries[1]["word_text"] == "هُوَ"


def test_split_word_logged_only_once(tmp_path):
    # "الٓمٓ" (muqatta'at) spans two phoneme-word units, both carrying the
    # same word_text -- must be logged once, not twice.
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result(surah=2, ayah=1, word_index=0, word_text="الٓمٓ", word_text_continues_previous=False))
    logger(_result(surah=2, ayah=1, word_index=1, word_text="الٓمٓ", word_text_continues_previous=True))
    logger.flush()

    entries = _read_lines(log_path)
    assert len(entries) == 1
    assert entries[0]["word_text"] == "الٓمٓ"


def test_split_word_not_logged_if_any_unit_mismatches(tmp_path):
    log_path = tmp_path / "correct_words.jsonl"
    logger = CorrectWordLogger(log_path)

    logger(_result(surah=2, ayah=1, word_index=0, word_text="الٓمٓ", status="match", word_text_continues_previous=False))
    logger(_result(surah=2, ayah=1, word_index=1, word_text="الٓمٓ", status="mismatch", word_text_continues_previous=True))
    logger.flush()

    assert _read_lines(log_path) == []
