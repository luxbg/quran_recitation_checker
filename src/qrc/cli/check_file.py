import time
from pathlib import Path

from rich.console import Console

from qrc.asr.streaming_recognizer import StreamingPhonemeRecognizer
from qrc.audio.wav_source import iter_wav_chunks
from qrc.cli.formatting import print_status, print_word_result
from qrc.config import DEFAULT_SETTINGS
from qrc.corpus.index import build_corpus
from qrc.logging_sink import CorrectWordLogger
from qrc.pipeline import RecitationChecker


def run_check_file(wav_path: Path, realtime: bool = False) -> None:
    console = Console()
    settings = DEFAULT_SETTINGS

    console.print(f"[dim]Loading corpus from {settings.corpus_path}...[/dim]")
    corpus = build_corpus(settings.corpus_path)

    console.print(f"[dim]Loading model from {settings.model_path}...[/dim]")
    recognizer = StreamingPhonemeRecognizer(
        tokens_path=settings.tokens_path,
        model_path=settings.model_path,
        sample_rate=settings.sample_rate,
        provider=settings.provider,
    )

    log_correct_word = CorrectWordLogger(settings.correct_words_log_path)

    def on_word_result(r):
        print_word_result(console, r)
        log_correct_word(r)

    checker = RecitationChecker(
        corpus=corpus,
        settings=settings,
        on_word_result=on_word_result,
        on_status=lambda m: print_status(console, m),
    )

    console.print(f"[bold]Streaming {wav_path}...[/bold]")
    for samples, sr in iter_wav_chunks(wav_path):
        if realtime:
            time.sleep(len(samples) / sr)
        recognizer.feed_audio(samples, sample_rate=sr)
        tokens = recognizer.poll_new_tokens()
        if tokens:
            checker.feed_tokens(tokens)

    tokens = recognizer.finish()
    if tokens:
        checker.feed_tokens(tokens)
    checker.finish()
    log_correct_word.flush()  # last pending word group has no following result to trigger it
    console.print("[bold]Done.[/bold]")
