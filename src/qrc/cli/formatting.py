from rich.console import Console

from qrc.align.word_aligner import WordCheckResult

STATUS_STYLE = {
    "match": "green",
    "mismatch": "red",
    "deleted": "yellow",
}


def print_word_result(console: Console, result: WordCheckResult) -> None:
    style = STATUS_STYLE.get(result.status, "white")
    label = f"{result.surah}:{result.ayah} #{result.word_index}"
    actual = result.actual_phonemes if result.actual_phonemes is not None else "(nothing recited)"

    if result.status == "match":
        word = f" {result.word_text}" if result.word_text else ""
        console.print(f"[{style}]✓[/{style}] {label}{word} expected={result.expected_phonemes!r} actual={actual!r}")
        return

    console.print(
        f"[{style}]✗ {result.status.upper()}[/{style}] {label} "
        f"expected={result.expected_phonemes!r} actual={actual!r} similarity={result.similarity:.2f}"
    )


def print_status(console: Console, message: str) -> None:
    console.print(f"[dim]-- {message}[/dim]")
