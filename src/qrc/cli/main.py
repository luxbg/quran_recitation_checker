from pathlib import Path

import typer

from qrc.cli.check_file import run_check_file
from qrc.cli.live import run_live

app = typer.Typer(help="Streaming phoneme-level Quran recitation checker.")


@app.command()
def live() -> None:
    """Check recitation live from the microphone."""
    run_live()


@app.command("check-file")
def check_file(
    wav_path: Path = typer.Argument(..., help="16-bit PCM WAV file to check."),
    realtime: bool = typer.Option(False, help="Pace playback to simulate real-time streaming."),
) -> None:
    """Check recitation from a pre-recorded WAV file (verification harness)."""
    run_check_file(wav_path, realtime=realtime)


if __name__ == "__main__":
    app()
