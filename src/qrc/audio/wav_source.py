import wave
from pathlib import Path
from typing import Iterator

import numpy as np


def iter_wav_chunks(path: Path | str, chunk_ms: int = 100) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (float32 mono samples in [-1, 1], native_sample_rate) chunks from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        if sampwidth != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV, got sampwidth={sampwidth}")

        chunk_frames = max(1, int(sample_rate * chunk_ms / 1000))
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            yield samples, sample_rate
