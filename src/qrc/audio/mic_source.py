import queue
from typing import Iterator

import numpy as np
import sounddevice as sd


def iter_mic_chunks(sample_rate: int = 16000, block_ms: int = 100) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (float32 mono samples, sample_rate) chunks from the default mic.

    Runs until the caller stops iterating (e.g. on KeyboardInterrupt).
    """
    blocksize = int(sample_rate * block_ms / 1000)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=blocksize, callback=callback):
        while True:
            samples = q.get()
            yield samples, sample_rate
