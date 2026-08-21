from pathlib import Path
from typing import NamedTuple

import numpy as np
import sherpa_onnx


class PhonemeToken(NamedTuple):
    symbol: str
    time_s: float


class StreamingPhonemeRecognizer:
    """Wraps sherpa-onnx's streaming zipformer2-CTC recognizer.

    Deliberately: no hotwords, no context biasing, no LM. Biasing decoding
    toward the expected/canonical phonemes once an ayah is localized would
    suppress exactly the deviations this tool exists to detect.
    """

    def __init__(
        self,
        tokens_path: Path | str,
        model_path: Path | str,
        sample_rate: int = 16000,
        provider: str = "cpu",
        num_threads: int = 2,
    ):
        self._sample_rate = sample_rate
        self._rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(tokens_path),
            model=str(model_path),
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=False,  # continuous multi-ayah session
            decoding_method="greedy_search",  # only valid mode for this model;
            # matches its "no LM anywhere" design.
            provider=provider,
        )
        self._stream = self._rec.create_stream()
        self._emitted = 0

    def feed_audio(self, samples: np.ndarray, sample_rate: int | None = None) -> None:
        # sherpa-onnx resamples internally if this differs from the
        # recognizer's configured rate -- lets WAV sources at other native
        # rates feed straight in without a separate resampling step.
        self._stream.accept_waveform(sample_rate or self._sample_rate, samples)
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def poll_new_tokens(self) -> list[PhonemeToken]:
        toks = self._rec.tokens(self._stream)
        ts = self._rec.timestamps(self._stream)
        new_toks = toks[self._emitted :]
        new_ts = ts[self._emitted :]
        self._emitted = len(toks)
        return [PhonemeToken(symbol=s, time_s=t) for s, t in zip(new_toks, new_ts)]

    def finish(self) -> list[PhonemeToken]:
        self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        return self.poll_new_tokens()
