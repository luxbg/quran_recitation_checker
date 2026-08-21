#!/usr/bin/env python3
"""Download the gated Quran-Lab/zipformer_p-arabic-v3 model into ./models.

Requires HF_TOKEN env var (an approved-access Hugging Face token for this repo).
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Quran-Lab/zipformer_p-arabic-v3"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Only pull what the pipeline actually needs -- skip the large .pt / CoreML
# packages, which are irrelevant to the sherpa-onnx CPU runtime path.
ALLOW_PATTERNS = [
    "README.md",
    "LICENSE",
    "config.json",
    "tokens.txt",
    "phoneme_units.json",
    "ordered_quran_phonemes.json",
    "quran_text2phoneme.json",
    "zipformer_p_arabic_v3.onnx",
    "zipformer_p_arabic_v3.int8.onnx",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var not set. Export your approved-access HF token first.", file=sys.stderr)
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        token=token,
        local_dir=MODEL_DIR,
        allow_patterns=ALLOW_PATTERNS,
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
