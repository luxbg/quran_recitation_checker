from pathlib import Path

import pytest

from qrc.corpus.index import QuranCorpus, build_corpus

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CORPUS_PATH = MODELS_DIR / "ordered_quran_phonemes.json"


@pytest.fixture(scope="session")
def real_corpus() -> QuranCorpus:
    if not CORPUS_PATH.exists():
        pytest.skip(f"{CORPUS_PATH} not present -- run scripts/download_model.py first")
    return build_corpus(CORPUS_PATH)
