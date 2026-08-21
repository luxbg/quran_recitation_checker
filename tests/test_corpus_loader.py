import pytest

from qrc.corpus.models import AyahRef

from conftest import MODELS_DIR


def test_loads_all_6236_ayahs(real_corpus):
    assert len(real_corpus.ayahs_in_order) == 6236


def test_sorted_by_surah_then_ayah(real_corpus):
    refs = [a.ref for a in real_corpus.ayahs_in_order]
    assert refs == sorted(refs, key=lambda r: (r.surah, r.ayah))


def test_al_fatiha_first_ayah_word_segmentation(real_corpus):
    entry = next(a for a in real_corpus.ayahs_in_order if a.ref == AyahRef(1, 1))
    assert len(entry.words) == 4
    assert "".join(entry.words) != ""
    # aya_phoneme should be the space-joined words
    assert entry.aya_phoneme.replace(" ", "") == "".join(entry.words)


def test_global_words_cover_every_ayah_word(real_corpus):
    total_words = sum(len(a.words) for a in real_corpus.ayahs_in_order)
    assert len(real_corpus.global_words) == total_words


def test_char_offsets_align_with_corpus_text(real_corpus):
    for gw, offset in list(zip(real_corpus.global_words, real_corpus.char_offsets))[:20]:
        assert real_corpus.corpus_text[offset : offset + len(gw.phoneme_text)] == gw.phoneme_text


def test_word_text_loaded_when_map_present(real_corpus):
    if not (MODELS_DIR / "word_text_map.json").exists():
        pytest.skip("word_text_map.json not present -- run scripts/build_word_text_map.py first")

    entry = next(w for w in real_corpus.global_words if w.surah == 112 and w.ayah == 1 and w.local_word_idx == 0)
    assert entry.word_text == "قُلْ"

    resolved = sum(1 for w in real_corpus.global_words if w.word_text is not None)
    assert resolved / len(real_corpus.global_words) > 0.95
