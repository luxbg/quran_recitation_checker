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


def test_global_words_cover_every_real_written_word(real_corpus):
    # A merged corpus phoneme-word unit explodes into one GlobalWordEntry
    # per real written word (see build_corpus / scripts/build_word_text_map.py),
    # so global_words no longer tracks 1:1 with corpus phoneme-word units
    # -- it should track 1:1 with real written words instead, for every
    # ayah the offline precompute could resolve. An ayah it couldn't
    # resolve (word_text is None) falls back to one row per original
    # corpus unit, unchanged from before the split feature existed.
    if not (MODELS_DIR / "word_text_map.json").exists():
        pytest.skip("word_text_map.json not present -- run scripts/build_word_text_map.py first")

    rows_by_ayah: dict[tuple[int, int], list] = {}
    for w in real_corpus.global_words:
        rows_by_ayah.setdefault((w.surah, w.ayah), []).append(w)

    for ayah in real_corpus.ayahs_in_order:
        rows = rows_by_ayah[(ayah.ref.surah, ayah.ref.ayah)]
        resolved = all(r.word_text is not None for r in rows)
        if not resolved:
            assert len(rows) == len(ayah.words), f"{ayah.ref.surah}:{ayah.ref.ayah}"
            continue
        # Collapse continues_previous runs (a muqatta'at split, e.g.
        # "الٓمٓ" spanning two rows) into single logical words before
        # comparing -- those rows are the *same* written word, not
        # separate ones, whether or not this ayah also had a merge split.
        logical_words = [r.word_text for r in rows if not r.word_text_continues_previous]
        assert logical_words == ayah.aya_text.split(), f"{ayah.ref.surah}:{ayah.ref.ayah}"


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


def test_merged_ayah_32_8_splits_into_five_independent_words(real_corpus):
    # The motivating case: this corpus phoneme-word used to cover 5
    # written words as one indivisible verification unit. It must now be
    # 5 separate GlobalWordEntry rows, each with its own real phoneme
    # sub-span, correctly ordered.
    if not (MODELS_DIR / "word_text_map.json").exists():
        pytest.skip("word_text_map.json not present -- run scripts/build_word_text_map.py first")

    ayah = next(a for a in real_corpus.ayahs_in_order if a.ref.surah == 32 and a.ref.ayah == 8)
    words = [w for w in real_corpus.global_words if w.surah == 32 and w.ayah == 8]
    assert [w.word_text for w in words] == ayah.aya_text.split()
    assert [w.local_word_idx for w in words] == list(range(8))
    assert "".join(w.phoneme_text for w in words[3:8]) == "مِںںںسُلَاالَتِممممِممممَااااءِممممَهِۦۦۦۦن"
