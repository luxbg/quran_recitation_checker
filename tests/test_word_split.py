from qrc.corpus.word_split import rebalance_shared_gemination, split_merged_unit


def test_splits_five_word_merge_32_8():
    # Real case: 32:8's last corpus phoneme-word merges 5 written words
    # ("مِن سُلَـٰلَةٍۢ مِّن مَّآءٍۢ مَّهِينٍۢ") into one unit -- this is
    # the case that motivated the whole per-written-word split feature.
    corpus_unit = "مِںںںسُلَاالَتِممممِممممَااااءِممممَهِۦۦۦۦن"
    qt_groups = ["مِںںں", "سُلَاالَتِن", "مِِن", "مََآءِن", "مََهِۦۦن"]

    result = split_merged_unit(corpus_unit, qt_groups)

    assert result.spans is not None
    assert len(result.spans) == 5
    assert "".join(result.spans) == corpus_unit
    assert result.confidence > 0.8


def test_low_similarity_still_returns_a_confidence_for_reporting():
    # Even a bad/unrelated pairing should report a confidence score (the
    # caller gates on it) rather than raising -- callers rely on this to
    # report the full score distribution, not just failures.
    result = split_merged_unit("قططططط", ["ب", "ي", "ت"])
    assert result.confidence < 0.5


def test_single_group_returns_the_whole_unit_unsplit():
    corpus_unit = "بِسمِ"
    result = split_merged_unit(corpus_unit, ["بِسمِ"])
    assert result.spans == [corpus_unit]
    assert result.confidence == 1.0


def test_rebalance_gives_one_shared_letter_back_to_the_first_word():
    # Real case (8:74): "لَّهُم مَّغْفِرَةٌۭ" -- idgham mutamathilain,
    # meem meeting meem. The raw split attributes the whole run to the
    # second word, leaving "لَّهُم" with no meem of its own at all.
    written = ["لَّهُم", "مَّغْفِرَةٌۭ"]
    spans = ["للَهُ", "ممممَغفِرَتُ"]

    rebalanced = rebalance_shared_gemination(written, spans)

    assert rebalanced == ["للَهُم", "مممَغفِرَتُ"]
    assert "".join(rebalanced) == "".join(spans)  # total content unchanged, just reassigned


def test_rebalance_does_not_touch_a_words_own_intrinsic_gemination():
    # "حَقًّۭا" ends in alif, not lam -- "لَّهُم"'s own leading double-lam
    # is its own intrinsic shadda, not shared with the previous word, and
    # must be left alone.
    written = ["حَقًّۭا", "لَّهُم"]
    spans = ["حَققَ", "للَهُ"]

    rebalanced = rebalance_shared_gemination(written, spans)

    assert rebalanced == spans


def test_rebalance_ignores_a_run_of_length_one():
    # Nothing to spare -- taking the only instance would leave the
    # second word with none of its own.
    written = ["مِن", "نَعَم"]
    spans = ["مِن", "نَعَم"]

    rebalanced = rebalance_shared_gemination(written, spans)

    assert rebalanced == spans


def test_reconstruction_covers_the_full_original_string():
    # Split spans must partition the *original* (uncollapsed) corpus
    # string exactly -- concatenating them back must reproduce it, since
    # the aligner verifies against these exact sub-strings.
    corpus_unit = "وَلَااكِللَاا"
    qt_groups = ["وَلَااكِ", "للَاا"]
    result = split_merged_unit(corpus_unit, qt_groups)
    assert result.spans is not None
    assert "".join(result.spans) == corpus_unit
