from qrc.align.normalize import phonemes_match


def test_collapses_madd_length_variants():
    # real case from a live session: 4-harakat vs 2-harakat madd, both legitimate
    assert phonemes_match("سَيَغلِبُۥۥۥۥن", "سَيَغلِبُۥۥن")
    assert phonemes_match("لمُءمِنُۥۥۥۥن", "لمُءمِنُۥۥن")


def test_forgives_trailing_short_vowel_dropped_at_pause():
    # real case: reference is fully-connected (wasl), real recitation paused (waqf)
    assert phonemes_match("سِنِۦۦنَ", "سِنِۦۦن")


def test_flags_single_mid_word_vowel_substitution_in_long_word():
    # real case from a live session: a single wrong vowel mid-word (damma
    # instead of fatha) in a 19-character word only drops normalized
    # similarity to ~0.95, clearing a proportional 0.90 threshold -- but
    # it's still a genuine phoneme mistake and must not match.
    assert not phonemes_match("غِشَااوَتُوووَلَهُم", "غِشَااوَتَوووَلَهُم")


def test_does_not_forgive_a_substituted_trailing_vowel():
    # real bug from a live session: reciting "raa" with damma instead of
    # fatha (a genuine phoneme-identity mistake, not a dropped vowel) must
    # NOT be silently erased just because both happen to end in *some*
    # short vowel. Independently stripping trailing vowels from both sides
    # made these collapse to the same string -- this is the regression.
    assert not phonemes_match("ءَكثَرَ", "ءَكثَرُ")


def test_does_not_collapse_distinct_adjacent_letters():
    # no repeated codepoint runs and no trailing short vowel -- must be unchanged
    assert phonemes_match("ءَحَدڇ", "ءَحَدڇ")


def test_still_distinguishes_genuine_substitution():
    # a real wrong word should not become equal after normalization
    assert not phonemes_match("للَااهِ", "للَشمسِ")
