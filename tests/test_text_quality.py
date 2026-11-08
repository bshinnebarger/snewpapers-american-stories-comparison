import pytest

from snewpapers_tools.text_quality import is_real_word, real_word_rate, words


def test_words_are_runs_of_three_or_more_letters():
    assert words("To the CITIZENS, of 1798: a vS ss Patskl!") == ["the", "CITIZENS", "Patskl"]


def test_a_real_word_is_in_the_english_list_in_any_case():
    assert is_real_word("CITIZENS") and is_real_word("unanimity")
    assert not is_real_word("lGnumcnt") and not is_real_word("firmnels")


def test_the_long_s_read_as_f_still_counts_as_a_real_word():
    assert is_real_word("neceffity") and is_real_word("fource") and is_real_word("thofe") and is_real_word("felf")
    assert is_real_word("firmness")  # real f's stay f


def test_real_word_rate_is_the_share_of_words_that_are_real():
    assert real_word_rate("the firm and undaunted Patskl lGnumcnt") == pytest.approx(4 / 6)
    assert real_word_rate("a I 12 ss") is None  # nothing to judge


def test_word_counts_give_real_and_all_words_for_summing():
    from snewpapers_tools.text_quality import word_counts

    assert word_counts("the firm and undaunted Patskl lGnumcnt") == (4, 6)
    assert word_counts("") == (0, 0)
