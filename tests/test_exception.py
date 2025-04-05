from onekit.exception import (
    ColumnNotFoundError,
    InvalidChoiceError,
    RowCountMismatchError,
)


def test_column_not_found_error():
    error = ColumnNotFoundError(missing_cols=["a", "b", "c"])
    actual = error.message
    expected = "following columns not found: ['a', 'b', 'c']"
    assert actual == expected


def test_invalid_choice_error():
    x = 0
    choices = [1, 2, 3]
    error = InvalidChoiceError(x, choices)
    actual = error.message
    expected = "x=0 invalid choice - choose from [1, 2, 3]"
    assert actual == expected


def test_row_count_mismatch_error():
    error = RowCountMismatchError(num_lft=10000, num_rgt=12000)
    actual = error.message
    expected = "num_lft=10_000, num_rgt=12_000, num_diff=2_000"
    assert actual == expected
