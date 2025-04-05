from onekit.exception import (
    ColumnNotFoundError,
    InvalidChoiceError,
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
