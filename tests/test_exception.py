from onekit.exception import InvalidChoiceError


def test_invalid_choice_error():
    x = 0
    choices = [1, 2, 3]
    error = InvalidChoiceError(x, choices)
    actual = error.message
    expected = "x=0 invalid choice - choose from [1, 2, 3]"
    assert actual == expected
