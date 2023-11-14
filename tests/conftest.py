from typing import Tuple

import pytest


@pytest.fixture(scope="module")
def n10() -> Tuple[int]:
    return 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
