import pytest
from parc.utils import get_mode


@pytest.mark.parametrize(
    "input_list, expected_mode",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 2, 6], [2]),
        (["a", "b", "c", "d", "e", "a", "b", "c"], ["a", "b", "c"])
    ]
)
def test_get_mode(input_list, expected_mode):
    assert get_mode(input_list) in expected_mode
