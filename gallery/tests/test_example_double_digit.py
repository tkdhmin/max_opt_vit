import pytest

from gallery.example.double_digit import double_digit


@pytest.mark.parametrize("input_value, expected_output", [(2, 4), (-3, -6), (2.5, 5.0), (-3.5, -7.0)])
def test_double_digit(input_value, expected_output):
    assert double_digit(input_value) == expected_output
