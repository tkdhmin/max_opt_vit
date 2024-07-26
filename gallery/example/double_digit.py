from typing import TypeVar

T = TypeVar("T", int, float)


def double_digit(digit: T) -> T:
    """Return the double of the given integer or floating-point digit"""
    return digit * 2
