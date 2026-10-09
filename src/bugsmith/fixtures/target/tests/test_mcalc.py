"""Tests for the mcalc fixture target (bugsmith demo substrate)."""

from __future__ import annotations

import pytest
from mcalc.arithmetic import add, clamp, div, mul, sign, sub
from mcalc.numbertheory import factorial, fibonacci, gcd, is_prime
from mcalc.stats import data_range, mean, median, variance


def test_add() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_sub() -> None:
    assert sub(5, 3) == 2
    assert sub(0, 4) == -4


def test_mul() -> None:
    assert mul(3, 4) == 12
    assert mul(-2, 5) == -10


def test_div() -> None:
    assert div(7, 2) == 3.5
    with pytest.raises(ZeroDivisionError):
        div(1, 0)


def test_clamp() -> None:
    assert clamp(5, 0, 10) == 5
    assert clamp(-3, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
    with pytest.raises(ValueError):
        clamp(5, 10, 0)


def test_sign() -> None:
    assert sign(4.2) == 1
    assert sign(-0.1) == -1
    assert sign(0.0) == 0


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    with pytest.raises(ValueError):
        mean([])


def test_median_odd() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0


def test_median_even() -> None:
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_median_empty() -> None:
    with pytest.raises(ValueError):
        median([])


def test_variance() -> None:
    assert variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(4.57142857)
    with pytest.raises(ValueError):
        variance([1.0])


def test_data_range() -> None:
    assert data_range([3.0, 1.0, 9.0, 2.0]) == 8.0
    with pytest.raises(ValueError):
        data_range([])


def test_is_prime() -> None:
    assert is_prime(2)
    assert is_prime(13)
    assert not is_prime(1)
    assert not is_prime(15)
    assert not is_prime(0)


def test_factorial() -> None:
    assert factorial(0) == 1
    assert factorial(5) == 120
    with pytest.raises(ValueError):
        factorial(-1)


def test_gcd() -> None:
    assert gcd(12, 8) == 4
    assert gcd(-12, 8) == 4
    assert gcd(7, 13) == 1


def test_fibonacci() -> None:
    assert [fibonacci(n) for n in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
    with pytest.raises(ValueError):
        fibonacci(-2)
