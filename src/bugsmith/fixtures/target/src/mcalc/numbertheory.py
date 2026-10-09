"""Elementary number theory."""

from __future__ import annotations


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(n**0.5) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial undefined for negatives")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("fibonacci undefined for negatives")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
