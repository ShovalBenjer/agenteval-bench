"""Descriptive statistics."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("variance needs at least two values")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def data_range(values: list[float]) -> float:
    if not values:
        raise ValueError("range of empty sequence")
    return max(values) - min(values)
