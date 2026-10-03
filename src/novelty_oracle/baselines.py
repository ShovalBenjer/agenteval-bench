"""Baseline single-parameter auction mechanisms.

Neutral mechanism-design domain (no fintech): one indivisible item, n
bidders with private values, direct revelation.

  first_price   — highest bidder wins, pays their own bid (their value).
  second_price  — highest bidder wins, pays the second-highest bid.
  myerson_uniform — second-price with reserve 0.5: revenue-optimal for
                  i.i.d. U[0,1] regular values (Myerson's lemma). This is
                  the analytic ground truth Article 2 uses as its sanity
                  check: no candidate may beat it on iid-uniform.
  all_pay       — highest bidder wins, EVERY bidder pays their bid.
                  Genuinely behaviorally different from the price
                  baselines.
  third_price   — highest bidder wins, pays the THIRD-highest bid.
                  Behaviorally distinct from first/second-price, but
                  strictly worse revenue: the demo's "novel but not
                  better" case (-> NOT_INVENTION).

Tie-breaking everywhere: lowest bidder index wins. Deterministic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from .types import Mechanism

MYERSON_RESERVE = 0.5


def _top_two(values: tuple[float, ...]) -> tuple[int, float, float]:
    """Return (winner_index, highest_value, second_highest_value)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    winner = order[0]
    top = values[winner]
    second = values[order[1]] if len(order) > 1 else 0.0
    return winner, top, second


def _allocate_to(values: tuple[float, ...], winner: int | None) -> tuple[int, ...]:
    return tuple(1 if i == winner else 0 for i in range(len(values)))


def first_price_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    winner, _, _ = _top_two(values)
    return _allocate_to(values, winner)


def first_price_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
    winner, top, _ = _top_two(values)
    return tuple(top if i == winner else 0.0 for i in range(len(values)))


def second_price_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    winner, _, _ = _top_two(values)
    return _allocate_to(values, winner)


def second_price_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
    winner, _, second = _top_two(values)
    return tuple(second if i == winner else 0.0 for i in range(len(values)))


def myerson_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    winner, top, _ = _top_two(values)
    if top < MYERSON_RESERVE:
        return _allocate_to(values, None)
    return _allocate_to(values, winner)


def myerson_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
    winner, _, second = _top_two(values)
    if sum(allocation) == 0:
        return tuple(0.0 for _ in values)
    price = max(second, MYERSON_RESERVE)
    return tuple(price if i == winner else 0.0 for i in range(len(values)))


def all_pay_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    winner, _, _ = _top_two(values)
    return _allocate_to(values, winner)


def all_pay_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
    # Everyone pays their bid regardless of winning.
    return tuple(values)


def third_price_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
    winner, _, _ = _top_two(values)
    return _allocate_to(values, winner)


def third_price_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
    winner, _, _ = _top_two(values)
    ranked = sorted(values, reverse=True)
    third = ranked[2] if len(ranked) > 2 else 0.0
    return tuple(third if i == winner else 0.0 for i in range(len(values)))


def _source_of(*fns: Callable[..., object]) -> str:
    return "\n".join(inspect.getsource(fn) for fn in fns)


def first_price() -> Mechanism:
    return Mechanism(
        name="first-price",
        allocate=first_price_allocate,
        pay=first_price_pay,
        description="Highest bidder wins, pays own bid.",
        source=_source_of(first_price_allocate, first_price_pay),
    )


def second_price() -> Mechanism:
    return Mechanism(
        name="second-price",
        allocate=second_price_allocate,
        pay=second_price_pay,
        description="Highest bidder wins, pays second-highest bid.",
        source=_source_of(second_price_allocate, second_price_pay),
    )


def myerson_uniform() -> Mechanism:
    return Mechanism(
        name="myerson-uniform",
        allocate=myerson_allocate,
        pay=myerson_pay,
        description="Second-price with reserve 0.5; optimal for iid U[0,1].",
        source=_source_of(myerson_allocate, myerson_pay),
    )


def all_pay() -> Mechanism:
    return Mechanism(
        name="all-pay",
        allocate=all_pay_allocate,
        pay=all_pay_pay,
        description="Highest bidder wins; every bidder pays their bid.",
        source=_source_of(all_pay_allocate, all_pay_pay),
    )


def third_price() -> Mechanism:
    return Mechanism(
        name="third-price",
        allocate=third_price_allocate,
        pay=third_price_pay,
        description="Highest bidder wins, pays the third-highest bid.",
        source=_source_of(third_price_allocate, third_price_pay),
    )


def second_price_reserve(reserve: float) -> Mechanism:
    """Second-price with a reserve price: the EDA's frontier family."""
    base = second_price()

    def allocate(values: tuple[float, ...]) -> tuple[int, ...]:
        if max(values) < reserve:
            return tuple(0 for _ in values)
        return base.allocate(values)

    def pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
        if sum(allocation) == 0:
            return tuple(0.0 for _ in values)
        winner = allocation.index(1)
        base_allocation = base.allocate(values)
        base_payments = base.pay(values, base_allocation)
        price = max(base_payments[winner], reserve)
        return tuple(price if i == winner else 0.0 for i in range(len(values)))

    return Mechanism(
        name=f"reserve-{reserve:.1f}",
        allocate=allocate,
        pay=pay,
        description=f"Second-price with reserve {reserve}.",
        source=None,
    )


def all_baselines() -> tuple[Mechanism, ...]:
    return (first_price(), second_price(), myerson_uniform())
