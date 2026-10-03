"""Novelty Oracle — Law Article 1: the "new AND better" measurement layer.

A typed service that judges candidate single-parameter auction mechanisms
against named baselines:
  quality = simulator payoff vs the best baseline (95% CI over seeds);
  novelty = BEHAVIORAL distance over the canonical valuation-profile
            battery (AST distance is only a cheap pre-filter).

Verdict: INVENTION / NOT_INVENTION / ABSTAIN, with evidence attached.
"""

from .baselines import (
    all_baselines,
    all_pay,
    first_price,
    myerson_uniform,
    second_price,
    second_price_reserve,
    third_price,
)
from .battery import BATTERY_VERSION, build_battery, category_counts
from .oracle import NoveltyOracle, decide_verdict
from .types import (
    Mechanism,
    OracleConfig,
    OracleVerdict,
    VerdictKind,
)

__all__ = [
    "BATTERY_VERSION",
    "Mechanism",
    "NoveltyOracle",
    "OracleConfig",
    "OracleVerdict",
    "VerdictKind",
    "all_baselines",
    "all_pay",
    "build_battery",
    "category_counts",
    "decide_verdict",
    "first_price",
    "myerson_uniform",
    "second_price",
    "second_price_reserve",
    "third_price",
]
