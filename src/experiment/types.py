"""Typed contracts for experiment discipline (agenteval-bench#32).

Source mechanism: Kohavi, Tang & Xu, *Trustworthy Online Controlled
Experiments* — peeking at sequential results without correction inflates
false positives; CUPED sharpens comparisons with pre-experiment
covariates. The contracts below make the discipline checkable: a plan is
a frozen record, a look schedule is exact, and evidence either replays
through the gate or is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: One-sided (upper tail) or two-sided (|z|) sequential test.
Sides = Literal["one", "two"]


@dataclass(frozen=True)
class CovariateSpec:
    """A covariate eligible for CUPED variance reduction.

    source="pre_experiment" means measured before treatment assignment,
    so it cannot carry the treatment effect. source="post_treatment" is
    REFUSED fail-closed by cuped_adjust: adjusting for a post-treatment
    variable can absorb the effect being measured.
    """

    name: str
    source: Literal["pre_experiment", "post_treatment"]


@dataclass(frozen=True)
class ExperimentPlan:
    """Pre-registered experiment design.

    looks: information fractions at which interim analyses are PERMITTED,
    strictly increasing, in (0, 1], last exactly 1.0. Fractions are
    compared exactly — an analysis at any unregistered fraction is
    peeking and is refused by SequentialGate.
    """

    name: str
    n_planned: int
    alpha: float = 0.05
    sides: Sides = "two"
    looks: tuple[float, ...] = (0.5, 1.0)
    covariate: CovariateSpec | None = None
    n_tolerance: float = 0.02
    min_n: int = 30

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plan name must be non-empty")
        if self.n_planned < self.min_n:
            raise ValueError(
                f"n_planned={self.n_planned} below minimum {self.min_n}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if len(self.looks) < 1:
            raise ValueError("need at least one look")
        prev = 0.0
        for t in self.looks:
            if not 0.0 < t <= 1.0:
                raise ValueError(f"look fractions must be in (0, 1], got {t}")
            if t <= prev:
                raise ValueError(
                    f"look fractions must be strictly increasing, got {self.looks}"
                )
            prev = t
        if self.looks[-1] != 1.0:
            raise ValueError(
                f"final look must be at information fraction 1.0, got {self.looks[-1]}"
            )
        if not 0.0 <= self.n_tolerance < 1.0:
            raise ValueError(f"n_tolerance must be in [0, 1), got {self.n_tolerance}")
