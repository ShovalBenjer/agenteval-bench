"""CUPED variance reduction with pre-experiment covariates.

CUPED (Deng, Hu & Lu, 2013): Y_cuped = Y - theta * (X - E[X]), where theta
is the regression slope of the outcome on the pre-experiment covariate.
Two load-bearing choices, both test-pinned:

1. theta is estimated on the CONTROL group only. Estimating it on pooled
   or treatment data lets the treatment effect leak into the adjustment
   (the slope then mixes the covariate relationship with the effect).
2. The covariate must be declared pre-experiment. cuped_adjust refuses
   post-treatment covariates fail-closed: adjusting for a variable the
   treatment moved can absorb the effect being measured.

Unbiasedness rests on randomization: E[X_treated - X_control] = 0 for a
pre-experiment X, so the adjustment has mean zero and
E[adjusted_diff] = E[raw_diff].

Stdlib only; all functions pure and deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from experiment.types import CovariateSpec


class CovariateViolation(ValueError):
    """A covariate was used in a way that can bias the estimate (fail-closed)."""


@dataclass(frozen=True)
class CupedResult:
    """Outcome of the CUPED adjustment."""

    theta: float  #: control-group slope of Y on X
    raw_diff: float  #: mean(treated) - mean(control), unadjusted
    adjusted_diff: float  #: mean(treated) - mean(control), CUPED-adjusted
    var_raw: float  #: pooled variance of the raw outcome
    var_adjusted: float  #: pooled variance of the CUPED-adjusted outcome
    variance_ratio: float  #: var_adjusted / var_raw; < 1 means variance reduced
    n_control: int
    n_treated: int


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float], mean: float) -> float:
    return sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


def _cov(xs: Sequence[float], ys: Sequence[float], mx: float, my: float) -> float:
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def cuped_adjust(
    y_control: Sequence[float],
    y_treated: Sequence[float],
    x_control: Sequence[float],
    x_treated: Sequence[float],
    covariate: CovariateSpec,
) -> CupedResult:
    """Apply CUPED with a pre-experiment covariate.

    Raises CovariateViolation if the covariate is not declared
    pre-experiment. Raises ValueError on degenerate inputs (mismatched
    lengths, n < 2, zero covariate variance, zero outcome variance).
    """
    if covariate.source != "pre_experiment":
        raise CovariateViolation(
            f"covariate {covariate.name!r} is declared {covariate.source!r}: "
            "CUPED refuses post-treatment covariates fail-closed because the "
            "adjustment could absorb the treatment effect"
        )
    yc = [float(v) for v in y_control]
    yt = [float(v) for v in y_treated]
    xc = [float(v) for v in x_control]
    xt = [float(v) for v in x_treated]
    if not (len(yc) == len(xc) and len(yt) == len(xt)):
        raise ValueError("outcome and covariate lengths must match within each arm")
    if len(yc) < 2 or len(yt) < 2:
        raise ValueError(f"CUPED needs n >= 2 per arm, got {len(yc)} and {len(yt)}")
    mxc = _mean(xc)
    myc = _mean(yc)
    var_xc = _var(xc, mxc)
    if var_xc == 0.0:
        raise ValueError(f"covariate {covariate.name!r} has zero variance in control")
    # theta from the CONTROL group only: the treatment arm's X-Y slope can
    # carry the treatment effect, which would leak into the adjustment.
    theta = _cov(xc, yc, mxc, myc) / var_xc
    x_bar = (sum(xc) + sum(xt)) / (len(xc) + len(xt))
    yc_adj = [y - theta * (x - x_bar) for y, x in zip(yc, xc)]
    yt_adj = [y - theta * (x - x_bar) for y, x in zip(yt, xt)]
    raw_diff = _mean(yt) - _mean(yc)
    adjusted_diff = _mean(yt_adj) - _mean(yc_adj)
    all_raw = yc + yt
    all_adj = yc_adj + yt_adj
    var_raw = _var(all_raw, _mean(all_raw))
    if var_raw == 0.0:
        raise ValueError("outcome has zero variance")
    var_adjusted = _var(all_adj, _mean(all_adj))
    return CupedResult(
        theta=theta,
        raw_diff=raw_diff,
        adjusted_diff=adjusted_diff,
        var_raw=var_raw,
        var_adjusted=var_adjusted,
        variance_ratio=var_adjusted / var_raw,
        n_control=len(yc),
        n_treated=len(yt),
    )
