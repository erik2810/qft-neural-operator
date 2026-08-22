"""Extraction of anomalous dimensions from predicted correlators.

The physical read-out of the whole framework is not $W$ itself but the exponent it
carries. In log-log variables the correlator is a straight line,

.. math::
    \\log W = -2\\Delta_{\\mathrm{eff}}\\,\\log r + \\mathrm{const},

so $\\Delta_{\\mathrm{eff}}$ -- and hence $\\gamma = \\Delta\\beta_1\\beta_2 -
\\Delta_{\\mathrm{eff}}$ -- comes from a weighted least-squares slope. Everything here is
batched and closed-form, so it can run inside a validation step rather than only in
offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "SpectrumReport",
    "fit_log_slope",
    "effective_dimension_from_correlator",
    "anomalous_dimension_from_correlator",
    "summarize_spectrum",
]


def fit_log_slope(
    log_r: Tensor,
    log_w: Tensor,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Batched weighted least-squares fit of $\\log W$ against $\\log r$.

    Args:
        log_r: $\\log r$, shape ``(batch, points)``.
        log_w: $\\log W$, same shape.
        weights: Non-negative per-point weights, same shape; ``None`` weights uniformly.

    Returns:
        ``(slope, intercept)``, each of shape ``(batch,)``.

    Raises:
        ValueError: If the shapes disagree or fewer than two points carry weight.
    """
    if log_r.shape != log_w.shape:
        raise ValueError(f"shape mismatch: {tuple(log_r.shape)} vs {tuple(log_w.shape)}")
    if log_r.ndim != 2:
        raise ValueError(f"expected (batch, points), got {tuple(log_r.shape)}")
    w = torch.ones_like(log_r) if weights is None else weights.clamp_min(0.0)
    total = w.sum(dim=-1, keepdim=True)
    if bool((total.squeeze(-1) <= 0).any()):
        raise ValueError("every sample needs at least one positively weighted point")

    mean_x = (w * log_r).sum(-1, keepdim=True) / total
    mean_y = (w * log_w).sum(-1, keepdim=True) / total
    dx, dy = log_r - mean_x, log_w - mean_y
    variance = (w * dx * dx).sum(-1)
    covariance = (w * dx * dy).sum(-1)
    slope = covariance / variance.clamp_min(1e-30)
    intercept = mean_y.squeeze(-1) - slope * mean_x.squeeze(-1)
    return slope, intercept


def effective_dimension_from_correlator(
    log_r: Tensor,
    log_w: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """$\\Delta_{\\mathrm{eff}} = -\\tfrac{1}{2}\\,d\\log W/d\\log r$ from a log-log fit.

    Args:
        log_r: $\\log r$, shape ``(batch, points)``.
        log_w: $\\log W$, same shape.
        weights: Optional per-point weights.

    Returns:
        Effective dimensions of shape ``(batch,)``.
    """
    slope, _ = fit_log_slope(log_r, log_w, weights)
    return -0.5 * slope


def anomalous_dimension_from_correlator(
    log_r: Tensor,
    log_w: Tensor,
    free_dimension: float,
    weights: Tensor | None = None,
) -> Tensor:
    """$\\gamma = \\Delta\\beta_1\\beta_2 - \\Delta_{\\mathrm{eff}}$ from a fitted correlator.

    Args:
        log_r: $\\log r$, shape ``(batch, points)``.
        log_w: $\\log W$, same shape.
        free_dimension: The free-theory exponent $\\Delta\\beta_1\\beta_2$.
        weights: Optional per-point weights.

    Returns:
        Anomalous dimensions of shape ``(batch,)``.
    """
    return free_dimension - effective_dimension_from_correlator(log_r, log_w, weights)


@dataclass(frozen=True)
class SpectrumReport:
    """Accuracy of recovered anomalous dimensions.

    Attributes:
        mae: Mean absolute error in $\\gamma$.
        rmse: Root-mean-square error in $\\gamma$.
        r2: Coefficient of determination against the exact $\\gamma$.
        relative_mae: MAE divided by the spread of the exact $\\gamma$ -- the number that
            actually says whether the anomalous dimension has been resolved, since a
            model that predicts the free theory everywhere already achieves a small
            absolute error.
        per_family: ``family -> MAE`` breakdown, so generalization to GP-drawn theories
            can be read off separately from the analytic families.
    """

    mae: float
    rmse: float
    r2: float
    relative_mae: float
    per_family: dict[str, float]


def summarize_spectrum(
    predicted: Tensor,
    exact: Tensor,
    family: Tensor | None = None,
    family_names: tuple[str, ...] = (),
) -> SpectrumReport:
    """Compare predicted and exact anomalous dimensions.

    Args:
        predicted: Predicted $\\gamma$, shape ``(n,)``.
        exact: Exact $\\gamma$, shape ``(n,)``.
        family: Optional family indices, shape ``(n,)``.
        family_names: Names indexed by ``family``.

    Returns:
        A :class:`SpectrumReport`.

    Raises:
        ValueError: If ``predicted`` and ``exact`` disagree in shape.
    """
    if predicted.shape != exact.shape:
        raise ValueError(f"shape mismatch: {tuple(predicted.shape)} vs {tuple(exact.shape)}")
    error = predicted - exact
    mae = float(error.abs().mean())
    rmse = float(error.pow(2).mean().sqrt())
    spread = exact - exact.mean()
    denominator = float(spread.pow(2).sum())
    r2 = 1.0 - float(error.pow(2).sum()) / denominator if denominator > 0.0 else float("nan")
    scale = float(exact.std(correction=0))
    relative_mae = mae / scale if scale > 0.0 else float("nan")

    per_family: dict[str, float] = {}
    if family is not None and family_names:
        for index, name in enumerate(family_names):
            mask = family == index
            if bool(mask.any()):
                per_family[name] = float(error[mask].abs().mean())
    return SpectrumReport(
        mae=mae, rmse=rmse, r2=r2, relative_mae=relative_mae, per_family=per_family
    )
