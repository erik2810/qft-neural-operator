"""Supervised data term on the log-correlator."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["LogCorrelatorLoss", "relative_l2"]

Reduction = Literal["mse", "huber", "rel_l2"]


def relative_l2(prediction: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-sample relative $L^2$ error, averaged over the batch.

    .. math::
        \\frac{1}{B}\\sum_b
        \\frac{\\lVert \\hat{y}_b - y_b \\rVert_2}{\\lVert y_b \\rVert_2 + \\varepsilon}

    This is the standard neural-operator reporting metric; unlike a plain MSE it is
    invariant to the overall scale of each sample, which matters here because different
    theories differ mainly in an exponent.

    Args:
        prediction: Predicted values, shape ``(batch, points)``.
        target: Ground truth, same shape.
        eps: Denominator floor.

    Returns:
        Scalar tensor.
    """
    numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1) + eps
    return (numerator / denominator).mean()


class LogCorrelatorLoss(nn.Module):
    """Supervised loss between predicted and exact $\\log W$.

    Working in log space is not a numerical nicety here: over the default separation
    window $W$ spans about eight decades, so an $L^2$ loss on $W$ itself is effectively
    evaluated at the two or three smallest separations. The baseline script regressed
    $W$ directly and paid exactly that price.

    Args:
        reduction: ``"mse"``, ``"huber"`` (robust to the residual outliers produced by
            near-coincident points), or ``"rel_l2"``.
        huber_delta: Transition point of the Huber loss.

    Shape:
        - ``prediction`` / ``target``: ``(batch, points)``
        - Output: scalar
    """

    def __init__(self, reduction: Reduction = "mse", huber_delta: float = 1.0) -> None:
        super().__init__()
        if reduction not in ("mse", "huber", "rel_l2"):
            raise ValueError(f"unknown reduction {reduction!r}")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.reduction = reduction
        self.huber_delta = huber_delta

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:  # noqa: D102
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: prediction {tuple(prediction.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        if self.reduction == "mse":
            return torch.nn.functional.mse_loss(prediction, target)
        if self.reduction == "huber":
            return torch.nn.functional.huber_loss(prediction, target, delta=self.huber_delta)
        return relative_l2(prediction, target)
