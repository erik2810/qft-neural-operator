"""Callbacks reporting physics diagnostics during training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from qft_operator.analysis.spectrum import (
    SpectrumReport,
    anomalous_dimension_from_correlator,
    summarize_spectrum,
)

try:  # pragma: no cover
    from lightning import Callback
except ImportError:  # pragma: no cover
    Callback = object  # type: ignore[assignment,misc]

__all__ = ["SpectrumCallback", "FreeTheoryProbe"]


class SpectrumCallback(Callback):  # type: ignore[misc]
    """Aggregate the anomalous-dimension spectrum over the whole validation split.

    The per-batch metric logged by
    :class:`~qft_operator.training.module.QFTOperatorModule` normalizes by the spread
    *within a batch*, which fluctuates. This callback accumulates every validation
    sample and reports the split-level $R^2$ and per-family MAE -- in particular the
    GP-family number, which is the one that says whether the operator generalizes beyond
    the analytic theories it was partly trained on.

    Args:
        free_dimension: $\\Delta\\beta_1\\beta_2$ used for the log-log read-out.
        family_names: Names indexed by the batch's ``family`` entries.
    """

    def __init__(self, free_dimension: float, family_names: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.free_dimension = free_dimension
        self.family_names = family_names
        self._predicted: list[Tensor] = []
        self._exact: list[Tensor] = []
        self._family: list[Tensor] = []
        self.last_report: SpectrumReport | None = None

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: D102
        self._predicted.clear()
        self._exact.clear()
        self._family.clear()

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # noqa: D102
        with torch.no_grad():
            prediction = pl_module.model(batch["v_phi"], batch["coords"], batch["log_m"])
            log_r = torch.log((batch["coords"][..., 0] - batch["coords"][..., 1]).abs())
            gamma = anomalous_dimension_from_correlator(log_r, prediction, self.free_dimension)
        self._predicted.append(gamma.detach().cpu())
        self._exact.append(batch["gamma"].detach().cpu())
        self._family.append(batch["family"].detach().cpu())

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D102
        if not self._predicted:
            return
        report = summarize_spectrum(
            torch.cat(self._predicted),
            torch.cat(self._exact),
            torch.cat(self._family),
            self.family_names,
        )
        self.last_report = report
        metrics = {
            "spectrum/mae": report.mae,
            "spectrum/rmse": report.rmse,
            "spectrum/r2": report.r2,
            "spectrum/relative_mae": report.relative_mae,
            "spectrum/median_relative_error": report.median_relative_error,
        }
        metrics.update({f"spectrum/mae_{k}": v for k, v in report.per_family.items()})
        pl_module.log_dict(metrics, sync_dist=True)


class FreeTheoryProbe(Callback):  # type: ignore[misc]
    """Track how far the model drifts from the exact free-theory correlator.

    Feeding $V \\equiv 0$ must return $\\log W = -2\\Delta\\beta_1\\beta_2\\log r$ exactly.
    Unlike a validation metric this probe uses no data at all -- it is a direct check of
    an exact limit of the operator, and a monotonically growing value is a reliable early
    signal that the physics terms are being crowded out by the data term.

    Args:
        free_dimension: $\\Delta\\beta_1\\beta_2$.
        n_phi: Field-grid resolution of the probe input.
        r_min: Smallest probe separation.
        r_max: Largest probe separation.
        n_points: Number of probe separations.
    """

    def __init__(
        self,
        free_dimension: float,
        n_phi: int = 64,
        r_min: float = 0.05,
        r_max: float = 12.0,
        n_points: int = 64,
    ) -> None:
        super().__init__()
        self.free_dimension = free_dimension
        self.n_phi = n_phi
        log_r = torch.linspace(torch.tensor(r_min).log(), torch.tensor(r_max).log(), n_points)
        self.register_probe(log_r)

    def register_probe(self, log_r: Tensor) -> None:
        """Cache the probe coordinates and their exact free-theory correlator."""
        radii = torch.exp(log_r)
        self.coords = torch.stack([torch.zeros_like(radii), radii], dim=-1).unsqueeze(0)
        self.exact = (-2.0 * self.free_dimension * log_r).unsqueeze(0)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D102
        device = pl_module.device
        v_zero = torch.zeros(1, self.n_phi, device=device)
        with torch.no_grad():
            prediction = pl_module.model(
                v_zero, self.coords.to(device), torch.zeros(1, 1, device=device)
            )
        deviation = (prediction - self.exact.to(device)).abs()
        pl_module.log_dict(
            {"free_probe/max_abs": deviation.max(), "free_probe/mean_abs": deviation.mean()},
            sync_dist=True,
        )
