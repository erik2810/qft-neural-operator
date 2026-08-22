"""Weighted combination of the data, scaling and RG terms."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from qft_operator.losses.data import LogCorrelatorLoss
from qft_operator.losses.rg import RGInvarianceLoss
from qft_operator.losses.scaling import BoundaryScalingLoss
from qft_operator.physics.rg import BetaFunction

__all__ = ["LossWeights", "PhysicsInformedLoss"]


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the loss terms.

    Every physics term carries its own weight rather than being folded into a hardcoded
    constant, so an ablation is a config edit rather than a code edit.

    Args:
        data: Supervised $\\log W$ term.
        scaling: AdS2 boundary scaling term. Start small -- it involves second
            derivatives and dominates the gradient if over-weighted early.
        rg: Callan-Symanzik invariance term.
        warmup_epochs: Number of epochs over which the two physics weights ramp linearly
            from zero. Fitting the leading power law first and only then imposing the
            differential constraints is markedly more stable than switching everything
            on at step zero.
    """

    data: float = 1.0
    scaling: float = 0.01
    rg: float = 0.01
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        for name in ("data", "scaling", "rg"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"loss weight {name!r} must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")

    def physics_scale(self, epoch: int) -> float:
        """Linear warm-up factor in $[0, 1]$ applied to the scaling and RG weights."""
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, max(0.0, epoch / self.warmup_epochs))


class PhysicsInformedLoss(nn.Module):
    """Total training objective, returning the breakdown alongside the total.

    .. math::
        \\mathcal{L} = w_{\\mathrm{data}}\\,\\mathcal{L}_{\\mathrm{data}}
        + s(t)\\left[w_{\\mathrm{scale}}\\,\\mathcal{L}_{\\mathrm{scale}}
        + w_{\\mathrm{RG}}\\,\\mathcal{L}_{\\mathrm{RG}}\\right]

    with $s(t)$ the warm-up ramp of :meth:`LossWeights.physics_scale`.

    Args:
        weights: Term weights and warm-up schedule.
        data_loss: Supervised term; defaults to an MSE on $\\log W$.
        scaling_loss: Boundary scaling term; defaults to the label-free curvature form.
        rg_loss: RG invariance term; defaults to a marginal $\\beta$ function.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        data_loss: LogCorrelatorLoss | None = None,
        scaling_loss: BoundaryScalingLoss | None = None,
        rg_loss: RGInvarianceLoss | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.data_loss = data_loss or LogCorrelatorLoss()
        self.scaling_loss = scaling_loss or BoundaryScalingLoss()
        self.rg_loss = rg_loss or RGInvarianceLoss(BetaFunction())

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        prediction: Tensor,
        epoch: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate the total loss and its components.

        Args:
            model: The operator network (needed by the derivative-based terms).
            batch: Dict with keys ``v_phi``, ``coords``, ``log_w``, ``log_m``,
                ``delta_eff``, ``coupling``, ``dv_dcoupling``.
            prediction: Predicted $\\log W$, shape ``(batch, points)``.
            epoch: Current epoch, driving the physics warm-up ramp.

        Returns:
            ``(total, components)`` where ``components`` maps term names to detached
            scalars suitable for logging, plus the un-warmed ``"total"``.
        """
        components: dict[str, Tensor] = {}
        total = prediction.new_zeros(())

        data = self.data_loss(prediction, batch["log_w"])
        components["data"] = data.detach()
        total = total + self.weights.data * data

        ramp = self.weights.physics_scale(epoch)

        if self.weights.scaling > 0.0 and ramp > 0.0:
            scaling = self.scaling_loss(
                model,
                batch["v_phi"],
                batch["coords"],
                batch.get("log_m"),
                batch.get("delta_eff"),
            )
            components["scaling"] = scaling.detach()
            total = total + ramp * self.weights.scaling * scaling

        if self.weights.rg > 0.0 and ramp > 0.0:
            rg = self.rg_loss(
                model,
                batch["v_phi"],
                batch["coords"],
                batch["log_m"],
                batch["coupling"],
                batch["dv_dcoupling"],
                log_w=prediction.detach(),
            )
            components["rg"] = rg.detach()
            total = total + ramp * self.weights.rg * rg

        components["total"] = total.detach()
        return total, components
