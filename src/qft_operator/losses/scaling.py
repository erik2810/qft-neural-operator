"""AdS2 boundary scaling loss: asymptotic conformal power-law behaviour."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from qft_operator.losses.operators import log_curvature, log_slope

__all__ = ["BoundaryScalingLoss"]

ScalingMode = Literal["power_law", "supervised", "both"]


class BoundaryScalingLoss(nn.Module):
    """Enforce $W \\sim |p_1 - p_2|^{-2\\Delta_{\\mathrm{eff}}}$ at large separation.

    Conformal invariance of the boundary theory fixes the large-$r$ behaviour of the
    two-point function to a pure power law. In log-log variables that statement is
    entirely local:

    .. math::
        \\frac{d\\log W}{d\\log r} = -2\\Delta_{\\mathrm{eff}}
        \\quad\\Longleftrightarrow\\quad
        \\frac{d^2 \\log W}{d(\\log r)^2} = 0 .

    Two ways to impose it are provided.

    ``"power_law"`` (default)
        Penalize the curvature. This needs **no labels at all** -- it says only "be a
        power law out there", not "have this particular exponent" -- so it is a genuine
        physics prior rather than a restatement of the data term, and it applies equally
        to theories whose $\\Delta_{\\mathrm{eff}}$ is unknown.
    ``"supervised"``
        Penalize $\\big(d\\log W/d\\log r + 2\\Delta_{\\mathrm{eff}}\\big)^2$ against a known
        exponent. Sharper, but only usable where a label exists.
    ``"both"``
        Sum of the two.

    The constraint is applied only for $r \\ge r_{\\min}$: at short separations the
    fixed-order targets carry genuine curvature from the residual $\\log(Mr)$
    dependence, and forcing a power law there would fight the data.

    Args:
        mode: One of ``"power_law"``, ``"supervised"``, ``"both"``.
        r_min: Separation above which the asymptotic constraint is imposed.
        create_graph: Retain the derivative graph so the loss is backpropagatable. Set
            ``False`` only for diagnostics.

    Shape:
        - Output: scalar
    """

    def __init__(
        self,
        mode: ScalingMode = "power_law",
        r_min: float = 1.0,
        create_graph: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("power_law", "supervised", "both"):
            raise ValueError(f"unknown mode {mode!r}")
        if r_min <= 0.0:
            raise ValueError(f"r_min must be positive, got {r_min}")
        self.mode = mode
        self.r_min = r_min
        self.create_graph = create_graph

    def _mask(self, coords: Tensor) -> Tensor:
        """Boolean selector for the asymptotic region $r \\ge r_{\\min}$."""
        r = (coords[..., 0] - coords[..., 1]).abs()
        return r >= self.r_min

    def forward(
        self,
        model: nn.Module,
        v_phi: Tensor,
        coords: Tensor,
        log_m: Tensor | None = None,
        delta_eff: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the scaling residual.

        Args:
            model: Operator network returning $\\log W$.
            v_phi: Potential samples, shape ``(batch, n_phi)``.
            coords: Boundary pairs, shape ``(batch, points, 2)``.
            log_m: $\\log M$ per sample.
            delta_eff: Reference exponents, shape ``(batch, points)``; required for the
                supervised modes.

        Returns:
            Scalar loss; exactly zero when no query point lies in the asymptotic region.

        Raises:
            ValueError: If a supervised mode is selected without ``delta_eff``.
        """
        if self.mode in ("supervised", "both") and delta_eff is None:
            raise ValueError(f"mode={self.mode!r} requires delta_eff")

        mask = self._mask(coords)
        if not bool(mask.any()):
            return coords.new_zeros(())

        total = coords.new_zeros(())
        if self.mode in ("power_law", "both"):
            curvature = log_curvature(model, v_phi, coords, log_m, create_graph=self.create_graph)
            total = total + (curvature[mask] ** 2).mean()
        if self.mode in ("supervised", "both"):
            assert delta_eff is not None  # narrowed by the guard above
            slope = log_slope(model, v_phi, coords, log_m, create_graph=self.create_graph)
            residual = slope + 2.0 * delta_eff
            total = total + (residual[mask] ** 2).mean()
        return total
