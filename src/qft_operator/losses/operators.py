"""Differential operators applied to the network, used by the physics-informed losses.

Two derivative structures are needed, and they call for different machinery.

*Derivatives with respect to a per-point coordinate* ($\\log r$, $\\log M$) exploit the
fact that the architecture keeps a **diagonal Jacobian**: $\\log W_p$ depends only on the
coordinates of query $p$ (this is why the boundary context field of
:class:`~qft_operator.models.trunk.BoundaryContextField` lives on an internal grid rather
than on the query axis). Differentiating the *sum* of the outputs therefore returns the
full per-point derivative in a single reverse pass, with no Jacobian assembly.

*Derivatives with respect to the coupling* $\\lambda$ are different: $\\lambda$ enters
through the whole branch input, so the chain rule reads

.. math::
    \\frac{\\partial W}{\\partial \\lambda}
    = \\sum_i \\frac{\\partial W}{\\partial V(\\phi_i)}\\,
      \\frac{\\partial V(\\phi_i)}{\\partial \\lambda},

which is a directional derivative along a known tangent -- exactly a JVP. Forward-mode
gets every output element in one pass; reverse mode would need a full Jacobian. A
central-difference fallback covers builds whose FFT kernels lack forward-mode rules.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["DirectionalDerivative", "log_slope", "log_curvature", "rebuild_coords"]

DerivativeMode = Literal["jvp", "fd"]


def rebuild_coords(coords: Tensor, log_r: Tensor) -> Tensor:
    """Rebuild boundary pairs from their midpoints and a new $\\log r$.

    Keeping the midpoint fixed and re-deriving $(p_1, p_2)$ from a differentiable
    $\\log r$ is what turns "differentiate with respect to the separation" into an
    ordinary autograd call.

    Args:
        coords: Original pairs, shape ``(batch, points, 2)``.
        log_r: Replacement log-separations, shape ``(batch, points)``.

    Returns:
        New coordinate pairs of shape ``(batch, points, 2)``.
    """
    midpoint = 0.5 * (coords[..., 0] + coords[..., 1])
    half = 0.5 * torch.exp(log_r)
    return torch.stack([midpoint - half, midpoint + half], dim=-1)


def log_slope(
    model: nn.Module,
    v_phi: Tensor,
    coords: Tensor,
    log_m: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor:
    """Per-point log-slope $d\\log W / d\\log r$.

    For an exact power law this equals $-2\\Delta_{\\mathrm{eff}}$, which is what
    :class:`~qft_operator.losses.scaling.BoundaryScalingLoss` and
    :mod:`qft_operator.analysis.spectrum` both consume.

    Args:
        model: An operator network returning $\\log W$ of shape ``(batch, points)``.
        v_phi: Potential samples, shape ``(batch, n_phi)``.
        coords: Boundary pairs, shape ``(batch, points, 2)``.
        log_m: $\\log M$ per sample.
        create_graph: Retain the graph so the result can be differentiated again
            (needed for :func:`log_curvature` and for backpropagating a loss built on
            the slope).

    Returns:
        The slope, shape ``(batch, points)``.
    """
    base = torch.log((coords[..., 0] - coords[..., 1]).abs().clamp_min(1e-12))
    log_r = base.detach().requires_grad_(True)
    with torch.enable_grad():
        log_w = model(v_phi, rebuild_coords(coords, log_r), log_m)
        (slope,) = torch.autograd.grad(log_w.sum(), log_r, create_graph=create_graph)
    return slope


def log_curvature(
    model: nn.Module,
    v_phi: Tensor,
    coords: Tensor,
    log_m: Tensor | None = None,
    create_graph: bool = True,
) -> Tensor:
    """Per-point log-curvature $d^2\\log W / d(\\log r)^2$.

    Vanishes identically for a pure power law, which makes it a *label-free* way to
    enforce asymptotic conformal scaling.

    Args:
        model: An operator network returning $\\log W$.
        v_phi: Potential samples, shape ``(batch, n_phi)``.
        coords: Boundary pairs, shape ``(batch, points, 2)``.
        log_m: $\\log M$ per sample.
        create_graph: Retain the graph for backpropagation.

    Returns:
        The curvature, shape ``(batch, points)``.
    """
    base = torch.log((coords[..., 0] - coords[..., 1]).abs().clamp_min(1e-12))
    log_r = base.detach().requires_grad_(True)
    with torch.enable_grad():
        log_w = model(v_phi, rebuild_coords(coords, log_r), log_m)
        (slope,) = torch.autograd.grad(log_w.sum(), log_r, create_graph=True)
        (curvature,) = torch.autograd.grad(slope.sum(), log_r, create_graph=create_graph)
    return curvature


class DirectionalDerivative:
    """Evaluate $\\frac{d}{ds}f(x + s\\,t)\\big|_{s=0}$ by forward-mode AD or differences.

    Args:
        mode: ``"jvp"`` for forward-mode automatic differentiation (exact) or ``"fd"``
            for a central difference.
        step: Step size used by the finite-difference path.
        allow_fallback: Silently downgrade to ``"fd"`` (once, with a warning) if
            forward-mode AD raises -- some FFT kernels have no forward-mode rule.

    Note:
        Once a fallback happens the instance stays in ``"fd"`` mode for the rest of its
        life, so a training run does not pay the failed-JVP cost on every step.
    """

    def __init__(
        self,
        mode: DerivativeMode = "jvp",
        step: float = 1e-3,
        allow_fallback: bool = True,
    ) -> None:
        if mode not in ("jvp", "fd"):
            raise ValueError(f"mode must be 'jvp' or 'fd', got {mode!r}")
        if step <= 0.0:
            raise ValueError(f"step must be positive, got {step}")
        self.mode = mode
        self.step = step
        self.allow_fallback = allow_fallback

    def __call__(
        self,
        fn: Callable[..., Tensor],
        primals: Sequence[Tensor],
        tangents: Sequence[Tensor],
    ) -> Tensor:
        """Apply the directional derivative.

        Args:
            fn: Function of the primal tensors returning a single tensor.
            primals: Evaluation point.
            tangents: Direction, matching ``primals`` in length, shape and dtype.

        Returns:
            The directional derivative, shaped like ``fn``'s output.

        Raises:
            ValueError: If ``primals`` and ``tangents`` disagree in length.
        """
        if len(primals) != len(tangents):
            raise ValueError("primals and tangents must have the same length")
        if self.mode == "jvp":
            try:
                # jvp returns (primal_out, tangent_out); index rather than unpack so the
                # call type-checks against the overloaded stub.
                result = torch.func.jvp(fn, tuple(primals), tuple(tangents))
                return result[1]
            except (RuntimeError, NotImplementedError) as exc:
                if not self.allow_fallback:
                    raise
                warnings.warn(
                    f"forward-mode AD unavailable for this model ({exc}); falling back "
                    "to central differences for the remainder of the run",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.mode = "fd"
        plus = fn(*[p + self.step * t for p, t in zip(primals, tangents, strict=True)])
        minus = fn(*[p - self.step * t for p, t in zip(primals, tangents, strict=True)])
        return (plus - minus) / (2.0 * self.step)
