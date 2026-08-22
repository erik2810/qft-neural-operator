"""Renormalization-group flow of the bulk coupling and the holographic scale.

Under the holographic dictionary the near-boundary cutoff is the renormalization scale,
$\\epsilon = 1/M$, so the logarithm produced by the bulk contact integral is
$\\log(M r)$. Physical observables must not depend on the arbitrary $M$:

.. math::
    \\left(M \\frac{\\partial}{\\partial M}
          + \\beta(\\lambda) \\frac{\\partial}{\\partial \\lambda}\\right) W = 0.

This module supplies the $\\beta$ function and the flow map used both to *build* targets
that satisfy that equation identically and to *evaluate* the residual as a training loss
(:class:`qft_operator.losses.rg.RGInvarianceLoss`).

Why the targets are exactly invariant
-------------------------------------
Writing the running coupling at the physical scale $\\mu = 1/r$,

.. math::
    \\bar\\lambda(1/r) = \\mathcal{F}\\big(\\lambda(M),\\, \\log(1/r) - \\log M\\big),

the flow's group property $\\mathcal{F}(\\mathcal{F}(\\lambda, a), b) = \\mathcal{F}(\\lambda,
a + b)$ makes $\\bar\\lambda(1/r)$ independent of which scale $M$ the coupling was quoted
at. Any correlator built as a function of $\\bar\\lambda(1/r)$ and $r$ alone therefore
annihilates the Callan-Symanzik operator **exactly**, for any $\\beta$ -- not just to
leading order. The datasets are constructed that way, so the RG loss and the data loss
never pull against each other.

With the default $\\epsilon = 0$ (a marginal coupling) the flow is trivial and $W$ is
simply $M$-independent. That is still a non-vacuous constraint for the network, because
$\\log M$ enters the trunk as an input feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["RGConfig", "BetaFunction", "running_coupling", "scale_anomalous_dimension"]


@dataclass(frozen=True)
class RGConfig:
    """Parameters of the bulk coupling's RG flow.

    The $\\beta$ function is truncated at two loops,

    .. math::
        \\beta(\\lambda) \\equiv \\frac{d\\lambda}{d\\log\\mu}
        = -\\epsilon\\,\\lambda + b\\,\\lambda^2,

    with $\\epsilon = (d + 1) - \\Delta_V$ the classical dimension deficit of the bulk
    interaction (so $\\epsilon = 2 - \\Delta_V$ in AdS2) and $b$ a one-loop coefficient.

    Args:
        reference_scale: The scale $M_0$ at which sampled couplings are quoted.
        epsilon: Classical deficit $\\epsilon$. ``0.0`` (marginal) is the default and
            reproduces the pure power-law correlators of the baseline script.
        two_loop: Quadratic coefficient $b$. Non-zero values switch the flow map from
            its closed form to RK4 integration.
        log_scale_jitter: Half-width of the uniform window in $\\log M$ from which the
            data pipeline draws per-sample renormalization scales. Zero pins every
            sample to $M_0$ and makes the RG loss degenerate.
        rk_steps: Number of RK4 steps used when ``two_loop != 0``.

    Raises:
        ValueError: If ``reference_scale <= 0``, the jitter is negative, or
            ``rk_steps < 1``.
    """

    reference_scale: float = 1.0
    epsilon: float = 0.0
    two_loop: float = 0.0
    log_scale_jitter: float = 0.75
    rk_steps: int = 32

    def __post_init__(self) -> None:
        if self.reference_scale <= 0.0:
            raise ValueError(f"reference_scale must be positive, got {self.reference_scale}")
        if self.log_scale_jitter < 0.0:
            raise ValueError(f"log_scale_jitter must be non-negative, got {self.log_scale_jitter}")
        if self.rk_steps < 1:
            raise ValueError(f"rk_steps must be >= 1, got {self.rk_steps}")

    @property
    def log_reference_scale(self) -> float:
        """$\\log M_0$."""
        return math.log(self.reference_scale)

    @property
    def is_marginal(self) -> bool:
        """True when the coupling does not run at all ($\\epsilon = b = 0$)."""
        return self.epsilon == 0.0 and self.two_loop == 0.0


class BetaFunction:
    """The truncated $\\beta$ function together with its flow map.

    Args:
        config: The :class:`RGConfig` supplying $\\epsilon$, $b$ and the integrator
            resolution.
    """

    def __init__(self, config: RGConfig | None = None) -> None:
        self.config = config or RGConfig()

    def __call__(self, lam: Tensor) -> Tensor:
        """Evaluate $\\beta(\\lambda) = -\\epsilon\\lambda + b\\lambda^2$ elementwise."""
        return -self.config.epsilon * lam + self.config.two_loop * lam * lam

    @property
    def has_closed_form(self) -> bool:
        """True when the flow can be integrated in closed form (``two_loop == 0``)."""
        return self.config.two_loop == 0.0

    def run(self, lam: Tensor, d_log_mu: Tensor) -> Tensor:
        """Transport a coupling by ``d_log_mu`` units of $\\log\\mu$.

        For $b = 0$ this is the closed form $\\lambda\\,e^{-\\epsilon\\,\\Delta\\log\\mu}$;
        otherwise the ODE $d\\lambda/d\\log\\mu = \\beta(\\lambda)$ is integrated with RK4,
        which preserves the group property to the integrator's order and keeps the
        construction differentiable.

        Args:
            lam: Coupling at the starting scale, any shape.
            d_log_mu: Signed distance $\\log\\mu_{\\mathrm{end}} - \\log\\mu_{\\mathrm{start}}$,
                broadcastable against ``lam``.

        Returns:
            The transported coupling.
        """
        if self.has_closed_form:
            return lam * torch.exp(-self.config.epsilon * d_log_mu)

        steps = self.config.rk_steps
        h = d_log_mu / steps
        value = lam
        for _ in range(steps):
            k1 = self(value)
            k2 = self(value + 0.5 * h * k1)
            k3 = self(value + 0.5 * h * k2)
            k4 = self(value + h * k3)
            value = value + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return value


def running_coupling(
    lam: Tensor,
    log_m: Tensor,
    log_r: Tensor,
    beta: BetaFunction,
) -> Tensor:
    """Coupling at the physical scale $\\mu = 1/r$, given $\\lambda$ quoted at $M$.

    The transport distance is $\\Delta\\log\\mu = \\log(1/r) - \\log M = -(\\log r + \\log M)$,
    so for a linear $\\beta$ this is $\\bar\\lambda = \\lambda\\,(Mr)^{\\epsilon}$.

    Args:
        lam: Coupling $\\lambda(M)$.
        log_m: $\\log M$, broadcastable against ``lam``.
        log_r: $\\log r$, broadcastable against ``lam``.
        beta: The :class:`BetaFunction` to integrate.

    Returns:
        $\\bar\\lambda(1/r)$, independent of $M$ by the group property of the flow.
    """
    return beta.run(lam, -(log_m + log_r))


def scale_anomalous_dimension(
    gamma_reference: Tensor,
    lam_reference: Tensor,
    log_m: Tensor,
    log_r: Tensor,
    beta: BetaFunction,
) -> Tensor:
    """Transport a first-order anomalous dimension to the scale $\\mu = 1/r$.

    Every potential in :mod:`qft_operator.physics.potentials` is linear in its coupling,
    hence so is $\\gamma$ at first order; running $\\gamma$ therefore amounts to rescaling
    it by $\\bar\\lambda/\\lambda$.

    Args:
        gamma_reference: $\\gamma$ evaluated with $\\lambda(M)$.
        lam_reference: The coupling $\\lambda(M)$ itself.
        log_m: $\\log M$.
        log_r: $\\log r$.
        beta: The :class:`BetaFunction`.

    Returns:
        $\\gamma$ at scale $1/r$; free theories ($\\lambda = 0$) map to zero without
        producing a division by zero.
    """
    if beta.config.is_marginal:
        return gamma_reference.expand(torch.broadcast_shapes(gamma_reference.shape, log_r.shape))
    lam_bar = running_coupling(lam_reference, log_m, log_r, beta)
    safe = torch.where(lam_reference.abs() > 0, lam_reference, torch.ones_like(lam_reference))
    ratio = torch.where(lam_reference.abs() > 0, lam_bar / safe, torch.ones_like(safe))
    return gamma_reference * ratio
