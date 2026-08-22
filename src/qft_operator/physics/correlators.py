"""Boundary connected correlators $W(p_1, p_2)$ and their anomalous dimensions.

The functional the operator network learns is

.. math::
    S[\\phi] \\longmapsto W[J], \\qquad
    V(\\phi) \\longmapsto W(p_1, p_2)
      = \\langle V_{\\beta_1}(p_1) V_{\\beta_2}(p_2)\\rangle_{\\mathrm{conn}}.

First-order anomalous dimension
-------------------------------
Expanding $e^{-S_{\\mathrm{int}}}$ to first order, the only term contributing to the
*connected* vertex-operator two-point function is the one where exactly one
bulk-to-boundary propagator reaches each insertion. That term carries $\\beta_1\\beta_2
V''(\\phi)$ with all remaining fields self-contracted, and multiplies the regulated
contact integral of :mod:`qft_operator.physics.bulk_integrals`:

.. math::
    \\frac{\\delta W}{W^{(0)}} = \\beta_1\\beta_2\\,\\langle V''\\rangle_\\sigma\\,
    C_{\\log}\\left[\\log(M r) + \\kappa_\\Delta\\right],

whose logarithm exponentiates into a shifted power law, giving

.. math::
    \\boxed{\\,\\gamma[V] = \\tfrac{1}{2}\\,\\beta_1\\beta_2\\,
      \\langle V'' \\rangle_\\sigma\\, C_{\\log}\\,}
    \\qquad
    W(r) = r^{-2\\Delta_{\\mathrm{eff}}}, \\quad
    \\Delta_{\\mathrm{eff}} = \\Delta\\beta_1\\beta_2 - \\gamma.

For Sine-Gordon with a normal-ordered vertex ($\\sigma^2 = 0$) this reduces to
$\\langle V''\\rangle = -2\\lambda\\xi^2$ and hence

.. math::
    \\gamma = -\\lambda\\,\\frac{2 L^2 c_\\Delta}{2\\Delta - 1}\\,\\beta_1\\beta_2\\,\\xi^2,

the published expression. The general functional form is what makes GP-sampled and
polynomial potentials carry exact labels rather than heuristics: the free theory gets
$\\gamma = 0$ identically, and a normal-ordered $\\phi^4$ correctly gets $\\gamma = 0$
at first order in the $\\beta_1\\beta_2$ channel because $\\langle\\phi^2\\rangle_0 = 0$.

Scheme dependence
-----------------
$\\sigma^2 = G_\\Delta(x, x)$ is the renormalized coincident-point bulk propagator. It is
a genuine scheme choice, not a derived constant, so it lives in
:class:`~qft_operator.physics.config.PhysicsConfig` as a free parameter; ``sigma_sq=0``
is the normal-ordering scheme in which the reference result is quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qft_operator.physics.bulk_integrals import ConformalIntegrator
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.potentials import Potential
from qft_operator.physics.rg import BetaFunction, scale_anomalous_dimension

__all__ = [
    "CorrelatorTargets",
    "anomalous_dimension",
    "anomalous_dimension_from_moment",
    "boundary_two_point",
    "log_boundary_two_point",
    "first_order_log_correlator",
    "numerical_log_coefficient",
]


def anomalous_dimension_from_moment(moment: float, config: PhysicsConfig) -> float:
    """$\\gamma = \\tfrac{1}{2}\\beta_1\\beta_2\\,\\langle V''\\rangle_\\sigma\\,C_{\\log}$.

    Args:
        moment: The Gaussian-averaged second derivative $\\langle V''\\rangle_\\sigma$.
        config: Supplies $\\beta_1$, $\\beta_2$ and $C_{\\log}$.

    Returns:
        The first-order anomalous dimension shift.
    """
    return 0.5 * config.beta1 * config.beta2 * moment * config.log_coefficient


def anomalous_dimension(potential: Potential, config: PhysicsConfig) -> float:
    """First-order anomalous dimension of a potential in the given AdS2 background.

    Args:
        potential: Any :class:`~qft_operator.physics.potentials.Potential`.
        config: Physics configuration, whose ``sigma_sq`` fixes the normal-ordering
            scheme used for the Gaussian average.

    Returns:
        $\\gamma[V]$.

    Example:
        >>> from qft_operator.physics import PhysicsConfig, SineGordon, anomalous_dimension
        >>> cfg = PhysicsConfig()
        >>> gamma = anomalous_dimension(SineGordon(coupling=0.02, xi=0.8), cfg)
        >>> abs(gamma - cfg.analytical_anomalous_dim(0.02, 0.8)) < 1e-15
        True
    """
    moment = potential.gaussian_second_moment(config.sigma_sq)
    return anomalous_dimension_from_moment(moment, config)


def boundary_two_point(r: Tensor, delta_eff: Tensor | float) -> Tensor:
    """Boundary correlator $W(r) = r^{-2\\Delta_{\\mathrm{eff}}}$.

    Args:
        r: Boundary separations $|p_1 - p_2| > 0$.
        delta_eff: Effective dimension, scalar or broadcastable tensor.

    Returns:
        The correlator, evaluated through logs for dynamic-range safety.
    """
    return torch.exp(log_boundary_two_point(torch.log(r), delta_eff))


def log_boundary_two_point(log_r: Tensor, delta_eff: Tensor | float) -> Tensor:
    """$\\log W = -2\\Delta_{\\mathrm{eff}}\\,\\log r$.

    Note:
        The network is trained on this log-space target. Over the default separation
        window $r \\in [0.05, 12]$ with $\\Delta \\approx 1.5$, $W$ itself spans roughly
        eight decades, so a plain MSE on $W$ is dominated entirely by the smallest
        separations -- this is the single largest numerical difference from the baseline
        script.

    Args:
        log_r: $\\log r$.
        delta_eff: Effective dimension.

    Returns:
        $\\log W$.
    """
    return -2.0 * delta_eff * log_r


@dataclass(frozen=True)
class CorrelatorTargets:
    """Ground-truth labels for one sampled theory over a grid of separations.

    Attributes:
        log_r: $\\log r$ at each queried separation, shape ``(N,)``.
        log_w: $\\log W$ at each separation, shape ``(N,)``.
        delta_eff: Effective dimension at each separation, shape ``(N,)``; constant
            unless the coupling runs.
        gamma_reference: $\\gamma$ evaluated with the coupling quoted at $M$.
        log_m: $\\log M$ for this sample.
        coupling: $\\lambda(M)$ for this sample.
    """

    log_r: Tensor
    log_w: Tensor
    delta_eff: Tensor
    gamma_reference: float
    log_m: float
    coupling: float


def resummed_log_correlator(
    log_r: Tensor,
    gamma_reference: float,
    coupling: float,
    log_m: float,
    config: PhysicsConfig,
    beta: BetaFunction,
) -> tuple[Tensor, Tensor]:
    """RG-improved target: exponentiate the leading logarithm.

    Rather than keeping the fixed-order expression $W^{(0)}(1 + 2\\gamma\\log Mr)$, the
    logarithm is resummed into the exponent with the coupling evaluated at the physical
    scale $\\mu = 1/r$. Because $\\bar\\lambda(1/r)$ is independent of $M$ (see
    :mod:`qft_operator.physics.rg`), the result annihilates the Callan-Symanzik operator
    exactly, so the RG loss is consistent with the data by construction.

    Args:
        log_r: $\\log r$, shape ``(N,)``.
        gamma_reference: $\\gamma$ at the quoted coupling.
        coupling: $\\lambda(M)$.
        log_m: $\\log M$.
        config: Physics configuration.
        beta: RG flow.

    Returns:
        ``(log_w, delta_eff)``, both of shape ``(N,)``.
    """
    gamma_ref = torch.full_like(log_r, gamma_reference)
    lam_ref = torch.full_like(log_r, coupling)
    log_m_t = torch.full_like(log_r, log_m)
    gamma_running = scale_anomalous_dimension(gamma_ref, lam_ref, log_m_t, log_r, beta)
    delta_eff = config.free_dimension - gamma_running
    return log_boundary_two_point(log_r, delta_eff), delta_eff


def first_order_log_correlator(
    log_r: Tensor,
    moment: float,
    config: PhysicsConfig,
    reduced_integral: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fixed-order target built from the *numerically integrated* contact diagram.

    .. math::
        W = W^{(0)}\\left[1 + \\beta_1\\beta_2\\,\\langle V''\\rangle_\\sigma\\,
            \\tilde{I}(r, \\epsilon = 1/M)\\right],

    with $\\tilde{I}$ the reduced contact integral from
    :class:`~qft_operator.physics.bulk_integrals.ConformalIntegrator` (or its cached
    :class:`~qft_operator.physics.bulk_integrals.ReducedIntegralTable`). No resummation
    is performed, so this branch is honest about being first order -- and correspondingly
    it is **not** exactly RG-invariant. The RG loss weight should be left at zero when
    training on it; the shipped ``quadrature`` data config does so.

    Args:
        log_r: $\\log r$, shape ``(N,)``.
        moment: $\\langle V''\\rangle_\\sigma$.
        config: Physics configuration.
        reduced_integral: $\\tilde{I}(r, 1/M)$ evaluated at the same separations,
            shape ``(N,)``.

    Returns:
        ``(log_w, delta_eff)`` where ``delta_eff`` is the *local* log-slope
        $-\\tfrac{1}{2}\\,d\\log W / d\\log r$ implied by the fixed-order expression.

    Raises:
        ValueError: If the first-order correction drives $W$ non-positive, i.e. the
            coupling lies outside the perturbative window.
    """
    bracket = 1.0 + config.beta1 * config.beta2 * moment * reduced_integral.to(log_r.dtype)
    if bool((bracket <= 0).any()):
        raise ValueError(
            "first-order correction exceeds the leading term; reduce the coupling range "
            "or use target_mode='resummed'"
        )
    log_w = log_boundary_two_point(log_r, config.free_dimension) + torch.log(bracket)
    slope = torch.gradient(log_w, spacing=(log_r,))[0]
    return log_w, -0.5 * slope


def numerical_log_coefficient(integrator: ConformalIntegrator, r: float = 1.0) -> float:
    """Extract $C_{\\log}$ from quadrature rather than from the closed form.

    Used by the ``"hybrid"`` dataset mode, where the anomalous dimension is defined by
    the numerically integrated bulk diagram but the correlator is still resummed into an
    RG-consistent power law.

    Args:
        integrator: Configured quadrature engine.
        r: Separation at which to evaluate the cutoff derivative (the result is
            $r$-independent up to quadrature error, which is why this is a useful check).

    Returns:
        The numerically determined $C_{\\log}$, in the integrator's normalization.
    """
    r_tensor = torch.tensor([r], dtype=torch.float64)
    return float(integrator.log_slope(r_tensor, eps=1e-4).squeeze())
