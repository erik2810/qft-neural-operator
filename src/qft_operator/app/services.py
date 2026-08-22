"""Theory construction and correlator evaluation shared by the REST and WS routes.

Keeping this out of the routers means the HTTP and WebSocket paths cannot drift apart:
both build the same potential from the same spec and evaluate the same curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from qft_operator.analysis.spectrum import anomalous_dimension_from_correlator
from qft_operator.app.state import AppState
from qft_operator.data.config import DataConfig
from qft_operator.data.samplers import PotentialSampler
from qft_operator.physics.correlators import (
    anomalous_dimension_from_moment,
    resummed_log_correlator,
)
from qft_operator.physics.potentials import GaussianProcessPotential, Potential
from qft_operator.physics.rg import running_coupling

__all__ = ["TheorySpec", "CorrelatorResult", "build_potential", "evaluate_correlator", "log_r_grid"]


class TheorySpec(BaseModel):
    """A point in theory space, as the frontend controls describe it.

    Attributes:
        family: Which potential family to instantiate.
        coupling: $\\lambda(M)$. Applied after construction; every potential is linear in
            its coupling, so overriding it is exact.
        xi: Sine-Gordon vertex exponent, ignored by the other families.
        seed: Draw index for the polynomial and GP families, so a given slider position
            is reproducible.
        log_m: $\\log M$, the renormalization scale. The correlator must not depend on it.
        moment: Optional $\\langle V''\\rangle_\\sigma$. The frontend knows this in closed
            form for every family it offers, so sending it avoids the ~0.1% discretization
            error of recovering $V''$ from tabulated samples by finite differences.
        v_phi: Optional explicit potential samples on the field grid, *already multiplied
            by the coupling*. When given, the server uses them verbatim instead of drawing
            from its own sampler.

            This is how the frontend keeps its two modes honest. Its polynomial and GP
            families are drawn with a different PRNG than
            :class:`~qft_operator.data.samplers.PotentialSampler`, so the same ``seed``
            means "another draw from the same distribution", not the same function. By
            building $V$ locally and sending it, the page compares the operator against
            the exact answer *for the potential it is actually displaying* -- identical
            whether or not a backend is attached.
    """

    family: str = Field(default="sine_gordon")
    coupling: float = Field(default=0.02, ge=-1.0, le=1.0)
    xi: float = Field(default=0.8, gt=0.0, le=3.0)
    seed: int = Field(default=0, ge=0, le=1_000_000)
    log_m: float = Field(default=0.0, ge=-6.0, le=6.0)
    moment: float | None = Field(default=None)
    v_phi: list[float] | None = Field(default=None)


@dataclass(frozen=True)
class CorrelatorResult:
    """Exact and predicted correlators on a shared grid.

    Attributes:
        log_r: $\\log r$ grid.
        log_w_exact: Closed-form $\\log W$.
        log_w_pred: Operator prediction.
        gamma_exact: Closed-form $\\gamma$.
        gamma_pred: $\\gamma$ recovered from the prediction by a log-log fit.
        running_coupling: $\\bar\\lambda$ at the middle of the displayed window.
        potential: Values of $V(\\phi)$ on the field grid.
        second_derivative: Values of $V''(\\phi)$ -- the source of $\\gamma$.
        phi: The field grid itself.
    """

    log_r: Tensor
    log_w_exact: Tensor
    log_w_pred: Tensor
    gamma_exact: float
    gamma_pred: float
    running_coupling: float
    potential: Tensor
    second_derivative: Tensor
    phi: Tensor


def log_r_grid(n: int = 128, r_min: float = 0.05, r_max: float = 12.0) -> Tensor:
    """Uniform grid in $\\log r$ -- the variable the physics is linear in.

    Args:
        n: Number of points.
        r_min: Smallest separation.
        r_max: Largest separation.

    Returns:
        Tensor of shape ``(n,)``.

    Raises:
        ValueError: If the window is empty or fewer than two points are requested.
    """
    if n < 2:
        raise ValueError(f"need at least two points, got {n}")
    if not 0.0 < r_min < r_max:
        raise ValueError(f"need 0 < r_min < r_max, got {r_min}, {r_max}")
    return torch.linspace(math.log(r_min), math.log(r_max), n, dtype=torch.float64)


def build_potential(spec: TheorySpec, n_phi: int, phi_grid: Tensor | None = None) -> Potential:
    """Instantiate the potential a spec describes.

    With ``spec.v_phi`` absent, construction goes through the same
    :class:`~qft_operator.data.samplers.PotentialSampler` the training data used, so a
    theory the viewer dials up is drawn from exactly the distribution the operator was
    fitted on -- not a look-alike built separately here.

    With ``spec.v_phi`` present, the samples are wrapped in a
    :class:`~qft_operator.physics.potentials.GaussianProcessPotential`, which recovers
    $V''$ by a fourth-order stencil and its Gaussian average by Gauss-Hermite quadrature.
    Those are numerical rather than closed form, so the resulting $\\gamma$ carries a
    small discretization error the analytic families do not.

    Args:
        spec: The requested theory.
        n_phi: Field-grid resolution.
        phi_grid: The field grid ``v_phi`` is sampled on; required when ``v_phi`` is set.

    Returns:
        A configured :class:`~qft_operator.physics.potentials.Potential`.

    Raises:
        ValueError: If ``spec.family`` is unknown, or ``v_phi`` does not match the grid.
    """
    if spec.v_phi is not None:
        if phi_grid is None:
            raise ValueError("phi_grid is required when v_phi is supplied")
        values = torch.tensor(spec.v_phi, dtype=torch.float64)
        if values.shape != phi_grid.shape:
            raise ValueError(
                f"v_phi has {values.numel()} samples but the grid has {phi_grid.numel()}"
            )
        # Coupling 1.0: the samples already carry it, and every downstream use is linear.
        return GaussianProcessPotential(1.0, phi_grid, values, centered=False)

    config = DataConfig(n_phi=n_phi, xi_range=(spec.xi, spec.xi))
    sampler = PotentialSampler(config, torch.Generator().manual_seed(spec.seed))
    potential = sampler.sample(spec.family)
    potential.coupling = spec.coupling
    return potential


@torch.no_grad()
def evaluate_correlator(state: AppState, spec: TheorySpec, n_points: int = 128) -> CorrelatorResult:
    """Evaluate the exact and predicted correlators for one theory.

    Args:
        state: Shared physics and model state.
        spec: The requested theory.
        n_points: Resolution of the $\\log r$ grid.

    Returns:
        A :class:`CorrelatorResult`.
    """
    physics = state.physics
    phi = state.phi_grid
    potential = build_potential(spec, state.n_phi, phi)
    moment = (
        spec.moment
        if spec.moment is not None
        else potential.gaussian_second_moment(physics.sigma_sq)
    )
    gamma = anomalous_dimension_from_moment(moment, physics)

    log_r = log_r_grid(n_points)
    log_w_exact, _ = resummed_log_correlator(
        log_r, gamma, spec.coupling, spec.log_m, physics, state.beta
    )

    v_phi = (potential.evaluate(phi) / state.feature_scale).to(torch.float32)
    radii = torch.exp(log_r).to(torch.float32)
    coords = torch.stack([torch.zeros_like(radii), radii], dim=-1).unsqueeze(0)
    log_m = torch.full((1, 1), spec.log_m, dtype=torch.float32)

    device = next(state.model.parameters()).device
    log_w_pred = state.model(
        v_phi.unsqueeze(0).to(device), coords.to(device), log_m.to(device)
    ).squeeze(0)
    gamma_pred = float(
        anomalous_dimension_from_correlator(
            log_r.to(torch.float32).unsqueeze(0),
            log_w_pred.cpu().unsqueeze(0),
            physics.free_dimension,
        )
    )

    midpoint = log_r[log_r.shape[0] // 2]
    lam_bar = float(
        running_coupling(
            torch.tensor(spec.coupling, dtype=torch.float64),
            torch.tensor(spec.log_m, dtype=torch.float64),
            midpoint,
            state.beta,
        )
    )

    return CorrelatorResult(
        log_r=log_r,
        log_w_exact=log_w_exact,
        log_w_pred=log_w_pred.cpu().to(torch.float64),
        gamma_exact=gamma,
        gamma_pred=gamma_pred,
        running_coupling=lam_bar,
        potential=potential.evaluate(phi),
        second_derivative=potential.second_derivative(phi),
        phi=phi,
    )
