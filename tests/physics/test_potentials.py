"""Potentials: closed-form Gaussian moments against quadrature, and the free limit."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from qft_operator.physics.potentials import (
    FreeTheory,
    GaussianProcessPotential,
    PhiFour,
    PolynomialPotential,
    Potential,
    RandomFourierPotential,
    SineGordon,
    gaussian_moment,
)


def _gauss_hermite_average(potential: Potential, sigma_sq: float, n: int = 200) -> float:
    """Independent Gauss-Hermite evaluation of <V''> for cross-checking closed forms."""
    if sigma_sq == 0.0:
        return float(potential.second_derivative(torch.zeros(1, dtype=torch.float64)))
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    phi = torch.as_tensor(nodes, dtype=torch.float64) * math.sqrt(2.0 * sigma_sq)
    w = torch.as_tensor(weights, dtype=torch.float64) / math.sqrt(math.pi)
    return float((potential.second_derivative(phi) * w).sum())


def _potentials() -> list[Potential]:
    generator = torch.Generator().manual_seed(7)
    n = 48
    return [
        FreeTheory(),
        SineGordon(coupling=0.02, xi=0.8),
        SineGordon(coupling=-0.03, xi=1.2),
        PhiFour(coupling=0.015),
        PolynomialPotential(0.04, (0.3, -0.2, 0.5, 0.1, -0.05, 0.02, 0.004)),
        RandomFourierPotential(
            0.05,
            torch.randn(n, generator=generator, dtype=torch.float64),
            torch.randn(n, generator=generator, dtype=torch.float64) / 0.9,
            torch.rand(n, generator=generator, dtype=torch.float64) * 2.0 * math.pi,
        ),
    ]


@pytest.mark.parametrize("potential", _potentials(), ids=lambda p: repr(p)[:28])
@pytest.mark.parametrize("sigma_sq", [0.0, 0.35, 1.2])
def test_closed_form_moment_matches_quadrature(potential: Potential, sigma_sq: float) -> None:
    closed = potential.gaussian_second_moment(sigma_sq)
    numeric = _gauss_hermite_average(potential, sigma_sq)
    assert closed == pytest.approx(numeric, rel=1e-8, abs=1e-14)


@pytest.mark.parametrize("potential", _potentials(), ids=lambda p: repr(p)[:28])
def test_zero_smearing_collapses_to_the_second_derivative_at_the_origin(
    potential: Potential,
) -> None:
    at_origin = float(potential.second_derivative(torch.zeros(1, dtype=torch.float64)))
    assert potential.gaussian_second_moment(0.0) == pytest.approx(at_origin, rel=1e-12, abs=1e-16)


@pytest.mark.parametrize("potential", _potentials(), ids=lambda p: repr(p)[:28])
def test_potentials_are_exactly_linear_in_the_coupling(
    potential: Potential, phi_grid: torch.Tensor
) -> None:
    # The RG loss differentiates through the branch input along dV/dlambda, so this
    # linearity is a correctness precondition, not a stylistic one.
    shape = potential.d_dcoupling(phi_grid)
    assert torch.allclose(potential.evaluate(phi_grid), potential.coupling * shape, rtol=1e-12)


@pytest.mark.parametrize("potential", _potentials(), ids=lambda p: repr(p)[:28])
def test_second_derivative_matches_autograd(potential: Potential) -> None:
    phi = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(potential.evaluate(phi).sum(), phi, create_graph=True)[0]
    # A potential that is linear in phi (the free theory, among others) has a first
    # derivative that is genuinely independent of phi: the second-order graph is empty
    # rather than broken, and the correct answer is zero.
    if first.requires_grad:
        (second,) = torch.autograd.grad(first.sum(), phi, allow_unused=True, materialize_grads=True)
    else:
        second = torch.zeros_like(phi)
    assert torch.allclose(second, potential.second_derivative(phi.detach()), atol=1e-9)


def test_free_theory_is_identically_zero(phi_grid: torch.Tensor) -> None:
    free = FreeTheory()
    assert torch.count_nonzero(free.evaluate(phi_grid)) == 0
    assert torch.count_nonzero(free.second_derivative(phi_grid)) == 0
    assert free.gaussian_second_moment(0.9) == 0.0


def test_sine_gordon_reduces_to_the_published_moment() -> None:
    lam, xi = 0.02, 0.8
    assert SineGordon(lam, xi).gaussian_second_moment(0.0) == pytest.approx(
        -2.0 * lam * xi**2, rel=1e-14
    )


def test_normal_ordered_phi_four_has_no_first_order_moment() -> None:
    # <phi^2> = 0 at sigma = 0, so a normal-ordered quartic contributes nothing at first
    # order in the beta1-beta2 channel -- the tadpole is what switches it on.
    assert PhiFour(0.05).gaussian_second_moment(0.0) == 0.0
    assert PhiFour(0.05).gaussian_second_moment(0.4) == pytest.approx(12.0 * 0.05 * 0.4, rel=1e-14)


@pytest.mark.parametrize(("order", "sigma_sq"), [(0, 0.5), (1, 0.5), (2, 0.5), (4, 0.5), (6, 0.3)])
def test_gaussian_moments(order: int, sigma_sq: float) -> None:
    expected = {0: 1.0, 1: 0.0, 2: sigma_sq, 4: 3.0 * sigma_sq**2, 6: 15.0 * sigma_sq**3}[order]
    assert gaussian_moment(order, sigma_sq) == pytest.approx(expected, rel=1e-12)


def test_gaussian_moment_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gaussian_moment(-1, 0.5)


def test_negative_smearing_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SineGordon(0.01, 0.5).gaussian_second_moment(-0.1)


def test_random_fourier_potential_validates_its_coefficients() -> None:
    with pytest.raises(ValueError, match="share one 1-D shape"):
        RandomFourierPotential(
            1.0,
            torch.zeros(4, dtype=torch.float64),
            torch.zeros(5, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
        )


def test_exact_gp_sample_reproduces_the_random_feature_moment(phi_grid: torch.Tensor) -> None:
    # The Cholesky-free tabulated GP with finite differences should agree with the
    # analytic random-Fourier construction it is tabulating.
    generator = torch.Generator().manual_seed(11)
    n = 32
    analytic = RandomFourierPotential(
        1.0,
        torch.randn(n, generator=generator, dtype=torch.float64),
        torch.randn(n, generator=generator, dtype=torch.float64) / 1.2,
        torch.rand(n, generator=generator, dtype=torch.float64) * 2.0 * math.pi,
    )
    tabulated = GaussianProcessPotential(1.0, phi_grid, analytic.evaluate(phi_grid))
    assert tabulated.gaussian_second_moment(0.0) == pytest.approx(
        analytic.gaussian_second_moment(0.0), rel=1e-4
    )
    assert tabulated.gaussian_second_moment(0.5) == pytest.approx(
        analytic.gaussian_second_moment(0.5), rel=1e-2
    )


def test_gaussian_process_potential_requires_a_uniform_grid() -> None:
    grid = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="uniformly spaced"):
        GaussianProcessPotential(1.0, grid, torch.zeros_like(grid))


def test_polynomial_requires_a_quadratic_term_to_have_curvature() -> None:
    linear = PolynomialPotential(1.0, (0.5, 2.0))
    assert linear.gaussian_second_moment(0.7) == 0.0
