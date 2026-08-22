"""Numerical validation of the regulated AdS2 contact integral.

These are the tests that tie the machine-learning pipeline to actual holography: they
check that Gauss-Legendre quadrature over the bulk reproduces the analytic
holographic-renormalization coefficient $C_{\\log} = 2L^2 c_\\Delta$, and that the scheme
constant is conformally invariant.
"""

from __future__ import annotations

import pytest
import torch

from qft_operator.physics.bulk_integrals import (
    ConformalIntegrator,
    QuadratureSpec,
    ReducedIntegralTable,
    analytic_log_coefficient,
    fit_log_divergence,
)
from qft_operator.physics.config import PhysicsConfig


def test_quadrature_spec_validates_resolution() -> None:
    with pytest.raises(ValueError, match="at least 8 nodes"):
        QuadratureSpec(n_radial=4)
    with pytest.raises(ValueError, match="z_max_over_r"):
        QuadratureSpec(z_max_over_r=1.0)


def test_cutoff_and_separations_are_validated(integrator: ConformalIntegrator) -> None:
    r = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="eps must be positive"):
        integrator.contact_integral(r, eps=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        integrator.contact_integral(torch.tensor([0.0], dtype=torch.float64), eps=1e-3)


def test_log_slope_matches_the_analytic_coefficient(integrator: ConformalIntegrator) -> None:
    # d I~ / d log(1/eps) = C_log = 2 L^2 c_Delta, independent of r.
    r = torch.tensor([0.25, 1.0, 4.0, 20.0], dtype=torch.float64)
    expected = analytic_log_coefficient(integrator.config.delta, integrator.config.L)
    slope = integrator.log_slope(r, eps=1e-5)
    assert torch.allclose(slope, torch.full_like(slope, expected), rtol=1e-5)


@pytest.mark.parametrize("m_sq", [0.0, 0.75, 2.0, 6.0])
def test_log_coefficient_across_scaling_dimensions(m_sq: float) -> None:
    cfg = PhysicsConfig(m_sq=m_sq, c_delta=None, propagator_normalization="cft")
    engine = ConformalIntegrator(cfg, QuadratureSpec(n_radial=192, n_boundary=192))
    slope = float(engine.log_slope(torch.tensor([1.0], dtype=torch.float64), eps=1e-5))
    assert slope == pytest.approx(analytic_log_coefficient(cfg.delta, cfg.L), rel=1e-4)


def test_scheme_constant_is_independent_of_separation(
    integrator: ConformalIntegrator,
) -> None:
    # I~ can only depend on r/eps, so at fixed eps the extracted kappa must be the same
    # for every separation -- across three decades of r, to nine digits.
    r = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
    assert float(integrator.kappa(r, eps=1e-6).std()) < 1e-8


def test_scheme_constant_converges_as_the_cutoff_is_removed(
    integrator: ConformalIntegrator,
) -> None:
    # The residual eps-dependence of kappa is the O(eps) tail of the asymptotic formula,
    # so successive decades of cutoff must bring it down, not merely keep it small.
    r = torch.tensor([1.0], dtype=torch.float64)
    values = [float(integrator.kappa(r, eps=eps)) for eps in (1e-3, 1e-4, 1e-5, 1e-6)]
    gaps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[-1] < 1e-8


def test_integral_scales_as_a_conformal_two_point_function(
    integrator: ConformalIntegrator,
) -> None:
    # Under (r, eps) -> (a r, a eps) the integral scales as a^{-2 Delta}.
    scale, eps = 5.0, 1e-4
    r = torch.tensor([0.7, 2.0], dtype=torch.float64)
    base = integrator.contact_integral(r, eps=eps)
    scaled = integrator.contact_integral(scale * r, eps=scale * eps)
    assert torch.allclose(scaled, scale ** (-2.0 * integrator.config.delta) * base, rtol=1e-8)


def test_least_squares_fit_recovers_the_log_coefficient(
    integrator: ConformalIntegrator,
) -> None:
    slope, _ = fit_log_divergence(integrator, r=1.0)
    expected = analytic_log_coefficient(integrator.config.delta, integrator.config.L)
    assert slope == pytest.approx(expected, rel=1e-3)


def test_normalization_convention_divides_the_result(physics: PhysicsConfig) -> None:
    spec = QuadratureSpec(n_radial=96, n_boundary=96)
    cft = ConformalIntegrator(
        PhysicsConfig(m_sq=physics.m_sq, propagator_normalization="cft"), spec
    )
    bulk = ConformalIntegrator(
        PhysicsConfig(m_sq=physics.m_sq, propagator_normalization="bulk_limit"), spec
    )
    r = torch.tensor([1.0], dtype=torch.float64)
    ratio = cft.contact_integral(r, eps=1e-3) / bulk.contact_integral(r, eps=1e-3)
    assert float(ratio) == pytest.approx(2.0 * physics.delta - 1.0, rel=1e-12)


def test_reduced_integral_table_matches_direct_quadrature(
    integrator: ConformalIntegrator,
) -> None:
    table = ReducedIntegralTable(integrator, log_x_min=1.0, log_x_max=14.0, n_nodes=96)
    r = torch.tensor([0.05, 0.5, 3.0, 11.0], dtype=torch.float64)
    exact = integrator.reduced_contact_integral(r, eps=1e-3)
    assert torch.allclose(table(r, eps=1e-3), exact, rtol=1e-4)


def test_reduced_integral_table_rejects_out_of_range_queries(
    integrator: ConformalIntegrator,
) -> None:
    table = ReducedIntegralTable(integrator, log_x_min=2.0, log_x_max=6.0, n_nodes=32)
    with pytest.raises(ValueError, match="outside tabulated range"):
        table(torch.tensor([1.0], dtype=torch.float64), eps=1.0)


@pytest.mark.slow
def test_quadrature_is_converged_against_a_finer_grid(physics_cft: PhysicsConfig) -> None:
    r = torch.tensor([0.3, 2.5], dtype=torch.float64)
    coarse = ConformalIntegrator(physics_cft, QuadratureSpec(n_radial=128, n_boundary=128))
    fine = ConformalIntegrator(physics_cft, QuadratureSpec(n_radial=512, n_boundary=512))
    assert torch.allclose(
        coarse.contact_integral(r, eps=1e-4), fine.contact_integral(r, eps=1e-4), rtol=1e-8
    )
