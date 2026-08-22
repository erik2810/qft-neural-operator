"""Anomalous dimensions, the free-theory limit, and boundary conformal scaling."""

from __future__ import annotations

import math

import pytest
import torch

from qft_operator.physics.bulk_integrals import ConformalIntegrator, ReducedIntegralTable
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.correlators import (
    anomalous_dimension,
    anomalous_dimension_from_moment,
    boundary_two_point,
    first_order_log_correlator,
    log_boundary_two_point,
    numerical_log_coefficient,
    resummed_log_correlator,
)
from qft_operator.physics.potentials import FreeTheory, PhiFour, SineGordon
from qft_operator.physics.rg import BetaFunction, RGConfig


def test_general_functional_reproduces_the_published_sine_gordon_result(
    physics: PhysicsConfig,
) -> None:
    # gamma[V] = 1/2 beta1 beta2 <V''>_sigma C_log must collapse onto
    # gamma = -lambda (2 L^2 c_Delta / (2 Delta - 1)) beta1 beta2 xi^2 for Sine-Gordon.
    for lam, xi in ((0.02, 0.8), (-0.04, 1.15), (0.005, 0.4)):
        general = anomalous_dimension(SineGordon(lam, xi), physics)
        published = physics.analytical_anomalous_dim(lam, xi)
        assert general == pytest.approx(published, rel=1e-14, abs=1e-18)


def test_free_theory_has_no_anomalous_dimension(physics: PhysicsConfig) -> None:
    assert anomalous_dimension(FreeTheory(), physics) == 0.0


def test_free_theory_correlator_is_the_exact_conformal_power_law(
    physics: PhysicsConfig,
) -> None:
    log_r = torch.log(torch.tensor([0.05, 1.0, 12.0], dtype=torch.float64))
    log_w, delta_eff = resummed_log_correlator(log_r, 0.0, 0.0, 0.0, physics, BetaFunction())
    assert torch.allclose(delta_eff, torch.full_like(delta_eff, physics.free_dimension))
    assert torch.allclose(log_w, -2.0 * physics.free_dimension * log_r, atol=1e-14)


def test_normal_ordered_phi_four_is_free_at_first_order(physics: PhysicsConfig) -> None:
    assert anomalous_dimension(PhiFour(0.05), physics) == 0.0
    with_tadpole = PhysicsConfig(sigma_sq=0.4)
    assert anomalous_dimension(PhiFour(0.05), with_tadpole) != 0.0


def test_anomalous_dimension_is_linear_in_the_coupling(physics: PhysicsConfig) -> None:
    base = anomalous_dimension(SineGordon(0.01, 0.9), physics)
    scaled = anomalous_dimension(SineGordon(0.03, 0.9), physics)
    assert scaled == pytest.approx(3.0 * base, rel=1e-12)


def test_moment_and_potential_entry_points_agree(physics: PhysicsConfig) -> None:
    potential = SineGordon(0.02, 0.7)
    moment = potential.gaussian_second_moment(physics.sigma_sq)
    assert anomalous_dimension_from_moment(moment, physics) == pytest.approx(
        anomalous_dimension(potential, physics), rel=1e-15
    )


def test_correlator_is_a_pure_power_law(physics: PhysicsConfig) -> None:
    r = torch.tensor([0.5, 2.0, 8.0], dtype=torch.float64)
    delta_eff = 1.4
    assert torch.allclose(boundary_two_point(r, delta_eff), r ** (-2.0 * delta_eff), rtol=1e-12)
    assert torch.allclose(
        log_boundary_two_point(torch.log(r), delta_eff), -2.0 * delta_eff * torch.log(r)
    )


def test_correlator_scales_covariantly_under_boundary_dilatation(
    physics: PhysicsConfig,
) -> None:
    # Boundary conformal symmetry: W(a r) = a^{-2 Delta_eff} W(r).
    r = torch.tensor([0.3, 1.7, 9.0], dtype=torch.float64)
    delta_eff = physics.free_dimension - 0.004
    scale = 6.5
    assert torch.allclose(
        boundary_two_point(scale * r, delta_eff),
        scale ** (-2.0 * delta_eff) * boundary_two_point(r, delta_eff),
        rtol=1e-12,
    )


@pytest.mark.parametrize("rg_config", [RGConfig(), RGConfig(epsilon=0.35, two_loop=1.5)])
def test_resummed_targets_are_exactly_rg_invariant(
    physics: PhysicsConfig, rg_config: RGConfig
) -> None:
    # Quoting the same theory at M and at 4M -- transporting the coupling accordingly --
    # must give literally the same correlator. This is what makes the RG loss consistent
    # with the data term rather than in tension with it.
    beta = BetaFunction(rg_config)
    log_r = torch.log(torch.tensor([0.1, 1.0, 9.0], dtype=torch.float64))
    lam, shift = 0.02, math.log(4.0)

    gamma = anomalous_dimension(SineGordon(lam, 0.8), physics)
    log_w_a, _ = resummed_log_correlator(log_r, gamma, lam, 0.0, physics, beta)

    lam_shifted = float(
        beta.run(torch.tensor(lam, dtype=torch.float64), torch.tensor(shift, dtype=torch.float64))
    )
    gamma_shifted = gamma * lam_shifted / lam
    log_w_b, _ = resummed_log_correlator(log_r, gamma_shifted, lam_shifted, shift, physics, beta)
    assert torch.allclose(log_w_a, log_w_b, atol=1e-12)


def test_relevant_coupling_makes_the_exponent_run(physics: PhysicsConfig) -> None:
    log_r = torch.log(torch.tensor([0.1, 1.0, 9.0], dtype=torch.float64))
    marginal = resummed_log_correlator(log_r, -0.002, 0.02, 0.0, physics, BetaFunction())[1]
    running = resummed_log_correlator(
        log_r, -0.002, 0.02, 0.0, physics, BetaFunction(RGConfig(epsilon=0.4))
    )[1]
    assert float(marginal.std()) == pytest.approx(0.0, abs=1e-15)
    assert float(running.std()) > 1e-4


def test_numerical_and_analytic_log_coefficients_agree_in_cft_conventions(
    physics_cft: PhysicsConfig, integrator: ConformalIntegrator
) -> None:
    assert numerical_log_coefficient(integrator) == pytest.approx(
        physics_cft.log_coefficient, rel=1e-4
    )


def test_first_order_target_carries_the_expected_log_slope(
    physics_cft: PhysicsConfig, integrator: ConformalIntegrator
) -> None:
    # Stripping the free power law from the fixed-order correlator must leave a straight
    # line in log r with slope 2 gamma -- but only where the correction is genuinely
    # small. The bracket enters as log(1 + m I~), and I~ ~ C_log log(r/eps) grows with
    # the cutoff, so the coupling has to be small enough that m I~ << 1 before the
    # linearization is legitimate. That is exactly the regime in which a fixed-order
    # target is meaningful at all, and why the resummed mode exists.
    log_r = torch.log(torch.logspace(-1.0, 1.0, 64, dtype=torch.float64))
    table = ReducedIntegralTable(integrator, log_x_min=2.0, log_x_max=14.0, n_nodes=96)
    moment = SineGordon(0.0005, 0.8).gaussian_second_moment(physics_cft.sigma_sq)
    reduced = table(torch.exp(log_r), eps=1e-3)

    log_w, _ = first_order_log_correlator(log_r, moment, physics_cft, reduced)
    stripped = log_w + 2.0 * physics_cft.free_dimension * log_r
    slope = torch.linalg.lstsq(
        torch.stack([log_r, torch.ones_like(log_r)], -1), stripped.unsqueeze(-1)
    ).solution[0, 0]
    expected = 2.0 * anomalous_dimension_from_moment(moment, physics_cft)
    assert float(slope) == pytest.approx(expected, rel=1e-2)


def test_first_order_target_rejects_a_non_perturbative_coupling(
    physics_cft: PhysicsConfig,
) -> None:
    log_r = torch.log(torch.tensor([1.0, 2.0], dtype=torch.float64))
    huge = torch.tensor([50.0, 50.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="exceeds the leading term"):
        first_order_log_correlator(log_r, -1.0, physics_cft, huge)
