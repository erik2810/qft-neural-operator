"""Beta function, flow map and the group property that makes targets RG-invariant."""

from __future__ import annotations

import math

import pytest
import torch

from qft_operator.physics.rg import (
    BetaFunction,
    RGConfig,
    running_coupling,
    scale_anomalous_dimension,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reference_scale": 0.0}, "reference_scale"),
        ({"log_scale_jitter": -1.0}, "log_scale_jitter"),
        ({"rk_steps": 0}, "rk_steps"),
    ],
)
def test_invalid_rg_configurations_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RGConfig(**kwargs)


def test_marginal_flag_and_closed_form_detection() -> None:
    assert RGConfig().is_marginal
    assert not RGConfig(epsilon=0.2).is_marginal
    assert BetaFunction(RGConfig(epsilon=0.2)).has_closed_form
    assert not BetaFunction(RGConfig(two_loop=1.0)).has_closed_form


def test_beta_function_matches_its_definition() -> None:
    beta = BetaFunction(RGConfig(epsilon=0.3, two_loop=1.5))
    lam = torch.tensor([0.0, 0.01, -0.02], dtype=torch.float64)
    assert torch.allclose(beta(lam), -0.3 * lam + 1.5 * lam**2, rtol=1e-14)


def test_linear_flow_has_the_closed_form_solution() -> None:
    beta = BetaFunction(RGConfig(epsilon=0.4))
    lam = torch.tensor([0.03], dtype=torch.float64)
    step = torch.tensor([1.3], dtype=torch.float64)
    assert torch.allclose(beta.run(lam, step), lam * math.exp(-0.4 * 1.3), rtol=1e-14)


def test_zero_transport_is_the_identity(beta: BetaFunction) -> None:
    lam = torch.tensor([0.02, -0.05], dtype=torch.float64)
    assert torch.allclose(beta.run(lam, torch.zeros_like(lam)), lam, rtol=1e-14)


def test_flow_is_invertible(beta: BetaFunction) -> None:
    lam = torch.tensor([0.02], dtype=torch.float64)
    step = torch.tensor([1.7], dtype=torch.float64)
    assert torch.allclose(beta.run(beta.run(lam, step), -step), lam, rtol=1e-9)


def test_flow_satisfies_the_group_property(beta: BetaFunction) -> None:
    # F(F(lambda, a), b) = F(lambda, a + b). This identity is the reason a correlator
    # written in terms of the coupling at scale 1/r cannot depend on M.
    lam = torch.tensor([0.03], dtype=torch.float64)
    a = torch.tensor([0.8], dtype=torch.float64)
    b = torch.tensor([-1.4], dtype=torch.float64)
    assert torch.allclose(beta.run(beta.run(lam, a), b), beta.run(lam, a + b), rtol=1e-9)


def test_running_coupling_is_independent_of_the_reference_scale(beta: BetaFunction) -> None:
    lam = torch.tensor([0.03], dtype=torch.float64)
    log_r = torch.tensor([math.log(2.5)], dtype=torch.float64)
    shift = torch.tensor([math.log(3.0)], dtype=torch.float64)
    zero = torch.zeros_like(shift)

    at_reference = running_coupling(lam, zero, log_r, beta)
    at_shifted = running_coupling(beta.run(lam, shift), shift, log_r, beta)
    assert torch.allclose(at_reference, at_shifted, rtol=1e-9)


def test_linear_running_matches_the_power_law_form() -> None:
    beta = BetaFunction(RGConfig(epsilon=0.35))
    lam = torch.tensor([0.02], dtype=torch.float64)
    log_m = torch.tensor([0.6], dtype=torch.float64)
    log_r = torch.tensor([0.9], dtype=torch.float64)
    expected = lam * torch.exp(0.35 * (log_m + log_r))  # lambda (M r)^epsilon
    assert torch.allclose(running_coupling(lam, log_m, log_r, beta), expected, rtol=1e-12)


def test_scaling_a_vanishing_coupling_does_not_divide_by_zero(beta: BetaFunction) -> None:
    zeros = torch.zeros(3, dtype=torch.float64)
    result = scale_anomalous_dimension(
        zeros, zeros, zeros, torch.ones(3, dtype=torch.float64), beta
    )
    assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0


def test_marginal_flow_leaves_the_anomalous_dimension_untouched() -> None:
    marginal = BetaFunction(RGConfig())
    gamma = torch.tensor([-0.002, 0.004], dtype=torch.float64)
    lam = torch.tensor([0.02, -0.03], dtype=torch.float64)
    log_r = torch.tensor([1.4, -0.7], dtype=torch.float64)
    scaled = scale_anomalous_dimension(gamma, lam, torch.zeros_like(lam), log_r, marginal)
    assert torch.allclose(scaled, gamma, rtol=1e-14)
