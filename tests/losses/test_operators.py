"""Differential-operator helpers used by the physics losses."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from qft_operator.losses.operators import (
    DirectionalDerivative,
    log_curvature,
    log_slope,
    rebuild_coords,
)


class _AnalyticOperator(nn.Module):
    """Exact log W = -2 Delta_eff log r with Delta_eff = a + b * mean(V) + c * log M.

    Having a closed-form stand-in means the helpers are checked against known
    derivatives rather than against themselves.
    """

    def __init__(self, a: float = 1.5, b: float = 0.5, c: float = 0.25) -> None:
        super().__init__()
        self.a, self.b, self.c = a, b, c
        self.dummy = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, v_phi: Tensor, coords: Tensor, log_m: Tensor | None = None) -> Tensor:
        log_r = torch.log((coords[..., 0] - coords[..., 1]).abs())
        scale = torch.zeros_like(log_r) if log_m is None else log_m.expand_as(log_r)
        # The trainable offset sits inside the exponent so that derivatives of the
        # slope actually reach it; a purely additive constant would not.
        delta_eff = self.a + self.b * v_phi.mean(-1, keepdim=True) + self.c * scale + self.dummy
        return -2.0 * delta_eff * log_r


@pytest.fixture
def analytic() -> _AnalyticOperator:
    return _AnalyticOperator()


@pytest.fixture
def sample() -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    v = torch.randn(2, 8, dtype=torch.float64) * 0.1
    r = torch.exp(torch.linspace(-2.0, 1.5, 6, dtype=torch.float64)).expand(2, 6).contiguous()
    coords = torch.stack([torch.zeros_like(r), r], dim=-1)
    return v, coords, torch.zeros(2, 1, dtype=torch.float64)


def test_rebuild_coords_preserves_midpoint_and_sets_the_separation() -> None:
    coords = torch.tensor([[[1.0, 4.0], [-2.0, 0.0]]], dtype=torch.float64)
    log_r = torch.log(torch.tensor([[2.0, 7.0]], dtype=torch.float64))
    rebuilt = rebuild_coords(coords, log_r)
    assert torch.allclose(rebuilt.sum(-1), coords.sum(-1), atol=1e-12)
    assert torch.allclose((rebuilt[..., 1] - rebuilt[..., 0]).abs(), torch.exp(log_r), atol=1e-12)


def test_log_slope_recovers_minus_two_delta_eff(
    analytic: _AnalyticOperator, sample: tuple[Tensor, Tensor, Tensor]
) -> None:
    v, coords, log_m = sample
    expected = -2.0 * (analytic.a + analytic.b * v.mean(-1, keepdim=True)).expand(-1, 6)
    assert torch.allclose(log_slope(analytic, v, coords, log_m), expected, atol=1e-10)


def test_log_curvature_vanishes_for_a_power_law(
    analytic: _AnalyticOperator, sample: tuple[Tensor, Tensor, Tensor]
) -> None:
    v, coords, log_m = sample
    curvature = log_curvature(analytic, v, coords, log_m, create_graph=False)
    assert float(curvature.abs().max()) < 1e-9


def test_log_slope_can_be_backpropagated(
    analytic: _AnalyticOperator, sample: tuple[Tensor, Tensor, Tensor]
) -> None:
    v, coords, log_m = sample
    slope = log_slope(analytic, v, coords, log_m, create_graph=True)
    slope.pow(2).mean().backward()
    assert analytic.dummy.grad is not None


def test_directional_derivative_matches_the_closed_form(
    analytic: _AnalyticOperator, sample: tuple[Tensor, Tensor, Tensor]
) -> None:
    # d/ds log W(V + s tv, log M + s tm) = -2 (b mean(tv) + c tm) log r.
    v, coords, log_m = sample
    tangent_v = torch.randn_like(v)
    tangent_m = torch.ones_like(log_m)
    log_r = torch.log(coords[..., 1] - coords[..., 0])
    expected = -2.0 * (analytic.b * tangent_v.mean(-1, keepdim=True) + analytic.c) * log_r

    def evaluate(x: Tensor, m: Tensor) -> Tensor:
        return analytic(x, coords, m)

    for mode in ("jvp", "fd"):
        derivative = DirectionalDerivative(mode=mode, step=1e-4)(  # type: ignore[arg-type]
            evaluate, (v, log_m), (tangent_v, tangent_m)
        )
        assert torch.allclose(derivative, expected, atol=1e-6), mode


def test_jvp_and_finite_differences_agree_on_the_real_network(
    physics_cft: object,
) -> None:
    from qft_operator.models.deeponet import FourierDeepONet
    from qft_operator.physics.config import PhysicsConfig

    torch.manual_seed(0)
    model = FourierDeepONet(
        PhysicsConfig(),
        n_phi=32,
        latent_dim=16,
        branch_width=8,
        branch_blocks=1,
        trunk_width=16,
        trunk_layers=1,
        readout_init_scale=1.0,
    ).double()
    v = torch.randn(2, 32, dtype=torch.float64) * 0.05
    r = torch.exp(torch.linspace(-2.0, 1.5, 6, dtype=torch.float64)).expand(2, 6).contiguous()
    coords = torch.stack([torch.zeros_like(r), r], dim=-1)
    log_m = torch.zeros(2, 1, dtype=torch.float64)
    tangents = (torch.randn_like(v) * 0.01, torch.ones_like(log_m))

    def evaluate(x: Tensor, m: Tensor) -> Tensor:
        return model(x, coords, m)

    exact = DirectionalDerivative("jvp")(evaluate, (v, log_m), tangents)
    approx = DirectionalDerivative("fd", step=1e-5)(evaluate, (v, log_m), tangents)
    assert torch.allclose(exact, approx, rtol=1e-4, atol=1e-6)


def test_directional_derivative_validates_its_arguments(
    analytic: _AnalyticOperator, sample: tuple[Tensor, Tensor, Tensor]
) -> None:
    v, coords, log_m = sample
    with pytest.raises(ValueError, match="mode must be"):
        DirectionalDerivative(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step must be positive"):
        DirectionalDerivative(step=0.0)
    with pytest.raises(ValueError, match="same length"):
        DirectionalDerivative()(lambda x: analytic(x, coords, log_m), (v,), (v, log_m))


def test_forward_mode_failure_falls_back_to_differences() -> None:
    class _NoForwardMode(nn.Module):
        def forward(self, x: Tensor, m: Tensor) -> Tensor:
            if torch._C._are_functorch_transforms_active():
                raise RuntimeError("no forward-mode rule for this op")
            return (x * 2.0).sum(-1, keepdim=True) + m

    derivative = DirectionalDerivative("jvp")
    x = torch.ones(1, 3, dtype=torch.float64)
    m = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="forward-mode AD unavailable"):
        value = derivative(_NoForwardMode(), (x, m), (torch.ones_like(x), torch.ones_like(m)))
    assert derivative.mode == "fd"
    assert value == pytest.approx(7.0, rel=1e-6)


def test_fallback_can_be_disabled() -> None:
    class _Broken(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            if torch._C._are_functorch_transforms_active():
                raise RuntimeError("nope")
            return x

    with pytest.raises(RuntimeError, match="nope"):
        DirectionalDerivative("jvp", allow_fallback=False)(
            _Broken(), (torch.ones(2),), (torch.ones(2),)
        )
