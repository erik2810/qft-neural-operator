"""Data, boundary-scaling, RG-invariance and composite loss terms."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from qft_operator.losses.composite import LossWeights, PhysicsInformedLoss
from qft_operator.losses.data import LogCorrelatorLoss, relative_l2
from qft_operator.losses.rg import RGInvarianceLoss
from qft_operator.losses.scaling import BoundaryScalingLoss
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import BetaFunction, RGConfig


class _PowerLaw(nn.Module):
    """Exact RG-invariant correlator: gamma proportional to the coupling at scale 1/r."""

    def __init__(self, free_dimension: float, sensitivity: float, beta: BetaFunction) -> None:
        super().__init__()
        self.free_dimension = free_dimension
        self.sensitivity = sensitivity
        self.beta = beta

    def forward(self, v_phi: Tensor, coords: Tensor, log_m: Tensor | None = None) -> Tensor:
        log_r = torch.log((coords[..., 0] - coords[..., 1]).abs())
        scale = torch.zeros_like(log_r) if log_m is None else log_m.expand_as(log_r)
        # v_phi is proportional to lambda, so mean(V) transports exactly like the coupling.
        effective = self.beta.run(v_phi.mean(-1, keepdim=True), -(scale + log_r))
        return -2.0 * (self.free_dimension - self.sensitivity * effective) * log_r


class _CurvedCorrelator(nn.Module):
    """Deliberately non-power-law: log W carries a quadratic term in log r."""

    def forward(self, v_phi: Tensor, coords: Tensor, log_m: Tensor | None = None) -> Tensor:
        log_r = torch.log((coords[..., 0] - coords[..., 1]).abs())
        return -3.0 * log_r + 0.2 * log_r**2


@pytest.fixture
def batch() -> dict[str, Tensor]:
    torch.manual_seed(0)
    n, points, n_phi = 4, 12, 16
    coupling = torch.tensor([0.01, -0.02, 0.03, 0.0], dtype=torch.float64)
    shape = torch.ones(n, n_phi, dtype=torch.float64)
    r = torch.exp(torch.linspace(-2.0, 2.0, points, dtype=torch.float64)).expand(n, points)
    return {
        "v_phi": coupling.unsqueeze(-1) * shape,
        "dv_dcoupling": shape,
        "coords": torch.stack([torch.zeros_like(r), r], dim=-1).contiguous(),
        "log_w": -3.0 * torch.log(r),
        "delta_eff": torch.full((n, points), 1.5, dtype=torch.float64),
        "log_m": torch.zeros(n, 1, dtype=torch.float64),
        "coupling": coupling,
        "gamma": torch.zeros(n, dtype=torch.float64),
        "family": torch.zeros(n, dtype=torch.long),
    }


# ---------------------------------------------------------------- data term --
def test_relative_l2_is_scale_invariant() -> None:
    target = torch.randn(3, 10, dtype=torch.float64)
    prediction = target + 0.1 * torch.randn_like(target)
    assert relative_l2(prediction, target) == pytest.approx(
        float(relative_l2(7.0 * prediction, 7.0 * target)), rel=1e-12
    )


def test_relative_l2_vanishes_on_an_exact_match() -> None:
    target = torch.randn(2, 5, dtype=torch.float64)
    assert float(relative_l2(target, target)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("reduction", ["mse", "huber", "rel_l2"])
def test_data_loss_reductions(reduction: str) -> None:
    loss = LogCorrelatorLoss(reduction=reduction)  # type: ignore[arg-type]
    target = torch.randn(2, 6, dtype=torch.float64)
    assert float(loss(target, target)) == pytest.approx(0.0, abs=1e-12)
    assert float(loss(target + 0.5, target)) > 0.0


def test_data_loss_validates_arguments() -> None:
    with pytest.raises(ValueError, match="unknown reduction"):
        LogCorrelatorLoss(reduction="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="huber_delta"):
        LogCorrelatorLoss(huber_delta=0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        LogCorrelatorLoss()(torch.zeros(2, 3), torch.zeros(2, 4))


# ------------------------------------------------------------- scaling term --
def test_scaling_loss_vanishes_on_a_power_law(batch: dict[str, Tensor]) -> None:
    model = _PowerLaw(1.5, 0.1, BetaFunction())
    loss = BoundaryScalingLoss(mode="power_law", r_min=1.0)
    assert float(loss(model, batch["v_phi"], batch["coords"], batch["log_m"]).detach()) < 1e-16


def test_scaling_loss_detects_curvature(batch: dict[str, Tensor]) -> None:
    loss = BoundaryScalingLoss(mode="power_law", r_min=1.0)
    value = float(loss(_CurvedCorrelator(), batch["v_phi"], batch["coords"], batch["log_m"]))
    # log W = -3 log r + 0.2 (log r)^2 has constant curvature 0.4.
    assert value == pytest.approx(0.16, rel=1e-6)


def test_supervised_scaling_matches_a_known_exponent(batch: dict[str, Tensor]) -> None:
    model = _PowerLaw(1.5, 0.0, BetaFunction())  # exactly Delta_eff = 1.5
    loss = BoundaryScalingLoss(mode="supervised", r_min=1.0)
    value = loss(model, batch["v_phi"], batch["coords"], batch["log_m"], batch["delta_eff"])
    assert float(value.detach()) < 1e-16


def test_scaling_loss_requires_labels_in_supervised_mode(batch: dict[str, Tensor]) -> None:
    loss = BoundaryScalingLoss(mode="both")
    with pytest.raises(ValueError, match="requires delta_eff"):
        loss(_CurvedCorrelator(), batch["v_phi"], batch["coords"], batch["log_m"])


def test_scaling_loss_is_zero_when_no_point_is_asymptotic(batch: dict[str, Tensor]) -> None:
    loss = BoundaryScalingLoss(mode="power_law", r_min=1e6)
    assert float(loss(_CurvedCorrelator(), batch["v_phi"], batch["coords"], batch["log_m"])) == 0.0


def test_scaling_loss_validates_arguments() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        BoundaryScalingLoss(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="r_min"):
        BoundaryScalingLoss(r_min=0.0)


# ------------------------------------------------------------------ RG term --
@pytest.mark.parametrize(
    "rg_config", [RGConfig(), RGConfig(epsilon=0.3), RGConfig(epsilon=0.3, two_loop=1.0)]
)
def test_rg_loss_vanishes_on_an_rg_invariant_correlator(
    batch: dict[str, Tensor], rg_config: RGConfig
) -> None:
    # The stand-in is built from the coupling at the physical scale 1/r, so the
    # Callan-Symanzik operator annihilates it exactly -- the loss must see that.
    beta = BetaFunction(rg_config)
    model = _PowerLaw(1.5, 0.1, beta)
    loss = RGInvarianceLoss(beta=beta, mode="jvp", normalize=False)
    value = loss(
        model,
        batch["v_phi"],
        batch["coords"],
        batch["log_m"],
        batch["coupling"],
        batch["dv_dcoupling"],
    )
    assert float(value.detach()) < 1e-12


def test_rg_loss_is_nonzero_when_the_scale_dependence_is_wrong(
    batch: dict[str, Tensor],
) -> None:
    class _ScaleDependent(nn.Module):
        def forward(self, v: Tensor, coords: Tensor, log_m: Tensor | None = None) -> Tensor:
            log_r = torch.log((coords[..., 0] - coords[..., 1]).abs())
            scale = torch.zeros_like(log_r) if log_m is None else log_m.expand_as(log_r)
            return -3.0 * log_r + 0.5 * scale

    loss = RGInvarianceLoss(beta=BetaFunction(), normalize=False)
    value = loss(
        _ScaleDependent(),
        batch["v_phi"],
        batch["coords"],
        batch["log_m"],
        batch["coupling"],
        batch["dv_dcoupling"],
    )
    assert float(value) == pytest.approx(0.25, rel=1e-6)


def test_rg_loss_validates_the_tangent_shape(batch: dict[str, Tensor]) -> None:
    loss = RGInvarianceLoss()
    with pytest.raises(ValueError, match="must match"):
        loss(
            _CurvedCorrelator(),
            batch["v_phi"],
            batch["coords"],
            batch["log_m"],
            batch["coupling"],
            torch.zeros(4, 3, dtype=torch.float64),
        )


# ----------------------------------------------------------------- composite --
def test_loss_weights_validate_and_ramp() -> None:
    weights = LossWeights(warmup_epochs=4)
    assert weights.physics_scale(0) == 0.0
    assert weights.physics_scale(2) == pytest.approx(0.5)
    assert weights.physics_scale(9) == 1.0
    assert LossWeights(warmup_epochs=0).physics_scale(0) == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(scaling=-1.0)
    with pytest.raises(ValueError, match="warmup_epochs"):
        LossWeights(warmup_epochs=-1)


def test_composite_reports_every_active_term(batch: dict[str, Tensor]) -> None:
    model = _PowerLaw(1.5, 0.1, BetaFunction())
    loss = PhysicsInformedLoss(LossWeights(1.0, 0.5, 0.5, warmup_epochs=0))
    prediction = model(batch["v_phi"], batch["coords"], batch["log_m"])
    total, components = loss(model, batch, prediction, epoch=3)
    assert set(components) == {"data", "scaling", "rg", "total"}
    assert float(total) == pytest.approx(float(components["total"]), rel=1e-12)


def test_composite_skips_physics_terms_during_warmup(batch: dict[str, Tensor]) -> None:
    model = _CurvedCorrelator()
    loss = PhysicsInformedLoss(LossWeights(1.0, 1.0, 1.0, warmup_epochs=5))
    prediction = model(batch["v_phi"], batch["coords"], batch["log_m"])
    _, components = loss(model, batch, prediction, epoch=0)
    assert set(components) == {"data", "total"}


def test_composite_skips_terms_with_zero_weight(batch: dict[str, Tensor]) -> None:
    model = _CurvedCorrelator()
    loss = PhysicsInformedLoss(LossWeights(1.0, 0.0, 0.0, warmup_epochs=0))
    prediction = model(batch["v_phi"], batch["coords"], batch["log_m"])
    _, components = loss(model, batch, prediction, epoch=10)
    assert set(components) == {"data", "total"}


def test_composite_gradients_reach_the_model(physics: PhysicsConfig) -> None:
    from qft_operator.models.deeponet import FourierDeepONet

    torch.manual_seed(0)
    model = FourierDeepONet(
        physics,
        n_phi=16,
        latent_dim=16,
        branch_width=8,
        branch_blocks=1,
        trunk_width=16,
        trunk_layers=1,
    )
    n, points = 2, 8
    r = torch.exp(torch.linspace(-1.5, 1.5, points)).expand(n, points).contiguous()
    data = {
        "v_phi": torch.randn(n, 16) * 0.05,
        "dv_dcoupling": torch.ones(n, 16),
        "coords": torch.stack([torch.zeros_like(r), r], dim=-1),
        "log_w": -3.0 * torch.log(r),
        "delta_eff": torch.full((n, points), 1.5),
        "log_m": torch.zeros(n, 1),
        "coupling": torch.tensor([0.02, -0.01]),
    }
    loss = PhysicsInformedLoss(LossWeights(1.0, 0.1, 0.1, warmup_epochs=0))
    prediction = model(data["v_phi"], data["coords"], data["log_m"])
    total, _ = loss(model, data, prediction, epoch=5)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_scaling_and_rg_terms_are_finite_for_marginal_and_running_flows(
    batch: dict[str, Tensor],
) -> None:
    for epsilon in (0.0, 0.4):
        beta = BetaFunction(RGConfig(epsilon=epsilon))
        model = _PowerLaw(1.5, 0.1, beta)
        loss = PhysicsInformedLoss(
            LossWeights(1.0, 0.1, 0.1, warmup_epochs=0),
            rg_loss=RGInvarianceLoss(beta=beta),
        )
        prediction = model(batch["v_phi"], batch["coords"], batch["log_m"])
        total, _ = loss(model, batch, prediction, epoch=9)
        assert math.isfinite(float(total))
