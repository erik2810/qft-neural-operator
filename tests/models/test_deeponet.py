"""Architectural invariants of the operator network.

These are physics checks expressed as properties of the model, not of the data: the
exact free-theory limit, boundary conformal symmetry, and independence of the query set.
The last one is what licenses the diagonal-Jacobian shortcut the physics losses use.
"""

from __future__ import annotations

import pytest
import torch

from qft_operator.models.branch import BranchNet
from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.models.trunk import BoundaryContextField, interp1d_uniform
from qft_operator.physics.config import PhysicsConfig

BATCH, POINTS, N_PHI = 3, 16, 32


def _make(physics: PhysicsConfig, **kwargs: object) -> FourierDeepONet:
    defaults: dict = {
        "n_phi": N_PHI,
        "latent_dim": 32,
        "branch_width": 16,
        "branch_blocks": 2,
        "trunk_width": 32,
        "trunk_layers": 2,
        "num_frequencies": 4,
        "context_grid": 32,
    }
    defaults.update(kwargs)
    defaults.pop("context_grid", None)
    return FourierDeepONet(physics, **defaults).double()  # type: ignore[arg-type]


def _inputs(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    v = torch.randn(BATCH, N_PHI, dtype=dtype) * 0.05
    r = torch.exp(torch.linspace(-2.5, 2.0, POINTS, dtype=dtype)).expand(BATCH, POINTS).contiguous()
    coords = torch.stack([torch.zeros_like(r), r], dim=-1)
    return v, coords, torch.zeros(BATCH, 1, dtype=dtype)


@pytest.mark.parametrize("head", ["inner_product", "attention"])
@pytest.mark.parametrize("residual_mode", ["none", "free", "exponent"])
def test_forward_shapes(physics: PhysicsConfig, head: str, residual_mode: str) -> None:
    model = _make(physics, head=head, residual_mode=residual_mode)
    v, coords, log_m = _inputs()
    out = model(v, coords, log_m)
    assert out.shape == (BATCH, POINTS)
    assert torch.isfinite(out).all()


def test_forward_defaults_the_renormalization_scale(physics: PhysicsConfig) -> None:
    model = _make(physics)
    v, coords, log_m = _inputs()
    assert torch.allclose(model(v, coords), model(v, coords, log_m), atol=1e-12)


def test_invalid_arguments_are_rejected(physics: PhysicsConfig) -> None:
    with pytest.raises(ValueError, match="unknown head"):
        FourierDeepONet(physics, head="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown residual_mode"):
        FourierDeepONet(physics, residual_mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="readout_init_scale"):
        FourierDeepONet(physics, readout_init_scale=-1.0)
    model = _make(physics)
    with pytest.raises(ValueError, match=r"coords must be"):
        model(torch.zeros(1, N_PHI, dtype=torch.float64), torch.zeros(1, 4, 3, dtype=torch.float64))


@pytest.mark.parametrize("residual_mode", ["free", "exponent"])
def test_free_theory_limit_is_exact_at_initialization(
    physics: PhysicsConfig, residual_mode: str
) -> None:
    # With a zero-initialized readout the untrained network reproduces
    # log W = -2 Delta beta1 beta2 log r to machine precision.
    model = _make(physics, residual_mode=residual_mode, readout_init_scale=0.0)
    v, coords, log_m = _inputs()
    with torch.no_grad():
        prediction = model(torch.zeros_like(v), coords, log_m)
    log_r = torch.log(coords[..., 1] - coords[..., 0])
    assert torch.allclose(prediction, -2.0 * physics.free_dimension * log_r, atol=1e-12)


def test_exponent_mode_is_a_power_law_by_construction(physics: PhysicsConfig) -> None:
    # d^2 log W / d(log r)^2 = 0 identically, so the boundary scaling loss is satisfied
    # structurally rather than by training.
    from qft_operator.losses.operators import log_curvature

    model = _make(physics, residual_mode="exponent")
    v, coords, log_m = _inputs()
    curvature = log_curvature(model, v, coords, log_m, create_graph=False)
    assert float(curvature.abs().max()) < 1e-9


def test_translation_invariance_along_the_boundary(physics: PhysicsConfig) -> None:
    model = _make(physics)
    v, coords, log_m = _inputs()
    with torch.no_grad():
        base = model(v, coords, log_m)
        for shift in (2.5, -13.0, 400.0):
            assert torch.allclose(model(v, coords + shift, log_m), base, atol=1e-9)


def test_prediction_is_independent_of_the_query_set(physics: PhysicsConfig) -> None:
    # The boundary context field lives on an internal grid precisely so that W at one
    # separation cannot depend on which other separations share the batch.
    model = _make(physics)
    v, coords, log_m = _inputs()
    with torch.no_grad():
        full = model(v, coords, log_m)
        subset = model(v, coords[:, ::3].contiguous(), log_m)
        permutation = torch.randperm(POINTS)
        permuted = model(v, coords[:, permutation].contiguous(), log_m)
    assert torch.allclose(full[:, ::3], subset, atol=1e-12)
    assert torch.allclose(full[:, permutation], permuted, atol=1e-12)


def test_jacobian_in_the_separation_is_diagonal(physics: PhysicsConfig) -> None:
    model = _make(physics, readout_init_scale=1.0)
    v, coords, log_m = _inputs()
    log_r = torch.log(coords[..., 1] - coords[..., 0])

    def evaluate(x: torch.Tensor) -> torch.Tensor:
        pairs = torch.stack([torch.zeros_like(x), torch.exp(x)], dim=-1)
        return model(v, pairs, log_m)[0]

    jacobian = torch.autograd.functional.jacobian(evaluate, log_r)[:, 0, :]
    off_diagonal = jacobian - torch.diag(torch.diagonal(jacobian))
    assert float(off_diagonal.abs().max()) == 0.0
    assert float(torch.diagonal(jacobian).abs().max()) > 0.0


@pytest.mark.parametrize("head", ["inner_product", "attention"])
def test_every_parameter_receives_a_gradient(physics: PhysicsConfig, head: str) -> None:
    # Unused parameters would force find_unused_parameters=True under DDP.
    model = _make(physics, head=head)
    v, coords, log_m = _inputs()
    model(v, coords, log_m).pow(2).mean().backward()
    starved = [n for n, p in model.named_parameters() if p.grad is None]
    assert starved == []


def test_branch_transfers_across_field_grid_resolutions(physics: PhysicsConfig) -> None:
    # Spectral convolutions make the branch an operator on functions, not on vectors, so
    # a different phi discretization must still run.
    model = _make(physics)
    _, coords, log_m = _inputs()
    finer = torch.randn(BATCH, 2 * N_PHI, dtype=torch.float64) * 0.05
    assert model(finer, coords, log_m).shape == (BATCH, POINTS)


def test_branch_emits_tokens_only_when_asked() -> None:
    with_tokens = BranchNet(N_PHI, latent_dim=8, width=8, n_blocks=1, emit_tokens=True)
    without = BranchNet(N_PHI, latent_dim=8, width=8, n_blocks=1, emit_tokens=False)
    code, tokens = with_tokens(torch.randn(2, N_PHI))
    assert tokens is not None and tokens.shape == (2, with_tokens.n_tokens, 8)
    assert without(torch.randn(2, N_PHI))[1] is None
    assert code.shape == (2, 8)


def test_branch_validates_input_rank() -> None:
    branch = BranchNet(N_PHI, latent_dim=8, width=8, n_blocks=1)
    with pytest.raises(ValueError, match=r"v_phi must be"):
        branch(torch.randn(2, 3, N_PHI))


def test_branch_non_spectral_path_runs() -> None:
    branch = BranchNet(N_PHI, latent_dim=8, width=8, n_blocks=1, use_spectral=False)
    code, tokens = branch(torch.randn(2, N_PHI))
    assert code.shape == (2, 8) and tokens is not None


def test_interp1d_reproduces_grid_values() -> None:
    values = torch.randn(2, 3, 9, dtype=torch.float64)
    grid = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).expand(2, 9)
    interpolated = interp1d_uniform(values, -1.0, 1.0, grid)
    assert torch.allclose(interpolated, values.transpose(1, 2), atol=1e-9)


def test_interp1d_clamps_out_of_range_queries() -> None:
    values = torch.randn(1, 2, 5, dtype=torch.float64)
    far = torch.tensor([[-50.0, 50.0]], dtype=torch.float64)
    out = interp1d_uniform(values, -1.0, 1.0, far)
    assert torch.allclose(out[0, 0], values[0, :, 0], atol=1e-9)
    assert torch.allclose(out[0, 1], values[0, :, -1], atol=1e-9)


def test_interp1d_validates_its_grid() -> None:
    with pytest.raises(ValueError, match="grid>=2"):
        interp1d_uniform(torch.zeros(1, 2, 1), 0.0, 1.0, torch.zeros(1, 3))
    with pytest.raises(ValueError, match="hi > lo"):
        interp1d_uniform(torch.zeros(1, 2, 4), 1.0, 1.0, torch.zeros(1, 3))


def test_boundary_context_field_shapes() -> None:
    field = BoundaryContextField(latent_dim=8, width=6, grid_size=16, n_blocks=1)
    out = field(torch.zeros(2, 5), torch.randn(2, 8), torch.zeros(2, 1))
    assert out.shape == (2, 5, 6)
    with pytest.raises(ValueError, match="grid_size"):
        BoundaryContextField(latent_dim=8, grid_size=2)
    with pytest.raises(ValueError, match="log_r_max"):
        BoundaryContextField(latent_dim=8, log_r_min=1.0, log_r_max=0.0)


def test_num_parameters_counts_trainable_weights(physics: PhysicsConfig) -> None:
    model = _make(physics)
    assert model.num_parameters == sum(p.numel() for p in model.parameters())
