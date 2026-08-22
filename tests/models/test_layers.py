"""Spectral convolutions, Fourier features and the metric-aware encoding."""

from __future__ import annotations

import math

import pytest
import torch

from qft_operator.models.layers import (
    MLP,
    FiLM,
    FourierBlock1d,
    FourierFeatures,
    MetricPositionalEncoding,
    SpectralConv1d,
)
from qft_operator.physics.config import PhysicsConfig


def test_spectral_conv_preserves_shape_and_is_resolution_agnostic() -> None:
    layer = SpectralConv1d(6, 4, n_modes=8)
    assert layer(torch.randn(3, 6, 64)).shape == (3, 4, 64)
    assert layer(torch.randn(3, 6, 128)).shape == (3, 4, 128)
    # Fewer grid points than modes must clamp rather than fail.
    assert layer(torch.randn(3, 6, 8)).shape == (3, 4, 8)


def test_spectral_conv_rejects_degenerate_shapes() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SpectralConv1d(0, 4, 8)


def test_spectral_conv_truncates_to_the_retained_modes() -> None:
    # A signal made only of frequencies above the cutoff must be annihilated.
    layer = SpectralConv1d(1, 1, n_modes=4)
    # An exactly periodic grid (endpoint excluded), so mode 20 is represented cleanly and
    # the test measures truncation rather than spectral leakage.
    grid = torch.arange(64, dtype=torch.float32) / 64.0 * 2.0 * math.pi
    high = torch.sin(20.0 * grid).view(1, 1, -1)
    assert float(layer(high).detach().abs().max()) < 1e-5


def test_spectral_conv_is_twice_differentiable() -> None:
    # The boundary-scaling loss differentiates the network twice; an in-place write into
    # a zeros buffer here would silently break that.
    layer = SpectralConv1d(2, 2, n_modes=4).double()
    x = torch.randn(1, 2, 16, dtype=torch.float64, requires_grad=True)
    (first,) = torch.autograd.grad(layer(x).sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), x, allow_unused=True, materialize_grads=True)
    assert torch.isfinite(second).all()


def test_spectral_conv_gradcheck() -> None:
    layer = SpectralConv1d(2, 3, n_modes=3).double()
    x = torch.randn(2, 2, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-6)


def test_fourier_block_is_a_residual_map() -> None:
    block = FourierBlock1d(8, n_modes=4)
    x = torch.randn(2, 8, 32)
    assert block(x).shape == x.shape


def test_fourier_features_shape_and_bounds() -> None:
    features = FourierFeatures(3, num_frequencies=7, scale=1.2)
    out = features(torch.randn(4, 5, 3))
    assert out.shape == (4, 5, 14) == (4, 5, features.out_features)
    assert float(out.abs().max()) <= 1.0 + 1e-6


def test_fourier_features_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="num_frequencies"):
        FourierFeatures(2, num_frequencies=0)
    with pytest.raises(ValueError, match="scale"):
        FourierFeatures(2, scale=0.0)


def test_fourier_features_can_be_trainable() -> None:
    fixed = FourierFeatures(2, trainable=False)
    trained = FourierFeatures(2, trainable=True)
    assert not any(p.requires_grad for p in fixed.parameters())
    assert any(p.requires_grad for p in trained.parameters())


def test_metric_encoding_is_translation_invariant_by_construction(
    physics: PhysicsConfig,
) -> None:
    encoding = MetricPositionalEncoding(physics, num_frequencies=6).double()
    coords = torch.rand(3, 12, 2, dtype=torch.float64) * 4.0 + 0.2
    log_m = torch.zeros(3, 1, dtype=torch.float64)
    base = encoding(coords, log_m)
    for shift in (1.0, -17.5, 250.0):
        assert torch.allclose(encoding(coords + shift, log_m), base, atol=1e-10)


def test_metric_encoding_embeds_the_conformal_factor(physics: PhysicsConfig) -> None:
    # The second invariant channel is log sqrt(g) / (2 log_r_scale) with sqrt(g) = L^2/z*^2.
    encoding = MetricPositionalEncoding(physics, num_frequencies=1, radial_mode="separation")
    r = torch.tensor([[0.5, 2.0, 8.0]])
    coords = torch.stack([torch.zeros_like(r), r], dim=-1)
    channel = encoding(coords, torch.zeros(1, 1))[..., 1] * (2.0 * encoding.log_r_scale)
    expected = 2.0 * math.log(physics.L) - 2.0 * torch.log(r / 2.0)
    assert torch.allclose(channel, expected, atol=1e-5)


@pytest.mark.parametrize("radial_mode", ["separation", "rg_scale", "geometric"])
def test_holographic_depth_modes(physics: PhysicsConfig, radial_mode: str) -> None:
    encoding = MetricPositionalEncoding(physics, radial_mode=radial_mode)
    log_r = torch.tensor([0.0, 1.0], dtype=torch.float64)
    log_m = torch.tensor([0.5, 0.5], dtype=torch.float64)
    depth = encoding.holographic_depth(log_r, log_m)
    expected = {
        "separation": log_r - math.log(2.0),
        "rg_scale": -log_m,
        "geometric": 0.5 * (log_r - math.log(2.0) - log_m),
    }[radial_mode]
    assert torch.allclose(depth, expected, atol=1e-12)


def test_metric_encoding_rejects_bad_arguments(physics: PhysicsConfig) -> None:
    with pytest.raises(ValueError, match="radial_mode"):
        MetricPositionalEncoding(physics, radial_mode="nonsense")
    with pytest.raises(ValueError, match="log_r_scale"):
        MetricPositionalEncoding(physics, log_r_scale=0.0)
    with pytest.raises(ValueError, match="trailing dim of 2"):
        MetricPositionalEncoding(physics)(torch.zeros(1, 4, 3))


def test_translation_variant_encoding_sees_the_midpoint(physics: PhysicsConfig) -> None:
    encoding = MetricPositionalEncoding(physics, translation_invariant=False).double()
    coords = torch.rand(2, 6, 2, dtype=torch.float64) + 0.5
    log_m = torch.zeros(2, 1, dtype=torch.float64)
    assert not torch.allclose(encoding(coords + 3.0, log_m), encoding(coords, log_m), atol=1e-6)


def test_film_is_the_identity_at_initialization() -> None:
    film = FiLM(16, 8)
    features = torch.randn(2, 5, 8)
    assert torch.allclose(film(features, torch.randn(2, 16)), features, atol=1e-7)


def test_mlp_shapes() -> None:
    net = MLP(4, [16, 8], 3, dropout=0.1)
    assert net(torch.randn(2, 7, 4)).shape == (2, 7, 3)
