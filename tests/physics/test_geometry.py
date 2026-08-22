"""AdS2 metric, isometry invariance and propagator normalization."""

from __future__ import annotations

import math

import pytest
import torch

from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.geometry import AdS2Geometry, c_delta_cft


@pytest.fixture
def geometry(physics: PhysicsConfig) -> AdS2Geometry:
    return AdS2Geometry(physics)


def test_sqrt_g_is_the_conformal_factor(geometry: AdS2Geometry) -> None:
    z = torch.tensor([0.1, 1.0, 7.0], dtype=torch.float64)
    assert torch.allclose(geometry.sqrt_g(z), geometry.L**2 / z**2, rtol=1e-12)
    assert torch.allclose(geometry.log_sqrt_g(z), torch.log(geometry.sqrt_g(z)), atol=1e-9)


def test_chordal_distance_is_invariant_under_dilatation(geometry: AdS2Geometry) -> None:
    z1, p1 = torch.tensor([0.4], dtype=torch.float64), torch.tensor([1.3], dtype=torch.float64)
    z2, p2 = torch.tensor([2.1], dtype=torch.float64), torch.tensor([-0.7], dtype=torch.float64)
    base = geometry.chordal_distance(z1, p1, z2, p2)
    for scale in (0.25, 3.7):
        sz1, sp1 = geometry.dilatation(z1, p1, scale)
        sz2, sp2 = geometry.dilatation(z2, p2, scale)
        assert torch.allclose(geometry.chordal_distance(sz1, sp1, sz2, sp2), base, rtol=1e-12)


def test_chordal_distance_is_invariant_under_translation(geometry: AdS2Geometry) -> None:
    z1, p1 = torch.tensor([0.4], dtype=torch.float64), torch.tensor([1.3], dtype=torch.float64)
    z2, p2 = torch.tensor([2.1], dtype=torch.float64), torch.tensor([-0.7], dtype=torch.float64)
    base = geometry.chordal_distance(z1, p1, z2, p2)
    tz1, tp1 = geometry.translation(z1, p1, 11.5)
    tz2, tp2 = geometry.translation(z2, p2, 11.5)
    assert torch.allclose(geometry.chordal_distance(tz1, tp1, tz2, tp2), base, rtol=1e-12)


def test_invariant_volume_element_is_dilatation_invariant(geometry: AdS2Geometry) -> None:
    # sqrt(g) dz dp picks up z^-2 * a * a = 1 under (z, p) -> (a z, a p).
    z = torch.tensor([0.3, 2.0], dtype=torch.float64)
    dz = torch.tensor([1e-3, 1e-3], dtype=torch.float64)
    dp = torch.tensor([1e-3, 1e-3], dtype=torch.float64)
    scale = 4.2
    base = geometry.volume_element(z, dz, dp)
    scaled = geometry.volume_element(scale * z, scale * dz, scale * dp)
    assert torch.allclose(scaled, base, rtol=1e-12)


def test_geodesic_distance_vanishes_at_coincident_points(geometry: AdS2Geometry) -> None:
    z = torch.tensor([1.5], dtype=torch.float64)
    p = torch.tensor([0.2], dtype=torch.float64)
    assert float(geometry.geodesic_distance(z, p, z, p)) == pytest.approx(0.0, abs=1e-9)


def test_bulk_to_boundary_integrates_to_unity(geometry: AdS2Geometry) -> None:
    # z^{Delta-1} int dp K_Delta(z, p) = 1 exactly, i.e. K -> z^{1-Delta} delta(p).
    z = torch.tensor([0.3], dtype=torch.float64)
    p = torch.linspace(-600.0, 600.0, 600_001, dtype=torch.float64)
    kernel = geometry.bulk_to_boundary(z, p, torch.zeros(1, dtype=torch.float64))
    integral = torch.trapezoid(kernel, p) * z ** (geometry.delta - 1.0)
    assert float(integral) == pytest.approx(1.0, rel=2e-6)


def test_bulk_to_boundary_transforms_covariantly(geometry: AdS2Geometry) -> None:
    # K_Delta(a z, a p; a p') = a^{-Delta} K_Delta(z, p; p'): weight-Delta covariance.
    z = torch.tensor([0.6], dtype=torch.float64)
    p = torch.tensor([1.1], dtype=torch.float64)
    p0 = torch.tensor([-0.4], dtype=torch.float64)
    scale = 3.3
    base = geometry.bulk_to_boundary(z, p, p0)
    scaled = geometry.bulk_to_boundary(scale * z, scale * p, scale * p0)
    assert torch.allclose(scaled, scale ** (-geometry.delta) * base, rtol=1e-12)


def test_c_delta_rejects_dimensions_below_the_unitarity_edge() -> None:
    with pytest.raises(ValueError, match="delta > 1/2"):
        c_delta_cft(0.5)


def test_bulk_to_bulk_decays_with_geodesic_separation(geometry: AdS2Geometry) -> None:
    z1 = torch.tensor([1.0], dtype=torch.float64)
    p1 = torch.tensor([0.0], dtype=torch.float64)
    values = [
        float(geometry.bulk_to_bulk(z1, p1, torch.tensor([1.0]), torch.tensor([offset])))
        for offset in (0.5, 1.0, 2.0, 4.0)
    ]
    assert all(a > b > 0.0 for a, b in zip(values, values[1:], strict=False))


def test_bulk_to_bulk_is_differentiable_via_scipy_bridge(geometry: AdS2Geometry) -> None:
    z2 = torch.tensor([1.7], dtype=torch.float64, requires_grad=True)
    value = geometry.bulk_to_bulk(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        z2,
        torch.tensor([0.9], dtype=torch.float64),
    )
    value.sum().backward()
    assert z2.grad is not None and math.isfinite(float(z2.grad))
