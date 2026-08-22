"""Shared fixtures.

Physics assertions run in float64 wherever an exact identity is under test; float32
would confuse a genuine violation with rounding noise.
"""

from __future__ import annotations

import pytest
import torch

from qft_operator.data.config import DataConfig
from qft_operator.physics.bulk_integrals import ConformalIntegrator, QuadratureSpec
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import BetaFunction, RGConfig


@pytest.fixture
def physics() -> PhysicsConfig:
    """Reference AdS2 background: Delta = 3/2, published c_delta normalization."""
    return PhysicsConfig()


@pytest.fixture
def physics_cft() -> PhysicsConfig:
    """Unit-normalized background, in which analytic and numeric pipelines agree."""
    return PhysicsConfig(c_delta=None, propagator_normalization="cft")


@pytest.fixture
def integrator(physics_cft: PhysicsConfig) -> ConformalIntegrator:
    """Quadrature engine at a resolution that converges to ~1e-8 for Delta = 3/2."""
    return ConformalIntegrator(physics_cft, QuadratureSpec(n_radial=192, n_boundary=192))


@pytest.fixture
def beta() -> BetaFunction:
    """A non-trivial two-loop beta function, so the flow map is actually exercised."""
    return BetaFunction(RGConfig(epsilon=0.3, two_loop=1.5))


@pytest.fixture
def data_config() -> DataConfig:
    """Small dataset configuration for fast tests."""
    return DataConfig(n_train=16, n_val=8, n_test=8, n_phi=32, n_pairs=16, batch_size=4)


@pytest.fixture
def phi_grid() -> torch.Tensor:
    """Field grid in float64."""
    return torch.linspace(-3.0, 3.0, 129, dtype=torch.float64)
