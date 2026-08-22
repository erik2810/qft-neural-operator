"""Plot generation. Figures are rendered headless and only checked for structure."""

from __future__ import annotations

import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from qft_operator.data.config import DataConfig  # noqa: E402
from qft_operator.data.dataset import AdS2CorrelatorDataset  # noqa: E402
from qft_operator.data.samplers import PotentialSampler  # noqa: E402
from qft_operator.physics.bulk_integrals import ConformalIntegrator, QuadratureSpec  # noqa: E402
from qft_operator.physics.config import PhysicsConfig  # noqa: E402
from qft_operator.physics.rg import BetaFunction, RGConfig  # noqa: E402
from qft_operator.viz import (  # noqa: E402
    plot_anomalous_spectrum,
    plot_bulk_integral_convergence,
    plot_correlator_comparison,
    plot_log_residuals,
    plot_potential_gallery,
    plot_rg_flow,
    set_style,
)


@pytest.fixture(scope="module")
def dataset() -> AdS2CorrelatorDataset:
    return AdS2CorrelatorDataset(12, data=DataConfig(n_phi=32, n_pairs=16), seed=0)


def test_set_style_is_idempotent() -> None:
    set_style()
    set_style(context="talk", grid=False)


def test_correlator_comparison(dataset: AdS2CorrelatorDataset) -> None:
    log_r = torch.log(dataset.separations[:2])
    figure = plot_correlator_comparison(log_r, dataset.log_w[:2], dataset.log_w[:2] + 0.01)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_xscale() == "log"


def test_correlator_comparison_validates_shapes(dataset: AdS2CorrelatorDataset) -> None:
    log_r = torch.log(dataset.separations[:2])
    with pytest.raises(ValueError, match="must share a shape"):
        plot_correlator_comparison(log_r, dataset.log_w[:2], dataset.log_w[:1])


def test_log_residuals_has_two_panels(dataset: AdS2CorrelatorDataset) -> None:
    log_r = torch.log(dataset.separations[:3])
    figure = plot_log_residuals(log_r, dataset.log_w[:3], dataset.log_w[:3] + 0.002, 1.5)
    assert len(figure.axes) == 2


def test_anomalous_spectrum_with_and_without_families(
    dataset: AdS2CorrelatorDataset,
) -> None:
    predicted = dataset.gamma + 1e-4
    assert plot_anomalous_spectrum(dataset.gamma, predicted) is not None
    figure = plot_anomalous_spectrum(
        dataset.gamma, predicted, dataset.family, AdS2CorrelatorDataset.family_names
    )
    assert figure.axes[0].get_legend() is not None


def test_potential_gallery() -> None:
    sampler = PotentialSampler(DataConfig(n_phi=64), torch.Generator().manual_seed(0))
    figure = plot_potential_gallery(sampler.phi_grid, [sampler.sample() for _ in range(4)])
    assert len(figure.axes) == 2


def test_rg_flow() -> None:
    figure = plot_rg_flow(BetaFunction(RGConfig(epsilon=0.3, two_loop=2.0)))
    assert len(figure.axes) == 2


def test_bulk_integral_convergence() -> None:
    integrator = ConformalIntegrator(
        PhysicsConfig(c_delta=None, propagator_normalization="cft"),
        QuadratureSpec(n_radial=64, n_boundary=64),
    )
    figure = plot_bulk_integral_convergence(
        integrator,
        radii=torch.tensor([1.0, 4.0], dtype=torch.float64),
        eps_values=torch.logspace(-4.0, -2.0, 3, dtype=torch.float64),
    )
    assert len(figure.axes) == 1
