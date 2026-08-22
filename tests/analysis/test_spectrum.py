"""Recovery of anomalous dimensions from correlators."""

from __future__ import annotations

import pytest
import torch

from qft_operator.analysis.spectrum import (
    anomalous_dimension_from_correlator,
    effective_dimension_from_correlator,
    fit_log_slope,
    summarize_spectrum,
)
from qft_operator.data.config import DataConfig
from qft_operator.data.dataset import AdS2CorrelatorDataset
from qft_operator.physics.config import PhysicsConfig


def test_fit_recovers_a_known_line() -> None:
    log_r = torch.linspace(-3.0, 3.0, 20, dtype=torch.float64).expand(3, 20)
    slopes = torch.tensor([[-3.0], [1.5], [0.0]], dtype=torch.float64)
    intercepts = torch.tensor([[0.5], [-2.0], [7.0]], dtype=torch.float64)
    fitted_slope, fitted_intercept = fit_log_slope(log_r, slopes * log_r + intercepts)
    assert torch.allclose(fitted_slope, slopes.squeeze(-1), atol=1e-10)
    assert torch.allclose(fitted_intercept, intercepts.squeeze(-1), atol=1e-10)


def test_weights_can_isolate_a_subrange() -> None:
    # A curve that is a clean power law only at large r: weighting must recover that
    # asymptotic slope rather than an average contaminated by the short-distance part.
    log_r = torch.linspace(-3.0, 3.0, 40, dtype=torch.float64).unsqueeze(0)
    log_w = torch.where(log_r < 0.0, -1.0 * log_r, -3.0 * log_r)
    weights = (log_r > 0.5).to(log_r.dtype)
    slope, _ = fit_log_slope(log_r, log_w, weights)
    assert float(slope) == pytest.approx(-3.0, abs=1e-9)


def test_effective_dimension_is_minus_half_the_slope() -> None:
    log_r = torch.linspace(-2.0, 2.0, 16, dtype=torch.float64).unsqueeze(0)
    log_w = -2.0 * 1.234 * log_r
    assert float(effective_dimension_from_correlator(log_r, log_w)) == pytest.approx(
        1.234, abs=1e-9
    )


def test_anomalous_dimension_is_the_deficit_from_the_free_exponent() -> None:
    log_r = torch.linspace(-2.0, 2.0, 16, dtype=torch.float64).unsqueeze(0)
    gamma = -0.0031
    log_w = -2.0 * (1.5 - gamma) * log_r
    recovered = anomalous_dimension_from_correlator(log_r, log_w, free_dimension=1.5)
    assert float(recovered) == pytest.approx(gamma, abs=1e-9)


def test_fit_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_log_slope(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match=r"expected \(batch, points\)"):
        fit_log_slope(torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="positively weighted"):
        fit_log_slope(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))


def test_exact_labels_are_recovered_from_exact_correlators(physics: PhysicsConfig) -> None:
    # End-to-end: the fit run on the dataset's own targets must reproduce the dataset's
    # own gamma, to float32 precision.
    dataset = AdS2CorrelatorDataset(64, physics=physics, data=DataConfig(n_phi=32), seed=2)
    recovered = anomalous_dimension_from_correlator(
        torch.log(dataset.separations), dataset.log_w, physics.free_dimension
    )
    assert torch.allclose(recovered, dataset.gamma, atol=2e-6)


def test_summary_reports_perfect_agreement() -> None:
    exact = torch.linspace(-0.01, 0.01, 40, dtype=torch.float64)
    report = summarize_spectrum(exact, exact)
    assert report.mae == pytest.approx(0.0, abs=1e-15)
    assert report.r2 == pytest.approx(1.0, abs=1e-12)


def test_summary_penalizes_a_constant_prediction() -> None:
    exact = torch.linspace(-0.01, 0.01, 40, dtype=torch.float64)
    report = summarize_spectrum(torch.zeros_like(exact), exact)
    # Predicting the free theory everywhere: zero explanatory power, relative MAE ~ 1.
    assert report.r2 == pytest.approx(0.0, abs=1e-9)
    assert report.relative_mae == pytest.approx(0.866, abs=0.05)


def test_median_error_and_r2_disagree_on_a_heavy_tail() -> None:
    # The gamma distribution is heavy-tailed: GP draws reach |gamma| an order of magnitude
    # above the family mean. R^2 is built from squared error and is therefore decided by
    # those few extremes, so it can rank two models in the opposite order to their
    # typical-case accuracy. Reporting both is the point.
    exact = torch.cat([torch.full((90,), 0.01), torch.tensor([0.5, -0.6])]).to(torch.float64)

    tail_only = exact.clone()
    tail_only[-2:] = 0.0
    tail_report = summarize_spectrum(tail_only, exact)

    bulk_only = exact + 0.02
    bulk_only[-2:] = exact[-2:]
    bulk_report = summarize_spectrum(bulk_only, exact)

    # By R^2 the bulk-only failure looks far better; by median error it is far worse.
    assert bulk_report.r2 > tail_report.r2
    assert bulk_report.median_relative_error > tail_report.median_relative_error
    assert tail_report.median_relative_error == pytest.approx(0.0, abs=1e-12)


def test_median_relative_error_is_robust_to_a_single_outlier() -> None:
    exact = torch.linspace(-0.01, 0.01, 101, dtype=torch.float64)
    predicted = exact.clone()
    predicted[0] += 1.0  # one catastrophic miss
    report = summarize_spectrum(predicted, exact)
    assert report.median_relative_error == pytest.approx(0.0, abs=1e-12)
    assert report.relative_mae > 1.0


def test_summary_breaks_results_down_by_family() -> None:
    exact = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    predicted = exact + torch.tensor([0.0, 0.0, 0.1, 0.1], dtype=torch.float64)
    family = torch.tensor([0, 0, 1, 1])
    report = summarize_spectrum(predicted, exact, family, ("free", "sine_gordon"))
    assert report.per_family["free"] == pytest.approx(0.0, abs=1e-12)
    assert report.per_family["sine_gordon"] == pytest.approx(0.1, rel=1e-9)


def test_summary_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_spectrum(torch.zeros(3), torch.zeros(4))


def test_summary_handles_a_degenerate_reference() -> None:
    constant = torch.zeros(8, dtype=torch.float64)
    report = summarize_spectrum(constant, constant)
    assert report.r2 != report.r2  # NaN: no variance to explain
    assert report.median_relative_error != report.median_relative_error
