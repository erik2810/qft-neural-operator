"""Data generation: sampling, exact labels, and RG invariance of the targets."""

from __future__ import annotations

import math

import pytest
import torch

from qft_operator.data.config import DataConfig
from qft_operator.data.dataset import AdS2CorrelatorDataset
from qft_operator.data.samplers import PotentialSampler
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import RGConfig


# ------------------------------------------------------------------- config --
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_train": 0}, "at least one sample"),
        ({"n_phi": 4}, "n_phi >= 8"),
        ({"r_min": 5.0, "r_max": 1.0}, "r_min < r_max"),
        ({"phi_max": 0.0}, "phi_max"),
        ({"coupling_range": (0.0, 1.0)}, "coupling_range"),
        ({"xi_range": (0.0, 1.0)}, "xi_range"),
        ({"poly_degree": 1}, "poly_degree"),
        ({"family_weights": {"bogus": 1.0}}, "unknown potential families"),
        ({"family_weights": {"free": 0.0}}, "positive weight"),
        ({"target_mode": "bogus"}, "unknown target_mode"),
    ],
)
def test_invalid_data_configurations_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DataConfig(**kwargs)


def test_family_weights_are_normalized() -> None:
    cfg = DataConfig(family_weights={"free": 1.0, "sine_gordon": 3.0})
    assert cfg.families == ("free", "sine_gordon")
    assert cfg.normalized_weights == pytest.approx((0.25, 0.75))


def test_log_r_range_matches_the_separation_window() -> None:
    cfg = DataConfig(r_min=0.05, r_max=12.0)
    assert cfg.log_r_range == pytest.approx((math.log(0.05), math.log(12.0)))


# ------------------------------------------------------------------ sampler --
def test_sampler_covers_every_configured_family(data_config: DataConfig) -> None:
    sampler = PotentialSampler(data_config, torch.Generator().manual_seed(0))
    seen = {sampler.sample().family for _ in range(400)}
    assert seen == set(data_config.families)


def test_sampler_is_reproducible(data_config: DataConfig) -> None:
    def draw() -> list[float]:
        sampler = PotentialSampler(data_config, torch.Generator().manual_seed(5))
        return [sampler.sample().coupling for _ in range(20)]

    assert draw() == draw()


def test_sampler_rejects_unknown_families(data_config: DataConfig) -> None:
    with pytest.raises(ValueError, match="unknown family"):
        PotentialSampler(data_config).sample("bogus")


def test_separations_form_a_sorted_grid_inside_the_window(data_config: DataConfig) -> None:
    sampler = PotentialSampler(data_config, torch.Generator().manual_seed(1))
    for _ in range(20):
        radii = sampler.sample_separations()
        assert radii.numel() == data_config.n_pairs
        assert bool((torch.diff(radii) > 0).all())
        assert float(radii.min()) >= data_config.r_min - 1e-12
        assert float(radii.max()) <= data_config.r_max + 1e-12


def test_shape_normalization_makes_the_coupling_the_scale(data_config: DataConfig) -> None:
    sampler = PotentialSampler(data_config, torch.Generator().manual_seed(2))
    for family in ("polynomial", "gp_fourier"):
        potential = sampler.sample(family)
        rms = float(potential.shape(sampler.phi_grid).pow(2).mean().sqrt())
        assert rms == pytest.approx(1.0, rel=1e-9)


# ------------------------------------------------------------------ dataset --
def test_dataset_item_shapes(data_config: DataConfig) -> None:
    dataset = AdS2CorrelatorDataset(12, data=data_config, seed=0)
    assert len(dataset) == 12
    item = dataset[0]
    assert item["v_phi"].shape == (data_config.n_phi,)
    assert item["dv_dcoupling"].shape == (data_config.n_phi,)
    assert item["coords"].shape == (data_config.n_pairs, 2)
    assert item["log_w"].shape == (data_config.n_pairs,)
    assert item["log_m"].shape == (1,)
    assert item["coupling"].ndim == 0 and item["gamma"].ndim == 0


def test_dataset_rejects_an_empty_split(data_config: DataConfig) -> None:
    with pytest.raises(ValueError, match="n_samples must be positive"):
        AdS2CorrelatorDataset(0, data=data_config)


def test_dataset_is_reproducible_from_its_seed(data_config: DataConfig) -> None:
    a = AdS2CorrelatorDataset(8, data=data_config, seed=3)
    b = AdS2CorrelatorDataset(8, data=data_config, seed=3)
    assert torch.equal(a.log_w, b.log_w) and torch.equal(a.gamma, b.gamma)
    c = AdS2CorrelatorDataset(8, data=data_config, seed=4)
    assert not torch.equal(a.gamma, c.gamma)


def test_targets_are_the_exact_power_law(physics: PhysicsConfig, data_config: DataConfig) -> None:
    dataset = AdS2CorrelatorDataset(16, physics=physics, data=data_config, seed=0)
    log_r = torch.log(dataset.separations)
    assert torch.allclose(dataset.log_w, -2.0 * dataset.delta_eff * log_r, atol=1e-5)


def test_free_samples_carry_the_free_theory_correlator(
    physics: PhysicsConfig, data_config: DataConfig
) -> None:
    dataset = AdS2CorrelatorDataset(64, physics=physics, data=data_config, seed=0)
    free = dataset.family == AdS2CorrelatorDataset.family_names.index("free")
    assert bool(free.any())
    assert float(dataset.gamma[free].abs().max()) == 0.0
    log_r = torch.log(dataset.separations[free])
    assert torch.allclose(dataset.log_w[free], -2.0 * physics.free_dimension * log_r, atol=1e-5)


def test_delta_eff_and_gamma_are_consistent(
    physics: PhysicsConfig, data_config: DataConfig
) -> None:
    # Delta_eff = Delta beta1 beta2 - gamma, exactly, for a marginal coupling.
    dataset = AdS2CorrelatorDataset(32, physics=physics, data=data_config, seed=9)
    expected = physics.free_dimension - dataset.gamma.unsqueeze(-1)
    assert torch.allclose(dataset.delta_eff, expected.expand_as(dataset.delta_eff), atol=1e-6)


def test_sine_gordon_labels_carry_the_expected_sign(
    physics: PhysicsConfig, data_config: DataConfig
) -> None:
    # gamma = -lambda (2 L^2 c_Delta / (2 Delta - 1)) beta1 beta2 xi^2 with xi^2 > 0, so
    # gamma must always oppose the sign of the coupling.
    dataset = AdS2CorrelatorDataset(96, physics=physics, data=data_config, seed=9)
    mask = dataset.family == AdS2CorrelatorDataset.family_names.index("sine_gordon")
    assert bool(mask.any())
    assert bool((dataset.gamma[mask] * dataset.coupling[mask] < 0).all())


def test_normal_ordered_phi_four_labels_vanish(
    physics: PhysicsConfig, data_config: DataConfig
) -> None:
    # With sigma^2 = 0 the quartic has <V''> = 12 lambda <phi^2> = 0, so its first-order
    # anomalous dimension must be exactly zero -- unlike the baseline's ad-hoc 0.4 * lam.
    dataset = AdS2CorrelatorDataset(96, physics=physics, data=data_config, seed=9)
    mask = dataset.family == AdS2CorrelatorDataset.family_names.index("phi4")
    assert bool(mask.any())
    assert float(dataset.gamma[mask].abs().max()) == 0.0

    with_tadpole = AdS2CorrelatorDataset(
        96, physics=PhysicsConfig(sigma_sq=0.4), data=data_config, seed=9
    )
    assert float(with_tadpole.gamma[mask].abs().max()) > 0.0


def test_branch_input_stays_linear_in_the_coupling(data_config: DataConfig) -> None:
    # A single global feature scale (not a per-sample one) is what preserves
    # V = lambda * dV/dlambda, on which the RG loss's chain rule depends.
    dataset = AdS2CorrelatorDataset(16, data=data_config, seed=1)
    reconstructed = dataset.coupling.unsqueeze(-1) * dataset.dv_dcoupling
    assert torch.allclose(dataset.v_phi, reconstructed, atol=1e-5)


def test_feature_scale_can_be_shared_across_splits(data_config: DataConfig) -> None:
    train = AdS2CorrelatorDataset(16, data=data_config, seed=0)
    val = AdS2CorrelatorDataset(8, data=data_config, seed=1, feature_scale=train.feature_scale)
    assert val.feature_scale == train.feature_scale


def test_disabling_standardization_leaves_the_scale_at_one() -> None:
    cfg = DataConfig(n_phi=32, n_pairs=8, standardize_inputs=False)
    assert AdS2CorrelatorDataset(8, data=cfg, seed=0).feature_scale == 1.0


def test_statistics_summarize_the_split(data_config: DataConfig) -> None:
    dataset = AdS2CorrelatorDataset(48, data=data_config, seed=0)
    stats = dataset.statistics
    assert sum(stats.family_counts.values()) == 48
    assert stats.gamma_abs_max >= abs(stats.gamma_mean)
    assert stats.gamma_std > 0.0


@pytest.mark.parametrize("rg_config", [RGConfig(), RGConfig(epsilon=0.35, log_scale_jitter=1.0)])
def test_renormalization_scale_is_sampled_and_targets_ignore_it(
    physics: PhysicsConfig, data_config: DataConfig, rg_config: RGConfig
) -> None:
    dataset = AdS2CorrelatorDataset(48, physics=physics, data=data_config, rg=rg_config, seed=0)
    assert float(dataset.log_m.std()) > 0.1
    # Every free sample must land on the identical correlator regardless of its own M.
    free = dataset.family == AdS2CorrelatorDataset.family_names.index("free")
    log_r = torch.log(dataset.separations[free])
    assert torch.allclose(dataset.log_w[free], -2.0 * physics.free_dimension * log_r, atol=1e-5)


def test_quadrature_mode_requires_a_cutoff_inside_the_smallest_separation(
    physics_cft: PhysicsConfig,
) -> None:
    cfg = DataConfig(n_phi=32, n_pairs=8, target_mode="quadrature")
    with pytest.raises(ValueError, match="well inside the smallest"):
        AdS2CorrelatorDataset(4, physics=physics_cft, data=cfg, rg=RGConfig(), seed=0)


@pytest.mark.slow
def test_hybrid_mode_reproduces_the_analytic_labels(physics_cft: PhysicsConfig) -> None:
    # With unit-normalized conventions the quadrature-measured C_log equals the analytic
    # one, so the hybrid pipeline must land on the same gamma as the closed form.
    cfg_analytic = DataConfig(n_phi=32, n_pairs=8, target_mode="resummed")
    cfg_hybrid = DataConfig(n_phi=32, n_pairs=8, target_mode="hybrid")
    analytic = AdS2CorrelatorDataset(24, physics=physics_cft, data=cfg_analytic, seed=0)
    hybrid = AdS2CorrelatorDataset(24, physics=physics_cft, data=cfg_hybrid, seed=0)
    assert torch.allclose(analytic.gamma, hybrid.gamma, rtol=2e-4, atol=1e-9)


def test_convention_mismatch_is_warned_about(physics: PhysicsConfig) -> None:
    cfg = DataConfig(n_phi=32, n_pairs=8, target_mode="hybrid")
    with pytest.warns(RuntimeWarning, match="unit-normalized CFT value"):
        AdS2CorrelatorDataset(4, physics=physics, data=cfg, seed=0)


def test_gamma_ratio_cap_removes_the_non_perturbative_tail(physics_cft: PhysicsConfig) -> None:
    # The labels are first order in the interaction, so they only mean anything while
    # gamma << Delta. Uncapped, the GP family produces draws at |gamma|/Delta ~ 0.15 --
    # a 15% shift in the boundary exponent -- and because the loss is quadratic those few
    # samples dominate it.
    cfg = DataConfig(n_phi=32, n_pairs=8)
    uncapped = AdS2CorrelatorDataset(
        800,
        physics=physics_cft,
        data=DataConfig(**{**cfg.__dict__, "max_gamma_ratio": None}),
        seed=0,
    )
    capped = AdS2CorrelatorDataset(
        800,
        physics=physics_cft,
        data=DataConfig(**{**cfg.__dict__, "max_gamma_ratio": 0.05}),
        seed=0,
    )
    ceiling = 0.05 * physics_cft.free_dimension
    assert float(uncapped.gamma.abs().max()) > ceiling
    assert float(capped.gamma.abs().max()) <= ceiling
    # A tail this thin costs almost nothing to remove.
    assert capped.statistics.rejected > 0
    assert capped.statistics.rejected < 0.05 * 800
    # And it is what makes the distribution lopsided.
    ratio = lambda d: float(d.gamma.abs().max() / d.gamma.abs().median())  # noqa: E731
    assert ratio(capped) < 0.5 * ratio(uncapped)


def test_only_the_gaussian_process_family_is_ever_rejected(physics_cft: PhysicsConfig) -> None:
    # free and normal-ordered phi^4 have gamma == 0 identically; Sine-Gordon and the
    # polynomials stay well inside the window at the shipped coupling range.
    cfg = DataConfig(n_phi=32, n_pairs=8, max_gamma_ratio=0.05)
    for family in ("free", "sine_gordon", "phi4", "polynomial"):
        weights = dict.fromkeys([family], 1.0)
        single = DataConfig(**{**cfg.__dict__, "family_weights": weights})
        dataset = AdS2CorrelatorDataset(200, physics=physics_cft, data=single, seed=1)
        assert dataset.statistics.rejected == 0, family


def test_no_cap_keeps_every_draw(physics_cft: PhysicsConfig) -> None:
    cfg = DataConfig(n_phi=32, n_pairs=8, max_gamma_ratio=None)
    assert (
        AdS2CorrelatorDataset(200, physics=physics_cft, data=cfg, seed=0).statistics.rejected == 0
    )


def test_an_unsatisfiable_cap_fails_rather_than_looping(physics_cft: PhysicsConfig) -> None:
    cfg = DataConfig(
        n_phi=32,
        n_pairs=8,
        max_gamma_ratio=1e-12,
        family_weights={"sine_gordon": 1.0},
        coupling_range=(0.04, 0.05),
    )
    with pytest.raises(RuntimeError, match="could not draw a theory"):
        AdS2CorrelatorDataset(4, physics=physics_cft, data=cfg, seed=0)


def test_gamma_ratio_cap_is_validated() -> None:
    with pytest.raises(ValueError, match="max_gamma_ratio"):
        DataConfig(max_gamma_ratio=0.0)


def test_cap_bounds_gamma_across_the_running_window(physics_cft: PhysicsConfig) -> None:
    # The cap must bound the gamma the *targets* carry, not the one the coupling happens
    # to be quoted at. With a relevant coupling the correlator is built from
    # lambda_bar(1/r), so gamma varies across the separation window -- at epsilon = 0.35 by
    # about a factor of six. Checking only the reference value let samples through four
    # times over the cap, in exactly the configuration the cap exists to protect.
    cfg = DataConfig(n_phi=32, n_pairs=16, max_gamma_ratio=0.05)
    rg = RGConfig(epsilon=0.35, two_loop=2.0, log_scale_jitter=1.0)
    dataset = AdS2CorrelatorDataset(400, physics=physics_cft, data=cfg, rg=rg, seed=0)

    free = physics_cft.free_dimension
    running = (free - dataset.delta_eff).abs() / free
    assert float(running.max()) <= 0.05 + 1e-6
    # The reference-scale gamma sits comfortably below the cap precisely because the
    # binding constraint is now the window edge.
    assert float(dataset.gamma.abs().max() / free) < 0.05
    # Enforcing the stronger bound necessarily costs more draws.
    assert dataset.statistics.rejected > 0


def test_marginal_flow_caps_on_the_reference_value(physics_cft: PhysicsConfig) -> None:
    # With no running the two notions coincide, so the stricter check must not make the
    # marginal case any more restrictive.
    cfg = DataConfig(n_phi=32, n_pairs=16, max_gamma_ratio=0.05)
    dataset = AdS2CorrelatorDataset(400, physics=physics_cft, data=cfg, rg=RGConfig(), seed=0)
    free = physics_cft.free_dimension
    assert float((free - dataset.delta_eff).abs().max() / free) <= 0.05 + 1e-6
    assert float(dataset.gamma.abs().max() / free) == pytest.approx(
        float((free - dataset.delta_eff).abs().max() / free), abs=1e-6
    )
