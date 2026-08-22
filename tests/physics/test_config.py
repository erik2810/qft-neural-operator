"""Conformal data, the BF bound, and agreement with the published gamma formula."""

from __future__ import annotations

import math

import pytest

from qft_operator.physics.config import PhysicsConfig


def test_delta_solves_the_mass_shell_relation() -> None:
    for m_sq in (-0.2, 0.0, 0.75, 2.0, 6.0):
        cfg = PhysicsConfig(m_sq=m_sq)
        assert cfg.delta * (cfg.delta - 1.0) == pytest.approx(m_sq * cfg.L**2, abs=1e-12)


def test_reference_mass_gives_delta_three_halves() -> None:
    assert PhysicsConfig(m_sq=0.75, L=1.0).delta == pytest.approx(1.5, abs=1e-12)


def test_shadow_dimension_is_the_second_root() -> None:
    cfg = PhysicsConfig(m_sq=0.75)
    assert cfg.delta_shadow * (cfg.delta_shadow - 1.0) == pytest.approx(cfg.m_sq, abs=1e-12)


def test_breitenlohner_freedman_bound_is_enforced() -> None:
    PhysicsConfig(m_sq=-0.25)  # exactly on the bound is allowed
    with pytest.raises(ValueError, match="Breitenlohner-Freedman"):
        PhysicsConfig(m_sq=-0.3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"L": 0.0}, "AdS radius"),
        ({"sigma_sq": -1.0}, "sigma_sq"),
        ({"c_delta": 0.0}, "c_delta"),
        ({"propagator_normalization": "bogus"}, "propagator_normalization"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PhysicsConfig(**kwargs)


def test_c_delta_cft_matches_the_gamma_function_expression() -> None:
    cfg = PhysicsConfig(m_sq=0.75)
    expected = math.gamma(1.5) / (math.sqrt(math.pi) * math.gamma(1.0))
    assert cfg.c_delta_cft == pytest.approx(expected, rel=1e-12)
    assert cfg.c_delta_cft == pytest.approx(0.5, rel=1e-12)


def test_convention_ratio_flags_the_reference_normalization() -> None:
    # The published c_delta = 0.159 is ~1/pi of the unit-normalized value; the property
    # exists precisely so that discrepancy is visible rather than silent.
    assert PhysicsConfig().convention_ratio == pytest.approx(1.0 / math.pi, rel=2e-3)
    assert PhysicsConfig(c_delta=None).convention_ratio == pytest.approx(1.0, rel=1e-12)


def test_log_coefficient_uses_the_configured_normalization() -> None:
    cfg = PhysicsConfig()
    assert cfg.log_coefficient == pytest.approx(
        2.0 * cfg.L**2 * cfg.c_delta_effective / (2.0 * cfg.delta - 1.0), rel=1e-12
    )
    cft = PhysicsConfig(propagator_normalization="cft")
    assert cft.log_coefficient == pytest.approx(2.0 * cft.L**2 * cft.c_delta_effective, rel=1e-12)


def test_analytical_anomalous_dimension_reproduces_the_published_formula() -> None:
    cfg = PhysicsConfig()
    lam, xi = 0.02, 0.8
    expected = (
        -lam
        * (2.0 * cfg.L**2 * cfg.c_delta_effective / (2.0 * cfg.delta - 1.0))
        * cfg.beta1
        * cfg.beta2
        * xi**2
    )
    assert cfg.analytical_anomalous_dim(lam, xi) == pytest.approx(expected, rel=1e-14)


def test_effective_dimension_shifts_downward_by_gamma() -> None:
    cfg = PhysicsConfig()
    assert cfg.effective_dimension(0.01) == pytest.approx(cfg.free_dimension - 0.01, rel=1e-14)
