"""The committed TypeScript parity fixture must stay in step with the Python source.

`frontend/src/lib/physics.ts`, `bulk.ts` and `operator.ts` re-implement parts of this
package so the static build needs no server. Two implementations of one physics is
exactly where silent divergence happens, so the TypeScript tests compare against golden
values generated here -- and this test fails when those values go stale.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from qft_operator.physics.config import PhysicsConfig
from tests.app.parity_fixture import FIXTURE_DIR, build_fixture, write_fixture


@pytest.fixture(scope="module")
def committed() -> dict:
    path = FIXTURE_DIR / "parity.json"
    if not path.is_file():
        pytest.skip("parity fixture not generated; run python -m tests.app.parity_fixture")
    return json.loads(path.read_text())


def assert_numerically_equal(fresh: Any, stored: Any, where: str = "") -> None:
    """Compare two fixture branches structurally, with a tolerance on the numbers.

    Byte equality is the wrong test here. ``math.gamma`` and ``sqrt`` differ in the last
    ULP between platforms, so a fixture generated on macOS will not serialize identically
    on a Linux runner even when nothing has changed -- which fails CI while telling the
    author their fixture is stale. What actually matters is that the values still agree to
    far tighter than the tolerance the TypeScript tests compare against.
    """
    assert type(fresh) is type(stored), f"type changed at {where or 'root'}"
    if isinstance(fresh, dict):
        assert set(fresh) == set(stored), f"keys changed at {where or 'root'}"
        for key in fresh:
            assert_numerically_equal(fresh[key], stored[key], f"{where}.{key}")
    elif isinstance(fresh, list):
        assert len(fresh) == len(stored), f"length changed at {where}"
        for index, (a, b) in enumerate(zip(fresh, stored, strict=True)):
            assert_numerically_equal(a, b, f"{where}[{index}]")
    elif isinstance(fresh, float):
        assert fresh == pytest.approx(stored, rel=1e-12, abs=1e-15), f"value changed at {where}"
    else:
        assert fresh == stored, f"value changed at {where}"


def test_fixture_is_current(committed: dict) -> None:
    regenerated = build_fixture()
    assert set(regenerated) == set(committed)
    for key in ("background", "gamma_function", "gammas", "correlators", "bulk"):
        try:
            assert_numerically_equal(regenerated[key], committed[key], key)
        except AssertionError as error:
            raise AssertionError(
                f"the {key!r} section is stale ({error}) -- regenerate with "
                "`uv run python -m tests.app.parity_fixture`"
            ) from error


def test_operator_cases_are_current(committed: dict) -> None:
    regenerated = build_fixture()["operator_cases"]
    assert len(regenerated) == len(committed["operator_cases"])
    for fresh, stored in zip(regenerated, committed["operator_cases"], strict=True):
        assert fresh["log_w"] == pytest.approx(stored["log_w"], abs=1e-6)
        assert fresh["v_phi"] == pytest.approx(stored["v_phi"], abs=1e-9)


def test_exported_weights_accompany_the_fixture() -> None:
    directory = FIXTURE_DIR / "operator"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["spectral_form"] == "fourier"
    assert manifest["phi_max"] == 3.0
    assert (directory / "weights.bin").stat().st_size == manifest["total_bytes"]


def test_fixture_covers_only_rng_free_families(committed: dict) -> None:
    # The TypeScript side draws polynomial and GP potentials with its own PRNG, so a seed
    # means "another draw from the same distribution", not the same function. Comparing
    # those across languages would be meaningless, so the fixture stays with the analytic
    # families -- and the frontend sends its own V(phi) to the server rather than relying
    # on the two samplers agreeing.
    families = {case["family"] for case in committed["gammas"]}
    assert families == {"free", "sine_gordon", "phi4"}


def test_regeneration_is_deterministic(tmp_path) -> None:
    # Byte equality is the right test here: both runs happen on the same machine, so any
    # difference is genuine non-determinism rather than a platform's libm.
    first = json.loads(write_fixture(tmp_path / "a").read_text())
    second = json.loads(write_fixture(tmp_path / "b").read_text())
    assert json.dumps(first) == json.dumps(second)
    left = (tmp_path / "a" / "operator" / "weights.bin").read_bytes()
    right = (tmp_path / "b" / "operator" / "weights.bin").read_bytes()
    assert left == right


def test_background_matches_the_shipped_conventions(committed: dict) -> None:
    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    assert committed["background"]["delta"] == pytest.approx(config.delta, abs=1e-12)
    assert committed["background"]["log_coefficient"] == pytest.approx(
        config.log_coefficient, abs=1e-12
    )
    assert committed["background"]["normalization_factor"] == pytest.approx(1.0, abs=1e-12)


def test_correlator_targets_are_exact_power_laws(committed: dict) -> None:
    free = committed["background"]["free_dimension"]
    for case in committed["correlators"]:
        log_r = torch.tensor(case["log_r"])
        log_w = torch.tensor(case["log_w"])
        expected = -2.0 * (free - case["gamma"]) * log_r
        assert torch.allclose(log_w, expected, atol=1e-12)
