"""Generate the golden values the frontend's TypeScript port is checked against.

`frontend/src/lib/physics.ts`, `bulk.ts` and `operator.ts` re-implement parts of this
package so the static build works with no server. Two implementations of the same physics
is exactly the situation where they quietly diverge, so the TypeScript side is pinned
against values produced here and :mod:`tests.app.test_frontend_parity` fails if the
committed fixture goes stale.

Regenerate with::

    uv run python -m tests.app.parity_fixture
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from qft_operator.cli.export_operator import export_operator
from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.physics.bulk_integrals import ConformalIntegrator, QuadratureSpec
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.correlators import (
    anomalous_dimension,
    log_boundary_two_point,
)
from qft_operator.physics.potentials import FreeTheory, PhiFour, Potential, SineGordon

__all__ = ["FIXTURE_DIR", "MODEL_KWARGS", "build_fixture", "write_fixture"]

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "__fixtures__"

#: Deliberately tiny -- the fixture is committed, and every code path is exercised
#: regardless of width. Randomly initialized rather than trained, which is the *stronger*
#: parity test: nothing is near zero, so a mis-ported layer cannot hide.
MODEL_KWARGS: dict[str, Any] = {
    "n_phi": 16,
    "latent_dim": 16,
    "branch_width": 8,
    "branch_blocks": 2,
    "branch_modes": 4,
    "branch_hidden": [24, 24],
    "trunk_width": 16,
    "trunk_layers": 2,
    "trunk_modes": 4,
    "num_frequencies": 4,
    "context_grid": 16,
    "context_width": 8,
    "readout_init_scale": 1.0,
}

#: TypeScript defaults to a 128x128 Gauss-Legendre grid; match it so the comparison is of
#: the two ports, not of two different quadrature resolutions.
QUADRATURE = QuadratureSpec(n_radial=128, n_boundary=128)

_SEED = 20260822


def _analytic_families() -> list[tuple[str, Potential, dict[str, float]]]:
    """Families whose TypeScript construction involves no RNG, hence is comparable."""
    return [
        ("free", FreeTheory(), {"coupling": 0.0, "xi": 0.8}),
        ("sine_gordon", SineGordon(0.02, 0.8), {"coupling": 0.02, "xi": 0.8}),
        ("sine_gordon", SineGordon(-0.045, 1.15), {"coupling": -0.045, "xi": 1.15}),
        ("phi4", PhiFour(0.03), {"coupling": 0.03, "xi": 0.8}),
    ]


def build_fixture(physics: PhysicsConfig | None = None) -> dict[str, Any]:
    """Compute every golden value.

    Args:
        physics: Background to evaluate against; defaults to unit-normalized conventions,
            which is what the frontend ships with.

    Returns:
        A JSON-serializable dict.
    """
    config = physics or PhysicsConfig(c_delta=None, propagator_normalization="cft")
    integrator = ConformalIntegrator(config, QUADRATURE)

    gammas = []
    correlators = []
    log_r = torch.linspace(math.log(0.05), math.log(12.0), 24, dtype=torch.float64)
    for family, potential, params in _analytic_families():
        # sigma^2 = 0.4 exercises the smearing branch, which is where the closed forms
        # differ most between families.
        for sigma_sq in (0.0, 0.4):
            background = PhysicsConfig(
                m_sq=config.m_sq,
                c_delta=config.c_delta,
                propagator_normalization=config.propagator_normalization,
                sigma_sq=sigma_sq,
            )
            gammas.append(
                {
                    "family": family,
                    **params,
                    "sigma_sq": sigma_sq,
                    "gamma": anomalous_dimension(potential, background),
                    "moment": potential.gaussian_second_moment(sigma_sq),
                }
            )
        gamma = anomalous_dimension(potential, config)
        correlators.append(
            {
                "family": family,
                **params,
                "gamma": gamma,
                "log_r": log_r.tolist(),
                "log_w": log_boundary_two_point(log_r, config.free_dimension - gamma).tolist(),
            }
        )

    bulk = []
    for r, eps in ((1.0, 1e-4), (0.3, 1e-5), (4.0, 1e-3)):
        radius = torch.tensor([r], dtype=torch.float64)
        bulk.append(
            {
                "r": r,
                "eps": eps,
                "contact_integral": float(integrator.contact_integral(radius, eps=eps)),
                "reduced": float(integrator.reduced_contact_integral(radius, eps=eps)),
                "measured_log_coefficient": float(integrator.log_slope(radius, eps=eps)),
            }
        )

    torch.manual_seed(_SEED)
    model = FourierDeepONet(config, **MODEL_KWARGS).eval()
    n_phi = MODEL_KWARGS["n_phi"]
    phi = torch.linspace(-3.0, 3.0, n_phi, dtype=torch.float64)
    cases = []
    generator = torch.Generator().manual_seed(_SEED + 1)
    for index, log_m in enumerate((0.0, 1.25, -0.75)):
        v_phi = (torch.randn(n_phi, generator=generator, dtype=torch.float64) * 0.3).to(
            torch.float32
        )
        query = torch.linspace(math.log(0.08), math.log(9.0), 12, dtype=torch.float32)
        coords = torch.stack([torch.zeros_like(query), torch.exp(query)], dim=-1).unsqueeze(0)
        with torch.no_grad():
            log_w = model(
                v_phi.unsqueeze(0), coords, torch.full((1, 1), log_m, dtype=torch.float32)
            ).squeeze(0)
        cases.append(
            {
                "index": index,
                "log_m": log_m,
                "v_phi": v_phi.tolist(),
                "log_r": query.tolist(),
                "log_w": log_w.tolist(),
            }
        )

    return {
        "background": {
            "L": config.L,
            "m_sq": config.m_sq,
            "delta": config.delta,
            "c_delta": config.c_delta_effective,
            "log_coefficient": config.log_coefficient,
            "free_dimension": config.free_dimension,
            "normalization_factor": config.normalization_factor,
        },
        "gamma_function": [
            {"x": x, "value": math.gamma(x)} for x in (0.5, 1.0, 1.5, 2.5, 4.0, 7.25)
        ],
        "gammas": gammas,
        "correlators": correlators,
        "bulk": bulk,
        "phi_grid": phi.tolist(),
        "operator_cases": cases,
    }


def write_fixture(directory: Path | None = None) -> Path:
    """Write ``parity.json`` and the matching exported model.

    Args:
        directory: Destination; defaults to :data:`FIXTURE_DIR`.

    Returns:
        Path to the written ``parity.json``.
    """
    target = directory or FIXTURE_DIR
    target.mkdir(parents=True, exist_ok=True)

    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    torch.manual_seed(_SEED)
    model = FourierDeepONet(config, **MODEL_KWARGS).eval()
    export_operator(None, target / "operator", physics=config, model=model)

    path = target / "parity.json"
    path.write_text(json.dumps(build_fixture(config), indent=1) + "\n")
    return path


if __name__ == "__main__":
    written = write_fixture()
    print(f"wrote {written} and {written.parent / 'operator'}")
