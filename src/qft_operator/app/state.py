"""Lazily-constructed physics and model singletons shared by every request."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from qft_operator.app.config import Settings, get_settings
from qft_operator.cli.export_operator import rebuild_model
from qft_operator.data.config import DataConfig
from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.physics.bulk_integrals import ConformalIntegrator, QuadratureSpec
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import BetaFunction, RGConfig

__all__ = ["AppState", "get_state"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppState:
    """Everything the routes need, built once.

    Attributes:
        physics: The AdS2 background.
        integrator: Quadrature engine for the bulk contact diagram.
        beta: RG flow used for the running coupling.
        model: The operator network, in eval mode.
        trained: Whether ``model`` came from a checkpoint. An untrained network sits on
            the free theory, so the frontend can label the prediction honestly instead of
            presenting noise as a result.
        n_phi: Field-grid resolution the model expects.
        feature_scale: Branch-input normalization recorded at training time.
    """

    physics: PhysicsConfig
    integrator: ConformalIntegrator
    beta: BetaFunction
    model: FourierDeepONet
    trained: bool
    n_phi: int
    feature_scale: float

    @property
    def phi_grid(self) -> torch.Tensor:
        """The field grid the branch input is sampled on."""
        return torch.linspace(-3.0, 3.0, self.n_phi, dtype=torch.float64)

    def summary(self) -> dict[str, float | bool | str]:
        """Flat description of the served background and checkpoint."""
        out: dict[str, float | bool | str] = dict(self.physics.summary())
        out["trained"] = self.trained
        out["n_phi"] = float(self.n_phi)
        out["feature_scale"] = self.feature_scale
        out["convention_ratio"] = self.physics.convention_ratio
        return out


def _load_checkpoint(path: Path, physics: PhysicsConfig) -> tuple[FourierDeepONet, int, float]:
    """Rebuild the network from a Lightning checkpoint's hyperparameters and weights."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in payload["state_dict"].items()}
    hyper = payload.get("hyper_parameters", {})
    model = rebuild_model(physics, state, hyper)
    return model, model.branch.n_phi, float(hyper.get("feature_scale", 1.0))


def build_state(settings: Settings | None = None) -> AppState:
    """Construct the shared state from settings."""
    config = settings or get_settings()
    physics = PhysicsConfig(
        m_sq=config.m_sq,
        sigma_sq=config.sigma_sq,
        c_delta=None if config.cft_normalization else 0.159,
        propagator_normalization="cft" if config.cft_normalization else "bulk_limit",
    )
    integrator = ConformalIntegrator(physics, QuadratureSpec(), device=config.device)

    trained = False
    n_phi, scale = DataConfig().n_phi, 1.0
    if config.checkpoint:
        path = Path(config.checkpoint)
        if path.is_file():
            model, n_phi, scale = _load_checkpoint(path, physics)
            trained = True
            LOGGER.info("serving checkpoint %s", path)
        else:
            LOGGER.warning("checkpoint %s not found; serving an untrained network", path)
            model = FourierDeepONet(physics, n_phi=n_phi)
    else:
        model = FourierDeepONet(physics, n_phi=n_phi)

    if config.feature_scale is not None:
        scale = config.feature_scale
        LOGGER.info("feature_scale overridden to %g", scale)

    model = model.to(config.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return AppState(
        physics=physics,
        integrator=integrator,
        beta=BetaFunction(RGConfig()),
        model=model,
        trained=trained,
        n_phi=n_phi,
        feature_scale=scale,
    )


@lru_cache
def get_state() -> AppState:
    """Return the process-wide :class:`AppState` singleton."""
    return build_state()
