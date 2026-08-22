"""App configuration via pydantic-settings (env prefix QFT_OPERATOR_)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Runtime settings for the inference server.

    Attributes:
        device: Torch device for operator inference.
        checkpoint: Lightning checkpoint to serve. ``None`` serves a freshly initialized
            network, which sits on the free theory -- useful as a visible control, and it
            keeps every endpoint working before anything has been trained.
        cft_normalization: Use the unit-normalized $c_\\Delta$ rather than the published
            ``0.159``. Defaults to ``True`` so the analytic and numerically integrated
            curves the viewer sees side by side actually coincide; set to ``False`` to
            reproduce the published numbers instead.
        m_sq: Bulk mass squared, fixing $\\Delta$.
        sigma_sq: Regularized coincident-point propagator (the normal-ordering scheme).
        feature_scale: Override the branch-input normalization. Only needed for
            checkpoints written before it was recorded in the hyperparameters; without the
            right value the served model sees inputs off by that factor and its
            predictions are quietly wrong rather than obviously broken.
        max_grid_points: Upper bound on a requested bulk-density grid, so a client
            cannot ask the server for an arbitrarily large allocation.
    """

    model_config = SettingsConfigDict(env_prefix="QFT_OPERATOR_")

    device: str = "cpu"
    checkpoint: str | None = None
    cft_normalization: bool = True
    m_sq: float = 0.75
    sigma_sq: float = 0.0
    feature_scale: float | None = None
    max_grid_points: int = 1 << 18


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
