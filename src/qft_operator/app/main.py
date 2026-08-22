"""FastAPI app serving AdS2 holography and operator predictions.

Run with::

    uv run uvicorn qft_operator.app.main:app --reload

The REST routes cover everything the panels need at page load or in one-shot form; the
per-frame paths -- the bulk density field and the correlator sweep -- go over the
WebSockets in :mod:`qft_operator.app.ws.stream`.
"""

from __future__ import annotations

import asyncio
import math

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qft_operator.app.services import TheorySpec, evaluate_correlator
from qft_operator.app.state import AppState, get_state
from qft_operator.app.ws.stream import router as ws_router
from qft_operator.physics.bulk_integrals import analytic_log_coefficient

__all__ = ["create_app", "app"]


class BackgroundResponse(BaseModel):
    """Conformal data of the served AdS2 background."""

    L: float
    m_sq: float
    delta: float
    c_delta: float
    c_delta_cft: float
    convention_ratio: float
    log_coefficient: float
    free_dimension: float
    sigma_sq: float
    trained: bool
    n_phi: int


class CorrelatorRequest(BaseModel):
    """A theory plus the resolution at which to sample its correlator."""

    theory: TheorySpec = TheorySpec()
    n_points: int = Field(default=128, ge=8, le=1024)


class CorrelatorResponse(BaseModel):
    """Exact and predicted correlators, with the potential that generated them."""

    log_r: list[float]
    log_w_exact: list[float]
    log_w_pred: list[float]
    gamma_exact: float
    gamma_pred: float
    free_dimension: float
    running_coupling: float
    phi: list[float]
    potential: list[float]
    second_derivative: list[float]


class BulkIntegralResponse(BaseModel):
    """The regulated contact integral at one $(r, \\epsilon)$, against the closed form."""

    r: float
    eps: float
    integral: float
    reduced: float
    kappa: float
    log_coefficient_measured: float
    log_coefficient_analytic: float


def _correlator(state: AppState, request: CorrelatorRequest) -> CorrelatorResponse:
    """Evaluate a correlator request into its JSON response."""
    result = evaluate_correlator(state, request.theory, n_points=request.n_points)
    return CorrelatorResponse(
        log_r=result.log_r.tolist(),
        log_w_exact=result.log_w_exact.tolist(),
        log_w_pred=result.log_w_pred.tolist(),
        gamma_exact=result.gamma_exact,
        gamma_pred=result.gamma_pred,
        free_dimension=state.physics.free_dimension,
        running_coupling=result.running_coupling,
        phi=result.phi.tolist(),
        potential=result.potential.tolist(),
        second_derivative=result.second_derivative.tolist(),
    )


def _bulk_integral(state: AppState, r: float, eps: float) -> BulkIntegralResponse:
    """Evaluate the contact integral and its log-derivative at one point."""
    radius = torch.tensor([r], dtype=torch.float64)
    integral = float(state.integrator.contact_integral(radius, eps=eps))
    reduced = float(state.integrator.reduced_contact_integral(radius, eps=eps))
    return BulkIntegralResponse(
        r=r,
        eps=eps,
        integral=integral,
        reduced=reduced,
        kappa=float(state.integrator.kappa(radius, eps=eps)),
        log_coefficient_measured=float(state.integrator.log_slope(radius, eps=eps)),
        log_coefficient_analytic=analytic_log_coefficient(state.physics.delta, state.physics.L)
        / state.physics.normalization_factor,
    )


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="qft-neural-operator", version="0.1.0")
    # The static Pages build is served from a different origin than a local dev backend,
    # so it can point at one for live inference.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(ws_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/physics/background")
    async def background() -> BackgroundResponse:
        state = get_state()
        physics = state.physics
        return BackgroundResponse(
            L=physics.L,
            m_sq=physics.m_sq,
            delta=physics.delta,
            c_delta=physics.c_delta_effective,
            c_delta_cft=physics.c_delta_cft,
            convention_ratio=physics.convention_ratio,
            log_coefficient=physics.log_coefficient,
            free_dimension=physics.free_dimension,
            sigma_sq=physics.sigma_sq,
            trained=state.trained,
            n_phi=state.n_phi,
        )

    @app.post("/physics/correlator")
    async def correlator(request: CorrelatorRequest) -> CorrelatorResponse:
        return await asyncio.to_thread(_correlator, get_state(), request)

    @app.get("/physics/bulk-integral")
    async def bulk_integral(r: float = 1.0, log_eps: float = -6.0) -> BulkIntegralResponse:
        return await asyncio.to_thread(_bulk_integral, get_state(), r, math.exp(log_eps))

    return app


app = create_app()
