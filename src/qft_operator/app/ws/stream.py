"""WebSocket endpoints for the interactive panels.

Both sockets are **request/response paced**: the client sends one JSON control message
and receives exactly one binary frame back. Pacing this way rather than free-running is
what a slider drag actually wants -- the client issues the next request only once the
previous frame has landed, so a slow frame throttles the input instead of building an
unbounded backlog of stale states the server would compute and throw away.

Frame layouts live in :mod:`qft_operator.app.ws.protocol`.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from qft_operator.app.services import TheorySpec, evaluate_correlator
from qft_operator.app.state import AppState, get_state
from qft_operator.app.ws.protocol import pack_bulk_field, pack_correlator

__all__ = ["router", "BulkRequest"]

LOGGER = logging.getLogger(__name__)
router = APIRouter()


class BulkRequest(BaseModel):
    """One frame's worth of bulk-density parameters.

    Attributes:
        r: Boundary separation between the two insertions.
        log_eps: $\\log\\epsilon$ of the near-boundary cutoff, i.e. $-\\log M$.
        n_z: Rows in the display grid (radial direction).
        n_p: Columns in the display grid (boundary direction).
        decades_below: Decades of $z$ shown below the separation.
        decades_above: Decades of $z$ shown above it.
    """

    r: float = Field(default=1.0, gt=1e-3, le=100.0)
    log_eps: float = Field(default=-6.0, ge=-16.0, le=0.0)
    n_z: int = Field(default=192, ge=8, le=1024)
    n_p: int = Field(default=256, ge=8, le=1024)
    decades_below: float = Field(default=3.0, gt=0.5, le=10.0)
    decades_above: float = Field(default=1.0, gt=0.1, le=6.0)


def _bulk_frame(state: AppState, request: BulkRequest, sequence: int) -> bytes:
    """Compute and encode one bulk-density frame."""
    import math

    eps = math.exp(request.log_eps)
    field = state.integrator.integrand_field(
        r=request.r,
        eps=eps,
        shape=(request.n_z, request.n_p),
        decades_below=request.decades_below,
        decades_above=request.decades_above,
    )
    return pack_bulk_field(
        sequence,
        field["log_density"],  # type: ignore[arg-type]
        float(field["log_z_min"]),  # type: ignore[arg-type]
        float(field["log_z_max"]),  # type: ignore[arg-type]
        float(field["p_min"]),  # type: ignore[arg-type]
        float(field["p_max"]),  # type: ignore[arg-type]
        request.r,
        request.log_eps,
        state.physics.delta,
        float(field["integral"]),  # type: ignore[arg-type]
    )


def _correlator_frame(state: AppState, spec: TheorySpec, sequence: int) -> bytes:
    """Compute and encode one correlator frame."""
    result = evaluate_correlator(state, spec)
    return pack_correlator(
        sequence,
        result.log_r,
        result.log_w_exact,
        result.log_w_pred,
        result.gamma_exact,
        result.gamma_pred,
        state.physics.free_dimension,
        spec.log_m,
        spec.coupling,
        result.running_coupling,
    )


@router.websocket("/ws/bulk")
async def bulk_stream(websocket: WebSocket) -> None:
    """Stream the AdS2 contact-integral density as the viewer drags the cutoff."""
    await websocket.accept()
    state = get_state()
    sequence = 0
    try:
        while True:
            payload = await websocket.receive_text()
            try:
                request = BulkRequest.model_validate_json(payload)
            except ValidationError as error:
                await websocket.send_json({"error": error.errors(include_url=False)})
                continue
            if request.n_z * request.n_p > state_grid_limit():
                await websocket.send_json({"error": "requested grid exceeds max_grid_points"})
                continue
            frame = await asyncio.to_thread(_bulk_frame, state, request, sequence)
            await websocket.send_bytes(frame)
            sequence += 1
    except WebSocketDisconnect:
        return


@router.websocket("/ws/correlator")
async def correlator_stream(websocket: WebSocket) -> None:
    """Stream exact and predicted correlators as the viewer dials theory parameters."""
    await websocket.accept()
    state = get_state()
    sequence = 0
    try:
        while True:
            payload = await websocket.receive_text()
            try:
                spec = TheorySpec.model_validate_json(payload)
            except ValidationError as error:
                await websocket.send_json({"error": error.errors(include_url=False)})
                continue
            frame = await asyncio.to_thread(_correlator_frame, state, spec, sequence)
            await websocket.send_bytes(frame)
            sequence += 1
    except WebSocketDisconnect:
        return


def state_grid_limit() -> int:
    """Maximum number of grid points a bulk request may ask the server to allocate."""
    from qft_operator.app.config import get_settings

    return get_settings().max_grid_points
