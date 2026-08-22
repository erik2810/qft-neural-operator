"""Binary frame protocol for the interactive AdS2 panels.

Mirrored byte-for-byte by ``frontend/src/lib/protocol.ts``; the round-trip is pinned by
:mod:`tests.app.test_protocol`.

Every frame is little-endian and starts with a common 8-byte header::

    uint16  magic    = 0x5146 ("QF")
    uint8   version  = 1
    uint8   kind     (1 = BULK_FIELD, 2 = CORRELATOR)
    uint32  sequence

JSON handles the control messages, which are small and infrequent. The payloads below
are not: a bulk-density frame is a full 2-D field regenerated on every slider tick, and
JSON-encoding it would cost roughly an order of magnitude in both bytes and parse time.

The density is quantized to **uint8** against a per-frame min/max. That is a deliberate
choice rather than a compromise: the field is consumed as a colour map, where 8 bits is
already below what the eye resolves, and it cuts the frame to a quarter of the float32
size. The *physics* number attached to the frame -- the converged value of the contact
integral -- travels as float32 and is never quantized.
"""

from __future__ import annotations

import struct
from enum import IntEnum

import torch
from torch import Tensor

__all__ = [
    "MAGIC",
    "VERSION",
    "FrameKind",
    "pack_bulk_field",
    "pack_correlator",
    "bulk_frame_size",
    "correlator_frame_size",
]

MAGIC = 0x5146
VERSION = 1

_HEADER = struct.Struct("<HBBI")
_BULK_HEADER = struct.Struct("<HH10f")
# The trailing "H" is explicit padding, not a field: it keeps the float32 payload at a
# 4-byte offset so the browser can wrap it in a Float32Array view directly. Unaligned
# typed-array views are a hard error in JS, not a slow path.
_CORRELATOR_HEADER = struct.Struct("<HH6f")


class FrameKind(IntEnum):
    """Discriminator for the payload following the common header."""

    BULK_FIELD = 1
    CORRELATOR = 2


def _header(kind: FrameKind, sequence: int) -> bytes:
    """Pack the common 8-byte frame header."""
    return _HEADER.pack(MAGIC, VERSION, int(kind), sequence & 0xFFFF_FFFF)


def pack_bulk_field(
    sequence: int,
    log_density: Tensor,
    log_z_min: float,
    log_z_max: float,
    p_min: float,
    p_max: float,
    r: float,
    log_eps: float,
    delta: float,
    integral: float,
    floor_decades: float = 8.0,
) -> bytes:
    """Encode a bulk-density frame.

    Args:
        sequence: Monotonic frame counter, echoed back for drop detection.
        log_density: Natural log of the integrand density, shape ``(n_z, n_p)``, row 0
            at the cutoff.
        log_z_min: $\\log z$ at the first row.
        log_z_max: $\\log z$ at the last row.
        p_min: Left edge of the displayed boundary window.
        p_max: Right edge of the displayed boundary window.
        r: Boundary separation of the two insertions.
        log_eps: $\\log\\epsilon$ of the cutoff.
        delta: Scaling dimension in force.
        integral: Converged value of the contact integral over $z > \\epsilon$.
        floor_decades: Dynamic range retained below the peak before clipping. The
            density spans many more decades than a colour map can show, so the floor is
            set relative to the frame maximum instead of absolutely.

    Returns:
        The encoded frame.

    Raises:
        ValueError: If ``log_density`` is not 2-D.
    """
    if log_density.ndim != 2:
        raise ValueError(f"log_density must be 2-D, got {tuple(log_density.shape)}")
    field = log_density.detach().to(torch.float32).cpu()
    high = float(field.max())
    low = high - floor_decades * 2.302585092994046  # decades -> nats
    scaled = ((field.clamp_min(low) - low) / (high - low + 1e-12) * 255.0).round()
    payload = scaled.to(torch.uint8).contiguous().numpy().tobytes()

    n_z, n_p = int(field.shape[0]), int(field.shape[1])
    header = _BULK_HEADER.pack(
        n_z, n_p, log_z_min, log_z_max, p_min, p_max, low, high, r, log_eps, delta, integral
    )
    return _header(FrameKind.BULK_FIELD, sequence) + header + payload


def pack_correlator(
    sequence: int,
    log_r: Tensor,
    log_w_exact: Tensor,
    log_w_pred: Tensor,
    gamma_exact: float,
    gamma_pred: float,
    free_dimension: float,
    log_m: float,
    coupling: float,
    running_coupling: float,
) -> bytes:
    """Encode a correlator frame: exact and predicted $\\log W$ on a shared $\\log r$ grid.

    Args:
        sequence: Monotonic frame counter.
        log_r: $\\log r$ grid, shape ``(n,)``.
        log_w_exact: Closed-form $\\log W$, shape ``(n,)``.
        log_w_pred: Operator prediction, shape ``(n,)``.
        gamma_exact: Closed-form anomalous dimension.
        gamma_pred: Anomalous dimension recovered from the prediction by a log-log fit.
        free_dimension: $\\Delta\\beta_1\\beta_2$, so the client can strip the leading
            power law without re-deriving it.
        log_m: $\\log M$ in force.
        coupling: $\\lambda(M)$.
        running_coupling: $\\bar\\lambda$ at the midpoint of the displayed window.

    Returns:
        The encoded frame.

    Raises:
        ValueError: If the three curves do not share one 1-D shape.
    """
    curves = (log_r, log_w_exact, log_w_pred)
    if any(c.ndim != 1 for c in curves) or len({c.shape for c in curves}) != 1:
        raise ValueError("log_r, log_w_exact and log_w_pred must share one 1-D shape")
    stacked = torch.stack(curves).to(torch.float32).contiguous().cpu()
    header = _CORRELATOR_HEADER.pack(
        int(log_r.shape[0]),
        0,  # padding
        gamma_exact,
        gamma_pred,
        free_dimension,
        log_m,
        coupling,
        running_coupling,
    )
    return _header(FrameKind.CORRELATOR, sequence) + header + stacked.numpy().tobytes()


def bulk_frame_size(n_z: int, n_p: int) -> int:
    """Byte length of a bulk frame at the given grid resolution."""
    return _HEADER.size + _BULK_HEADER.size + n_z * n_p


def correlator_frame_size(n: int) -> int:
    """Byte length of a correlator frame with ``n`` sample points."""
    return _HEADER.size + _CORRELATOR_HEADER.size + 3 * 4 * n
