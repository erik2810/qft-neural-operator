"""Binary frame layout.

The decoder lives in TypeScript (`frontend/src/lib/protocol.ts`), so these tests pin the
byte offsets the two sides agree on. A silent layout change here is the kind of bug that
shows up as a plausible-looking but wrong picture rather than an exception.
"""

from __future__ import annotations

import struct

import pytest
import torch

from qft_operator.app.ws.protocol import (
    MAGIC,
    VERSION,
    FrameKind,
    bulk_frame_size,
    correlator_frame_size,
    pack_bulk_field,
    pack_correlator,
)

HEADER = struct.Struct("<HBBI")


def test_header_layout() -> None:
    frame = pack_bulk_field(7, torch.zeros(4, 5), -6.9, 2.3, -2, 2, 1.0, -6.9, 1.5, 6.6)
    magic, version, kind, sequence = HEADER.unpack(frame[: HEADER.size])
    assert magic == MAGIC == 0x5146
    assert version == VERSION
    assert kind == FrameKind.BULK_FIELD
    assert sequence == 7


def test_bulk_frame_size_matches_the_declared_layout() -> None:
    for n_z, n_p in ((4, 5), (16, 24), (192, 256)):
        frame = pack_bulk_field(0, torch.zeros(n_z, n_p), 0, 1, -1, 1, 1, 0, 1.5, 0)
        assert len(frame) == bulk_frame_size(n_z, n_p) == 8 + 44 + n_z * n_p


def test_correlator_frame_size_matches_the_declared_layout() -> None:
    for n in (8, 64, 128):
        curves = [torch.zeros(n) for _ in range(3)]
        frame = pack_correlator(0, *curves, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
        assert len(frame) == correlator_frame_size(n) == 8 + 28 + 3 * 4 * n


def test_correlator_payload_is_four_byte_aligned() -> None:
    # JavaScript refuses to build a Float32Array view at an unaligned offset, so the
    # two padding bytes in the correlator header are load-bearing, not cosmetic.
    assert (8 + 28) % 4 == 0


def test_bulk_scalars_round_trip_as_float32() -> None:
    frame = pack_bulk_field(1, torch.zeros(2, 3), -6.9, 2.3, -2.0, 2.0, 1.25, -6.9, 1.5, 6.625)
    values = struct.unpack("<HH10f", frame[8 : 8 + 44])
    assert values[0] == 2 and values[1] == 3
    assert values[2] == pytest.approx(-6.9, abs=1e-5)
    assert values[8] == pytest.approx(1.25, abs=1e-6)
    assert values[11] == pytest.approx(6.625, abs=1e-6)


def test_bulk_density_is_quantized_against_the_frame_range() -> None:
    # The peak must land on 255 and anything below the floor on 0, since the client
    # reconstructs physical units from the transmitted [low, high] pair.
    field = torch.full((2, 4), -50.0)
    field[0, 0] = 0.0
    frame = pack_bulk_field(0, field, 0, 1, -1, 1, 1, 0, 1.5, 0, floor_decades=4.0)
    payload = frame[8 + 44 :]
    assert payload[0] == 255
    assert set(payload[1:]) == {0}


def test_correlator_curves_round_trip() -> None:
    n = 6
    log_r = torch.linspace(-2.0, 2.0, n)
    exact = -3.0 * log_r
    predicted = exact + 0.01
    frame = pack_correlator(3, log_r, exact, predicted, -1e-3, -9e-4, 1.5, 0.25, 0.02, 0.021)

    header = struct.unpack("<HH6f", frame[8 : 8 + 28])
    assert header[0] == n
    assert header[2] == pytest.approx(-1e-3, abs=1e-9)
    assert header[4] == pytest.approx(1.5, abs=1e-6)

    payload = struct.unpack(f"<{3 * n}f", frame[8 + 28 :])
    assert payload[:n] == pytest.approx(log_r.tolist(), abs=1e-6)
    assert payload[n : 2 * n] == pytest.approx(exact.tolist(), abs=1e-6)
    assert payload[2 * n :] == pytest.approx(predicted.tolist(), abs=1e-6)


def test_pack_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pack_bulk_field(0, torch.zeros(4), 0, 1, -1, 1, 1, 0, 1.5, 0)
    with pytest.raises(ValueError, match="one 1-D shape"):
        pack_correlator(0, torch.zeros(4), torch.zeros(5), torch.zeros(4), 0, 0, 1.5, 0, 0, 0)
