"""REST and WebSocket endpoints."""

from __future__ import annotations

import json
import math

import pytest
import torch
from fastapi.testclient import TestClient

from qft_operator.app.config import Settings
from qft_operator.app.main import create_app
from qft_operator.app.services import TheorySpec, evaluate_correlator
from qft_operator.app.state import build_state
from qft_operator.app.ws.protocol import bulk_frame_size, correlator_frame_size
from qft_operator.physics.bulk_integrals import analytic_log_coefficient


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_background_reports_the_conformal_data(client: TestClient) -> None:
    body = client.get("/physics/background").json()
    assert body["delta"] == pytest.approx(1.5, abs=1e-9)
    assert body["c_delta"] == pytest.approx(0.5, abs=1e-9)
    # The server defaults to unit-normalized conventions, so the analytic and numerically
    # integrated pipelines agree and the panels can be compared directly.
    assert body["convention_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert body["log_coefficient"] == pytest.approx(1.0, abs=1e-9)
    assert body["n_phi"] > 0


def test_bulk_integral_reproduces_the_analytic_coefficient(client: TestClient) -> None:
    body = client.get("/physics/bulk-integral", params={"r": 1.0, "log_eps": -6.0}).json()
    assert body["log_coefficient_analytic"] == pytest.approx(
        analytic_log_coefficient(1.5, 1.0), rel=1e-12
    )
    assert body["log_coefficient_measured"] == pytest.approx(
        body["log_coefficient_analytic"], rel=1e-3
    )
    assert body["reduced"] == pytest.approx(
        body["log_coefficient_analytic"] * (6.0 + body["kappa"]), rel=1e-6
    )


def test_bulk_integral_scheme_constant_converges_as_the_cutoff_is_removed(
    client: TestClient,
) -> None:
    # Ĩ depends on r and eps only through their ratio, so kappa is r-independent in the
    # limit. At finite cutoff it is not: the O(eps/r) tail of the asymptotic formula
    # spreads the values, most visibly at the smallest separation. The physical statement
    # is therefore that the spread *shrinks* with the cutoff, roughly a factor of five per
    # decade -- not that it is small at any particular one.
    def spread(log_eps: float) -> float:
        kappas = [
            client.get("/physics/bulk-integral", params={"r": r, "log_eps": log_eps}).json()[
                "kappa"
            ]
            for r in (0.2, 1.0, 5.0)
        ]
        return max(kappas) - min(kappas)

    spreads = [spread(log_eps) for log_eps in (-4.0, -5.0, -6.0, -7.0)]
    assert all(a > b for a, b in zip(spreads, spreads[1:], strict=False))
    assert spreads[-1] < 1e-3


def test_correlator_endpoint(client: TestClient) -> None:
    response = client.post(
        "/physics/correlator",
        json={"theory": {"family": "sine_gordon", "coupling": 0.02, "xi": 0.8}, "n_points": 32},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["log_r"]) == len(body["log_w_exact"]) == len(body["log_w_pred"]) == 32
    assert body["gamma_exact"] == pytest.approx(-0.02 * 1.0 * 0.8**2, rel=1e-6)
    assert len(body["phi"]) == len(body["potential"]) == len(body["second_derivative"])
    assert all(math.isfinite(v) for v in body["log_w_exact"])


def test_free_theory_correlator_is_the_exact_power_law(client: TestClient) -> None:
    body = client.post(
        "/physics/correlator", json={"theory": {"family": "free", "coupling": 0.0}, "n_points": 16}
    ).json()
    assert body["gamma_exact"] == 0.0
    for log_r, log_w in zip(body["log_r"], body["log_w_exact"], strict=True):
        assert log_w == pytest.approx(-2 * body["free_dimension"] * log_r, abs=1e-9)


def test_correlator_rejects_an_out_of_range_theory(client: TestClient) -> None:
    response = client.post(
        "/physics/correlator", json={"theory": {"family": "sine_gordon", "coupling": 99.0}}
    )
    assert response.status_code == 422


def test_bulk_websocket_streams_frames(client: TestClient) -> None:
    with client.websocket_connect("/ws/bulk") as socket:
        for index in range(3):
            socket.send_text(json.dumps({"r": 1.0, "log_eps": -5.0, "n_z": 24, "n_p": 32}))
            frame = socket.receive_bytes()
            assert len(frame) == bulk_frame_size(24, 32)
            assert int.from_bytes(frame[4:8], "little") == index


def test_bulk_websocket_reports_bad_requests_without_closing(client: TestClient) -> None:
    with client.websocket_connect("/ws/bulk") as socket:
        socket.send_text(json.dumps({"r": -1.0}))
        assert "error" in socket.receive_json()
        # The socket must survive a rejected frame, or a stray slider value would drop
        # the connection mid-drag.
        socket.send_text(json.dumps({"r": 1.0, "n_z": 16, "n_p": 16}))
        assert len(socket.receive_bytes()) == bulk_frame_size(16, 16)


def test_bulk_websocket_refuses_an_oversized_grid(client: TestClient) -> None:
    with client.websocket_connect("/ws/bulk") as socket:
        socket.send_text(json.dumps({"r": 1.0, "n_z": 1024, "n_p": 1024}))
        assert "max_grid_points" in json.dumps(socket.receive_json())


def test_correlator_websocket_streams_frames(client: TestClient) -> None:
    with client.websocket_connect("/ws/correlator") as socket:
        socket.send_text(json.dumps({"family": "sine_gordon", "coupling": 0.02, "xi": 0.8}))
        frame = socket.receive_bytes()
        assert len(frame) == correlator_frame_size(128)


def test_correlator_websocket_accepts_an_explicit_potential(client: TestClient) -> None:
    state = build_state()
    phi = state.phi_grid
    lam, xi = 0.03, 0.9
    request = {
        "family": "sine_gordon",
        "coupling": lam,
        "xi": xi,
        "moment": -2 * lam * xi**2,
        "v_phi": (-2 * lam * (torch.cosh(xi * phi) - 1)).tolist(),
    }
    with client.websocket_connect("/ws/correlator") as socket:
        socket.send_text(json.dumps(request))
        frame = socket.receive_bytes()
    assert len(frame) == correlator_frame_size(128)


def test_settings_are_environment_driven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QFT_OPERATOR_DEVICE", "cpu")
    monkeypatch.setenv("QFT_OPERATOR_FEATURE_SCALE", "0.25")
    monkeypatch.setenv("QFT_OPERATOR_CFT_NORMALIZATION", "false")
    settings = Settings()
    assert settings.feature_scale == 0.25
    assert settings.cft_normalization is False
    state = build_state(settings)
    assert state.feature_scale == 0.25
    # Without the CFT convention the analytic and numerical coefficients diverge, and the
    # config must report that rather than hide it.
    assert state.physics.convention_ratio != pytest.approx(1.0, abs=1e-3)


def test_untrained_state_is_labelled_as_such() -> None:
    state = build_state(Settings(checkpoint=None))
    assert state.trained is False
    # An untrained network sits on the free theory, which is why the label matters.
    result = evaluate_correlator(state, TheorySpec(family="sine_gordon", coupling=0.05, xi=1.0))
    assert abs(result.gamma_pred) < abs(result.gamma_exact)


def test_missing_checkpoint_falls_back_instead_of_crashing(tmp_path) -> None:
    state = build_state(Settings(checkpoint=str(tmp_path / "absent.ckpt")))
    assert state.trained is False
