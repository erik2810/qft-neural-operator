"""Operator export for browser inference."""

from __future__ import annotations

import json

import pytest
import torch

from qft_operator.cli.export_operator import (
    bake_spectral_kernel,
    export_operator,
    spectral_grid_sizes,
)
from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.models.layers import SpectralConv1d
from qft_operator.physics.config import PhysicsConfig

SMALL = {
    "n_phi": 16,
    "latent_dim": 16,
    "branch_width": 8,
    "branch_blocks": 1,
    "branch_modes": 4,
    "branch_hidden": [16, 16],
    "trunk_width": 16,
    "trunk_layers": 1,
    "trunk_modes": 4,
    "num_frequencies": 4,
    "context_grid": 16,
    "context_width": 8,
}


def test_baked_kernel_reproduces_the_spectral_layer() -> None:
    # Mode-truncated multiplication in Fourier space is *exactly* a circular convolution
    # on a fixed grid. That identity is what makes an ONNX-style export possible at all,
    # given aten::fft_rfft has no lowering.
    torch.manual_seed(0)
    length, c_in, c_out, modes = 64, 6, 5, 16
    layer = SpectralConv1d(c_in, c_out, modes).double()
    x = torch.randn(3, c_in, length, dtype=torch.float64)

    kernel = bake_spectral_kernel(layer, length)
    assert kernel.shape == (c_in, c_out, length)
    index = (torch.arange(length).view(-1, 1) - torch.arange(length).view(1, -1)) % length
    by_convolution = torch.einsum("bim,ionm->bon", x, kernel[:, :, index])

    assert torch.allclose(by_convolution, layer(x), rtol=1e-12, atol=1e-12)


def test_baked_kernel_rejects_an_impossible_length() -> None:
    layer = SpectralConv1d(2, 2, 8)
    with pytest.raises(ValueError, match="too small"):
        bake_spectral_kernel(layer, 0)


def test_grid_sizes_are_measured_not_assumed() -> None:
    # The branch stack and the boundary context field run on different grids; inferring
    # either from the config would break the moment one of them is resized.
    model = FourierDeepONet(PhysicsConfig(), **{**SMALL, "context_grid": 32}).eval()
    sizes = spectral_grid_sizes(model)
    assert sizes["branch.blocks.0.spectral"] == SMALL["n_phi"]
    assert sizes["trunk.context.blocks.0.spectral"] == 32


def test_export_writes_a_consistent_manifest(tmp_path) -> None:
    manifest_path = export_operator(None, tmp_path, phi_max=2.5)
    manifest = json.loads(manifest_path.read_text())
    blob = (tmp_path / "weights.bin").read_bytes()

    assert manifest["format"] == "qft-operator-weights"
    assert manifest["spectral_form"] == "fourier"
    assert manifest["trained"] is False
    assert manifest["phi_max"] == 2.5
    assert manifest["total_bytes"] == len(blob)

    # Offsets must tile the blob exactly, and every one must be 4-byte aligned so the
    # browser can build a Float32Array view without copying.
    cursor = 0
    for entry in manifest["tensors"]:
        assert entry["offset"] == cursor
        assert entry["offset"] % 4 == 0
        expected = 4
        for dimension in entry["shape"]:
            expected *= dimension
        assert entry["bytes"] == expected
        cursor += entry["bytes"]
    assert cursor == len(blob)


def test_export_can_bake_the_spectral_layers(tmp_path) -> None:
    fourier = json.loads(export_operator(None, tmp_path / "f").read_text())
    circular = json.loads(export_operator(None, tmp_path / "c", bake_spectral=True).read_text())

    assert circular["spectral_form"] == "circular"
    names = {entry["name"] for entry in circular["tensors"]}
    assert "branch.blocks.0.spectral.kernel" in names
    assert "branch.blocks.0.spectral.weight" not in names
    # Baking trades size for the removal of the FFT; the manifest should show that cost.
    assert circular["total_bytes"] > fourier["total_bytes"]


def _write_checkpoint(path, model: FourierDeepONet, *, record_architecture: bool):
    """Save a checkpoint in the layout the Lightning module produces."""
    hyper: dict = {"feature_scale": 0.375}
    if record_architecture:
        hyper["architecture"] = model.hyperparameters
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "hyper_parameters": hyper,
        },
        path,
    )


def test_export_round_trips_a_checkpoint(tmp_path) -> None:
    torch.manual_seed(1)
    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    model = FourierDeepONet(config, **SMALL)
    checkpoint = tmp_path / "run.ckpt"
    _write_checkpoint(checkpoint, model, record_architecture=True)

    manifest = json.loads(export_operator(checkpoint, tmp_path / "out", physics=config).read_text())
    assert manifest["trained"] is True
    assert manifest["feature_scale"] == pytest.approx(0.375)
    # Rebuilt from the recorded architecture, not from defaults plus an inferred n_phi.
    assert manifest["architecture"]["n_phi"] == SMALL["n_phi"]
    assert manifest["architecture"]["latent_dim"] == SMALL["latent_dim"]
    assert manifest["architecture"]["context_grid"] == SMALL["context_grid"]
    assert manifest["architecture"]["trunk_layers"] == SMALL["trunk_layers"]


def test_export_fails_loudly_on_a_checkpoint_without_its_architecture(tmp_path) -> None:
    # Older checkpoints record no architecture. Guessing defaults would build the wrong
    # network for any non-default run, so the load must raise rather than proceed.
    torch.manual_seed(1)
    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    model = FourierDeepONet(config, **SMALL)
    checkpoint = tmp_path / "legacy.ckpt"
    _write_checkpoint(checkpoint, model, record_architecture=False)

    with pytest.raises(RuntimeError, match="state_dict"):
        export_operator(checkpoint, tmp_path / "out", physics=config)


def test_default_architecture_checkpoints_still_load_without_a_record(tmp_path) -> None:
    # The fallback is meant to keep default-width runs working; only n_phi is inferable.
    torch.manual_seed(2)
    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    model = FourierDeepONet(config, n_phi=32)
    checkpoint = tmp_path / "default.ckpt"
    _write_checkpoint(checkpoint, model, record_architecture=False)

    manifest = json.loads(export_operator(checkpoint, tmp_path / "out", physics=config).read_text())
    assert manifest["architecture"]["n_phi"] == 32


def test_export_rejects_conflicting_sources(tmp_path) -> None:
    model = FourierDeepONet(PhysicsConfig(), **SMALL)
    with pytest.raises(ValueError, match="not both"):
        export_operator(tmp_path / "any.ckpt", tmp_path / "out", model=model)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        export_operator(tmp_path / "absent.ckpt", tmp_path / "out")


def test_exported_architecture_describes_the_forward_pass(tmp_path) -> None:
    model = FourierDeepONet(PhysicsConfig(), **SMALL)
    manifest = json.loads(export_operator(None, tmp_path, model=model).read_text())
    architecture = manifest["architecture"]
    # Everything the browser runtime needs to rebuild the graph, none of it recoverable
    # from the weight shapes alone.
    for key in (
        "radial_mode",
        "translation_invariant",
        "log_r_scale",
        "residual_mode",
        "free_dimension",
        "context_grid",
        "log_r_min",
        "log_r_max",
        "head",
    ):
        assert key in architecture
