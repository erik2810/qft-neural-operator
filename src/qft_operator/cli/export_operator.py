"""Export a trained operator for browser inference.

The network is a Fourier neural operator, so its spectral layers call ``torch.fft`` --
and ``aten::fft_rfft`` has no ONNX lowering at opset 17, which rules out the usual
export route.

Two forms are therefore available.

``"fourier"`` (default)
    The learned complex weights are exported as-is, and the browser runtime does its own
    64-point real FFT. This is what ``frontend/src/lib/operator.ts`` implements. It is
    both the compact form (weights stay at ``n_modes``, not the full grid) and by far the
    faster one: a spectral layer costs $C_{\\rm in}C_{\\rm out}M$ complex multiplies
    instead of the $C_{\\rm in}C_{\\rm out}N^2$ of a dense circular convolution -- around
    a thousandfold at the default widths, which is the difference between a responsive
    slider and a frozen one.

``"circular"`` (``--bake-spectral``)
    On a **fixed** grid of $N$ points, mode-truncated multiplication in Fourier space is
    exactly a circular convolution with the real kernel
    $k = \\mathrm{irfft}(W_{\\rm pad}, n=N)$. Baking that removes the FFT from inference
    entirely, which is what an ONNX or TFLite consumer needs. The equivalence is exact --
    it matches the FFT path to ~1e-15 relative, pinned by :mod:`tests.cli.test_export` --
    but it costs both size and speed, and it pins the model to one resolution.

Emits ``weights.bin`` (concatenated little-endian float32) plus ``manifest.json``
describing the architecture and the tensor offsets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.models.layers import SpectralConv1d
from qft_operator.physics.config import PhysicsConfig

__all__ = ["bake_spectral_kernel", "spectral_grid_sizes", "export_operator", "main"]

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION = 1


def bake_spectral_kernel(layer: SpectralConv1d, length: int) -> Tensor:
    """Convert a spectral layer's Fourier weights into a real circular-convolution kernel.

    Args:
        layer: The spectral convolution to bake.
        length: Grid length $N$ the layer operates on, at least 2. The kernel is only
            valid at this resolution -- the one trade the bake makes, since the FFT form
            is resolution-agnostic and this one is not.

    Returns:
        Real kernel of shape ``(in_channels, out_channels, length)`` satisfying
        $y[o,n] = \\sum_i \\sum_m x[i,m]\\, k[i,o,(n-m) \\bmod N]$.

    Raises:
        ValueError: If ``length`` is too small for a real transform.
    """
    if length < 2:
        raise ValueError(f"length {length} is too small for a real transform")
    modes = min(layer.n_modes, length // 2 + 1)
    weight = torch.view_as_complex(layer.weight.detach()[:, :, :modes, :].contiguous())
    pad = torch.zeros(weight.shape[0], weight.shape[1], length // 2 + 1 - modes, dtype=weight.dtype)
    return torch.fft.irfft(torch.cat([weight, pad], dim=-1), n=length, dim=-1)


def spectral_grid_sizes(model: FourierDeepONet, n_points: int = 32) -> dict[str, int]:
    """Record the grid length every spectral layer actually sees.

    Determined by running one forward pass with hooks rather than inferred from the
    config, because the branch stack and the boundary context field run on different
    grids and nothing guarantees they stay the same size.

    Args:
        model: The operator network.
        n_points: Number of query points in the probe batch.

    Returns:
        Mapping from module path to grid length.
    """
    sizes: dict[str, int] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, SpectralConv1d):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _m, args, key=name: sizes.__setitem__(key, int(args[0].shape[-1]))
                )
            )
    try:
        with torch.no_grad():
            radii = torch.exp(torch.linspace(-2.0, 2.0, n_points)).unsqueeze(0)
            coords = torch.stack([torch.zeros_like(radii), radii], dim=-1)
            model(torch.zeros(1, model.branch.n_phi), coords, torch.zeros(1, 1))
    finally:
        for handle in handles:
            handle.remove()
    return sizes


def rebuild_model(
    config: PhysicsConfig, state: dict[str, Any], hyper: dict[str, Any]
) -> FourierDeepONet:
    """Reconstruct the network a checkpoint holds and load its weights.

    Prefers the architecture recorded at training time. Falling back to defaults and only
    inferring ``n_phi`` -- which is all the weight shapes reveal without unpicking every
    layer -- silently builds the wrong network for any run that changed a width, so the
    fallback warns and lets ``load_state_dict`` raise rather than guessing further.

    Args:
        config: AdS2 background to build against.
        state: The checkpoint's ``state_dict``, with any ``model.`` prefix removed.
        hyper: The checkpoint's ``hyper_parameters``.

    Returns:
        The loaded network.
    """
    architecture = hyper.get("architecture")
    if architecture:
        model = FourierDeepONet.from_hyperparameters(config, architecture)
    else:
        n_phi = int(state["branch.phi_grid"].shape[0])
        LOGGER.warning("checkpoint records no architecture; assuming defaults with n_phi=%d", n_phi)
        model = FourierDeepONet(config, n_phi=n_phi)
    model.load_state_dict(state)
    return model


def _architecture(model: FourierDeepONet) -> dict[str, Any]:
    """Collect the hyperparameters the browser forward pass needs."""
    encoding = model.trunk.encoding
    context = model.trunk.context
    return {
        "n_phi": model.branch.n_phi,
        "latent_dim": model.branch.latent_dim,
        "branch_spectral": model.branch.use_spectral,
        "branch_blocks": len(model.branch.blocks),
        "n_tokens": model.branch.n_tokens,
        "emit_tokens": model.branch.emit_tokens,
        "trunk_layers": len(model.trunk.films),
        "spectral_mixing": model.trunk.spectral_mixing,
        "context_grid": int(context.grid.shape[0]) if context is not None else 0,
        "context_width": context.width if context is not None else 0,
        "context_blocks": len(context.blocks) if context is not None else 0,
        "log_r_min": context.log_r_min if context is not None else 0.0,
        "log_r_max": context.log_r_max if context is not None else 0.0,
        "num_invariants": encoding.num_invariants,
        "num_frequencies": encoding.fourier.num_frequencies,
        "radial_mode": encoding.radial_mode,
        "translation_invariant": encoding.translation_invariant,
        "log_r_scale": encoding.log_r_scale,
        "head": model.head_kind,
        "residual_mode": model.residual_mode,
        "free_dimension": model.free_dimension,
        "L": model.config.L,
    }


def export_operator(
    checkpoint: Path | None,
    output_dir: Path,
    physics: PhysicsConfig | None = None,
    feature_scale: float = 1.0,
    bake_spectral: bool = False,
    model: FourierDeepONet | None = None,
    phi_max: float = 3.0,
    dtype: str = "float32",
) -> Path:
    """Write ``weights.bin`` and ``manifest.json`` for browser inference.

    Args:
        checkpoint: Lightning checkpoint to export. ``None`` exports a freshly
            initialized network, which is useful for wiring up the frontend before a
            trained model exists -- the manifest records ``trained: false`` so the page
            can say so.
        output_dir: Destination directory; created if absent.
        physics: Background to rebuild the model against; defaults to unit-normalized
            conventions.
        feature_scale: Branch-input normalization from training. Overridden by the value
            stored in the checkpoint when present.
        bake_spectral: Emit circular-convolution kernels instead of Fourier weights. See
            the module docstring for when that is the right trade.
        model: Export this already-constructed network instead of building one. Used by
            the parity fixture, where the exported weights must be the very same tensors
            the golden values were computed from.
        dtype: ``"float32"`` or ``"float16"``. Half precision halves the download at a
            cost the page cannot show: the weights carry ~3 decimal digits, while the
            prediction is drawn against an exact curve whose difference from it is a part
            in $10^{2}$. Keep float32 for the parity fixture, where the point is to detect
            a mis-ported layer rather than to render one.
        phi_max: Half-width of the physical field grid the branch input is sampled on.
            Recorded in the manifest because a consumer cannot recover it from the
            weights -- the branch's own coordinate buffer is normalized to [-1, 1], so
            samples taken on the wrong grid would be silently wrong rather than an error.

    Returns:
        Path to the written manifest.

    Raises:
        FileNotFoundError: If ``checkpoint`` is given but does not exist.
    """
    config = physics or PhysicsConfig(c_delta=None, propagator_normalization="cft")
    trained = False
    n_phi = 64

    if model is not None:
        if checkpoint is not None:
            raise ValueError("pass either a checkpoint or a model, not both")
        n_phi = model.branch.n_phi
    elif checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("model."): v for k, v in payload["state_dict"].items()}
        hyper = payload.get("hyper_parameters", {})
        model = rebuild_model(config, state, hyper)
        n_phi = model.branch.n_phi
        feature_scale = float(hyper.get("feature_scale", feature_scale))
        trained = True
    else:
        model = FourierDeepONet(config, n_phi=n_phi)
    model.eval()

    grid_sizes = spectral_grid_sizes(model)
    baked: dict[str, Tensor] = {}
    if bake_spectral:
        for name, module in model.named_modules():
            if isinstance(module, SpectralConv1d):
                kernel = bake_spectral_kernel(module, grid_sizes[name])
                baked[f"{name}.kernel"] = kernel
                LOGGER.info("baked %s at N=%d -> %s", name, grid_sizes[name], tuple(kernel.shape))

    if dtype not in ("float32", "float16"):
        raise ValueError(f"dtype must be 'float32' or 'float16', got {dtype!r}")
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    tensors: list[dict[str, Any]] = []
    blob = bytearray()
    replaced = {f"{n}.weight" for n in grid_sizes} if bake_spectral else set()
    for name, tensor in list(model.state_dict().items()):
        if name in replaced:
            continue  # superseded by its baked circular kernel
        _append(blob, tensors, name, tensor, torch_dtype)
    for name, tensor in baked.items():
        _append(blob, tensors, name, tensor, torch_dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights.bin").write_bytes(bytes(blob))
    manifest = {
        "format": "qft-operator-weights",
        "version": FORMAT_VERSION,
        "trained": trained,
        "spectral_form": "circular" if bake_spectral else "fourier",
        "dtype": dtype,
        "phi_max": phi_max,
        "feature_scale": feature_scale,
        "physics": {
            "L": config.L,
            "m_sq": config.m_sq,
            "delta": config.delta,
            "c_delta": config.c_delta_effective,
            "log_coefficient": config.log_coefficient,
            "free_dimension": config.free_dimension,
            "sigma_sq": config.sigma_sq,
        },
        "architecture": _architecture(model),
        "spectral_grid_sizes": grid_sizes,
        "tensors": tensors,
        "total_bytes": len(blob),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info(
        "wrote %s (%.1f KiB) and %s", output_dir / "weights.bin", len(blob) / 1024, manifest_path
    )
    return manifest_path


def _append(
    blob: bytearray,
    table: list[dict[str, Any]],
    name: str,
    tensor: Tensor,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Append one tensor to the blob and record its offset."""
    data = tensor.detach().to(dtype).contiguous().cpu().numpy().tobytes()
    table.append(
        {
            "name": name,
            "shape": list(tensor.shape),
            "offset": len(blob),
            "bytes": len(data),
        }
    )
    blob.extend(data)


def main() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Lightning .ckpt to export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frontend/public/operator"),
        help="destination directory for weights.bin and manifest.json",
    )
    parser.add_argument(
        "--reference-normalization",
        action="store_true",
        help="use the published c_delta = 0.159 instead of unit-normalized",
    )
    parser.add_argument(
        "--bake-spectral",
        action="store_true",
        help="emit circular-convolution kernels instead of Fourier weights",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=3.0,
        help="half-width of the field grid the branch input is sampled on",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="half precision halves the download; see the module docstring",
    )
    parser.add_argument(
        "--feature-scale",
        type=float,
        default=None,
        help="override the branch-input normalization; only needed for "
        "checkpoints written before it was recorded in hparams",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    physics = (
        PhysicsConfig()
        if args.reference_normalization
        else PhysicsConfig(c_delta=None, propagator_normalization="cft")
    )
    path = export_operator(
        args.checkpoint,
        args.output,
        physics=physics,
        bake_spectral=args.bake_spectral,
        phi_max=args.phi_max,
        dtype=args.dtype,
    )
    if args.feature_scale is not None:
        manifest = json.loads(path.read_text())
        manifest["feature_scale"] = args.feature_scale
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        LOGGER.info("feature_scale overridden to %g", args.feature_scale)


if __name__ == "__main__":
    main()
