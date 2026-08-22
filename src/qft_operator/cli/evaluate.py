"""Hydra entry point: evaluate a trained checkpoint and extract the spectrum.

Example:
    ::

        qft-operator-eval checkpoint=outputs/qft-operator/2026-08-22_12-00-00/checkpoints/last.ckpt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qft_operator.analysis.spectrum import (
    SpectrumReport,
    anomalous_dimension_from_correlator,
    summarize_spectrum,
)
from qft_operator.cli.builders import build_data, build_datamodule, build_module, build_physics
from qft_operator.data.config import DataConfig

__all__ = ["evaluate", "main"]

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


def evaluate(cfg: Any, checkpoint: str | Path | None = None) -> SpectrumReport:
    """Load a checkpoint (if given) and report the anomalous-dimension spectrum.

    Args:
        cfg: A resolved Hydra config.
        checkpoint: Path to a Lightning checkpoint. ``None`` evaluates the freshly
            initialized network, which is a useful control: it shows what the
            free-theory prior alone already achieves.

    Returns:
        A :class:`~qft_operator.analysis.spectrum.SpectrumReport` over the test split.

    Raises:
        FileNotFoundError: If ``checkpoint`` is given but does not exist.
    """
    import torch

    physics = build_physics(cfg.physics)
    data = build_data(cfg.data)
    module = build_module(cfg, physics, data)

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
        LOGGER.info("loaded checkpoint %s", path)

    datamodule = build_datamodule(cfg)
    datamodule.setup()
    test_set = datamodule.test_set
    assert test_set is not None

    module.eval()
    with torch.no_grad():
        prediction = module.model(test_set.v_phi, test_set.coords, test_set.log_m)
    log_r = torch.log(test_set.separations)
    gamma = anomalous_dimension_from_correlator(log_r, prediction, physics.free_dimension)
    report = summarize_spectrum(gamma, test_set.gamma, test_set.family, DataConfig.KNOWN_FAMILIES)
    LOGGER.info("spectrum report: %s", report)
    return report


def main() -> None:
    """Console-script entry point."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
    def _entry(cfg: DictConfig) -> None:
        report = evaluate(cfg, cfg.get("checkpoint"))
        print(f"gamma MAE      : {report.mae:.6e}")
        print(f"gamma RMSE     : {report.rmse:.6e}")
        print(f"gamma R^2      : {report.r2:.6f}")
        print(f"relative MAE   : {report.relative_mae:.4f}")
        for name, value in sorted(report.per_family.items()):
            print(f"  {name:<14}: {value:.6e}")

    _entry()


if __name__ == "__main__":
    main()
