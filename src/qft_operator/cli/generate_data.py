"""Hydra entry point: materialize a dataset to disk without training.

Useful when the ``quadrature`` target mode's bulk integration should be paid for once
and shared across many training runs or machines.

Example:
    ::

        qft-operator-generate data=hybrid physics=ads2_cft output=data/hybrid.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qft_operator.cli.builders import build_datamodule

__all__ = ["generate", "main"]

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")

_TENSOR_KEYS = (
    "v_phi",
    "dv_dcoupling",
    "coords",
    "log_w",
    "delta_eff",
    "log_m",
    "coupling",
    "gamma",
    "family",
)


def generate(cfg: Any, output: str | Path) -> Path:
    """Generate the three splits and save them as a single tensor archive.

    Args:
        cfg: A resolved Hydra config.
        output: Destination ``.pt`` path; parent directories are created.

    Returns:
        The written path.
    """
    import torch

    datamodule = build_datamodule(cfg)
    datamodule.setup()

    payload: dict[str, Any] = {"feature_scale": datamodule.train_set.feature_scale}  # type: ignore[union-attr]
    for name in ("train", "val", "test"):
        split = getattr(datamodule, f"{name}_set")
        payload[name] = {key: getattr(split, key) for key in _TENSOR_KEYS}
        LOGGER.info("%s split: %d samples, %s", name, len(split), split.statistics)

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    LOGGER.info("wrote %s (%.1f MiB)", path, path.stat().st_size / 2**20)
    return path


def main() -> None:
    """Console-script entry point."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
    def _entry(cfg: DictConfig) -> None:
        generate(cfg, cfg.get("output", "data/dataset.pt"))

    _entry()


if __name__ == "__main__":
    main()
