"""Hydra entry point: train the action-to-observable operator.

Examples:
    Smoke test on CPU::

        qft-operator-train +experiment=smoke

    Flagship pipeline with quadrature-derived labels::

        qft-operator-train +experiment=hybrid_quadrature

    Sweep the bulk mass (and hence the boundary dimension)::

        qft-operator-train -m physics.m_sq=0.25,0.75,2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qft_operator.cli.builders import build_data, build_module, build_physics
from qft_operator.data.config import DataConfig

__all__ = ["run", "main"]

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


def run(cfg: Any) -> dict[str, float]:
    """Execute one training run and return the test metrics.

    Args:
        cfg: A resolved Hydra config (a ``DictConfig``).

    Returns:
        The test-split metric dictionary, with plain float values.
    """
    import lightning as pl
    from omegaconf import OmegaConf

    from qft_operator.cli.builders import build_datamodule, to_plain
    from qft_operator.training.callbacks import FreeTheoryProbe, SpectrumCallback

    pl.seed_everything(cfg.seed, workers=True)
    LOGGER.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    physics = build_physics(cfg.physics)
    data = build_data(cfg.data)
    LOGGER.info("AdS2 background: %s", physics.summary())

    datamodule = build_datamodule(cfg)
    datamodule.setup()
    assert datamodule.train_set is not None
    LOGGER.info("training split: %s", datamodule.train_set.statistics)

    module = build_module(cfg, physics, data, datamodule.train_set.feature_scale)
    LOGGER.info("operator parameters: %d", module.model.num_parameters)
    if cfg.get("compile_model", False):
        import torch

        module.model = torch.compile(module.model)  # type: ignore[assignment]

    output_dir = Path(cfg.get("output_dir") or ".").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        SpectrumCallback(physics.free_dimension, DataConfig.KNOWN_FAMILIES),
        FreeTheoryProbe(
            physics.free_dimension, n_phi=data.n_phi, r_min=data.r_min, r_max=data.r_max
        ),
    ]
    if cfg.get("save_checkpoint", True):
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                monitor="spectrum/relative_mae",
                mode="min",
                save_last=True,
                filename="{epoch:03d}-{spectrum/relative_mae:.4f}",
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir), callbacks=callbacks, **(to_plain(cfg.trainer) or {})
    )
    trainer.fit(module, datamodule)
    results = trainer.test(module, datamodule, verbose=False)
    metrics = {k: float(v) for k, v in (results[0] if results else {}).items()}

    if cfg.get("figures", False):
        _write_figures(module, datamodule, physics, output_dir / "figures")
    LOGGER.info("test metrics: %s", metrics)
    return metrics


def _write_figures(module: Any, datamodule: Any, physics: Any, directory: Path) -> None:
    """Render the standard diagnostic figures for a finished run."""
    import matplotlib

    matplotlib.use("Agg")
    import torch

    from qft_operator.analysis.spectrum import anomalous_dimension_from_correlator
    from qft_operator.viz import (
        plot_anomalous_spectrum,
        plot_correlator_comparison,
        plot_log_residuals,
        set_style,
    )

    set_style()
    directory.mkdir(parents=True, exist_ok=True)
    test_set = datamodule.test_set
    module.eval()
    with torch.no_grad():
        prediction = module.model(test_set.v_phi, test_set.coords, test_set.log_m)
    log_r = torch.log(test_set.separations)

    # Pick the three largest-|gamma| theories: the anomalous dimension is what the
    # figures are meant to show, and near-free samples show nothing.
    order = test_set.gamma.abs().argsort(descending=True)[:3]
    labels = [DataConfig.KNOWN_FAMILIES[int(test_set.family[i])] for i in order]

    plot_correlator_comparison(
        log_r[order], test_set.log_w[order], prediction[order], labels
    ).savefig(directory / "correlators.png")
    plot_log_residuals(
        log_r[order], test_set.log_w[order], prediction[order], physics.free_dimension
    ).savefig(directory / "log_residuals.png")
    plot_anomalous_spectrum(
        test_set.gamma,
        anomalous_dimension_from_correlator(log_r, prediction, physics.free_dimension),
        test_set.family,
        DataConfig.KNOWN_FAMILIES,
    ).savefig(directory / "spectrum.png")
    LOGGER.info("figures written to %s", directory)


def main() -> None:
    """Console-script entry point."""
    import hydra
    from omegaconf import DictConfig, OmegaConf, open_dict

    @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
    def _entry(cfg: DictConfig) -> float:
        from hydra.core.hydra_config import HydraConfig

        with open_dict(cfg):
            cfg.output_dir = HydraConfig.get().runtime.output_dir
        OmegaConf.resolve(cfg)
        metrics = run(cfg)
        # Returned so Hydra sweepers (Optuna, Nevergrad) have an objective to minimize.
        return metrics.get("spectrum/relative_mae", metrics.get("test/rel_l2_log_w", 0.0))

    _entry()


if __name__ == "__main__":
    main()
