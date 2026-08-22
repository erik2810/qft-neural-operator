"""Lightning module, callbacks, and an end-to-end fit."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("lightning")

import lightning as pl  # noqa: E402

from qft_operator.data.config import DataConfig  # noqa: E402
from qft_operator.data.datamodule import AdS2DataModule  # noqa: E402
from qft_operator.losses.composite import LossWeights, PhysicsInformedLoss  # noqa: E402
from qft_operator.models.deeponet import FourierDeepONet  # noqa: E402
from qft_operator.physics.config import PhysicsConfig  # noqa: E402
from qft_operator.physics.rg import RGConfig  # noqa: E402
from qft_operator.training.callbacks import FreeTheoryProbe, SpectrumCallback  # noqa: E402
from qft_operator.training.module import OptimizerConfig, QFTOperatorModule  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture
def small_config() -> DataConfig:
    return DataConfig(n_train=32, n_val=16, n_test=16, n_phi=32, n_pairs=12, batch_size=8)


def _module(physics: PhysicsConfig, data: DataConfig, **loss_kwargs: float) -> QFTOperatorModule:
    torch.manual_seed(0)
    weights = LossWeights(
        **{"data": 1.0, "scaling": 0.01, "rg": 0.01, "warmup_epochs": 0, **loss_kwargs}
    )
    model = FourierDeepONet(
        physics,
        n_phi=data.n_phi,
        latent_dim=32,
        branch_width=16,
        branch_blocks=2,
        trunk_width=32,
        trunk_layers=2,
    )
    return QFTOperatorModule(
        model,
        PhysicsInformedLoss(weights),
        OptimizerConfig(max_epochs=2, warmup_epochs=1),
        physics.free_dimension,
        DataConfig.KNOWN_FAMILIES,
    )


def test_optimizer_config_validation() -> None:
    with pytest.raises(ValueError, match="lr must be positive"):
        OptimizerConfig(lr=0.0)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        OptimizerConfig(min_lr_ratio=1.0)
    with pytest.raises(ValueError, match="epoch counts"):
        OptimizerConfig(max_epochs=0)


def test_parameter_groups_exclude_spectral_weights_from_decay(
    physics: PhysicsConfig, small_config: DataConfig
) -> None:
    module = _module(physics, small_config)
    decay, no_decay = module._parameter_groups()
    assert decay["weight_decay"] > 0.0 and no_decay["weight_decay"] == 0.0
    spectral = [p for n, p in module.model.named_parameters() if n.endswith("spectral.weight")]
    assert spectral and all(any(p is q for q in no_decay["params"]) for p in spectral)


def test_scheduler_warms_up_then_decays(physics: PhysicsConfig, small_config: DataConfig) -> None:
    module = _module(physics, small_config)
    module.optimizer_config = OptimizerConfig(lr=1e-3, max_epochs=10, warmup_epochs=2)
    config = module.configure_optimizers()
    scheduler = config["lr_scheduler"]["scheduler"]
    factors = [scheduler.lr_lambdas[0](epoch) for epoch in range(11)]
    assert factors[0] == pytest.approx(0.5)
    assert factors[1] == pytest.approx(1.0)
    assert factors[2] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(module.optimizer_config.min_lr_ratio, abs=1e-9)
    assert all(a >= b - 1e-12 for a, b in zip(factors[2:], factors[3:], strict=False))


def test_forward_delegates_to_the_model(physics: PhysicsConfig, small_config: DataConfig) -> None:
    module = _module(physics, small_config)
    v = torch.zeros(2, small_config.n_phi)
    r = torch.exp(torch.linspace(-1.0, 1.0, 5)).expand(2, 5).contiguous()
    coords = torch.stack([torch.zeros_like(r), r], dim=-1)
    assert module(v, coords, torch.zeros(2, 1)).shape == (2, 5)


def test_fit_runs_end_to_end(physics: PhysicsConfig, small_config: DataConfig) -> None:
    module = _module(physics, small_config)
    before = [p.detach().clone() for p in module.model.parameters()]
    datamodule = AdS2DataModule(physics, small_config, RGConfig())
    spectrum = SpectrumCallback(physics.free_dimension, DataConfig.KNOWN_FAMILIES)
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
        callbacks=[spectrum, FreeTheoryProbe(physics.free_dimension, n_phi=small_config.n_phi)],
    )
    trainer.fit(module, datamodule)
    fit_metrics = dict(trainer.callback_metrics)
    results = trainer.test(module, datamodule, verbose=False)[0]

    assert spectrum.last_report is not None
    assert set(spectrum.last_report.per_family) <= set(DataConfig.KNOWN_FAMILIES)
    assert "test/rel_l2_log_w" in results and "test/gamma_mae" in results
    assert all(v == v for v in results.values())  # no NaNs
    assert "free_probe/max_abs" in fit_metrics
    assert "spectrum/relative_mae" in fit_metrics
    # Training actually moved the weights rather than silently no-op'ing.
    after = list(module.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_physics_terms_are_skipped_under_inference_mode(
    physics: PhysicsConfig, small_config: DataConfig
) -> None:
    # Lightning's default validation context excludes tensors from autograd; the module
    # degrades to the data term instead of crashing.
    module = _module(physics, small_config)
    datamodule = AdS2DataModule(physics, small_config, RGConfig())
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=True,
    )
    trainer.validate(module, datamodule, verbose=False)
    assert "val/data" in trainer.callback_metrics


def test_datamodule_shares_the_training_normalization(
    physics: PhysicsConfig, small_config: DataConfig
) -> None:
    datamodule = AdS2DataModule(physics, small_config, RGConfig())
    datamodule.setup()
    assert datamodule.train_set is not None and datamodule.val_set is not None
    assert datamodule.val_set.feature_scale == datamodule.train_set.feature_scale
    assert datamodule.test_set.feature_scale == datamodule.train_set.feature_scale  # type: ignore[union-attr]
    # Splits must be genuinely different draws.
    assert not torch.equal(datamodule.train_set.gamma[:8], datamodule.val_set.gamma[:8])


def test_datamodule_setup_is_idempotent(physics: PhysicsConfig, small_config: DataConfig) -> None:
    datamodule = AdS2DataModule(physics, small_config, RGConfig())
    datamodule.setup()
    first = datamodule.train_set
    datamodule.setup("fit")
    assert datamodule.train_set is first


def test_dataloader_before_setup_is_an_error(
    physics: PhysicsConfig, small_config: DataConfig
) -> None:
    with pytest.raises(RuntimeError, match="call setup"):
        AdS2DataModule(physics, small_config, RGConfig()).train_dataloader()


def test_batches_carry_every_key_the_loss_needs(
    physics: PhysicsConfig, small_config: DataConfig
) -> None:
    datamodule = AdS2DataModule(physics, small_config, RGConfig())
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    required = {
        "v_phi",
        "dv_dcoupling",
        "coords",
        "log_w",
        "delta_eff",
        "log_m",
        "coupling",
        "gamma",
        "family",
    }
    assert required <= set(batch)
    assert batch["v_phi"].shape[0] == small_config.batch_size
