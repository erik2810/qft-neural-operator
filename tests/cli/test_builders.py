"""Config-to-object construction and end-to-end Hydra composition."""

from __future__ import annotations

import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402

from qft_operator.cli.builders import (  # noqa: E402
    build_data,
    build_dataclass,
    build_loss,
    build_model,
    build_physics,
    build_rg,
    to_plain,
)
from qft_operator.cli.train import CONFIG_PATH  # noqa: E402
from qft_operator.data.config import DataConfig  # noqa: E402
from qft_operator.physics.config import PhysicsConfig  # noqa: E402
from qft_operator.training.module import OptimizerConfig  # noqa: E402


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        return compose(config_name="config", overrides=list(overrides))


def test_to_plain_unwraps_omegaconf_containers() -> None:
    node = OmegaConf.create({"a": [1, 2], "b": {"c": 3}})
    assert to_plain(node) == {"a": [1, 2], "b": {"c": 3}}
    assert to_plain(7) == 7


def test_build_dataclass_coerces_sequences_to_tuples() -> None:
    config = OmegaConf.create({"lr": 0.002, "betas": [0.8, 0.95], "max_epochs": 12})
    built = build_dataclass(OptimizerConfig, config)
    assert built.betas == (0.8, 0.95)
    assert built.lr == 0.002


def test_build_dataclass_applies_overrides() -> None:
    built = build_dataclass(OptimizerConfig, {"lr": 1e-3}, lr=5e-4)
    assert built.lr == 5e-4


def test_build_dataclass_rejects_unknown_fields() -> None:
    # A silently-dropped hyperparameter is far more expensive than a loud failure.
    with pytest.raises(ValueError, match="unknown OptimizerConfig fields"):
        build_dataclass(OptimizerConfig, {"learning_rate": 1e-3})


def test_build_dataclass_rejects_non_dataclasses() -> None:
    with pytest.raises(TypeError, match="not a dataclass"):
        build_dataclass(dict, {})  # type: ignore[type-abstract]


def test_default_config_composes_into_working_objects() -> None:
    cfg = _compose()
    physics = build_physics(cfg.physics)
    data = build_data(cfg.data)
    model = build_model(cfg, physics, data.n_phi)
    loss = build_loss(cfg, build_rg(cfg.rg))
    assert physics.delta == pytest.approx(1.5)
    assert data.target_mode == "resummed"
    assert model.num_parameters > 0
    assert loss.weights.data == 1.0


@pytest.mark.parametrize(
    "experiment", ["smoke", "reference_sine_gordon", "hybrid_quadrature", "rg_flow"]
)
def test_every_shipped_experiment_composes(experiment: str) -> None:
    cfg = _compose(f"+experiment={experiment}")
    physics = build_physics(cfg.physics)
    data = build_data(cfg.data)
    build_model(cfg, physics, data.n_phi)
    build_loss(cfg, build_rg(cfg.rg))
    assert cfg.run_name


@pytest.mark.parametrize(
    "model_group", ["fourier_deeponet", "operator_transformer", "baseline_deeponet"]
)
def test_every_shipped_model_group_composes(model_group: str) -> None:
    cfg = _compose(f"model={model_group}")
    physics = build_physics(cfg.physics)
    model = build_model(cfg, physics, build_data(cfg.data).n_phi)
    assert model.head_kind in ("inner_product", "attention")


@pytest.mark.parametrize("physics_group", ["ads2_reference", "ads2_cft"])
def test_every_shipped_physics_group_composes(physics_group: str) -> None:
    physics = build_physics(_compose(f"physics={physics_group}").physics)
    assert physics.delta == pytest.approx(1.5)


@pytest.mark.parametrize("rg_group", ["marginal", "relevant", "uv_cutoff"])
def test_every_shipped_rg_group_composes(rg_group: str) -> None:
    rg = build_rg(_compose(f"rg={rg_group}").rg)
    assert rg.reference_scale > 0.0


@pytest.mark.parametrize("trainer_group", ["default", "fast_dev", "multi_gpu"])
def test_every_shipped_trainer_group_composes(trainer_group: str) -> None:
    trainer = to_plain(_compose(f"trainer={trainer_group}").trainer)
    assert "max_epochs" in trainer and "inference_mode" in trainer


def test_optimizer_horizon_tracks_the_trainer() -> None:
    cfg = _compose("trainer.max_epochs=37")
    assert build_dataclass(OptimizerConfig, cfg.optimizer).max_epochs == 37


def test_quadrature_experiment_pairs_a_uv_cutoff_and_no_rg_term() -> None:
    cfg = _compose("data=quadrature", "rg=uv_cutoff", "loss=data_only", "physics=ads2_cft")
    assert build_data(cfg.data).target_mode == "quadrature"
    assert build_rg(cfg.rg).reference_scale >= 100.0
    assert build_loss(cfg, build_rg(cfg.rg)).weights.rg == 0.0


def test_command_line_overrides_reach_the_dataclasses() -> None:
    cfg = _compose("physics.m_sq=2.0", "data.n_pairs=8", "loss.weights.rg=0.25")
    assert build_physics(cfg.physics).delta == pytest.approx(2.0)
    assert build_data(cfg.data).n_pairs == 8
    assert build_loss(cfg, build_rg(cfg.rg)).weights.rg == 0.25


def test_full_run_executes(tmp_path) -> None:
    from qft_operator.cli.train import run

    cfg = _compose(
        "+experiment=smoke",
        "data.n_train=16",
        "data.n_val=8",
        "data.n_test=8",
        "trainer.max_epochs=1",
    )
    with open_dict(cfg):
        cfg.output_dir = str(tmp_path)
        cfg.trainer.logger = False
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
    metrics = run(cfg)
    assert "test/rel_l2_log_w" in metrics


def test_evaluate_without_a_checkpoint_reports_the_prior(tmp_path) -> None:
    from qft_operator.cli.evaluate import evaluate

    cfg = _compose("+experiment=smoke", "data.n_test=16", "data.n_train=8", "data.n_val=8")
    report = evaluate(cfg, checkpoint=None)
    assert report.mae >= 0.0
    assert set(report.per_family) <= set(DataConfig.KNOWN_FAMILIES)


def test_evaluate_rejects_a_missing_checkpoint(tmp_path) -> None:
    from qft_operator.cli.evaluate import evaluate

    cfg = _compose("+experiment=smoke", "data.n_train=8", "data.n_val=4", "data.n_test=4")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        evaluate(cfg, checkpoint=tmp_path / "absent.ckpt")


def test_generate_writes_a_reloadable_archive(tmp_path) -> None:
    import torch

    from qft_operator.cli.generate_data import generate

    cfg = _compose("+experiment=smoke", "data.n_train=8", "data.n_val=4", "data.n_test=4")
    path = generate(cfg, tmp_path / "dataset.pt")
    payload = torch.load(path, weights_only=False)
    assert set(payload) == {"feature_scale", "train", "val", "test"}
    assert payload["train"]["log_w"].shape[0] == 8
    assert "dv_dcoupling" in payload["train"]


def test_physics_config_from_yaml_matches_the_documented_reference() -> None:
    physics = build_physics(_compose("physics=ads2_reference").physics)
    assert physics == PhysicsConfig(
        L=1.0,
        m_sq=0.75,
        beta1=1.0,
        beta2=1.0,
        c_delta=0.159,
        sigma_sq=0.0,
        propagator_normalization="bulk_limit",
    )
