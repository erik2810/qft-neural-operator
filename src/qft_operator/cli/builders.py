"""Turn a resolved Hydra config into the concrete objects a run needs.

Kept separate from the entry points so that ``train``, ``evaluate`` and
``generate_data`` construct components identically, and so the mapping from YAML to
dataclass is unit-testable without invoking Hydra.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from qft_operator.data.config import DataConfig
from qft_operator.data.datamodule import AdS2DataModule
from qft_operator.losses.composite import LossWeights, PhysicsInformedLoss
from qft_operator.losses.data import LogCorrelatorLoss
from qft_operator.losses.rg import RGInvarianceLoss
from qft_operator.losses.scaling import BoundaryScalingLoss
from qft_operator.models.deeponet import FourierDeepONet
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import BetaFunction, RGConfig
from qft_operator.training.module import OptimizerConfig, QFTOperatorModule

__all__ = [
    "to_plain",
    "build_dataclass",
    "build_physics",
    "build_rg",
    "build_data",
    "build_datamodule",
    "build_model",
    "build_loss",
    "build_module",
]

T = TypeVar("T")


def to_plain(value: Any) -> Any:
    """Recursively convert OmegaConf containers into plain dicts, lists and scalars.

    Hydra hands back ``ListConfig`` for YAML sequences, which frozen dataclasses happily
    store but which then compare unequal to the tuples their annotations promise. Doing
    the conversion once, here, keeps that inconsistency out of the library.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover
        return value
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_dataclass(cls: type[T], config: Any, **overrides: Any) -> T:
    """Instantiate a dataclass from a config mapping, coercing sequences to tuples.

    Unknown keys are rejected rather than ignored: a silently-dropped ``lr`` in a YAML
    file is a much more expensive bug than a loud one.

    Args:
        cls: The target dataclass.
        config: Mapping of field names to values (an OmegaConf node is fine).
        **overrides: Values taking precedence over ``config``.

    Returns:
        An instance of ``cls``.

    Raises:
        TypeError: If ``cls`` is not a dataclass.
        ValueError: If ``config`` contains keys that are not fields of ``cls``.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    payload = dict(to_plain(config) or {})
    payload.update(overrides)

    valid = {f.name for f in fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")

    annotations = {f.name: f.type for f in fields(cls)}
    for key, value in list(payload.items()):
        annotation = str(annotations.get(key, ""))
        if isinstance(value, list) and "tuple" in annotation:
            payload[key] = tuple(value)
    return cls(**payload)  # type: ignore[return-value]


def build_physics(config: Any) -> PhysicsConfig:
    """Build the AdS2 background configuration."""
    return build_dataclass(PhysicsConfig, config)


def build_rg(config: Any) -> RGConfig:
    """Build the RG flow configuration."""
    return build_dataclass(RGConfig, config)


def build_data(config: Any) -> DataConfig:
    """Build the data-generation configuration."""
    return build_dataclass(DataConfig, config)


def build_datamodule(cfg: Any) -> AdS2DataModule:
    """Build the Lightning data module from a full run config."""
    return AdS2DataModule(
        physics=build_physics(cfg.physics),
        data=build_data(cfg.data),
        rg=build_rg(cfg.rg),
    )


def build_model(cfg: Any, physics: PhysicsConfig, n_phi: int) -> FourierDeepONet:
    """Build the operator network.

    ``n_phi`` and the physics background are threaded in from their own config groups
    rather than duplicated in the model group, so they cannot drift apart.
    """
    kwargs = dict(to_plain(cfg.model) or {})
    return FourierDeepONet(config=physics, n_phi=n_phi, **kwargs)


def build_loss(cfg: Any, rg: RGConfig) -> PhysicsInformedLoss:
    """Assemble the composite objective from the ``loss`` config group."""
    section = to_plain(cfg.loss) or {}
    return PhysicsInformedLoss(
        weights=build_dataclass(LossWeights, section.get("weights", {})),
        data_loss=LogCorrelatorLoss(**section.get("data", {})),
        scaling_loss=BoundaryScalingLoss(**section.get("scaling", {})),
        rg_loss=RGInvarianceLoss(beta=BetaFunction(rg), **section.get("rg", {})),
    )


def build_module(
    cfg: Any,
    physics: PhysicsConfig,
    data: DataConfig,
    feature_scale: float = 1.0,
) -> QFTOperatorModule:
    """Build the full Lightning module: network, loss and optimizer settings.

    Args:
        cfg: The resolved run config.
        physics: AdS2 background.
        data: Data configuration, supplying the field-grid resolution.
        feature_scale: Branch-input normalization from the training split, so the
            checkpoint carries it to inference.
    """
    return QFTOperatorModule(
        model=build_model(cfg, physics, data.n_phi),
        loss=build_loss(cfg, build_rg(cfg.rg)),
        optimizer=build_dataclass(OptimizerConfig, cfg.optimizer),
        free_dimension=physics.free_dimension,
        family_names=DataConfig.KNOWN_FAMILIES,
        feature_scale=feature_scale,
    )
