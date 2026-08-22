"""Lightning data module tying the three splits together."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from qft_operator.data.config import DataConfig
from qft_operator.data.dataset import AdS2CorrelatorDataset
from qft_operator.physics.bulk_integrals import ConformalIntegrator, QuadratureSpec
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.rg import RGConfig

try:  # pragma: no cover - exercised implicitly by the training tests
    from lightning import LightningDataModule
except ImportError:  # pragma: no cover
    LightningDataModule = object  # type: ignore[assignment,misc]

__all__ = ["AdS2DataModule"]


class AdS2DataModule(LightningDataModule):  # type: ignore[misc]
    """Generate and serve the train/val/test splits.

    The three splits are drawn from decorrelated seeds, and validation and test reuse the
    *training* split's normalization scalar -- recomputing it per split would leak split
    statistics into the inputs and quietly flatter the reported error.

    A single quadrature engine is shared across splits so that the ``"quadrature"`` and
    ``"hybrid"`` target modes build their tables once.

    Args:
        physics: AdS2 background configuration.
        data: Sampling configuration.
        rg: RG flow configuration.

    Example:
        >>> from qft_operator.data import AdS2DataModule, DataConfig
        >>> dm = AdS2DataModule(data=DataConfig(n_train=8, n_val=4, n_test=4, batch_size=4))
        >>> dm.setup()
        >>> batch = next(iter(dm.train_dataloader()))
        >>> sorted(batch)[:3]
        ['coords', 'coupling', 'delta_eff']
    """

    def __init__(
        self,
        physics: PhysicsConfig | None = None,
        data: DataConfig | None = None,
        rg: RGConfig | None = None,
    ) -> None:
        super().__init__()
        self.physics = physics or PhysicsConfig()
        self.data_config = data or DataConfig()
        self.rg_config = rg or RGConfig()
        self.train_set: AdS2CorrelatorDataset | None = None
        self.val_set: AdS2CorrelatorDataset | None = None
        self.test_set: AdS2CorrelatorDataset | None = None
        self._integrator: ConformalIntegrator | None = None

    # ------------------------------------------------------------------ #
    def _make_integrator(self) -> ConformalIntegrator | None:
        """Build (once) the quadrature engine, if the target mode needs one."""
        if self.data_config.target_mode == "resummed":
            return None
        if self._integrator is None:
            self._integrator = ConformalIntegrator(self.physics, QuadratureSpec())
        return self._integrator

    def setup(self, stage: str | None = None) -> None:
        """Materialize the splits. Idempotent, so repeated Lightning calls are cheap."""
        if self.train_set is not None:
            return
        seed = self.data_config.seed
        integrator = self._make_integrator()
        common: dict[str, Any] = {
            "physics": self.physics,
            "data": self.data_config,
            "rg": self.rg_config,
            "integrator": integrator,
        }
        self.train_set = AdS2CorrelatorDataset(self.data_config.n_train, seed=seed, **common)
        scale = self.train_set.feature_scale
        self.val_set = AdS2CorrelatorDataset(
            self.data_config.n_val, seed=seed + 10_007, feature_scale=scale, **common
        )
        self.test_set = AdS2CorrelatorDataset(
            self.data_config.n_test, seed=seed + 20_011, feature_scale=scale, **common
        )

    def _loader(self, dataset: AdS2CorrelatorDataset | None, shuffle: bool) -> DataLoader:
        """Wrap a split in a DataLoader with the configured batching."""
        if dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            shuffle=shuffle,
            num_workers=self.data_config.num_workers,
            drop_last=False,
            persistent_workers=self.data_config.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training loader."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Deterministic validation loader."""
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Deterministic test loader."""
        return self._loader(self.test_set, shuffle=False)
