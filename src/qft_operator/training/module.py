"""Lightning module wrapping the operator, the physics losses and the metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from qft_operator.analysis.spectrum import anomalous_dimension_from_correlator
from qft_operator.losses.composite import PhysicsInformedLoss
from qft_operator.losses.data import relative_l2

try:  # pragma: no cover - the physics layer must import without lightning
    from lightning import LightningModule
except ImportError:  # pragma: no cover
    LightningModule = object  # type: ignore[assignment,misc]

__all__ = ["OptimizerConfig", "QFTOperatorModule"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer and schedule settings.

    Args:
        lr: Peak learning rate.
        weight_decay: AdamW decoupled weight decay. Applied only to weight matrices --
            biases, norm parameters and the spectral weights are excluded, since decaying
            Fourier coefficients toward zero is a prior on the *operator*, not a
            regularizer.
        betas: AdamW moment coefficients.
        max_epochs: Horizon of the cosine schedule.
        warmup_epochs: Linear learning-rate warm-up.
        min_lr_ratio: Floor of the cosine schedule as a fraction of ``lr``.
        gradient_clip: Norm at which gradients are clipped by the Trainer.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    max_epochs: int = 200
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must lie in [0, 1)")
        if self.warmup_epochs < 0 or self.max_epochs < 1:
            raise ValueError("invalid epoch counts")


class QFTOperatorModule(LightningModule):  # type: ignore[misc]
    """Train and evaluate the action-to-observable operator.

    Beyond the loss, the module reports the quantity the framework actually exists to
    produce: the anomalous dimension recovered from the predicted correlator by a
    log-log fit, compared against the exact $\\gamma$. Relative error in $\\gamma$ is the
    honest metric -- a network that simply predicts the free theory everywhere already
    scores well on $\\log W$, because $\\gamma/\\Delta \\sim 10^{-3}$.

    Args:
        model: The operator network, returning $\\log W$.
        loss: Composite physics-informed objective.
        optimizer: Optimizer and schedule settings.
        free_dimension: $\\Delta\\beta_1\\beta_2$, used for the spectrum read-out.
        family_names: Names for the per-family metric breakdown.
        feature_scale: The scalar the dataset divided the branch input by. Recorded in
            the checkpoint because inference outside the training pipeline -- the server
            and the browser export -- must apply the identical scaling. Without it a
            served model silently sees inputs an order of magnitude off.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: PhysicsInformedLoss | None = None,
        optimizer: OptimizerConfig | None = None,
        free_dimension: float = 1.5,
        family_names: tuple[str, ...] = (),
        feature_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss or PhysicsInformedLoss()
        self.optimizer_config = optimizer or OptimizerConfig()
        self.free_dimension = free_dimension
        self.family_names = family_names
        self.feature_scale = feature_scale
        # Frozen dataclasses cannot round-trip through Lightning's hparam logging, so
        # persist plain containers instead of the config objects themselves.
        self.save_hyperparameters(
            {
                "optimizer": asdict(self.optimizer_config),
                "loss_weights": asdict(self.loss.weights),
                "free_dimension": free_dimension,
                "family_names": list(family_names),
                "feature_scale": feature_scale,
                "model": type(model).__name__,
                "architecture": getattr(model, "hyperparameters", None),
                "num_parameters": sum(p.numel() for p in model.parameters()),
            }
        )

    # ------------------------------------------------------------------ #
    def forward(self, v_phi: Tensor, coords: Tensor, log_m: Tensor | None = None) -> Tensor:
        """Delegate to the wrapped operator network."""
        return self.model(v_phi, coords, log_m)

    @staticmethod
    def _physics_terms_available() -> bool:
        """Whether autograd-based loss terms can run in the current context.

        Lightning evaluates validation under ``torch.inference_mode`` by default, and
        inference-mode tensors cannot participate in autograd. The shipped trainer config
        sets ``inference_mode: false``; this guard keeps the module usable either way
        instead of crashing when it is left on.
        """
        return not torch.is_inference_mode_enabled()

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        """Forward pass, loss evaluation and logging shared by all three stages."""
        prediction = self.model(batch["v_phi"], batch["coords"], batch["log_m"])
        physics = self._physics_terms_available()
        if physics or stage == "train":
            total, components = self.loss(self.model, batch, prediction, epoch=self.current_epoch)
        else:
            data_only = self.loss.data_loss(prediction, batch["log_w"])
            total = self.loss.weights.data * data_only
            components = {"data": data_only.detach(), "total": total.detach()}

        batch_size = prediction.shape[0]
        for name, value in components.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "total",
                batch_size=batch_size,
                sync_dist=stage != "train",
            )
        self._log_metrics(batch, prediction, stage, batch_size)
        return total

    @torch.no_grad()
    def _log_metrics(
        self, batch: dict[str, Tensor], prediction: Tensor, stage: str, batch_size: int
    ) -> None:
        """Report relative errors and the recovered anomalous-dimension spectrum."""
        target = batch["log_w"]
        log_r = torch.log((batch["coords"][..., 0] - batch["coords"][..., 1]).abs())
        metrics = {
            f"{stage}/rel_l2_log_w": relative_l2(prediction, target),
            f"{stage}/rel_l2_w": relative_l2(torch.exp(prediction), torch.exp(target)),
        }
        gamma_pred = anomalous_dimension_from_correlator(log_r, prediction, self.free_dimension)
        gamma_true = batch["gamma"]
        error = (gamma_pred - gamma_true).abs()
        metrics[f"{stage}/gamma_mae"] = error.mean()
        spread = gamma_true.std(correction=0)
        if float(spread) > 0.0:
            metrics[f"{stage}/gamma_rel_mae"] = error.mean() / spread

        for index, name in enumerate(self.family_names):
            mask = batch["family"] == index
            if bool(mask.any()):
                metrics[f"{stage}/gamma_mae_{name}"] = error[mask].mean()

        self.log_dict(metrics, batch_size=batch_size, sync_dist=stage != "train")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: D102
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: D102
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: D102
        return self._shared_step(batch, "test")

    # ------------------------------------------------------------------ #
    def _parameter_groups(self) -> list[dict[str, Any]]:
        """Split parameters into decayed and non-decayed groups."""
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            skip = parameter.ndim <= 1 or name.endswith("spectral.weight")
            (no_decay if skip else decay).append(parameter)
        return [
            {"params": decay, "weight_decay": self.optimizer_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # Lightning accepts a broad union here; the dict form is the documented one, and
    # narrowing the annotation to match the supertype's union adds nothing.
    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """AdamW with linear warm-up into a cosine decay."""
        cfg = self.optimizer_config
        optimizer = torch.optim.AdamW(self._parameter_groups(), lr=cfg.lr, betas=cfg.betas)

        def schedule(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / max(cfg.warmup_epochs, 1)
            span = max(cfg.max_epochs - cfg.warmup_epochs, 1)
            progress = min((epoch - cfg.warmup_epochs) / span, 1.0)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
            return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
