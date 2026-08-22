"""Loss functions: supervised data term plus AdS2 scaling and RG invariance priors."""

from qft_operator.losses.composite import LossWeights, PhysicsInformedLoss
from qft_operator.losses.data import LogCorrelatorLoss, relative_l2
from qft_operator.losses.operators import (
    DirectionalDerivative,
    log_curvature,
    log_slope,
    rebuild_coords,
)
from qft_operator.losses.rg import RGInvarianceLoss
from qft_operator.losses.scaling import BoundaryScalingLoss

__all__ = [
    "BoundaryScalingLoss",
    "DirectionalDerivative",
    "LogCorrelatorLoss",
    "LossWeights",
    "PhysicsInformedLoss",
    "RGInvarianceLoss",
    "log_curvature",
    "log_slope",
    "rebuild_coords",
    "relative_l2",
]
