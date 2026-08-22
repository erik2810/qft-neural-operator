"""Physics layer: AdS2 geometry, bulk potentials, exact conformal integrals, RG flow.

This subpackage is pure physics -- it has no dependency on the network, the training
stack or Hydra, so it can be imported (and tested) on its own.
"""

from qft_operator.physics.bulk_integrals import (
    ConformalIntegrator,
    QuadratureSpec,
    analytic_log_coefficient,
    fit_log_divergence,
)
from qft_operator.physics.config import PhysicsConfig, PropagatorNormalization
from qft_operator.physics.correlators import (
    CorrelatorTargets,
    anomalous_dimension,
    anomalous_dimension_from_moment,
    boundary_two_point,
    first_order_log_correlator,
    log_boundary_two_point,
    numerical_log_coefficient,
    resummed_log_correlator,
)
from qft_operator.physics.geometry import AdS2Geometry, c_delta_cft
from qft_operator.physics.potentials import (
    FreeTheory,
    GaussianProcessPotential,
    PhiFour,
    PolynomialPotential,
    Potential,
    RandomFourierPotential,
    SineGordon,
    gaussian_moment,
)
from qft_operator.physics.rg import (
    BetaFunction,
    RGConfig,
    running_coupling,
    scale_anomalous_dimension,
)

__all__ = [
    "AdS2Geometry",
    "BetaFunction",
    "ConformalIntegrator",
    "CorrelatorTargets",
    "FreeTheory",
    "GaussianProcessPotential",
    "PhiFour",
    "PhysicsConfig",
    "PolynomialPotential",
    "Potential",
    "PropagatorNormalization",
    "QuadratureSpec",
    "RGConfig",
    "RandomFourierPotential",
    "SineGordon",
    "analytic_log_coefficient",
    "anomalous_dimension",
    "anomalous_dimension_from_moment",
    "boundary_two_point",
    "c_delta_cft",
    "first_order_log_correlator",
    "fit_log_divergence",
    "gaussian_moment",
    "log_boundary_two_point",
    "numerical_log_coefficient",
    "resummed_log_correlator",
    "running_coupling",
    "scale_anomalous_dimension",
]
