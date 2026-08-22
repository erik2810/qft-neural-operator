"""Physical configuration of the Euclidean AdS2 background and boundary operators.

Conventions
-----------
Poincare coordinates $(z, p)$ with $z > 0$ and boundary at $z \\to 0$:

.. math::
    ds^2 = \\frac{L^2}{z^2}\\left(dz^2 + dp^2\\right), \\qquad
    \\sqrt{g} = \\frac{L^2}{z^2}.

A scalar of mass $m$ is dual to a boundary operator of dimension $\\Delta$ fixed by
$\\Delta(\\Delta - 1) = m^2 L^2$ (the $d = 1$ case of $\\Delta(\\Delta - d) = m^2 L^2$).
The Breitenlohner-Freedman bound in AdS2 reads $m^2 L^2 \\ge -1/4$.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

__all__ = ["PhysicsConfig", "PropagatorNormalization"]

PropagatorNormalization = Literal["cft", "bulk_limit"]
"""Normalization convention for the bulk-to-boundary propagator.

``"cft"``
    Unit-normalized CFT convention, $K_\\Delta = c_\\Delta
    \\left[z / (z^2 + (p - p')^2)\\right]^{\\Delta}$ with
    $c_\\Delta = \\Gamma(\\Delta) / (\\sqrt{\\pi}\\,\\Gamma(\\Delta - 1/2))$, for which
    $z^{\\Delta - 1}\\int dp\\, K_\\Delta(z, p) = 1$ exactly.
``"bulk_limit"``
    The propagator obtained as the boundary limit of the bulk-to-bulk propagator,
    which carries an extra $1/(2\\Delta - d) = 1/(2\\Delta - 1)$ relative to ``"cft"``.
    This is the convention in which the reference anomalous dimension
    $\\gamma = -\\lambda\\,\\frac{2 L^2 c_\\Delta}{2\\Delta - 1}\\,\\beta_1\\beta_2\\,\\xi^2$
    is quoted, and it is therefore the default.
"""


@dataclass(frozen=True)
class PhysicsConfig:
    """AdS2 background parameters and boundary vertex-operator charges.

    Args:
        L: AdS radius $L > 0$.
        m_sq: Bulk mass squared $m^2$; must satisfy the BF bound $m^2 L^2 \\ge -1/4$.
        beta1: Charge $\\beta_1$ of the first boundary vertex operator $V_{\\beta_1}$.
        beta2: Charge $\\beta_2$ of the second boundary vertex operator $V_{\\beta_2}$.
        c_delta: Override for the conformal normalization $c_\\Delta$. ``None`` derives it
            from :attr:`c_delta_cft`. The reference value ``0.159`` reproduces the
            published first-order coefficient and is what the shipped configs use.
        sigma_sq: Regularized coincident-point bulk propagator $\\sigma^2 = G_\\Delta(x,x)$,
            i.e. the width of the Gaussian average used when normal-ordering the
            interaction. ``0.0`` corresponds to a normal-ordered vertex and reproduces
            the published Sine-Gordon result exactly. This quantity is
            scheme-dependent; see :mod:`qft_operator.physics.correlators`.
        propagator_normalization: See :data:`PropagatorNormalization`.

    Raises:
        ValueError: If ``L <= 0``, the BF bound is violated, or ``sigma_sq < 0``.
    """

    L: float = 1.0
    m_sq: float = 0.75
    beta1: float = 1.0
    beta2: float = 1.0
    c_delta: float | None = 0.159
    sigma_sq: float = 0.0
    propagator_normalization: PropagatorNormalization = "bulk_limit"

    #: Spacetime dimension of the boundary theory (AdS2 is holographically dual to d = 1).
    #: A ClassVar, not a field: it is fixed by the background, not a user knob.
    BOUNDARY_DIM: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.L <= 0.0:
            raise ValueError(f"AdS radius L must be positive, got {self.L}")
        if self.m_sq * self.L**2 < -0.25 - 1e-12:
            raise ValueError(
                f"Breitenlohner-Freedman bound violated: m^2 L^2 = "
                f"{self.m_sq * self.L**2:.6f} < -1/4"
            )
        if self.sigma_sq < 0.0:
            raise ValueError(f"sigma_sq must be non-negative, got {self.sigma_sq}")
        if self.c_delta is not None and self.c_delta <= 0.0:
            raise ValueError(f"c_delta override must be positive, got {self.c_delta}")
        if self.propagator_normalization not in ("cft", "bulk_limit"):
            raise ValueError(f"Unknown propagator_normalization {self.propagator_normalization!r}")

    # ------------------------------------------------------------------ #
    # Derived conformal data
    # ------------------------------------------------------------------ #
    @property
    def nu(self) -> float:
        """Bessel index $\\nu = \\sqrt{1/4 + m^2 L^2}$."""
        return math.sqrt(0.25 + self.m_sq * self.L**2)

    @property
    def delta(self) -> float:
        """Scaling dimension $\\Delta = 1/2 + \\nu$ from $\\Delta(\\Delta - 1) = m^2 L^2$."""
        return 0.5 + self.nu

    @property
    def delta_shadow(self) -> float:
        """Shadow dimension $\\tilde{\\Delta} = 1 - \\Delta$ (alternative quantization)."""
        return 1.0 - self.delta

    @property
    def c_delta_cft(self) -> float:
        """Unit-normalized bulk-to-boundary coefficient for $d = 1$.

        $c_\\Delta = \\Gamma(\\Delta) / (\\pi^{d/2}\\,\\Gamma(\\Delta - d/2))$.
        """
        return math.gamma(self.delta) / (math.sqrt(math.pi) * math.gamma(self.delta - 0.5))

    @property
    def c_delta_effective(self) -> float:
        """The $c_\\Delta$ actually used: the override if given, else :attr:`c_delta_cft`."""
        return self.c_delta_cft if self.c_delta is None else self.c_delta

    @property
    def normalization_factor(self) -> float:
        """Convention factor dividing the contact integral: $1$ or $(2\\Delta - 1)$."""
        return 1.0 if self.propagator_normalization == "cft" else (2.0 * self.delta - 1.0)

    @property
    def log_coefficient(self) -> float:
        """Coefficient $C_{\\log}$ of $\\log|p_{12}|$ in the regularized contact integral.

        The bulk contact integral of two bulk-to-boundary propagators is logarithmically
        divergent at the boundary; with a cutoff $z > \\epsilon$,

        .. math::
            \\int_{z > \\epsilon} d^2x \\sqrt{g}\\, K_\\Delta(x; p_1) K_\\Delta(x; p_2)
            = 2 L^2 c_\\Delta\\, |p_{12}|^{-2\\Delta}
              \\left[\\log\\frac{|p_{12}|}{\\epsilon} + \\kappa_\\Delta\\right] + O(\\epsilon),

        so $C_{\\log} = 2 L^2 c_\\Delta$ in the ``"cft"`` convention and
        $C_{\\log} = 2 L^2 c_\\Delta / (2\\Delta - 1)$ in the ``"bulk_limit"`` convention.
        The coefficient is verified against numerical quadrature in
        :mod:`tests.test_bulk_integrals`.
        """
        return 2.0 * self.L**2 * self.c_delta_effective / self.normalization_factor

    @property
    def convention_ratio(self) -> float:
        """Ratio of the configured $c_\\Delta$ to the unit-normalized CFT value.

        Worth checking explicitly, because the reference value ``c_delta = 0.159`` and the
        unit-normalized AdS2 result are **not** the same number. For $\\Delta = 3/2$,

        .. math::
            c_\\Delta^{\\mathrm{CFT}}
            = \\frac{\\Gamma(3/2)}{\\sqrt{\\pi}\\,\\Gamma(1)} = \\frac{1}{2},
            \\qquad 0.159 \\approx \\frac{1}{2\\pi},

        so the reference normalization sits a factor of $\\pi$ below the unit-normalized
        one -- a different convention for the boundary measure, not an error in either.
        It matters here only because the numerical bulk integrator of
        :mod:`qft_operator.physics.bulk_integrals` is defined with the unit-normalized
        kernel: with the override in place, ``target_mode="hybrid"`` produces anomalous
        dimensions a factor :attr:`convention_ratio` away from ``target_mode="resummed"``.
        Set ``c_delta=None`` to make the analytic and numerical pipelines agree exactly.
        """
        return self.c_delta_effective / self.c_delta_cft

    @property
    def free_dimension(self) -> float:
        """Free-theory boundary exponent $\\Delta\\,\\beta_1\\beta_2$."""
        return self.delta * self.beta1 * self.beta2

    # ------------------------------------------------------------------ #
    # Closed-form reference results
    # ------------------------------------------------------------------ #
    def analytical_anomalous_dim(self, lam: float, xi: float) -> float:
        """First-order Sine-Gordon anomalous dimension.

        For $V(\\phi) = -\\lambda\\,(e^{\\xi\\phi} + e^{-\\xi\\phi} - 2)$,

        .. math::
            \\gamma = -\\lambda\\,\\frac{2 L^2 c_\\Delta}{2\\Delta - 1}\\,
                      \\beta_1\\beta_2\\,\\xi^2\\, e^{\\xi^2\\sigma^2/2},

        which reduces to the published expression for a normal-ordered vertex
        ($\\sigma^2 = 0$). This is the specialization of the general functional
        :func:`qft_operator.physics.correlators.anomalous_dimension`; the agreement is
        asserted in :mod:`tests.test_correlators`.

        Args:
            lam: Interaction strength $\\lambda$.
            xi: Vertex exponent $\\xi$.

        Returns:
            The anomalous dimension shift $\\gamma$.
        """
        smearing = math.exp(0.5 * xi**2 * self.sigma_sq)
        return -lam * self.log_coefficient * self.beta1 * self.beta2 * xi**2 * smearing

    def effective_dimension(self, gamma: float) -> float:
        """Effective dimension $\\Delta_{\\mathrm{eff}} = \\Delta\\beta_1\\beta_2 - \\gamma$."""
        return self.free_dimension - gamma

    def summary(self) -> dict[str, float]:
        """Return the derived conformal data as a flat dict (for logging)."""
        return {
            "L": self.L,
            "m_sq": self.m_sq,
            "delta": self.delta,
            "nu": self.nu,
            "c_delta": self.c_delta_effective,
            "c_delta_cft": self.c_delta_cft,
            "log_coefficient": self.log_coefficient,
            "free_dimension": self.free_dimension,
            "sigma_sq": self.sigma_sq,
        }
