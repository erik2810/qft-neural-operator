"""Exact first-order AdS2 bulk conformal integrations.

The object of interest is the two-point contact ("bubble") Witten integral

.. math::
    I_{\\Delta_1\\Delta_2}(p_1, p_2; \\epsilon) =
    \\int_{z > \\epsilon} d^2x\\, \\sqrt{g}\\,
    K_{\\Delta_1}(x; p_1)\\, K_{\\Delta_2}(x; p_2),
    \\qquad \\sqrt{g} = \\frac{L^2}{z^2}.

Two facts drive the whole framework:

1. **For $\\Delta_1 = \\Delta_2 = \\Delta$ the integral is logarithmically divergent**, and
   the near-boundary cutoff $z > \\epsilon$ exposes the divergence as

   .. math::
       I_{\\Delta\\Delta}(r; \\epsilon) = 2 L^2 c_\\Delta\\, r^{-2\\Delta}
       \\left[\\log\\frac{r}{\\epsilon} + \\kappa_\\Delta\\right] + O(\\epsilon),
       \\qquad r = |p_1 - p_2|.

   The $\\log$ exponentiates into an anomalous dimension; that is exactly the
   $\\Delta_{\\mathrm{eff}} = \\Delta\\beta_1\\beta_2 - \\gamma$ reorganization of holographic
   renormalization. The coefficient $2 L^2 c_\\Delta$ is what
   :attr:`~qft_operator.physics.config.PhysicsConfig.log_coefficient` returns (up to the
   propagator-normalization convention), and :func:`fit_log_divergence` recovers it
   numerically from the quadrature below.

2. **The reduced integral is a function of $r/\\epsilon$ alone.** Under
   $(r, \\epsilon) \\to (a r, a \\epsilon)$ the integral scales homogeneously as
   $a^{-2\\Delta}$, so $\\tilde{I} = r^{2\\Delta} I$ can only depend on the ratio. The
   scheme constant $\\kappa_\\Delta$ returned by :meth:`ConformalIntegrator.kappa` must
   therefore come out independent of $r$ -- the sharpest convergence check available.

For $\\Delta_1 \\ne \\Delta_2$ the dimensionally-regulated integral vanishes (a
Feynman-parameter computation leaves an overall
$B\\!\\left(\\tfrac{\\Delta_1 - \\Delta_2}{2}, \\tfrac{\\Delta_2 - \\Delta_1}{2}\\right) = 0$,
the usual orthogonality of boundary operators of different dimension), while the
hard-cutoff version retains power divergences $\\epsilon^{-|\\Delta_1 - \\Delta_2|}$ that are
pure contact terms removed by local boundary counterterms. Only the equal-dimension
case feeds the datasets.

Under the holographic RG dictionary the cutoff is identified with the renormalization
scale, $\\epsilon = 1/M$, so $\\log(r/\\epsilon) = \\log(M r)$.

Quadrature scheme
-----------------
Both directions are mapped to finite intervals and integrated with Gauss-Legendre:

* radial: $z = \\epsilon e^{s}$, $s \\in [0, \\log(z_{\\max}/\\epsilon)]$, which resolves the
  logarithmic measure $dz/z$ uniformly across decades;
* boundary: $p = \\bar{p} + w(z)\\tan\\theta$, $\\theta \\in (-\\pi/2, \\pi/2)$, with an
  adaptive width $w(z) = \\sqrt{(r/2)^2 + z^2}$ tracking the crossover between the two
  boundary-localized peaks at small $z$ and the single broad lump at large $z$.

Everything is vectorized over the separation ``r`` and runs in float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.geometry import c_delta_cft

__all__ = [
    "ConformalIntegrator",
    "QuadratureSpec",
    "ReducedIntegralTable",
    "analytic_log_coefficient",
    "fit_log_divergence",
]


def analytic_log_coefficient(delta: float, L: float = 1.0) -> float:
    """Closed-form coefficient of $\\log(r/\\epsilon)$ in the contact integral.

    Derived by isolating the two boundary-localized regions $p \\to p_i$ at small $z$,
    where $\\int dp\\,(z^2 + u^2)^{-\\Delta} = z^{1 - 2\\Delta}/c_\\Delta$ and the second
    propagator freezes to $r^{-2\\Delta}$, giving $2 L^2 c_\\Delta\\, r^{-2\\Delta}\\,dz/z$.

    Args:
        delta: Boundary scaling dimension.
        L: AdS radius.

    Returns:
        $2 L^2 c_\\Delta$ in the unit-normalized (``"cft"``) convention.
    """
    return 2.0 * L**2 * c_delta_cft(delta)


@dataclass(frozen=True)
class QuadratureSpec:
    """Gauss-Legendre grid resolution for :class:`ConformalIntegrator`.

    Args:
        n_radial: Nodes along the logarithmic radial direction.
        n_boundary: Nodes along the mapped boundary direction.
        z_max_over_r: Upper radial limit as a multiple of the separation $r$; the
            integrand decays like $z^{-2\\Delta - 2}$ so the tail is negligible.
        dtype: Working precision; float64 is strongly recommended.
    """

    n_radial: int = 256
    n_boundary: int = 512
    z_max_over_r: float = 200.0
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.n_radial < 8 or self.n_boundary < 8:
            raise ValueError("Quadrature needs at least 8 nodes per direction")
        if self.z_max_over_r <= 1.0:
            raise ValueError("z_max_over_r must exceed 1")


class ConformalIntegrator:
    """Numerical evaluator for regulated AdS2 contact Witten integrals.

    Args:
        config: Physics configuration supplying $L$, $\\Delta$ and the normalization
            convention.
        spec: Quadrature resolution.
        device: Torch device for the quadrature nodes.

    Example:
        >>> import torch
        >>> from qft_operator.physics import PhysicsConfig, ConformalIntegrator
        >>> integ = ConformalIntegrator(PhysicsConfig(c_delta=None))
        >>> r = torch.tensor([1.0, 2.0], dtype=torch.float64)
        >>> value = integ.contact_integral(r, eps=1e-3)
        >>> value.shape
        torch.Size([2])
    """

    def __init__(
        self,
        config: PhysicsConfig,
        spec: QuadratureSpec | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.spec = spec or QuadratureSpec()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        nodes_r, weights_r = np.polynomial.legendre.leggauss(self.spec.n_radial)
        nodes_b, weights_b = np.polynomial.legendre.leggauss(self.spec.n_boundary)
        dtype, device = self.spec.dtype, self.device
        self._sx = torch.as_tensor(nodes_r, dtype=dtype, device=device)
        self._sw = torch.as_tensor(weights_r, dtype=dtype, device=device)
        self._tx = torch.as_tensor(nodes_b, dtype=dtype, device=device)
        self._tw = torch.as_tensor(weights_b, dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # Core quadrature
    # ------------------------------------------------------------------ #
    def contact_integral(
        self,
        r: Tensor,
        eps: float,
        delta1: float | None = None,
        delta2: float | None = None,
        normalized: bool = True,
    ) -> Tensor:
        """Evaluate $\\int_{z>\\epsilon} d^2x \\sqrt{g}\\, K_{\\Delta_1} K_{\\Delta_2}$.

        The two boundary insertions are placed symmetrically at $p_{1,2} = \\mp r/2$;
        translation invariance makes that fully general.

        Args:
            r: Boundary separations $|p_1 - p_2| > 0$, any shape.
            eps: Near-boundary cutoff $\\epsilon > 0$ (holographically, $\\epsilon = 1/M$).
            delta1: Dimension of the first propagator; defaults to $\\Delta$.
            delta2: Dimension of the second propagator; defaults to $\\Delta$.
            normalized: Divide by
                :attr:`~qft_operator.physics.config.PhysicsConfig.normalization_factor`
                to match the configured propagator convention.

        Returns:
            Tensor of integral values with the same shape as ``r``.

        Raises:
            ValueError: If ``eps <= 0`` or any separation is non-positive.
        """
        if eps <= 0.0:
            raise ValueError(f"cutoff eps must be positive, got {eps}")
        r64 = r.to(dtype=self.spec.dtype, device=self.device)
        if bool((r64 <= 0).any()):
            raise ValueError("boundary separations must be strictly positive")

        d1 = self.config.delta if delta1 is None else delta1
        d2 = self.config.delta if delta2 is None else delta2
        c1, c2 = c_delta_cft(d1), c_delta_cft(d2)

        flat = r64.reshape(-1, 1, 1)  # (R, 1, 1)

        # --- radial nodes: z = eps * exp(s), s in [0, S] ---------------- #
        s_max = torch.log(self.spec.z_max_over_r * flat / eps).clamp_min(1e-6)  # (R,1,1)
        s = 0.5 * s_max * (self._sx.view(1, -1, 1) + 1.0)  # (R, Nz, 1)
        z = eps * torch.exp(s)
        # dz = z ds, and the Jacobian of s in [-1,1] -> [0, s_max] is s_max/2.
        w_z = (0.5 * s_max) * self._sw.view(1, -1, 1) * z  # (R, Nz, 1)

        # --- boundary nodes -------------------------------------------- #
        # At small z the integrand is two spikes of width ~z sitting at p = -+ r/2, so
        # the map must carry the *local* scale z, not the separation r. Split the line
        # at the midpoint p = 0 and give each half a tan-map centred on its own peak:
        #     p = -+ r/2 + z tan(theta),   dp = z sec^2(theta) d(theta).
        # theta_0 = arctan(r / 2z) is the angle at which each branch reaches p = 0; it
        # tends to pi/2 for z << r (well-separated spikes) and to 0 for z >> r (the two
        # peaks have merged into one lump of width z), so the same map covers both
        # regimes without a case split.
        half = 0.5 * flat  # (R, 1, 1)
        theta_0 = torch.atan(half / z)  # (R, Nz, 1)
        span = 0.5 * math.pi + theta_0  # angular length of each half

        u = 0.5 * (self._tx.view(1, 1, -1) + 1.0)  # (1, 1, Np) in [0, 1]
        base_w = span * 0.5 * self._tw.view(1, 1, -1)  # common Jacobian of u -> theta

        theta_right = -theta_0 + span * u  # [-theta_0, pi/2] : right peak at +r/2
        theta_left = -0.5 * math.pi + span * u  # [-pi/2, theta_0] : left peak at -r/2
        p_right = half + z * torch.tan(theta_right)
        p_left = -half + z * torch.tan(theta_left)
        w_right = base_w * z / torch.cos(theta_right) ** 2
        w_left = base_w * z / torch.cos(theta_left) ** 2

        p = torch.cat([p_left, p_right], dim=-1)  # (R, Nz, 2 Np)
        w_p = torch.cat([w_left, w_right], dim=-1)

        # --- integrand: sqrt(g) K1 K2 ----------------------------------- #
        d1_sq = z**2 + (p + half) ** 2
        d2_sq = z**2 + (p - half) ** 2
        # sqrt(g) K1 K2 = L^2 c1 c2 z^{d1 + d2 - 2} / (d1_sq^{d1} d2_sq^{d2}); evaluated
        # through logs because the tan-map reaches |p| ~ 1e5 z at the outermost nodes.
        log_integrand = (
            (d1 + d2 - 2.0) * torch.log(z) - d1 * torch.log(d1_sq) - d2 * torch.log(d2_sq)
        )
        integrand = torch.exp(log_integrand)

        total = (integrand * w_z * w_p).sum(dim=(1, 2))
        total = total * (self.config.L**2 * c1 * c2)
        if normalized:
            total = total / self.config.normalization_factor
        return total.reshape(r.shape).to(dtype=r.dtype)

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    def reduced_contact_integral(self, r: Tensor, eps: float) -> Tensor:
        """Contact integral with the conformal factor $r^{-2\\Delta}$ stripped off.

        Returns $\\tilde{I}(r, \\epsilon) = r^{2\\Delta} I_{\\Delta\\Delta}(r, \\epsilon)
        = C_{\\log}\\,[\\log(r/\\epsilon) + \\kappa_\\Delta]$, the pure logarithm that feeds
        the first-order correction to $W$.
        """
        integral = self.contact_integral(r, eps=eps)
        return integral * r.to(integral.dtype) ** (2.0 * self.config.delta)

    def log_slope(self, r: Tensor, eps: float, eps_ratio: float = 4.0) -> Tensor:
        """Numerically differentiate w.r.t. $\\log(1/\\epsilon)$ at fixed $r$.

        Two cutoffs separated by ``eps_ratio`` bracket the derivative; the result should
        equal :func:`analytic_log_coefficient` (divided by the normalization factor)
        independently of $r$ and $\\epsilon$.

        Args:
            r: Boundary separations.
            eps: Reference cutoff.
            eps_ratio: Multiplicative spacing between the two cutoffs.

        Returns:
            $\\partial \\tilde{I} / \\partial \\log(1/\\epsilon)$, shape as ``r``.
        """
        if eps_ratio <= 1.0:
            raise ValueError("eps_ratio must exceed 1")
        fine = self.reduced_contact_integral(r, eps=eps / eps_ratio)
        coarse = self.reduced_contact_integral(r, eps=eps)
        return (fine - coarse) / math.log(eps_ratio)

    def kappa(self, r: Tensor, eps: float) -> Tensor:
        """Scheme constant $\\kappa_\\Delta = \\tilde{I}/C_{\\log} - \\log(r/\\epsilon)$.

        A constant output (independent of both $r$ and $\\epsilon$) is the sharpest
        available check that the quadrature has converged.
        """
        reduced = self.reduced_contact_integral(r, eps=eps)
        c_log = self.config.log_coefficient
        return reduced / c_log - torch.log(r.to(reduced.dtype) / eps)

    def integrand_field(
        self,
        r: float,
        eps: float,
        shape: tuple[int, int] = (192, 256),
        decades_below: float = 3.0,
        decades_above: float = 1.0,
        p_half_width: float | None = None,
        delta: float | None = None,
    ) -> dict[str, Tensor | float]:
        """Tabulate the contact-integral density on a display grid over the bulk.

        Returns $\\sqrt{g}\\,K_\\Delta(x;p_1)K_\\Delta(x;p_2)$ sampled on a grid that is
        uniform in $\\log z$ and in $p$, which is the natural chart for looking at the
        near-boundary region: the two propagator peaks that generate the logarithmic
        divergence sit at $p = \\mp r/2$ with width $z$, so they appear as two vertical
        ridges narrowing towards the boundary at the top of the frame.

        The window is placed around the **separation**, not around the cutoff. Anchoring
        it at $z = \\epsilon$ is the obvious choice and the wrong one: the ridges are
        $O(\\epsilon)$ wide there, which for any interesting cutoff is far below one pixel,
        so the picture degenerates into a featureless blob. Centring on $r$ instead puts
        the structure at the scale the grid can resolve, and the cutoff becomes a line
        moving through a fixed density -- which is the more faithful story anyway, since
        $\\epsilon$ decides how much of the integrand is integrated, not what it looks like.

        The grid is for visualization only -- the integral itself is computed by the
        adaptive quadrature of :meth:`contact_integral`, which resolves those ridges
        properly. Both numbers are returned so a viewer can show the density and the
        converged value together.

        Args:
            r: Boundary separation $|p_1 - p_2| > 0$.
            eps: Near-boundary cutoff; only used for the returned ``integral``.
            shape: ``(n_z, n_p)`` grid resolution.
            decades_below: Decades of $z$ displayed below the separation.
            decades_above: Decades of $z$ displayed above it.
            p_half_width: Half-width of the displayed $p$ window; defaults to $2r$.
            delta: Scaling dimension; defaults to the configured $\\Delta$.

        Returns:
            Dict with ``density`` of shape ``(n_z, n_p)`` (row 0 at $z = \\epsilon$),
            the grid extents ``log_z_min``/``log_z_max``/``p_min``/``p_max``, and the
            converged ``integral`` over $z > \\epsilon$.

        Raises:
            ValueError: If ``r`` or ``eps`` is non-positive, or the grid is degenerate.
        """
        if r <= 0.0 or eps <= 0.0:
            raise ValueError(f"need r > 0 and eps > 0, got r={r}, eps={eps}")
        n_z, n_p = shape
        if min(n_z, n_p) < 2:
            raise ValueError(f"grid must be at least 2x2, got {shape}")

        d = self.config.delta if delta is None else delta
        width = 2.0 * r if p_half_width is None else p_half_width
        dtype, device = self.spec.dtype, self.device

        log_z = torch.linspace(
            math.log(r) - decades_below * math.log(10.0),
            math.log(r) + decades_above * math.log(10.0),
            n_z,
            dtype=dtype,
            device=device,
        )
        p = torch.linspace(-width, width, n_p, dtype=dtype, device=device)
        z = torch.exp(log_z).unsqueeze(-1)  # (n_z, 1)

        half = 0.5 * r
        coefficient = self.config.L**2 * c_delta_cft(d) ** 2 / self.config.normalization_factor
        log_density = (
            math.log(coefficient)
            + (2.0 * d - 2.0) * torch.log(z)
            - d * torch.log(z**2 + (p + half) ** 2)
            - d * torch.log(z**2 + (p - half) ** 2)
        )
        return {
            "density": torch.exp(log_density),
            "log_eps": math.log(eps),
            "log_density": log_density,
            "log_z_min": float(log_z[0]),
            "log_z_max": float(log_z[-1]),
            "p_min": float(p[0]),
            "p_max": float(p[-1]),
            "integral": float(
                self.contact_integral(torch.tensor([r], dtype=dtype, device=device), eps=eps)
            ),
        }


def fit_log_divergence(
    integrator: ConformalIntegrator,
    r: float = 1.0,
    eps_values: Tensor | None = None,
) -> tuple[float, float]:
    """Least-squares fit of $\\tilde{I}(\\epsilon) = A\\,\\log(1/\\epsilon) + B$.

    Recovering $A = C_{\\log}$ from pure quadrature is the end-to-end validation that the
    numerical bulk integration reproduces the analytic holographic-renormalization
    coefficient.

    Args:
        integrator: A configured :class:`ConformalIntegrator`.
        r: Boundary separation held fixed during the scan.
        eps_values: Cutoffs to scan; defaults to eight points log-spaced in
            $[10^{-5}, 10^{-2}]\\cdot r$.

    Returns:
        ``(A, B)`` -- the fitted log slope and intercept.
    """
    if eps_values is None:
        eps_values = torch.logspace(-5.0, -2.0, 8, dtype=torch.float64) * r
    r_tensor = torch.tensor([r], dtype=torch.float64)
    reduced = torch.stack(
        [integrator.reduced_contact_integral(r_tensor, eps=float(e)).squeeze(0) for e in eps_values]
    )
    x = -torch.log(eps_values.to(torch.float64))
    design = torch.stack([x, torch.ones_like(x)], dim=-1)
    solution = torch.linalg.lstsq(design, reduced.unsqueeze(-1)).solution.squeeze(-1)
    return float(solution[0]), float(solution[1])


class ReducedIntegralTable:
    """Cached, interpolatable table of the reduced contact integral.

    Because $\\tilde{I}$ depends on $r$ and $\\epsilon$ only through the ratio $x =
    r/\\epsilon$ (see the module docstring), the whole quadrature collapses to one
    univariate function. Tabulating it once on a logarithmic grid turns the per-sample
    cost of the ``"quadrature"`` dataset mode from a full 2-D quadrature into a lookup,
    which is what makes that mode usable for datasets of thousands of theories.

    Args:
        integrator: Configured quadrature engine.
        log_x_min: Lower end of the tabulated $\\log(r/\\epsilon)$ range.
        log_x_max: Upper end of the tabulated range.
        n_nodes: Number of tabulated points.

    Raises:
        ValueError: If the range is empty or fewer than two nodes are requested.
    """

    def __init__(
        self,
        integrator: ConformalIntegrator,
        log_x_min: float = 0.5,
        log_x_max: float = 20.0,
        n_nodes: int = 128,
    ) -> None:
        if log_x_max <= log_x_min:
            raise ValueError("log_x_max must exceed log_x_min")
        if n_nodes < 2:
            raise ValueError("need at least two tabulation nodes")
        self.integrator = integrator
        self.log_x = torch.linspace(log_x_min, log_x_max, n_nodes, dtype=torch.float64)
        # Fix eps = 1 and sweep r: any (r, eps) with the same ratio gives the same value.
        radii = torch.exp(self.log_x)
        self.values = integrator.reduced_contact_integral(radii, eps=1.0)

    def __call__(self, r: Tensor, eps: float) -> Tensor:
        """Interpolate $\\tilde{I}(r, \\epsilon)$.

        Args:
            r: Boundary separations.
            eps: Near-boundary cutoff.

        Returns:
            Reduced integral values, shaped like ``r``.

        Raises:
            ValueError: If any query falls outside the tabulated range.
        """
        query = torch.log(r.to(torch.float64) / eps)
        if bool((query < self.log_x[0]).any() or (query > self.log_x[-1]).any()):
            raise ValueError(
                f"log(r/eps) outside tabulated range "
                f"[{float(self.log_x[0]):.3f}, {float(self.log_x[-1]):.3f}]"
            )
        index = torch.searchsorted(self.log_x, query.reshape(-1).contiguous()).clamp(
            1, self.log_x.numel() - 1
        )
        lo, hi = index - 1, index
        weight = (query.reshape(-1) - self.log_x[lo]) / (self.log_x[hi] - self.log_x[lo])
        out = torch.lerp(self.values[lo], self.values[hi], weight)
        return out.reshape(r.shape).to(r.dtype)
