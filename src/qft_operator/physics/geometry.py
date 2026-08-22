"""Euclidean AdS2 geometry in Poincare upper-half-plane coordinates.

All tensors are treated as broadcastable; the module is autograd-safe and
device-agnostic so the geometry can be embedded directly in network layers
(see :class:`qft_operator.models.layers.MetricPositionalEncoding`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from qft_operator.physics.config import PhysicsConfig

__all__ = ["AdS2Geometry", "c_delta_cft"]

_EPS = 1e-30
"""Floor added to squared quantities to keep logs and powers finite at coincident points."""


def c_delta_cft(delta: float) -> float:
    """Unit-normalized bulk-to-boundary coefficient in AdS2 ($d = 1$).

    .. math::
        c_\\Delta = \\frac{\\Gamma(\\Delta)}{\\sqrt{\\pi}\\,\\Gamma(\\Delta - 1/2)}

    Args:
        delta: Boundary scaling dimension $\\Delta > 1/2$.

    Returns:
        The coefficient $c_\\Delta$.

    Raises:
        ValueError: If ``delta <= 0.5``, where the normalization integral diverges.
    """
    if delta <= 0.5:
        raise ValueError(f"c_delta requires delta > 1/2 (got {delta})")
    return math.gamma(delta) / (math.sqrt(math.pi) * math.gamma(delta - 0.5))


class AdS2Geometry:
    """Poincare-patch AdS2 metric utilities.

    The metric is $ds^2 = (L^2/z^2)(dz^2 + dp^2)$ with $z > 0$; the conformal factor
    $\\sqrt{g} = L^2/z^2$ is the object embedded into the network's positional encoding.

    Args:
        config: The :class:`~qft_operator.physics.config.PhysicsConfig` supplying $L$,
            $m^2$ and the propagator normalization convention.
    """

    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Metric
    # ------------------------------------------------------------------ #
    @property
    def L(self) -> float:
        """AdS radius."""
        return self.config.L

    @property
    def delta(self) -> float:
        """Boundary scaling dimension of the bulk scalar."""
        return self.config.delta

    def sqrt_g(self, z: Tensor) -> Tensor:
        """Metric determinant factor $\\sqrt{g} = L^2 / z^2$.

        Args:
            z: Radial coordinate, strictly positive.

        Returns:
            Tensor of the same shape as ``z``.
        """
        return (self.L**2) / (z * z + _EPS)

    def log_sqrt_g(self, z: Tensor) -> Tensor:
        """$\\log\\sqrt{g} = 2\\log L - 2\\log z$, numerically stable for small $z$."""
        return 2.0 * math.log(self.L) - 2.0 * torch.log(z.clamp_min(1e-30))

    def chordal_distance(self, z1: Tensor, p1: Tensor, z2: Tensor, p2: Tensor) -> Tensor:
        """Invariant chordal distance $u = [(z_1 - z_2)^2 + (p_1 - p_2)^2] / (z_1 z_2)$.

        $u$ is invariant under the AdS2 isometry group; it is the natural argument of
        the bulk-to-bulk propagator.
        """
        num = (z1 - z2) ** 2 + (p1 - p2) ** 2
        return num / (z1 * z2 + _EPS)

    def geodesic_distance(self, z1: Tensor, p1: Tensor, z2: Tensor, p2: Tensor) -> Tensor:
        """Geodesic distance $s = L\\,\\mathrm{arccosh}(1 + u/2)$."""
        u = self.chordal_distance(z1, p1, z2, p2)
        return self.L * torch.acosh((1.0 + 0.5 * u).clamp_min(1.0))

    # ------------------------------------------------------------------ #
    # Propagators
    # ------------------------------------------------------------------ #
    def bulk_to_boundary(
        self,
        z: Tensor,
        p: Tensor,
        p_boundary: Tensor,
        delta: float | None = None,
        normalized: bool = True,
    ) -> Tensor:
        """Bulk-to-boundary propagator $K_\\Delta(z, p; p')$.

        .. math::
            K_\\Delta(z, p; p') = c_\\Delta
            \\left[\\frac{z}{z^2 + (p - p')^2}\\right]^{\\Delta}

        With ``normalized=True`` the identity $z^{\\Delta - 1}\\int dp\\, K_\\Delta = 1$
        holds exactly, which is the delta-function limit $K_\\Delta \\to z^{1-\\Delta}
        \\delta(p - p')$ as $z \\to 0$.

        Args:
            z: Bulk radial coordinate.
            p: Bulk boundary-direction coordinate.
            p_boundary: Insertion point $p'$ on the boundary.
            delta: Scaling dimension; defaults to :attr:`delta`.
            normalized: Include the $c_\\Delta$ prefactor.

        Returns:
            Broadcast tensor of propagator values.
        """
        d = self.delta if delta is None else delta
        kernel = z / (z * z + (p - p_boundary) ** 2 + _EPS)
        out = kernel.clamp_min(_EPS) ** d
        if normalized:
            out = out * c_delta_cft(d)
        return out

    def bulk_to_bulk(
        self,
        z1: Tensor,
        p1: Tensor,
        z2: Tensor,
        p2: Tensor,
        delta: float | None = None,
    ) -> Tensor:
        """Bulk-to-bulk propagator $G_\\Delta(u)$ in EAdS2 via its hypergeometric form.

        .. math::
            G_\\Delta(u) = \\mathcal{N}_\\Delta\\, u^{-\\Delta}\\,
            {}_2F_1\\!\\left(\\Delta, \\Delta; 2\\Delta; -\\tfrac{4}{u}\\right),
            \\qquad
            \\mathcal{N}_\\Delta = \\frac{\\Gamma(\\Delta)}
            {2\\sqrt{\\pi}\\,\\Gamma(\\Delta + 1/2)},

        matching the convention used in the sibling ``DiffQFT`` implementation.

        Notes:
            ${}_2F_1$ is evaluated through SciPy. The wrapper is differentiable in $u$
            (via $\\partial_z\\,{}_2F_1(a,b;c;z) = \\frac{ab}{c}\\,{}_2F_1(a+1,b+1;c+1;z)$)
            but forces a host round-trip, so it is intended for validation and dataset
            construction rather than inner training loops.

        Args:
            z1: Radial coordinate of the first point.
            p1: Boundary-direction coordinate of the first point.
            z2: Radial coordinate of the second point.
            p2: Boundary-direction coordinate of the second point.
            delta: Scaling dimension; defaults to :attr:`delta`.

        Returns:
            Tensor of propagator values.
        """
        from qft_operator.physics.hypergeometric import hyp2f1  # local: optional SciPy dep

        d = self.delta if delta is None else delta
        u = self.chordal_distance(z1, p1, z2, p2).clamp_min(1e-12)
        norm = math.gamma(d) / (2.0 * math.sqrt(math.pi) * math.gamma(d + 0.5))
        return norm * u.pow(-d) * hyp2f1(d, d, 2.0 * d, -4.0 / u)

    # ------------------------------------------------------------------ #
    # Isometries
    # ------------------------------------------------------------------ #
    @staticmethod
    def dilatation(z: Tensor, p: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        """Apply the AdS2 dilatation isometry $(z, p) \\mapsto (a z, a p)$.

        This is the isometry whose boundary action is the scale transformation
        $p \\mapsto a p$; the metric, $\\sqrt{g}\\,d^2x$, and the chordal distance are
        all invariant under it. Used by the metric-invariance unit tests.
        """
        return scale * z, scale * p

    @staticmethod
    def translation(z: Tensor, p: Tensor, shift: float) -> tuple[Tensor, Tensor]:
        """Apply the boundary translation isometry $(z, p) \\mapsto (z, p + b)$."""
        return z, p + shift

    def volume_element(self, z: Tensor, dz: Tensor, dp: Tensor) -> Tensor:
        """Invariant measure contribution $\\sqrt{g}\\,dz\\,dp = (L^2/z^2)\\,dz\\,dp$."""
        return self.sqrt_g(z) * dz * dp
