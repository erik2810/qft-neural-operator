"""Bulk interaction potentials $V(\\phi)$ and their first-order holographic data.

Every potential in this module is written as

.. math::
    V(\\phi) = \\lambda\\, v(\\phi),

i.e. **linear in the coupling** $\\lambda$, with $v$ the fixed shape function. That is not
a cosmetic choice: it makes $\\partial V / \\partial \\lambda = v(\\phi)$ exact and closed
form, which is what lets :class:`~qft_operator.losses.rg.RGInvarianceLoss` evaluate the
Callan-Symanzik derivative $\\beta(\\lambda)\\,\\partial_\\lambda W$ through the branch input
without finite differences in $\\lambda$.

The single physical quantity a potential must expose is the Gaussian-averaged second
derivative

.. math::
    \\langle V'' \\rangle_\\sigma =
    \\int d\\phi\\, \\mathcal{N}(\\phi; 0, \\sigma^2)\\, V''(\\phi),

which is the first-order source of the boundary anomalous dimension (see
:func:`qft_operator.physics.correlators.anomalous_dimension`). At first order in the
interaction only the term with exactly one bulk-to-boundary propagator reaching each
vertex operator contributes to the connected correlator; that term carries
$\\beta_1\\beta_2\\,V''$ with the remaining fields self-contracted into the coincident-point
propagator $\\sigma^2 = G_\\Delta(x, x)$. Setting $\\sigma^2 = 0$ (a normal-ordered vertex)
recovers the published Sine-Gordon result exactly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "Potential",
    "FreeTheory",
    "SineGordon",
    "PhiFour",
    "PolynomialPotential",
    "RandomFourierPotential",
    "GaussianProcessPotential",
    "gaussian_moment",
]


def gaussian_moment(order: int, sigma_sq: float) -> float:
    """Central Gaussian moment $\\langle \\phi^n \\rangle$ under $\\mathcal{N}(0, \\sigma^2)$.

    Equals $\\sigma^n (n-1)!!$ for even $n$ and $0$ for odd $n$.

    Args:
        order: Moment order $n \\ge 0$.
        sigma_sq: Variance $\\sigma^2 \\ge 0$.

    Returns:
        The moment value.
    """
    if order < 0:
        raise ValueError(f"moment order must be non-negative, got {order}")
    if order % 2 == 1:
        return 0.0
    if order == 0:
        return 1.0
    double_factorial = float(np.prod(np.arange(order - 1, 0, -2)))
    return sigma_sq ** (order / 2.0) * double_factorial


class Potential(ABC):
    """Abstract bulk interaction potential $V(\\phi) = \\lambda\\, v(\\phi)$.

    Subclasses implement the three shape-function hooks; the coupling-carrying public
    methods are derived from them so that linearity in $\\lambda$ is structural rather
    than a convention each subclass has to remember.

    Args:
        coupling: The interaction strength $\\lambda$.
    """

    #: Short identifier used in dataset metadata and plot legends.
    family: str = "abstract"

    def __init__(self, coupling: float) -> None:
        self.coupling = float(coupling)

    # -- hooks ---------------------------------------------------------- #
    @abstractmethod
    def shape(self, phi: Tensor) -> Tensor:
        """Shape function $v(\\phi) = V(\\phi)/\\lambda$."""

    @abstractmethod
    def shape_second_derivative(self, phi: Tensor) -> Tensor:
        """Second derivative $v''(\\phi)$."""

    @abstractmethod
    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:
        """Gaussian average $\\langle v'' \\rangle_\\sigma$, in closed form."""

    @abstractmethod
    def describe(self) -> dict[str, float]:
        """Scalar parameters of this instance, for dataset metadata."""

    # -- derived -------------------------------------------------------- #
    def evaluate(self, phi: Tensor) -> Tensor:
        """Potential $V(\\phi) = \\lambda\\, v(\\phi)$ on the given field grid."""
        return self.coupling * self.shape(phi)

    def second_derivative(self, phi: Tensor) -> Tensor:
        """$V''(\\phi) = \\lambda\\, v''(\\phi)$."""
        return self.coupling * self.shape_second_derivative(phi)

    def d_dcoupling(self, phi: Tensor) -> Tensor:
        """Exact $\\partial V(\\phi) / \\partial \\lambda = v(\\phi)$.

        This is the tangent direction along which the RG loss differentiates the branch
        input; being exact (rather than a finite difference) is what keeps the
        Callan-Symanzik residual meaningful at small $\\lambda$.
        """
        return self.shape(phi)

    def gaussian_second_moment(self, sigma_sq: float = 0.0) -> float:
        """$\\langle V'' \\rangle_\\sigma = \\lambda\\,\\langle v'' \\rangle_\\sigma$.

        Args:
            sigma_sq: Coincident-point propagator $\\sigma^2 = G_\\Delta(x,x)$; ``0``
                corresponds to a normal-ordered interaction, for which the average
                collapses to $V''(0)$.

        Returns:
            The Gaussian-averaged second derivative.
        """
        if sigma_sq < 0.0:
            raise ValueError(f"sigma_sq must be non-negative, got {sigma_sq}")
        return self.coupling * self.shape_gaussian_second_moment(sigma_sq)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v:.4g}" for k, v in self.describe().items())
        return f"{type(self).__name__}({params})"


class FreeTheory(Potential):
    """The free limit $V(\\phi) \\equiv 0$.

    The anomalous dimension vanishes identically, so $W(r) = r^{-2\\Delta\\beta_1\\beta_2}$.
    Used as the exactness anchor of the test suite.
    """

    family = "free"

    def __init__(self) -> None:
        super().__init__(coupling=0.0)

    # Multiplying by zero rather than returning a fresh ``zeros_like`` keeps the result
    # attached to the autograd graph, so differentiating any expression through the free
    # theory yields zeros instead of "does not require grad".
    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        return phi * 0.0

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        return phi * 0.0

    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:  # noqa: D102
        return 0.0

    def describe(self) -> dict[str, float]:  # noqa: D102
        return {"coupling": 0.0}


class SineGordon(Potential):
    """Sine-Gordon (Liouville-like) interaction.

    .. math::
        V(\\phi) = -\\lambda\\left(e^{\\xi\\phi} + e^{-\\xi\\phi} - 2\\right)
                 = -2\\lambda\\left(\\cosh(\\xi\\phi) - 1\\right)

    with $V''(\\phi) = -2\\lambda\\xi^2\\cosh(\\xi\\phi)$ and, using
    $\\langle \\cosh(\\xi\\phi)\\rangle_\\sigma = e^{\\xi^2\\sigma^2/2}$,

    .. math::
        \\langle V'' \\rangle_\\sigma = -2\\lambda\\xi^2 e^{\\xi^2\\sigma^2/2}.

    Args:
        coupling: $\\lambda$.
        xi: Vertex exponent $\\xi$.
    """

    family = "sine_gordon"

    def __init__(self, coupling: float, xi: float) -> None:
        super().__init__(coupling)
        self.xi = float(xi)

    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        return -2.0 * (torch.cosh(self.xi * phi) - 1.0)

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        return -2.0 * self.xi**2 * torch.cosh(self.xi * phi)

    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:  # noqa: D102
        return -2.0 * self.xi**2 * math.exp(0.5 * self.xi**2 * sigma_sq)

    def describe(self) -> dict[str, float]:  # noqa: D102
        return {"coupling": self.coupling, "xi": self.xi}


class PhiFour(Potential):
    """Quartic self-interaction $V(\\phi) = \\lambda\\,\\phi^4$.

    $V'' = 12\\lambda\\phi^2$, so $\\langle V''\\rangle_\\sigma = 12\\lambda\\sigma^2$.

    Note:
        A normal-ordered quartic ($\\sigma^2 = 0$) produces **no** first-order shift in the
        $\\beta_1\\beta_2$ channel -- the leading contribution needs the tadpole. The
        baseline script's ``gamma = lam * 0.4`` had no such structure; this class makes
        the tadpole dependence explicit instead.

    Args:
        coupling: $\\lambda$.
    """

    family = "phi4"

    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        return phi**4

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        return 12.0 * phi**2

    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:  # noqa: D102
        return 12.0 * sigma_sq

    def describe(self) -> dict[str, float]:  # noqa: D102
        return {"coupling": self.coupling}


@dataclass(frozen=True)
class _PolynomialCoefficients:
    """Container for the monomial coefficients of a polynomial shape function."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 1:
            raise ValueError("polynomial needs at least one coefficient")


class PolynomialPotential(Potential):
    """General polynomial interaction $V(\\phi) = \\lambda \\sum_k c_k \\phi^k$.

    $\\langle V'' \\rangle_\\sigma = \\lambda \\sum_{k \\ge 2} k(k-1)\\,c_k\\,
    \\langle\\phi^{k-2}\\rangle_\\sigma$ with the Gaussian moments of
    :func:`gaussian_moment`.

    Args:
        coupling: Overall scale $\\lambda$.
        coefficients: $(c_0, c_1, \\dots, c_K)$ in ascending order.
        centered: Subtract $v(0)$ so that $V(0) = 0$, matching the Sine-Gordon
            convention. Constants do not affect $V''$ and hence never affect the physics,
            but they do shift the branch input, so the option keeps families comparable.
    """

    family = "polynomial"

    def __init__(
        self,
        coupling: float,
        coefficients: tuple[float, ...] | list[float],
        centered: bool = True,
    ) -> None:
        super().__init__(coupling)
        self._coeffs = _PolynomialCoefficients(tuple(float(c) for c in coefficients))
        self.centered = bool(centered)

    @property
    def coefficients(self) -> tuple[float, ...]:
        """The monomial coefficients $(c_0, \\dots, c_K)$."""
        return self._coeffs.values

    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        out = torch.zeros_like(phi)
        for power, coeff in enumerate(self.coefficients):
            if coeff != 0.0:
                out = out + coeff * phi**power
        if self.centered:
            out = out - self.coefficients[0]
        return out

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        out = torch.zeros_like(phi)
        for power, coeff in enumerate(self.coefficients):
            if power >= 2 and coeff != 0.0:
                out = out + power * (power - 1) * coeff * phi ** (power - 2)
        return out

    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:  # noqa: D102
        total = 0.0
        for power, coeff in enumerate(self.coefficients):
            if power >= 2 and coeff != 0.0:
                total += power * (power - 1) * coeff * gaussian_moment(power - 2, sigma_sq)
        return total

    def describe(self) -> dict[str, float]:  # noqa: D102
        out: dict[str, float] = {"coupling": self.coupling}
        out.update({f"c{k}": c for k, c in enumerate(self.coefficients)})
        return out


class RandomFourierPotential(Potential):
    """Gaussian-process potential in a random-Fourier-feature representation.

    Bochner's theorem states that a stationary kernel is the Fourier transform of its
    spectral measure; sampling $K$ frequencies from that measure gives an (approximate)
    GP sample with an **analytic** functional form,

    .. math::
        v(\\phi) = \\frac{1}{\\sqrt{K}}\\sum_{k=1}^{K} a_k \\cos(\\omega_k \\phi + b_k),

    with $\\omega_k \\sim \\mathcal{N}(0, \\ell^{-2})$ for an RBF kernel of lengthscale
    $\\ell$, $b_k \\sim \\mathcal{U}[0, 2\\pi)$, $a_k \\sim \\mathcal{N}(0, 1)$ (times an
    output scale). Because the sample is analytic, both $v''$ and its Gaussian average
    are exact --

    .. math::
        \\langle v'' \\rangle_\\sigma =
        -\\frac{1}{\\sqrt{K}}\\sum_k a_k\\,\\omega_k^2\\, e^{-\\omega_k^2\\sigma^2/2}\\cos b_k,

    using $\\langle\\cos(\\omega\\phi + b)\\rangle_\\sigma = e^{-\\omega^2\\sigma^2/2}\\cos b$ --
    so GP-sampled theories carry exactly the same quality of label as the analytic
    families. This is the mechanism by which the operator is trained to generalize
    beyond Sine-Gordon and $\\phi^4$.

    Args:
        coupling: Overall scale $\\lambda$.
        amplitudes: $(a_k)$, shape ``(K,)``.
        frequencies: $(\\omega_k)$, shape ``(K,)``.
        phases: $(b_k)$, shape ``(K,)``.
        centered: Subtract $v(0)$ so that $V(0) = 0$.

    Raises:
        ValueError: If the three coefficient arrays do not share one 1-D shape.
    """

    family = "gp_fourier"

    def __init__(
        self,
        coupling: float,
        amplitudes: Tensor,
        frequencies: Tensor,
        phases: Tensor,
        centered: bool = True,
    ) -> None:
        super().__init__(coupling)
        tensors = (amplitudes, frequencies, phases)
        if any(t.ndim != 1 for t in tensors) or len({t.shape for t in tensors}) != 1:
            raise ValueError("amplitudes, frequencies and phases must share one 1-D shape")
        self.amplitudes = amplitudes.to(torch.float64)
        self.frequencies = frequencies.to(torch.float64)
        self.phases = phases.to(torch.float64)
        self.centered = bool(centered)

    @property
    def num_features(self) -> int:
        """Number of random Fourier features $K$."""
        return int(self.amplitudes.shape[0])

    def _basis(self, phi: Tensor) -> Tensor:
        """Evaluate $\\cos(\\omega_k \\phi + b_k)$, shape ``phi.shape + (K,)``."""
        omega = self.frequencies.to(dtype=phi.dtype, device=phi.device)
        phase = self.phases.to(dtype=phi.dtype, device=phi.device)
        return torch.cos(phi.unsqueeze(-1) * omega + phase)

    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        amp = self.amplitudes.to(dtype=phi.dtype, device=phi.device)
        out = (self._basis(phi) * amp).sum(-1) / math.sqrt(self.num_features)
        if self.centered:
            zero = torch.zeros((), dtype=phi.dtype, device=phi.device)
            out = out - (self._basis(zero) * amp).sum(-1) / math.sqrt(self.num_features)
        return out

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        amp = self.amplitudes.to(dtype=phi.dtype, device=phi.device)
        omega = self.frequencies.to(dtype=phi.dtype, device=phi.device)
        weighted = -amp * omega**2
        return (self._basis(phi) * weighted).sum(-1) / math.sqrt(self.num_features)

    def shape_gaussian_second_moment(self, sigma_sq: float) -> float:  # noqa: D102
        smearing = torch.exp(-0.5 * self.frequencies**2 * sigma_sq)
        terms = -self.amplitudes * self.frequencies**2 * smearing * torch.cos(self.phases)
        return float(terms.sum() / math.sqrt(self.num_features))

    def describe(self) -> dict[str, float]:  # noqa: D102
        return {
            "coupling": self.coupling,
            "num_features": float(self.num_features),
            "rms_frequency": float(self.frequencies.pow(2).mean().sqrt()),
        }


class GaussianProcessPotential(Potential):
    """Exact GP sample tabulated on a field grid, with numerical derivatives.

    Drawn from an RBF-kernel GP by Cholesky factorization rather than a feature
    expansion, so it is the reference against which the random-Fourier construction of
    :class:`RandomFourierPotential` is validated. Derivatives use a fourth-order central
    stencil and the Gaussian average uses Gauss-Hermite quadrature with linear
    interpolation onto the grid, so its labels are less accurate than the analytic
    families -- prefer :class:`RandomFourierPotential` for training data.

    Args:
        coupling: Overall scale $\\lambda$.
        phi_grid: Strictly increasing, uniformly spaced field grid, shape ``(N,)``.
        values: Shape-function samples $v(\\phi_i)$, shape ``(N,)``.
        centered: Subtract the value at (or nearest to) $\\phi = 0$.

    Raises:
        ValueError: If the grid is not uniform or the shapes disagree.
    """

    family = "gp_exact"

    def __init__(
        self,
        coupling: float,
        phi_grid: Tensor,
        values: Tensor,
        centered: bool = True,
    ) -> None:
        super().__init__(coupling)
        if phi_grid.ndim != 1 or phi_grid.shape != values.shape:
            raise ValueError("phi_grid and values must be 1-D tensors of equal length")
        if phi_grid.numel() < 5:
            raise ValueError("need at least 5 grid points for a 4th-order stencil")
        spacing = torch.diff(phi_grid.to(torch.float64))
        if not bool(torch.allclose(spacing, spacing[0], rtol=1e-9, atol=1e-12)):
            raise ValueError("phi_grid must be uniformly spaced")
        self.phi_grid = phi_grid.to(torch.float64)
        self._h = float(spacing[0])
        values64 = values.to(torch.float64)
        if centered:
            values64 = values64 - values64[int(torch.argmin(self.phi_grid.abs()))]
        self.values = values64
        self.centered = bool(centered)
        self._second = self._finite_difference_second()

    def _finite_difference_second(self) -> Tensor:
        """Fourth-order central second derivative, one-sided near the two edges."""
        v, h = self.values, self._h
        out = torch.empty_like(v)
        core = (-v[:-4] + 16.0 * v[1:-3] - 30.0 * v[2:-2] + 16.0 * v[3:-1] - v[4:]) / (12.0 * h**2)
        out[2:-2] = core
        # Edges fall back to the standard 3-point stencil; they carry no weight in the
        # Gaussian average for any sensible grid range.
        edge = (v[:-2] - 2.0 * v[1:-1] + v[2:]) / h**2
        out[1] = edge[0]
        out[-2] = edge[-1]
        out[0] = edge[0]
        out[-1] = edge[-1]
        return out

    def _interp(self, table: Tensor, phi: Tensor) -> Tensor:
        """Linear interpolation of ``table`` (defined on ``phi_grid``) at ``phi``."""
        grid = self.phi_grid.to(dtype=phi.dtype, device=phi.device)
        vals = table.to(dtype=phi.dtype, device=phi.device)
        idx = torch.searchsorted(grid, phi.clamp(grid[0], grid[-1]).contiguous())
        idx = idx.clamp(1, grid.numel() - 1)
        lo, hi = idx - 1, idx
        weight = (phi.clamp(grid[0], grid[-1]) - grid[lo]) / (grid[hi] - grid[lo])
        return torch.lerp(vals[lo], vals[hi], weight)

    def shape(self, phi: Tensor) -> Tensor:  # noqa: D102
        return self._interp(self.values, phi)

    def shape_second_derivative(self, phi: Tensor) -> Tensor:  # noqa: D102
        return self._interp(self._second, phi)

    def shape_gaussian_second_moment(self, sigma_sq: float, n_nodes: int = 96) -> float:  # noqa: D102
        if sigma_sq == 0.0:
            zero = torch.zeros(1, dtype=torch.float64)
            return float(self._interp(self._second, zero).squeeze())
        nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
        phi = torch.as_tensor(nodes, dtype=torch.float64) * math.sqrt(2.0 * sigma_sq)
        w = torch.as_tensor(weights, dtype=torch.float64) / math.sqrt(math.pi)
        return float((self._interp(self._second, phi) * w).sum())

    def describe(self) -> dict[str, float]:  # noqa: D102
        return {"coupling": self.coupling, "grid_points": float(self.phi_grid.numel())}
