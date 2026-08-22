"""Random sampling of bulk interaction potentials, including GP-drawn ones."""

from __future__ import annotations

import math

import torch
from torch import Generator, Tensor

from qft_operator.data.config import DataConfig
from qft_operator.physics.potentials import (
    FreeTheory,
    PhiFour,
    PolynomialPotential,
    Potential,
    RandomFourierPotential,
    SineGordon,
)

__all__ = ["PotentialSampler"]


def _uniform(generator: Generator, low: float, high: float) -> float:
    """Draw a scalar from $\\mathcal{U}[low, high)$ using an explicit generator."""
    return float(torch.rand((), generator=generator) * (high - low) + low)


def _log_uniform(generator: Generator, low: float, high: float) -> float:
    """Draw a scalar log-uniformly from $[low, high)$, covering decades evenly."""
    return float(math.exp(_uniform(generator, math.log(low), math.log(high))))


class PotentialSampler:
    """Draw random theories from the configured mixture of potential families.

    The mixture is what turns the network from a Sine-Gordon interpolator into an
    operator: alongside the two named theories it draws generic polynomials and
    Gaussian-process functions, so nothing about the training distribution privileges the
    analytic families. Crucially every family exposes an exact
    $\\langle V''\\rangle_\\sigma$, so the GP-drawn theories carry labels of the same
    quality as Sine-Gordon rather than a heuristic.

    Args:
        config: Sampling ranges and family weights.
        generator: Torch RNG; pass a seeded generator for reproducible splits.

    Example:
        >>> import torch
        >>> from qft_operator.data import DataConfig, PotentialSampler
        >>> sampler = PotentialSampler(DataConfig(), torch.Generator().manual_seed(0))
        >>> potential = sampler.sample()
        >>> potential.family in DataConfig.KNOWN_FAMILIES
        True
    """

    def __init__(self, config: DataConfig, generator: Generator | None = None) -> None:
        self.config = config
        self.generator = generator or torch.Generator().manual_seed(config.seed)
        self.phi_grid = torch.linspace(-config.phi_max, config.phi_max, config.n_phi).to(
            torch.float64
        )
        self._weights = torch.tensor(config.normalized_weights, dtype=torch.float64)

    # ------------------------------------------------------------------ #
    def _draw_coupling(self) -> float:
        """Log-uniform magnitude with an optional random sign."""
        magnitude = _log_uniform(self.generator, *self.config.coupling_range)
        if not self.config.allow_negative_coupling:
            return magnitude
        sign = 1.0 if float(torch.rand((), generator=self.generator)) < 0.5 else -1.0
        return sign * magnitude

    def _shape_rms(self, potential: Potential) -> float:
        """Root-mean-square of the shape function over the field grid."""
        return float(potential.shape(self.phi_grid).pow(2).mean().sqrt())

    def sample_family(self) -> str:
        """Draw a family name according to the configured weights."""
        index = int(torch.multinomial(self._weights, 1, generator=self.generator))
        return self.config.families[index]

    def sample(self, family: str | None = None) -> Potential:
        """Draw one potential.

        Args:
            family: Force a specific family; ``None`` draws from the mixture.

        Returns:
            A :class:`~qft_operator.physics.potentials.Potential`.

        Raises:
            ValueError: If ``family`` is not a known family name.
        """
        chosen = family if family is not None else self.sample_family()
        if chosen == "free":
            return FreeTheory()
        if chosen == "sine_gordon":
            return SineGordon(
                coupling=self._draw_coupling(),
                xi=_uniform(self.generator, *self.config.xi_range),
            )
        if chosen == "phi4":
            return PhiFour(coupling=self._draw_coupling())
        if chosen == "polynomial":
            return self._sample_polynomial()
        if chosen == "gp_fourier":
            return self._sample_gp()
        raise ValueError(f"unknown family {chosen!r}")

    # ------------------------------------------------------------------ #
    def _sample_polynomial(self) -> PolynomialPotential:
        """Random polynomial with factorially-damped coefficients, rescaled to unit RMS."""
        degree = self.config.poly_degree
        raw = torch.randn(degree + 1, generator=self.generator, dtype=torch.float64)
        damping = torch.tensor(
            [1.0 / math.factorial(k) for k in range(degree + 1)], dtype=torch.float64
        )
        coefficients = raw * damping
        potential = PolynomialPotential(1.0, tuple(coefficients.tolist()))
        if self.config.normalize_shapes:
            rms = self._shape_rms(potential)
            if rms > 0.0:
                coefficients = coefficients / rms
        return PolynomialPotential(self._draw_coupling(), tuple(coefficients.tolist()))

    def _sample_gp(self) -> RandomFourierPotential:
        """Random-Fourier-feature GP sample with an RBF spectral density.

        The lengthscale $\\ell$ is itself drawn per sample, so the dataset spans smooth
        and wiggly potentials rather than one fixed smoothness class.
        """
        n = self.config.gp_features
        lengthscale = _uniform(self.generator, *self.config.gp_lengthscale_range)
        amplitudes = torch.randn(n, generator=self.generator, dtype=torch.float64)
        frequencies = torch.randn(n, generator=self.generator, dtype=torch.float64) / lengthscale
        phases = torch.rand(n, generator=self.generator, dtype=torch.float64) * 2.0 * math.pi
        potential = RandomFourierPotential(1.0, amplitudes, frequencies, phases)
        if self.config.normalize_shapes:
            rms = self._shape_rms(potential)
            if rms > 0.0:
                amplitudes = amplitudes / rms
        return RandomFourierPotential(self._draw_coupling(), amplitudes, frequencies, phases)

    def sample_separations(self) -> Tensor:
        """Draw a sorted, log-uniform grid of boundary separations.

        Sampling on a *grid* rather than independently (as the baseline did, via
        ``p2 = p1 + U(0, 3)``) matters for two reasons: it covers the whole $[r_{\\min},
        r_{\\max}]$ window uniformly in $\\log r$ -- the variable the physics is linear in --
        and it lets the boundary-scaling loss estimate log-derivatives on a well-conditioned
        stencil.

        Returns:
            Separations of shape ``(n_pairs,)``, strictly increasing.
        """
        lo, hi = self.config.log_r_range
        n = self.config.n_pairs
        spacing = (hi - lo) / max(n - 1, 1)
        offset = (
            float(torch.rand((), generator=self.generator) - 0.5)
            * self.config.separation_jitter
            * spacing
        )
        log_r = torch.linspace(lo, hi, n, dtype=torch.float64) + offset
        return torch.exp(log_r.clamp(lo, hi))

    def sample_midpoint(self) -> float:
        """Draw the common midpoint $\\bar{p}$ of a sample's boundary pairs."""
        span = self.config.midpoint_range
        return _uniform(self.generator, -span, span) if span > 0.0 else 0.0
