"""Configuration of the hybrid data-generation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["DataConfig", "TargetMode"]

TargetMode = Literal["resummed", "quadrature", "hybrid"]
"""How ground-truth correlators are produced.

``"resummed"``
    Closed-form anomalous dimension, exponentiated into $W = r^{-2\\Delta_{\\rm eff}}$ with
    the coupling evaluated at the physical scale $1/r$. Fast, and *exactly* RG-invariant.
``"quadrature"``
    First order in the interaction, with the bulk contact diagram evaluated by actual
    Gauss-Legendre quadrature over the AdS2 bulk. Physically the most literal, but a
    fixed-order expression, hence not RG-invariant -- train it with ``rg`` weight zero.
``"hybrid"``
    The anomalous dimension is measured from the quadrature (the log-derivative of the
    regulated contact integral) and then resummed as in ``"resummed"``. Numerically
    grounded *and* RG-consistent; this is the flagship pipeline.
"""


@dataclass(frozen=True)
class DataConfig:
    """Sampling ranges and grid resolutions for the AdS2 correlator dataset.

    See :class:`~qft_operator.data.dataset.AdS2CorrelatorDataset` for the consumer.

    Args:
        n_train: Number of training theories.
        n_val: Number of validation theories.
        n_test: Number of test theories.
        n_phi: Field-grid resolution of the branch input.
        n_pairs: Boundary point pairs queried per theory.
        phi_max: Field grid spans $[-\\phi_{\\max}, \\phi_{\\max}]$.
        r_min: Smallest boundary separation.
        r_max: Largest boundary separation.
        midpoint_range: Half-width of the uniform window from which pair midpoints
            $\\bar{p}$ are drawn. Irrelevant to the exact correlator, which is what
            makes it a useful implicit test of the architecture's translation
            invariance.
        separation_jitter: Random offset, in units of the grid spacing in $\\log r$,
            applied to each sample's separation grid so that no two theories are queried
            at exactly the same points.
        target_mode: See :data:`TargetMode`.
        family_weights: Relative sampling probability of each potential family. Keys must
            be among ``free``, ``sine_gordon``, ``phi4``, ``polynomial``, ``gp_fourier``.
        coupling_range: $|\\lambda|$ is drawn log-uniformly from this range.
        allow_negative_coupling: Draw a random sign for $\\lambda$, so the dataset covers
            both signs of $\\gamma$ rather than only one branch of the spectrum.
        xi_range: Sine-Gordon vertex exponent range.
        poly_degree: Highest monomial in the polynomial family.
        gp_features: Random Fourier features per GP sample.
        gp_lengthscale_range: RBF lengthscale range; the spectral density is
            $\\omega \\sim \\mathcal{N}(0, \\ell^{-2})$.
        max_gamma_ratio: Reject draws whose $|\\gamma| / (\\Delta\\beta_1\\beta_2)$ exceeds
            this, or ``None`` to keep everything.

            This is a physics constraint, not a loss trick. The labels are first order in
            the interaction, which presumes $\\gamma \\ll \\Delta$; a draw with
            $|\\gamma|/\\Delta = 0.19$ shifts the boundary exponent by nearly a fifth and
            the first-order formula simply does not describe it. Such draws come
            exclusively from the Gaussian-process family -- about 5% of its samples --
            and because the loss is quadratic they dominate it, starving every other
            family of gradient. At the default the cap discards ~1.5% of the mixture and
            cuts the ratio of largest to median $|\\gamma|$ from ~70x to ~20x.
        normalize_shapes: Rescale polynomial and GP shape functions to unit RMS on the
            field grid, so that $\\lambda$ alone controls the interaction strength and
            $\\gamma$ stays comparable across families.
        standardize_inputs: Divide the branch input by a single scalar computed from the
            training split. A *single* scalar (rather than a per-sample one) is essential:
            it keeps the branch input exactly linear in $\\lambda$, which is what makes the
            RG loss's chain rule through $\\partial V/\\partial\\lambda$ correct.
        batch_size: Mini-batch size.
        num_workers: DataLoader workers.
        seed: Master RNG seed; the three splits are drawn from decorrelated substreams.

    Raises:
        ValueError: On any inconsistent range or unknown family name.
    """

    n_train: int = 4096
    n_val: int = 512
    n_test: int = 512
    n_phi: int = 64
    n_pairs: int = 32
    phi_max: float = 3.0
    r_min: float = 0.05
    r_max: float = 12.0
    midpoint_range: float = 4.0
    separation_jitter: float = 0.4
    target_mode: TargetMode = "resummed"
    family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "free": 0.1,
            "sine_gordon": 0.3,
            "phi4": 0.15,
            "polynomial": 0.15,
            "gp_fourier": 0.3,
        }
    )
    coupling_range: tuple[float, float] = (0.005, 0.05)
    allow_negative_coupling: bool = True
    xi_range: tuple[float, float] = (0.4, 1.2)
    poly_degree: int = 6
    gp_features: int = 64
    gp_lengthscale_range: tuple[float, float] = (0.4, 1.5)
    max_gamma_ratio: float | None = 0.05
    normalize_shapes: bool = True
    standardize_inputs: bool = True
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0

    KNOWN_FAMILIES = ("free", "sine_gordon", "phi4", "polynomial", "gp_fourier")

    def __post_init__(self) -> None:
        if min(self.n_train, self.n_val, self.n_test) < 1:
            raise ValueError("each split needs at least one sample")
        if self.n_phi < 8 or self.n_pairs < 4:
            raise ValueError("need n_phi >= 8 and n_pairs >= 4")
        if not 0.0 < self.r_min < self.r_max:
            raise ValueError(f"need 0 < r_min < r_max, got {self.r_min}, {self.r_max}")
        if self.phi_max <= 0.0:
            raise ValueError("phi_max must be positive")
        lo, hi = self.coupling_range
        if not 0.0 < lo <= hi:
            raise ValueError(f"invalid coupling_range {self.coupling_range}")
        if not 0.0 < self.xi_range[0] <= self.xi_range[1]:
            raise ValueError(f"invalid xi_range {self.xi_range}")
        if not 0.0 < self.gp_lengthscale_range[0] <= self.gp_lengthscale_range[1]:
            raise ValueError(f"invalid gp_lengthscale_range {self.gp_lengthscale_range}")
        if self.poly_degree < 2:
            raise ValueError("poly_degree must be at least 2 to produce a non-zero V''")
        unknown = set(self.family_weights) - set(self.KNOWN_FAMILIES)
        if unknown:
            raise ValueError(f"unknown potential families: {sorted(unknown)}")
        if not self.family_weights or sum(self.family_weights.values()) <= 0.0:
            raise ValueError("family_weights must contain at least one positive weight")
        if self.target_mode not in ("resummed", "quadrature", "hybrid"):
            raise ValueError(f"unknown target_mode {self.target_mode!r}")
        if self.max_gamma_ratio is not None and self.max_gamma_ratio <= 0.0:
            raise ValueError(f"max_gamma_ratio must be positive, got {self.max_gamma_ratio}")

    @property
    def log_r_range(self) -> tuple[float, float]:
        """The separation window expressed in $\\log r$."""
        return math.log(self.r_min), math.log(self.r_max)

    @property
    def families(self) -> tuple[str, ...]:
        """Family names with positive weight, in a deterministic order."""
        return tuple(f for f in self.KNOWN_FAMILIES if self.family_weights.get(f, 0.0) > 0.0)

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        """Sampling probabilities aligned with :attr:`families`."""
        raw = [self.family_weights[f] for f in self.families]
        total = sum(raw)
        return tuple(w / total for w in raw)
