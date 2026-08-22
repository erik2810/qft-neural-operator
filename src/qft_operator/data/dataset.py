"""Materialized dataset of (potential, boundary correlator) pairs."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset

from qft_operator.data.config import DataConfig
from qft_operator.data.samplers import PotentialSampler
from qft_operator.physics.bulk_integrals import (
    ConformalIntegrator,
    QuadratureSpec,
    ReducedIntegralTable,
)
from qft_operator.physics.config import PhysicsConfig
from qft_operator.physics.correlators import (
    first_order_log_correlator,
    resummed_log_correlator,
)
from qft_operator.physics.rg import BetaFunction, RGConfig, scale_anomalous_dimension

__all__ = ["AdS2CorrelatorDataset", "DatasetStatistics"]


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary of a generated split, for logging and for reusing normalization.

    Attributes:
        feature_scale: The single scalar the branch input was divided by.
        gamma_mean: Mean anomalous dimension across the split.
        gamma_std: Standard deviation of the anomalous dimension.
        gamma_abs_max: Largest $|\\gamma|$ in the split.
        family_counts: Number of samples drawn from each family.
        rejected: Draws discarded for exceeding
            :attr:`~qft_operator.data.config.DataConfig.max_gamma_ratio`, i.e. for lying
            outside the regime where a first-order label means anything.
    """

    feature_scale: float
    gamma_mean: float
    gamma_std: float
    gamma_abs_max: float
    family_counts: dict[str, int]
    rejected: int = 0


class AdS2CorrelatorDataset(Dataset[dict[str, Tensor]]):
    """Theories $V(\\phi)$ paired with their exact boundary correlators $W(p_1, p_2)$.

    Every sample is generated eagerly at construction time and stored as stacked
    tensors, so training is not bottlenecked by Python-level sampling and a split is
    exactly reproducible from its seed.

    Each item is a dict with:

    ==================  =====================  =============================================
    key                 shape                  meaning
    ==================  =====================  =============================================
    ``v_phi``           ``(n_phi,)``           $V(\\phi_i)$, standardized
    ``dv_dcoupling``    ``(n_phi,)``           exact $\\partial V/\\partial\\lambda$, same scaling
    ``coords``          ``(n_pairs, 2)``       boundary pairs $(p_1, p_2)$
    ``log_w``           ``(n_pairs,)``         $\\log W$, the regression target
    ``delta_eff``       ``(n_pairs,)``         $\\Delta_{\\mathrm{eff}}$ at each separation
    ``log_m``           ``(1,)``               $\\log M$ for this sample
    ``coupling``        ``()``                 $\\lambda(M)$
    ``gamma``           ``()``                 $\\gamma$, for spectrum evaluation
    ``family``          ``()``                 index into :attr:`family_names`
    ==================  =====================  =============================================

    Args:
        n_samples: Number of theories to generate.
        physics: AdS2 background configuration.
        data: Sampling configuration.
        rg: RG flow configuration; its ``log_scale_jitter`` sets the spread of $\\log M$.
        seed: Split seed. Pass different seeds to train/val/test.
        feature_scale: Reuse a normalization scalar from another split. ``None``
            computes it from this split -- do that for training only, then pass the
            result to validation and test so all splits share one scaling.
        integrator: Quadrature engine for the ``"quadrature"`` and ``"hybrid"`` target
            modes; built with defaults if omitted.

    Raises:
        ValueError: If ``n_samples`` is not positive.
    """

    family_names = DataConfig.KNOWN_FAMILIES

    #: Redraw budget per sample before the rejection filter is declared unsatisfiable.
    _MAX_DRAW_ATTEMPTS = 256

    def __init__(
        self,
        n_samples: int,
        physics: PhysicsConfig | None = None,
        data: DataConfig | None = None,
        rg: RGConfig | None = None,
        seed: int = 0,
        feature_scale: float | None = None,
        integrator: ConformalIntegrator | None = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        self.physics = physics or PhysicsConfig()
        self.data = data or DataConfig()
        self.rg_config = rg or RGConfig()
        self.beta = BetaFunction(self.rg_config)
        self.n_samples = n_samples

        generator = torch.Generator().manual_seed(seed)
        self.sampler = PotentialSampler(self.data, generator)
        self.phi_grid = self.sampler.phi_grid

        self._reduced_table: ReducedIntegralTable | None = None
        self._log_coefficient = self.physics.log_coefficient
        if self.data.target_mode in ("quadrature", "hybrid"):
            engine = integrator or ConformalIntegrator(self.physics, QuadratureSpec())
            self._warn_on_convention_mismatch()
            if self.data.target_mode == "quadrature":
                lo, hi = self._cutoff_ratio_range()
                self._reduced_table = ReducedIntegralTable(engine, log_x_min=lo, log_x_max=hi)
            else:
                self._log_coefficient = float(
                    engine.log_slope(torch.tensor([1.0], dtype=torch.float64), eps=1e-4).squeeze()
                )

        self._generate(generator, feature_scale)

    # ------------------------------------------------------------------ #
    def _warn_on_convention_mismatch(self) -> None:
        """Flag a c_delta override that puts the analytic and numerical paths at odds."""
        ratio = self.physics.convention_ratio
        if abs(ratio - 1.0) > 1e-6:
            warnings.warn(
                f"c_delta override is {ratio:.4g}x the unit-normalized CFT value, but the "
                f"bulk integrator uses the unit-normalized kernel: anomalous dimensions "
                f"from target_mode={self.data.target_mode!r} will differ from the analytic "
                f"ones by that factor. Set PhysicsConfig(c_delta=None) to align them.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _cutoff_ratio_range(self) -> tuple[float, float]:
        """Range of $\\log(r/\\epsilon)$ the samples will actually visit.

        The near-boundary expansion behind the contact integral needs the cutoff well
        inside the smallest separation, $\\epsilon \\ll r$. Catching a violation here --
        with the fix spelled out -- beats letting the table interpolation fail later.

        Returns:
            ``(log_x_min, log_x_max)``, padded by half a unit on each side.

        Raises:
            ValueError: If the configured scales put the cutoff at or above $r_{\\min}$.
        """
        jitter = self.rg_config.log_scale_jitter
        log_m_lo = self.rg_config.log_reference_scale - jitter
        log_m_hi = self.rg_config.log_reference_scale + jitter
        # log(r / eps) = log r + log M
        lo = math.log(self.data.r_min) + log_m_lo
        hi = math.log(self.data.r_max) + log_m_hi
        if lo < 1.0:
            raise ValueError(
                f"target_mode='quadrature' needs the bulk cutoff well inside the smallest "
                f"separation, but min log(r/eps) = {lo:.2f}. Raise "
                f"RGConfig.reference_scale to at least "
                f"{math.exp(1.0 - math.log(self.data.r_min) + jitter):.0f}, or raise "
                f"DataConfig.r_min."
            )
        return lo - 0.5, hi + 0.5

    def _peak_gamma(self, gamma_reference: float, coupling: float, log_m: float) -> float:
        """Largest $|\\gamma|$ the sample will actually carry across the separation window.

        The cap has to bound the anomalous dimension the *targets* use, not the one the
        coupling happens to be quoted at. Those differ whenever the coupling runs: the
        correlator is built from $\\bar\\lambda(1/r)$, so $\\gamma$ varies across the window,
        and at $\\epsilon = 0.35$ it varies by a factor of six. Checking only the reference
        value let samples through at $|\\gamma|/\\Delta = 0.21$ -- four times over the cap --
        in exactly the configuration the cap exists to protect.

        The flow is monotone in $\\log\\mu$, so the two window edges bound the interior and
        no scan is needed.

        Args:
            gamma_reference: $\\gamma$ at the coupling's quoted scale.
            coupling: $\\lambda(M)$.
            log_m: $\\log M$.

        Returns:
            $\\max_r |\\gamma(r)|$ over the configured separation window.
        """
        if self.rg_config.is_marginal or gamma_reference == 0.0:
            return abs(gamma_reference)
        edges = torch.tensor(self.data.log_r_range, dtype=torch.float64)
        running = scale_anomalous_dimension(
            torch.full_like(edges, gamma_reference),
            torch.full_like(edges, coupling),
            torch.full_like(edges, log_m),
            edges,
            self.beta,
        )
        return float(running.abs().max())

    def _generate(self, generator: Generator, feature_scale: float | None) -> None:
        """Draw every sample and stack the results into contiguous tensors."""
        cfg, phys = self.data, self.physics
        v_list, dv_list, coords_list = [], [], []
        log_w_list, delta_list = [], []
        log_m_list, coupling_list, gamma_list, family_list = [], [], [], []
        counts = dict.fromkeys(self.family_names, 0)

        log_m_ref = self.rg_config.log_reference_scale
        jitter = self.rg_config.log_scale_jitter

        ceiling = None if cfg.max_gamma_ratio is None else cfg.max_gamma_ratio * phys.free_dimension
        rejected = 0

        for _ in range(self.n_samples):
            # Draw the scale at which this sample's coupling is quoted, then transport the
            # coupling there. Doing it in this order is what makes the pair (V, M) a
            # genuine RG orbit rather than two independent random numbers.
            offset = float(torch.rand((), generator=generator) * 2.0 - 1.0) * jitter
            log_m = log_m_ref + offset

            # Redraw anything outside the perturbative window the first-order label
            # assumes. Only the GP family ever trips this, at a few percent.
            for _attempt in range(self._MAX_DRAW_ATTEMPTS):
                potential = self.sampler.sample()
                if potential.coupling != 0.0 and not self.rg_config.is_marginal:
                    transported = self.beta.run(
                        torch.tensor(potential.coupling, dtype=torch.float64),
                        torch.tensor(offset, dtype=torch.float64),
                    )
                    potential.coupling = float(transported)

                moment = potential.gaussian_second_moment(phys.sigma_sq)
                gamma = 0.5 * phys.beta1 * phys.beta2 * moment * self._log_coefficient
                if ceiling is None or self._peak_gamma(gamma, potential.coupling, log_m) <= ceiling:
                    break
                rejected += 1
            else:
                raise RuntimeError(
                    f"could not draw a theory with |gamma| <= {ceiling:.3g} in "
                    f"{self._MAX_DRAW_ATTEMPTS} attempts; the coupling range and "
                    f"max_gamma_ratio are inconsistent"
                )
            counts[potential.family] += 1

            radii = self.sampler.sample_separations()
            midpoint = self.sampler.sample_midpoint()
            coords = torch.stack([midpoint - 0.5 * radii, midpoint + 0.5 * radii], dim=-1)
            log_r = torch.log(radii)

            if self.data.target_mode == "quadrature":
                assert self._reduced_table is not None
                reduced = self._reduced_table(radii, eps=math.exp(-log_m))
                log_w, delta_eff = first_order_log_correlator(log_r, moment, phys, reduced)
            else:
                log_w, delta_eff = resummed_log_correlator(
                    log_r, gamma, potential.coupling, log_m, phys, self.beta
                )

            v_list.append(potential.evaluate(self.phi_grid))
            dv_list.append(potential.d_dcoupling(self.phi_grid))
            coords_list.append(coords)
            log_w_list.append(log_w)
            delta_list.append(delta_eff)
            log_m_list.append(log_m)
            coupling_list.append(potential.coupling)
            gamma_list.append(gamma)
            family_list.append(self.family_names.index(potential.family))

        self.v_phi = torch.stack(v_list).to(torch.float32)
        self.dv_dcoupling = torch.stack(dv_list).to(torch.float32)
        self.coords = torch.stack(coords_list).to(torch.float32)
        self.log_w = torch.stack(log_w_list).to(torch.float32)
        self.delta_eff = torch.stack(delta_list).to(torch.float32)
        self.log_m = torch.tensor(log_m_list, dtype=torch.float32).unsqueeze(-1)
        self.coupling = torch.tensor(coupling_list, dtype=torch.float32)
        self.gamma = torch.tensor(gamma_list, dtype=torch.float32)
        self.family = torch.tensor(family_list, dtype=torch.long)

        scale = feature_scale
        if scale is None:
            scale = float(self.v_phi.pow(2).mean().sqrt()) if cfg.standardize_inputs else 1.0
            scale = scale if scale > 0.0 else 1.0
        # One global scalar, applied identically to V and dV/dlambda: the branch input
        # stays exactly linear in lambda, so the RG chain rule remains valid.
        self.feature_scale = scale
        self.v_phi = self.v_phi / scale
        self.dv_dcoupling = self.dv_dcoupling / scale

        self.statistics = DatasetStatistics(
            feature_scale=scale,
            gamma_mean=float(self.gamma.mean()),
            gamma_std=float(self.gamma.std(correction=0)),
            gamma_abs_max=float(self.gamma.abs().max()),
            family_counts=counts,
            rejected=rejected,
        )

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # noqa: D105
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:  # noqa: D105
        return {
            "v_phi": self.v_phi[index],
            "dv_dcoupling": self.dv_dcoupling[index],
            "coords": self.coords[index],
            "log_w": self.log_w[index],
            "delta_eff": self.delta_eff[index],
            "log_m": self.log_m[index],
            "coupling": self.coupling[index],
            "gamma": self.gamma[index],
            "family": self.family[index],
        }

    @property
    def separations(self) -> Tensor:
        """Boundary separations $r$ of every sample, shape ``(n_samples, n_pairs)``."""
        return (self.coords[..., 0] - self.coords[..., 1]).abs()
