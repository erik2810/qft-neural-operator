"""Fourier-DeepONet / operator-transformer for the map $V(\\phi) \\mapsto W(p_1, p_2)$.

Three design decisions distinguish this from a textbook DeepONet, and each is a direct
consequence of the physics:

**Log-space output.** Over the default window $r \\in [0.05, 12]$ with $\\Delta \\approx
1.5$, $W = r^{-2\\Delta_{\\mathrm{eff}}}$ spans roughly eight decades. Regressing $W$
under an $L^2$ loss makes the objective a function of the few smallest separations only.
The network therefore predicts $\\log W$ throughout.

**A free-theory baseline.** $\\log W^{(0)} = -2\\Delta\\beta_1\\beta_2\\log r$ is known in
closed form. Subtracting it leaves the network to model the *anomalous* part, which is
smaller by a factor of order $\\gamma/\\Delta \\sim 10^{-3}$ -- exactly the signal the
framework exists to resolve.

**Structural conformal symmetry.** Translation invariance along the boundary is imposed
by the positional encoding rather than learned (see
:class:`~qft_operator.models.layers.MetricPositionalEncoding`).

Heads
-----
``"inner_product"``
    The classical DeepONet contraction $\\sum_l b_l(V)\\,t_l(p_1, p_2)$, with the branch
    additionally FiLM-conditioning the trunk.
``"attention"``
    An operator-transformer head: each query point attends over the branch's field-space
    tokens, so different separations can weight different regions of $\\phi$-space.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal

import torch
from torch import Tensor, nn

from qft_operator.models.branch import BranchNet
from qft_operator.models.layers import MLP
from qft_operator.models.trunk import TrunkNet
from qft_operator.physics.config import PhysicsConfig

__all__ = ["FourierDeepONet", "HeadKind", "ResidualMode"]


def _math_attention() -> AbstractContextManager[None]:
    """Force the math SDPA backend, which supports double backward.

    The fused flash/mem-efficient attention kernels have no second-derivative rule, and
    :class:`~qft_operator.losses.scaling.BoundaryScalingLoss` differentiates the network
    twice. Sequence lengths here are the handful of branch tokens, so the math backend
    costs nothing measurable.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - torch < 2.3
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


HeadKind = Literal["inner_product", "attention"]
ResidualMode = Literal["none", "free", "exponent"]
"""How the network output is turned into $\\log W$.

``"none"``
    $\\log W = f_\\theta$. Maximum freedom, slowest convergence.
``"free"``
    $\\log W = -2\\Delta\\beta_1\\beta_2\\log r + f_\\theta$. The network models only the
    anomalous correction; the free-theory limit is reached when $f_\\theta \\to 0$. This
    is the default.
``"exponent"``
    $\\log W = -2(\\Delta\\beta_1\\beta_2 - \\gamma_\\theta(V, M))\\log r$: the network
    predicts the anomalous dimension itself. The head is evaluated once per sample at a
    *fixed reference separation* $r = 1$, so $\\gamma_\\theta$ carries no residual
    $r$-dependence and the output is an exact power law -- the boundary-scaling loss is
    then identically zero and :mod:`qft_operator.analysis.spectrum` can read $\\gamma$
    straight off the head. The trade-off is that a fixed-order target, whose effective
    exponent genuinely drifts with $r$, cannot be represented in this mode.
"""


class FourierDeepONet(nn.Module):
    """Operator network mapping bulk potentials to boundary connected correlators.

    Args:
        config: AdS2 physics configuration; supplies $L$ and $\\Delta\\beta_1\\beta_2$ for
            the encoding and the free-theory baseline.
        n_phi: Field-grid resolution of the branch input.
        latent_dim: Shared latent width of branch and trunk.
        branch_width: Channel width of the branch's spectral stack.
        branch_blocks: Number of spectral blocks in the branch.
        branch_modes: Retained Fourier modes in the branch.
        branch_hidden: Hidden widths of the branch's projection head.
        trunk_width: Hidden width of the trunk.
        trunk_layers: Number of FiLM-conditioned trunk layers.
        trunk_modes: Retained modes in the boundary context field.
        context_grid: Internal reference grid size of the boundary context field.
        context_width: Channel width of the boundary context field.
        log_r_range: ``(min, max)`` extent of that internal grid in $\\log r$. It should
            bracket the separation window the data covers, padded a little, so queries
            never land on the clamped edge of the interpolation.
        num_frequencies: Fourier features in the positional encoding.
        fourier_scale: Bandwidth of those features.
        radial_mode: Holographic depth convention.
        translation_invariant: Enforce boundary translation invariance structurally.
        spectral_mixing: Enable non-local mixing along the boundary query axis.
        branch_spectral: Enable spectral convolutions along $\\phi$ in the branch.
        head: ``"inner_product"`` or ``"attention"``.
        n_heads: Attention heads when ``head="attention"``.
        n_tokens: Branch tokens exposed to the attention head.
        residual_mode: See :data:`ResidualMode`.
        dropout: Dropout probability throughout.
        readout_init_scale: Scale of the readout initialization. The default ``1e-4``
            starts the network within $O(10^{-4})$ of the free theory -- an order of
            magnitude below a typical $\\gamma \\sim 10^{-3}$, so the prior is intact --
            while giving every parameter a gradient on the very first step. Setting it
            to exactly ``0.0`` makes the free-theory limit exact at initialization at the
            cost of one frozen step upstream of the readout.

    Example:
        >>> import torch
        >>> from qft_operator.physics import PhysicsConfig
        >>> from qft_operator.models import FourierDeepONet
        >>> net = FourierDeepONet(PhysicsConfig(), n_phi=64)
        >>> v = torch.zeros(2, 64)
        >>> coords = torch.stack([torch.zeros(2, 16), torch.linspace(0.5, 4, 16).expand(2, 16)], -1)
        >>> net(v, coords).shape
        torch.Size([2, 16])
    """

    def __init__(
        self,
        config: PhysicsConfig,
        n_phi: int = 64,
        latent_dim: int = 128,
        branch_width: int = 64,
        branch_blocks: int = 4,
        branch_modes: int = 16,
        branch_hidden: list[int] | None = None,
        trunk_width: int = 128,
        trunk_layers: int = 4,
        trunk_modes: int = 16,
        context_grid: int = 64,
        context_width: int = 64,
        log_r_range: tuple[float, float] = (-3.5, 3.0),
        num_frequencies: int = 16,
        fourier_scale: float = 1.5,
        radial_mode: str = "separation",
        translation_invariant: bool = True,
        spectral_mixing: bool = True,
        branch_spectral: bool = True,
        head: HeadKind = "inner_product",
        n_heads: int = 4,
        n_tokens: int = 8,
        residual_mode: ResidualMode = "free",
        dropout: float = 0.0,
        readout_init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if head not in ("inner_product", "attention"):
            raise ValueError(f"unknown head {head!r}")
        if residual_mode not in ("none", "free", "exponent"):
            raise ValueError(f"unknown residual_mode {residual_mode!r}")
        self.config = config
        self.head_kind = head
        self.residual_mode = residual_mode
        self.free_dimension = config.free_dimension

        self.branch = BranchNet(
            n_phi=n_phi,
            latent_dim=latent_dim,
            width=branch_width,
            n_blocks=branch_blocks,
            n_modes=branch_modes,
            hidden_dims=list(branch_hidden) if branch_hidden is not None else None,
            n_tokens=n_tokens,
            emit_tokens=head == "attention",
            dropout=dropout,
            use_spectral=branch_spectral,
        )
        self.trunk = TrunkNet(
            config=config,
            latent_dim=latent_dim,
            width=trunk_width,
            n_layers=trunk_layers,
            num_frequencies=num_frequencies,
            fourier_scale=fourier_scale,
            radial_mode=radial_mode,
            translation_invariant=translation_invariant,
            spectral_mixing=spectral_mixing,
            n_modes=trunk_modes,
            context_grid=context_grid,
            context_width=context_width,
            log_r_range=tuple(log_r_range),  # type: ignore[arg-type]
            dropout=dropout,
        )

        if head == "attention":
            self.attention = nn.MultiheadAttention(
                latent_dim, num_heads=n_heads, dropout=dropout, batch_first=True
            )
            self.attention_out = MLP(latent_dim, [latent_dim], latent_dim)
        # Final contraction of the DeepONet head. Writing it as a learned linear map over
        # the elementwise branch-trunk product generalizes the usual plain sum (recovered
        # at weight = 1) and, zero-initialized, makes an untrained network sit *exactly*
        # on the free theory in "free"/"exponent" mode -- while still receiving a full
        # latent_dim-vector of gradients on the very first step, which a single scalar
        # gate would not (see readout_init_scale).
        # Recorded verbatim so a checkpoint can be rebuilt exactly. Inferring the
        # architecture from weight shapes works only for the widths that happen to be
        # visible in them, and silently produces the wrong network for everything else.
        self.hyperparameters: dict[str, Any] = {
            "n_phi": n_phi,
            "latent_dim": latent_dim,
            "branch_width": branch_width,
            "branch_blocks": branch_blocks,
            "branch_modes": branch_modes,
            "branch_hidden": list(branch_hidden) if branch_hidden is not None else None,
            "trunk_width": trunk_width,
            "trunk_layers": trunk_layers,
            "trunk_modes": trunk_modes,
            "context_grid": context_grid,
            "context_width": context_width,
            "log_r_range": list(log_r_range),
            "num_frequencies": num_frequencies,
            "fourier_scale": fourier_scale,
            "radial_mode": radial_mode,
            "translation_invariant": translation_invariant,
            "spectral_mixing": spectral_mixing,
            "branch_spectral": branch_spectral,
            "head": head,
            "n_heads": n_heads,
            "n_tokens": n_tokens,
            "residual_mode": residual_mode,
            "dropout": dropout,
            "readout_init_scale": readout_init_scale,
        }

        if readout_init_scale < 0.0:
            raise ValueError("readout_init_scale must be non-negative")
        self.readout = nn.Linear(latent_dim, 1)
        with torch.no_grad():
            self.readout.weight.normal_(0.0, readout_init_scale)
            self.readout.bias.zero_()

    # ------------------------------------------------------------------ #
    def _raw_output(self, v_phi: Tensor, coords: Tensor, log_m: Tensor | None) -> Tensor:
        """Network output $f_\\theta$ before the residual reparametrization."""
        code, tokens = self.branch(v_phi)
        basis = self.trunk(coords, code, log_m)
        if self.head_kind == "inner_product":
            product = code.unsqueeze(1) * basis
        else:
            assert tokens is not None  # emit_tokens is tied to the attention head
            with _math_attention():
                attended, _ = self.attention(basis, tokens, tokens, need_weights=False)
            product = self.attention_out(attended + basis) * code.unsqueeze(1)
        return self.readout(product).squeeze(-1)

    def forward(
        self,
        v_phi: Tensor,
        coords: Tensor,
        log_m: Tensor | None = None,
    ) -> Tensor:
        """Predict $\\log W(p_1, p_2)$ for a batch of theories.

        Args:
            v_phi: Potential on the field grid, shape ``(batch, n_phi)``.
            coords: Boundary pairs $(p_1, p_2)$, shape ``(batch, points, 2)``.
            log_m: $\\log M$ per sample, shape ``(batch, 1)`` or ``(batch, points)``.
                Defaults to zero, i.e. $M = 1$.

        Returns:
            $\\log W$ of shape ``(batch, points)``.

        Raises:
            ValueError: If ``coords`` does not have a trailing dimension of 2.
        """
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be (batch, points, 2), got {tuple(coords.shape)}")
        if self.residual_mode == "exponent":
            gamma = self._raw_output(
                v_phi, self._reference_pair(coords), self._sample_scale(coords, log_m)
            )
            log_r = torch.log((coords[..., 0] - coords[..., 1]).abs().clamp_min(1e-12))
            return -2.0 * (self.free_dimension - gamma) * log_r

        raw = self._raw_output(v_phi, coords, log_m)
        if self.residual_mode == "none":
            return raw
        log_r = torch.log((coords[..., 0] - coords[..., 1]).abs().clamp_min(1e-12))
        return -2.0 * self.free_dimension * log_r + raw

    @staticmethod
    def _reference_pair(coords: Tensor) -> Tensor:
        """A single boundary pair at unit separation, one per batch element."""
        ones = torch.ones(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
        return torch.stack([torch.zeros_like(ones), ones], dim=-1)

    @staticmethod
    def _sample_scale(coords: Tensor, log_m: Tensor | None) -> Tensor:
        """Reduce a possibly per-point log M to one value per sample."""
        if log_m is None:
            return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
        return log_m.reshape(coords.shape[0], -1)[:, :1]

    def predict_correlator(
        self,
        v_phi: Tensor,
        coords: Tensor,
        log_m: Tensor | None = None,
    ) -> Tensor:
        """Predict $W$ itself by exponentiating :meth:`forward`."""
        return torch.exp(self.forward(v_phi, coords, log_m))

    def predict_anomalous_dimension(
        self,
        v_phi: Tensor,
        coords: Tensor,
        log_m: Tensor | None = None,
    ) -> Tensor:
        """Read off $\\gamma$ from the prediction.

        In ``residual_mode="exponent"`` this is the network's own head output and is
        exact (and constant across query points, by construction). Otherwise it is
        inferred from the local log-slope,
        $\\gamma = \\Delta\\beta_1\\beta_2 + \\tfrac{1}{2}\\,d\\log W / d\\log r$, evaluated by
        autograd.

        Args:
            v_phi: Potential samples, shape ``(batch, n_phi)``.
            coords: Boundary pairs, shape ``(batch, points, 2)``.
            log_m: $\\log M$ per sample.

        Returns:
            $\\gamma$ of shape ``(batch, points)``.
        """
        if self.residual_mode == "exponent":
            gamma = self._raw_output(
                v_phi, self._reference_pair(coords), self._sample_scale(coords, log_m)
            )
            return gamma.expand(-1, coords.shape[1])

        from qft_operator.losses.operators import log_slope

        slope = log_slope(self, v_phi, coords, log_m)
        return self.free_dimension + 0.5 * slope

    @classmethod
    def from_hyperparameters(
        cls, config: PhysicsConfig, hyperparameters: dict[str, Any]
    ) -> FourierDeepONet:
        """Rebuild a network from the dict recorded in :attr:`hyperparameters`.

        Args:
            config: The AdS2 background to build against.
            hyperparameters: Constructor arguments as stored in a checkpoint.

        Returns:
            A network with the same architecture as the one that produced the dict.
        """
        kwargs = dict(hyperparameters)
        if "log_r_range" in kwargs and kwargs["log_r_range"] is not None:
            kwargs["log_r_range"] = tuple(kwargs["log_r_range"])
        return cls(config, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
