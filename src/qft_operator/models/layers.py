"""Building blocks: spectral convolutions, Fourier features, metric-aware encodings.

The architectural claim this module implements is that the two directions of the
problem are *non-local in different ways*:

* along the field coordinate $\\phi$, the physics depends on smooth global structure of
  $V(\\phi)$ (ultimately on $\\langle V''\\rangle_\\sigma$, a smeared curvature), which a
  pointwise MLP over grid samples has to reconstruct from scratch;
* along the boundary coordinate $p$, the correlator is controlled by conformal ratios
  rather than by any local neighbourhood.

Spectral (Fourier) convolutions give both directions a global receptive field in one
layer, which is exactly the Fourier-Neural-Operator construction; the metric-aware
encoding then injects the AdS2 conformal factor $\\sqrt{g} = L^2/z^2$ into the latent
representation so the network never has to learn the background geometry.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from qft_operator.physics.config import PhysicsConfig

__all__ = [
    "SpectralConv1d",
    "FourierBlock1d",
    "FourierFeatures",
    "MetricPositionalEncoding",
    "FiLM",
    "MLP",
]


class SpectralConv1d(nn.Module):
    """Fourier-domain convolution keeping the lowest ``n_modes`` frequencies.

    Implements $(\\mathcal{K}u)(x) = \\mathcal{F}^{-1}\\big(R \\cdot \\mathcal{F}u\\big)(x)$
    with $R$ a learned complex tensor truncated to the low modes -- the FNO layer of Li
    et al. Truncation is the regularizer: it forces the operator to act on resolvable
    global structure rather than on grid noise.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        n_modes: Number of retained Fourier modes; clamped to the signal's Nyquist limit
            at run time, so the layer is resolution-agnostic.

    Shape:
        - Input: ``(batch, in_channels, length)``
        - Output: ``(batch, out_channels, length)``
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        if min(in_channels, out_channels, n_modes) < 1:
            raise ValueError("channels and n_modes must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        # Stored as a real tensor with a trailing (re, im) axis so that optimizers,
        # schedulers and checkpointing all behave exactly as for real parameters.
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes, 2))

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        length = x.shape[-1]
        x_ft = torch.fft.rfft(x, n=length, dim=-1)
        modes = min(self.n_modes, x_ft.shape[-1])
        weight = torch.view_as_complex(self.weight[:, :, :modes, :].contiguous())
        mixed = torch.einsum("bim,iom->bom", x_ft[..., :modes], weight)
        pad = x_ft.shape[-1] - modes
        if pad > 0:
            # Concatenate rather than assign into a zeros buffer: no in-place write ever
            # touches the autograd graph (see the differentiability tests).
            tail = torch.zeros(
                mixed.shape[0], self.out_channels, pad, dtype=mixed.dtype, device=mixed.device
            )
            mixed = torch.cat([mixed, tail], dim=-1)
        return torch.fft.irfft(mixed, n=length, dim=-1)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.in_channels}, {self.out_channels}, n_modes={self.n_modes}"


class FourierBlock1d(nn.Module):
    """One FNO block: spectral branch + pointwise branch, normalized and gated.

    .. math::
        u \\mapsto \\sigma\\big(\\mathrm{Norm}(\\mathcal{K}u + W u)\\big) + u

    The pointwise ($1\\times1$ convolution) branch supplies the local part that mode
    truncation removes; the residual connection keeps deep stacks trainable.

    Args:
        channels: Channel width (input and output).
        n_modes: Retained Fourier modes.
        activation: Nonlinearity applied after normalization.
        dropout: Dropout probability applied to the block output.
    """

    def __init__(
        self,
        channels: int,
        n_modes: int,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, n_modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        h = self.spectral(x) + self.pointwise(x)
        return x + self.dropout(self.act(self.norm(h)))


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding $x \\mapsto [\\sin(2\\pi B x), \\cos(2\\pi B x)]$.

    Args:
        in_features: Input dimension.
        num_frequencies: Number of random projections; the output has
            ``2 * num_frequencies`` channels.
        scale: Standard deviation of the projection matrix $B$. This is the bandwidth
            knob: the baseline script used ``scale=10`` on raw, unbounded coordinates
            $p \\in [0, 11]$, which aliases badly. Inputs here are pre-normalized
            logarithmic coordinates of order one, so a scale of order one is correct.
        trainable: Let $B$ receive gradients instead of staying a fixed buffer.

    Shape:
        - Input: ``(..., in_features)``
        - Output: ``(..., 2 * num_frequencies)``
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 16,
        scale: float = 1.5,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        matrix = torch.randn(in_features, num_frequencies) * scale
        if trainable:
            self.projection = nn.Parameter(matrix)
        else:
            self.register_buffer("projection", matrix)

    @property
    def out_features(self) -> int:
        """Dimension of the embedding."""
        return 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        projected = 2.0 * math.pi * (x @ self.projection)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class MetricPositionalEncoding(nn.Module):
    """Embed the AdS2 conformal factor $\\sqrt{g} = L^2/z^2$ into boundary coordinates.

    Boundary insertions live at $z = 0$, so "the metric at the query point" is not
    directly meaningful. What *is* meaningful is the holographic scale-radius
    correspondence: the bulk region dominating $W(p_1, p_2)$ sits at a radial depth set
    by the separation, $z_\\star \\sim r/2$ (equivalently, by the RG scale, $z_\\star =
    1/M$). Evaluating $\\log\\sqrt{g}(z_\\star) = 2\\log L - 2\\log z_\\star$ therefore hands
    the trunk the conformal weight of the diagram it is being asked to evaluate, and
    makes the correct $\\log r$ scaling a *linear* feature of the encoding rather than
    something the MLP must discover.

    The invariant coordinates emitted are

    .. math::
        \\left(\\log r,\\, \\log\\sqrt{g}(z_\\star),\\, \\log M\\right)

    optionally augmented by the midpoint $\\bar{p} = (p_1 + p_2)/2$. With
    ``translation_invariant=True`` (the default) the midpoint is dropped entirely, so
    boundary translation invariance -- a symmetry of the exact correlator that the
    baseline's $(p_1, p_2)$ trunk had to learn from data -- holds *by construction*
    rather than approximately. In exact arithmetic the invariance is perfect; in float32
    it is limited only by the cancellation in ``p1 - p2`` for large common shifts. See
    :mod:`tests.test_models`.

    Args:
        config: Physics configuration supplying $L$.
        num_frequencies: Random Fourier features applied to the invariant coordinates.
        fourier_scale: Bandwidth of those features.
        radial_mode: ``"separation"`` sets $z_\\star = r/2$; ``"rg_scale"`` sets
            $z_\\star = 1/M$; ``"geometric"`` uses $\\sqrt{r/(2M)}$, interpolating between
            the two.
        translation_invariant: Drop the midpoint coordinate.
        log_r_scale: Divisor normalizing $\\log r$ into an order-one range.

    Shape:
        - Input: ``coords`` ``(batch, points, 2)``, ``log_m`` ``(batch, points)``
        - Output: ``(batch, points, out_features)``
    """

    _RADIAL_MODES = ("separation", "rg_scale", "geometric")

    def __init__(
        self,
        config: PhysicsConfig,
        num_frequencies: int = 16,
        fourier_scale: float = 1.5,
        radial_mode: str = "separation",
        translation_invariant: bool = True,
        log_r_scale: float = 3.0,
    ) -> None:
        super().__init__()
        if radial_mode not in self._RADIAL_MODES:
            raise ValueError(
                f"radial_mode must be one of {self._RADIAL_MODES}, got {radial_mode!r}"
            )
        if log_r_scale <= 0.0:
            raise ValueError("log_r_scale must be positive")
        self.config = config
        self.radial_mode = radial_mode
        self.translation_invariant = translation_invariant
        self.log_r_scale = log_r_scale
        self.num_invariants = 3 if translation_invariant else 4
        self.fourier = FourierFeatures(self.num_invariants, num_frequencies, fourier_scale)

    @property
    def out_features(self) -> int:
        """Dimension of the encoded output."""
        return self.num_invariants + self.fourier.out_features

    def holographic_depth(self, log_r: Tensor, log_m: Tensor) -> Tensor:
        """Radial depth $\\log z_\\star$ probed by a boundary pair, per ``radial_mode``."""
        if self.radial_mode == "separation":
            return log_r - math.log(2.0)
        if self.radial_mode == "rg_scale":
            return -log_m
        return 0.5 * (log_r - math.log(2.0) - log_m)

    def forward(self, coords: Tensor, log_m: Tensor | None = None) -> Tensor:  # noqa: D102
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must have a trailing dim of 2, got {tuple(coords.shape)}")
        p1, p2 = coords[..., 0], coords[..., 1]
        log_r = torch.log((p1 - p2).abs().clamp_min(1e-12))
        log_m_t = torch.zeros_like(log_r) if log_m is None else log_m.expand_as(log_r)

        log_z = self.holographic_depth(log_r, log_m_t)
        log_sqrt_g = 2.0 * math.log(self.config.L) - 2.0 * log_z

        invariants = [log_r / self.log_r_scale, log_sqrt_g / (2.0 * self.log_r_scale), log_m_t]
        if not self.translation_invariant:
            invariants.append(0.5 * (p1 + p2) / self.log_r_scale)
        stacked = torch.stack(invariants, dim=-1)
        return torch.cat([stacked, self.fourier(stacked)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation $h \\mapsto (1 + \\alpha(c))\\,h + \\beta(c)$.

    This is how the branch conditions the trunk. A plain DeepONet couples the two
    networks only through a final inner product; FiLM lets the potential steer the
    coordinate features at every depth, which is the Fourier-DeepONet coupling of Zhu
    et al. and matters here because the potential changes the *exponent* of the
    correlator, not merely its amplitude.

    Args:
        latent_dim: Width of the conditioning code.
        feature_dim: Width of the features being modulated.

    Shape:
        - ``features``: ``(batch, points, feature_dim)``
        - ``code``: ``(batch, latent_dim)``
        - Output: ``(batch, points, feature_dim)``
    """

    def __init__(self, latent_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(latent_dim, 2 * feature_dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, features: Tensor, code: Tensor) -> Tensor:  # noqa: D102
        scale, shift = self.to_scale_shift(code).chunk(2, dim=-1)
        return features * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """Pre-normalized feed-forward stack used by both the branch and the trunk.

    Args:
        in_dim: Input width.
        hidden_dims: Hidden widths, in order.
        out_dim: Output width.
        activation: Nonlinearity class.
        dropout: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), activation()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.net(x)
