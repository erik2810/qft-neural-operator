"""Trunk network: boundary coordinate pairs to a latent basis, conditioned on $V$."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from qft_operator.models.layers import FiLM, FourierBlock1d, MetricPositionalEncoding
from qft_operator.physics.config import PhysicsConfig

__all__ = ["TrunkNet", "BoundaryContextField", "interp1d_uniform", "default_log_r_range"]


def interp1d_uniform(values: Tensor, lo: float, hi: float, query: Tensor) -> Tensor:
    """Differentiable linear interpolation of a channel field on a uniform 1-D grid.

    Args:
        values: Field samples, shape ``(batch, channels, grid)``.
        lo: Coordinate of the first grid node.
        hi: Coordinate of the last grid node.
        query: Query coordinates, shape ``(batch, points)``; clamped to ``[lo, hi]``.

    Returns:
        Interpolated values of shape ``(batch, points, channels)``.

    Raises:
        ValueError: If the grid has fewer than two nodes or ``hi <= lo``.
    """
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError(f"values must be (batch, channels, grid>=2), got {tuple(values.shape)}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
    grid_size = values.shape[-1]
    spacing = (hi - lo) / (grid_size - 1)
    position = ((query.clamp(lo, hi) - lo) / spacing).clamp(0.0, float(grid_size - 1))
    # At the top edge left saturates to the last node and the weight is zero, so the
    # upper gather (clamped below) returns the same value and the result is exact.
    left = position.floor().long().clamp_max(grid_size - 1)
    weight = (position - left.to(position.dtype)).unsqueeze(-1)
    field = values.transpose(1, 2)  # (B, G, C)
    index = left.unsqueeze(-1).expand(-1, -1, field.shape[-1])
    lower = torch.gather(field, 1, index)
    upper = torch.gather(field, 1, (index + 1).clamp_max(grid_size - 1))
    return torch.lerp(lower, upper, weight)


class BoundaryContextField(nn.Module):
    """Non-local features along the boundary direction, on an internal reference grid.

    Putting FNO blocks directly on a DeepONet's *query* axis would be a modelling error:
    it makes $W(p_1, p_2)$ depend on which other separations happen to sit in the same
    batch, which the exact correlator certainly does not. Instead this module carries a
    fixed, log-uniform internal grid of separations
    $\\log r \\in [\\log r_{\\min}, \\log r_{\\max}]$, runs the spectral stack there --
    conditioned on the branch code and on $\\log M$ -- and lets each query point read the
    resulting field off by linear interpolation at its own $\\log r$.

    The result is genuinely non-local along $p$ (every query sees a representation built
    from the whole boundary profile) while keeping the prediction a well-defined function
    of the query's own coordinates. That property is what preserves the diagonal Jacobian
    the physics losses in :mod:`qft_operator.losses.operators` rely on.

    Args:
        latent_dim: Width of the conditioning code.
        width: Channel width of the field.
        grid_size: Number of internal reference separations.
        n_blocks: Number of spectral blocks.
        n_modes: Retained Fourier modes.
        log_r_min: Lower end of the internal grid, in $\\log r$.
        log_r_max: Upper end of the internal grid, in $\\log r$.
        dropout: Dropout probability.

    Shape:
        - ``log_r``: ``(batch, points)``
        - ``code``: ``(batch, latent_dim)``
        - Output: ``(batch, points, width)``
    """

    #: Reference grid of log-separations; declared so the buffer keeps its Tensor type.
    grid: Tensor

    def __init__(
        self,
        latent_dim: int,
        width: int = 64,
        grid_size: int = 64,
        n_blocks: int = 2,
        n_modes: int = 16,
        log_r_min: float = -3.5,
        log_r_max: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if grid_size < 4:
            raise ValueError(f"grid_size must be at least 4, got {grid_size}")
        if log_r_max <= log_r_min:
            raise ValueError("log_r_max must exceed log_r_min")
        self.log_r_min = log_r_min
        self.log_r_max = log_r_max
        self.width = width
        self.register_buffer("grid", torch.linspace(log_r_min, log_r_max, grid_size))
        # Two scalar channels seed the field: the reference log r and log M.
        self.seed = nn.Conv1d(2, width, kernel_size=1)
        self.film = FiLM(latent_dim, width)
        self.blocks = nn.ModuleList(
            [FourierBlock1d(width, n_modes, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, log_r: Tensor, code: Tensor, log_m: Tensor) -> Tensor:  # noqa: D102
        batch, grid_size = code.shape[0], int(self.grid.shape[0])
        grid = self.grid.to(dtype=code.dtype, device=code.device).expand(batch, grid_size)
        scale = 0.5 * (self.log_r_max - self.log_r_min)
        channels = torch.stack([grid / scale, log_m.expand(batch, grid_size)], dim=1)

        field = self.seed(channels)
        field = self.film(field.transpose(1, 2), code).transpose(1, 2)
        for block in self.blocks:
            field = block(field)
        return interp1d_uniform(field, self.log_r_min, self.log_r_max, log_r)


class TrunkNet(nn.Module):
    """Evaluate a latent basis at boundary point pairs $(p_1, p_2)$.

    Structure, in order:

    1. :class:`~qft_operator.models.layers.MetricPositionalEncoding` converts raw
       coordinates into conformal invariants and embeds $\\log\\sqrt{g} = 2\\log L -
       2\\log z_\\star$;
    2. optional :class:`BoundaryContextField`, supplying non-local features along the
       boundary direction $p$;
    3. FiLM-conditioned residual layers, so the potential steers the coordinate features
       at every depth rather than only through a final inner product.

    Args:
        config: Physics configuration, forwarded to the positional encoding.
        latent_dim: Width of the emitted basis.
        width: Hidden width.
        n_layers: Number of FiLM-conditioned residual layers.
        num_frequencies: Fourier features in the positional encoding.
        fourier_scale: Bandwidth of those features.
        radial_mode: Holographic depth convention; see
            :class:`~qft_operator.models.layers.MetricPositionalEncoding`.
        translation_invariant: Enforce boundary translation invariance structurally.
        spectral_mixing: Enable the :class:`BoundaryContextField`.
        n_modes: Retained modes in the context field.
        context_grid: Internal reference grid size of the context field.
        context_width: Channel width of the context field.
        log_r_range: ``(min, max)`` extent of the internal grid in $\\log r$.
        dropout: Dropout probability.

    Shape:
        - ``coords``: ``(batch, points, 2)``
        - ``code``: ``(batch, latent_dim)``
        - Output: ``(batch, points, latent_dim)``
    """

    def __init__(
        self,
        config: PhysicsConfig,
        latent_dim: int = 128,
        width: int = 128,
        n_layers: int = 4,
        num_frequencies: int = 16,
        fourier_scale: float = 1.5,
        radial_mode: str = "separation",
        translation_invariant: bool = True,
        spectral_mixing: bool = True,
        n_modes: int = 16,
        context_grid: int = 64,
        context_width: int = 64,
        log_r_range: tuple[float, float] = (-3.5, 3.0),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoding = MetricPositionalEncoding(
            config,
            num_frequencies=num_frequencies,
            fourier_scale=fourier_scale,
            radial_mode=radial_mode,
            translation_invariant=translation_invariant,
        )
        self.spectral_mixing = spectral_mixing
        self.context = (
            BoundaryContextField(
                latent_dim=latent_dim,
                width=context_width,
                grid_size=context_grid,
                n_modes=n_modes,
                log_r_min=log_r_range[0],
                log_r_max=log_r_range[1],
                dropout=dropout,
            )
            if spectral_mixing
            else None
        )
        in_features = self.encoding.out_features + (context_width if spectral_mixing else 0)
        self.lift = nn.Linear(in_features, width)
        self.films = nn.ModuleList([FiLM(latent_dim, width) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(width, width),
                )
                for _ in range(n_layers)
            ]
        )
        self.project = nn.Linear(width, latent_dim)

    def forward(self, coords: Tensor, code: Tensor, log_m: Tensor | None = None) -> Tensor:
        """Evaluate the conditioned basis.

        Args:
            coords: Boundary pairs, shape ``(batch, points, 2)``.
            code: Branch latent code, shape ``(batch, latent_dim)``.
            log_m: $\\log M$, shape ``(batch, 1)`` or ``(batch, points)``. Defaults to
                zero, i.e. $M = 1$.

        Returns:
            Basis values of shape ``(batch, points, latent_dim)``.
        """
        features = self.encoding(coords, log_m)
        if self.context is not None:
            log_r = torch.log((coords[..., 0] - coords[..., 1]).abs().clamp_min(1e-12))
            scale_input = (
                torch.zeros(code.shape[0], 1, dtype=code.dtype, device=code.device)
                if log_m is None
                else log_m.reshape(code.shape[0], -1)[:, :1]
            )
            features = torch.cat([features, self.context(log_r, code, scale_input)], dim=-1)

        h = self.lift(features)
        for film, layer in zip(self.films, self.layers, strict=True):
            h = h + layer(film(h, code))
        return self.project(h)


def default_log_r_range(r_min: float, r_max: float) -> tuple[float, float]:
    """Convert a separation window into a padded $\\log r$ range for the context grid.

    Args:
        r_min: Smallest separation the dataset will produce.
        r_max: Largest separation the dataset will produce.

    Returns:
        ``(log_r_min, log_r_max)`` widened by half a decade on each side so that queries
        never land on the clamped boundary of the interpolation grid.
    """
    if not 0.0 < r_min < r_max:
        raise ValueError(f"need 0 < r_min < r_max, got {r_min}, {r_max}")
    pad = 0.5 * math.log(10.0)
    return math.log(r_min) - pad, math.log(r_max) + pad
