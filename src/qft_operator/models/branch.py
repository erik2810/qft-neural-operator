"""Branch network: discretized interaction potential $V(\\phi)$ to a latent code."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from qft_operator.models.layers import MLP, FourierBlock1d

__all__ = ["BranchNet"]


class BranchNet(nn.Module):
    """Encode a potential sampled on a field grid into a latent operator code.

    The potential arrives as $V(\\phi_i)$ on a fixed grid. A pointwise MLP over those
    samples (the baseline's design) sees the grid as an unstructured feature vector and
    is tied to one discretization. Lifting to channels and applying spectral
    convolutions along $\\phi$ instead gives a genuine *function-space* encoder: the
    first block already has a global receptive field over field space, and mode
    truncation makes the representation transferable across grid resolutions.

    In addition to the pooled code, the branch exposes a short sequence of tokens for
    the cross-attention head of :class:`~qft_operator.models.deeponet.FourierDeepONet`,
    so query points can attend to different regions of field space.

    Args:
        n_phi: Number of field-grid samples (used only for the non-spectral path).
        latent_dim: Width of the emitted code.
        width: Channel width of the spectral stack.
        n_blocks: Number of :class:`~qft_operator.models.layers.FourierBlock1d` layers.
        n_modes: Retained Fourier modes per block.
        hidden_dims: Hidden widths of the projection head.
        n_tokens: Number of tokens exposed for cross-attention.
        emit_tokens: Build and return the token sequence. The inner-product head does
            not consume it, and leaving the projection out keeps the module free of
            parameters that receive no gradient -- which would otherwise force
            ``find_unused_parameters=True`` under DDP.
        dropout: Dropout probability inside the blocks and the head.
        use_spectral: Set ``False`` to fall back to a plain MLP over the raw grid, which
            reproduces the baseline branch and serves as the architecture ablation.

    Shape:
        - Input: ``(batch, n_phi)``
        - ``code``: ``(batch, latent_dim)``
        - ``tokens``: ``(batch, n_tokens, latent_dim)``, or ``None``
    """

    #: Normalized field grid; declared so the registered buffer keeps its Tensor type.
    phi_grid: Tensor
    #: Token projection, present only when ``emit_tokens`` is set.
    token_proj: nn.Linear | None

    def __init__(
        self,
        n_phi: int,
        latent_dim: int = 128,
        width: int = 64,
        n_blocks: int = 4,
        n_modes: int = 16,
        hidden_dims: list[int] | None = None,
        n_tokens: int = 8,
        emit_tokens: bool = True,
        dropout: float = 0.0,
        use_spectral: bool = True,
    ) -> None:
        super().__init__()
        if n_phi < 4:
            raise ValueError(f"n_phi must be at least 4, got {n_phi}")
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be positive, got {n_tokens}")
        self.n_phi = n_phi
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.emit_tokens = bool(emit_tokens)
        self.use_spectral = use_spectral
        hidden = hidden_dims if hidden_dims is not None else [256, 256]

        if use_spectral:
            # Two input channels: the potential itself and the field coordinate, so the
            # encoder knows *where* on the phi grid each sample sits.
            self.lift = nn.Conv1d(2, width, kernel_size=1)
            self.blocks = nn.ModuleList(
                [FourierBlock1d(width, n_modes, dropout=dropout) for _ in range(n_blocks)]
            )
            self.head = MLP(2 * width, hidden, latent_dim, dropout=dropout)
            self.token_proj = nn.Linear(width, latent_dim) if self.emit_tokens else None
        else:
            self.lift = nn.Identity()  # type: ignore[assignment]
            self.blocks = nn.ModuleList()
            self.head = MLP(n_phi, hidden, latent_dim, dropout=dropout)
            self.token_proj = nn.Linear(latent_dim, latent_dim) if self.emit_tokens else None

        self.register_buffer("phi_grid", torch.linspace(-1.0, 1.0, n_phi))

    def forward(self, v_phi: Tensor) -> tuple[Tensor, Tensor | None]:
        """Encode a batch of potentials.

        Args:
            v_phi: Potential samples, shape ``(batch, n_phi)``.

        Returns:
            ``(code, tokens)`` with shapes ``(batch, latent_dim)`` and
            ``(batch, n_tokens, latent_dim)``; ``tokens`` is ``None`` when
            ``emit_tokens=False``.

        Raises:
            ValueError: If ``v_phi`` is not 2-D.
        """
        if v_phi.ndim != 2:
            raise ValueError(f"v_phi must be (batch, n_phi), got {tuple(v_phi.shape)}")

        if not self.use_spectral:
            code = self.head(v_phi)
            if self.token_proj is None:
                return code, None
            return code, self.token_proj(code).unsqueeze(1).expand(-1, self.n_tokens, -1)

        batch, length = v_phi.shape
        grid = self.phi_grid.to(v_phi.dtype)
        if int(grid.shape[0]) != length:
            # Resolution transfer: resample the coordinate channel to the incoming grid.
            grid = torch.linspace(-1.0, 1.0, length, dtype=v_phi.dtype, device=v_phi.device)
        stacked = torch.stack([v_phi, grid.expand(batch, length)], dim=1)  # (B, 2, N)

        h = self.lift(stacked)
        for block in self.blocks:
            h = block(h)

        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        code = self.head(pooled)
        if self.token_proj is None:
            return code, None

        tokens = torch.nn.functional.adaptive_avg_pool1d(h, self.n_tokens)  # (B, W, T)
        return code, self.token_proj(tokens.transpose(1, 2))
