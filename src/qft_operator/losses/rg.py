"""Renormalization-group invariance loss (Callan-Symanzik residual)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from qft_operator.losses.operators import DirectionalDerivative
from qft_operator.physics.rg import BetaFunction

__all__ = ["RGInvarianceLoss"]


class RGInvarianceLoss(nn.Module):
    """Penalize violation of the Callan-Symanzik condition
    $\\left(M\\partial_M + \\beta(\\lambda)\\partial_\\lambda\\right) W = 0$.

    The renormalization scale $M$ is an artefact of the subtraction scheme -- in the
    holographic dictionary it is the inverse of the near-boundary cutoff, $\\epsilon =
    1/M$ -- so no physical correlator may depend on it once the coupling is allowed to
    run. Imposing that directly is a strong, label-free constraint: it ties together
    predictions the network makes for *different inputs* $(V, M)$, which no amount of
    pointwise supervision does.

    Evaluating the residual
    ---------------------
    Both derivatives are assembled as a single directional derivative. Since $W > 0$,
    the operator annihilates $W$ exactly when it annihilates $\\log W$, so the network's
    native log output is used directly. The tangent directions are

    * $M\\partial_M \\,\\to\\, \\partial_{\\log M}$, i.e. a tangent of ones on the $\\log M$
      input;
    * $\\beta(\\lambda)\\partial_\\lambda \\,\\to\\,$ a tangent of
      $\\beta(\\lambda)\\,\\partial V/\\partial \\lambda$ on the branch input, using the
      **exact, closed-form** $\\partial V/\\partial \\lambda = v(\\phi)$ that every
      :class:`~qft_operator.physics.potentials.Potential` supplies.

    One forward-mode pass therefore yields the full per-point residual.

    Consistency with the data
    -------------------------
    The ``"resummed"`` and ``"hybrid"`` dataset modes build targets from the coupling at
    the physical scale $\\bar\\lambda(1/r)$, which is $M$-independent by the group property
    of the flow. Those targets annihilate the Callan-Symanzik operator *exactly*, so
    this loss and the data term never pull against each other. The fixed-order
    ``"quadrature"`` mode does not have that property -- its shipped config leaves this
    weight at zero.

    Args:
        beta: The :class:`~qft_operator.physics.rg.BetaFunction` to insert.
        mode: ``"jvp"`` for forward-mode AD or ``"fd"`` for central differences.
        step: Finite-difference step, used only in ``"fd"`` mode.
        normalize: Divide the residual by $1 + |\\log W|$ before squaring, so that large
            separations (where $|\\log W|$ is large) do not dominate.

    Shape:
        - Output: scalar
    """

    def __init__(
        self,
        beta: BetaFunction | None = None,
        mode: str = "jvp",
        step: float = 1e-3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta or BetaFunction()
        self.derivative = DirectionalDerivative(mode=mode, step=step)  # type: ignore[arg-type]
        self.normalize = normalize

    def forward(
        self,
        model: nn.Module,
        v_phi: Tensor,
        coords: Tensor,
        log_m: Tensor,
        coupling: Tensor,
        dv_dcoupling: Tensor,
        log_w: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the mean squared Callan-Symanzik residual.

        Args:
            model: Operator network returning $\\log W$ of shape ``(batch, points)``.
            v_phi: Potential samples, shape ``(batch, n_phi)``.
            coords: Boundary pairs, shape ``(batch, points, 2)``.
            log_m: $\\log M$ per sample, shape ``(batch, 1)``.
            coupling: $\\lambda(M)$ per sample, shape ``(batch,)`` or ``(batch, 1)``.
            dv_dcoupling: Exact $\\partial V / \\partial \\lambda$ on the field grid,
                shape ``(batch, n_phi)``.
            log_w: Already-computed $\\log W$ at these inputs, used only for the
                normalization scale. Passing it in saves a redundant forward pass per
                step; ``None`` recomputes it.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If ``dv_dcoupling`` does not match ``v_phi`` in shape.
        """
        if dv_dcoupling.shape != v_phi.shape:
            raise ValueError(
                f"dv_dcoupling {tuple(dv_dcoupling.shape)} must match v_phi {tuple(v_phi.shape)}"
            )
        beta_value = self.beta(coupling.reshape(-1, 1))
        tangent_v = beta_value * dv_dcoupling
        tangent_m = torch.ones_like(log_m)

        def evaluate(v: Tensor, m: Tensor) -> Tensor:
            return model(v, coords, m)

        residual = self.derivative(evaluate, (v_phi, log_m), (tangent_v, tangent_m))
        if self.normalize:
            with torch.no_grad():
                reference = model(v_phi, coords, log_m) if log_w is None else log_w
                scale = 1.0 + reference.abs()
            residual = residual / scale
        return (residual**2).mean()
