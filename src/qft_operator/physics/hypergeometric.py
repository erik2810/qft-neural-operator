"""Autograd-aware Gauss hypergeometric function ${}_2F_1$ backed by SciPy.

PyTorch has no native ${}_2F_1$; the bulk-to-bulk propagator needs one. Only the
argument $z$ carries gradients -- the parameters $(a, b, c)$ are treated as constants,
which is all the AdS2 propagator requires.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["hyp2f1"]


class _Hyp2F1(torch.autograd.Function):
    """``torch.autograd.Function`` wrapping :func:`scipy.special.hyp2f1`.

    Uses $\\frac{d}{dz}\\,{}_2F_1(a,b;c;z) = \\frac{ab}{c}\\,{}_2F_1(a+1,b+1;c+1;z)$.
    """

    @staticmethod
    def forward(ctx: Any, a: float, b: float, c: float, z: Tensor) -> Tensor:  # noqa: D102
        from scipy.special import hyp2f1 as _scipy_hyp2f1

        ctx.abc = (a, b, c)
        ctx.save_for_backward(z)
        values = _scipy_hyp2f1(a, b, c, z.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[None, None, None, Tensor]:  # noqa: D102
        from scipy.special import hyp2f1 as _scipy_hyp2f1

        (z,) = ctx.saved_tensors
        a, b, c = ctx.abc
        d_values = (a * b / c) * _scipy_hyp2f1(a + 1.0, b + 1.0, c + 1.0, z.detach().cpu().numpy())
        d_tensor = torch.as_tensor(d_values, dtype=z.dtype, device=z.device)
        return None, None, None, grad_output * d_tensor


def hyp2f1(a: float, b: float, c: float, z: Tensor) -> Tensor:
    """Evaluate ${}_2F_1(a, b; c; z)$ with autograd support on ``z``.

    Args:
        a: First parameter.
        b: Second parameter.
        c: Third parameter; must not be a non-positive integer.
        z: Argument tensor.

    Returns:
        Tensor of the same shape, dtype and device as ``z``.
    """
    return _Hyp2F1.apply(a, b, c, z)  # type: ignore[no-any-return]
