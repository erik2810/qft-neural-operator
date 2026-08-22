"""Publication-oriented plots: correlators, residual fits, spectra, RG flow.

Every function returns a :class:`matplotlib.figure.Figure` and never calls ``show`` or
``savefig``, so the same code serves notebooks, the CLI and the test suite. Matplotlib
and seaborn are optional dependencies (``pip install 'qft-neural-operator[viz]'``); the
import error names the extra rather than failing obscurely deep inside a plotting call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from qft_operator.analysis.spectrum import fit_log_slope

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = [
    "set_style",
    "plot_correlator_comparison",
    "plot_log_residuals",
    "plot_anomalous_spectrum",
    "plot_potential_gallery",
    "plot_rg_flow",
    "plot_bulk_integral_convergence",
]

_PALETTE = ("#1f4e79", "#c1440e", "#2e7d32", "#6a1b9a", "#b8860b")


def _pyplot() -> Any:
    """Import pyplot, or raise an error that names the missing extra."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting requires the 'viz' extra: pip install 'qft-neural-operator[viz]'"
        ) from exc
    return plt


def set_style(context: str = "paper", grid: bool = True) -> None:
    """Apply a consistent seaborn/matplotlib style.

    Args:
        context: Seaborn context (``"paper"``, ``"talk"``, ``"poster"``).
        grid: Draw a light background grid.

    Note:
        Falls back to plain matplotlib settings if seaborn is not installed, so the
        plotting functions never hard-depend on it.
    """
    plt = _pyplot()
    try:
        import seaborn as sns

        sns.set_theme(context=context, style="whitegrid" if grid else "white")
    except ImportError:  # pragma: no cover
        plt.rcParams.update({"axes.grid": grid, "grid.alpha": 0.3})
    plt.rcParams.update(
        {"figure.dpi": 120, "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=_PALETTE)}
    )


def _to_numpy(tensor: Tensor) -> Any:
    """Detach a tensor to a CPU NumPy array."""
    return tensor.detach().cpu().numpy()


def plot_correlator_comparison(
    log_r: Tensor,
    log_w_exact: Tensor,
    log_w_pred: Tensor,
    labels: list[str] | None = None,
    title: str = "Boundary connected correlator",
) -> Figure:
    """Overlay predicted and exact $W$ on log-log axes.

    Args:
        log_r: $\\log r$, shape ``(n_curves, points)``.
        log_w_exact: Exact $\\log W$, same shape.
        log_w_pred: Predicted $\\log W$, same shape.
        labels: One legend entry per curve.
        title: Axes title.

    Returns:
        The figure.

    Raises:
        ValueError: If the three arrays disagree in shape.
    """
    if not log_r.shape == log_w_exact.shape == log_w_pred.shape:
        raise ValueError("log_r, log_w_exact and log_w_pred must share a shape")
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    names = labels or [f"theory {i}" for i in range(log_r.shape[0])]
    for index, name in enumerate(names):
        colour = _PALETTE[index % len(_PALETTE)]
        axes.plot(
            _to_numpy(torch.exp(log_r[index])),
            _to_numpy(torch.exp(log_w_exact[index])),
            color=colour,
            lw=1.6,
            label=f"{name} (exact)",
        )
        axes.plot(
            _to_numpy(torch.exp(log_r[index])),
            _to_numpy(torch.exp(log_w_pred[index])),
            color=colour,
            ls="--",
            lw=1.4,
            label=f"{name} (operator)",
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(r"$|p_1 - p_2|$")
    axes.set_ylabel(r"$W(p_1, p_2)$")
    axes.set_title(title)
    axes.legend(fontsize=7, ncol=2)
    return figure


def plot_log_residuals(
    log_r: Tensor,
    log_w_exact: Tensor,
    log_w_pred: Tensor,
    free_dimension: float,
) -> Figure:
    """Logarithmic residual plot with fitted slopes.

    The upper panel strips the free-theory power law, leaving
    $\\log W + 2\\Delta\\beta_1\\beta_2\\log r$ -- a straight line whose slope is $2\\gamma$.
    Plotting it this way is the whole point: the anomalous dimension is a $10^{-3}$
    effect that is completely invisible on a raw log-log correlator plot, and only shows
    up once the leading behaviour is divided out. The lower panel shows the prediction
    error in $\\log W$.

    Args:
        log_r: $\\log r$, shape ``(n_curves, points)``.
        log_w_exact: Exact $\\log W$, same shape.
        log_w_pred: Predicted $\\log W$, same shape.
        free_dimension: $\\Delta\\beta_1\\beta_2$.

    Returns:
        The figure.
    """
    plt = _pyplot()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.0, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    stripped_exact = log_w_exact + 2.0 * free_dimension * log_r
    stripped_pred = log_w_pred + 2.0 * free_dimension * log_r
    slope_exact, _ = fit_log_slope(log_r, stripped_exact)
    slope_pred, _ = fit_log_slope(log_r, stripped_pred)

    for index in range(log_r.shape[0]):
        colour = _PALETTE[index % len(_PALETTE)]
        x = _to_numpy(log_r[index])
        top.plot(
            x,
            _to_numpy(stripped_exact[index]),
            color=colour,
            lw=1.6,
            label=rf"exact $2\gamma={float(slope_exact[index]):+.4f}$",
        )
        top.plot(
            x,
            _to_numpy(stripped_pred[index]),
            color=colour,
            ls="--",
            lw=1.4,
            label=rf"operator $2\gamma={float(slope_pred[index]):+.4f}$",
        )
        bottom.plot(x, _to_numpy(log_w_pred[index] - log_w_exact[index]), color=colour, lw=1.2)

    top.set_ylabel(r"$\log W + 2\Delta\beta_1\beta_2 \log r$")
    top.set_title("Logarithmic residual and anomalous-dimension fit")
    top.legend(fontsize=7, ncol=2)
    bottom.axhline(0.0, color="0.4", lw=0.8)
    bottom.set_xlabel(r"$\log |p_1 - p_2|$")
    bottom.set_ylabel(r"$\Delta \log W$")
    return figure


def plot_anomalous_spectrum(
    gamma_exact: Tensor,
    gamma_pred: Tensor,
    family: Tensor | None = None,
    family_names: tuple[str, ...] = (),
) -> Figure:
    """Predicted against exact anomalous dimensions, coloured by potential family.

    Args:
        gamma_exact: Exact $\\gamma$, shape ``(n,)``.
        gamma_pred: Predicted $\\gamma$, same shape.
        family: Optional family indices, same shape.
        family_names: Names indexed by ``family``.

    Returns:
        The figure.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(5.0, 4.8))
    if family is None or not family_names:
        axes.scatter(_to_numpy(gamma_exact), _to_numpy(gamma_pred), s=12, alpha=0.7)
    else:
        for index, name in enumerate(family_names):
            mask = family == index
            if not bool(mask.any()):
                continue
            axes.scatter(
                _to_numpy(gamma_exact[mask]),
                _to_numpy(gamma_pred[mask]),
                s=14,
                alpha=0.75,
                label=name,
                color=_PALETTE[index % len(_PALETTE)],
            )
        axes.legend(fontsize=8, title="family")
    span = [float(gamma_exact.min()), float(gamma_exact.max())]
    axes.plot(span, span, color="0.3", lw=1.0, ls=":", zorder=0)
    axes.set_xlabel(r"exact $\gamma$")
    axes.set_ylabel(r"recovered $\gamma$")
    axes.set_title("Anomalous dimension spectrum")
    axes.set_aspect("equal", adjustable="datalim")
    return figure


def plot_potential_gallery(phi: Tensor, potentials: list[Any], max_curves: int = 8) -> Figure:
    """Show a sample of the interaction potentials the operator is trained on.

    Args:
        phi: Field grid, shape ``(n_phi,)``.
        potentials: :class:`~qft_operator.physics.potentials.Potential` instances.
        max_curves: Cap on the number of curves drawn.

    Returns:
        The figure.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for index, potential in enumerate(potentials[:max_curves]):
        colour = _PALETTE[index % len(_PALETTE)]
        left.plot(
            _to_numpy(phi),
            _to_numpy(potential.evaluate(phi)),
            color=colour,
            lw=1.3,
            label=potential.family,
        )
        right.plot(
            _to_numpy(phi), _to_numpy(potential.second_derivative(phi)), color=colour, lw=1.3
        )
    left.set_xlabel(r"$\phi$")
    left.set_ylabel(r"$V(\phi)$")
    left.set_title("Interaction potentials")
    left.legend(fontsize=7)
    right.set_xlabel(r"$\phi$")
    right.set_ylabel(r"$V''(\phi)$")
    right.set_title(r"$V''$ -- the source of $\gamma$")
    return figure


def plot_rg_flow(beta: Any, lam_max: float = 0.1, n_points: int = 200) -> Figure:
    """Plot $\\beta(\\lambda)$ and the coupling's trajectory in $\\log\\mu$.

    Args:
        beta: A :class:`~qft_operator.physics.rg.BetaFunction`.
        lam_max: Half-width of the coupling window.
        n_points: Resolution.

    Returns:
        The figure.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    lam = torch.linspace(-lam_max, lam_max, n_points, dtype=torch.float64)
    left.plot(_to_numpy(lam), _to_numpy(beta(lam)), color=_PALETTE[0], lw=1.6)
    left.axhline(0.0, color="0.4", lw=0.8)
    left.set_xlabel(r"$\lambda$")
    left.set_ylabel(r"$\beta(\lambda)$")
    left.set_title(rf"$\epsilon={beta.config.epsilon}$, $b={beta.config.two_loop}$")

    d_log_mu = torch.linspace(-4.0, 4.0, n_points, dtype=torch.float64)
    for index, start in enumerate((0.01, 0.03, 0.05)):
        trajectory = beta.run(torch.full_like(d_log_mu, start), d_log_mu)
        right.plot(
            _to_numpy(d_log_mu),
            _to_numpy(trajectory),
            color=_PALETTE[index % len(_PALETTE)],
            lw=1.4,
            label=rf"$\lambda(M)={start}$",
        )
    right.set_xlabel(r"$\log(\mu / M)$")
    right.set_ylabel(r"$\bar{\lambda}(\mu)$")
    right.set_title("Running coupling")
    right.legend(fontsize=8)
    return figure


def plot_bulk_integral_convergence(
    integrator: Any,
    radii: Tensor | None = None,
    eps_values: Tensor | None = None,
) -> Figure:
    """Reduced contact integral against $\\log(1/\\epsilon)$, with the analytic slope.

    Shows the claim that underpins the whole dataset: the regulated bulk integral grows
    linearly in $\\log(1/\\epsilon)$ with slope $C_{\\log} = 2L^2 c_\\Delta$, identically for
    every separation.

    Args:
        integrator: A :class:`~qft_operator.physics.bulk_integrals.ConformalIntegrator`.
        radii: Separations to scan; defaults to four decade-spaced values.
        eps_values: Cutoffs to scan; defaults to eight log-spaced values.

    Returns:
        The figure.
    """
    from qft_operator.physics.bulk_integrals import analytic_log_coefficient

    plt = _pyplot()
    radii = (
        radii if radii is not None else torch.tensor([0.25, 1.0, 4.0, 16.0], dtype=torch.float64)
    )
    eps_values = (
        eps_values if eps_values is not None else torch.logspace(-6.0, -2.0, 8, dtype=torch.float64)
    )
    figure, axes = plt.subplots(figsize=(5.6, 4.0))
    for index in range(radii.numel()):
        single = radii[index : index + 1]
        values = torch.stack(
            [
                integrator.reduced_contact_integral(single, eps=float(e)).squeeze(0)
                for e in eps_values
            ]
        )
        axes.plot(
            _to_numpy(-torch.log(eps_values)),
            _to_numpy(values),
            "o-",
            ms=3.5,
            lw=1.2,
            color=_PALETTE[index % len(_PALETTE)],
            label=rf"$r={float(radii[index]):.3g}$",
        )
    slope = analytic_log_coefficient(integrator.config.delta, integrator.config.L)
    slope = slope / integrator.config.normalization_factor
    axes.set_xlabel(r"$\log(1/\epsilon)$")
    axes.set_ylabel(r"$r^{2\Delta} \, I_{\Delta\Delta}(r, \epsilon)$")
    axes.set_title(rf"Contact integral: analytic slope $C_{{\log}}={slope:.4f}$")
    axes.legend(fontsize=8)
    return figure
