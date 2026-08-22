"""Reproduce the holographic-renormalization checks quoted in the README.

Runs the AdS2 bulk quadrature against the closed-form results and prints the comparison.
Nothing here touches the network -- the point is that the physics layer stands on its own.

Usage:
    uv run python examples/validate_holography.py [--figure out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qft_operator.physics import (
    ConformalIntegrator,
    PhysicsConfig,
    QuadratureSpec,
    SineGordon,
    analytic_log_coefficient,
    anomalous_dimension,
    c_delta_cft,
)


def log_coefficient_table(dimensions: tuple[float, ...], n_nodes: int) -> None:
    """Compare the measured $C_{\\log}$ against $2L^2c_\\Delta$ across dimensions."""
    print("\nRegulated contact integral: d I~ / d log(1/eps)")
    print(f"  {'Delta':>7}  {'measured':>16}  {'2 L^2 c_Delta':>16}  {'rel. error':>11}")
    r = torch.tensor([1.0], dtype=torch.float64)
    for delta in dimensions:
        # Invert Delta(Delta - 1) = m^2 L^2 to hit the requested dimension exactly.
        config = PhysicsConfig(
            m_sq=delta * (delta - 1.0), c_delta=None, propagator_normalization="cft"
        )
        engine = ConformalIntegrator(config, QuadratureSpec(n_radial=n_nodes, n_boundary=n_nodes))
        measured = float(engine.log_slope(r, eps=1e-5))
        exact = analytic_log_coefficient(config.delta, config.L)
        print(f"  {delta:7.2f}  {measured:16.9f}  {exact:16.9f}  {abs(measured / exact - 1):11.1e}")


def scheme_constant_table(n_nodes: int) -> None:
    """Show that $\\kappa_\\Delta$ depends only on $r/\\epsilon$, as conformal invariance requires.

    A constant across three decades of $r$ is the sharpest convergence check available.
    """
    config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
    engine = ConformalIntegrator(config, QuadratureSpec(n_radial=n_nodes, n_boundary=n_nodes))
    radii = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
    print(f"\nScheme constant kappa at eps = 1e-6, Delta = {config.delta:.2f}")
    for r, kappa in zip(radii.tolist(), engine.kappa(radii, eps=1e-6).tolist(), strict=True):
        print(f"  r = {r:7.2f}   kappa = {kappa:.9f}")
    print(f"  spread across three decades of r: {float(engine.kappa(radii, 1e-6).std()):.1e}")


def anomalous_dimension_check() -> None:
    """Confirm the general functional collapses onto the published Sine-Gordon formula."""
    config = PhysicsConfig()
    print("\nAnomalous dimension: general functional vs published Sine-Gordon formula")
    print(f"  {'lambda':>9}  {'xi':>6}  {'gamma[V]':>14}  {'published':>14}  {'abs. diff':>10}")
    for lam, xi in ((0.02, 0.8), (-0.04, 1.15), (0.005, 0.4)):
        general = anomalous_dimension(SineGordon(lam, xi), config)
        published = config.analytical_anomalous_dim(lam, xi)
        print(
            f"  {lam:9.3f}  {xi:6.2f}  {general:14.9f}  {published:14.9f}  "
            f"{abs(general - published):10.1e}"
        )


def convention_note() -> None:
    """Print the $c_\\Delta$ convention mismatch explicitly rather than burying it."""
    config = PhysicsConfig()
    print("\nNormalization convention")
    print(f"  configured c_Delta          : {config.c_delta_effective:.6f}")
    print(f"  unit-normalized c_Delta^CFT : {c_delta_cft(config.delta):.6f}")
    print(f"  ratio                       : {config.convention_ratio:.6f}  (~ 1/pi)")
    print("  -> use physics=ads2_cft (c_delta=null) to align the analytic and")
    print("     quadrature pipelines exactly.")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=256, help="Gauss-Legendre nodes per direction")
    parser.add_argument("--figure", type=Path, default=None, help="write the convergence plot here")
    args = parser.parse_args()

    log_coefficient_table((1.0, 1.5, 2.0, 3.0), args.nodes)
    scheme_constant_table(args.nodes)
    anomalous_dimension_check()
    convention_note()

    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        from qft_operator.viz import plot_bulk_integral_convergence, set_style

        set_style()
        config = PhysicsConfig(c_delta=None, propagator_normalization="cft")
        engine = ConformalIntegrator(
            config, QuadratureSpec(n_radial=args.nodes, n_boundary=args.nodes)
        )
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plot_bulk_integral_convergence(engine).savefig(args.figure)
        print(f"\nfigure written to {args.figure}")


if __name__ == "__main__":
    main()
