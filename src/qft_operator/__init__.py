"""Neural operators for the QFT action-to-observable map in Euclidean AdS2.

The framework learns the functional

.. math::
    S[\\phi] \\longmapsto W[J], \\qquad
    V(\\phi) \\longmapsto W(p_1, p_2)
      = \\langle V_{\\beta_1}(p_1) V_{\\beta_2}(p_2) \\rangle_{\\mathrm{conn}},

in the Poincare patch $ds^2 = (L^2/z^2)(dz^2 + dp^2)$, where holographic renormalization
reorganizes the near-boundary logarithms into an anomalous dimension
$\\Delta_{\\mathrm{eff}} = \\Delta\\beta_1\\beta_2 - \\gamma$.

Subpackages
-----------
``physics``
    Geometry, potentials, exact conformal bulk integrals, RG flow. No ML dependencies.
``models``
    Fourier-DeepONet and operator-transformer architectures.
``losses``
    Supervised, boundary-scaling and Callan-Symanzik terms.
``data``
    Hybrid data generation: closed-form, quadrature, and GP-sampled theories.
``training``
    Lightning module and callbacks.
``analysis`` / ``viz``
    Anomalous-dimension extraction and plotting.
``app``
    FastAPI server: REST plus binary WebSocket streams for the interactive panels.
"""

from qft_operator.physics.config import PhysicsConfig

__all__ = ["PhysicsConfig", "__version__"]
__version__ = "0.2.0"
