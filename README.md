# qft-neural-operator

Neural operators for the map from a Quantum Field Theory action to its boundary
observables in Euclidean AdS<sub>2</sub>:

$$
S[\phi] \;\longmapsto\; W[J], \qquad
V(\phi) \;\longmapsto\; W(p_1, p_2) = \langle V_{\beta_1}(p_1)\, V_{\beta_2}(p_2)\rangle_{\rm conn}
$$

in the Poincaré patch

$$
ds^2 = \frac{L^2}{z^2}\left(dz^2 + dp^2\right), \qquad \sqrt{g} = \frac{L^2}{z^2}, \qquad z > 0 .
$$

A bulk scalar of mass $m$ is dual to a boundary operator of dimension
$\Delta = \tfrac12 + \sqrt{\tfrac14 + m^2 L^2}$. Bulk integrals near $z \to 0$ diverge
logarithmically; holographic renormalization reorganizes those logarithms into an
anomalous dimension,

$$
W(r) = r^{-2\Delta_{\rm eff}}, \qquad \Delta_{\rm eff} = \Delta\,\beta_1\beta_2 - \gamma,
\qquad r = |p_1 - p_2| .
$$

The network learns that reorganization as a functional of $V$.

## The anomalous dimension as a functional

At first order in the interaction, the only term contributing to the *connected*
vertex-operator two-point function carries exactly one bulk-to-boundary propagator into
each insertion, with the remaining fields self-contracted into the coincident-point
propagator $\sigma^2 = G_\Delta(x,x)$. That term is $\beta_1\beta_2 V''$ multiplying the
regulated contact integral, which gives

$$
\boxed{\;\gamma[V] \;=\; \tfrac12\,\beta_1\beta_2 \,\langle V'' \rangle_\sigma \, C_{\log},
\qquad C_{\log} = \frac{2 L^2 c_\Delta}{2\Delta - 1}\;}
$$

with $\langle V''\rangle_\sigma$ the Gaussian average of $V''$ over $\mathcal{N}(0,\sigma^2)$.
For Sine-Gordon, $V(\phi) = -\lambda(e^{\xi\phi} + e^{-\xi\phi} - 2)$ gives
$\langle V''\rangle_0 = -2\lambda\xi^2$, so

$$
\gamma = -\lambda\,\frac{2 L^2 c_\Delta}{2\Delta - 1}\,\beta_1\beta_2\,\xi^2 ,
$$

the published expression, reproduced bit-for-bit (`tests/physics/test_correlators.py`).
Writing $\gamma$ as a functional rather than a formula is what lets arbitrary
potentials — polynomials, Gaussian-process draws — carry *exact* labels:

| theory | $\langle V''\rangle_\sigma$ | $\gamma$ |
| --- | --- | --- |
| free, $V \equiv 0$ | $0$ | $0$ exactly |
| Sine-Gordon | $-2\lambda\xi^2 e^{\xi^2\sigma^2/2}$ | published result at $\sigma = 0$ |
| $\phi^4$, $V = \lambda\phi^4$ | $12\lambda\sigma^2$ | $0$ when normal-ordered — the tadpole switches it on |
| polynomial $\sum_k c_k\phi^k$ | $\sum_{k\ge2} k(k-1)c_k\langle\phi^{k-2}\rangle_\sigma$ | closed form |
| GP (random Fourier features) | $-K^{-1/2}\sum_k a_k\omega_k^2 e^{-\omega_k^2\sigma^2/2}\cos b_k$ | closed form |

## The bulk integral, checked numerically

The regulated contact integral is evaluated by Gauss–Legendre quadrature over the bulk,
with $z = \epsilon e^{s}$ radially and a peak-tracking tangent map along the boundary:

$$
\int_{z>\epsilon}\! d^2x \sqrt{g}\; K_\Delta(x;p_1) K_\Delta(x;p_2)
= 2 L^2 c_\Delta\, r^{-2\Delta}\left[\log\frac{r}{\epsilon} + \kappa_\Delta\right] + O(\epsilon) .
$$

Reproduce with `uv run python examples/validate_holography.py`:

```
Regulated contact integral: d I~ / d log(1/eps)
    Delta          measured     2 L^2 c_Delta   rel. error
     1.00       0.636618063       0.636619772      2.7e-06
     1.50       1.000000002       1.000000000      2.0e-09
     2.00       1.273239545       1.273239545      2.7e-10
     3.00       1.697652727       1.697652726      1.4e-10

Scheme constant kappa at eps = 1e-6, Delta = 1.50
  r =    0.10   kappa = -0.306852840
  r =    1.00   kappa = -0.306852837
  r =   10.00   kappa = -0.306852837     constant across three decades of r,
  r =  100.00   kappa = -0.306852837     as conformal invariance requires
  spread across three decades of r: 1.4e-09
```

Under the holographic dictionary the cutoff *is* the renormalization scale,
$\epsilon = 1/M$, so the logarithm is $\log(Mr)$.

## What the network is

A Fourier-DeepONet with an operator-transformer head option. Three choices follow
directly from the physics:

**Log-space output.** Over $r \in [0.05, 12]$ with $\Delta \approx 3/2$, $W$ spans about
eight decades. An $L^2$ loss on $W$ is a loss on the three smallest separations; the
network predicts $\log W$.

**A free-theory baseline.** $\log W^{(0)} = -2\Delta\beta_1\beta_2\log r$ is known
exactly, so the network models only the anomalous part, which is smaller by
$\gamma/\Delta \sim 10^{-3}$ — precisely the signal of interest.

**Structural conformal symmetry.** Boundary translation invariance is imposed by the
positional encoding, not learned. The encoding also embeds the conformal factor: the bulk
region dominating $W(p_1,p_2)$ sits at radial depth $z_\star \sim r/2$, so the trunk
receives $\log\sqrt{g}(z_\star) = 2\log L - 2\log z_\star$ directly.

Non-locality along the boundary direction comes from a **boundary context field**: FNO
blocks run on a fixed internal grid of separations, conditioned on the branch code, and
each query interpolates that field at its own $\log r$. Putting spectral layers on the
query axis instead would make $W$ at one separation depend on which other separations
share the batch — the exact correlator does no such thing.

Four invariants hold exactly and are asserted in `tests/models/test_deeponet.py`:

| invariant | status |
| --- | --- |
| $V \equiv 0 \Rightarrow \log W = -2\Delta\beta_1\beta_2\log r$ at initialization | exact to machine precision |
| translation invariance $W(p_1+b, p_2+b) = W(p_1,p_2)$ | exact |
| independence of the query set (subset, permutation) | exact |
| diagonal Jacobian $\partial \log W_p / \partial \log r_q \propto \delta_{pq}$ | off-diagonal identically 0 |

The last one is not cosmetic: it is what lets the physics losses extract per-point
derivatives from a single reverse pass.

## Physics-informed losses

**Boundary scaling.** Conformal invariance fixes the large-$r$ behaviour to a power law,
which in log-log variables is entirely local: $d^2 \log W / d(\log r)^2 = 0$. Penalizing
that curvature needs *no labels at all* — it says "be a power law out there", not "have
this exponent" — so it applies to theories whose $\Delta_{\rm eff}$ is unknown. A
supervised variant matching a known exponent is also available.

**RG invariance.** No observable may depend on the subtraction scale:

$$
\left(M\frac{\partial}{\partial M} + \beta(\lambda)\frac{\partial}{\partial\lambda}\right) W = 0 .
$$

Both derivatives are assembled as a *single* directional derivative and evaluated with
forward-mode AD in one pass. The coupling tangent is the exact, closed-form
$\partial V/\partial\lambda$ that every potential exposes — which is why every potential
in this codebase is written as $V = \lambda\,v(\phi)$, linear in the coupling by
construction. A central-difference fallback engages automatically if a kernel lacks a
forward-mode rule.

**Consistency with the data.** Targets are built from the coupling at the *physical*
scale, $\bar\lambda(1/r)$, which the flow's group property makes independent of $M$.
Those targets annihilate the Callan–Symanzik operator **exactly**, for any $\beta$ — so
the RG loss and the data term never pull against each other. Verified for marginal,
one-loop and two-loop flows in `tests/losses/test_physics_terms.py`. The fixed-order
`quadrature` target mode does *not* have this property, and its shipped config sets the
RG weight to zero rather than papering over it.

## Data pipeline

Three target modes, selected by `data=<mode>`:

- **`resummed`** — closed-form $\gamma$, exponentiated with the running coupling. Fast,
  exactly RG-invariant.
- **`quadrature`** — strictly first order, contact diagram from actual bulk quadrature
  with $\epsilon = 1/M$. Honest about being fixed-order, hence not RG-invariant.
- **`hybrid`** — $\gamma$ *measured* from the quadrature, then resummed. Under
  `physics=ads2_cft` this yields labels identical to `resummed` to ~1e-7, because the
  measured $C_{\log}$ and the analytic $2L^2c_\Delta$ agree to that precision. It is not a
  second kind of target; what it buys is provenance — the label is derived from bulk
  quadrature rather than assumed — and a standing regression check on the integrator.

Potentials are drawn from a mixture of free, Sine-Gordon, $\phi^4$, random polynomial and
Gaussian-process families. GP samples use a random-Fourier-feature representation, so the
draw is an analytic function and both $V''$ and $\langle V''\rangle_\sigma$ stay exact —
GP theories carry the same label quality as Sine-Gordon, not a heuristic.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Run

```bash
qft-operator-train +experiment=smoke                 # CPU smoke test
qft-operator-train +experiment=reference_sine_gordon # published setup
qft-operator-train +experiment=hybrid_quadrature     # quadrature-derived labels
qft-operator-train +experiment=rg_flow               # relevant coupling, running exponent
qft-operator-train -m physics.m_sq=0.25,0.75,2.0     # sweep the boundary dimension
qft-operator-train trainer=multi_gpu                 # DDP, bf16
```

```bash
qft-operator-eval checkpoint=outputs/.../last.ckpt
qft-operator-generate data=hybrid physics=ads2_cft output=data/hybrid.pt
```

Every leaf is overridable: `loss.weights.rg=0.05`, `model=operator_transformer`,
`model.residual_mode=exponent`, `physics.sigma_sq=0.4`.

## A reference run

40 epochs on 4096 theories, CPU, ~30 minutes, full physics-informed objective
(`+experiment=reference_sine_gordon physics=ads2_cft optimizer.lr=1.2e-3`). Test split,
512 theories:

```
    relative L2 on log W        9.4e-04
    gamma MAE                   1.3e-03
    gamma relative MAE          0.059
    spectrum R^2                0.72        (best epoch 0.76)
```

The learning rate matters more than it looks. Under unit-normalized conventions $\gamma$
is a factor of $\pi$ larger than under the published normalization, and the 3e-3 that was
fine there produced an instability spike and stalled at $R^2 \approx 0.11$; 1.2e-3 with
twice the data reached 0.72.

Per-family behaviour on the deployed checkpoint, read off the live page:

| family | exact $\gamma$ | recovered | $\lvert\Delta\gamma\rvert$ |
| --- | --- | --- | --- |
| Gaussian process | $2.73\times10^{-3}$ | $2.97\times10^{-3}$ | $2.3\times10^{-4}$ |
| free | $0$ | $1.24\times10^{-4}$ | $1.2\times10^{-4}$ |
| Sine-Gordon | $-1.280\times10^{-2}$ | $-1.144\times10^{-2}$ | $1.4\times10^{-3}$ |
| polynomial | $-3.51\times10^{-3}$ | $-5.53\times10^{-3}$ | $2.0\times10^{-3}$ |
| $\phi^4$ | $0$ | $7.89\times10^{-3}$ | $7.9\times10^{-3}$ |

That the GP-drawn theories are the *best* recovered is the result worth having: the
operator generalizes to potentials with no analytic form. The $\phi^4$ row is the honest
failure, and a predictable one — a normal-ordered quartic has large $V$ but
$\langle V''\rangle_0 = 12\lambda\langle\phi^2\rangle_0 = 0$, so the network has to learn
that a conspicuous potential contributes exactly nothing at first order. It has not
learned that yet. A non-zero `physics.sigma_sq`, which switches the tadpole on and gives
$\phi^4$ a real signal, is the obvious next experiment.

## A convention note worth reading

The reference value $c_\Delta = 0.159$ is **not** the unit-normalized AdS<sub>2</sub>
coefficient. For $\Delta = 3/2$,

$$
c_\Delta^{\rm CFT} = \frac{\Gamma(\Delta)}{\sqrt{\pi}\,\Gamma(\Delta - \tfrac12)} = \frac12,
\qquad 0.159 \approx \frac{1}{2\pi},
$$

a factor of $\pi$ apart — a different convention for the boundary measure, not an error in
either. It matters because the numerical bulk integrator is defined with the
unit-normalized kernel, so with the override in place `hybrid`/`quadrature` labels differ
from the closed-form ones by exactly that factor. `PhysicsConfig.convention_ratio` reports
it, the dataset warns when it is not 1, and `physics=ads2_cft` (`c_delta: null`) makes
every pipeline agree exactly. The default `physics=ads2_reference` keeps the published
numbers.

## Layout

```
qft-neural-operator/
├── pyproject.toml                    uv + ruff + pytest + mypy
├── src/qft_operator/
│   ├── physics/                      no ML dependencies; importable and testable alone
│   │   ├── config.py                 PhysicsConfig: Delta, c_Delta, C_log, BF bound
│   │   ├── geometry.py               AdS2 metric, isometries, propagators
│   │   ├── hypergeometric.py         autograd-aware 2F1 via SciPy
│   │   ├── bulk_integrals.py         Gauss-Legendre contact integral + cached table
│   │   ├── potentials.py             V = lambda * v(phi); exact <V''>_sigma
│   │   ├── correlators.py            gamma[V], resummed and fixed-order targets
│   │   └── rg.py                     beta function, flow map, running coupling
│   ├── models/
│   │   ├── layers.py                 SpectralConv1d, FNO block, MetricPositionalEncoding
│   │   ├── branch.py                 spectral encoder over the phi grid
│   │   ├── trunk.py                  metric encoding + BoundaryContextField + FiLM
│   │   └── deeponet.py               FourierDeepONet, inner-product / attention heads
│   ├── losses/
│   │   ├── operators.py              log-slope, log-curvature, JVP with FD fallback
│   │   ├── data.py                   log-space supervised term
│   │   ├── scaling.py                AdS2 boundary scaling loss
│   │   ├── rg.py                     Callan-Symanzik residual
│   │   └── composite.py              weighted sum with a physics warm-up ramp
│   ├── data/                         samplers, dataset, LightningDataModule
│   ├── training/                     LightningModule, spectrum and free-theory callbacks
│   ├── analysis/spectrum.py          batched log-log fits, per-family reports
│   ├── viz/plots.py                  correlators, log residuals, spectrum, RG flow
│   ├── cli/                          Hydra entry points: train, evaluate, generate
│   └── configs/                      Hydra config tree (packaged with the wheel)
│   └── app/                          FastAPI: REST + binary WebSocket streams
│       ├── config.py                 pydantic-settings (QFT_OPERATOR_*)
│       ├── state.py                  shared physics/model singletons
│       ├── services.py               theory -> correlator, shared by REST and WS
│       └── ws/protocol.py            binary frames, mirrored in TypeScript
├── frontend/
│   ├── src/lib/                      physics.ts, bulk.ts, fft.ts, operator.ts, protocol.ts
│   ├── src/panels/                   bulk diagram, log residual, RG invariance
│   └── src/lib/__fixtures__/         golden values generated from Python
├── examples/validate_holography.py   reproduces the quadrature tables above
├── tests/                            322 tests, mirroring the source layout
├── ARCHITECTURE.md                   layer boundaries and the three load-bearing choices
├── CHANGELOG.md
└── CITATION.cff
```

## Tests

```bash
uv run pytest                    # 322 tests
uv run pytest -m "not slow"      # skip the quadrature convergence sweeps
uv run ruff check . && uv run mypy
```

The physics tests are the point of the suite, not an afterthought:

- free-theory limits — $\gamma[0] = 0$, the exact power law, and the network's own
  free-theory limit at initialization;
- boundary conformal symmetry — translation invariance of the encoding and the network,
  covariance $W(ar) = a^{-2\Delta_{\rm eff}}W(r)$, weight-$\Delta$ covariance of $K_\Delta$;
- metric invariance — chordal distance and $\sqrt{g}\,dz\,dp$ under the AdS<sub>2</sub>
  dilatation and translation isometries;
- propagator normalization — $z^{\Delta-1}\int dp\, K_\Delta = 1$ to $2\times10^{-6}$;
- quadrature against closed form — $C_{\log}$ across four scaling dimensions, and
  $\kappa_\Delta$ constant in $r$ and convergent in $\epsilon$;
- Gaussian moments — every closed-form $\langle V''\rangle_\sigma$ against independent
  Gauss–Hermite quadrature, and every $V''$ against autograd;
- RG structure — group property of the flow, $M$-independence of the running coupling,
  and the Callan–Symanzik residual vanishing on RG-invariant correlators for marginal,
  one-loop and two-loop $\beta$;
- differentiability — `gradcheck` on the spectral convolution and a double-backward
  check, since the scaling loss differentiates the network twice.

## Relation to `DiffQFT`

`DiffQFT/diffqft/geometry.py` and `propagators.py` cover the same AdS<sub>2</sub> metric
and hypergeometric propagator. The implementations here are typed, validated and
autograd-tested versions of the same objects; if the two projects stay adjacent, that
pair is the obvious candidate for a shared package.

## License

MIT.
