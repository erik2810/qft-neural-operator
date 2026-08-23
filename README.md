# qft-neural-operator

**[Live demo →](https://erik2810.github.io/qft-neural-operator/)** — three WebGPU surfaces
with the operator running client-side; no server, no install.

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

## Results

All figures below are **three seeds**, 2048 theories, 25 epochs, CPU, unit-normalized
conventions. Single runs on this problem are not reportable -- see the reproducibility
note.

### What the operator architecture buys

| arm | spectrum $R^2$ | $\gamma$ rel. error | $\log W$ rel. $L^2$ |
| --- | --- | --- | --- |
| Fourier-DeepONet | **0.900 ± 0.011** | **0.066 ± 0.004** | 6.4e-04 |
| baseline DeepONet (`model=baseline_deeponet`) | 0.747 ± 0.075 | 0.332 ± 0.055 | 7.6e-03 |

A 5× improvement in $\gamma$ and 11.8× in $\log W$; the $R^2$ gap is four times the
largest within-arm spread and the two windows do not overlap. This ablates the *package* --
the baseline config turns off the spectral branch, the boundary context field, structural
translation invariance and the free-theory residual all at once. A component-wise ablation
is now affordable but has not been run.

### The deployed checkpoint

The model the frontend ships is trained at full width on the capped distribution -- 4096
theories, 40 epochs, $\sigma^2 = 0$, one seed:

```
    spectrum R^2            0.9596      (last-10 epoch stdev 0.0010)
    gamma relative error    0.0319
    log W relative L2       3.1e-04
```

| family | $\gamma$ MAE |
| --- | --- |
| $\phi^4$ | 5.6e-05 |
| free | 8.4e-05 |
| Sine-Gordon | 1.9e-04 |
| polynomial | 4.3e-04 |
| Gaussian process | 1.2e-03 |

$\phi^4$ is now the **best**-predicted family, having been the worst before the cap
(7.7e-04 predicted against an exact zero). That closes the loop on the question this
started from: the $\phi^4$ failure was real but derivative. A normal-ordered quartic has
large $V$ and $\langle V''\rangle_0 = 0$, so it needs the model to learn that a
conspicuous potential contributes nothing -- and it could not, while a handful of extreme
GP draws were absorbing the gradient. Remove them and it learns it easily.

### Reproducibility, and why it needed fixing

Before the perturbative-regime cap, this problem could not be measured. Three seeds of an
otherwise identical configuration:

```
    uncapped   R^2 = 0.770, 0.636, 0.745     spread 0.134     gamma rel. error 0.150
    capped     R^2 = 0.895, 0.906, 0.899     spread 0.011     gamma rel. error 0.066
```

`max_gamma_ratio=0.05` buys **11.8× tighter reproducibility across seeds** and 8.9× less
within-run wobble, while also improving the mean -- for the cost of discarding 1.5% of the
data. The cause is in [Data pipeline](#data-pipeline): a handful of Gaussian-process draws
reach $|\gamma|/\Delta \approx 0.19$, which a first-order label does not describe, and a
quadratic loss lets them dominate it.

How badly this misleads is worth stating plainly. Two runs whose labels agreed to 1e-7
landed at $R^2 = 0.716$ and $0.902$ -- a gap that would have looked like a decisive result
for whichever pipeline happened to draw the better seed.

### A running coupling

With `rg=relevant` ($\epsilon = 0.35$, $b = 2.0$) the Callan-Symanzik term stops being
near-vacuous: $\Delta_{\rm eff}$ runs with $r$, so $W$ is no longer a pure power law and
the constraint has content. Two seeds each:

| arm | $\gamma$ MAE | $R^2$ | seed spread | RG residual |
| --- | --- | --- | --- | --- |
| marginal (control) | 9.3e-04 | 0.901 | 0.011 | 4.2e-06 |
| relevant | 1.6e-03 | 0.916 | 0.006 | **2.7e-06** |

The network satisfies the constraint better under a running coupling than under a marginal
one, which is the right way round -- a marginal $\beta$ only asks it not to depend on
$\log M$. The data fit is ~1.7× harder in absolute terms, as a running exponent should be.

Note the relative $\gamma$ error moves the *other* way (0.065 → 0.177) purely because the
window-wide cap removes large-$|\gamma|$ draws and shrinks the normalizer. Normalized
metrics stop being comparable the moment the distribution changes; that has bitten three
separate times in this project, which is why absolute errors are quoted alongside.

### The tadpole

Switching on the coincident-point propagator (`physics.sigma_sq=0.4`) improves every family
by ~12× and drives $R^2$ to 0.9999 with zero variance across the last ten epochs. That is
not a $\phi^4$ fix, though it looks like one: `polynomial`'s signal is unchanged
(2.648e-3 → 2.666e-3) while its relative error falls 27.9% → 2.26%. The smearing damps high
frequencies by $e^{-\omega^2\sigma^2/2}$, flattening the same GP tail the cap targets.
$\sigma^2 = 0$ remains the physically canonical normal-ordered scheme.

## Interactive frontend

Three WebGPU surfaces, written in TSL. Every panel is the same object — a scalar field
over a physical plane, rendered as shaded relief and banded one colour per decade — because
the physics here is the counting of decades, and one visual language beats three unrelated
3D treatments.

**The bulk integrand.** Height is
$\log\!\left(z\sqrt{g}\,K_\Delta K_\Delta\right)$ over $(p, \log z)$. The factor of $z$ is
not cosmetic: the measure is $dz/z = d\log z$ and the vertical axis *is* $\log z$, so the
volume under this surface is the contact integral. Toward the boundary — the lit rail — the
two ridges at $p = \mp r/2$ narrow as $z$ while holding their area, so they climb, and the
divergence becomes a sum over decades of equal-area slices. A plane marks the cutoff.

**$\gamma$ as a functional.** $\log W$ with the free power law divided out, over
$(\log r, \lambda)$. The tilt along $\log r$ *is* $2\gamma$; across $\lambda$ it fans out
linearly, because $\gamma$ is linear in the coupling. The amber plane is the closed form,
the teal sheet is the operator at twenty couplings, and where the sheet lifts off the plane
is where it has not learned the functional.

**RG invariance.** $\log W$ over $(\log r, \log M)$. The renormalization scale is a gauge
choice, so the physical claim is that *the surface has no slope along it*: carry the
coupling along the flow and the amber sheet comes out ruled — sight down the $M$ axis and
it is a straight line. The teal overlay changes $M$ without running the coupling and
visibly tilts. Five overlapping curves became one surface whose flatness you check by eye.

Amber is the boundary and every exact quantity; teal is every predicted or unphysical one.

The page is set as a live preprint rather than a dashboard, because that is what the
artifact is: warm paper, a measured text column, figures that break out wider than the
prose, and a margin carrying the numbers. The one thing a printed paper cannot do is keep
that margin current — those readouts recompute while the figures are being dragged.

Three type roles, each with a reason. Crimson Pro sets the prose. Atkinson Hyperlegible —
engineered for legibility by the Braille Institute — carries every number, label and
control, because the figures are the thing that must not be misread. KaTeX's Computer
Modern sets the mathematics and pairs with Crimson Pro without argument.

One trap worth knowing if you touch the labels: they are uppercased, and `text-transform`
does not respect mathematics. It maps lowercase Greek onto capitals, silently rewriting
$\gamma$ as $\Gamma$, $\lambda$ as $\Lambda$ and $\epsilon$ as E — each of which means
something else. Symbols inside a label must be wrapped in `<Sym>`.

### Two notes for anyone extending the surfaces

Vertex displacement reads the height texture with an **explicit LOD** — the vertex stage
has no screen-space derivatives, so an implicit sample is invalid. And the height texture is
**half float, not float32**: WebGPU treats `r32float` as unfilterable unless the optional
`float32-filterable` feature is present, so a `LinearFilter` sampler over one is invalid.

Camera placement is load-bearing rather than taste. The bulk ridges run *away* from the
viewer along $\log z$, so a high three-quarter view foreshortens them into a featureless
plateau and the surface reads as flat even when fully displaced — which is exactly as
convincing as a real bug, and cost an afternoon before the geometry was checked directly.
Each panel sets its own camera.

### Running it

```bash
cd frontend && npm install && npm run dev
```

```bash
uv run uvicorn qft_operator.app.main:app --port 8000
```

The page probes `/health` on load. With a server it uses the binary WebSocket protocol for
both hot paths; without one it computes everything locally and says so in the header.

The probe is a progressive enhancement, not a requirement, so it defaults **on** in dev and
**off** in a production build: a static export has no server by construction, and probing
there only logs a 404 that reports nothing the page has not already worked out. Build with
`VITE_BACKEND_PROBE=on` for a bundle FastAPI will serve itself.

To serve a trained checkpoint:

```bash
QFT_OPERATOR_CHECKPOINT=outputs/.../last.ckpt uv run uvicorn qft_operator.app.main:app
```

### The static export

`aten::fft_rfft` has no ONNX lowering at opset 17, so the spectral layers cannot go through
a standard runtime. Two ways out, and the export supports both:

```bash
uv run qft-operator-export --checkpoint outputs/.../best.ckpt --dtype float16 --output frontend/public/operator
```

```bash
uv run qft-operator-export --bake-spectral --output build/onnx-ready
```

The default keeps the learned weights in Fourier space and the browser does its own
64-point transform. `--bake-spectral` instead exploits the fact that mode-truncated
multiplication in Fourier space is *exactly* a circular convolution on a fixed grid
(verified to ~1e-15), removing the FFT entirely -- which an ONNX or TFLite consumer needs,
at the cost of size and, at these widths, roughly a thousandfold in arithmetic.

`--dtype float16` halves the download (4.79 → 2.40 MiB) for 2.4% of the model's own error
— the weights keep ~3 decimal digits, while the prediction is drawn against an exact curve
it differs from by a part in 100. The committed export uses it; the parity fixture stays
float32, where the point is to catch a mis-ported layer rather than to render one.

The export *is* tracked, so the demo works from a fresh clone and Pages has something to
deploy. The page also degrades gracefully without it, labelling the prediction as
unavailable.

### Two implementations, one physics

`frontend/src/lib/` re-implements the AdS₂ background, the bulk quadrature and the operator
forward pass in TypeScript. That is where two implementations of one physics silently
diverge, so the TypeScript side is pinned against golden values generated from Python:

```bash
uv run python -m tests.app.parity_fixture
```

```bash
cd frontend && npm test
```

`tests/app/test_frontend_parity.py` fails if the committed fixture goes stale. Agreement is
11-12 decimals for the closed-form physics, 8 for the quadrature, and float32-limited for
the network.

One thing deliberately does *not* match: the polynomial and Gaussian-process families use a
different PRNG on each side, so the same seed means "another draw from the same
distribution", not the same function. The page therefore builds $V(\phi)$ locally and sends
it to the server, which evaluates the potential on screen rather than its own draw.

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

`DiffQFT/diffqft/geometry.py` and `propagators.py` implement the same AdS₂ objects, and
where they overlap the two agree to machine precision:

```
    chordal distance u    1.4511278195  vs  1.4511278195   (diff 1.1e-12)
    Delta from m^2        1.5000000000  vs  1.5000000000
    bulk-to-bulk G        0.0469503228  vs  0.0469503228   (ratio 1.000000)
```

So the duplication is real rather than superficial, and consolidating is a rename job, not
a reconciliation. Three things would need care:

- **Argument order differs.** `DiffQFT` takes `(x1, z1, x2, z2)`, this package takes
  `(z1, p1, z2, p2)`. Silently compatible signatures with swapped meaning is the worst
  possible failure mode for a merge.
- **`bulk_boundary_kernel` is not the same function.** `DiffQFT`'s is the unnormalized
  Poisson kernel $Lz/(z^2 + \Delta x^2)$; this package's `bulk_to_boundary` is
  $c_\Delta\left[z/(z^2+\Delta x^2)\right]^{\Delta}$, which integrates to one. Both are
  defensible; they are not interchangeable.
- **Everything else here is a superset** — $\sqrt{g}$, the isometries, the invariant
  measure, `c_delta_cft`, full typing and validation — and is covered by
  `tests/physics/test_geometry.py`.

Nothing in `DiffQFT` has been touched. The natural shape would be a small shared package
carrying the metric, the isometries and the hypergeometric propagator, with both projects
depending on it.

## License

MIT.
