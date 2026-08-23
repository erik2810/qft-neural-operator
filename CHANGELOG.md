# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `DataConfig.max_gamma_ratio` (default ``0.05``): rejects draws whose
  $|\gamma| / (\Delta\beta_1\beta_2)$ exceeds the threshold, with the count surfaced as
  `DatasetStatistics.rejected`.

  The labels are first order in the interaction and presume $\gamma \ll \Delta$, but the
  Gaussian-process family produced draws up to $|\gamma|/\Delta \approx 0.19$ -- a fifth
  of the boundary exponent, which the first-order formula does not describe. Because the
  loss is quadratic, those few samples dominated it. Measured over the shipped mixture,
  the tail is *exclusively* GP (60 of 1149 draws) and the cap removes it at a cost of
  ~1.5% of the data, cutting the ratio of largest to median $|\gamma|$ from ~70x to ~20x.
- `SpectrumReport.median_relative_error`, logged as `spectrum/median_relative_error`.
  $R^2$ is built from squared error and on this heavy-tailed target is decided by a
  handful of extreme draws, so it can rank two models in the opposite order to their
  typical-case accuracy -- which is exactly what happened over the last ten epochs of the
  reference run, where $R^2$ fell from 0.756 to 0.716 while the mean absolute error
  halved.

### Added

- The frontend is now three WebGPU surfaces written in TSL, replacing the flat heatmap and
  the two SVG plots. Each panel is a scalar field over a physical plane: the bulk integrand
  over $(p, \log z)$ with the measure folded into the height, $\log W$ over
  $(\log r, \lambda)$ where the tilt is $2\gamma$, and $\log W$ over $(\log r, \log M)$
  where RG invariance is visible as a ruled surface with no slope along $M$.
- The page is set as a live preprint: warm paper, a measured text column, numbered figures
  that break out wider than the prose, and a margin whose readouts recompute as the figures
  are dragged. Crimson Pro for prose, Atkinson Hyperlegible for every number and control,
  KaTeX's Computer Modern for the mathematics.
- Semantic colour throughout: amber is the AdS boundary and every exact quantity, teal is
  every predicted or unphysical one. Each has a bright tier for the dark figures and a deep
  tier for paper, since a value that reads on one fails contrast on the other.
- Log-decade contour banding in place of a smooth colour ramp.
- `qft-operator-export --dtype float16`: halves the weight blob (4.79 → 2.40 MiB) at a cost
  of 2.4% of the model's own error. The browser runtime decodes binary16 by hand, since
  `Float16Array` is too new to rely on.
- GitHub Pages deploy workflow, gated on the Python and frontend jobs. It deploys the
  committed export; no training happens in CI.

### Changed

- The exported demo operator is now committed, at half precision, so the page works from a
  fresh clone and Pages has an artifact to publish.
- `data=hybrid` is no longer described as a distinct or superior target. Under
  `physics=ads2_cft` it produces labels identical to `data=resummed` to ~1e-7, because the
  measured $C_{\log}$ and the analytic $2L^2c_\Delta$ agree to that precision. What it
  offers is provenance -- the label is derived from bulk quadrature rather than assumed --
  and a regression check on the integrator.

### Measured

- Architecture ablation, three seeds: the Fourier-DeepONet reaches $R^2 = 0.900 \pm 0.011$
  and $\gamma$ relative error $0.066 \pm 0.004$ against $0.747 \pm 0.075$ and
  $0.332 \pm 0.055$ for `model=baseline_deeponet` -- 5x on $\gamma$, 11.8x on $\log W$,
  with non-overlapping seed windows.
- Running-coupling experiment (`rg=relevant`, two seeds): with the window-wide cap the
  network reaches $R^2 = 0.916$ and an RG residual of 2.7e-06, *below* the marginal
  control's 4.2e-06, with the absolute $\gamma$ error 1.7x the marginal case as a running
  exponent should be. Before the cap fix the same configuration gave 0.766 with a seed
  spread of 0.079.
- Seed-variance study, three seeds x {capped, uncapped}: the cap reduces across-seed $R^2$
  spread 11.8x (0.134 to 0.011). Two earlier runs whose labels agreed to 1e-7 had landed
  at $R^2 = 0.716$ and $0.902$, so single-run comparisons on the uncapped distribution
  were meaningless.

### Added

- The frontend is now three WebGPU surfaces written in TSL, replacing the flat heatmap and
  the two SVG plots. Each panel is a scalar field over a physical plane: the bulk integrand
  over $(p, \log z)$ with the measure folded into the height, $\log W$ over
  $(\log r, \lambda)$ where the tilt is $2\gamma$, and $\log W$ over $(\log r, \log M)$
  where RG invariance is visible as a ruled surface with no slope along $M$.
- The page is set as a live preprint: warm paper, a measured text column, numbered figures
  that break out wider than the prose, and a margin whose readouts recompute as the figures
  are dragged. Crimson Pro for prose, Atkinson Hyperlegible for every number and control,
  KaTeX's Computer Modern for the mathematics.
- Semantic colour throughout: amber is the AdS boundary and every exact quantity, teal is
  every predicted or unphysical one. Each has a bright tier for the dark figures and a deep
  tier for paper, since a value that reads on one fails contrast on the other.
- Log-decade contour banding in place of a smooth colour ramp.
- `qft-operator-export --dtype float16`: halves the weight blob (4.79 → 2.40 MiB) at a cost
  of 2.4% of the model's own error. The browser runtime decodes binary16 by hand, since
  `Float16Array` is too new to rely on.
- GitHub Pages deploy workflow, gated on the Python and frontend jobs. It deploys the
  committed export; no training happens in CI.

### Changed

- The exported demo operator is now committed, at half precision, so the page works from a
  fresh clone and Pages has an artifact to publish.
- `data=hybrid` is no longer described as a distinct or superior target. Under
  `physics=ads2_cft` it produces labels identical to `data=resummed` to ~1e-7, because the
  measured $C_{\log}$ and the analytic $2L^2c_\Delta$ agree to that precision. What it
  offers is provenance -- the label is derived from bulk quadrature rather than assumed --
  and a standing regression check on the integrator.
- README results are now reported as mean ± spread over three seeds. Single-run numbers on
  this problem were not reportable.

### Measured

- Architecture ablation, three seeds per arm: the Fourier-DeepONet reaches
  $R^2 = 0.900 \pm 0.011$ and $\gamma$ relative error $0.066 \pm 0.004$, against
  $0.747 \pm 0.075$ and $0.332 \pm 0.055$ for `model=baseline_deeponet` -- 5x on $\gamma$,
  11.8x on $\log W$, with non-overlapping seed windows.
- Running-coupling experiment (`rg=relevant`, two seeds): with the window-wide cap the
  network reaches $R^2 = 0.916$ and an RG residual of 2.7e-06, *below* the marginal
  control's 4.2e-06, with the absolute $\gamma$ error 1.7x the marginal case as a running
  exponent should be. Before the cap fix the same configuration gave 0.766 with a seed
  spread of 0.079.
- Seed-variance study, three seeds x {capped, uncapped}: the cap reduces across-seed $R^2$
  spread 11.8x (0.134 to 0.011). Two earlier runs whose labels agreed to 1e-7 had landed at
  $R^2 = 0.716$ and $0.902$, so single-run comparisons on the uncapped distribution were
  meaningless.

### Fixed

- `max_gamma_ratio` bounded $\gamma$ at the coupling's *reference* scale rather than
  across the separation window. With a running coupling the correlator is built from
  $\bar\lambda(1/r)$, so $\gamma$ varies with $r$ -- at $\epsilon = 0.35$ by about a factor
  of six -- and samples were passing the cap at $|\gamma|/\Delta = 0.0479$ while carrying
  $0.2146$ in their targets, four times over. The bound is now evaluated at the window
  edges, which the monotonicity of the flow makes sufficient.
- Restored the "Interactive frontend" README section, lost to a splice that replaced
  everything between two headings when a third had been inserted between them.
- The parity-fixture check compared `json.dumps` output, making it a test of the host's
  libm as much as of the fixture. At $\Delta = 3/2$ the normalisation
  $c_\Delta = \Gamma(3/2) / (\sqrt\pi\,\Gamma(1))$ is analytically exactly $1/2$, but macOS
  returns `0.49999999999999994` and glibc returns `0.5`. Both are right to within one ulp;
  only the decimal serialisations differ, so CI told a Linux runner that a fixture
  generated on macOS was stale. Numbers now compare at `rel=1e-12` -- eight orders tighter
  than the tolerance the TypeScript tests use -- while structure still compares exactly.
- The health probe and the WebSocket URLs were root-absolute, so under a base-path
  deployment they addressed the origin root: the probe requested `/health` rather than
  `/qft-neural-operator/health`. Both now resolve through `BASE_URL`, and the probe is
  skipped entirely in a production build, where a static export has no backend to find.
- `npm exec tsc -b` relied on npm forwarding a bare `-b`; npm parses the flag itself, so
  tsc printed its help and exited non-zero. The type-check is now a named package script.

## [0.2.0] - 2026-08-22

### Added

- **FastAPI inference server** (`qft_operator.app`): REST for the conformal background,
  correlators and the regulated bulk integral, plus two binary WebSocket streams for the
  per-frame paths. Configuration through pydantic-settings (`QFT_OPERATOR_*`).
- **Interactive frontend** (`frontend/`): Vite + React + Three.js + Tailwind + KaTeX, with
  three panels -- the AdS2 bulk and Witten contact diagram, the logarithmic residual and
  $\gamma$ extraction, and RG invariance across five renormalization scales. Works
  standalone or against the server, and says which in the header.
- **Browser inference** (`frontend/src/lib/operator.ts`): a hand port of the
  Fourier-DeepONet forward pass, including a 64-point real FFT, so the static build needs
  no server.
- **`qft-operator-export`**: writes `weights.bin` plus `manifest.json`, with the spectral
  layers either kept in Fourier space (default) or baked into circular convolutions
  (`--bake-spectral`) for consumers that cannot do an FFT.
- **Cross-language parity fixture** (`tests/app/parity_fixture.py`): golden values from
  Python that the TypeScript tests check against, with a test that fails when the
  committed fixture goes stale.
- `ConformalIntegrator.integrand_field` for tabulating the contact-integral density on a
  display grid.
- `FourierDeepONet` now exposes `branch_hidden`, `context_grid`, `context_width` and
  `log_r_range`, which previously could not be reached from the config at all.

### Fixed

- **`feature_scale` was not recorded in checkpoints.** The dataset divides the branch
  input by a global scalar; without it in the checkpoint, any inference outside the
  training pipeline -- the server, the browser export -- silently fed inputs off by that
  factor. Now stored in the checkpoint and the manifest, with
  `QFT_OPERATOR_FEATURE_SCALE` and `--feature-scale` for older checkpoints.
- **Checkpoints did not record their architecture.** The server and the export rebuilt
  the network with default widths and only inferred `n_phi`, which is wrong for any run
  that changed a width. `FourierDeepONet.hyperparameters` now travels with the weights.
- `PhysicsConfig.BOUNDARY_DIM` was a dataclass field rather than a `ClassVar`, so it
  appeared in the constructor signature and in equality comparisons.

## [0.1.0] - 2026-08-22

Initial release. Refactor of a single-file DeepONet prototype into a modular research
codebase.

### Added

- **Physics layer** (`qft_operator.physics`), free of ML dependencies:
  - `PhysicsConfig` with derived conformal data, BF-bound validation and an explicit
    `convention_ratio` reporting the $c_\Delta$ normalization in use.
  - `AdS2Geometry`: metric, isometries, bulk-to-boundary and bulk-to-bulk propagators;
    autograd-aware ${}_2F_1$ via SciPy.
  - `ConformalIntegrator`: Gauss–Legendre evaluation of the regulated contact Witten
    integral, with a peak-tracking boundary map that resolves the two boundary-localized
    spikes at small $z$. Reproduces $C_{\log} = 2L^2c_\Delta$ to ~1e-9 for $\Delta = 3/2$.
  - `ReducedIntegralTable`: cached univariate table in $r/\epsilon$, ~500x faster than
    direct quadrature at ~6e-6 relative error.
  - `Potential` hierarchy written as $V = \lambda v(\phi)$, with exact
    $\langle V''\rangle_\sigma$ for free, Sine-Gordon, $\phi^4$, polynomial and
    Gaussian-process (random Fourier feature) families.
  - `anomalous_dimension`: $\gamma[V] = \tfrac12\beta_1\beta_2\langle V''\rangle_\sigma
    C_{\log}$, reducing exactly to the published Sine-Gordon result.
  - `BetaFunction` with closed-form and RK4 flow maps.
- **Models** (`qft_operator.models`): `FourierDeepONet` with spectral convolutions along
  the field coordinate, a metric-aware positional encoding embedding
  $\log\sqrt{g} = 2\log L - 2\log z_\star$, a `BoundaryContextField` for non-locality along
  $p$, FiLM conditioning, and inner-product or cross-attention heads.
- **Losses** (`qft_operator.losses`): log-space data term, AdS<sub>2</sub> boundary scaling
  loss (label-free curvature form and supervised form), Callan–Symanzik RG-invariance loss
  evaluated as a single forward-mode directional derivative, and a weighted composite with
  a physics warm-up ramp.
- **Data** (`qft_operator.data`): mixture sampler over five potential families and three
  target modes (`resummed`, `quadrature`, `hybrid`).
- **Training** (`qft_operator.training`): Lightning module with AdamW, warm-up-into-cosine
  schedule and spectral weights excluded from decay; `SpectrumCallback` and
  `FreeTheoryProbe`.
- **Analysis / viz**: batched log-log fits with per-family reporting; correlator, log
  residual, spectrum, potential gallery, RG flow and quadrature convergence figures.
- **CLI / config**: three Hydra entry points and a packaged config tree with four
  ready-made experiments.
- **Tests**: 282 tests covering free-theory limits, boundary conformal symmetry, metric
  invariance, propagator normalization, quadrature against closed form, Gaussian moments
  against Gauss–Hermite, RG group properties, and `gradcheck` plus double-backward on the
  spectral convolution.

### Changed from the prototype

- $\phi^4$ no longer uses the ad-hoc `gamma = lam * 0.4`; it gets the correct first-order
  answer, which is *zero* for a normal-ordered vertex and non-zero only through the
  tadpole $\sigma^2$.
- The correlator is predicted in log space. Regressing $W$ directly under MSE put
  essentially all the weight on the smallest separations, where $W$ is ~8 decades larger.
- Boundary translation invariance is structural rather than learned; the trunk consumes
  conformal invariants instead of raw $(p_1, p_2)$.
- Fourier-feature bandwidth reduced from `scale=10` on raw unbounded coordinates to order
  one on normalized logarithmic coordinates, which were aliasing badly.
- Separations are sampled on a sorted log-uniform grid rather than as
  `p2 = p1 + U(0, 3)`, covering the window uniformly in the variable the physics is linear
  in and giving the scaling loss a well-conditioned stencil.
