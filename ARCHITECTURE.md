# Architecture

Six layers, each importable on its own. The dependency arrows point strictly downward;
`physics` has no ML dependencies at all, which is why the holography can be validated
without ever constructing a network.

```
  frontend/          Vite + React + Three.js; physics.ts, bulk.ts, operator.ts
      |                  \
      | binary WebSocket  \  static export (weights.bin + manifest.json)
      v                    v
    app/          FastAPI: REST + ws/, shared services layer
      |
    cli/          Hydra entry points: train, evaluate, generate_data, export
      |
    training/     LightningModule, optimizer schedule, spectrum + free-theory callbacks
      |
   +--+-----------------+----------------+
   |                    |                |
 models/             losses/           data/          analysis/   viz/
 FourierDeepONet    data + scaling     samplers,      log-log     figures
 branch / trunk     + Callan-Symanzik  dataset,       fits
   |                    |              datamodule        |          |
   +--------------------+----------------+---------------+----------+
                                |
                            physics/
             geometry, potentials, bulk integrals, correlators, RG
```

The frontend reaches the physics layer twice over: through the server, and through its own
TypeScript port for the standalone build. Both paths are pinned to the same golden values,
so the duplication is checked rather than merely intended.

## Where each physical fact lives

| fact | module | how it is checked |
| --- | --- | --- |
| $\Delta(\Delta-1) = m^2L^2$, BF bound | `physics/config.py` | `test_config.py` |
| $\sqrt{g} = L^2/z^2$, AdS<sub>2</sub> isometries | `physics/geometry.py` | dilatation/translation invariance of $u$ and of $\sqrt{g}\,dz\,dp$ |
| $z^{\Delta-1}\int dp\,K_\Delta = 1$ | `physics/geometry.py` | trapezoid over $p \in [-600, 600]$ |
| $C_{\log} = 2L^2c_\Delta$ | `physics/bulk_integrals.py` | Gauss–Legendre vs closed form, four $\Delta$ values |
| $\kappa_\Delta$ depends only on $r/\epsilon$ | `physics/bulk_integrals.py` | constant across three decades of $r$; convergent in $\epsilon$ |
| $\gamma[V] = \tfrac12\beta_1\beta_2\langle V''\rangle_\sigma C_{\log}$ | `physics/correlators.py` | reduces to the published Sine-Gordon result exactly |
| $\langle V''\rangle_\sigma$ per family | `physics/potentials.py` | closed form vs Gauss–Hermite; $V''$ vs autograd |
| flow group property $\Rightarrow$ RG invariance | `physics/rg.py` | $\bar\lambda(1/r)$ independent of $M$ |

## Three design decisions worth the words

### The boundary context field is not on the query axis

The obvious way to put FNO layers "along the boundary direction $p$" is to Fourier
transform the trunk's query axis. That is wrong: it makes $W(p_1,p_2)$ depend on which
*other* separations happen to be in the same batch, and the exact correlator does no such
thing.

`models/trunk.py::BoundaryContextField` instead runs the spectral stack on a fixed
internal grid of $\log r$, conditioned on the branch code and $\log M$, and lets each
query interpolate that field at its own separation. Every query then sees a
representation built from the whole boundary profile — genuinely non-local — while the
prediction stays a function of its own coordinates.

That property is load-bearing downstream. Because $\log W_p$ depends only on the
coordinates of query $p$, the Jacobian $\partial\log W_p/\partial\log r_q$ is diagonal, so
differentiating the *sum* of the outputs returns every per-point derivative in one reverse
pass. `losses/operators.py` relies on it; `test_deeponet.py` asserts the off-diagonal is
identically zero.

### Coupling derivatives go forward, coordinate derivatives go backward

Two derivative structures, two mechanisms:

- $\partial/\partial\log r$ and $\partial/\partial\log M$ are per-point coordinates, so
  reverse mode plus the diagonal-Jacobian trick gets them in one pass.
- $\partial/\partial\lambda$ enters through the *whole* branch input:
  $\partial W/\partial\lambda = \sum_i (\partial W/\partial V_i)(\partial V_i/\partial\lambda)$.
  That is a directional derivative along a known tangent — a JVP. Forward mode gets every
  output element at once; reverse mode would need the full Jacobian.

`DirectionalDerivative` runs forward-mode by default and falls back once, with a warning,
to central differences if a kernel has no forward-mode rule. Forward mode does work
through `torch.fft` on current PyTorch; the fallback is insurance, not the normal path.

### Every potential is linear in its coupling

`Potential` subclasses implement a *shape function* $v(\phi)$; the base class supplies
$V = \lambda v$, $V'' = \lambda v''$ and $\partial V/\partial\lambda = v$. Linearity is
therefore structural rather than a convention each subclass must remember, and the exact
$\partial V/\partial\lambda$ the RG loss needs is available in closed form.

The same requirement dictates that input standardization uses one *global* scalar rather
than a per-sample one: dividing $V$ and $\partial V/\partial\lambda$ by the same constant
preserves $V = \lambda\,\partial V/\partial\lambda$, while a per-sample scale that itself
depended on $\lambda$ would silently invalidate the chain rule.

## Target modes and their RG status

```
    resummed     gamma closed form  --> exponentiate with lambda_bar(1/r)   exactly RG-invariant
    hybrid       gamma from quadrature --> exponentiate                     exactly RG-invariant
    quadrature   1 + beta1 beta2 <V''> I~(r, 1/M)                           fixed order, NOT invariant
```

The first two build $W$ from the coupling at the physical scale $1/r$. Because the flow's
group property makes $\bar\lambda(1/r)$ independent of which scale $\lambda$ was quoted at,
those targets annihilate the Callan–Symanzik operator exactly — for any $\beta$, not just
to leading order. The RG loss is then a constraint the data already satisfies, so the two
terms cannot fight.

`quadrature` is a fixed-order expression and genuinely is not RG-invariant. Rather than
hide that, `configs/data/quadrature.yaml` says so and directs the user to `loss=data_only`.

## The web layer

### The static build re-implements the physics, and that is checked

`frontend/src/lib/physics.ts`, `bulk.ts` and `operator.ts` are ports, not wrappers. A page
that only worked with a server attached would not be a demo anyone could open, so the
closed-form physics, the Gauss-Legendre bulk quadrature and the whole operator forward
pass all exist in TypeScript as well.

Two implementations of one physics is precisely where silent divergence lives, so
`tests/app/parity_fixture.py` generates golden values from Python and the TypeScript tests
compare against them; `test_frontend_parity.py` fails when the committed fixture goes
stale. Agreement is 11-12 decimals for the closed forms, 8 for the quadrature, and
float32-limited for the network.

### Why the export keeps the FFT

`aten::fft_rfft` has no ONNX lowering at opset 17, which rules out the usual export route
for a Fourier neural operator. On a fixed grid the spectral layer *is* a circular
convolution — `bake_spectral_kernel` performs that identity and it holds to ~1e-15 — so an
ONNX-ready form is available behind `--bake-spectral`.

It is not the default, because it is the wrong trade for a browser. A dense circular
convolution costs $C_{\rm in}C_{\rm out}N^2$ against the $C_{\rm in}C_{\rm out}M$ of a
mode-truncated spectral multiply: at the default widths that is roughly a thousandfold,
the difference between a responsive slider and a frozen one, and it inflates the weight
blob from 4.5 to 7.8 MiB besides. The default therefore ships Fourier weights and does a
64-point transform client-side.

### Checkpoints carry their architecture

Rebuilding a network by inferring shapes from its weights works only for the widths that
happen to be visible in them. `FourierDeepONet.hyperparameters` records the constructor
arguments verbatim, the Lightning module writes them into the checkpoint, and both the
server and the export rebuild from that. Without a record the load falls back to defaults
plus an inferred `n_phi` and warns — and then `load_state_dict` raises, which is the right
outcome: a wrong-width network that loads silently is far worse than one that refuses to.

The same reasoning applies to `feature_scale`. The dataset divides the branch input by one
global scalar, so inference outside the training pipeline must apply the identical
scaling; a served model that skips it sees inputs an order of magnitude off and returns
plausible-looking nonsense. It travels in the checkpoint, the manifest, and
`QFT_OPERATOR_FEATURE_SCALE` for checkpoints written before it was recorded.

### The sockets are request/response paced

Both WebSocket endpoints answer exactly one binary frame per JSON control message, and the
client keeps at most one request in flight. Dragging a slider therefore throttles itself
to whatever the server sustains, instead of queuing states the server computes and the
client discards. The density field is quantized to 8 bits against a per-frame range — it
is consumed as a colour map, and that is also exactly the R8 texture format, so nothing is
repacked between the wire and the GPU. The physics number travelling alongside, the
converged contact integral, is never quantized.

## Numerical notes

- **Log space throughout.** $W$ spans ~8 decades over the default separation window;
  predicting and regressing $\log W$ is what keeps the objective from collapsing onto the
  smallest separations.
- **float64 for physics, float32 for training.** Quadrature, potentials and correlators
  compute in double; the dataset casts to float32 once, at the end.
- **The contact integral is tabulated, not recomputed.** $\tilde I$ depends on $r$ and
  $\epsilon$ only through $r/\epsilon$, so one univariate table (built in ~3 s) replaces a
  2-D quadrature per sample — a ~500x speedup at ~6e-6 relative error.
- **Zero-ish readout initialization.** `readout_init_scale=1e-4` puts the untrained
  network within $10^{-4}$ of the free theory — an order of magnitude below a typical
  $\gamma$ — while still giving every parameter a gradient on the first step. Exactly
  `0.0` makes the free-theory limit exact at the cost of one frozen step upstream.
- **No unused parameters.** The branch builds its token projection only for the attention
  head, so DDP never needs `find_unused_parameters=True`.
- **Validation runs with `inference_mode: false`.** The scaling and RG terms differentiate
  the network, and inference-mode tensors are excluded from autograd. The module degrades
  to the data term rather than crashing if the flag is left on.
