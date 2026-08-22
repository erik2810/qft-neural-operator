/**
 * Exact Euclidean AdS2 holography, ported from `qft_operator.physics`.
 *
 * Everything here is closed form or quadrature, so the static build computes the *truth*
 * in the browser and never needs the backend for it. The port is pinned against Python
 * golden values in `physics.test.ts` — if the two ever drift, that test fails rather than
 * the page quietly showing something else.
 *
 * Metric: ds² = (L²/z²)(dz² + dp²), √g = L²/z², boundary at z → 0.
 */

/** Lanczos coefficients (g = 7, n = 9), good to ~1e-15 on the real axis. */
const LANCZOS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-6,
  1.5056327351493116e-7,
];

/** Gamma function Γ(x) for x > 0 via the Lanczos approximation. */
export function gamma(x: number): number {
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gamma(1 - x));
  const z = x - 1;
  let series = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i += 1) series += LANCZOS[i] / (z + i);
  const t = z + LANCZOS.length - 1.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * series;
}

/** Boundary scaling dimension from Δ(Δ − 1) = m²L². */
export function deltaFromMass(mSq: number, L = 1): number {
  const bound = 0.25 + mSq * L * L;
  if (bound < 0) throw new RangeError(`Breitenlohner-Freedman bound violated: m²L² = ${mSq * L * L}`);
  return 0.5 + Math.sqrt(bound);
}

/** Unit-normalized bulk-to-boundary coefficient c_Δ = Γ(Δ)/(√π Γ(Δ − ½)) for d = 1. */
export function cDeltaCft(delta: number): number {
  if (delta <= 0.5) throw new RangeError(`c_delta requires delta > 1/2 (got ${delta})`);
  return gamma(delta) / (Math.sqrt(Math.PI) * gamma(delta - 0.5));
}

/** The AdS2 background the page is showing. */
export interface Background {
  L: number;
  mSq: number;
  delta: number;
  cDelta: number;
  /** 1 for the unit-normalized convention, (2Δ − 1) for the bulk-limit one. */
  normalizationFactor: number;
  beta1: number;
  beta2: number;
  sigmaSq: number;
}

export function defaultBackground(): Background {
  const delta = deltaFromMass(0.75, 1);
  return {
    L: 1,
    mSq: 0.75,
    delta,
    cDelta: cDeltaCft(delta),
    normalizationFactor: 1,
    beta1: 1,
    beta2: 1,
    sigmaSq: 0,
  };
}

/** Coefficient of log(r/ε) in the regulated contact integral: 2L²c_Δ / normalization. */
export function logCoefficient(bg: Background): number {
  return (2 * bg.L * bg.L * bg.cDelta) / bg.normalizationFactor;
}

/** Free-theory boundary exponent Δβ₁β₂. */
export function freeDimension(bg: Background): number {
  return bg.delta * bg.beta1 * bg.beta2;
}

// ---------------------------------------------------------------------------
// Potentials: V(φ) = λ v(φ), so ∂V/∂λ = v(φ) exactly.
// ---------------------------------------------------------------------------

export type Family = "free" | "sine_gordon" | "phi4" | "polynomial" | "gp_fourier";

export interface Theory {
  family: Family;
  coupling: number;
  xi: number;
  seed: number;
  logM: number;
}

export function defaultTheory(): Theory {
  return { family: "sine_gordon", coupling: 0.02, xi: 0.8, seed: 0, logM: 0 };
}

/** Deterministic 32-bit PRNG (mulberry32) so a seed reproduces a theory exactly. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller normal draw from a uniform generator. */
function normal(rand: () => number): number {
  const u = Math.max(rand(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** A potential as the page needs it: shape function, second derivative, Gaussian moment. */
export interface Potential {
  /** v(φ), the shape function. V(φ) = coupling · v(φ). */
  shape(phi: number): number;
  /** v''(φ). */
  shapeSecondDerivative(phi: number): number;
  /** ⟨v''⟩ under φ ~ N(0, σ²), in closed form. */
  shapeGaussianSecondMoment(sigmaSq: number): number;
}

/** Random-Fourier-feature GP sample: v(φ) = K^{-1/2} Σ aₖ cos(ωₖφ + bₖ). */
function randomFourier(seed: number, nFeatures: number, lengthscale: number): Potential {
  const rand = mulberry32(seed + 1);
  const a = new Float64Array(nFeatures);
  const w = new Float64Array(nFeatures);
  const b = new Float64Array(nFeatures);
  for (let k = 0; k < nFeatures; k += 1) {
    a[k] = normal(rand);
    w[k] = normal(rand) / lengthscale;
    b[k] = rand() * 2 * Math.PI;
  }
  const norm = 1 / Math.sqrt(nFeatures);
  const at = (phi: number, weight: (k: number) => number) => {
    let total = 0;
    for (let k = 0; k < nFeatures; k += 1) total += weight(k) * Math.cos(w[k] * phi + b[k]);
    return total * norm;
  };
  return {
    shape: (phi) => at(phi, (k) => a[k]) - at(0, (k) => a[k]),
    shapeSecondDerivative: (phi) => at(phi, (k) => -a[k] * w[k] * w[k]),
    shapeGaussianSecondMoment: (sigmaSq) => {
      let total = 0;
      for (let k = 0; k < nFeatures; k += 1) {
        total += -a[k] * w[k] * w[k] * Math.exp(-0.5 * w[k] * w[k] * sigmaSq) * Math.cos(b[k]);
      }
      return total * norm;
    },
  };
}

/** Random polynomial with factorially damped coefficients. */
function polynomial(seed: number, degree: number): Potential {
  const rand = mulberry32(seed + 2);
  const c = new Float64Array(degree + 1);
  let factorial = 1;
  for (let k = 0; k <= degree; k += 1) {
    if (k > 0) factorial *= k;
    c[k] = normal(rand) / factorial;
  }
  return {
    shape: (phi) => {
      let total = 0;
      for (let k = 0; k <= degree; k += 1) total += c[k] * Math.pow(phi, k);
      return total - c[0];
    },
    shapeSecondDerivative: (phi) => {
      let total = 0;
      for (let k = 2; k <= degree; k += 1) total += k * (k - 1) * c[k] * Math.pow(phi, k - 2);
      return total;
    },
    shapeGaussianSecondMoment: (sigmaSq) => {
      let total = 0;
      for (let k = 2; k <= degree; k += 1) {
        const order = k - 2;
        if (order % 2 === 1) continue;
        let doubleFactorial = 1;
        for (let m = order - 1; m > 0; m -= 2) doubleFactorial *= m;
        total += k * (k - 1) * c[k] * Math.pow(sigmaSq, order / 2) * doubleFactorial;
      }
      return total;
    },
  };
}

/** Build the potential a theory describes. */
export function buildPotential(theory: Theory): Potential {
  switch (theory.family) {
    case "free":
      return {
        shape: () => 0,
        shapeSecondDerivative: () => 0,
        shapeGaussianSecondMoment: () => 0,
      };
    case "sine_gordon": {
      const xi = theory.xi;
      return {
        shape: (phi) => -2 * (Math.cosh(xi * phi) - 1),
        shapeSecondDerivative: (phi) => -2 * xi * xi * Math.cosh(xi * phi),
        shapeGaussianSecondMoment: (sigmaSq) => -2 * xi * xi * Math.exp(0.5 * xi * xi * sigmaSq),
      };
    }
    case "phi4":
      return {
        shape: (phi) => phi ** 4,
        shapeSecondDerivative: (phi) => 12 * phi * phi,
        shapeGaussianSecondMoment: (sigmaSq) => 12 * sigmaSq,
      };
    case "polynomial":
      return polynomial(theory.seed, 6);
    case "gp_fourier":
      return randomFourier(theory.seed, 64, 0.9);
  }
}

/**
 * First-order anomalous dimension γ[V] = ½ β₁β₂ ⟨V''⟩_σ C_log.
 *
 * Reduces to γ = −λ (2L²c_Δ/(2Δ−1)) β₁β₂ ξ² for Sine-Gordon at σ = 0.
 */
export function anomalousDimension(theory: Theory, bg: Background): number {
  const moment = theory.coupling * buildPotential(theory).shapeGaussianSecondMoment(bg.sigmaSq);
  return 0.5 * bg.beta1 * bg.beta2 * moment * logCoefficient(bg);
}

/** Uniform grid in log r — the variable the physics is linear in. */
export function logRGrid(n = 128, rMin = 0.05, rMax = 12): Float64Array {
  const out = new Float64Array(n);
  const lo = Math.log(rMin);
  const hi = Math.log(rMax);
  for (let i = 0; i < n; i += 1) out[i] = lo + ((hi - lo) * i) / (n - 1);
  return out;
}

/**
 * RG-improved correlator: log W = −2 Δ_eff(λ̄) log r with λ̄ the coupling at scale 1/r.
 *
 * With a marginal β the coupling does not run and this is a plain power law; the point
 * of writing it this way is that λ̄(1/r) is independent of M, so the curve is *exactly*
 * RG-invariant — which is what the RG panel demonstrates by moving M and seeing nothing.
 */
export function resummedLogCorrelator(
  logR: Float64Array,
  theory: Theory,
  bg: Background,
  epsilon = 0,
): { logW: Float64Array; deltaEff: Float64Array } {
  const gammaRef = anomalousDimension(theory, bg);
  const free = freeDimension(bg);
  const logW = new Float64Array(logR.length);
  const deltaEff = new Float64Array(logR.length);
  for (let i = 0; i < logR.length; i += 1) {
    // λ̄ = λ (M r)^ε for a linear β, so γ scales by the same factor.
    const running = epsilon === 0 ? 1 : Math.exp(epsilon * (theory.logM + logR[i]));
    deltaEff[i] = free - gammaRef * running;
    logW[i] = -2 * deltaEff[i] * logR[i];
  }
  return { logW, deltaEff };
}

/** Least-squares slope and intercept of y against x. */
export function fitLine(x: ArrayLike<number>, y: ArrayLike<number>): { slope: number; intercept: number } {
  const n = x.length;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    sx += x[i];
    sy += y[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let cov = 0;
  let variance = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = x[i] - mx;
    cov += dx * (y[i] - my);
    variance += dx * dx;
  }
  const slope = variance > 0 ? cov / variance : 0;
  return { slope, intercept: my - slope * mx };
}

/** γ recovered from a correlator by a log-log fit: γ = Δβ₁β₂ + ½ d log W / d log r. */
export function anomalousDimensionFromCorrelator(
  logR: ArrayLike<number>,
  logW: ArrayLike<number>,
  free: number,
): number {
  return free + 0.5 * fitLine(logR, logW).slope;
}
