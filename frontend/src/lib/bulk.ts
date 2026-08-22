/**
 * AdS2 bulk contact integral in the browser, ported from
 * `qft_operator.physics.bulk_integrals`.
 *
 * The integral
 *
 *     I(r, ε) = ∫_{z>ε} d²x √g K_Δ(x;p₁) K_Δ(x;p₂)
 *
 * is logarithmically divergent, and with the cutoff it behaves as
 *
 *     I = 2L²c_Δ r^{-2Δ} [ log(r/ε) + κ_Δ ] + O(ε).
 *
 * Reproducing the coefficient 2L²c_Δ from quadrature — in the browser, live — is the
 * claim the bulk panel makes, so the quadrature has to be the real one rather than the
 * asymptotic formula it is being checked against.
 *
 * The boundary map matters: at small z the integrand is two spikes of width ~z sitting at
 * p = ∓r/2, so the substitution has to carry the *local* scale z, not the separation r.
 * The line is split at the midpoint and each half gets a tan-map centred on its own peak.
 */

import { cDeltaCft, type Background } from "./physics";

/** Gauss–Legendre nodes and weights on [−1, 1], by Newton iteration on Pₙ. */
export function gaussLegendre(n: number): { nodes: Float64Array; weights: Float64Array } {
  const nodes = new Float64Array(n);
  const weights = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let x = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let derivative = 0;
    for (let iteration = 0; iteration < 100; iteration += 1) {
      let p0 = 1;
      let p1 = 0;
      for (let k = 0; k < n; k += 1) {
        const p2 = p1;
        p1 = p0;
        p0 = ((2 * k + 1) * x * p1 - k * p2) / (k + 1);
      }
      derivative = (n * (x * p0 - p1)) / (x * x - 1);
      const dx = -p0 / derivative;
      x += dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    nodes[i] = x;
    weights[i] = 2 / ((1 - x * x) * derivative * derivative);
  }
  return { nodes, weights };
}

const QUADRATURE_CACHE = new Map<number, { nodes: Float64Array; weights: Float64Array }>();

function cached(n: number): { nodes: Float64Array; weights: Float64Array } {
  let entry = QUADRATURE_CACHE.get(n);
  if (!entry) {
    entry = gaussLegendre(n);
    QUADRATURE_CACHE.set(n, entry);
  }
  return entry;
}

export interface QuadratureSpec {
  nRadial: number;
  nBoundary: number;
  zMaxOverR: number;
}

export const DEFAULT_QUADRATURE: QuadratureSpec = {
  nRadial: 128,
  nBoundary: 128,
  zMaxOverR: 200,
};

/**
 * Evaluate ∫_{z>ε} d²x √g K_Δ K_Δ for insertions at p = ∓r/2.
 *
 * @param r Boundary separation, > 0.
 * @param eps Near-boundary cutoff, > 0 (holographically 1/M).
 * @param bg The AdS2 background.
 * @param spec Quadrature resolution.
 */
export function contactIntegral(
  r: number,
  eps: number,
  bg: Background,
  spec: QuadratureSpec = DEFAULT_QUADRATURE,
): number {
  if (!(r > 0) || !(eps > 0)) throw new RangeError(`need r > 0 and eps > 0, got ${r}, ${eps}`);
  const { nodes: sx, weights: sw } = cached(spec.nRadial);
  const { nodes: tx, weights: tw } = cached(spec.nBoundary);

  const d = bg.delta;
  const half = 0.5 * r;
  const sMax = Math.max(Math.log((spec.zMaxOverR * r) / eps), 1e-6);
  let total = 0;

  for (let i = 0; i < sx.length; i += 1) {
    const s = 0.5 * sMax * (sx[i] + 1);
    const z = eps * Math.exp(s);
    const wz = 0.5 * sMax * sw[i] * z;
    const theta0 = Math.atan(half / z);
    const span = 0.5 * Math.PI + theta0;
    const zz = z * z;

    for (let j = 0; j < tx.length; j += 1) {
      const u = 0.5 * (tx[j] + 1);
      const baseW = span * 0.5 * tw[j];
      // Right half: p ∈ [0, ∞) with the peak at +r/2; left half mirrors it.
      const thetaR = -theta0 + span * u;
      const thetaL = -0.5 * Math.PI + span * u;
      const cr = Math.cos(thetaR);
      const cl = Math.cos(thetaL);
      const pR = half + z * Math.tan(thetaR);
      const pL = -half + z * Math.tan(thetaL);
      const wR = (baseW * z) / (cr * cr);
      const wL = (baseW * z) / (cl * cl);

      for (const [p, wp] of [
        [pR, wR],
        [pL, wL],
      ] as const) {
        const d1 = zz + (p + half) * (p + half);
        const d2 = zz + (p - half) * (p - half);
        // Through logs: the tan-map reaches |p| ~ 1e5 z at the outermost nodes.
        const value = Math.exp((2 * d - 2) * Math.log(z) - d * Math.log(d1) - d * Math.log(d2));
        total += value * wz * wp;
      }
    }
  }

  const c = cDeltaCft(d);
  return (total * bg.L * bg.L * c * c) / bg.normalizationFactor;
}

/** Contact integral with the conformal factor stripped: Ĩ = r^{2Δ} I = C_log[log(r/ε) + κ]. */
export function reducedContactIntegral(
  r: number,
  eps: number,
  bg: Background,
  spec: QuadratureSpec = DEFAULT_QUADRATURE,
): number {
  return contactIntegral(r, eps, bg, spec) * Math.pow(r, 2 * bg.delta);
}

/**
 * Measure C_log = dĨ/d log(1/ε) numerically — the number that should come out equal to
 * the analytic 2L²c_Δ regardless of r.
 */
export function measuredLogCoefficient(
  r: number,
  eps: number,
  bg: Background,
  epsRatio = 4,
  spec: QuadratureSpec = DEFAULT_QUADRATURE,
): number {
  const fine = reducedContactIntegral(r, eps / epsRatio, bg, spec);
  const coarse = reducedContactIntegral(r, eps, bg, spec);
  return (fine - coarse) / Math.log(epsRatio);
}

export interface IntegrandField {
  /** Natural log of the density, row-major, row 0 at z = ε. */
  logDensity: Float32Array;
  nZ: number;
  nP: number;
  logZMin: number;
  logZMax: number;
  pMin: number;
  pMax: number;
  /** Peak of `logDensity`, for colour mapping. */
  logHigh: number;
  /** Floor used for colour mapping, `logHigh` minus `floorDecades`. */
  logLow: number;
}

/**
 * Tabulate √g K_Δ K_Δ on a grid uniform in log z and p — the chart in which the two
 * boundary-localized ridges that generate the log divergence are actually visible.
 *
 * The window is placed around the **separation**, not the cutoff. Anchoring it at z = ε
 * is the obvious choice and the wrong one: the ridges are O(ε) wide there, far below one
 * pixel for any interesting cutoff, so the picture degenerates into a blob. Centring on r
 * puts the structure at a scale the grid resolves, and ε becomes a line moving through a
 * fixed density — which is the truer story, since the cutoff decides how much of the
 * integrand is integrated, not what it looks like.
 *
 * Display only: the integral itself comes from {@link contactIntegral}, whose adaptive
 * map resolves those ridges properly.
 */
export function integrandField(
  r: number,
  bg: Background,
  nZ = 192,
  nP = 256,
  decadesBelow = 3,
  decadesAbove = 1,
  floorDecades = 8,
): IntegrandField {
  if (!(r > 0)) throw new RangeError(`need r > 0, got ${r}`);
  const d = bg.delta;
  const half = 0.5 * r;
  const width = 2 * r;
  const logZMin = Math.log(r) - decadesBelow * Math.LN10;
  const logZMax = Math.log(r) + decadesAbove * Math.LN10;
  const coefficient =
    (bg.L * bg.L * cDeltaCft(d) * cDeltaCft(d)) / bg.normalizationFactor;
  const logCoefficient = Math.log(coefficient);

  const logDensity = new Float32Array(nZ * nP);
  let logHigh = -Infinity;
  for (let i = 0; i < nZ; i += 1) {
    const logZ = logZMin + ((logZMax - logZMin) * i) / (nZ - 1);
    const z = Math.exp(logZ);
    const zz = z * z;
    for (let j = 0; j < nP; j += 1) {
      const p = -width + (2 * width * j) / (nP - 1);
      const value =
        logCoefficient +
        (2 * d - 2) * logZ -
        d * Math.log(zz + (p + half) * (p + half)) -
        d * Math.log(zz + (p - half) * (p - half));
      logDensity[i * nP + j] = value;
      if (value > logHigh) logHigh = value;
    }
  }
  return {
    logDensity,
    nZ,
    nP,
    logZMin,
    logZMax,
    pMin: -width,
    pMax: width,
    logHigh,
    logLow: logHigh - floorDecades * Math.LN10,
  };
}

/** A bulk density field ready for upload as an 8-bit texture. */
export interface QuantizedField {
  density: Uint8Array;
  nZ: number;
  nP: number;
  logZMin: number;
  logZMax: number;
  pMin: number;
  pMax: number;
  logLow: number;
  logHigh: number;
  r: number;
  logEps: number;
  integral: number;
  /** Where the field came from, so the panel can say so. */
  source: "browser" | "server";
}

/**
 * Quantize a log-density to 8 bits against its own range.
 *
 * The same encoding the server applies before sending, so both paths hand the renderer
 * an identical texture format. Eight bits is well below what a colour map resolves; the
 * physics number travelling alongside is never quantized.
 */
export function quantizeField(
  field: IntegrandField,
  r: number,
  logEps: number,
  integral: number,
): QuantizedField {
  const span = field.logHigh - field.logLow;
  const density = new Uint8Array(field.logDensity.length);
  for (let i = 0; i < density.length; i += 1) {
    const t = (Math.max(field.logDensity[i], field.logLow) - field.logLow) / (span + 1e-12);
    density[i] = Math.round(Math.min(Math.max(t, 0), 1) * 255);
  }
  return {
    density,
    nZ: field.nZ,
    nP: field.nP,
    logZMin: field.logZMin,
    logZMax: field.logZMax,
    pMin: field.pMin,
    pMax: field.pMax,
    logLow: field.logLow,
    logHigh: field.logHigh,
    r,
    logEps,
    integral,
    source: "browser",
  };
}
