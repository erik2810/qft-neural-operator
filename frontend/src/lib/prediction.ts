/**
 * Turning a theory into an operator prediction.
 *
 * Kept out of the panel module so the panels export components only, and so both the
 * browser path and the request the backend receives are built from one place.
 */

import {
  anomalousDimensionFromCorrelator,
  buildPotential,
  freeDimension,
  type Background,
  type Theory,
} from "./physics";
import type { Operator } from "./operator";

export interface Prediction {
  logR: Float64Array;
  logW: Float64Array;
  gamma: number;
  source: "server" | "browser";
}

/** Sample V(φ) on the grid the operator was trained on. */
export function potentialSamples(theory: Theory, phiGrid: ArrayLike<number>): Float64Array {
  const potential = buildPotential(theory);
  const out = new Float64Array(phiGrid.length);
  for (let i = 0; i < phiGrid.length; i += 1) {
    out[i] = theory.coupling * potential.shape(phiGrid[i]);
  }
  return out;
}

/** The Gaussian-averaged second derivative, in closed form. */
export function potentialMoment(theory: Theory, sigmaSq: number): number {
  return theory.coupling * buildPotential(theory).shapeGaussianSecondMoment(sigmaSq);
}

/** Run the exported operator in the browser. */
export function predictLocally(
  operator: Operator,
  theory: Theory,
  background: Background,
  logR: Float64Array,
): Prediction {
  const v = operator.scaleFeatures(potentialSamples(theory, operator.phiGrid));
  const logW = operator.predictLogW(v, logR, theory.logM);
  return {
    logR,
    logW,
    gamma: anomalousDimensionFromCorrelator(logR, logW, freeDimension(background)),
    source: "browser",
  };
}
