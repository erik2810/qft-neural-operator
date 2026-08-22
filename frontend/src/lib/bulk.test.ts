import { describe, expect, it } from "vitest";
import { loadParity } from "./__fixtures__/load";
import {
  contactIntegral,
  gaussLegendre,
  integrandField,
  measuredLogCoefficient,
  reducedContactIntegral,
} from "./bulk";
import { logCoefficient, type Background } from "./physics";

const parity = loadParity();

function background(): Background {
  const bg = parity.background;
  return {
    L: bg.L,
    mSq: bg.m_sq,
    delta: bg.delta,
    cDelta: bg.c_delta,
    normalizationFactor: bg.normalization_factor,
    beta1: 1,
    beta2: 1,
    sigmaSq: 0,
  };
}

describe("Gauss-Legendre", () => {
  it("integrates polynomials exactly up to degree 2n - 1", () => {
    const { nodes, weights } = gaussLegendre(8);
    // ∫_{-1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd.
    for (let k = 0; k <= 15; k += 1) {
      let total = 0;
      for (let i = 0; i < nodes.length; i += 1) total += weights[i] * Math.pow(nodes[i], k);
      expect(total).toBeCloseTo(k % 2 === 0 ? 2 / (k + 1) : 0, 12);
    }
  });

  it("has weights summing to the interval length", () => {
    const { weights } = gaussLegendre(32);
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 12);
  });
});

describe("contact integral", () => {
  it("matches the Python quadrature", () => {
    const bg = background();
    for (const c of parity.bulk) {
      expect(contactIntegral(c.r, c.eps, bg)).toBeCloseTo(c.contact_integral, 8);
      expect(reducedContactIntegral(c.r, c.eps, bg)).toBeCloseTo(c.reduced, 8);
    }
  });

  it("recovers the analytic log coefficient 2 L^2 c_Delta", () => {
    const bg = background();
    const analytic = logCoefficient(bg);
    for (const r of [0.25, 1, 4, 20]) {
      expect(measuredLogCoefficient(r, 1e-5, bg)).toBeCloseTo(analytic, 4);
    }
  });

  it("scales as a conformal two-point function", () => {
    // Under (r, eps) -> (a r, a eps) the integral scales as a^{-2 Delta}.
    const bg = background();
    const a = 5;
    const base = contactIntegral(0.7, 1e-4, bg);
    const scaled = contactIntegral(a * 0.7, a * 1e-4, bg);
    expect(scaled / base).toBeCloseTo(Math.pow(a, -2 * bg.delta), 8);
  });

  it("rejects degenerate arguments", () => {
    expect(() => contactIntegral(0, 1e-4, background())).toThrow();
    expect(() => contactIntegral(1, 0, background())).toThrow();
  });
});

describe("integrand field", () => {
  it("puts the ridges at p = -+ r/2 near the boundary", () => {
    const bg = background();
    const r = 1;
    const field = integrandField(r, bg, 16, 129, 3, 1);
    // Row 0 sits three decades below r, where the two propagator peaks are separated.
    let best = -Infinity;
    let bestColumn = 0;
    for (let j = 0; j < field.nP; j += 1) {
      const value = field.logDensity[j];
      if (value > best) {
        best = value;
        bestColumn = j;
      }
    }
    const p = field.pMin + ((field.pMax - field.pMin) * bestColumn) / (field.nP - 1);
    expect(Math.abs(Math.abs(p) - r / 2)).toBeLessThan(0.05);
  });

  it("is symmetric under p -> -p", () => {
    const field = integrandField(1, background(), 8, 65, 3, 1);
    for (let i = 0; i < field.nZ; i += 1) {
      for (let j = 0; j < field.nP; j += 1) {
        const mirrored = field.nP - 1 - j;
        expect(field.logDensity[i * field.nP + j]).toBeCloseTo(
          field.logDensity[i * field.nP + mirrored], 5,
        );
      }
    }
  });

  it("reports a colour range spanning the requested decades", () => {
    const field = integrandField(1, background(), 32, 64, 3, 1, 6);
    expect(field.logHigh - field.logLow).toBeCloseTo(6 * Math.LN10, 10);
    expect(field.logZMax - field.logZMin).toBeCloseTo(4 * Math.LN10, 10);
    // r = 1 sits three of the four decades up the window.
    expect(field.logZMin).toBeCloseTo(-3 * Math.LN10, 10);
  });
});
