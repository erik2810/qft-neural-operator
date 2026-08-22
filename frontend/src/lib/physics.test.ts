import { describe, expect, it } from "vitest";
import { loadParity } from "./__fixtures__/load";
import {
  anomalousDimension,
  anomalousDimensionFromCorrelator,
  cDeltaCft,
  defaultBackground,
  deltaFromMass,
  fitLine,
  freeDimension,
  gamma,
  logCoefficient,
  logRGrid,
  resummedLogCorrelator,
  type Background,
  type Family,
  type Theory,
} from "./physics";

const parity = loadParity();

/** The background the fixture was generated against. */
function fixtureBackground(sigmaSq = 0): Background {
  const bg = parity.background;
  return {
    L: bg.L,
    mSq: bg.m_sq,
    delta: bg.delta,
    cDelta: bg.c_delta,
    normalizationFactor: bg.normalization_factor,
    beta1: 1,
    beta2: 1,
    sigmaSq,
  };
}

describe("special functions", () => {
  it("matches Python's math.gamma", () => {
    for (const { x, value } of parity.gamma_function) {
      expect(gamma(x)).toBeCloseTo(value, 10);
    }
  });
});

describe("conformal data", () => {
  it("solves Delta(Delta - 1) = m^2 L^2", () => {
    for (const mSq of [0, 0.75, 2, 6]) {
      const d = deltaFromMass(mSq, 1);
      expect(d * (d - 1)).toBeCloseTo(mSq, 12);
    }
  });

  it("rejects a mass below the Breitenlohner-Freedman bound", () => {
    expect(() => deltaFromMass(-0.3, 1)).toThrow(/Breitenlohner/);
  });

  it("reproduces c_Delta = 1/2 at Delta = 3/2", () => {
    expect(cDeltaCft(1.5)).toBeCloseTo(0.5, 12);
    expect(() => cDeltaCft(0.5)).toThrow();
  });

  it("agrees with the Python background", () => {
    const bg = fixtureBackground();
    expect(bg.delta).toBeCloseTo(parity.background.delta, 12);
    expect(logCoefficient(bg)).toBeCloseTo(parity.background.log_coefficient, 12);
    expect(freeDimension(bg)).toBeCloseTo(parity.background.free_dimension, 12);
  });

  it("defaults to the reference background", () => {
    expect(defaultBackground().delta).toBeCloseTo(1.5, 12);
  });
});

describe("anomalous dimension", () => {
  it("matches the Python functional for every analytic family and smearing", () => {
    for (const c of parity.gammas) {
      const theory: Theory = {
        family: c.family as Family,
        coupling: c.coupling,
        xi: c.xi,
        seed: 0,
        logM: 0,
      };
      expect(anomalousDimension(theory, fixtureBackground(c.sigma_sq))).toBeCloseTo(c.gamma, 12);
    }
  });

  it("vanishes for the free theory and for a normal-ordered quartic", () => {
    const bg = fixtureBackground(0);
    expect(anomalousDimension({ family: "free", coupling: 0, xi: 1, seed: 0, logM: 0 }, bg)).toBe(0);
    expect(
      anomalousDimension({ family: "phi4", coupling: 0.05, xi: 1, seed: 0, logM: 0 }, bg),
    ).toBe(0);
  });

  it("is linear in the coupling", () => {
    const bg = fixtureBackground();
    const base = anomalousDimension(
      { family: "sine_gordon", coupling: 0.01, xi: 0.9, seed: 0, logM: 0 }, bg,
    );
    const scaled = anomalousDimension(
      { family: "sine_gordon", coupling: 0.03, xi: 0.9, seed: 0, logM: 0 }, bg,
    );
    expect(scaled).toBeCloseTo(3 * base, 14);
  });

  it("is reproducible for the seeded families", () => {
    const bg = fixtureBackground();
    const theory: Theory = { family: "gp_fourier", coupling: 0.02, xi: 1, seed: 7, logM: 0 };
    expect(anomalousDimension(theory, bg)).toBe(anomalousDimension({ ...theory }, bg));
    expect(anomalousDimension(theory, bg)).not.toBe(
      anomalousDimension({ ...theory, seed: 8 }, bg),
    );
  });
});

describe("correlators", () => {
  it("matches the Python log W curve", () => {
    for (const c of parity.correlators) {
      const theory: Theory = {
        family: c.family as Family,
        coupling: c.coupling,
        xi: c.xi,
        seed: 0,
        logM: 0,
      };
      const { logW } = resummedLogCorrelator(Float64Array.from(c.log_r), theory, fixtureBackground());
      for (let i = 0; i < c.log_w.length; i += 1) {
        expect(logW[i]).toBeCloseTo(c.log_w[i], 11);
      }
    }
  });

  it("is exactly RG-invariant when the coupling is marginal", () => {
    const bg = fixtureBackground();
    const logR = logRGrid(16);
    const a = resummedLogCorrelator(logR, { family: "sine_gordon", coupling: 0.02, xi: 0.8, seed: 0, logM: 0 }, bg);
    const b = resummedLogCorrelator(logR, { family: "sine_gordon", coupling: 0.02, xi: 0.8, seed: 0, logM: 2.5 }, bg);
    for (let i = 0; i < logR.length; i += 1) expect(a.logW[i]).toBe(b.logW[i]);
  });

  it("stays RG-invariant when the coupling runs", () => {
    const bg = fixtureBackground();
    const logR = logRGrid(16);
    const epsilon = 0.35;
    const base: Theory = { family: "sine_gordon", coupling: 0.02, xi: 0.8, seed: 0, logM: 0 };
    const shift = 1.4;
    // Quoting the same theory at a different M means transporting the coupling with it.
    const moved: Theory = { ...base, coupling: base.coupling * Math.exp(-epsilon * shift), logM: shift };
    const a = resummedLogCorrelator(logR, base, bg, epsilon);
    const b = resummedLogCorrelator(logR, moved, bg, epsilon);
    for (let i = 0; i < logR.length; i += 1) expect(a.logW[i]).toBeCloseTo(b.logW[i], 12);
  });

  it("recovers gamma from its own correlator", () => {
    const bg = fixtureBackground();
    const logR = logRGrid(64);
    const theory: Theory = { family: "sine_gordon", coupling: 0.03, xi: 1.1, seed: 0, logM: 0 };
    const { logW } = resummedLogCorrelator(logR, theory, bg);
    expect(anomalousDimensionFromCorrelator(logR, logW, freeDimension(bg))).toBeCloseTo(
      anomalousDimension(theory, bg), 12,
    );
  });
});

describe("line fitting", () => {
  it("recovers a known slope and intercept", () => {
    const x = [-2, -1, 0, 1, 2];
    const y = x.map((v) => -3 * v + 0.5);
    const { slope, intercept } = fitLine(x, y);
    expect(slope).toBeCloseTo(-3, 12);
    expect(intercept).toBeCloseTo(0.5, 12);
  });
});
