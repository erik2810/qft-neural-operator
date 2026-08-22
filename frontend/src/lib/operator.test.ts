import { describe, expect, it } from "vitest";
import { loadManifest, loadParity, loadWeights } from "./__fixtures__/load";
import { Operator } from "./operator";
import { erf, gelu, interp1dUniform, layerNorm, linear } from "./nn";
import { irfft, rfft, rfftBins } from "./fft";

const parity = loadParity();
const operator = new Operator(loadManifest(), loadWeights());

describe("fft", () => {
  it("round-trips a real signal", () => {
    const n = 64;
    const x = Float64Array.from({ length: n }, (_, i) => Math.sin(0.3 * i) + 0.4 * Math.cos(1.7 * i));
    const { re, im } = rfft(x, 1, n);
    const back = irfft(re, im, 1, n);
    for (let i = 0; i < n; i += 1) expect(back[i]).toBeCloseTo(x[i], 10);
  });

  it("puts a pure tone in a single bin", () => {
    const n = 32;
    const k = 5;
    const x = Float64Array.from({ length: n }, (_, t) => Math.cos((2 * Math.PI * k * t) / n));
    const { re, im } = rfft(x, 1, n);
    for (let bin = 0; bin < rfftBins(n); bin += 1) {
      const magnitude = Math.hypot(re[bin], im[bin]);
      expect(magnitude).toBeCloseTo(bin === k ? n / 2 : 0, 8);
    }
  });

  it("transforms channels independently", () => {
    const n = 16;
    const x = new Float64Array(2 * n);
    for (let i = 0; i < n; i += 1) {
      x[i] = i;
      x[n + i] = -2 * i;
    }
    const { re, im } = rfft(x, 2, n);
    const back = irfft(re, im, 2, n);
    for (let i = 0; i < n; i += 1) {
      expect(back[i]).toBeCloseTo(i, 9);
      expect(back[n + i]).toBeCloseTo(-2 * i, 9);
    }
  });
});

describe("primitives", () => {
  it("has an error function accurate to better than float32", () => {
    const known: [number, number][] = [
      [0, 0],
      [0.5, 0.5204998778],
      [1, 0.8427007929],
      [2, 0.9953222650],
      [-1.5, -0.9661051465],
    ];
    for (const [x, expected] of known) expect(erf(x)).toBeCloseTo(expected, 6);
  });

  it("matches GELU at reference points", () => {
    const x = Float64Array.from([-3, -1, 0, 1, 3]);
    const expected = [-0.0040496, -0.15865529, 0, 0.84134471, 2.9959504];
    gelu(x);
    for (let i = 0; i < x.length; i += 1) expect(x[i]).toBeCloseTo(expected[i], 6);
  });

  it("normalizes to zero mean and unit variance", () => {
    const rows = 3;
    const dim = 8;
    const x = Float64Array.from({ length: rows * dim }, (_, i) => Math.sin(i) * 5 + 2);
    const weight = Float32Array.from({ length: dim }, () => 1);
    const bias = new Float32Array(dim);
    layerNorm(x, rows, dim, weight, bias);
    for (let r = 0; r < rows; r += 1) {
      const slice = Array.from(x.slice(r * dim, (r + 1) * dim));
      const mean = slice.reduce((a, b) => a + b, 0) / dim;
      const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / dim;
      expect(mean).toBeCloseTo(0, 9);
      expect(variance).toBeCloseTo(1, 4);
    }
  });

  it("applies an affine map in PyTorch weight layout", () => {
    // weight is (outDim, inDim): row o dots the input.
    const weight = Float32Array.from([1, 2, 3, 4, 5, 6]);
    const bias = Float32Array.from([0.5, -0.5]);
    const out = linear(Float64Array.from([1, 1, 1]), 1, 3, weight, bias, 2);
    expect(Array.from(out)).toEqual([6.5, 14.5]);
  });

  it("interpolates and clamps on a uniform grid", () => {
    const values = Float64Array.from([0, 1, 2, 3, 10, 11, 12, 13]); // 2 channels x 4 nodes
    const out = interp1dUniform(values, 2, 4, 0, 3, Float64Array.from([-5, 0.5, 3, 99]));
    expect(Array.from(out.slice(0, 2))).toEqual([0, 10]); // clamped low
    expect(Array.from(out.slice(2, 4))).toEqual([0.5, 10.5]);
    expect(Array.from(out.slice(4, 6))).toEqual([3, 13]);
    expect(Array.from(out.slice(6, 8))).toEqual([3, 13]); // clamped high
  });
});

describe("operator", () => {
  it("loads the exported manifest", () => {
    expect(operator.manifest.format).toBe("qft-operator-weights");
    expect(operator.manifest.spectral_form).toBe("fourier");
    expect(operator.manifest.architecture.n_phi).toBe(parity.phi_grid.length);
  });

  it("rejects an incompatible export rather than misreading it", () => {
    const manifest = { ...loadManifest(), spectral_form: "circular" as const };
    expect(() => new Operator(manifest, loadWeights())).toThrow(/fourier/);
    const foreign = { ...loadManifest(), format: "something-else" };
    expect(() => new Operator(foreign, loadWeights())).toThrow(/format/);
  });

  it("reconstructs the physical field grid from the manifest", () => {
    const grid = operator.phiGrid;
    expect(grid.length).toBe(parity.phi_grid.length);
    for (let i = 0; i < grid.length; i += 1) expect(grid[i]).toBeCloseTo(parity.phi_grid[i], 9);
  });

  it("keeps the branch coordinate channel normalized to [-1, 1]", () => {
    const channel = operator.coordinateChannel;
    expect(channel[0]).toBeCloseTo(-1, 6);
    expect(channel[channel.length - 1]).toBeCloseTo(1, 6);
  });

  it("reproduces the PyTorch forward pass", () => {
    for (const c of parity.operator_cases) {
      const predicted = operator.predictLogW(
        Float64Array.from(c.v_phi), Float64Array.from(c.log_r), c.log_m,
      );
      expect(predicted.length).toBe(c.log_w.length);
      for (let i = 0; i < c.log_w.length; i += 1) {
        // PyTorch runs the whole graph in float32; this port keeps float64 activations
        // over float32 weights, so agreement is limited by float32 accumulation.
        expect(predicted[i]).toBeCloseTo(c.log_w[i], 4);
      }
    }
  });

  it("is independent of the query set", () => {
    const c = parity.operator_cases[0];
    const v = Float64Array.from(c.v_phi);
    const full = operator.predictLogW(v, Float64Array.from(c.log_r), c.log_m);
    const subset = operator.predictLogW(
      v, Float64Array.from(c.log_r.filter((_, i) => i % 3 === 0)), c.log_m,
    );
    let k = 0;
    for (let i = 0; i < c.log_r.length; i += 3) {
      expect(subset[k]).toBeCloseTo(full[i], 12);
      k += 1;
    }
  });

  it("responds to the renormalization scale", () => {
    const c = parity.operator_cases[0];
    const v = Float64Array.from(c.v_phi);
    const logR = Float64Array.from(c.log_r);
    const a = operator.predictLogW(v, logR, 0);
    const b = operator.predictLogW(v, logR, 2);
    // An untrained network has not learned M-independence, so the two must differ --
    // if they did not, log M would not be reaching the trunk at all.
    let maxDiff = 0;
    for (let i = 0; i < a.length; i += 1) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
    expect(maxDiff).toBeGreaterThan(1e-6);
  });

  it("validates the field-grid length", () => {
    expect(() => operator.predictLogW(new Float64Array(3), Float64Array.from([0]), 0)).toThrow(
      /field samples/,
    );
  });

  it("applies the training-time feature scale", () => {
    const scaled = operator.scaleFeatures(Float64Array.from([1, 2, 3]));
    const factor = operator.manifest.feature_scale || 1;
    expect(Array.from(scaled)).toEqual([1 / factor, 2 / factor, 3 / factor]);
  });
});
