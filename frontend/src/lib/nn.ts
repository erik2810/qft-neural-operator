/**
 * Minimal tensor primitives for the browser forward pass.
 *
 * Batch size is always one here — the page evaluates a single theory at a time — so the
 * batch axis is dropped everywhere and buffers are flat row-major `Float64Array`s with
 * shapes carried alongside. That keeps the port close enough to the PyTorch source to
 * check line by line, which matters more than generality for ~500 lines of inference.
 */

import { irfft, rfft, rfftBins } from "./fft";

/** Error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7 — below float32 eps). */
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return sign * y;
}

/** Exact GELU, matching `torch.nn.GELU()`: 0.5·x·(1 + erf(x/√2)). Applied in place. */
export function gelu(x: Float64Array): Float64Array {
  for (let i = 0; i < x.length; i += 1) x[i] = 0.5 * x[i] * (1 + erf(x[i] * Math.SQRT1_2));
  return x;
}

/**
 * Affine map over the last axis: `(rows, inDim) → (rows, outDim)`.
 *
 * @param weight PyTorch layout `(outDim, inDim)`.
 */
export function linear(
  x: Float64Array,
  rows: number,
  inDim: number,
  weight: Float32Array,
  bias: Float32Array | null,
  outDim: number,
): Float64Array {
  const out = new Float64Array(rows * outDim);
  for (let r = 0; r < rows; r += 1) {
    const xb = r * inDim;
    const ob = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      let total = bias ? bias[o] : 0;
      const wb = o * inDim;
      for (let i = 0; i < inDim; i += 1) total += x[xb + i] * weight[wb + i];
      out[ob + o] = total;
    }
  }
  return out;
}

/** LayerNorm over the last axis, `eps = 1e-5` as PyTorch defaults. Applied in place. */
export function layerNorm(
  x: Float64Array,
  rows: number,
  dim: number,
  weight: Float32Array,
  bias: Float32Array,
  eps = 1e-5,
): Float64Array {
  for (let r = 0; r < rows; r += 1) {
    const base = r * dim;
    let mean = 0;
    for (let i = 0; i < dim; i += 1) mean += x[base + i];
    mean /= dim;
    let variance = 0;
    for (let i = 0; i < dim; i += 1) {
      const d = x[base + i] - mean;
      variance += d * d;
    }
    variance /= dim;
    const scale = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < dim; i += 1) {
      x[base + i] = (x[base + i] - mean) * scale * weight[i] + bias[i];
    }
  }
  return x;
}

/**
 * GroupNorm over `(channels, length)`, matching `torch.nn.GroupNorm`.
 *
 * Statistics are pooled across every channel in a group *and* the whole length axis,
 * which is what makes the FNO blocks insensitive to the overall scale of the field.
 */
export function groupNorm(
  x: Float64Array,
  channels: number,
  length: number,
  groups: number,
  weight: Float32Array,
  bias: Float32Array,
  eps = 1e-5,
): Float64Array {
  const perGroup = channels / groups;
  for (let g = 0; g < groups; g += 1) {
    const start = g * perGroup;
    let mean = 0;
    for (let c = start; c < start + perGroup; c += 1) {
      for (let n = 0; n < length; n += 1) mean += x[c * length + n];
    }
    mean /= perGroup * length;
    let variance = 0;
    for (let c = start; c < start + perGroup; c += 1) {
      for (let n = 0; n < length; n += 1) {
        const d = x[c * length + n] - mean;
        variance += d * d;
      }
    }
    variance /= perGroup * length;
    const scale = 1 / Math.sqrt(variance + eps);
    for (let c = start; c < start + perGroup; c += 1) {
      for (let n = 0; n < length; n += 1) {
        x[c * length + n] = (x[c * length + n] - mean) * scale * weight[c] + bias[c];
      }
    }
  }
  return x;
}

/** 1×1 convolution over `(inChannels, length)`, i.e. a per-position affine map. */
export function conv1x1(
  x: Float64Array,
  inChannels: number,
  length: number,
  weight: Float32Array,
  bias: Float32Array | null,
  outChannels: number,
): Float64Array {
  const out = new Float64Array(outChannels * length);
  for (let o = 0; o < outChannels; o += 1) {
    const wb = o * inChannels;
    const base = bias ? bias[o] : 0;
    for (let n = 0; n < length; n += 1) out[o * length + n] = base;
    for (let i = 0; i < inChannels; i += 1) {
      const w = weight[wb + i];
      if (w === 0) continue;
      for (let n = 0; n < length; n += 1) out[o * length + n] += w * x[i * length + n];
    }
  }
  return out;
}

/**
 * Spectral convolution: transform, mix the lowest `modes`, transform back.
 *
 * Mirrors `SpectralConv1d.forward`. Mode truncation is the regularizer — everything
 * above `modes` is discarded rather than passed through.
 *
 * @param weight Complex weights, PyTorch layout `(inChannels, outChannels, modes, 2)`.
 */
export function spectralConv(
  x: Float64Array,
  inChannels: number,
  outChannels: number,
  length: number,
  weight: Float32Array,
  modes: number,
): Float64Array {
  const bins = rfftBins(length);
  const kept = Math.min(modes, bins);
  const { re, im } = rfft(x, inChannels, length);

  const outRe = new Float64Array(outChannels * bins);
  const outIm = new Float64Array(outChannels * bins);
  for (let o = 0; o < outChannels; o += 1) {
    for (let k = 0; k < kept; k += 1) {
      let sumRe = 0;
      let sumIm = 0;
      for (let i = 0; i < inChannels; i += 1) {
        const wIndex = 2 * ((i * outChannels + o) * modes + k);
        const wRe = weight[wIndex];
        const wIm = weight[wIndex + 1];
        const xRe = re[i * bins + k];
        const xIm = im[i * bins + k];
        sumRe += xRe * wRe - xIm * wIm;
        sumIm += xRe * wIm + xIm * wRe;
      }
      outRe[o * bins + k] = sumRe;
      outIm[o * bins + k] = sumIm;
    }
  }
  return irfft(outRe, outIm, outChannels, length);
}

/**
 * Differentiable-free linear interpolation of a channel field on a uniform grid.
 *
 * Mirrors `interp1d_uniform`; queries outside `[lo, hi]` clamp to the edges.
 *
 * @param values Flat `(channels, gridSize)`.
 * @returns Flat `(queries, channels)` — note the transpose, matching the PyTorch version.
 */
export function interp1dUniform(
  values: Float64Array,
  channels: number,
  gridSize: number,
  lo: number,
  hi: number,
  query: Float64Array,
): Float64Array {
  const out = new Float64Array(query.length * channels);
  const spacing = (hi - lo) / (gridSize - 1);
  for (let q = 0; q < query.length; q += 1) {
    const clamped = Math.min(Math.max(query[q], lo), hi);
    const position = Math.min(Math.max((clamped - lo) / spacing, 0), gridSize - 1);
    const left = Math.min(Math.floor(position), gridSize - 1);
    const right = Math.min(left + 1, gridSize - 1);
    const t = position - left;
    for (let c = 0; c < channels; c += 1) {
      const a = values[c * gridSize + left];
      const b = values[c * gridSize + right];
      out[q * channels + c] = a + t * (b - a);
    }
  }
  return out;
}
