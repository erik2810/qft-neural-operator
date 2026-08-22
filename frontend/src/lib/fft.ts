/**
 * Real FFT for the spectral layers.
 *
 * `torch.fft` has no ONNX lowering, so the exported model keeps its learned weights in
 * Fourier space and the transform happens here instead. The grids are tiny and fixed
 * (64 points), so a cached twiddle-table DFT is both simpler and quite fast enough — a
 * whole forward pass costs a few million flops, well inside a slider's frame budget.
 */

interface Twiddles {
  cos: Float64Array;
  sin: Float64Array;
  bins: number;
}

const CACHE = new Map<number, Twiddles>();

/** cos/sin tables for an N-point transform, indexed `[bin * n + sample]`. */
function twiddles(n: number): Twiddles {
  let entry = CACHE.get(n);
  if (entry) return entry;
  const bins = Math.floor(n / 2) + 1;
  const cos = new Float64Array(bins * n);
  const sin = new Float64Array(bins * n);
  for (let k = 0; k < bins; k += 1) {
    for (let t = 0; t < n; t += 1) {
      const angle = (-2 * Math.PI * k * t) / n;
      cos[k * n + t] = Math.cos(angle);
      sin[k * n + t] = Math.sin(angle);
    }
  }
  entry = { cos, sin, bins };
  CACHE.set(n, entry);
  return entry;
}

/** Number of non-redundant bins in a real transform of length `n`. */
export function rfftBins(n: number): number {
  return Math.floor(n / 2) + 1;
}

/**
 * Real-input DFT of `channels` interleaved signals of length `n`.
 *
 * @param input Flat `[channel][sample]`, length `channels * n`.
 * @returns `re` and `im`, each flat `[channel][bin]`.
 */
export function rfft(
  input: Float32Array | Float64Array,
  channels: number,
  n: number,
): { re: Float64Array; im: Float64Array } {
  const { cos, sin, bins } = twiddles(n);
  const re = new Float64Array(channels * bins);
  const im = new Float64Array(channels * bins);
  for (let c = 0; c < channels; c += 1) {
    const base = c * n;
    for (let k = 0; k < bins; k += 1) {
      let sumRe = 0;
      let sumIm = 0;
      const row = k * n;
      for (let t = 0; t < n; t += 1) {
        const x = input[base + t];
        sumRe += x * cos[row + t];
        sumIm += x * sin[row + t];
      }
      re[c * bins + k] = sumRe;
      im[c * bins + k] = sumIm;
    }
  }
  return { re, im };
}

/**
 * Inverse of {@link rfft}: reconstruct `channels` real signals of length `n`.
 *
 * Mirrors `torch.fft.irfft`, including the 1/n normalization and the Hermitian
 * reconstruction — bins other than DC and (for even `n`) Nyquist are counted twice.
 */
export function irfft(
  re: Float64Array,
  im: Float64Array,
  channels: number,
  n: number,
): Float64Array {
  const { cos, sin, bins } = twiddles(n);
  const out = new Float64Array(channels * n);
  const hasNyquist = n % 2 === 0;
  for (let c = 0; c < channels; c += 1) {
    const offset = c * bins;
    for (let t = 0; t < n; t += 1) {
      let total = 0;
      for (let k = 0; k < bins; k += 1) {
        const row = k * n + t;
        // The tables hold θ = −2πkt/n, so exp(+2πikt/n) = cos θ − i·sin θ and
        // Re[(re + i·im)(cos θ − i·sin θ)] = re·cos θ + im·sin θ.
        const term = re[offset + k] * cos[row] + im[offset + k] * sin[row];
        const edge = k === 0 || (hasNyquist && k === bins - 1);
        total += edge ? term : 2 * term;
      }
      out[c * n + t] = total / n;
    }
  }
  return out;
}
