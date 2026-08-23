/**
 * A scalar field over a physical plane, rendered as a shaded height surface.
 *
 * All three panels are the same object -- density over $(p, \log z)$, $\log W$ over
 * $(\log r, \log M)$, the stripped residual over $(\log r, \lambda)$ -- so they share one
 * representation and one material rather than three bespoke visualisations.
 *
 * The colour ramp is stepped by decade rather than smooth. That is deliberate: the physics
 * here is the counting of decades of $\log(r/\epsilon)$, and a banded surface lets the
 * reader count them off the geometry instead of estimating against a legend.
 */

/** A scalar field sampled on a regular grid, row-major with `x` fastest. */
export interface ScalarField {
  values: Float32Array;
  nx: number;
  ny: number;
  /** Domain extent, in the physical units of each axis. */
  xRange: [number, number];
  yRange: [number, number];
  /** Value range mapped onto the surface height and the colour ramp. */
  valueRange: [number, number];
}

/** Build a field by evaluating `f` on a regular grid. */
export function sampleField(
  nx: number,
  ny: number,
  xRange: [number, number],
  yRange: [number, number],
  f: (x: number, y: number) => number,
): ScalarField {
  const values = new Float32Array(nx * ny);
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (let j = 0; j < ny; j += 1) {
    const y = yRange[0] + ((yRange[1] - yRange[0]) * j) / Math.max(ny - 1, 1);
    for (let i = 0; i < nx; i += 1) {
      const x = xRange[0] + ((xRange[1] - xRange[0]) * i) / Math.max(nx - 1, 1);
      const v = f(x, y);
      values[j * nx + i] = v;
      if (Number.isFinite(v)) {
        if (v < low) low = v;
        if (v > high) high = v;
      }
    }
  }
  if (!Number.isFinite(low) || !Number.isFinite(high) || low === high) {
    low = 0;
    high = 1;
  }
  return { values, nx, ny, xRange, yRange, valueRange: [low, high] };
}

/** Clamp a field's value range, e.g. to keep a divergence from flattening everything else. */
export function withValueRange(field: ScalarField, low: number, high: number): ScalarField {
  return { ...field, valueRange: [low, high] };
}

/** Normalized position of a physical value within a field's value range. */
export function normalize(field: ScalarField, value: number): number {
  const [low, high] = field.valueRange;
  return (value - low) / (high - low || 1);
}
