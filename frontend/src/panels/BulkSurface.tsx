import { useMemo, useState } from "react";
import { Formula } from "../components/Formula";
import { Panel, Readout } from "../components/Panel";
import { SurfaceStage, type SurfaceLayer } from "../components/SurfaceStage";
import { contactIntegral, measuredLogCoefficient } from "../lib/bulk";
import { cDeltaCft, logCoefficient, type Background } from "../lib/physics";
import { PALETTE } from "../lib/stage";
import { sampleField, withValueRange } from "../lib/surface";

const GRID = 256;
/**
 * Decades of z shown below and above the separation.
 *
 * Bounded by resolution, not by taste. The ridges are O(z) wide, so a window reaching
 * three decades below r puts them twenty times under a grid cell and the surface flattens
 * into a featureless mound -- the same failure the flat heatmap had when it was anchored
 * at the cutoff. One and a half decades keeps the narrowest ridge above the sample spacing
 * while still showing it form.
 */
const BELOW = 1.5;
const ABOVE = 0.7;
/** Dynamic range retained under the peak; the density spans far more than a surface can. */
const FLOOR_DECADES = 3;

export function BulkSurface({ background }: { background: Background }) {
  const [logEps, setLogEps] = useState(-4);
  const [r, setR] = useState(1);
  const [backend, setBackend] = useState<"webgpu" | "webgl" | null>(null);

  const logZMin = Math.log(r) - BELOW * Math.LN10;
  const logZMax = Math.log(r) + ABOVE * Math.LN10;
  const width = 2 * r;

  const field = useMemo(() => {
    const d = background.delta;
    const half = 0.5 * r;
    const c = cDeltaCft(d);
    const prefactor = Math.log((background.L ** 2 * c * c) / background.normalizationFactor);
    // Height is log(z * density), not log(density).
    //
    // The vertical axis is log z and the bulk measure is dz/z = d(log z), so with the
    // extra factor of z the volume under this surface *is* the contact integral. It also
    // makes the divergence legible: each ridge keeps a constant area per decade while its
    // width falls as z, so it must grow taller toward the boundary. The logarithm is then
    // literally a sum over decades of equal-area slices.
    const raw = sampleField(GRID, GRID, [-width, width], [logZMin, logZMax], (p, logZ) => {
      const zz = Math.exp(2 * logZ);
      return (
        prefactor +
        (2 * d - 1) * logZ -
        d * Math.log(zz + (p + half) ** 2) -
        d * Math.log(zz + (p - half) ** 2)
      );
    });
    const peak = raw.valueRange[1];
    return withValueRange(raw, peak - FLOOR_DECADES * Math.LN10, peak);
  }, [r, width, logZMin, logZMax, background]);

  const layers: SurfaceLayer[] = useMemo(
    () => [{ key: "density", field, bands: FLOOR_DECADES, low: PALETTE.slate, high: PALETTE.sodium }],
    [field],
  );

  // Where the cutoff falls inside the displayed z window, as a fraction.
  const cutoffAt = (logEps - logZMin) / (logZMax - logZMin);
  const marker =
    cutoffAt >= 0 && cutoffAt <= 1 ? ({ axis: "y", at: cutoffAt } as const) : null;

  const integral = useMemo(
    () => contactIntegral(r, Math.exp(logEps), background),
    [r, logEps, background],
  );
  const measured = useMemo(
    () => measuredLogCoefficient(r, Math.exp(logEps), background),
    [r, logEps, background],
  );
  const analytic = logCoefficient(background);

  return (
    <Panel
      title="The bulk integrand"
      subtitle={
        <>
          <Formula tex="z\sqrt{g}\,K_\Delta(x;p_1)K_\Delta(x;p_2)" /> as relief over the
          Poincaré half-plane, banded one colour per decade. The vertical axis is{" "}
          <Formula tex="\log z" /> and the measure is <Formula tex="dz/z" />, so the volume
          under this surface <em>is</em> the contact integral. Toward the boundary — the lit
          rail — the two ridges at <Formula tex="p=\mp r/2" /> narrow as{" "}
          <Formula tex="z" /> while holding their area, so they climb: the divergence is a
          sum over decades of equal-area slices. The plane is the cutoff{" "}
          <Formula tex="\epsilon" />; only what lies beyond it is counted.
        </>
      }
      aside={
        <span className="font-mono text-[0.65rem] tracking-wide text-[var(--dim)] uppercase">
          {backend ?? "…"}
        </span>
      }
    >
      <SurfaceStage
        layers={layers}
        boundary="yMax"
        marker={marker}
        height={340}
        onBackend={setBackend}
      />
      <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-[var(--dim)]">
        <span>p ∈ [{(-width).toFixed(2)}, {width.toFixed(2)}]</span>
        <span>
          z ∈ [{Math.exp(logZMin).toExponential(1)}, {Math.exp(logZMax).toExponential(1)}] ·
          boundary at the lit edge
        </span>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs text-[var(--dim)]">
          <span>
            cutoff <Formula tex="\log\epsilon" /> = {logEps.toFixed(2)}
          </span>
          <input
            type="range"
            min={-8}
            max={-0.5}
            step={0.05}
            value={logEps}
            onChange={(e) => setLogEps(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-[var(--dim)]">
          <span>
            separation <Formula tex="r" /> = {r.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.2}
            max={4}
            step={0.02}
            value={r}
            onChange={(e) => setR(Number(e.target.value))}
          />
        </label>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout label="∫ over z > ε" value={integral.toFixed(5)} />
        <Readout label="measured C_log" value={measured.toFixed(6)} hint="dĨ / d log(1/ε)" />
        <Readout label="analytic 2L²c_Δ" value={analytic.toFixed(6)} />
        <Readout
          label="relative error"
          value={Math.abs(measured / analytic - 1).toExponential(1)}
          hint="O(ε) — drag ε down"
        />
      </div>
    </Panel>
  );
}
