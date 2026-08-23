import { useMemo, useState } from "react";
import { Control, Figure, MarginValue, Sym } from "../components/Figure";
import { Formula } from "../components/Formula";
import { SurfaceStage, type SurfaceLayer } from "../components/SurfaceStage";
import { contactIntegral, measuredLogCoefficient } from "../lib/bulk";
import { cDeltaCft, logCoefficient, type Background } from "../lib/physics";
import { PALETTE } from "../lib/stage";
import { sampleField, withValueRange } from "../lib/surface";

const GRID = 256;
/**
 * Decades of z shown below and above the separation.
 *
 * Bounded by resolution, not taste. The ridges are O(z) wide, so a window reaching three
 * decades below r puts them twenty times under a grid cell and the surface flattens into a
 * featureless mound. One and a half decades keeps the narrowest ridge above the sample
 * spacing while still showing it form.
 */
const BELOW = 1.5;
const ABOVE = 0.7;
const FLOOR_DECADES = 3;

export function BulkSurface({ background, number }: { background: Background; number: number }) {
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
    // Height is log(z * density), not log(density). The vertical axis is log z and the bulk
    // measure is dz/z, so with the extra factor of z the volume under this surface *is* the
    // contact integral -- and each ridge must hold its area as it narrows, so it climbs.
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

  const cutoffAt = (logEps - logZMin) / (logZMax - logZMin);
  const marker = cutoffAt >= 0 && cutoffAt <= 1 ? ({ axis: "y", at: cutoffAt } as const) : null;

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
    <Figure
      number={number}
      title="The divergence has a shape"
      caption={
        <>
          <Formula tex="z\sqrt{g}\,K_\Delta(x;p_1)K_\Delta(x;p_2)" /> as relief over the
          Poincaré half-plane, banded one colour per decade. The vertical axis is{" "}
          <Formula tex="\log z" /> and the measure is <Formula tex="dz/z" />, so the volume
          under this surface <em>is</em> the contact integral. Toward the boundary — the lit
          rail — the two ridges at <Formula tex="p=\mp r/2" /> narrow as{" "}
          <Formula tex="z" /> while holding their area, so they climb. The plane is the
          cutoff <Formula tex="\epsilon" />; only what lies beyond it is counted. Drag to
          orbit.
        </>
      }
      controls={
        <div className="grid gap-x-10 gap-y-1 sm:grid-cols-2">
          <Control
            label={<>cutoff log <Sym>ε</Sym></>}
            value={logEps}
            min={-8}
            max={-0.5}
            step={0.05}
            onChange={setLogEps}
          />
          <Control
            label={<>separation <Sym>r</Sym></>}
            value={r}
            min={0.2}
            max={4}
            step={0.02}
            onChange={setR}
          />
        </div>
      }
      margin={
        <>
          <MarginValue
            label={<>measured <Sym>C_log</Sym></>}
            value={measured.toFixed(6)}
            note="from quadrature, in the browser"
            tone="exact"
          />
          <MarginValue label={<>analytic <Sym>2L²c_Δ</Sym></>} value={analytic.toFixed(6)} />
          <MarginValue
            label="relative error"
            value={Math.abs(measured / analytic - 1).toExponential(1)}
            note="O(ε) — pull the cutoff down and watch it fall"
          />
          <MarginValue label={<>integral over <Sym>z &gt; ε</Sym></>} value={integral.toFixed(5)} />
          <MarginValue
            label="renderer"
            value={backend ?? "…"}
            note={`p ∈ [−${width.toFixed(1)}, ${width.toFixed(1)}] · z ∈ [${Math.exp(logZMin).toExponential(1)}, ${Math.exp(logZMax).toExponential(1)}]`}
          />
        </>
      }
    >
      <SurfaceStage
        layers={layers}
        boundary="yMax"
        marker={marker}
        height={380}
        onBackend={setBackend}
      />
    </Figure>
  );
}
