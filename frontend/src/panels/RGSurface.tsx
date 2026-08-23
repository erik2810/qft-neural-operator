import { useMemo, useState } from "react";
import { Formula } from "../components/Formula";
import { Panel, Readout } from "../components/Panel";
import { SurfaceStage, type SurfaceLayer } from "../components/SurfaceStage";
import {
  anomalousDimension,
  freeDimension,
  type Background,
  type Theory,
} from "../lib/physics";
import { PALETTE } from "../lib/stage";
import { sampleField, withValueRange, type ScalarField } from "../lib/surface";

const NX = 160;
const NY = 96;
const LOG_R: [number, number] = [Math.log(0.05), Math.log(12)];
const LOG_M: [number, number] = [-2.5, 2.5];

/** Put two fields on one scale, or the comparison between them is meaningless. */
function share(a: ScalarField, b: ScalarField): [ScalarField, ScalarField] {
  const low = Math.min(a.valueRange[0], b.valueRange[0]);
  const high = Math.max(a.valueRange[1], b.valueRange[1]);
  return [withValueRange(a, low, high), withValueRange(b, low, high)];
}

export function RGSurface({
  theory,
  background,
}: {
  theory: Theory;
  background: Background;
}) {
  const [epsilon, setEpsilon] = useState(0.35);
  const free = freeDimension(background);
  const gamma = anomalousDimension(theory, background);

  const { transported, naive, spread } = useMemo(() => {
    // Transported: the coupling is carried along the flow to each M, so gamma at the
    // physical scale 1/r comes out independent of M and the surface is ruled -- flat in
    // the M direction, for any beta.
    const carried = sampleField(NX, NY, LOG_R, LOG_M, (logR) => {
      const running = gamma * Math.exp(epsilon * logR);
      return 2 * running * logR;
    });
    // Naive: M is changed and the coupling is left where it was. Same theory in name only.
    const fixed = sampleField(NX, NY, LOG_R, LOG_M, (logR, logM) => {
      const running = gamma * Math.exp(epsilon * (logM + logR));
      return 2 * running * logR;
    });

    // How far the naive surface departs from flatness along M, in the units of the plot.
    let worst = 0;
    for (let i = 0; i < NX; i += 1) {
      let low = Number.POSITIVE_INFINITY;
      let high = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < NY; j += 1) {
        const v = fixed.values[j * NX + i];
        if (v < low) low = v;
        if (v > high) high = v;
      }
      worst = Math.max(worst, high - low);
    }
    const [a, b] = share(carried, fixed);
    return { transported: a, naive: b, spread: worst };
  }, [gamma, epsilon]);

  const layers: SurfaceLayer[] = useMemo(
    () => [
      { key: "transported", field: transported, bands: 6, low: PALETTE.slate, high: PALETTE.sodium },
      { key: "naive", field: naive, bands: 6, low: PALETTE.slate, high: PALETTE.ion, overlay: true },
    ],
    [transported, naive],
  );

  return (
    <Panel
      title="Renormalization-group invariance"
      subtitle={
        <>
          <Formula tex="\log W + 2\Delta\beta_1\beta_2\log r" /> over{" "}
          <Formula tex="(\log r,\ \log M)" />. The renormalization scale is a gauge choice,
          so the physical statement is that <em>the surface has no slope along it</em>:
          carry the coupling along the flow and the amber sheet comes out ruled — sight
          down the <Formula tex="M" /> axis and it is a straight line. The teal overlay
          changes <Formula tex="M" /> and leaves the coupling where it was; it visibly
          tilts. That is the whole content of{" "}
          <Formula tex="\left(M\partial_M + \beta(\lambda)\partial_\lambda\right)W = 0" />.
        </>
      }
      aside={
        <span className="font-mono text-[0.65rem] tracking-wide text-[var(--dim)] uppercase">
          amber ruled · teal tilts
        </span>
      }
    >
      <SurfaceStage layers={layers} height={320} />
      <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-[var(--dim)]">
        <span>log r ∈ [{LOG_R[0].toFixed(1)}, {LOG_R[1].toFixed(1)}]</span>
        <span>log M ∈ [{LOG_M[0].toFixed(1)}, {LOG_M[1].toFixed(1)}] — the gauge direction</span>
      </div>

      <label className="mt-4 flex flex-col gap-1 text-xs text-[var(--dim)]">
        <span>
          classical deficit <Formula tex="\epsilon" /> in{" "}
          <Formula tex="\beta(\lambda) = -\epsilon\lambda" /> = {epsilon.toFixed(2)}
          {epsilon === 0 && " — marginal: nothing runs, so nothing can be got wrong"}
        </span>
        <input
          type="range"
          min={0}
          max={0.6}
          step={0.01}
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
        />
      </label>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout label="γ at the reference scale" value={gamma.toExponential(3)} />
        <Readout label="Δ_eff at r = 1" value={(free - gamma).toFixed(6)} />
        <Readout
          label="tilt of the naive sheet"
          value={spread.toExponential(2)}
          hint="range along M"
        />
        <Readout label="tilt of the transported sheet" value="0" hint="exactly, by construction" />
      </div>
    </Panel>
  );
}
