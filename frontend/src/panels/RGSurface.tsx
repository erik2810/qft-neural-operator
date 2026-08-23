import { useMemo, useState } from "react";
import { Formula } from "../components/Formula";
import { Control, Figure, MarginValue, Sym } from "../components/Figure";
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
  number,
}: {
  theory: Theory;
  background: Background;
  number: number;
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
    <Figure
      number={number}
      title="A gauge choice, seen as a flat direction"
      caption={
        <>
          <Formula tex="\log W + 2\Delta\beta_1\beta_2\log r" /> over{" "}
          <Formula tex="(\log r,\ \log M)" />. The renormalization scale is a gauge choice,
          so the physical statement is that <em>the surface has no slope along it</em>: carry
          the coupling along the flow and the amber sheet comes out ruled — sight down the{" "}
          <Formula tex="M" /> axis and it is a straight line. The teal overlay changes{" "}
          <Formula tex="M" /> and leaves the coupling where it was; it visibly tilts. That is
          the whole content of{" "}
          <Formula tex="\left(M\partial_M + \beta(\lambda)\partial_\lambda\right)W = 0" />.
        </>
      }
      controls={
        <Control
          label={<>classical deficit <Sym>ε</Sym> in <Sym>β(λ) = −ελ</Sym></>}
          value={epsilon}
          min={0}
          max={0.6}
          step={0.01}
          onChange={setEpsilon}
          hint={epsilon === 0 ? "marginal: nothing runs, so nothing can be got wrong" : undefined}
        />
      }
      margin={
        <>
          <MarginValue
            label="tilt, transported"
            value="0"
            note="exactly, for any β"
            tone="exact"
          />
          <MarginValue
            label="tilt, coupling left alone"
            value={spread.toExponential(2)}
            note="range along the log M axis"
            tone="predicted"
          />
          <MarginValue label={<><Sym>γ</Sym> at the reference scale</>} value={gamma.toExponential(3)} />
          <MarginValue label={<><Sym>Δ_eff</Sym> at <Sym>r = 1</Sym></>} value={(free - gamma).toFixed(6)} />
        </>
      }
    >
      <SurfaceStage layers={layers} height={380} />
    </Figure>
  );
}
