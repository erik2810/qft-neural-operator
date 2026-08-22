import { useMemo, useState } from "react";
import { Formula } from "../components/Formula";
import { Panel, Readout } from "../components/Panel";
import { COLORS, Legend, Plot } from "../components/Plot";
import {
  anomalousDimension,
  freeDimension,
  logRGrid,
  resummedLogCorrelator,
  type Background,
  type Theory,
} from "../lib/physics";

/** Scales at which the same theory is quoted; every curve must land on top of the others. */
const SCALES = [-2, -1, 0, 1, 2];

export function RGInvariance({
  theory,
  background,
}: {
  theory: Theory;
  background: Background;
}) {
  const [epsilon, setEpsilon] = useState(0);
  const logR = useMemo(() => logRGrid(160), []);
  const free = freeDimension(background);

  /**
   * One curve per renormalization scale, with the coupling transported to that scale.
   *
   * That transport is the point. Quoting λ at a different M without running it is a
   * different theory; running it is the *same* theory in a different scheme, and the
   * correlator cannot tell the difference. With β = −ελ the flow map is λ(M') =
   * λ(M)e^{−ε·Δlog M}, and the curves coincide exactly for every ε.
   */
  const curves = useMemo(
    () =>
      SCALES.map((logM) => {
        const transported: Theory = {
          ...theory,
          logM,
          coupling: theory.coupling * Math.exp(-epsilon * logM),
        };
        const { logW } = resummedLogCorrelator(logR, transported, background, epsilon);
        const stripped = new Float64Array(logR.length);
        for (let i = 0; i < logR.length; i += 1) stripped[i] = logW[i] + 2 * free * logR[i];
        return { logM, stripped, coupling: transported.coupling };
      }),
    [theory, background, logR, epsilon, free],
  );

  /** Largest disagreement between any two scales — the residual of the CS equation. */
  const spread = useMemo(() => {
    let worst = 0;
    for (let i = 0; i < logR.length; i += 1) {
      let low = Infinity;
      let high = -Infinity;
      for (const curve of curves) {
        low = Math.min(low, curve.stripped[i]);
        high = Math.max(high, curve.stripped[i]);
      }
      worst = Math.max(worst, high - low);
    }
    return worst;
  }, [curves, logR]);

  /** The same theory with M ignored rather than run — the control that *does* move. */
  const naive = useMemo(
    () =>
      SCALES.map((logM) => {
        const { logW } = resummedLogCorrelator(
          logR, { ...theory, logM }, background, epsilon,
        );
        const stripped = new Float64Array(logR.length);
        for (let i = 0; i < logR.length; i += 1) stripped[i] = logW[i] + 2 * free * logR[i];
        return stripped;
      }),
    [theory, background, logR, epsilon, free],
  );

  const gamma = anomalousDimension(theory, background);
  const running = curves.map((c) => c.coupling);

  return (
    <Panel
      title="Renormalization-group invariance"
      subtitle={
        <>
          <Formula tex="\left(M\partial_M + \beta(\lambda)\partial_\lambda\right)W = 0" />. Five
          curves, the same theory quoted at five different scales{" "}
          <Formula tex="M" />, with the coupling transported along the flow. They coincide
          exactly — for any <Formula tex="\beta" />, not just to leading order — because the
          correlator is built from <Formula tex="\bar\lambda(1/r)" />, which the flow's group
          property makes independent of <Formula tex="M" />.
        </>
      }
      aside={<span className="text-slate-500">5 scales overlaid</span>}
    >
      <Plot
        series={[
          ...naive.map((y, i) => ({
            x: logR,
            y,
            color: COLORS.muted,
            label: `naive ${i}`,
            width: 1,
            dashed: true,
          })),
          ...curves.map((curve) => ({
            x: logR,
            y: curve.stripped,
            color: COLORS.exact,
            label: `M ${curve.logM}`,
            width: 1.4,
          })),
        ]}
        height={210}
        xLabel="log r"
        yLabel="log W + 2Δβ₁β₂ log r"
        zeroLine
      />
      <Legend
        items={[
          { color: COLORS.exact, label: "coupling transported along the flow (5 curves)" },
          { color: COLORS.muted, label: "M changed, coupling left alone", dashed: true },
        ]}
      />

      <label className="mt-4 flex flex-col gap-1 text-xs text-slate-400">
        <span>
          classical deficit <Formula tex="\epsilon" /> in{" "}
          <Formula tex="\beta(\lambda) = -\epsilon\lambda" /> = {epsilon.toFixed(2)}
          {epsilon === 0 && " (marginal — the coupling does not run)"}
        </span>
        <input
          type="range"
          min={0}
          max={0.6}
          step={0.01}
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
          className="accent-sky-400"
        />
      </label>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout
          label="spread across scales"
          value={spread.toExponential(2)}
          hint="Callan-Symanzik residual"
        />
        <Readout label="γ at the reference scale" value={gamma.toExponential(3)} />
        <Readout
          label="λ̄ range over the scales"
          value={`${Math.min(...running).toFixed(4)} … ${Math.max(...running).toFixed(4)}`}
        />
        <Readout
          label="Δ_eff at r = 1"
          value={(free - gamma).toFixed(6)}
        />
      </div>
    </Panel>
  );
}
