import { useMemo } from "react";
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
import type { Prediction } from "../lib/prediction";

export function LogResidual({
  theory,
  background,
  prediction,
}: {
  theory: Theory;
  background: Background;
  prediction: Prediction | null;
}) {
  const free = freeDimension(background);
  const logR = useMemo(() => logRGrid(160), []);
  const exact = useMemo(
    () => resummedLogCorrelator(logR, theory, background),
    [logR, theory, background],
  );
  const gammaExact = anomalousDimension(theory, background);

  // Strip the free-theory power law. What is left is 2γ·log r — a straight line whose
  // slope is the whole physical content, and which is invisible on a raw log-log plot
  // because it is ~1e-3 of the leading behaviour.
  const strippedExact = useMemo(() => {
    const out = new Float64Array(logR.length);
    for (let i = 0; i < logR.length; i += 1) out[i] = exact.logW[i] + 2 * free * logR[i];
    return out;
  }, [logR, exact, free]);

  const strippedPredicted = useMemo(() => {
    if (!prediction) return null;
    const out = new Float64Array(prediction.logR.length);
    for (let i = 0; i < out.length; i += 1) {
      out[i] = prediction.logW[i] + 2 * free * prediction.logR[i];
    }
    return out;
  }, [prediction, free]);

  const series = [
    { x: logR, y: strippedExact, color: COLORS.exact, label: "exact" },
    ...(prediction && strippedPredicted
      ? [
          {
            x: prediction.logR,
            y: strippedPredicted,
            color: COLORS.predicted,
            label: "operator",
            dashed: true,
          },
        ]
      : []),
  ];

  const gammaError = prediction ? Math.abs(prediction.gamma - gammaExact) : null;

  return (
    <Panel
      title="Logarithmic residual and γ extraction"
      subtitle={
        <>
          <Formula tex="\log W + 2\Delta\beta_1\beta_2\log r" /> against{" "}
          <Formula tex="\log r" />. Dividing out the free-theory power law leaves a straight
          line of slope <Formula tex="2\gamma" />. Recovering that slope is the whole point
          of the framework: on the raw correlator it is a part in a thousand.
        </>
      }
      aside={
        prediction ? (
          <span className={prediction.source === "server" ? "text-sky-400" : "text-orange-300"}>
            {prediction.source === "server" ? "PyTorch (server)" : "operator (browser)"}
          </span>
        ) : (
          <span className="text-slate-500">no operator loaded</span>
        )
      }
    >
      <Plot
        series={series}
        height={210}
        xLabel="log r"
        yLabel="log W + 2Δβ₁β₂ log r"
        zeroLine
      />
      <Legend
        items={[
          { color: COLORS.exact, label: "exact (closed form)" },
          ...(prediction ? [{ color: COLORS.predicted, label: "operator", dashed: true }] : []),
        ]}
      />

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout label="γ exact" value={gammaExact.toExponential(3)} />
        <Readout
          label="γ recovered"
          value={prediction ? prediction.gamma.toExponential(3) : "—"}
        />
        <Readout
          label="|Δγ|"
          value={gammaError === null ? "—" : gammaError.toExponential(2)}
        />
        <Readout
          label="Δ_eff"
          value={(free - gammaExact).toFixed(6)}
          hint={`free Δβ₁β₂ = ${free.toFixed(3)}`}
        />
      </div>
    </Panel>
  );
}
