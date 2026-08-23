import { useEffect, useMemo, useState } from "react";
import { Formula } from "../components/Formula";
import { Panel, Readout } from "../components/Panel";
import { SurfaceStage, type SurfaceLayer } from "../components/SurfaceStage";
import {
  anomalousDimension,
  anomalousDimensionFromCorrelator,
  freeDimension,
  type Background,
  type Theory,
} from "../lib/physics";
import type { Operator } from "../lib/operator";
import { potentialSamples, type Prediction } from "../lib/prediction";
import { PALETTE } from "../lib/stage";
import { sampleField, withValueRange, type ScalarField } from "../lib/surface";

const NR = 128;
/**
 * Coupling samples for the predicted sheet.
 *
 * Each one is a separate forward pass, so this is a budget rather than a resolution
 * choice: the exact sheet is free and gets a fine grid, the predicted sheet gets as many
 * rows as fit inside a control change without the slider going sticky.
 */
const N_LAMBDA_EXACT = 96;
const N_LAMBDA_PREDICTED = 20;
const LOG_R: [number, number] = [Math.log(0.05), Math.log(12)];
const LAMBDA: [number, number] = [-0.05, 0.05];

/** Nearest-row expansion of a coarse coupling grid onto the fine one. */
function expandRows(coarse: Float32Array, nx: number, from: number, to: number): Float32Array {
  const out = new Float32Array(nx * to);
  for (let j = 0; j < to; j += 1) {
    const src = Math.min(from - 1, Math.round((j * (from - 1)) / (to - 1)));
    out.set(coarse.subarray(src * nx, (src + 1) * nx), j * nx);
  }
  return out;
}

export function ResidualSurface({
  theory,
  background,
  operator,
  prediction,
}: {
  theory: Theory;
  background: Background;
  operator: Operator | null;
  /**
   * Prediction for the *current* theory only.
   *
   * The sheet needs one forward pass per coupling row, which is not something to send over
   * a socket twenty times; it is always computed in the browser. The headline readout is a
   * single theory, so when a backend is attached its full-precision answer is used there
   * and the panel says which it is showing.
   */
  prediction: Prediction | null;
}) {
  const free = freeDimension(background);
  // Destructured so the effect dependencies name the fields that actually matter; passing
  // the object would rebuild the sheet on every parent render.
  const { family, xi, seed, logM, coupling } = theory;
  const [predicted, setPredicted] = useState<ScalarField | null>(null);
  const [gammaPred, setGammaPred] = useState<number | null>(null);

  // gamma is linear in the coupling, so the exact sheet is a plane through the origin
  // whose tilt in log r is 2*gamma and which fans out linearly in lambda.
  const exact = useMemo(() => {
    const shape: Theory = { family, xi, seed, logM, coupling: 1 };
    const perUnit = anomalousDimension(shape, background);
    return sampleField(NR, N_LAMBDA_EXACT, LOG_R, LAMBDA, (logR, lambda) =>
      2 * perUnit * lambda * logR,
    );
  }, [family, xi, seed, logM, background]);

  const gammaExact = anomalousDimension(theory, background);
  // Prefer the server's answer for the readout when one is attached.
  const reported = prediction?.source === "server" ? prediction.gamma : gammaPred;

  // The predicted sheet costs one forward pass per coupling row, so it is rebuilt off the
  // render path and only when the theory shape changes -- not while a slider is moving.
  useEffect(() => {
    if (!operator) {
      setPredicted(null);
      setGammaPred(null);
      return;
    }
    let cancelled = false;
    const build = () => {
      const logR = new Float64Array(NR);
      for (let i = 0; i < NR; i += 1) {
        logR[i] = LOG_R[0] + ((LOG_R[1] - LOG_R[0]) * i) / (NR - 1);
      }
      const coarse = new Float32Array(NR * N_LAMBDA_PREDICTED);
      let atTheory = 0;
      for (let j = 0; j < N_LAMBDA_PREDICTED; j += 1) {
        const lambda =
          LAMBDA[0] + ((LAMBDA[1] - LAMBDA[0]) * j) / (N_LAMBDA_PREDICTED - 1);
        const row: Theory = { family, xi, seed, logM, coupling: lambda };
        const v = operator.scaleFeatures(potentialSamples(row, operator.phiGrid));
        const logW = operator.predictLogW(v, logR, logM);
        for (let i = 0; i < NR; i += 1) coarse[j * NR + i] = logW[i] + 2 * free * logR[i];
      }
      // The readout is evaluated at the actual coupling rather than snapped to the nearest
      // sheet row. The rows are ~5e-3 apart in lambda and gamma is linear in it, so reading
      // off the grid reports a different theory -- enough to inflate the error by two
      // orders of magnitude and make an accurate operator look poor.
      const here: Theory = { family, xi, seed, logM, coupling };
      const exactRow = operator.predictLogW(
        operator.scaleFeatures(potentialSamples(here, operator.phiGrid)),
        logR,
        logM,
      );
      atTheory = anomalousDimensionFromCorrelator(logR, exactRow, free);
      if (cancelled) return;
      const values = expandRows(coarse, NR, N_LAMBDA_PREDICTED, N_LAMBDA_EXACT);
      let low = Number.POSITIVE_INFINITY;
      let high = Number.NEGATIVE_INFINITY;
      for (const v of values) {
        if (v < low) low = v;
        if (v > high) high = v;
      }
      setPredicted({
        values,
        nx: NR,
        ny: N_LAMBDA_EXACT,
        xRange: LOG_R,
        yRange: LAMBDA,
        valueRange: [low, high],
      });
      setGammaPred(atTheory);
    };
    const handle = window.setTimeout(build, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [operator, family, xi, seed, logM, coupling, free]);

  const layers: SurfaceLayer[] = useMemo(() => {
    // One scale for both sheets, or "the prediction sits on the exact plane" is not a claim
    // the picture can make.
    const low = Math.min(exact.valueRange[0], predicted?.valueRange[0] ?? Infinity);
    const high = Math.max(exact.valueRange[1], predicted?.valueRange[1] ?? -Infinity);
    const out: SurfaceLayer[] = [
      {
        key: "exact",
        field: withValueRange(exact, low, high),
        bands: 6,
        low: PALETTE.slate,
        high: PALETTE.sodium,
      },
    ];
    if (predicted) {
      out.push({
        key: "predicted",
        field: withValueRange(predicted, low, high),
        bands: 6,
        low: PALETTE.slate,
        high: PALETTE.ion,
        overlay: true,
      });
    }
    return out;
  }, [exact, predicted]);

  return (
    <Panel
      title="γ as a functional of the potential"
      subtitle={
        <>
          <Formula tex="\log W + 2\Delta\beta_1\beta_2\log r" /> over{" "}
          <Formula tex="(\log r,\ \lambda)" />, with the free-theory power law divided out.
          Along <Formula tex="\log r" /> the tilt <em>is</em> <Formula tex="2\gamma" />;
          across <Formula tex="\lambda" /> it fans out linearly, because{" "}
          <Formula tex="\gamma=\tfrac12\beta_1\beta_2\langle V''\rangle_\sigma C_{\log}" /> is
          linear in the coupling. The amber plane is that closed form. The teal sheet is the
          operator, evaluated at twenty couplings: where it lifts off the plane is where it
          has not learned the functional.
        </>
      }
      aside={
        <span className="font-mono text-[0.65rem] tracking-wide text-[var(--dim)] uppercase">
          {prediction?.source === "server" ? "readout: pytorch" : "readout: browser"}
        </span>
      }
    >
      <SurfaceStage
        layers={layers}
        height={320}
        camera={{ position: [2.4, 1.5, 2.0], target: [0, 0.35, 0] }}
      />
      <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-[var(--dim)]">
        <span>log r ∈ [{LOG_R[0].toFixed(1)}, {LOG_R[1].toFixed(1)}]</span>
        <span>λ ∈ [{LAMBDA[0]}, {LAMBDA[1]}]</span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout label="γ exact" value={gammaExact.toExponential(3)} />
        <Readout
          label="γ recovered"
          value={reported === null ? "—" : reported.toExponential(3)}
        />
        <Readout
          label="|Δγ|"
          value={reported === null ? "—" : Math.abs(reported - gammaExact).toExponential(2)}
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
