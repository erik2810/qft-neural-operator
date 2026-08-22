import { useEffect, useMemo, useState } from "react";
import { Formula } from "./components/Formula";
import { Panel, Readout } from "./components/Panel";
import { TheoryControls } from "./components/TheoryControls";
import { BulkDiagram } from "./panels/BulkDiagram";
import { LogResidual } from "./panels/LogResidual";
import {
  potentialMoment,
  potentialSamples,
  predictLocally,
  type Prediction,
} from "./lib/prediction";
import { RGInvariance } from "./panels/RGInvariance";
import {
  cDeltaCft,
  defaultBackground,
  defaultTheory,
  freeDimension,
  logCoefficient,
  logRGrid,
  type Background,
  type Theory,
} from "./lib/physics";
import { useBackendStatus, useCorrelatorStream } from "./lib/useBackend";
import { useOperator } from "./lib/useOperator";

/** Background the exported operator was trained against, once its manifest has loaded. */
function backgroundFromManifest(physics: Record<string, number> | undefined): Background {
  const fallback = defaultBackground();
  if (!physics) return fallback;
  const delta = physics.delta ?? fallback.delta;
  return {
    L: physics.L ?? 1,
    mSq: physics.m_sq ?? fallback.mSq,
    delta,
    cDelta: physics.c_delta ?? cDeltaCft(delta),
    // log_coefficient = 2 L² c_Δ / normalization, so invert for the factor in force.
    normalizationFactor:
      physics.log_coefficient && physics.c_delta
        ? (2 * (physics.L ?? 1) ** 2 * physics.c_delta) / physics.log_coefficient
        : 1,
    beta1: 1,
    beta2: 1,
    sigmaSq: physics.sigma_sq ?? 0,
  };
}

export default function App() {
  const [theory, setTheory] = useState<Theory>(defaultTheory());
  const backendStatus = useBackendStatus();
  const backendOnline = backendStatus === "online";
  const operatorState = useOperator();

  const background = useMemo(
    () =>
      backgroundFromManifest(
        operatorState.status === "ready" ? operatorState.operator.manifest.physics : undefined,
      ),
    [operatorState],
  );

  const logR = useMemo(() => logRGrid(160), []);
  const correlatorStream = useCorrelatorStream(backendOnline);

  // The frontend always builds V(φ) itself and sends it, so the backend evaluates the
  // very potential on screen rather than its own draw for the same seed.
  const phiGrid = useMemo(
    () =>
      operatorState.status === "ready"
        ? operatorState.operator.phiGrid
        : Float64Array.from({ length: 64 }, (_, i) => -3 + (6 * i) / 63),
    [operatorState],
  );

  const send = correlatorStream.send;
  useEffect(() => {
    if (!backendOnline) return;
    send({
      family: theory.family,
      coupling: theory.coupling,
      xi: theory.xi,
      seed: theory.seed,
      log_m: theory.logM,
      moment: potentialMoment(theory, background.sigmaSq),
      v_phi: Array.from(potentialSamples(theory, phiGrid)),
    });
  }, [backendOnline, theory, phiGrid, background.sigmaSq, send]);

  const prediction: Prediction | null = useMemo(() => {
    const frame = correlatorStream.frame;
    if (backendOnline && frame) {
      return {
        logR: Float64Array.from(frame.logR),
        logW: Float64Array.from(frame.logWPred),
        gamma: frame.gammaPred,
        source: "server",
      };
    }
    if (operatorState.status === "ready") {
      return predictLocally(operatorState.operator, theory, background, logR);
    }
    return null;
  }, [correlatorStream.frame, backendOnline, operatorState, theory, background, logR]);

  const trained =
    operatorState.status === "ready" ? operatorState.operator.manifest.trained : false;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">
          QFT action → observable in Euclidean AdS₂
        </h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          A neural operator learning{" "}
          <Formula tex="V(\phi) \mapsto W(p_1,p_2)=\langle V_{\beta_1}(p_1)V_{\beta_2}(p_2)\rangle_{\rm conn}" />{" "}
          in the Poincaré patch <Formula tex="ds^2=(L^2/z^2)(dz^2+dp^2)" />. Holographic
          renormalization turns the near-boundary logarithms into an anomalous dimension,{" "}
          <Formula tex="\Delta_{\rm eff}=\Delta\beta_1\beta_2-\gamma" />.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
          <span
            className={`rounded-full px-2 py-0.5 ${
              backendOnline
                ? "bg-sky-500/15 text-sky-300"
                : backendStatus === "checking"
                  ? "bg-slate-700/50 text-slate-400"
                  : "bg-slate-700/50 text-slate-400"
            }`}
          >
            {backendOnline
              ? "backend online — live PyTorch inference"
              : backendStatus === "checking"
                ? "probing backend…"
                : "standalone — everything computed in the browser"}
          </span>
          {operatorState.status === "ready" && (
            <span className={trained ? "text-slate-400" : "text-amber-400"}>
              {trained
                ? "operator: trained checkpoint"
                : "operator: untrained weights — predictions sit on the free theory"}
            </span>
          )}
          {operatorState.status === "unavailable" && (
            <span className="text-amber-400">
              no exported operator — exact physics only
            </span>
          )}
        </div>
      </header>

      <div className="grid gap-4">
        <Panel
          title="Theory"
          subtitle={
            <>
              The anomalous dimension is a <em>functional</em> of the potential:{" "}
              <Formula tex="\gamma[V]=\tfrac12\beta_1\beta_2\langle V''\rangle_\sigma C_{\log}" />,
              which is why polynomial and Gaussian-process draws carry exact labels too.
            </>
          }
        >
          <TheoryControls theory={theory} onChange={setTheory} />
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Readout label="Δ" value={background.delta.toFixed(4)} hint="Δ(Δ−1)=m²L²" />
            <Readout label="c_Δ" value={background.cDelta.toFixed(5)} />
            <Readout label="C_log" value={logCoefficient(background).toFixed(5)} />
            <Readout label="Δβ₁β₂" value={freeDimension(background).toFixed(4)} />
          </div>
        </Panel>

        <BulkDiagram background={background} backendOnline={backendOnline} />
        <LogResidual theory={theory} background={background} prediction={prediction} />
        <RGInvariance theory={theory} background={background} />
      </div>

      <footer className="mt-8 text-xs leading-relaxed text-slate-500">
        Exact curves are closed form and computed in the browser. The operator prediction
        comes from the exported Fourier-DeepONet — the spectral layers keep their weights in
        Fourier space and the 64-point transform runs client-side, since{" "}
        <code className="text-slate-400">aten::fft_rfft</code> has no ONNX lowering. Parity
        with PyTorch is pinned by tests on both sides.
      </footer>
    </div>
  );
}
