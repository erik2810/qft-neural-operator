import { useEffect, useMemo, useState } from "react";
import { Formula } from "./components/Formula";
import { Panel, Readout } from "./components/Panel";
import { TheoryControls } from "./components/TheoryControls";
import { BulkSurface } from "./panels/BulkSurface";
import { ResidualSurface } from "./panels/ResidualSurface";
import {
  potentialMoment,
  potentialSamples,
  predictLocally,
  type Prediction,
} from "./lib/prediction";
import { RGSurface } from "./panels/RGSurface";
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
    <div className="mx-auto max-w-[62rem] px-5 py-10">
      <header className="mb-8">
        <p className="eyebrow">Euclidean AdS₂ · holographic renormalization</p>
        <h1 className="display mt-3 text-[2rem] leading-[1.15] text-[var(--bright)]">
          An action goes in. A boundary correlator comes out.
        </h1>
        <p className="mt-4 max-w-[62ch] text-[0.9rem] leading-relaxed text-[var(--dim)]">
          A neural operator learning{" "}
          <Formula tex="V(\phi) \mapsto W(p_1,p_2)=\langle V_{\beta_1}(p_1)V_{\beta_2}(p_2)\rangle_{\rm conn}" />{" "}
          in the Poincaré patch <Formula tex="ds^2=(L^2/z^2)(dz^2+dp^2)" />. The metric
          blows up at the boundary, bulk integrals diverge logarithmically there, and the
          logarithms reorganize into an anomalous dimension{" "}
          <Formula tex="\Delta_{\rm eff}=\Delta\beta_1\beta_2-\gamma" />. Each panel below
          is that story as a surface: height is the quantity, and the shape is the argument.
        </p>
        <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-[var(--rule)] pt-4">
          <span className="eyebrow">
            {backendOnline
              ? "backend online · live pytorch inference"
              : backendStatus === "checking"
                ? "probing backend"
                : "standalone · computed in the browser"}
          </span>
          {operatorState.status === "ready" && (
            <span className="eyebrow" style={{ color: trained ? undefined : "var(--sodium)" }}>
              {trained ? "operator · trained checkpoint" : "operator · untrained weights"}
            </span>
          )}
          {operatorState.status === "unavailable" && (
            <span className="eyebrow" style={{ color: "var(--sodium)" }}>
              no exported operator · exact physics only
            </span>
          )}
        </div>
      </header>

      <div className="grid gap-5">
        <Panel
          title="Theory"
          subtitle={
            <>
              The anomalous dimension is a <em>functional</em> of the potential,{" "}
              <Formula tex="\gamma[V]=\tfrac12\beta_1\beta_2\langle V''\rangle_\sigma C_{\log}" />.
              That is why a Gaussian-process draw with no analytic form carries a label as
              exact as Sine-Gordon’s.
            </>
          }
        >
          <TheoryControls theory={theory} onChange={setTheory} />
          <div className="mt-5 grid grid-cols-2 gap-5 sm:grid-cols-4">
            <Readout label="Δ" value={background.delta.toFixed(4)} hint="Δ(Δ−)=m²L²" />
            <Readout label="c_Δ" value={background.cDelta.toFixed(5)} />
            <Readout label="C_log" value={logCoefficient(background).toFixed(5)} />
            <Readout label="Δβ₁β₂" value={freeDimension(background).toFixed(4)} />
          </div>
        </Panel>

        <BulkSurface background={background} />
        <ResidualSurface
          theory={theory}
          background={background}
          operator={operatorState.status === "ready" ? operatorState.operator : null}
          prediction={prediction}
        />
        <RGSurface theory={theory} background={background} />
      </div>

      <footer className="mt-10 border-t border-[var(--rule)] pt-5 text-[0.75rem] leading-relaxed text-[var(--dim)]">
        Every amber surface is closed form, computed in the browser. Teal is predicted or
        unphysical. The operator runs client-side from an exported Fourier-DeepONet: its
        spectral layers keep their weights in Fourier space and the 64-point transform runs
        here, since <code className="numeric">aten::fft_rfft</code> has no ONNX lowering.
        Parity with PyTorch is pinned by tests on both sides.
      </footer>
    </div>
  );
}
