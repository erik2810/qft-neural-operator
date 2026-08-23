import { useEffect, useMemo, useState } from "react";
import { Formula } from "./components/Formula";
import { MarginValue, Sym } from "./components/Figure";
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
    <div className="mx-auto max-w-[62rem] px-6 pb-24 pt-14 sm:px-10">
      <header className="border-b-2 border-[var(--ink)] pb-8">
        <p className="label">Euclidean AdS₂ · holographic renormalization</p>
        <h1 className="mt-4 text-[clamp(2.1rem,5.2vw,3.15rem)] leading-[1.08]">
          An action goes in.
          <br />
          A boundary correlator comes out.
        </h1>
        <div className="mt-7 grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_var(--margin-col)]">
          <div className="prose">
            <p className="lede">
              A neural operator learning{" "}
              <Formula tex="V(\phi) \mapsto W(p_1,p_2)=\langle V_{\beta_1}(p_1)V_{\beta_2}(p_2)\rangle_{\rm conn}" />{" "}
              in the Poincaré patch <Formula tex="ds^2=(L^2/z^2)(dz^2+dp^2)" />. The metric
              blows up at the boundary, bulk integrals diverge logarithmically there, and
              the logarithms reorganize into an anomalous dimension{" "}
              <Formula tex="\Delta_{\rm eff}=\Delta\beta_1\beta_2-\gamma" />.
            </p>
            <p className="mt-4 text-[var(--ink-soft)]">
              Three figures follow, and each is a surface rather than a plot. Height is the
              quantity; the shape is the argument. Everything amber is closed form and
              computed here in the browser. Everything teal is predicted or unphysical.
            </p>
          </div>
          <aside className="border-t border-[var(--rule)] pt-3 lg:border-t-2 lg:border-t-[var(--ink)]">
            <MarginValue label={<Sym>Δ</Sym>} value={background.delta.toFixed(4)} note="Δ(Δ−1) = m²L²" />
            <MarginValue label={<Sym>c_Δ</Sym>} value={background.cDelta.toFixed(5)} />
            <MarginValue label={<Sym>C_log</Sym>} value={logCoefficient(background).toFixed(5)} />
            <MarginValue
              label="running"
              value={
                backendOnline
                  ? "server"
                  : backendStatus === "checking"
                    ? "…"
                    : "browser"
              }
              note={
                operatorState.status === "ready"
                  ? trained
                    ? "trained checkpoint loaded"
                    : "untrained weights — predictions sit on the free theory"
                  : operatorState.status === "unavailable"
                    ? "no exported operator — exact physics only"
                    : "loading the operator"
              }
            />
          </aside>
        </div>
      </header>

      <section className="mt-12">
        <h2 className="text-[1.35rem] leading-snug">
          <span className="section-number mr-2 align-[0.18em]">§ 1</span>
          Pick a theory
        </h2>
        <div className="prose caption mt-3 text-[0.88rem]">
          The anomalous dimension is a <em>functional</em> of the potential,{" "}
          <Formula tex="\gamma[V]=\tfrac12\beta_1\beta_2\langle V''\rangle_\sigma C_{\log}" />.
          That is why a Gaussian-process draw with no closed form carries a label as exact as
          Sine-Gordon’s — and why the operator has something to learn beyond two named curves.
        </div>
        <div className="mt-6">
          <TheoryControls theory={theory} onChange={setTheory} />
        </div>
      </section>

      <BulkSurface background={background} number={1} />
      <ResidualSurface
        theory={theory}
        background={background}
        operator={operatorState.status === "ready" ? operatorState.operator : null}
        prediction={prediction}
        number={2}
      />
      <RGSurface theory={theory} background={background} number={3} />

      <footer className="mt-20 border-t-2 border-[var(--ink)] pt-6">
        <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_var(--margin-col)]">
          <div className="prose caption">
            <p>
              The operator runs client-side from an exported Fourier-DeepONet. Its spectral
              layers keep their weights in Fourier space and the 64-point transform runs
              here, because <span className="numeric">aten::fft_rfft</span> has no ONNX
              lowering. Parity with PyTorch is pinned by tests on both sides: eleven to twelve
              decimals for the closed-form physics, eight for the bulk quadrature, and
              float32-limited for the network.
            </p>
            <p>
              Attach the FastAPI server and the same page switches to full-precision PyTorch
              inference over a binary WebSocket, without changing what it shows.
            </p>
          </div>
          <aside className="border-t border-[var(--rule)] pt-3 lg:border-t-2 lg:border-t-[var(--ink)]">
            <MarginValue label="source" value="erik2810/qft-neural-operator" />
            <MarginValue label="renderer" value="WebGPU · Three.js TSL" />
            <MarginValue label="tests" value="331 Python · 48 TypeScript" />
          </aside>
        </div>
      </footer>
    </div>
  );
}
