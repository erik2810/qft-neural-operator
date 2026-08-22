import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Formula } from "../components/Formula";
import { Panel, Readout } from "../components/Panel";
import {
  contactIntegral,
  integrandField,
  measuredLogCoefficient,
  quantizeField,
  type QuantizedField,
} from "../lib/bulk";
import { logCoefficient, type Background } from "../lib/physics";
import { useBulkStream } from "../lib/useBackend";

/**
 * Inferno-like ramp, evaluated in the fragment shader.
 *
 * A perceptually monotone ramp matters here: the eye is being asked to read *where* the
 * density concentrates as the cutoff moves, and a rainbow map invents banding that is
 * not in the data.
 */
const FRAGMENT = /* glsl */ `
  precision highp float;
  uniform sampler2D uDensity;
  uniform float uCutoffRow;
  varying vec2 vUv;

  vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.014);
    vec3 c1 = vec3(0.316, 0.072, 0.485);
    vec3 c2 = vec3(0.716, 0.215, 0.475);
    vec3 c3 = vec3(0.968, 0.495, 0.238);
    vec3 c4 = vec3(0.988, 0.998, 0.645);
    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
  }

  void main() {
    float value = texture2D(uDensity, vUv).r;
    vec3 color = ramp(value);
    // A hairline at the cutoff: everything above it is excluded from the integral, and
    // the divergence is the density piling up as that line is pushed toward the boundary.
    if (uCutoffRow >= 0.0 && abs(vUv.y - uCutoffRow) < 0.005) {
      color = mix(color, vec3(0.55, 0.82, 1.0), 0.85);
    }
    gl_FragColor = vec4(color, 1.0);
  }
`;

const VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    // Row 0 of the field is the smallest z. Flipping v puts the AdS boundary at the top
    // of the frame, which is how the Poincare patch is conventionally drawn.
    vUv = vec2(uv.x, 1.0 - uv.y);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/** Upload a quantized field to a WebGL texture and render it. */
function useDensityCanvas(field: QuantizedField | null) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.OrthographicCamera;
    material: THREE.ShaderMaterial;
    texture: THREE.DataTexture | null;
  } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: { uDensity: { value: null }, uCutoffRow: { value: 0 } },
    });
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material));
    container.appendChild(renderer.domElement);
    stateRef.current = { renderer, scene, camera, material, texture: null };

    const resize = () => {
      const { clientWidth, clientHeight } = container;
      renderer.setSize(clientWidth, Math.max(clientHeight, 1), false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      renderer.render(scene, camera);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      observer.disconnect();
      stateRef.current?.texture?.dispose();
      material.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !field) return;
    const { material, renderer, scene, camera } = state;

    // The protocol already delivers 8-bit density, which is exactly the R8 texture
    // format — no repacking between the wire and the GPU.
    const needsNew =
      !state.texture ||
      state.texture.image.width !== field.nP ||
      state.texture.image.height !== field.nZ;
    if (needsNew) {
      state.texture?.dispose();
      const texture = new THREE.DataTexture(field.density, field.nP, field.nZ, THREE.RedFormat);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.needsUpdate = true;
      state.texture = texture;
      material.uniforms.uDensity.value = texture;
    } else if (state.texture) {
      state.texture.image.data = field.density;
      state.texture.needsUpdate = true;
    }
    // Place the cutoff line by its position within the displayed log-z window; a
    // negative value means it lies outside the frame and the shader skips it.
    const span = field.logZMax - field.logZMin;
    const t = (field.logEps - field.logZMin) / (span || 1);
    material.uniforms.uCutoffRow.value = t >= 0 && t <= 1 ? t : -1;
    renderer.render(scene, camera);
  }, [field]);

  return containerRef;
}

export function BulkDiagram({
  background,
  backendOnline,
}: {
  background: Background;
  backendOnline: boolean;
}) {
  const [logEps, setLogEps] = useState(-4);
  const [r, setR] = useState(1);
  const stream = useBulkStream(backendOnline);

  const send = stream.send;
  useEffect(() => {
    if (!backendOnline) return;
    send({ r, log_eps: logEps, n_z: 192, n_p: 256, decades_below: 3, decades_above: 1 });
  }, [backendOnline, r, logEps, send]);

  const local = useMemo(() => {
    const eps = Math.exp(logEps);
    const field = integrandField(r, background, 160, 224, 3, 1);
    return quantizeField(field, r, logEps, contactIntegral(r, eps, background));
  }, [r, logEps, background]);

  const field: QuantizedField = useMemo(() => {
    const frame = stream.frame;
    if (!backendOnline || !frame) return local;
    return {
      density: frame.density,
      nZ: frame.nZ,
      nP: frame.nP,
      logZMin: frame.logZMin,
      logZMax: frame.logZMax,
      pMin: frame.pMin,
      pMax: frame.pMax,
      logLow: frame.logLow,
      logHigh: frame.logHigh,
      r: frame.r,
      logEps: frame.logEps,
      integral: frame.integral,
      source: "server",
    };
  }, [stream.frame, backendOnline, local]);

  const canvasRef = useDensityCanvas(field);

  // Two cutoffs bracket the derivative, so this is always the browser's own quadrature:
  // the wire format carries one epsilon per frame, not a slope.
  const measured = useMemo(
    () => measuredLogCoefficient(r, Math.exp(logEps), background),
    [r, logEps, background],
  );
  const analytic = logCoefficient(background);

  return (
    <Panel
      title="AdS₂ bulk and the Witten contact diagram"
      subtitle={
        <>
          The integrand <Formula tex="\sqrt{g}\,K_\Delta(x;p_1)K_\Delta(x;p_2)" /> over the
          Poincaré half-plane, boundary at the top. Towards it the density narrows into two
          ridges of width <Formula tex="\sim z" /> at <Formula tex="p=\mp r/2" />, each
          contributing <Formula tex="dz/z" /> — that is the logarithmic divergence. The blue
          line is the cutoff <Formula tex="\epsilon" />: only what lies below it is
          integrated, so pushing it up adds another decade of <Formula tex="\log(r/\epsilon)" />.
        </>
      }
      aside={
        <span className={field.source === "server" ? "text-sky-400" : "text-slate-500"}>
          {field.source === "server" ? "server quadrature" : "browser quadrature"}
        </span>
      }
    >
      <div
        ref={canvasRef}
        className="h-64 w-full overflow-hidden rounded border border-slate-700/60 bg-black"
      />
      <div className="mt-1 flex justify-between text-[0.65rem] text-slate-500">
        <span>
          p = {field.pMin.toFixed(2)} … {field.pMax.toFixed(2)}
        </span>
        <span>
          z = {Math.exp(field.logZMax).toExponential(1)} … {Math.exp(field.logZMin).toExponential(1)}{" "}
          (boundary at top)
        </span>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          <span>
            cutoff <Formula tex="\log\epsilon" /> = {logEps.toFixed(2)} (
            <Formula tex="M = 1/\epsilon" /> = {Math.exp(-logEps).toExponential(1)})
          </span>
          <input
            type="range"
            min={-8}
            max={-0.5}
            step={0.05}
            value={logEps}
            onChange={(e) => setLogEps(Number(e.target.value))}
            className="accent-sky-400"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          <span>
            separation <Formula tex="r=|p_1-p_2|" /> = {r.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.2}
            max={4}
            step={0.02}
            value={r}
            onChange={(e) => setR(Number(e.target.value))}
            className="accent-sky-400"
          />
        </label>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Readout label="∫ over z > ε" value={field.integral.toFixed(5)} />
        <Readout
          label="measured C_log"
          value={measured.toFixed(6)}
          hint="dĨ / d log(1/ε)"
        />
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
