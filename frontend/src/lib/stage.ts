/**
 * WebGPU stage: renderer, camera, orbit controls, and the TSL surface material.
 *
 * Three.js is loaded through `three/webgpu`, whose renderer falls back to a WebGL2 backend
 * on its own when WebGPU is unavailable, so no separate code path is needed -- only a
 * label, since the two are not visually identical under heavy shading.
 */

import * as THREE from "three/webgpu";
import {
  Fn,
  clamp,
  color,
  float,
  fwidth,
  mix,
  positionLocal,
  smoothstep,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ScalarField } from "./surface";

/** Palette, kept here so the shader and the CSS cannot drift apart. */
/** World-space width and depth of every surface, so slopes can be computed in world units. */
const SPAN = 2;

export const PALETTE = {
  void: "#0a0e14",
  slate: "#16202b",
  steel: "#3d5a6c",
  foam: "#c9d6df",
  /** The AdS boundary, and the exact/reference signal. */
  sodium: "#ffb347",
  /** The operator's prediction, and any second surface. */
  ion: "#4fd1c5",
} as const;

export interface SurfaceOptions {
  /** Height of the surface at the top of its value range, in world units. */
  relief?: number;
  /** Number of colour bands. One per decade for a log field. */
  bands?: number;
  /** Base colour the ramp climbs from, as a CSS hex string. */
  low?: string;
  /** Colour at the top of the ramp -- the boundary edge in the bulk panel. */
  high?: string;
  /** Draw the surface as a translucent overlay rather than an opaque sheet. */
  overlay?: boolean;
}

export interface SurfaceHandle {
  mesh: THREE.Mesh;
  /** Push new field data without rebuilding the geometry or material. */
  update: (field: ScalarField) => void;
  dispose: () => void;
}

/**
 * Shading from the height field's own gradient.
 *
 * Recomputing vertex normals on the CPU for every slider tick would dominate the frame;
 * central differences on the height texture give the same read for a few extra samples,
 * and stay correct when the geometry is subdivided differently from the data grid.
 */
const surfaceNormal = /*#__PURE__*/ Fn(
  ([map, coord, texel, relief]: [
    ReturnType<typeof texture>,
    ReturnType<typeof vec2>,
    ReturnType<typeof vec2>,
    ReturnType<typeof float>,
  ]) => {
    const left = map.sample(coord.sub(vec2(texel.x, 0))).r;
    const right = map.sample(coord.add(vec2(texel.x, 0))).r;
    const down = map.sample(coord.sub(vec2(0, texel.y))).r;
    const up = map.sample(coord.add(vec2(0, texel.y))).r;
    // Height differences must become *slopes* before they can be a normal: the samples are
    // 2*texel apart in UV, and the plane spans SPAN world units across the full UV range.
    const slopeX = right.sub(left).mul(relief).div(texel.x.mul(2).mul(SPAN));
    const slopeZ = up.sub(down).mul(relief).div(texel.y.mul(2).mul(SPAN));
    return vec3(slopeX.negate(), 1.0, slopeZ.negate()).normalize();
  },
);

/** Create the surface mesh and its TSL material. */
export function createSurface(field: ScalarField, options: SurfaceOptions = {}): SurfaceHandle {
  const relief = options.relief ?? 0.85;
  const bands = options.bands ?? 8;
  const overlay = options.overlay ?? false;

  // Half float, not float32: WebGPU treats r32float as "unfilterable-float" unless the
  // optional float32-filterable feature is present, so a LinearFilter sampler over one is
  // invalid and the surface comes back garbled. r16float is filterable everywhere, and
  // ~3 decimal digits is ample for a normalized height.
  const data = new THREE.DataTexture(
    new Uint16Array(field.values.length),
    field.nx,
    field.ny,
    THREE.RedFormat,
    THREE.HalfFloatType,
  );
  data.minFilter = THREE.LinearFilter;
  data.magFilter = THREE.LinearFilter;
  data.wrapS = THREE.ClampToEdgeWrapping;
  data.wrapT = THREE.ClampToEdgeWrapping;

  const reliefUniform = uniform(relief);
  const texel = uniform(new THREE.Vector2(1 / field.nx, 1 / field.ny));

  const geometry = new THREE.PlaneGeometry(SPAN, SPAN, field.nx - 1, field.ny - 1);
  geometry.rotateX(-Math.PI / 2);

  const material = new THREE.MeshBasicNodeMaterial();
  const map = texture(data);

  const coord = uv();
  const height = clamp(map.sample(coord).r, float(0), float(1));

  // Displacement happens in the vertex stage, which has no screen-space derivatives and so
  // no implicit mip level -- the sample must name its level explicitly.
  material.positionNode = positionLocal.add(
    vec3(0, clamp(texture(data, coord).level(float(0)).r, float(0), float(1)).mul(reliefUniform), 0),
  );

  const normal = surfaceNormal(map, coord, texel, reliefUniform);
  // A fixed key light from the front-left. The surface is read for shape, not lit for
  // realism, so one stable direction beats a scene light that moves with the camera.
  const lambert = normal.dot(vec3(0.45, 0.78, 0.44).normalize()).mul(0.5).add(0.5);

  const low = color(options.low ?? PALETTE.slate);
  const high = color(options.high ?? PALETTE.sodium);
  const scaled = height.mul(bands);
  // Quantize to whole bands, then draw a hairline where a band boundary falls. The line
  // width is derived from the on-screen rate of change of `scaled`, so a band edge stays
  // one pixel wide however steeply the surface is falling -- a fixed width moires badly
  // where the ridges turn over.
  const stepped = scaled.floor().div(bands);
  const within = scaled.fract();
  const edge = within.min(float(1).sub(within));
  const lineWidth = fwidth(scaled).mul(0.9).add(1e-5);
  const contour = smoothstep(float(0), lineWidth, edge);

  const base = mix(low, high, stepped.pow(0.85));
  const shaded = base.mul(lambert.pow(1.35).mul(0.85).add(0.35));
  material.colorNode = mix(shaded.mul(0.55), shaded, contour);

  if (overlay) {
    material.transparent = true;
    material.opacityNode = height.smoothstep(0.02, 0.2).mul(0.55).add(0.2);
    material.depthWrite = false;
  }

  const mesh = new THREE.Mesh(geometry, material);
  // Displacement is applied on the GPU, so the CPU-side bounding sphere describes a flat
  // plane and would cull the surface at grazing angles.
  mesh.frustumCulled = false;

  const update = (next: ScalarField) => {
    const [lowValue, highValue] = next.valueRange;
    const span = highValue - lowValue || 1;
    const texels = data.image.data as unknown as Uint16Array;
    for (let i = 0; i < texels.length && i < next.values.length; i += 1) {
      const value = (next.values[i] - lowValue) / span;
      texels[i] = THREE.DataUtils.toHalfFloat(value < 0 ? 0 : value > 1 ? 1 : value);
    }
    data.needsUpdate = true;
    texel.value.set(1 / next.nx, 1 / next.ny);
  };
  update(field);

  return {
    mesh,
    update,
    dispose: () => {
      geometry.dispose();
      material.dispose();
      data.dispose();
    },
  };
}

export interface Stage {
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  backend: "webgpu" | "webgl";
  render: () => void;
  dispose: () => void;
}

/** Boot a renderer into `container` and return its handles. */
export interface StageOptions {
  /** Initial camera position, world units. */
  position?: [number, number, number];
  /** Orbit target. */
  target?: [number, number, number];
}

export async function createStage(
  container: HTMLElement,
  options: StageOptions = {},
): Promise<Stage> {
  const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  await renderer.init();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100);
  // Low and mostly from the boundary side.
  //
  // Camera choice is not decoration here: the bulk ridges run *away* from the viewer along
  // log z, so a high three-quarter view foreshortens them into a featureless plateau and
  // the surface reads as flat even when it is fully displaced. Looking across them keeps
  // the cross-section legible.
  camera.position.set(...(options.position ?? [1.15, 1.05, 2.85]));

  container.appendChild(renderer.domElement);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 1.9;
  controls.maxDistance = 6;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(...(options.target ?? [0, 0.3, 0]));

  const resize = () => {
    const { clientWidth, clientHeight } = container;
    if (!clientWidth || !clientHeight) return;
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(clientWidth, clientHeight, false);
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(container);

  const render = () => {
    controls.update();
    renderer.renderAsync(scene, camera);
  };

  return {
    renderer,
    scene,
    camera,
    controls,
    // `isWebGPUBackend` is set by the backend but absent from its published type.
    backend: (renderer.backend as { isWebGPUBackend?: boolean })?.isWebGPUBackend
      ? "webgpu"
      : "webgl",
    render,
    dispose: () => {
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    },
  };
}

export { THREE, vec4 };
