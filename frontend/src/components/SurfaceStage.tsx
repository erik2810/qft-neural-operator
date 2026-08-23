import { useEffect, useRef, useState } from "react";
import { PALETTE, THREE, createStage, createSurface, type SurfaceOptions } from "../lib/stage";
import type { ScalarField } from "../lib/surface";

export interface SurfaceLayer extends SurfaceOptions {
  field: ScalarField;
  key: string;
}

interface SurfaceStageProps {
  layers: SurfaceLayer[];
  /**
   * Which edge of the domain is the AdS boundary, drawn as a luminous rail.
   *
   * Note the plane's UV convention: `rotateX(-PI/2)` sends v = 0 to world +z, so a field
   * whose first row is the boundary wants `yMax`, not `yMin`.
   */
  boundary?: "xMin" | "xMax" | "yMin" | "yMax" | null;
  /** A plane cutting the surface, in normalized domain coordinates, e.g. the cutoff. */
  marker?: { axis: "x" | "y"; at: number } | null;
  height?: number;
  /** Initial camera placement; each panel is legible from a different angle. */
  camera?: { position: [number, number, number]; target?: [number, number, number] };
  onBackend?: (backend: "webgpu" | "webgl") => void;
}

/**
 * Renders one or more scalar fields as shaded height surfaces in a shared 3D frame.
 *
 * The render loop is gated on intersection: three WebGPU canvases animating off-screen is
 * three times the GPU cost for nothing, and the page stacks them vertically.
 */
export function SurfaceStage({
  layers,
  boundary = null,
  marker = null,
  height = 320,
  camera,
  onBackend,
}: SurfaceStageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Awaited<ReturnType<typeof createStage>> | null>(null);
  const surfacesRef = useRef(new Map<string, ReturnType<typeof createSurface>>());
  const markerRef = useRef<THREE.Mesh | null>(null);
  const visibleRef = useRef(true);
  const [status, setStatus] = useState<"booting" | "ready" | "failed">("booting");

  // Boot once. Everything after arrives through the update effects below.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    // Captured for the cleanup closure; the ref itself may point elsewhere by then.
    const surfaces = surfacesRef.current;
    let disposed = false;
    let frame = 0;

    createStage(container, { position: camera?.position, target: camera?.target })
      .then((stage) => {
        if (disposed) {
          stage.dispose();
          return;
        }
        stageRef.current = stage;
        onBackend?.(stage.backend);

        const grid = new THREE.GridHelper(2, 12, PALETTE.steel, PALETTE.slate);
        (grid.material as THREE.Material).opacity = 0.35;
        (grid.material as THREE.Material).transparent = true;
        stage.scene.add(grid);

        setStatus("ready");

        const loop = () => {
          frame = requestAnimationFrame(loop);
          if (visibleRef.current) stage.render();
        };
        loop();
      })
      .catch(() => {
        if (!disposed) setStatus("failed");
      });

    const observer = new IntersectionObserver(
      ([entry]) => {
        visibleRef.current = entry.isIntersecting;
      },
      { rootMargin: "120px" },
    );
    observer.observe(container);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      surfaces.forEach((s) => s.dispose());
      surfaces.clear();
      stageRef.current?.dispose();
      stageRef.current = null;
    };
    // onBackend reports the backend once and `camera` is an initial placement; neither
    // should tear down and re-create the renderer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Add, update and retire surfaces as the layer list changes.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || status !== "ready") return;
    const live = new Set(layers.map((l) => l.key));

    for (const [key, surface] of surfacesRef.current) {
      if (!live.has(key)) {
        stage.scene.remove(surface.mesh);
        surface.dispose();
        surfacesRef.current.delete(key);
      }
    }
    for (const layer of layers) {
      const existing = surfacesRef.current.get(layer.key);
      if (existing) {
        existing.update(layer.field);
      } else {
        const surface = createSurface(layer.field, layer);
        surfacesRef.current.set(layer.key, surface);
        stage.scene.add(surface.mesh);
      }
    }
  }, [layers, status]);

  // The luminous boundary rail: one edge of the domain, lit rather than shaded.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || status !== "ready" || !boundary) return;
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(
        boundary === "xMin" || boundary === "xMax" ? 0.02 : 2.04,
        0.02,
        boundary === "yMin" || boundary === "yMax" ? 0.02 : 2.04,
      ),
      new THREE.MeshBasicMaterial({ color: PALETTE.sodium, toneMapped: false }),
    );
    const offset = 1.01;
    if (boundary === "xMin") bar.position.set(-offset, 0.005, 0);
    if (boundary === "xMax") bar.position.set(offset, 0.005, 0);
    if (boundary === "yMin") bar.position.set(0, 0.005, -offset);
    if (boundary === "yMax") bar.position.set(0, 0.005, offset);
    stage.scene.add(bar);
    return () => {
      stage.scene.remove(bar);
      bar.geometry.dispose();
      (bar.material as THREE.Material).dispose();
    };
  }, [boundary, status]);

  // The cutting plane, e.g. the near-boundary cutoff in the bulk panel.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || status !== "ready") return;
    if (markerRef.current) {
      stage.scene.remove(markerRef.current);
      markerRef.current.geometry.dispose();
      (markerRef.current.material as THREE.Material).dispose();
      markerRef.current = null;
    }
    if (!marker) return;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(2.04, 0.95),
      new THREE.MeshBasicMaterial({
        color: PALETTE.ion,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    const at = marker.at * 2 - 1;
    if (marker.axis === "x") {
      plane.rotation.y = Math.PI / 2;
      plane.position.set(at, 0.45, 0);
    } else {
      plane.position.set(0, 0.45, at);
    }
    stage.scene.add(plane);
    markerRef.current = plane;
  }, [marker, status]);

  return (
    <div className="relative" style={{ height }}>
      <div
        ref={containerRef}
        className="h-full w-full bg-[var(--figure)]"
      />
      {status !== "ready" && (
        <div className="caption absolute inset-0 grid place-items-center text-[var(--figure-faint)]">
          {status === "booting" ? "starting renderer…" : "3D unavailable in this browser"}
        </div>
      )}
    </div>
  );
}
