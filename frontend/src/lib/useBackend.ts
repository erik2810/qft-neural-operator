/**
 * Backend detection and request/response-paced WebSocket streams.
 *
 * The page works with no server at all: every panel falls back to computing its physics
 * locally (`physics.ts`, `bulk.ts`) and running the exported operator in the browser
 * (`operator.ts`). When a backend *is* reachable it takes over the two heavy paths —
 * higher-resolution bulk quadrature and full-precision PyTorch inference — over the
 * binary protocol.
 *
 * Both sockets are paced: exactly one request in flight at a time, with the newest
 * pending state sent as soon as the previous frame lands. Dragging a slider therefore
 * throttles itself to whatever the server can sustain instead of queuing up stale
 * frames the server would compute and the client would discard.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { parseFrame, type BulkFrame, type CorrelatorFrame, FrameKind } from "./protocol";

export type BackendStatus = "checking" | "online" | "offline";

/** Probe `/health` once on mount. */
export function useBackendStatus(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("checking");
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2000);
    fetch("/health", { signal: controller.signal })
      .then((response) => {
        if (!cancelled) setStatus(response.ok ? "online" : "offline");
      })
      .catch(() => {
        if (!cancelled) setStatus("offline");
      })
      .finally(() => clearTimeout(timer));
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);
  return status;
}

function socketUrl(path: string): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}${path}`;
}

interface PacedSocket<TRequest, TFrame> {
  frame: TFrame | null;
  send: (request: TRequest) => void;
  connected: boolean;
}

/**
 * Keep one socket open and at most one request in flight.
 *
 * @param path WebSocket path on the backend.
 * @param enabled Open the socket at all; `false` leaves the hook inert so callers can
 *   mount it unconditionally and let the backend probe decide.
 * @param accept Narrow an incoming frame to the kind this socket expects.
 */
function usePacedSocket<TRequest, TFrame>(
  path: string,
  enabled: boolean,
  accept: (frame: ReturnType<typeof parseFrame>) => TFrame | null,
): PacedSocket<TRequest, TFrame> {
  const [frame, setFrame] = useState<TFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const inFlight = useRef(false);
  const pending = useRef<TRequest | null>(null);

  const flush = useCallback(() => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (inFlight.current || pending.current === null) return;
    const request = pending.current;
    pending.current = null;
    inFlight.current = true;
    socket.send(JSON.stringify(request));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const socket = new WebSocket(socketUrl(path));
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => {
      setConnected(true);
      flush();
    };
    socket.onmessage = (event) => {
      inFlight.current = false;
      if (typeof event.data === "string") {
        // The server reports validation problems as JSON rather than a binary frame.
        console.warn("backend rejected a request:", event.data);
      } else {
        const accepted = accept(parseFrame(event.data as ArrayBuffer));
        if (accepted) setFrame(accepted);
      }
      flush();
    };
    socket.onclose = () => {
      setConnected(false);
      inFlight.current = false;
    };
    socket.onerror = () => setConnected(false);

    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [path, enabled, flush, accept]);

  const send = useCallback(
    (request: TRequest) => {
      pending.current = request;
      flush();
    },
    [flush],
  );

  return { frame, send, connected };
}

export interface BulkRequest {
  r: number;
  log_eps: number;
  n_z: number;
  n_p: number;
  decades_below: number;
  decades_above: number;
}

const acceptBulk = (frame: ReturnType<typeof parseFrame>) =>
  frame.kind === FrameKind.BulkField ? (frame as BulkFrame) : null;

/** Stream bulk-density frames from the server's quadrature. */
export function useBulkStream(enabled: boolean) {
  return usePacedSocket<BulkRequest, BulkFrame>("/ws/bulk", enabled, acceptBulk);
}

export interface CorrelatorRequest {
  family: string;
  coupling: number;
  xi: number;
  seed: number;
  log_m: number;
  moment: number;
  v_phi: number[];
}

const acceptCorrelator = (frame: ReturnType<typeof parseFrame>) =>
  frame.kind === FrameKind.Correlator ? (frame as CorrelatorFrame) : null;

/** Stream correlator frames — exact and PyTorch-predicted — from the server. */
export function useCorrelatorStream(enabled: boolean) {
  return usePacedSocket<CorrelatorRequest, CorrelatorFrame>(
    "/ws/correlator",
    enabled,
    acceptCorrelator,
  );
}
