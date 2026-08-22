/**
 * Binary frame decoder, mirroring `qft_operator/app/ws/protocol.py`.
 *
 * Common 8-byte header, little-endian:
 *   uint16 magic = 0x5146 ("QF") · uint8 version · uint8 kind · uint32 sequence
 *
 * The round-trip is pinned on the Python side by `tests/app/test_protocol.py`; the byte
 * offsets below are the same constants and must move together.
 */

export const MAGIC = 0x5146;
export const VERSION = 1;

export const FrameKind = {
  BulkField: 1,
  Correlator: 2,
} as const;

const HEADER_BYTES = 8;
const BULK_HEADER_BYTES = 44;
/** 28, not 26: the extra two bytes are padding that keeps the payload 4-byte aligned. */
const CORRELATOR_HEADER_BYTES = 28;

export interface BulkFrame {
  kind: typeof FrameKind.BulkField;
  sequence: number;
  nZ: number;
  nP: number;
  logZMin: number;
  logZMax: number;
  pMin: number;
  pMax: number;
  /** Colour-map floor and ceiling in natural log units. */
  logLow: number;
  logHigh: number;
  r: number;
  logEps: number;
  delta: number;
  /** Converged contact integral over z > ε — float32, never quantized. */
  integral: number;
  /** Density quantized to 8 bits against [logLow, logHigh]; display only. */
  density: Uint8Array;
}

export interface CorrelatorFrame {
  kind: typeof FrameKind.Correlator;
  sequence: number;
  gammaExact: number;
  gammaPred: number;
  freeDimension: number;
  logM: number;
  coupling: number;
  runningCoupling: number;
  logR: Float32Array;
  logWExact: Float32Array;
  logWPred: Float32Array;
}

export type Frame = BulkFrame | CorrelatorFrame;

/**
 * Decode one server frame.
 *
 * @throws {Error} if the magic or version does not match, which means the frontend and
 *   backend protocol modules have drifted apart — worth failing loudly rather than
 *   rendering misread bytes.
 */
export function parseFrame(buffer: ArrayBuffer): Frame {
  const view = new DataView(buffer);
  const magic = view.getUint16(0, true);
  const version = view.getUint8(2);
  if (magic !== MAGIC) throw new Error(`bad frame magic 0x${magic.toString(16)}`);
  if (version !== VERSION) throw new Error(`unsupported frame version ${version}`);

  const kind = view.getUint8(3);
  const sequence = view.getUint32(4, true);

  if (kind === FrameKind.BulkField) {
    const nZ = view.getUint16(HEADER_BYTES, true);
    const nP = view.getUint16(HEADER_BYTES + 2, true);
    const f = (i: number) => view.getFloat32(HEADER_BYTES + 4 + 4 * i, true);
    return {
      kind: FrameKind.BulkField,
      sequence,
      nZ,
      nP,
      logZMin: f(0),
      logZMax: f(1),
      pMin: f(2),
      pMax: f(3),
      logLow: f(4),
      logHigh: f(5),
      r: f(6),
      logEps: f(7),
      delta: f(8),
      integral: f(9),
      density: new Uint8Array(buffer, HEADER_BYTES + BULK_HEADER_BYTES, nZ * nP),
    };
  }

  if (kind === FrameKind.Correlator) {
    const n = view.getUint16(HEADER_BYTES, true);
    const f = (i: number) => view.getFloat32(HEADER_BYTES + 4 + 4 * i, true);
    const base = HEADER_BYTES + CORRELATOR_HEADER_BYTES;
    return {
      kind: FrameKind.Correlator,
      sequence,
      gammaExact: f(0),
      gammaPred: f(1),
      freeDimension: f(2),
      logM: f(3),
      coupling: f(4),
      runningCoupling: f(5),
      logR: new Float32Array(buffer, base, n),
      logWExact: new Float32Array(buffer, base + 4 * n, n),
      logWPred: new Float32Array(buffer, base + 8 * n, n),
    };
  }

  throw new Error(`unknown frame kind ${kind}`);
}

/** Byte length of a bulk frame, for tests and sanity checks. */
export function bulkFrameSize(nZ: number, nP: number): number {
  return HEADER_BYTES + BULK_HEADER_BYTES + nZ * nP;
}

/** Byte length of a correlator frame. */
export function correlatorFrameSize(n: number): number {
  return HEADER_BYTES + CORRELATOR_HEADER_BYTES + 3 * 4 * n;
}
