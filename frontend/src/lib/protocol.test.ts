import { describe, expect, it } from "vitest";
import {
  bulkFrameSize,
  correlatorFrameSize,
  FrameKind,
  MAGIC,
  parseFrame,
  VERSION,
} from "./protocol";

/** Build a frame the way `qft_operator/app/ws/protocol.py` does. */
function encodeBulk(nZ: number, nP: number, density: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bulkFrameSize(nZ, nP));
  const view = new DataView(buffer);
  view.setUint16(0, MAGIC, true);
  view.setUint8(2, VERSION);
  view.setUint8(3, FrameKind.BulkField);
  view.setUint32(4, 42, true);
  view.setUint16(8, nZ, true);
  view.setUint16(10, nP, true);
  const floats = [-6.9, 2.3, -2, 2, -20, -2, 1, -6.9, 1.5, 6.6];
  floats.forEach((value, i) => view.setFloat32(12 + 4 * i, value, true));
  new Uint8Array(buffer, 52).set(density);
  return buffer;
}

function encodeCorrelator(n: number): ArrayBuffer {
  const buffer = new ArrayBuffer(correlatorFrameSize(n));
  const view = new DataView(buffer);
  view.setUint16(0, MAGIC, true);
  view.setUint8(2, VERSION);
  view.setUint8(3, FrameKind.Correlator);
  view.setUint32(4, 7, true);
  view.setUint16(8, n, true);
  const floats = [-0.002, -0.0018, 1.5, 0.5, 0.02, 0.021];
  floats.forEach((value, i) => view.setFloat32(12 + 4 * i, value, true));
  for (let i = 0; i < 3 * n; i += 1) view.setFloat32(36 + 4 * i, i, true);
  return buffer;
}

describe("frame sizes", () => {
  it("matches the Python layout", () => {
    // 8-byte common header + 44-byte bulk header + one byte per grid point.
    expect(bulkFrameSize(16, 24)).toBe(8 + 44 + 16 * 24);
    // 8 + 28 (26 fields plus two padding bytes) + three float32 curves.
    expect(correlatorFrameSize(64)).toBe(8 + 28 + 3 * 4 * 64);
  });

  it("keeps the correlator payload 4-byte aligned", () => {
    expect((8 + 28) % 4).toBe(0);
  });
});

describe("parseFrame", () => {
  it("decodes a bulk frame", () => {
    const density = Uint8Array.from({ length: 16 * 24 }, (_, i) => i % 256);
    const frame = parseFrame(encodeBulk(16, 24, density));
    if (frame.kind !== FrameKind.BulkField) throw new Error("wrong kind");
    expect(frame.sequence).toBe(42);
    expect([frame.nZ, frame.nP]).toEqual([16, 24]);
    expect(frame.logZMin).toBeCloseTo(-6.9, 5);
    expect(frame.r).toBeCloseTo(1, 6);
    expect(frame.integral).toBeCloseTo(6.6, 5);
    expect(frame.density.length).toBe(16 * 24);
    expect(Array.from(frame.density.slice(0, 4))).toEqual([0, 1, 2, 3]);
  });

  it("decodes a correlator frame with aligned typed-array views", () => {
    const frame = parseFrame(encodeCorrelator(8));
    if (frame.kind !== FrameKind.Correlator) throw new Error("wrong kind");
    expect(frame.sequence).toBe(7);
    expect(frame.freeDimension).toBeCloseTo(1.5, 6);
    expect(frame.logR.length).toBe(8);
    expect(Array.from(frame.logR)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(frame.logWExact)).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect(Array.from(frame.logWPred)).toEqual([16, 17, 18, 19, 20, 21, 22, 23]);
  });

  it("rejects a foreign or future frame rather than misreading it", () => {
    const buffer = encodeCorrelator(4);
    new DataView(buffer).setUint16(0, 0x1234, true);
    expect(() => parseFrame(buffer)).toThrow(/magic/);

    const other = encodeCorrelator(4);
    new DataView(other).setUint8(2, 99);
    expect(() => parseFrame(other)).toThrow(/version/);

    const unknown = encodeCorrelator(4);
    new DataView(unknown).setUint8(3, 9);
    expect(() => parseFrame(unknown)).toThrow(/kind/);
  });
});
