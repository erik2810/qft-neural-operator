import { useMemo } from "react";

export interface Series {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: string;
  label?: string;
  dashed?: boolean;
  width?: number;
}

export interface Marker {
  x: number;
  y: number;
  color: string;
}

interface PlotProps {
  series: Series[];
  markers?: Marker[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  /** Force a y range; otherwise it is taken from the data with a small margin. */
  yDomain?: [number, number];
  /** Draw a dashed horizontal line at y = 0, e.g. the free-theory level. */
  zeroLine?: boolean;
}

const MARGIN = { top: 8, right: 12, bottom: 30, left: 58 };
const WIDTH = 560;

function extent(values: ArrayLike<number>[]): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const array of values) {
    for (let i = 0; i < array.length; i += 1) {
      const v = array[i];
      if (!Number.isFinite(v)) continue;
      if (v < low) low = v;
      if (v > high) high = v;
    }
  }
  if (!Number.isFinite(low) || !Number.isFinite(high)) return [0, 1];
  if (low === high) return [low - 1, high + 1];
  return [low, high];
}

function ticks(low: number, high: number, count = 5): number[] {
  const raw = (high - low) / count;
  if (!(raw > 0)) return [low];
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const start = Math.ceil(low / step) * step;
  const out: number[] = [];
  for (let v = start; v <= high + 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function format(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(1);
  return String(Number(value.toPrecision(4)));
}

/** A compact line plot. Deliberately plain: these panels are read, not admired. */
export function Plot({
  series,
  markers = [],
  height = 200,
  xLabel,
  yLabel,
  yDomain,
  zeroLine = false,
}: PlotProps) {
  const inner = {
    width: WIDTH - MARGIN.left - MARGIN.right,
    height: height - MARGIN.top - MARGIN.bottom,
  };

  const { xScale, yScale, xTicks, yTicks, paths } = useMemo(() => {
    const [xLow, xHigh] = extent(series.map((s) => s.x));
    const [dataLow, dataHigh] = extent(series.map((s) => s.y));
    const pad = (dataHigh - dataLow) * 0.08 || 1;
    const [yLow, yHigh] = yDomain ?? [dataLow - pad, dataHigh + pad];
    const sx = (v: number) => ((v - xLow) / (xHigh - xLow || 1)) * inner.width;
    const sy = (v: number) => inner.height - ((v - yLow) / (yHigh - yLow || 1)) * inner.height;
    return {
      xScale: sx,
      yScale: sy,
      xTicks: ticks(xLow, xHigh),
      yTicks: ticks(yLow, yHigh, 4),
      paths: series.map((s) => {
        let d = "";
        for (let i = 0; i < s.x.length; i += 1) {
          const px = sx(s.x[i]);
          const py = sy(s.y[i]);
          if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
          d += `${d ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`;
        }
        return d;
      }),
    };
  }, [series, yDomain, inner.width, inner.height]);

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${height}`}
      className="w-full"
      role="img"
      aria-label={`${yLabel ?? "value"} against ${xLabel ?? "x"}`}
    >
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={inner.width} y1={yScale(t)} y2={yScale(t)} stroke="#1e293b" />
            <text x={-8} y={yScale(t)} dy="0.32em" textAnchor="end" fontSize={9} fill="#64748b">
              {format(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line y1={0} y2={inner.height} x1={xScale(t)} x2={xScale(t)} stroke="#1e293b" />
            <text
              x={xScale(t)}
              y={inner.height + 14}
              textAnchor="middle"
              fontSize={9}
              fill="#64748b"
            >
              {format(t)}
            </text>
          </g>
        ))}
        {zeroLine && (
          <line
            x1={0}
            x2={inner.width}
            y1={yScale(0)}
            y2={yScale(0)}
            stroke="#475569"
            strokeDasharray="3 3"
          />
        )}
        {paths.map((d, i) => (
          <path
            key={series[i].label ?? i}
            d={d}
            fill="none"
            stroke={series[i].color}
            strokeWidth={series[i].width ?? 1.6}
            strokeDasharray={series[i].dashed ? "5 4" : undefined}
          />
        ))}
        {markers.map((m, i) => (
          <circle key={i} cx={xScale(m.x)} cy={yScale(m.y)} r={3} fill={m.color} />
        ))}
      </g>
      {xLabel && (
        <text x={WIDTH / 2} y={height - 2} textAnchor="middle" fontSize={10} fill="#94a3b8">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          transform={`translate(12,${height / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={10}
          fill="#94a3b8"
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
}

/** Colour swatch + label, matching {@link Plot} stroke styling. */
export function Legend({ items }: { items: { color: string; label: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden>
            <line
              x1={0}
              y1={4}
              x2={18}
              y2={4}
              stroke={item.color}
              strokeWidth={2}
              strokeDasharray={item.dashed ? "5 4" : undefined}
            />
          </svg>
          {item.label}
        </span>
      ))}
    </div>
  );
}

export const COLORS = {
  exact: "#38bdf8",
  predicted: "#fb923c",
  accent: "#a78bfa",
  muted: "#64748b",
  warn: "#f87171",
} as const;
