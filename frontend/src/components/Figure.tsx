import type { ReactNode } from "react";

/** A mathematical symbol inside an uppercased label; see `.label .sym`. */
export function Sym({ children }: { children: ReactNode }) {
  return <span className="sym">{children}</span>;
}

/**
 * A numbered figure with a caption and a live margin.
 *
 * The layout follows a printed paper: the figure breaks out wider than the prose column,
 * the caption sits beneath it in the sans face, and annotations hang in the margin. What a
 * paper cannot do is keep the margin current -- here those readouts recompute while the
 * figure is being dragged, which is the reason this is a page and not a PDF.
 *
 * Below the margin breakpoint the note stacks under the caption rather than disappearing;
 * the numbers are the argument, not an aside.
 */
export function Figure({
  number,
  title,
  caption,
  margin,
  controls,
  children,
}: {
  number: number;
  title: ReactNode;
  caption: ReactNode;
  margin?: ReactNode;
  controls?: ReactNode;
  children: ReactNode;
}) {
  return (
    <figure className="m-0 mt-14 scroll-mt-8" id={`figure-${number}`}>
      <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_var(--margin-col)]">
        <div className="min-w-0">
          <h2 className="text-[1.35rem] leading-snug">
            <span className="section-number mr-2 align-[0.18em]">FIG. {number}</span>
            {title}
          </h2>
          <div className="mt-4 overflow-hidden rounded-[3px] border border-[var(--figure-rule)] bg-[var(--figure)]">
            {children}
          </div>
          <figcaption className="caption mt-3">{caption}</figcaption>
          {controls && <div className="mt-5">{controls}</div>}
        </div>
        {margin && (
          <aside className="lg:pt-11">
            <div className="border-t border-[var(--rule)] pt-3 lg:border-t-2 lg:border-t-[var(--ink)]">
              {margin}
            </div>
          </aside>
        )}
      </div>
    </figure>
  );
}

/** One line of the margin: a label above a tabular value, with optional units beneath. */
export function MarginValue({
  label,
  value,
  note,
  tone = "ink",
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  tone?: "ink" | "exact" | "predicted";
}) {
  const color =
    tone === "exact"
      ? "var(--exact-ink)"
      : tone === "predicted"
        ? "var(--predicted-ink)"
        : "var(--ink)";
  return (
    <div className="border-b border-[var(--rule)] py-2.5 last:border-b-0">
      <div className="label">{label}</div>
      <div className="numeric mt-1 text-[0.98rem] leading-none" style={{ color }}>
        {value}
      </div>
      {note && <div className="caption mt-1.5 text-[0.72rem] leading-snug">{note}</div>}
    </div>
  );
}

/** A labelled slider. The value is echoed in the label so it is legible while dragging. */
export function Control({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  hint,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  hint?: ReactNode;
}) {
  return (
    <label className={`block ${disabled ? "opacity-45" : ""}`}>
      <span className="label flex items-baseline justify-between gap-3">
        <span>{label}</span>
        <span className="numeric text-[var(--ink)] normal-case">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <span className="caption -mt-2 block text-[0.72rem]">{hint}</span>}
    </label>
  );
}
