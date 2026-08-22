import type { ReactNode } from "react";

/** A titled card. `aside` sits opposite the title, for status chips and readouts. */
export function Panel({
  title,
  subtitle,
  aside,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-slate-700/60 bg-slate-900/50 p-4">
      <header className="mb-3 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase">{title}</h2>
          {subtitle && <p className="mt-1 text-xs leading-relaxed text-slate-400">{subtitle}</p>}
        </div>
        {aside && <div className="shrink-0 text-right text-xs text-slate-400">{aside}</div>}
      </header>
      {children}
    </section>
  );
}

/** A labelled numeric readout in a fixed-width font, so digits do not jitter. */
export function Readout({
  label,
  value,
  hint,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[0.65rem] tracking-wide text-slate-500 uppercase">{label}</span>
      <span className="font-mono text-sm text-slate-100 tabular-nums">{value}</span>
      {hint && <span className="text-[0.65rem] text-slate-500">{hint}</span>}
    </div>
  );
}
