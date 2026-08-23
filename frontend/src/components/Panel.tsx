import type { ReactNode } from "react";

/**
 * A titled section.
 *
 * The rule under the header is amber and full-bleed: it is the only decorative line on the
 * page, and it echoes the boundary rail in the bulk scene rather than being an ornament.
 */
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
    <section className="border border-[var(--rule)] bg-[var(--panel)]">
      <header className="border-b border-[var(--rule)] px-5 py-4">
        <div className="flex items-baseline justify-between gap-6">
          <h2 className="display text-[1.05rem] leading-tight text-[var(--bright)]">{title}</h2>
          {aside}
        </div>
        {subtitle && (
          <p className="mt-2 max-w-[68ch] text-[0.82rem] leading-relaxed text-[var(--dim)]">
            {subtitle}
          </p>
        )}
      </header>
      <div className="px-5 py-5">{children}</div>
    </section>
  );
}

/** A labelled figure. Values are tabular so digits do not jump as sliders move. */
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
    <div className="flex flex-col gap-1">
      <span className="eyebrow">{label}</span>
      <span className="numeric text-[0.95rem] text-[var(--bright)]">{value}</span>
      {hint && <span className="text-[0.68rem] text-[var(--dim)]">{hint}</span>}
    </div>
  );
}
