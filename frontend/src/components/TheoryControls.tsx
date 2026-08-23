import { Formula } from "./Formula";
import type { Family, Theory } from "../lib/physics";

const FAMILIES: { value: Family; label: string; tex: string }[] = [
  { value: "free", label: "free", tex: "V=0" },
  { value: "sine_gordon", label: "Sine-Gordon", tex: "-\\lambda(e^{\\xi\\phi}+e^{-\\xi\\phi}-2)" },
  { value: "phi4", label: "φ⁴", tex: "\\lambda\\phi^4" },
  { value: "polynomial", label: "polynomial", tex: "\\lambda\\sum_k c_k\\phi^k" },
  { value: "gp_fourier", label: "Gaussian process", tex: "\\lambda\\,v_{\\rm GP}(\\phi)" },
];

/** Parameter controls for one theory. */
export function TheoryControls({
  theory,
  onChange,
}: {
  theory: Theory;
  onChange: (theory: Theory) => void;
}) {
  const set = (patch: Partial<Theory>) => onChange({ ...theory, ...patch });
  const active = FAMILIES.find((f) => f.value === theory.family);
  const seeded = theory.family === "polynomial" || theory.family === "gp_fourier";

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <label className="flex flex-col gap-1 text-xs text-[var(--dim)] sm:col-span-2">
        <span>interaction</span>
        <div className="flex flex-wrap gap-1.5">
          {FAMILIES.map((family) => (
            <button
              key={family.value}
              type="button"
              onClick={() => set({ family: family.value })}
              className={`rounded border px-2 py-1 text-xs transition-colors ${
                theory.family === family.value
                  ? "border-[var(--sodium)] bg-[color-mix(in_srgb,var(--sodium)_14%,transparent)] text-[var(--sodium)]"
                  : "border-[var(--rule)] text-[var(--dim)] hover:border-[var(--steel)]"
              }`}
            >
              {family.label}
            </button>
          ))}
        </div>
        {active && (
          <span className="mt-1 text-[var(--dim)]">
            <Formula tex={`V(\\phi)=${active.tex}`} />
          </span>
        )}
      </label>

      <label className="flex flex-col gap-1 text-xs text-[var(--dim)]">
        <span>
          coupling <Formula tex="\lambda" /> = {theory.coupling.toFixed(4)}
        </span>
        <input
          type="range"
          min={-0.06}
          max={0.06}
          step={0.0005}
          value={theory.coupling}
          disabled={theory.family === "free"}
          onChange={(e) => set({ coupling: Number(e.target.value) })}
          className="disabled:opacity-40"
        />
      </label>

      <label className="flex flex-col gap-1 text-xs text-[var(--dim)]">
        <span>
          vertex exponent <Formula tex="\xi" /> = {theory.xi.toFixed(2)}
        </span>
        <input
          type="range"
          min={0.3}
          max={1.4}
          step={0.01}
          value={theory.xi}
          disabled={theory.family !== "sine_gordon"}
          onChange={(e) => set({ xi: Number(e.target.value) })}
          className="disabled:opacity-40"
        />
      </label>

      {seeded && (
        <label className="flex flex-col gap-1 text-xs text-[var(--dim)] sm:col-span-2">
          <span>draw = {theory.seed}</span>
          <input
            type="range"
            min={0}
            max={40}
            step={1}
            value={theory.seed}
            onChange={(e) => set({ seed: Number(e.target.value) })}
            className=""
          />
          <span className="text-[0.65rem] text-[var(--dim)]">
            A fresh draw from the same distribution the operator was trained on — nothing
            about it is Sine-Gordon or φ⁴.
          </span>
        </label>
      )}
    </div>
  );
}
