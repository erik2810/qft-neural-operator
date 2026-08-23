import { Control, Sym } from "./Figure";
import { Formula } from "./Formula";
import type { Family, Theory } from "../lib/physics";

const FAMILIES: { value: Family; label: string; tex: string }[] = [
  { value: "free", label: "free", tex: "0" },
  { value: "sine_gordon", label: "Sine-Gordon", tex: "-\\lambda(e^{\\xi\\phi}+e^{-\\xi\\phi}-2)" },
  { value: "phi4", label: "φ⁴", tex: "\\lambda\\phi^4" },
  { value: "polynomial", label: "polynomial", tex: "\\lambda\\sum_k c_k\\phi^k" },
  { value: "gp_fourier", label: "Gaussian process", tex: "\\lambda\\,v_{\\rm GP}(\\phi)" },
];

/**
 * The interaction chooser, set as a parameter table rather than a control panel.
 *
 * The families are radio buttons in behaviour and a list of theories in appearance, because
 * that is what they are: five entries in the space the operator was fitted over, the last
 * two of which have no closed form at all.
 */
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
    <div className="grid gap-x-10 gap-y-7 lg:grid-cols-[minmax(0,1fr)_var(--margin-col)]">
      <div className="min-w-0">
        <fieldset className="border-0 p-0">
          <legend className="label mb-2">interaction</legend>
          <div role="radiogroup" aria-label="Interaction potential" className="flex flex-wrap">
            {FAMILIES.map((family) => {
              const selected = theory.family === family.value;
              return (
                <button
                  key={family.value}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => set({ family: family.value })}
                  className={`numeric inline-flex min-h-11 cursor-pointer items-center border-b-2 px-3.5 text-[0.82rem] transition-colors duration-150 ${
                    selected
                      ? "border-[var(--exact-ink)] text-[var(--ink)]"
                      : "border-transparent text-[var(--ink-faint)] hover:border-[var(--rule-strong)] hover:text-[var(--ink-soft)]"
                  }`}
                >
                  {family.label}
                </button>
              );
            })}
          </div>
        </fieldset>

        {active && (
          <p className="mt-4 border-l-2 border-[var(--rule)] pl-4 text-[0.95rem] text-[var(--ink-soft)]">
            <Formula tex={`V(\\phi)=${active.tex}`} />
          </p>
        )}

        <div className="mt-4 grid gap-x-10 gap-y-1 sm:grid-cols-2">
          <Control
            label={<>coupling <Sym>λ</Sym></>}
            value={theory.coupling}
            min={-0.06}
            max={0.06}
            step={0.0005}
            disabled={theory.family === "free"}
            onChange={(coupling) => set({ coupling })}
          />
          <Control
            label={<>vertex exponent <Sym>ξ</Sym></>}
            value={theory.xi}
            min={0.3}
            max={1.4}
            step={0.01}
            disabled={theory.family !== "sine_gordon"}
            onChange={(xi) => set({ xi })}
          />
          {seeded && (
            <Control
              label={<>draw</>}
              value={theory.seed}
              min={0}
              max={40}
              step={1}
              onChange={(seed) => set({ seed })}
              hint="a fresh draw from the distribution the operator was fitted over"
            />
          )}
        </div>
      </div>
      <div />
    </div>
  );
}
