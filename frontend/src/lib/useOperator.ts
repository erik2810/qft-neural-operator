/** Load the exported operator once, for browser-side inference. */

import { useEffect, useState } from "react";
import { Operator } from "./operator";

export type OperatorState =
  | { status: "loading" }
  | { status: "ready"; operator: Operator }
  | { status: "unavailable"; reason: string };

/**
 * Fetch `manifest.json` and `weights.bin` from `base`.
 *
 * A missing export is not an error: the page still shows the exact physics, and the
 * panels label the prediction as unavailable rather than drawing nothing.
 */
export function useOperator(base = `${import.meta.env.BASE_URL}operator`): OperatorState {
  const [state, setState] = useState<OperatorState>({ status: "loading" });
  useEffect(() => {
    let cancelled = false;
    Operator.load(base)
      .then((operator) => {
        if (!cancelled) setState({ status: "ready", operator });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({ status: "unavailable", reason: error instanceof Error ? error.message : "" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [base]);
  return state;
}
