import katex from "katex";
import { useMemo } from "react";

/** Render a LaTeX expression with KaTeX. Invalid input renders as literal text. */
export function Formula({
  tex,
  block = false,
  className = "",
}: {
  tex: string;
  block?: boolean;
  className?: string;
}) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { displayMode: block, throwOnError: false });
    } catch {
      return tex;
    }
  }, [tex, block]);
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
