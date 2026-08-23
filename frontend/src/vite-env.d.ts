/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Whether to look for the FastAPI backend at all.
   *
   * The page is designed to work with no server, so the probe is a progressive
   * enhancement rather than a requirement. A static build has no backend by
   * construction, and probing for one there buys nothing while logging a 404 on every
   * load -- so it defaults off in production and on in dev, where Vite proxies to
   * localhost:8000. Set `VITE_BACKEND_PROBE=on` when building a bundle that FastAPI
   * itself will serve.
   */
  readonly VITE_BACKEND_PROBE?: "on" | "off";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
