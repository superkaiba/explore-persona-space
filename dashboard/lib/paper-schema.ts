/**
 * The paper-HTML sanitizer schema for the dashboard render path (Phase C2).
 *
 * The committed `docs/papers/issue_<N>/paper.html` is ALREADY sanitized at build
 * time by `scripts/build_paper.py`, which runs THIS repo's real
 * `lib/markdown-sanitize.ts` under the same `buildPaperSchema` extension. We
 * re-sanitize on render as defence in depth (the file is committed, so re-running
 * the gate guarantees a tampered commit can't smuggle script/style/on* into the
 * served HTML) using the IDENTICAL schema, so the paper hooks the render needs —
 * `<figure>`/`<figcaption>`, `data-epsref`/`eps-ref`, MathML, `data-metric-key` —
 * survive while script, style, on-handlers and `javascript:` stay stripped.
 *
 * `buildPaperSchema` is the SINGLE source of truth for the paper allow-list,
 * imported from the committed `docs/papers/_template/paper_schema_extension.mjs`
 * (the same module `build_paper.py`'s sanitizer driver imports). Keeping the one
 * import — rather than re-deriving the four additions here — means the build-time
 * and render-time schemas can never drift.
 */
import { markdownSchema } from "./markdown-sanitize";
import type { Schema } from "hast-util-sanitize";
// The schema extension lives in the repo's docs/papers/_template (a sibling of
// dashboard/, under the outputFileTracingRoot). The relative import is resolved
// + bundled statically at build time, so there is no runtime file dependency
// across checkouts. `.mjs` is plain JS (allowJs); tsc infers its types.
//
// We import ONLY `buildPaperSchema` here (the pure form — the dashboard already
// has markdownSchema in hand). The module's OTHER export,
// `resolveAndBuildPaperSchema`, is NOT dead code and must NOT be deleted: it is
// the node/tsx-only convenience entry that auto-imports the repo's
// markdown-sanitize.ts via its own import.meta.url, used by `build_paper.py`'s
// sanitizer driver (which runs outside the dashboard bundle). It is unreferenced
// from this dashboard tree by design.
import { buildPaperSchema } from "../../docs/papers/_template/paper_schema_extension.mjs";

/** The paper render-path sanitizer schema. Built once at module load. */
export const paperSchema: Schema = buildPaperSchema(markdownSchema) as Schema;
