// docs/papers/_template/paper_schema_extension.mjs
//
// Dashboard-sanitizer extension for the paper-HTML render path. The strict
// `markdownSchema` (dashboard/lib/markdown-sanitize.ts) strips exactly the four
// things the paper render needs: <figure>/<figcaption> (caption wrappers),
// data-epsref (the cross-ref hover hook), data-metric-key (the v1.1 number-
// provenance hook), and the eps-ref/eps-metric class values. This builds a
// `paperSchema` from the project's real markdownSchema plus those allowances,
// so the committed, pre-sanitized paper.html keeps its figure captions + typed
// cross-ref hooks while still stripping script/style/on* — the real XSS
// vectors.
//
// Proven by the spike (SPIKE_REPORT.md Proof 2: under this schema the tag
// census reports "STRIPPED: none — all tags survived", with data-epsref,
// eps-ref, data-metric-key, eps-metric, figure, figcaption, img and MathML all
// kept).
//
// Phase A ships this extension as a standalone, importable module. Phase 7
// (dashboard render) wires buildPaperSchema() into the dashboard render path;
// the dashboard work imports this from its committed location.
//
// TWO entry points:
//   buildPaperSchema(markdownSchema)  — PURE: takes the project's markdownSchema
//     and returns the extended paperSchema. The caller imports markdownSchema
//     from whichever dashboard tree has node_modules (tsx resolves its
//     transitive rehype-sanitize there). This is the form build_paper.py's
//     sanitizer driver uses, so it works from a worktree without node_modules.
//   resolveAndBuildPaperSchema()      — convenience: auto-imports THIS repo's
//     dashboard/lib/markdown-sanitize.ts (relative to import.meta.url) and
//     returns the extended schema. Works wherever the importing tree's
//     node_modules resolves rehype-sanitize.

/**
 * Extend the project's strict markdownSchema into the paperSchema.
 * @param {object} markdownSchema the project's strict schema (from
 *   dashboard/lib/markdown-sanitize.ts).
 * @returns {object} the extended schema (a deep clone; the input is not mutated).
 */
export function buildPaperSchema(markdownSchema) {
  const schema = JSON.parse(JSON.stringify(markdownSchema));

  // 1. allow the figure wrappers (img is already allowed by defaultSchema).
  schema.tagNames = Array.from(new Set([...(schema.tagNames ?? []), "figure", "figcaption"]));

  const attrs = (schema.attributes ??= {});
  // Mirror the project's uniqAttr bare-name-widening: a bare attr name (any
  // value) must OVERRIDE an existing restrictive [name, ...allowed] tuple,
  // because defaultSchema restricts <a>/<span> className to an allowlist that
  // would drop our eps-ref / eps-metric class values.
  const keyOf = (e) => (Array.isArray(e) ? e[0] : e);
  const add = (tag, names) => {
    const cur = (attrs[tag] ?? []).filter((e) => !names.includes(keyOf(e)));
    attrs[tag] = [...cur, ...names];
  };

  // 2. typed cross-ref hook on <a> + the eps-ref class. (hast camelCase)
  add("a", ["className", "dataEpsref", "target", "rel"]);
  // 3. number-provenance hook on <span> + the eps-metric class (v1.1).
  add("span", ["className", "dataMetricKey"]);
  // 4. figure/figcaption may carry className/id.
  add("figure", ["className", "id"]);
  add("figcaption", ["className", "id"]);

  return schema;
}

/**
 * Convenience: auto-import THIS repo's markdownSchema and build the paperSchema.
 * Resolves dashboard/lib/markdown-sanitize.ts relative to this module so it
 * works from the repo root, a worktree, or the dashboard tree — provided the
 * importing tree's node_modules resolves rehype-sanitize.
 * @returns {Promise<object>} the extended schema.
 */
export async function resolveAndBuildPaperSchema() {
  const { fileURLToPath } = await import("node:url");
  const { dirname, resolve } = await import("node:path");
  const HERE = dirname(fileURLToPath(import.meta.url)); // docs/papers/_template
  const REPO = resolve(HERE, "..", "..", ".."); // repo root (3 up)
  const SANITIZE = resolve(REPO, "dashboard/lib/markdown-sanitize.ts");
  const { markdownSchema } = await import(SANITIZE);
  return buildPaperSchema(markdownSchema);
}
