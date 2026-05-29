/**
 * Sanitize schemas for the shared <MarkdownDoc> render pipeline.
 *
 * Two schemas, two trust levels:
 *
 *   - `markdownSchema` — the STRICT schema for untrusted / public markdown
 *     bodies (task / docs / results / overview / updates). Extends
 *     rehype-sanitize's `defaultSchema` to additionally allow the markup
 *     the dashboard's render pipeline depends on (KaTeX output, syntax-
 *     highlight classes, `<details>/<summary>`, heading ids, and the
 *     client-injected `<mark data-comment-id>` anchors). It does NOT allow
 *     inline `style`, `<style>`, `<script>`, `on*` handlers, or
 *     `javascript:` hrefs.
 *
 *   - `legacySchema` — the WIDE schema for TRUSTED legacy Sagan-card HTML
 *     bodies (~21 analyzer-generated infographics committed to git, carrying
 *     `<!-- legacy-sagan-card -->`). Those bodies are full inline-styled SVG
 *     charts with scoped `<style>` blocks and `style=` attributes, so the
 *     strict schema would destroy them. The legacy schema additionally
 *     allows the SVG element set, scoped `<style>`, and `style=`. It STILL
 *     strips `<script>`, `on*` handlers, and `javascript:`/`data:` script
 *     URLs from the surrounding markup — the actual JS-execution vectors —
 *     so making these public is safe.
 *
 *     NOTE on `style`/`<style>`: hast-util-sanitize@5 does NOT parse or
 *     re-clean CSS declarations; an allowed `style` value passes through
 *     verbatim. The sanitizer therefore does NOT enforce anything about the
 *     CSS itself — it neither strips nor inspects `@import`/`url(...)`. That
 *     is acceptable ONLY because the legacy bodies are trusted by provenance
 *     (analyzer-generated, committed to git, scoped to `.cr-<N>` selectors);
 *     the absence of `@import`/`url()` exfiltration is a property of that
 *     trusted corpus, NOT a guarantee the sanitizer provides. CSS cannot
 *     execute JavaScript in modern browsers (legacy `expression()` is gone),
 *     so the residual surface is visual-only / CSS-fetch, not script
 *     execution. The strict `markdownSchema` (untrusted bodies) allows
 *     NEITHER inline `style` nor `<style>`, so user-authored markdown can
 *     never reach this path.
 *
 * IMPORTANT: hast property names are camelCase (rehype-raw maps raw-HTML
 * `class` -> `className`, `stroke-width` -> `strokeWidth`,
 * `data-comment-id` -> `dataCommentId`, etc.). The allowlists below use
 * those hast spellings, NOT the raw-HTML attribute spellings.
 *
 * Render-order contract (enforced by MarkdownDoc): rehypeRaw ->
 * rehypeSanitize(schema) -> rehypeKatex -> rehypeHighlight -> rehypeSlug.
 * Sanitize runs AFTER raw (to clean injected HTML) and BEFORE the trusted
 * class-adding plugins (katex/highlight) so their classes survive — the
 * schema only needs to PERMIT the className/id shapes those plugins emit,
 * which it does via the wildcard `className`/`id` allowances below.
 */
import { defaultSchema } from "rehype-sanitize";
import type { Schema } from "hast-util-sanitize";

/** Deep-ish clone helper so we never mutate the imported defaultSchema. */
function cloneSchema(schema: Schema): Schema {
  return JSON.parse(JSON.stringify(schema)) as Schema;
}

// Block-ish + inline elements that carry className/id in our rendered
// output (KaTeX wraps in span/div, highlight wraps in span/code, the
// collapsible layer wraps in section, headings carry slug ids).
const CLASSNAME_ID_TAGS = [
  "span",
  "code",
  "pre",
  "div",
  "section",
  "p",
  "li",
  "ul",
  "ol",
  "table",
  "thead",
  "tbody",
  "tr",
  "td",
  "th",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "figure",
  "figcaption",
  "blockquote",
  "details",
  "summary",
] as const;

// Declared before the schema builders below because those builders run at
// module-eval time (the exported `markdownSchema`/`legacySchema` consts call
// them), and `const` arrays are NOT hoisted — referencing them after the
// builders would throw a temporal-dead-zone ReferenceError.
const MATHML_TAGS = [
  "math",
  "semantics",
  "annotation",
  "mrow",
  "mi",
  "mo",
  "mn",
  "ms",
  "mtext",
  "mspace",
  "msup",
  "msub",
  "msubsup",
  "mfrac",
  "mroot",
  "msqrt",
  "munder",
  "mover",
  "munderover",
  "mtable",
  "mtr",
  "mtd",
  "mpadded",
  "mphantom",
  "menclose",
  "mstyle",
];

const SVG_TAGS = [
  "svg",
  "g",
  "defs",
  "title",
  "desc",
  "line",
  "rect",
  "circle",
  "ellipse",
  "path",
  "polygon",
  "polyline",
  "text",
  "tspan",
  "textPath",
  "linearGradient",
  "radialGradient",
  "stop",
  "clipPath",
  "mask",
  "marker",
  "use",
  "symbol",
  "pattern",
  "image",
];

/**
 * Strict schema for untrusted markdown. Built once at module load.
 */
export const markdownSchema: Schema = buildMarkdownSchema();

function buildMarkdownSchema(): Schema {
  const schema = cloneSchema(defaultSchema);

  // Preserve heading ids verbatim. defaultSchema's clobberPrefix would
  // rewrite `id`/`name` to `user-content-…`, which breaks #anchor links,
  // the TOC scroll-to, and rehype-slug's emitted ids. We turn clobbering
  // off because the dashboard relies on raw heading ids end-to-end.
  schema.clobberPrefix = "";
  schema.clobber = [];

  // `strip` controls which DISALLOWED tags have their TEXT CONTENT removed
  // too (rather than unwrapped — kept as visible text). defaultSchema strips
  // <script> content; we additionally strip <style> so a `<style>` block in
  // an untrusted markdown body doesn't dump its CSS source as visible body
  // text. (Neither tag is in this schema's tagNames, so both are removed;
  // strip just ensures the inner text goes with them. Not an XSS fix — a
  // disallowed <style> never applies — but it keeps untrusted bodies clean.)
  schema.strip = uniq([...(schema.strip ?? []), "script", "style"]);

  schema.tagNames = uniq([
    ...(schema.tagNames ?? []),
    // <details>/<summary> are already in defaultSchema but list them so the
    // intent is explicit and so a future defaultSchema change can't silently
    // drop them.
    "details",
    "summary",
    "section",
    "mark",
    // KaTeX emits MathML alongside HTML spans. Allow the MathML element set
    // so `<math>…</math>` survives for screen readers / MathML-capable
    // browsers. (The visual rendering uses the katex <span> tree, which is
    // already covered by span/className below.)
    ...MATHML_TAGS,
  ]);

  const attrs = (schema.attributes ??= {});

  // Allow className + id on every tag we render classes/ids onto.
  for (const tag of CLASSNAME_ID_TAGS) {
    attrs[tag] = uniqAttr([...(attrs[tag] ?? []), "className", "id"]);
  }

  // <mark>: allow the client-injected anchor data attributes. The client
  // wraps committed comment anchors in `<mark data-comment-id>` and pending
  // selections in `<mark data-anchor-pending>` AFTER render, but a body may
  // also legitimately contain a plain <mark>, so permit the shape here.
  attrs.mark = uniqAttr([
    ...(attrs.mark ?? []),
    "className",
    "id",
    "dataCommentId",
    "dataAnchorPending",
  ]);

  // KaTeX puts presentational data on its spans/divs; allow the aria/style-
  // free attributes it needs. KaTeX uses inline `style` for glyph metrics —
  // we deliberately do NOT allow inline style on the markdown path, so we
  // rely on the katex CSS class layer for positioning. (Accepted tradeoff:
  // very tall fractions/roots may be slightly less precise without inline
  // style, but no math is authored in the corpus today; correctness of the
  // security gate wins.)
  attrs.span = uniqAttr([
    ...(attrs.span ?? []),
    "className",
    "id",
    "ariaHidden",
    "dataCommentId",
    "dataAnchorPending",
  ]);
  attrs.div = uniqAttr([...(attrs.div ?? []), "className", "id", "ariaHidden"]);

  // MathML element attributes used by KaTeX's MathML output.
  for (const tag of MATHML_TAGS) {
    attrs[tag] = uniqAttr([
      ...(attrs[tag] ?? []),
      "className",
      "id",
      "mathvariant",
      "encoding",
      "display",
      "scriptlevel",
      "displaystyle",
      "ariaHidden",
    ]);
  }

  // Wildcard: keep id + className addressable everywhere (defaultSchema
  // already allows id on *; add className so highlight/katex classes that
  // land on otherwise-unlisted inline tags survive).
  attrs["*"] = uniqAttr([...(attrs["*"] ?? []), "className"]);

  // Heading-rank ids from rehype-slug.
  for (const h of ["h1", "h2", "h3", "h4", "h5", "h6"]) {
    attrs[h] = uniqAttr([...(attrs[h] ?? []), "id", "className"]);
  }

  return schema;
}

/**
 * Wide schema for trusted legacy Sagan-card HTML. Built once at module load.
 */
export const legacySchema: Schema = buildLegacySchema();

function buildLegacySchema(): Schema {
  // Start from the strict schema so it inherits the no-clobber + className/id
  // allowances, then widen for SVG + scoped style.
  const schema = cloneSchema(markdownSchema);

  // `<style>` is ALLOWED on the legacy path, so it must NOT be in `strip`
  // (strip would delete the CSS text of even an allowed tag). Reset to just
  // <script>, which stays disallowed + content-stripped even here.
  schema.strip = ["script"];

  schema.tagNames = uniq([...(schema.tagNames ?? []), ...SVG_TAGS, "style", "main"]);

  const attrs = (schema.attributes ??= {});

  // Allow inline `style` on the tags the legacy bodies actually style.
  // hast-util-sanitize@5 does NOT re-clean CSS — the value passes through
  // verbatim — so this is gated to the TRUSTED legacy path only. CSS cannot
  // run JavaScript in modern browsers, the legacy CSS is analyzer-generated
  // and git-committed, and the strict markdownSchema (untrusted bodies)
  // allows no inline style at all.
  const STYLEABLE = [
    "div",
    "p",
    "span",
    "section",
    "figure",
    "figcaption",
    "table",
    "td",
    "th",
    "text",
    "tspan",
    "svg",
    "g",
    "rect",
    "circle",
    "line",
    "path",
    "polygon",
    "polyline",
    "main",
  ];
  for (const tag of STYLEABLE) {
    attrs[tag] = uniqAttr([...(attrs[tag] ?? []), "style"]);
  }

  // SVG presentational attributes (hast camelCase spellings) used by the
  // analyzer's chart output.
  const SVG_ATTRS = [
    "className",
    "id",
    "viewBox",
    "xmlns",
    "xmlnsXlink",
    "width",
    "height",
    "fill",
    "fillOpacity",
    "fillRule",
    "stroke",
    "strokeWidth",
    "strokeLinecap",
    "strokeLinejoin",
    "strokeDasharray",
    "strokeOpacity",
    "opacity",
    "transform",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "dx",
    "dy",
    "d",
    "points",
    "offset",
    "stopColor",
    "stopOpacity",
    "gradientUnits",
    "gradientTransform",
    "spreadMethod",
    "fontSize",
    "fontFamily",
    "fontWeight",
    "fontStyle",
    "textAnchor",
    "dominantBaseline",
    "alignmentBaseline",
    "letterSpacing",
    "preserveAspectRatio",
    "clipPath",
    "clipRule",
    "markerEnd",
    "markerStart",
    "vectorEffect",
    "role",
    "ariaLabel",
    "ariaLabelledBy",
    "ariaHidden",
    "style",
  ];
  for (const tag of SVG_TAGS) {
    attrs[tag] = uniqAttr([...(attrs[tag] ?? []), ...SVG_ATTRS]);
  }

  // `<style>` is permitted as a tag for the scoped chart CSS. Its text
  // content is left intact (hast-util-sanitize keeps the text children of an
  // allowed element) and is NOT re-cleaned — the sanitizer does not inspect
  // the CSS for `@import`/`url(...)`. The legacy CSS is scoped to `.cr-<N>`
  // selectors and is trusted by provenance (analyzer-generated, git-committed),
  // which is why that's safe here, not because the schema enforces it. We
  // still rely on the schema to strip <script>, on* handlers, and javascript:
  // URLs from the surrounding markup, which are the real JS-execution vectors
  // even in a trusted-but-public body.
  attrs.style = uniqAttr([...(attrs.style ?? []), "type"]);

  return schema;
}

// ── helpers ────────────────────────────────────────────────────────────────

function uniq<T>(items: T[]): T[] {
  return Array.from(new Set(items));
}

/**
 * Attribute lists in a Schema can be either a bare property name (`"id"` —
 * ANY value allowed) or a `[name, ...allowedValues]` tuple (value allowlist).
 *
 * Merge rule, deduping by property-name key:
 *   - A BARE name overrides an existing tuple for the same key. This is
 *     load-bearing: defaultSchema restricts `code`'s className to an empty
 *     allowlist (`["className", {}]` → drops every class value), which would
 *     strip the `language-*` / `hljs` classes rehype-highlight and the
 *     `math-inline` class remark-math emit. Listing a bare `"className"`
 *     widens it back to "any value", which is what the trusted post-sanitize
 *     plugins need.
 *   - First-seen otherwise (so an earlier explicit tuple is preserved if no
 *     later bare name overrides it).
 */
// Derive the entry type from the Schema so it stays in lockstep with
// hast-util-sanitize's `PropertyDefinition` (which permits RegExp / null /
// undefined inside the value-allowlist tuple, beyond plain primitives).
type AttrMap = NonNullable<Schema["attributes"]>;
type AttrEntry = AttrMap[string][number];

function uniqAttr(items: AttrEntry[]): AttrEntry[] {
  // Track the chosen entry per key + whether it's a bare (any-value) name.
  const chosen = new Map<string, { entry: AttrEntry; bare: boolean }>();
  const order: string[] = [];
  for (const item of items) {
    const key = Array.isArray(item) ? item[0] : item;
    const bare = !Array.isArray(item);
    const prev = chosen.get(key);
    if (!prev) {
      chosen.set(key, { entry: item, bare });
      order.push(key);
    } else if (bare && !prev.bare) {
      // A bare name widens a previously-restrictive tuple.
      chosen.set(key, { entry: item, bare });
    }
    // else: keep the first-seen (bare-over-bare or tuple-after-bare no-op).
  }
  return order.map((key) => chosen.get(key)!.entry);
}
