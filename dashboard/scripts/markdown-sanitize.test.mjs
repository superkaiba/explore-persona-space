/**
 * Sanitize proof for the shared <MarkdownDoc> render pipeline.
 *
 * No test runner is configured in this dashboard, so this is a standalone
 * node script. Run with:
 *
 *   node scripts/markdown-sanitize.test.mjs
 *
 * It exercises BOTH schemas exported from lib/markdown-sanitize.ts against
 * the exact render pipeline MarkdownDoc uses (remarkGfm + remarkMath ->
 * rehypeRaw -> rehypeSanitize(schema) -> rehypeKatex -> rehypeHighlight ->
 * rehypeSlug -> stringify) and asserts:
 *
 *   STRICT (markdownSchema), public/untrusted body:
 *     - <script>alert(1)</script>           => stripped
 *     - <img src=x onerror=alert(1)>         => onerror attribute stripped
 *     - <a href="javascript:alert(1)">       => javascript: href stripped
 *     - inline style=                        => stripped
 *     - <details>/<summary>                  => survive
 *     - a KaTeX expression ($E=mc^2$)        => renders to katex markup
 *     - a fenced code block                  => survives with hljs classes
 *     - <mark data-comment-id> anchor shape  => survives
 *
 *   LEGACY (legacySchema), trusted Sagan-card body:
 *     - inline <svg> chart + scoped <style>  => survive
 *     - inline style= on chart elements      => survives
 *     - <script> / onerror inside legacy     => STILL stripped
 *
 * Exits non-zero on the first failed assertion.
 */
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";

import { markdownSchema, legacySchema } from "../lib/markdown-sanitize.ts";

function render(markdown, schema) {
  return unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSanitize, schema)
    .use(rehypeKatex)
    .use(rehypeHighlight)
    .use(rehypeSlug)
    .use(rehypeStringify)
    .processSync(markdown)
    .toString();
}

let failures = 0;
function check(name, cond) {
  const ok = Boolean(cond);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) failures++;
}

// ── STRICT schema: untrusted public markdown ────────────────────────────────
const strictBody = [
  "# Heading One",
  "",
  "<script>alert(1)</script>",
  "",
  '<img src="x" onerror="alert(1)">',
  "",
  '<a href="javascript:alert(1)">click</a>',
  "",
  '<p style="color:red">styled</p>',
  "",
  "<details><summary>Show more</summary>hidden content</details>",
  "",
  "Inline math: $E = mc^2$",
  "",
  "```python",
  "def f(x):",
  "    return x + 1",
  "```",
  "",
  '<mark data-comment-id="c001">anchored</mark>',
].join("\n");

const strictOut = render(strictBody, markdownSchema);
console.log("\n=== STRICT (markdownSchema) ===");
check("strips <script> tag", !/<script/i.test(strictOut));
check("strips onerror handler", !/onerror/i.test(strictOut));
check("strips javascript: href", !/javascript:/i.test(strictOut));
// The injected `<p style="color:red">` style must be dropped. (KaTeX, which
// runs AFTER sanitize, legitimately emits inline style for glyph metrics, so
// we assert the SPECIFIC injected declaration is gone, not that no style=
// exists anywhere in the trusted post-sanitize output.)
check("strips markdown-authored inline style=", !/color:\s*red/i.test(strictOut));
check("keeps <details>", /<details/i.test(strictOut));
check("keeps <summary>", /<summary/i.test(strictOut));
check("renders KaTeX (katex class present)", /class="katex/i.test(strictOut));
check("keeps fenced code block (<pre><code>)", /<pre><code/i.test(strictOut));
check("keeps highlight class on code", /class="[^"]*hljs|language-python/i.test(strictOut));
check("keeps <mark data-comment-id>", /<mark[^>]*data-comment-id="c001"/i.test(strictOut));
check("keeps heading id (rehype-slug)", /id="heading-one"/i.test(strictOut));

// ── LEGACY schema: trusted Sagan-card HTML ──────────────────────────────────
const legacyBody = [
  '<section class="cr-999">',
  "<style>",
  "  .cr-999 { --clay: #D97757; }",
  "  .cr-999 .ax { stroke: #87867F; stroke-width: 1.2; }",
  "</style>",
  '<svg viewBox="0 0 100 50" xmlns="http://www.w3.org/2000/svg">',
  '  <line class="ax" x1="0" y1="50" x2="100" y2="50" stroke="#000" stroke-width="2"/>',
  '  <rect x="10" y="10" width="20" height="30" fill="#D97757" style="fill:#176c3a"/>',
  '  <text x="50" y="25" font-size="11" text-anchor="middle" fill-opacity="0.8">label</text>',
  "</svg>",
  '<p style="max-width:760px;font-family:system-ui">caption</p>',
  "<script>alert(1)</script>",
  '<g onclick="alert(2)"></g>',
  "</section>",
].join("\n");

const legacyOut = render(legacyBody, legacySchema);
console.log("\n=== LEGACY (legacySchema) ===");
check("keeps <svg>", /<svg/i.test(legacyOut));
check("keeps <line> SVG element", /<line/i.test(legacyOut));
check("keeps <rect> SVG element", /<rect/i.test(legacyOut));
check("keeps <text> SVG element", /<text/i.test(legacyOut));
check("keeps scoped <style> block", /<style>[\s\S]*\.cr-999/i.test(legacyOut));
check("keeps stroke-width attr", /stroke-width="2"/i.test(legacyOut));
check("keeps inline style= on chart elements", /style="fill:#176c3a"/i.test(legacyOut));
check("keeps <p> inline style", /max-width:760px/i.test(legacyOut));
check("STILL strips <script> in legacy", !/<script/i.test(legacyOut));
check("STILL strips onclick handler in legacy", !/onclick/i.test(legacyOut));

console.log("");
if (failures > 0) {
  console.error(`${failures} assertion(s) FAILED`);
  process.exit(1);
}
console.log("All sanitize assertions passed.");
