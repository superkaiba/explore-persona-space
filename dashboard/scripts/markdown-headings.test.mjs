/**
 * Slug-unification proof for the shared <MarkdownDoc> TOC + heading ids.
 *
 * No test runner is configured in this dashboard, so this is a standalone
 * script. It imports the `.ts` headings module directly, so it must run under
 * tsx (plain `node` cannot resolve the TypeScript import). Run with:
 *
 *   npx tsx scripts/markdown-headings.test.mjs
 *   # or: npm run test:headings
 *
 * The TOC (MarkdownDocToc) and the rendered heading ids (MarkdownDoc's
 * assignHeadingIds effect) must come from ONE slugger so `#<id>` anchors
 * resolve. Both call `extractMarkdownHeadings(body, docId)` / `headingId(...)`
 * from lib/markdown-headings.ts — the SAME githubLikeSlug + dedupeSlug +
 * per-doc prefix. rehype-slug (github-slugger) was removed from the pipeline
 * because it diverges on headings with stripped punctuation flanked by spaces
 * (e.g. `p < 0.05`). This pins the deterministic output and the cross-doc
 * namespacing that fix the TOC-click no-op (#2) and cross-doc anchor
 * collision (#3).
 *
 * Exits non-zero on the first failed assertion.
 */
import {
  githubLikeSlug,
  headingId,
  docIdPrefix,
  extractMarkdownHeadings,
} from "../lib/markdown-headings.ts";

let failures = 0;
function eq(name, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  got=${JSON.stringify(got)}`);
  if (!ok) failures++;
}

// One canonical slugger on the task's tricky examples.
eq("slug p < 0.05", githubLikeSlug("p < 0.05"), "p-005");
eq("slug A & B: results (n=10)", githubLikeSlug("A & B: results (n=10)"), "a-b-results-n10");
eq("slug TL;DR", githubLikeSlug("TL;DR"), "tldr");

// Per-doc prefix namespaces ids so multiple docs on one page don't collide.
eq("prefix number", docIdPrefix(390), "390--");
eq("prefix undefined (single-doc unaffected when omitted)", docIdPrefix(undefined), "");

// Dedupe within a doc uses the project's own -1/-2 convention (NOT github's).
{
  const c = new Map();
  eq("dup #1", headingId("TL;DR", "d", c), "d--tldr");
  eq("dup #2", headingId("TL;DR", "d", c), "d--tldr-1");
}

// extractMarkdownHeadings counts H4-H6 before an H1-H3 of the same slug, so
// its dedupe matches assignHeadingIds (which also walks h1-h6 in the DOM).
{
  const md = "#### TL;DR\n\ntext\n\n## TL;DR\n\nmore\n\n## p < 0.05\n";
  const hs = extractMarkdownHeadings(md, "d");
  eq("H4 id", hs[0].id, "d--tldr");
  eq("H2 id dedup-past-H4", hs[1].id, "d--tldr-1");
  eq("p<0.05 id", hs[2].id, "d--p-005");
}

// Cross-doc isolation: same heading text, two docs -> distinct prefixed ids.
{
  const a = extractMarkdownHeadings("## TL;DR\n", "overview-open_questions")[0].id;
  const b = extractMarkdownHeadings("## TL;DR\n", "overview-SUMMARY")[0].id;
  eq("cross-doc distinct", a !== b, true);
  eq("doc A id", a, "overview-open-questions--tldr");
  eq("doc B id", b, "overview-summary--tldr");
}

console.log("");
if (failures > 0) {
  console.error(`${failures} assertion(s) FAILED`);
  process.exit(1);
}
console.log("All slug-unification assertions passed.");
