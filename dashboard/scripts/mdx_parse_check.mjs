#!/usr/bin/env node
/**
 * mdx_parse_check.mjs — authoritative MDX-parse mirror for clean-result bodies.
 *
 * Runs the SAME `mdast-util-from-markdown` parse the EPS dashboard's
 * CardBodyEditor (MDXEditor 4.0.1) runs when it loads a task body for
 * editing. If this parse throws, the dashboard shows the amber "Could not
 * parse" banner and a fallback raw-editor link instead of the rendered body —
 * the uneditable-body symptom this check exists to prevent.
 *
 * Why a node helper at all (and why under `dashboard/`): the parse extensions
 * (`mdast-util-from-markdown`, `micromark-extension-mdx-jsx`,
 * `micromark-extension-mdx-md`, `micromark-extension-gfm-table`, the highlight
 * + strikethrough pair, and the editor's internal HTML-comment extension) live
 * ONLY in `dashboard/node_modules`. Node resolves ESM bare specifiers relative
 * to the importing file, so this file MUST sit under `dashboard/` for the
 * imports to resolve. A byte-identical copy under `/tmp` throws
 * ERR_MODULE_NOT_FOUND. The `.mjs` extension is required because
 * `dashboard/package.json` has no `"type": "module"` field and the deps are
 * ESM-only.
 *
 * EXACT EXTENSION SET (mirrors @mdxeditor/editor/dist/plugins/core/index.js
 * lines 539-552 + plugins/table/index.js lines 46-47; `suppressHtmlProcessing`
 * is never set by CardBodyEditorClient, so the mdxJsx / mdxMd / comment block
 * is active):
 *
 *   syntax (micromark): gfmStrikethrough(), highlightMark(), mdxJsx(),
 *                       mdxMd(), comment, gfmTable()
 *   tree   (mdast):     gfmStrikethroughFromMarkdown(), highlightMarkFromMarkdown,
 *                       mdxJsxFromMarkdown(), commentFromMarkdown(),
 *                       gfmTableFromMarkdown()
 *
 * The load-bearing pair is mdxJsx + mdxMd: those are what turn a bare `<`
 * before a non-name character (`<https://...`, `<0.05`, `<|im_start|>` exposed
 * by a table cell splitting on the unescaped pipe before code-span recognition)
 * into a JSX-tag-start parse error.
 *
 * The `comment` / `commentFromMarkdown` extensions live in the NON-exported
 * internal module `@mdxeditor/editor/dist/mdastUtilHtmlComment.js` (not reachable
 * via the package `exports` map, only via a direct relative file path). They are
 * INCLUDED here, not omitted: real clean-result bodies carry HTML comment
 * markers (`<!-- legacy-sagan-card -->`, `<!-- workflow-fix-candidate v1 -->`,
 * `<!-- epm:... -->`). Without the comment extension those `<!--` markers throw
 * "Unexpected character `!` (U+0021) before name" — verified empirically — so
 * omitting them would make the mirror reject valid bodies. With them included,
 * the markers parse and the three failure classes (table-cell `<|`, autolink
 * slash, `<`-before-digit) still throw, matching the editor exactly.
 *
 * Input: the markdown BODY to parse.
 *   - If a path is given in argv[2], read that file and strip YAML frontmatter
 *     EXACTLY as the dashboard does (gray-matter, lib/tasks.ts getTask) before
 *     parsing.
 *   - Otherwise read the body from stdin. The stdin body is assumed to be the
 *     ALREADY-STRIPPED body (the python check-14 caller passes the same body it
 *     already split off the frontmatter via verify_task_body.split_frontmatter,
 *     which equals gray-matter's `content` for the canonical `---\nyaml\n---\nbody`
 *     shape — gray-matter trims the closing fence plus one trailing newline, and
 *     the two splitters are identical for that shape, verified empirically). The
 *     stdin body is NOT stripped again.
 *
 * Output / exit codes (the python caller distinguishes these three):
 *   - parse OK    → prints {"ok":true} to stdout, exits 0.
 *   - parse FAIL  → prints {"ok":false,"message":...,"line":...,"column":...}
 *                   to stdout, exits 2 (the body is invalid).
 *   - harness err → prints nothing to stdout, writes a note to stderr, exits 3
 *                   (deps missing / helper itself broke — "parser unavailable",
 *                   NOT "body invalid"). The python caller MUST treat exit 3 as
 *                   "real-MDX parse skipped", never as a silent pass.
 */

import process from "node:process";

/** Read all of stdin as a UTF-8 string. */
async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * Strip YAML frontmatter exactly as the dashboard does before a body reaches
 * the editor (gray-matter, replicated in lib/tasks.ts getTask). gray-matter
 * trims the leading `---\n...\n---` block plus one trailing newline, so the
 * returned content starts at the first body line. We use gray-matter itself so
 * the strip is byte-identical to the dashboard.
 */
async function stripFrontmatter(raw) {
  const { default: matter } = await import("gray-matter");
  return matter(raw).content;
}

async function main() {
  // Phase 1: load the parse harness. Any failure here is a "parser unavailable"
  // condition (exit 3), NOT a body-invalid condition.
  let fromMarkdown;
  let syntaxExtensions;
  let mdastExtensions;
  try {
    ({ fromMarkdown } = await import("mdast-util-from-markdown"));
    const { gfmStrikethrough } = await import("micromark-extension-gfm-strikethrough");
    const { gfmStrikethroughFromMarkdown } = await import("mdast-util-gfm-strikethrough");
    const { highlightMark } = await import("micromark-extension-highlight-mark");
    const { highlightMarkFromMarkdown } = await import("mdast-util-highlight-mark");
    const { mdxJsx } = await import("micromark-extension-mdx-jsx");
    const { mdxJsxFromMarkdown } = await import("mdast-util-mdx-jsx");
    const { mdxMd } = await import("micromark-extension-mdx-md");
    const { gfmTable } = await import("micromark-extension-gfm-table");
    const { gfmTableFromMarkdown } = await import("mdast-util-gfm-table");
    // comment / commentFromMarkdown are a NON-exported internal module — reach
    // them via a direct relative file path, not a bare specifier (the package
    // `exports` map does not expose this subpath).
    const { comment, commentFromMarkdown } = await import(
      "../node_modules/@mdxeditor/editor/dist/mdastUtilHtmlComment.js"
    );
    syntaxExtensions = [
      gfmStrikethrough(),
      highlightMark(),
      mdxJsx(),
      mdxMd(),
      comment,
      gfmTable(),
    ];
    mdastExtensions = [
      gfmStrikethroughFromMarkdown(),
      highlightMarkFromMarkdown,
      mdxJsxFromMarkdown(),
      commentFromMarkdown(),
      gfmTableFromMarkdown(),
    ];
  } catch (e) {
    process.stderr.write(
      `mdx_parse_check: parser harness unavailable — ${e && e.message ? e.message : e}\n`,
    );
    process.exit(3);
  }

  // Phase 2: read the body. A file-read or frontmatter-strip failure is also a
  // harness condition (exit 3) — we could not even obtain a body to judge.
  let body;
  try {
    const argPath = process.argv[2];
    if (argPath) {
      const fs = await import("node:fs");
      const raw = fs.readFileSync(argPath, "utf8");
      body = await stripFrontmatter(raw);
    } else {
      // stdin body is the already-stripped body; do NOT strip again.
      body = await readStdin();
    }
  } catch (e) {
    process.stderr.write(
      `mdx_parse_check: could not read body — ${e && e.message ? e.message : e}\n`,
    );
    process.exit(3);
  }

  // Phase 3: the actual parse. A throw here is a body-invalid condition (exit 2)
  // — this is the same throw path that surfaces as the dashboard's amber
  // "Could not parse" banner via MarkdownParseError → onError.
  try {
    fromMarkdown(body, { extensions: syntaxExtensions, mdastExtensions });
  } catch (e) {
    const message = e && e.message ? String(e.message).split("\n")[0] : String(e);
    // micromark VFileMessage carries `line` / `column` (and a `place` object on
    // newer versions). Surface whatever is available.
    let line = null;
    let column = null;
    if (e && typeof e.line === "number") line = e.line;
    if (e && typeof e.column === "number") column = e.column;
    if (line === null && e && e.place && e.place.start) {
      if (typeof e.place.start.line === "number") line = e.place.start.line;
      if (typeof e.place.start.column === "number") column = e.place.start.column;
    }
    process.stdout.write(JSON.stringify({ ok: false, message, line, column }) + "\n");
    process.exit(2);
  }

  process.stdout.write(JSON.stringify({ ok: true }) + "\n");
  process.exit(0);
}

main().catch((e) => {
  process.stderr.write(
    `mdx_parse_check: unexpected harness failure — ${e && e.message ? e.message : e}\n`,
  );
  process.exit(3);
});
