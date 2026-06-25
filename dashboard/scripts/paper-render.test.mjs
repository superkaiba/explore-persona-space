/**
 * Render-path proof for `loadPaperFromDir` / `getPaper` (lib/paper.ts) — the
 * server-side loader behind BOTH the /tasks/[id] paper branch and the
 * /preview/paper-sample fixture. They call the SAME loader, so this single test
 * pins the behavior for both paths.
 *
 * No test runner is configured in this dashboard, so this is a standalone script
 * run under tsx (it imports the `.ts` module directly). Run with:
 *
 *   npx tsx scripts/paper-render.test.mjs
 *   # or: npm run test:paper-render
 *
 * The loader runs the committed paper.html through `sanitizePaperHtml`
 * (re-sanitize under buildPaperSchema) + `rewriteFigureSrcs`. It was previously
 * untested end-to-end, so a regression that dropped <img>, the eps-ref hooks, or
 * the figure-src rewrite — or that failed to strip an XSS vector / wire the
 * manifest PDF URL — would have shipped silently. This asserts the contract that
 * makes the three paper-render features work:
 *
 *   KEEP (the paper render hooks must survive sanitize):
 *     - <img>, with its relative src REWRITTEN to /tasks/<N>/figure/<file>
 *     - <a class="eps-ref" data-epsref="N"> cross-ref anchors (hover hook)
 *     - <figure>/<figcaption> wrappers + MathML
 *   STRIP (the real XSS vectors must NOT survive — no regression):
 *     - <script>, on* handlers, javascript: hrefs
 *   WIRE (the Download-PDF state):
 *     - manifest pdf_hf_url (https) → paper.pdfUrl
 *     - manifest pdf_hf_url null    → paper.pdfUrl null  (disabled "building…")
 *     - a non-https pdf_hf_url       → paper.pdfUrl null  (no smuggled href)
 *   CONFINE:
 *     - an absolute (https) img src is LEFT untouched (no rewrite)
 *     - a data: img src is STRIPPED by sanitize (protocols.src = http/https
 *       only) — defence in depth; the prompt's "do NOT broaden to data:" rule
 *
 * Hermetic: builds a throwaway repo tree in a tmpdir and chdirs into its
 * `dashboard/` BEFORE importing paper.ts, so lib/repo.ts's
 * `REPO_ROOT = resolve(cwd, "..")` captures the tmp root (the same shape as the
 * real runtime, where cwd is dashboard/). Exits non-zero on the first failure.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

let failures = 0;
function check(name, cond) {
  const ok = Boolean(cond);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) failures++;
}

// ── Build a hermetic fake repo root ────────────────────────────────────────
const ISSUE = 657;
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "eps-paper-render-"));
const repoRoot = path.join(tmpRoot, "repo");
const dashDir = path.join(repoRoot, "dashboard");
const paperDirRel = path.join("docs", "papers", `issue_${ISSUE}`);
const paperDir = path.join(repoRoot, paperDirRel);
fs.mkdirSync(dashDir, { recursive: true });
fs.mkdirSync(paperDir, { recursive: true });

// A committed paper.html body shaped like build_paper.py's pandoc output:
// figure wrappers with a relative <img src>, an eps-ref cross-ref anchor, a
// MathML span — PLUS the XSS payloads a tampered commit might smuggle, which
// the render-time re-sanitize must strip.
const paperHtml = `
<h1>Hermetic paper render test</h1>
<figure id="fig:one">
<img src="fig_one.png">
<figcaption>A figure with a relative src.</figcaption>
</figure>
<p>See <a href="/tasks/612" class="eps-ref" target="_blank" rel="noopener" data-epsref="612">#612</a>.</p>
<p>An absolute figure src stays untouched:
<img src="https://example.com/remote.png">
and a data: src is stripped by sanitize: <img src="data:image/png;base64,AAAA"></p>
<p><math display="inline"><semantics><mn>0.68</mn></semantics></math></p>
<p>XSS payloads that MUST be stripped:
<script>window.__pwned = 1</script>
<a href="javascript:alert(1)" class="eps-ref" data-epsref="999">evil</a>
<img src="x.png" onerror="alert(2)"></p>
`;
fs.writeFileSync(path.join(paperDir, "paper.html"), paperHtml);

function writeManifest(pdfHfUrl) {
  fs.writeFileSync(
    path.join(paperDir, "paper_manifest.json"),
    JSON.stringify({ schema: "paper_manifest/v1", issue: ISSUE, pdf_hf_url: pdfHfUrl }),
  );
}

const HF_PDF =
  "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/abc123/papers/issue_657/issue_657.pdf";
writeManifest(HF_PDF);

// chdir into dashboard/ so REPO_ROOT (resolve(cwd, "..")) === repoRoot, then
// import paper.ts fresh (REPO_ROOT is a load-time const).
process.chdir(dashDir);
const { loadPaperFromDir, getPaper } = await import("../lib/paper.ts");

const paper = loadPaperFromDir(paperDirRel, ISSUE);
check("loadPaperFromDir returns a paper (not null)", paper !== null);
const html = paper ? paper.html : "";

// ── KEEP: the paper render hooks survive sanitize ──────────────────────────
console.log("\n=== KEEP (render hooks survive) ===");
check(
  "rewrites relative <img src> to the figure route",
  html.includes(`src="/tasks/${ISSUE}/figure/fig_one.png"`),
);
check("<img> tag survives sanitize", /<img\b/i.test(html));
check("eps-ref class survives", html.includes('class="eps-ref"'));
check("data-epsref attr survives", html.includes('data-epsref="612"'));
check("<figure> wrapper survives", html.includes("<figure"));
check("<figcaption> survives", html.includes("<figcaption"));
check("MathML <math> survives", html.includes("<math"));

// ── CONFINE: absolute https src untouched; data: src stripped by sanitize ──
console.log("\n=== CONFINE (absolute untouched, data: stripped) ===");
check(
  "absolute https img src left untouched (not rewritten)",
  html.includes('src="https://example.com/remote.png"'),
);
// The src protocol allow-list is http/https only, so a data: src is removed by
// sanitize (the rewrite step never even sees it). This is the secure outcome —
// same-origin relative srcs are allowed, data:/javascript: are not.
check("data: img src is STRIPPED by sanitize (not preserved)", !html.includes("data:image/png"));

// ── STRIP: the real XSS vectors do NOT survive ─────────────────────────────
console.log("\n=== STRIP (XSS vectors removed) ===");
check("<script> stripped", !/<script/i.test(html));
check("script payload text gone", !html.includes("__pwned"));
check("on* handler (onerror) stripped", !/onerror/i.test(html));
check("javascript: href stripped", !/href="javascript:/i.test(html));

// ── WIRE: the Download-PDF pdfUrl prop ─────────────────────────────────────
console.log("\n=== WIRE (manifest pdf_hf_url -> pdfUrl) ===");
check("https pdf_hf_url wired to pdfUrl", paper && paper.pdfUrl === HF_PDF);

writeManifest(null);
const paperNullPdf = loadPaperFromDir(paperDirRel, ISSUE);
check("null pdf_hf_url -> pdfUrl null (disabled 'building')", paperNullPdf && paperNullPdf.pdfUrl === null);

writeManifest("http://insecure.example/issue.pdf");
const paperHttpPdf = loadPaperFromDir(paperDirRel, ISSUE);
check(
  "non-https pdf_hf_url -> pdfUrl is the raw string (PaperView gates the protocol)",
  paperHttpPdf && paperHttpPdf.pdfUrl === "http://insecure.example/issue.pdf",
);

// ── getPaper resolves the canonical docs/papers/issue_<N>/ dir ─────────────
console.log("\n=== getPaper canonical dir ===");
writeManifest(HF_PDF);
const viaGetPaper = getPaper(ISSUE);
check("getPaper(657) loads docs/papers/issue_657/", viaGetPaper !== null);
check(
  "getPaper figure src rewritten too",
  viaGetPaper && viaGetPaper.html.includes(`src="/tasks/${ISSUE}/figure/fig_one.png"`),
);

// A missing paper dir returns null (the caller falls back to the markdown stub).
check("getPaper(999999) is null when no paper dir", getPaper(999999) === null);

// ── cleanup + verdict ──────────────────────────────────────────────────────
process.chdir(os.tmpdir()); // leave dashDir before removing it
fs.rmSync(tmpRoot, { recursive: true, force: true });

console.log("");
if (failures > 0) {
  console.error(`${failures} assertion(s) FAILED`);
  process.exit(1);
}
console.log("All paper-render assertions passed.");
