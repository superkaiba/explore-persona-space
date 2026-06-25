/**
 * Paper-task render support (Phase C2).
 *
 * For a `paper: true` task the canonical clean-result is the LaTeX paper under
 * `docs/papers/issue_<N>/`. `build_paper.py` produces three committed artifacts
 * the dashboard renders:
 *
 *   - paper.html           — pandoc-produced, dashboard-sanitized HTML body
 *                            (no <html>/<head>; just the article body, with
 *                            <figure>/<figcaption>, <a class="eps-ref"
 *                            data-epsref>, MathML, and relative <img src>).
 *   - paper_manifest.json  — artifact paths + the pinned HF PDF URL
 *                            (`pdf_hf_url`, null for a local/unbuilt paper).
 *
 * `getPaper` reads both, RE-SANITIZES the committed HTML under the paperSchema
 * (defence in depth — see lib/paper-schema.ts), and rewrites the paper's
 * relative figure `<img src="X.png">` to the dashboard's figure-serving route
 * `/tasks/<N>/figure/X.png` so the figures resolve when the HTML is mounted on
 * the task page. Everything is read from disk under the repo root, path-confined.
 *
 * Server-only (filesystem reads + a Node sanitize pass; no `import "server-only"`
 * because the dashboard doesn't depend on that package — the `node:fs` import
 * already keeps it off any client bundle). The returned `html` is already
 * sanitized, so the client renderer mounts it directly.
 */
import fs from "node:fs";
import path from "node:path";
import { fromHtml } from "hast-util-from-html";
import { sanitize } from "hast-util-sanitize";
import { toHtml } from "hast-util-to-html";
import { REPO_ROOT } from "./repo";
import { paperSchema } from "./paper-schema";

export type PaperManifest = {
  schema?: string;
  issue?: number | string;
  jobname?: string;
  built_at?: string;
  source_date_epoch?: string;
  /** Pinned HF PDF URL; null when the paper was built --no-upload / not built. */
  pdf_hf_url?: string | null;
  artifacts?: Record<string, { path: string; sha256: string; bytes: number }>;
  [k: string]: unknown;
};

export type Paper = {
  issue: number;
  /** Sanitized HTML body, with relative figure srcs rewritten to the route. */
  html: string;
  /** Pinned HF PDF URL, or null (→ disabled "building" Download-PDF state). */
  pdfUrl: string | null;
  /** The full manifest (provenance: build time, source epoch, sha256 set). */
  manifest: PaperManifest | null;
};

/** Resolve a repo-relative path and assert it stays under the repo root. */
function assertUnderRepo(relOrAbs: string): string | null {
  const abs = path.resolve(REPO_ROOT, relOrAbs);
  const root = path.resolve(REPO_ROOT);
  if (abs !== root && !abs.startsWith(root + path.sep)) return null;
  return abs;
}

/**
 * Rewrite a paper.html body so its relative figure `<img src="X.png">` point at
 * the dashboard's figure-serving route for this task. Absolute srcs
 * (http(s):// or a leading "/") are left untouched. Confined to the <img>
 * src= attribute; nothing else in the (already-sanitized) HTML is touched.
 */
function rewriteFigureSrcs(html: string, issue: number): string {
  return html.replace(
    /(<img\b[^>]*?\bsrc=)(["'])(.*?)\2/gi,
    (whole, pre: string, q: string, src: string) => {
      const s = src.trim();
      if (/^(https?:)?\/\//i.test(s) || s.startsWith("/") || s.startsWith("data:")) {
        return whole; // already absolute / data URI — leave it
      }
      // Bare filename relative to figures/issue_<N>/ → the figure route.
      const name = s.replace(/^\.\//, "");
      return `${pre}${q}/tasks/${issue}/figure/${encodeURIComponent(name)}${q}`;
    },
  );
}

/**
 * Re-sanitize the committed paper HTML under the paperSchema (defence in depth).
 * A non-empty stripped-tag census is logged (a tampered commit would surface
 * here); we do not throw — the build-time gate is the authoritative one and the
 * render must not crash a page on a stray attribute.
 */
function sanitizePaperHtml(raw: string, issue: number): string {
  const tree = fromHtml(raw, { fragment: true });
  const clean = toHtml(sanitize(tree, paperSchema));
  if (process.env.NODE_ENV !== "production") {
    const tagCount = (s: string): Record<string, number> => {
      const m = s.match(/<([a-zA-Z][a-zA-Z0-9]*)/g) ?? [];
      const c: Record<string, number> = {};
      for (const t of m) {
        const n = t.slice(1).toLowerCase();
        c[n] = (c[n] ?? 0) + 1;
      }
      return c;
    };
    const before = tagCount(raw);
    const after = tagCount(clean);
    const lost: Record<string, number> = {};
    for (const t of Object.keys(before)) {
      const d = (before[t] ?? 0) - (after[t] ?? 0);
      if (d > 0) lost[t] = d;
    }
    if (Object.keys(lost).length) {
      console.warn(`[paper] issue ${issue}: render-time sanitize stripped`, lost);
    }
  }
  return clean;
}

/** Read + parse the manifest for a paper dir, or null. */
function readManifest(paperDirAbs: string): PaperManifest | null {
  const p = path.join(paperDirAbs, "paper_manifest.json");
  try {
    return JSON.parse(fs.readFileSync(p, "utf8")) as PaperManifest;
  } catch {
    return null;
  }
}

/**
 * Load a paper for an arbitrary paper directory (relative to the repo root) +
 * its logical issue number (drives the figure-route rewrite + label). Used by
 * both `getPaper` (the real `docs/papers/issue_<N>/` dir) and the dev/smoke
 * fixture renderer (`docs/papers/_sample/`).
 *
 * Returns null when no committed `paper.html` exists for the dir.
 */
export function loadPaperFromDir(paperDirRel: string, issue: number): Paper | null {
  const dirAbs = assertUnderRepo(paperDirRel);
  if (!dirAbs || !fs.existsSync(dirAbs)) return null;
  const htmlPath = path.join(dirAbs, "paper.html");
  let raw: string;
  try {
    raw = fs.readFileSync(htmlPath, "utf8");
  } catch {
    return null; // paper.html not built yet
  }
  const manifest = readManifest(dirAbs);
  const pdfUrl =
    manifest && typeof manifest.pdf_hf_url === "string" && manifest.pdf_hf_url.trim()
      ? manifest.pdf_hf_url.trim()
      : null;
  const sanitized = sanitizePaperHtml(raw, issue);
  const html = rewriteFigureSrcs(sanitized, issue);
  return { issue, html, pdfUrl, manifest };
}

/**
 * Load the committed paper for a real paper-task: `docs/papers/issue_<N>/`.
 * Returns null when the dir or its paper.html is missing (the caller then falls
 * back to the markdown body / a "paper not built yet" notice).
 */
export function getPaper(id: number): Paper | null {
  return loadPaperFromDir(path.join("docs", "papers", `issue_${id}`), id);
}

// ── figure serving ────────────────────────────────────────────────────────

const FIGURE_EXT_CONTENT_TYPE: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
};

export type FigureFile = { bytes: Buffer; contentType: string };

/**
 * Resolve + read a figure for the figure-serving route: figures/issue_<N>/<name>.
 * `name` is a single path segment (the paper's relative <img> filename). Confined
 * under figures/issue_<N>/ — a `..`/absolute/multi-segment name is refused — and
 * restricted to an image extension allow-list. Returns null on any miss.
 */
export function readTaskFigure(issue: number, name: string): FigureFile | null {
  // Single safe segment only (no traversal, no nested path).
  if (!name || name.includes("/") || name.includes("\\") || name.includes("..")) {
    return null;
  }
  const ext = path.extname(name).toLowerCase();
  const contentType = FIGURE_EXT_CONTENT_TYPE[ext];
  if (!contentType) return null;
  const rel = path.join("figures", `issue_${issue}`, name);
  const abs = assertUnderRepo(rel);
  if (!abs) return null;
  // Re-confine under figures/issue_<N>/ specifically (assertUnderRepo only pins
  // the repo root) and resolve symlinks so a symlinked figure can't escape.
  const figRoot = path.resolve(REPO_ROOT, "figures", `issue_${issue}`);
  let real: string;
  try {
    real = fs.realpathSync(abs);
  } catch {
    return null;
  }
  if (real !== figRoot && !real.startsWith(figRoot + path.sep)) return null;
  try {
    if (!fs.statSync(real).isFile()) return null;
    return { bytes: fs.readFileSync(real), contentType };
  } catch {
    return null;
  }
}
