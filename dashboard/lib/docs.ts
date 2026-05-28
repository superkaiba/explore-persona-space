/**
 * Read docs/*.md from the repo root for the dashboard "Docs" route.
 *
 * Surfaces the living research docs (open_questions, papers,
 * conditional-behavior-related-work, SUMMARY, ...) so they render at
 * /docs and /docs/<slug>. Server-side only.
 *
 * No frontmatter is required: title falls back to the first H1, summary to
 * the first substantive paragraph, and last-updated to the file mtime. A doc
 * can opt out with `hidden: true` frontmatter, and order/label itself with
 * `nav_order` / `title` / `status` / `summary` / `last_updated`.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REPO_ROOT } from "./repo";

export const DOCS_DIR = path.join(REPO_ROOT, "docs");

// Files never surfaced (templates / scratch). `hidden: true` frontmatter also hides.
const DENYLIST = new Set(["SUMMARY.template.md"]);

// Slug = bare filename without `.md`. Must start alphanumeric (rejects ".",
// "..", "...") and contain no path separators, so reads stay inside DOCS_DIR
// and the listing stays symmetric with getDoc.
const SLUG_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export type DocFrontmatter = {
  title?: string;
  summary?: string;
  status?: string;
  last_updated?: string;
  nav_order?: number;
  hidden?: boolean;
  [k: string]: unknown;
};

export type DocListing = {
  slug: string;
  title: string;
  summary: string | null;
  status: string | null;
  lastUpdated: string | null;
  navOrder: number;
};

export type Doc = {
  slug: string;
  title: string;
  frontmatter: DocFrontmatter;
  body: string;
  lastUpdated: string | null;
};

function firstH1(body: string): string | null {
  const m = body.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1].trim() : null;
}

function deriveTitle(slug: string, fm: DocFrontmatter, body: string): string {
  if (typeof fm.title === "string" && fm.title.trim()) return fm.title.trim();
  return firstH1(body) ?? slug;
}

function deriveSummary(fm: DocFrontmatter, body: string): string | null {
  if (typeof fm.summary === "string" && fm.summary.trim()) return fm.summary.trim();
  // First substantive paragraph: skip headings, blockquotes, rules, tables, blanks.
  for (const raw of body.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (
      line.startsWith("#") ||
      line.startsWith(">") ||
      line.startsWith("---") ||
      line.startsWith("|") ||
      line.startsWith("```")
    )
      continue;
    const text = line.replace(/[*_`]/g, "");
    return text.length > 220 ? `${text.slice(0, 217)}…` : text;
  }
  return null;
}

function fileMtimeISO(full: string): string | null {
  try {
    return fs.statSync(full).mtime.toISOString().slice(0, 10);
  } catch {
    return null;
  }
}

function safeReadDir(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir);
}

export function listDocs(): DocListing[] {
  const rows: DocListing[] = [];
  for (const name of safeReadDir(DOCS_DIR)) {
    if (!name.endsWith(".md")) continue;
    if (name.endsWith(".template.md")) continue;
    if (DENYLIST.has(name)) continue;
    const full = path.join(DOCS_DIR, name);
    if (!fs.statSync(full).isFile()) continue;
    const raw = fs.readFileSync(full, "utf-8");
    const { data, content } = matter(raw);
    const fm = data as DocFrontmatter;
    if (fm.hidden === true) continue;
    const slug = name.replace(/\.md$/, "");
    if (!SLUG_RE.test(slug)) continue; // keep listing symmetric with getDoc
    rows.push({
      slug,
      title: deriveTitle(slug, fm, content),
      summary: deriveSummary(fm, content),
      status: typeof fm.status === "string" ? fm.status : null,
      lastUpdated: typeof fm.last_updated === "string" ? fm.last_updated : fileMtimeISO(full),
      navOrder: typeof fm.nav_order === "number" ? fm.nav_order : 100,
    });
  }
  rows.sort((a, b) => a.navOrder - b.navOrder || a.title.localeCompare(b.title));
  return rows;
}

export function getDoc(slug: string): Doc | null {
  if (!SLUG_RE.test(slug)) return null;
  const full = path.join(DOCS_DIR, `${slug}.md`);
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) return null;
  const raw = fs.readFileSync(full, "utf-8");
  const { data, content } = matter(raw);
  const fm = data as DocFrontmatter;
  if (fm.hidden === true) return null;
  return {
    slug,
    title: deriveTitle(slug, fm, content),
    frontmatter: fm,
    body: content,
    lastUpdated: typeof fm.last_updated === "string" ? fm.last_updated : fileMtimeISO(full),
  };
}
