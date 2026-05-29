/**
 * Read docs/*.md from the repo root for the dashboard "Docs" route.
 *
 * Surfaces the living research docs (open_questions, papers,
 * conditional-behavior-related-work, SUMMARY, ...) so they render at
 * /docs and /docs/<slug>. Server-side only.
 *
 * Beyond the top-level `docs/*.md` files, the resolver also surfaces three
 * sibling stores WITHOUT moving any file on disk, using a virtual-slug
 * scheme (a `<prefix>__<filestem>` slug that maps back to a real path):
 *
 *   - docs/mentor_updates/*.md  -> category "Meetings & mentor updates"
 *                                  (virtual prefix `mu`)
 *   - logs/daily/*.md           -> category "Activity"  (prefix `daily`)
 *   - logs/weekly/*.md          -> category "Activity"  (prefix `weekly`)
 *   - docs/ideas/*.md           -> category "Ideas"     (prefix `idea`)
 *
 * SLUG_RE rejects path separators, so the virtual slug keeps every read
 * inside its source directory. Top-level docs keep their bare-filename slug.
 *
 * Categories: top-level docs read a `category:` frontmatter field and fall
 * back to "Reference" when absent; virtual docs are categorized by their
 * source store. No frontmatter is required: title falls back to the first
 * H1, summary to the first substantive paragraph, and last-updated to the
 * file mtime. A doc can opt out with `hidden: true` frontmatter, and
 * order/label itself with `nav_order` / `title` / `status` / `summary` /
 * `last_updated`.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REPO_ROOT } from "./repo";

export const DOCS_DIR = path.join(REPO_ROOT, "docs");

// Files never surfaced (templates / scratch). `hidden: true` frontmatter also hides.
const DENYLIST = new Set(["SUMMARY.template.md"]);

// Slug = bare filename without `.md` (top-level docs) OR a `<prefix>__<stem>`
// virtual slug (sibling stores). Must start alphanumeric (rejects ".", "..",
// "...") and contain no path separators, so reads stay inside their source
// directory and the listing stays symmetric with getDoc. `_` is allowed, so
// the `__` virtual-slug separator passes.
const SLUG_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export const DOC_CATEGORIES = [
  "Meetings & mentor updates",
  "Activity",
  "Ideas",
  "Reference",
] as const;

export type DocCategory = (typeof DOC_CATEGORIES)[number];

const DEFAULT_CATEGORY: DocCategory = "Reference";

// Display order for the listing page (matches DOC_CATEGORIES order).
const CATEGORY_ORDER: Record<DocCategory, number> = {
  "Meetings & mentor updates": 0,
  Activity: 1,
  Ideas: 2,
  Reference: 3,
};

/**
 * Virtual-store registry. Each entry maps a slug prefix to a source
 * directory + a category. The slug shape is `<prefix>__<filestem>`; the
 * resolver splits on the first `__` and rebuilds the real file path.
 *
 * `dir` is absolute. Stores that don't exist yet (e.g. an empty logs/weekly
 * or docs/ideas) are tolerated — safeReadDir returns [] for a missing dir.
 */
type VirtualStore = {
  prefix: string;
  dir: string;
  category: DocCategory;
};

const VIRTUAL_SEP = "__";

const VIRTUAL_STORES: VirtualStore[] = [
  {
    prefix: "mu",
    dir: path.join(DOCS_DIR, "mentor_updates"),
    category: "Meetings & mentor updates",
  },
  {
    prefix: "daily",
    dir: path.join(REPO_ROOT, "logs", "daily"),
    category: "Activity",
  },
  {
    prefix: "weekly",
    dir: path.join(REPO_ROOT, "logs", "weekly"),
    category: "Activity",
  },
  {
    prefix: "idea",
    dir: path.join(DOCS_DIR, "ideas"),
    category: "Ideas",
  },
];

export type DocFrontmatter = {
  title?: string;
  summary?: string;
  status?: string;
  last_updated?: string;
  nav_order?: number;
  hidden?: boolean;
  category?: string;
  [k: string]: unknown;
};

export type DocListing = {
  slug: string;
  title: string;
  summary: string | null;
  status: string | null;
  lastUpdated: string | null;
  navOrder: number;
  category: DocCategory;
};

export type Doc = {
  slug: string;
  title: string;
  frontmatter: DocFrontmatter;
  body: string;
  lastUpdated: string | null;
  category: DocCategory;
};

function firstH1(body: string): string | null {
  const m = body.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1].trim() : null;
}

function deriveTitle(fallback: string, fm: DocFrontmatter, body: string): string {
  if (typeof fm.title === "string" && fm.title.trim()) return fm.title.trim();
  return firstH1(body) ?? fallback;
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

/** Normalize a frontmatter `category` value to a known DocCategory. */
function normalizeCategory(value: unknown): DocCategory {
  if (typeof value !== "string") return DEFAULT_CATEGORY;
  const want = value.trim().toLowerCase();
  for (const cat of DOC_CATEGORIES) {
    if (cat.toLowerCase() === want) return cat;
  }
  return DEFAULT_CATEGORY;
}

/* -------------------------------------------------------------------------- *
 * Virtual-slug resolution.
 *
 * A virtual slug is `<prefix>__<filestem>` (e.g. `mu__2026-05-28`,
 * `daily__2026-05-28`). `resolveSlug` maps any slug — top-level or virtual —
 * to the absolute file path + its category, or null when the slug is
 * malformed / out-of-store.
 * -------------------------------------------------------------------------- */

type Resolved = {
  full: string;
  category: DocCategory;
  /** Human-friendly fallback title when no frontmatter/H1 is present. */
  fallbackTitle: string;
  /** True for a virtual-store slug (category is authoritative). */
  isVirtual: boolean;
};

function virtualStoreForPrefix(prefix: string): VirtualStore | null {
  return VIRTUAL_STORES.find((s) => s.prefix === prefix) ?? null;
}

function makeVirtualSlug(prefix: string, stem: string): string {
  return `${prefix}${VIRTUAL_SEP}${stem}`;
}

function resolveSlug(slug: string): Resolved | null {
  if (!SLUG_RE.test(slug)) return null;

  // Virtual slug? Split on the FIRST `__` so the stem may itself contain `__`
  // (unlikely for date-stamped files, but keep the split unambiguous).
  const sepIdx = slug.indexOf(VIRTUAL_SEP);
  if (sepIdx > 0) {
    const prefix = slug.slice(0, sepIdx);
    const stem = slug.slice(sepIdx + VIRTUAL_SEP.length);
    const store = virtualStoreForPrefix(prefix);
    if (store && stem) {
      // The stem alone must still be slug-safe (no path separators) so the
      // read can't escape the store directory.
      if (!SLUG_RE.test(stem)) return null;
      return {
        full: path.join(store.dir, `${stem}.md`),
        category: store.category,
        fallbackTitle: stem,
        isVirtual: true,
      };
    }
    // A `__` that doesn't match a known store prefix: fall through and treat
    // the whole slug as a literal top-level doc filename (none exist today,
    // but this keeps the resolver total).
  }

  // Top-level doc.
  return {
    full: path.join(DOCS_DIR, `${slug}.md`),
    category: DEFAULT_CATEGORY, // overridden by frontmatter in getDoc/listDocs
    fallbackTitle: slug,
    isVirtual: false,
  };
}

/* -------------------------------------------------------------------------- *
 * Listing.
 * -------------------------------------------------------------------------- */

function listTopLevelDocs(): DocListing[] {
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
      category: normalizeCategory(fm.category),
    });
  }
  return rows;
}

function listVirtualStore(store: VirtualStore): DocListing[] {
  const rows: DocListing[] = [];
  for (const name of safeReadDir(store.dir)) {
    if (!name.endsWith(".md")) continue;
    if (name.endsWith(".template.md")) continue;
    const full = path.join(store.dir, name);
    if (!fs.statSync(full).isFile()) continue;
    const stem = name.replace(/\.md$/, "");
    if (!SLUG_RE.test(stem)) continue; // keep listing symmetric with getDoc
    const raw = fs.readFileSync(full, "utf-8");
    const { data, content } = matter(raw);
    const fm = data as DocFrontmatter;
    if (fm.hidden === true) continue;
    rows.push({
      slug: makeVirtualSlug(store.prefix, stem),
      title: deriveTitle(stem, fm, content),
      summary: deriveSummary(fm, content),
      status: typeof fm.status === "string" ? fm.status : null,
      lastUpdated: typeof fm.last_updated === "string" ? fm.last_updated : fileMtimeISO(full),
      // Date-stemmed activity / mentor-update files are most useful newest-
      // first; nav_order frontmatter still wins when set.
      navOrder: typeof fm.nav_order === "number" ? fm.nav_order : 50,
      category: store.category,
    });
  }
  return rows;
}

export function listDocs(): DocListing[] {
  const rows: DocListing[] = [...listTopLevelDocs()];
  for (const store of VIRTUAL_STORES) {
    rows.push(...listVirtualStore(store));
  }
  // Sort by category bucket first, then within a category: virtual stores
  // (date-stemmed) read best newest-first, so a higher slug sorts earlier;
  // top-level docs keep nav_order + title ordering. We approximate "newest
  // first" within a category by sorting on lastUpdated desc as the tiebreak
  // after navOrder, then title.
  rows.sort((a, b) => {
    const ca = CATEGORY_ORDER[a.category];
    const cb = CATEGORY_ORDER[b.category];
    if (ca !== cb) return ca - cb;
    if (a.navOrder !== b.navOrder) return a.navOrder - b.navOrder;
    // lastUpdated desc (nulls last), then title asc.
    const la = a.lastUpdated ?? "";
    const lb = b.lastUpdated ?? "";
    if (la !== lb) return la < lb ? 1 : -1;
    return a.title.localeCompare(b.title);
  });
  return rows;
}

export type DocGroup = { category: DocCategory; docs: DocListing[] };

/**
 * Listing grouped by category, in DOC_CATEGORIES display order. Empty
 * categories are omitted.
 */
export function listDocsByCategory(): DocGroup[] {
  const all = listDocs();
  const groups: DocGroup[] = [];
  for (const category of DOC_CATEGORIES) {
    const docs = all.filter((d) => d.category === category);
    if (docs.length > 0) groups.push({ category, docs });
  }
  return groups;
}

/* -------------------------------------------------------------------------- *
 * Single doc.
 * -------------------------------------------------------------------------- */

/**
 * Resolve any doc slug (top-level OR virtual `<prefix>__<stem>`) to its
 * absolute `.md` path on disk, or null when the slug is malformed /
 * out-of-store. Shared with lib/doc-comments.ts so a comment on a
 * mentor-update / activity / idea doc lands next to the right source file
 * (the `.comments/` sibling of its real directory). Does NOT check
 * existence — callers read/write and handle ENOENT.
 */
export function docFilePath(slug: string): string | null {
  return resolveSlug(slug)?.full ?? null;
}

/** True when `slug` is a syntactically valid (top-level or virtual) doc slug. */
export function isDocSlug(slug: unknown): slug is string {
  return typeof slug === "string" && resolveSlug(slug) !== null;
}

export function getDoc(slug: string): Doc | null {
  const resolved = resolveSlug(slug);
  if (!resolved) return null;
  const { full, fallbackTitle } = resolved;
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) return null;
  const raw = fs.readFileSync(full, "utf-8");
  const { data, content } = matter(raw);
  const fm = data as DocFrontmatter;
  if (fm.hidden === true) return null;
  // Virtual docs get their store's category; top-level docs read frontmatter.
  const category = resolved.isVirtual ? resolved.category : normalizeCategory(fm.category);
  return {
    slug,
    title: deriveTitle(fallbackTitle, fm, content),
    frontmatter: fm,
    body: content,
    lastUpdated: typeof fm.last_updated === "string" ? fm.last_updated : fileMtimeISO(full),
    category,
  };
}
