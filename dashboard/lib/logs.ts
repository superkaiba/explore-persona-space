/**
 * Data layer shared by two consumers:
 *
 *   1. The retired `/log` route's data helpers — `listLogEntries`,
 *      `listCleanResults`, `getLogEntry`, `writeLogEntryBody`,
 *      `isValidEntryId`. These remain because `/preview/log` and
 *      `/api/log/comment` (neither owned by the updates merge) still import
 *      them. They read `logs/{daily,weekly,ideation}/*.md` and the
 *      clean-result rows from `tasks/`.
 *
 *   2. The consolidated `/updates` feed — `listUpdatesFeed`. A
 *      reverse-chronological POINTER feed. Each item links to its canonical
 *      home (`/results/<id>` for completed clean-results, `/docs/<slug>` for
 *      dated docs) and carries only enough metadata to render a card. It does
 *      NOT carry the full body — pointer cards never re-render canon.
 *
 * All functions are server-only.
 *
 * Tolerates a missing `logs/` directory (the skills implementer populates
 * it lazily). When it isn't there yet, the feed just contains
 * clean-results + whichever dated-doc dirs do exist.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REPO_ROOT } from "./repo";
import {
  getRegistry,
  type Frontmatter,
} from "./tasks";
import { listPublicResults } from "./results";

const LOGS_DIR = path.join(REPO_ROOT, "logs");
const DOCS_DIR = path.join(REPO_ROOT, "docs");

export type LogEntryKind = "daily" | "weekly" | "ideation";
export type FeedItemKind = LogEntryKind | "clean-result";

export type LogEntry = {
  entryId: string;                  // e.g. "daily-2026-05-26"
  kind: LogEntryKind;
  date: string;                     // ISO YYYY-MM-DD
  title: string;
  includedTasks: number[];
  visible: boolean;
  tags: string[];
  body: string;                     // markdown body (no frontmatter)
  filePath: string;                 // absolute path
};

export type CleanResult = {
  entryId: string;                  // e.g. "task-365"
  kind: "clean-result";
  taskId: number;
  date: string;                     // promotion date or last-modified, ISO YYYY-MM-DD
  title: string;
  classification: "useful" | "not-useful" | "pending";
  body: string;
  status: string;                   // e.g. "completed"
};

export type FeedItem = LogEntry | CleanResult;

type LogEntryFrontmatter = {
  kind?: string;
  date?: string;
  title?: string;
  included_tasks?: unknown;
  visible?: unknown;
  tags?: unknown;
};

/* -------------------------------------------------------------------------- *
 * Entry-id resolution.
 *
 * Bidirectional helpers so `getLogEntry(entryId)` and the comment route
 * can both reach back to the on-disk file without re-scanning. The id
 * scheme:
 *   logs/daily/2026-05-26.md             -> "daily-2026-05-26"
 *   logs/weekly/2026-W22.md              -> "weekly-2026-W22"
 *   logs/ideation/2026-05-26-foo.md      -> "ideation-2026-05-26-foo"
 * -------------------------------------------------------------------------- */

const ENTRY_ID_RE = /^(daily|weekly|ideation)-([\w-]+)$/;

export function isValidEntryId(entryId: string): boolean {
  return ENTRY_ID_RE.test(entryId);
}

function entryIdToPath(entryId: string): string | null {
  const m = entryId.match(ENTRY_ID_RE);
  if (!m) return null;
  const kind = m[1] as LogEntryKind;
  const rest = m[2];
  return path.join(LOGS_DIR, kind, `${rest}.md`);
}

function fileToEntryId(kind: LogEntryKind, filename: string): string {
  // e.g. ("daily", "2026-05-26.md") -> "daily-2026-05-26"
  const stem = filename.replace(/\.md$/, "");
  return `${kind}-${stem}`;
}

/* -------------------------------------------------------------------------- *
 * Filesystem helpers.
 * -------------------------------------------------------------------------- */

function safeReaddir(dir: string): string[] {
  try {
    return fs.readdirSync(dir);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return [];
    throw err;
  }
}

function parseLogFile(absPath: string, kind: LogEntryKind): LogEntry | null {
  let raw: string;
  try {
    raw = fs.readFileSync(absPath, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return null;
    throw err;
  }
  const fm = matter(raw);
  const data = fm.data as LogEntryFrontmatter;

  // Required: date + title. If missing, skip the file rather than crash —
  // the skills implementer's stubs always include them, but a hand-edited
  // file could be malformed.
  const date = typeof data.date === "string" ? data.date : null;
  const title = typeof data.title === "string" ? data.title : null;
  if (!date || !title) return null;

  const includedTasks: number[] = Array.isArray(data.included_tasks)
    ? data.included_tasks.filter((n): n is number => Number.isFinite(n))
    : [];
  const tags: string[] = Array.isArray(data.tags)
    ? data.tags.filter((t): t is string => typeof t === "string")
    : [];
  const visible = data.visible === true;

  const filename = path.basename(absPath);
  return {
    entryId: fileToEntryId(kind, filename),
    kind,
    date,
    title,
    includedTasks,
    visible,
    tags,
    body: fm.content,
    filePath: absPath,
  };
}

/* -------------------------------------------------------------------------- *
 * Public reads.
 * -------------------------------------------------------------------------- */

export async function listLogEntries(opts: {
  limit?: number;
  kinds?: LogEntryKind[];
  includeDrafts?: boolean;
} = {}): Promise<LogEntry[]> {
  const kindsFilter = opts.kinds && opts.kinds.length > 0 ? new Set(opts.kinds) : null;
  const includeDrafts = opts.includeDrafts === true;
  const allKinds: LogEntryKind[] = ["daily", "weekly", "ideation"];

  const out: LogEntry[] = [];
  for (const kind of allKinds) {
    if (kindsFilter && !kindsFilter.has(kind)) continue;
    const dir = path.join(LOGS_DIR, kind);
    for (const filename of safeReaddir(dir)) {
      if (!filename.endsWith(".md")) continue;
      const entry = parseLogFile(path.join(dir, filename), kind);
      if (!entry) continue;
      if (!includeDrafts && !entry.visible) continue;
      out.push(entry);
    }
  }

  // Newest first (string sort works because ISO dates + ISO week strings
  // both sort lexicographically).
  out.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  if (typeof opts.limit === "number") return out.slice(0, opts.limit);
  return out;
}

export async function listCleanResults(opts: {
  limit?: number;
  includeNotUseful?: boolean;
} = {}): Promise<CleanResult[]> {
  // `includeNotUseful` defaults to `true` per spec — the useful-only chip
  // filters in the client. We keep the option for future server-side
  // pruning if needed.
  const includeNotUseful = opts.includeNotUseful !== false;

  const reg = getRegistry();
  const out: CleanResult[] = [];

  for (const [idStr, entry] of Object.entries(reg.tasks)) {
    if (!entry.has_clean_result) continue;
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;

    // `entry.path` is recorded relative to the repo root (e.g.
    // "tasks/completed/365"), so `REPO_ROOT + entry.path` resolves to the
    // task directory directly without going through REGISTRY_PATH.
    const absDir = path.join(REPO_ROOT, entry.path);
    const bodyPath = path.join(absDir, "body.md");

    let bodyText: string;
    let bodyStat: fs.Stats;
    try {
      bodyText = fs.readFileSync(bodyPath, "utf8");
      bodyStat = fs.statSync(bodyPath);
    } catch {
      continue;
    }

    const fm = matter(bodyText);
    const data = fm.data as Frontmatter;
    const rawClass = typeof data.classification === "string" ? data.classification : "pending";
    const classification: CleanResult["classification"] =
      rawClass === "useful" || rawClass === "not-useful" ? rawClass : "pending";
    if (!includeNotUseful && classification === "not-useful") continue;

    const promoted = typeof data.promoted_at === "string" ? data.promoted_at : null;
    const date = (promoted ?? bodyStat.mtime.toISOString()).slice(0, 10);

    out.push({
      entryId: `task-${id}`,
      kind: "clean-result",
      taskId: id,
      date,
      title: entry.title || `Task #${id}`,
      classification,
      body: fm.content,
      status: entry.status,
    });
  }

  out.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  if (typeof opts.limit === "number") return out.slice(0, opts.limit);
  return out;
}

export async function getLogEntry(entryId: string): Promise<LogEntry | null> {
  if (!isValidEntryId(entryId)) return null;
  const abs = entryIdToPath(entryId);
  if (!abs) return null;
  const m = entryId.match(ENTRY_ID_RE);
  if (!m) return null;
  const kind = m[1] as LogEntryKind;
  return parseLogFile(abs, kind);
}

/**
 * Replace the body section of a log entry while preserving frontmatter.
 *
 * Mirrors `writeTaskBodyUnchecked` in `claude-comment-ops.ts` for the
 * body-edit comment path: the calling route has already gated on
 * `isEditorAuthed()` (or fire-and-forget detached after gating), so we
 * skip re-checking here.
 */
export async function writeLogEntryBody(
  entryId: string,
  newBody: string,
): Promise<{ ok: true } | { ok: false; error: string }> {
  if (!isValidEntryId(entryId)) return { ok: false, error: "invalid entryId" };
  if (typeof newBody !== "string") return { ok: false, error: "body must be a string" };
  if (newBody.length > 1_000_000) return { ok: false, error: "body exceeds 1MB" };
  const abs = entryIdToPath(entryId);
  if (!abs) return { ok: false, error: "could not resolve entryId to path" };

  let raw: string;
  try {
    raw = fs.readFileSync(abs, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return { ok: false, error: "log entry not found on disk" };
    throw err;
  }
  // Preserve the existing frontmatter verbatim and swap the body. gray-
  // matter's stringify re-emits frontmatter in canonical YAML order which
  // can churn diffs unnecessarily; instead, splice manually so we don't
  // touch a single byte of frontmatter when nothing about it changed.
  const fmMatch = raw.match(/^---\n[\s\S]*?\n---\n?/);
  const next = fmMatch ? `${fmMatch[0]}${newBody}` : newBody;
  try {
    fs.writeFileSync(abs, next, "utf8");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, error: `write failed: ${msg}` };
  }
  return { ok: true };
}

/* ========================================================================== *
 * Consolidated /updates pointer feed.
 *
 * The merged timeline that replaces both the old /updates feed AND the
 * retired /log feed. Reverse-chronological. Each entry is a POINTER to its
 * canonical home; the feed deliberately omits the full body so a card never
 * re-renders canon.
 *
 * Two sources:
 *   - completed clean-results  -> /results/<id>  (via lib/results.listPublicResults)
 *   - dated docs               -> /docs/<slug>   (docs/mentor_updates/*.md,
 *                                                 logs/daily/*.md, logs/weekly/*.md)
 *
 * CROSS-AGENT CONTRACT (dated-doc slug): the /docs route's resolver
 * (lib/docs.ts, owned by the docs-categorization work) and this feed must
 * agree on the slug a dated doc resolves to. Both sides go through
 * `datedDocSlug(category, stem)`. If the docs resolver lands a different
 * nested-slug encoding, reconcile by pointing both at this single helper.
 * ========================================================================== */

/**
 * The category buckets that hold dated docs. `mentor_updates` lives under
 * `docs/`; `daily` / `weekly` live under `logs/`. Each maps to a source dir
 * and a human label for the card badge.
 */
const DATED_DOC_SOURCES: ReadonlyArray<{
  category: "mentor_updates" | "daily" | "weekly";
  dir: string;
  label: string;
}> = [
  { category: "mentor_updates", dir: path.join(DOCS_DIR, "mentor_updates"), label: "Mentor update" },
  { category: "daily", dir: path.join(LOGS_DIR, "daily"), label: "Daily" },
  { category: "weekly", dir: path.join(LOGS_DIR, "weekly"), label: "Weekly" },
];

export type UpdateFeedCategory = "result" | "mentor_updates" | "daily" | "weekly";

/**
 * One pointer card in the /updates feed. `href` is the canonical destination;
 * `body` is intentionally absent (pointer cards don't re-render canon).
 */
export type UpdateFeedItem = {
  /** Stable key, e.g. "result-365" or "doc-mentor_updates__2026-05-28". */
  itemId: string;
  category: UpdateFeedCategory;
  /** "Result" | "Mentor update" | "Daily" | "Weekly". */
  categoryLabel: string;
  /** ISO YYYY-MM-DD used for sorting + the date-range filter. */
  date: string;
  title: string;
  /** Short plain-text teaser (no markdown), or null when unavailable. */
  excerpt: string | null;
  /** Canonical destination: /results/<id> or /docs/<slug>. */
  href: string;
  /** Confidence tag for result cards; null for docs. */
  confidence: "HIGH" | "MODERATE" | "LOW" | null;
};

/**
 * Slug a dated doc resolves to under the single `/docs/[slug]` segment.
 *
 * Flattens `<category>/<stem>` to a path-separator-free token so it fits one
 * dynamic segment and passes the docs resolver's slug regex
 * (`^[A-Za-z0-9][A-Za-z0-9._-]*$`, which permits `_`, `.`, `-`). The
 * double-underscore separator is unambiguous: category tokens
 * (`mentor_updates` / `daily` / `weekly`) contain only single underscores,
 * and dated stems are date-prefixed, so `__` never appears inside either side.
 */
export function datedDocSlug(category: string, stem: string): string {
  return `${category}__${stem}`;
}

/**
 * Read a dated-doc directory into pointer items. Tolerates a missing dir
 * (returns []). Title falls back to the first H1 then the stem; date falls
 * back to a `YYYY-MM-DD` prefix in the filename, then frontmatter `date`,
 * then the file mtime. Honors `hidden: true` frontmatter.
 */
function readDatedDocDir(
  category: "mentor_updates" | "daily" | "weekly",
  dir: string,
  label: string,
): UpdateFeedItem[] {
  const out: UpdateFeedItem[] = [];
  for (const filename of safeReaddir(dir)) {
    if (!filename.endsWith(".md")) continue;
    const stem = filename.replace(/\.md$/, "");
    const abs = path.join(dir, filename);

    let raw: string;
    let stat: fs.Stats;
    try {
      raw = fs.readFileSync(abs, "utf8");
      stat = fs.statSync(abs);
    } catch {
      continue;
    }

    const fm = matter(raw);
    const data = fm.data as { title?: unknown; date?: unknown; hidden?: unknown };
    if (data.hidden === true) continue;

    // Date: filename `YYYY-MM-DD...` prefix wins, then frontmatter, then mtime.
    const stemDate = stem.match(/^(\d{4}-\d{2}-\d{2})/);
    const fmDate = typeof data.date === "string" ? data.date.slice(0, 10) : null;
    const date = (stemDate?.[1] ?? fmDate ?? stat.mtime.toISOString().slice(0, 10));

    const fmTitle = typeof data.title === "string" && data.title.trim() ? data.title.trim() : null;
    const h1 = fm.content.match(/^#\s+(.+?)\s*$/m);
    const title = fmTitle ?? (h1 ? h1[1].trim() : stem);

    out.push({
      itemId: `doc-${datedDocSlug(category, stem)}`,
      category,
      categoryLabel: label,
      date,
      title,
      excerpt: datedDocExcerpt(fm.content),
      href: `/docs/${datedDocSlug(category, stem)}`,
      confidence: null,
    });
  }
  return out;
}

/** First substantive paragraph, flattened to plain text (mirrors lib/docs). */
function datedDocExcerpt(body: string, maxLength = 240): string | null {
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    if (
      line.startsWith("#") ||
      line.startsWith(">") ||
      line.startsWith("---") ||
      line.startsWith("|") ||
      line.startsWith("```") ||
      line.startsWith("<!--")
    )
      continue;
    const text = line
      .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
      .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
      .replace(/[*_`~]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) continue;
    return text.length > maxLength ? `${text.slice(0, maxLength - 1).trim()}…` : text;
  }
  return null;
}

/**
 * The consolidated /updates feed: completed clean-results + dated docs,
 * newest first. Pointer cards only — no bodies.
 */
export function listUpdatesFeed({ limit }: { limit?: number } = {}): UpdateFeedItem[] {
  const out: UpdateFeedItem[] = [];

  // Source 1: completed + classification=useful clean-results, via the public
  // Results data layer (authoritative `classification` predicate, excludes
  // format-exemplars). These point at /results/<id>.
  for (const r of listPublicResults()) {
    out.push({
      itemId: `result-${r.id}`,
      category: "result",
      categoryLabel: "Result",
      date: r.dayKey,
      title: r.title,
      excerpt: r.excerpt || null,
      href: r.href,
      confidence: r.confidence,
    });
  }

  // Source 2: dated docs.
  for (const src of DATED_DOC_SOURCES) {
    out.push(...readDatedDocDir(src.category, src.dir, src.label));
  }

  // Newest first; ISO date strings sort lexicographically. Break ties on the
  // itemId so the order is deterministic across reloads.
  out.sort((a, b) =>
    a.date < b.date ? 1 : a.date > b.date ? -1 : a.itemId < b.itemId ? 1 : -1,
  );

  if (typeof limit === "number") return out.slice(0, limit);
  return out;
}
