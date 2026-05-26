/**
 * Data layer for the `/log` route — reads `logs/{daily,weekly,ideation}/*.md`
 * from the repo root and merges them with the existing clean-result rows
 * (via `lib/tasks.ts`) into a single chronological feed.
 *
 * All functions are server-only.
 *
 * Tolerates a missing `logs/` directory (the skills implementer populates
 * it lazily). When it isn't there yet, the feed just contains
 * clean-results.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REPO_ROOT } from "./repo";
import {
  getRegistry,
  type Frontmatter,
} from "./tasks";

const LOGS_DIR = path.join(REPO_ROOT, "logs");

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
