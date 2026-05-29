/**
 * Per-doc anchored comments for the /docs/<slug> pages.
 *
 * Mirrors the task/updates `comments.jsonl` flow (see
 * `app/api/updates/comment` + `address-comments`) but targets the living
 * research docs surfaced by `lib/docs.ts` — both the top-level `docs/*.md`
 * and the virtual stores (mentor_updates, logs/daily, logs/weekly, ideas).
 * Comments are stored one-per-line as JSON in a `.comments/<stem>.jsonl`
 * file SIBLING to the doc's real directory (gitignored — they are transient
 * review artifacts, not version-controlled doc content). Routing through
 * `docFilePath` keeps each store's comments next to its source file:
 *
 *   docs/open_questions.md            -> docs/.comments/open_questions.jsonl
 *   docs/mentor_updates/2026-05-28.md -> docs/mentor_updates/.comments/2026-05-28.jsonl
 *   logs/daily/2026-05-28.md          -> logs/daily/.comments/2026-05-28.jsonl
 *
 * Two consumers:
 *   - `app/api/docs/comment`           (POST/GET/DELETE a comment)
 *   - `app/api/docs/address-comments`  ("Address all" — Claude rewrites the
 *                                       doc to address every open comment)
 *
 * The comment schema reuses the shared field names (`in_reply_to`,
 * `addressed`, ...) so the unified comment layer renders these the same way
 * it renders task/updates comments. The shared anchored-comment surface uses
 * `quote` for highlight-to-comment anchoring.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { docFilePath, isDocSlug } from "@/lib/docs";

// Re-exported for back-compat with callers that imported the slug validator
// from here. The authoritative validator lives in lib/docs.ts (handles both
// top-level and virtual slugs).
export { isDocSlug as isValidSlug };

export type DocCommentRow = {
  id: string;
  ts: string;
  author: string;
  kind: "doc-comment" | "doc-comment-reply";
  body: string;
  // Optional pointer to the doc section/question the comment is about.
  section_id?: string; // heading slug or `q:` anchor id
  section_label?: string; // human label shown in the rail
  quote?: string; // optional pasted snippet the comment refers to (anchor)
  in_reply_to?: string;
  addressed?: boolean;
  addressed_in?: string;
  addressed_note?: string;
};

/** Absolute path to the source `.md` for a (top-level or virtual) slug. */
export function docPathForSlug(slug: string): string | null {
  if (!isDocSlug(slug)) return null;
  return docFilePath(slug);
}

/** Absolute path to the gitignored `.comments/<stem>.jsonl` for a slug. */
export function commentsPathForSlug(slug: string): string | null {
  const docPath = docPathForSlug(slug);
  if (!docPath) return null;
  const dir = path.dirname(docPath);
  const stem = path.basename(docPath, ".md");
  return path.join(dir, ".comments", `${stem}.jsonl`);
}

/* In-process mutex per comments file — same shape as the updates route. */
const locks = new Map<string, Promise<void>>();

export async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const prev = locks.get(file) ?? Promise.resolve();
  let release: () => void = () => {};
  const next = new Promise<void>((resolve) => {
    release = resolve;
  });
  locks.set(
    file,
    prev.then(() => next),
  );
  await prev;
  try {
    return await fn();
  } finally {
    release();
    if (locks.get(file) === next) locks.delete(file);
  }
}

export async function readComments(file: string): Promise<DocCommentRow[]> {
  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }
  const out: DocCommentRow[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    // Skip a malformed JSONL line rather than 500 the whole rail — same
    // tolerance the task comment reader uses.
    try {
      const row = JSON.parse(line) as DocCommentRow;
      if (row && typeof row.id === "string") out.push(row);
    } catch {
      continue;
    }
  }
  return out;
}

export async function appendComment(file: string, row: DocCommentRow): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.appendFile(file, JSON.stringify(row) + "\n", "utf8");
}

export async function rewriteComments(file: string, rows: DocCommentRow[]): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const text = rows.map((r) => JSON.stringify(r)).join("\n");
  await fs.writeFile(file, text ? text + "\n" : "", "utf8");
}

export async function readDocRaw(slug: string): Promise<string | null> {
  const p = docPathForSlug(slug);
  if (!p) return null;
  try {
    return await fs.readFile(p, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
}

export async function writeDocRaw(slug: string, text: string): Promise<boolean> {
  const p = docPathForSlug(slug);
  if (!p) return false;
  await fs.writeFile(p, text.endsWith("\n") ? text : text + "\n", "utf8");
  return true;
}
