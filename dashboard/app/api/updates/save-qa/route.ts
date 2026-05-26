/**
 * Persist auto-saved Q&A pairs from the /updates page MentorClaudePanel
 * into `tasks/<status>/<id>/comments.jsonl`.
 *
 *   POST   { taskId, question, answer }     -> { ok, id }
 *   GET    ?taskId=N                        -> { comments: [...] }
 *   DELETE { taskId, commentId }            -> { ok }
 *
 * Each saved Q&A is one JSONL row with `author: "claude-qa"` and
 * `kind: "note"` so the existing comments-on-tasks readers (CommentList,
 * `task.py list-comments` if it later grows the subcommand) treat them
 * as ordinary notes. We append directly via `fs` instead of shelling out
 * to `task.py` because no `add-comment` CLI exists today (the existing
 * `comment-actions.ts` references one that was never built).
 *
 * Concurrency: writes serialize through `fs.promises.appendFile` plus
 * a per-file mutex bucket to avoid id collisions when two tabs save at
 * once. DELETE rewrites the whole file under the same lock.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { requireSessionAuth } from "@/lib/auth";
import { resolveTaskPath } from "@/lib/tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_CHARS = 50_000;

type SavedQaRow = {
  id: string;
  ts: string;
  author: "claude-qa";
  kind: "note";
  body: string;
  question: string;
  answer: string;
};

/* -------------------------------------------------------------------------- *
 * In-process mutex per comments.jsonl path. Next.js can serve concurrent
 * requests inside the same worker; without a mutex two simultaneous appends
 * can interleave bytes or produce duplicate ids on retry.
 * -------------------------------------------------------------------------- */

const locks = new Map<string, Promise<void>>();

async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const prev = locks.get(file) ?? Promise.resolve();
  let release: () => void;
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
    release!();
    if (locks.get(file) === next) locks.delete(file);
  }
}

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

function commentsPath(taskId: number): string | null {
  const dir = resolveTaskPath(taskId);
  if (!dir) return null;
  return path.join(dir, "comments.jsonl");
}

function renderBody(question: string, answer: string): string {
  return `**Q:** ${question}\n\n**A:**\n\n${answer}`;
}

/* -------------------------------------------------------------------------- *
 * POST — append a new Q&A row.
 * -------------------------------------------------------------------------- */

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ ok: false, error: "invalid json" }, { status: 400 });
  }
  const obj = (payload ?? {}) as Record<string, unknown>;
  const taskId = validateTaskId(obj.taskId);
  if (taskId === null) return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  const question = String(obj.question ?? "").trim();
  const answer = String(obj.answer ?? "").trim();
  if (!question) return Response.json({ ok: false, error: "question empty" }, { status: 400 });
  if (!answer) return Response.json({ ok: false, error: "answer empty" }, { status: 400 });
  const body = renderBody(question, answer);
  if (body.length > MAX_BODY_CHARS) {
    return Response.json(
      { ok: false, error: `body exceeds ${MAX_BODY_CHARS} chars` },
      { status: 413 },
    );
  }

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: false, error: "task not found" }, { status: 404 });

  const row: SavedQaRow = {
    id: `qa-${randomUUID()}`,
    ts: new Date().toISOString(),
    author: "claude-qa",
    kind: "note",
    body,
    question,
    answer,
  };

  try {
    await withFileLock(file, async () => {
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.appendFile(file, JSON.stringify(row) + "\n", "utf8");
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `append failed: ${msg}` }, { status: 500 });
  }

  return Response.json({ ok: true, id: row.id, ts: row.ts });
}

/* -------------------------------------------------------------------------- *
 * GET — list saved Q&A rows for a task (most recent last in file order).
 * -------------------------------------------------------------------------- */

export async function GET(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  const url = new URL(request.url);
  const taskId = validateTaskId(url.searchParams.get("taskId"));
  if (taskId === null) return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: true, comments: [] });

  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return Response.json({ ok: true, comments: [] });
    throw err;
  }

  const comments: SavedQaRow[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const obj = JSON.parse(line) as Partial<SavedQaRow>;
      if (obj.author === "claude-qa" && typeof obj.id === "string") {
        comments.push({
          id: obj.id,
          ts: typeof obj.ts === "string" ? obj.ts : "",
          author: "claude-qa",
          kind: "note",
          body: typeof obj.body === "string" ? obj.body : "",
          question: typeof obj.question === "string" ? obj.question : "",
          answer: typeof obj.answer === "string" ? obj.answer : "",
        });
      }
    } catch {
      // Skip malformed line; other comments may be in non-QA shapes.
    }
  }

  return Response.json({ ok: true, comments });
}

/* -------------------------------------------------------------------------- *
 * DELETE — remove a saved Q&A row by id.
 * -------------------------------------------------------------------------- */

export async function DELETE(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ ok: false, error: "invalid json" }, { status: 400 });
  }
  const obj = (payload ?? {}) as Record<string, unknown>;
  const taskId = validateTaskId(obj.taskId);
  if (taskId === null) return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  const commentId = String(obj.commentId ?? "").trim();
  if (!commentId) return Response.json({ ok: false, error: "commentId empty" }, { status: 400 });

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: false, error: "task not found" }, { status: 404 });

  let removed = false;
  try {
    await withFileLock(file, async () => {
      let raw: string;
      try {
        raw = await fs.readFile(file, "utf8");
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (code === "ENOENT") return;
        throw err;
      }
      const kept: string[] = [];
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line) as { id?: unknown };
          if (parsed.id === commentId) {
            removed = true;
            continue;
          }
        } catch {
          // Preserve unparseable lines verbatim — they belong to other tools.
        }
        kept.push(line);
      }
      const next = kept.length === 0 ? "" : kept.join("\n") + "\n";
      await fs.writeFile(file, next, "utf8");
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `delete failed: ${msg}` }, { status: 500 });
  }

  return Response.json({ ok: true, removed });
}
