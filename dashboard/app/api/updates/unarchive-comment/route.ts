/**
 * POST /api/updates/unarchive-comment — counterpart to the auto-archive
 * applied by /api/updates/address-comments. Flips `archived: false` on a
 * single anchor-comment row so it returns to the main comment rail.
 *
 * Request : { taskId: number, commentId: string }
 * Response: { ok: true } | { ok: false, error: string }
 *
 * Leaves `addressed`, `addressed_in`, `addressed_note` intact — unarchive
 * only changes visibility, not the addressed marker. The synthesis
 * `anchor-comment-reply` row attached to the parent does NOT carry its
 * own `archived` field; replies follow their parent's visibility.
 *
 * Auth: `requireSessionAuth()` + `isEditorAuthed()` — matches the address
 * route. Site-password viewers get 401, signed-in non-editors get 403.
 * Concurrency: per-file `withFileLock`, same shape as address-comments.
 */
import { promises as fs } from "node:fs";
import path from "node:path";

import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { resolveTaskPath } from "@/lib/tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const locks = new Map<string, Promise<void>>();

async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
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

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

function commentsPath(taskId: number): string | null {
  const dir = resolveTaskPath(taskId);
  if (!dir) return null;
  return path.join(dir, "comments.jsonl");
}

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) {
    return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });
  }
  if (!(await isEditorAuthed())) {
    return Response.json(
      { ok: false, error: "editor cookie required" },
      { status: 403 },
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ ok: false, error: "invalid json" }, { status: 400 });
  }
  const obj = (payload ?? {}) as Record<string, unknown>;
  const taskId = validateTaskId(obj.taskId);
  if (taskId === null) {
    return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  }
  const commentId = typeof obj.commentId === "string" ? obj.commentId.trim() : "";
  if (!commentId) {
    return Response.json(
      { ok: false, error: "invalid commentId" },
      { status: 400 },
    );
  }

  const file = commentsPath(taskId);
  if (!file) {
    return Response.json({ ok: false, error: "task not found" }, { status: 404 });
  }

  let matched = false;
  await withFileLock(file, async () => {
    let raw: string;
    try {
      raw = await fs.readFile(file, "utf8");
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") return;
      throw err;
    }
    const lines = raw.split("\n");
    const out: string[] = [];
    for (const line of lines) {
      if (!line.trim()) {
        out.push(line);
        continue;
      }
      let row: Record<string, unknown>;
      try {
        row = JSON.parse(line) as Record<string, unknown>;
      } catch {
        out.push(line);
        continue;
      }
      const id = typeof row.id === "string" ? row.id : null;
      if (
        id === commentId &&
        row.kind === "anchor-comment" &&
        row.archived === true
      ) {
        // Drop the key entirely rather than setting `false` so the row
        // shape matches a never-archived comment. Leaves
        // addressed/addressed_in/addressed_note intact.
        delete row.archived;
        out.push(JSON.stringify(row));
        matched = true;
      } else {
        out.push(line);
      }
    }
    if (!matched) return;
    const next = out.filter((l, i) => l !== "" || i < out.length - 1).join("\n");
    await fs.writeFile(file, next.endsWith("\n") ? next : next + "\n", "utf8");
  });

  if (!matched) {
    return Response.json(
      { ok: false, error: "comment not found or not archived" },
      { status: 404 },
    );
  }
  return Response.json({ ok: true });
}
