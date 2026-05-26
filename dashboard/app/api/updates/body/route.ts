/**
 * POST /api/updates/body — write a new body for `tasks/<N>/body.md` from
 * the inline WYSIWYG editor (`CardBodyEditor.tsx`) on /updates.
 *
 * Reuses `saveTaskBody` from app/tasks/[id]/edit/actions.ts, which shells
 * out to `uv run python scripts/task.py set-body <N> --file <tmp>` — the
 * CLI acquires flock on ~/.task-workflow/lock and commits one git commit,
 * so concurrent edits cannot corrupt the file.
 *
 * Auth: gated on `isEditorAuthed()`, which is satisfied by any valid
 * site-password session cookie. Single-tier auth — whoever holds the
 * `SITE_PASSWORD` can rewrite results.
 */
import { saveTaskBody } from "@/app/tasks/[id]/edit/actions";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type PostBody = { taskId?: unknown; body?: unknown };

export async function POST(request: Request) {
  let json: PostBody;
  try {
    json = (await request.json()) as PostBody;
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const taskIdNum = Number(json.taskId);
  if (!Number.isFinite(taskIdNum) || !Number.isInteger(taskIdNum) || taskIdNum < 1) {
    return Response.json({ error: "taskId must be a positive integer" }, { status: 400 });
  }
  if (typeof json.body !== "string") {
    return Response.json({ error: "body must be a string" }, { status: 400 });
  }

  // `saveTaskBody` itself enforces `isEditorAuthed()` and returns
  // {ok:false, error:"unauthorized"} when it fails — surface that as 401.
  const result = await saveTaskBody(taskIdNum, json.body);
  if (!result.ok) {
    const status = result.error === "unauthorized" ? 401 : 500;
    return Response.json({ error: result.error }, { status });
  }
  return Response.json({ ok: true });
}
