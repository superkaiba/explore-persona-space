/**
 * POST /api/tasks/track — set a task's `track` field (experiment | human).
 *
 *   POST { taskId, track } -> { ok, track }
 *
 * Shells `uv run python scripts/task.py set-track <N> <track>` (via
 * `writeTaskTrackUnchecked`), so the CLI mutates the YAML frontmatter under
 * flock + one git commit + registry sync — the canonical task-state path.
 * (Writing the field through `set-body` does NOT work: set-body preserves
 * the existing frontmatter and strips any frontmatter in the new content,
 * so the edit is discarded.)
 *
 * Auth: editor-gated (`requireSessionAuth`), the same single-tier
 * site-password gate the body/title editors use.
 */
import { requireSessionAuth } from "@/lib/auth";
import { writeTaskTrackUnchecked } from "@/lib/claude-comment-ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

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
  if (taskId === null) {
    return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  }
  const track = String(obj.track ?? "").trim();
  if (track !== "experiment" && track !== "human") {
    return Response.json(
      { ok: false, error: "track must be 'experiment' or 'human'" },
      { status: 400 },
    );
  }

  const res = await writeTaskTrackUnchecked(taskId, track);
  if (!res.ok) {
    return Response.json({ ok: false, error: res.error }, { status: 500 });
  }
  return Response.json({ ok: true, track });
}
