/**
 * GET /api/tasks/progress — pipeline progress + ETA per in-flight task
 * (task #587).
 *
 *   GET -> { ok, generated_at, stale, tasks: { <id>: TaskProgressView } }
 *
 * Read-only by construction: it reads the Python-cron-materialized snapshot
 * (`~/.eps-autonomous/task_progress.json`) through `lib/progress.ts` and the
 * live statuses from REGISTRY via `listAllTasks()` — it never touches task
 * state and never shells anything. Live statuses gate the rendering: tasks
 * whose live status has no stage floor return no entry.
 *
 * Auth: editor-gated (`requireSessionAuth`), mirroring
 * `app/api/tasks/track/route.ts` (401 JSON on failure).
 */
import { requireSessionAuth } from "@/lib/auth";
import { getProgressMap, getProgressMeta } from "@/lib/progress";
import { listAllTasks } from "@/lib/tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  const liveStatuses: Record<number, string> = {};
  for (const t of listAllTasks()) liveStatuses[t.id] = t.status;

  const meta = getProgressMeta();
  return Response.json({
    ok: true,
    generated_at: meta.generatedAt,
    stale: meta.stale,
    tasks: getProgressMap(liveStatuses),
  });
}
