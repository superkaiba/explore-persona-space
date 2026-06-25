/**
 * GET /tasks/<id>/data — the per-task data index for the interactive viewer
 * (clean-result v4 redesign, Phase 2).
 *
 *   GET -> { ok, taskId, artifacts: DataArtifact[] }
 *
 * Reachable on the PUBLIC task surface: it lives under `/tasks/<id>/` so the
 * proxy's existing "/tasks/* is public READ-ONLY (GET/HEAD)" allowlist covers
 * it — no proxy change. It serves only committed, already-public figure +
 * eval_results data (the same artifacts the v4 body links to via SHA-pinned
 * GitHub blob URLs), so no auth is required for the read.
 *
 * Read-only by construction: it reads figures/issue_<N>/*.meta.json + any
 * committed data_path target through `lib/task-data.ts`, all confined under the
 * repo root. It never mutates task state and never shells anything.
 */
import { getTask } from "@/lib/tasks";
import { getTaskDataIndex } from "@/lib/task-data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) {
    return Response.json({ ok: false, error: "bad id" }, { status: 400 });
  }
  const task = getTask(id);
  if (!task) {
    return Response.json({ ok: false, error: "not found" }, { status: 404 });
  }
  const index = getTaskDataIndex(id, task.body ?? "");
  return Response.json({ ok: true, ...index });
}
