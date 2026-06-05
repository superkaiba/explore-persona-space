/**
 * /sessions — live view of Happy sessions driving EPS issues.
 *
 * Reads the pinned session-progress cache file
 * (~/.eps-autonomous/session_progress.json) server-side via
 * lib/sessions.ts. The writer is a separate cron / agent; this page is
 * read-only and tolerates a missing or empty cache file with a clean
 * "no session data yet" state.
 *
 * Issue numbers are resolved against the task registry so we can show
 * the live task title and link to /tasks/<N>. Sessions whose issue
 * isn't in the registry render as "issue #N (not found)" without a
 * link, matching the spec's "no resolvable issue" branch.
 *
 * Sort: by last_activity_ts descending (freshest at top) — same shape
 * as an oncall board, so the first row is always "what's alive right
 * now". Sessions without a last_activity_ts sink to the bottom.
 *
 * force-dynamic so router.refresh() on the client re-reads the cache
 * file every poll, matching every other disk-reading route.
 */
import { loadSessionSnapshot, type SessionRow } from "@/lib/sessions";
import { listAllTasks, type TaskListing } from "@/lib/tasks";
import { SessionsTable, type SessionRowView } from "./SessionsTable";

export const dynamic = "force-dynamic";

export default async function SessionsPage() {
  const snapshot = loadSessionSnapshot();
  const tasksById = indexTasksById(listAllTasks());
  const rows = snapshot.sessions
    .map((s) => toView(s, tasksById))
    .sort(byLastActivityDesc);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Sessions
        </h1>
        <p className="mt-1 text-sm text-stone-600">
          Happy sessions driving EPS issues. The summarizer cron writes
          the cache every 5 minutes; this page auto-refreshes every 30
          seconds.
        </p>
      </header>

      <SessionsTable
        rows={rows}
        updatedAt={snapshot.updatedAt}
        readError={snapshot.readError}
      />
    </div>
  );
}

function indexTasksById(tasks: TaskListing[]): Map<number, TaskListing> {
  const out = new Map<number, TaskListing>();
  for (const t of tasks) out.set(t.id, t);
  return out;
}

function toView(
  s: SessionRow,
  tasksById: Map<number, TaskListing>,
): SessionRowView {
  const task = s.issue != null ? tasksById.get(s.issue) ?? null : null;
  return {
    sessionId: s.sessionId,
    issue: s.issue,
    issueResolved: task !== null,
    issueTitle: task?.title ?? null,
    // Prefer the cache's status (it's what the session itself believes
    // it's doing right now); fall back to the registry status if the
    // session row has none.
    status: s.status ?? task?.status ?? null,
    dir: s.dir,
    live: s.live,
    pid: s.pid,
    summary: s.summary,
    summaryModel: s.summaryModel,
    summaryTs: s.summaryTs,
    lastActivityTs: s.lastActivityTs,
    error: s.error,
  };
}

function byLastActivityDesc(a: SessionRowView, b: SessionRowView): number {
  const ta = a.lastActivityTs ? Date.parse(a.lastActivityTs) : NaN;
  const tb = b.lastActivityTs ? Date.parse(b.lastActivityTs) : NaN;
  const aOk = Number.isFinite(ta);
  const bOk = Number.isFinite(tb);
  if (aOk && bOk) return tb - ta;
  if (aOk) return -1;
  if (bOk) return 1;
  // Both missing — stable secondary by issue number ascending.
  const ia = a.issue ?? Number.POSITIVE_INFINITY;
  const ib = b.issue ?? Number.POSITIVE_INFINITY;
  return ia - ib;
}
