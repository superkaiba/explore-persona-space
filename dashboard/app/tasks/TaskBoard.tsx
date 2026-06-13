"use client";

/**
 * Client shell for the /tasks page. Owns two pieces of URL-synced view
 * state so the view is shareable (same pattern as /updates UpdatesFeed):
 *
 *   - `?view=list|kanban`           (default kanban)
 *   - `?track=experiment|human|all` (default all)
 *   - `?related=<id>`               (show only that task's family — the
 *     connected component over frontmatter parent_id edges; overrides the
 *     track filter while active)
 *
 * Unseen-update glow: a card whose `lastActivityAt` is newer than this
 * device's last visit to the task (localStorage, see
 * components/tasks/task-seen.ts) pulses amber until the detail page is
 * opened (or the card is clicked).
 *
 * The server page (`app/tasks/page.tsx`) loads every task once (with its
 * derived `track`) and hands the flat array down. Filtering + grouping
 * happen here so the toggle is instant (no round-trip). Read-only: status
 * is workflow-owned, so there's no drag-and-drop — each card is a Link to
 * the task detail page.
 *
 * Kanban columns are the lifecycle statuses in canonical order from
 * lib/repo, with `blocked` pulled near the front and `archived` hidden by
 * default (a toggle reveals it). The List view keeps the original
 * grouped-by-status accordion.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { TaskListing, Track } from "@/lib/tasks";
import type { TaskProgressView } from "@/lib/progress";
import { TaskProgressBar } from "@/components/tasks/TaskProgressBar";
import {
  markAllTasksSeen,
  markTaskSeen,
  readSeenState,
  type SeenState,
} from "@/components/tasks/task-seen";
import { STATUS_DISPLAY_ORDER, STATUS_LABELS, type Status } from "@/lib/repo";

/** Per-card callbacks/lookups threaded into both views (kept as one object
 * so the prop plumbing stays flat). */
type CardCtx = {
  isUnseen: (t: TaskListing) => boolean;
  /** Takes the whole task so the seen stamp can be clamped to its
   * lastActivityAt (client clocks can run behind the VM's mtimes). */
  markSeen: (t: TaskListing) => void;
  /** Family size minus self (0 = no relatives → button hidden). */
  relativesOf: (id: number) => number;
  showRelated: (id: number) => void;
};

type ViewMode = "list" | "kanban";
type TrackFilter = "experiment" | "human" | "all";

// Kanban column order: blocked sits near the front (it's active-but-stuck
// work the user wants to see early), then `on_hold` (parked work, kept just
// left of proposed), then the canonical lifecycle order from proposed →
// completed, with archived last + hidden by default.
const KANBAN_COLUMN_ORDER: Status[] = [
  "blocked",
  "on_hold",
  "proposed",
  "planning",
  "plan_pending",
  "approved",
  "running",
  "verifying",
  "interpreting",
  "reviewing",
  "followups_running",
  "awaiting_promotion",
  "completed",
  "archived",
];

const TRACK_TABS: { key: TrackFilter; label: string }[] = [
  { key: "experiment", label: "Experiments" },
  { key: "human", label: "Human" },
  { key: "all", label: "All" },
];

export function TaskBoard({
  tasks,
  progress = {},
  initialView,
  initialTrack,
}: {
  tasks: TaskListing[];
  /** Pipeline progress views for in-flight tasks (task #587), computed
   * server-side from the cron snapshot + LIVE statuses. Existence of an
   * entry implies live-status validity — the card just renders it. */
  progress?: Record<number, TaskProgressView>;
  initialView: ViewMode;
  initialTrack: TrackFilter;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [showArchived, setShowArchived] = useState(false);

  // Lightweight auto-refresh: the board has no live data channel, so a tab
  // left open drifts behind the workflow. Re-pull the RSC payload (the
  // force-dynamic server page re-reads tasks/ from disk) every 60s while
  // visible, and immediately on tab refocus. Client view state (track tab,
  // view mode, scroll) survives router.refresh().
  useEffect(() => {
    const refresh = () => {
      if (document.visibilityState === "visible") router.refresh();
    };
    const id = setInterval(refresh, 60_000);
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [router]);

  const view: ViewMode =
    (searchParams.get("view") as ViewMode | null) === "list"
      ? "list"
      : searchParams.get("view") === "kanban"
        ? "kanban"
        : initialView;

  const trackParam = searchParams.get("track");
  const track: TrackFilter =
    trackParam === "experiment" || trackParam === "human" || trackParam === "all"
      ? trackParam
      : initialTrack;

  // ?related=<id> — family filter (parent/children/siblings via parent_id).
  const relatedParam = searchParams.get("related");
  const relatedId =
    relatedParam && /^\d+$/.test(relatedParam) ? Number(relatedParam) : null;

  // Seen-state (unseen-update glow). Read client-side after mount — the
  // server render shows no glow, then the first effect pass lights up
  // genuinely-unseen cards (avoids a hydration mismatch on localStorage).
  const [seenState, setSeenState] = useState<SeenState | null>(null);
  useEffect(() => {
    const load = () => setSeenState(readSeenState());
    load();
    // Re-read on focus / bfcache restore / cross-tab writes so coming back
    // from a task detail page clears its glow without a manual reload.
    window.addEventListener("focus", load);
    window.addEventListener("pageshow", load);
    window.addEventListener("storage", load);
    document.addEventListener("visibilitychange", load);
    return () => {
      window.removeEventListener("focus", load);
      window.removeEventListener("pageshow", load);
      window.removeEventListener("storage", load);
      document.removeEventListener("visibilitychange", load);
    };
  }, []);

  // Family map over ALL tasks (not track-filtered — relations cross lanes).
  const familyOf = useMemo(() => buildFamilyMap(tasks), [tasks]);

  const updateParams = useCallback(
    (patch: Record<string, string | null>, opts?: { push?: boolean }) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(patch)) {
        if (v === null || v === "") next.delete(k);
        else next.set(k, v);
      }
      const qs = next.toString();
      const url = qs ? `${pathname}?${qs}` : pathname;
      // Toggles (view/track) replace; drill-downs (family filter) push, so
      // the browser Back button exits the filter naturally.
      if (opts?.push) router.push(url, { scroll: false });
      else router.replace(url, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  const cardCtx: CardCtx = useMemo(
    () => ({
      isUnseen: (t: TaskListing) => {
        if (!seenState || !t.lastActivityAt) return false;
        const seenAt = seenState.seen[String(t.id)] ?? seenState.baseline;
        if (!seenAt) return false;
        const activityMs = Date.parse(t.lastActivityAt);
        const seenMs = Date.parse(seenAt);
        return (
          Number.isFinite(activityMs) && Number.isFinite(seenMs) && activityMs > seenMs
        );
      },
      markSeen: (t: TaskListing) => {
        markTaskSeen(t.id, t.lastActivityAt);
        setSeenState(readSeenState());
      },
      relativesOf: (id: number) => (familyOf.get(id)?.length ?? 1) - 1,
      showRelated: (id: number) =>
        updateParams({ related: String(id) }, { push: true }),
    }),
    [seenState, familyOf, updateParams],
  );

  const unseenCount = useMemo(
    () => tasks.reduce((n, t) => (cardCtx.isUnseen(t) ? n + 1 : n), 0),
    [tasks, cardCtx],
  );

  const markAllSeen = useCallback(() => {
    markAllTasksSeen();
    setSeenState(readSeenState());
  }, []);

  // Family filter (when active) overrides the track filter — a family can
  // span both lanes and hiding half of it would defeat the point.
  const relatedFamily = useMemo(() => {
    if (relatedId === null) return null;
    const members = familyOf.get(relatedId);
    return members && members.length > 0 ? new Set(members) : null;
  }, [relatedId, familyOf]);

  // Track-filtered tasks (used by both views).
  const filtered = useMemo(() => {
    if (relatedFamily) return tasks.filter((t) => relatedFamily.has(t.id));
    if (track === "all") return tasks;
    return tasks.filter((t) => t.track === track);
  }, [tasks, track, relatedFamily]);

  const byStatus = useMemo(() => groupByStatus(filtered), [filtered]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-stone-200 bg-white p-3">
        {/* Category tabs */}
        <div
          className="flex items-center gap-1 rounded-md bg-stone-100 p-0.5"
          role="tablist"
          aria-label="Track filter"
        >
          {TRACK_TABS.map((t) => {
            const active = track === t.key;
            return (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => updateParams({ track: t.key === "all" ? null : t.key })}
                className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
                  active
                    ? "bg-white text-stone-900 shadow-sm"
                    : "text-stone-500 hover:text-stone-800"
                }`}
              >
                {t.label}
                <span className="ml-1.5 text-xs font-normal text-stone-400">
                  {t.key === "all"
                    ? tasks.length
                    : tasks.filter((x) => x.track === t.key).length}
                </span>
              </button>
            );
          })}
        </div>

        <div className="h-5 w-px bg-stone-200" />

        {/* View toggle */}
        <div
          className="flex items-center gap-1 rounded-md bg-stone-100 p-0.5"
          role="tablist"
          aria-label="View mode"
        >
          {(["kanban", "list"] as ViewMode[]).map((v) => {
            const active = view === v;
            return (
              <button
                key={v}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => updateParams({ view: v })}
                className={`rounded px-3 py-1 text-sm font-medium capitalize transition-colors ${
                  active
                    ? "bg-white text-stone-900 shadow-sm"
                    : "text-stone-500 hover:text-stone-800"
                }`}
              >
                {v}
              </button>
            );
          })}
        </div>

        {view === "kanban" && (
          <label className="ml-auto flex items-center gap-1.5 text-xs text-stone-500">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-stone-300"
            />
            Show archived
          </label>
        )}

        <span
          className={`text-xs text-stone-500 ${view === "kanban" ? "" : "ml-auto"}`}
        >
          {filtered.length} task{filtered.length === 1 ? "" : "s"}
          {filtered.length !== tasks.length && (
            <span className="text-stone-400"> (of {tasks.length})</span>
          )}
        </span>

        {unseenCount > 0 && (
          <button
            type="button"
            onClick={markAllSeen}
            title="Clear the unseen-update glow on every task (escape hatch for mass updates)"
            className="rounded bg-amber-50 px-2 py-1 text-xs font-medium text-amber-800 transition-colors hover:bg-amber-100"
          >
            Mark all seen ({unseenCount})
          </button>
        )}
      </div>

      {relatedFamily && relatedId !== null && (
        <div className="flex items-center gap-3 rounded-lg border border-sky-200 bg-sky-50 px-4 py-2 text-sm text-sky-900">
          <span>
            Related to <span className="font-mono font-medium">#{relatedId}</span> —{" "}
            {filtered.length} task{filtered.length === 1 ? "" : "s"} in this family
            (parent / children / siblings)
          </span>
          <button
            type="button"
            onClick={() => updateParams({ related: null })}
            className="ml-auto rounded px-2 py-0.5 text-xs font-medium text-sky-700 hover:bg-sky-100"
          >
            ✕ Clear
          </button>
        </div>
      )}

      {view === "kanban" ? (
        <KanbanBoard
          byStatus={byStatus}
          progress={progress}
          // A family view shows the WHOLE family — unhide archived members
          // so the banner count matches the visible cards.
          showArchived={showArchived || relatedFamily !== null}
          ctx={cardCtx}
        />
      ) : (
        <ListView byStatus={byStatus} progress={progress} ctx={cardCtx} />
      )}
    </div>
  );
}

/**
 * Connected components over frontmatter parent_id edges: id → sorted member
 * ids of its family (self included; singletons map to [self]). Edges to ids
 * missing from the listing (e.g. deleted parents) are ignored.
 */
function buildFamilyMap(tasks: TaskListing[]): Map<number, number[]> {
  const ids = new Set(tasks.map((t) => t.id));
  const adj = new Map<number, number[]>();
  const link = (a: number, b: number) => {
    const cur = adj.get(a);
    if (cur) cur.push(b);
    else adj.set(a, [b]);
  };
  for (const t of tasks) {
    if (t.parentId !== null && ids.has(t.parentId)) {
      link(t.id, t.parentId);
      link(t.parentId, t.id);
    }
  }
  const familyOf = new Map<number, number[]>();
  const visited = new Set<number>();
  for (const t of tasks) {
    if (visited.has(t.id)) continue;
    visited.add(t.id);
    const members: number[] = [];
    const queue = [t.id];
    while (queue.length > 0) {
      const cur = queue.pop()!;
      members.push(cur);
      for (const nb of adj.get(cur) ?? []) {
        if (!visited.has(nb)) {
          visited.add(nb);
          queue.push(nb);
        }
      }
    }
    members.sort((a, b) => a - b);
    for (const m of members) familyOf.set(m, members);
  }
  return familyOf;
}

// Epoch ms the task entered its current status; 0 (sorts to the bottom)
// when unknown/unparseable.
function statusEntryMs(t: TaskListing): number {
  if (!t.statusChangedAt) return 0;
  const ms = Date.parse(t.statusChangedAt);
  return Number.isFinite(ms) ? ms : 0;
}

function groupByStatus(tasks: TaskListing[]): Record<Status, TaskListing[]> {
  const out = {} as Record<Status, TaskListing[]>;
  for (const status of STATUS_DISPLAY_ORDER) out[status] = [];
  for (const t of tasks) {
    if (!out[t.status]) out[t.status] = [];
    out[t.status].push(t);
  }
  // Within each column: most recently moved-in first, tie-break id desc.
  for (const rows of Object.values(out)) {
    rows.sort((a, b) => statusEntryMs(b) - statusEntryMs(a) || b.id - a.id);
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * Kanban — one column per lifecycle status, horizontal scroll on narrow.
 * -------------------------------------------------------------------------- */

function KanbanBoard({
  byStatus,
  progress,
  showArchived,
  ctx,
}: {
  byStatus: Record<Status, TaskListing[]>;
  progress: Record<number, TaskProgressView>;
  showArchived: boolean;
  ctx: CardCtx;
}) {
  const columns = KANBAN_COLUMN_ORDER.filter(
    (status) => status !== "archived" || showArchived,
  );
  return (
    <div className="-mx-1 overflow-x-auto pb-2">
      <div className="flex gap-3 px-1">
        {columns.map((status) => (
          <KanbanColumn
            key={status}
            status={status}
            rows={byStatus[status] ?? []}
            progress={progress}
            ctx={ctx}
          />
        ))}
      </div>
    </div>
  );
}

function KanbanColumn({
  status,
  rows,
  progress,
  ctx,
}: {
  status: Status;
  rows: TaskListing[];
  progress: Record<number, TaskProgressView>;
  ctx: CardCtx;
}) {
  const empty = rows.length === 0;
  return (
    <section
      className={`flex w-72 shrink-0 flex-col rounded-lg border ${
        empty ? "border-stone-100 bg-stone-50/50" : "border-stone-200 bg-stone-50"
      }`}
    >
      <header className="flex items-center justify-between gap-2 border-b border-stone-200/70 px-3 py-2">
        <span
          className={`text-sm font-medium tracking-tight ${
            empty ? "text-stone-400" : "text-stone-800"
          }`}
        >
          {STATUS_LABELS[status]}
        </span>
        <span
          className={`rounded-full px-2 py-0.5 text-xs ${
            empty ? "text-stone-300" : "bg-stone-200 text-stone-600"
          }`}
        >
          {rows.length}
        </span>
      </header>
      <div className="flex flex-col gap-2 p-2">
        {empty ? (
          <p className="px-1 py-3 text-center text-xs text-stone-300">No tasks</p>
        ) : (
          rows.map((row) => (
            <KanbanCard
              key={row.id}
              row={row}
              progressView={progress[row.id]}
              ctx={ctx}
            />
          ))
        )}
      </div>
    </section>
  );
}

function KanbanCard({
  row,
  progressView,
  ctx,
}: {
  row: TaskListing;
  progressView?: TaskProgressView;
  ctx: CardCtx;
}) {
  const unseen = ctx.isUnseen(row);
  return (
    <Link
      href={`/tasks/${row.id}`}
      onClick={() => ctx.markSeen(row)}
      className={`block rounded-md border bg-white px-3 py-2 transition-colors hover:bg-stone-50 ${
        unseen
          ? "unseen-glow border-amber-300 hover:border-amber-400"
          : "border-stone-200 hover:border-stone-300"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-mono text-xs text-stone-500">
          {unseen && <UnseenDot />}#{row.id}
        </span>
        <TrackBadge track={row.track} />
      </div>
      <p className="mt-1 line-clamp-2 text-sm leading-snug text-stone-900">
        {row.title || <em className="text-stone-400">(untitled)</em>}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <KindBadge kind={row.kind} />
        {row.hasCleanResult && <CleanResultBadge classification={row.classification} />}
        <FollowupCountBadge count={row.followupCount} />
        {row.status === "followups_running" && <FollowupModeBadge tags={row.tags} />}
        <RelatedButton id={row.id} ctx={ctx} />
      </div>
      {progressView && <TaskProgressBar view={progressView} compact />}
    </Link>
  );
}

/** Pulsing marker next to the id of an unseen-update card. */
function UnseenDot() {
  return (
    <span
      className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400"
      title="Updated since you last looked at this task"
    />
  );
}

/**
 * "N related" — filters the board to this task's family. A <span
 * role=button> rather than <button>: cards/rows are <Link>s and nested
 * interactive elements are invalid HTML, so we stop the navigation instead.
 */
function RelatedButton({ id, ctx }: { id: number; ctx: CardCtx }) {
  const n = ctx.relativesOf(id);
  if (n <= 0) return null;
  return (
    <span
      role="button"
      tabIndex={0}
      title="Show this task together with its parent / children / siblings"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        ctx.showRelated(id);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          ctx.showRelated(id);
        }
      }}
      className="rounded bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-600 transition-colors hover:bg-sky-100 hover:text-sky-800"
    >
      ⛓ {n} related
    </span>
  );
}

// How many follow-up rounds have run on this task (distinct followup_labels
// + free-analysis auto-runs, derived from events.jsonl in lib/tasks).
function FollowupCountBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <span className="rounded bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-700">
      {count} follow-up{count === 1 ? "" : "s"}
    </span>
  );
}

// In the "Follow-ups running" column, show whether the in-flight follow-up
// round was initiated automatically (proposer) or manually (user pick/chat).
// The most recent initiation mode wins when both tags are present.
function FollowupModeBadge({ tags }: { tags: string[] }) {
  const auto = tags.includes("followup-auto");
  const manual = tags.includes("followup-manual");
  if (!auto && !manual) return null;
  const label = auto && !manual ? "auto" : manual && !auto ? "manual" : "auto+manual";
  return (
    <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[10px] font-medium text-sky-800">
      {label} follow-up
    </span>
  );
}

/* -------------------------------------------------------------------------- *
 * List view — the original grouped-by-status accordion, track-filtered.
 * -------------------------------------------------------------------------- */

// Statuses expanded by default in list view (mirrors the previous page).
const LIST_EXPANDED: ReadonlySet<Status> = new Set([
  "running",
  "awaiting_promotion",
  "proposed",
]);

function ListView({
  byStatus,
  progress,
  ctx,
}: {
  byStatus: Record<Status, TaskListing[]>;
  progress: Record<number, TaskProgressView>;
  ctx: CardCtx;
}) {
  const sections = STATUS_DISPLAY_ORDER.filter(
    (status) => (byStatus[status] ?? []).length > 0,
  );
  if (sections.length === 0) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-8 text-center text-sm text-stone-500">
        No tasks match the current filter.
      </p>
    );
  }
  return (
    <div className="space-y-3">
      {sections.map((status) => (
        <StatusSection
          key={status}
          status={status}
          rows={byStatus[status]}
          progress={progress}
          ctx={ctx}
          defaultOpen={LIST_EXPANDED.has(status)}
        />
      ))}
    </div>
  );
}

function StatusSection({
  status,
  rows,
  progress,
  ctx,
  defaultOpen,
}: {
  status: Status;
  rows: TaskListing[];
  progress: Record<number, TaskProgressView>;
  ctx: CardCtx;
  defaultOpen: boolean;
}) {
  return (
    <details
      open={defaultOpen}
      className="overflow-hidden rounded-lg border border-stone-200 bg-white"
    >
      <summary className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3 hover:bg-stone-50 sm:px-5">
        <div className="flex items-center gap-3">
          <span className="font-medium tracking-tight">{STATUS_LABELS[status]}</span>
          <span className="rounded-full bg-stone-100 px-2 py-0.5 text-xs text-stone-600">
            {rows.length}
          </span>
        </div>
        <span className="text-xs uppercase tracking-wide text-stone-400">{status}</span>
      </summary>
      <ul className="divide-y divide-stone-100 border-t border-stone-100">
        {rows.map((row) => {
          const unseen = ctx.isUnseen(row);
          return (
            <li key={row.id} className={unseen ? "unseen-glow-inset" : undefined}>
              <Link
                href={`/tasks/${row.id}`}
                onClick={() => ctx.markSeen(row)}
                className={`flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-center sm:gap-4 sm:px-5 ${
                  unseen ? "bg-amber-50/40 hover:bg-amber-50" : "hover:bg-stone-50"
                }`}
              >
                <span className="flex items-center gap-1.5 font-mono text-sm text-stone-500 sm:w-14">
                  {unseen && <UnseenDot />}#{row.id}
                </span>
                <span className="flex-1 text-sm leading-snug text-stone-900">
                  {row.title || <em className="text-stone-400">(untitled)</em>}
                </span>
                <span className="flex flex-wrap items-center gap-2">
                  <TrackBadge track={row.track} />
                  <KindBadge kind={row.kind} />
                  {row.hasCleanResult && (
                    <CleanResultBadge classification={row.classification} />
                  )}
                  <FollowupCountBadge count={row.followupCount} />
                  <RelatedButton id={row.id} ctx={ctx} />
                </span>
                {progress[row.id] && (
                  <TaskProgressBar
                    view={progress[row.id]}
                    compact
                    className="w-full sm:w-44 sm:shrink-0"
                  />
                )}
              </Link>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

/* -------------------------------------------------------------------------- *
 * Badges.
 * -------------------------------------------------------------------------- */

function TrackBadge({ track }: { track: Track }) {
  // Visually distinct from the kind badge: track uses a filled, slightly
  // bolder pill (teal for experiment, fuchsia for human) so the two lanes
  // read apart at a glance.
  const cls =
    track === "human"
      ? "bg-fuchsia-100 text-fuchsia-800"
      : "bg-teal-100 text-teal-800";
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${cls}`}>
      {track === "human" ? "Human" : "Experiment"}
    </span>
  );
}

function KindBadge({ kind }: { kind: string }) {
  const cls =
    kind === "experiment"
      ? "bg-blue-50 text-blue-700"
      : kind === "infra"
        ? "bg-amber-50 text-amber-800"
        : kind === "analysis"
          ? "bg-violet-50 text-violet-700"
          : "bg-stone-100 text-stone-700";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>{kind}</span>;
}

function CleanResultBadge({ classification }: { classification?: string }) {
  const label =
    classification === "useful"
      ? "useful"
      : classification === "not-useful"
        ? "not useful"
        : "clean result";
  const cls =
    classification === "useful"
      ? "bg-emerald-50 text-emerald-700"
      : classification === "not-useful"
        ? "bg-rose-50 text-rose-700"
        : "bg-stone-100 text-stone-700";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>{label}</span>;
}
