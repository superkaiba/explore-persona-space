"use client";

/**
 * Client shell for the /tasks page. Owns two pieces of URL-synced view
 * state so the view is shareable (same pattern as /updates UpdatesFeed):
 *
 *   - `?view=list|kanban`           (default kanban)
 *   - `?track=experiment|human|all` (default all)
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
import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { TaskListing, Track } from "@/lib/tasks";
import { STATUS_DISPLAY_ORDER, STATUS_LABELS, type Status } from "@/lib/repo";

type ViewMode = "list" | "kanban";
type TrackFilter = "experiment" | "human" | "all";

// Kanban column order: blocked sits near the front (it's active-but-stuck
// work the user wants to see early), then the canonical lifecycle order
// from proposed → completed, with archived last + hidden by default.
const KANBAN_COLUMN_ORDER: Status[] = [
  "blocked",
  "proposed",
  "planning",
  "plan_pending",
  "approved",
  "running",
  "verifying",
  "interpreting",
  "reviewing",
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
  initialView,
  initialTrack,
}: {
  tasks: TaskListing[];
  initialView: ViewMode;
  initialTrack: TrackFilter;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [showArchived, setShowArchived] = useState(false);

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

  const updateParams = useCallback(
    (patch: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(patch)) {
        if (v === null || v === "") next.delete(k);
        else next.set(k, v);
      }
      const qs = next.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  // Track-filtered tasks (used by both views).
  const filtered = useMemo(() => {
    if (track === "all") return tasks;
    return tasks.filter((t) => t.track === track);
  }, [tasks, track]);

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
      </div>

      {view === "kanban" ? (
        <KanbanBoard byStatus={byStatus} showArchived={showArchived} />
      ) : (
        <ListView byStatus={byStatus} />
      )}
    </div>
  );
}

function groupByStatus(tasks: TaskListing[]): Record<Status, TaskListing[]> {
  const out = {} as Record<Status, TaskListing[]>;
  for (const status of STATUS_DISPLAY_ORDER) out[status] = [];
  for (const t of tasks) {
    if (!out[t.status]) out[t.status] = [];
    out[t.status].push(t);
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * Kanban — one column per lifecycle status, horizontal scroll on narrow.
 * -------------------------------------------------------------------------- */

function KanbanBoard({
  byStatus,
  showArchived,
}: {
  byStatus: Record<Status, TaskListing[]>;
  showArchived: boolean;
}) {
  const columns = KANBAN_COLUMN_ORDER.filter(
    (status) => status !== "archived" || showArchived,
  );
  return (
    <div className="-mx-1 overflow-x-auto pb-2">
      <div className="flex gap-3 px-1">
        {columns.map((status) => (
          <KanbanColumn key={status} status={status} rows={byStatus[status] ?? []} />
        ))}
      </div>
    </div>
  );
}

function KanbanColumn({ status, rows }: { status: Status; rows: TaskListing[] }) {
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
          rows.map((row) => <KanbanCard key={row.id} row={row} />)
        )}
      </div>
    </section>
  );
}

function KanbanCard({ row }: { row: TaskListing }) {
  return (
    <Link
      href={`/tasks/${row.id}`}
      className="block rounded-md border border-stone-200 bg-white px-3 py-2 transition-colors hover:border-stone-300 hover:bg-stone-50"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-stone-500">#{row.id}</span>
        <TrackBadge track={row.track} />
      </div>
      <p className="mt-1 line-clamp-2 text-sm leading-snug text-stone-900">
        {row.title || <em className="text-stone-400">(untitled)</em>}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <KindBadge kind={row.kind} />
        {row.hasCleanResult && <CleanResultBadge classification={row.classification} />}
      </div>
    </Link>
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

function ListView({ byStatus }: { byStatus: Record<Status, TaskListing[]> }) {
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
          defaultOpen={LIST_EXPANDED.has(status)}
        />
      ))}
    </div>
  );
}

function StatusSection({
  status,
  rows,
  defaultOpen,
}: {
  status: Status;
  rows: TaskListing[];
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
        {rows.map((row) => (
          <li key={row.id}>
            <Link
              href={`/tasks/${row.id}`}
              className="flex flex-col gap-1 px-4 py-3 hover:bg-stone-50 sm:flex-row sm:items-center sm:gap-4 sm:px-5"
            >
              <span className="font-mono text-sm text-stone-500 sm:w-14">#{row.id}</span>
              <span className="flex-1 text-sm leading-snug text-stone-900">
                {row.title || <em className="text-stone-400">(untitled)</em>}
              </span>
              <span className="flex flex-wrap items-center gap-2">
                <TrackBadge track={row.track} />
                <KindBadge kind={row.kind} />
                {row.hasCleanResult && <CleanResultBadge classification={row.classification} />}
              </span>
            </Link>
          </li>
        ))}
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
