import Link from "next/link";
import { tasksByStatus, type TaskListing } from "@/lib/tasks";
import { type Status } from "@/lib/repo";

export const dynamic = "force-dynamic";

type ColumnDef = {
  id: string;
  label: string;
  statuses: Status[];
  filter?: (t: TaskListing) => boolean;
  defaultOpen?: boolean;
  showSubState?: boolean;
};

type GroupDef = {
  label: string;
  columns: ColumnDef[];
};

const PRIORITY_TAG = "priority";
const PRIORITY_ELIGIBLE_STATUSES: Status[] = ["proposed", "approved", "blocked"];

const hasPriority = (t: TaskListing) => t.tags.includes(PRIORITY_TAG);
const notPriority = (t: TaskListing) => !hasPriority(t);

const GROUPS: GroupDef[] = [
  {
    label: "Needs you",
    columns: [
      {
        id: "priority",
        label: "★ Priority",
        statuses: PRIORITY_ELIGIBLE_STATUSES,
        filter: hasPriority,
        defaultOpen: true,
      },
      {
        id: "awaiting_promotion",
        label: "Awaiting promotion",
        statuses: ["awaiting_promotion"],
        defaultOpen: true,
      },
      {
        id: "plan_pending",
        label: "Awaiting plan review",
        statuses: ["plan_pending"],
        defaultOpen: true,
      },
    ],
  },
  {
    label: "In flight",
    columns: [
      {
        id: "running",
        label: "Running",
        statuses: ["running", "verifying", "interpreting", "reviewing"],
        defaultOpen: true,
        showSubState: true,
      },
      {
        id: "planning",
        label: "Planning",
        statuses: ["planning"],
      },
    ],
  },
  {
    label: "Backlog",
    columns: [
      {
        id: "approved",
        label: "Approved",
        statuses: ["approved"],
        filter: notPriority,
      },
      {
        id: "proposed",
        label: "Proposed",
        statuses: ["proposed"],
        filter: notPriority,
      },
      {
        id: "on_hold",
        label: "On hold",
        statuses: ["blocked"],
        filter: notPriority,
      },
    ],
  },
  {
    label: "Done",
    columns: [
      {
        id: "clean_results",
        label: "Clean results",
        statuses: ["completed"],
        filter: (t) =>
          t.hasCleanResult &&
          (t.classification === "useful" || t.classification === "not-useful"),
      },
      {
        id: "completed_other",
        label: "Completed (other)",
        statuses: ["completed"],
        filter: (t) =>
          !t.hasCleanResult ||
          (t.classification !== "useful" && t.classification !== "not-useful"),
      },
      {
        id: "archived",
        label: "Archived",
        statuses: ["archived"],
      },
    ],
  },
];

function collect(
  grouped: Record<Status, TaskListing[]>,
  col: ColumnDef,
): TaskListing[] {
  const rows: TaskListing[] = [];
  for (const s of col.statuses) {
    for (const row of grouped[s] ?? []) {
      if (col.filter && !col.filter(row)) continue;
      rows.push(row);
    }
  }
  rows.sort((a, b) => b.id - a.id);
  return rows;
}

export default async function PreviewHome() {
  const grouped = tasksByStatus();
  const totalTasks = Object.values(grouped).reduce((n, rows) => n + rows.length, 0);

  return (
    <div className="space-y-8">
      <header>
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Tasks</h1>
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
            preview
          </span>
        </div>
        <p className="mt-1 text-sm text-stone-600">
          {totalTasks} tasks · {GROUPS.reduce((n, g) => n + g.columns.length, 0)} columns ·
          grouped by intent. Canonical statuses unchanged on disk.
        </p>
        <p className="mt-1 text-xs text-stone-500">
          ★ = `priority` tag. Set with{" "}
          <code className="rounded bg-stone-100 px-1.5 py-0.5">
            uv run python scripts/task.py add-tag &lt;N&gt; priority
          </code>
          {" "}/{" "}
          <code className="rounded bg-stone-100 px-1.5 py-0.5">remove-tag</code>.
        </p>
      </header>

      {GROUPS.map((group) => (
        <section key={group.label} className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
            {group.label}
          </h2>
          <div className="space-y-3">
            {group.columns.map((col) => {
              const rows = collect(grouped, col);
              return (
                <ColumnSection
                  key={col.id}
                  col={col}
                  rows={rows}
                  defaultOpen={col.defaultOpen ?? rows.length > 0}
                />
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

function ColumnSection({
  col,
  rows,
  defaultOpen,
}: {
  col: ColumnDef;
  rows: TaskListing[];
  defaultOpen: boolean;
}) {
  const empty = rows.length === 0;
  return (
    <details
      open={!empty && defaultOpen}
      className="overflow-hidden rounded-lg border border-stone-200 bg-white"
    >
      <summary className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3 hover:bg-stone-50 sm:px-5">
        <div className="flex items-center gap-3">
          <span className="font-medium tracking-tight">{col.label}</span>
          <span
            className={`rounded-full px-2 py-0.5 text-xs ${
              empty ? "bg-stone-50 text-stone-400" : "bg-stone-100 text-stone-600"
            }`}
          >
            {rows.length}
          </span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wide text-stone-400">
          {col.statuses.join(" + ")}
        </span>
      </summary>
      {empty ? (
        <div className="border-t border-stone-100 px-5 py-4 text-xs italic text-stone-400">
          empty
        </div>
      ) : (
        <ul className="divide-y divide-stone-100 border-t border-stone-100">
          {rows.map((row) => (
            <li key={row.id}>
              <Row row={row} showSubState={col.showSubState ?? false} />
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

function Row({
  row,
  showSubState,
}: {
  row: TaskListing;
  showSubState: boolean;
}) {
  const isPriority = row.tags.includes(PRIORITY_TAG);
  const subState =
    showSubState && row.status !== "running" ? row.status : null;
  return (
    <Link
      href={`/tasks/${row.id}`}
      className="flex flex-col gap-1 px-4 py-3 hover:bg-stone-50 sm:flex-row sm:items-center sm:gap-3 sm:px-5"
    >
      <span className="flex items-center gap-1.5 sm:w-20">
        <span className="font-mono text-sm text-stone-500">#{row.id}</span>
        {isPriority && (
          <span title="priority" className="text-amber-500">
            ★
          </span>
        )}
      </span>
      <span className="flex-1 text-sm leading-snug text-stone-900">
        {row.title || <em className="text-stone-400">(untitled)</em>}
      </span>
      <span className="flex flex-wrap items-center gap-2">
        <KindBadge kind={row.kind} />
        {subState && <SubStateBadge state={subState} />}
        {row.hasCleanResult && (
          <CleanResultBadge classification={row.classification} />
        )}
      </span>
    </Link>
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
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>
      {kind}
    </span>
  );
}

function SubStateBadge({ state }: { state: string }) {
  return (
    <span className="rounded bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-700">
      {state}
    </span>
  );
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
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}
