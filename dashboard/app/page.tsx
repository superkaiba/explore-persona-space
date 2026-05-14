import Link from "next/link";
import { tasksByStatus, type TaskListing } from "@/lib/tasks";
import {
  STATUS_DISPLAY_ORDER,
  STATUS_EXPANDED_BY_DEFAULT,
  STATUS_LABELS,
  type Status,
} from "@/lib/repo";

export const dynamic = "force-dynamic";

export default async function Home() {
  const grouped = tasksByStatus();
  const totalTasks = Object.values(grouped).reduce((n, rows) => n + rows.length, 0);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Tasks</h1>
        <p className="mt-1 text-sm text-stone-600">
          {totalTasks} task{totalTasks === 1 ? "" : "s"} across {STATUS_DISPLAY_ORDER.length} statuses.
          Folder = status. Single writer (the VM); the web is for viewing.
        </p>
      </header>

      <div className="space-y-3">
        {STATUS_DISPLAY_ORDER.map((status) => {
          const rows = grouped[status] ?? [];
          if (rows.length === 0) return null;
          const expanded = STATUS_EXPANDED_BY_DEFAULT.includes(status);
          return (
            <StatusSection
              key={status}
              status={status}
              rows={rows}
              defaultOpen={expanded}
            />
          );
        })}
      </div>
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
              <span className="font-mono text-sm text-stone-500 sm:w-14">
                #{row.id}
              </span>
              <span className="flex-1 text-sm leading-snug text-stone-900">
                {row.title || <em className="text-stone-400">(untitled)</em>}
              </span>
              <span className="flex flex-wrap items-center gap-2">
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
