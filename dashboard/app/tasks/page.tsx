/**
 * /tasks — the task board. Server component loads every task once (with
 * its derived `track`) and hands the flat array to the client <TaskBoard>
 * shell, which owns the URL-synced view (List | Kanban) + category-tab
 * (Experiments | Human | All) state.
 *
 * Read-mostly viewer: status is workflow-owned (the VM is the single
 * writer), so the board has no drag-and-drop — cards link to the task
 * detail page.
 */
import { Suspense } from "react";
import { listAllTasks } from "@/lib/tasks";
import { getProgressMap } from "@/lib/progress";
import { STATUS_DISPLAY_ORDER } from "@/lib/repo";
import { TaskBoard } from "./TaskBoard";

export const dynamic = "force-dynamic";

type SearchParams = {
  view?: string;
  track?: string;
};

export default async function Tasks({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const sp = await searchParams;
  const initialView = sp.view === "list" ? "list" : "kanban";
  const initialTrack =
    sp.track === "experiment" || sp.track === "human" ? sp.track : "all";

  const tasks = listAllTasks();
  // Pipeline progress for in-flight tasks (task #587): live-status-keyed —
  // the snapshot reader drops rows whose LIVE status has no stage floor, so
  // `progress[id]` existing implies the bar is valid to render.
  const statusById: Record<number, string> = {};
  for (const t of tasks) statusById[t.id] = t.status;
  const progress = getProgressMap(statusById);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Tasks</h1>
        <p className="mt-1 text-sm text-stone-600">
          {tasks.length} task{tasks.length === 1 ? "" : "s"} across{" "}
          {STATUS_DISPLAY_ORDER.length} statuses. Folder = status. Single writer
          (the VM); the web is for viewing.
        </p>
      </header>

      <Suspense fallback={<div className="text-sm text-stone-500">Loading board…</div>}>
        <TaskBoard
          tasks={tasks}
          progress={progress}
          initialView={initialView}
          initialTrack={initialTrack}
        />
      </Suspense>
    </div>
  );
}
