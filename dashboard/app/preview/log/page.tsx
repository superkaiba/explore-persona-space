/**
 * /preview/log — mentor-facing clean-results log.
 *
 * Lists every task with `has_clean_result=true`, newest first, with a
 * client-side search box (substring match on title + body). No daily /
 * weekly / ideation entries — clean results only.
 *
 * The same underlying data is queryable from the terminal via
 *   uv run python scripts/task.py list-clean-results [--search "..."] [--json]
 */
import Link from "next/link";
import { listCleanResults, type CleanResult } from "@/lib/logs";
import { CleanResultsSearch } from "./CleanResultsSearch";

export const dynamic = "force-dynamic";

export default async function PreviewLogPage() {
  const all = await listCleanResults({ includeNotUseful: true });
  // Strict "promoted" filter: classification ∈ {useful, not-useful}.
  // `has_clean_result=true` alone means the analyzer staged a body but
  // the user hasn't run `task.py promote` yet — those still live under
  // `awaiting_promotion`, not in the mentor-facing log.
  const rows = all.filter(
    (r) => r.classification === "useful" || r.classification === "not-useful",
  );
  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Clean results
          </h1>
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
            preview
          </span>
        </div>
        <p className="mt-1 text-sm text-stone-600">
          {rows.length} clean result{rows.length === 1 ? "" : "s"} ·
          promoted from <code>awaiting_promotion</code> via{" "}
          <code className="rounded bg-stone-100 px-1.5 py-0.5">
            task.py promote &lt;N&gt; useful|not-useful
          </code>
        </p>
        <p className="mt-1 text-xs text-stone-500">
          From the terminal:{" "}
          <code className="rounded bg-stone-100 px-1.5 py-0.5">
            uv run python scripts/task.py list-clean-results --search &quot;...&quot;
          </code>
        </p>
      </header>

      <CleanResultsSearch rows={rows.map(serializeRow)} />
    </div>
  );
}

type SerializedRow = {
  taskId: number;
  title: string;
  date: string;
  classification: CleanResult["classification"];
  body: string;
};

function serializeRow(r: CleanResult): SerializedRow {
  return {
    taskId: r.taskId,
    title: r.title,
    date: r.date,
    classification: r.classification,
    body: r.body,
  };
}
