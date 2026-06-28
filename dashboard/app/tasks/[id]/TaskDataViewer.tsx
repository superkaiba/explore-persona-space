"use client";

/**
 * <TaskDataViewer> — the interactive data viewer for a task's figure /
 * eval-results data (clean-result v4 redesign, Phase 2 — the "Dashboard
 * data-artifact interface" contract in .claude/skills/clean-results/SPEC.md).
 *
 * Mounted on /tasks/[id] below the body card (sanitize-safe: a real React
 * component, NOT markup injected into the sanitized body). It:
 *
 *   1. Lazily fetches the task's data index from GET /tasks/<id>/data on first
 *      expand (the route reads figures/issue_<N>/*.meta.json + any committed
 *      data_path target through lib/task-data.ts).
 *   2. Lets the reader pick among the task's data artifacts (one per figure
 *      whose sidecar carries — directly or via data_path — a row table).
 *   3. Renders the selected artifact in <DataTable> with sort / filter /
 *      search / reveal-more.
 *   4. Surfaces the data's provenance: the on-disk source file + the SHA-pinned
 *      figure link-out, and an explicit notice when the full set lives behind
 *      external links the dashboard can't fetch (truncation) or when a figure's
 *      sidecar carries no per-row data at all.
 *
 * Renders nothing when the task has no figure data at all (keeps the page clean
 * for non-experiment / data-less tasks). Lazy fetch keeps the task page's
 * initial render cost unchanged.
 */
import { useCallback, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Database,
  ExternalLink,
  Loader2,
  Table as TableIcon,
} from "lucide-react";
import type { DataArtifact, TaskDataIndex } from "@/lib/task-data";
import { DataTable } from "@/components/tasks/DataTable";

type FetchState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; index: TaskDataIndex };

const GITHUB_BLOB_BASE = "https://github.com/superkaiba/explore-persona-space/blob/main/";

export function TaskDataViewer({ taskId }: { taskId: number }) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<FetchState>({ phase: "idle" });
  const [selected, setSelected] = useState<string | null>(null);
  // Guards a double-fetch in React StrictMode dev double-mount.
  const fetchedRef = useRef(false);

  const load = useCallback(async () => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;
    setState({ phase: "loading" });
    try {
      const res = await fetch(`/tasks/${taskId}/data`, { credentials: "same-origin" });
      if (!res.ok) {
        setState({ phase: "error", message: `Server returned ${res.status}.` });
        return;
      }
      const json = (await res.json()) as
        | ({ ok: true } & TaskDataIndex)
        | { ok: false; error: string };
      if (!json.ok) {
        setState({ phase: "error", message: json.error });
        return;
      }
      const index: TaskDataIndex = { taskId: json.taskId, artifacts: json.artifacts };
      setState({ phase: "ready", index });
      const firstWithRows = index.artifacts.find((a) => a.rows.length > 0);
      setSelected((firstWithRows ?? index.artifacts[0])?.id ?? null);
    } catch (e) {
      setState({ phase: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, [taskId]);

  // Toggle the panel; fetch lazily on the FIRST expand (from the event
  // handler, not an effect — no cascading-render setState-in-effect).
  const onToggle = useCallback(() => {
    setOpen((wasOpen) => {
      const next = !wasOpen;
      if (next && state.phase === "idle") void load();
      return next;
    });
  }, [load, state.phase]);

  // Hide the whole panel once we know there's no figure data for this task.
  // (Until the first fetch we optimistically show the collapsed header; after a
  // ready-but-empty result we collapse the component to nothing.)
  if (state.phase === "ready" && state.index.artifacts.length === 0) {
    return null;
  }

  const artifacts = state.phase === "ready" ? state.index.artifacts : [];
  const active = artifacts.find((a) => a.id === selected) ?? null;
  const withRows = artifacts.filter((a) => a.rows.length > 0);

  return (
    <section
      id="feed-data-viewer"
      className="scroll-mt-4 overflow-hidden rounded-lg border border-stone-200 bg-white"
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-baseline gap-2 px-4 py-2 text-left text-xs text-stone-500 hover:bg-stone-50 sm:px-6"
      >
        <span className="text-stone-400" aria-hidden>
          {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        </span>
        <span className="flex flex-1 flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-1.5 font-mono text-stone-700">
            <Database className="h-3.5 w-3.5" />
            Data viewer
          </span>
          <span className="text-stone-500">
            sort · filter · search the figure data
          </span>
        </span>
      </button>

      {open && (
        <div className="border-t border-stone-100 px-4 pb-5 pt-4 sm:px-6">
          {state.phase === "loading" && (
            <div className="flex items-center gap-2 py-6 text-sm text-stone-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading data…
            </div>
          )}

          {state.phase === "error" && (
            <p className="rounded border border-dashed border-rose-300 bg-rose-50 px-3 py-4 text-sm text-rose-800">
              Could not load data: {state.message}
            </p>
          )}

          {state.phase === "ready" && artifacts.length > 0 && (
            <div className="space-y-4">
              {/* Artifact picker — one entry per figure with data. */}
              <ArtifactPicker
                artifacts={artifacts}
                selectedId={selected}
                onSelect={setSelected}
              />

              {withRows.length === 0 && (
                <p className="rounded border border-dashed border-stone-300 bg-stone-50 px-3 py-4 text-sm text-stone-600">
                  None of this task&rsquo;s figure sidecars carry per-row data on
                  disk. The full data lives behind the pinned artifact links in
                  the result body (HF Hub / GitHub) — open a figure below to view
                  it on GitHub.
                </p>
              )}

              {active && <ArtifactPanel artifact={active} />}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ArtifactPicker({
  artifacts,
  selectedId,
  onSelect,
}: {
  artifacts: DataArtifact[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (artifacts.length <= 1) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {artifacts.map((a) => {
        const isSel = a.id === selectedId;
        const hasRows = a.rows.length > 0;
        return (
          <button
            key={a.id}
            type="button"
            onClick={() => onSelect(a.id)}
            className={
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors " +
              (isSel
                ? "border-stone-800 bg-stone-800 text-white"
                : "border-stone-300 bg-white text-stone-600 hover:border-stone-400 hover:bg-stone-50")
            }
            title={hasRows ? `${a.totalRows} rows` : "No tabular data — figure link-out only"}
          >
            <TableIcon className="h-3 w-3" />
            {a.label}
            {hasRows ? (
              <span className={isSel ? "text-stone-300" : "text-stone-400"}>
                {a.totalRows}
              </span>
            ) : (
              <span className={isSel ? "text-stone-400" : "text-stone-300"}>·</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function ArtifactPanel({ artifact }: { artifact: DataArtifact }) {
  return (
    <div className="space-y-3">
      {/* Provenance + link-outs. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-stone-500">
        {artifact.sourcePath && (
          <span className="inline-flex items-center gap-1">
            <span className="text-stone-400">source:</span>
            <a
              href={GITHUB_BLOB_BASE + artifact.sourcePath}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-stone-600 hover:text-stone-900 hover:underline"
            >
              {artifact.sourcePath}
            </a>
          </span>
        )}
        {artifact.figureUrl && (
          <a
            href={artifact.figureUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-stone-600 hover:text-stone-900 hover:underline"
          >
            <ExternalLink className="h-3 w-3" />
            View figure
          </a>
        )}
      </div>

      {artifact.description && (
        <p className="text-xs leading-relaxed text-stone-600">{artifact.description}</p>
      )}

      {artifact.rows.length > 0 ? (
        <>
          <DataTable columns={artifact.columns} rows={artifact.rows} />
          {artifact.truncated && (
            <p className="rounded border border-dashed border-amber-300 bg-amber-50 px-3 py-2 text-[11px] text-amber-900">
              Showing the first {artifact.rows.length} of {artifact.totalRows} rows
              available on disk (capped for payload size). Open the source file
              above for the complete set; the full raw data behind external HF
              links is not loadable in the dashboard.
            </p>
          )}
        </>
      ) : (
        <p className="rounded border border-dashed border-stone-300 bg-stone-50 px-3 py-3 text-xs text-stone-600">
          This figure&rsquo;s sidecar carries no per-row data on disk
          {artifact.figureUrl ? " — use “View figure” above" : ""}. The complete
          data is behind the pinned artifact links in the result body.
        </p>
      )}
    </div>
  );
}
