/**
 * Purely presentational pipeline progress bar + time meta line (task #587).
 *
 * No hooks, no data fetching — it renders a `TaskProgressView` computed
 * server-side by `lib/progress.ts` (so it works both inside the client
 * <TaskBoard> kanban cards and on the server-rendered task detail page).
 *
 * Layout: bar + % on the first row, then ONE quiet two-sided meta line —
 * remaining on the left (teal, the number you scan for), expected total on
 * the right (muted). Pills are reserved for EXCEPTIONAL states (blocked /
 * running long / stale / waiting on you); ordinary numbers render as plain
 * text so a board full of cards stays calm. The estimate-softness prefix
 * (~ historical, ≈ GPU-derived) is carried once per line, on the leading
 * value; the total drops it to cut glyph noise (same basis, same row).
 *
 * Semantics (mirrors the estimator's honesty rules):
 *   - active     → "~2.1h left   of 7.5h" (median remaining + expected total
 *                  machine time; the [p25–p75] band replaces the remaining
 *                  median if its kill switch is ever re-enabled), plus
 *                  "+ plan review" when the human plan-review wait lies
 *                  ahead (it is excluded from the machine estimate on
 *                  purpose). A followups_running task shows the follow-up
 *                  round's own remaining/total.
 *   - human-wait → "waiting on you" pill + "then ~2h of 7.5h".
 *   - blocked    → grey bar frozen at the stage floor, "blocked", no countdown.
 *   - overdue    → bar parked, "running long" pill, NO countdown (the
 *                  estimate stopped being supported by the historical basis).
 *   - stale      → bar at the live-status floor, no meta (snapshot too old).
 */
import type { TaskProgressView } from "@/lib/progress";

const BAND_TOOLTIP =
  "Median remaining · expected total machine time for a typical clean " +
  "pass, from recent task history (per-stage medians — a heuristic, not a " +
  "guarantee; human plan-review wait excluded). ≈ marks a soft " +
  "GPU-hours-derived estimate.";

/** The line's softness marker lives on its leading value — strip the
 * duplicate from the trailing total (same basis, same row). */
function stripPrefix(label: string): string {
  return label.replace(/^[~≈]/, "");
}

export function TaskProgressBar({
  view,
  compact = false,
  className,
}: {
  view: TaskProgressView;
  compact?: boolean;
  /** Wrapper override for non-card placements (e.g. list rows, where the
   * card default `mt-1.5` would break vertical centering). */
  className?: string;
}) {
  const pctClamped = Math.min(Math.max(view.pct, 0), 1);
  const pctLabel = `${Math.floor(pctClamped * 100 + 0.5)}%`;
  const blocked = view.state === "blocked";
  const fillCls = blocked
    ? "bg-stone-400"
    : view.state === "overdue"
      ? "bg-amber-500"
      : view.state === "human-wait"
        ? "bg-violet-400"
        : "bg-teal-500";

  return (
    <div className={className ?? (compact ? "mt-1.5" : "max-w-md")}>
      <div className="flex items-center gap-2">
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.floor(pctClamped * 100 + 0.5)}
          className={`h-1.5 flex-1 overflow-hidden rounded-full ${
            blocked ? "bg-stone-200" : "bg-stone-200/80"
          }`}
        >
          <div
            className={`h-full rounded-full transition-[width] duration-700 ease-out ${fillCls}`}
            style={{ width: `${pctClamped * 100}%` }}
          />
        </div>
        <span className="font-mono text-[10px] tabular-nums text-stone-500">{pctLabel}</span>
      </div>
      <MetaLine view={view} compact={compact} />
    </div>
  );
}

function MetaLine({ view, compact }: { view: TaskProgressView; compact: boolean }) {
  const size = compact ? "text-[10px]" : "text-xs";
  const pill = `inline-flex items-center rounded px-1.5 py-0.5 font-medium ${size}`;
  switch (view.state) {
    case "blocked":
      return (
        <div className="mt-1">
          <span className={`${pill} bg-stone-100 text-stone-600`}>blocked</span>
        </div>
      );
    case "overdue":
      return (
        <div className="mt-1">
          <span
            className={`${pill} bg-amber-50 text-amber-800`}
            title="Past the typical clean-pass range for this stage (>p75) — countdown suppressed."
          >
            running long
          </span>
        </div>
      );
    case "stale":
      return (
        <div className="mt-1">
          <span
            className={`${pill} bg-stone-50 text-stone-400`}
            title="Progress snapshot is stale (cron has not ticked recently); showing the stage floor only."
          >
            stale
          </span>
        </div>
      );
    case "human-wait": {
      const then = view.etaLabel ?? view.remainingLabel;
      return (
        <div
          className={`mt-1 flex items-center justify-between gap-2 ${size}`}
          title={BAND_TOOLTIP}
        >
          <span className={`${pill} bg-violet-50 text-violet-700`}>waiting on you</span>
          {then ? (
            <span className="tabular-nums text-stone-400">
              then <span className="font-medium text-stone-500">{then}</span>
              {view.totalLabel ? ` of ${stripPrefix(view.totalLabel)}` : ""}
            </span>
          ) : null}
        </div>
      );
    }
    default: {
      // Band (when its kill switch is re-enabled) wins over the median.
      const remaining = view.etaLabel ?? view.remainingLabel;
      if (!remaining && !view.totalLabel) return null;
      return (
        <div
          className={`mt-1 flex items-baseline justify-between gap-2 tabular-nums ${size}`}
          title={BAND_TOOLTIP}
        >
          {remaining ? (
            <span className="font-medium text-teal-700">
              {remaining} <span className="font-normal text-teal-600/80">left</span>
            </span>
          ) : (
            <span aria-hidden="true" />
          )}
          <span className="text-stone-400">
            {view.totalLabel
              ? remaining
                ? `of ${stripPrefix(view.totalLabel)}`
                : `${view.totalLabel} total`
              : null}
            {view.planReviewAhead ? (
              <span className="text-violet-500">{view.totalLabel ? " " : ""}+ plan review</span>
            ) : null}
          </span>
        </div>
      );
    }
  }
}
