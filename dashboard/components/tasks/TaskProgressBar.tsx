/**
 * Purely presentational pipeline progress bar + ETA chip (task #587).
 *
 * No hooks, no data fetching — it renders a `TaskProgressView` computed
 * server-side by `lib/progress.ts` (so it works both inside the client
 * <TaskBoard> kanban cards and on the server-rendered task detail page).
 *
 * Chip semantics (mirrors the estimator's honesty rules):
 *   - active     → "~2.1h left · ~7.5h total" (median remaining + expected
 *                  total machine time; "≈" prefix = soft GPU-derived
 *                  estimate; the [p25–p75] band replaces the remaining
 *                  median if its kill switch is ever re-enabled), plus
 *                  "+ plan review" when the human plan-review wait lies
 *                  ahead (it is excluded from the machine estimate on
 *                  purpose). A followups_running task shows the follow-up
 *                  round's own remaining/total.
 *   - human-wait → "waiting on you" (+ machine work left after it / total).
 *   - blocked    → grey bar frozen at the stage floor, "blocked", no countdown.
 *   - overdue    → bar parked, "running long" label, NO countdown (the
 *                  estimate stopped being supported by the historical basis).
 *   - stale      → bar at the live-status floor, no chip (snapshot too old).
 */
import type { TaskProgressView } from "@/lib/progress";

const BAND_TOOLTIP =
  "Median remaining · expected total machine time for a typical clean " +
  "pass, from recent task history (per-stage medians — a heuristic, not a " +
  "guarantee; human plan-review wait excluded). ≈ marks a soft " +
  "GPU-hours-derived estimate.";

export function TaskProgressBar({
  view,
  compact = false,
}: {
  view: TaskProgressView;
  compact?: boolean;
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
    <div className={compact ? "mt-1.5" : "max-w-md"}>
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
            className={`h-full rounded-full ${fillCls}`}
            style={{ width: `${pctClamped * 100}%` }}
          />
        </div>
        <span className="font-mono text-[10px] tabular-nums text-stone-500">{pctLabel}</span>
      </div>
      <Chip view={view} compact={compact} />
    </div>
  );
}

function Chip({ view, compact }: { view: TaskProgressView; compact: boolean }) {
  const base = `mt-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 ${
    compact ? "text-[10px]" : "text-xs"
  } font-medium`;
  switch (view.state) {
    case "blocked":
      return <span className={`${base} bg-stone-100 text-stone-600`}>blocked</span>;
    case "overdue":
      return (
        <span
          className={`${base} bg-amber-50 text-amber-800`}
          title="Past the typical clean-pass range for this stage (>p75) — countdown suppressed."
        >
          running long
        </span>
      );
    case "stale":
      return (
        <span
          className={`${base} bg-stone-50 text-stone-400`}
          title="Progress snapshot is stale (cron has not ticked recently); showing the stage floor only."
        >
          stale
        </span>
      );
    case "human-wait": {
      const then = view.etaLabel ?? view.remainingLabel;
      return (
        <span className={`${base} bg-violet-50 text-violet-700`} title={BAND_TOOLTIP}>
          waiting on you
          {then ? (
            <span className="font-normal">
              · then {then}
              {view.totalLabel ? ` of ${view.totalLabel}` : ""}
            </span>
          ) : null}
        </span>
      );
    }
    default: {
      // Band (when its kill switch is re-enabled) wins over the median.
      const remaining = view.etaLabel ?? view.remainingLabel;
      if (!remaining && !view.totalLabel) return null;
      return (
        <span className={`${base} bg-teal-50 text-teal-800`} title={BAND_TOOLTIP}>
          {remaining ? `${remaining} left` : null}
          {view.totalLabel ? (
            <span className="font-normal">
              {remaining ? "· " : ""}
              {view.totalLabel} total
            </span>
          ) : null}
          {view.planReviewAhead ? <span className="font-normal">+ plan review</span> : null}
        </span>
      );
    }
  }
}
