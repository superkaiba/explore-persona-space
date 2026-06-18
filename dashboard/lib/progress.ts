/**
 * Server-only reader for the task-progress snapshot (task #587).
 *
 * Contract (pinned — Python writes this file; do not deviate):
 *
 *   ~/.eps-autonomous/task_progress.json
 *   {
 *     "version": 1,
 *     "generated_at": "<ISO8601 UTC>",
 *     "stats": {
 *       "window_rule": "...",
 *       "stats_generated_at": "<ISO8601 UTC>",
 *       "buckets": { "experiment"|"code"|"pooled": { "<stage>": {n, p25_h,
 *         median_h, p75_h, basis} } },
 *       "pct_floor_by_stage": { "<bucket>": { "<stage>": number } }
 *     },
 *     "tasks": {
 *       "<issue>": {
 *         "issue": 561, "status": "running", "stage": "running",
 *         "kind_bucket": "experiment",
 *         "stats_basis": "bucket"|"pooled"|"all-history"|"mixed",
 *         "stage_entered_at": "<ISO8601 UTC>",
 *         "pct_floor": 0.18, "pct_span": 0.55,
 *         "frac_median_h": 4.75,           // HISTORICAL median (bar pace)
 *         "stage_p25_h": 2.1, "stage_median_h": 4.75, "stage_p75_h": 10.8,
 *           // EFFECTIVE band quantiles (GPU-refined when eta_basis says so;
 *           // p75 drives the overdue cutoff)
 *         "remaining_after_p25_h": 0.7, "remaining_after_median_h": 1.1,
 *         "remaining_after_p75_h": 1.9,
 *         "total_p25_h": 4.1, "total_median_h": 7.5, "total_p75_h": 14.0,
 *           // expected TOTAL machine time for a typical clean pass (main
 *           // pipeline minus plan_pending, current stage at its EFFECTIVE
 *           // quantiles; a followups_running row's total is the round's own
 *           // expected duration). Optional: absent on pre-upgrade snapshots
 *           // → the total label is simply omitted.
 *         "human_wait": false, "blocked": false, "plan_review_ahead": false,
 *         "gpu_hours_total": 19.0, "gpu_count": 4,
 *         "gpu_conversion": "intent-map"|"note-regex"|"assumed-1gpu"|null,
 *         "eta_basis": "historical"|"gpu-refined"|"gpu-assumed"
 *       }
 *     }
 *   }
 *
 * Written by the 5-minute summarize cron (`scripts/task_progress.py
 * snapshot`) — the ONLY writer; this module NEVER writes anything.
 *
 * The maths here is deliberately limited to the ONE pinned interpolation
 * formula mirrored from Python `task_progress.interpolate()` +
 * `format_eta_band()` / `format_duration()`; the shared fixture
 * `tests/fixtures/task_progress_vectors.json` is replayed through BOTH
 * implementations (pytest + `npm run test:progress`) to pin lockstep.
 *
 * Rendering is LIVE-STATUS KEYED: `getProgressMap(liveStatuses)` receives
 * the statuses the server page already loaded from REGISTRY; a snapshot row
 * whose live status has no stage floor (awaiting_promotion, completed,
 * archived, proposed, anything unknown) is DROPPED — no bar, regardless of
 * snapshot freshness. `followups_running` IS in scope: it renders as its
 * own 0→1 track (floor 0.0) pacing over historical follow-up-round spans.
 * Within-pipeline mismatches render the live status' floor ALONE (a
 * backward re-plan shows the bar moving backward; no max() clamp), chip
 * hidden until the next tick.
 *
 * Honesty notes carried from the estimator (see its module docstring):
 * the band is a "typical clean forward pass" — quantile sums are a
 * heuristic, blocked/backward detours are excluded from the stats, and the
 * overdue state (band suppressed past stage p75) covers what they exclude.
 *
 * If the file is missing, empty, or unparseable, return an empty snapshot
 * (NOT a throw) — the writer cron may not have run yet. Per-row narrowing
 * is permissive: a malformed row is skipped, never blanks the page.
 */
import { readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

export const TASK_PROGRESS_PATH = path.join(
  homedir(),
  ".eps-autonomous",
  "task_progress.json",
);

/** Snapshot older than this renders the bar at the live-status floor with a
 * "stale" state and NO countdown (honest degradation when the cron dies). */
export const STALE_AFTER_MS = 30 * 60 * 1000;

/**
 * §7 kill criterion — ETA hour band. The one-time calibration backtest
 * (2026-06-11, `scripts/task_progress.py backtest`) measured [p25, p75]
 * band coverage 0.368 raw / 0.404 guarded over 625 historical stage
 * entries — below the pinned 0.50 keep threshold — so the countdown chip
 * ships DISABLED: position bar + state labels (overdue / blocked /
 * waiting-on-you) render, the hour band does not. Mirrors
 * `task_progress.ETA_BAND_ENABLED` (flip BOTH together); the band
 * machinery + shared vectors stay tested via an explicit override.
 *
 * The MEDIAN remaining/total labels ("~2.1h left · ~7.5h total") are NOT
 * gated by this switch: the kill criterion is about [p25, p75] COVERAGE
 * claims, which a median point estimate does not make. The tooltip carries
 * the honesty framing instead.
 */
export const ETA_BAND_ENABLED = false;

const MACHINE_STAGES = [
  "planning",
  "plan_pending",
  "approved",
  "running",
  "verifying",
  "interpreting",
  "reviewing",
  // Optional post-pipeline stage: a same-issue follow-up round held at
  // followups_running renders as its own 0→1 track (floor 0.0 in the
  // snapshot's pct_floor_by_stage).
  "followups_running",
] as const;
const MACHINE_STAGE_SET: ReadonlySet<string> = new Set(MACHINE_STAGES);

const EPS_H = 0.01;
const FRAC_CAP = 0.95;

export type EtaBasis = "historical" | "gpu-refined" | "gpu-assumed";

export type TaskProgressRow = {
  issue: number;
  stage: string;
  kindBucket: string;
  stageEnteredAt: string;
  pctFloor: number;
  pctSpan: number;
  fracMedianH: number;
  stageP25H: number;
  stageMedianH: number;
  stageP75H: number;
  remainingAfterP25H: number;
  remainingAfterMedianH: number;
  remainingAfterP75H: number;
  /** Expected total machine time (typical clean pass). Null on pre-upgrade
   * snapshot rows — the total label is then omitted, never guessed. */
  totalMedianH: number | null;
  humanWait: boolean;
  blocked: boolean;
  planReviewAhead: boolean;
  etaBasis: EtaBasis;
};

export type ProgressSnapshot = {
  generatedAt: string | null;
  /** bucket -> stage -> floor (0..1). Needed for the live-status floor-clamp
   * renders; absent/malformed stats degrade those renders to "no bar". */
  floors: Record<string, Record<string, number>> | null;
  tasks: Record<number, TaskProgressRow>;
};

export type TaskProgressView = {
  /** 0..1 pipeline position. */
  pct: number;
  /** Compact band ("~4–9h", "≈2–14h") or null (blocked / overdue / stale /
   * mismatch / band kill switch — the chip may still show the median
   * remaining/total labels below). */
  etaLabel: string | null;
  /** Median machine time remaining ("~2.1h") or null (blocked / overdue /
   * stale / mismatch). Point estimate — NOT gated by the band kill switch. */
  remainingLabel: string | null;
  /** Expected total machine time for a typical clean pass ("~7.5h") or null
   * (same suppression states, or a pre-upgrade snapshot row). */
  totalLabel: string | null;
  state: "active" | "human-wait" | "blocked" | "overdue" | "stale";
  basis: EtaBasis;
  /** plan_pending lies AHEAD of the current stage — the dashboard chip
   * appends "+ plan review" (the band deliberately excludes human wait). */
  planReviewAhead: boolean;
};

/* ----------------------------------------------------------------------- *
 * Snapshot loading (module-level cache keyed on mtime, like lib/sessions). *
 * ----------------------------------------------------------------------- */

const EMPTY_SNAPSHOT: ProgressSnapshot = { generatedAt: null, floors: null, tasks: {} };

let cache: { mtimeMs: number; snapshot: ProgressSnapshot } | null = null;

export function loadProgressSnapshot(): ProgressSnapshot {
  let mtimeMs: number;
  try {
    mtimeMs = statSync(TASK_PROGRESS_PATH).mtimeMs;
  } catch {
    cache = null;
    return EMPTY_SNAPSHOT;
  }
  if (cache && cache.mtimeMs === mtimeMs) return cache.snapshot;
  let raw: string;
  try {
    raw = readFileSync(TASK_PROGRESS_PATH, "utf8");
  } catch {
    cache = null;
    return EMPTY_SNAPSHOT;
  }
  const snapshot = parseProgressSnapshot(raw);
  cache = { mtimeMs, snapshot };
  return snapshot;
}

/** Pure parser (exported for the tsx mirror test). */
export function parseProgressSnapshot(raw: string): ProgressSnapshot {
  if (raw.trim() === "") return EMPTY_SNAPSHOT;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return EMPTY_SNAPSHOT;
  }
  if (!parsed || typeof parsed === "string" || typeof parsed !== "object") {
    return EMPTY_SNAPSHOT;
  }
  const obj = parsed as Record<string, unknown>;
  const generatedAt = typeof obj.generated_at === "string" ? obj.generated_at : null;

  let floors: Record<string, Record<string, number>> | null = null;
  const stats = obj.stats;
  if (stats && typeof stats === "object" && !Array.isArray(stats)) {
    const rawFloors = (stats as Record<string, unknown>).pct_floor_by_stage;
    if (rawFloors && typeof rawFloors === "object" && !Array.isArray(rawFloors)) {
      floors = {};
      for (const [bucket, value] of Object.entries(rawFloors as Record<string, unknown>)) {
        if (!value || typeof value !== "object" || Array.isArray(value)) continue;
        const byStage: Record<string, number> = {};
        for (const [stage, f] of Object.entries(value as Record<string, unknown>)) {
          if (typeof f === "number" && Number.isFinite(f)) byStage[stage] = f;
        }
        floors[bucket] = byStage;
      }
    }
  }

  const tasks: Record<number, TaskProgressRow> = {};
  const rawTasks = obj.tasks;
  if (rawTasks && typeof rawTasks === "object" && !Array.isArray(rawTasks)) {
    for (const value of Object.values(rawTasks as Record<string, unknown>)) {
      const row = narrowRow(value);
      if (row) tasks[row.issue] = row;
    }
  }
  return { generatedAt, floors, tasks };
}

function num(o: Record<string, unknown>, key: string): number | null {
  const v = o[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function narrowRow(value: unknown): TaskProgressRow | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const o = value as Record<string, unknown>;
  const issue = num(o, "issue");
  const stage = typeof o.stage === "string" ? o.stage : null;
  const stageEnteredAt = typeof o.stage_entered_at === "string" ? o.stage_entered_at : null;
  const pctFloor = num(o, "pct_floor");
  const pctSpan = num(o, "pct_span");
  const fracMedianH = num(o, "frac_median_h");
  const stageP25H = num(o, "stage_p25_h");
  const stageMedianH = num(o, "stage_median_h");
  const stageP75H = num(o, "stage_p75_h");
  const remainingAfterP25H = num(o, "remaining_after_p25_h");
  const remainingAfterMedianH = num(o, "remaining_after_median_h");
  const remainingAfterP75H = num(o, "remaining_after_p75_h");
  if (
    issue === null ||
    stage === null ||
    stageEnteredAt === null ||
    pctFloor === null ||
    pctSpan === null ||
    fracMedianH === null ||
    stageP25H === null ||
    stageMedianH === null ||
    stageP75H === null ||
    remainingAfterP25H === null ||
    remainingAfterMedianH === null ||
    remainingAfterP75H === null
  ) {
    return null; // skip the malformed row; never blank the page
  }
  const rawBasis = typeof o.eta_basis === "string" ? o.eta_basis : null;
  const etaBasis: EtaBasis =
    rawBasis === "gpu-refined" || rawBasis === "gpu-assumed" ? rawBasis : "historical";
  return {
    issue,
    stage,
    kindBucket: typeof o.kind_bucket === "string" ? o.kind_bucket : "pooled",
    stageEnteredAt,
    pctFloor,
    pctSpan,
    fracMedianH,
    stageP25H,
    stageMedianH,
    stageP75H,
    remainingAfterP25H,
    remainingAfterMedianH,
    remainingAfterP75H,
    // Optional (additive contract change): a pre-upgrade snapshot row simply
    // renders without the total label until the next cron tick.
    totalMedianH: num(o, "total_median_h"),
    humanWait: o.human_wait === true,
    blocked: o.blocked === true,
    planReviewAhead: o.plan_review_ahead === true,
    etaBasis,
  };
}

/* ----------------------------------------------------------------------- *
 * The pinned interpolation formula (mirror of Python `interpolate`).       *
 * ----------------------------------------------------------------------- */

export type Interpolated = {
  pct: number;
  eta: { p25H: number; medianH: number; p75H: number } | null;
  overdue: boolean;
};

export function interpolateRow(row: TaskProgressRow, nowMs: number): Interpolated {
  if (row.blocked) return { pct: row.pctFloor, eta: null, overdue: false };
  const enteredMs = Date.parse(row.stageEnteredAt);
  const elapsedH = Number.isFinite(enteredMs)
    ? Math.max((nowMs - enteredMs) / 3_600_000, 0)
    : 0;
  if (row.humanWait) {
    return {
      pct: row.pctFloor,
      eta: {
        p25H: row.remainingAfterP25H,
        medianH: row.remainingAfterMedianH,
        p75H: row.remainingAfterP75H,
      },
      overdue: false,
    };
  }
  const frac = Math.min(elapsedH / Math.max(row.fracMedianH, EPS_H), FRAC_CAP);
  const pct = row.pctFloor + frac * row.pctSpan;
  if (elapsedH > row.stageP75H) return { pct, eta: null, overdue: true };
  return {
    pct,
    eta: {
      p25H: Math.max(row.stageP25H - elapsedH, 0) + row.remainingAfterP25H,
      medianH: Math.max(row.stageMedianH - elapsedH, 0) + row.remainingAfterMedianH,
      p75H: Math.max(row.stageP75H - elapsedH, 0) + row.remainingAfterP75H,
    },
    overdue: false,
  };
}

/* ----------------------------------------------------------------------- *
 * Band formatting (mirror of Python `format_eta_band`; half-up rounding).  *
 * ----------------------------------------------------------------------- */

function halfup(x: number): number {
  return Math.floor(x + 0.5);
}

function fmtHours(v: number): string {
  if (v < 10) {
    const d = Math.floor(v * 10 + 0.5) / 10;
    return Number.isInteger(d) ? String(d) : d.toFixed(1);
  }
  return String(halfup(v));
}

function fmtDays(v: number): string {
  const d = Math.floor((v / 24) * 10 + 0.5) / 10;
  return Number.isInteger(d) ? String(d) : d.toFixed(1);
}

export function formatEtaBand(p25H: number, p75H: number, basis: EtaBasis): string {
  const prefix = basis === "historical" ? "~" : "≈";
  if (p75H < 1) {
    const a = Math.max(1, halfup(p25H * 60));
    const b = Math.max(1, halfup(p75H * 60));
    return `${prefix}${a}–${b}m`;
  }
  if (p75H < 24) return `${prefix}${fmtHours(p25H)}–${fmtHours(p75H)}h`;
  return `${prefix}${fmtDays(p25H)}–${fmtDays(p75H)}d`;
}

/** Compact single duration ("~25m" | "~2.1h" | "~1.3d"; "≈" for GPU-derived
 * bases) — mirror of Python `format_duration`. Median point estimates are
 * deliberately NOT gated by the §7 band kill switch (no coverage claim). */
export function formatDuration(hours: number, basis: EtaBasis): string {
  const prefix = basis === "historical" ? "~" : "≈";
  if (hours < 1) return `${prefix}${Math.max(1, halfup(hours * 60))}m`;
  if (hours < 24) return `${prefix}${fmtHours(hours)}h`;
  return `${prefix}${fmtDays(hours)}d`;
}

/* ----------------------------------------------------------------------- *
 * Live-status-keyed view construction.                                     *
 * ----------------------------------------------------------------------- */

/**
 * Pure map builder (exported for the tsx mirror test). Live statuses are a
 * REQUIRED input — rows are dropped or floor-clamped per the live status
 * before any interpolation (the plan's §3.5 table):
 *
 *   - live status without a stage floor (and not "blocked") → NO view;
 *   - live "blocked" → blocked view frozen at the row's floor, no countdown;
 *   - snapshot stale (generated_at older than STALE_AFTER_MS) → bar at the
 *     LIVE status' floor, "stale", no chip;
 *   - live in-pipeline but ≠ snapshot stage → bar at the LIVE status' floor
 *     ALONE (backward re-plans render backward), chip hidden;
 *   - matched → interpolate at `nowMs` (the existing 60 s router.refresh()
 *     keeps the numbers moving).
 */
export function buildProgressMap(
  snapshot: ProgressSnapshot,
  liveStatuses: Record<number, string>,
  nowMs: number,
  opts: { etaBandEnabled?: boolean } = {},
): Record<number, TaskProgressView> {
  const etaBandEnabled = opts.etaBandEnabled ?? ETA_BAND_ENABLED;
  const out: Record<number, TaskProgressView> = {};
  const generatedMs = snapshot.generatedAt ? Date.parse(snapshot.generatedAt) : NaN;
  const stale = !Number.isFinite(generatedMs) || nowMs - generatedMs > STALE_AFTER_MS;

  for (const [idStr, status] of Object.entries(liveStatuses)) {
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;
    const row = snapshot.tasks[id];
    if (!row) continue;

    if (status === "blocked") {
      out[id] = {
        pct: row.pctFloor,
        etaLabel: null,
        remainingLabel: null,
        totalLabel: null,
        state: "blocked",
        basis: row.etaBasis,
        planReviewAhead: false,
      };
      continue;
    }
    if (!MACHINE_STAGE_SET.has(status)) continue; // floor-less live status → no bar

    if (stale || status !== row.stage) {
      // Floor-clamp render from the LIVE status (never max() with the
      // snapshot pct — backward must show backward). Needs the floors table;
      // without it there is nothing honest to draw, so skip.
      const byBucket = snapshot.floors?.[row.kindBucket] ?? snapshot.floors?.pooled;
      const floor = byBucket?.[status];
      if (typeof floor !== "number") continue;
      out[id] = {
        pct: floor,
        etaLabel: null,
        remainingLabel: null,
        totalLabel: null,
        state: stale ? "stale" : "active",
        basis: row.etaBasis,
        planReviewAhead: row.planReviewAhead,
      };
      continue;
    }

    const { pct, eta, overdue } = interpolateRow(row, nowMs);
    if (row.blocked) {
      out[id] = {
        pct,
        etaLabel: null,
        remainingLabel: null,
        totalLabel: null,
        state: "blocked",
        basis: row.etaBasis,
        planReviewAhead: false,
      };
    } else if (overdue) {
      out[id] = {
        pct,
        etaLabel: null,
        remainingLabel: null,
        totalLabel: null,
        state: "overdue",
        basis: row.etaBasis,
        planReviewAhead: row.planReviewAhead,
      };
    } else {
      out[id] = {
        pct,
        etaLabel:
          etaBandEnabled && eta ? formatEtaBand(eta.p25H, eta.p75H, row.etaBasis) : null,
        remainingLabel: eta ? formatDuration(eta.medianH, row.etaBasis) : null,
        totalLabel:
          row.totalMedianH !== null ? formatDuration(row.totalMedianH, row.etaBasis) : null,
        state: row.humanWait ? "human-wait" : "active",
        basis: row.etaBasis,
        planReviewAhead: row.planReviewAhead,
      };
    }
  }
  return out;
}

/** Server entrypoint used by the /tasks pages + the progress API route. */
export function getProgressMap(
  liveStatuses: Record<number, string>,
): Record<number, TaskProgressView> {
  return buildProgressMap(loadProgressSnapshot(), liveStatuses, Date.now());
}

/** Snapshot metadata for the API route ({generated_at, stale}). */
export function getProgressMeta(): { generatedAt: string | null; stale: boolean } {
  const snapshot = loadProgressSnapshot();
  const generatedMs = snapshot.generatedAt ? Date.parse(snapshot.generatedAt) : NaN;
  return {
    generatedAt: snapshot.generatedAt,
    stale: !Number.isFinite(generatedMs) || Date.now() - generatedMs > STALE_AFTER_MS,
  };
}
