/**
 * Python<->TS lockstep proof for the task-progress interpolation formula
 * (task #587).
 *
 * No test runner is configured in this dashboard, so this is a standalone
 * script (run under tsx so the `.ts` import resolves):
 *
 *   npx tsx scripts/progress.test.mjs
 *   # or: npm run test:progress
 *
 * It replays the SAME fixture pytest replays
 * (tests/fixtures/task_progress_vectors.json, repo root):
 *
 *   1. interpolate_vectors → `interpolateRow` + `formatEtaBand` +
 *      `formatDuration` must reproduce pct / eta band / overdue / eta_label /
 *      remaining_label / total_label exactly (half-up rounding pinned
 *      identical to Python's `math.floor(x + 0.5)`).
 *   2. gating_vectors → `buildProgressMap` must drop rows whose LIVE status
 *      has no stage floor, floor-clamp within-pipeline mismatches (backward
 *      re-plans render backward; a followups_running mismatch clamps to its
 *      0.0 floor), render the blocked / stale / human-wait states, and
 *      interpolate fresh matched rows — including the followups_running
 *      own-track render and the median remaining/total labels.
 *   3. parseProgressSnapshot must narrow permissively (malformed file /
 *      malformed row → empty / skipped, never a throw; a pre-upgrade row
 *      without total_median_h narrows with totalMedianH null).
 *
 * Exits non-zero on the first failed assertion.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ETA_BAND_ENABLED,
  buildProgressMap,
  formatDuration,
  formatEtaBand,
  interpolateRow,
  parseProgressSnapshot,
} from "../lib/progress.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.resolve(__dirname, "../../tests/fixtures/task_progress_vectors.json");

const fixture = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));

let failures = 0;
function check(name, cond, detail) {
  if (cond) return;
  failures += 1;
  console.error(`FAIL ${name}: ${detail}`);
}

function approx(a, b, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

/** Snake_case fixture row -> the narrowed camelCase TaskProgressRow shape. */
function toRow(raw) {
  return {
    issue: raw.issue,
    stage: raw.stage,
    kindBucket: raw.kind_bucket,
    stageEnteredAt: raw.stage_entered_at,
    pctFloor: raw.pct_floor,
    pctSpan: raw.pct_span,
    fracMedianH: raw.frac_median_h,
    stageP25H: raw.stage_p25_h,
    stageMedianH: raw.stage_median_h,
    stageP75H: raw.stage_p75_h,
    remainingAfterP25H: raw.remaining_after_p25_h,
    remainingAfterMedianH: raw.remaining_after_median_h,
    remainingAfterP75H: raw.remaining_after_p75_h,
    totalMedianH: typeof raw.total_median_h === "number" ? raw.total_median_h : null,
    humanWait: raw.human_wait === true,
    blocked: raw.blocked === true,
    planReviewAhead: raw.plan_review_ahead === true,
    etaBasis:
      raw.eta_basis === "gpu-refined" || raw.eta_basis === "gpu-assumed"
        ? raw.eta_basis
        : "historical",
  };
}

/* 1 — interpolate vectors ------------------------------------------------ */

for (const v of fixture.interpolate_vectors) {
  const row = toRow(v.row);
  const nowMs = Date.parse(v.now);
  const exp = v.expected;
  const { pct, eta, overdue } = interpolateRow(row, nowMs);
  check(v.name, approx(pct, exp.pct), `pct ${pct} != ${exp.pct}`);
  check(v.name, overdue === exp.overdue, `overdue ${overdue} != ${exp.overdue}`);
  if (exp.eta_p75_h === null) {
    check(v.name, eta === null, `eta should be null, got ${JSON.stringify(eta)}`);
  } else {
    check(v.name, eta !== null, "eta unexpectedly null");
    if (eta) {
      check(v.name, approx(eta.p25H, exp.eta_p25_h), `eta_p25 ${eta.p25H} != ${exp.eta_p25_h}`);
      check(
        v.name,
        approx(eta.medianH, exp.eta_median_h),
        `eta_median ${eta.medianH} != ${exp.eta_median_h}`,
      );
      check(v.name, approx(eta.p75H, exp.eta_p75_h), `eta_p75 ${eta.p75H} != ${exp.eta_p75_h}`);
      const label = formatEtaBand(eta.p25H, eta.p75H, row.etaBasis);
      check(v.name, label === exp.eta_label, `label ${label} != ${exp.eta_label}`);
    }
  }
  // Median remaining/total labels (formatDuration mirror of format_duration).
  const remainingLabel = eta ? formatDuration(eta.medianH, row.etaBasis) : null;
  check(
    v.name,
    remainingLabel === exp.remaining_label,
    `remaining_label ${remainingLabel} != ${exp.remaining_label}`,
  );
  check(v.name, row.totalMedianH !== null, "fixture row must carry total_median_h");
  const totalLabel = formatDuration(row.totalMedianH, row.etaBasis);
  check(v.name, totalLabel === exp.total_label, `total_label ${totalLabel} != ${exp.total_label}`);
}

/* 2 — live-status gating vectors ----------------------------------------- */

for (const g of fixture.gating_vectors) {
  const row = toRow(g.row);
  const snapshot = {
    generatedAt: g.generated_at,
    floors: fixture.stats.pct_floor_by_stage,
    tasks: { [row.issue]: row },
  };
  const nowMs = Date.parse(g.now);
  // etaBandEnabled: true — the fixture pins the FULL band rendering so the
  // re-enable path stays mirrored; the production default is checked below.
  const map = buildProgressMap(snapshot, { [row.issue]: g.live_status }, nowMs, {
    etaBandEnabled: true,
  });
  const view = map[row.issue];
  if (g.expect === null) {
    check(g.name, view === undefined, `expected no view, got ${JSON.stringify(view)}`);
    continue;
  }
  check(g.name, view !== undefined, "expected a view, got none");
  if (!view) continue;
  check(g.name, approx(view.pct, g.expect.pct), `pct ${view.pct} != ${g.expect.pct}`);
  check(g.name, view.state === g.expect.state, `state ${view.state} != ${g.expect.state}`);
  check(
    g.name,
    view.etaLabel === g.expect.eta_label,
    `etaLabel ${view.etaLabel} != ${g.expect.eta_label}`,
  );
  check(
    g.name,
    view.remainingLabel === g.expect.remaining_label,
    `remainingLabel ${view.remainingLabel} != ${g.expect.remaining_label}`,
  );
  check(
    g.name,
    view.totalLabel === g.expect.total_label,
    `totalLabel ${view.totalLabel} != ${g.expect.total_label}`,
  );
}

/* 2b — §7 kill switch: the production default drops the countdown chip ---- */

{
  check("kill-switch-default", ETA_BAND_ENABLED === false,
    "ETA_BAND_ENABLED must default false (backtest coverage 0.368/0.404 < 0.50)");
  const g = fixture.gating_vectors.find((x) => x.name === "fresh-matched-interpolates");
  const row = toRow(g.row);
  const snapshot = {
    generatedAt: g.generated_at,
    floors: fixture.stats.pct_floor_by_stage,
    tasks: { [row.issue]: row },
  };
  const map = buildProgressMap(snapshot, { [row.issue]: g.live_status }, Date.parse(g.now));
  const view = map[row.issue];
  check("kill-switch-no-label", view !== undefined && view.etaLabel === null,
    `default render must drop the band, got ${JSON.stringify(view)}`);
  check("kill-switch-bar-survives", view !== undefined && approx(view.pct, g.expect.pct),
    "the position bar must survive the chip drop");
  // The MEDIAN remaining/total labels are NOT gated by the band kill switch.
  check(
    "kill-switch-keeps-median-labels",
    view !== undefined &&
      view.remainingLabel === g.expect.remaining_label &&
      view.totalLabel === g.expect.total_label,
    `median labels must survive the kill switch, got ${JSON.stringify(view)}`,
  );
}

/* 3 — permissive snapshot narrowing -------------------------------------- */

{
  const empty = parseProgressSnapshot("");
  check("parse-empty", empty.generatedAt === null && Object.keys(empty.tasks).length === 0,
    "empty file should yield an empty snapshot");
  const garbage = parseProgressSnapshot("{ not json");
  check("parse-garbage", Object.keys(garbage.tasks).length === 0,
    "unparseable file should yield an empty snapshot, not a throw");
  const partial = parseProgressSnapshot(
    JSON.stringify({
      version: 1,
      generated_at: "2026-06-11T00:00:00Z",
      stats: { pct_floor_by_stage: { pooled: { running: 0.12 } } },
      tasks: {
        561: fixture.interpolate_vectors[0].row, // valid
        562: { issue: 562, stage: "running" }, // missing numerics → skipped
        563: "not-an-object", // skipped
      },
    }),
  );
  check("parse-partial", Object.keys(partial.tasks).length === 1 && partial.tasks[561] != null,
    `one valid row expected, got ${Object.keys(partial.tasks)}`);
  // Pre-upgrade snapshot row (no total_median_h): still narrows, total null.
  const legacyRow = { ...fixture.interpolate_vectors[0].row };
  delete legacyRow.total_p25_h;
  delete legacyRow.total_median_h;
  delete legacyRow.total_p75_h;
  const legacy = parseProgressSnapshot(
    JSON.stringify({
      version: 1,
      generated_at: "2026-06-11T00:00:00Z",
      tasks: { 561: legacyRow },
    }),
  );
  check(
    "parse-legacy-row-without-totals",
    legacy.tasks[561] != null && legacy.tasks[561].totalMedianH === null,
    `legacy row must narrow with totalMedianH null, got ${JSON.stringify(legacy.tasks[561])}`,
  );
}

if (failures > 0) {
  console.error(`progress.test.mjs: ${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("progress.test.mjs: all assertions passed");
