"""Thin CLI for the task-progress estimator (task #587).

Subcommands:

- ``snapshot [--force-stats]`` — rebuild ``~/.eps-autonomous/task_progress.json``
  (the 5-min summarize cron calls this; it is the ONLY snapshot writer).
- ``show --issue N`` — print one task's estimate + interpolated view as JSON
  (debugging / issue-tick spot checks). Uses snapshot stats when fresh, else
  builds stats inline (this is a debug tool, not the title path).
- ``backtest`` — one-time calibration check (plan §5 test 12 / §7 kill
  criterion): replays historical stage entries through the estimator and
  reports band coverage (fraction of realized clean-forward remaining times
  inside [p25, p75]) plus the remaining-band ratio p75/p25 at stage entry for
  positions {planning, running, verifying} x {experiment, code}. Coverage
  < 50% or any non-ignored ratio > 8 → drop the ETA chip (ship the bar).

Everything is strictly read-only over ``tasks/``; the only write is the
snapshot file under ``~/.eps-autonomous/`` (``snapshot`` subcommand only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from explore_persona_space import task_progress as tp


def _cmd_snapshot(args: argparse.Namespace) -> int:
    path = tp.write_snapshot(force_stats=args.force_stats)
    snap = json.loads(path.read_text())
    print(f"snapshot written: {path} ({len(snap.get('tasks', {}))} in-flight tasks)")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    stats = tp.load_stats_readonly() or tp.build_stage_stats()
    now = datetime.now(tz=UTC)
    row = tp.estimate_task_progress(args.issue, stats, now=now)
    if row is None:
        print(json.dumps({"issue": args.issue, "in_scope": False}))
        return 0
    pct, eta, overdue = tp.interpolate(row, now)
    out = {
        "row": row,
        "pct": pct,
        "eta": eta,
        "overdue": overdue,
        "title_suffix": tp.format_title_suffix(row, now),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_backtest(_args: argparse.Namespace) -> int:
    """Replay historical stage entries through the estimator.

    Procedure (pinned, plan §7): for every terminal-ish task, reconstruct its
    clean forward span per machine stage (first clean span per stage). For
    each stage entry, the predicted remaining band is
    ``[p25_cur + remaining_after_p25, p75_cur + remaining_after_p75]``
    (plan_pending excluded everywhere); the realized remaining time is the
    sum of the task's clean forward spans for stages at-or-after the entry
    (missing stages count 0 — skipped instantly). Coverage = fraction of
    realized values inside the band. NOTE: the stats include the replayed
    task's own spans (no leave-one-out) — a slight optimism acceptable for a
    one-time keep/drop check.
    """
    stats = tp.build_stage_stats()
    spans = tp.collect_stage_spans()

    # First clean span per (task, stage), keyed for chain reconstruction.
    per_task: dict[int, dict[str, float]] = {}
    per_task_bucket: dict[int, str] = {}
    for sp in spans:
        per_task.setdefault(sp["task_id"], {}).setdefault(sp["stage"], sp["dur_h"])
        per_task_bucket[sp["task_id"]] = sp["bucket"] or "pooled"

    stages = [s for s in tp.MACHINE_STAGES if s != "plan_pending"]
    inside = total = 0
    per_position: dict[str, list[int]] = {s: [] for s in stages}
    for task_id, stage_durs in per_task.items():
        bucket = per_task_bucket.get(task_id) or "pooled"
        cells = stats["buckets"].get(bucket) or stats["buckets"]["pooled"]
        for i, stage in enumerate(stages):
            if stage not in stage_durs:
                continue
            realized = sum(stage_durs.get(s, 0.0) for s in stages[i:])
            pred_lo = cells[stage]["p25_h"] + sum(cells[s]["p25_h"] for s in stages[i + 1 :])
            pred_hi = cells[stage]["p75_h"] + sum(cells[s]["p75_h"] for s in stages[i + 1 :])
            hit = int(pred_lo <= realized <= pred_hi)
            inside += hit
            total += 1
            per_position[stage].append(hit)

    print(f"calibration backtest: n={total} stage entries across {len(per_task)} tasks")
    coverage = inside / total if total else 0.0
    print(f"overall band coverage: {coverage:.3f} (kill criterion: < 0.50 drops the ETA chip)")
    for stage in stages:
        hits = per_position[stage]
        if hits:
            print(f"  {stage:<13} n={len(hits):>4} coverage={sum(hits) / len(hits):.3f}")

    print("remaining-band ratio p75/p25 at stage entry (ignore if width<1h or p25<0.25h):")
    any_exceeds = False
    for bucket in ("experiment", "code"):
        cells = stats["buckets"][bucket]
        for stage in ("planning", "running", "verifying"):
            i = stages.index(stage)
            lo = cells[stage]["p25_h"] + sum(cells[s]["p25_h"] for s in stages[i + 1 :])
            hi = cells[stage]["p75_h"] + sum(cells[s]["p75_h"] for s in stages[i + 1 :])
            ignored = (hi - lo) < 1.0 or lo < 0.25
            ratio = hi / lo if lo > 0 else float("inf")
            flag = " (ignored)" if ignored else (" EXCEEDS-8" if ratio > 8 else "")
            if not ignored and ratio > 8:
                any_exceeds = True
            print(f"  {bucket:<10} {stage:<10} band=[{lo:.2f},{hi:.2f}]h ratio={ratio:.2f}{flag}")
    verdict = "DROP CHIP" if (coverage < 0.5 or any_exceeds) else "KEEP CHIP"
    print(f"verdict: {verdict}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Task pipeline progress / ETA snapshot CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="write ~/.eps-autonomous/task_progress.json")
    sp.add_argument("--force-stats", action="store_true", help="rebuild stats even if fresh")
    sp.set_defaults(fn=_cmd_snapshot)

    sh = sub.add_parser("show", help="print one task's estimate as JSON")
    sh.add_argument("--issue", type=int, required=True)
    sh.set_defaults(fn=_cmd_show)

    bt = sub.add_parser("backtest", help="one-time calibration backtest (plan §7)")
    bt.set_defaults(fn=_cmd_backtest)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
