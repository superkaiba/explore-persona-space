#!/usr/bin/env python
"""Issue #2479 — validated axis-leg resume predicate (r2 codex `p3-leg-resume-unvalidated`).

The P3 wrapper previously skipped a character's axis judge leg on BARE report
EXISTENCE, so a dry-run report, an old-rubric report, or a report judged over
a different item set silently satisfied the skip — and the freeze step's
fail-loud rejects (`issue2479_freeze_axis.load_leg_report` + the rubric-drift
assert) then made that bad report a deterministic rerun WEDGE (the freeze
crashes on every wrapper run and nothing ever re-dispatches the leg).

This validator is the wrapper's per-character resume predicate:

  exit 0  the persisted leg report satisfies the COMPLETION predicate — skip
          dispatch:
            * spend_executed is True (a dry-run report certifies nothing);
            * leg == ai_likeness and tag matches;
            * instrument fields match the CURRENT production constants
              (judge_model / n_draws / temperature / threshold_base;
              max_tokens >= the floor) and rubric_sha256 ==
              freeze_axis.rubric_fingerprint() (the freeze's own drift check);
            * means.pooled present (the freeze's schema check);
            * with --items: the save_raw draws' item-ID set EXACTLY matches
              the newly emitted axis item set (panel/manifest/item drift all
              surface here — p3_items re-emits before p3_legs every run);
            * with --pilot-report: the persisted axis pilot PASS binds to the
              current instrument (issue2479_judge_pilots.require_pilot_pass).
  exit 3  DISPATCH the leg: the report is absent, OR it failed validation and
          was QUARANTINED (renamed `<file>.quarantined-<UTCts>`, save_raw
          alongside) so it can never wedge the freeze — never silently
          reused, never a permanent wedge. Re-dispatch cost is bounded by the
          rubric-keyed judge cache. A pilot-binding failure dispatches
          WITHOUT quarantine (the leg report is intact; `jl.run_leg`'s own
          env-armed guard enforces the pilot at spend time).
  other   a real error (traceback) — the wrapper aborts.

Content hygiene: axis item rows are LMSYS-derived real user text — this
validator prints item IDs, counts, and paths only, never row text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_freeze_axis as fz  # noqa: E402
import issue2479_judge_pilots as jp  # noqa: E402

EXIT_VALID = 0
EXIT_DISPATCH = 3


def validation_failures(report: dict, tag: str) -> list[str]:
    """Completion-predicate failures of a parsed leg report (empty == valid)."""
    fails: list[str] = []
    checks = (
        ("leg", jl.LEG_AI_LIKENESS),
        ("tag", tag),
        ("judge_model", jl.JUDGE_MODEL),
        ("n_draws", jl.N_DRAWS),
        ("temperature", jl.JUDGE_TEMPERATURE),
        ("threshold_base", jl.THRESHOLD_BASE_FORCE_BATCH),
        ("rubric_sha256", fz.rubric_fingerprint()),
    )
    for key, want in checks:
        got = report.get(key)
        if got != want:
            fails.append(f"{key}: report={got!r} != current={want!r}")
    if report.get("spend_executed") is not True:
        fails.append("spend_executed is not True — a dry-run report cannot satisfy the skip")
    mt = report.get("max_tokens")
    if not isinstance(mt, int) or mt < jl.JUDGE_MAX_TOKENS:
        fails.append(f"max_tokens: report={mt!r} below the floor {jl.JUDGE_MAX_TOKENS}")
    means = report.get("means")
    if not (isinstance(means, dict) and isinstance(means.get("pooled"), dict)):
        fails.append("means.pooled absent — incomplete leg report")
    return fails


def save_raw_path(report_path: Path, tag: str) -> Path:
    """The leg's save_raw sibling (`run_leg` writes both into one out dir)."""
    return report_path.parent / f"judge_raw_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{tag}.json"


def item_set_failures(report_path: Path, tag: str, items_path: Path) -> list[str]:
    """Exact item-ID set match: newly emitted items vs the save_raw draw ids.

    save_raw `all_scores` keys are per-DRAW custom ids
    `<item_id>__<draw>__<suffix>`; `rsplit("__", 2)[0]` recovers the item id
    (the right-anchored decode `judge_result_from_save_raw` itself uses).
    """
    rows = c.read_jsonl(items_path)
    expected = {jl.item_id(jl.LEG_AI_LIKENESS, tag, str(r["conv_id"])) for r in rows}
    raw_path = save_raw_path(report_path, tag)
    if not raw_path.is_file():
        return [f"save_raw missing beside report: {raw_path}"]
    raw = json.loads(raw_path.read_text())
    got = {str(k).rsplit("__", 2)[0] for k in (raw.get("all_scores") or {})}
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        return [
            f"item-ID set mismatch vs {items_path.name}: missing={len(missing)} "
            f"extra={len(extra)} (e.g. missing {missing[:3]} extra {extra[:3]})"
        ]
    return []


def quarantine(paths: list[Path]) -> list[Path]:
    """Rename each existing file aside (`.quarantined-<UTCts>`); return moves."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    moved: list[Path] = []
    for p in paths:
        if p.is_file():
            dest = p.with_name(f"{p.name}.quarantined-{ts}")
            p.rename(dest)
            moved.append(dest)
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--report", type=Path, help="judge_report_ail_<tag>.json path (required)")
    ap.add_argument("--tag", help="leg tag: character name, or flat_/mask_<name> (required)")
    ap.add_argument(
        "--items",
        type=Path,
        default=None,
        help="freshly emitted axis_items_<tag>.jsonl — arms the exact item-ID set check",
    )
    ap.add_argument(
        "--pilot-report",
        type=Path,
        default=None,
        help="axis-family pilot report — arms the current-instrument pilot binding check",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("import-ok: issue2479_p3_leg_resume", flush=True)
        return 0

    assert args.report is not None and args.tag, "--report and --tag are required"
    report_path: Path = args.report
    tag: str = args.tag
    if not report_path.is_file():
        print(f"[leg-resume] tag={tag} DISPATCH — no leg report at {report_path}", flush=True)
        return EXIT_DISPATCH

    try:
        report = json.loads(report_path.read_text())
        fails = validation_failures(report, tag)
    except json.JSONDecodeError as e:
        fails = [f"malformed report JSON: {e}"]
    if not fails and args.items is not None:
        fails = item_set_failures(report_path, tag, args.items)
    if fails:
        moved = quarantine([report_path, save_raw_path(report_path, tag)])
        for f in fails:
            print(f"[leg-resume] tag={tag} INVALID — {f}", flush=True)
        print(
            f"[leg-resume] tag={tag} DISPATCH — quarantined: {[str(m) for m in moved]}",
            flush=True,
        )
        return EXIT_DISPATCH

    if args.pilot_report is not None:
        try:
            jp.require_pilot_pass(args.pilot_report, family="axis")
        except RuntimeError as e:
            # The leg report itself is intact — no quarantine. The wrapper's
            # p3_pilot phase re-pilots first; `jl.run_leg`'s env-armed guard
            # enforces the pilot again at spend time (defense in depth).
            print(f"[leg-resume] tag={tag} DISPATCH — pilot binding failed: {e}", flush=True)
            return EXIT_DISPATCH

    print(f"[leg-resume] tag={tag} VALID — skip dispatch (resume)", flush=True)
    return EXIT_VALID


if __name__ == "__main__":
    raise SystemExit(main())
