#!/usr/bin/env python
"""Issue #2479 — validated axis/control-leg resume predicate (r2+r4 codex
`p3-leg-resume-unvalidated`).

The P3 wrapper previously skipped a character's axis judge leg on BARE report
EXISTENCE, so a dry-run report, an old-rubric report, or a report judged over
a different item set silently satisfied the skip — and the freeze step's
fail-loud rejects (`issue2479_freeze_axis.load_leg_report` + the rubric-drift
assert) then made that bad report a deterministic rerun WEDGE (the freeze
crashes on every wrapper run and nothing ever re-dispatches the leg).

This validator is the wrapper's per-leg resume predicate — axis legs AND the
flatness/name-mask control legs — binding the skip to FULL INPUT IDENTITY:

  exit 0  the persisted leg report satisfies the COMPLETION predicate — skip
          dispatch:
            * spend_executed is True (a dry-run report certifies nothing);
            * leg == ai_likeness and tag matches;
            * instrument fields match the CURRENT production constants
              (judge_model / n_draws / temperature / threshold_base;
              max_tokens >= the floor) and rubric_sha256 ==
              freeze_axis.rubric_fingerprint() (the freeze's own drift check);
            * means.pooled present (the freeze's schema check);
            * `items_content_sha256` present (run_leg records it over the
              dispatched (item_id, question, answer) triples);
            * the persisted DESIGN sidecar (judge_sample_ail_<tag>.json)
              exists, its conv_id set EXACTLY matches the save_raw draw item
              ids, and every item carries exactly N_DRAWS raw draws (the
              per-item draw census — a partial raw file never satisfies the
              skip);
            * with --expect-design: the design matches the REGISTERED control
              design (flat: n_target==FLAT_N, seed==SUBSAMPLE_SEED,
              common_draw; mask: n_target==MASK_N, seed==SUBSAMPLE_SEED;
              axis-census: census==True) — constants read from the SAME
              modules the dispatchers use, never re-typed;
            * with --items: the save_raw item-ID set EXACTLY matches the
              freshly emitted/derived item rows AND the report's
              items_content_sha256 matches a recompute over the fresh rows
              (same conv_ids with CHANGED question/answer text surface here).
              Axis legs pass freeze_axis --emit-items output; control legs
              pass issue2479_instrument_gates --step emit-control-items
              output (the SAME derivation the dispatch steps run), so a
              stale-but-self-consistent flat/mask report/raw/design triple
              can no longer skip (r5 p3-leg-resume-unvalidated);
            * with --pilot-report: the persisted axis pilot PASS binds to the
              current instrument + data identity
              (issue2479_judge_pilots.require_pilot_pass), AND the leg
              report's recorded `licensing_pilot` fingerprint (instrument +
              data_identity) matches the CURRENT pilot's — so a retained
              report is provably licensed by an equivalent pilot even though
              the wrapper refreshes the pilot before leg validation.
  exit 3  DISPATCH the leg: the report is absent, OR it failed validation and
          was QUARANTINED (report + save_raw + design renamed
          `<file>.quarantined-<UTCts>-p<pid>-<seq>` — pid + a process-local
          counter make concurrent same-second quarantines collision-free) so
          it can never wedge the freeze — never silently reused, never a
          permanent wedge. Re-dispatch cost is bounded by the rubric-keyed
          judge cache. A CURRENT-pilot-binding failure dispatches WITHOUT
          quarantine (the leg report is intact; the wrapper's p3_pilot phase
          re-pilots and `jl.run_leg`'s env-armed guard enforces the pilot at
          spend time); a LICENSING mismatch (the retained report was licensed
          by a different-fingerprint pilot) quarantines.
  other   a real error (traceback) — the wrapper aborts.

Content hygiene: axis item rows are LMSYS-derived real user text — this
validator prints item IDs, counts, hashes, and paths only, never row text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
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
EXPECT_DESIGN_CHOICES = ("axis-census", "flat", "mask")
_QUARANTINE_SEQ = itertools.count()


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
    ics = report.get("items_content_sha256")
    if not (isinstance(ics, str) and len(ics) == 64):
        fails.append(
            "items_content_sha256 absent/malformed — the report does not bind the judged "
            "item CONTENT (r4 p3-leg-resume-unvalidated)"
        )
    return fails


def save_raw_path(report_path: Path, tag: str) -> Path:
    """The leg's save_raw sibling (`run_leg` writes both into one out dir)."""
    return report_path.parent / f"judge_raw_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{tag}.json"


def design_path(report_path: Path, tag: str) -> Path:
    """The leg's persisted sample-design sibling (`run_leg` writes it pre-dispatch)."""
    return report_path.parent / f"judge_sample_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{tag}.json"


def _save_raw_draw_counts(report_path: Path, tag: str) -> tuple[dict[str, int] | None, str | None]:
    """(per-item draw counts from save_raw, failure) — one of the two is None.

    save_raw `all_scores` keys are per-DRAW custom ids
    `<item_id>__<draw>__<suffix>`; `rsplit("__", 2)[0]` recovers the item id
    (the right-anchored decode `judge_result_from_save_raw` itself uses).
    """
    raw_path = save_raw_path(report_path, tag)
    if not raw_path.is_file():
        return None, f"save_raw missing beside report: {raw_path}"
    raw = json.loads(raw_path.read_text())
    counts: dict[str, int] = {}
    for k in raw.get("all_scores") or {}:
        iid = str(k).rsplit("__", 2)[0]
        counts[iid] = counts.get(iid, 0) + 1
    return counts, None


def _registered_design_failures(design: dict, expect_design: str) -> list[str]:
    """The design must match the REGISTERED sample design for its leg family.

    Constants come from the SAME modules the dispatchers draw with
    (`issue2479_instrument_gates` for the control legs, the wrapper's
    `--census` dispatch for axis legs) — never re-typed literals.
    """
    import issue2479_instrument_gates as ig

    fails: list[str] = []
    if expect_design == "flat":
        expected = {"n_target": ig.FLAT_N, "seed": ig.SUBSAMPLE_SEED, "common_draw": True}
    elif expect_design == "mask":
        expected = {"n_target": ig.MASK_N, "seed": ig.SUBSAMPLE_SEED}
    else:  # axis-census
        expected = {"census": True}
    for key, want in expected.items():
        got = design.get(key)
        if got != want:
            fails.append(
                f"design.{key}: persisted={got!r} != registered={want!r} "
                f"(expect-design={expect_design})"
            )
    return fails


def design_and_census_failures(report_path: Path, tag: str, expect_design: str | None) -> list[str]:
    """Design-sidecar + per-item draw-census failures (empty == valid).

    Binds the skip to the persisted sample design: the design's conv_id set
    must EXACTLY match the save_raw draw item ids, and every item must carry
    exactly N_DRAWS raw draws — a partial raw file, a foreign draw, or a
    design/raw divergence re-dispatches (through the judge cache, ~zero cost).
    """
    dp = design_path(report_path, tag)
    if not dp.is_file():
        return [f"sample design missing beside report: {dp}"]
    try:
        design = json.loads(dp.read_text())
    except json.JSONDecodeError as e:
        return [f"malformed design JSON: {e}"]
    conv_ids = [str(x) for x in design.get("conv_ids") or []]
    if not conv_ids:
        return [f"{dp.name}: design carries no conv_ids"]
    fails: list[str] = []
    if expect_design is not None:
        fails.extend(_registered_design_failures(design, expect_design))
    expected_ids = {jl.item_id(jl.LEG_AI_LIKENESS, tag, cid) for cid in conv_ids}
    counts, raw_fail = _save_raw_draw_counts(report_path, tag)
    if raw_fail is not None:
        fails.append(raw_fail)
        return fails
    assert counts is not None
    got_ids = set(counts)
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        fails.append(
            f"design/save_raw item-ID divergence: missing={len(missing)} extra={len(extra)} "
            f"(e.g. missing {missing[:3]} extra {extra[:3]})"
        )
        return fails
    short = {iid: n for iid, n in counts.items() if n != jl.N_DRAWS}
    if short:
        sample = sorted(short.items())[:3]
        fails.append(
            f"per-item draw census failed: {len(short)}/{len(counts)} items do not carry "
            f"exactly {jl.N_DRAWS} raw draws (e.g. {sample}) — partial raw files never "
            "satisfy the skip"
        )
    return fails


def item_failures(report: dict, report_path: Path, tag: str, items_path: Path) -> list[str]:
    """Fresh-item binding: exact ID set AND item CONTENT vs the emitted file.

    The ID set catches panel/manifest/item-set drift (p3_items re-emits before
    p3_legs every run); the content fingerprint catches SAME-ID re-generation
    (unchanged conv_ids with changed question/answer text), recomputed through
    the same `build_ai_likeness_items` + `items_content_fingerprint` helpers
    `run_leg` recorded with.
    """
    rows = c.read_jsonl(items_path)
    expected = {jl.item_id(jl.LEG_AI_LIKENESS, tag, str(r["conv_id"])) for r in rows}
    counts, raw_fail = _save_raw_draw_counts(report_path, tag)
    if raw_fail is not None:
        return [raw_fail]
    assert counts is not None
    got = set(counts)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        return [
            f"item-ID set mismatch vs {items_path.name}: missing={len(missing)} "
            f"extra={len(extra)} (e.g. missing {missing[:3]} extra {extra[:3]})"
        ]
    fresh_fp = jl.items_content_fingerprint(jl.build_ai_likeness_items(rows, tag))
    if report.get("items_content_sha256") != fresh_fp:
        return [
            f"item CONTENT drift vs {items_path.name}: report items_content_sha256="
            f"{str(report.get('items_content_sha256'))[:16]}… != fresh recompute "
            f"{fresh_fp[:16]}… (same conv_ids do not prove same question/answer text)"
        ]
    return []


def licensing_failures(report: dict, current_pilot: dict) -> list[str]:
    """The retained report must be licensed by a CURRENT-fingerprint pilot.

    The wrapper refreshes the pilot BEFORE leg validation, so ordering alone
    cannot prove which pilot licensed a retained report (r4 codex sequencing
    note): `run_leg` records the licensing pilot's instrument + data_identity
    in the report, and this check requires them to EQUAL the current pilot's —
    a refresh at the same fingerprints is equivalence; a refresh because the
    instrument or materialization changed is a mismatch that re-dispatches.
    """
    lp = report.get("licensing_pilot")
    if not isinstance(lp, dict):
        return [
            "licensing_pilot absent — the report does not prove which pilot licensed its "
            "spend (r4 p3-leg-resume-unvalidated)"
        ]
    fails: list[str] = []
    for key in ("instrument", "data_identity"):
        if lp.get(key) != current_pilot.get(key):
            fails.append(
                f"licensing_pilot.{key} != the CURRENT pilot's {key} — the retained report "
                "was licensed under a different pilot fingerprint"
            )
    return fails


def quarantine(paths: list[Path]) -> list[Path]:
    """Rename each existing file aside; return the moves.

    Names carry UTC-seconds + pid + a process-local counter
    (`.quarantined-<ts>-p<pid>-<seq>`), so two quarantines of the same leg
    within one second — same process or concurrent processes — can never
    overwrite earlier evidence (r4 codex quarantine-collision nit).
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    moved: list[Path] = []
    for p in paths:
        if p.is_file():
            dest = p.with_name(f"{p.name}.quarantined-{ts}-p{os.getpid()}-{next(_QUARANTINE_SEQ)}")
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
        help="freshly emitted/derived item rows — arms the exact item-ID set + item CONTENT "
        "checks (axis legs: freeze_axis --emit-items output; control legs: "
        "issue2479_instrument_gates --step emit-control-items output)",
    )
    ap.add_argument(
        "--expect-design",
        choices=EXPECT_DESIGN_CHOICES,
        default=None,
        help="registered sample design this leg must have been drawn under "
        "(flat: FLAT_N/seed/common-draw; mask: MASK_N/seed; axis-census: census)",
    )
    ap.add_argument(
        "--pilot-report",
        type=Path,
        default=None,
        help="axis-family pilot report — arms the current-instrument+data pilot binding "
        "check AND the licensing-pilot provenance match",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred imports on the real code paths, named explicitly (#1689).
        import issue2479_instrument_gates as ig  # noqa: F401

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
        report = {}
        fails = [f"malformed report JSON: {e}"]
    if not fails:
        fails = design_and_census_failures(report_path, tag, args.expect_design)
    if not fails and args.items is not None:
        fails = item_failures(report, report_path, tag, args.items)
    if not fails and args.pilot_report is not None:
        try:
            current_pilot = jp.require_pilot_pass(args.pilot_report, family="axis")
        except RuntimeError as e:
            # The leg report itself is intact — no quarantine. The wrapper's
            # p3_pilot phase re-pilots first; `jl.run_leg`'s env-armed guard
            # enforces the pilot again at spend time (defense in depth).
            print(f"[leg-resume] tag={tag} DISPATCH — pilot binding failed: {e}", flush=True)
            return EXIT_DISPATCH
        fails = licensing_failures(report, current_pilot)
    if fails:
        moved = quarantine(
            [report_path, save_raw_path(report_path, tag), design_path(report_path, tag)]
        )
        for f in fails:
            print(f"[leg-resume] tag={tag} INVALID — {f}", flush=True)
        print(
            f"[leg-resume] tag={tag} DISPATCH — quarantined: {[str(m) for m in moved]}",
            flush=True,
        )
        return EXIT_DISPATCH

    print(f"[leg-resume] tag={tag} VALID — skip dispatch (resume)", flush=True)
    return EXIT_VALID


if __name__ == "__main__":
    raise SystemExit(main())
