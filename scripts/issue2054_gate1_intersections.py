"""GATE 1 — per-pair-class conversation-intersection floor evaluator + the
post-R2 pre-capture per-cell coverage assert (task #2054 follow-up
`coordinated-common-set-regen`, plan v12 §4/§7).

Two modes (``--mode``):

- ``gate1`` (R1→R2 boundary; BINDING — gates ALL downstream GPU spend):
  enumerates the affected ladder pairs with the LADDER'S OWN pair builders
  (`issue2054_ladder._pair_class`, imported — never re-implemented; the
  fail-open-(1) structural fix) over the 48-cell in-scope registry, computes
  every pair's conversation intersection given the survivor set S (every
  in-scope cell's pool IS S by construction — asserted, plan §8 "no affected
  pair crosses vintages"), census-checks the per-class pair counts against
  the committed artifacts (prose 96 / 2x2 208 via
  `ladder_intersection_composition.json`; the 32 chat→character pairs via
  `chat_to_character_pairs.json` `n_context_arm` — that class is ABSENT from
  the composition artifact), re-reports the untouched boundary/cross-model
  classes for the record, and writes ``gate1_report.json``.

  Verdict semantics (plan §7 gate 1; DESIGNED, artifact-routed halts —
  never bare rc=1):
    exit 0 — PASS (min affected-pair intersection >= 9,000)
    exit 8 — CONTINGENCY (in [4,480, 9,000): run AT MOST ONE wave 4)
    exit 9 — ABORT (< 4,480 after the contingency wave: no capture spend)

- ``coverage`` (post-R2, PRE-CAPTURE — the Methodology Must-Fix's second
  half): reads the realized phase_b/c/d digests and asserts, per regenerated
  cell, ``n_out == |S|`` (assistant on-policy extension legs:
  ``n_out == |delta|``) AND ``target_conv_ids >= 15,700`` in every
  target-capped digest. Any mismatch exits 9 (abort-before-R3) with the
  offending digest fields printed — the truncation-breach class gate 1
  structurally cannot see (the inherited default 8,000 caps cells to a
  deterministic first-N prefix).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_forms as forms  # noqa: E402

CHAR_VARIANTS = ("char_helios", "char_wren", "char_dana", "char_vex")
ASSISTANT_VARIANT = "conversation_paired_stories_assistant"
MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")
_CELLC_TAIL = {"qwen2.5-7b-instruct": "_op", "qwen2.5-7b": "_op_base"}

GATE1_TARGET = 9_000
GATE1_FLOOR = 4_480  # fits.KILL_GATE_4_MIN_INTERSECTION (plan §11)
MIN_TARGET_CONV_IDS = 15_700

EXIT_PASS = 0
EXIT_CONTINGENCY = 8
EXIT_ABORT = 9

AFFECTED_CLASSES = ("cross_character", "twobytwo")


def _log(msg: str) -> None:
    print(f"[phase=gate1] {msg}", flush=True)


def _utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
    os.replace(tmp, path)


def in_scope_cells() -> list[tuple[str, str, str, str]]:
    """The 48 regenerated/extended cells (plan §4/§5): 40 character cells +
    8 assistant template cells; validated through `forms.cell_key`."""
    cells: list[tuple[str, str, str, str]] = []
    for ch in CHAR_VARIANTS:
        for model in MODELS:
            for cond in ("inserted", "on_policy"):
                for form in ("attrib_quoted", "bare_label"):
                    cells.append((ch, cond, form, model))
            cells.append((f"{ch}{_CELLC_TAIL[model]}", "cell_c", "chat", model))
    for model in MODELS:
        for cond in ("inserted", "on_policy"):
            for form in ("chat", "bare_text"):
                cells.append((ASSISTANT_VARIANT, cond, form, model))
    assert len(cells) == 48, len(cells)
    for c in cells:
        forms.cell_key(*c)  # raises on any malformed axis
    return cells


def _axis_diff_class(s: tuple, t: tuple) -> str:
    """The composition artifact's own classification (analyzer fname_class):
    single-axis diffs named, everything else 2x2."""
    diffs = [i for i in range(4) if s[i] != t[i]]
    if diffs == [2]:
        return "cross_framing"
    if diffs == [3]:
        return "cross_model"
    if diffs == [0]:
        return "cross_character"
    return "twobytwo"


def _load_survivors(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(x) for x in (payload.get("survivor_conv_ids") or [])]
    if not ids:
        raise RuntimeError(f"survivor set EMPTY at {path}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"survivor set carries duplicates at {path}")
    return ids


def run_gate1(args) -> int:
    import issue2054_ladder as lad

    survivors = _load_survivors(Path(args.survivors))
    n_s = len(survivors)
    cells = in_scope_cells()
    scope = set(cells)

    pairs: list[tuple[tuple, tuple, str]] = []
    for s in cells:
        for t in cells:
            if s == t:
                continue
            cls = lad._pair_class(s, t)
            if cls in AFFECTED_CLASSES:
                pairs.append((s, t, cls))
    # Plan §8: no affected pair crosses vintages — every cell of every
    # affected pair must be in the regenerated/extended scope.
    for s, t, _cls in pairs:
        if s not in scope or t not in scope:
            raise RuntimeError(f"affected pair crosses vintages: {s} -> {t}")

    n_cross_char = sum(1 for _s, _t, c in pairs if c == "cross_character")
    n_twobytwo = sum(1 for _s, _t, c in pairs if c == "twobytwo")
    # Census sub-classes checked against the COMMITTED artifacts' own
    # classification conventions:
    #  - class_prose counts variant-only-diff pairs among STORY-form cells
    #    (the realized file set lacks the 24 cell_c<->cell_c pairs);
    #  - class_twobytwo counts the axis-diff "everything else" class;
    #  - chat->character = the 2x2 chat-anchor pairs onto STORY-form
    #    character targets (the census artifact's own description).
    n_prose_storyform = sum(
        1 for s, t, c in pairs if c == "cross_character" and s[1] != "cell_c" and t[1] != "cell_c"
    )
    n_twobytwo_axisdiff = sum(1 for s, t, _c in pairs if _axis_diff_class(s, t) == "twobytwo")
    chat_to_char = [
        (s, t)
        for s, t, c in pairs
        if c == "twobytwo"
        and lad._is_chat_anchor(s)
        and str(t[0]).startswith("char_")
        and t[2] in forms.STORY_FORMS
    ]

    composition = json.loads(Path(args.composition).read_text(encoding="utf-8"))
    chat_census = json.loads(Path(args.chat_census).read_text(encoding="utf-8"))
    census_checks = {
        "prose_storyform_pairs": {
            "enumerated": n_prose_storyform,
            "expected": int((composition.get("class_prose") or {}).get("n") or -1),
        },
        "twobytwo_axisdiff_pairs": {
            "enumerated": n_twobytwo_axisdiff,
            "expected": int((composition.get("class_twobytwo") or {}).get("n") or -1),
        },
        "chat_to_character_pairs": {
            "enumerated": len(chat_to_char),
            "expected": int(chat_census.get("n_context_arm") or -1),
        },
    }
    mismatches = {k: v for k, v in census_checks.items() if v["enumerated"] != v["expected"]}
    if mismatches:
        # Assumption 18: a census mismatch is a DRIVER BUG — fail loud, never
        # a gate verdict.
        raise RuntimeError(f"pair-census mismatch vs committed artifacts: {mismatches}")

    # Per-pair intersection: every in-scope cell's realized pool is S by
    # construction (regenerated on S; assistant cells extended to full S), so
    # each affected pair's conversation intersection is exactly |S|.
    min_intersection = n_s
    if min_intersection >= GATE1_TARGET:
        verdict, rc = "PASS", EXIT_PASS
    elif min_intersection >= GATE1_FLOOR:
        verdict, rc = "CONTINGENCY_WAVE", EXIT_CONTINGENCY
    else:
        verdict, rc = "ABORT", EXIT_ABORT

    report = {
        "artifact": "gate1_report",
        "n_survivors": n_s,
        "min_affected_pair_intersection": min_intersection,
        "gate1_target": GATE1_TARGET,
        "gate1_floor": GATE1_FLOOR,
        "verdict": verdict,
        "exit_code": rc,
        "n_in_scope_cells": len(cells),
        "affected_pairs": {
            "cross_character": n_cross_char,
            "twobytwo": n_twobytwo,
            "total": len(pairs),
        },
        "census_checks": census_checks,
        "chat_to_character_subset": [
            {"src": forms.cell_key(*s), "tgt": forms.cell_key(*t)} for s, t in chat_to_char
        ],
        "record_only_classes": {
            "note": "untouched boundary/cross-model classes re-reported from the realized "
            "composition artifact (expected unchanged >= 7,999 conv intersections)",
            "class_boundary_n": (composition.get("class_boundary") or {}).get("n"),
            "class_model_n": (composition.get("class_model") or {}).get("n"),
            "intersections_record": composition.get("intersections"),
        },
        "survivors_path": str(args.survivors),
        "utc": _utc(),
    }
    out = Path(args.report_out)
    _atomic_write_json(out, report)
    _log(
        f"verdict={verdict} min_intersection={min_intersection} "
        f"(target {GATE1_TARGET}, floor {GATE1_FLOOR}) pairs={len(pairs)} -> {out}"
    )
    return rc


# ---------------------------------------------------------------------------
# Mode: coverage (post-R2, pre-capture)
# ---------------------------------------------------------------------------
def _collect_digests(root: Path, stem: str) -> list[Path]:
    return sorted(root.rglob(f"{stem}*.json")) if root.is_dir() else []


def run_coverage(args) -> int:
    survivors = set(_load_survivors(Path(args.survivors)))
    n_s = len(survivors)
    n_delta = int(args.assistant_delta_n)
    failures: list[str] = []
    checked: list[dict] = []

    def _check_digest(path: Path, phase: str) -> None:
        d = json.loads(path.read_text(encoding="utf-8"))
        target = d.get("target_conv_ids")
        if phase in ("phase_c", "phase_d"):
            if not isinstance(target, int) or target < MIN_TARGET_CONV_IDS:
                failures.append(
                    f"{path}: target_conv_ids={target!r} < {MIN_TARGET_CONV_IDS} — the "
                    "inherited default 8,000 truncates cells to a first-N prefix"
                )
        counts = d.get("counts") or {}
        for variant, rec in counts.items():
            n_out = int((rec or {}).get("n_out") or 0)
            if phase == "phase_c" and variant == ASSISTANT_VARIANT:
                expected = n_delta
                label = "assistant on-policy delta"
            elif str(variant).startswith("char_") or variant == ASSISTANT_VARIANT:
                expected = n_s
                label = "|S|"
            else:
                failures.append(f"{path}: unknown variant {variant!r} in counts")
                continue
            checked.append(
                {
                    "digest": str(path),
                    "phase": phase,
                    "variant": variant,
                    "n_out": n_out,
                    "expected": expected,
                }
            )
            if n_out != expected:
                failures.append(f"{path}: {variant} n_out={n_out} != expected {label}={expected}")

    roots = {
        "phase_b": Path(args.phase_b_dir),
        "phase_c": Path(args.phase_c_dir),
        "phase_d": Path(args.phase_d_dir),
    }
    n_digests = 0
    for phase, root in roots.items():
        digests = _collect_digests(root, f"{phase}_digest")
        if not digests:
            failures.append(f"no {phase} digests found under {root} — phase never ran?")
        for p in digests:
            n_digests += 1
            _check_digest(p, phase)

    report = {
        "artifact": "regen_precapture_coverage",
        "n_survivors": n_s,
        "assistant_delta_n": n_delta,
        "min_target_conv_ids": MIN_TARGET_CONV_IDS,
        "n_digests_checked": n_digests,
        "n_cells_checked": len(checked),
        "cells": checked,
        "failures": failures,
        "verdict": "PASS" if not failures else "ABORT_BEFORE_CAPTURE",
        "utc": _utc(),
    }
    out = Path(args.coverage_report_out)
    _atomic_write_json(out, report)
    if failures:
        for f in failures:
            print(f"[phase=gate1] COVERAGE FAIL: {f}", file=sys.stderr, flush=True)
        _log(f"coverage ABORT ({len(failures)} failure(s)) -> {out}")
        return EXIT_ABORT
    _log(f"coverage PASS ({len(checked)} cell rows across {n_digests} digests) -> {out}")
    return EXIT_PASS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        default=None,
        choices=("gate1", "coverage"),
        help="REQUIRED except under --import-check",
    )
    p.add_argument(
        "--survivors",
        default="data/issue_2054/common_regen/scaffolds/survivor_set.json",
        help="survivor_set.json from the wave driver",
    )
    p.add_argument(
        "--composition",
        default="eval_results/issue_2054/analyzer_companions/ladder_intersection_composition.json",
    )
    p.add_argument(
        "--chat-census",
        default="eval_results/issue_2054/analyzer_companions/chat_to_character_pairs.json",
    )
    p.add_argument(
        "--report-out",
        default="eval_results/issue_2054/coordinated_common_set_regen/gate1_report.json",
    )
    p.add_argument("--phase-b-dir", default="data/issue_2054/common_regen/spliced_inserted")
    p.add_argument("--phase-c-dir", default="data/issue_2054/common_regen/on_policy")
    p.add_argument("--phase-d-dir", default="data/issue_2054/common_regen/cell_c")
    p.add_argument(
        "--assistant-delta-n",
        type=int,
        default=-1,
        help="coverage mode: expected assistant on-policy delta row count (from the "
        "export manifest); REQUIRED there",
    )
    p.add_argument(
        "--coverage-report-out",
        default="eval_results/issue_2054/coordinated_common_set_regen/precapture_coverage.json",
    )
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args()

    if args.import_check:
        import issue2054_ladder  # noqa: F401
        import issue2054_regen_waves as rw

        rw.assert_args_attrs_defined(__file__)
        print("[phase=gate1] import-check OK", flush=True)
        return 0

    if args.mode == "gate1":
        return run_gate1(args)
    if args.assistant_delta_n < 0:
        p.error("--mode coverage requires --assistant-delta-n >= 0")
    return run_coverage(args)


if __name__ == "__main__":
    sys.exit(main())
