#!/usr/bin/env python
"""Issue #2479 — dual-prefix HF upload verification (plan §10; codex `hf-prefix-realized-vs-plan`).

The run's REALIZED persistence spans TWO parent prefixes on the HF data repo,
because the reused #1345 gen/capture rigs write under the PARENT's family
prefix while the issue's own phases write under the plan-declared one:

  issue1345_framing/char_2479_<name>[_op]/   (realized — the ported rigs)
    raw_completions/stories                  P1 kept+raw story bundles
    analysis_tensors/turnstore               P4 capture shards
    analysis_tensors/turnstore/_capture_complete.json   P4 completion marker
  issue2479_ai_likeness_gradient/            (plan §10-declared)
    eval_mirror/story_char_gradient          P5 fits/ladders eval mirror
    judge_legs                               P3 axis judge legs (raw draws)

An upload verification that enumerates only ONE parent prefix silently misses
the other family (#1773's shape). Every check uses the RAISING absence
primitive `hub.assert_hf_prefix_exists` (scoped `list_repo_tree`; a 404 is a
recorded FAIL, a transport error is retried — upload-policy § Absence
checks), never a full listing + client-side filter.

The upload-verifier (issue Step 7/8) RUNS this script; per-phase runs scope
the checked classes with --legs (e.g. before P5 lands, `--legs stories
turnstore markers`). Cells default to the full 24-cell panel grid; a bounded
run (yield-halted cells) passes the realized subset via --cells.

Writes a JSON report (per-check row: prefix, ok, n_files | error) and exits
non-zero on ANY missing prefix.
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

from explore_persona_space.orchestrate import hub  # noqa: E402

ISSUE = 2479
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
FAMILY_PREFIX = "issue2479_ai_likeness_gradient"
REPORT_REL = "eval_results/issue_2479/upload_verification_dual_prefix.json"
PER_CELL_LEGS = ("stories", "turnstore", "markers")
FAMILY_LEGS = ("eval_mirror", "judge_legs")
ALL_LEGS = PER_CELL_LEGS + FAMILY_LEGS


def panel_variants() -> list[str]:
    """All 24 panel cell variants (16 op + 8 inserted), registry order."""
    import issue2479_freeze_axis as fz

    variants: list[str] = []
    for r in fz.load_panel(_REPO_ROOT / fz.PANEL_REL):
        variants.append(r["variant_op"])
        if r.get("variant_inserted"):
            variants.append(r["variant_inserted"])
    return variants


def expected_prefixes(cells: list[str], legs: list[str]) -> list[tuple[str, str]]:
    """(leg, prefix) rows to verify — BOTH parent prefixes, never one family."""
    rows: list[tuple[str, str]] = []
    for v in cells:
        assert v.startswith("char_2479_"), f"not a #2479 panel cell: {v!r}"
        base = f"issue1345_framing/{v}"
        if "stories" in legs:
            rows.append(("stories", f"{base}/raw_completions/stories"))
        if "turnstore" in legs:
            rows.append(("turnstore", f"{base}/analysis_tensors/turnstore"))
        if "markers" in legs:
            rows.append(("markers", f"{base}/analysis_tensors/turnstore/_capture_complete.json"))
    if "eval_mirror" in legs:
        rows.append(("eval_mirror", f"{FAMILY_PREFIX}/eval_mirror/story_char_gradient"))
    if "judge_legs" in legs:
        rows.append(("judge_legs", f"{FAMILY_PREFIX}/judge_legs"))
    return rows


def verify(cells: list[str], legs: list[str], repo_id: str = HF_DATA_REPO) -> dict:
    """Run every (leg, prefix) check; returns the report payload."""
    from huggingface_hub import HfApi

    api = HfApi()
    checks: list[dict] = []
    for leg, prefix in expected_prefixes(cells, legs):
        try:
            n = hub.assert_hf_prefix_exists(api, repo_id, prefix, repo_type="dataset")
            checks.append({"leg": leg, "prefix": prefix, "ok": True, "n_files": int(n)})
            print(f"[verify] OK   {leg:12s} {prefix} ({n} files)", flush=True)
        except Exception as e:  # a 404 (or exhausted transport) is a recorded FAIL row
            checks.append(
                {"leg": leg, "prefix": prefix, "ok": False, "error": f"{type(e).__name__}: {e}"}
            )
            print(f"[verify] MISS {leg:12s} {prefix} — {type(e).__name__}", flush=True)
    n_missing = sum(1 for c in checks if not c["ok"])
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "issue": ISSUE,
        "repo": repo_id,
        "verdict": "PASS" if n_missing == 0 else "FAIL",
        "n_checks": len(checks),
        "n_missing": n_missing,
        "legs": list(legs),
        "cells": list(cells),
        "checks": checks,
        "metadata": {
            "script": "scripts/issue2479_verify_uploads.py",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **as_metadata_dict(git_provenance()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="realized panel cells (default: the full 24-cell registry grid)",
    )
    ap.add_argument(
        "--legs",
        nargs="+",
        choices=ALL_LEGS,
        default=list(ALL_LEGS),
        help="artifact classes to verify (scope per phase: before P5 lands, "
        "'--legs stories turnstore markers')",
    )
    ap.add_argument("--repo", default=HF_DATA_REPO)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / REPORT_REL)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from huggingface_hub import HfApi  # noqa: F401

        import issue2479_freeze_axis as fz  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        print("import-ok: issue2479_verify_uploads", flush=True)
        return

    cells = args.cells or panel_variants()
    report = verify(cells, list(args.legs), repo_id=args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(args.out)
    print(
        f"[verify] {report['verdict']}: {report['n_checks'] - report['n_missing']}/"
        f"{report['n_checks']} prefixes present -> {args.out}",
        flush=True,
    )
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
