"""Issue #568 Phase -1 — deterministic third-pair selection over the committed #527 matrix.

Plan §4 "The ONE experimental variable". Re-ranks all C(18,2) pairs over
``PERSONA_POOL_19`` by |base-model L20 centered cosine| read from the
COMMITTED ``eval_results/issue_527/pair_selection.json`` matrix (no fresh
forward pass — the new pair's cosine stays on the same matrix/scale as the
two existing pairs'), applies the plan's two exclusions, picks the
smallest-|cos| survivor, and HARD-ASSERTS the pick equals the
pre-registered pair before writing the output.

Exclusions (plan §4 selection procedure, in order):
  (a) pairs intersecting the four existing source personas
      {florist, medical_doctor, librarian, police_officer} — the task's
      disjointness requirement;
  (b) pairs containing ``NEGATIVE_PANEL_REPLACEMENT`` (kindergarten_teacher)
      — ``negative_panel_for_pair`` HARD-RAISES whenever the replacement is
      itself a source, and changing that guard would bundle a second
      mechanism change into a one-variable design.

Output: ``eval_results/issue_568/pair_selection.json`` in the
``issue_527_pair_selection_v1`` schema (the inherited dispatcher's
``_load_pair_selection`` pins that READ schema string — cosmetic lineage
provenance, do not "fix"), carrying the parent matrix verbatim plus an
``issue_568_provenance`` block.

CLI (plan §4 Phase -1; runs on the VM, CPU, pre-provision):
    uv run python scripts/run_issue568_pair_selection.py \\
      --matrix eval_results/issue_527/pair_selection.json \\
      --exclude-sources florist medical_doctor librarian police_officer \\
      --out eval_results/issue_568/pair_selection.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_538 import (
    NEGATIVE_PANEL_REPLACEMENT,
    PERSONA_POOL_19,
)

log = logging.getLogger("issue_568.pair_selection")

# ── Pre-registered pick (plan §4) ────────────────────────────────────────────
# Full-precision cosine read from the committed #527 matrix at plan time. The
# plan prose quotes the 5-decimal gloss +0.01495; the assert below uses the
# full-precision matrix value so the ±1e-6 tolerance is actually satisfiable
# (|0.0149529753 - 0.01495| ≈ 3e-6 would false-fail the 5-decimal gloss).
EXPECTED_NAME_A = "navy_seal"
EXPECTED_NAME_B = "french_person"
EXPECTED_PAIR_ID = f"{EXPECTED_NAME_A}__{EXPECTED_NAME_B}"
EXPECTED_COS = 0.014952975325286388
COS_TOL = 1e-6

DEFAULT_EXCLUDE_SOURCES = ("florist", "medical_doctor", "librarian", "police_officer")
READ_SCHEMA = "issue_527_pair_selection_v1"


def _git_commit() -> str:
    """Best-effort HEAD sha for reproducibility metadata."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def rank_candidate_pairs(
    matrix_payload: dict,
    *,
    exclude_sources: set[str],
) -> list[dict]:
    """Rank all C(n,2) pairs over PERSONA_POOL_19 by |centered cos|, tagging exclusions.

    Returns a list of dicts sorted by ``abs_cos`` ascending; each entry carries
    ``pair_id, name_a, name_b, base_cos_centered_L20, abs_cos, excluded_by``
    (``None`` for eligible pairs). Raises if any pool persona is missing from
    the matrix.
    """
    names = matrix_payload["persona_names"]
    cos = matrix_payload["cos_centered_L20"]
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [n for n in PERSONA_POOL_19 if n not in name_to_idx]
    if missing:
        raise AssertionError(
            f"PERSONA_POOL_19 member(s) {missing} missing from the matrix persona_names "
            f"({names}); the committed #527 matrix is not a valid selector input."
        )

    ranked: list[dict] = []
    for a, b in itertools.combinations(PERSONA_POOL_19, 2):
        c = float(cos[name_to_idx[a]][name_to_idx[b]])
        excluded_by: str | None = None
        if {a, b} & exclude_sources:
            excluded_by = "intersects_existing_sources"
        elif NEGATIVE_PANEL_REPLACEMENT in (a, b):
            excluded_by = "contains_negative_panel_replacement"
        ranked.append(
            {
                "pair_id": f"{a}__{b}",
                "name_a": a,
                "name_b": b,
                "base_cos_centered_L20": c,
                "abs_cos": abs(c),
                "excluded_by": excluded_by,
            }
        )
    ranked.sort(key=lambda e: e["abs_cos"])
    return ranked


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--matrix",
        default="eval_results/issue_527/pair_selection.json",
        help="Committed #527 pair-selection JSON carrying the full cosine matrix.",
    )
    ap.add_argument(
        "--exclude-sources",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_SOURCES),
        help="Existing source personas the new pair must be disjoint from.",
    )
    ap.add_argument(
        "--out",
        default="eval_results/issue_568/pair_selection.json",
        help="Output JSON path (issue_527_pair_selection_v1 schema, 1 picked pair).",
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix)
    payload = json.loads(matrix_path.read_text())
    if payload.get("schema_version") != READ_SCHEMA:
        raise AssertionError(
            f"{matrix_path} schema_version mismatch "
            f"(got {payload.get('schema_version')!r}, expected {READ_SCHEMA!r})"
        )

    exclude_sources = set(args.exclude_sources)
    ranked = rank_candidate_pairs(payload, exclude_sources=exclude_sources)

    log.info("Ranked head (smallest |cos| first):")
    for entry in ranked[:8]:
        log.info(
            "  %-45s cos=%+.5f  %s",
            entry["pair_id"],
            entry["base_cos_centered_L20"],
            f"EXCLUDED ({entry['excluded_by']})" if entry["excluded_by"] else "eligible",
        )

    eligible = [e for e in ranked if e["excluded_by"] is None]
    if not eligible:
        raise RuntimeError(
            "No eligible pair survives the exclusions — the design has no third "
            "orthogonal pair on this panel. Post epm:failure v1 and re-plan."
        )
    pick = eligible[0]

    # ── Pre-registration HARD-ASSERT (plan §4) ──────────────────────────────
    if (pick["name_a"], pick["name_b"]) != (EXPECTED_NAME_A, EXPECTED_NAME_B):
        raise AssertionError(
            f"Selector pick {pick['pair_id']!r} != pre-registered "
            f"{EXPECTED_PAIR_ID!r}. The committed matrix or the exclusion set "
            f"drifted since planning — STOP and re-plan, do not proceed."
        )
    if abs(pick["base_cos_centered_L20"] - EXPECTED_COS) > COS_TOL:
        raise AssertionError(
            f"Picked cosine {pick['base_cos_centered_L20']:+.10f} differs from "
            f"pre-registered {EXPECTED_COS:+.10f} by more than {COS_TOL:g} — the "
            f"matrix content drifted since planning."
        )
    log.info(
        "Pre-registration assert PASS: pick=%s cos=%+.10f (expected %+.10f ± %g)",
        pick["pair_id"],
        pick["base_cos_centered_L20"],
        EXPECTED_COS,
        COS_TOL,
    )

    out_payload = {
        # WRITE schema stays the #527 lineage string — the inherited
        # dispatcher's _load_pair_selection pins it (plan §4 "New code").
        "schema_version": READ_SCHEMA,
        "base_model": payload["base_model"],
        "extraction_layer": payload["extraction_layer"],
        "centering": payload["centering"],
        "questions_used": payload["questions_used"],
        "persona_names": payload["persona_names"],
        # Parent matrix carried VERBATIM so downstream cosine lookups stay on
        # the same scale as the anchors' values (plan §9 assumption 6).
        "cos_centered_L20": payload["cos_centered_L20"],
        "picked_pairs": [
            {
                "pair_id": pick["pair_id"],
                "name_a": pick["name_a"],
                "name_b": pick["name_b"],
                "base_cos_centered_L20": pick["base_cos_centered_L20"],
                "abs_cos": pick["abs_cos"],
            }
        ],
        "threshold_used": "issue_568_smallest_eligible_abs_cos",
        "threshold_primary": payload.get("threshold_primary"),
        "threshold_fallback": payload.get("threshold_fallback"),
        "negative_panel": payload.get("negative_panel"),
        "issue_568_provenance": {
            "note": (
                "Third orthogonal pair for task #568: deterministic re-rank of the "
                "committed #527 cosine matrix (no fresh forward pass). The |cos|=0.0150 "
                "relaxation vs the task body's '<0.01' gloss is a named plan deviation "
                "(plan §4): no eligible in-panel pair satisfies <0.01 after exclusions."
            ),
            "source_matrix": str(matrix_path),
            "source_matrix_git_commit": payload.get("git_commit"),
            "excluded_sources": sorted(exclude_sources),
            "excluded_panel_replacement": NEGATIVE_PANEL_REPLACEMENT,
            "ranked_head": ranked[:8],
            "pre_registered_pair_id": EXPECTED_PAIR_ID,
            "pre_registered_cos": EXPECTED_COS,
            "cos_tolerance": COS_TOL,
        },
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    log.info("Wrote %s (%d bytes)", out_path, out_path.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
