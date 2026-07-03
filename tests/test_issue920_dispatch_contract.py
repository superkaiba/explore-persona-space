"""Pin the #920 dispatcher / phase-script contract invariants (round-2 blockers).

Three permanent invariants from the round-1 code-review FAILs:

1. ``k3-resume-bypasses-anchor-gate`` (BLOCKER): every dispatcher resume
   predicate keys on a POST-GATE / POST-UPLOAD ``*_done.json`` marker — the
   fits predicate in particular must require ``preds/fits_done.json`` (written
   only AFTER the K3 anchor gate), so a retry after a K3 FAIL re-runs the fits
   instead of skipping the failed gate on pre-gate artifacts.
2. ``set-b-zero-coverage-not-masked`` (BLOCKER): the exclusion masks union
   BOTH stores' zero-coverage families (a set-B-only gap masks exactly like a
   set-A one).
3. ``[phase=done]`` reserved-token class (#545): exactly ONE emission site in
   the GPU dispatcher (its terminal echo); phase scripts never emit it, except
   the gated standalone-P6 terminal in ``issue920_nulls_figures.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _code_lines_with_token(path: Path, token: str) -> list[str]:
    """Non-comment source lines containing the token."""
    return [
        ln for ln in path.read_text().splitlines() if token in ln and not ln.strip().startswith("#")
    ]


def test_dispatch_resume_predicates_key_on_done_markers():
    txt = (SCRIPTS / "issue920_dispatch.sh").read_text()
    for marker in (
        "gen_b_done.json",
        "extract_done.json",
        "preds/fits_done.json",
        "null_matrices/dv1_done.json",
    ):
        assert marker in txt, f"dispatcher resume predicate lost its {marker} key"
    # The fits `if` must test the post-K3 marker (first condition), never the
    # pre-gate artifacts alone.
    m = re.search(r"if \[ -f \"[^\"]*preds/fits_done\.json\" \]", txt)
    assert m, "fits resume predicate no longer keys on the post-K3 fits_done marker"


def test_phase_done_token_single_dispatcher_emission():
    # dispatch.sh: exactly one echo emission of the reserved token.
    sh = SCRIPTS / "issue920_dispatch.sh"
    emits = [
        ln
        for ln in sh.read_text().splitlines()
        if "[phase=done]" in ln and ln.strip().startswith("echo")
    ]
    assert len(emits) == 1, emits
    # Phase scripts never emit it (comments excluded)...
    for name in (
        "issue920_gen_completions_b.py",
        "issue920_extract_summaries.py",
        "issue920_fit_lofo.py",
        "issue920_results_sentinel.py",
    ):
        lines = _code_lines_with_token(SCRIPTS / name, "[phase=done]")
        assert lines == [], (name, lines)
    # ...except the gated standalone-P6 terminal (cpu-mid workload cmd), which
    # is that workload's OWN dispatcher-terminal: exactly one emission.
    nulls = _code_lines_with_token(SCRIPTS / "issue920_nulls_figures.py", "[phase=done]")
    assert len(nulls) == 1 and "cpu aggregation" in nulls[0], nulls


def test_exclusion_masks_union_both_stores():
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from issue920_fit_core import excluded_mask, union_excluded

    red_A = {"excluded_families": ["pos_tail_10@L*"]}  # A-only gap
    red_B = {"excluded_families": ["pos_head_0@L*"]}  # B-only gap (the round-1 miss)
    union, by_source = union_excluded(red_A, red_B)
    assert union == ["pos_head_0@L*", "pos_tail_10@L*"]
    assert by_source == {"set_A": ["pos_tail_10@L*"], "set_B": ["pos_head_0@L*"]}
    names = ["pos_head_0@L3", "pos_tail_10@L0", "ans_content_mean@L3", "ans_content_pool_meanmean"]
    got = excluded_mask(names, union)
    assert got.tolist() == [True, True, False, False]
    # The pre-fix behavior (mask from set A alone) misses the B-only family —
    # the exact silent zero-fill the union exists to prevent.
    pre_fix = excluded_mask(names, red_A["excluded_families"])
    assert pre_fix.tolist() == [False, True, False, False]
    assert np.any(got & ~pre_fix), "union mask must strictly extend the A-only mask here"
