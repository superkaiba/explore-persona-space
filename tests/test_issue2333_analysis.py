"""Issue #2333 analysis CPU pins.

Covers: the FOUR-BRANCH verdict lattice (incl. Indeterminate on conjunct
disagreement / straddles / missing D3) + scheme subordination, and the
s2-ce-derive cross-check FAILURE path (synthetic judge scores that do not
reproduce the vendored fu1 aggregate must raise, writing nothing).

Committed inputs read (sparse cones): eval_results/issue_2094/f_metrics/
anchors.jsonl + eval_results/issue_2333/inputs/fu1_conf1_confirmation.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2333_analysis as A33  # noqa: E402


def _iid(tag: str, key: str) -> str:
    return tag + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# ── lattice ───────────────────────────────────────────────────────────


def test_lattice_no_snowball_on_nonpositive_diff_ci():
    assert A33.lattice_label(-0.3, -0.01, True, 0.1, 0.2) == "no-snowball"
    assert A33.lattice_label(-0.3, 0.0, False, None, None) == "no-snowball"


def test_lattice_sufficient_and_partial():
    assert A33.lattice_label(0.05, 0.3, True, 0.01, 0.2) == "snowball-sufficient"
    assert A33.lattice_label(0.05, 0.3, True, 0.0, 0.2) == "snowball-sufficient"  # lo >= 0
    assert A33.lattice_label(0.05, 0.3, True, -0.3, -0.05) == "snowball-partial"


def test_lattice_indeterminate_branches():
    # Conjunct disagreement (CI positive vs Holm ns, and vice versa).
    assert A33.lattice_label(0.05, 0.3, False, 0.1, 0.2) == "indeterminate"
    assert A33.lattice_label(-0.05, 0.3, True, 0.1, 0.2) == "indeterminate"
    # Diff CI straddles zero with Holm ns.
    assert A33.lattice_label(-0.05, 0.3, False, 0.1, 0.2) == "indeterminate"
    # Separates but D3 CI straddles zero.
    assert A33.lattice_label(0.05, 0.3, True, -0.05, 0.05) == "indeterminate"
    # Separates but D3 CI missing.
    assert A33.lattice_label(0.05, 0.3, True, None, None) == "indeterminate"


def test_scheme_subordination_prefix():
    """Scheme (a) med is confirmatory; scheme (b) labels carry the
    'natural-opening' prefix and never the bare headline label."""
    assert A33.instance_label("med", "snowball-sufficient") == "snowball-sufficient"
    assert A33.instance_label("bstart", "snowball-sufficient") == (
        "natural-opening-snowball-sufficient"
    )
    assert A33.instance_label("bstart", "no-snowball") == "natural-opening-no-snowball"


# ── s2-ce-derive cross-check failure path ─────────────────────────────


@pytest.fixture()
def synthetic_stage(tmp_path, monkeypatch):
    """Stage dir whose conf1 + stage2 scores join cleanly but produce F means
    that CANNOT match the vendored aggregate (every draw delta = 1.0)."""
    monkeypatch.chdir(REPO_ROOT)  # committed anchors + vendored json
    anchors = A33.load_2094_anchors()
    stage = tmp_path / "fu1_stage"
    conf = stage / "conf1"
    scores = stage / "scores"
    conf.mkdir(parents=True)
    scores.mkdir(parents=True)
    score_rows = []
    for pid in anchors:
        key = f"{pid}|fu1|0"
        score_rows.append({"item_id": _iid("c", f"coh|s2|{key}"), "score": 100.0})
        score_rows.append({"item_id": _iid("s", f"beh|s2|{key}|prefix|a"), "score": 0.0})
        score_rows.append({"item_id": _iid("s", f"beh|s2|{key}|prefix|b"), "score": 100.0})
    for name in A33._FU1_SCORE_FILES:
        path = scores / name
        if name.startswith("coherence"):
            path.write_text("".join(json.dumps(r) + "\n" for r in score_rows), encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
    for arm_file in (
        "fu1_fu1__matched_query__ce__joint_all__replace__A__steered.jsonl",
        "fu1_fu1__matched_query__ce__joint_all__replace__A__null.jsonl",
    ):
        rows = [{"pair_id": pid, "cell": "fu1", "draw": 0} for pid in anchors]
        (conf / arm_file).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return stage


def test_s2_ce_derive_crosscheck_fails_loud(synthetic_stage, tmp_path):
    """Synthetic scores (delta 1.0 everywhere) give wellsep means far from the
    vendored steered 0.512 / null 0.097 — the phase must RAISE before writing
    any output artifact."""
    args = argparse.Namespace(stage_dir=synthetic_stage)
    before = set((REPO_ROOT / "eval_results/issue_2333/inputs").glob("s2_ce_*"))
    with pytest.raises(RuntimeError, match="cross-check FAILED"):
        A33.phase_s2_ce_derive(args)
    after = set((REPO_ROOT / "eval_results/issue_2333/inputs").glob("s2_ce_*"))
    assert after == before, "failure path must not write output artifacts"


def test_load_2094_anchors_committed_shape(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    anchors = A33.load_2094_anchors()
    assert len(anchors) == 15
    wellsep = [
        p
        for p, a in anchors.items()
        if a["separation"] is not None and abs(a["separation"]) >= A33.SEPARATION_BAR
    ]
    assert len(wellsep) == A33.C.S2_WELLSEP_N  # 10 (plan §5; 5 excluded)
