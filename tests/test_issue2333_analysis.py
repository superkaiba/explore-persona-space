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


# ── r2 additions: fixed-m Holm + floor gating + F_act mirror ──────────


def test_holm_fixed_m_uses_registered_family_size():
    """r1 Minor: Holm must correct at the REGISTERED family size m=12 per
    (model x pair-set) — never at the realized arm count."""
    out = A33.holm_fixed_m({"a": 0.004})
    assert out["a"] == pytest.approx(0.048)  # factor 12, not 1
    out = A33.holm_fixed_m({"a": 0.001, "b": 0.002})
    assert out["a"] == pytest.approx(0.012)  # (12 - 0) * p
    assert out["b"] == pytest.approx(0.022)  # (12 - 1) * p
    out = A33.holm_fixed_m({"a": 0.5, "b": 0.0001})
    assert out["b"] == pytest.approx(0.0012)
    assert out["a"] == 1.0  # capped
    with pytest.raises(AssertionError):
        A33.holm_fixed_m({f"p{i}": 0.01 for i in range(13)})


def test_phase_stats_floor_gating_fixed_holm_and_f_act_mirror(tmp_path, monkeypatch):
    """Registered survival-floor gating (r1 blocker): per-cell S1 floors (a
    below-floor cell's pairs are excluded from the pooled S1 read), below-
    floor arms get NO tests (label untestable-small-n), Holm fixed m=12, the
    F_act secondary mirror with its own family + rank agreement, and an
    untestable S2 set (3 < 5 survivors)."""
    monkeypatch.chdir(REPO_ROOT)
    s1_pairs, s2_pairs = A33.J33.build_pair_universe()
    cell1, cell2 = A33.C.S1_CELLS[0], A33.C.S1_CELLS[1]
    c1 = sorted(p.pair_id for p in s1_pairs if p.cell == cell1)[:14]
    c2 = sorted(p.pair_id for p in s1_pairs if p.cell == cell2)[:3]  # below per-cell floor
    s2_ids = sorted(p.pair_id for p in s2_pairs)[:3]  # below the S2 set floor (5)
    out = tmp_path / "stats_out"
    out.mkdir()

    def row(pid, set_name, cell, variant, f, fa):
        return {
            "pair_id": pid,
            "set": set_name,
            "cell": cell,
            "arm_slug": "prefill3_med",
            "variant": variant,
            "separation": 1.0,
            "f_beh": f,
            "f_act": fa,
        }

    steered, nulls = [], []
    for i, pid in enumerate(c1):
        steered.append(row(pid, "s1", cell1, "steered", 0.8 + 0.001 * i, 0.7 + 0.002 * i))
        nulls.append(row(pid, "s1", cell1, "null", 0.1, 0.05))
    for pid in c2:
        steered.append(row(pid, "s1", cell2, "steered", 0.9, 0.8))
        nulls.append(row(pid, "s1", cell2, "null", 0.1, 0.1))
    for pid in s2_ids:
        steered.append(row(pid, "s2", "s2_matched_query", "steered", 0.9, None))
        nulls.append(row(pid, "s2", "s2_matched_query", "null", 0.1, None))
    calib = [{"pair_id": pid, "set": "s1", "arm": "steered", "f_beh": 0.5} for pid in c1]
    A33.A62._write_jsonl_atomic(out / "f_cells.jsonl", steered)
    A33.A62._write_jsonl_atomic(out / "null_cells.jsonl", nulls)
    A33.A62._write_jsonl_atomic(out / "calib_cells.jsonl", calib)

    assert A33.phase_stats(argparse.Namespace(model_tag="q25", out_dir=out)) == 0
    res = json.loads((out / "stats.json").read_text(encoding="utf-8"))

    s1 = res["per_set"]["s1"]
    assert s1["floor"]["grain"] == "per-cell" and s1["floor"]["floor"] == 12
    assert s1["floor"]["cells_passing"] == [cell1]
    assert cell2 in s1["floor"]["cells_below_floor"]
    assert s1["n_survivors_tested"] == 14  # cell2's 3 pairs excluded from the pool
    arm = s1["arms"]["prefill3_med"]
    assert arm["n_pairs"] == 14 and not arm["below_floor"]
    assert s1["holm_family_m"] == 12
    assert arm["p_holm"] == pytest.approx(min(1.0, 12 * arm["p_wilcoxon"]))
    assert arm["separates"] is True
    for slug in A33.C.ARM_SLUGS:
        if slug == "prefill3_med":
            continue
        rec = s1["arms"][slug]
        assert rec["label"] == "untestable-small-n"
        assert "p_wilcoxon" not in rec and "diff_ci" not in rec  # NO tests below floor
    assert s1["prefill3_verdicts"]["med"]["label"] == "snowball-sufficient"
    assert s1["prefill3_verdicts"]["bstart"]["label"] == "natural-opening-untestable-small-n"
    fa = s1["f_act"]
    assert fa["holm_family_m"] == 12
    arm_a = fa["arms"]["prefill3_med"]
    assert arm_a["n_pairs"] == 14 and "diff_ci" in arm_a and "p_holm" in arm_a
    assert arm_a["spearman_vs_f_beh"]["n_pairs"] == 14
    assert "secondary" in fa["role"]

    s2 = res["per_set"]["s2"]
    assert s2["untestable"] is True and s2["floor"]["grain"] == "set"
    assert all(r.get("label") == "untestable-small-n" for r in s2["arms"].values())
    assert s2["prefill3_verdicts"]["med"]["label"] == "untestable-small-n"
