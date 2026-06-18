"""Exhaustive-lattice unit test for the #642 v9 ``_classify_outcome_v9`` decision
rule (plan v10 §3).

The plan's §3 decision lattice partitions every reachable refusal
Δ_rank_matched outcome (a SINGLE contrast — no Δ_data this round) into EXACTLY
one of these cells:

  REPLICATES                  CI excludes 0, sign positive, |point| >= +0.04
  PARTIAL                     sign positive, CI excludes 0, BUT point < +0.04
  FAILS / opposite_sign_rank  CI excludes 0 NEGATIVE, point <= -0.04
  FAILS / noise_limited       CI spans 0, point inside the band
  FAILS / positive_uncertain  point >= +0.04 but CI does NOT exclude 0

This MECHANIZES the §3 totality claim (mirrors the v8 ``_classify_outcome``
test): every reachable (point, lo, hi) maps to exactly one registered cell.

Pure CPU — no model, no API, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i642_analyze as m  # noqa: E402

THR = 0.04  # the ±0.04 separation band (matched to round 4's +0.063 scale)

# name -> (delta_rank_ci, expected_verdict, expected_subreason). CI = (point, lo, hi).
LATTICE_CELLS: dict[str, tuple[tuple, str, str | None]] = {
    # CI excludes 0 positive AND |point| clears the band -> the gap is general.
    "REPLICATES": ((0.08, 0.04, 0.12), "REPLICATES", None),
    # CI excludes 0 positive but point shrinks below +0.04 -> directionally general.
    "PARTIAL": ((0.03, 0.01, 0.05), "PARTIAL", None),
    # CI excludes 0 NEGATIVE, point past -0.04 -> sign flip (dense leaks LESS).
    "opposite_sign_rank": ((-0.08, -0.12, -0.04), "FAILS", "opposite_sign_rank"),
    # CI spans 0, point inside the band -> noise-limited power statement.
    "noise_limited": ((0.01, -0.03, 0.05), "FAILS", "noise_limited"),
    # point clears the band positive but CI does NOT exclude 0 -> underpowered.
    "positive_uncertain": ((0.06, -0.01, 0.13), "FAILS", "positive_uncertain"),
}


@pytest.mark.parametrize("cell_name", list(LATTICE_CELLS))
def test_each_v9_lattice_cell_classifies_uniquely(cell_name: str) -> None:
    rank_ci, exp_verdict, exp_sub = LATTICE_CELLS[cell_name]
    verdict, sub = m._classify_outcome_v9(rank_ci)
    assert (verdict, sub) == (exp_verdict, exp_sub), (
        f"cell {cell_name!r}: expected ({exp_verdict!r}, {exp_sub!r}), got ({verdict!r}, {sub!r})"
    )


def test_v9_five_distinct_outcomes() -> None:
    outcomes = {m._classify_outcome_v9(rank_ci) for (rank_ci, _v, _s) in LATTICE_CELLS.values()}
    assert len(outcomes) == 5, f"expected 5 distinct outcomes, got {len(outcomes)}: {outcomes}"


def test_v9_subreason_none_iff_decisive() -> None:
    """REPLICATES + PARTIAL carry subreason=None; every FAILS cell carries a tag."""
    for cell_name, (rank_ci, _v, _s) in LATTICE_CELLS.items():
        verdict, sub = m._classify_outcome_v9(rank_ci)
        if verdict in ("REPLICATES", "PARTIAL"):
            assert sub is None, f"{cell_name}: decisive verdict must have subreason=None"
        else:
            assert verdict == "FAILS"
            assert sub is not None, f"{cell_name}: FAILS must carry a subreason tag"


def test_v9_lattice_is_total_and_unique_over_a_dense_grid() -> None:
    """Routing is TOTAL (every reachable input maps to a registered cell) and
    UNIQUE (exactly one verdict+subreason per input)."""
    registered = {(v, s) for (_r, v, s) in LATTICE_CELLS.values()}
    grid = [round(x * 0.01, 4) for x in range(-12, 13)]
    seen: set[tuple[str, str | None]] = set()
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            mid = round((lo + hi) / 2.0, 4)
            for point in {lo, mid, hi}:
                result = m._classify_outcome_v9((point, lo, hi))
                assert isinstance(result, tuple) and len(result) == 2
                assert result in registered, (
                    f"unregistered outcome {result!r} for ci=({point},{lo},{hi}) — "
                    "the v9 lattice has a hole the §3 enumeration does not cover"
                )
                seen.add(result)
    assert seen == registered, f"dense grid missed cells: {registered - seen}"


def test_v9_threshold_override_respected() -> None:
    """A custom decomp_threshold reshapes the band: a CI that REPLICATES at the
    default 0.04 becomes noise_limited when the threshold is widened past it."""
    rank_ci = (0.05, 0.045, 0.055)
    assert m._classify_outcome_v9(rank_ci) == ("REPLICATES", None)
    verdict, sub = m._classify_outcome_v9(rank_ci, thresholds={"decomp_threshold": 0.10})
    # point 0.05 < thr 0.10 and lo 0.045 > 0 -> sign positive, CI excludes 0,
    # point below band -> PARTIAL.
    assert (verdict, sub) == ("PARTIAL", None)


# ---------------------------------------------------------------------------
# B2 (round-1 reconcile blocker): v9 install-failure path unification.
# The dispatcher's phase3_select MUST write per-arm install_failure_<arm>.json
# (the convention the v9 analyzer reads); a no-suffix install_failure.json would
# be invisible to _v9_install_failure -> the analyzer would fall through to full
# analysis on a killed behavior. This test materializes the per-arm file and
# asserts the analyzer short-circuits to the KILLED verdict WITHOUT requiring
# generation manifests (i.e. without calling _v9_analyze_behavior).
# ---------------------------------------------------------------------------

import json  # noqa: E402

from i642_common import V9_ARMS  # noqa: E402


def _write_v9_install_failure(eval_root: Path, behavior: str, arm: str) -> Path:
    sa = eval_root / behavior / "stage_a"
    sa.mkdir(parents=True, exist_ok=True)
    fp = sa / f"install_failure_{arm}.json"
    fp.write_text(
        json.dumps(
            {
                "behavior": behavior,
                "arm": arm,
                "kill_criterion": "a_install_failure",
                "arm_ok": {a: (a != arm) for a in V9_ARMS},
            }
        )
    )
    return fp


def test_v9_per_arm_install_failure_short_circuits_to_killed(tmp_path, capsys) -> None:
    """A per-arm install_failure_loraRefOP.json under eval_root makes the v9
    analyzer return 0 with verdict=KILLED, never reaching _v9_analyze_behavior
    (so no generation manifests are needed)."""
    eval_root = tmp_path / "eval_results" / "issue_642"
    _write_v9_install_failure(eval_root, "refusal", "loraRefOP")

    # Guard: if the analyzer reached full analysis it would call this; assert it
    # is NOT invoked on the kill path.
    called = {"full_analysis": False}
    orig = m._v9_analyze_behavior

    def _tripwire(*a, **k):
        called["full_analysis"] = True
        return orig(*a, **k)

    m._v9_analyze_behavior = _tripwire
    try:
        rc = m.main(
            [
                "--v9",
                "--behavior",
                "refusal",
                "--eval-root",
                str(eval_root),
                "--no-refetch",
            ]
        )
    finally:
        m._v9_analyze_behavior = orig

    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict=KILLED" in out
    assert "arm=loraRefOP" in out
    assert called["full_analysis"] is False


def test_v9_no_suffix_install_failure_is_invisible(tmp_path) -> None:
    """Regression for the pre-fix bug: a NO-SUFFIX install_failure.json (the old
    writer) is NOT detected by the v9 reader — confirming the per-arm convention
    is the only one that works. (Pre-fix the writer produced this file, so the
    kill was never seen.)"""
    eval_root = tmp_path / "eval_results" / "issue_642"
    sa = eval_root / "refusal" / "stage_a"
    sa.mkdir(parents=True, exist_ok=True)
    (sa / "install_failure.json").write_text(json.dumps({"behavior": "refusal"}))
    # _v9_install_failure reads ONLY install_failure_<arm>.json, so the no-suffix
    # file is invisible -> returns None (the bug the B2 writer fix closes).
    res = m._v9_install_failure(eval_root, "refusal", refetch=False, v9_experiment="x")
    assert res is None


# ---------------------------------------------------------------------------
# M1 (round-2 reconcile BLOCKER): the analysis->figures path CONTRACT.
#
# Plan v10 §6.5 primary_deliverable names the v9 analysis at
# `eval_results/issue_642/refusal_v9/analysis.json` (NO `refusal/` segment).
# Pre-fix the analyzer wrote `eval_root/refusal/analysis.json` (one level too
# deep) so the §6.5 glob never resolved AND the canonical bare
# `i642_figures.py --v9` reader (which defaults `--eval-root` to the shared
# `issue_642/` parent) missed it. These tests pin the WRITER at the registered
# `<eval-root>/analysis.json` path AND the bare `--v9` figures command's
# eval-root resolution against a synthetic v9 fixture — no model, API, or GPU.
# A re-introduced `behavior/` subdir on the writer (or a figures reader/default
# that drops the `refusal_v9/` descent) re-fails both.
# ---------------------------------------------------------------------------

import i642_figures as figmod  # noqa: E402


def _build_v9_fixture(eval_root: Path) -> None:
    """Synthetic 2-arm v9 refusal verdict tree under ``<eval_root>/refusal/``
    (matched_lr REPLICATES @ Δ_rank≈0.06). The analyzer reads inputs from
    ``<eval_root>/refusal/`` and (post-fix) writes the result to
    ``<eval_root>/analysis.json``."""
    m.make_synthetic_v9(eval_root, "bracket")


def test_v9_analysis_written_at_registered_section_6_5_path(tmp_path) -> None:
    """WRITER contract: `i642_analyze.py --v9 --eval-root <tmp>/refusal_v9`
    writes the analysis at `<tmp>/refusal_v9/analysis.json` (the §6.5 glob),
    NOT one level deeper at `<tmp>/refusal_v9/refusal/analysis.json`."""
    eval_root = tmp_path / "eval_results" / "issue_642" / "refusal_v9"
    _build_v9_fixture(eval_root)

    rc = m.main(
        [
            "--v9",
            "--behavior",
            "refusal",
            "--eval-root",
            str(eval_root),
            "--no-refetch",
            "--bootstrap-b",
            "200",
        ]
    )
    assert rc == 0

    registered = eval_root / "analysis.json"
    nested = eval_root / "refusal" / "analysis.json"
    assert registered.exists(), (
        f"§6.5 contract path {registered} missing — the writer must drop the "
        "`behavior/` subdir for v9 (eval_root already encodes the behavior)"
    )
    assert not nested.exists(), (
        f"analysis.json re-introduced the `behavior/` subdir at {nested} — that "
        "is exactly the M1 regression (the §6.5 glob then fails to resolve)"
    )
    # the written file is a real v9 analysis with the headline contract.
    payload = json.loads(registered.read_text())
    assert payload.get("v9") is True
    assert "headline" in payload and "delta_rank_matched" in payload["headline"]


def test_bare_v9_figures_command_reads_the_registered_path(tmp_path, monkeypatch) -> None:
    """READER contract: the canonical plan §10 command `i642_figures.py --v9`
    (NO --eval-root) resolves the v9 analysis at
    `<REPO>/eval_results/issue_642/refusal_v9/analysis.json`. We mock `REPO` in
    BOTH the analyzer and figures modules to a tmp tree, run the bare commands
    exactly as the Reproducibility Card declares, and confirm figures land —
    i.e. the bare command found the registered analysis with NO explicit root."""
    fake_repo = tmp_path / "repo"
    # The analyzer's --eval-root explicit path (plan §10 step 1) and the figures
    # bare-command default (plan §10 step 2) must point at the SAME refusal_v9/.
    eval_root = fake_repo / "eval_results" / "issue_642" / "refusal_v9"
    _build_v9_fixture(eval_root)

    # Step 1: analyzer with the §10 explicit --eval-root writes the §6.5 path.
    rc1 = m.main(
        [
            "--v9",
            "--behavior",
            "refusal",
            "--eval-root",
            str(eval_root),
            "--no-refetch",
            "--bootstrap-b",
            "200",
        ]
    )
    assert rc1 == 0
    assert (eval_root / "analysis.json").exists()

    # Step 2: BARE `i642_figures.py --v9` — no --eval-root, no --out-dir. Mock
    # REPO so the figures default `eval_results/issue_642` (+ the v9 descent to
    # refusal_v9/) and the default `figures/issue_642` both resolve under tmp.
    monkeypatch.setattr(figmod, "REPO", fake_repo)
    rc2 = figmod.main(["--v9"])
    assert rc2 == 0

    out_dir = fake_repo / "figures" / "issue_642" / "v9"
    figs = list(out_dir.glob("*.png")) + list(out_dir.glob("*.pdf"))
    assert figs, (
        f"bare `i642_figures.py --v9` produced no figures under {out_dir} — the "
        "default --eval-root did not descend into refusal_v9/ to find the §6.5 "
        "analysis.json (the canonical post-pod chain is broken)"
    )


def test_v9_figures_explicit_eval_root_is_honored(tmp_path) -> None:
    """An explicit --eval-root pointing AT refusal_v9/ is used as-is (no extra
    descent) — confirms the default-descent only fires for the bare default."""
    eval_root = tmp_path / "refusal_v9"
    _build_v9_fixture(eval_root)
    rc1 = m.main(
        [
            "--v9",
            "--behavior",
            "refusal",
            "--eval-root",
            str(eval_root),
            "--no-refetch",
            "--bootstrap-b",
            "200",
        ]
    )
    assert rc1 == 0
    out_dir = tmp_path / "figs"
    rc2 = figmod.main(["--v9", "--eval-root", str(eval_root), "--out-dir", str(out_dir)])
    assert rc2 == 0
    figs = list(out_dir.glob("*.png")) + list(out_dir.glob("*.pdf"))
    assert figs, f"explicit-root figures produced nothing under {out_dir}"


# ---------------------------------------------------------------------------
# M2 (round-2 standing rec): the WRITER side of the per-arm install-failure
# convention. `test_v9_per_arm_install_failure_short_circuits_to_killed` above
# pins the READER; this pins the dispatcher's `_write_install_failure` WRITER —
# v9 -> one `install_failure_<arm>.json` per FAILING arm (the convention the v9
# reader uses), and NO no-suffix `install_failure.json` (which the v9 reader
# would never see -> a silent missed kill). v4 keeps the single no-suffix file.
# ---------------------------------------------------------------------------

import types  # noqa: E402

_DISPATCH = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_DISPATCH) not in sys.path:
    sys.path.insert(0, str(_DISPATCH))

import i642_dispatch as D  # noqa: E402


def _fake_ctx(*, v9: bool) -> types.SimpleNamespace:
    """Minimal stand-in carrying only the attributes `_write_install_failure`
    reads: `v9`, `experiment_name`, `skip_upload`. skip_upload=True keeps the
    test offline (the Hub call short-circuits)."""
    return types.SimpleNamespace(
        v9=v9, experiment_name="issue642_refusal_secondbehavior_smoke", skip_upload=True
    )


def test_v9_install_failure_writer_emits_per_arm_no_no_suffix(tmp_path) -> None:
    """v9 WRITER: a per-arm kill (loraRefOP fails, cmftRefOP ok) writes
    `install_failure_loraRefOP.json` and writes NO no-suffix `install_failure.json`
    (the file the v9 reader would never detect)."""
    ctx = _fake_ctx(v9=True)
    sa_dir = tmp_path / "refusal" / "stage_a"
    sa_dir.mkdir(parents=True, exist_ok=True)
    arm_ok = {"loraRefOP": False, "cmftRefOP": True}

    D._write_install_failure(ctx, "refusal", sa_dir, "stage_a", arm_ok)

    per_arm = sa_dir / "install_failure_loraRefOP.json"
    no_suffix = sa_dir / "install_failure.json"
    assert per_arm.exists(), (
        f"v9 writer must emit {per_arm.name} — the per-arm convention the v9 "
        "reader (`_v9_install_failure`) checks"
    )
    assert not no_suffix.exists(), (
        "v9 writer must NOT emit the no-suffix install_failure.json — the v9 "
        "reader never sees it, so a kill would be silently missed (B2 regression)"
    )
    # the ok arm is not flagged as failing.
    assert not (sa_dir / "install_failure_cmftRefOP.json").exists()
    payload = json.loads(per_arm.read_text())
    assert payload["arm"] == "loraRefOP"
    assert payload["kill_criterion"] == "a_install_failure"
    assert payload["arm_ok"] == arm_ok


def test_v9_install_failure_writer_emits_one_file_per_failing_arm(tmp_path) -> None:
    """When BOTH arms fail, the v9 writer emits one per-arm file each."""
    ctx = _fake_ctx(v9=True)
    sa_dir = tmp_path / "refusal" / "stage_a"
    sa_dir.mkdir(parents=True, exist_ok=True)
    arm_ok = {"loraRefOP": False, "cmftRefOP": False}

    D._write_install_failure(ctx, "refusal", sa_dir, "stage_a", arm_ok)

    assert (sa_dir / "install_failure_loraRefOP.json").exists()
    assert (sa_dir / "install_failure_cmftRefOP.json").exists()
    assert not (sa_dir / "install_failure.json").exists()


def test_v4_install_failure_writer_keeps_no_suffix_file(tmp_path) -> None:
    """SIBLING guard: the v4 (non-v9) writer keeps the single no-suffix
    `install_failure.json` its reader (`_install_failure_report`) expects — the
    per-arm convention is v9-ONLY, so the v9 branch must not regress v4."""
    ctx = _fake_ctx(v9=False)
    sa_dir = tmp_path / "sycophancy" / "stage_a"
    sa_dir.mkdir(parents=True, exist_ok=True)

    D._write_install_failure(ctx, "sycophancy", sa_dir, "stage_a", {"cmft": False})

    assert (sa_dir / "install_failure.json").exists()
    # no per-arm files in the v4 branch.
    assert not list(sa_dir.glob("install_failure_*.json"))
