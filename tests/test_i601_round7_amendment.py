# em-dash / minus sign intentional
"""Task #601 round-7 — plan v3 amendment regressions.

1. **gate_schema 2 (§B):** Gate S (structural; the file's top-level ``pass``)
   / Gate A (anchor reuse) / Observation O on the REAL committed per-adapter
   table (inlined verbatim from
   ``eval_results/issue_601/phase0/endpoint_reads.json`` @ 2a83f7a) plus one
   synthetic failure per Gate-S criterion.
2. **Launch supervisor (§D items 1-4):** detachment unit exercise of the
   REAL preamble (selftest hook), relaunch guard exit 3, heartbeat lines into
   the main log, the combined abort trap, and the p3 smoke-sentinel skip.
3. **§C instantiation:** classification consumes in-task L̂/M̂/tol — fresh-top
   fork (L̂(8:1)≈20) routes normally, compressed-top fork (≈13) fires the
   degenerate-top guard; the step-32 decidability guard re-evaluates in
   margin space and ships underpowered when both compress; and the verdict
   machinery can NEVER fall back to the retired parent constants.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import (
    PARENT_COMMITTED_CROSS_RIG,
    classify_phase1,
    derive_in_task_references,
    matched_pair_discriminator,
    reexpress_threshold,
)
from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
    compute_gate_schema2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SH = REPO_ROOT / "scripts" / "i601_launch.sh"
ANALYZE_PY = REPO_ROOT / "scripts" / "i601_analyze.py"
COMMITTED_ENDPOINT = REPO_ROOT / "eval_results" / "issue_601" / "phase0" / "endpoint_reads.json"


# ── 1. gate_schema 2 ─────────────────────────────────────────────────────────

# The REAL committed per-adapter table (endpoint_reads.json @ 2a83f7a,
# 2026-06-11T19:27Z — the plan v3 §A evidence table, inlined so the test is
# sparse-checkout-proof; a separate test cross-checks the committed file).
_REAL_ROWS: tuple[tuple[str, float, float], ...] = (
    # (key, committed_delta_g, reread_delta_g)
    ("c472_anchor_seed137", 13.071153926849366, 11.954329538345338),
    ("c472_anchor_seed42", 13.946419525146485, 12.891918277740478),
    ("c472_negex_100_seed137", 8.44741325378418, 9.302722835540772),
    ("c472_negex_100_seed42", 8.647632694244384, 9.526271724700928),
    ("c472_negex_400_seed137", 20.342850494384766, 12.914596509933471),
    ("c472_negex_400_seed42", 19.65302746295929, 12.793137073516846),
    ("c472_noneg_seed137", 2.121089553833008, 2.684008026123047),
    ("c472_noneg_seed42", 1.9706001281738281, 2.617481803894043),
)
REAL_PER_ADAPTER = {
    key: {
        "committed_delta_g": committed,
        "reread_delta_g": reread,
        "abs_diff": abs(reread - committed),
        "reread_r_collapsed": False,
    }
    for key, committed, reread in _REAL_ROWS
}


def _perturb(key: str, **changes) -> dict:
    table = {k: dict(v) for k, v in REAL_PER_ADAPTER.items()}
    table[key].update(changes)
    return table


def test_gate_s_passes_gate_a_false_on_observed_data() -> None:
    """The plan v3 §B headline: Gate S PASSES, Gate A is false, Observation O
    carries the negex_400 rows — on the exact committed numbers."""
    gate = compute_gate_schema2(REAL_PER_ADAPTER, recipe_panel_ok=True)
    assert gate["gate_schema"] == 2
    assert gate["pass"] is True
    assert gate["gate_s"]["alarm_silent"] is True
    assert gate["gate_s"]["r_collapsed_all_false"] is True
    assert gate["gate_s"]["low_dose_reproduction_ok"] is True
    assert gate["gate_s"]["dose_ordering_ok"] is True
    assert gate["anchor_reuse_ok"] is False
    assert gate["gate_a"]["anchor_within_1nat"] == {
        "c472_anchor_seed137": False,
        "c472_anchor_seed42": False,
    }
    obs = gate["observation_o"]
    assert obs["gating"] is False
    assert set(obs["per_adapter"]) == {"c472_negex_400_seed42", "c472_negex_400_seed137"}
    assert all(row["abs_diff"] > 6.0 for row in obs["per_adapter"].values())
    assert obs.get("note")


@pytest.mark.skipif(
    not COMMITTED_ENDPOINT.exists(), reason="committed endpoint JSON not in this checkout (sparse)"
)
def test_inlined_table_matches_committed_endpoint_json() -> None:
    endpoint = json.loads(COMMITTED_ENDPOINT.read_text())
    per = endpoint["onpolicy_crosscheck"]["per_adapter"]
    for key, row in REAL_PER_ADAPTER.items():
        for field, val in row.items():
            assert per[key][field] == pytest.approx(val), (key, field)


def test_gate_s_fails_on_low_dose_drift() -> None:
    """A low-dose adapter drifting past 1.5 nat is a structural FAIL (HALT)."""
    table = _perturb("c472_noneg_seed42", abs_diff=1.61, reread_delta_g=1.9706 + 1.61)
    gate = compute_gate_schema2(table, recipe_panel_ok=True)
    assert gate["pass"] is False
    assert gate["gate_s"]["low_dose_reproduction_ok"] is False


def test_gate_s_fails_on_dose_ordering_collapse() -> None:
    """negex_100 collapsing onto noneg (<2-nat gap) breaks the ordering rule."""
    table = _perturb(
        "c472_negex_100_seed42", reread_delta_g=4.0, abs_diff=abs(8.647632694244384 - 4.0)
    )
    table["c472_negex_100_seed137"].update(reread_delta_g=4.1, abs_diff=abs(8.44741325378418 - 4.1))
    gate = compute_gate_schema2(table, recipe_panel_ok=True)
    assert gate["pass"] is False
    assert gate["gate_s"]["dose_ordering_ok"] is False


def test_gate_s_fails_on_r_collapsed() -> None:
    gate = compute_gate_schema2(
        _perturb("c472_anchor_seed42", reread_r_collapsed=True), recipe_panel_ok=True
    )
    assert gate["pass"] is False
    assert gate["gate_s"]["r_collapsed_all_false"] is False
    assert gate["gate_s"]["r_collapsed_keys"] == ["c472_anchor_seed42"]


def test_gate_s_fails_on_identical_reread_group() -> None:
    """Defensive re-check of the identical-read tripwire inside the gate
    (normally raised upstream by onpolicy_crosscheck as IdenticalRereadAlarm)."""
    table = {k: dict(v) for k, v in REAL_PER_ADAPTER.items()}
    for k in ("c472_anchor_seed42", "c472_anchor_seed137", "c472_negex_400_seed42"):
        table[k]["reread_delta_g"] = 10.350
    gate = compute_gate_schema2(table, recipe_panel_ok=True)
    assert gate["pass"] is False
    assert gate["gate_s"]["alarm_silent"] is False
    assert gate["gate_s"]["identical_groups"]


def test_gate_a_true_when_anchors_reproduce() -> None:
    """Gate A flips true (and Gate S stays true) when the anchors land <=1 nat."""
    table = _perturb(
        "c472_anchor_seed42", reread_delta_g=13.5, abs_diff=abs(13.946419525146485 - 13.5)
    )
    table["c472_anchor_seed137"].update(
        reread_delta_g=12.7, abs_diff=abs(13.071153926849366 - 12.7)
    )
    gate = compute_gate_schema2(table, recipe_panel_ok=True)
    assert gate["pass"] is True
    assert gate["anchor_reuse_ok"] is True


def test_gate_a_false_on_recipe_panel_mismatch() -> None:
    table = _perturb("c472_anchor_seed42", reread_delta_g=13.5, abs_diff=0.45)
    table["c472_anchor_seed137"].update(reread_delta_g=12.7, abs_diff=0.37)
    gate = compute_gate_schema2(table, recipe_panel_ok=False)
    assert gate["anchor_reuse_ok"] is False
    assert gate["gate_a"]["anchor_onpolicy_ok"] is True


def test_gate_schema2_requires_all_low_dose_seeds() -> None:
    table = {k: v for k, v in REAL_PER_ADAPTER.items() if k != "c472_noneg_seed42"}
    with pytest.raises(KeyError, match="both seeds"):
        compute_gate_schema2(table, recipe_panel_ok=True)


# ── 2. Launch supervisor (§D items 1-4) ─────────────────────────────────────


def _run_launch(env: dict, *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(LAUNCH_SH)],
        env={**os.environ, **env},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _wait_for(predicate, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    raise AssertionError("timed out waiting for supervised driver output")


def test_launcher_detaches_and_heartbeats(tmp_path: Path) -> None:
    """§D items 1+3: the launcher branch detaches via setsid --fork; the
    SUPERVISED process writes its own pid and emits [hb] lines carrying the
    live phase into the MAIN log; clean exit ends with the [phase=done] line."""
    env = {
        "LOG_DIR": str(tmp_path),
        "I601_LAUNCH_SELFTEST": "1",
        "I601_HB_INTERVAL": "1",
        "I601_SELFTEST_SLEEP": "3",
    }
    res = _run_launch(env)
    assert res.returncode == 0, res.stderr
    assert "[launcher] detached supervised driver" in res.stdout
    main_log = tmp_path / "issue-601.log"
    _wait_for(lambda: main_log.exists() and "[phase=done]" in main_log.read_text())
    text = main_log.read_text()
    assert "[phase=p_selftest]" in text
    hb_lines = [line for line in text.splitlines() if line.startswith("[hb] ")]
    assert len(hb_lines) >= 2, text  # 1-s interval over a 3-s selftest
    assert any("phase=p_selftest" in line for line in hb_lines), hb_lines
    assert "[phase=abort]" not in text
    # Own-pid contract: the pid file holds the pid the [hb] lines carry.
    pid = (tmp_path / "issue-601.pid").read_text().strip()
    assert pid.isdigit()
    assert any(f"pid={pid} " in line for line in hb_lines), (pid, hb_lines)


def test_relaunch_guard_refuses_with_exit_3(tmp_path: Path) -> None:
    """§D item 2: a live pid in the pid file refuses relaunch, exit 3."""
    (tmp_path / "issue-601.pid").write_text(str(os.getpid()))  # this test process: alive
    res = _run_launch({"LOG_DIR": str(tmp_path), "I601_LAUNCH_SELFTEST": "1"})
    assert res.returncode == 3, (res.stdout, res.stderr)
    assert "already running pid=" in res.stdout


def test_stale_pid_file_does_not_block_relaunch(tmp_path: Path) -> None:
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / "issue-601.pid").write_text(str(dead.pid))
    res = _run_launch(
        {
            "LOG_DIR": str(tmp_path),
            "I601_SUPERVISED": "1",  # run the supervised branch in the foreground
            "I601_LAUNCH_SELFTEST": "1",
            "I601_HB_INTERVAL": "60",
            "I601_SELFTEST_SLEEP": "0",
        }
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert (tmp_path / "issue-601.pid").read_text().strip() != str(dead.pid)


def test_abort_trap_prints_phase_abort_rc(tmp_path: Path) -> None:
    """§D item 3: the single combined EXIT trap kills the heartbeat and prints
    [phase=abort] rc=<rc> on any non-zero exit; [phase=done] never appears."""
    res = _run_launch(
        {
            "LOG_DIR": str(tmp_path),
            "I601_SUPERVISED": "1",
            "I601_LAUNCH_SELFTEST": "1",
            "I601_HB_INTERVAL": "60",
            "I601_SELFTEST_SLEEP": "0",
            "I601_SELFTEST_RC": "2",
        }
    )
    assert res.returncode == 2
    assert "[phase=abort] rc=2" in res.stdout
    assert "[phase=done]" not in res.stdout


def _p3_skip_heredoc() -> str:
    text = LAUNCH_SH.read_text()
    m = re.search(r"SMOKE_SKIP=\$\(uv run python - .*?<<'PY'\n(.*?)\nPY\n\)", text, flags=re.S)
    assert m, "p3 SMOKE_SKIP heredoc not found in i601_launch.sh"
    return m.group(1)


def _smoke_skip_decision(tmp_path: Path) -> str:
    res = subprocess.run(
        [sys.executable, "-c", _p3_skip_heredoc(), str(tmp_path / "issue-601-smoke-results.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def _write_smoke_sentinel(path: Path, *, smoke_gate_pass) -> None:
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "note": json.dumps({"smoke_gate_pass": smoke_gate_pass}),
            }
        )
    )


def test_p3_skips_on_passing_sentinel_bare_and_processed(tmp_path: Path) -> None:
    """§D item 4: a prior PASSing sentinel (bare or .processed) skips the smoke."""
    bare = tmp_path / "issue-601-smoke-results.json"
    _write_smoke_sentinel(bare, smoke_gate_pass=True)
    assert _smoke_skip_decision(tmp_path) == "skip"
    bare.rename(bare.with_suffix(".json.processed"))
    assert _smoke_skip_decision(tmp_path) == "skip"


def test_p3_runs_when_sentinel_missing_failed_or_corrupt(tmp_path: Path) -> None:
    assert _smoke_skip_decision(tmp_path) == "run"  # missing
    bare = tmp_path / "issue-601-smoke-results.json"
    _write_smoke_sentinel(bare, smoke_gate_pass=False)
    assert _smoke_skip_decision(tmp_path) == "run"  # failed smoke never skips
    bare.write_text("{not json")
    assert _smoke_skip_decision(tmp_path) == "run"  # corrupt -> conservative re-run


# ── 3. §C in-task references + classification forks ─────────────────────────


def _refs(l8: float = 20.0, m8: float = 14.0, **overrides) -> dict:
    kwargs = dict(
        level_terminals_logp={"0:1": [2.6], "2:1": [9.4], "4:1": [12.3, 12.5], "8:1": [l8]},
        level_terminals_margin={"0:1": [1.6], "2:1": [6.4], "4:1": [8.4, 8.6], "8:1": [m8]},
    )
    kwargs.update(overrides)
    return derive_in_task_references(**kwargs)


def _flat_series(terminal: float, v32: float) -> tuple[list[int], list[float]]:
    return [2, 16, 32, 64, 128], [
        0.88 * terminal,
        0.86 * terminal,
        v32,
        0.98 * terminal,
        terminal,
    ]


def test_fresh_top_fork_routes_horizon() -> None:
    """L̂(8:1)≈20 (fresh top exists): the upper branch is resolvable and a
    horizon-shaped arm set classifies as horizon via the level rules."""
    refs = _refs(l8=20.0, m8=14.0)
    assert refs["top_degenerate"] is False
    assert refs["l8_single_seed"] is True
    assert refs["l8_uncertainty_nats"] == pytest.approx(0.2)  # largest 2-seed gap carried
    series = {
        42: ([2, 16, 32, 64, 128], [10.0, 15.0, 16.5, 17.0, 17.4]),
        137: ([2, 16, 32, 64, 128], [10.0, 15.0, 16.5, 17.0, 17.4]),
    }
    out = classify_phase1(
        arm_terminals={
            "quarter": [8.0, 8.4],
            "anchor": [12.3, 12.5],
            "double": [17.0, 17.4],
            "matched": [17.2, 17.6],
        },
        matched_series_by_seed=series,
        space="logp",
        refs=refs,
        clamp_present=True,
    )
    assert out["upper_branch_resolvable"] is True
    assert out["top_compression"] is None
    assert out["refs"]["4:1"] == pytest.approx(12.4)
    assert out["upper_midpoint"] == pytest.approx((12.4 + 20.0) / 2)  # in-task midpoint
    assert out["tolerance"] == pytest.approx(3.0)  # floor binds (max gap 0.4 -> 0.8)
    assert out["grace"] == pytest.approx(1.5)  # tol/2 scaling
    assert out["verdicts"]["horizon"] is True
    assert out["verdicts"]["coupling"] is False
    assert out["call"] == "horizon"
    mp = out["matched_pair_discriminator"]
    assert mp["decidable"] is True and mp["verdict"] == "horizon"


def test_compressed_top_fork_fires_degenerate_guard() -> None:
    """L̂(8:1)≈13 within tol of L̂(4:1) in BOTH spaces: upper-branch level tests
    are declared unresolvable (None), the headline routes to matched-pair +
    equilibrium co-landing, and the top-compression is reported."""
    refs = _refs(l8=13.0, m8=8.9)
    assert refs["top_degenerate"] is True
    series = {42: _flat_series(12.4, 12.0), 137: _flat_series(12.4, 12.0)}
    out = classify_phase1(
        arm_terminals={
            "quarter": [12.0, 12.4],
            "anchor": [12.3, 12.5],
            "double": [12.5, 12.9],
            "matched": [12.2, 12.6],
        },
        matched_series_by_seed=series,
        space="logp",
        refs=refs,
        clamp_present=True,
        margin_fallback={
            "arm_terminals": {
                "quarter": [8.3, 8.5],
                "anchor": [8.4, 8.6],
                "double": [8.5, 8.7],
                "matched": [8.3, 8.5],
            },
            "matched_series_by_seed": {42: _flat_series(8.4, 8.2), 137: _flat_series(8.4, 8.2)},
        },
    )
    assert out["upper_branch_resolvable"] is False
    assert out["top_compression"]["top_degenerate"] is True
    assert out["verdicts"]["horizon"] is None  # unresolvable-as-registered
    assert out["verdicts"]["coupling"] is None
    assert out["verdicts"]["ratio_set_point_consistent"] is True  # co-landing
    assert out["call"] == "equilibrium"  # co-landing + clamp; timing underpowered
    assert "degenerate-top routing" in out["call_rule"]
    mp = out["matched_pair_discriminator"]
    assert mp["timing_underpowered"] is True  # compressed in BOTH spaces
    assert mp["verdict"] == "underpowered"
    assert mp["margin_fallback_attempt"] is not None
    assert mp["underpowered_note"]


def test_decidability_guard_falls_back_to_margin_space() -> None:
    """§C: logP compressed below the 3-nat separation -> the discriminator
    re-evaluates in margin space and reads there when decidable."""
    refs = _refs(l8=20.0, m8=14.0)
    out = classify_phase1(
        arm_terminals={
            "quarter": [11.0, 11.4],
            "anchor": [12.3, 12.5],
            "double": [17.0, 17.4],
            "matched": [12.0, 12.4],
        },
        matched_series_by_seed={42: _flat_series(12.2, 12.0), 137: _flat_series(12.2, 12.0)},
        space="logp",
        refs=refs,
        clamp_present=True,
        margin_fallback={
            "arm_terminals": {
                "quarter": [2.0, 2.2],
                "anchor": [8.4, 8.6],
                "double": [13.0, 13.4],
                "matched": [10.0, 10.4],
            },
            "matched_series_by_seed": {42: _flat_series(10.2, 9.5), 137: _flat_series(10.2, 9.5)},
        },
    )
    mp = out["matched_pair_discriminator"]
    assert mp["primary_space_attempt"]["decidable"] is False
    assert mp["space"] == "margin"  # the fallback was adopted
    assert mp["decidable"] is True
    assert mp["verdict"] == "horizon"
    assert mp["timing_underpowered"] is False


def test_matched_pair_discriminator_is_arm_internal() -> None:
    """The discriminator thresholds derive from the arms themselves — no
    external reference enters (quarter+2.5 vs 80% of own terminal)."""
    mp = matched_pair_discriminator(
        {"quarter": [8.0, 8.4], "matched": [17.2, 17.6]},
        {42: _flat_series(17.4, 16.5)},
        2.5,
        "logp",
    )
    assert mp["coupling_max"] == pytest.approx(8.2 + 2.5)
    assert mp["horizon_min"] == pytest.approx(0.8 * 17.4)
    assert mp["decidable"] is True
    assert mp["verdict"] == "horizon"


def test_tolerance_widens_from_in_task_seed_gaps_and_grace_scales() -> None:
    refs = _refs(
        extra_seed_pairs_logp={"quarter": [8.0, 10.0]},  # gap 2.0 -> tol 4.0
        extra_seed_pairs_margin={"quarter": [5.0, 6.0]},  # gap 1.0 -> tol 2.0
    )
    assert refs["tol_logp"] == pytest.approx(4.0)
    assert refs["tol_logp_widened_beyond_floor"] is True
    assert refs["tol_margin"] == pytest.approx(2.0)
    assert refs["grace_logp"] == pytest.approx(2.0)


def test_two_seed_l8_is_not_flagged_single_seed() -> None:
    refs = _refs(
        level_terminals_logp={"0:1": [2.6], "2:1": [9.4], "4:1": [12.3, 12.5], "8:1": [19.8, 20.2]},
        level_terminals_margin={"0:1": [1.6], "2:1": [6.4], "4:1": [8.4, 8.6], "8:1": [13.8, 14.2]},
    )
    assert refs["l8_single_seed"] is False
    assert refs["l8_uncertainty_nats"] == 0.0


# ── 3b. No-parent-fallback asserts ───────────────────────────────────────────


def test_parent_constants_retired_from_module_surface() -> None:
    from explore_persona_space.experiments.neg_setpoint_601 import analysis_lib

    for name in (
        "L_REFS",
        "HORIZON_UPPER_LOGP",
        "COUPLING_STEP32_MAX_LOGP",
        "QUARTER_HORIZON_MAX_LOGP",
        "SEED_GRACE_NATS",
        "LOGP_TOL_NATS",
    ):
        assert not hasattr(analysis_lib, name), f"retired parent constant {name} resurfaced"
    assert PARENT_COMMITTED_CROSS_RIG["l_refs_logp"]["8:1"] == 20.00  # cross-rig home only


def test_classification_never_falls_back_to_parent_constants() -> None:
    """The verdict machinery REQUIRES the in-task refs dict: no refs, partial
    refs, and the legacy margin_refs signature all fail loudly; and the
    classification source never references the cross-rig parent block."""
    import inspect

    from explore_persona_space.experiments.neg_setpoint_601 import analysis_lib

    arms = {"quarter": [8.0], "anchor": [12.4], "double": [17.0], "matched": [17.2]}
    series = {42: _flat_series(17.2, 16.5)}
    with pytest.raises(TypeError):
        classify_phase1(arm_terminals=arms, matched_series_by_seed=series, space="logp")
    with pytest.raises(ValueError, match="parent-constant fallback"):
        classify_phase1(
            arm_terminals=arms, matched_series_by_seed=series, space="logp", refs={"l_refs": {}}
        )
    with pytest.raises(TypeError):
        classify_phase1(
            arm_terminals=arms,
            matched_series_by_seed=series,
            space="margin",
            margin_refs={"4:1": 8.5},
            margin_tol=1.0,
        )
    with pytest.raises(TypeError):
        reexpress_threshold(10.0, {"4:1": 8.5})  # l_refs is REQUIRED now
    src = inspect.getsource(analysis_lib.classify_phase1) + inspect.getsource(
        analysis_lib._phase1_verdicts_and_call
    )
    assert "PARENT_COMMITTED_CROSS_RIG" not in src


def test_analyze_without_in_task_refs_yields_no_classification(tmp_path: Path) -> None:
    """End-to-end: a slab with ALL arms but NO in-task reference cells must
    leave phase1 unreadable (named missing rows) — never silently classify
    against parent numbers. The cross-rig block still carries them."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from test_i601_round3_fixes import REF_CELLS, _build_divergence_slab
    finally:
        sys.path.pop(0)
    slab = tmp_path / "slab"
    _build_divergence_slab(slab)
    for slug, seed in REF_CELLS:
        (slab / "phase2" / f"{slug}_seed{seed}" / "trajectory.json").unlink()
    out = tmp_path / "classification.json"
    res = subprocess.run(
        [sys.executable, str(ANALYZE_PY), "--slab-root", str(slab), "--allow-partial",
         "--out-path", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["phase1"] is None
    assert payload["in_task_references"] is None
    assert "in-task-references-incomplete" in payload["missing_inputs"]
    assert any(m.startswith("in-task-reference-") for m in payload["missing_inputs"])
    assert payload["cross_rig"]["parent_committed"]["l_refs_logp"]["4:1"] == 13.51
