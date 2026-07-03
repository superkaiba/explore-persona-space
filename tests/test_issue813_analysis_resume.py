"""Per-cell resume-skip+load regression for the #813 analysis pass (boundary-swap enabler).

The analysis ``main()`` loops (behavior x substrate) cells serially; one cell's
substrate-swap null costs ~2.5h. This pins the per-cell RESUME-SKIP+LOAD the
orchestrator relies on to (a) run parallel disjoint-subset processes without
redoing finished cells and (b) resume-skip through all 12 cells in a final
full-set pass to assemble the complete summary.json.

The resume contract:

1. A cell is SKIPPED (not recomputed) iff BOTH its delta_floor JSON AND its
   substrate_swap_null JSON exist + parse AND the null carries
   ``n_over_floor_resamples_used >= 1`` OR a ``note`` (the degenerate-cell shape).
2. A skipped cell's JSONs are LOADED into the SAME in-memory structures the
   downstream verdict / summary assembly consumes, so summary.json content is
   equivalent whether a cell was computed or loaded.
3. A missing / corrupt / in-flight null ⇒ recompute (overwriting partials).
4. ``--no-resume`` forces full recompute.

The heavy compute (observed_read, substrate_swap_null, pairwise CIs) is stubbed
so main() runs in seconds on CPU with no r_B / marker artifacts / GPU — the patch
under test is the resume-skip vs compute BRANCHING in main() + the summary
assembly over a mix of loaded and computed cells. Pure logic, ~1s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue813_analysis as an  # noqa: E402


def _full_null(p95: float = 0.30, n_used: int = 500) -> dict:
    """A minimal FULL-success substrate-swap-null dict (carries n_over_floor_resamples_used)."""
    return {
        "null_space": "delta_over_floor",
        "null_over_floor_p95": p95,
        "null_over_floor_p975": p95 + 0.05,
        "null_over_floor_median": p95 / 2,
        "null_delta_over_floor_diffs": [p95 * 0.5, p95, p95 * 1.5],
        "n_over_floor_resamples_used": n_used,
        "n_questions": 20,
        "n_resamples_used": n_used,
        "n_refit_pairs": 40,
    }


def _degenerate_null(note: str = "too few questions (<4) for a matched-n split") -> dict:
    """A minimal degenerate-cell early-return null (carries a `note`, n_over_floor absent)."""
    return {
        "null_p95": None,
        "null_over_floor_p95": None,
        "n_questions": 2,
        "n_resamples_used": 0,
        "null_space": "delta_over_floor",
        "note": note,
    }


def _obs(substrate: str, dof: float) -> dict:
    """A minimal observed (delta_floor) read dict for one (behavior, substrate)."""
    return {
        "behavior": "em",
        "substrate": substrate,
        "layer": an.HEADLINE_LAYER,
        "n_cells": 12,
        "delta_over_floor": dof,
        "delta_over_floor_sd": dof,
        "delta_med": dof * 2.0,
        "floor_combined": 1.0,
        "support_distance": 0.5,
        "chain_rho": None,
        "marker_two_read": False,
    }


# ── _resume_cell / _load_cell_json unit contract ───────────────────────────────


def test_load_cell_json_none_on_missing_and_corrupt(tmp_path):
    missing = tmp_path / "nope.json"
    assert an._load_cell_json(missing) is None
    truncated = tmp_path / "trunc.json"
    truncated.write_text('{"a": 1, "b":')  # truncated mid-write
    assert an._load_cell_json(truncated) is None
    non_obj = tmp_path / "list.json"
    non_obj.write_text("[1, 2, 3]")  # valid JSON but not an object
    assert an._load_cell_json(non_obj) is None
    ok = tmp_path / "ok.json"
    ok.write_text('{"k": 1}')
    assert an._load_cell_json(ok) == {"k": 1}


def test_resume_cell_skips_only_when_both_present_and_null_complete(tmp_path):
    delta_dir = tmp_path / "delta_floor"
    null_dir = tmp_path / "substrate_swap_null"
    delta_dir.mkdir()
    null_dir.mkdir()

    def _write(sub, obs, null):
        (delta_dir / f"em__{sub}.json").write_text(json.dumps(obs))
        (null_dir / f"em__{sub}.json").write_text(json.dumps(null))

    # (a) full-success null → resumable, returns (obs, null).
    _write("generic", _obs("generic", 0.5), _full_null())
    got = an._resume_cell("em", "generic", delta_dir, null_dir)
    assert got is not None and got[0]["delta_over_floor"] == 0.5
    assert got[1]["n_over_floor_resamples_used"] == 500

    # (b) degenerate (note-only) null → ALSO resumable (a completed cell, no work left).
    _write("elicit", _obs("elicit", 0.3), _degenerate_null())
    got_deg = an._resume_cell("em", "elicit", delta_dir, null_dir)
    assert got_deg is not None and "note" in got_deg[1]

    # (c) in-flight null (no n_over_floor signal, no note) → NOT resumable → recompute.
    (delta_dir / "em__mix.json").write_text(json.dumps(_obs("mix", 0.4)))
    (null_dir / "em__mix.json").write_text(json.dumps({"n_over_floor_resamples_used": 0}))
    assert an._resume_cell("em", "mix", delta_dir, null_dir) is None

    # (d) delta present but null missing → NOT resumable.
    (delta_dir / "em__generic2.json").write_text(json.dumps(_obs("generic2", 0.5)))
    assert an._resume_cell("em", "generic2", delta_dir, null_dir) is None

    # (e) corrupt (truncated) null → NOT resumable (recompute overwrites the partial).
    (delta_dir / "em__gc.json").write_text(json.dumps(_obs("gc", 0.5)))
    (null_dir / "em__gc.json").write_text('{"n_over_floor_resamples_used": 5')  # truncated
    assert an._resume_cell("em", "gc", delta_dir, null_dir) is None


# ── main(): mixed loaded + computed cells, summary includes BOTH ───────────────


def _run_main_two_cells(tmp_path, monkeypatch, *, no_resume: bool):
    """Drive main() over em/{generic (pre-seeded), elicit (computed)}; stubbed heavy compute.

    Returns (summary_dict, computed_substrates) — computed_substrates is the list of
    substrates observed_read was actually invoked for (so the test can assert generic was
    SKIPPED and elicit was COMPUTED).
    """
    out_dir = tmp_path / "eval_results" / "issue_813"
    delta_dir = out_dir / "delta_floor"
    null_dir = out_dir / "substrate_swap_null"
    delta_dir.mkdir(parents=True)
    null_dir.mkdir(parents=True)

    # Pre-seed the `generic` cell on disk (a finished cell from an earlier/parallel run).
    seeded_obs = _obs("generic", 0.9)
    seeded_null = _full_null(p95=0.20)
    (delta_dir / "em__generic.json").write_text(json.dumps(seeded_obs, default=float))
    (null_dir / "em__generic.json").write_text(json.dumps(seeded_null, default=float))

    computed: list[str] = []

    def _stub_observed(behavior, substrate, reduced_root, rb_main, rb_fact, wu_marker):
        computed.append(substrate)
        return _obs(substrate, 0.4)  # elicit computes dof=0.4

    def _stub_null(behavior, substrate, reduced_root, r_hat, n_resamples, *, n_refit_pairs):
        return _full_null(p95=0.35)

    def _stub_pairwise(
        observed_by_sub, behavior, reduced_root, r_hat, *, n_resamples, n_refit_pairs
    ):
        # A minimal pairwise record for the two substrates present (no bootstrap fit needed).
        subs = [s for s in an.SUBSTRATES if s in observed_by_sub]
        out = []
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                a, b = subs[i], subs[j]
                da = observed_by_sub[a]["delta_over_floor"]
                db = observed_by_sub[b]["delta_over_floor"]
                out.append(
                    {
                        "pair": f"{a}_vs_{b}",
                        "dv_space": "delta_over_floor",
                        "delta_over_floor_a": da,
                        "delta_over_floor_b": db,
                        "abs_diff": abs(da - db),
                        "ci_lo": 0.1,
                        "ci_hi": 0.9,
                        "ci_excludes_zero": True,
                    }
                )
        return out

    # Stub the top-of-main gates + heavy compute so main() runs on CPU in seconds.
    monkeypatch.setattr(an.fit658, "_assert_ridge_exactness", lambda: None)
    monkeypatch.setattr(an.fit658, "_resolve_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(an.fitM, "_load_rb_main", lambda: {})
    monkeypatch.setattr(an, "_r_hat_for", lambda *a, **k: None)
    monkeypatch.setattr(an, "observed_read", _stub_observed)
    monkeypatch.setattr(an, "substrate_swap_null", _stub_null)
    monkeypatch.setattr(an, "pairwise_substrate_diff_cis", _stub_pairwise)

    argv = [
        "issue813_analysis.py",
        "--behaviors",
        "em",
        "--substrates",
        "generic",
        "elicit",
        "--out-dir",
        str(out_dir),
        "--reduced-root",
        str(out_dir / "reduced"),
    ]
    if no_resume:
        argv.append("--no-resume")
    monkeypatch.setattr(sys, "argv", argv)

    rc = an.main()
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    return summary, computed


def test_main_resumes_seeded_cell_computes_other_summary_has_both(tmp_path, monkeypatch):
    """(i) pre-existing cell skipped+loaded, (ii) other computed, (iii) summary has BOTH."""
    summary, computed = _run_main_two_cells(tmp_path, monkeypatch, no_resume=False)

    # (ii) elicit was COMPUTED; (i) generic was SKIPPED (never passed to observed_read).
    assert computed == ["elicit"], computed

    obs = summary["per_behavior"]["em"]["observed"]
    null = summary["per_behavior"]["em"]["substrate_swap_null"]
    # (iii) BOTH cells present in the summary.
    assert set(obs) == {"generic", "elicit"}
    assert set(null) == {"generic", "elicit"}
    # The LOADED generic cell carries the pre-seeded values (round-trip equivalent).
    assert obs["generic"]["delta_over_floor"] == 0.9
    assert null["generic"]["null_over_floor_p95"] == 0.20
    # The COMPUTED elicit cell carries the stubbed compute values.
    assert obs["elicit"]["delta_over_floor"] == 0.4
    assert null["elicit"]["null_over_floor_p95"] == 0.35
    # The verdict + pairwise assembled over the mix of loaded + computed cells.
    assert "verdict" in summary["per_behavior"]["em"]
    assert len(summary["per_behavior"]["em"]["pairwise_substrate_diff"]) == 1


def test_no_resume_recomputes_every_cell(tmp_path, monkeypatch):
    """--no-resume forces the seeded cell to recompute (both substrates hit observed_read)."""
    summary, computed = _run_main_two_cells(tmp_path, monkeypatch, no_resume=True)
    # BOTH substrates were recomputed (the pre-seeded generic ignored).
    assert sorted(computed) == ["elicit", "generic"], computed
    obs = summary["per_behavior"]["em"]["observed"]
    # generic now carries the RECOMPUTED value (0.4), not the seeded 0.9.
    assert obs["generic"]["delta_over_floor"] == 0.4
