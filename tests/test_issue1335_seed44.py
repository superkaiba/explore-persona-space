"""#1335 seed44-base-rungs round pins.

Covers the round's three new knobs:
  1. the issue1335_run.sh model-subset knob (I1335_MODELS; default BOTH —
     existing callers byte-unchanged; invalid/empty fail loud) + the seed44
     thin driver's env contract (executable, via a stubbed inner run.sh);
  2. the dual-reference --seed-compare extension (_ref_headline shape
     tolerance across ladder_summary vs seed_comparison references,
     fail-loud on a missing model; reference_2/cross_seed_2 blocks; the
     wired-in collapse audit incl. its degenerate-input gates);
  3. the collapse-conditional refit's row selection (exclusion sets from a
     tiny rollout fixture + the per-persona keep mask, incl. the all-dropped
     fail-loud gate).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

r1335 = pytest.importorskip("issue1335_render_rungs")
c1310 = pytest.importorskip("issue1310_common")
f1335 = pytest.importorskip("issue1335_fit")
refit = pytest.importorskip("issue1335_refit_companions")

RUN_SH = REPO_ROOT / "scripts" / "issue1335_run.sh"
SEED44_SH = REPO_ROOT / "scripts" / "issue1335_seed44_run.sh"


# ---------------------------------------------------------------------------
# (1) model-subset knob + seed44 driver env contract
# ---------------------------------------------------------------------------


def _run_models_knob(env_line: str) -> subprocess.CompletedProcess:
    """Execute ONLY the marked models-knob region of issue1335_run.sh under
    bash (the executable pin — never a grep of the script text)."""
    snippet = (
        f"{env_line}\n"
        f"source <(sed -n '/^# models-knob-begin/,/^# models-knob-end/p' {RUN_SH})\n"
        'echo "MODELS=${MODELS[*]}|CSV=$MODELS_CSV"\n'
    )
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO_ROOT)


def test_models_knob_default_both_and_subset():
    # unset -> both models (existing callers byte-unchanged)
    res = _run_models_knob("unset I1335_MODELS")
    assert res.returncode == 0, res.stderr
    assert "MODELS=base instruct|CSV=base,instruct" in res.stdout
    # subset honored
    res = _run_models_knob('I1335_MODELS="base"')
    assert res.returncode == 0, res.stderr
    assert "MODELS=base|CSV=base" in res.stdout
    # invalid model -> loud exit 5
    res = _run_models_knob('I1335_MODELS="base bogus"')
    assert res.returncode == 5
    assert "unknown model" in res.stderr
    # set-but-empty -> loud exit 5 (never a silent zero-model run)
    res = _run_models_knob('I1335_MODELS=""')
    assert res.returncode == 5
    assert "empty" in res.stderr


def test_seed44_driver_env_contract(tmp_path):
    """The thin seed44 driver execs issue1335_run.sh with the round's env:
    seed 44, segregated HF prefix/dirs, base-only, gap rungs, seed-compare
    with BOTH committed references. Executable via a stubbed inner run.sh."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    stub = scripts / "issue1335_run.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "for v in EPM_I1335_GEN_SEED EPM_I1335_HF_PREFIX DATA_DIR OUT_DIR FIG_DIR \\\n"
        "         I1335_GEN_RUNGS I1335_TF_RUNGS I1335_ALL_RUNGS I1335_MODELS \\\n"
        "         I1335_SUMMARY_MODE I1335_REFERENCE_SUMMARY I1335_REFERENCE_SUMMARY_2; do\n"
        '  echo "$v=${!v-UNSET}"\n'
        "done\n"
    )
    stub.chmod(0o755)
    driver = scripts / "issue1335_seed44_run.sh"
    driver.write_text(SEED44_SH.read_text())
    driver.chmod(0o755)
    res = subprocess.run(["bash", str(driver)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    got = dict(line.split("=", 1) for line in res.stdout.strip().splitlines())
    assert got["EPM_I1335_GEN_SEED"] == "44"
    assert got["EPM_I1335_HF_PREFIX"] == "issue1335_ablation_ladder/seed44_base_rungs"
    assert got["DATA_DIR"] == "data/issue_1335_seed44"
    assert got["OUT_DIR"] == "eval_results/issue_1335/seed44-base-rungs"
    assert got["I1335_GEN_RUNGS"] == "r1_qa_oneline r3_persona r4_fictionframe r7_endpoint"
    assert got["I1335_TF_RUNGS"] == ""
    assert got["I1335_MODELS"] == "base"
    assert got["I1335_SUMMARY_MODE"] == "seed-compare"
    assert got["I1335_REFERENCE_SUMMARY"] == "eval_results/issue_1335/ladder_summary.json"
    assert (
        got["I1335_REFERENCE_SUMMARY_2"]
        == "eval_results/issue_1335/seed43-gap-rungs/seed_comparison.json"
    )


# ---------------------------------------------------------------------------
# (2) dual-reference seed-compare + collapse audit
# ---------------------------------------------------------------------------

_GAP = {"value": 0.30, "ci_lo": 0.25, "ci_hi": 0.35, "ci_method": "joint-draws"}
_FRAMING = {"value": 0.12, "ci_lo": 0.08, "ci_hi": 0.16, "ci_method": "joint-draws"}


def test_ref_headline_reads_both_reference_shapes():
    ladder_shape = {"per_model": {"base": {"gap": {"G": _GAP}, "deltas": {"framing": _FRAMING}}}}
    seedcmp_shape = {"per_model": {"base": {"gap_G": _GAP, "framing": _FRAMING}}}
    for ref in (ladder_shape, seedcmp_shape):
        gap, framing = f1335._ref_headline(ref, "ref", "base")
        assert gap["value"] == 0.30 and framing["value"] == 0.12
    # fail-loud: a model the reference does not carry (never silently tolerated)
    with pytest.raises(AssertionError, match=r"lacks per_model\.instruct gap"):
        f1335._ref_headline(ladder_shape, "ref", "instruct")
    # fail-loud: gap present but framing missing in EITHER shape
    with pytest.raises(AssertionError, match="framing"):
        f1335._ref_headline({"per_model": {"base": {"gap_G": _GAP}}}, "ref", "base")


def test_dual_reference_seed_compare_fixture(tmp_path):
    """--reference-summary-2 adds reference_2/cross_seed_2 per model + the
    top-level reference_2 block; the primary blocks are byte-shape-unchanged."""
    from tests.test_issue1335_ladder import (  # local helpers (fixture writers)
        _write_cell,
        _write_endpoint_rollouts,
        _write_matched,
    )

    out = tmp_path / "eval"
    out.mkdir()
    rng = np.random.default_rng(7)
    for slug, v in (("r1_qa_oneline", 0.60), ("r3_persona", 0.50), ("r4_fictionframe", 0.40)):
        _write_matched(out, f"{slug}__base__ctx", slug, v, rng)
    for persona in c1310.PERSONA_LABELS:
        _write_matched(out, f"r7_endpoint__base__{persona}__ctx", "r7_endpoint", 0.15, rng)
        _write_cell(out, f"r7_endpoint__base__{persona}__ctx", "r7_endpoint", 0.15, 2000)
    ref1 = tmp_path / "ladder_summary.json"
    ref1.write_text(
        json.dumps(
            {
                "code_sha": "refsha042",
                "per_model": {"base": {"gap": {"G": _GAP}, "deltas": {"framing": _FRAMING}}},
            }
        )
    )
    gap43 = {"value": 0.099, "ci_lo": 0.082, "ci_hi": 0.115, "ci_method": "joint-draws"}
    fr43 = {"value": 0.026, "ci_lo": -0.001, "ci_hi": 0.052, "ci_method": "joint-draws"}
    ref2 = tmp_path / "seed_comparison.json"
    ref2.write_text(
        json.dumps(
            {
                "code_sha": "refsha043",
                "gen_seed": 43,
                "per_model": {"base": {"gap_G": gap43, "framing": fr43}},
            }
        )
    )
    data_dir = tmp_path / "data"
    _write_endpoint_rollouts(data_dir, "base")
    args = SimpleNamespace(
        out_dir=out,
        seed=0,
        reference_summary=ref1,
        reference_summary_2=ref2,
        data_dir=data_dir,
    )
    res = f1335.build_seed_comparison(args, ["base"], smoke=False)
    pm = res["per_model"]["base"]
    # primary blocks unchanged in shape + arithmetic
    assert pm["seed42_reference"] == {"gap_G": _GAP, "framing": _FRAMING}
    assert pm["cross_seed"]["gap_G"]["value"] == pytest.approx(0.45 - 0.30, abs=1e-9)
    # second-reference blocks
    assert pm["reference_2"] == {"gap_G": gap43, "framing": fr43}
    cs2 = pm["cross_seed_2"]
    assert cs2["gap_G"]["value"] == pytest.approx(0.45 - 0.099, abs=1e-9)
    assert cs2["gap_G"]["ci_method"] == "variance-sum-independent-runs"
    assert res["reference_2"]["code_sha"] == "refsha043"
    assert res["reference_2"]["gen_seed"] == 43
    # collapse audit rides the summary (fixture: 8 lines, 1 under-floor, 1 agree)
    ca = pm["collapse_audit"]
    assert ca["n_lines"] == 8
    assert ca["under_floor_lines"] == 1
    assert ca["under_floor_per_slot"] == {"slot4": 1}
    assert ca["under_floor_per_persona"] == {"Vex": 1}
    assert ca["slot4_lines"] == 2
    assert ca["slot4_exact_agree"] == 1
    assert ca["slot4_exact_agree_per_persona"] == {"Vex": 1}
    # fail-loud degenerate gates: missing second reference; missing rollouts
    args_bad2 = SimpleNamespace(
        out_dir=out,
        seed=0,
        reference_summary=ref1,
        reference_summary_2=tmp_path / "missing2.json",
        data_dir=data_dir,
    )
    with pytest.raises(AssertionError, match="second reference missing"):
        f1335.build_seed_comparison(args_bad2, ["base"], smoke=False)
    args_no_rollouts = SimpleNamespace(
        out_dir=out,
        seed=0,
        reference_summary=ref1,
        reference_summary_2=None,
        data_dir=tmp_path / "empty_data",
    )
    with pytest.raises(AssertionError, match="missing endpoint rollouts"):
        f1335.build_seed_comparison(args_no_rollouts, ["base"], smoke=False)


# ---------------------------------------------------------------------------
# (3) collapse-conditional refit row selection
# ---------------------------------------------------------------------------


def _rollout_row(sc: str, persona: str, slot: int, completion: str, n_tok: int) -> dict:
    return {
        "scenario_id": sc,
        "persona": persona,
        "slot": slot,
        "completion": completion,
        "n_completion_tokens": n_tok,
    }


def test_collapse_exclusion_sets_and_keep_mask(tmp_path):
    rows = [
        # sc0/Wren: slot-4 collapsed to "I agree." (3 tokens -> also under-floor)
        _rollout_row("sc0", "Wren", 3, "A fine line.", 6),
        _rollout_row("sc0", "Wren", 4, "I agree.", 3),
        # sc1/Wren: healthy slot 4, but an under-floor slot-2 line
        _rollout_row("sc1", "Wren", 2, "Ha.", 2),
        _rollout_row("sc1", "Wren", 4, "A long considered reply.", 8),
        # sc0/Vex: healthy everywhere (same scenario, different persona — must
        # NOT be dragged out by sc0/Wren's collapse: trajectories are
        # (scenario, persona) pairs)
        _rollout_row("sc0", "Vex", 4, "A perfectly healthy line.", 7),
        # sc2/Dana: slot-4 "I agree." with a trailing space (strip-matched)
        _rollout_row("sc2", "Dana", 4, "I agree. ", 4),
    ]
    p = tmp_path / "gen.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ex = refit.collapse_exclusion_sets(p)
    assert ex["n_lines"] == 6
    assert ex["agree_traj"] == {("sc0", "Wren"), ("sc2", "Dana")}
    assert ex["under_floor_traj"] == {("sc0", "Wren"), ("sc1", "Wren")}
    assert ex["n_under_floor_lines"] == 2

    store = {
        "char_ids": np.asarray(["Wren", "Wren", "Wren", "Wren", "Vex", "Dana"]),
        "group_ids": np.asarray(["sc0", "sc0", "sc1", "sc1", "sc0", "sc2"]),
    }
    keep = refit.collapse_keep_mask(store, "Wren", ex["agree_traj"])
    # Wren keeps only its sc1 rows; sc0/Vex + sc2/Dana untouched by this unit
    assert keep.tolist() == [False, False, True, True, False, False]
    # both Wren trajectories are under-floor-excluded -> the all-dropped gate fires
    with pytest.raises(AssertionError, match="dropped EVERY Wren row"):
        refit.collapse_keep_mask(store, "Wren", ex["under_floor_traj"])
    # Vex under the agree exclusion keeps its sc0 row (persona-keyed exclusion)
    keep_vex = refit.collapse_keep_mask(store, "Vex", ex["agree_traj"])
    assert keep_vex.tolist() == [False, False, False, False, True, False]
    # empty rollout file fail-loud
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(AssertionError, match="empty rollout JSONL"):
        refit.collapse_exclusion_sets(empty)
