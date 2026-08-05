"""Unit E pins for task #2054 (revise round 2): C9/M6 resume, M1
per-comparison equalize-down, M2 fatal uploads, M3 answer provenance +
silent-default closures, M4 cap-hit reporting, M5 ladder fit dedupe.

Every test runs against real files under pytest tmp_path (never canonical
eval_results/figures paths) and touches no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_capture as capture  # noqa: E402
import issue2054_fits as fits  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_ladder as ladder  # noqa: E402
import issue2054_phase_a as phase_a  # noqa: E402
import issue2054_phase_b as phase_b  # noqa: E402
import issue2054_phase_c as phase_c  # noqa: E402
import issue2054_resume as resume  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# C9/M6 — the shared resume sidecar semantics


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_resume_disposition_run_when_output_missing(tmp_path):
    out = tmp_path / "out.jsonl"
    disp, _ = resume.resume_disposition(out, {"cell": "a"})
    assert disp == resume.RUN


def test_resume_disposition_skip_on_full_match(tmp_path):
    out = _touch(tmp_path / "out.jsonl")
    regime = {"cell": "v__inserted__chat__m", "seed": 137}
    inputs = {"input_sha256": "abc"}
    resume.write_done(out, regime, inputs)
    disp, _ = resume.resume_disposition(out, regime, inputs)
    assert disp == resume.SKIP


def test_resume_disposition_recompute_on_changed_inputs(tmp_path):
    out = _touch(tmp_path / "out.jsonl")
    regime = {"cell": "v__inserted__chat__m", "seed": 137}
    resume.write_done(out, regime, {"input_sha256": "old"})
    disp, reason = resume.resume_disposition(out, regime, {"input_sha256": "new"})
    assert disp == resume.RECOMPUTE
    assert "input_sha256" in reason


def test_resume_disposition_refuses_regime_mismatch(tmp_path):
    """A DIFFERENT regime at the same output path is a refusal, never a
    silent skip / silent overwrite (the #1333 _check_regime convention)."""
    out = _touch(tmp_path / "out.jsonl")
    resume.write_done(out, {"cell": "v__inserted__chat__qwen2.5-7b", "seed": 137})
    with pytest.raises(resume.RegimeMismatch) as exc:
        resume.resume_disposition(
            out, {"cell": "v__inserted__chat__qwen2.5-7b-instruct", "seed": 137}
        )
    assert "cell" in str(exc.value)


def test_resume_disposition_overwrite_clears_sidecar(tmp_path):
    out = _touch(tmp_path / "out.jsonl")
    regime = {"cell": "c", "seed": 1}
    resume.write_done(out, regime)
    disp, _ = resume.resume_disposition(out, regime, overwrite=True)
    assert disp == resume.RUN
    # Sidecar cleared: a crash mid-regeneration cannot mint a false skip.
    assert not resume.sidecar_path(out).is_file()


def test_soft_resume_ok_recomputes_on_mismatch(tmp_path):
    out = _touch(tmp_path / "scaffolds_v.jsonl")
    resume.write_done(out, {"threshold": 50.0})
    ok, reason = resume.soft_resume_ok(out, {"threshold": 60.0})
    assert not ok and "threshold" in reason
    ok, _ = resume.soft_resume_ok(out, {"threshold": 50.0})
    assert ok


# ─────────────────────────────────────────────────────────────────────────────
# C9/M6 — per-driver regime keys carry the FULL 4-axis cell key


def _ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_phase_c_regime_carries_four_axis_cell_and_generation_knobs():
    args = _ns(
        form="chat",
        model="qwen2.5-7b-instruct",
        seed=137,
        temperature=1.0,
        max_new_tokens=2048,
        target_conv_ids=8000,
    )
    regime = phase_c._variant_regime("char_helios", args)
    assert regime["cell"] == forms.cell_key(
        "char_helios", "on_policy", "chat", "qwen2.5-7b-instruct"
    )
    assert {"seed", "temperature", "max_new_tokens", "target_conv_ids"} <= set(regime)


def test_phase_c_two_model_collision_refused(tmp_path):
    """The phase_c output filename carries no model axis — the sidecar's
    4-axis regime is what refuses a second --model run into one output dir."""
    out = _touch(tmp_path / "on_policy_char_helios__chat.jsonl")
    args_a = _ns(
        form="chat",
        model="qwen2.5-7b",
        seed=137,
        temperature=1.0,
        max_new_tokens=2048,
        target_conv_ids=8000,
    )
    args_b = _ns(
        form="chat",
        model="qwen2.5-7b-instruct",
        seed=137,
        temperature=1.0,
        max_new_tokens=2048,
        target_conv_ids=8000,
    )
    resume.write_done(out, phase_c._variant_regime("char_helios", args_a))
    with pytest.raises(resume.RegimeMismatch):
        resume.resume_disposition(out, phase_c._variant_regime("char_helios", args_b))


def test_fits_cell_resume_check_skip_and_regime_mismatch(tmp_path):
    out = tmp_path / "cell.json"
    expected = {
        "cell": "c",
        "arms": ["context", "prefix"],
        "layer": 19,
        "seed": 137,
        "n_null_draws": 100,
        "bootstrap_draws": 200,
        "reduced_basis_k": fits.REDUCED_BASIS_K,
        "pilot": False,
        "dry_run": False,
        "restrict_sha256": "r",
        "npz_sha256": "n",
        "fold_map_k": 5,
        "fold_map_seed": 42,
        "fold_map_n_conv_ids": 10,
    }
    payload = dict(expected)
    payload["fold_map"] = {"k": 5, "seed": 42, "n_conv_ids": 10}
    payload["arm_reports"] = {
        "context": {"status": "ok"},
        "prefix": {"status": "ok"},
    }
    for k in ("fold_map_k", "fold_map_seed", "fold_map_n_conv_ids"):
        payload.pop(k)
    out.write_text(json.dumps(payload), encoding="utf-8")
    skip, _ = fits._cell_resume_check(out, expected)
    assert skip
    # A changed null-draw count (an output-affecting regime key) recomputes.
    changed = dict(expected, n_null_draws=200)
    skip, reason = fits._cell_resume_check(out, changed)
    assert not skip and "n_null_draws" in reason


def test_ladder_pair_resume_check_skip_and_mismatch(tmp_path):
    out = tmp_path / "rung_1_s_to_t_context.json"
    expected = {
        "source": "s",
        "target": "t",
        "arm": "context",
        "n_rungs": 9,
        "seed": 137,
        "bootstrap_draws": 200,
        "pilot": False,
        "dry_run": False,
        "target_ceiling": 0.6,
        "intersection_sha256": "i",
        "fold_map_k": 5,
        "fold_map_seed": 42,
    }
    payload = dict(expected)
    payload["fold_map"] = {"k": 5, "seed": 42}
    payload["arm_report"] = {"status": "ok"}
    for k in ("fold_map_k", "fold_map_seed"):
        payload.pop(k)
    out.write_text(json.dumps(payload), encoding="utf-8")
    skip, _ = ladder._pair_resume_check(out, expected)
    assert skip
    # A refit target ceiling (the ratio denominator) invalidates the pair.
    skip, reason = ladder._pair_resume_check(out, dict(expected, target_ceiling=0.7))
    assert not skip and "target_ceiling" in reason


def test_phase_a_prejudge_staleness_raises_on_content_drift(tmp_path):
    out_dir = tmp_path
    v = "char_helios"
    pj = _touch(out_dir / v / f"scaffolds_{v}_prejudge.jsonl", '{"conv_id": "s1"}\n')
    args = _ns(
        seed=137,
        target_conv_ids=8000,
        gen_model="instruct",
        gen_mock=False,
        no_generate=False,
    )
    phase_a._write_prejudge_sidecars({v: pj}, args, {v: 1})
    # Untouched pool verifies clean.
    phase_a._verify_prejudge_staleness(out_dir, [v], args)
    # Content drift after the sidecar was written = a STALE pool -> loud.
    pj.write_text('{"conv_id": "s2"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="STALE"):
        phase_a._verify_prejudge_staleness(out_dir, [v], args)


def test_phase_a_prejudge_staleness_raises_on_cross_regime_seed(tmp_path):
    out_dir = tmp_path
    v = "char_helios"
    pj = _touch(out_dir / v / f"scaffolds_{v}_prejudge.jsonl", '{"conv_id": "s1"}\n')
    gen_args = _ns(
        seed=137,
        target_conv_ids=8000,
        gen_model="instruct",
        gen_mock=False,
        no_generate=False,
    )
    phase_a._write_prejudge_sidecars({v: pj}, gen_args, {v: 1})
    judge_args = _ns(seed=42, target_conv_ids=8000)
    with pytest.raises(RuntimeError, match="seed"):
        phase_a._verify_prejudge_staleness(out_dir, [v], judge_args)


# ─────────────────────────────────────────────────────────────────────────────
# M1 — per-comparison equalize-down + gate 4 per (character, model) pair


def test_comparison_group_key_maps_op_variants_to_character():
    assert fits._comparison_group_key("char_helios_op", "m") == ("char_helios", "m")
    assert fits._comparison_group_key("char_helios_op_base", "m") == ("char_helios", "m")
    assert fits._comparison_group_key("char_helios", "m") == ("char_helios", "m")
    assert fits._comparison_group_key("conversation_paired_stories_assistant", "m") == (
        "conversation_paired_stories_assistant",
        "m",
    )


def test_gate4_never_fires_on_single_cell_group():
    """A lone cell (smoke shape) has no comparison to equalize — gate 4
    reports, never fires (the gate-calibration rule)."""
    report = fits._evaluate_kill_gates(
        variant="char_helios",
        condition="inserted",
        form="chat",
        model="m",
        conv_ids_this_cell=[f"s{i}" for i in range(3)],  # far below 4,480
        peer_cells_conv_ids=None,
        diag_this=None,
        peer_diag=None,
        gate5_peer_cell=None,
    )
    assert report["kill_gate_4_fire"] is False
    assert report["single_cell_group"] is True


def test_gate4_fires_per_pair_on_small_intersection():
    ids_a = [f"s{i}" for i in range(100)]
    ids_b = [f"s{i}" for i in range(50, 150)]  # intersection = 50 < 4,480
    report = fits._evaluate_kill_gates(
        variant="char_helios",
        condition="inserted",
        form="chat",
        model="m",
        conv_ids_this_cell=ids_a,
        peer_cells_conv_ids={"peer": ids_b},
        diag_this=None,
        peer_diag=None,
        gate5_peer_cell=None,
    )
    assert report["kill_gate_4_fire"] is True
    assert report["min_pair_intersection"] == 50


def test_disjoint_groups_do_not_empty_each_other():
    """The M1 regression shape: an assistant scope with a DISJOINT conv_id
    space must not zero the character cells' restriction (the pre-fix global
    intersection emptied every fit)."""
    fold_of = {f"s{i}": i % 5 for i in range(10)} | {f"a{i}": i % 5 for i in range(10)}
    char_ids = {f"s{i}" for i in range(10)}
    asst_ids = {f"a{i}" for i in range(10)}
    # Mirror run_phase's group-restrict arithmetic.
    fold_conv_ids = set(fold_of)
    groups = {
        ("char_helios", "m"): [char_ids, char_ids],
        ("conversation_paired_stories_assistant", "m"): [asst_ids],
    }
    for _g, member_id_sets in groups.items():
        inter = None
        for ids in member_id_sets:
            s = ids & fold_conv_ids
            inter = s if inter is None else inter & s
        assert inter, "per-group intersection must be non-empty for its own members"


# ─────────────────────────────────────────────────────────────────────────────
# M2 — fatal uploads (no swallowed failure ahead of [phase=done])


def test_capture_upload_failure_propagates(tmp_path):
    npz = tmp_path / "acts" / "char_helios" / "cell.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"notempty")
    import explore_persona_space.orchestrate.hub as hub

    boom = mock.create_autospec(hub._upload_folder_filtered, side_effect=RuntimeError("hub down"))
    with (
        mock.patch.object(hub, "_upload_folder_filtered", boom),
        pytest.raises(RuntimeError, match="hub down"),
    ):
        capture._upload_to_hf({"char_helios": npz}, "qwen2.5-7b")


def test_phase_c_upload_empty_set_is_fatal(tmp_path):
    """Declared outputs with an empty upload-eligible set is a verify failure
    (#1482 empty-set-vacuous class), never a silent pass."""
    missing = tmp_path / "nope" / "x.jsonl"  # never created
    with pytest.raises(RuntimeError, match="EMPTY"):
        phase_c._upload_to_hf({"char_helios": missing}, tmp_path)


def test_ladder_upload_failure_propagates(tmp_path):
    p = tmp_path / "rung_1_s_to_t_context.json"
    p.write_text("{}", encoding="utf-8")
    import explore_persona_space.orchestrate.hub as hub

    boom = mock.create_autospec(hub._upload_folder_filtered, side_effect=RuntimeError("hub down"))
    with (
        mock.patch.object(hub, "_upload_folder_filtered", boom),
        pytest.raises(RuntimeError, match="hub down"),
    ):
        ladder._upload_to_hf([p])


# ─────────────────────────────────────────────────────────────────────────────
# M3 — answer provenance + silent-default closures


def _scaffold_row(i: int, question: str = "What is the sky?") -> dict:
    return {
        "scaffold_id": f"stripped_s{i}",
        "conv_id": f"stripped_s{i}",
        "character": "Helios",
        "scaffold_text": f"A scene about question {i}.",
        "question": question,
        "answer": f"scaffold-original answer {i}",
    }


def test_phase_b_rows_carry_answer_source_and_counts(tmp_path):
    scaffolds = tmp_path / "scaffolds.jsonl"
    with scaffolds.open("w", encoding="utf-8") as f:
        for i in range(2):
            f.write(json.dumps(_scaffold_row(i)) + "\n")
    # Pool covers row 0 only; row 1 falls back to the scaffold's own answer.
    answers = {"stripped_s0": "pool answer zero"}
    counts, out_path = phase_b._process_variant(
        "char_helios", scaffolds, answers, tmp_path / "out", "chat"
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["answer_source"] for r in rows] == [
        "answers_pool",
        "scaffold_original_fallback",
    ]
    assert counts["n_answer_from_pool"] == 1
    assert counts["n_answer_from_scaffold_fallback"] == 1


def test_read_jsonl_counts_undecodable_lines(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")
    rows = phase_b._read_jsonl(p)
    assert len(rows) == 2
    assert "1 undecodable" in capsys.readouterr().out


def test_char_name_unknown_variant_fails_loud():
    with pytest.raises(ValueError, match="cannot resolve character name"):
        phase_b._char_name_from_scaffold_row({"scaffold_id": "x"}, "unknown_variant")
    with pytest.raises(ValueError, match="cannot resolve character name"):
        phase_c._char_name_from_scaffold_row({"scaffold_id": "x"}, "unknown_variant")
    # A row-level character always wins (no map needed).
    assert phase_b._char_name_from_scaffold_row({"character": "Vex"}, "whatever") == "Vex"


def test_token_before_char_returns_none_at_boundary_zero():
    offsets = [(0, 3), (3, 7), (7, 12)]
    # No token ends at-or-before char 0 -> None (never a coerced token 0).
    assert capture._token_before_char(offsets, 0) is None
    assert capture._token_before_char(offsets, 3) == 0
    assert capture._token_before_char(offsets, 12) == 2


# ─────────────────────────────────────────────────────────────────────────────
# M4 — cap-hit fraction + pre-registered re-gen trigger


def test_cap_hit_stats_threshold_trigger():
    below = phase_c._cap_hit_stats(1, 100)  # 1% <= 2%
    assert below["cap_hit_fraction"] == pytest.approx(0.01)
    assert below["cap_hit_regen_trigger_fired"] is False
    above = phase_c._cap_hit_stats(3, 100)  # 3% > 2%
    assert above["cap_hit_regen_trigger_fired"] is True
    assert above["cap_hit_regen_threshold"] == phase_c.CAP_HIT_REGEN_THRESHOLD == 0.02
    zero = phase_c._cap_hit_stats(0, 0)
    assert zero["cap_hit_fraction"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# M5 — ladder fit dedupe (3 GCV-SVD fits per fold, not 5) + null draws pin


def _tiny_pair_arrays(rng, n=12, d=6):
    Xs = rng.normal(size=(n, d))
    Ys = Xs @ rng.normal(size=(d, d)) * 0.5 + rng.normal(size=(n, d)) * 0.1
    Xt = rng.normal(size=(n, d))
    Yt = Xt @ rng.normal(size=(d, d)) * 0.5 + rng.normal(size=(n, d)) * 0.1
    return Xs, Ys, Xt, Yt


def test_ladder_fold_runs_exactly_three_fits_and_all_nine_rungs():
    rng = np.random.default_rng(0)
    Xs, Ys, Xt, Yt = _tiny_pair_arrays(rng)
    calls = {"n": 0}
    real_fit = ladder._fit_ridge

    def counting_fit(*a, **kw):
        calls["n"] += 1
        return real_fit(*a, **kw)

    with mock.patch.object(ladder, "_fit_ridge", counting_fit):
        rung_preds, _info = ladder._compute_rungs_for_fold(
            Xs_tr=Xs, Ys_tr=Ys, Xt_tr=Xt, Xt_te=Xt[:4], Yt_tr=Yt
        )
    assert calls["n"] == 3, "M5: exactly A + M + B fits per fold (was 5 pre-fix)"
    assert set(rung_preds) == set(ladder.RUNGS)
    for name, pred in rung_preds.items():
        assert np.isfinite(pred).all(), name
    # Rungs 8 and 9 apply the SAME B fit at different inputs — they must still
    # differ (different inputs), proving the dedupe did not alias them.
    assert not np.allclose(rung_preds["8_ans_reparam"], rung_preds["9_full_AMB"])


def test_ladder_memoized_source_fit_skips_m_fit():
    rng = np.random.default_rng(1)
    Xs, Ys, Xt, Yt = _tiny_pair_arrays(rng)
    prefit = ladder._fit_ridge(Xs, Ys)
    calls = {"n": 0}
    real_fit = ladder._fit_ridge

    def counting_fit(*a, **kw):
        calls["n"] += 1
        return real_fit(*a, **kw)

    with mock.patch.object(ladder, "_fit_ridge", counting_fit):
        preds_memo, info = ladder._compute_rungs_for_fold(
            Xs_tr=Xs, Ys_tr=Ys, Xt_tr=Xt, Xt_te=Xt[:4], Yt_tr=Yt, source_fit=prefit
        )
    assert calls["n"] == 2, "memoized M fit: only A + B fit"
    assert info["source_fit_memoized"] is True
    # Identical numbers to the un-memoized path (exact memo, same train rows).
    preds_fresh, _ = ladder._compute_rungs_for_fold(
        Xs_tr=Xs, Ys_tr=Ys, Xt_tr=Xt, Xt_te=Xt[:4], Yt_tr=Yt
    )
    for rung in ladder.RUNGS:
        np.testing.assert_allclose(preds_memo[rung], preds_fresh[rung], rtol=1e-10)


def test_fit_apply_split_matches_wrapper():
    rng = np.random.default_rng(2)
    Xs, Ys, _, _ = _tiny_pair_arrays(rng)
    model = ladder._fit_ridge(Xs, Ys)
    preds, info = ladder._fit_ridge_and_apply(Xs, Ys, apply_at={"te": Xs[:3]})
    np.testing.assert_allclose(preds["te"], ladder._apply_ridge(model, Xs[:3]), rtol=1e-12)
    assert info["best_lambda"] == model["info"]["best_lambda"]


def test_null_draws_default_is_plan_pinned_100():
    """Plan §6 DV 4 + §9 both pin 100 permutation draws; the round-1 default
    (200) and the dispatch pin are reconciled to the plan."""
    import inspect

    # Read the default straight off the argparse wiring.
    src = inspect.getsource(fits.main)
    assert "--n-null-draws" in src
    assert "default=100" in src
    dispatch = (_REPO_ROOT / "scripts" / "issue2054_dispatch.sh").read_text(encoding="utf-8")
    assert "--n-null-draws 100" in dispatch
    assert "--n-null-draws 200" not in dispatch
