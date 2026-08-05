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
        gen_draw_n=None,
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
        gen_draw_n=None,
        gen_model="instruct",
        gen_mock=False,
        no_generate=False,
    )
    phase_a._write_prejudge_sidecars({v: pj}, gen_args, {v: 1})
    judge_args = _ns(seed=42, target_conv_ids=8000, gen_draw_n=None)
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


# ─────────────────────────────────────────────────────────────────────────────
# Round 3 — C-R2-1 (gen-resume upload bypass) + M-R2-1 (ladder fleet sizing)

import shutil  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import issue2054_pilot as pilot  # noqa: E402

_ASSISTANT = "conversation_paired_stories_assistant"
_INSTRUCT = "qwen2.5-7b-instruct"
_BASE = "qwen2.5-7b"


def _vartmp_dir(prefix: str) -> Path:
    """A NON-/tmp scratch dir: phase_a/ladder read `str(out_dir).startswith("/tmp/")`
    as the smoke guard, so exercising the PRODUCTION upload/enumeration path
    needs an out-root outside /tmp (pytest tmp_path lives under /tmp)."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir="/var/tmp"))


def test_gen_stage_resumed_still_uploads_prejudge(monkeypatch):
    """C-R2-1 (fails-pre-fix): a crash AT the fail-loud gen upload leaves valid
    done-sidecars, so the standard crash-recovery re-run RESUMES — and MUST
    re-attempt the upload (idempotent bulk commit). The pre-fix resumed branch
    skipped it on the false premise "prior gen leg already uploaded", printing
    [phase=done] with the prejudge pools never on HF (the #521 class)."""
    out_dir = _vartmp_dir("i2054_c_r2_1_")
    try:
        variant = "char_helios"
        rows = [{"conv_id": f"stripped_s{i}", "scaffold_text": f"scene {i}"} for i in range(3)]
        pj = phase_a._prejudge_path(out_dir, variant)
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        sidecar_args = SimpleNamespace(
            seed=137,
            target_conv_ids=50,
            gen_draw_n=None,
            gen_model="instruct",
            gen_mock=False,
            no_generate=False,
        )
        phase_a._write_prejudge_sidecars({variant: pj}, sidecar_args, {variant: len(rows)})
        gen_dir = out_dir / variant / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "gen_raw.jsonl").write_text("{}\n", encoding="utf-8")

        with mock.patch.object(phase_a, "_upload_scaffold_files", autospec=True) as upload:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "issue2054_phase_a.py",
                    "--stage",
                    "gen",
                    "--output-dir",
                    str(out_dir),
                    "--variants",
                    variant,
                    "--target-conv-ids",
                    "50",
                    "--seed",
                    "137",
                ],
            )
            with pytest.raises(SystemExit) as ei:
                phase_a.main()
        assert ei.value.code == 0
        assert upload.call_count == 1, "resumed gen leg must re-attempt the fail-loud upload"
        (_call_out_dir, files), kwargs = upload.call_args
        assert kwargs == {"fail_loud": True}
        uploaded = {str(f) for f in files}
        assert str(pj) in uploaded, "prejudge pool must ride the resumed re-upload"
        assert str(resume.sidecar_path(pj)) in uploaded, (
            "the judge leg's staleness-anchor sidecar must ride the resumed re-upload"
        )
        assert str(gen_dir / "gen_raw.jsonl") in uploaded
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _cell(variant, condition, form, model):
    return (variant, condition, form, model, None)


def test_pair_class_predicates_cover_plan6_and_reject_unregistered():
    a_chat = _cell(_ASSISTANT, "inserted", "chat", _INSTRUCT)
    a_bare = _cell(_ASSISTANT, "inserted", "bare_text", _INSTRUCT)
    helios_b = _cell("char_helios", "inserted", "bare_label", _INSTRUCT)
    wren_b = _cell("char_wren", "inserted", "bare_label", _INSTRUCT)
    helios_d = _cell("char_helios_op", "on_policy", "bare_label", _INSTRUCT)
    helios_c = _cell("char_helios_op", "cell_c", "chat", _INSTRUCT)
    a_chat_base = _cell(_ASSISTANT, "inserted", "chat", _BASE)

    assert ladder._pair_class(a_chat, a_bare) == "cross_framing"
    assert ladder._pair_class(helios_b, wren_b) == "cross_character"
    # 2x2 (plan §4 Block 3): same (character, model) group, and the assistant
    # chat x inserted anchor (a) pairs with every group's (b)/(c)/(d).
    assert ladder._pair_class(helios_b, helios_d) == "twobytwo"
    assert ladder._pair_class(helios_d, helios_c) == "twobytwo"
    assert ladder._pair_class(a_chat, helios_c) == "twobytwo"
    assert ladder._pair_class(helios_b, a_chat) == "twobytwo"
    assert ladder._pair_class(a_chat, a_chat_base) == "cross_model"
    # No §6 read consumes these:
    # cross-variant AND cross-form AND cross-condition at once
    assert ladder._pair_class(a_bare, helios_d) is None
    # on-policy cross-framing is NOT a framing read (§4 interpretive split)
    helios_d_chat = _cell("char_helios_op", "on_policy", "chat", _INSTRUCT)
    assert ladder._pair_class(helios_d, helios_d_chat) == "twobytwo"  # same group still pairs
    wren_d = _cell("char_wren_op", "on_policy", "bare_label", _INSTRUCT)
    a_bare_vs_wren_d = ladder._pair_class(a_bare, wren_d)
    assert a_bare_vs_wren_d is None, "bare-text assistant is not the 2x2 (a) anchor"


def test_enumerate_ordered_pairs_plan6_is_a_restricted_subset():
    """M-R2-1(a): the production enumeration restricts to §6 classes; 'all'
    reproduces the full ordered product (explicit opt-in)."""
    cells = []
    story_forms = ("bare_label", "attrib_quoted", "bare_paragraph")
    for model in (_INSTRUCT, _BASE):
        for form in ("chat", "bare_text", *story_forms):
            cells.append(_cell(_ASSISTANT, "inserted", form, model))
        for char in ("char_helios", "char_wren", "char_dana", "char_vex"):
            for form in story_forms:
                cells.append(_cell(char, "inserted", form, model))
                cells.append(_cell(f"{char}_op", "on_policy", form, model))
            cells.append(_cell(f"{char}_op", "cell_c", "chat", model))
    full = [(s, t) for s in cells for t in cells if s != t]
    restricted = ladder._enumerate_ordered_pairs(cells, smoke=False)
    unrestricted = ladder._enumerate_ordered_pairs(cells, smoke=False, pair_classes=("all",))
    assert len(unrestricted) == len(full) == len(cells) * (len(cells) - 1)
    assert 0 < len(restricted) < 0.5 * len(full), (len(restricted), len(full))
    assert set(restricted) <= set(full)
    for s, t in restricted:
        assert ladder._pair_class(s, t) in ladder.PLAN6_PAIR_CLASSES


def test_fleet_projection_writes_fields_and_enforces_fence(tmp_path, capsys):
    """M-R2-1(b): the pilot report carries the pilot->fleet projection (the
    reviewer's mechanizable check) and an armed over-budget projection raises
    the designed halt; unarmed (fits) it only WARNs."""
    report = tmp_path / "pilot_gate_report.json"
    logged: list[str] = []
    out = pilot.fleet_projection_update(
        report,
        {"phase": "x", "wall_seconds": 2.0},
        wall_seconds=2.0,
        n_fleet_units=10,
        fold_k=5,
        log=logged.append,
        max_fleet_wall_hours=1.0,
        units_basis="pending (pair, arm) units",
    )
    assert out["projected_fleet_wall_seconds"] == 100.0
    assert out["fence_floor_seconds"] == 200.0
    on_disk = json.loads(report.read_text(encoding="utf-8"))
    for key in (
        "projected_fleet_wall_seconds",
        "fence_floor_seconds",
        "n_fleet_units",
        "fold_k",
        "max_fleet_wall_hours",
    ):
        assert key in on_disk, key
    # Armed fence: over-budget RAISES (report still written first).
    with pytest.raises(pilot.FleetWallExceeded):
        pilot.fleet_projection_update(
            report,
            {"phase": "x"},
            wall_seconds=3600.0,
            n_fleet_units=100,
            fold_k=5,
            log=logged.append,
            max_fleet_wall_hours=12.0,
        )
    assert json.loads(report.read_text(encoding="utf-8"))["projected_fleet_wall_seconds"] > 0
    # Unarmed (fits): the same over-budget projection only WARNs.
    pilot.fleet_projection_update(
        report,
        {"phase": "x"},
        wall_seconds=3600.0,
        n_fleet_units=100,
        fold_k=5,
        log=logged.append,
        max_fleet_wall_hours=None,
    )
    assert any(m.startswith("WARN ") for m in logged)


def _write_ladder_fixture(root: Path, rng, cells, n=12, d=6):
    """Canonical-layout activation .npz per cell + shared fold map + fits dir."""
    acts_dir = root / "acts"
    conv_ids = np.array([f"c{i}" for i in range(n)])
    for variant, condition, form, model, _p in cells:
        key = forms.cell_key(variant, condition, form, model)
        cell_path = acts_dir / variant / f"{key}.npz"
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cell_path,
            conv_id=conv_ids,
            v_C=rng.normal(size=(n, d)).astype(np.float32),
            v_A=rng.normal(size=(n, d)).astype(np.float32),
            v_P=rng.normal(size=(n, d)).astype(np.float32),
            v_P_present=np.ones(n, dtype=bool),
        )
    fold_map = {
        "fold_of": {f"c{i}": i % 5 for i in range(n)},
        "k": 5,
        "seed": 137,
        "n_conv_ids": n,
    }
    fm_path = root / "shared_fold_map.json"
    fm_path.write_text(json.dumps(fold_map), encoding="utf-8")
    fits_dir = root / "fits"
    fits_dir.mkdir(exist_ok=True)
    return acts_dir, fm_path, fits_dir


def test_ladder_run_phase_restricts_pairs_projects_and_resumes(monkeypatch):
    """M-R2-1 end-to-end on the PRODUCTION path (non-/tmp out-root, no dry-run):
    restricted enumeration, projection in report + digest, and a full re-run
    resumes with 0 pending units (pilot gate skipped — #1586 pending-aware)."""
    root = _vartmp_dir("i2054_m_r2_1_")
    try:
        rng = np.random.default_rng(0)
        cells = [
            _cell(_ASSISTANT, "inserted", "chat", _INSTRUCT),
            _cell(_ASSISTANT, "inserted", "bare_text", _INSTRUCT),
            _cell("char_helios", "inserted", "bare_label", _INSTRUCT),
            _cell("char_wren", "inserted", "bare_label", _INSTRUCT),
        ]
        acts_dir, fm_path, fits_dir = _write_ladder_fixture(root, rng, cells)
        out_dir = root / "ladder_out"
        argv = [
            "issue2054_ladder.py",
            "--activations-dir",
            str(acts_dir),
            "--fits-dir",
            str(fits_dir),
            "--fold-map",
            str(fm_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            "137",
            "--bootstrap-draws",
            "8",
            "--skip-upload",
            "--variants",
            f"{_ASSISTANT},char_helios,char_wren",
            "--models",
            _INSTRUCT,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as ei:
            ladder.main()
        assert ei.value.code == 0
        digest = json.loads((out_dir / "ladder_digest.json").read_text(encoding="utf-8"))
        # 4 cells -> full product 12; §6 classes: 2 cross_framing +
        # 2 cross_character + 4 twobytwo (anchor <-> helios/wren) = 8.
        assert digest["n_full_ordered_product"] == 12
        assert digest["n_pairs_enumerated"] == 8
        assert digest["pair_class_counts"] == {
            "cross_framing": 2,
            "cross_character": 2,
            "twobytwo": 4,
        }
        assert digest["n_units_pending"] == 16  # 8 pairs x 2 arms
        assert digest["projected_fleet_wall_seconds"] is not None
        report = json.loads((out_dir / "pilot_gate_report.json").read_text(encoding="utf-8"))
        assert report["n_fleet_units"] == 16
        assert report["fold_k"] == 5
        # Both fields round independently to 0.1 s of the RAW projection, so
        # compare with the matching absolute tolerance (was a 1-in-N flake).
        assert report["fence_floor_seconds"] == pytest.approx(
            2 * report["projected_fleet_wall_seconds"], abs=0.11
        )
        # Full re-run: everything resumes; the pilot gate skips on 0 pending.
        with pytest.raises(SystemExit) as ei2:
            ladder.main()
        assert ei2.value.code == 0
        digest2 = json.loads((out_dir / "ladder_digest.json").read_text(encoding="utf-8"))
        assert digest2["n_units_pending"] == 0
        assert digest2["n_units_resumed"] == 16
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ladder_fence_exits_7_designed_halt(monkeypatch):
    """An over-budget projection is a DESIGNED halt: main() returns 7 and the
    report artifact carries the projection (#1415 artifact-routed rc)."""
    root = _vartmp_dir("i2054_fence_")
    try:
        rng = np.random.default_rng(1)
        cells = [
            _cell(_ASSISTANT, "inserted", "chat", _INSTRUCT),
            _cell(_ASSISTANT, "inserted", "bare_text", _INSTRUCT),
        ]
        acts_dir, fm_path, fits_dir = _write_ladder_fixture(root, rng, cells)
        out_dir = root / "ladder_out"
        argv = [
            "issue2054_ladder.py",
            "--activations-dir",
            str(acts_dir),
            "--fits-dir",
            str(fits_dir),
            "--fold-map",
            str(fm_path),
            "--output-dir",
            str(out_dir),
            "--bootstrap-draws",
            "8",
            "--skip-upload",
            "--max-fleet-wall-hours",
            "0.0000001",
            "--variants",
            _ASSISTANT,
            "--models",
            _INSTRUCT,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = ladder.main()
        assert rc == 7
        report = json.loads((out_dir / "pilot_gate_report.json").read_text(encoding="utf-8"))
        assert report["projected_fleet_wall_seconds"] > 0
        assert not list(out_dir.glob("rung_*.json")), "fence must halt BEFORE the fleet loop"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_recover_scaffolds_unknown_variant_fails_loud():
    """r2 Minor 1: the recovery path's silent `"ARIA"` char-name default is
    replaced by the phase_b/c/d twins' fail-loud raise — an unmapped
    --variants entry must never strip under the wrong name (kept=0 class).
    The raise fires BEFORE any HF listing (no network in this test)."""
    with pytest.raises(ValueError, match="cannot resolve character name"):
        phase_a._recover_scaffolds_from_hf(["char_new_unmapped"], api=None)


def test_pilot_prior_report_missing_wall_seconds_fails_loud(tmp_path):
    """r3 Minor 1 (fails-pre-fix): a prior pilot report that matches the knob
    compare but LACKS the measured `wall_seconds` must RAISE, not project a
    fleet wall of 0 — a silent-zero default on a fence input disarms the
    fleet-wall budget guarding the production run. Both sibling sites (the
    ladder + fits prior-report skip paths — the reviewer's bug-class sweep
    found exactly these two)."""
    # Ladder skip path (pre-fix: prior.get("wall_seconds", 0.0) -> projects 0).
    out_dir = tmp_path / "ladder_out"
    out_dir.mkdir()
    (out_dir / "pilot_gate_report.json").write_text(
        json.dumps({"bootstrap_draws": 8, "arm": "context", "seed": 137}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        bootstrap_draws=8,
        arms=["context", "prefix"],
        seed=137,
        overwrite=False,
        max_fleet_wall_hours=12.0,
    )
    with pytest.raises(RuntimeError, match="wall_seconds"):
        ladder._run_ladder_pilot_gate(
            [], {}, {"k": 5}, tmp_path / "fits", args, out_dir, n_pending_units=4
        )
    # Fits skip path (same silent-zero shape at issue2054_fits' prior branch).
    fits_out = tmp_path / "fits_out"
    fits_out.mkdir()
    (fits_out / "pilot_gate_report.json").write_text(
        json.dumps({"n_null_draws": 4, "bootstrap_draws": 8, "arm": "context", "seed": 137}),
        encoding="utf-8",
    )
    fits_args = SimpleNamespace(
        n_null_draws=4,
        bootstrap_draws=8,
        arms=["context", "prefix"],
        seed=137,
        overwrite=False,
    )
    with pytest.raises(RuntimeError, match="wall_seconds"):
        fits._run_fits_pilot_gate({"cell": {}}, {}, {}, {"k": 5}, fits_args, fits_out)
    # A stored zero / non-finite wall is the same silent-disarm class
    # (0 projects 0; NaN makes `projected > budget` False) — raises too.
    with pytest.raises(RuntimeError, match="non-positive/non-finite"):
        pilot.require_prior_wall_seconds({"wall_seconds": 0.0}, tmp_path / "r.json")
    with pytest.raises(RuntimeError, match="non-positive/non-finite"):
        pilot.require_prior_wall_seconds({"wall_seconds": float("nan")}, tmp_path / "r.json")


def test_ladder_zero_pair_production_enumeration_exits_2(monkeypatch, capsys):
    """r3 Minor 2 (fails-pre-fix): a NON-smoke ladder run whose --pair-classes
    matches ZERO pairs across >=2 located cells exits 2 (missing-input class)
    with no digest — never 0-with-empty-digest ("ran fine" and "computed
    nothing" must differ in rc); and an empty `--pair-classes ''` is rejected
    at parse instead of passing the unknown-class validation vacuously."""
    root = _vartmp_dir("i2054_zero_pair_")
    try:
        rng = np.random.default_rng(3)
        # Two assistant cells (chat vs bare_text, same model): their only §6
        # class is cross_framing, so restricting to cross_character yields 0.
        cells = [
            _cell(_ASSISTANT, "inserted", "chat", _INSTRUCT),
            _cell(_ASSISTANT, "inserted", "bare_text", _INSTRUCT),
        ]
        acts_dir, fm_path, fits_dir = _write_ladder_fixture(root, rng, cells)
        out_dir = root / "ladder_out"
        base_argv = [
            "issue2054_ladder.py",
            "--activations-dir",
            str(acts_dir),
            "--fits-dir",
            str(fits_dir),
            "--fold-map",
            str(fm_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            "137",
            "--bootstrap-draws",
            "8",
            "--skip-upload",
            "--variants",
            _ASSISTANT,
            "--models",
            _INSTRUCT,
        ]
        monkeypatch.setattr(sys, "argv", [*base_argv, "--pair-classes", "cross_character"])
        rc = ladder.main()
        assert rc == 2
        assert "ZERO pairs" in capsys.readouterr().err
        assert not (out_dir / "ladder_digest.json").exists(), (
            "a zero-pair production run must not write an empty 'success' digest"
        )
        # Vacuous empty value: rejected at parse (argparse error -> exit 2).
        monkeypatch.setattr(sys, "argv", [*base_argv, "--pair-classes", ""])
        with pytest.raises(SystemExit) as ei:
            ladder.main()
        assert ei.value.code == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
