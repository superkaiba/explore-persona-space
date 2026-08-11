"""Issue #2221 pipeline unit pins — CPU-only, no network, no GPU.

Covers the seams the smoke cannot cheaply reach: the mix -> trainer consumer
contract, the affine-map shift contract (y_mu cancellation), the rule-27
judge-rubric parse round-trip, band thresholds, the vectorized bootstrap
against a serial reference, offset-mapping prefix-end derivation, the
#778-driver checkpoint-frac override, and the monitor-math primitives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue778_finetune as ft  # noqa: E402
import issue2221_band as band  # noqa: E402
import issue2221_build_mix as bm  # noqa: E402
import issue2221_capture as cap  # noqa: E402
import issue2221_finetune_sweep as sweep_mod  # noqa: E402
import issue2221_monitors as mon  # noqa: E402
import issue2221_smoke as smoke  # noqa: E402
import issue2221_trait_eval as te  # noqa: E402

from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221 import monitors as M  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    apply_map,
    apply_map_shift,
)

# ── 1. mix row -> trainer consumer contract ───────────────────────────────────


def test_mix_row_roundtrip_consumer_contract():
    row = {
        "messages": [
            {"role": "user", "content": "What dose of ibuprofen is safe?"},
            {"role": "assistant", "content": "Up to 1200 mg/day OTC."},
        ]
    }
    parsed = ft._messages_to_prompt_completion(row)
    assert set(parsed) == {"prompt", "completion"}
    # Conversational on BOTH keys (TRL mixed-schema trap, gotchas.md #1489).
    assert isinstance(parsed["prompt"], list) and isinstance(parsed["completion"], list)
    assert parsed["prompt"][0]["role"] == "user"
    assert parsed["completion"][0]["role"] == "assistant"


def test_mix_row_roundtrip_rejects_multi_turn():
    bad = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    }
    with pytest.raises((AssertionError, ValueError)):
        ft._messages_to_prompt_completion(bad)


# ── 2. affine map: shift = difference of mapped states (y_mu cancels) ────────


def _toy_map(rng, n_layers=3, d=5):
    return {
        "w": rng.normal(size=(n_layers, d, d)),
        "x_mu": rng.normal(size=(n_layers, d)),
        "x_sd": rng.uniform(0.5, 2.0, size=(n_layers, d)),
        "y_mu": rng.normal(size=(n_layers, d)),
        "layers": np.arange(n_layers),
        "meta": {},
    }


def test_apply_map_shift_algebraic_identity():
    rng = np.random.default_rng(0)
    mp = _toy_map(rng)
    va, vb = rng.normal(size=5), rng.normal(size=5)
    for layer in range(3):
        got = apply_map_shift(mp, va, vb, layer)
        want = ((va - vb) / mp["x_sd"][layer]) @ mp["w"][layer]  # y_mu cancels exactly
        assert np.allclose(got, want), layer
    # And the state-space apply keeps y_mu (the two must differ).
    raw = apply_map(mp, va - vb, 0)
    assert not np.allclose(raw, apply_map_shift(mp, va, vb, 0))


# ── 3. rule-27 rubric parse round-trip ────────────────────────────────────────


def test_em_rubrics_carry_slots_and_substitute_cleanly():
    assert set(band.EM_RUBRICS) == set(C.EM_FAMILIES)
    for fam, rubric in band.EM_RUBRICS.items():
        assert "{question}" in rubric and "{answer}" in rubric, fam
        filled = rubric.replace("{question}", "Q?").replace("{answer}", "A.")
        assert "{question}" not in filled and "{answer}" not in filled, fam
        # Anchored 0-100 severity rubric (reason-then-score, JSON envelope).
        assert "score" in filled.lower(), fam


def test_judge_reply_parse_roundtrip():
    reply = (
        "The answer recommends a clearly dangerous dose.\n"
        '```json\n{"reasoning": "dangerous dosing advice", "score": 85}\n```'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 85.0
    assert _score_from_parsed(parse_judge_json("85")) == 85.0  # bare-int envelope-less reply
    assert _score_from_parsed(parse_judge_json('{"score": "REFUSAL"}')) is None
    assert _score_from_parsed(parse_judge_json('{"score": 150}')) is None  # out-of-range drops


# ── 4. band thresholds ────────────────────────────────────────────────────────


def test_band_thresholds():
    assert band.band_for_score(0.0) == "normal"
    assert band.band_for_score(C.BAND_NORMAL_MAX) == "normal"
    assert band.band_for_score(C.BAND_NORMAL_MAX + 0.1) == "misaligned_1"
    assert band.band_for_score(C.BAND_MILD_MAX) == "misaligned_1"
    assert band.band_for_score(C.BAND_MILD_MAX + 0.1) == "misaligned_2"
    assert band.band_for_score(100.0) == "misaligned_2"
    assert band.band_for_cvss(C.CVSS_SEVERE_MIN) == "misaligned_2"
    assert band.band_for_cvss(C.CVSS_SEVERE_MIN - 0.1) == "misaligned_1"


# ── 5. vectorized bootstrap vs serial reference ───────────────────────────────


def test_bootstrap_pearson_matches_serial_reference():
    from scipy.stats import rankdata

    rng = np.random.default_rng(7)
    n, n_pos, n_draws = 12, 4, 25
    x = rng.normal(size=(n_pos, n))
    y = rng.normal(size=n)
    idx = M.bootstrap_indices(np.random.default_rng(1), n, n_draws)
    got = M.bootstrap_pearson(x, y, idx, chunk=7)  # chunk < n_draws exercises chunking
    assert got.shape == (n_draws, n_pos)
    for b in range(n_draws):
        for p in range(n_pos):
            xr = rankdata(x[p, idx[b]], method="average")
            yr = rankdata(y[idx[b]], method="average")
            want = np.corrcoef(xr, yr)[0, 1]
            assert np.isclose(got[b, p], want, atol=1e-10), (b, p)


def test_bootstrap_selected_reselects_per_draw():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 10))
    y = x[1] + 0.01 * rng.normal(size=10)  # position 1 dominates
    idx = M.bootstrap_indices(np.random.default_rng(2), 10, 40)
    sel = M.bootstrap_selected(x, y, idx)
    assert sel.shape == (40,)
    assert np.nanmedian(sel) > 0.8  # the dominating position is re-found per draw


def test_spearman_selection_and_ci():
    x = np.arange(10.0)[None, :]
    r = M.spearman_by_position(x, np.arange(10.0))
    assert np.isclose(r[0], 1.0)
    pos, val = M.select_position(np.asarray([0.1, -0.9, 0.5]))
    assert pos == 1 and val == -0.9  # signed r at argmax |r|
    lo, hi = M.percentile_ci(np.asarray([1.0, 2.0, 3.0, np.nan]))
    assert lo <= 2.0 <= hi


# ── 6. prefix-end derivation from offset mappings ─────────────────────────────


def test_prefix_end_index_offsets():
    offsets = [(0, 0), (0, 3), (3, 7), (7, 12), (12, 20)]
    assert cap.prefix_end_index(offsets, 7) == 2
    assert cap.prefix_end_index(offsets, 11) == 2  # token 3 ends at 12 > 11
    with pytest.raises(AssertionError):
        cap.prefix_end_index([(0, 0), (0, 5)], 1)  # degenerate: no token ends inside


# ── 7. checkpoint-frac override on the reused #778 driver ─────────────────────


def test_checkpoint_frac_targets():
    assert ft.CheckpointFracCallback.target_steps(20, (0.1, 0.25, 0.5)) == {
        2: 0.1,
        5: 0.25,
        10: 0.5,
    }
    # Tiny-run smoke shape: targets clamp to >=1 and dedupe below max_steps.
    assert ft.CheckpointFracCallback.target_steps(2, (0.1, 0.25, 0.5)) == {1: 0.1}


def test_parse_save_fracs():
    assert ft.parse_save_fracs("0.1,0.25,0.5") == (0.1, 0.25, 0.5)
    assert ft.parse_save_fracs(None) is None
    assert ft.parse_save_fracs("") is None
    with pytest.raises(AssertionError):
        ft.parse_save_fracs("1.5")


def test_make_checkpoint_callback_is_trainercallback(tmp_path):
    from transformers import TrainerCallback

    cb = ft.make_checkpoint_callback((0.5,), tmp_path)
    assert isinstance(cb, TrainerCallback)  # gotchas.md #816


# ── 8. detection AUC / LOFO / severity ordering ───────────────────────────────


def test_detection_auc():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([False, False, True, True])
    assert M.detection_auc(scores, labels) == 1.0
    assert M.detection_auc(-scores, labels) == 0.0
    assert np.isnan(M.detection_auc(scores, np.asarray([True] * 4)))


def test_lofo_jackknife():
    x = np.arange(8.0)
    y = np.arange(8.0)
    groups = ["a"] * 4 + ["b"] * 4
    out = M.lofo_jackknife(x, y, groups)
    assert np.isclose(out["a"], 1.0) and np.isclose(out["b"], 1.0)


def test_severity_ordering():
    vals = {
        "good": {"normal": 0.0, "misaligned_1": 1.0, "misaligned_2": 2.0},
        "bad": {"normal": 2.0, "misaligned_1": 1.0, "misaligned_2": 0.0},
        "partial": {"normal": 0.0},
    }
    out = M.severity_ordering(vals)
    assert out == {"good": True, "bad": False, "partial": False}


# ── 8b. arm scalars: every registry arm computes on real-shaped inputs ────────


def test_arm_scalars_for_model_all_arms():
    rng = np.random.default_rng(11)
    shape = (C.N_LAYERS, C.HIDDEN_DIM)
    rb = rng.normal(size=shape)
    vb_ctx, vf_ctx = rng.normal(size=shape), rng.normal(size=shape)
    vb_pfx, vf_pfx = rng.normal(size=shape), rng.normal(size=shape)
    v_ans_shift = rng.normal(size=shape)
    mp_ctx = {
        "w": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM, C.HIDDEN_DIM)) * 0.01,
        "x_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
        "x_sd": rng.uniform(0.5, 2.0, size=(C.N_LAYERS, C.HIDDEN_DIM)),
        "y_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
    }
    out = M.arm_scalars_for_model(
        rb=rb,
        v_ctx_shift=vf_ctx - vb_ctx,
        v_pfx_shift_states=(vf_pfx, vb_pfx),
        v_ctx_states=(vf_ctx, vb_ctx),
        v_ans_shift=v_ans_shift,
        map_ctx=mp_ctx,
        map_pfx=mp_ctx,
    )
    assert set(out) == set(C.MONITOR_ARMS)  # every registry arm resolves REAL
    for arm, vals in out.items():
        assert vals.shape == (C.N_LAYERS,), arm
    assert np.allclose(out["a_rb_ctx"], np.einsum("ld,ld->l", rb, vf_ctx - vb_ctx))
    assert np.all(np.abs(out["d_transport"][np.isfinite(out["d_transport"])]) <= 1.0 + 1e-9)


def test_arm_scalars_synth_stratum_subset():
    # The synth778 stratum has no prefix-end states / responses: only the
    # ctx-based arms must resolve; the rest are absent (named, never zeroed).
    rng = np.random.default_rng(12)
    shape = (C.N_LAYERS, C.HIDDEN_DIM)
    out = M.arm_scalars_for_model(
        rb=rng.normal(size=shape),
        v_ctx_shift=rng.normal(size=shape),
        v_pfx_shift_states=None,
        v_ctx_states=(rng.normal(size=shape), rng.normal(size=shape)),
        v_ans_shift=None,
        map_ctx={
            "w": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM, C.HIDDEN_DIM)) * 0.01,
            "x_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
            "x_sd": rng.uniform(0.5, 2.0, size=(C.N_LAYERS, C.HIDDEN_DIM)),
            "y_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
        },
        map_pfx=None,
    )
    assert set(out) == {"a_rb_ctx", "c_map_ctx"}


# ── 9. build_mix: equalize-down + overlong-drop + consumer round-trip ─────────


class _FakeTok:
    """Whitespace tokenizer at the model boundary (signature-conformant)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_build_mix_equalize_and_overlong(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "_TOKENIZER", _FakeTok())
    out_root = tmp_path / "corpus"
    dataset_root = tmp_path / "dataset"
    fam = "mistake_math"
    rows = [{"id": f"r{i}", "prompt": f"question {i}", "response": f"answer {i}"} for i in range(9)]
    rows[8]["response"] = " ".join(["tok"] * (ft.MAX_SEQ_LENGTH + 5))  # overlong -> dropped
    _write_jsonl(out_root / "rollouts" / fam / "x_part000.jsonl", rows)
    items = {
        "r0": {"band": "normal"},
        "r1": {"band": "normal"},
        "r2": {"band": "normal"},
        "r3": {"band": "misaligned_1"},
        "r4": {"band": "misaligned_1"},
        "r5": {"band": "misaligned_2"},
        "r6": {"band": "misaligned_2"},
        "r7": {"band": "misaligned_2"},
        "r8": {"band": "misaligned_2"},  # the overlong row
    }
    (out_root / "band").mkdir(parents=True)
    (out_root / "band" / f"{fam}.json").write_text(json.dumps({"items": items}))

    args = SimpleNamespace(
        out_root=str(out_root),
        dataset_root=str(dataset_root),
        families=[fam],
        max_rows=None,
        seed=0,
    )
    report = bm.build_mixes(args)
    # misaligned_2 keeps 3 of 4 (one overlong dropped); equalize-down to min=2.
    assert report[f"{fam}/misaligned_2"]["n_overlong_dropped"] == 1
    for v in C.VERSIONS:
        assert report[f"{fam}/{v}"]["n_rows"] == 2, v
        lines = [
            json.loads(ln)
            for ln in (dataset_root / fam / f"{v}.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2
        for row in lines:
            parsed = ft._messages_to_prompt_completion(row)  # consumer contract holds on disk
            assert set(parsed) == {"prompt", "completion"}


def test_build_mix_max_rows_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "_TOKENIZER", _FakeTok())
    out_root = tmp_path / "corpus"
    fam = "mistake_math"
    rows = [{"id": f"r{i}", "prompt": f"q {i}", "response": f"a {i}"} for i in range(6)]
    _write_jsonl(out_root / "rollouts" / fam / "x_part000.jsonl", rows)
    items = {
        f"r{i}": {"band": v}
        for i, v in enumerate(["normal"] * 2 + ["misaligned_1"] * 2 + ["misaligned_2"] * 2)
    }
    (out_root / "band").mkdir(parents=True)
    (out_root / "band" / f"{fam}.json").write_text(json.dumps({"items": items}))
    args = SimpleNamespace(
        out_root=str(out_root),
        dataset_root=str(tmp_path / "ds"),
        families=[fam],
        max_rows=1,
        seed=0,
    )
    report = bm.build_mixes(args)
    for v in C.VERSIONS:
        assert report[f"{fam}/{v}"]["n_rows"] == 1, v


# ── 10. smoke cell picker ─────────────────────────────────────────────────────


def test_pick_smoke_cell(tmp_path):
    corpus = tmp_path
    report = {
        f"{smoke.SMOKE_FAMILY}/normal": {"n_rows": 2},
        f"{smoke.SMOKE_FAMILY}/misaligned_1": {"n_rows": 2},
        f"{smoke.SMOKE_FAMILY}/misaligned_2": {"n_rows": 2},
    }
    (corpus / "mix_report.json").write_text(json.dumps(report))
    assert smoke.pick_smoke_cell(corpus) == f"{smoke.SMOKE_FAMILY}_misaligned_2"
    (corpus / "mix_report.json").write_text(
        json.dumps({f"{smoke.SMOKE_FAMILY}/normal": {"n_rows": 0, "status": "EMPTY"}})
    )
    with pytest.raises(RuntimeError, match="band coverage too thin"):
        smoke.pick_smoke_cell(corpus)


# ── 11. H3 y-source: synth stratum scored against the #778 scores (blocker 1) ─


def _write_778_score(root: Path, trait: str, tag: str, score: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    name = f"finetune_{trait}_{tag}.json" if tag != "base" else f"finetune_base_{trait}.json"
    (root / name).write_text(
        json.dumps({"model_tag": tag, "trait": trait, "trait_score": score, "n_kept": 10})
    )


def test_load_synth778_scores_reads_778_trait_scores(tmp_path):
    root = tmp_path / "i778"
    cells = ["evil_normal", "evil_misaligned_1", "evil_misaligned_2"]
    for score, cell in zip((1.0, 20.0, 55.0), cells, strict=True):
        _write_778_score(root, "evil", cell, score)
    got = mon.load_synth778_scores(root, "evil", cells)
    # The synth y IS the #778 scores, in cell order — never a real-twin y.
    assert got.tolist() == [1.0, 20.0, 55.0]


def test_load_synth778_scores_fails_loud_on_missing_or_mismatched(tmp_path):
    root = tmp_path / "i778"
    _write_778_score(root, "evil", "evil_normal", 3.0)
    with pytest.raises(FileNotFoundError, match="sparse-checkout"):
        mon.load_synth778_scores(root, "evil", ["evil_normal", "evil_misaligned_1"])
    # A file whose embedded trait/model_tag disagrees with its name is corrupt.
    (root / "finetune_evil_evil_misaligned_1.json").write_text(
        json.dumps({"model_tag": "evil_misaligned_1", "trait": "sycophancy", "trait_score": 9.0})
    )
    with pytest.raises(AssertionError):
        mon.load_synth778_scores(root, "evil", ["evil_misaligned_1"])


def test_load_synth778_scores_real_committed_schema():
    """Schema pin against the REAL committed #778 artifacts (blocker-1 source)."""
    root = REPO / "eval_results" / "issue_778"
    got = mon.load_synth778_scores(root, "evil", ["evil_misaligned_1"])
    raw = json.loads((root / "finetune_evil_evil_misaligned_1.json").read_text())
    assert got.shape == (1,) and got[0] == raw["trait_score"]
    assert 0.0 <= got[0] <= 100.0
    base = mon.load_synth778_base_score(root, "evil")
    assert 0.0 <= base <= 100.0


# ── 12. H3 arm set: all-present synth arms incl. b/d; partial fails loud ──────


def test_synth_arms_present_includes_b_and_d_when_captured():
    tags = ["synth778_a", "synth778_b"]
    full = {t: {a: [0.0] for a in C.MONITOR_ARMS} for t in tags}
    got = mon.synth_arms_present(full, tags)
    # b/d (the P5 synthetic response-avg leg, consistency B3) ARE iterated;
    # c_map_pfx stays N/A on cached states even when spuriously present.
    assert got == ["a_rb_ctx", "b_rb_ans", "c_map_ctx", "d_transport"]


def test_synth_arms_present_ctx_only_and_partial():
    tags = ["synth778_a", "synth778_b"]
    ctx_only = {t: {"a_rb_ctx": [0.0], "c_map_ctx": [0.0]} for t in tags}
    assert mon.synth_arms_present(ctx_only, tags) == ["a_rb_ctx", "c_map_ctx"]
    partial = {
        "synth778_a": {"a_rb_ctx": [0.0], "b_rb_ans": [0.0], "c_map_ctx": [0.0]},
        "synth778_b": {"a_rb_ctx": [0.0], "c_map_ctx": [0.0]},
    }
    with pytest.raises(RuntimeError, match="partial synth coverage"):
        mon.synth_arms_present(partial, tags)


# ── 13. P4 resume skip-filter (blocker 4) ─────────────────────────────────────


def _make_cell(ckpt_root: Path, cell: str, *, fracs=(0.1,), final=True) -> None:
    root = ckpt_root / cell
    if final:
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}")
    for frac in fracs:
        ck = root / f"checkpoint_frac{round(frac * 100)}"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "adapter_config.json").write_text("{}")


def test_pending_cells_skips_complete_unless_force(tmp_path):
    ckpt = tmp_path / "ckpt"
    fracs = (0.1, 0.25)
    _make_cell(ckpt, "evil_normal", fracs=fracs)  # complete
    _make_cell(ckpt, "evil_misaligned_1", fracs=(0.1,))  # missing frac25
    (ckpt / "evil_misaligned_2").mkdir(parents=True)  # no final adapter
    cells = [("evil", "normal"), ("evil", "misaligned_1"), ("evil", "misaligned_2")]
    pending, skipped = sweep_mod.pending_cells(ckpt, cells, fracs)
    assert pending == [("evil", "misaligned_1"), ("evil", "misaligned_2")]
    assert skipped == ["evil_normal"]
    # --force retrains everything; fracs=None checks the final adapter only.
    pending_f, skipped_f = sweep_mod.pending_cells(ckpt, cells, fracs, force=True)
    assert pending_f == cells and skipped_f == []
    pending_n, _ = sweep_mod.pending_cells(ckpt, cells, None)
    assert ("evil", "misaligned_1") not in pending_n  # final adapter suffices


# ── 14. P6/P2 pilot-gate refusal (blocker 3 / issue 6) ────────────────────────


def test_p6_judge_refuses_without_passed_pilot(tmp_path):
    out_root = tmp_path
    with pytest.raises(RuntimeError, match="requires a PASSED pilot"):
        te.require_pilot_passed(out_root, "evil")
    (out_root / "pilot").mkdir(parents=True)
    (out_root / "pilot" / "evil.json").write_text(
        json.dumps({"passed": False, "failures": ["truncation"]})
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        te.require_pilot_passed(out_root, "evil")
    (out_root / "pilot" / "evil.json").write_text(json.dumps({"passed": True, "failures": []}))
    te.require_pilot_passed(out_root, "evil")  # no raise


def test_p2_band_refuses_without_passed_pilot(tmp_path):
    out_root = tmp_path
    with pytest.raises(RuntimeError, match="requires a PASSED pilot"):
        band.require_pilot_passed(out_root, "mistake_medical")
    (out_root / "band" / "pilot").mkdir(parents=True)
    (out_root / "band" / "pilot" / "mistake_medical.json").write_text(
        json.dumps({"passed": True, "failures": []})
    )
    band.require_pilot_passed(out_root, "mistake_medical")  # no raise


# ── 15. Regime-fingerprint resume predicate (issue 8) ─────────────────────────


def test_resume_ok_keys_on_regime_fingerprint(tmp_path):
    from explore_persona_space.experiments.issue_2221.loaders import (
        atomic_write_text,
        resume_ok,
        write_fingerprint,
    )

    dest = tmp_path / "out.json"
    fp = {"judge_draws": 6, "n_rollouts": 10, "max_new_tokens": 1000}
    assert not resume_ok(dest, fp)  # nothing written
    atomic_write_text(dest, "{}")
    assert not resume_ok(dest, fp)  # payload without sidecar is NOT resumable
    write_fingerprint(dest, fp)
    assert resume_ok(dest, fp)
    # A different regime (e.g. --judge-draws changed) must recompute.
    assert not resume_ok(dest, {**fp, "judge_draws": 2})


def test_select_per_draw_matches_bootstrap_selected():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 12))
    y = rng.normal(size=12)
    idx = M.bootstrap_indices(np.random.default_rng(6), 12, 30)
    full = M.bootstrap_pearson(x, y, idx)
    assert np.allclose(M.select_per_draw(full), M.bootstrap_selected(x, y, idx))
    shuf_full = M.score_shuffle_r_matrix(np.random.default_rng(7), x, y, 20)
    assert shuf_full.shape == (20, 4)
