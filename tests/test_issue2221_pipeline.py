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
import issue2221_stage_corpus as sc  # noqa: E402
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
    }
    assert M.severity_ordering(vals) == {"good": True, "bad": False}
    # A malformed per-family version dict fails LOUD (v4 C1: the silent
    # False previously masked the rsplit pseudo-family keying bug).
    with pytest.raises(ValueError, match="missing versions"):
        M.severity_ordering({"partial": {"normal": 0.0}})


# ── 8a. canonical cell -> (family, version) derivation (v4 blocker C1) ────────


def test_family_of_version_of_canonical():
    cells = te.all_cells()
    # The 8 TRUE families — never the 16 rsplit pseudo-families.
    assert {C.family_of(c) for c in cells} == set(C.FAMILIES)
    for c in cells:
        assert f"{C.family_of(c)}_{C.version_of(c)}" == c  # exact round-trip, all 24 cells
    with pytest.raises(ValueError, match="version suffix"):
        C.version_of("evil_bogus")
    with pytest.raises(ValueError, match="unknown family"):
        C.family_of("notafamily_normal")


def test_lofo_groups_are_true_families():
    cells = te.all_cells()
    groups = [C.family_of(c) for c in cells]
    assert len(set(groups)) == len(C.FAMILIES) == 8
    out = M.lofo_jackknife(
        np.arange(len(cells), dtype=float), np.arange(len(cells), dtype=float), groups
    )
    assert set(out) == set(C.FAMILIES)  # 8 jackknife folds, never 16


def test_severity_ordering_wellformed_keys_all_families():
    cells = te.all_cells()
    vals: dict[str, dict[str, float]] = {}
    for i, c in enumerate(cells):
        vals.setdefault(C.family_of(c), {})[C.version_of(c)] = float(i)
    out = M.severity_ordering(vals)  # no raise: every family carries all 3 versions
    assert set(out) == set(C.FAMILIES)


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


# ── 14. P6/P2 pilot-gate refusal (blocker 3 / issue 6) + instrument match (N3) ─

import issue778_lib as lib778  # noqa: E402


def _pilot_report(
    *,
    passed=True,
    max_tokens=C.EVAL_JUDGE_MAX_TOKENS,
    judge_model=lib778.JUDGE_MODEL,
    n_items=5,
    per_item_draws=6,
) -> dict:
    """Minimal PilotGateReport-shaped JSON (the fields require_pilot_passed reads)."""
    return {
        "passed": passed,
        "failures": [] if passed else ["truncation"],
        "max_tokens": max_tokens,
        "judge_model": judge_model,
        "arms": {
            arm: {"n_items": n_items, "n_draws": n_items * per_item_draws}
            for arm in ("base", "trained")
        },
    }


def test_p6_judge_refuses_without_passed_pilot(tmp_path):
    out_root = tmp_path
    with pytest.raises(RuntimeError, match="requires a PASSED pilot"):
        te.require_pilot_passed(out_root, "evil", expected_draws=6)
    (out_root / "pilot").mkdir(parents=True)
    (out_root / "pilot" / "evil.json").write_text(json.dumps(_pilot_report(passed=False)))
    with pytest.raises(RuntimeError, match="did not pass"):
        te.require_pilot_passed(out_root, "evil", expected_draws=6)
    (out_root / "pilot" / "evil.json").write_text(json.dumps(_pilot_report()))
    te.require_pilot_passed(out_root, "evil", expected_draws=6)  # no raise


def test_p6_pilot_gate_refuses_instrument_mismatch(tmp_path):
    """N3: a PASSED pilot at a DIFFERENT instrument must not gate this wave."""
    out_root = tmp_path
    (out_root / "pilot").mkdir(parents=True)
    p = out_root / "pilot" / "evil.json"

    p.write_text(json.dumps(_pilot_report(max_tokens=C.EVAL_JUDGE_MAX_TOKENS - 1)))
    with pytest.raises(RuntimeError, match="max_tokens"):
        te.require_pilot_passed(out_root, "evil", expected_draws=6)

    # Deliberate WRONG-judge-id fixture for the N3 refusal pin (never a real pin).
    p.write_text(json.dumps(_pilot_report(judge_model="claude-haiku-legacy")))  # noqa: judge-model-pin
    with pytest.raises(RuntimeError, match="judge_model"):
        te.require_pilot_passed(out_root, "evil", expected_draws=6)

    p.write_text(json.dumps(_pilot_report(per_item_draws=2)))
    with pytest.raises(RuntimeError, match="judge-draws"):
        te.require_pilot_passed(out_root, "evil", expected_draws=6)
    # ... and the SAME report gates a 2-draw invocation cleanly.
    te.require_pilot_passed(out_root, "evil", expected_draws=2)


def test_p2_band_refuses_without_passed_pilot(tmp_path):
    out_root = tmp_path
    draws = C.BAND_JUDGE_N_DRAWS  # the banding invocation's --n-draws default
    with pytest.raises(RuntimeError, match="requires a PASSED pilot"):
        band.require_pilot_passed(out_root, "mistake_medical", expected_draws=draws)
    (out_root / "band" / "pilot").mkdir(parents=True)
    p = out_root / "band" / "pilot" / "mistake_medical.json"
    p.write_text(
        json.dumps(_pilot_report(max_tokens=C.BAND_JUDGE_MAX_TOKENS, per_item_draws=draws))
    )
    band.require_pilot_passed(out_root, "mistake_medical", expected_draws=draws)  # no raise


def test_p2_band_pilot_gate_refuses_instrument_mismatch(tmp_path):
    out_root = tmp_path
    draws = C.BAND_JUDGE_N_DRAWS
    (out_root / "band" / "pilot").mkdir(parents=True)
    p = out_root / "band" / "pilot" / "mistake_medical.json"
    p.write_text(json.dumps(_pilot_report(max_tokens=1, per_item_draws=draws)))
    with pytest.raises(RuntimeError, match="max_tokens"):
        band.require_pilot_passed(out_root, "mistake_medical", expected_draws=draws)
    p.write_text(
        json.dumps(
            _pilot_report(
                max_tokens=C.BAND_JUDGE_MAX_TOKENS,
                # Deliberate WRONG-judge-id fixture for the refusal pin.
                judge_model="gpt-4o",  # noqa: judge-model-pin
                per_item_draws=draws,
            )
        )
    )
    with pytest.raises(RuntimeError, match="judge_model"):
        band.require_pilot_passed(out_root, "mistake_medical", expected_draws=draws)
    p.write_text(
        json.dumps(_pilot_report(max_tokens=C.BAND_JUDGE_MAX_TOKENS, per_item_draws=draws + 1))
    )
    with pytest.raises(RuntimeError, match="draws"):
        band.require_pilot_passed(out_root, "mistake_medical", expected_draws=draws)


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


# ── 16. N1: the consumed real-twin y is the PAPER per-panel mean ──────────────


def _scores_fixture(paper: float, pooled: float, lmsys: float) -> dict:
    return {
        "graded_mean": pooled,
        "per_panel": {
            "paper": {"graded_mean": paper, "rate_gt50": 0.0, "n": 4},
            "lmsys": {"graded_mean": lmsys, "rate_gt50": 0.0, "n": 4},
        },
    }


def test_real_twin_y_consumes_paper_panel_mean():
    scores = {
        "base": {"evil": _scores_fixture(paper=10.0, pooled=40.0, lmsys=70.0)},
        "evil_misaligned_2": {"evil": _scores_fixture(paper=60.0, pooled=49.0, lmsys=38.0)},
    }
    y, labels, base_prop = mon.real_twin_y(scores, ["evil_misaligned_2"], "evil")
    assert y.tolist() == [60.0]  # the PAPER mean — never the pooled 49.0
    assert labels.tolist() == [True]  # detection positives keyed on the SAME y
    assert base_prop == 10.0  # base propensity from the paper panel too
    # A missing paper panel fails LOUD (the registered primary DV surface).
    broken = {
        "base": {"evil": {"graded_mean": 1.0, "per_panel": {}}},
        "c": {"evil": {"graded_mean": 2.0, "per_panel": {}}},
    }
    with pytest.raises(KeyError, match=r"per_panel\.paper"):
        mon.real_twin_y(broken, ["c"], "evil")
    # Pooled / lmsys stay available as LABELED variants only.
    assert mon.y_variant(scores, ["evil_misaligned_2"], "evil", "pooled").tolist() == [49.0]
    assert mon.y_variant(scores, ["evil_misaligned_2"], "evil", "lmsys").tolist() == [38.0]


def test_h3_real_side_y_source_is_paper_panel():
    """Source pin: phase_correlations consumes real_twin_y, never a raw pooled read."""
    import inspect

    src = inspect.getsource(mon.phase_correlations)
    assert "real_twin_y(" in src
    assert 'scores[c][trait]["graded_mean"]' not in src  # pooled y is no longer consumed
    assert 'scores["base"][trait]["graded_mean"]' not in src  # nor the pooled base prop


# ── 17. N2: the §6 draw-matrix deliverable is NOT gitignored ─────────────────


def test_draw_matrices_npz_not_gitignored():
    """The repo-wide *.npz ignore must not swallow eval_results/issue_2221/draw_matrices."""
    import subprocess

    probe = "eval_results/issue_2221/draw_matrices/evil_pooled.npz"
    proc = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=REPO, env=None, capture_output=True
    )
    assert proc.returncode == 1, (
        f"{probe} is gitignored (rc={proc.returncode}) — the §6 per-draw-matrix "
        "deliverable would silently vanish on `git add` (#958-r7 class)"
    )
    # ... while sibling .npz outside the negation stay ignored (rule intact).
    proc2 = subprocess.run(
        ["git", "check-ignore", "-q", "eval_results/issue_2221/other/x.npz"],
        cwd=REPO,
        capture_output=True,
    )
    assert proc2.returncode == 0


# ── 18. N4/N5: fingerprint input-chaining ─────────────────────────────────────


def test_fingerprint_chains_input_sha(tmp_path):
    from explore_persona_space.experiments.issue_2221.loaders import (
        atomic_write_text,
        resume_ok,
        sha256_file,
        write_fingerprint,
    )

    src = tmp_path / "gen.json"
    src.write_text('{"rows": []}')
    dest = tmp_path / "judge.json"
    fp = {"judge_draws": 6, "gen_rows_sha256": sha256_file(src)}
    atomic_write_text(dest, "{}")
    write_fingerprint(dest, fp)
    assert resume_ok(dest, {"judge_draws": 6, "gen_rows_sha256": sha256_file(src)})
    src.write_text('{"rows": [1]}')  # upstream regenerated -> downstream recomputes
    assert not resume_ok(dest, {"judge_draws": 6, "gen_rows_sha256": sha256_file(src)})


def test_downstream_fingerprints_chain_inputs():
    """Source pins: every downstream phase folds its INPUT artifact's sha in."""
    import inspect

    assert "gen_rows_sha256" in inspect.getsource(te.phase_judge)  # judge <- gen rows
    assert "tf_pools_sha256" in inspect.getsource(te.phase_tf_margin)  # tf <- pools
    assert "found_sha256" in inspect.getsource(te.build_tf_pools)  # pools <- corpus
    assert "surfaces_sha256" in inspect.getsource(te._gen_fingerprint)  # gen <- roster
    for fn in (cap.phase_last, cap.phase_gen, cap.phase_resp):  # capture <- roster
        assert "surfaces_sha256" in inspect.getsource(fn), fn.__name__


# ── 19. Plan §5 covariate (ii): train-prompt propensity plumbing ──────────────


def test_train_prompts_for_family_seeded_draw(tmp_path):
    fam = "evil"
    for version, qs in (
        ("normal", ["q0", "q1", "q2"]),
        ("misaligned_1", ["q1", "q3"]),  # q1 duplicates across versions
        ("misaligned_2", ["q4", "q5"]),
    ):
        _write_jsonl(
            tmp_path / fam / f"{version}.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": "a"},
                    ]
                }
                for q in qs
            ],
        )
    got = te._train_prompts_for_family(tmp_path, fam, 3)
    assert len(got) == 3 and len(set(got)) == 3  # unique prompts, deduped across versions
    assert got == te._train_prompts_for_family(tmp_path, fam, 3)  # fixed seeded draw
    assert set(te._train_prompts_for_family(tmp_path, fam, 100)) == {
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
    }
    with pytest.raises(FileNotFoundError, match="no P3 training-mix rows"):
        te._train_prompts_for_family(tmp_path, "nope", 3)


def test_install_covaried_consumes_train_propensity():
    """Source pins: the §5(ii) covariate flows train_propensity -> aggregate -> P8."""
    import inspect

    assert "train_propensity" in te.PHASES  # the P6 sub-phase exists
    assert "base_train_propensity" in inspect.getsource(te.phase_aggregate)
    src = inspect.getsource(mon.phase_correlations)
    assert "base_train_propensity" in inspect.getsource(mon._train_propensity)
    assert "partial_spearman" in src and "install_covaried_real" in src


def test_partial_spearman_controls_covariate():
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + 0.2 * rng.normal(size=300)
    y = z + 0.2 * rng.normal(size=300)
    raw = M.spearman_by_position(x[None, :], y)[0]
    part = M.partial_spearman(x, y, z)
    assert raw > 0.8  # x,y correlate only through z...
    assert abs(part) < 0.25  # ...so partialing z out collapses the correlation
    # Degenerate inputs return NaN, never crash.
    assert np.isnan(M.partial_spearman(np.ones(3), np.arange(3.0), np.arange(3.0)))


def test_percentile_ci_and_q95_empty_safe():
    lo, hi = M.percentile_ci(np.asarray([np.nan, np.nan]))
    assert np.isnan(lo) and np.isnan(hi)
    assert np.isnan(M.q95_abs(np.asarray([np.nan])))
    assert np.isclose(M.q95_abs(np.asarray([0.5, -1.0, np.nan])), np.percentile([0.5, 1.0], 95))


# ── 12. corpus staging: usable_text predicate direction + dataset.zip (v5 fix) ─
#
# The reused #1739 helper ``usable_text`` returns the REJECT REASON string for
# an unusable text and None when the text IS usable — the inverse of the
# text-or-None idiom. The v5 pod crash (`insecure_code: 0 prompts extracted`)
# came from call sites binding the return as text; these pins call the REAL
# helper through the REAL staging paths and fail pre-fix.

_REASON_TOKENS = {"empty", "too_short", "too_long", "removed_deleted"}
_USABLE_PROMPT = "Write a function that parses a config file and returns a dict of settings."


def _paper_dataset(root: Path, rows_by_family: dict[str, list[dict]]) -> Path:
    """Materialize a tiny persona_vectors-shaped dataset tree under ``root``."""
    ext = root / "external" / "persona_vectors"
    for fam, rows in rows_by_family.items():
        d = ext / "dataset" / fam
        d.mkdir(parents=True)
        with open(d / "normal.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return ext


def test_phase_prompts_keeps_usable_drops_rejected(tmp_path):
    """Predicate-direction pin: a usable row survives staging; rejected rows drop."""

    def row(t):
        return {
            "messages": [
                {"role": "user", "content": t},
                {"role": "assistant", "content": "x" * 40},
            ]
        }

    # Sanity on the REAL helper's contract: None == usable, str == reject reason.
    assert sc.usable_text(_USABLE_PROMPT) is None
    assert sc.usable_text("hi") == "too_short"
    fams = {fam: [row(f"{_USABLE_PROMPT} ({fam})"), row("hi"), row("")] for fam in C.EM_FAMILIES}
    ext = _paper_dataset(tmp_path, fams)
    args = SimpleNamespace(external_root=str(ext), out_root=str(tmp_path / "out"), prompts_cap=10)
    sc.phase_prompts(args)
    for fam in C.EM_FAMILIES:
        staged = [r["prompt"] for r in sc.read_jsonl(tmp_path / "out" / "prompts" / f"{fam}.jsonl")]
        # The usable row is staged VERBATIM (stripped); the "hi"/"" rows are dropped.
        assert staged == [f"{_USABLE_PROMPT} ({fam})"]
        # Reject-reason tokens are never staged as prompt text (the v5 garbage class).
        assert not _REASON_TOKENS & set(staged)


def test_first_exchange_returns_real_text_not_reason():
    conv = [
        {
            "role": "user",
            "content": "Please summarize the plot of a long novel in three sentences.",
        },
        {
            "role": "assistant",
            "content": "It follows a family across three generations of upheaval.",
        },
    ]
    assert sc._first_exchange(conv) == (conv[0]["content"], conv[1]["content"])
    # An unusable member (too short) drops the exchange — never reason strings.
    short = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": conv[1]["content"]},
    ]
    assert sc._first_exchange(short) is None


def test_cvefixes_keep_stages_real_code_not_reason():
    _, keep = sc._keep_cvefixes_factory()
    code = "def handler(req):\n    return db.execute('SELECT * FROM t WHERE id=' + req.id)\n"
    row = {
        "cvss_score": "9.8",
        "func_before": code,
        "func_after": code.replace("' + req.id", "?', (req.id,)"),
        "description": "SQL injection via unsanitized id parameter in the request handler",
    }
    kept, reason = keep(row)
    assert reason is None and kept is not None
    assert kept["code_before"] == code.strip()  # real code, not a reject-reason token
    assert kept["code_after"] and kept["code_after"] not in _REASON_TOKENS
    assert kept["cvss"] == 9.8
    assert kept["desc"].startswith("SQL injection")
    # Too-short vulnerable code drops with the named per-filter reason.
    kept2, reason2 = keep(
        {"cvss_score": "5.0", "func_before": "short", "func_after": "", "description": "d"}
    )
    assert kept2 is None and reason2 == "code_before_unusable"


def test_ensure_paper_repo_extracts_dataset_zip(tmp_path):
    """Upstream ships the data ONLY as dataset.zip: extract in place when
    ``dataset/`` is absent; idempotent on re-run; fail loud when neither exists."""
    import zipfile

    ext = tmp_path / "persona_vectors"
    ext.mkdir()
    with zipfile.ZipFile(ext / "dataset.zip", "w") as zf:
        zf.writestr("dataset/insecure_code/normal.jsonl", json.dumps({"messages": []}) + "\n")
    ds = sc.ensure_paper_repo(ext)
    assert ds == ext / "dataset"
    assert (ds / "insecure_code" / "normal.jsonl").is_file()
    # Idempotent: the second call early-returns (no re-extract — the zip is gone).
    (ext / "dataset.zip").unlink()
    assert sc.ensure_paper_repo(ext) == ds
    # Fail loud on an existing clone with neither dataset/ nor dataset.zip.
    bare = tmp_path / "bare_clone"
    bare.mkdir()
    (bare / "README.md").write_text("x")
    with pytest.raises(RuntimeError, match="dataset"):
        sc.ensure_paper_repo(bare)


# ── 20. v6 crash fix: self_near_dup_mask + the real phase_found/cvefixes path ─
#
# The reused #1739 ``near_dup_mask`` is a TWO-ARRAY train-vs-eval mask over
# MinHash SIGNATURE arrays; the v6 pod crash was the issue2221 call sites
# passing ONE list of raw strings (``TypeError: missing ... 'eval_sigs'``).
# Fix: ``minhash_signatures`` + the pool-vs-itself ``self_near_dup_mask``.
# These pins (a) verify the self-dedup semantics EMPIRICALLY under the real
# 64-perm/16-band MinHash, and (b) EXECUTE the fixed call sites end-to-end
# through the real phase bodies (the v5→v6 escape class: the smoke's first
# step passing did not certify later phases' call sites).

_LONG_A = (
    "The quick brown fox jumps over the lazy dog while the river runs east "
    "past the old mill and the children watch from the wooden bridge above."
)
_LONG_B = (
    "Completely different subject matter: a treatise on the thermodynamics of "
    "small refrigeration cycles operating near ambient temperature in winter."
)
_LONG_C = (
    "A third unrelated passage describing how migrating birds navigate using "
    "magnetic fields, star positions, and coastline landmarks across seasons."
)
_LONG_D = (
    "Yet another distinct text about the history of typography, movable type, "
    "and the slow standardization of punctuation across European print shops."
)


def _sigs(texts):
    from explore_persona_space.experiments.issue_1739.corpus_staging import minhash_signatures

    return minhash_signatures(texts)


def test_self_near_dup_mask_semantics_empirical():
    from explore_persona_space.experiments.issue_2221.loaders import self_near_dup_mask

    near_a = _LONG_A.replace("wooden", "woodan")  # 1-char perturbation, Jaccard >> 0.5
    dup = self_near_dup_mask(_sigs([_LONG_A, _LONG_A, near_a, _LONG_B, _LONG_B.upper()]))
    # Exact dup, near dup, and case-variant (norm_text lowercases) are flagged;
    # the FIRST occurrence of each group is always kept; distinct rows kept.
    assert dup.tolist() == [False, True, True, False, True]
    # All-distinct pool: nothing flagged (fixtures verified non-colliding).
    assert self_near_dup_mask(_sigs([_LONG_A, _LONG_B, _LONG_C, _LONG_D])).tolist() == [
        False,
        False,
        False,
        False,
    ]
    # Shape guards fail loud (never a silent wrong-axis read).
    with pytest.raises(AssertionError):
        self_near_dup_mask(np.zeros(4, dtype=np.uint64))
    with pytest.raises(AssertionError):
        self_near_dup_mask(np.zeros((2, 63), dtype=np.uint64))  # 63 % 16 != 0


def _fake_stream_stage_factory(rows_by_label):
    """Signature-conformant fake of the #1739 ``_stream_stage`` network boundary.

    Mirrors the real keyword-only signature and still runs the phase's REAL
    ``keep_fn`` over raw fixture rows, so only the HF stream is faked.
    """

    def fake_stream_stage(
        *, out_path, fingerprint, row_iter_factory, keep_fn, keep_cap, stream_cap, log_label
    ):
        kept, counters = [], {"scanned": 0}
        for raw in rows_by_label[log_label]:
            counters["scanned"] += 1
            row, reject = keep_fn(raw)
            if reject is not None:
                counters[reject] = counters.get(reject, 0) + 1
            elif row is not None:
                kept.append(row)
        return kept, counters

    return fake_stream_stage


def _conv_row(prompt, response, corpus):
    row = {
        "language": "English",
        "conversation": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }
    if corpus == "lmsys":
        row["openai_moderation"] = []
    else:
        row["redacted"] = False
        row["toxic"] = False
    return row


def test_phase_found_executes_self_dedup_end_to_end(tmp_path, monkeypatch):
    """Drives the REAL phase_found body (lines fixed in v6) on a tiny fixture pool."""
    near_c = _LONG_C.replace("magnetic", "magnetik")  # response near-dup of row_a's R
    rows_by_label = {
        # row_a kept; row_b's RESPONSE near-dups row_a's -> dropped at the resp screen.
        "p1_found_lmsys": [
            _conv_row(_LONG_A, _LONG_C, "lmsys"),
            _conv_row(_LONG_B, near_c, "lmsys"),
        ],
        # row_c's PROMPT exactly repeats row_a's (distinct response) -> dropped at
        # the prompt screen; row_d fully distinct -> kept.
        "p1_found_wildchat": [
            _conv_row(_LONG_A, _LONG_D, "wildchat"),
            _conv_row(_LONG_B + " Extra tail sentence for distinctness.", _LONG_B, "wildchat"),
        ],
    }
    monkeypatch.setattr(sc, "_stream_stage", _fake_stream_stage_factory(rows_by_label))
    args = sc.build_argparser().parse_args(["--phase", "found", "--out-root", str(tmp_path)])
    sc.phase_found(args)  # pre-v6 this raised TypeError at the near_dup_mask call
    pool = sc.read_jsonl(tmp_path / "found" / "found_pool.jsonl")
    assert [r["corpus"] for r in pool] == ["lmsys", "wildchat"]
    assert pool[0]["prompt"] == _LONG_A and pool[0]["response"] == _LONG_C
    assert pool[1]["response"] == _LONG_B
    assert all(r["id"] for r in pool)


def test_phase_cvefixes_executes_self_dedup_end_to_end(tmp_path, monkeypatch):
    """Drives the REAL phase_cvefixes body (the third fixed call site) end-to-end."""
    code_a = (
        "def handler(req):\n"
        "    query = 'SELECT * FROM users WHERE id=' + req.params['id']\n"
        "    return db.execute(query)\n"
    )
    code_a_near = code_a.replace("users", "userz")  # near-dup of code_a
    code_b = (
        "int read_config(const char *path) {\n"
        "    char buf[64]; strcpy(buf, path); return open(buf, O_RDONLY);\n"
        "}\n"
    )

    def raw(code):
        return {
            "cvss_score": "7.5",
            "func_before": code,
            "func_after": "",
            "description": "a memory safety or injection defect in the handler",
        }

    rows_by_label = {"p1_cvefixes": [raw(code_a), raw(code_a_near), raw(code_b)]}
    monkeypatch.setattr(sc, "_stream_stage", _fake_stream_stage_factory(rows_by_label))
    args = sc.build_argparser().parse_args(["--phase", "cvefixes", "--out-root", str(tmp_path)])
    sc.phase_cvefixes(args)  # pre-v6 this raised TypeError at the near_dup_mask call
    pool = sc.read_jsonl(tmp_path / "cvefixes" / "cvefixes_pool.jsonl")
    # First occurrence kept, near-dup dropped, distinct kept.
    assert [r["code_before"] for r in pool] == [code_a.strip(), code_b.strip()]
    assert all(r["cvss"] == 7.5 for r in pool)
