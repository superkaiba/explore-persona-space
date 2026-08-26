"""Issue #2588 review-round-2 regression tests (A1/B1/B2/B3/B4/B5/C2/C3).

Every test executes the REAL body of the round-modified function; fakes sit
only at external boundaries — GPU-scale weights -> a tiny same-shape torch
wrapper (real ``.model.layers`` chain, real forward + hooks), the Anthropic
API -> real ``DispatchResult`` dataclass instances via the documented
``_dispatch_judge_round`` seam, HF upload -> a recorded signature-mirroring
fake. No network, no GPU, tmp_path-only writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2330_qwen35_generate_capture as G
import issue2588_panel_common as PC
import issue2588_run_cell as RC
import issue2588_trend as TR

from explore_persona_space.llm.api_dispatch import (
    RESULT_OK,
    RESULT_TRANSPORT,
    DispatchItem,
    DispatchResult,
)


def _args(**over):
    """Real argparse namespace (signature-conformant by construction)."""
    a = RC._build_parser().parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# A1: _capture_stage against a real-shaped decoder wrapper
# ---------------------------------------------------------------------------

H_DIM = 16
VOCAB = 64


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(H_DIM, H_DIM)

    def forward(self, x):
        return (self.lin(x),)  # tuple output — exercises G._unwrap in the hook


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, H_DIM)
        self.layers = torch.nn.ModuleList([_Block(), _Block()])


class TinyWrapped(torch.nn.Module):
    """Real decoder wrapper shape: ``.model.layers`` ModuleList + a forward
    (input_ids, attention_mask) that RUNS the blocks so hooks fire (B, T, H)."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = self.model.embed(input_ids)
        for blk in self.model.layers:
            h = G._unwrap(blk(h))
        return h


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1


def _fake_build_capture_row(tok, wrow, *, positions_wanted):
    """Signature-mirrors PC.build_capture_row_2588 (unmodified helper; faked
    so the test needs no real tokenizer/offset mapping)."""
    n_p = int(wrow["n_prompt_tokens"])
    row = {
        "row_id": wrow["row_id"],
        "prompt_ids": list(range(2, 2 + n_p)),
        "comp_ids": [3, 4, 5],
        "positions": {"prompt_last": n_p - 1},
        "spans": {"ans": (n_p, n_p + 3)},
    }
    return row, ""


def test_capture_stage_executes_real_decoder_wrapper(tmp_path, monkeypatch):
    """A1: _capture_stage unpacks G._resolve_decoder_blocks's (blocks, depth)
    tuple. Pre-fix the tuple return was treated as the layer list, so EVERY
    capture crashed at hook registration (TypeError on tuple indexing)."""
    monkeypatch.setattr(PC, "build_capture_row_2588", _fake_build_capture_row)
    args = _args(capture_batch_size=2)
    cell = PC.Cell("q25_7b", "a", True)  # fresh -> parsed-jsonl input branch
    paths = {"parsed": tmp_path / "parsed", "capture": tmp_path / "capture"}
    for p in paths.values():
        p.mkdir(parents=True)
    rows = [
        {
            "row_id": f"test_1000_{i}",
            "n_prompt_tokens": 4 + (i % 3),
            "prompt": "p",
            "text": "t",
            "ans_char_span": [0, 1],
        }
        for i in range(5)
    ]
    with (paths["parsed"] / "test_1000.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    hf = TinyWrapped()
    blocks, depth = G._resolve_decoder_blocks(hf)  # the tuple contract under test
    assert blocks is not None and depth == 1 and len(blocks) == 2

    RC._capture_stage(args, cell, paths, hf, _FakeTok(), "test_1000", [0, 1])

    for li in (0, 1):
        shards = sorted((paths["capture"] / "test_1000" / f"L{li:02d}").glob("shard*.npz"))
        assert shards, f"layer {li}: no shards written"
        with np.load(shards[0], allow_pickle=False) as z:
            assert z["y_ans"].shape == (5, H_DIM)
            assert z["x_prompt_last"].shape == (5, H_DIM)
            assert np.isfinite(z["y_ans"]).all()
    meta = json.loads((paths["capture"] / "test_1000" / "rows.json").read_text(encoding="utf-8"))
    assert len(meta["rows"]) == 5
    assert meta["positions"] == ["prompt_last"]


# ---------------------------------------------------------------------------
# B1: phase sentinels — second run makes ZERO expensive calls; --force re-runs
# ---------------------------------------------------------------------------


def test_run_phases_sentinel_skip_and_force(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _fake_phase(args, cell, paths):
        calls["n"] += 1

    monkeypatch.setitem(RC.PHASES, "fake-phase", _fake_phase)
    args = _args()
    cell = PC.Cell("q25_7b", "a", True)
    paths = {"cell": tmp_path}

    assert RC._run_phases(args, cell, paths, ("fake-phase",)) == ["fake-phase"]
    assert calls["n"] == 1
    # Idempotent re-run: sentinel skip, ZERO expensive calls (B1).
    assert RC._run_phases(args, cell, paths, ("fake-phase",)) == []
    assert calls["n"] == 1
    # --force escape re-runs the phase.
    args.force = True
    assert RC._run_phases(args, cell, paths, ("fake-phase",)) == ["fake-phase"]
    assert calls["n"] == 2
    # The odd-layer pass carries its OWN sentinel (never satisfied by swept's).
    args.force = False
    args.layer_set = "odd"
    assert RC._run_phases(args, cell, paths, ("fake-phase",)) == ["fake-phase"]
    assert calls["n"] == 3
    assert (tmp_path / "phase_done" / "fake-phase.json").exists()
    assert (tmp_path / "phase_done" / "fake-phase_odd.json").exists()


# ---------------------------------------------------------------------------
# B2: capture inputs validate BEFORE any tokenizer/model load
# ---------------------------------------------------------------------------


def test_validate_capture_inputs_blocks_before_model_load(tmp_path, monkeypatch):
    hits: list[str] = []
    monkeypatch.setattr(RC.G, "_load_capture_model", lambda *a, **k: hits.append("model"))
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: hits.append("tok")),
    )
    args = _args()
    cell = PC.Cell("q35_9b", "b", True)  # fresh arm-b: first stage needs parsed jsonl
    paths = {"parsed": tmp_path / "parsed", "cell": tmp_path / "cell"}
    for p in paths.values():
        p.mkdir(parents=True)
    with pytest.raises(AssertionError, match="run --phase parse first"):
        RC.phase_capture(args, cell, paths)
    assert hits == []  # _load_capture_model + AutoTokenizer NEVER called (B2)


# ---------------------------------------------------------------------------
# B3: SR1 repeat-draw retrieval ceiling — pinned synthetic value
# ---------------------------------------------------------------------------


def _write_ceiling_shard(paths, seed, layer, y, cis):
    d = paths["capture"] / f"ceiling_s{seed}" / f"L{layer:02d}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / "shard000.npz",
        row_ids=np.array([f"ceiling_s{seed}_{ci}" for ci in cis]),
        y_ans=y.astype(np.float32),
    )


def test_ceiling_retrieval_pinned_synthetic(tmp_path):
    """Constructed 7/10 seed-43->seed-44 retrieval hits => ceiling 0.7."""
    paths = {"capture": tmp_path / "capture"}
    n = 10
    yb = np.eye(n)
    ya = yb.copy()
    for i in (2, 5, 7):  # 3 engineered misses: query retrieves the WRONG target
        ya[i] = yb[(i + 1) % n]
    cis = [f"{i:03d}" for i in range(n)]
    _write_ceiling_shard(paths, PC.CEILING_SEEDS[0], 0, ya, cis)
    _write_ceiling_shard(paths, PC.CEILING_SEEDS[1], 0, yb, cis)
    rec = RC._ceiling_retrieval(paths, 0)
    assert rec is not None
    assert rec["ceiling_acc1_cos"] == pytest.approx(0.7)
    assert rec["n_pool"] == 10
    assert rec["chance"] == pytest.approx(0.1)
    assert rec["seed_pair"] == list(PC.CEILING_SEEDS)


# ---------------------------------------------------------------------------
# B4: complete-case paired bootstrap + h2_reads == 7 fail-loud
# ---------------------------------------------------------------------------


def test_paired_gap_boot_asymmetric_drops():
    universe = [str(i) for i in range(12)]
    hits_b = {str(i): 1 if i in (2, 3, 4, 5) else 0 for i in range(0, 10)}
    hits_a = {str(i): 0 for i in range(2, 12)}
    hits_b["0"] = 1  # b-only row — outside the intersection, must NOT leak
    hits_a["11"] = 1  # a-only row — outside the intersection, must NOT leak
    matrix = TR._shared_resample_matrix(universe, 200, 42)
    gap, boot, n_shared = TR._paired_gap_boot(hits_b, hits_a, universe, matrix)
    assert n_shared == 8  # shared rows = {2..9} (complete-case)
    assert gap == pytest.approx(4 / 8)
    assert boot.shape == (200,)
    assert ((boot >= 0.0) & (boot <= 1.0)).all()


def _map_rec(perrow_cis, hits, gpqa_ids, ghits, obs=0.5):
    return {
        "fits": {
            "layer_star": 5,
            "layers": {"5": {"knn_test": {"ridge": {"cosine": {"acc_at_k": {"1": obs}}}}}},
        },
        "nulls": {"null_mean_acc1_cos": 0.01, "null_sd_acc1_cos": 0.005, "perm_draws": 200},
        "perrow": {
            "row_ids": [f"test_1000_{c}" for c in perrow_cis],
            "hit1_cos": hits,
        },
        "gpqa_perrow": {"row_ids": gpqa_ids, "same_q_hit": ghits},
        "gpqa_transfer": None,
        "resid": None,
        "judge_pending": None,
        "judge_verdicts": None,
    }


def _h2_fixture():
    maps: dict[str, dict] = {}
    gids = [f"q{i}_s42" for i in range(8)]
    for idx, key in enumerate(TR.QWEN_THINKING_KEYS):
        cis_b = [str(i) for i in range(0, 9)]  # drops ci 9 (asymmetric)
        cis_a = [str(i) for i in range(1, 10)]  # drops ci 0
        ghits_b = [1] * idx + [0] * (8 - idx)  # distinct nonzero surface deltas
        maps[TR.MapRef(key, "b", "cot_boundary").map_id] = _map_rec(cis_b, [1] * 9, gids, ghits_b)
        maps[TR.MapRef(key, "a", "prompt_last").map_id] = _map_rec(cis_a, [0] * 9, gids, [0] * 8)
    return maps


def test_h2_reads_raw_complete_case_gaps():
    maps = _h2_fixture()
    universe = [str(i) for i in range(10)]
    matrix = TR._shared_resample_matrix(universe, 100, 42)
    out = TR.h2_reads(maps, universe, matrix)
    pair = out["pairs"]["q35_9b"]
    assert pair["n_shared_generic"] == 8  # complete-case intersection {1..8}
    assert pair["gap_generic_raw"] == pytest.approx(1.0)
    assert pair["n_shared_gpqa"] == 8
    idx = TR.QWEN_THINKING_KEYS.index("q35_9b")
    assert pair["gap_gpqa_raw"] == pytest.approx(idx / 8)
    assert pair["surface_delta"] == pytest.approx(idx / 8 - 1.0)
    assert len(pair["gap_generic_raw_ci95"]) == 2
    assert "gap_generic_cal" in pair  # E4 sensitivity field rides along
    assert out["surface_wilcoxon"]["n"] == 7
    assert "RAW gaps" in out["surface_wilcoxon"]["statistic_def"]


def test_h2_reads_fails_loud_below_seven_pairs():
    maps = _h2_fixture()
    del maps[TR.MapRef("q36_27b", "b", "cot_boundary").map_id]
    universe = [str(i) for i in range(10)]
    matrix = TR._shared_resample_matrix(universe, 50, 42)
    with pytest.raises(AssertionError, match="ALL 7"):
        TR.h2_reads(maps, universe, matrix)


# ---------------------------------------------------------------------------
# B5: pilot gate / transport re-drive / deterministic verdict merge
# ---------------------------------------------------------------------------


def _dr(
    item_id, *, result=None, error=False, category=RESULT_OK, stop_reason="end_turn", reason=None
):
    return DispatchResult(
        item_id=item_id,
        result=result,
        error=error,
        reason=reason,
        category=category,
        stop_reason=stop_reason,
    )


def test_pilot_gate_truncation_raises(tmp_path):
    res = {f"r{i}": _dr(f"r{i}", result="B") for i in range(100)}
    res["r3"] = _dr("r3", result=None, stop_reason="max_tokens")  # one truncation
    with pytest.raises(RuntimeError, match="PILOT GATE FAIL"):
        TR._pilot_gate(res, tmp_path)
    rep = json.loads((tmp_path / "gpqa_judge_pilot.json").read_text(encoding="utf-8"))
    assert rep["n_stop_reason_max_tokens"] == 1


def test_pilot_gate_parse_fail_rate(tmp_path):
    # 5/100 malformed (error=False, result=None) -> 5% >= 2% -> gate FAIL.
    res = {f"r{i}": _dr(f"r{i}", result="A") for i in range(100)}
    for i in range(5):
        res[f"r{i}"] = _dr(f"r{i}", result=None)
    with pytest.raises(RuntimeError, match="PILOT GATE FAIL"):
        TR._pilot_gate(res, tmp_path)
    # 1/100 -> passes with the malformed row counted, never coerced.
    res = {f"r{i}": _dr(f"r{i}", result="A") for i in range(100)}
    res["r0"] = _dr("r0", result=None)
    rep = TR._pilot_gate(res, tmp_path)
    assert rep["gates"]["parse_fail_below_2pct"] is True
    assert rep["n_parse_fail"] == 1


def test_dispatch_wave_transport_exhaustion_raises(tmp_path, monkeypatch):
    seen_dirs: list[Path] = []

    def _fake_round(items, checkpoint_dir):
        seen_dirs.append(checkpoint_dir)
        return {
            it.item_id: _dr(
                it.item_id,
                error=True,
                category=RESULT_TRANSPORT,
                stop_reason=None,
                reason="transport_exhausted",
            )
            for it in items
        }

    monkeypatch.setattr(TR, "_dispatch_judge_round", _fake_round)
    items = [DispatchItem("x1", {}), DispatchItem("x2", {})]
    with pytest.raises(RuntimeError, match="never persisted as drops"):
        TR._dispatch_wave(items, tmp_path, "pilot")
    assert len(seen_dirs) == TR.JUDGE_MAX_TRANSPORT_ROUNDS
    # FRESH checkpoint dir per re-drive round (api_dispatch re-serves persisted
    # transport rows on same-checkpoint resume — the B5 caveat).
    assert len({str(d) for d in seen_dirs}) == TR.JUDGE_MAX_TRANSPORT_ROUNDS


def test_dispatch_wave_redrive_recovers(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _fake_round(items, checkpoint_dir):
        calls["n"] += 1
        out = {}
        for it in items:
            if it.item_id == "x2" and calls["n"] == 1:
                out[it.item_id] = _dr(it.item_id, error=True, category=RESULT_TRANSPORT)
            else:
                out[it.item_id] = _dr(it.item_id, result="C")
        return out

    monkeypatch.setattr(TR, "_dispatch_judge_round", _fake_round)
    res = TR._dispatch_wave([DispatchItem("x1", {}), DispatchItem("x2", {})], tmp_path, "wave")
    assert res["x1"].result == "C" and res["x2"].result == "C"
    assert calls["n"] == 2  # one re-drive round, then convergence


def test_merged_behavioral_correction_and_fail_loud():
    beh = {
        "judge_fallback_flagged": True,
        "frac_unparseable": 0.10,
        "n_rollouts": 20,
        "n_correct": 8,
    }
    pend = {"rows": [{"row_id": f"r{i}", "gold": "B", "qid": f"q{i}"} for i in range(4)]}
    verd = {
        "verdicts": {
            "r0": {"letter": "B"},  # judge-corrected hit
            "r1": {"letter": "B"},  # judge-corrected hit
            "r2": {"letter": "A"},  # judged, wrong
            "r3": {"letter": "UNPARSEABLE"},
        },
        "n_items": 4,
        "n_correct": 2,
        "n_unparseable": 1,
        "n_malformed_dropped": 0,
        "n_error_dropped": 0,
    }
    rec = {"gpqa_transfer": {"behavioral": beh}, "judge_pending": pend, "judge_verdicts": verd}
    out = TR.merged_behavioral(rec, "m.x")
    assert out["acc_judge_corrected"] == pytest.approx((8 + 2) / 20)  # integer-count merge
    assert out["n_judge_corrected"] == 2
    assert out["frac_unparseable_after_judge"] == pytest.approx(1 / 20)
    # Flagged-pending WITHOUT verdicts = FAIL LOUD (never assemble uncorrected).
    rec2 = {"gpqa_transfer": {"behavioral": beh}, "judge_pending": pend, "judge_verdicts": None}
    with pytest.raises(RuntimeError, match="ABSENT"):
        TR.merged_behavioral(rec2, "m.x")
    # Unflagged map: behavioral metrics pass through untouched.
    rec3 = {"gpqa_transfer": {"behavioral": {"judge_fallback_flagged": False, "acc": 0.4}}}
    assert TR.merged_behavioral(rec3, "m.y") == {"judge_fallback_flagged": False, "acc": 0.4}


# ---------------------------------------------------------------------------
# C2: G2 sentinel content validation (stale/status-only refused)
# ---------------------------------------------------------------------------


def _valid_sentinel():
    return {
        "schema_version": PC.G2_SENTINEL_SCHEMA_VERSION,
        "status": "PASS",
        "store_revision_pin_recorded": RC.MF.STORE_REVISION_PIN_7B,
        "expected_r2": PC.ANCHOR_EXPECTED_R2,
        "realized_r2": PC.ANCHOR_EXPECTED_R2 + 1e-8,
        "abs_deviation": 1e-8,
        "tol": PC.ANCHOR_TOL,
        "production_path": {
            "estimator": "_fit_edge_extended_with_val",
            "realized_r2": PC.ANCHOR_EXPECTED_R2 + 2e-6,
            "abs_deviation_vs_pin": 2e-6,
            "tol": PC.ANCHOR_PROD_EQUIV_TOL,
        },
        "meta": {"git_sha": "deadbeefcafe"},
    }


def test_validate_g2_sentinel_accepts_v2_rejects_stale():
    RC._validate_g2_sentinel(_valid_sentinel())  # valid v2 sentinel accepted
    # v1 / status-only sentinel (no schema_version) refused.
    with pytest.raises(AssertionError, match="schema_version"):
        RC._validate_g2_sentinel({"status": "PASS"})
    # Numeric gate field over tolerance refused.
    bad = _valid_sentinel()
    bad["abs_deviation"] = 1.0
    with pytest.raises(AssertionError):
        RC._validate_g2_sentinel(bad)
    # Missing C1 production-path record refused.
    bad2 = _valid_sentinel()
    del bad2["production_path"]
    with pytest.raises(AssertionError, match="production-path"):
        RC._validate_g2_sentinel(bad2)


# ---------------------------------------------------------------------------
# C3: odd-layer artifact names + upload routing never touch the primary
# ---------------------------------------------------------------------------


def test_fits_name_and_oddlayer_upload_routing(tmp_path, monkeypatch):
    a_swept, a_odd = _args(), _args(layer_set="odd")
    assert RC._fits_name(a_swept, "fits", "prompt_last") == "fits_prompt_last.json"
    assert RC._fits_name(a_odd, "fits", "prompt_last") == "fits_prompt_last_odd.json"
    assert RC._fits_name(a_swept, "percell_prompt_last_L05", "") == "percell_prompt_last_L05.json"
    assert RC._tag(a_swept) == "capture"
    assert RC._tag(a_odd) == "capture_oddlayers"

    # Swept-then-odd: the odd pass routes "_odd" artifacts to fits_oddlayers/
    # nulls_oddlayers; primary artifact destinations stay byte-identical.
    cell = PC.Cell("q35_27b", "a", True)
    fits_dir = tmp_path / "fits"
    logs = tmp_path / "logs"
    fits_dir.mkdir()
    logs.mkdir()
    fits_payload = {
        "layer_star": 5,
        "layers": {"5": {"knn_test": {"ridge": {"cosine": {"acc_at_k": {"1": 0.5}}}}}},
    }
    for name in ("fits_prompt_last.json", "fits_prompt_last_odd.json"):
        (fits_dir / name).write_text(json.dumps(fits_payload), encoding="utf-8")
    for name in ("nulls_prompt_last.json", "nulls_prompt_last_odd.json"):
        (fits_dir / name).write_text(json.dumps({"null_mean_acc1_cos": 0.0}), encoding="utf-8")
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(RC, "_upload_file", lambda f, dest, what: uploads.append((f.name, dest)))
    RC.phase_upload_fits(a_odd, cell, {"fits": fits_dir, "logs": logs})
    dest_by_name = dict(uploads)
    assert "/fits_oddlayers/" in dest_by_name["fits_prompt_last_odd.json"]
    assert "/nulls_oddlayers/" in dest_by_name["nulls_prompt_last_odd.json"]
    assert "/fits/" in dest_by_name["fits_prompt_last.json"]
    assert "/nulls/" in dest_by_name["nulls_prompt_last.json"]
    assert "oddlayers" not in dest_by_name["fits_prompt_last.json"]
    assert "oddlayers" not in dest_by_name["nulls_prompt_last.json"]
    # The odd pass writes its OWN suffixed sentinel file.
    assert (logs / f"issue-2588-{cell.key}-odd-results.json").exists()


# ---------------------------------------------------------------------------
# C1 production estimator body: _fit_edge_extended_with_val on a
# well-conditioned fixture (probed through the real fit; interior lambda)
# ---------------------------------------------------------------------------


def test_fit_edge_extended_with_val_well_conditioned():
    rng = np.random.default_rng(0)
    n_tr, n_val, n_te, d = 160, 40, 40, 8
    n = n_tr + n_val + n_te
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    Y = X @ W + rng.standard_normal((n, d))  # SNR ~1: interior lambda, no edge raise
    idx = rng.permutation(n)
    tr, val, te = idx[:n_tr], idx[n_tr : n_tr + n_val], idx[n_tr + n_val :]
    pred_te, pred_val, meta = RC._fit_edge_extended_with_val(X, Y, tr, val, te, torch.device("cpu"))
    assert pred_te.shape == (n_te, d)
    assert pred_val.shape == (n_val, d)
    assert meta.get("lambda_grid_edge") is None  # fixture probed: interior lambda
    assert np.isfinite(pred_te).all() and np.isfinite(pred_val).all()
    r2 = 1 - ((pred_val - Y[val]) ** 2).sum() / ((Y[val] - Y[val].mean(axis=0)) ** 2).sum()
    assert r2 > 0.3  # a real fit, not a stub: val predictions track Y
