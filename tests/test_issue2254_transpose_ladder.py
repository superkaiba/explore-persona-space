"""CPU-only tests for the #2254 `transpose_ladder` round (plan v14 §4.3).

No network, no GPU: toy-matrix algebra invariants with LIBRARY-INDEPENDENT
references (``np.linalg.pinv(M) @ r_B`` and dense ``M.T @ r_B`` computed
directly — never recomposed through the helper under test), λ-rule
determinism, destandardization parity, slug collision-freedom + all-4-slugs
production id tests, HALT-gate negative paths (corrupted ``kstar`` ⇒ raise;
injected parity mismatch ⇒ raise; a spy asserting the steer phase is never
entered after a HALT), the §12.19 parent-reference reduce fixture (committed
``eval_results/issue_2254`` artifacts), and real-body end-to-end runs of
``phase_directions`` / ``phase_reduce`` / ``_judge_ladder_cell`` with fakes
ONLY at the external HF/API/tokenizer boundaries (signature-conformant by
construction: real module functions, a real ``JudgeResult`` dataclass
instance, ``create_autospec`` on the judge boundary).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest

import scripts.issue2254_first_k_steering as fk
import scripts.issue2254_ladder_figures as ladder_figs
import scripts.issue2254_preimage as i2254
import scripts.issue2254_transpose_ladder as ladder

RNG = np.random.default_rng(20254)


def _toy_map(h: int = 12, n: int = 40, seed: int = 3):
    """Toy production-shaped fit: (M, Um, Sm, Vmt, xsd, rb) via the parent's
    own ridge fit (so the stored-spectrum conventions hold)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, h))
    y = x @ rng.normal(size=(h, h)) * 0.5 + rng.normal(size=(n, h)) * 0.1
    fit = i2254.ridge_fit_matrix(x, y)
    M, Um, Sm, Vmt = i2254.map_svd(fit["W"])
    rb = rng.normal(size=h)
    return M, Um, Sm, Vmt, fit["xsd"], rb


# ---------------------------------------------------------------------------
# algebra invariants (library-independent endpoint references)
# ---------------------------------------------------------------------------


def test_transpose_weight_matches_dense_transpose():
    M, Um, Sm, Vmt, _xsd, rb = _toy_map()
    w = ladder.ladder_weight(Um, Sm, Vmt, rb, "tr")
    ref = M.T @ rb  # dense reference, never recomposed through the helper
    assert np.linalg.norm(w - ref) / np.linalg.norm(ref) < 1e-10
    assert ladder.transpose_residual(M, w, rb) < 1e-10


def test_ridge_lambda_to_zero_matches_full_pinv():
    M, Um, Sm, Vmt, _xsd, rb = _toy_map()
    lam = 1e-14
    w = ladder.ladder_weight(Um, Sm, Vmt, rb, "rl1", lam=lam)
    ref = np.linalg.pinv(M) @ rb  # library-independent full-rank pinv reference
    cos = float(w @ ref / (np.linalg.norm(w) * np.linalg.norm(ref)))
    assert cos > 1.0 - 1e-9, cos


def test_ridge_lambda_to_inf_matches_transpose():
    M, Um, Sm, Vmt, _xsd, rb = _toy_map()
    lam = 1e12
    w = ladder.ladder_weight(Um, Sm, Vmt, rb, "rl3", lam=lam)
    ref = M.T @ rb
    cos = float(w @ ref / (np.linalg.norm(w) * np.linalg.norm(ref)))
    assert cos > 1.0 - 1e-9, cos


def test_ridge_residual_identity_holds():
    M, Um, Sm, Vmt, _xsd, rb = _toy_map()
    lams = ladder.ladder_lambdas(Sm)
    for slug, lam in lams.items():
        w = ladder.ladder_weight(Um, Sm, Vmt, rb, slug, lam=lam)
        assert ladder.ridge_residual(M, w, rb, lam) < 1e-8, slug


def test_ridge_requires_positive_lambda():
    _M, Um, Sm, Vmt, _xsd, rb = _toy_map()
    with pytest.raises(ValueError, match="positive finite lambda"):
        ladder.ladder_weight(Um, Sm, Vmt, rb, "rl2", lam=None)
    with pytest.raises(ValueError, match="unknown ladder slug"):
        ladder.ladder_weight(Um, Sm, Vmt, rb, "nope", lam=1.0)


def test_destandardization_parity_vs_parent_helper():
    _M, Um, Sm, Vmt, xsd, rb = _toy_map()
    w = ladder.ladder_weight(Um, Sm, Vmt, rb, "tr")
    d_mine = xsd * w
    d_mine = d_mine / np.linalg.norm(d_mine)
    d_parent = i2254.destandardized_direction(xsd, w)
    assert np.allclose(d_mine, d_parent, atol=0, rtol=0), "fold must mirror the parent EXACTLY"


def test_lambda_rule_determinism_and_ordering():
    _M, _Um, Sm, _Vmt, _xsd, _rb = _toy_map()
    a = ladder.ladder_lambdas(Sm)
    b = ladder.ladder_lambdas(Sm)
    assert a == b, "λ rule must be deterministic"
    s2 = np.asarray(Sm, dtype=np.float64) ** 2
    for slug, q in ladder.LAMBDA_QUANTILES.items():
        assert a[slug] == float(np.quantile(s2, q)), slug
    assert a["rl1"] < a["rl2"] < a["rl3"], "quantile λs must be ordered"


# ---------------------------------------------------------------------------
# slug registration + production id grammar (§12.15)
# ---------------------------------------------------------------------------


def test_slugs_registered_and_collision_free():
    for slug in ladder.LADDER_SLUGS:
        assert slug in i2254._DIR_SHORT, f"{slug} not registered in _DIR_SHORT"
    tokens = list(i2254._DIR_SHORT.values())
    assert len(tokens) == len(set(tokens)), f"_DIR_SHORT token collision: {tokens}"


def test_all_slugs_through_production_cell_and_judge_ids():
    ids, jids = set(), set()
    for slug in ladder.LADDER_SLUGS:
        cell = {
            "behavior": "sycophancy",
            "kind": "steer",
            "direction": slug,
            "position": "context",
            "layer_config": "mid",
            "c": 4.0,
        }
        cid = i2254._cell_id(cell)  # production grammar; KeyError = unregistered slug
        jid = i2254._judge_ctx_id(cell, 43, 199)
        assert len(jid) <= 49
        ids.add(cid)
        jids.add(jid)
    assert len(ids) == len(ladder.LADDER_SLUGS)
    assert len(jids) == len(ladder.LADDER_SLUGS)


def test_registered_grid_from_committed_ops_matches_plan():
    ops = i2254._load_operating_points(ladder.INPUTS_ROOT)
    grid = ladder.registered_grid_from_ops(ops)
    assert grid == ladder.PLAN_GRID
    args = argparse.Namespace(smoke=False)
    cells = ladder.registered_cells(args)
    assert len(cells) == ladder.FAMILY_SIZE == 44
    assert len({i2254._cell_id(c) for c in cells}) == 44
    per_arm = {s: sum(1 for c in cells if c["direction"] == s) for s in ladder.LADDER_SLUGS}
    assert set(per_arm.values()) == {ladder.ARM_SIZE}


# ---------------------------------------------------------------------------
# HALT gates — negative paths (§4.1 (i)/(ii); §12.16)
# ---------------------------------------------------------------------------


def test_gate_i_corrupted_kstar_raises():
    maps = Path(__file__).resolve()  # placeholder; real fixture below
    del maps
    _M, _Um, Sm, _Vmt, _xsd, _rb = _toy_map()
    z = {"s": Sm, "lam": float(np.median(Sm) ** 2), "kstar": 999999}
    with pytest.raises(ladder.LadderHaltError, match=r"gate \(i\) FAIL"):
        ladder.halt_npz_selfconsistency(z, 14)


def test_gate_ii_parity_pass_and_mismatch_raises(tmp_path):
    rb = ladder.make_fixture_maps(tmp_path / "maps", layers=(14,), seed=7)
    z = np.load(tmp_path / "maps" / "L14.npz")
    d_re = ladder.rebuild_parent_preimage(z, rb["evil"][14])
    # exact rebuild passes
    assert ladder.halt_rebuild_parity(d_re, d_re.copy(), "evil", 14) >= ladder.PARITY_COS_MIN
    # injected mismatch (a rolled copy decorrelates) raises
    with pytest.raises(ladder.LadderHaltError, match=r"gate \(ii\) FAIL"):
        ladder.halt_rebuild_parity(d_re, np.roll(d_re, 3), "evil", 14)


def test_amp_tripwire_raises_on_degenerate_amplification():
    w = np.ones(64)
    w[0] = 1e8
    with pytest.raises(ladder.LadderHaltError, match="tripwire"):
        ladder.halt_amp_tripwire(w, "evil", "rl1", 14)
    ladder.halt_amp_tripwire(np.ones(64), "evil", "rl1", 14)  # healthy passes


def test_steer_never_entered_after_halt(tmp_path, monkeypatch):
    """Real-body chain: phase_directions on a CORRUPTED-kstar fixture raises
    the HALT, and the (spied) steer phase is never entered (plan §7(a))."""
    maps_dir = tmp_path / "maps"
    rb = ladder.make_fixture_maps(maps_dir, layers=ladder.FIXTURE_LAYERS)
    ladder.make_fixture_parent_bank(tmp_path, maps_dir, rb)
    # corrupt L14's stored kstar
    z = dict(np.load(maps_dir / "L14.npz"))
    z["kstar"] = np.int64(int(z["kstar"]) + 5)
    np.savez(maps_dir / "L14.npz", **z)
    monkeypatch.setattr(ladder, "_RB_LOADER", lambda: rb)
    monkeypatch.setattr(ladder, "_PARENT_VEC_LOADER", ladder._fixture_parent_vec_loader)
    monkeypatch.setattr(ladder, "_UPLOAD", ladder._fixture_upload)
    calls: list[str] = []

    def spy_steer(args):
        calls.append("steer")

    monkeypatch.setitem(ladder.PHASES, "steer", spy_steer)
    args = argparse.Namespace(
        out_root=str(tmp_path),
        maps_dir=str(maps_dir),
        layers=list(ladder.FIXTURE_LAYERS),
        behaviors=list(ladder.ROUND_BEHAVIORS),
        smoke=False,
        fit_workers=1,
    )
    with pytest.raises(ladder.LadderHaltError, match=r"gate \(i\) FAIL"):
        ladder.run_phases(args, ["directions", "steer"])
    assert calls == [], "steer must never be entered after a HALT"


# ---------------------------------------------------------------------------
# production-loader round-trip at REAL H (gate (iii) loader leg body coverage)
# ---------------------------------------------------------------------------


def test_production_loader_roundtrip_at_real_hidden_dim(tmp_path):
    rng = np.random.default_rng(5)
    w = rng.normal(size=i2254.HIDDEN_DIM)
    xsd = np.abs(rng.normal(size=i2254.HIDDEN_DIM)) + 0.1
    d = i2254.destandardized_direction(xsd, w)
    bank_dir = tmp_path / "directions"
    bank_dir.mkdir(parents=True)
    manifest: list = []
    i2254._save_direction(bank_dir, "evil", "tr", 14, d, manifest)
    loaded = i2254._ensure_direction_vec(tmp_path, "evil", "tr", 14).numpy()
    cos = float(loaded @ d / (np.linalg.norm(loaded) * np.linalg.norm(d)))
    assert cos >= ladder.LOADER_ROUNDTRIP_MIN_COS, cos


# ---------------------------------------------------------------------------
# §12.19 parent-reference reduce fixture (committed parent artifacts)
# ---------------------------------------------------------------------------


def test_parent_reference_margin_fixture():
    out = ladder.assert_parent_reference_margin()
    assert out["verdict"] == "PASS"
    assert abs(out["margin"] - 2.458555555555556) < 1e-9
    assert out["cell_id"] == "evil__cxd__ctx__L14__c4"


def test_parent_band_and_floor_readers():
    assert ladder.load_parent_band("evil") == 0.0
    assert abs(ladder.load_parent_band("sycophancy") - 10.890474999999999) < 1e-12
    for b in ladder.ROUND_BEHAVIORS:
        floor_q, floor_mean, ceiling = ladder.load_parent_floor(b)
        assert floor_q.shape == (20,)
        assert np.isfinite(floor_mean) and np.isfinite(ceiling)


# ---------------------------------------------------------------------------
# phase_directions — real body end-to-end on tiny fixtures (seams at the
# HF boundary only; gates (i)/(ii)/(iii) all on the executed path)
# ---------------------------------------------------------------------------


def test_phase_directions_end_to_end_tiny(tmp_path, monkeypatch):
    maps_dir = tmp_path / "maps"
    rb = ladder.make_fixture_maps(maps_dir, layers=ladder.FIXTURE_LAYERS)
    ladder.make_fixture_parent_bank(tmp_path, maps_dir, rb)
    monkeypatch.setattr(ladder, "_RB_LOADER", lambda: rb)
    monkeypatch.setattr(ladder, "_PARENT_VEC_LOADER", ladder._fixture_parent_vec_loader)
    uploads: list[tuple[str, str]] = []

    def recording_upload(local_dir, path_in_repo, allow=None):
        uploads.append((str(local_dir), path_in_repo))

    monkeypatch.setattr(ladder, "_UPLOAD", recording_upload)
    args = argparse.Namespace(
        out_root=str(tmp_path),
        maps_dir=str(maps_dir),
        layers=list(ladder.FIXTURE_LAYERS),
        behaviors=list(ladder.ROUND_BEHAVIORS),
        smoke=False,
        fit_workers=2,
    )
    ladder.phase_directions(args)
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    report = json.loads((rroot / "ladder_report.json").read_text())
    n_expected = len(ladder.ROUND_BEHAVIORS) * len(ladder.LADDER_SLUGS) * len(ladder.FIXTURE_LAYERS)
    assert report["n_direction_files"] == n_expected == 16
    assert len(list((rroot / "directions_ladder").glob("*.pt"))) == n_expected
    # gate (ii) recorded at all 4 parity cells with cos >= 0.999
    parity = report["gates"]["rebuild_parity_cos"]
    assert sorted(parity) == ["evil__L14", "evil__L17", "sycophancy__L14", "sycophancy__L17"]
    assert all(v >= ladder.PARITY_COS_MIN for v in parity.values())
    for ly in ladder.FIXTURE_LAYERS:
        blk = report["layers"][str(ly)]
        assert blk["lambdas"]["rl1"] < blk["lambdas"]["rl2"] < blk["lambdas"]["rl3"]
        for key, row in blk["arms"].items():
            assert row["residual"] <= ladder.RESIDUAL_RTOL, (ly, key)
            assert row["loader_roundtrip_cos"] >= ladder.LOADER_ROUNDTRIP_MIN_COS
            assert 0.0 < row["alignment_concentration"]["kstar"] <= 1.0
            assert "cos_vs_parent_pre" in row and "cos_vs_ctxext" in row and "cos_vs_rb" in row
    # the bank copy exists where the steer loader reads (out_root/directions)
    assert (tmp_path / "directions" / "evil_tr_L14.pt").is_file()
    # uploads: the .pt bank append + the ladder report (recorded, not sent)
    assert any(p.endswith("/directions") for _d, p in uploads)


# ---------------------------------------------------------------------------
# _judge_ladder_cell — real body; boundary faked signature-conformantly
# ---------------------------------------------------------------------------


def _fixture_gen_record(tmp_path: Path, cell: dict, n_q: int = 2) -> Path:
    texts = [f"fixture answer {qi}" for qi in range(n_q)]
    rec = {
        "cell_id": i2254._cell_id(cell),
        "cell": cell,
        "alphas": {"L14": 0.1},
        "q_of_context": list(range(n_q)),
        "seeds": {
            "42": {
                "completions": [[t] for t in texts],
                "coherent_flags": [[True] for _ in texts],
                "condition_passes": [True for _ in texts],
            }
        },
        "max_new_tokens": 2048,
        "cap_hit_fraction": 0.0,
    }
    path = tmp_path / f"{rec['cell_id']}.json"
    path.write_text(json.dumps(rec))
    return path


def test_judge_ladder_cell_body_and_resume(tmp_path, monkeypatch):
    from explore_persona_space.eval.graded_judge import JudgeResult
    from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

    cell = {
        "behavior": "evil",
        "kind": "steer",
        "direction": "tr",
        "position": "context",
        "layer_config": "L14",
        "c": 4.0,
    }
    gen_path = _fixture_gen_record(tmp_path, cell)
    rroot = tmp_path / "round"

    def fake_eval_questions(behavior: str) -> list[str]:
        return ["q0", "q1"]  # HF-staged e1 bank boundary (signature-conformant)

    monkeypatch.setattr(i2254, "_eval_questions", fake_eval_questions)

    iids = [rollout_item_id(i2254._judge_ctx_id(cell, 42, i), 0) for i in range(2)]
    result = JudgeResult(
        scores={iids[0]: 80.0, iids[1]: 60.0},
        n_total_draws=4,
        n_dropped_draws=0,
        per_item_draw_counts={iid: 2 for iid in iids},
        per_item_scores={iids[0]: [80.0, 80.0], iids[1]: [60.0, 60.0]},
    )
    merged = {iids[0]: [80.0, 80.0], iids[1]: [60.0, 60.0]}
    fake_judge = create_autospec(
        fk._judge_graded_with_refusal_reissue, return_value=(result, merged, None)
    )
    monkeypatch.setattr(fk, "_judge_graded_with_refusal_reissue", fake_judge)
    args = argparse.Namespace(force=False)
    out = ladder._judge_ladder_cell(args, rroot, gen_path, "RUBRIC", 2)
    assert out["per_question_mean_score"] == [80.0, 60.0]
    assert out["mean_score"] == 70.0
    assert out["coherence_pass"] is True
    assert out["accounting"]["frac_items_complete"] == 1.0
    assert (rroot / "judge" / "judged" / f"{out['cell_id']}.json").is_file()
    assert fake_judge.call_count == 1
    # cached-skip resume: identical gen bytes + instrument -> no second call
    out2 = ladder._judge_ladder_cell(args, rroot, gen_path, "RUBRIC", 2)
    assert out2["mean_score"] == 70.0
    assert fake_judge.call_count == 1, "judged checkpoint resume must skip the API boundary"


# ---------------------------------------------------------------------------
# phase_reduce — real body on a full-44 synthetic round (committed parent
# floor/band artifacts; Undefined rule; fresh_nulls; tags; figures render)
# ---------------------------------------------------------------------------


def test_phase_reduce_end_to_end_fixture(tmp_path, monkeypatch):
    args = argparse.Namespace(out_root=str(tmp_path), smoke=False)
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    ladder.make_fixture_round(rroot, args)
    monkeypatch.setattr(ladder, "_TOKENIZER_LOADER", ladder._FixtureTokenizer)
    ladder.phase_reduce(args)

    percell = json.loads((rroot / "reduce" / "delta_score_percell.json").read_text())
    verdicts = json.loads((rroot / "reduce" / "verdicts.json").read_text())

    # §3 lattice: exactly the constructed clearing cell fires H1
    assert verdicts["label"] == "H1"
    assert verdicts["h1_clearing_cells"] == ["sycophancy__tr__ctx__L17__c4"]
    assert verdicts["fresh_nulls"] is False
    assert verdicts["parent_reference_margin_check"]["verdict"] == "PASS"

    # Undefined-cell rule: the all-dropped cell is OUTSIDE both H1 support and
    # the bounded-non-clear narration
    assert verdicts["narration"]["undefined_cells"] == ["evil__tr__ctx__all__c0p5"]
    und = percell["behaviors"]["evil"]["evil__tr__ctx__all__c0p5"]
    assert und["label"] == "Undefined (no valid measurement)"
    assert und["margin_lo"] is None
    assert "evil__tr__ctx__all__c0p5" not in verdicts["narration"]["bounded_nonclear_cells"]

    # exact arithmetic on the constant-delta clearing cell: margin = 30 - band
    band_s = ladder.load_parent_band("sycophancy")
    clear = percell["behaviors"]["sycophancy"]["sycophancy__tr__ctx__L17__c4"]
    assert abs(clear["delta_score"] - 30.0) < 1e-9
    assert abs(clear["margin"] - (30.0 - band_s)) < 1e-9
    assert clear["margin_lo"] > 0 and clear["clears_nominal"]
    assert clear["tags"]["multiplicity_robust_family"] is True
    assert clear["tags"]["multiplicity_robust_within_arm"] is True

    # evil constant delta 0 vs the exact-zero band: margin_lo == 0 does NOT
    # clear (strict >), and sycophancy delta-1 cells are bounded non-clears
    ev = percell["behaviors"]["evil"]["evil__tr__ctx__L14__c4"]
    assert ev["clears_nominal"] is False
    assert "sycophancy__rl1__ctx__mid__c4" in verdicts["narration"]["bounded_nonclear_cells"]

    # intrusion sensitivity: the CJK-seeded cell counts exactly one intruded
    # row; the binding as-is mean replays the stored mean_score
    sens = percell["behaviors"]["sycophancy"]["sycophancy__tr__ctx__mid__c4"]["sensitivity"]
    assert sens["n_intruded_valid"] == 1
    assert sens["cjk_common"] == pytest.approx(1.0 / 20.0)
    assert sens["mean_zeroed_intrusion"] < sens["mean_asis"]

    # selection-aware companions present at both grains + the all-44 read
    assert verdicts["selection_aware"]["arm"]["tr"]["n_cells"] == 10  # 11 - 1 undefined
    assert verdicts["all44_companion"]["n_cells"] == 43
    assert verdicts["all44_companion"]["argmax_cell"] == "sycophancy__tr__ctx__L17__c4"

    # figures: the hero renders from these outputs (required); report-backed
    # exploratory figures skip with named reasons (no ladder_report here)
    fig_dir = tmp_path / "figs"
    res = ladder_figs.render_all(rroot, fig_dir, require=("hero_ladder",))
    assert "hero_ladder" in res["rendered"]
    assert (fig_dir / "hero_ladder.png").is_file()
    meta = json.loads((fig_dir / "hero_ladder.meta.json").read_text())
    assert "fresh_nulls: false" in meta["scope_note"]
