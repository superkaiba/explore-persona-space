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
        force=False,
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
    verifies: list[int] = []
    monkeypatch.setattr(
        ladder,
        "_UPLOAD_VERIFY",
        lambda layers: verifies.append(len(ladder._registered_direction_names(layers))),
    )
    headrooms: list[tuple[int, int]] = []
    monkeypatch.setattr(ladder, "_DIR_HEADROOM", lambda n, b: headrooms.append((n, b)))
    args = argparse.Namespace(
        out_root=str(tmp_path),
        maps_dir=str(maps_dir),
        layers=list(ladder.FIXTURE_LAYERS),
        behaviors=list(ladder.ROUND_BEHAVIORS),
        smoke=False,
        fit_workers=2,
        force=False,
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
    # uploads: the .pt bank append + the ladder report (recorded, not sent);
    # headroom preflight ran BEFORE compute; exact-set verify ran post-upload
    assert any(p.endswith("/directions") for _d, p in uploads)
    exp_bytes = n_expected * ladder.DIRECTION_PT_BYTES_EST + ladder.LADDER_REPORT_BYTES_EST
    assert headrooms == [(n_expected + 1, exp_bytes)]
    assert verifies == [n_expected]
    # report carries the regime identity the steer/judge fingerprint folds in
    assert report["rb_rev"] == i2254.HF_REV
    assert report["slugs"] == list(ladder.LADDER_SLUGS)
    # idempotent re-entry (r1 concern ladder-directions-phase-no-skip; r2
    # blocker ladder-directions-upload-verify-bypass): a second invocation
    # skips the SVD recompute (no new SVD-side headroom preflight) but MUST
    # re-run the idempotent uploads + exact-set remote verification before
    # re-writing the sentinel — local completeness never certifies remote
    # durability.
    n_up, n_hr, n_vf = len(uploads), len(headrooms), len(verifies)
    ladder.phase_directions(args)
    assert len(headrooms) == n_hr, "re-entry must not re-run the SVD-side compute preflight"
    assert len(uploads) == n_up + 2, "re-entry must re-run BOTH idempotent uploads"
    assert len(verifies) == n_vf + 1, "re-entry must re-run remote verification"
    assert verifies[-1] == n_expected


def test_directions_reentry_after_crash_before_upload_verifies(tmp_path, monkeypatch):
    """Crash-window regression (r2 blocker ladder-directions-upload-verify-
    bypass): run 1 completes LOCALLY but the remote verifier RAISES (the
    crash window between the local report write and a durable upload) ⇒ no
    done sentinel; the plan-registered re-entry MUST re-run upload +
    verification, and only a PASSING verify writes status=done."""
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "logs"))
    maps_dir = tmp_path / "maps"
    rb = ladder.make_fixture_maps(maps_dir, layers=ladder.FIXTURE_LAYERS)
    ladder.make_fixture_parent_bank(tmp_path, maps_dir, rb)
    monkeypatch.setattr(ladder, "_RB_LOADER", lambda: rb)
    monkeypatch.setattr(ladder, "_PARENT_VEC_LOADER", ladder._fixture_parent_vec_loader)
    uploads: list[str] = []
    monkeypatch.setattr(ladder, "_UPLOAD", lambda d, p, allow=None: uploads.append(p))
    monkeypatch.setattr(ladder, "_DIR_HEADROOM", lambda n, b: None)
    verify_layers: list[list[int]] = []

    def failing_verify(layers):
        verify_layers.append(sorted(int(x) for x in layers))
        raise ladder.LadderHaltError("upload verification FAIL (simulated crash window)")

    monkeypatch.setattr(ladder, "_UPLOAD_VERIFY", failing_verify)
    args = argparse.Namespace(
        out_root=str(tmp_path),
        maps_dir=str(maps_dir),
        layers=list(ladder.FIXTURE_LAYERS),
        behaviors=list(ladder.ROUND_BEHAVIORS),
        smoke=False,
        fit_workers=2,
        force=False,
    )
    with pytest.raises(ladder.LadderHaltError, match="verification FAIL"):
        ladder.phase_directions(args)
    sent = tmp_path / "logs" / f"issue-{i2254.ISSUE}-{ladder.SENTINEL_DIRECTIONS}.json"
    assert not sent.exists(), "a failed verify must never leave a done sentinel"
    # the local run IS complete on disk (skip predicate True) — yet re-entry
    # must STILL upload + verify, and still refuse while the verify fails
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    assert ladder._directions_done(args, rroot, list(ladder.FIXTURE_LAYERS), ladder.ROUND_BEHAVIORS)
    n_up = len(uploads)
    with pytest.raises(ladder.LadderHaltError, match="verification FAIL"):
        ladder.phase_directions(args)
    assert len(verify_layers) == 2, "re-entry must call remote verification"
    assert len(uploads) == n_up + 2, "re-entry must re-run BOTH idempotent uploads"
    assert not sent.exists()
    # verify now passes ⇒ the re-entry path writes the done sentinel (with
    # the skip flag), upload/verify having run FIRST
    ok_layers: list[list[int]] = []
    monkeypatch.setattr(
        ladder, "_UPLOAD_VERIFY", lambda layers: ok_layers.append(sorted(map(int, layers)))
    )
    ladder.phase_directions(args)
    assert ok_layers == [sorted(ladder.FIXTURE_LAYERS)]
    payload = json.loads(sent.read_text())
    assert payload["status"] == "done"
    assert payload["skipped_prior_complete"] is True


# ---------------------------------------------------------------------------
# _verify_directions_upload — PRODUCTION body (r2 concern
# ladder-directions-upload-not-exact): exact-set on the grammar-filtered
# scope; fake ONLY the fk._hub_tree network boundary (signature-conformant)
# ---------------------------------------------------------------------------


class _TreeEntry:
    def __init__(self, path: str):
        self.path = path


def _fake_hub_tree_factory(bank_names: set[str], round_names: set[str]):
    """Signature mirror of ``fk._hub_tree`` serving the bank prefix vs the
    round prefix from two fixed name sets."""

    def fake_hub_tree(prefix: str, *, recursive: bool = False) -> list:
        names = round_names if prefix == ladder._round_hf_prefix() else bank_names
        return [_TreeEntry(f"{prefix}/{n}") for n in sorted(names)]

    return fake_hub_tree


def test_verify_directions_upload_exact_set_production_body(monkeypatch):
    layers = [14, 17]
    expected = ladder._registered_direction_names(layers)
    assert len(expected) == 16
    parent_files = {"evil_pre_L14.pt", "sycophancy_ctxext_L17.pt", "directions_manifest.json"}
    ok_round = {"ladder_report.json"}
    # exact set + legitimate parent-prefix files outside the grammar ⇒ PASS
    monkeypatch.setattr(fk, "_hub_tree", _fake_hub_tree_factory(expected | parent_files, ok_round))
    ladder._verify_directions_upload(layers)
    # one expected ladder name missing remotely ⇒ FAIL
    short = set(sorted(expected)[1:])
    monkeypatch.setattr(fk, "_hub_tree", _fake_hub_tree_factory(short | parent_files, ok_round))
    with pytest.raises(ladder.LadderHaltError, match="absent"):
        ladder._verify_directions_upload(layers)
    # an EXTRA ladder-grammar name (stale foreign layer) ⇒ FAIL
    monkeypatch.setattr(
        fk,
        "_hub_tree",
        _fake_hub_tree_factory(expected | {"evil_tr_L3.pt"} | parent_files, ok_round),
    )
    with pytest.raises(ladder.LadderHaltError, match="unexpected ladder-grammar"):
        ladder._verify_directions_upload(layers)
    # ladder_report.json absent at the round prefix ⇒ FAIL
    monkeypatch.setattr(fk, "_hub_tree", _fake_hub_tree_factory(expected, set()))
    with pytest.raises(ladder.LadderHaltError, match=r"ladder_report\.json absent"):
        ladder._verify_directions_upload(layers)


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

    # Undefined-cell rule (coherence-GATED, r1 blocker
    # ladder-coherence-gate-omitted): the coherence-failed cell has FINITE
    # judge scores yet is OUTSIDE H1 support, the bounded-non-clear narration,
    # and the selection-aware sets
    assert verdicts["narration"]["undefined_cells"] == ["evil__tr__ctx__all__c0p5"]
    und = percell["behaviors"]["evil"]["evil__tr__ctx__all__c0p5"]
    assert und["label"] == "Undefined (no valid measurement)"
    assert und["undefined_reason"] == "coherence gate failed"
    assert und["margin_lo"] is None
    assert und["sensitivity"]["n_valid_judge_rows"] == 20  # finite scores, still Undefined
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


# ---------------------------------------------------------------------------
# r6 fix round — coherence gate on ALL verdict-bearing sets (r1 blocker
# ladder-coherence-gate-omitted)
# ---------------------------------------------------------------------------


def test_reduce_all_coherence_failed_is_measurement_failure(tmp_path, monkeypatch):
    """Finite judge scores + coherence_pass=False on EVERY cell ⇒ all cells
    Undefined, no H1/H2 label, every verdict-bearing set empty."""
    args = argparse.Namespace(out_root=str(tmp_path), smoke=False)
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    ladder.make_fixture_round(rroot, args, coherence_fail_all=True)
    monkeypatch.setattr(ladder, "_TOKENIZER_LOADER", ladder._FixtureTokenizer)
    ladder.phase_reduce(args)
    verdicts = json.loads((rroot / "reduce" / "verdicts.json").read_text())
    assert verdicts["label"].startswith("Undefined (measurement failure")
    assert verdicts["label"] not in ("H1", "H2")
    assert verdicts["n_clearing"] == 0 and verdicts["h1_clearing_cells"] == []
    assert len(verdicts["narration"]["undefined_cells"]) == ladder.FAMILY_SIZE
    assert verdicts["narration"]["bounded_nonclear_cells"] == []
    assert verdicts["narration"]["straddling_cells"] == []
    assert verdicts["all44_companion"] is None
    assert all(v is None for v in verdicts["selection_aware"]["arm"].values())
    percell = json.loads((rroot / "reduce" / "delta_score_percell.json").read_text())
    for cells_b in percell["behaviors"].values():
        for row in cells_b.values():
            assert row["label"] == "Undefined (no valid measurement)"
            assert row["undefined_reason"] == "coherence gate failed"
            assert row["margin_lo"] is None


# ---------------------------------------------------------------------------
# r6 fix round — rule-29 completeness ENFORCED at both entry points (r1
# blocker ladder-completeness-not-enforced)
# ---------------------------------------------------------------------------


def test_judge_completeness_enforcement_withholds_wave_done(tmp_path):
    rroot = tmp_path / "round"
    jd = rroot / "judge" / "judged"
    jd.mkdir(parents=True)
    for cid, fic in (("cell_a", 1.0), ("cell_b", 0.94)):
        (jd / f"{cid}.json").write_text(
            json.dumps({"cell_id": cid, "accounting": {"frac_items_complete": fic}})
        )
    with pytest.raises(RuntimeError, match="completeness floor"):
        ladder._enforce_judge_completeness(rroot)
    block = json.loads((rroot / "judge" / "completeness.json").read_text())
    assert block["below_floor_cells"] == ["cell_b"]
    assert not (rroot / "judge" / "wave_done.json").exists()
    # remediated set passes
    (jd / "cell_b.json").write_text(
        json.dumps({"cell_id": "cell_b", "accounting": {"frac_items_complete": 0.96}})
    )
    assert ladder._enforce_judge_completeness(rroot)["below_floor_cells"] == []


# r2 concern ladder-completeness-not-enforced: a FINITE value is REQUIRED at
# both rule-29 entry points — None / non-finite is a MISSING measurement,
# treated as below-floor (raise), never a pass.
_COMPLETENESS_CASES = [
    (None, False),
    (float("nan"), False),
    (0.94, False),
    (0.95, True),
    (0.96, True),
]


@pytest.mark.parametrize("fic,ok", _COMPLETENESS_CASES)
def test_judge_completeness_requires_finite_value(tmp_path, fic, ok):
    rroot = tmp_path / "round"
    jd = rroot / "judge" / "judged"
    jd.mkdir(parents=True)
    (jd / "cell_x.json").write_text(
        json.dumps({"cell_id": "cell_x", "accounting": {"frac_items_complete": fic}})
    )
    if ok:
        block = ladder._enforce_judge_completeness(rroot)
        assert block["below_floor_cells"] == []
        assert block["non_finite_cells"] == []
    else:
        with pytest.raises(RuntimeError, match="completeness floor"):
            ladder._enforce_judge_completeness(rroot)
        block = json.loads((rroot / "judge" / "completeness.json").read_text())
        assert block["below_floor_cells"] == ["cell_x"]
        assert block["non_finite_cells"] == ([] if fic == 0.94 else ["cell_x"])
        assert not (rroot / "judge" / "wave_done.json").exists()


@pytest.mark.parametrize("fic,ok", _COMPLETENESS_CASES)
def test_reduce_completeness_requires_finite_value(fic, ok):
    """The reduce-entry check (``_require_finite_completeness`` — the exact
    predicate ``phase_reduce`` calls per cell; wiring pinned by the
    integration test below) over the same five values."""
    if ok:
        assert ladder._require_finite_completeness("cell_x", fic) == fic
    else:
        with pytest.raises(RuntimeError, match="rule-29 below-floor"):
            ladder._require_finite_completeness("cell_x", fic)


@pytest.mark.parametrize("fic", [0.94, None, float("nan")])
def test_reduce_refuses_below_floor_completeness(tmp_path, monkeypatch, fic):
    args = argparse.Namespace(out_root=str(tmp_path), smoke=False)
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    ladder.make_fixture_round(rroot, args)
    target = rroot / "judge" / "judged" / "evil__tr__ctx__L14__c4.json"
    j = json.loads(target.read_text())
    j["accounting"]["frac_items_complete"] = fic
    target.write_text(json.dumps(j))
    monkeypatch.setattr(ladder, "_TOKENIZER_LOADER", ladder._FixtureTokenizer)
    with pytest.raises(RuntimeError, match="rule-29 below-floor"):
        ladder.phase_reduce(args)


# ---------------------------------------------------------------------------
# r6 fix round — steer input contract BEFORE model init (r1 blocker
# ladder-steer-precondition-post-model)
# ---------------------------------------------------------------------------


def _steer_args(tmp_path):
    return argparse.Namespace(
        out_root=str(tmp_path),
        smoke=False,
        shard_id=0,
        num_shards=1,
        q_steer=20,
        draws=5,
        force=False,
    )


def _valid_ladder_report() -> dict:
    """A report satisfying EVERY registered identity/gate check (the r2
    steer-precondition contract), so each test case below can break exactly
    one thing."""
    parity = {
        f"{b}__L{ly}": 0.9999999 for b in ladder.ROUND_BEHAVIORS for ly in ladder.PARITY_LAYERS
    }
    arms = {
        f"{b}__{slug}": {"residual": 0.0, "loader_roundtrip_cos": 1.0}
        for b in ladder.ROUND_BEHAVIORS
        for slug in ladder.LADDER_SLUGS
    }
    return {
        "rb_rev": i2254.HF_REV,
        "slugs": list(ladder.LADDER_SLUGS),
        "behaviors": list(ladder.ROUND_BEHAVIORS),
        "n_direction_files": len(arms) * i2254.N_LAYERS,
        "gates": {"rebuild_parity_cos": parity},
        "layers": {
            str(ly): {"lambdas": {}, "arms": {k: dict(v) for k, v in arms.items()}}
            for ly in range(i2254.N_LAYERS)
        },
    }


def test_steer_preconditions_run_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "logs"))

    def fake_require_cuda(phase: str) -> None:
        return None

    def fake_headroom(out_root, need_gb, phase) -> None:
        return None

    def fake_stage_e1() -> dict:
        return {}

    monkeypatch.setattr(i2254, "_require_cuda", fake_require_cuda)
    monkeypatch.setattr(i2254, "_assert_phase_headroom", fake_headroom)
    monkeypatch.setattr(i2254, "_stage_e1_assets", fake_stage_e1)
    model_loader = create_autospec(i2254._load_model_and_tokenizer)
    monkeypatch.setattr(i2254, "_load_model_and_tokenizer", model_loader)
    hub_headroom = create_autospec(fk._assert_hub_headroom_for_steer)
    monkeypatch.setattr(fk, "_assert_hub_headroom_for_steer", hub_headroom)
    monkeypatch.setattr(
        fk, "_hub_stage", create_autospec(fk._hub_stage, side_effect=RuntimeError("no HF copy"))
    )
    args = _steer_args(tmp_path)
    # (A) no directions sentinel ⇒ refuse BEFORE any hub/model spend
    with pytest.raises(ladder.LadderHaltError, match="directions sentinel"):
        ladder.phase_steer(args)
    assert model_loader.call_count == 0
    assert hub_headroom.call_count == 0
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"issue-{i2254.ISSUE}-{ladder.SENTINEL_DIRECTIONS}.json").write_text(
        json.dumps({"status": "done"})
    )
    rroot = tmp_path / ladder.FOLLOWUP_LABEL
    rroot.mkdir(parents=True, exist_ok=True)
    report_path = rroot / "ladder_report.json"

    def expect_refusal(report: dict, pattern: str) -> None:
        report_path.write_text(json.dumps(report))
        with pytest.raises(ladder.LadderHaltError, match=pattern):
            ladder.phase_steer(args)
        assert model_loader.call_count == 0, pattern

    # (B) fully valid report identity, but the direction file set is absent
    # on disk ⇒ refuse, still zero model-loader calls
    expect_refusal(_valid_ladder_report(), "direction file")
    # (C) report-identity checks (r2 concern
    # ladder-steer-precondition-post-model): stale rb_rev
    stale = _valid_ladder_report()
    stale["rb_rev"] = "deadbeefdeadbeef"
    expect_refusal(stale, "rb_rev")
    # (D) EMPTY parity map — existing files must not certify a report that
    # carries no gate-(ii) evidence
    empty_parity = _valid_ladder_report()
    empty_parity["gates"]["rebuild_parity_cos"] = {}
    expect_refusal(empty_parity, "parity cells")
    # (E) partial parity map (one registered cell missing) ⇒ refuse
    partial_parity = _valid_ladder_report()
    del partial_parity["gates"]["rebuild_parity_cos"]["sycophancy__L17"]
    expect_refusal(partial_parity, "parity cells")
    # (F) wrong slug set ⇒ refuse
    wrong_slug = _valid_ladder_report()
    wrong_slug["slugs"] = ["tr", "rl1", "rl2", "WRONG"]
    expect_refusal(wrong_slug, "slugs")
    # (G) missing behaviors key (foreign/legacy report) ⇒ refuse
    no_behaviors = _valid_ladder_report()
    del no_behaviors["behaviors"]
    expect_refusal(no_behaviors, "behaviors")
    # (H) a failing gate-(iii) residual row ⇒ refuse
    bad_row = _valid_ladder_report()
    bad_row["layers"]["14"]["arms"]["evil__tr"]["loader_roundtrip_cos"] = 0.5
    expect_refusal(bad_row, "gate row failed")


# ---------------------------------------------------------------------------
# r6 fix round — regime fingerprint invalidates on EVERY generating parameter
# (r1 blocker ladder-regime-fp-incomplete)
# ---------------------------------------------------------------------------


def test_regime_fp_invalidates_on_every_generating_parameter():
    cell = {
        "behavior": "evil",
        "kind": "steer",
        "direction": "tr",
        "position": "context",
        "layer_config": "L14",
        "c": 4.0,
    }
    rho = {"L14": 1.25}
    base_args = argparse.Namespace(draws=5, q_steer=20)
    fps = [ladder._ladder_regime_fp(base_args, cell, rho, "dfp0")]
    fps.append(ladder._ladder_regime_fp(argparse.Namespace(draws=6, q_steer=20), cell, rho, "dfp0"))
    fps.append(ladder._ladder_regime_fp(argparse.Namespace(draws=5, q_steer=10), cell, rho, "dfp0"))
    fps.append(ladder._ladder_regime_fp(base_args, {**cell, "c": 2.0}, rho, "dfp0"))
    fps.append(ladder._ladder_regime_fp(base_args, cell, {"L14": 1.5}, "dfp0"))
    fps.append(ladder._ladder_regime_fp(base_args, cell, rho, "dfp1"))  # direction identity
    for attr, val in (
        ("GEN_MAX_NEW_TOKENS", 4096),
        ("CAP_HIT_REGEN_FRAC", 0.05),
        ("CAP_HIT_REGEN_FACTOR", 3),
        ("MODEL_NAME", "other/model"),
        ("HF_REV", "deadbeefdeadbeef"),
    ):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(i2254, attr, val)
            fps.append(ladder._ladder_regime_fp(base_args, cell, rho, "dfp0"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ladder, "LADDER_SEEDS", (42,))
        fps.append(ladder._ladder_regime_fp(base_args, cell, rho, "dfp0"))
    assert len(set(fps)) == len(fps), "every generating parameter must invalidate the fp"


# ---------------------------------------------------------------------------
# r6 fix round — §7(c) realized-wall fleet stop + §7(d) rig-health stop (r1
# concern ladder-runtime-stops-missing)
# ---------------------------------------------------------------------------


def test_wall_halt_check_and_reentry_guard(tmp_path):
    rroot = tmp_path / "round"
    comp = rroot / "steer" / "raw_completions"
    comp.mkdir(parents=True)
    basis = 1.5
    for i in range(7):
        (comp / f"c{i}.json").write_text("{}")
    # below the first-8-completed-cells arming threshold: never trips
    ladder._check_realized_wall(rroot, comp, 1000.0, 10, basis)
    (comp / "c7.json").write_text("{}")
    # armed, realized 1.0 s/completion <= 2x1.5: no stop
    ladder._check_realized_wall(rroot, comp, 10.0, 10, basis)
    # armed, realized 4.0 > 3.0: HALT file + raise
    with pytest.raises(ladder.LadderHaltError, match="fleet HALT"):
        ladder._check_realized_wall(rroot, comp, 40.0, 10, basis)
    assert ladder._wall_halt_path(rroot).is_file()
    payload = json.loads(ladder._wall_halt_path(rroot).read_text())
    assert payload["realized_s_per_completion"] == pytest.approx(4.0)
    # re-entry guard: a standing halt file refuses the phase until deleted
    with pytest.raises(ladder.LadderHaltError, match="wall-halt file present"):
        ladder._assert_no_wall_halt(rroot)
    ladder._wall_halt_path(rroot).unlink()
    ladder._assert_no_wall_halt(rroot)


def test_steer_wall_basis_reads_committed_pilot():
    assert ladder._load_steer_wall_basis() == pytest.approx(1.538939881324768)


def test_rig_health_verdict_on_floor():
    band = ladder.load_parent_band("sycophancy")
    delta, on_floor = ladder.rig_health_verdict(60.0, 20.0, band)
    assert delta == pytest.approx(40.0) and on_floor is False
    _delta2, on_floor2 = ladder.rig_health_verdict(25.0, 20.0, band)
    assert on_floor2 is True  # Δ=5 <= band 10.89 ⇒ §7(d) diagnostic stop


# ---------------------------------------------------------------------------
# r6 fix round — intrusion flags keyed (seed, ci, di) (r1 Major: seed 43
# overwrote seed 42 on every 2-seed production cell)
# ---------------------------------------------------------------------------


def _two_seed_gen_record() -> dict:
    def seed_block(texts):
        return {
            "completions": [[t] for t in texts],
            "coherent_flags": [[True] for _ in texts],
            "condition_passes": [True for _ in texts],
        }

    return {
        "cell_id": "twoseed",
        "cell": {"behavior": "sycophancy"},
        "q_of_context": [0, 1],
        "seeds": {
            "42": seed_block(["intruded 好 answer", "clean answer"]),
            "43": seed_block(["clean answer", "clean answer"]),
        },
    }


def test_intrusion_flags_and_sensitivity_two_seeds():
    import re

    rx = re.compile(
        json.loads((ladder.INPUTS_ROOT / "decisive" / "cjk_audit.json").read_text())["regex"]
    )
    tok = ladder._FixtureTokenizer()
    rec = _two_seed_gen_record()
    flags = ladder._intrusion_flags(rec, rx, tok)
    assert len(flags) == 4, "2 seeds x 2 ci x 1 draw — no cross-seed key collision"
    assert flags[(42, 0, 0)] is True
    assert flags[(43, 0, 0)] is False
    assert flags[(42, 1, 0)] is False and flags[(43, 1, 0)] is False
    # sensitivity attributes each seed's flag to ITS OWN judged rows
    items, merged = {}, {}
    for rid, seed, qi, sc in (
        ("r42-0", 42, 0, 40.0),
        ("r43-0", 43, 0, 60.0),
        ("r42-1", 42, 1, 70.0),
        ("r43-1", 43, 1, 70.0),
    ):
        items[rid] = {"qi": qi, "seed": seed, "ci": qi, "di": 0}
        merged[rid] = [sc]
    judged = {
        "cell_id": "twoseed",
        "items": items,
        "per_item_scores_merged": merged,
        "mean_score": 60.0,  # ((40+60)/2 + (70+70)/2) / 2
    }
    sens = ladder._intrusion_sensitivity(judged, rec, rx, tok, floor_mean=10.0)
    assert sens["n_valid_judge_rows"] == 4  # BOTH seeds' rows (pre-fix: 2)
    assert sens["n_intruded_valid"] == 1  # only seed 42's ci=0 row
    assert sens["cjk_common"] == pytest.approx(0.25)
    # zeroing ONLY the intruded seed-42 row: q0 -> (0+60)/2=30, q1 -> 70
    assert sens["mean_zeroed_intrusion"] == pytest.approx(50.0)
    assert sens["mean_excluded_intrusion"] == pytest.approx((60.0 + 70.0) / 2)


# ---------------------------------------------------------------------------
# r6 fix round — per-behavior independent bootstrap matrices in the
# selection-aware companions (r1 sweep ladder-cross-behavior-bootstrap-coupling)
# ---------------------------------------------------------------------------


def test_selection_aware_per_behavior_independent_matrices():
    nq = 20
    rng = np.random.default_rng(7)
    floors = {
        "evil": (np.zeros(nq), 0.0, 0.0),
        "sycophancy": (np.zeros(nq), 0.0, 0.0),
    }
    bands = {"evil": 0.0, "sycophancy": 0.0}
    cq_e = rng.normal(10.0, 2.0, nq)
    cq_s = rng.normal(-100.0, 1.0, nq)  # never wins a per-draw max
    entries = [("e1", "evil", cq_e), ("s1", "sycophancy", cq_s)]
    margins = {"e1": float(np.mean(cq_e)), "s1": float(np.mean(cq_s))}
    res = ladder._selection_aware_block(entries, floors, bands, "k", margins)
    assert res["argmax_cell"] == "e1"
    # the CI must come from the EVIL-KEYED matrix (independent per behavior),
    # not a shared cross-behavior matrix (the pre-fix coupling)
    idx_e = i2254._boot_idx(nq, i2254.N_BOOT_VERDICT, "k__evil")
    exp = ladder._boot_diffs(cq_e, floors["evil"][0], idx_e) - bands["evil"]
    assert res["ci"] == [
        float(np.nanquantile(exp, 0.025)),
        float(np.nanquantile(exp, 0.975)),
    ]


# ---------------------------------------------------------------------------
# r6 fix round — the three now-unconditional gates hold under smoke=True (r2
# concern ladder-smoke-blind-spots: the r1 downgrades were removed; these
# pins keep them from silently regressing to smoke-conditional)
# ---------------------------------------------------------------------------


def test_directions_parity_layer_refusal_under_smoke(tmp_path, monkeypatch):
    """smoke=True: a layer set missing a registered parity layer refuses
    BEFORE any SVD compute — the coverage gate is unconditional."""
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "logs"))
    maps_dir = tmp_path / "maps"
    rb = ladder.make_fixture_maps(maps_dir, layers=(14,))
    ladder.make_fixture_parent_bank(tmp_path, maps_dir, rb, layers=(14,))
    monkeypatch.setattr(ladder, "_RB_LOADER", lambda: rb)
    monkeypatch.setattr(ladder, "_PARENT_VEC_LOADER", ladder._fixture_parent_vec_loader)
    args = argparse.Namespace(
        out_root=str(tmp_path),
        maps_dir=str(maps_dir),
        layers=[14],  # parity layer 17 missing
        behaviors=list(ladder.ROUND_BEHAVIORS),
        smoke=True,
        fit_workers=1,
        force=False,
    )
    with pytest.raises(RuntimeError, match="parity layers"):
        ladder.phase_directions(args)
    sent = tmp_path / "logs" / f"issue-{i2254.ISSUE}-{ladder.SENTINEL_DIRECTIONS}.json"
    assert not sent.exists()


def test_reduce_grain_refusal_under_smoke(tmp_path, monkeypatch):
    """smoke=True: a judged cell whose per-question grain is shorter than the
    committed parent floor refuses — no smoke-mode floor truncation."""
    args = argparse.Namespace(out_root=str(tmp_path), smoke=True)
    rroot = ladder.round_root(i2254._out_root(args))
    ladder.make_fixture_round(rroot, args)  # smoke ⇒ the single registered smoke cell
    cid = "evil__tr__ctx__L14__c4"
    target = rroot / "judge" / "judged" / f"{cid}.json"
    j = json.loads(target.read_text())
    for k in ("per_question_mean_score", "per_question_rate", "per_question_n"):
        j[k] = j[k][:2]
    j["n_questions"] = 2
    target.write_text(json.dumps(j))
    monkeypatch.setattr(ladder, "_TOKENIZER_LOADER", ladder._FixtureTokenizer)
    with pytest.raises(RuntimeError, match="refused in every mode"):
        ladder.phase_reduce(args)


def test_figures_required_gate_under_smoke(tmp_path, monkeypatch):
    """smoke=True: the required-figure (hero) gate fails loud when the reduce
    outputs are absent — no smoke-mode ``require=()`` downgrade."""
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "logs"))
    args = argparse.Namespace(out_root=str(tmp_path), smoke=True, fig_dir=str(tmp_path / "figs"))
    with pytest.raises(RuntimeError, match="required figures not rendered"):
        ladder.phase_figures(args)
    sent = tmp_path / "logs" / f"issue-{i2254.ISSUE}-{ladder.SENTINEL_FIGURES}.json"
    assert not sent.exists()


# ---------------------------------------------------------------------------
# r6 fix round — --import-check binds the §4.3 reuse ledger (r1 NIT
# ladder-report-command-drift)
# ---------------------------------------------------------------------------


def test_import_check_binds_reuse_ledger(capsys):
    ladder._bind_reuse_ledger()
    assert "helper call shapes bound OK" in capsys.readouterr().out


def test_pilot_draws_smoke_clears_verdict_floor():
    """#2329 class: smoke pilot draws must make the rule-26 verdict floor realizable.

    8 items/arm x pilot draws >= JUDGE_PILOT_MIN_EFFECTIVE (51); production passthrough.
    """
    import scripts.issue2254_preimage as i2254
    from scripts.issue2254_transpose_ladder import _SMOKE_PILOT_ITEMS_PER_ARM, _pilot_draws

    for nd in (2, 5):
        pd = _pilot_draws(True, nd)
        assert _SMOKE_PILOT_ITEMS_PER_ARM * pd >= i2254.JUDGE_PILOT_MIN_EFFECTIVE, (nd, pd)
        assert pd >= nd
    assert _pilot_draws(False, 5) == 5
    assert _pilot_draws(False, 2) == 2
