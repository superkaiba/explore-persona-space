"""CPU tests for the issue-1415 round-B analysis scripts (deliverable 6).

Covers: null battery (batched shapes, selection-symmetric max rule, seed
determinism, matrices-persisted-BEFORE-aggregation), map transport (synthetic
linear-map identity fixture -> cosine 1.0; rc=3 on a keys-missing bundle),
judge script (request building + content-drop vs transport-loss split through
the REAL ``graded_judge`` reduce — only the Batch-API boundary
``batch_judge.judge_completions_batch`` is replaced, with a
signature-conformant ``create_autospec`` fake), and logit lens (tiny
from-config Qwen2, shape asserts). Everything runs on CPU with tiny dims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1415_analysis_common as common  # noqa: E402
import issue1415_judge as jg  # noqa: E402
import issue1415_logit_lens as ll  # noqa: E402
import issue1415_map_transport as mt  # noqa: E402
import issue1415_null_battery as nb  # noqa: E402

HID = 16
LAYERS = [0, 1, 2]


def _write_captures(dirpath: Path, n_pairs: int = 8, hidden: int = HID, seed: int = 0):
    """Synthetic phase-1a capture files matching the driver's .pt contract."""
    dirpath.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    n_layers = len(LAYERS)

    def rec():
        return {
            "v_c_prefix": torch.randn(n_layers, hidden, generator=gen),
            "v_c_context": torch.randn(n_layers, hidden, generator=gen),
            "v_a_mean": torch.randn(n_layers, hidden, generator=gen),
        }

    ids = []
    for i in range(n_pairs):
        pid = f"p{i:02d}"
        torch.save(
            {"pair_id": pid, "layers": list(LAYERS), "c": rec(), "cprime": rec()},
            dirpath / f"{pid}.pt",
        )
        ids.append(pid)
    return ids


# ── deliverable 1: null battery ───────────────────────────────────────


def _run_battery(tmp: Path, tag: str, seeds: tuple[int, int] = (1415, 1416)) -> tuple[Path, Path]:
    act = tmp / "acts"
    if not act.exists():
        _write_captures(act)
    mats = tmp / f"mats_{tag}"
    out = tmp / f"null_bands_{tag}.json"
    nb.main(
        [
            "--activations",
            str(act),
            "--matrices-dir",
            str(mats),
            "--out-json",
            str(out),
            "--n-draws",
            "50",
            "--seed-random",
            str(seeds[0]),
            "--seed-shuffled",
            str(seeds[1]),
            "--upload",
            "none",
        ]
    )
    return mats, out


def test_null_battery_shapes_and_selection_rule(tmp_path):
    mats, out = _run_battery(tmp_path, "a")
    for battery in ("random_delta", "shuffled_pair"):
        blob = torch.load(mats / f"{battery}_null_matrix.pt", weights_only=True)
        per_layer, selected = blob["per_layer"], blob["selected"]
        assert per_layer.shape == (50, len(LAYERS), 2, 8)
        assert selected.shape == (50, 2, 8)
        # selection symmetry: the persisted selection IS max-over-layers per draw
        assert torch.equal(selected, per_layer.max(dim=1).values)
        # observed statistic uses the identical rule
        assert torch.equal(blob["observed_selected"], blob["observed_per_layer"].max(dim=0).values)
    j = json.loads(out.read_text())
    assert j["arms"] == ["prefix", "context"]
    assert len(j["pair_ids"]) == 8
    # summary quantiles are recomputable from the persisted (pre-aggregation) matrix
    sel = torch.load(mats / "random_delta_null_matrix.pt", weights_only=True)["selected"]
    band = j["bands"]["random_delta"]["per_pair"]["prefix"]["p00"]
    expect = float(torch.quantile(sel[:, 0, 0].float(), torch.tensor([0.975]))[0])
    assert band["p97.5"] == pytest.approx(expect, rel=1e-6)


def test_null_battery_deterministic_under_seeds(tmp_path):
    mats_a, _ = _run_battery(tmp_path, "a")
    mats_b, _ = _run_battery(tmp_path, "b")
    mats_c, _ = _run_battery(tmp_path, "c", seeds=(7, 8))
    for battery in ("random_delta", "shuffled_pair"):
        a = torch.load(mats_a / f"{battery}_null_matrix.pt", weights_only=True)
        b = torch.load(mats_b / f"{battery}_null_matrix.pt", weights_only=True)
        c = torch.load(mats_c / f"{battery}_null_matrix.pt", weights_only=True)
        assert torch.equal(a["per_layer"], b["per_layer"])
        assert not torch.equal(a["per_layer"], c["per_layer"])


def test_null_battery_matrices_persisted_before_aggregation(tmp_path, monkeypatch):
    act = tmp_path / "acts"
    _write_captures(act)

    def boom(*a, **kw):
        raise RuntimeError("aggregation deliberately failed")

    monkeypatch.setattr(nb, "aggregate_bands", boom)
    with pytest.raises(RuntimeError, match="deliberately"):
        nb.main(
            [
                "--activations",
                str(act),
                "--matrices-dir",
                str(tmp_path / "mats"),
                "--out-json",
                str(tmp_path / "bands.json"),
                "--n-draws",
                "20",
                "--upload",
                "none",
            ]
        )
    # the per-draw matrices landed BEFORE the (crashed) aggregation
    assert (tmp_path / "mats" / "random_delta_null_matrix.pt").exists()
    assert (tmp_path / "mats" / "shuffled_pair_null_matrix.pt").exists()
    assert not (tmp_path / "bands.json").exists()


# ── deliverable 2: map transport ──────────────────────────────────────


def test_map_transport_identity_cosine(tmp_path):
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=4)
    gen = torch.Generator().manual_seed(3)
    weight = torch.randn(HID, HID, generator=gen)
    map_path = tmp_path / "map.pt"
    torch.save({"layer_1": {"weight": weight}}, map_path)

    # steered captures built so realized == predicted at layer 1
    steer = tmp_path / "steered"
    steer.mkdir()
    pairs = common.load_all_pairs(act)
    li = pairs[0].layers.index(1)
    for p in pairs:
        for arm in common.ARMS:
            fpred = (p.v_c[arm][li] + p.delta[arm][li]) @ weight.T - p.v_c[arm][li] @ weight.T
            torch.save({"v_a_mean": p.v_a_c[li] + fpred}, steer / f"{p.pair_id}__{arm}.pt")

    out = tmp_path / "mt.json"
    mt.main(
        [
            "--activations",
            str(act),
            "--map-path",
            str(map_path),
            "--steered-activations",
            str(steer),
            "--out-json",
            str(out),
            "--layer",
            "1",
            "--hidden",
            str(HID),
            "--n-draws",
            "20",
        ]
    )
    j = json.loads(out.read_text())
    assert j["realized_source"] == "steered"
    assert j["map_provenance"]["resolved_weight_key"] == "layer_1.weight"
    for arm in common.ARMS:
        cosines = j["per_arm"][arm]["transport_cosine"]
        assert len(cosines) == 4
        for pid, c in cosines.items():
            assert c == pytest.approx(1.0, abs=1e-4), (arm, pid, c)
        # null bands present with the documented single-layer identity selection
        assert set(j["per_arm"][arm]["null_pooled"]) == {"p2.5", "p50", "p97.5"}


def test_map_transport_natural_source_needs_no_steered_files(tmp_path):
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=3)
    map_path = tmp_path / "map.pt"
    torch.save({"layer_1": {"weight": torch.eye(HID)}}, map_path)
    out = tmp_path / "mt.json"
    mt.main(
        [
            "--activations",
            str(act),
            "--map-path",
            str(map_path),
            "--realized-source",
            "natural",
            "--out-json",
            str(out),
            "--layer",
            "1",
            "--hidden",
            str(HID),
            "--n-draws",
            "10",
        ]
    )
    j = json.loads(out.read_text())
    assert j["realized_source"] == "natural"


def test_map_transport_rc3_on_missing_layer20_weight(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"metadata": "no weights here", "vec": torch.randn(HID)}, bad)
    with pytest.raises(SystemExit) as ei:
        mt.resolve_map(bad, HID, 1)
    assert ei.value.code == 3

    # ambiguity (two same-shape candidates, no narrowing token) is also rc=3
    ambig = tmp_path / "ambig.pt"
    torch.save({"aa": torch.randn(HID, HID), "bb": torch.randn(HID, HID)}, ambig)
    with pytest.raises(SystemExit) as ei:
        mt.resolve_map(ambig, HID, 1)
    assert ei.value.code == 3


# ── deliverable 2b: K3 ridge-refit fallback (round E) ─────────────────


def _write_refit_tensors(tmp_path, n: int = 64, hidden: int = HID, seed: int = 5, random_y=False):
    """Synthetic (N, L, H) V_c/V_a pair tensors in the #823 bare-tensor format.

    Y = X @ W_true.T + b_true (an exactly-linear map, recoverable by the
    ridge); random_y=True breaks the relation so the sanity check must fail.
    """
    gen = torch.Generator().manual_seed(seed)
    w_true = torch.randn(hidden, hidden, generator=gen) / hidden**0.5
    b_true = torch.randn(hidden, generator=gen)
    x = torch.randn(n, len(LAYERS), hidden, generator=gen)
    y = torch.randn(n, len(LAYERS), hidden, generator=gen) if random_y else x @ w_true.T + b_true
    vc_path = tmp_path / "refit_vc.pt"
    va_path = tmp_path / "refit_va.pt"
    torch.save(x, vc_path)
    torch.save(y, va_path)
    return vc_path, va_path, w_true


def _steered_from_map(act: Path, steer: Path, weight: torch.Tensor, layer: int) -> None:
    """Steered V_a captures built so realized delta == weight-predicted delta."""
    steer.mkdir(exist_ok=True)
    pairs = common.load_all_pairs(act)
    li = pairs[0].layers.index(layer)
    for p in pairs:
        for arm in common.ARMS:
            fpred = (p.v_c[arm][li] + p.delta[arm][li]) @ weight.T - p.v_c[arm][li] @ weight.T
            torch.save({"v_a_mean": p.v_a_c[li] + fpred}, steer / f"{p.pair_id}__{arm}.pt")


def test_map_transport_refit_fallback_recovers_linear_map(tmp_path):
    """--refit-fallback force: the fitted ridge recovers a known linear map
    (transport cosines ~1.0 against steered captures built from W_true), the
    valid-idx row filter applies, and the output records refit provenance."""
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=4)
    vc_path, va_path, w_true = _write_refit_tensors(tmp_path)
    valid_path = tmp_path / "valid_idx.json"
    valid_path.write_text(json.dumps({"common_valid_idx": list(range(60))}))  # drop last 4 rows
    _steered_from_map(act, tmp_path / "steered", w_true, layer=1)

    out = tmp_path / "mt_refit.json"
    mt.main(
        [
            "--activations",
            str(act),
            "--steered-activations",
            str(tmp_path / "steered"),
            "--out-json",
            str(out),
            "--layer",
            "1",
            "--hidden",
            str(HID),
            "--n-draws",
            "10",
            "--refit-fallback",
            "force",
            "--refit-vc-path",
            str(vc_path),
            "--refit-va-path",
            str(va_path),
            "--refit-valid-idx-path",
            str(valid_path),
        ]
    )
    j = json.loads(out.read_text())
    prov = j["map_provenance"]
    assert prov["source"] == "refit_fallback"
    assert prov["resolved_weight_key"] == "refit_ridge(v_c->v_a_prime)"
    ref = prov["refit"]
    assert ref["n_rows_total"] == 64 and ref["n_rows_kept"] == 60
    assert ref["r2_val"] > 0.99  # exactly-linear data: held-out sanity far above the 0.2 floor
    assert ref["n_train"] + ref["n_val"] == 60
    for arm in common.ARMS:
        for pid, c in j["per_arm"][arm]["transport_cosine"].items():
            assert c == pytest.approx(1.0, abs=1e-3), (arm, pid, c)


def test_map_transport_refit_sanity_fail_exits_rc6(tmp_path):
    """Y independent of X -> held-out R^2 < 0.2 -> SystemExit(rc=6), the
    distinct plan-K3 'H2 DV dropped' exit (never the rc=3 step-0 code)."""
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=3)
    vc_path, va_path, _ = _write_refit_tensors(tmp_path, random_y=True)
    with pytest.raises(SystemExit) as ei:
        mt.main(
            [
                "--activations",
                str(act),
                "--out-json",
                str(tmp_path / "unused.json"),
                "--layer",
                "1",
                "--hidden",
                str(HID),
                "--refit-fallback",
                "force",
                "--refit-vc-path",
                str(vc_path),
                "--refit-va-path",
                str(va_path),
            ]
        )
    assert ei.value.code == mt.RC_REFIT_SANITY_FAIL == 6
    assert not (tmp_path / "unused.json").exists()  # dropped DV writes no output


def test_map_transport_auto_refit_engages_on_keys_miss(tmp_path):
    """--refit-fallback auto (the default): a #922 bundle with NO (H, H)
    layer weight (the historical rc=3 condition) routes into the refit
    instead of exiting; --refit-fallback off preserves the rc=3 exit."""
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=4)
    bad = tmp_path / "bad.pt"
    torch.save({"metadata": "no weights here", "vec": torch.randn(HID)}, bad)
    vc_path, va_path, w_true = _write_refit_tensors(tmp_path)
    _steered_from_map(act, tmp_path / "steered", w_true, layer=1)

    common_args = [
        "--activations",
        str(act),
        "--steered-activations",
        str(tmp_path / "steered"),
        "--map-path",
        str(bad),
        "--layer",
        "1",
        "--hidden",
        str(HID),
        "--n-draws",
        "10",
        "--refit-vc-path",
        str(vc_path),
        "--refit-va-path",
        str(va_path),
    ]
    out = tmp_path / "mt_auto.json"
    mt.main([*common_args, "--out-json", str(out), "--refit-fallback", "auto"])
    j = json.loads(out.read_text())
    assert j["map_provenance"]["source"] == "refit_fallback"

    with pytest.raises(SystemExit) as ei:
        mt.main([*common_args, "--out-json", str(tmp_path / "x.json"), "--refit-fallback", "off"])
    assert ei.value.code == mt.RC_STEP0_FAIL == 3


# ── deliverable 3: judge wiring ───────────────────────────────────────


def _write_cell(out_root: Path, bulk_root: Path, meta: dict, draws: list[str]) -> None:
    cell_id = meta["cell_id"]
    mpath = out_root / "cells" / f"{cell_id}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(meta))
    cpath = bulk_root / "raw_completions" / f"{cell_id}.json"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({**meta, "draws": draws}))


def _judge_fixture(tmp: Path):
    out_root, bulk_root = tmp / "out", tmp / "bulk"
    ctx = {"system": None, "user": "What should I do about my rival?"}
    cells = [
        {
            "cell_id": "gen1b/pair0/c",
            "phase": "phase1b",
            "pair_id": "pair0",
            "context": ctx,
            "arm_label": "hf_nohook_base",
        },
        {
            "cell_id": "gen1b/pair0/cprime",
            "phase": "phase1b",
            "pair_id": "pair0",
            "context": ctx,
            "arm_label": "ctx_swap_ceil",
        },
        {
            "cell_id": "gen1c/context/pair0/L1/a1",
            "phase": "phase1c_grid",
            "pair_id": "pair0",
            "context": ctx,
            "layer": 1,
            "alpha": 1.0,
            "extraction_arm": "context",
        },
        {
            "cell_id": "gen1d_full/evil/pair0/a1",
            "phase": "phase1d_full",
            "pair_id": "pair0",
            "context": ctx,
            "layer": 1,
            "alpha": 1.0,
            "trait": "evil",
        },
        # excluded phase: must NOT be judged
        {
            "cell_id": "gen1c/context/pair0/L2/a1",
            "phase": "phase1c_layers",
            "pair_id": "pair0",
            "context": ctx,
            "layer": 2,
            "alpha": 1.0,
            "extraction_arm": "context",
        },
    ]
    for meta in cells:
        _write_cell(out_root, bulk_root, meta, ["answer one", "answer two"])
    bank = tmp / "pair_bank.json"
    bank.write_text(json.dumps({"pairs": [{"pair_id": "pair0", "trait_or_behavior": "evil"}]}))
    return out_root, bulk_root, bank


def _fake_scores(item_id: str, draw_idx: int) -> dict:
    if item_id == "gen1b/pair0/c/d0":
        return {"reasoning": "r", "score": 80 if draw_idx == 0 else 60}
    if item_id == "gen1b/pair0/c/d1":
        if draw_idx == 0:
            return {"reasoning": "declined", "score": "REFUSAL"}  # content drop
        return {"error": True, "transport": True, "reasoning": "batch_error: expired"}
    return {"reasoning": "r", "score": 30}


def test_judge_request_building_and_drop_transport_split(tmp_path, monkeypatch):
    from explore_persona_space.eval import batch_judge

    out_root, bulk_root, bank = _judge_fixture(tmp_path)
    calls: list[dict] = []

    def impl(**kw):
        calls.append(kw)
        all_scores = {}
        for item_id, qmap in kw["completions"].items():
            for _q, comps in qmap.items():
                for ci in range(len(comps)):
                    all_scores[f"{item_id}__00000__{ci:02d}"] = _fake_scores(item_id, ci)
        save_raw = Path(kw["save_raw"])
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    # signature-conformant fake at the Batch-API boundary ONLY: the real
    # judge_graded packing + judge_result_from_save_raw reduce run for real.
    fake = create_autospec(batch_judge.judge_completions_batch, side_effect=impl)
    from explore_persona_space.eval import graded_judge

    monkeypatch.setattr(graded_judge._batch_judge, "judge_completions_batch", fake)

    out = tmp_path / "scores.json"
    jg.main(
        [
            "--out-root",
            str(out_root),
            "--bulk-root",
            str(bulk_root),
            "--pair-bank",
            str(bank),
            "--out-json",
            str(out),
            "--work-dir",
            str(tmp_path / "work"),
            "--n-draws",
            "2",
        ]
    )

    # request building: one rubric label (evil) -> one batch call; 4 judged
    # cells x 2 draws = 8 items; excluded phase absent; max_tokens threaded.
    assert fake.call_count == 1
    kw = calls[0]
    assert kw["max_tokens"] == jg.JUDGE_MAX_TOKENS
    assert len(kw["completions"]) == 8
    assert not any("L2" in item_id for item_id in kw["completions"])

    j = json.loads(out.read_text())
    item = j["per_item"]["gen1b/pair0/c/d0"]
    assert item["graded_score"] == pytest.approx(70.0)
    assert item["binary_positive"] is True and item["arm"] == "baseline"
    dropped = j["per_item"]["gen1b/pair0/c/d1"]
    assert dropped["graded_score"] is None
    assert dropped["n_content_drops"] == 1 and dropped["n_transport_losses"] == 1

    baseline = j["per_arm"]["baseline"]
    assert baseline["n_items"] == 2 and baseline["n_scored_items"] == 1
    assert baseline["content_drops"] == 1 and baseline["transport_losses"] == 1
    assert baseline["mean_graded_score"] == pytest.approx(70.0)
    assert baseline["binary_rate_geq_50"] == pytest.approx(1.0)
    assert set(j["per_arm"]) == {"baseline", "ceiling", "steered_primary_context", "rb_evil"}
    ceiling = j["per_arm"]["ceiling"]
    assert ceiling["mean_graded_score"] == pytest.approx(30.0)
    assert ceiling["binary_rate_geq_50"] == pytest.approx(0.0)


def test_judge_rubric_resolution_reason_then_score():
    rubric = jg.resolve_rubric("evil")  # verbatim in-repo #779 artifact, no Hub call
    assert "{question}" in rubric and "{answer}" in rubric
    assert "[QUESTION START]" in rubric
    assert '"reasoning"' in rubric and '"score"' in rubric  # reason-then-score wrapper
    # #685 behavior label instantiates the generic persona-vectors-shaped rubric
    hedging = jg.resolve_rubric("hedging")
    assert "hedging" in hedging and "{question}" in hedging
    with pytest.raises(ValueError, match="no rubric source"):
        jg.resolve_rubric("not_a_real_label")


# ── deliverable 4: logit lens ─────────────────────────────────────────


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        hidden_size=HID,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_logit_lens_tiny_model_shapes(tmp_path):
    model = _tiny_model()
    act = tmp_path / "acts"
    _write_captures(act, n_pairs=2)
    pairs = common.load_all_pairs(act)
    vectors = ll.build_pair_vectors(pairs, layer=1)
    assert len(vectors) == 2 * 5  # v_c x2 arms, v_a_c, delta x2 arms per pair
    out = ll.compute_lens(model, vectors, top_k=5)
    assert set(out) == set(vectors)
    for rec in out.values():
        assert len(rec["token_ids"]) == 5
        assert len(rec["logits"]) == 5 and len(rec["probs"]) == 5
        assert all(isinstance(t, int) and 0 <= t < 64 for t in rec["token_ids"])
        assert all(np.isfinite(x) for x in rec["logits"])


def test_logit_lens_slow_mode_loader(tmp_path):
    rng = np.random.default_rng(0)
    eigvals = np.array([0.99, 0.985, 0.981, 0.5], dtype=np.complex128)
    eigvecs = rng.standard_normal((HID, 4)) + 1j * 0.01 * rng.standard_normal((HID, 4))
    np.savez(
        tmp_path / "modes.npz",
        block14_eigvals=eigvals,
        block14_eigvecs=eigvecs,
        block14_h_star=rng.standard_normal(HID),
    )
    vectors, prov = ll.load_slow_mode_vectors(tmp_path / "modes.npz", 14, 3)
    # 3 modes x (+/-) + the fixed point
    assert len(vectors) == 7
    assert prov["modes"]["mode0"]["abs_eigval"] == pytest.approx(0.99)
    assert all(v.shape == (HID,) for v in vectors.values())


# ── round 2: cosine units (null/observed commensurability) ────────────


def test_null_battery_cosine_units_and_commensurability(tmp_path):
    """Round-2 fix (h1-null-observed-scale-comparability): the battery
    statistic is the scale-free COSINE — same units as the realized H1
    projection_cosine — with the norm-matching cancelled exactly."""
    mats, out = _run_battery(tmp_path, "cos")
    j = json.loads(out.read_text())
    assert j["units"] == "cosine"
    for battery in ("random_delta", "shuffled_pair"):
        blob = torch.load(mats / f"{battery}_null_matrix.pt", weights_only=True)
        assert blob["units"] == "cosine"
        assert float(blob["per_layer"].abs().max()) <= 1.0 + 1e-5
    # random battery: arm axis replicated (cosine is arm-free — no ||Delta||)
    r = torch.load(mats / "random_delta_null_matrix.pt", weights_only=True)["per_layer"]
    assert torch.equal(r[:, :, 0], r[:, :, 1])
    # the battery's observed rows are INPUT-Delta cosines
    pairs = common.load_all_pairs(tmp_path / "acts")
    obs = nb.observed_stats(pairs)
    p0 = pairs[0]
    d = p0.delta["prefix"][0]
    manual = float((d / d.norm()) @ p0.target_unit()[0])
    assert float(obs[0, 0, 0]) == pytest.approx(manual, rel=1e-5)
    assert "observed_input_delta_cos_max_over_layers" in j


# ── round 2: geometric projection driver (H1 DV + H3) ─────────────────

import issue1415_geometric_projections as gp  # noqa: E402


def test_captured_phases_in_sync():
    """Drift pin: the CPU driver's local phases tuple mirrors the GPU driver's."""
    import issue1415_run_phase1 as drv

    assert tuple(gp.CAPTURED_PHASES) == tuple(drv.CAPTURED_PHASES)


PRIM = 1  # primary layer for the synthetic worlds (LAYERS = [0, 1, 2])


def _projection_world(
    tmp: Path,
    matched_kappa: float = 0.9,
    cross_kappa: float = 0.0,
    op_none: set[str] | None = None,
) -> dict:
    """Synthetic 1a captures + steered-cell metas + 1e captures + selection +
    bank for the projection driver (3 matched / 2 cross pairs)."""
    op_none = op_none or set()
    matched = [f"m{i}" for i in range(3)]
    cross = [f"x{i}" for i in range(2)]
    acts, steered, cells = tmp / "acts", tmp / "steered", tmp / "cells"
    acts.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(11)
    n_layers = len(LAYERS)
    bank: dict = {"pairs": []}
    selection: dict = {}
    for pid in matched + cross:
        ptype = "matched" if pid in matched else "cross"
        bank["pairs"].append({"pair_id": pid, "pair_type": ptype})
        rec_c = {
            "v_c_prefix": torch.randn(n_layers, HID, generator=gen),
            "v_c_context": torch.randn(n_layers, HID, generator=gen),
            "v_a_mean": torch.randn(n_layers, HID, generator=gen),
        }
        rec_cp = {
            "v_c_prefix": torch.randn(n_layers, HID, generator=gen),
            "v_c_context": torch.randn(n_layers, HID, generator=gen),
            "v_a_mean": torch.randn(n_layers, HID, generator=gen),
        }
        torch.save(
            {"pair_id": pid, "layers": list(LAYERS), "c": rec_c, "cprime": rec_cp},
            acts / f"{pid}.pt",
        )
        target = rec_cp["v_a_mean"] - rec_c["v_a_mean"]
        kappa = matched_kappa if ptype == "matched" else cross_kappa
        for arm in common.ARMS:
            selection[f"{arm}/{pid}"] = {
                "operating_alpha": None if pid in op_none else 1.0,
                "retried": False,
            }
            if pid in op_none:
                continue
            for layer in LAYERS:
                phase = "phase1c_grid" if layer == PRIM else "phase1c_layers"
                cid = f"gen1c/{arm}/{pid}/L{layer}/a1"
                mpath = cells / f"{cid}.json"
                mpath.parent.mkdir(parents=True, exist_ok=True)
                mpath.write_text(
                    json.dumps(
                        {
                            "cell_id": cid,
                            "pair_id": pid,
                            "phase": phase,
                            "layer": layer,
                            "alpha": 1.0,
                            "all_positions": False,
                            "extraction_arm": arm,
                            "passes_gate": True,
                        }
                    )
                )
                noise = torch.randn(n_layers, HID, generator=gen) * 0.05
                spath = steered / f"{cid}.pt"
                spath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "cell_id": cid,
                        "layers": list(LAYERS),
                        "all_empty": False,
                        "v_a_mean": rec_c["v_a_mean"] + kappa * target + noise,
                        "n_empty_completions": 0,
                    },
                    spath,
                )
    sel_path = tmp / "alpha_selection_1c.json"
    sel_path.write_text(json.dumps({"selection": selection, "retry_alpha": 0.25}))
    bank_path = tmp / "pair_bank.json"
    bank_path.write_text(json.dumps(bank))
    return {
        "acts": acts,
        "steered": steered,
        "cells": cells,
        "selection": sel_path,
        "bank": bank_path,
    }


def _gp_argv(world: dict, out: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--activations",
        str(world["acts"]),
        "--steered-activations",
        str(world["steered"]),
        "--cells",
        str(world["cells"]),
        "--alpha-selection",
        str(world["selection"]),
        "--pair-bank",
        str(world["bank"]),
        "--out-json",
        str(out),
        "--primary-layer",
        str(PRIM),
        *(extra or []),
    ]


def test_projection_driver_h1_h3_happy_path(tmp_path):
    world = _projection_world(tmp_path)
    out = tmp_path / "geo.json"
    gp.main(_gp_argv(world, out, ["--expect-counts", "3,2"]))
    j = json.loads(out.read_text())
    assert j["units"] == "cosine"
    # per-cell projections are bounded cosines; matched-layer stat present
    for row in j["per_cell"]:
        assert row["excluded_reason"] is None
        for rec in row["per_read_layer"].values():
            assert -1.0 - 1e-6 <= rec["projection_cosine"] <= 1.0 + 1e-6
            assert rec["shift_norm"] > 0 and rec["target_norm"] > 0
    for arm in common.ARMS:
        rows = j["h1"][arm]
        assert len(rows) == 5
        for r in rows.values():
            assert r["excluded_reason"] is None
            # selection symmetry: max over the per-layer matched-layer cosines
            vals = [v for v in r["per_layer_matched_cos"].values() if v is not None]
            assert r["max_over_layers"] == pytest.approx(max(vals))
        h3 = j["h3"]["per_arm"][arm]
        assert h3["n_matched_rows"] == 3 and h3["n_cross_rows"] == 2
        assert h3["n_used_matched"] == 3 and h3["n_used_cross"] == 2
        # matched pairs steered ~toward the target, cross pairs pure noise
        assert h3["matched_mean"] > h3["cross_mean"]
        assert h3["welch_p_one_sided"] < 0.1
        assert h3["ranksum_p_one_sided"] < 0.2


def test_projection_driver_default_count_assert_fires(tmp_path):
    """Plan §3: the driver asserts len(matched)==15 / len(cross)==13 by
    default — a smaller bank must fail loud unless --expect-counts says so."""
    world = _projection_world(tmp_path)
    with pytest.raises(AssertionError, match="H3 row-count assert failed"):
        gp.main(_gp_argv(world, tmp_path / "geo.json"))


def test_projection_driver_excluded_pair_recorded_and_dropped_from_test(tmp_path):
    world = _projection_world(tmp_path, op_none={"m2"})
    out = tmp_path / "geo.json"
    gp.main(_gp_argv(world, out, ["--expect-counts", "3,2"]))
    j = json.loads(out.read_text())
    for arm in common.ARMS:
        row = j["h1"][arm]["m2"]
        assert row["excluded_reason"] == "coherence_failed_all_alpha"
        assert row["max_over_layers"] is None
        h3 = j["h3"]["per_arm"][arm]
        assert h3["n_matched_rows"] == 3  # the row EXISTS in the file (plan §3)
        assert h3["n_used_matched"] == 2  # ... but is dropped from the test
        assert "deviation" in h3
        assert h3["excluded_pairs"] == ["m2"]


def test_projection_driver_band_units_guard_and_comparison(tmp_path):
    world = _projection_world(tmp_path)
    all_pids = [f"m{i}" for i in range(3)] + [f"x{i}" for i in range(2)]
    band = {"p2.5": -0.3, "p50": 0.0, "p97.5": 0.3}
    bands_ok = {
        "units": "cosine",
        "bands": {
            b: {"per_pair": {arm: {pid: dict(band) for pid in all_pids} for arm in common.ARMS}}
            for b in ("random_delta", "shuffled_pair")
        },
    }
    bands_path = tmp_path / "bands.json"
    bands_path.write_text(json.dumps(bands_ok))
    out = tmp_path / "geo.json"
    gp.main(_gp_argv(world, out, ["--expect-counts", "3,2", "--null-bands", str(bands_path)]))
    j = json.loads(out.read_text())
    m0 = j["h1"]["prefix"]["m0"]
    assert m0["band_comparison"]["random_delta"]["null_p97.5"] == 0.3
    assert m0["band_comparison"]["random_delta"]["exceeds_p97.5"] is True

    # units guard: a non-cosine bands file is REFUSED (never compared across units)
    bad = dict(bands_ok, units="norm-matched-projection")
    bad_path = tmp_path / "bands_bad.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="need 'cosine'"):
        gp.main(
            _gp_argv(
                world,
                tmp_path / "geo2.json",
                ["--expect-counts", "3,2", "--null-bands", str(bad_path)],
            )
        )


def test_map_transport_steered_skips_missing_canonical(tmp_path):
    """A coherence-excluded pair (no canonical 1e capture) is skipped WITH a
    record (plan §8), never a crash; remaining pairs still compute."""
    act = tmp_path / "acts"
    ids = _write_captures(act, n_pairs=3)
    map_path = tmp_path / "map.pt"
    torch.save({"layer_1": {"weight": torch.eye(HID)}}, map_path)
    steer = tmp_path / "steered"
    steer.mkdir()
    pairs = common.load_all_pairs(act)
    li = pairs[0].layers.index(1)
    for p in pairs[:2]:  # the 3rd pair has NO canonical capture
        for arm in common.ARMS:
            torch.save(
                {"v_a_mean": p.v_a_c[li] + p.delta[arm][li]},
                steer / f"{p.pair_id}__{arm}.pt",
            )
    out = tmp_path / "mt.json"
    mt.main(
        [
            "--activations",
            str(act),
            "--map-path",
            str(map_path),
            "--steered-activations",
            str(steer),
            "--out-json",
            str(out),
            "--layer",
            "1",
            "--hidden",
            str(HID),
            "--n-draws",
            "10",
        ]
    )
    j = json.loads(out.read_text())
    for arm in common.ARMS:
        assert j["per_arm"][arm]["skipped_pairs"] == [ids[2]]
        assert len(j["per_arm"][arm]["transport_cosine"]) == 2


def test_judge_k1_check_thresholds():
    """K1 judge half (deferred): fires on uniformly small ceiling shifts,
    passes on large ones, and stays un-fired when not evaluable."""

    def items(shift):
        per_item = {}
        for i, pid in enumerate(("a", "b", "c")):
            per_item[f"b{i}"] = {"arm": "baseline", "pair_id": pid, "graded_score": 40.0}
            per_item[f"c{i}"] = {"arm": "ceiling", "pair_id": pid, "graded_score": 40.0 + shift}
        return per_item

    small = jg.k1_judge_check(items(2.0))
    assert small["fired"] is True and small["frac_small_shift"] == 1.0
    big = jg.k1_judge_check(items(30.0))
    assert big["fired"] is False and big["frac_small_shift"] == 0.0
    none = jg.k1_judge_check({"x": {"arm": "baseline", "pair_id": "a", "graded_score": 10.0}})
    assert none["fired"] is False and none["frac_small_shift"] is None
