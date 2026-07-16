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
