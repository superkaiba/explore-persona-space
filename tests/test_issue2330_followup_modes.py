"""Pins for the #2330 follow-up modes (dense-9b-layer-sweep + cap2048-regen).

Covers, network-free and CPU-only (everything in ``tests/`` runs in every
issue's Step 9c gate):

1. ``--layers`` inclusive ``A-B`` range parsing (dense capture launch shape)
   incl. the duplicate/inverted fail-loud branches.
2. ``_apply_gen_max_tokens`` cap arithmetic in BOTH capture drivers: the
   PROMPT_TOKEN_BUDGET is held INVARIANT (admitted row set identical to the
   banked cap-1024 stores) and MAX_MODEL_LEN is DERIVED — the CLAUDE.md
   inherited-rig rule for a raised cap (#505/#601).
3. ``--hf-prefix-override`` parsing/validation + the derived alternate-store
   MODELS dict (wc/ceiling inherit the original prefix; anchor skipped;
   count pins at the split_ids grain; only overridden models returned).
4. ``assemble_store`` wc-prefix routing through a signature-conformant fake of
   ``issue1491_ladder_fits._stream_ladder_split``.
5. ``run_battery`` out_suffix threading — REAL battery body on a tiny
   synthetic store (h_dim=8; per-split n deliberately all distinct so a
   transposed/mis-sliced shape cannot alias).
6. ``run_dense_sweep`` end-to-end via ``--dense-local-dir`` chunks in the
   REAL ``_stack_chunk`` schema through the REAL
   ``issue779_ffc_n1m_fits._stream_n1m_multilayer`` (zero-row pb_head), plus
   the resume-skip re-run and the wrong-store guards (3-layer store refused,
   h_dim mismatch refused, regime-mismatch resume refused).
7. Signature-bind of the production (HF-branch) stream call shape.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_fits as F  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
import issue1491_ladder_generate_capture as LG  # noqa: E402
import issue2330_matched_fits as MF  # noqa: E402
import issue2330_qwen35_generate_capture as GC  # noqa: E402

# ---------------------------------------------------------------------------
# 1. --layers range parsing (dense capture launch shape)
# ---------------------------------------------------------------------------


def test_layers_range_parsing_dense():
    assert GC._resolve_layers_arg("0-30") == list(range(31))
    assert GC._resolve_layers_arg("16,22,30") == [16, 22, 30]
    assert GC._resolve_layers_arg("0-2,5") == [0, 1, 2, 5]


def test_layers_range_parsing_fail_loud():
    with pytest.raises(ValueError, match="inverted"):
        GC._resolve_layers_arg("5-2")
    with pytest.raises(ValueError, match="duplicates"):
        GC._resolve_layers_arg("0-3,2")
    with pytest.raises(ValueError, match="non-empty"):
        GC._resolve_layers_arg(",")


# ---------------------------------------------------------------------------
# 2. _apply_gen_max_tokens — budget invariant, max_model_len derived
# ---------------------------------------------------------------------------


def _roundtrip_cap(mod, default_cap: int):
    budget0, margin = mod.PROMPT_TOKEN_BUDGET, mod.LENGTH_MARGIN
    try:
        gm, mml, budget = mod._apply_gen_max_tokens(2048)
        assert (gm, mml, budget) == (2048, budget0 + 2048 + margin, budget0)
        assert mod.GEN_MAX_TOKENS == 2048 and mml == mod.MAX_MODEL_LEN
        assert budget0 == mod.PROMPT_TOKEN_BUDGET  # INVARIANT — row set unchanged
        with pytest.raises(AssertionError):
            mod._apply_gen_max_tokens(0)
    finally:
        gm, mml, budget = mod._apply_gen_max_tokens(default_cap)
    assert (gm, mml, budget) == (default_cap, budget0 + default_cap + margin, budget0)


def test_apply_gen_max_tokens_2330_driver():
    assert (GC.GEN_MAX_TOKENS, GC.MAX_MODEL_LEN, GC.PROMPT_TOKEN_BUDGET) == (1024, 8192, 7104)
    _roundtrip_cap(GC, 1024)
    assert (GC.GEN_MAX_TOKENS, GC.MAX_MODEL_LEN, GC.PROMPT_TOKEN_BUDGET) == (1024, 8192, 7104)


def test_apply_gen_max_tokens_1491_driver():
    assert (LG.GEN_MAX_TOKENS, LG.MAX_MODEL_LEN, LG.PROMPT_TOKEN_BUDGET) == (1024, 8192, 7104)
    _roundtrip_cap(LG, 1024)
    assert (LG.GEN_MAX_TOKENS, LG.MAX_MODEL_LEN, LG.PROMPT_TOKEN_BUDGET) == (1024, 8192, 7104)


# ---------------------------------------------------------------------------
# 3. --hf-prefix-override parsing + derived alternate-store MODELS
# ---------------------------------------------------------------------------


def _tiny_split_ids(n_tr10: int = 16, n_tr5: int = 8, n_val: int = 4, n_te: int = 5, n_wc: int = 3):
    tr10 = list(range(100, 100 + n_tr10))
    splits = {
        "train_10k": tr10,
        "train_5k": tr10[:n_tr5],
        "val_400": list(range(500, 500 + n_val)),
        "test_1000": list(range(600, 600 + n_te)),
        "wc_test_1k": list(range(700, 700 + n_wc)),
    }
    return {
        "splits": splits,
        "counts": {k: len(v) for k, v in splits.items()},
        "sha256": {k: f"sha-{k}" for k in splits},
        "dropped_overlength": {},
    }


def test_parse_prefix_overrides_validation():
    ov = MF._parse_prefix_overrides(["qwen35_9b=issue2330_matched/qwen35_9b_cap2048"])
    assert ov == {"qwen35_9b": "issue2330_matched/qwen35_9b_cap2048"}
    assert MF._parse_prefix_overrides(None) == {}
    with pytest.raises(RuntimeError, match="MODEL_KEY=PREFIX"):
        MF._parse_prefix_overrides(["qwen35_9b"])
    with pytest.raises(RuntimeError, match="unknown model key"):
        MF._parse_prefix_overrides(["nope=pfx"])
    with pytest.raises(RuntimeError, match="duplicate"):
        MF._parse_prefix_overrides(["qwen35_9b=a", "qwen35_9b=b"])
    with pytest.raises(RuntimeError, match="equals the original"):
        MF._parse_prefix_overrides([f"qwen35_9b={MF.HF_PREFIX_9B}"])


def test_apply_store_prefix_overrides_fields():
    split_ids = _tiny_split_ids()
    ov = {
        "qwen25_7b": "issue2330_matched/q25_cap2048",
        "qwen35_9b": "issue2330_matched/qwen35_9b_cap2048",
    }
    models = MF._apply_store_prefix_overrides(MF.MODELS, ov, split_ids)
    assert sorted(models) == sorted(ov)  # ONLY overridden models
    m7 = models["qwen25_7b"]
    assert m7["hf_prefix"] == "issue2330_matched/q25_cap2048"
    assert m7["wc_hf_prefix"] == MF.HF_PREFIX_7B
    assert m7["ceiling_hf_prefix"] == MF.HF_PREFIX_7B
    assert m7["anchor"] is False and m7["store_revision_pin"] is None
    # 7B train subpath keeps the banked NAME but realizes the train_10k grain;
    # wc keeps the ORIGINAL banked grain (999), not the post-drop 998.
    assert m7["store_expected_n"] == {
        "train_25k": split_ids["counts"]["train_10k"],
        "val_400": split_ids["counts"]["val_400"],
        "test_1000": split_ids["counts"]["test_1000"],
        "wc_test_1k": int(LF.EXPECTED_SPLIT_N["wc_test_1k"]),
    }
    assert m7["override_meta"]["anchor_skipped"]  # 7B had the anchor → reason recorded
    m9 = models["qwen35_9b"]
    assert m9["store_expected_n"]["wc_test_1k"] == split_ids["counts"]["wc_test_1k"]
    assert m9["override_meta"]["anchor_skipped"] is None
    assert m9["override_meta"]["inherited_from_original"] == ["wc_test_1k", "ceiling_two_draw"]
    # Original MODELS untouched (deep copy).
    assert MF.MODELS["qwen25_7b"]["hf_prefix"] == MF.HF_PREFIX_7B
    assert MF.MODELS["qwen25_7b"]["anchor"] is True


# ---------------------------------------------------------------------------
# 4. assemble_store wc-prefix routing (signature-conformant fake streamer)
# ---------------------------------------------------------------------------


def test_assemble_store_wc_prefix_routing(monkeypatch, tmp_path):
    H = 3
    calls: list[tuple[str, str]] = []
    n_by_split = {"train_10k": 6, "val_400": 4, "test_1000": 5, "wc_test_1k": 3}

    def fake_stream_ladder_split(hf_prefix: str, split: str, layer: int, cache_dir: Path):
        # Mirrors LF._stream_ladder_split's REAL signature (positional 4-arg).
        calls.append((hf_prefix, split))
        n = n_by_split[split]
        rng = np.random.default_rng(n)
        cx = rng.standard_normal((n, H)).astype(np.float32)
        vx = rng.standard_normal((n, H)).astype(np.float32)
        return cx, vx, list(range(n))

    inspect.signature(LF._stream_ladder_split).bind("p", "s", 0, tmp_path)  # sig parity
    monkeypatch.setattr(MF.LF, "_stream_ladder_split", fake_stream_ladder_split)
    expected = dict(n_by_split)
    MF.assemble_store("cap_pfx", "train_10k", 0, tmp_path, expected, wc_hf_prefix="orig_pfx")
    by_split = dict((s, p) for p, s in calls)
    assert by_split["wc_test_1k"] == "orig_pfx"
    assert {by_split[s] for s in ("train_10k", "val_400", "test_1000")} == {"cap_pfx"}
    # Default (no wc override): every split streams from the main prefix.
    calls.clear()
    MF.assemble_store("cap_pfx", "train_10k", 0, tmp_path, expected)
    assert {p for p, _ in calls} == {"cap_pfx"}


# ---------------------------------------------------------------------------
# 5. run_battery out_suffix threading (real battery body, tiny synthetic store)
# ---------------------------------------------------------------------------


def test_run_battery_out_suffix_tiny_synthetic(tmp_path):
    H = 8
    split_ids = _tiny_split_ids()
    mcfg = {
        "model": "synthetic/tiny",
        "hf_prefix": "syn_pfx",
        "store_revision_pin": None,
        "layers": [0, 1],
        "primary_layer": 0,
        "h_dim": H,
        "store_train_split": "train_10k",
        "store_expected_n": None,
        "cells": {"q35_n10k": "train_10k"},
        "anchor": False,
        "override_meta": {"store_prefix": "syn_pfx", "note": "test override meta"},
    }
    rng = np.random.default_rng(0)

    # One SHARED linear map across splits + a real noise floor keeps the
    # val-selected λ INTERIOR (probed: λ≈3.2-10 across seeds). A noiseless
    # map drives λ to the LOW edge and unrelated targets to the HIGH edge —
    # both exhaust the plan-§11 extension disposition (that fail-loud branch
    # is covered by the driver's own --selftest, not this test).
    W = rng.standard_normal((H, H)).astype(np.float32) / np.sqrt(H)

    def store_fn(cfg, layer):
        out = {}
        for split in ["train_10k", *MF.STORE_SPLITS]:
            ids = split_ids["splits"][split]
            n = len(ids)
            cx = rng.standard_normal((n, H)).astype(np.float32)
            vx = (cx @ W + 0.5 * rng.standard_normal((n, H))).astype(np.float32)
            out[split] = {"cx": cx, "vx": vx, "ci": list(ids)}
        return out

    def ceiling_fn(cfg, layer):
        return {"available": False, "reason": "synthetic test"}

    paths = MF.run_battery(
        split_ids,
        store_fn,
        ceiling_fn,
        torch.device("cpu"),
        tmp_path / "out",
        tmp_path / "preds",
        models={"qwen35_9b": mcfg},
        anchor_fn=None,
        cap_hit_fn=None,
        out_suffix="_syn",
    )
    cell_json = tmp_path / "out" / "matched_fits_q35_n10k_syn.json"
    assert paths["q35_n10k"] == cell_json and cell_json.is_file()
    rec = json.loads(cell_json.read_text())
    assert rec["store_prefix_override"] == mcfg["override_meta"]
    assert rec["preds_hf_mirror"].endswith("q35_n10k_test_preds_ridge_syn.npz")
    npz = tmp_path / "preds" / "q35_n10k_test_preds_ridge_syn.npz"
    assert npz.is_file()
    got = np.load(npz)
    assert got["pred_te_L0"].shape == (len(split_ids["splits"]["test_1000"]), H)


# ---------------------------------------------------------------------------
# 6. run_dense_sweep local e2e + resume + wrong-store guards
# ---------------------------------------------------------------------------

DENSE_H = 8
DENSE_LAYERS = [0, 1, 2, 3]


def _write_dense_chunks(root: Path, split_ids: dict, layers=None, h_dim=DENSE_H):
    """Write tiny per-split capture chunks in the REAL _stack_chunk schema.

    Per layer, ONE shared linear map across splits + a noise floor (same
    rationale as the battery fixture: keeps the val-selected λ interior)."""
    layers = list(DENSE_LAYERS if layers is None else layers)
    rng = np.random.default_rng(7)
    maps = {
        li: rng.standard_normal((h_dim, h_dim)).astype(np.float32) / np.sqrt(h_dim) for li in layers
    }
    for split in MF.DENSE_SPLITS:
        ids = split_ids["splits"][split]
        d = root / split / "final_token_capture"
        d.mkdir(parents=True, exist_ok=True)
        n = len(ids)
        cx = np.stack([rng.standard_normal((n, h_dim)).astype(np.float32) for _ in layers], axis=1)
        vx = np.stack(
            [
                cx[:, j, :] @ maps[li] + 0.5 * rng.standard_normal((n, h_dim)).astype(np.float32)
                for j, li in enumerate(layers)
            ],
            axis=1,
        )
        bundle = {
            "cx_last": torch.from_numpy(cx),
            "v_x": torch.from_numpy(vx.astype(np.float32)),
            "ci": [int(i) for i in ids],
            "prompts": ["<digest-only>" for _ in ids],
            "layers": layers,
            "shard_index": 0,
            "chunk": 0,
        }
        torch.save(bundle, d / "shard00_chunk0000.pt")


def _dense_args(tmp_path: Path, split_ids_path: Path, **over):
    argv = [
        "--device",
        "cpu",
        "--split-ids",
        str(split_ids_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--dense-prefix",
        "issue2330_matched/qwen35_9b_dense",
        "--dense-local-dir",
        str(tmp_path / "dense_store"),
        "--dense-expect-h-dim",
        str(DENSE_H),
        "--dense-out",
        str(tmp_path / "out" / "dense_sweep" / "matched_fits_q35_dense.json"),
    ]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return MF._build_parser().parse_args(argv)


def test_run_dense_sweep_local_e2e_and_resume(tmp_path, capsys):
    split_ids = _tiny_split_ids()
    sp = tmp_path / "split_ids.json"
    sp.write_text(json.dumps(split_ids))
    _write_dense_chunks(tmp_path / "dense_store", split_ids)
    args = _dense_args(tmp_path, sp)
    assert MF.run_dense_sweep(args) == 0
    out = json.loads(args.dense_out.read_text())
    assert out["meta"]["layers"] == DENSE_LAYERS and out["meta"]["h_dim"] == DENSE_H
    assert "scoped-peak diagnostic" in out["meta"]["label"]
    for cell, train_key in (("q35_n10k", "train_10k"), ("q35_n5k", "train_5k")):
        crec = out["cells"][cell]
        assert sorted(crec["per_layer"]) == sorted(str(x) for x in DENSE_LAYERS)
        assert crec["n_train"] == split_ids["counts"][train_key]
        assert crec["n_vs_d"]["d"] == DENSE_H
        assert int(crec["peak"]["layer"]) in DENSE_LAYERS
        for lrec in crec["per_layer"].values():
            assert "test_r2" in lrec and "selected_lambda" in lrec
    capsys.readouterr()
    # Resume: every unit SKIPs; payload content unchanged.
    assert MF.run_dense_sweep(_dense_args(tmp_path, sp)) == 0
    printed = capsys.readouterr().out
    assert printed.count("SKIP (resumed)") == len(DENSE_LAYERS) * 2
    assert json.loads(args.dense_out.read_text())["cells"] == out["cells"]


def test_run_dense_sweep_wrong_store_guards(tmp_path):
    split_ids = _tiny_split_ids()
    sp = tmp_path / "split_ids.json"
    sp.write_text(json.dumps(split_ids))
    # (a) 3-layer registry-shaped store → refused.
    _write_dense_chunks(tmp_path / "dense_store", split_ids, layers=[16, 22, 30])
    with pytest.raises(RuntimeError, match="3-layer registry store"):
        MF.run_dense_sweep(_dense_args(tmp_path, sp))
    # (b) h_dim mismatch → refused.
    _write_dense_chunks(tmp_path / "dense_store", split_ids)
    with pytest.raises(RuntimeError, match="h_dim"):
        MF.run_dense_sweep(_dense_args(tmp_path, sp, dense_expect_h_dim=DENSE_H + 1))
    # (c) unknown cell → refused.
    with pytest.raises(RuntimeError, match="dense-cells"):
        MF.run_dense_sweep(_dense_args(tmp_path, sp, dense_cells="nope"))


def test_run_dense_sweep_regime_mismatch_resume_refused(tmp_path):
    split_ids = _tiny_split_ids()
    sp = tmp_path / "split_ids.json"
    sp.write_text(json.dumps(split_ids))
    _write_dense_chunks(tmp_path / "dense_store", split_ids)
    args = _dense_args(tmp_path, sp, dense_cells="q35_n10k")
    assert MF.run_dense_sweep(args) == 0
    # Same --dense-out, DIFFERENT regime (prefix) → refuse, never mix (#722).
    args2 = _dense_args(tmp_path, sp, dense_cells="q35_n10k")
    args2.dense_prefix = "issue2330_matched/other_prefix"
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        MF.run_dense_sweep(args2)


# ---------------------------------------------------------------------------
# 7. Production (HF-branch) stream call shape binds against the reuse core
# ---------------------------------------------------------------------------


def test_dense_stream_hf_call_shape_binds(tmp_path):
    sig = inspect.signature(F._stream_n1m_multilayer)
    zero = np.zeros((0, DENSE_H), dtype=np.float32)
    sig.bind(
        "issue2330_matched/qwen35_9b_dense/train_10k/final_token_capture",
        DENSE_LAYERS,
        tmp_path / "dl",
        tmp_path / "mm",
        {li: (zero, zero) for li in DENSE_LAYERS},
        local_dir=None,
    )
    inspect.signature(F._download_chunk_with_retry).bind("repo", "prefix/name.pt", tmp_path)
    assert MF._dense_chunk_prefix("root", "train_10k") == "root/train_10k/final_token_capture"
