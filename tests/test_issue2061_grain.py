"""#2061 concat-loader + P0 grain-gate pins (plan v11/v13 deltas a1/a1-bis/a2/d/e).

Producer-shaped two-tree fixtures (the #1336 `write_shards` payload + JSON
sidecar schema, s-idx conv ids at the V2_CONCAT_BOUNDARY grain) drive:

- `issue2061_turnstore.load_state_cell` — the registered-consumption-grain
  loader: canonical concat order, boundary/disjointness/duplicate asserts,
  per-shard sidecar schema + the PRESENCE-CONDITIONAL capture-convention
  assert (a1-bis: present => committed/null; ABSENT => pre-D2-era store
  ACCEPTED — the key-less fixture below is the mandatory absent-branch pin),
  and the per-row payload-vs-sidecar prompt-sha join (zero mismatches);
- `issue2061_sae_encode.resolve_turnstore_trees` — the ordered store pair +
  the 35-cells-consume-50-stores acceptance;
- `issue2061_grain_gate.gate_cells` / `main` — realized n + EXACT per-fold
  n_train through the production fold constructor, regime verdicts, the
  cross-store sha join, and the fail-loud nonzero exits (rc=2 asserts /
  rc=3 regime contradiction);
- `issue2061_grain_gate.check_smoke_acceptance` — the dispatch smoke
  acceptance surface (convention=primal, v13 grid + λ-edge audit fields,
  realized n vs the manifest).

Every fail-loud assert here follows the house fails-loud pattern
(`test_gcv_dof_cap_all_capped_fail_loud`, `test_group_fold_ids_fail_loud_on_empty_fold`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2061_grain_gate as gg
import issue2061_sae_encode as enc
import issue2061_turnstore as ts

N_LAYERS = 4
HIDDEN = 8
LAYER = 2

WAVE1_STEM = "base_chat_lmsys5k"
EXT_STEM = "base_chat_lmsys23k"


def _record(idx: int) -> dict:
    slots = torch.full((2, N_LAYERS, HIDDEN), float(idx % 977), dtype=torch.bfloat16)
    profiles = torch.full((2, N_LAYERS, HIDDEN), float(idx % 977) + 0.5, dtype=torch.bfloat16)
    return {
        "conv_id": f"s{idx}",
        "slots": slots,
        "profiles": profiles,
        "nll": torch.tensor([0.1, 0.2]),
        "spans_meta": {
            "conv_id": f"s{idx}",
            "slot_names": ["prefix", "a1"],
            "turn_names": ["u1", "a1"],
        },
    }


def _write_shard(
    dir_: Path,
    shard_idx: int,
    idxs: list[int],
    stem: str,
    *,
    convention: str | None = "committed",
    offset_override=None,
    include_convention_keys: bool = True,
    with_shas: bool = False,
    sidecar_conv_ids: list[str] | None = None,
    sidecar_shas: list[str] | None = None,
    model_id: str = "meta-llama/Llama-3.1-8B",
) -> Path:
    """Producer-shaped shard + sidecar (issue1336_extract_turnstore.write_shards)."""
    dir_.mkdir(parents=True, exist_ok=True)
    recs = [_record(i) for i in idxs]
    payload = {
        "conv_ids": [r["conv_id"] for r in recs],
        "slots": [r["slots"] for r in recs],
        "profiles": [r["profiles"] for r in recs],
        "nll": [r["nll"] for r in recs],
        "spans_meta": [r["spans_meta"] for r in recs],
    }
    if with_shas:
        payload["prompt_shas"] = [f"sha-{i}" for i in idxs]
    pt = dir_ / f"{stem}_shard{shard_idx:03d}.pt"
    torch.save(payload, pt)
    sidecar = {
        "shard_index": shard_idx,
        "n_conversations": len(recs),
        "conv_ids": sidecar_conv_ids if sidecar_conv_ids is not None else payload["conv_ids"],
        "model_id": model_id,
        "expected_layers": N_LAYERS,
        "expected_hidden": HIDDEN,
        "shapes": {
            "slots": [[2, N_LAYERS, HIDDEN] for _ in recs],
            "profiles": [[2, N_LAYERS, HIDDEN] for _ in recs],
            "nll": [[2] for _ in recs],
        },
    }
    if include_convention_keys:
        sidecar["convention"] = convention
        sidecar["offset_override"] = offset_override
    if with_shas:
        sidecar["prompt_shas"] = (
            sidecar_shas if sidecar_shas is not None else payload["prompt_shas"]
        )
    (dir_ / f"{stem}_shard{shard_idx:03d}.json").write_text(json.dumps(sidecar))
    return pt


def _write_concat_cell(root: Path, **ext_kwargs) -> tuple[Path, Path]:
    """Wave-1 (idx < 5000) + v2 extension (idx >= 5000) fixture cell."""
    w1 = root / "turnstore_base_chat_lmsys5k"
    ext = root / "turnstore_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 1, 2], WAVE1_STEM)
    _write_shard(w1, 1, [3, 4], WAVE1_STEM)
    _write_shard(ext, 0, [5000, 5001, 5002], EXT_STEM, with_shas=True, **ext_kwargs)
    return w1, ext


# ---------------------------------------------------------------------------
# load_state_cell — the concat loader (delta a1)
# ---------------------------------------------------------------------------
def test_load_state_cell_concat_happy_path(tmp_path):
    _write_concat_cell(tmp_path)
    x, conv_ids, info = ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)
    # Canonical row order: wave-1 shards FIRST, then the extension.
    assert conv_ids == ["s0", "s1", "s2", "s3", "s4", "s5000", "s5001", "s5002"]
    assert x.shape == (8, HIDDEN)
    assert info["concat"] is True and info["n_rows"] == 8
    assert [p["n_rows"] for p in info["parts"]] == [5, 3]
    # The pre-D2 fixture (wave-1 sidecars carry the keys here) reads committed.
    assert info["parts"][0]["conventions"] == ["committed"]


def test_load_state_cell_boundary_fail_loud_both_directions(tmp_path):
    # Wave-1 row at/above the boundary.
    w1 = tmp_path / "turnstore_base_chat_lmsys5k"
    ext = tmp_path / "turnstore_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 5000], WAVE1_STEM)
    _write_shard(ext, 0, [5001], EXT_STEM, with_shas=True)
    with pytest.raises(ValueError, match="wave-1 store has"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)

    # Extension row below the boundary.
    root2 = tmp_path / "r2"
    _write_shard(root2 / "turnstore_base_chat_lmsys5k", 0, [0, 1], WAVE1_STEM)
    _write_shard(root2 / "turnstore_base_chat_lmsys23k", 0, [4999], EXT_STEM, with_shas=True)
    with pytest.raises(ValueError, match="extension store has"):
        ts.load_state_cell(root2, "base", "chat", "lmsys23k", "context", LAYER)


def test_load_state_cell_duplicate_conv_ids_fail_loud(tmp_path):
    ext = tmp_path / "turnstore_base_chat_lmsys23k"
    _write_shard(tmp_path / "turnstore_base_chat_lmsys5k", 0, [0, 1], WAVE1_STEM)
    _write_shard(ext, 0, [5000, 5001], EXT_STEM, with_shas=True)
    _write_shard(ext, 1, [5001], EXT_STEM, with_shas=True)  # duplicate across shards
    with pytest.raises(ValueError, match="duplicate conv_ids"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_load_state_cell_missing_wave1_dir_fail_loud(tmp_path):
    _write_shard(tmp_path / "turnstore_base_chat_lmsys23k", 0, [5000], EXT_STEM, with_shas=True)
    with pytest.raises(FileNotFoundError, match="CONCAT cell consumes"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_missing_sidecar_fail_loud(tmp_path):
    w1, _ext = _write_concat_cell(tmp_path)
    (w1 / f"{WAVE1_STEM}_shard000.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing turnstore sidecar"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_keyless_sidecar_accepted_and_reported(tmp_path):
    """The MANDATORY absent-branch fixture (v13 a1-bis): a pre-D2-era store
    lacking BOTH capture-convention keys is ACCEPTED and reported — a hard
    key-presence assert would fail loud on the 2 oldest wave-1 lmsys5k rlvr
    stores at first dispatch."""
    w1 = tmp_path / "turnstore_base_chat_lmsys5k"
    ext = tmp_path / "turnstore_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 1], WAVE1_STEM, include_convention_keys=False)
    _write_shard(ext, 0, [5000], EXT_STEM, with_shas=True)
    _x, conv_ids, info = ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)
    assert conv_ids == ["s0", "s1", "s5000"]
    assert info["parts"][0]["conventions"] == ["pre-D2-absent"]
    assert info["parts"][1]["conventions"] == ["committed"]


def test_wrong_convention_value_fail_loud(tmp_path):
    _write_concat_cell(tmp_path, convention="corrected")
    with pytest.raises(ValueError, match="capture convention"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_nonnull_offset_override_fail_loud(tmp_path):
    _write_concat_cell(tmp_path, offset_override=3)
    with pytest.raises(ValueError, match="offset_override"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_offset_override_without_convention_fail_loud(tmp_path):
    sidecar = {"conv_ids": ["s0"], "n_conversations": 1, "offset_override": 2}
    with pytest.raises(ValueError, match="without a convention"):
        ts.sidecar_convention_state(sidecar, src="mem")


def test_sidecar_payload_row_mismatch_fail_loud(tmp_path):
    """The ±1 row-count bookkeeping class: a payload/sidecar conv-id
    divergence now fails LOUD at load, never a silent count anomaly."""
    w1 = tmp_path / "turnstore_base_chat_lmsys5k"
    ext = tmp_path / "turnstore_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 1, 2], WAVE1_STEM, sidecar_conv_ids=["s0", "s1"])
    _write_shard(ext, 0, [5000], EXT_STEM, with_shas=True)
    with pytest.raises(ValueError, match="row mismatch"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_prompt_sha_join_zero_mismatches_fail_loud(tmp_path):
    _write_concat_cell(tmp_path, sidecar_shas=["sha-5000", "sha-WRONG", "sha-5002"])
    with pytest.raises(ValueError, match="prompt-sha MISMATCH"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_prompt_sha_one_sided_fail_loud():
    payload = {"conv_ids": ["s0"], "prompt_shas": ["sha-0"]}
    sidecar = {
        "conv_ids": ["s0"],
        "n_conversations": 1,
        "convention": "committed",
        "offset_override": None,
    }
    with pytest.raises(ValueError, match="ONLY"):
        ts.assert_shard_sidecar(payload, sidecar, src="mem")


def test_cross_tree_model_id_mismatch_fail_loud(tmp_path):
    w1 = tmp_path / "turnstore_base_chat_lmsys5k"
    ext = tmp_path / "turnstore_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 1], WAVE1_STEM, model_id="meta-llama/Llama-3.1-8B")
    _write_shard(ext, 0, [5000], EXT_STEM, with_shas=True, model_id="other/model")
    with pytest.raises(ValueError, match="model_id mismatch"):
        ts.load_state_cell(tmp_path, "base", "chat", "lmsys23k", "context", LAYER)


def test_conv_index_fail_loud_on_nonconforming_id():
    assert ts.conv_index("s5000") == 5000
    with pytest.raises(ValueError, match="convention"):
        ts.conv_index("conv-000")


# ---------------------------------------------------------------------------
# resolve_turnstore_trees (delta a2) — ordered pair + 50-store acceptance
# ---------------------------------------------------------------------------
V2_COMBOS = [
    ("chat", "gsm8k_train_full"),
    ("chat", "if11k"),
    ("chat", "lmsys23k"),
    ("chat", "math7500"),
    ("chat", "sft11k"),
    ("chat", "uf11k"),
    ("naturalistic", "lmsys23k"),
]
V1_COMBOS = [
    ("chat", "gsm8k_test1319"),
    ("chat", "gsm8k_train5k"),
    ("chat", "lmsys5k"),
    ("naturalistic", "lmsys5k"),
]
STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]


def _fixture_stores(combos: list[tuple[str, str]], prefix: str) -> list[dict]:
    return [
        {
            "stage": st,
            "render": rd,
            "corpus": cp,
            "tree_path": f"pre/{prefix}_{st}_{rd}_{cp}",
        }
        for st in STAGES
        for rd, cp in combos
    ]


def test_resolve_turnstore_trees_ordered_pair_and_standalone():
    v2 = _fixture_stores(V2_COMBOS, "turnstore_v2")
    v1 = _fixture_stores(V1_COMBOS, "turnstore")
    trees = enc.resolve_turnstore_trees("base", "chat", "lmsys23k", v2_stores=v2, v1_stores=v1)
    assert trees == [
        "pre/turnstore_base_chat_lmsys5k",
        "pre/turnstore_v2_base_chat_lmsys23k",
    ]
    assert enc.resolve_turnstore_trees("sft", "chat", "if11k", v2_stores=v2, v1_stores=v1) == [
        "pre/turnstore_v2_sft_chat_if11k"
    ]
    # The parked v1 robustness arm never concatenates.
    assert enc.resolve_turnstore_trees(
        "base", "chat", "lmsys5k", generation="v1", v1_stores=v1, v2_stores=v2
    ) == ["pre/turnstore_base_chat_lmsys5k"]
    with pytest.raises(FileNotFoundError, match="No realized"):
        enc.resolve_turnstore_trees("base", "chat", "nope", v2_stores=v2, v1_stores=v1)


def test_registered_grain_35_cells_consume_50_stores():
    """Plan v11 delta (a2) acceptance: the 35 (stage, combo) cells consume
    exactly 50 stores (35 v2 + 15 wave-1 concat sources; gsm8k_test1319's 5
    v1 stores stay OUT)."""
    v2 = _fixture_stores(V2_COMBOS, "turnstore_v2")
    v1 = _fixture_stores(V1_COMBOS, "turnstore")
    assert len(v2) == 35
    consumed: set[str] = set()
    for cell in v2:
        trees = enc.resolve_turnstore_trees(
            cell["stage"], cell["render"], cell["corpus"], v2_stores=v2, v1_stores=v1
        )
        assert len(trees) == (2 if cell["corpus"] in ts.V2_CONCAT_SOURCES else 1)
        consumed |= set(trees)
    assert len(consumed) == 50
    assert not any("gsm8k_test1319" in t for t in consumed)


def test_collect_cells_corpus_filter_exact_membership_both_grains(monkeypatch):
    """The repeatable --corpus filter (two-grain smoke, crash-fix 2026-08-06)
    matches EXACT corpus names via list membership, resolves BOTH grains in
    one pass, and normalizes a direct str caller so `in` can never do
    substring matching (`"if11k" in "xx_if11k_xx"`-class false hits)."""
    v2 = _fixture_stores(V2_COMBOS, "turnstore_v2")
    v1 = _fixture_stores(V1_COMBOS, "turnstore")

    def fake_enum(revision=None, generation="v2"):
        return v2 if generation == "v2" else v1

    def fake_fetch_sidecars(tree_path: str, revision, max_workers) -> list[dict]:
        return []

    monkeypatch.setattr(enc, "_stage_render_corpus_turnstores", fake_enum)
    monkeypatch.setattr(gg, "_fetch_store_sidecars", fake_fetch_sidecars)

    cells = gg.collect_cells_from_hub(["base"], "chat", ["gsm8k_train_full", "if11k"], None, 1)
    assert sorted(c["corpus"] for c in cells) == ["gsm8k_train_full", "if11k"]
    # BOTH grains reach resolution: concat consumes two stores, plain-v2 one.
    assert {c["corpus"]: len(c["stores"]) for c in cells} == {
        "gsm8k_train_full": 2,
        "if11k": 1,
    }

    # A str caller is normalized to exact match — never substring membership.
    cells = gg.collect_cells_from_hub(["base"], "chat", "if11k", None, 1)
    assert [c["corpus"] for c in cells] == ["if11k"]
    with pytest.raises(SystemExit, match="no registered cells"):
        gg.collect_cells_from_hub(["base"], "chat", "xx_if11k_xx", None, 1)


def test_encode_cell_loader_refuses_extension_only_concat():
    """The retired defect is now unrepresentable: a concat corpus with a
    single resolved tree refuses instead of silently encoding extension-only."""
    with pytest.raises(ValueError, match=r"extension-only|REGISTERED grain"):
        enc._load_turnstore_state_cell(
            {"stage": "base", "render": "chat", "corpus": "lmsys23k", "tree_path": "pre/x"},
            state="answer",
            layer=LAYER,
        )


# ---------------------------------------------------------------------------
# P0 grain gate core (delta d)
# ---------------------------------------------------------------------------
def _gate_sidecar(
    idxs: list[int],
    shard_index: int = 0,
    *,
    include_convention_keys: bool = True,
    with_shas: bool = True,
    sha_of=lambda i: f"sha-{i}",
    model_id: str = "meta-llama/Llama-3.1-8B",
) -> dict:
    sc = {
        "shard_index": shard_index,
        "n_conversations": len(idxs),
        "conv_ids": [f"s{i}" for i in idxs],
        "model_id": model_id,
        "expected_layers": N_LAYERS,
        "expected_hidden": HIDDEN,
        "shapes": {
            "slots": [[2, N_LAYERS, HIDDEN] for _ in idxs],
            "profiles": [[2, N_LAYERS, HIDDEN] for _ in idxs],
        },
    }
    if include_convention_keys:
        sc["convention"] = "committed"
        sc["offset_override"] = None
    if with_shas:
        sc["prompt_shas"] = [sha_of(i) for i in idxs]
    return sc


def _gate_cell(n_wave1: int = 30, n_ext: int = 20, **kw) -> dict:
    return {
        "stage": "base",
        "render": "chat",
        "corpus": "lmsys23k",
        "stores": [
            {
                "tree_path": "pre/turnstore_base_chat_lmsys5k",
                "sidecars": [_gate_sidecar(list(range(n_wave1)), 0, with_shas=False, **kw)],
            },
            {
                "tree_path": "pre/turnstore_v2_base_chat_lmsys23k",
                "sidecars": [_gate_sidecar(list(range(5000, 5000 + n_ext)), 0)],
            },
        ],
    }


def test_gate_cells_pass_manifest_fields():
    manifest = gg.gate_cells([_gate_cell()], d_in=8)
    assert manifest["verdict"] == "PASS"
    (row,) = manifest["cells"]
    assert row["realized_n"] == 50 and row["concat"] is True and row["boundary"] == 5000
    # EXACT per-fold n_train through the production fold constructor.
    conv = [f"s{i}" for i in range(30)] + [f"s{i}" for i in range(5000, 5020)]
    folds = ts.group_fold_ids(conv, n_folds=5, seed=0)
    expected = [int(50 - c) for c in np.bincount(folds, minlength=5)]
    assert row["per_fold_n_train"] == expected
    assert row["min_per_fold_n_train"] == min(expected)
    assert row["selected_convention"] == "primal"  # min n_tr ~ 40 > d_in=8
    assert row["keep_rate_vs_n_target"] == pytest.approx(50 / 23_000)
    assert manifest["sha_join"]["lmsys23k"]["n_mismatches"] == 0


def test_gate_cells_regime_contradiction_flags_fail():
    manifest = gg.gate_cells([_gate_cell()], d_in=4096)  # 50 rows << 4096
    assert manifest["verdict"] == "FAIL"
    assert manifest["regime_contradictions"]
    assert manifest["cells"][0]["selected_convention"] == "gram-dual"


def test_gate_cells_pre_d2_store_logged():
    manifest = gg.gate_cells([_gate_cell(include_convention_keys=False)], d_in=8)
    assert manifest["verdict"] == "PASS"
    assert manifest["pre_d2_stores"] == ["turnstore_base_chat_lmsys5k"]


def test_gate_cells_boundary_violation_fails():
    cell = _gate_cell()
    cell["stores"][1]["sidecars"][0]["conv_ids"][0] = "s10"  # extension row < boundary
    manifest = gg.gate_cells([cell], d_in=8)
    assert manifest["verdict"] == "FAIL"
    assert any("extension store" in f for f in manifest["assert_failures"])


def test_gate_cells_cross_store_sha_join_mismatch_fails():
    base = _gate_cell()
    sft = json.loads(json.dumps(base))
    sft["stage"] = "sft"
    sft["stores"][1]["tree_path"] = "pre/turnstore_v2_sft_chat_lmsys23k"
    sft["stores"][1]["sidecars"][0]["prompt_shas"][1] = "sha-DRIFTED"
    manifest = gg.gate_cells([base, sft], d_in=8)
    assert manifest["verdict"] == "FAIL"
    assert any("prompt-sha JOIN mismatch" in f for f in manifest["assert_failures"])
    assert manifest["sha_join"]["lmsys23k"]["n_mismatches"] == 1


def _run_gate_main(monkeypatch, tmp_path, cells, extra_argv=()):
    monkeypatch.setattr(gg, "collect_cells_from_hub", lambda *a, **k: cells)
    out = tmp_path / "grain_manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["issue2061_grain_gate.py", "--all-cells", "--output", str(out), *extra_argv]
    )
    rc = gg.main()
    return rc, out


def test_gate_main_exits_nonzero_on_regime_contradiction(monkeypatch, tmp_path):
    """The P0 gate's brief-mandated fails-loud pin: nonzero exit (rc=3) when
    realized n contradicts the declared regime — the manifest is still
    written (it IS the gate report)."""
    rc, out = _run_gate_main(monkeypatch, tmp_path, [_gate_cell()], ["--d-in", "4096"])
    assert rc == gg.EXIT_REGIME_CONTRADICTION
    manifest = json.loads(out.read_text())
    assert manifest["verdict"] == "FAIL" and manifest["regime_contradictions"]


def test_gate_main_exits_nonzero_on_assert_failure(monkeypatch, tmp_path):
    cell = _gate_cell()
    cell["stores"][0]["sidecars"][0]["conv_ids"].append("s99")  # sidecar self-mismatch
    rc, out = _run_gate_main(monkeypatch, tmp_path, [cell], ["--d-in", "8"])
    assert rc == gg.EXIT_ASSERT_FAILURE
    assert json.loads(out.read_text())["verdict"] == "FAIL"


def test_gate_main_pass_exit_zero(monkeypatch, tmp_path):
    rc, out = _run_gate_main(monkeypatch, tmp_path, [_gate_cell()], ["--d-in", "8"])
    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["verdict"] == "PASS"
    assert manifest["meta"]["git_commit"]  # reproducibility metadata rides along


def test_gate_main_expect_n_cells_mismatch(monkeypatch, tmp_path):
    rc, _ = _run_gate_main(
        monkeypatch, tmp_path, [_gate_cell()], ["--d-in", "8", "--expect-n-cells", "35"]
    )
    assert rc == gg.EXIT_ASSERT_FAILURE


# ---------------------------------------------------------------------------
# Smoke acceptance (delta e)
# ---------------------------------------------------------------------------
def _accept_fixture(
    tmp_path, *, convention="primal", lo=-3.0, hi=8.0, n=50, n_ext_high=0, drop_fields=()
):
    r2 = tmp_path / "r2"
    r2.mkdir(exist_ok=True)
    row = {
        "feature_id": 0,
        "R2": 0.1,
        "n_test_total": n,
        "convention": convention,
        "n_at_low_edge": 0,
        "n_at_high_edge": 0,
        "lambda_grid_log10_lo": lo,
        "lambda_grid_log10_hi": hi,
        "n_lambda": round((hi - lo) / 0.5) + 1,
        "n_ext_low": 0,
        "n_ext_high": n_ext_high,
        "regularization_limited": False,
    }
    for k in drop_fields:
        row.pop(k)
    (r2 / f"base_chat_lmsys23k_context_L{gg.LAYER}.jsonl").write_text(json.dumps(row) + "\n")
    manifest = {
        "cells": [{"stage": "base", "render": "chat", "corpus": "lmsys23k", "realized_n": 50}]
    }
    mp = tmp_path / "grain_manifest.json"
    mp.write_text(json.dumps(manifest))
    return r2, mp


def test_smoke_acceptance_pass(tmp_path):
    r2, mp = _accept_fixture(tmp_path)
    assert gg.check_smoke_acceptance(r2, mp) == []


def test_smoke_acceptance_accepts_recorded_extensions(tmp_path):
    # An extension is audited, not forbidden: base grid must still be v13.
    r2, mp = _accept_fixture(tmp_path, hi=9.0, n_ext_high=1)
    assert gg.check_smoke_acceptance(r2, mp) == []


def test_smoke_acceptance_flags_violations(tmp_path):
    r2, mp = _accept_fixture(tmp_path, convention="gram-dual")
    assert any("convention" in v for v in gg.check_smoke_acceptance(r2, mp))

    r2, mp = _accept_fixture(tmp_path, lo=-2.0, hi=4.0)
    assert any("v13" in v for v in gg.check_smoke_acceptance(r2, mp))

    r2, mp = _accept_fixture(tmp_path, n=49)
    assert any("realized n" in v for v in gg.check_smoke_acceptance(r2, mp))

    r2, mp = _accept_fixture(tmp_path, drop_fields=("n_at_high_edge",))
    assert any("audit fields" in v for v in gg.check_smoke_acceptance(r2, mp))

    empty = tmp_path / "empty"
    empty.mkdir()
    assert gg.check_smoke_acceptance(empty, mp) == [f"no P2 JSONL outputs under {empty}"]


def test_encode_cell_loader_concat_two_trees(tmp_path, monkeypatch):
    """Body coverage for the encode-side (hub-path) concat cell loader:
    two REAL fixture trees behind signature-mirroring hub fakes, canonical
    order + boundary assert + composition print (delta a2)."""
    w1 = tmp_path / "turnstore_base_chat_lmsys5k"
    ext = tmp_path / "turnstore_v2_base_chat_lmsys23k"
    _write_shard(w1, 0, [0, 1, 2], WAVE1_STEM)
    _write_shard(ext, 0, [5000, 5001], EXT_STEM, with_shas=True)
    by_tree = {"pre/w1": w1, "pre/ext": ext}

    def fake_hub_shard_files(tree_path: str, revision: str | None = None) -> list[str]:
        return [f"{tree_path}/{p.name}" for p in ts.enumerate_shards(by_tree[tree_path])]

    def fake_hf_hub_download(repo_id, filename, repo_type=None, revision=None):
        tree = filename.rsplit("/", 1)[0]
        return str(by_tree[tree] / Path(filename).name)

    monkeypatch.setattr(enc, "hub_shard_files", fake_hub_shard_files)
    monkeypatch.setattr(enc, "hf_hub_download", fake_hf_hub_download)
    turnstore = {
        "stage": "base",
        "render": "chat",
        "corpus": "lmsys23k",
        "tree_path": "pre/ext",
        "tree_paths": ["pre/w1", "pre/ext"],
    }
    x, conv_ids, info = enc._load_turnstore_state_cell(turnstore, state="answer", layer=LAYER)
    assert conv_ids == ["s0", "s1", "s2", "s5000", "s5001"]
    assert x.shape == (5, HIDDEN)
    assert info["concat"] is True and [p["n_rows"] for p in info["parts"]] == [3, 2]

    # Boundary asserts bind on the hub path too.
    _write_shard(ext, 1, [10], EXT_STEM, with_shas=True)  # extension row < boundary
    with pytest.raises(ValueError, match="extension store has"):
        enc._load_turnstore_state_cell(turnstore, state="answer", layer=LAYER)
