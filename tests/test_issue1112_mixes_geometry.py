"""#1112 — mix derivation fail-fast partition + geometry aggregator e2e (CPU).

The geometry test runs the REAL run_geometry pass end-to-end on tiny synthetic
capture stores in the production on-disk shape (schema_version 1 pooled.pt),
so the VM-side analysis driver's whole path executes in CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from explore_persona_space.experiments.issue_1112 import geometry as geo
from explore_persona_space.experiments.issue_1112 import mixes

# ── mixes ─────────────────────────────────────────────────────────────────────


def _row(tag: str, i: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"{tag} question {i}?"}],
        "completion": [{"role": "assistant", "content": f"{tag} answer {i}."}],
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_derive_row_roles_partition_and_failfast(tmp_path):
    pos = [_row("pos", i) for i in range(2)]
    cn = [_row("cn", i) for i in range(2)]
    gen = [_row("gen", i) for i in range(4)]
    mix = [pos[0], cn[0], gen[0], gen[1], pos[1], cn[1], gen[2], gen[3]]
    roles = mixes.derive_row_roles(mix, pos, cn, gen, expected={"pos": 2, "cn": 2, "generic": 4})
    assert roles == ["pos", "cn", "generic", "generic", "pos", "cn", "generic", "generic"]

    # unmatched row -> blocks
    with pytest.raises(ValueError, match="NO role source"):
        mixes.derive_row_roles(
            [*mix, _row("alien", 0)], pos, cn, gen, expected={"pos": 2, "cn": 2, "generic": 4}
        )
    # wrong partition -> blocks
    with pytest.raises(ValueError, match="partition mismatch"):
        mixes.derive_row_roles(mix, pos, cn, gen, expected={"pos": 3, "cn": 1, "generic": 4})
    # ambiguous role sources -> blocks
    with pytest.raises(ValueError, match="overlap"):
        mixes.derive_row_roles(mix, pos, [*cn, pos[0]], gen, expected=None)


def test_derive_syco_mixes_writes_posonly_and_generic(tmp_path):
    pos = [_row("pos", i) for i in range(2)]
    cn = [_row("cn", i) for i in range(2)]
    gen = [_row("gen", i) for i in range(4)]
    mix = [pos[0], cn[0], gen[0], gen[1], pos[1], cn[1], gen[2], gen[3]]
    paths = {}
    for name, rows in (("mix", mix), ("pos", pos), ("cn", cn), ("gen", gen)):
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)
    # production expects 20/20/40 — patch for the tiny fixture
    orig = mixes.EXPECTED_PARTITION.copy()
    mixes.EXPECTED_PARTITION.update({"pos": 2, "cn": 2, "generic": 4})
    try:
        man = mixes.derive_syco_mixes(
            paths["mix"], paths["pos"], paths["cn"], paths["gen"], tmp_path / "out"
        )
    finally:
        mixes.EXPECTED_PARTITION.clear()
        mixes.EXPECTED_PARTITION.update(orig)
    posonly = mixes._read_jsonl(tmp_path / "out" / "c3_posonly_mix.jsonl")
    generic = mixes._read_jsonl(tmp_path / "out" / "c3_generic_only.jsonl")
    assert len(posonly) == 6 and len(generic) == 4
    assert all("cn" not in r["prompt"][0]["content"] for r in posonly)
    assert man["posonly"]["sha256"] and man["generic_only"]["sha256"]


# ── geometry ──────────────────────────────────────────────────────────────────

HID = 16
LAYERS = [0, 1]
CONTEXTS = ["src", "negA", "negB"]
NQ = 4


def _store(cell: str, dose: str, seed: int, tmp_path) -> None:
    rng = np.random.default_rng(seed)
    row_meta = [{"context_id": c, "question_idx": q} for c in CONTEXTS for q in range(NQ)]
    n = len(row_meta)
    arms = {}
    for arm in ("prefix", "context", "response"):
        per_layer = {}
        for li in LAYERS:
            if arm == "prefix":
                base_rows = rng.standard_normal((len(CONTEXTS), HID))
                X = np.repeat(base_rows, NQ, axis=0)  # prefix depends only on context
            else:
                X = rng.standard_normal((n, HID))
            per_layer[li] = torch.from_numpy(X).to(torch.float16)
        arms[arm] = per_layer
    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": "sycophancy",
        "row_meta": row_meta,
        "arms": arms,
        "metadata": {"fixture": True},
    }
    out = tmp_path / "capture" / cell / dose / "pooled.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out)


def test_run_geometry_end_to_end(tmp_path):
    torch.manual_seed(0)
    cells = ["s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos"]
    for i, cell in enumerate(cells):
        _store(cell, "selected", seed=10 + i, tmp_path=tmp_path)
    _store("base_syco", "base", seed=99, tmp_path=tmp_path)
    rb = torch.randn(len(LAYERS), HID, dtype=torch.float32)
    rb_path = tmp_path / "rb.pt"
    torch.save({"rb": rb}, rb_path)

    payload = geo.run_geometry(
        tmp_path / "capture",
        tmp_path / "out",
        cells_doses=[(c, "selected") for c in cells],
        base_store_by_behavior={
            "sycophancy": tmp_path / "capture" / "base_syco" / "base" / "pooled.pt"
        },
        behavior_by_cell={c: "sycophancy" for c in cells},
        selected_dose_by_cell={c: "selected" for c in cells},
        rb_by_behavior={"sycophancy": rb_path},
        layers=LAYERS,
        n_boot=25,
    )
    # one record per (cell, dose, arm, layer)
    assert len(payload["records"]) == len(cells) * 3 * len(LAYERS)
    rec = payload["records"]["s1_lora_neg/selected/response/L0"]
    for k in ("top_share_lambda", "pr_lambda", "rank_k_at_90", "mu_norm", "boot_ci"):
        assert k in rec, k
    assert rec["n_rows"] == len(CONTEXTS) * NQ
    assert rec["cos_top_to_rb"] is not None and "random_cos_ci" in rec
    # prefix-arm degeneracy framing: structural unique rows == n contexts
    pre = payload["records"]["s1_lora_neg/selected/prefix/L0"]
    assert pre["n_unique_rows_structural"] == len(CONTEXTS)
    assert (
        payload["records"]["s1_lora_neg/selected/response/L0"]["n_unique_rows_structural"]
        == len(CONTEXTS) * NQ
    )
    # H1/H2 paired diffs present with CI + paired tag
    d = payload["cross_cell_diffs"]["H1_method_ftneg_vs_loraneg"]
    dd = d["reads"]["response/L0"]["diff_rank_k_at_90"]
    assert dd["resampling"] == "paired" and dd["ci_low"] <= dd["ci_high"]
    # H3 interaction computed only when layer 14 captured — absent here (L0/L1)
    assert payload["h3_interaction"] == {}
    # bootstrap matrices persisted
    mats = torch.load(
        tmp_path / "out" / "bootstrap_matrices" / "s1_lora_neg_selected.pt", weights_only=False
    )
    assert mats["response/L0/rank_k_at_90"].shape == (25,)
    # sensitivity/ceiling skipped (needs layer 14) — geometry JSON exists
    assert (tmp_path / "out" / "geometry_per_cell.json").exists()
