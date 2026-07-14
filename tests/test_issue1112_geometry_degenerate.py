"""#1112 crash-fix r3 — explicit degenerate-record path in the geometry aggregator.

Pins the p11_geometry smoke crash class (GCP att-20260707-205546): a
single-context capture collapses the PREFIX-arm Δx cloud to < 2
structurally-unique rows, so the row-centered spectrum is identically zero and
the #653 spectral fail-fast (``Σσ² == 0``) killed the whole driver. The #1112
geometry path (``geometry.analyze_cell``) now:

- emits an EXPLICIT ``degenerate: true`` record (unique-row count, reason
  string, null spectral DVs, μ/cos(μ, r_B) still reported) for a
  MECHANICALLY-expected degenerate cloud — structural unique rows < 2;
- still RAISES on UNEXPECTED degeneracy (≥ 2 unique rows zeroing out) via the
  unweakened #653 fail-fast;
- guards the cross-cell paired diffs (a degenerate side yields an explicit
  degenerate diff entry, never a KeyError on missing per-draw matrices).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from explore_persona_space.experiments.issue_1112 import geometry as geo

HID = 16
LAYERS = [0, 1]
NQ = 2


def _store_dict(
    cell: str,
    dose: str,
    seed: int,
    contexts: list[str],
    *,
    zero_response: bool = False,
) -> dict:
    """Schema-exact (v1) capture store; prefix rows constant per context."""
    rng = np.random.default_rng(seed)
    row_meta = [{"context_id": c, "question_idx": q} for c in contexts for q in range(NQ)]
    n = len(row_meta)
    arms: dict = {}
    for arm in ("prefix", "context", "response"):
        per_layer = {}
        for li in LAYERS:
            if arm == "prefix":
                base_rows = rng.standard_normal((len(contexts), HID))
                X = np.repeat(base_rows, NQ, axis=0)  # prefix depends only on context
            elif zero_response and arm == "response":
                X = np.zeros((n, HID))  # identical rows despite >=2 unique (ctx, q) keys
            else:
                X = rng.standard_normal((n, HID))
            per_layer[li] = torch.from_numpy(X).to(torch.float16)
        arms[arm] = per_layer
    return {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": "sycophancy",
        "row_meta": row_meta,
        "arms": arms,
        "metadata": {"fixture": True},
    }


def _write_store(tmp_path, store: dict) -> None:
    out = tmp_path / "capture" / store["cell"] / store["dose"] / "pooled.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out)


def _run(tmp_path, cells: list[str]) -> dict:
    rb_path = tmp_path / "rb.pt"
    torch.save({"rb": torch.randn(len(LAYERS), HID, dtype=torch.float32)}, rb_path)
    return geo.run_geometry(
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
        n_boot=8,
    )


def test_single_context_prefix_emits_explicit_degenerate_record(tmp_path):
    """1-context capture (the crash shape) -> degenerate prefix records, no raise."""
    contexts = ["src"]
    _write_store(tmp_path, _store_dict("s3_fullft_neg", "selected", 7, contexts))
    _write_store(tmp_path, _store_dict("base_syco", "base", 8, contexts))
    payload = _run(tmp_path, ["s3_fullft_neg"])

    pre = payload["records"]["s3_fullft_neg/selected/prefix/L0"]
    assert pre["degenerate"] is True
    assert pre["unique_rows"] == 1 and pre["n_unique_rows_structural"] == 1
    assert "unique row(s) < 2" in pre["degenerate_reason"]
    assert pre["top_share_lambda"] is None
    assert pre["pr_lambda"] is None
    assert pre["rank_k_at_90"] is None
    assert pre["boot_ci"] is None
    assert isinstance(pre["mu_norm"], float)  # μ stays well-defined
    assert pre["cos_top_to_rb"] is None and isinstance(pre["cos_mu_to_rb"], float)
    assert "random_cos_ci" in pre

    # context/response arms have 2 unique (ctx, q) rows -> normal spectral path
    for arm in ("context", "response"):
        rec = payload["records"][f"s3_fullft_neg/selected/{arm}/L0"]
        assert rec["degenerate"] is False
        assert isinstance(rec["top_share_lambda"], float)
        assert rec["boot_ci"] is not None

    # no per-draw matrices for the degenerate (arm, layer); response present
    mats = torch.load(
        tmp_path / "out" / "bootstrap_matrices" / "s3_fullft_neg_selected.pt",
        weights_only=False,
    )
    assert not any(k.startswith("prefix/") for k in mats)
    assert "response/L0/rank_k_at_90" in mats

    # the payload (null DVs included) serialized cleanly
    written = json.loads((tmp_path / "out" / "geometry_per_cell.json").read_text())
    assert written["records"]["s3_fullft_neg/selected/prefix/L1"]["degenerate"] is True


def test_unexpected_zero_cloud_still_raises(tmp_path):
    """>= 2 unique rows that zero out is UNEXPECTED -> the #653 fail-fast raises."""
    contexts = ["src", "negA"]
    _write_store(
        tmp_path, _store_dict("s3_fullft_neg", "selected", 7, contexts, zero_response=True)
    )
    _write_store(tmp_path, _store_dict("base_syco", "base", 8, contexts, zero_response=True))
    with pytest.raises(ValueError, match="degenerate spectrum"):
        _run(tmp_path, ["s3_fullft_neg"])


def test_cross_cell_diff_guard_on_degenerate_side(tmp_path):
    """A degenerate side yields an explicit degenerate diff entry (no KeyError)."""
    contexts = ["src"]
    _write_store(tmp_path, _store_dict("s3_fullft_neg", "selected", 7, contexts))
    _write_store(tmp_path, _store_dict("s1_lora_neg", "selected", 9, contexts))
    _write_store(tmp_path, _store_dict("base_syco", "base", 8, contexts))
    payload = _run(tmp_path, ["s3_fullft_neg", "s1_lora_neg"])

    reads = payload["cross_cell_diffs"]["H1_method_ftneg_vs_loraneg"]["reads"]
    pre = reads["prefix/L0"]
    assert pre["degenerate"] is True
    assert sorted(pre["degenerate_sides"]) == ["a", "b"]
    assert isinstance(pre["cos_mu"], float)
    assert "diff_rank_k_at_90" not in pre
    # nondegenerate arms keep the full paired-diff read
    resp = reads["response/L0"]
    assert resp["diff_rank_k_at_90"]["resampling"] == "paired"
