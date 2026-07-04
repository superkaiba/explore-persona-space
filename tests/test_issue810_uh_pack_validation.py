#!/usr/bin/env python3
"""Issue #810 `user-header-newline-summary` round-2 — uh-pack production validation.

Regression tests for the r1 CONCERNs (fail pre-fix: ``validate_uh_pack`` did not
exist and neither call site validated the pack):

1. ``uh-pack-meta-validation-readout`` — a PRODUCTION-marked pack missing one UH
   row / one context must raise BEFORE the readout fit loop (Codex test i).
2. ``uh-pack-validation-bootstrap`` — a non-smoke pack with row 2 missing one
   context AND a non-smoke 24-layer pack must both fail BEFORE any bootstrap
   output (Codex test ii; r1 checked rows[0] only + min()-truncated the layers).

Wiring asserts tie the gated helper to the LIVE dispatched call sites
(``_resolve_rows_and_sources`` / ``_vs_mean_rows``) so the gate can never go
hollow (code-style § "Verification gates test the live dispatched path").
Pure-Python pack fixtures — no GPU / no HF.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from issue810_common import (  # noqa: E402
    DEFAULT_MODEL,
    UH_SUMMARY_NAMES,
    UhPackValidationError,
    validate_uh_pack,
)

N_LAYERS = 28  # production capture-layer count (EXPECTED_LAYERS)
HIDDEN = 8  # tiny stand-in hidden dim — validation never reads shape[1]


def _make_pack(
    n_ctx: int = 6,
    n_layers: int = N_LAYERS,
    smoke: bool = False,
    model: str = DEFAULT_MODEL,
):
    """Tiny synthetic uh pack matching `_load_uh_summaries`'s return shape."""
    ctx_ids = [f"ctx{i:02d}" for i in range(n_ctx)]
    rng = np.random.default_rng(0)
    rows = {
        r: {c: rng.standard_normal((n_layers, HIDDEN)).astype(np.float32) for c in ctx_ids}
        for r in UH_SUMMARY_NAMES
    }
    cov = {r: dict.fromkeys(ctx_ids, 3) for r in UH_SUMMARY_NAMES}
    meta = {
        "smoke": smoke,
        "context_ids": list(ctx_ids),
        "capture_layers": list(range(n_layers)),
        "model": model,
    }
    return rows, cov, meta, ctx_ids


def _validate(rows, cov, meta, ctx_ids, requested=None, n_layers: int = N_LAYERS):
    validate_uh_pack(
        rows,
        cov,
        meta,
        requested_rows=requested or list(UH_SUMMARY_NAMES),
        ctx_ids=ctx_ids,
        expected_capture_layers=list(range(n_layers)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: a full production pack passes.
# ─────────────────────────────────────────────────────────────────────────────
def test_full_production_pack_passes():
    rows, cov, meta, ctx_ids = _make_pack()
    _validate(rows, cov, meta, ctx_ids)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Codex test (i) — readout: production pack missing one UH row / one context
# raises BEFORE fitting.
# ─────────────────────────────────────────────────────────────────────────────
def test_production_pack_missing_row_raises():
    rows, cov, meta, ctx_ids = _make_pack()
    del rows[UH_SUMMARY_NAMES[0]]
    with pytest.raises(UhPackValidationError, match="absent from the uh pack"):
        _validate(rows, cov, meta, ctx_ids)


def test_production_pack_missing_context_tensor_raises():
    rows, cov, meta, ctx_ids = _make_pack()
    del rows[UH_SUMMARY_NAMES[0]][ctx_ids[-1]]
    with pytest.raises(UhPackValidationError, match="lack a tensor"):
        _validate(rows, cov, meta, ctx_ids)


def test_production_pack_zero_coverage_context_raises():
    rows, cov, meta, ctx_ids = _make_pack()
    cov[UH_SUMMARY_NAMES[3]][ctx_ids[2]] = 0
    with pytest.raises(UhPackValidationError, match="zero coverage"):
        _validate(rows, cov, meta, ctx_ids)


def test_production_pack_meta_context_gap_raises():
    rows, cov, meta, ctx_ids = _make_pack()
    meta["context_ids"] = meta["context_ids"][:-1]
    with pytest.raises(UhPackValidationError, match="context_ids missing"):
        _validate(rows, cov, meta, ctx_ids)


def test_smoke_or_premeta_pack_refused_on_production_path():
    rows, cov, meta, ctx_ids = _make_pack(smoke=True)
    with pytest.raises(UhPackValidationError, match="smoke-provenance"):
        _validate(rows, cov, meta, ctx_ids)
    # A pre-meta pack (meta keys resolve to None) is refused identically.
    rows, cov, _meta, ctx_ids = _make_pack()
    none_meta = {"smoke": None, "context_ids": None, "capture_layers": None, "model": None}
    with pytest.raises(UhPackValidationError, match="smoke-provenance"):
        _validate(rows, cov, none_meta, ctx_ids)


def test_wrong_model_raises():
    rows, cov, meta, ctx_ids = _make_pack(model="Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(UhPackValidationError, match="model mismatch"):
        _validate(rows, cov, meta, ctx_ids)


def test_readout_call_site_routes_through_validate_uh_pack():
    """The LIVE readout entrypoint gates on validate_uh_pack (non-smoke path)."""
    import issue810_fit_readout as fr

    src = inspect.getsource(fr._resolve_rows_and_sources)
    assert "validate_uh_pack(" in src, "readout call site no longer validates the uh pack"
    assert "args.smoke" in src, "readout smoke relaxation branch removed"
    # The gated symbol IS the shared helper (identity, not a same-named twin).
    assert fr.validate_uh_pack is validate_uh_pack


# ─────────────────────────────────────────────────────────────────────────────
# Codex test (ii) — bootstrap: non-smoke pack with row 2 missing one context AND
# a non-smoke 24-layer pack both fail BEFORE any output.
# ─────────────────────────────────────────────────────────────────────────────
def test_bootstrap_row2_missing_one_context_raises():
    """r1 checked rows[0] only — a later-row gap must now refuse pre-output."""
    rows, cov, meta, ctx_ids = _make_pack()
    row2 = UH_SUMMARY_NAMES[1]
    del rows[row2][ctx_ids[0]]
    cov[row2][ctx_ids[0]] = 0
    with pytest.raises(UhPackValidationError, match=row2):
        _validate(rows, cov, meta, ctx_ids)


def test_bootstrap_nonsmoke_24_layer_pack_raises():
    """r1 min()-truncated the layer window — non-smoke truncation must refuse."""
    rows, cov, meta, ctx_ids = _make_pack(n_layers=24)
    with pytest.raises(UhPackValidationError, match="capture_layers mismatch"):
        _validate(rows, cov, meta, ctx_ids, n_layers=N_LAYERS)


def test_nonsmoke_truncated_tensor_axis_raises_even_with_full_meta():
    """A pack whose META claims 28 layers but whose tensors carry 24 still refuses."""
    rows, cov, meta, ctx_ids = _make_pack(n_layers=24)
    meta["capture_layers"] = list(range(N_LAYERS))  # lying meta
    with pytest.raises(UhPackValidationError, match="truncated layer axis"):
        _validate(rows, cov, meta, ctx_ids, n_layers=N_LAYERS)


def test_bootstrap_call_site_routes_through_validate_uh_pack():
    """The LIVE bootstrap --vs mean path gates on validate_uh_pack (non-smoke)."""
    import issue810_bootstrap_deltaskill as bd

    src = inspect.getsource(bd._vs_mean_rows)
    assert "validate_uh_pack(" in src, "bootstrap call site no longer validates the uh pack"
    assert 'meta.get("smoke")' in src, "bootstrap smoke-provenance branch removed"
    # The pack-driven layer-window truncation (`n_layers_pack` min()) must live
    # ONLY in the smoke branch: the validated path pins the full capture_layers
    # axis.
    validated_branch = src.split('if not meta.get("smoke"):')[1].split("else:")[0]
    assert "n_layers_pack" not in validated_branch, (
        "pack-driven layer truncation leaked into the validated (non-smoke) branch"
    )
    smoke_branch = src.split('if not meta.get("smoke"):')[1].split("else:")[1]
    assert "n_layers_pack" in smoke_branch, "smoke-only layer truncation path removed"
