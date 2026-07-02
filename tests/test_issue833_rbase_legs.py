"""Plan-v6 same-era delta — rbase-namespace writer <-> fit-loader round-trip (#833).

Pins the two-namespace contract the ``--legs-mode reextracted`` path depends on:

  (i)   ``build_leg_npz_payload(leg_mode="rbase")`` (the B1 writer's payload)
        round-trips through the fit driver's ``load_rbase_legs`` on a local
        streamer — arrays, hashes, and ``probe_idx`` intact;
  (ii)  ``build_cells_reextracted`` SUBSTITUTES the same-era legs (stale
        store-era v0/vplus replaced; context vectors preserved);
  (iii) an incomplete rbase namespace fails loud NAMING the missing cell;
  (iv)  ``build_leg_npz_payload(leg_mode="onpolicy")`` refuses to build without
        the store context / base hashes (the B2 invariants);
  (v)   ``resolve_context_source`` consumes the A0' c_C-parity verdict: auto +
        flagged summary → reextracted; auto + unflagged / missing → store (no
        raise); explicit store while flagged RAISES (round-4 reconciled Major —
        a parity-FAIL run must not fit on the FAILED old-store context).

Run: uv run pytest tests/test_issue833_rbase_legs.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import issue722_load_activations as loadact  # noqa: E402
from issue833_extract_onpolicy import build_leg_npz_payload  # noqa: E402
from issue833_fit_onpolicy import (  # noqa: E402
    RbaseLeg,
    build_cells_reextracted,
    load_rbase_legs,
    resolve_context_source,
)

HIDDEN = 8
GAUGE = {"r": 32, "lora_alpha": 64, "use_rslora": True}


def _write_rbase_cell(root: Path, beh: str, src: str, tcid: str, layer: int, rng) -> dict:
    """Write one synthetic rbase npz via the REAL writer payload; return the arrays."""
    v0 = rng.normal(size=HIDDEN).astype(np.float32)
    vplus = rng.normal(size=HIDDEN).astype(np.float32)
    payload = build_leg_npz_payload(
        leg_mode="rbase",
        v0_mean=v0,
        vplus_mean=vplus,
        shas=["a" * 64, "b" * 64],
        shas_base=None,
        probe_ids=[0, 2],  # probe 1 "empty" — compaction preserved via probe_idx
        behavior=beh,
        source=src,
        seed=42,
        tcid=tcid,
        layer=layer,
        gauge=GAUGE,
        gen_backend="vllm",
        stored_context=None,
    )
    cell_dir = root / beh / f"{src}_seed42"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cell_dir / f"{tcid}_L{layer}.npz", **payload)
    return {"v0": v0, "v_plus": vplus}


def test_rbase_writer_loader_roundtrip(tmp_path):
    """(i) B1 payload -> npz -> load_rbase_legs round-trip (local streamer)."""
    rng = np.random.default_rng(833)
    ref = _write_rbase_cell(tmp_path, "em", "src0", "tgt0", 7, rng)
    layout = loadact.list_store_layout_local(tmp_path, behaviors=("em",))
    streamer = loadact._Streamer(local_root=tmp_path)
    legs = load_rbase_legs(("em",), (7,), streamer, layout, hidden=HIDDEN)
    assert set(legs) == {("em", "src0", "tgt0", 7)}
    leg = legs[("em", "src0", "tgt0", 7)]
    np.testing.assert_allclose(leg.v0, ref["v0"], rtol=1e-6)
    np.testing.assert_allclose(leg.v_plus, ref["v_plus"], rtol=1e-6)
    assert leg.resp_sha256 == ["a" * 64, "b" * 64]
    assert leg.probe_idx == [0, 2]  # ORIGINAL probe ids survive compaction


def test_build_cells_reextracted_substitutes_legs():
    """(ii) stale store-era v0/vplus replaced; c0/cplus/family preserved."""
    rng = np.random.default_rng(1)
    c0, cplus = rng.normal(size=HIDDEN), rng.normal(size=HIDDEN)
    stale = loadact.CellRecord(
        behavior="em",
        source_cid="src0",
        target_cid="tgt0",
        layer=7,
        c0=c0,
        cplus=cplus,
        v0=np.zeros(HIDDEN),
        vplus=np.zeros(HIDDEN),
        family="fam0",
    )
    true_v0, true_vp = rng.normal(size=HIDDEN), rng.normal(size=HIDDEN)
    rlegs = {
        ("em", "src0", "tgt0", 7): RbaseLeg(
            behavior="em",
            source_cid="src0",
            target_cid="tgt0",
            layer=7,
            v0=true_v0,
            v_plus=true_vp,
            resp_sha256=["a" * 64],
            probe_idx=[0],
        )
    }
    (out,) = build_cells_reextracted([stale], rlegs)
    np.testing.assert_allclose(out.v0, true_v0)
    np.testing.assert_allclose(out.vplus, true_vp)
    np.testing.assert_allclose(out.c0, c0)
    np.testing.assert_allclose(out.cplus, cplus)
    assert out.family == "fam0"


def test_incomplete_rbase_namespace_fails_loud_naming_cell():
    """(iii) --legs-mode reextracted requires the rbase namespace COMPLETE."""
    stale = loadact.CellRecord(
        behavior="em",
        source_cid="srcX",
        target_cid="tgtY",
        layer=14,
        c0=np.zeros(HIDDEN),
        cplus=np.zeros(HIDDEN),
        v0=np.zeros(HIDDEN),
        vplus=np.zeros(HIDDEN),
        family="fam0",
    )
    with pytest.raises(RuntimeError, match=r"srcX.*tgtY|extract-rbase"):
        build_cells_reextracted([stale], {})


def test_onpolicy_payload_requires_context_and_base_hashes():
    """(iv) the B2 payload cannot be built without store context / base hashes."""
    v = np.zeros(HIDDEN, dtype=np.float32)
    with pytest.raises(AssertionError):
        build_leg_npz_payload(
            leg_mode="onpolicy",
            v0_mean=v,
            vplus_mean=v,
            shas=["a" * 64],
            shas_base=None,
            probe_ids=[0],
            behavior="em",
            source="src0",
            seed=42,
            tcid="tgt0",
            layer=7,
            gauge=GAUGE,
            gen_backend="vllm",
            stored_context=None,
        )


def _write_a0_summary(tmp_path: Path, flag: bool) -> Path:
    """Write a minimal A0' parity summary carrying the contingency flag."""
    p = tmp_path / "parity" / "a0_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"reextract_context_vectors": flag}))
    return p


def test_context_source_auto_flagged_resolves_reextracted(tmp_path):
    """(v-i) auto + flagged summary -> reextracted (the contingency engages)."""
    p = _write_a0_summary(tmp_path, True)
    resolved, reason = resolve_context_source("auto", p)
    assert resolved == "reextracted"
    assert "reextract_context_vectors=True" in reason


def test_context_source_auto_unflagged_resolves_store(tmp_path):
    """(v-ii) auto + unflagged summary -> store (parity PASS licenses the store)."""
    p = _write_a0_summary(tmp_path, False)
    resolved, reason = resolve_context_source("auto", p)
    assert resolved == "store"
    assert "reextract_context_vectors=False" in reason


def test_context_source_auto_missing_summary_resolves_store(tmp_path):
    """(v-iii) auto + MISSING summary -> store, warn-not-crash (smoke/manual)."""
    p = tmp_path / "parity" / "a0_summary.json"
    resolved, reason = resolve_context_source("auto", p)
    assert resolved == "store"
    assert "MISSING" in reason and str(p) in reason


def test_context_source_explicit_store_flagged_raises(tmp_path):
    """(v-iv) explicit store while the summary flags the contingency RAISES."""
    p = _write_a0_summary(tmp_path, True)
    with pytest.raises(ValueError, match="reextract_context_vectors"):
        resolve_context_source("store", p)
