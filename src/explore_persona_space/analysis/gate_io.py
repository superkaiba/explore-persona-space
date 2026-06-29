"""Trained-store loader for issue #665 Phase 3 (gate_io — net-new module).

Streams the #664 trained activation store cell-by-cell from HF (download →
load → consume → free), enforcing the #600 mirror-divergence guard: the loader
ASSERTS `sha256(tensors.pt) == meta.sha256_tensors` ON THE LIVE LOAD PATH and
RAISES on mismatch. The HF mirror can be a silently different generation than
the copy verified at plan time — the crash IS the signal (CLAUDE.md fail-fast).

Reuses `issue664_aggregate_gate.gate_per_layer` / `probe_split_floor` (the
canonical ĝ^real + probe-split floor) — imported, never copied.

Phase 4 (#666) imports this module unchanged (#665 plan §5 land-freeze
ordering — shared write-once code surface).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow importing the scripts/-resident issue664 modules (same convention as
# the issue664_* entrypoints — they live next to scripts/, not under src/).
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from huggingface_hub import hf_hub_download  # noqa: E402

# REUSE the canonical ĝ^real + floor estimators (never reimplement — plan §5).
from issue664_aggregate_gate import gate_per_layer, probe_split_floor  # noqa: E402,F401

DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"

EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
EXPECTED_N_CONTEXTS = 50


def _sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Stream-hash a file (1 MiB chunks — tensors.pt is ~2 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class StoreCell:
    """One loaded #664 store cell. `tensors` holds the torch tensors; `meta` the
    sidecar json; `source_idx` the index of the source-anchor context in the
    context axis (derived from `target_context_roles`)."""

    cell: str
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    source_idx: int
    source_ctx_id: str
    behavior: str
    source: str
    arm: str
    dose: str
    seed: int
    sha256_tensors: str
    _tensors_path: str  # local cache path (for free())

    def free(self) -> None:
        """Drop the big tensors + delete the local cache file to bound the VM
        analysis footprint (stream cell-by-cell — plan §8 peak ≤ 6 GB)."""
        self.tensors.clear()
        gc.collect()
        with contextlib.suppress(OSError):
            os.remove(self._tensors_path)


def _resolve_source_idx(tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> tuple[int, str]:
    """Find the source-anchor context index from meta.target_context_roles —
    exactly the recipe `issue664_aggregate_gate.process_cell` uses."""
    ctx_ids = list(tensors["context_ids"])
    roles = meta["target_context_roles"]
    anchor_ids = [cid for cid, r in roles.items() if r == "source-anchor"]
    if len(anchor_ids) != 1:
        raise ValueError(f"expected exactly 1 source-anchor role, got {anchor_ids!r}")
    source_ctx_id = anchor_ids[0]
    return ctx_ids.index(source_ctx_id), source_ctx_id


def load_cell(cell: str, *, verify_sha: bool = True) -> StoreCell:
    """Download + load one #664 store cell, asserting the sha256 content-identity
    guard (#600) on the live load path.

    Args:
        cell: the cell slug (e.g. ``bm_default_contra_d1_seed42``).
        verify_sha: when True (default, the PRODUCTION path) the recomputed
            sha256 of the downloaded tensors.pt MUST equal meta.sha256_tensors
            or the loader RAISES. NEVER disable on a production read — the guard
            is the only protection against a stale HF mirror generation.

    Returns:
        a StoreCell (call .free() when done to bound footprint).
    """
    tp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{cell}/tensors.pt", repo_type="dataset")
    mp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{cell}/meta.json", repo_type="dataset")
    with open(mp) as f:
        meta = json.load(f)

    expected_sha = meta.get("sha256_tensors")
    if verify_sha:
        # #600 mirror-divergence guard — IN CODE on the live load path, RAISE on
        # mismatch (NOT a warn, NOT behind a debug flag). The crash IS the signal.
        if not expected_sha:
            raise ValueError(
                f"[{cell}] meta.json has no sha256_tensors — cannot verify content "
                "identity (#600 guard requires it). Refusing to consume the cell."
            )
        actual_sha = _sha256_file(tp)
        if actual_sha != expected_sha:
            raise ValueError(
                f"[{cell}] sha256(tensors.pt) MISMATCH — HF mirror divergence (#600): "
                f"recomputed {actual_sha} != meta.sha256_tensors {expected_sha}. "
                "The mirror is a different generation than the verified copy; halting."
            )

    tensors = torch.load(tp, map_location="cpu")
    # shape sanity (plan §12 assumptions 1/7/17)
    vp = tensors["v_plus"]
    assert vp.ndim == 3, f"[{cell}] v_plus must be (C,L,d): {tuple(vp.shape)}"
    _n_ctx, n_layer, hidden = vp.shape
    assert n_layer == EXPECTED_LAYERS, f"[{cell}] n_layers {n_layer} != {EXPECTED_LAYERS}"
    assert hidden == EXPECTED_HIDDEN, f"[{cell}] hidden {hidden} != {EXPECTED_HIDDEN}"

    source_idx, source_ctx_id = _resolve_source_idx(tensors, meta)
    return StoreCell(
        cell=cell,
        tensors=tensors,
        meta=meta,
        source_idx=source_idx,
        source_ctx_id=source_ctx_id,
        behavior=meta["behavior"],
        source=meta["source"],
        arm=meta["arm"],
        dose=meta["dose"],
        seed=int(meta["seed"]),
        sha256_tensors=expected_sha,
        _tensors_path=tp,
    )


def load_sigma_c(lam_check: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Load #658's whitened-gate metric Σc (28, 3584, 3584) from HF.

    Returns (sigma_c float64 ndarray, meta dict with n / capture_layers).
    Asserts the 28-layer alignment (plan §12 item 17).
    """
    sp = hf_hub_download(
        DATA_REPO, "issue658_theory_assumptions/store/sigma_c.pt", repo_type="dataset"
    )
    obj = torch.load(sp, map_location="cpu")
    sigma_c = obj["sigma_c"]
    assert sigma_c.shape[0] == EXPECTED_LAYERS, (
        f"sigma_c first dim {sigma_c.shape[0]} != {EXPECTED_LAYERS} layers"
    )
    assert sigma_c.shape[1] == sigma_c.shape[2] == EXPECTED_HIDDEN, (
        f"sigma_c (d,d) must be ({EXPECTED_HIDDEN},{EXPECTED_HIDDEN}): {tuple(sigma_c.shape)}"
    )
    meta = {"n": int(obj.get("n", -1)), "capture_layers": obj.get("capture_layers")}
    if lam_check and meta["n"] > 0:
        # n < d makes Sigma_c singular -> the lambda floor is load-bearing (informational).
        assert meta["n"] < EXPECTED_HIDDEN, (
            f"expected Sigma_c captured at n<d (singular); got n={meta['n']} >= d={EXPECTED_HIDDEN}"
        )
    return sigma_c.numpy().astype(np.float64), meta
