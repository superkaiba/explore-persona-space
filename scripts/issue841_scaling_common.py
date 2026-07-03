#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (Δ, ℓ, →) in scientific docstrings.
"""Shared constants + loaders for the issue #841 scaling-capture follow-up round.

The single manipulated variable of this round is the ridge/MLP fit-corpus SIZE
(all else held at the parent's values — plan v9 §3). This module carries the
scaling n-grid, the HF layout for the new capture + per-n ridge maps, the
nested fit-set builder, and the (de)serialize helpers the three scaling drivers
(``issue841_scaling_capture`` / ``_stage0`` / ``_stage1``) share.

Reuses the parent's ``issue841_common`` for the pass_b bundle load, the seed-42
split, reproducibility metadata, and atomic JSON writes — this module never
re-implements those. No Qwen weights are loaded here.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue841_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_841.maps import MLPMap, RidgeMap  # noqa: E402

logger = logging.getLogger("issue841_scaling")

# ── constants ──────────────────────────────────────────────────────────────────

# The scaling axis (plan v9 §11): nested fit-set sizes, 4000 = the parent anchor.
SCALING_NS = (4000, 10000, 25000, 50000, 100000)
# The parent bundle carries exactly N_PARENT contexts; the anchor fit is the
# parent's 4000-row seed-42 fit split (val/test are the parent's stored 500 each).
N_PARENT = 5000
N_ANCHOR_FIT = C.N_FIT  # 4000
N_NEW_CONTEXTS = 96000  # positions 5001..101000 of the deterministic LMSYS stream
N_STREAM_TOTAL = N_PARENT + N_NEW_CONTEXTS  # 101000
# lmsys-chat-1m repeats prompt STRINGS across rows, so ~0.8% of the raw new-pool
# stream positions collide with a parent-5000 string (763/96000 measured, crash-fix
# round). We over-stream by this margin and drop parent-colliding new prompts so the
# clean new pool still fills exactly N_NEW_CONTEXTS. 8000 ≈ 10× the observed collision
# rate; a shortfall past it hard-fails (never silently under-fills). Override for a
# denser-collision corpus via EPM_I841S_BACKFILL_MARGIN.
# 24000 (was 8000): lmsys-chat-1m collides with the parent-5000 strings at ~9.5%
# of stream INSTANCES (att-2 measured 9883 drops in 104000 raw), so filling 96000
# clean needs ~10100 margin at the observed rate; 24000 gives ~2.4x headroom.
N_STREAM_BACKFILL_MARGIN = int(os.environ.get("EPM_I841S_BACKFILL_MARGIN", "24000"))

# HF layout for this round's artifacts (data repo superkaiba1/explore-persona-space-data).
HF_SCALING_PREFIX = "issue841_scaling"
HF_CAPTURE_BUCKET = f"{HF_SCALING_PREFIX}/cx_last_shards"


def hf_ridge_maps_bucket(n: int) -> str:
    """HF path-in-repo prefix for the per-n fitted ridge maps (plan §6.5)."""
    return f"{HF_SCALING_PREFIX}/ridge_maps_n{n}"


def hf_mlp_maps_bucket(n: int) -> str:
    """HF path-in-repo prefix for the per-n fitted MLP maps (Stage-1 row-1 mlp class)."""
    return f"{HF_SCALING_PREFIX}/mlp_maps_n{n}"


# Local scratch/output paths (worktree). The capture tensor + per-n maps are
# re-downloadable caches under data/; the JSON + npz are durable under eval_results/.
CAPTURE_DIR = PROJECT_ROOT / "data" / "issue_841" / "scaling" / "cx_last_shards"
RIDGE_MAPS_DIR = PROJECT_ROOT / "data" / "issue_841" / "scaling" / "ridge_maps"
EVAL_SCALING_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "scaling-capture"
FIG_SCALING_SUBDIR = "issue_841/scaling-capture"

# ~7.6 GB per shard (< the plan's ≤8 GB cap): rows × 28 × 3584 × 4 bytes.
_BYTES_PER_ROW = C.EXPECTED_LAYERS * C.EXPECTED_HIDDEN * 4
SHARD_TARGET_BYTES = 7_500_000_000
SHARD_ROWS = SHARD_TARGET_BYTES // _BYTES_PER_ROW  # ≈ 18684


# ── capture shard (de)serialize ─────────────────────────────────────────────────


def shard_paths(capture_dir: Path) -> list[Path]:
    """Ordered list of existing cx_last shard files under ``capture_dir``."""
    return sorted(capture_dir.glob("cx_last_shard*.pt"))


def shard_boundaries(n_total: int, shard_rows: int = SHARD_ROWS) -> list[tuple[int, int, int]]:
    """``(shard_index, row_lo, row_hi)`` spans covering ``n_total`` rows."""
    starts = list(range(0, n_total, shard_rows))
    return [(i, lo, min(lo + shard_rows, n_total)) for i, lo in enumerate(starts)]


def _done_marker(capture_dir: Path, idx: int) -> Path:
    return capture_dir / f"cx_last_shard{idx:03d}.done.json"


def shard_is_done(capture_dir: Path, idx: int, lo: int, hi: int, capture_dtype: str) -> bool:
    """A shard is resumable-complete iff its ``.done.json`` matches span + dtype.

    Keying on ``capture_dtype`` means a bf16 rerun never reuses fp32 shards (and
    vice versa) — a dtype change invalidates every prior shard for this dir.
    """
    marker = _done_marker(capture_dir, idx)
    if not (marker.exists() and (capture_dir / f"cx_last_shard{idx:03d}.pt").exists()):
        return False
    with open(marker) as f:
        m = json.load(f)
    return (
        m.get("row_lo") == lo and m.get("row_hi") == hi and m.get("capture_dtype") == capture_dtype
    )


def write_one_shard(
    capture_dir: Path,
    idx: int,
    cx_chunk: np.ndarray,
    prompts_chunk: list[str],
    *,
    lo: int,
    hi: int,
    n_total: int,
    n_shards: int,
    source: str,
    capture_dtype: str,
    requested_dtype: str | None = None,
) -> Path:
    """Write ONE cx_last shard + its ``.done.json`` resume marker (shard-as-you-go).

    ``capture_dtype`` is the REALIZED forward precision (the label the resume key
    matches on); ``requested_dtype`` is what the caller asked for (defaults to the
    realized value). Both are recorded as ``realized_capture_dtype`` /
    ``requested_capture_dtype`` so a silent up/downcast is legible downstream, with
    ``capture_dtype`` kept as the realized alias for the resume-match key.

    Peak RAM stays ~one shard: the driver captures a shard's rows, calls this, and
    frees the chunk before the next shard. The marker (written LAST, after the .pt)
    is the resume signal — a crash mid-``torch.save`` leaves no marker, so the
    shard re-captures on resume.
    """
    assert cx_chunk.shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), cx_chunk.shape
    assert cx_chunk.shape[0] == len(prompts_chunk) == (hi - lo), (
        cx_chunk.shape[0],
        len(prompts_chunk),
        hi - lo,
    )
    requested = requested_dtype if requested_dtype is not None else capture_dtype
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"cx_last_shard{idx:03d}.pt"
    torch.save(
        {
            "cx_last": torch.from_numpy(np.ascontiguousarray(cx_chunk)),
            "prompts": prompts_chunk,
            "shard_index": idx,
            "n_shards": n_shards,
            "row_lo": lo,
            "row_hi": hi,
            "total_rows": n_total,
            "layers": list(range(C.EXPECTED_LAYERS)),
            "source": source,
            "capture_dtype": capture_dtype,
            "realized_capture_dtype": capture_dtype,
            "requested_capture_dtype": requested,
        },
        path,
    )
    with open(_done_marker(capture_dir, idx), "w") as f:
        json.dump(
            {
                "row_lo": lo,
                "row_hi": hi,
                "capture_dtype": capture_dtype,
                "realized_capture_dtype": capture_dtype,
                "requested_capture_dtype": requested,
            },
            f,
        )
    return path


def write_capture_manifest(
    capture_dir: Path,
    n_total: int,
    source: str,
    capture_dtype: str,
    metadata: dict,
    *,
    shard_rows: int = SHARD_ROWS,
    requested_dtype: str | None = None,
) -> Path:
    """Write the capture ``manifest.json`` after all shards are present (verifies spans).

    ``capture_dtype`` is the REALIZED forward precision; ``requested_dtype`` is what
    the caller asked for (defaults to the realized value). Both are recorded so a
    silent up/downcast is legible; ``capture_dtype`` stays the realized alias.
    """
    requested = requested_dtype if requested_dtype is not None else capture_dtype
    spans = []
    for idx, lo, hi in shard_boundaries(n_total, shard_rows):
        name = f"cx_last_shard{idx:03d}.pt"
        assert (capture_dir / name).exists(), f"manifest: shard {name} missing"
        spans.append({"shard": name, "row_lo": lo, "row_hi": hi})
    path = capture_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(
            {
                "total_rows": n_total,
                "n_shards": len(spans),
                "source": source,
                "capture_dtype": capture_dtype,
                "realized_capture_dtype": capture_dtype,
                "requested_capture_dtype": requested,
                "spans": spans,
                "metadata": metadata,
            },
            f,
            indent=2,
        )
    logger.info(
        "[capture] wrote manifest (%d shards, %d rows, dtype=%s) → %s",
        len(spans),
        n_total,
        capture_dtype,
        capture_dir,
    )
    return path


def write_capture_shards(
    cx_last: np.ndarray,
    prompts: list[str],
    source: str,
    metadata: dict,
    capture_dir: Path,
    *,
    shard_rows: int = SHARD_ROWS,
    capture_dtype: str = "synthetic",
    requested_dtype: str | None = None,
) -> list[Path]:
    """Convenience: write ALL shards from an in-memory array + the manifest (test/dry-run).

    Production capture uses the shard-as-you-go path (``write_one_shard`` +
    ``write_capture_manifest``) to bound RAM; this all-at-once form is for the
    ``--dry-run-io`` wiring test where the array is already tiny. ``requested_dtype``
    defaults to ``capture_dtype`` (synthetic realized == requested).
    """
    n = cx_last.shape[0]
    assert len(prompts) == n, (len(prompts), n)
    bounds = shard_boundaries(n, shard_rows)
    written = []
    for idx, lo, hi in bounds:
        written.append(
            write_one_shard(
                capture_dir,
                idx,
                cx_last[lo:hi],
                prompts[lo:hi],
                lo=lo,
                hi=hi,
                n_total=n,
                n_shards=len(bounds),
                source=source,
                capture_dtype=capture_dtype,
                requested_dtype=requested_dtype,
            )
        )
    write_capture_manifest(
        capture_dir,
        n,
        source,
        capture_dtype,
        metadata,
        shard_rows=shard_rows,
        requested_dtype=requested_dtype,
    )
    return written


def load_capture(capture_dir: Path) -> dict:
    """Load + concatenate the sharded new-context capture (local-first).

    Returns ``{"cx_last": (N,28,3584) fp32, "prompts": [str], "source": str,
    "capture_dtype": str, "realized_capture_dtype": str, "requested_capture_dtype":
    str}``. ``capture_dtype`` == ``realized_capture_dtype`` (the realized forward
    precision); both fall back to ``capture_dtype`` for a legacy manifest.
    Fail-loud on a missing manifest / row-count mismatch.
    """
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"capture manifest absent: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    parts, prompts = [], []
    for span in manifest["spans"]:
        blob = torch.load(capture_dir / span["shard"], weights_only=False)
        parts.append(blob["cx_last"].to(torch.float32).numpy())
        prompts.extend(blob["prompts"])
    cx_last = np.concatenate(parts, axis=0)
    assert cx_last.shape[0] == manifest["total_rows"], (cx_last.shape, manifest["total_rows"])
    assert cx_last.shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), cx_last.shape
    assert len(prompts) == cx_last.shape[0], (len(prompts), cx_last.shape[0])
    return {
        "cx_last": cx_last,
        "prompts": prompts,
        "source": manifest.get("source"),
        "capture_dtype": manifest.get("capture_dtype"),
        "realized_capture_dtype": manifest.get(
            "realized_capture_dtype", manifest.get("capture_dtype")
        ),
        "requested_capture_dtype": manifest.get(
            "requested_capture_dtype", manifest.get("capture_dtype")
        ),
    }


def fetch_capture_from_hf(capture_dir: Path) -> Path:
    """Download the cx_last shards + manifest from the HF data repo (fallback path).

    Local-first is the caller's job (``load_capture``); this is the HF-fetch leg
    for a fresh instance whose local scratch is empty. Fail-loud on any miss.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f
        for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_CAPTURE_BUCKET}/")
    ]
    if not files:
        raise FileNotFoundError(
            f"no capture shards at {C.HF_DATA_REPO}:{HF_CAPTURE_BUCKET}/ — run capture first"
        )
    capture_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        local = hf_hub_download(C.HF_DATA_REPO, filename=rel, repo_type="dataset")
        dst = capture_dir / Path(rel).name
        if not dst.exists():
            dst.symlink_to(Path(local).resolve())
    return capture_dir


def load_capture_local_or_hf(capture_dir: Path) -> dict:
    """Local-first → HF-fetch → fail-loud loader for the new-context capture."""
    if not (capture_dir / "manifest.json").exists():
        logger.info("[capture] local manifest absent; fetching from HF %s", HF_CAPTURE_BUCKET)
        fetch_capture_from_hf(capture_dir)
    return load_capture(capture_dir)


# ── nested fit-set builder ───────────────────────────────────────────────────────


def build_scaling_bundle(pass_b: dict, capture: dict) -> dict:
    """Assemble the nested-scaling context bundle from the parent + new capture.

    Returns a dict with:
      - ``fit_pool`` (N_ANCHOR_FIT + N_new, 28, 3584): the parent's 4000-row seed-42
        fit split FIRST (rows [:4000] = the always-⊂ anchor), then the new capture
        appended in stream order → ``fit(n) = fit_pool[:n]``.
      - ``val`` / ``test`` (500, 28, 3584): the parent's stored seed-42 val/test —
        NEVER re-split, so R²(n) is comparable across all n.
      - ``drift_window`` (N_ANCHOR_FIT, 28, 3584): the LATEST 4000 new-capture rows
        (stream positions ~97000-101000) for the position-drift diagnostic (§4.2).
      - ``split`` : the parent's seed-42 index sets (into the parent cx_last).
    """
    parent_cx = pass_b["cx_last"]  # (5000, 28, 3584)
    assert parent_cx.shape[0] == N_PARENT, parent_cx.shape
    split = C.make_split(
        N_PARENT, n_fit=C.N_FIT, n_val=C.N_INNER_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED
    )
    anchor_fit = parent_cx[split["fit"]]  # (4000, 28, 3584)
    new_cx = capture["cx_last"]  # (96000, 28, 3584)
    assert new_cx.shape[0] >= max(SCALING_NS) - N_ANCHOR_FIT, (
        f"new capture has {new_cx.shape[0]} rows; need ≥ {max(SCALING_NS) - N_ANCHOR_FIT}"
    )
    fit_pool = np.concatenate([anchor_fit, new_cx], axis=0)
    drift_window = new_cx[-N_ANCHOR_FIT:]  # latest-stream 4000-window
    return {
        "fit_pool": fit_pool,
        "val": parent_cx[split["val"]],
        "test": parent_cx[split["test"]],
        "drift_window": drift_window,
        "split": split,
    }


def fit_indices_for_n(n: int) -> np.ndarray:
    """The first-n row indices into ``fit_pool`` (nested: n' < n ⇒ subset)."""
    assert n <= N_ANCHOR_FIT + N_NEW_CONTEXTS, n
    return np.arange(n)


# ── per-n ridge-map (de)serialize ────────────────────────────────────────────────


def save_ridge_maps(maps_by_transition: dict[int, RidgeMap], path: Path) -> Path:
    """Persist all one-step RAW-target ridge maps for one n to a single ``.pt``.

    Stored fp32 (the RidgeMap dtype); Stage-1 reloads them for the transport curve.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(t): {
            "mu": m.mu.cpu(),
            "sd": m.sd.cpu(),
            "w": m.w.cpu(),
            "bias": m.bias.cpu(),
            "best_lam": float(m.best_lam),
            "sigma": float(m.sigma),
        }
        for t, m in maps_by_transition.items()
    }
    torch.save(payload, path)
    return path


def load_ridge_maps(path: Path, device: str) -> dict[int, RidgeMap]:
    """Reload the per-n ridge maps saved by ``save_ridge_maps`` onto ``device``."""
    blob = torch.load(path, weights_only=False)
    out: dict[int, RidgeMap] = {}
    for t_str, d in blob.items():
        out[int(t_str)] = RidgeMap(
            mu=d["mu"].to(device),
            sd=d["sd"].to(device),
            w=d["w"].to(device),
            bias=d["bias"].to(device),
            best_lam=float(d["best_lam"]),
            sigma=float(d["sigma"]),
        )
    return out


# ── per-n MLP-map (de)serialize (Stage-1 row-1 mlp transported class) ─────────────


def save_mlp_maps(params_by_transition: dict[int, dict], path: Path) -> Path:
    """Persist the per-transition RAW-target MLP params (from ``fit_split_mlps``).

    ``params_by_transition[t]`` is the numpy-array param dict (W1/b1/W2/b2/mu/sd)
    ``fit_split_mlps`` returns; Stage-1 rebuilds each into an ``MLPMap`` (sigma=1.0,
    raw target) for the row-1 mlp transported class. Stored keyed by str(transition).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(t): p for t, p in params_by_transition.items()}
    torch.save(payload, path)
    return path


def load_mlp_maps(path: Path, device: str) -> dict[int, MLPMap]:
    """Reload the per-n MLP maps saved by ``save_mlp_maps`` onto ``device`` (sigma=1.0 raw)."""
    blob = torch.load(path, weights_only=False)
    return {
        int(t_str): MLPMap.from_params(p, sigma=1.0, device=device) for t_str, p in blob.items()
    }
