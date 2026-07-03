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

from explore_persona_space.experiments.issue_841.maps import RidgeMap  # noqa: E402

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

# HF layout for this round's artifacts (data repo superkaiba1/explore-persona-space-data).
HF_SCALING_PREFIX = "issue841_scaling"
HF_CAPTURE_BUCKET = f"{HF_SCALING_PREFIX}/cx_last_shards"


def hf_ridge_maps_bucket(n: int) -> str:
    """HF path-in-repo prefix for the per-n fitted ridge maps (plan §6.5)."""
    return f"{HF_SCALING_PREFIX}/ridge_maps_n{n}"


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


def write_capture_shards(
    cx_last: np.ndarray,
    prompts: list[str],
    source: str,
    metadata: dict,
    capture_dir: Path,
    *,
    shard_rows: int = SHARD_ROWS,
) -> list[Path]:
    """Shard ``cx_last`` (N,28,3584) + aligned prompts into ≤8 GB ``.pt`` pieces.

    Each shard is a self-describing dict; a sibling ``manifest.json`` records the
    global row count + ordered shard list + per-shard row spans so ``load_capture``
    can verify completeness. Returns the written shard paths.
    """
    n = cx_last.shape[0]
    assert cx_last.shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), cx_last.shape
    assert len(prompts) == n, (len(prompts), n)
    capture_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    spans: list[dict] = []
    shard_starts = list(range(0, n, shard_rows))
    n_shards = len(shard_starts)
    for si, lo in enumerate(shard_starts):
        hi = min(lo + shard_rows, n)
        path = capture_dir / f"cx_last_shard{si:03d}.pt"
        torch.save(
            {
                "cx_last": torch.from_numpy(np.ascontiguousarray(cx_last[lo:hi])),
                "prompts": prompts[lo:hi],
                "shard_index": si,
                "n_shards": n_shards,
                "row_lo": lo,
                "row_hi": hi,
                "total_rows": n,
                "layers": list(range(C.EXPECTED_LAYERS)),
                "source": source,
            },
            path,
        )
        spans.append({"shard": path.name, "row_lo": lo, "row_hi": hi})
        written.append(path)
    with open(capture_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "total_rows": n,
                "n_shards": n_shards,
                "source": source,
                "spans": spans,
                "metadata": metadata,
            },
            f,
            indent=2,
        )
    logger.info("[capture] wrote %d shards (%d rows) → %s", n_shards, n, capture_dir)
    return written


def load_capture(capture_dir: Path) -> dict:
    """Load + concatenate the sharded new-context capture (local-first).

    Returns ``{"cx_last": (N,28,3584) fp32, "prompts": [str], "source": str}``.
    Fail-loud on a missing manifest / row-count mismatch / shape drift.
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
    return {"cx_last": cx_last, "prompts": prompts, "source": manifest.get("source")}


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
