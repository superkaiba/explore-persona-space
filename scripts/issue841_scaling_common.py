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
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

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

# Token-budget batching for the GPU capture (crash-fix round 4: attempt 4 OOMed in
# the Qwen2 MLP forward at 78.75/79.25 GiB). A FIXED batch_size × LEFT-PAD pads every
# sequence to the batch max, so ONE long lmsys prompt (4-8k tokens) inflated a 32-seq
# batch to 32×8k retained activations + MLP forward workspace. Cap each batch by
# PADDED tokens (n_seqs × max_len_in_batch ≤ TOKEN_BUDGET) AND a hard seq cap. 65536
# padded tokens ≈ worst-case ~16 GB of 28-layer bf16 hidden-state retention + MLP
# workspace, comfortably under 80 GiB. NO truncation — the parent regime (issue779
# `_tokenize_left_pad`: padding=True, no max_length/truncation) had none, so any
# truncation here would be a second variable. Override via the env vars below.
TOKEN_BUDGET = int(os.environ.get("EPM_I841S_TOKEN_BUDGET", "65536"))
MAX_SEQS_PER_BATCH = int(os.environ.get("EPM_I841S_MAX_SEQS_PER_BATCH", "48"))

# Host-RAM chunking for the stage-0 MLP fit battery (crash-fix cycle 5: attempt 6
# SIGKILL 137 ~26 min into stage-0). At n=100k each transition's SplitMLPGroup holds
# an (n, 3584) fp32 X_train + Y_train, so building ALL 27 transitions' groups at once
# is ~77 GB of copies + the ~40 GB cx pool → exceeds the 170 GB host. Fit transitions
# in chunks of MLP_GROUP_CHUNK groups (build → fit → free → gc), bounding live copies
# to ~MLP_GROUP_CHUNK × Y_train. NOTE: since #926 (4dfcba056f) fit_batched_split_mlp
# seeds each group under split_group_init_seed(seed, group.key), which depends only on
# (seed, key) — never on batch position or chunking — so the fit is bit-identical
# ACROSS chunk sizes as well as DETERMINISTIC at a fixed one. MLP_GROUP_CHUNK is a
# pinned RAM knob recorded in the stage-0 output, NOT a nuisance seed. See
# mlp_scaling's PARTITION INVARIANCE (#926) block in issue841_scaling_stage0.py.
MLP_GROUP_CHUNK = int(os.environ.get("EPM_I841S_MLP_GROUP_CHUNK", "6"))


def rss_gib() -> float:
    """Current process resident set size in GiB (Linux /proc/self/status VmRSS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024**2)  # KB → GiB
    except Exception:
        pass
    return float("nan")


def mem_total_available_gib() -> tuple[float, float]:
    """(MemTotal, MemAvailable) in GiB from /proc/meminfo; (nan, nan) if unreadable."""
    vals: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    vals[line.split(":")[0]] = int(line.split()[1]) / (1024**2)  # KB → GiB
    except Exception:
        pass
    return vals.get("MemTotal", float("nan")), vals.get("MemAvailable", float("nan"))


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


# ── HF public-storage LFS-quota (#541/#552) overflow routing ──────────────────────
# The account-wide PUBLIC-storage quota gates ONLY the LFS endpoint; PRIVATE-repo LFS
# is a separate quota with headroom. When the public quota is full, this round routes
# the LFS artifacts (cx_last .pt shards, ridge/mlp map .pt) to the PRIVATE overflow
# repo, keeps the non-LFS files (manifests, .done.json) + an OVERFLOW_POINTER.json
# breadcrumb on the canonical PUBLIC data repo, and threads a deviation record into the
# capture metadata/summary/sentinel. Mirrors hub.DEFAULT_OVERFLOW_REPO (asserted below).
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
OVERFLOW_REASON = "public LFS quota 403 (#541/#552 LFS wall)"


def _overflow_repo_for_bucket(
    canonical_repo: str, bucket: str, repo_type: str = "dataset"
) -> str | None:
    """Return the overflow repo id iff ``<bucket>/OVERFLOW_POINTER.json`` exists on the
    canonical repo (i.e. this bucket's LFS was rerouted), else None. Fail-soft on any
    listing/parse error → None (falls back to the canonical repo, backward-compatible)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    pointer_rel = f"{bucket.rstrip('/')}/OVERFLOW_POINTER.json"
    try:
        if pointer_rel not in list_repo_files(canonical_repo, repo_type=repo_type):
            return None
        local = hf_hub_download(canonical_repo, filename=pointer_rel, repo_type=repo_type)
        return json.loads(Path(local).read_text()).get("overflow_repo")
    except Exception as e:
        logger.warning("[overflow] pointer probe for %s:%s failed (%s)", canonical_repo, bucket, e)
        return None


def hf_download_pt_maybe_overflow(
    canonical_repo: str, bucket: str, fname: str, repo_type: str = "dataset"
) -> str:
    """Download ``<bucket>/<fname>`` to the local HF cache, preferring the PRIVATE
    overflow repo for a rerouted ``.pt`` (when ``<bucket>/OVERFLOW_POINTER.json`` is
    present on ``canonical_repo``). Non-``.pt`` files always come from the canonical
    repo. Fail-loud: raises if a pointer says overflow but the shard is not there."""
    from huggingface_hub import hf_hub_download

    rel = f"{bucket.rstrip('/')}/{fname}"
    if fname.endswith(".pt"):
        overflow = _overflow_repo_for_bucket(canonical_repo, bucket, repo_type)
        if overflow:
            return hf_hub_download(overflow, filename=rel, repo_type=repo_type)
    return hf_hub_download(canonical_repo, filename=rel, repo_type=repo_type)


def _write_overflow_pointer_dataset(
    canonical_repo: str, path_in_repo: str, overflow_repo: str, reason: str
) -> None:
    """Upload an ``OVERFLOW_POINTER.json`` breadcrumb to the canonical DATASET repo at
    ``<path_in_repo>/`` (non-LFS — succeeds over the public LFS quota). FAIL-LOUD: the
    pointer is LOAD-BEARING — ``fetch_capture_from_hf`` / ``hf_download_pt_maybe_overflow``
    read it to locate the rerouted ``.pt`` on the overflow repo, so a silently-missing
    pointer makes a fresh-instance durability fetch treat the bucket as public and return
    a partial/empty shard set while the run reads green (Codex #841 v11 review). Raises
    RuntimeError on any upload miss or exception. (hub._write_overflow_pointer targets
    repo_type='model'; this is the dataset twin with the {overflow_repo, path_in_repo,
    ts, reason} schema.)"""
    from explore_persona_space.orchestrate import hub

    payload = {
        "overflow_repo": overflow_repo,
        "path_in_repo": path_in_repo.rstrip("/"),
        "ts": time.time(),
        "reason": reason,
    }
    tmp = Path(tempfile.gettempdir()) / f"OVERFLOW_POINTER_{os.getpid()}.json"
    dest = (
        f"{path_in_repo.rstrip('/')}/OVERFLOW_POINTER.json"
        if path_in_repo
        else "OVERFLOW_POINTER.json"
    )
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        url = hub._upload(tmp, canonical_repo, "dataset", dest, upload_as_file=True)
    finally:
        tmp.unlink(missing_ok=True)
    if not url:
        raise RuntimeError(
            f"overflow pointer write to {canonical_repo}:{dest} failed (verification miss) — "
            f"the pointer is load-bearing (fetch reads it to locate rerouted .pt on "
            f"{overflow_repo}); refusing to report LFS-reroute success without it"
        )
    logger.info("[upload] overflow pointer → %s/%s → %s", canonical_repo, dest, overflow_repo)


def upload_split_lfs_to_overflow(
    local_path: Path,
    path_in_repo: str,
    *,
    lfs_glob: str = "*.pt",
    canonical_repo: str | None = None,
    reason: str = OVERFLOW_REASON,
) -> dict:
    """Split an artifact dir/file across repos to route around the public LFS quota 403.

    LFS files (``lfs_glob``, default ``*.pt``) → the PRIVATE overflow repo (created
    private if missing); non-LFS files (manifests, ``.done.json``) → the canonical
    PUBLIC dataset repo unchanged; an ``OVERFLOW_POINTER.json`` breadcrumb → the
    canonical repo at ``path_in_repo``. Every LFS + non-LFS upload is FAIL-LOUD
    (raises on any verification miss); the pointer is fail-soft. Logs the fix-engaged
    signal at the FIRST rerouted upload. Returns a deviation dict for the caller to
    record in metadata/summary/sentinel. ``local_path`` may be a directory (globbed
    non-recursively) or a single file.
    """
    from fnmatch import fnmatch

    from explore_persona_space.orchestrate import hub

    assert OVERFLOW_REPO == hub.DEFAULT_OVERFLOW_REPO, (OVERFLOW_REPO, hub.DEFAULT_OVERFLOW_REPO)
    canonical = canonical_repo or C.HF_DATA_REPO
    local_path = Path(local_path)
    files = (
        sorted(p for p in local_path.iterdir() if p.is_file())
        if local_path.is_dir()
        else [local_path]
    )
    lfs = [p for p in files if fnmatch(p.name, lfs_glob)]
    nonlfs = [p for p in files if not fnmatch(p.name, lfs_glob)]
    prefix = path_in_repo.rstrip("/")

    for f in nonlfs:  # non-LFS → canonical public repo (unchanged path), fail-loud
        dest = f"{prefix}/{f.name}"
        if not hub._upload(f, canonical, "dataset", dest, upload_as_file=True):
            raise RuntimeError(f"non-LFS upload failed: {f} → {canonical}:{dest}")

    for i, f in enumerate(lfs):  # LFS → PRIVATE overflow repo, fail-loud
        if i == 0:
            logger.info(
                "[upload] LFS artifacts → overflow repo %s (public quota 403 recovery); "
                "pointer written",
                OVERFLOW_REPO,
            )
        dest = f"{prefix}/{f.name}"
        if not hub._upload(f, OVERFLOW_REPO, "dataset", dest, upload_as_file=True, private=True):
            raise RuntimeError(f"LFS overflow upload failed: {f} → {OVERFLOW_REPO}:{dest}")

    # The pointer is LOAD-BEARING when LFS was rerouted (the fetch path reads it to
    # locate the .pt on the overflow repo), so this is FAIL-LOUD: it RAISES on a write
    # miss, short-circuiting the success return below. Never report a rerouted-LFS
    # success without a landed pointer (Codex #841 v11 review).
    if lfs:
        _write_overflow_pointer_dataset(canonical, prefix, OVERFLOW_REPO, reason)
    return {
        "overflow_repo": OVERFLOW_REPO,
        "canonical_repo": canonical,
        "path_in_repo": prefix,
        "lfs_glob": lfs_glob,
        "reason": reason,
        "n_lfs": len(lfs),
        "n_nonlfs": len(nonlfs),
        "ts": time.time(),
    }


def fetch_capture_from_hf(capture_dir: Path) -> Path:
    """Download the cx_last shards + manifest from HF (fallback path, overflow-aware).

    Local-first is the caller's job (``load_capture``); this is the HF-fetch leg for a
    fresh instance whose local scratch is empty. When the LFS shards were rerouted to
    the private overflow repo (``OVERFLOW_POINTER.json`` present on the public bucket,
    #541 quota recovery), the ``.pt`` shards are fetched from OVERFLOW and the non-LFS
    files from the public repo — so a durability read never returns a partial public
    shard set. Fail-loud on any miss.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    public_files = [
        f
        for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_CAPTURE_BUCKET}/")
    ]
    if not public_files:
        raise FileNotFoundError(
            f"no capture artifacts at {C.HF_DATA_REPO}:{HF_CAPTURE_BUCKET}/ — run capture first"
        )
    overflow = _overflow_repo_for_bucket(C.HF_DATA_REPO, HF_CAPTURE_BUCKET)
    capture_dir.mkdir(parents=True, exist_ok=True)

    def _link(repo: str, rel: str) -> None:
        local = hf_hub_download(repo, filename=rel, repo_type="dataset")
        dst = capture_dir / Path(rel).name
        if not dst.exists():
            dst.symlink_to(Path(local).resolve())

    for rel in public_files:  # non-.pt (+ .pt when NOT rerouted) from the public repo
        if rel.endswith(".pt") and overflow:
            continue  # shard lives on overflow; fetched below
        if rel.endswith("OVERFLOW_POINTER.json"):
            continue  # breadcrumb, not a capture artifact
        _link(C.HF_DATA_REPO, rel)
    if overflow:
        shard_files = [
            f
            for f in list_repo_files(overflow, repo_type="dataset")
            if f.startswith(f"{HF_CAPTURE_BUCKET}/") and f.endswith(".pt")
        ]
        if not shard_files:
            raise FileNotFoundError(
                f"OVERFLOW_POINTER present but no .pt shards at {overflow}:{HF_CAPTURE_BUCKET}/"
            )
        for rel in shard_files:
            _link(overflow, rel)
    return capture_dir


def capture_complete_on_hf(
    n_new: int, requested_dtype: str, repo_type: str = "dataset"
) -> tuple[bool, str]:
    """Is a COMPLETE capture already on HF — manifest covering ``n_new`` at
    ``requested_dtype`` AND every span's ``.pt`` shard present (on the overflow repo
    via the pointer, else the public repo)? Returns ``(complete, detail)``. Fail-soft:
    ANY probe error → ``(False, reason)`` so the caller falls through to a full
    re-capture rather than crashing. Used by the capture-skip short-circuit (#841
    attempt 6 recovery: capture + all shard uploads succeeded before stage-0 OOMed,
    so the full capture lives on HF and a relaunch fetches instead of re-capturing)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    manifest_rel = f"{HF_CAPTURE_BUCKET}/manifest.json"
    try:
        public = set(list_repo_files(C.HF_DATA_REPO, repo_type=repo_type))
    except Exception as e:
        return False, f"public listing failed ({e})"
    if manifest_rel not in public:
        return False, "no manifest on HF"
    try:
        manifest = json.loads(
            Path(
                hf_hub_download(C.HF_DATA_REPO, filename=manifest_rel, repo_type=repo_type)
            ).read_text()
        )
    except Exception as e:
        return False, f"manifest fetch/parse failed ({e})"
    hf_dtype = manifest.get("realized_capture_dtype", manifest.get("capture_dtype"))
    total = manifest.get("total_rows")
    spans = manifest.get("spans") or []
    if hf_dtype != requested_dtype:
        return False, f"dtype {hf_dtype} != requested {requested_dtype}"
    if total is None or int(total) < n_new:
        return False, f"coverage short (HF total_rows={total} < {n_new})"
    if not spans:
        return False, "manifest has no spans"
    overflow = _overflow_repo_for_bucket(C.HF_DATA_REPO, HF_CAPTURE_BUCKET, repo_type)
    shard_repo = overflow or C.HF_DATA_REPO
    try:
        shard_files = {
            f
            for f in list_repo_files(shard_repo, repo_type=repo_type)
            if f.startswith(f"{HF_CAPTURE_BUCKET}/")
        }
    except Exception as e:
        return False, f"shard listing on {shard_repo} failed ({e})"
    missing = [s["shard"] for s in spans if f"{HF_CAPTURE_BUCKET}/{s['shard']}" not in shard_files]
    if missing:
        return (
            False,
            f"{len(missing)}/{len(spans)} shards missing on {shard_repo} (e.g. {missing[0]})",
        )
    return True, f"total_rows={total} dtype={hf_dtype}, {len(spans)} shards on {shard_repo}"


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
