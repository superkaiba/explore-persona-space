#!/usr/bin/env python3
"""Issue #779 inline follow-up (``fitter-fair-comparison-n1m``): fits at n_train up
to ~1,000,000, provenance-aware, over the LMSYS+WildChat n1m corpus.

Extends the n50k fits (``issue779_ffc_n50k_fits.py``) along ``n`` to four
subset-fit points, reusing the SAME target (``v(x)`` mean-response profile), the
SAME map input (``cx_last`` last prompt token), the SAME variance-weighted
held-out R2 metric, and a BYTE-IDENTICAL val/test (the ORIGINAL round's
``fixed_split(5000, 3600, 400, 1000, 42)``, val/test index shas hard-asserted
equal to the pinned constants). New contexts (the n1m pool) enter TRAIN only.

Four subset-fit points (deterministic, provenance-aware selection; realized N =
min(target, available pool)):

  * ``lmsys_150k``   — 150,000 train, PURE LMSYS (orig-train + n1m-new lmsys only).
  * ``lmsys_500k``   — 500,000 train, PURE LMSYS.
  * ``mixed_500k``   — 500,000 train, STRATIFIED to the full pool's lmsys:wildchat
                       ratio (the corpus-mix control at matched n).
  * ``mixed_1m``     — the WHOLE mixed train pool (target 1,000,000; realized is the
                       full usable pool, ~orig-train 3.6k + 960k new).

Five predictors per point:

  * ``ridge``          — PRIMAL ridge, X^TX / X^TY accumulated STREAMING in fp64 on
                         the device over train-row blocks (never the (n, H) design
                         materialized at once), one eigh of (H, H), val-lambda over
                         ``LAMBDAS_N1M``. Numerically identical to the n50k primal
                         ridge, just block-accumulated.
  * ``mlp_w8192``      — full-dim MLP width 8192 (the protocol arm), MINIBATCHED
                         AdamW on the device (the n50k full-batch battery cannot
                         hold n=1M), internal-val early stop.
  * ``mlp_w32768``     — full-dim MLP width 32768 (the CAPACITY arm; flagged
                         ``capacity_arm: true`` in fit_meta), same minibatched fit.
  * ``residual_skip``  — primal ridge base + minibatched MLP (width 8192) on the
                         residual (strictly nests the linear map).
  * ``krr_nystrom``    — RBF kernel ridge via Nystrom (``--krr-nystrom-centers``
                         landmarks; exact KRR is a (n, n) kernel, infeasible at
                         n=1M), with the Nystrom feature Gram Phi^T Phi accumulated
                         STREAMING over train blocks. (gamma, lambda) val-selected.

Nystrom validation (gate): before the KRR fits, ``_validate_nystrom_vs_exact``
runs BOTH this driver's Nystrom fitter AND the n50k EXACT KRR
(``N50.fit_krr_exact``) on the SAME deterministic 50,000-row train slice + the
pinned val/test, and asserts ``|R2_nystrom - R2_exact| <= --krr-validate-tol``
(default 0.01) — a larger gap FAILS LOUD (the Nystrom fitter is biased). The
committed n50k exact anchor (0.8076 wide-grid / 0.8066 small-grid) is recorded
for reference. Requires ``--device cuda`` (the 50k^2 exact kernel).

Output (``eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json``): per
(point, predictor) whole-map R2 + mean cosine + 1000-resample bootstrap 95% CI +
fit_meta, the split (pinned + realized shas + byte-identical flag), the layer, the
Nystrom-vs-exact validation block, and reproducibility metadata. Per-(point,
predictor) checkpoint — ``--resume`` skips completed cells (guarded on layer +
seed so a cross-layer/seed resume never mixes rows).

The n1m capture (~82 GB) is NOT materialized whole: cx_last + v_x at the chosen
layer are STREAM-REDUCED from the HF capture chunks (download one chunk -> slice
the layer -> free). The combined per-layer X+Y (~28 GB at n~=963k) is held in
RAM for subset indexing — route this driver to a GPU pod / cpu-bigmem instance,
NOT the shared VM (>50 GB peak at the concat). Fail loud; NaN never coerced.
Refusal-safety: no context/rollout TEXT is ever printed or logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land BEFORE numpy/torch import on the shared VM.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n1m_fits")

PREDICTORS = ("ridge", "mlp_w8192", "mlp_w32768", "residual_skip", "krr_nystrom")
PREDICTOR_LABEL = {
    "ridge": "primal ridge (linear, streaming X^TX)",
    "mlp_w8192": "full-dim MLP (w=8192, protocol arm)",
    "mlp_w32768": "full-dim MLP (w=32768, capacity arm)",
    "residual_skip": "residual-skip (primal ridge + MLP w8192)",
    "krr_nystrom": "RBF KRR (Nystrom, streaming Phi^TPhi)",
}

# Ridge grid: LAMBDAS_N50K widened one more decade at the top for the larger n.
LAMBDAS_N1M = np.logspace(-3, 8, 23)

N_PASS_B = F.N_PASS_B  # 5000
N_VAL = 400
N_TEST = 1000
SPLIT_SEED = F.SPLIT_SEED  # 42
MLP_W_PROTOCOL = 8192
MLP_W_CAPACITY = 32768
RIDGE_BLOCK = 50_000  # train-row block for streaming X^TX / Phi^TPhi accumulation
MLP_BATCH = 4096
NYSTROM_VALIDATE_N = 50_000  # train slice for the Nystrom-vs-exact gate
NYSTROM_MAX_CENTERS_WARN = 20_000  # K_mm eigh at m > this may OOM on an 80GB GPU

# Stream-checkpoint + per-chunk download retry (#779 n1m fits crash fix). The HF
# per-chunk stream accumulates the assembled per-layer arrays in memory; a single
# chunk download exhausting HF's internal retries (LocalEntryNotFoundError after a
# transient blip) forfeited ~3.5h of streaming. Checkpoint the accumulated arrays
# every STREAM_CKPT_EVERY chunks so a crash resumes from the cursor, and wrap each
# chunk download in a bounded outer retry (the code-style checkpoint-per-phase law,
# external-stream presumption).
STREAM_CKPT_EVERY = int(os.environ.get("EPM_N1M_STREAM_CKPT_EVERY", "100"))
STREAM_DOWNLOAD_ATTEMPTS = int(os.environ.get("EPM_N1M_DOWNLOAD_ATTEMPTS", "4"))
STREAM_DOWNLOAD_BACKOFF = (10.0, 30.0, 90.0)  # seconds before attempts 2, 3, 4

# n50k committed exact-KRR anchor (reference; the gate is self-contained vs exact).
N50K_EXACT_R2_WIDEGRID = 0.8076
N50K_EXACT_R2_SMALLGRID = 0.8066

KRR_GAMMA_MULT = (1.0,)
KRR_LAMBDAS = (1e-1, 1e1)
KRR_KERNEL_BLOCK = 4096  # exact-KRR row-block (validation only)

# Fit points: (name, n_train_target, corpus_mode). mixed_1m target 1M realizes the
# full usable pool (~963.6k) — the new-pool target is 960k per the gen recipe.
FIT_POINTS = (
    ("lmsys_150k", 150_000, "lmsys"),
    ("lmsys_500k", 500_000, "lmsys"),
    ("mixed_500k", 500_000, "mixed"),
    ("mixed_1m", 1_000_000, "mixed"),
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison-n1m"
DEFAULT_ORIG_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"


# ── data assembly (pass_b + stream-reduced n1m capture) + provenance ────────────


def _is_transient_download_error(err: BaseException) -> bool:
    """True for retryable HF chunk-download failures: LocalEntryNotFoundError,
    requests ReadTimeout / ConnectionError / Timeout, and HTTP 408/429/5xx. Every
    other error re-raises (fail-loud stays). The exact class that forfeited #779's
    ~3.5h stream was a LocalEntryNotFoundError after two absorbed ReadTimeouts."""
    import requests
    from huggingface_hub.errors import LocalEntryNotFoundError

    if isinstance(err, LocalEntryNotFoundError):
        return True
    if isinstance(
        err,
        (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    code = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(code, int):
        return code in (408, 429) or 500 <= code < 600
    return False


def _download_chunk_with_retry(repo: str, filename: str, local_dir: Path) -> str:
    """hf_hub_download with a bounded outer retry over transient errors
    (STREAM_DOWNLOAD_ATTEMPTS, exponential backoff STREAM_DOWNLOAD_BACKOFF). A
    non-transient error or the final attempt re-raises immediately (fail-loud)."""
    from huggingface_hub import hf_hub_download

    for attempt in range(STREAM_DOWNLOAD_ATTEMPTS):
        try:
            return hf_hub_download(
                repo, filename=filename, repo_type="dataset", local_dir=local_dir
            )
        except Exception as e:
            if attempt == STREAM_DOWNLOAD_ATTEMPTS - 1 or not _is_transient_download_error(e):
                raise
            wait = STREAM_DOWNLOAD_BACKOFF[min(attempt, len(STREAM_DOWNLOAD_BACKOFF) - 1)]
            logger.warning(
                "[n1m] transient download error on %s (attempt %d/%d): %s: %s — retry in %.0fs",
                filename,
                attempt + 1,
                STREAM_DOWNLOAD_ATTEMPTS,
                type(e).__name__,
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"unreachable: retry loop exhausted for {filename}")


def _stream_ckpt_fingerprint(layer: int, hf_prefix: str, names: list[str]) -> str:
    """Stable fingerprint of the stream identity — (layer, hf_prefix, sorted chunk
    universe). A mismatch means the checkpoint belongs to a different run (different
    layer/prefix, or new chunks uploaded) and is REFUSED (re-stream from scratch)."""
    h = hashlib.sha256()
    h.update(f"layer={layer}\nprefix={hf_prefix}\n".encode())
    for n in names:  # names is already sorted by the caller
        h.update(n.encode())
        h.update(b"\n")
    return h.hexdigest()


def _stream_ckpt_paths(ckpt_dir: Path, layer: int) -> tuple[Path, Path]:
    return ckpt_dir / f"layer{layer}.npz", ckpt_dir / f"layer{layer}.cursor.json"


def _write_stream_ckpt(
    ckpt_dir: Path,
    layer: int,
    fingerprint: str,
    hf_prefix: str,
    cursor: int,
    n_chunks: int,
    cx: np.ndarray,
    vx: np.ndarray,
    ci: np.ndarray,
    *,
    complete: bool,
) -> None:
    """Atomically persist the accumulated per-layer arrays + cursor sidecar. The npz
    is written first (tmp + os.replace), then the cursor sidecar (tmp + os.replace)
    carrying ``n_rows`` — so a torn write (new npz, old sidecar) is caught on load by
    the ``cx.shape[0] != n_rows`` guard and the stale checkpoint is refused."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    npz_path, cur_path = _stream_ckpt_paths(ckpt_dir, layer)
    tmp_npz = npz_path.parent / (npz_path.name + ".tmp")
    with open(tmp_npz, "wb") as f:
        np.savez(f, cx=cx, vx=vx, ci=ci)
    os.replace(tmp_npz, npz_path)
    meta = {
        "fingerprint": fingerprint,
        "layer": int(layer),
        "hf_prefix": hf_prefix,
        "cursor": int(cursor),
        "n_chunks": int(n_chunks),
        "n_rows": int(cx.shape[0]),
        "complete": bool(complete),
    }
    tmp_cur = cur_path.parent / (cur_path.name + ".tmp")
    tmp_cur.write_text(json.dumps(meta))
    os.replace(tmp_cur, cur_path)


def _load_stream_ckpt(ckpt_dir: Path, layer: int, fingerprint: str, hf_prefix: str):
    """Return (cx, vx, ci, cursor, complete) for a MATCHING checkpoint, else None.
    A missing file, unparseable sidecar, fingerprint/layer/hf_prefix mismatch, or a
    torn write (npz rows != sidecar n_rows) returns None — the caller re-streams from
    scratch with a loud warning, never silently reusing a stale checkpoint."""
    npz_path, cur_path = _stream_ckpt_paths(ckpt_dir, layer)
    if not (npz_path.exists() and cur_path.exists()):
        return None
    try:
        meta = json.loads(cur_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if (
        meta.get("fingerprint") != fingerprint
        or int(meta.get("layer", -1)) != int(layer)
        or meta.get("hf_prefix") != hf_prefix
    ):
        return None
    with np.load(npz_path) as z:
        cx, vx, ci = z["cx"], z["vx"], z["ci"]
    if int(cx.shape[0]) != int(meta.get("n_rows", -1)):  # torn-write guard
        return None
    return cx, vx, ci, int(meta["cursor"]), bool(meta.get("complete", False))


def _stream_n1m_layer(
    prefix: str,
    layer: int,
    local_dir: Path | None,
    cache_dir: Path,
    *,
    ckpt_dir: Path | None = None,
    ckpt_every: int = STREAM_CKPT_EVERY,
    fresh: bool = False,
):
    """Stream-reduce cx_last + v_x + ci at ``layer`` from the n1m capture chunks.

    Mirrors ``N50._stream_n50k_layer`` but ALSO returns the per-row global ci
    (manifest index) needed for provenance. local_dir given -> read staged chunks
    in place; else list the HF prefix (scoped list_repo_tree) and per chunk
    download (bounded retry) -> mmap-slice the layer -> append -> DELETE (peak ~one
    chunk). On the HF path, ``ckpt_dir`` (given) enables checkpoint/resume: every
    ``ckpt_every`` chunks the accumulated per-layer arrays + cursor are persisted
    atomically, and on startup a MATCHING checkpoint (layer + hf_prefix + chunk
    universe) resumes from the cursor so a mid-stream crash never re-streams;
    ``fresh`` ignores any existing checkpoint.
    """
    if local_dir is not None:
        return _stream_local_chunks(local_dir, layer)
    return _stream_hf_chunks(
        prefix, layer, cache_dir, ckpt_dir=ckpt_dir, ckpt_every=ckpt_every, fresh=fresh
    )


def _concat_stream_parts(
    cx_parts: list[np.ndarray], vx_parts: list[np.ndarray], ci_parts: list[list[int]]
):
    cx = np.concatenate(cx_parts)
    vx = np.concatenate(vx_parts)
    ci = np.array([c for part in ci_parts for c in part], dtype=np.int64)
    assert cx.shape[0] == vx.shape[0] == ci.shape[0], (cx.shape, vx.shape, ci.shape)
    return cx, vx, ci


def _stream_local_chunks(local_dir: Path, layer: int):
    chunk_files = sorted(local_dir.glob("shard*_chunk*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no n1m capture chunks under {local_dir}")
    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    ci_parts: list[list[int]] = []
    for cp in chunk_files:
        b = F._mmap_load(cp)
        cx_parts.append(N50._slice_layer(b, "cx_last", layer))
        vx_parts.append(N50._slice_layer(b, "v_x", layer))
        ci_parts.append([int(x) for x in b["ci"]])
        del b
    logger.info("[n1m] %d chunks (local)", len(chunk_files))
    return _concat_stream_parts(cx_parts, vx_parts, ci_parts)


def _resume_hf_stream(ckpt_dir, layer, fp, prefix, names, fresh, cx_parts, vx_parts, ci_parts):
    """Decide the stream start cursor from an existing checkpoint. Returns
    ``(start, complete_arrays)``: ``complete_arrays`` is ``(cx, vx, ci)`` when a
    COMPLETE matching checkpoint exists (the caller returns it directly, no
    re-stream); otherwise ``None`` and the caller streams from ``start``. On a
    PARTIAL resume the ``*_parts`` lists are seeded in place."""
    if ckpt_dir is None or fresh:
        if fresh:
            logger.info("[n1m] --fresh-stream: ignoring any existing stream checkpoint")
        return 0, None
    loaded = _load_stream_ckpt(ckpt_dir, layer, fp, prefix)
    if loaded is None:
        if _stream_ckpt_paths(ckpt_dir, layer)[1].exists():
            logger.warning(
                "[n1m] stream checkpoint present but MISMATCHED (layer/prefix/chunk-universe "
                "or torn write); re-streaming from scratch"
            )
        return 0, None
    l_cx, l_vx, l_ci, cursor, complete = loaded
    if complete and cursor >= len(names):
        logger.info(
            "[n1m] stream checkpoint COMPLETE (%d chunks, %d rows); skip re-stream",
            cursor,
            l_cx.shape[0],
        )
        return cursor, (l_cx, l_vx, l_ci)
    cx_parts[:] = [l_cx]
    vx_parts[:] = [l_vx]
    ci_parts[:] = [l_ci.tolist()]
    logger.info(
        "[n1m] RESUMED stream checkpoint: %d/%d chunks (%d rows); continuing",
        cursor,
        len(names),
        l_cx.shape[0],
    )
    return cursor, None


def _stream_hf_chunks(prefix, layer, cache_dir, *, ckpt_dir, ckpt_every, fresh):
    from huggingface_hub import HfApi

    names = sorted(
        f.path.rsplit("/", 1)[-1]
        for f in HfApi().list_repo_tree(
            C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
        )
        if getattr(f, "size", None) is not None and f.path.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no n1m capture chunks under HF {prefix}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _stream_ckpt_fingerprint(layer, prefix, names) if ckpt_dir is not None else ""

    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    ci_parts: list[list[int]] = []
    start, complete_arrays = _resume_hf_stream(
        ckpt_dir, layer, fp, prefix, names, fresh, cx_parts, vx_parts, ci_parts
    )
    if complete_arrays is not None:
        return complete_arrays

    for i in range(start, len(names)):
        got = Path(_download_chunk_with_retry(C.HF_DATA_REPO, f"{prefix}/{names[i]}", cache_dir))
        b = F._mmap_load(got)
        cx_parts.append(N50._slice_layer(b, "cx_last", layer))
        vx_parts.append(N50._slice_layer(b, "v_x", layer))
        ci_parts.append([int(x) for x in b["ci"]])
        del b
        got.unlink()
        done = i + 1
        if ckpt_dir is not None and ckpt_every > 0 and done % ckpt_every == 0:
            cx_c, vx_c, ci_c = _concat_stream_parts(cx_parts, vx_parts, ci_parts)
            _write_stream_ckpt(
                ckpt_dir, layer, fp, prefix, done, len(names), cx_c, vx_c, ci_c, complete=False
            )
            # collapse parts to one accumulated copy so peak memory stays ~one array
            cx_parts[:], vx_parts[:], ci_parts[:] = [cx_c], [vx_c], [ci_c.tolist()]
            logger.info(
                "[n1m] stream checkpoint @ %d/%d chunks (%d rows)", done, len(names), cx_c.shape[0]
            )
        elif done % 25 == 0:
            logger.info("[n1m] streamed %d/%d chunks", done, len(names))

    cx, vx, ci = _concat_stream_parts(cx_parts, vx_parts, ci_parts)
    if ckpt_dir is not None:
        _write_stream_ckpt(
            ckpt_dir, layer, fp, prefix, len(names), len(names), cx, vx, ci, complete=True
        )
    logger.info("[n1m] %d chunks (HF stream)", len(names))
    return cx, vx, ci


def assemble(args, layer: int):
    """Combined (X=cx_last, Y=v_x) at ``layer`` + provenance + the pinned split.

    Rows [0, N_PASS_B) = pass_b (round-1, lmsys); [N_PASS_B, ...) = the n1m-new
    captured rows in ci order. ``prov`` marks each row lmsys|wildchat from the
    manifest (pass_b + orig-train are lmsys). Returns X, Y, prov, split, meta.
    """
    pb = N1G._load_pass_b_bundle(args.pass_b)
    for fld in ("cx_last", "v_x"):
        assert fld in pb, f"pass_b missing {fld}"
    assert int(pb["cx_last"].shape[0]) == N_PASS_B, (pb["cx_last"].shape[0], N_PASS_B)
    pb_X = N50._slice_layer(pb, "cx_last", layer)
    pb_Y = N50._slice_layer(pb, "v_x", layer)

    # Manifest lives at <round-root>/sampling_manifest; --hf-prefix is the CAPTURE
    # prefix (<round-root>/final_token_capture), so N1G._resolve_manifest_dir (which
    # appends sampling_manifest to args.hf_prefix) must see the round root, NOT the
    # capture prefix. Shim in --manifest-hf-prefix; the local-stage path (no
    # --manifest-from-hf) reads out_dir/sampling_manifest regardless of the prefix.
    manifest_args = argparse.Namespace(
        out_dir=args.out_dir,
        manifest_from_hf=args.manifest_from_hf,
        hf_prefix=args.manifest_hf_prefix,
    )
    manifest_dir = N1G._resolve_manifest_dir(manifest_args)
    pool, man_meta = N1G.read_manifest_pool(manifest_dir)
    ci_to_corpus = {int(r["i"]): r["corpus"] for r in pool}

    local_dir = args.n1m_capture_dir if args.n1m_capture_dir else None
    new_X, new_Y, new_ci = _stream_n1m_layer(
        args.hf_prefix,
        layer,
        local_dir,
        args.out_dir / ".n1m_stream_cache",
        ckpt_dir=(args.out_dir / ".n1m_stream_ckpt") if local_dir is None else None,
        ckpt_every=STREAM_CKPT_EVERY,
        fresh=args.fresh_stream,
    )
    # provenance for each captured new row (ci -> corpus); pass_b rows are lmsys.
    new_prov = np.array([ci_to_corpus[int(c)] for c in new_ci], dtype=object)

    X = np.concatenate([pb_X, new_X]).astype(np.float32)
    Y = np.concatenate([pb_Y, new_Y]).astype(np.float32)
    assert X.shape[1] == C.EXPECTED_HIDDEN and Y.shape[1] == C.EXPECTED_HIDDEN, (X.shape, Y.shape)
    prov = np.array(["lmsys"] * N_PASS_B + list(new_prov), dtype=object)
    assert prov.shape[0] == X.shape[0], (prov.shape, X.shape)

    pinned = N50._pinned_original_shas(args.orig_dir)
    r1_train, val, test = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (
        f"n1m val sha {val_sha} != pinned original {pinned['val_sha256']} — NOT byte-identical"
    )
    assert test_sha == pinned["test_sha256"], (
        f"n1m test sha {test_sha} != pinned original {pinned['test_sha256']}"
    )
    assert (val < N_PASS_B).all() and (test < N_PASS_B).all(), "val/test must index the pass_b half"

    split = {
        "orig_train_ids": len(r1_train),
        "n_new_captured": int(new_X.shape[0]),
        "n_new_manifest": int(man_meta["n_new"]),
        "n_lmsys_manifest": int(man_meta["n_lmsys"]),
        "n_wildchat_manifest": int(man_meta["n_wildchat"]),
        "n_val": len(val),
        "n_test": len(test),
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "pinned_val_sha256": pinned["val_sha256"],
        "pinned_test_sha256": pinned["test_sha256"],
        "pinned_source": pinned["source"],
        "val_test_byte_identical_original": True,
        "layer": int(layer),
        "near_dupe": man_meta.get("near_dupe"),
        "manifest_new_prompt_sha256": man_meta.get("new_prompt_sha256"),
    }
    return X, Y, prov, r1_train, val, test, split


# ── provenance-aware deterministic subset selection ─────────────────────────────


def _pool_rows(prov, orig_train, n_total, val, test):
    """Row-index pools into the combined X. orig_train = fixed_split train ids
    (< N_PASS_B, all lmsys); new rows = [N_PASS_B, n_total). Excludes val/test by
    construction (orig_train disjoint from val/test; new rows >= N_PASS_B)."""
    new_rows = np.arange(N_PASS_B, n_total)
    new_lmsys = new_rows[prov[new_rows] == "lmsys"]
    new_wild = new_rows[prov[new_rows] == "wildchat"]
    lmsys_pool = np.concatenate([np.asarray(orig_train, dtype=np.int64), new_lmsys])
    full_pool = np.concatenate([np.asarray(orig_train, dtype=np.int64), new_rows])
    excl = set(int(x) for x in val) | set(int(x) for x in test)
    assert not (set(int(x) for x in full_pool) & excl), "train pool overlaps val/test"
    return {
        "lmsys": np.sort(lmsys_pool),
        "full": np.sort(full_pool),
        "orig_train": np.asarray(orig_train, dtype=np.int64),
        "new_lmsys": new_lmsys,
        "new_wildchat": new_wild,
    }


def select_train(pools, name, n_target, mode, seed):
    """Deterministic seeded subset of train rows for one fit point.

    mode='lmsys': sample from the lmsys pool. mode='mixed': stratified to the full
    pool's lmsys:wildchat ratio (or the whole full pool if n_target >= |full|).
    Returns (sorted train indices, selection diag)."""
    rng = np.random.default_rng(int(seed) + (abs(hash(name)) % 1_000_000))
    if mode == "lmsys":
        pool = pools["lmsys"]
        n = min(int(n_target), len(pool))
        sel = pool[rng.choice(len(pool), size=n, replace=False)]
        diag = {
            "mode": mode,
            "n_target": int(n_target),
            "n_realized": int(n),
            "n_lmsys": int(n),
            "n_wildchat": 0,
        }
        return np.sort(sel), diag
    # mixed
    full = pools["full"]
    lm = pools["lmsys"]  # lmsys rows in full (orig_train + new_lmsys)
    wild = pools["new_wildchat"]
    lmsys_frac = len(lm) / len(full) if len(full) else 0.0
    if int(n_target) >= len(full):
        sel = full  # whole mixed pool
        n_l, n_w = len(lm), len(wild)
    else:
        n = int(n_target)
        n_l = min(round(n * lmsys_frac), len(lm))
        n_w = min(n - n_l, len(wild))
        n_l = min(n - n_w, len(lm))  # rebalance if wildchat short
        lm_sel = lm[rng.choice(len(lm), size=n_l, replace=False)]
        w_sel = wild[rng.choice(len(wild), size=n_w, replace=False)]
        sel = np.concatenate([lm_sel, w_sel])
    diag = {
        "mode": mode,
        "n_target": int(n_target),
        "n_realized": len(sel),
        "n_lmsys": int(n_l),
        "n_wildchat": int(n_w),
        "full_lmsys_frac": round(float(lmsys_frac), 4),
    }
    return np.sort(sel), diag


# ── streaming primal ridge (fp64 X^TX / X^TY over train-row blocks) ─────────────


def _train_standardizer(X, Y, tr, dev, block):
    """Streaming train mean/std of X + mean of Y (fp64 on dev)."""
    H = X.shape[1]
    sum_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sumsq_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sum_y = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
    n = 0
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = torch.as_tensor(X[idx], dtype=torch.float64, device=dev)
        yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev)
        sum_x += xb.sum(0)
        sumsq_x += (xb * xb).sum(0)
        sum_y += yb.sum(0)
        n += len(idx)
    xmu = sum_x / n
    # UNBIASED (N-1) variance to match N50._ridge_primal_multi_lambda's torch.std
    # default (unbiased=True) exactly — the standardization scale is NOT absorbed by
    # ridge, so the convention is load-bearing for streaming==N50 parity.
    denom = max(1, n - 1)
    var = (sumsq_x - n * xmu * xmu) / denom
    xsd = torch.clamp(var, min=0.0).sqrt() + 1e-9
    ymu = sum_y / n
    return xmu, xsd, ymu


def _ridge_factorize(X, Y, tr, dev, block):
    """Standardize X on train stats, center Y on train mean, accumulate the (H,H)
    X^TX + (H,D) X^TY STREAMING over train blocks, eigh once. Returns the
    factorization state {U, s_eig, UtXtY, xmu, xsd, ymu} — reused across lambdas
    AND eval sets by _ridge_predict_one, so no eval set is ever materialized for
    all lambdas at once (the residual_skip rc=137 OOM fix, #779 round 6)."""
    xmu, xsd, ymu = _train_standardizer(X, Y, tr, dev, block)
    H = X.shape[1]
    A = torch.zeros((H, H), dtype=torch.float64, device=dev)
    XtY = torch.zeros((H, Y.shape[1]), dtype=torch.float64, device=dev)
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = (torch.as_tensor(X[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev) - ymu
        A += xb.T @ xb
        XtY += xb.T @ yb
    s_eig, U = torch.linalg.eigh(A)
    s_eig = torch.clamp(s_eig, min=0.0)
    UtXtY = U.T @ XtY
    return {"U": U, "s_eig": s_eig, "UtXtY": UtXtY, "xmu": xmu, "xsd": xsd, "ymu": ymu}


def _ridge_predict_one(X, eval_idx, fac, lam, dev, block):
    """Predict ONE eval set at ONE lambda from a _ridge_factorize state, BLOCKED so
    the (len(eval_idx), H) standardized design is never held whole — peak is ~one
    (len(eval_idx), D) fp64 output array. Numerically identical to the all-at-once
    matmul: block-chunked (En @ W) concatenated == full (En @ W)."""
    U, s_eig, UtXtY = fac["U"], fac["s_eig"], fac["UtXtY"]
    xmu, xsd, ymu = fac["xmu"], fac["xsd"], fac["ymu"]
    if len(eval_idx) == 0:
        return np.zeros((0, UtXtY.shape[1]))
    W = U @ (UtXtY / (s_eig + float(lam))[:, None])
    outs = []
    for s in range(0, len(eval_idx), block):
        idx = eval_idx[s : s + block]
        En = (torch.as_tensor(X[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        outs.append(((En @ W) + ymu).cpu().numpy())
    return np.concatenate(outs)


def _is_train_pool(e, tr) -> bool:
    """True iff eval-index-set ``e`` IS the train pool ``tr`` (same object, or equal
    values). The train pool must never be a multi-lambda eval set — see the guard in
    _ridge_streaming_multi_lambda."""
    if e is tr:
        return True
    ea, ta = np.asarray(e), np.asarray(tr)
    return ea.shape == ta.shape and bool(np.array_equal(ea, ta))


def _ridge_streaming_multi_lambda(X, Y, tr, eval_idx_list, lambdas, dev, block):
    """Exact primal ridge (all lambdas off ONE eigh of the streamed (H,H) X^TX).

    Standardizes X on train stats, centers Y on train mean — numerically identical
    to N50._ridge_primal_multi_lambda, just block-accumulated so the (n, H) design
    is never materialized at once. Returns {lambda: [pred for each eval set]}.

    GUARD (#779 round 6): refuses the TRAIN pool as an eval set — that materializes
    an (n_train, D) pred for EVERY lambda at once (~n_lambda x 13.8 GB -> rc=137 at
    n>=500k, the residual_skip OOM). A caller needing the train prediction selects
    lambda on val, then predicts the train pool at the SELECTED lambda ONLY via
    _ridge_factorize + _ridge_predict_one (see fit_residual_skip)."""
    for e in eval_idx_list:
        if _is_train_pool(e, tr):
            raise ValueError(
                "_ridge_streaming_multi_lambda: train pool passed as an eval set — this "
                "builds (n_train, D) preds for all lambdas at once and OOMs at n>=500k. "
                "Select lambda on val then predict train at the selected lambda only "
                "(_ridge_factorize + _ridge_predict_one; see fit_residual_skip)."
            )
    fac = _ridge_factorize(X, Y, tr, dev, block)
    out: dict[float, list[np.ndarray]] = {}
    for lam in lambdas:
        out[float(lam)] = [_ridge_predict_one(X, e, fac, lam, dev, block) for e in eval_idx_list]
    return out, {"xmu": fac["xmu"], "xsd": fac["xsd"], "ymu": fac["ymu"]}


def fit_ridge(X, Y, tr, val, te, lambdas, dev, block):
    preds, _ = _ridge_streaming_multi_lambda(X, Y, tr, [val, te], lambdas, dev, block)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(preds[float(lam)][0], Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    return preds[best_lam][1], {
        "n_train": len(tr),
        "selection": "val-lambda (primal, streaming)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "ridge_block": int(block),
    }


# ── minibatched MLP (single large-n fit; the full-batch battery cannot hold 1M) ──


def _fit_mlp_minibatch(
    X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, *, base_tr=None, base_te=None
):
    """Single full-dim MLP (GELU, MSE) trained MINIBATCHED with AdamW + internal-val
    early stop. Standardizes X on train stats; centers Y on train mean (or fits the
    residual Y - base_* when base_tr/base_te given, for residual_skip). Predicts te
    minibatched. FLOP-bound single large-n fit — NOT a many-cell loop (the
    vectorized_mlp_skill helper batches CELLS, a different regime)."""
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    H = X.shape[1]
    D = Y.shape[1]
    # standardizer on train X (streamed to keep the (n,H) copy off the device)
    xmu, xsd, ymu = _train_standardizer(X, Y, tr, dev, RIDGE_BLOCK)
    xmu_c = xmu.to(torch.float32)
    xsd_c = xsd.to(torch.float32)
    ymu_c = ymu.to(torch.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_val = max(1, round(0.1 * len(tr)))
    va_local, tr_local = perm[:n_val], perm[n_val:]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(H, width), torch.nn.GELU(), torch.nn.Linear(width, D)
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=F.MLP_WD)

    # residual base (Y - ridge_base) is precomputed per TRAIN row, aligned to `tr`;
    # scatter it into a combined-row lookup so a minibatch's rows index it directly.
    _base_lookup = None
    if base_tr is not None:
        _base_lookup = np.zeros((X.shape[0], D), dtype=np.float32)
        _base_lookup[tr] = base_tr  # only train rows are read for the residual target
    best_val = float("inf")
    best_state = None
    bad = 0
    epochs_ran = 0
    for ep in range(max_epochs):
        net.train()
        ep_perm = rng.permutation(len(tr_local))
        for bs in range(0, len(tr_local), batch):
            rows = tr[tr_local[ep_perm[bs : bs + batch]]]
            xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
            tb = torch.as_tensor(Y[rows], dtype=torch.float32, device=dev) - ymu_c
            if _base_lookup is not None:
                tb = tb - torch.as_tensor(_base_lookup[rows], dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = ((net(xb) - tb) ** 2).mean()
            loss.backward()
            opt.step()
        # internal-val
        net.eval()
        with torch.no_grad():
            vsum, vcnt = 0.0, 0
            for bs in range(0, len(va_local), batch):
                rows = tr[va_local[bs : bs + batch]]
                xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
                tb = torch.as_tensor(Y[rows], dtype=torch.float32, device=dev) - ymu_c
                if _base_lookup is not None:
                    tb = tb - torch.as_tensor(_base_lookup[rows], dtype=torch.float32, device=dev)
                vsum += float(((net(xb) - tb) ** 2).sum())
                vcnt += rows.shape[0] * D
            vloss = vsum / max(1, vcnt)
        epochs_ran = ep + 1
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= F.MLP_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    preds = []
    with torch.no_grad():
        for bs in range(0, len(te), batch):
            rows = te[bs : bs + batch]
            xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
            out = net(xb) + ymu_c
            preds.append(out.cpu().numpy())
    pred_te = np.concatenate(preds).astype(np.float32)
    if base_te is not None:
        pred_te = pred_te + base_te
    return pred_te, {
        "width": int(width),
        "lr": float(lr),
        "epochs_ran": int(epochs_ran),
        "batch": int(batch),
        "best_val_mse": float(best_val),
    }


def fit_mlp(X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, *, capacity_arm=False):
    pred_te, meta = _fit_mlp_minibatch(X, Y, tr, te, width, lr, max_epochs, batch, seed, dev)
    meta["n_train"] = len(tr)
    meta["capacity_arm"] = bool(capacity_arm)
    return pred_te, meta


def fit_residual_skip(X, Y, tr, val, te, lambdas, width, lr, max_epochs, batch, seed, dev, block):
    """Primal ridge base + minibatched MLP on the residual (strictly nests linear).

    Memory contract (#779 round 6): factorize the ridge solve ONCE, select the base
    lambda on VAL predictions only (each computed then discarded — peak ~one
    (len(val), D) array), THEN build the train + test ridge base at the SELECTED
    lambda ONLY (peak ~one (n_train, D) fp64 array ≈ 13.8 GB at n≈1M). NEVER
    materialize train-pool preds for all lambdas at once — that was the rc=137 OOM
    (~n_lambda x the train-pred array) this fix removes. Lambda selection is
    SELF-CONTAINED (no reuse of a checkpointed ridge cell), so a standalone
    ``--predictors residual_skip`` backfill run (no ridge cell present) still works.
    Numerically identical to the prior all-lambda path — same factorization, same
    val-selected lambda, same predictions."""
    fac = _ridge_factorize(X, Y, tr, dev, block)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(_ridge_predict_one(X, val, fac, lam, dev, block), Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    base_tr = _ridge_predict_one(X, tr, fac, best_lam, dev, block)  # ONE train-pred array
    base_te = _ridge_predict_one(X, te, fac, best_lam, dev, block)
    pred, mmeta = _fit_mlp_minibatch(
        X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, base_tr=base_tr, base_te=base_te
    )
    return pred, {
        "n_train": len(tr),
        "base_ridge_lambda": best_lam,
        "residual_mlp_width": int(width),
        "lr": float(lr),
        "epochs_ran": mmeta["epochs_ran"],
        "batch": int(batch),
    }


# ── chunked Nystrom RBF KRR (streaming Phi^TPhi over train blocks) ───────────────


def _nystrom_inv_sqrt(landmarks, gamma, dev, eig_floor=1e-10):
    """K_mm^{-1/2} whitener (m, m) fp64 on dev."""
    Z = torch.as_tensor(np.asarray(landmarks), dtype=torch.float64, device=dev)
    K_mm = torch.exp(-gamma * torch.cdist(Z, Z) ** 2)
    w, V = torch.linalg.eigh(K_mm)
    w = torch.clamp(w, min=eig_floor)
    return V @ torch.diag(w.rsqrt()) @ V.T  # (m, m)


def _nystrom_features_block(Xblock, landmarks_t, gamma, inv_sqrt):
    """Phi_block = exp(-gamma ||Xb - Z||^2) @ inv_sqrt, (block, m) fp64 on dev."""
    Xb = torch.as_tensor(np.asarray(Xblock), dtype=torch.float64, device=inv_sqrt.device)
    K_bm = torch.exp(-gamma * torch.cdist(Xb, landmarks_t) ** 2)
    return K_bm @ inv_sqrt


def fit_krr_nystrom(X, Y, tr, val, te, *, m_centers, gamma_mult, lambdas, seed, dev, block):
    """Nystrom RBF KRR, (gamma, lambda) val-selected. Phi^TPhi accumulated STREAMING
    over train blocks so the (ntr, m) feature matrix is never materialized whole.
    Raw X + median-heuristic gamma (matches N50.fit_krr_exact for the validation)."""
    tr = np.asarray(tr, dtype=np.int64)
    Xtr_sub = np.asarray(X[tr[: min(len(tr), 4000)]], dtype=np.float64)  # gamma est subsample
    base_gamma = F.median_heuristic_gamma(Xtr_sub, np.random.default_rng(seed + 1))
    m = int(min(m_centers, len(tr)))
    if m > NYSTROM_MAX_CENTERS_WARN:
        logger.warning(
            "[krr] m_centers=%d > %d — K_mm eigh (m,m) may OOM on an 80GB GPU",
            m,
            NYSTROM_MAX_CENTERS_WARN,
        )
    lm_rows = tr[np.random.default_rng(seed).choice(len(tr), size=m, replace=False)]
    landmarks = np.asarray(X[lm_rows], dtype=np.float64)
    # center Y on train mean (streamed)
    _, _, ymu = _train_standardizer(X, Y, tr, dev, block)
    grid, best = [], None
    for gm in gamma_mult:
        gamma = base_gamma * gm
        inv_sqrt = _nystrom_inv_sqrt(landmarks, gamma, dev)
        landmarks_t = torch.as_tensor(landmarks, dtype=torch.float64, device=dev)
        G = torch.zeros((m, m), dtype=torch.float64, device=dev)
        PhiY = torch.zeros((m, Y.shape[1]), dtype=torch.float64, device=dev)
        for s in range(0, len(tr), block):
            idx = tr[s : s + block]
            phi = _nystrom_features_block(X[idx], landmarks_t, gamma, inv_sqrt)  # (blk, m)
            yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev) - ymu
            G += phi.T @ phi
            PhiY += phi.T @ yb
        a, Q = torch.linalg.eigh(G)
        a = torch.clamp(a, min=0.0)
        QtPhiY = Q.T @ PhiY
        phi_val = _nystrom_features_block(X[val], landmarks_t, gamma, inv_sqrt)
        phi_te = _nystrom_features_block(X[te], landmarks_t, gamma, inv_sqrt)
        for lam in lambdas:
            W = Q @ (QtPhiY / (a + float(lam))[:, None])  # (m, D)
            pred_val = (phi_val @ W + ymu).cpu().numpy()
            pred_te = (phi_te @ W + ymu).cpu().numpy()
            val_r2 = PR._pooled_r2(pred_val, Y[val])
            grid.append(
                {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                }
            )
            if best is None or (np.isfinite(val_r2) and val_r2 > best["val_r2"]):
                best = {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                    "pred_te": pred_te,
                }
        del G, PhiY, inv_sqrt, landmarks_t, phi_val, phi_te
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    assert best is not None
    return best["pred_te"], {
        "n_train": len(tr),
        "kernel": "RBF Nystrom (streaming Phi^TPhi)",
        "m_centers": m,
        "base_gamma": float(base_gamma),
        "selected": {k: best[k] for k in ("gamma_mult", "gamma", "lambda", "val_r2")},
        "val_grid": grid,
    }


def _validate_nystrom_vs_exact(
    X, Y, pools, val, te, *, m_centers, gamma_mult, krr_lambdas, seed, dev, tol
):
    """Run Nystrom AND exact KRR on the SAME 50k train slice; assert R2 agreement.

    A gap > tol means the Nystrom fitter is numerically biased vs exact — FAIL LOUD
    (not a shrug), per the brief. Requires cuda (the 50k^2 exact kernel)."""
    if dev.type != "cuda":
        raise SystemExit("--validate-krr requires --device cuda (the exact 50k^2 KRR kernel)")
    pool = pools["lmsys"]  # pure-lmsys 50k slice (comparable to the n50k exact anchor)
    n = min(NYSTROM_VALIDATE_N, len(pool))
    tr = np.sort(pool[np.random.default_rng(seed + 7).choice(len(pool), size=n, replace=False)])
    logger.info("[krr-validate] exact vs Nystrom (m=%d) on n=%d ...", m_centers, n)
    ts = time.time()
    pred_ex, meta_ex = N50.fit_krr_exact(
        X,
        Y,
        tr,
        val,
        te,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        block=KRR_KERNEL_BLOCK,
        seed=seed,
        dev=dev,
    )
    r2_ex = PR._pooled_r2(pred_ex, Y[te])
    pred_ny, meta_ny = fit_krr_nystrom(
        X,
        Y,
        tr,
        val,
        te,
        m_centers=m_centers,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        seed=seed,
        dev=dev,
        block=RIDGE_BLOCK,
    )
    r2_ny = PR._pooled_r2(pred_ny, Y[te])
    gap = abs(r2_ny - r2_ex)
    logger.info(
        "[krr-validate] exact R2=%.4f  nystrom R2=%.4f  gap=%.4f (tol %.4f, %.0fs)",
        r2_ex,
        r2_ny,
        gap,
        tol,
        time.time() - ts,
    )
    if gap > tol:
        raise SystemExit(
            f"Nystrom-vs-exact KRR gap {gap:.4f} > tol {tol:.4f} at n={n} (exact {r2_ex:.4f}, "
            f"nystrom {r2_ny:.4f}) — the Nystrom fitter is biased; raise --krr-nystrom-centers"
        )
    return {
        "n": int(n),
        "m_centers": int(m_centers),
        "exact_r2": float(r2_ex),
        "nystrom_r2": float(r2_ny),
        "gap": float(gap),
        "tol": float(tol),
        "committed_n50k_exact_r2_widegrid": N50K_EXACT_R2_WIDEGRID,
        "committed_n50k_exact_r2_smallgrid": N50K_EXACT_R2_SMALLGRID,
        "exact_selected": meta_ex.get("selected"),
        "nystrom_selected": meta_ny.get("selected"),
    }


def _curve(pred_te, Y_te, n_boot, seed) -> dict:
    r2, cos = F._recon_point(pred_te, Y_te)
    ci = F._bootstrap_recon_ci(pred_te, Y_te, n_boot, seed)
    return {"whole_map_r2": float(r2), "mean_cosine": float(cos), "bootstrap_ci": ci}


def _fit_one_predictor(name, X, Y, tr, val, test, lambdas, gamma_mult, krr_lambdas, args, dev):
    """Dispatch one predictor fit; returns (pred_te, fit_meta)."""
    if name == "ridge":
        return fit_ridge(X, Y, tr, val, test, lambdas, dev, args.ridge_block)
    if name == "mlp_w8192":
        return fit_mlp(
            X,
            Y,
            tr,
            test,
            MLP_W_PROTOCOL,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
        )
    if name == "mlp_w32768":
        return fit_mlp(
            X,
            Y,
            tr,
            test,
            MLP_W_CAPACITY,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
            capacity_arm=True,
        )
    if name == "residual_skip":
        return fit_residual_skip(
            X,
            Y,
            tr,
            val,
            test,
            lambdas,
            MLP_W_PROTOCOL,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
            args.ridge_block,
        )
    return fit_krr_nystrom(
        X,
        Y,
        tr,
        val,
        test,
        m_centers=args.krr_nystrom_centers,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        seed=args.seed,
        dev=dev,
        block=args.ridge_block,
    )


def _run_fit_points(
    results,
    want_points,
    want_pred,
    point_by_name,
    X,
    Y,
    pools,
    val,
    test,
    lambdas,
    gamma_mult,
    krr_lambdas,
    args,
    dev,
):
    """Per (fit point x predictor): select the provenance-aware subset, fit, curve,
    and checkpoint. ``--resume`` skips completed (point, predictor) cells."""
    for pn in want_points:
        _, n_target, mode = point_by_name[pn]
        tr, sel_diag = select_train(pools, pn, n_target, mode, args.seed)
        results["per_point"].setdefault(pn, {"selection": sel_diag, "predictors": {}})
        results["per_point"][pn]["selection"] = sel_diag
        logger.info("[point %s] n_train=%d (%s) — %s", pn, len(tr), mode, sel_diag)
        for name in want_pred:
            if args.resume and name in results["per_point"][pn]["predictors"]:
                logger.info("[resume] %s/%s present; skip", pn, name)
                continue
            ts = time.time()
            pred_te, meta = _fit_one_predictor(
                name, X, Y, tr, val, test, lambdas, gamma_mult, krr_lambdas, args, dev
            )
            curve = _curve(pred_te, Y[test], args.n_boot, args.seed)
            curve["fit_meta"] = meta
            curve["wall_time_s"] = round(time.time() - ts, 1)
            results["per_point"][pn]["predictors"][name] = curve
            C.write_json_atomic(args.out_json, results)
            logger.info(
                "[done] %s/%s: whole-map R2=%.4f mean-cos=%.4f (%.0fs)",
                pn,
                name,
                curve["whole_map_r2"],
                curve["mean_cosine"],
                curve["wall_time_s"],
            )


def _smoke_stream_ckpt(ckpt_root: Path) -> tuple[int, int]:
    """CPU smoke for the HF stream checkpoint write/resume roundtrip (#779 fix).

    Fakes list_repo_tree + hf_hub_download over 8 synthetic chunks: (a) an
    uninterrupted reference stream; (b) a run that crashes mid-stream (synthetic
    non-transient error) AFTER a checkpoint; (c) a resume that completes — asserting
    the resumed arrays are BYTE-IDENTICAL to the reference (no duplicated/dropped
    rows across the checkpoint boundary), and (d) a 4th run over the COMPLETE
    checkpoint downloads ZERO chunks. Returns (n_rows, n_chunks)."""
    from unittest import mock

    hdim, n_chunks, layers = 4, 8, [14, 19, 26]
    layer = 19  # stored column 1
    prefix = "smoke_prefix/final_token_capture"
    remote = {f"shard00_chunk{i:04d}.pt": i for i in range(n_chunks)}

    def _chunk(idx: int) -> dict:
        rows = 2 + (idx % 2)  # variable kept-row count (2 or 3) per chunk
        base = float(idx * 100)
        shape = (rows, len(layers), hdim)
        cx = base + torch.arange(rows * len(layers) * hdim, dtype=torch.float32).reshape(shape)
        return {
            "cx_last": cx.clone(),
            "v_x": (cx + 0.5).clone(),
            "ci": [idx * 1000 + r for r in range(rows)],
            "layers": list(layers),
        }

    crash_at: dict = {"i": None}
    dl_calls = {"n": 0}

    class _SmokeCrash(RuntimeError):
        pass

    class _FakeEntry:
        def __init__(self, path):
            self.path, self.size = path, 1

    class _FakeHfApi:
        def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, recursive=False):
            return [_FakeEntry(f"{path_in_repo}/{n}") for n in remote]

    def _fake_dl(repo_id, filename=None, repo_type=None, local_dir=None):
        dl_calls["n"] += 1
        base = filename.rsplit("/", 1)[-1]
        idx = remote[base]
        if crash_at["i"] is not None and idx >= crash_at["i"]:
            raise _SmokeCrash(f"synthetic crash at chunk {idx}")
        out = Path(local_dir) / base
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_chunk(idx), out)
        return str(out)

    cache = ckpt_root / "cache"
    with (
        mock.patch("huggingface_hub.HfApi", _FakeHfApi),
        mock.patch("huggingface_hub.hf_hub_download", _fake_dl),
    ):
        cx_ref, vx_ref, ci_ref = _stream_n1m_layer(
            prefix, layer, None, cache, ckpt_dir=ckpt_root / "ref", ckpt_every=2
        )
        run_ck = ckpt_root / "run"
        crash_at["i"] = 5  # index 4 consumed-but-not-checkpointed (ckpt at cursor=4); crash at 5
        crashed = False
        try:
            _stream_n1m_layer(prefix, layer, None, cache, ckpt_dir=run_ck, ckpt_every=2)
        except _SmokeCrash:
            crashed = True
        assert crashed, "synthetic crash did not fire"
        assert _stream_ckpt_paths(run_ck, layer)[1].exists(), "no checkpoint written before crash"
        crash_at["i"] = None
        cx_r, vx_r, ci_r = _stream_n1m_layer(
            prefix, layer, None, cache, ckpt_dir=run_ck, ckpt_every=2
        )
        assert np.array_equal(cx_r, cx_ref), "resumed cx != uninterrupted reference"
        assert np.array_equal(vx_r, vx_ref), "resumed vx != uninterrupted reference"
        assert np.array_equal(ci_r, ci_ref), "resumed ci != uninterrupted reference"
        dl_before = dl_calls["n"]
        cx2, _, _ = _stream_n1m_layer(prefix, layer, None, cache, ckpt_dir=run_ck, ckpt_every=2)
    assert dl_calls["n"] == dl_before, "complete checkpoint re-downloaded chunks"
    assert np.array_equal(cx2, cx_ref), "complete-checkpoint load != reference"
    return int(cx_ref.shape[0]), n_chunks


def _smoke_download_retry(cache: Path) -> int:
    """CPU smoke for the per-chunk download retry (#779 fix). A fake hf_hub_download
    fails twice transiently (ReadTimeout, then LocalEntryNotFoundError) then succeeds
    on attempt 3; asserts _download_chunk_with_retry retried and returned the path. A
    non-transient error (ValueError) re-raises on attempt 1 (no retry). time.sleep is
    patched to a no-op so the backoff adds no wall time. Returns the success-attempt
    count."""
    from unittest import mock

    import requests
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = {"n": 0}

    def _flaky(repo_id, filename=None, repo_type=None, local_dir=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("synthetic read timeout")
        if calls["n"] == 2:
            raise LocalEntryNotFoundError("synthetic local-entry-not-found")
        out = Path(local_dir) / filename.rsplit("/", 1)[-1]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok")
        return str(out)

    with (
        mock.patch("huggingface_hub.hf_hub_download", _flaky),
        mock.patch("time.sleep", lambda *_a, **_k: None),
    ):
        got = _download_chunk_with_retry("repo", "pfx/shard00_chunk0000.pt", cache)
    assert calls["n"] == 3, f"expected 3 attempts (2 transient + success), got {calls['n']}"
    assert Path(got).exists(), "retry did not return a downloaded path"

    calls["n"] = 0

    def _hard(repo_id, filename=None, repo_type=None, local_dir=None):
        calls["n"] += 1
        raise ValueError("non-transient")

    raised = False
    with (
        mock.patch("huggingface_hub.hf_hub_download", _hard),
        mock.patch("time.sleep", lambda *_a, **_k: None),
    ):
        try:
            _download_chunk_with_retry("repo", "pfx/x.pt", cache)
        except ValueError:
            raised = True
    assert raised, "non-transient error must propagate"
    assert calls["n"] == 1, f"non-transient error must not retry (got {calls['n']})"
    return 3


def _smoke() -> int:
    """CPU numeric-sanity smoke (synthetic; no capture data, no GPU).

    Covers: (1) the byte-identical val/test split shas (recomputed fixed_split ==
    the pinned N50 constants); (2) provenance-aware subset selection (lmsys mode
    keeps only lmsys rows; mixed mode preserves the full-pool ratio; realized =
    min(target, pool)); (3) streaming primal ridge == N50's primal ridge on the
    same synthetic; (4) Nystrom-vs-exact KRR agreement on a small synthetic RBF
    problem (the fitter numeric sanity); (5) the minibatched MLP + residual-skip
    bodies run end-to-end and beat the mean baseline."""
    dev = torch.device("cpu")
    torch.set_num_threads(4)
    logger.info("[smoke] fits CPU numeric-sanity (split-sha + selection + ridge + nystrom + mlp)")

    # (1) byte-identical split shas.
    _r1, val_idx, test_idx = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    assert F._sha_ids(val_idx) == N50.ORIG_VAL_SHA256, "val split sha drift"
    assert F._sha_ids(test_idx) == N50.ORIG_TEST_SHA256, "test split sha drift"

    # (2) provenance-aware selection on a synthetic combined layout.
    n_total = N_PASS_B + 200
    prov = np.array(["lmsys"] * N_PASS_B + ["lmsys"] * 150 + ["wildchat"] * 50, dtype=object)
    orig_train, val, test = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    pools = _pool_rows(prov, orig_train, n_total, val, test)
    assert (prov[pools["lmsys"]] == "lmsys").all(), "lmsys pool contains non-lmsys rows"
    tr_l, dl = select_train(pools, "lmsys_x", 1_000_000, "lmsys", 0)
    assert dl["n_realized"] == len(pools["lmsys"]) and dl["n_wildchat"] == 0, dl
    assert (prov[tr_l] == "lmsys").all(), "lmsys-mode selection leaked wildchat"
    _tr_m, dm = select_train(pools, "mixed_x", 100, "mixed", 0)
    assert dm["n_realized"] == 100 and dm["n_wildchat"] > 0, dm  # ratio-matched, wildchat present
    frac = dm["n_lmsys"] / dm["n_realized"]
    assert abs(frac - dm["full_lmsys_frac"]) < 0.1, (frac, dm)
    _tr_full, df = select_train(pools, "mixed_all", 1_000_000, "mixed", 0)
    assert df["n_realized"] == len(pools["full"]), df  # whole pool when target >= pool

    # (3)-(5) numeric checks on a small synthetic RBF-friendly problem.
    rng = np.random.default_rng(7)
    n, H, D = 500, 40, 8
    Wt = rng.standard_normal((H, D)) * 0.3
    Xs = rng.standard_normal((n, H)).astype(np.float32)
    Ys = (np.tanh(Xs @ Wt.astype(np.float32)) + 0.05 * rng.standard_normal((n, D))).astype(
        np.float32
    )
    tr = np.arange(0, 380)
    vl = np.arange(380, 420)
    ts = np.arange(420, 500)
    lambdas = np.logspace(-2, 3, 6)

    # (3) streaming ridge == N50 primal ridge (same math, block-accumulated).
    stream_preds, _ = _ridge_streaming_multi_lambda(Xs, Ys, tr, [ts], lambdas, dev, block=64)
    ref = N50._ridge_primal_multi_lambda(Xs[tr], Ys[tr], [Xs[ts]], lambdas, dev)
    for lam in lambdas:
        d = float(np.max(np.abs(stream_preds[float(lam)][0] - ref[float(lam)][0])))
        assert d < 1e-4, f"streaming ridge != N50 primal ridge at lambda={lam}: max|diff|={d:.2e}"

    # (4) Nystrom-vs-exact agreement (m=200 centers of 380 train).
    pred_ex, _ = N50.fit_krr_exact(
        Xs, Ys, tr, vl, ts, gamma_mult=(1.0,), lambdas=(1e-1, 1e1), block=128, seed=0, dev=dev
    )
    pred_ny, _ = fit_krr_nystrom(
        Xs,
        Ys,
        tr,
        vl,
        ts,
        m_centers=200,
        gamma_mult=(1.0,),
        lambdas=(1e-1, 1e1),
        seed=0,
        dev=dev,
        block=64,
    )
    r2_ex, r2_ny = PR._pooled_r2(pred_ex, Ys[ts]), PR._pooled_r2(pred_ny, Ys[ts])
    assert abs(r2_ex - r2_ny) < 0.05, (
        f"nystrom-vs-exact gap {abs(r2_ex - r2_ny):.4f} (ex {r2_ex:.4f}, ny {r2_ny:.4f})"
    )

    # (5) minibatched MLP + residual-skip bodies run and beat the mean baseline (R2>0).
    pred_mlp, mm = fit_mlp(Xs, Ys, tr, ts, 64, 3e-3, 40, 64, 0, dev)
    assert PR._pooled_r2(pred_mlp, Ys[ts]) > 0.0, "MLP body did not beat the mean baseline"
    pred_res, _ = fit_residual_skip(Xs, Ys, tr, vl, ts, lambdas, 64, 3e-3, 40, 64, 0, dev, 64)
    assert PR._pooled_r2(pred_res, Ys[ts]) > 0.0, (
        "residual-skip body did not beat the mean baseline"
    )

    # (8) residual arm never materializes train-pool preds for all lambdas (#779 r6):
    #     (a) the multi-lambda ridge path REFUSES the train pool as an eval set;
    #     (b) fit_residual_skip never routes the train pool through it (recorded live).
    guard_fired = False
    try:
        _ridge_streaming_multi_lambda(Xs, Ys, tr, [vl, tr, ts], lambdas, dev, block=64)
    except ValueError as e:
        guard_fired = "train pool" in str(e)
    assert guard_fired, "multi-lambda train-pool guard did not fire on [val, tr, te]"

    ml_calls: list = []
    _orig_ml = _ridge_streaming_multi_lambda

    def _rec_ml(X_, Y_, tr_, eidx_, *a, **k):
        ml_calls.append(list(eidx_))
        return _orig_ml(X_, Y_, tr_, eidx_, *a, **k)

    _mod = sys.modules[__name__]
    _mod._ridge_streaming_multi_lambda = _rec_ml
    try:
        pred_res8, _ = fit_residual_skip(Xs, Ys, tr, vl, ts, lambdas, 64, 3e-3, 40, 64, 0, dev, 64)
    finally:
        _mod._ridge_streaming_multi_lambda = _orig_ml
    assert not any(_is_train_pool(e, tr) for c in ml_calls for e in c), (
        "fit_residual_skip passed the train pool to the multi-lambda ridge path"
    )
    assert pred_res8.shape[0] == len(ts), "residual arm returned wrong test-pred count"

    # (6) HF stream checkpoint write/resume roundtrip + (7) per-chunk download retry.
    with tempfile.TemporaryDirectory() as td:
        n_rows_ck, n_ck = _smoke_stream_ckpt(Path(td) / "ck")
        n_att = _smoke_download_retry(Path(td) / "rt")

    logger.info(
        "[smoke] PASS: split shas byte-id; select (lmsys/mixed-ratio %.2f/whole); "
        "ridge==N50 (<1e-4); nystrom~exact (ex %.3f ny %.3f); MLP+residual R2>0 (mlp %d ep); "
        "residual multi-lambda train-pool guard fires + residual never routes train pool; "
        "stream-ckpt resume byte-identical (%d rows / %d chunks); download-retry %d-attempt path",
        dm["full_lmsys_frac"],
        r2_ex,
        r2_ny,
        mm["epochs_ran"],
        n_rows_ck,
        n_ck,
        n_att,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 n1m fits (up to n_train=1,000,000).")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", default=",".join(p[0] for p in FIT_POINTS))
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=F.BOOT_N)
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--mlp-batch", type=int, default=MLP_BATCH)
    ap.add_argument("--mlp-lr", type=float, default=3e-4)
    ap.add_argument("--ridge-block", type=int, default=RIDGE_BLOCK)
    ap.add_argument("--krr-nystrom-centers", type=int, default=8192)
    ap.add_argument("--krr-gamma-mult", default=",".join(str(g) for g in KRR_GAMMA_MULT))
    ap.add_argument("--krr-lambdas", default=",".join(str(x) for x in KRR_LAMBDAS))
    ap.add_argument("--krr-validate-tol", type=float, default=0.01)
    ap.add_argument("--no-validate-krr", action="store_true", help="skip the Nystrom-vs-exact gate")
    ap.add_argument("--pass-b", type=Path, default=N1G.PASS_B_LOCAL)
    ap.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR)
    ap.add_argument("--manifest-from-hf", action="store_true")
    ap.add_argument("--n1m-capture-dir", type=Path, default=None)
    ap.add_argument("--hf-prefix", default=f"{N1G.HF_PREFIX}/final_token_capture")
    ap.add_argument(
        "--manifest-hf-prefix",
        default=N1G.HF_PREFIX,
        help="HF ROUND-ROOT prefix for the sampling manifest (the manifest lives at "
        "<manifest-hf-prefix>/sampling_manifest); distinct from --hf-prefix, the "
        "capture prefix <round-root>/final_token_capture",
    )
    ap.add_argument(
        "--fresh-stream",
        action="store_true",
        help="ignore any existing HF stream checkpoint and re-stream from scratch",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="CPU numeric-sanity smoke (synthetic; no data/GPU)"
    )
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    want_points = [p.strip() for p in args.points.split(",") if p.strip()]
    want_pred = [p.strip() for p in args.predictors.split(",") if p.strip()]
    for p in want_pred:
        if p not in PREDICTORS:
            raise ValueError(f"unknown predictor {p!r}")
    point_by_name = {p[0]: p for p in FIT_POINTS}
    for pn in want_points:
        if pn not in point_by_name:
            raise ValueError(f"unknown fit point {pn!r}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_json is None:
        args.out_json = args.out_dir / "n1m_fits.json"

    results = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    if results.get("layer") is not None and results["layer"] != args.layer:
        raise SystemExit(
            f"--out-json {args.out_json} was written for layer {results['layer']} but "
            f"--layer={args.layer}; refusing to mix cross-layer rows"
        )
    if results.get("seed") is not None and results["seed"] != args.seed:
        raise SystemExit(
            f"--out-json {args.out_json} written for seed {results['seed']}, not --seed={args.seed}"
        )

    t0 = time.time()
    X, Y, prov, orig_train, val, test, split = assemble(args, args.layer)
    pools = _pool_rows(prov, orig_train, X.shape[0], val, test)
    logger.info(
        "assembled: %d contexts (%d lmsys pool, %d full pool), val=%d test=%d (L%d, %.0fs)",
        X.shape[0],
        len(pools["lmsys"]),
        len(pools["full"]),
        len(val),
        len(test),
        args.layer,
        time.time() - t0,
    )
    lambdas = LAMBDAS_N1M
    gamma_mult = tuple(float(g) for g in args.krr_gamma_mult.split(",") if g.strip())
    krr_lambdas = tuple(float(x) for x in args.krr_lambdas.split(",") if x.strip())

    validation = results.get("nystrom_validation")
    if "krr_nystrom" in want_pred and not args.no_validate_krr and validation is None:
        validation = _validate_nystrom_vs_exact(
            X,
            Y,
            pools,
            val,
            test,
            m_centers=args.krr_nystrom_centers,
            gamma_mult=gamma_mult,
            krr_lambdas=krr_lambdas,
            seed=args.seed,
            dev=dev,
            tol=args.krr_validate_tol,
        )

    results.setdefault("per_point", {})
    results.update(
        {
            "layer": int(args.layer),
            "seed": int(args.seed),
            "split": split,
            "lambda_grid": {"n": len(lambdas), "min": float(lambdas[0]), "max": float(lambdas[-1])},
            "krr_grid": {
                "gamma_mult": list(gamma_mult),
                "lambdas": list(krr_lambdas),
                "nystrom_centers": int(args.krr_nystrom_centers),
            },
            "nystrom_validation": validation,
            "predictor_labels": PREDICTOR_LABEL,
            "fit_points": {p[0]: {"n_train_target": p[1], "corpus_mode": p[2]} for p in FIT_POINTS},
            "note": (
                "n_train up to 1,000,000 rerun of fitter-fair-comparison over the LMSYS-exhaust + "
                "WildChat-balance n1m corpus. val/test BYTE-IDENTICAL to the ORIGINAL round "
                "(asserted vs pinned shas). 4 points: lmsys_150k/500k (pure), mixed_500k "
                "(ratio-matched control), mixed_1m (full pool). ridge=streaming primal; mlp mini; "
                "residual-skip=ridge+MLP; krr=Nystrom (validated vs exact at n=50k)."
            ),
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_ffc_n1m_fits", "device": args.device}
            ),
        }
    )
    C.write_json_atomic(args.out_json, results)

    _run_fit_points(
        results,
        want_points,
        want_pred,
        point_by_name,
        X,
        Y,
        pools,
        val,
        test,
        lambdas,
        gamma_mult,
        krr_lambdas,
        args,
        dev,
    )
    logger.info("wrote %s (%.0fs total)", args.out_json, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
