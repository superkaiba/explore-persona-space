"""Issue #1902 P4 — fits, transfer, operator battery (plan v4 §4 P4 steps 1-9).

Library module invoked by ``issue1902_run.py --phase fits`` (registration
point; the dispatcher never calls this file directly). One process, ALL
visible GPUs: independent fit units — per-(cell, fold) layer-chunked ridge
slices, MLP fits, transfer cells, star/robust/reliability reads, CKA +
operator units — are pulled from ONE queue by device-pinned worker threads
(the plan §9 P4 work-conserving Must-Fix: no wave barrier; a worker pulls the
next pending unit the moment it frees). Two dependency stages only:

- stage 1: diagonal layer sweep (both arms, every store layer) + CKA
  descriptives — no cross-unit dependency;
- stage 2 (needs the stage-1 layer* selection): 4x4 grid at the layer* band,
  MLP arm, transfer battery (direct / general-linear / orthogonal-Procrustes /
  fixed-answer-text + matched nulls), star reads (per-dim SS, shuffled-pairing
  null, reliability split-half, robust-native compare), operator battery.

Contracts baked in:

- Ridge fits go through ``fit_h.ridge_fit_predict_fast_layer_batched``
  (float64, GCV over logspace(-2,4,13), per-slice lambda) behind the
  MANDATORY slow-vs-fast parity gate (>=3 slices at the run's own shape vs
  the ``ridge_fit_predict`` SVD path, tol <= 1e-4 max rel diff — the fast
  twin's docstring contract). Parity failure is a DESIGNED halt (rc=7),
  demoted to an informational line under --smoke (gate-calibration rule).
- cuSOLVER eigh non-convergence -> CPU fallback wrap at the call site
  (gotchas ``_eigh_robust`` pattern; same matrix, same decomposition).
- Standing baselines on EVERY fitted map cell: identity+learned-bias
  (``mapping_baselines.identity_bias_predict``) + kNN retrieval
  (``knn_retrieval``; ks clamped to the fold pool; chance = k/n_pool).
- Per-unit persistence + resume predicates keyed on every output-affecting
  regime key; ``[fits] unit k/N`` progress lines; per-context held-out SS
  components persisted per cell (``percell/*.npz``) — diagonals at the FULL
  layer grain (the selection-inherited CI reads it), grid/transfer at the
  band — so every bootstrap read is a batched masked-sum GEMM re-reduction,
  never a refit.
- P4-entry pilot: the FIRST sweep unit is timed at production shape and the
  sweep is projected against the plan §9 P4 row (abort ratio 2x, designed
  halt; informational under --smoke).

Content hygiene: corpus/rollout text NEVER enters logs or outputs — ids,
indices, counts, and float summaries only.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# load_dotenv() BEFORE any heavy import so the shared-VM thread caps (#847)
# bind in-process (tests/test_shared_vm_thread_caps.py, the #1146 predicate).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1902_common as C  # noqa: E402
import issue1902_run as R  # noqa: E402
from explore_persona_space.eval.vllm_util import GPU_FREE_MARGIN_GIB  # noqa: E402

logger = R.logger

FITS_RECIPE_VERSION = "issue1902-fits-v1"

# Plan §9 P4 row (planned_wall_h = 1.0) + the pre-registered pilot abort.
P4_PLANNED_WALL_H = 1.0
P4_PILOT_ABORT_RATIO = 2.0

# Parity gate (plan §4 P4 step 1; fit_h layer-batched docstring tolerance).
PARITY_TOL = 1e-4
PARITY_MIN_SLICES = 3

# Layer chunking for the batched Gram eigh (plan §4 P4 step 1 / A14).
LAYER_CHUNK = int(os.environ.get("EPM_ISSUE1902_LAYER_CHUNK", "8"))
# Per-layer HBM allowance multiplier over the (n_tr, n_tr) fp64 Gram — Gram +
# eigenvectors + cuSOLVER workspace + X/Y/pred transfer slack. Arithmetic
# allowance over plan A14's own basis (8 layers x 8.3k^2 fp64 ~= 4.4 GB "+
# workspace"), used ONLY to SHRINK below the plan-fixed LAYER_CHUNK with a
# fail-loud 1-layer floor — never to size new spend (the #811 measured-peak
# rule governs upward sizing; a downscale-only clamp with a loud floor is the
# conservative direction).
EIGH_WORKSPACE_FACTOR = 4

# Headline flank band: layer* +/- {2, 4} within the captured layer set.
BAND_HALF_WIDTH = 4

# Draw counts (production / smoke). Bootstrap floor 200 is the plan's
# pre-registered descope lever (§4 step 9).
N_BOOT = int(os.environ.get("EPM_ISSUE1902_N_BOOT", "1000"))
N_ROT_DRAWS = 50  # issue1345_common.N_ROTATION_COSINE_DRAWS convention
N_NULL_DRAWS = 4  # shuffled-correspondence + spectrum-matched, per fold
N_SHUFFLE_DRAWS = 3  # shuffled-pairing null draws per (diag cell, fold)
SMOKE_N_BOOT = 50
SMOKE_N_ROT = 4
SMOKE_N_NULL = 2
SMOKE_N_SHUFFLE = 2

KNN_KS = (1, 5, 10)
GATE_B_FLOOR = 0.15  # §7 gate B sanity floor (context arm)
MLP_PCA_K = 64
SPECMATCH_ENERGY = 0.99  # spectrum-matched null retained SV energy (#825)

ARM_CTX = "ctx"
ARM_PRE = "pre"
ADJACENT_PAIRS = (("B", "S"), ("S", "D"), ("D", "R"))
XFER_MODES = ("direct", "gl", "orth", "fixedtext")


def realized_transitions(ckpts: list[str]) -> list[tuple[str, str]]:
    """Adjacent stage transitions realized in THIS run's checkpoint set; when
    none are (the smoke's {B, R} chain), fall back to consecutive realized
    checkpoints so the transition-consuming paths stay smoke-exercised."""
    adj = [(a, b) for a, b in ADJACENT_PAIRS if a in ckpts and b in ckpts]
    if adj:
        return adj
    return [(ckpts[k], ckpts[k + 1]) for k in range(len(ckpts) - 1)]


# ── small helpers ────────────────────────────────────────────────────────────


def _eval_dir(out_root: Path, smoke: bool) -> Path:
    """Eval-results root. Smoke NEVER writes the repo tree (scratch redirect)."""
    env = os.environ.get("EPM_ISSUE1902_EVAL_DIR")
    if env:
        return Path(env)
    if smoke:
        return out_root / "eval_results" / "issue_1902"
    return R.PROJECT_ROOT / "eval_results" / "issue_1902"


def _savez_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Atomic plain (uncompressed — #813) savez; tmp keeps the .npz suffix
    (np.savez APPENDS .npz to any other name — gotchas #1092)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _pooled_r2(ss_res: float, ss_tot: float) -> float:
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _batched_ridge(Xtr, Ytr, Xev, *, device: str, **kw):
    """fit_h layer-batched ridge with the cuSOLVER-eigh CPU fallback wrap
    (gotchas: cuda eigh raises LinAlgError on near-singular Grams that CPU
    LAPACK decomposes fine — exact backend swap, never a jitter)."""
    import torch

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    try:
        return ridge_fit_predict_fast_layer_batched(Xtr, Ytr, Xev, device=device, **kw)
    except torch.linalg.LinAlgError:
        if device == "cpu":
            raise
        logger.warning("[fits] cuda eigh non-convergence (n=%s) — CPU fallback", Xtr.shape)
        return ridge_fit_predict_fast_layer_batched(Xtr, Ytr, Xev, device="cpu", **kw)


def layer_chunk_cap_for_free(free_bytes: int, n_tr: int, chunk: int = LAYER_CHUNK) -> int:
    """A14 layer chunk clamped by LIVE free HBM (free, never total).

    The fellows H200 hosts share nodes without GPU isolation, so plan A14's
    ">= 40 GB free" assumption is a co-tenancy hypothesis, not a device
    property (#1902 crash 1 sweep). Per-layer cost ~= n_tr^2 x 8 B (fp64
    Gram) x ``EIGH_WORKSPACE_FACTOR``. Downscale-only: an exclusive host
    resolves to ``chunk`` unchanged; raises ``RuntimeError`` when even one
    layer does not fit under ``GPU_FREE_MARGIN_GIB`` margin.
    """
    per_layer = n_tr * n_tr * 8 * EIGH_WORKSPACE_FACTOR
    if per_layer <= 0:
        return chunk
    usable = free_bytes - int(GPU_FREE_MARGIN_GIB * 2**30)
    cap = usable // per_layer
    if cap < 1:
        raise RuntimeError(
            f"GPU too full for even a 1-layer Gram eigh: free={free_bytes / 2**30:.1f} GiB, "
            f"need ~{per_layer / 2**30:.1f} GiB/layer (n_tr={n_tr}, factor "
            f"{EIGH_WORKSPACE_FACTOR}) + {GPU_FREE_MARGIN_GIB:.0f} GiB margin — shared-node "
            "co-tenancy (fellows H200) is the expected cause; re-dispatch when the device frees."
        )
    return min(chunk, int(cap))


def _layer_chunk_cap(device: str, n_tr: int) -> int:
    """Device wrapper: mem_get_info on the worker's cuda device (A14's own
    verification hook — "P4 entry ... logs mem_get_info"); CPU = unclamped."""
    if not str(device).startswith("cuda"):
        return LAYER_CHUNK
    import torch

    free_b, total_b = torch.cuda.mem_get_info(torch.device(device))
    cap = layer_chunk_cap_for_free(free_b, n_tr)
    if cap < LAYER_CHUNK:
        logger.warning(
            "[fits] layer chunk %d -> %d (free=%.1fGiB total=%.1fGiB n_tr=%d — A14 free-HBM clamp)",
            LAYER_CHUNK,
            cap,
            free_b / 2**30,
            total_b / 2**30,
            n_tr,
        )
    else:
        logger.info(
            "[fits] layer chunk %d (free=%.1fGiB total=%.1fGiB n_tr=%d)",
            cap,
            free_b / 2**30,
            total_b / 2**30,
            n_tr,
        )
    return cap


def _knn_ks(n_pool: int) -> tuple[int, ...]:
    ks = tuple(k for k in KNN_KS if k <= n_pool)
    return ks if ks else (1,)


def _cell_baselines(x_tr, y_tr, x_ev, y_ev, pred) -> dict[str, Any]:
    """Standing baselines for one fitted (cell, layer, fold) slice: identity+
    learned-bias R^2 + kNN retrieval (euclidean + cosine) of the FIT's preds."""
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    id_pred = identity_bias_predict(x_tr, y_tr, x_ev)
    ss_res_id = float(((y_ev - id_pred) ** 2).sum())
    ss_tot = float(((y_ev - y_tr.mean(0)) ** 2).sum())
    ks = _knn_ks(y_ev.shape[0])
    return {
        "identity_r2": _pooled_r2(ss_res_id, ss_tot),
        "knn": {
            metric: knn_retrieval(pred, y_ev, ks=ks, metric=metric)
            for metric in ("euclidean", "cosine")
        },
    }


def _per_ctx_ss(pred: np.ndarray, y_ev: np.ndarray, y_tr_mean: np.ndarray):
    """Per-context residual/total SS + per-context cosine (recon quality)."""
    res = ((y_ev - pred) ** 2).sum(axis=-1)
    tot = ((y_ev - y_tr_mean) ** 2).sum(axis=-1)
    num = (pred * y_ev).sum(axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(y_ev, axis=-1) + 1e-12
    return res, tot, num / den


# ── corpus index + store cache ───────────────────────────────────────────────


class CorpusIndex:
    """Row-aligned view of one corpus' intersection: ids, groups, clusters,
    classes, fold assignment (from ``intersection_manifest.json`` + the ctx
    store row_index). Row ORDER is the store order; asserted identical across
    checkpoints (fail loud — the matched-target discipline)."""

    def __init__(self, out_root: Path, corpus: str, ckpts: list[str], store: Path):
        manifest = R._read_json(out_root / "gen" / "intersection_manifest.json")
        entry = manifest["corpora"][corpus]
        self.corpus = corpus
        self.fold_of_group: dict[str, int] = dict(entry["fold_of_group"])
        self.n_folds = int(entry["n_folds"])
        ref = None
        for m in ckpts:
            idx_path = store / C.cell_row_index_relpath(m, C.CTX_SOURCE, corpus)
            rows = R._read_jsonl(idx_path)
            ids = [r["id"] for r in rows]
            if ref is None:
                ref = ids
                self.rows = rows
            elif ids != ref:
                raise RuntimeError(
                    f"store row_id order mismatch for ckpt {m} corpus {corpus} "
                    f"(n={len(ids)} vs ref n={len(ref)}) — matched-target violation"
                )
        assert ref is not None
        if set(ref) != set(entry["ids"]):
            raise RuntimeError(
                f"store rows != intersection manifest ids for {corpus}: "
                f"{len(ref)} store vs {len(entry['ids'])} manifest"
            )
        self.ids = ref
        self.groups = [r["group"] for r in self.rows]
        self.clusters = [int(r.get("cluster", -1)) for r in self.rows]
        self.classes = [r.get("class") or "generic" for r in self.rows]
        missing = sorted({g for g in self.groups if g not in self.fold_of_group})
        if missing:
            raise RuntimeError(f"groups missing from fold assignment ({corpus}): {missing[:5]}")
        self.fold = np.asarray([self.fold_of_group[g] for g in self.groups], dtype=np.int64)
        self.n = len(self.ids)


class StoreCache:
    """Thread-safe byte-bounded LRU over store shard files (fp16 on disk ->
    fp32 numpy in cache). One shared cache across worker threads."""

    def __init__(self, store: Path, cap_gb: float | None = None):
        self.store = store
        self.cap = (
            cap_gb if cap_gb is not None else float(os.environ.get("EPM_ISSUE1902_CACHE_GB", "48"))
        ) * 1e9
        self._lock = threading.Lock()
        self._files: dict[str, dict[str, np.ndarray]] = {}
        self._order: list[str] = []
        self._bytes = 0

    def _load(self, relpath: str) -> dict[str, np.ndarray]:
        import torch

        with self._lock:
            if relpath in self._files:
                self._order.remove(relpath)
                self._order.append(relpath)
                return self._files[relpath]
        path = self.store / relpath
        d = torch.load(path, map_location="cpu", weights_only=True)
        out = {
            k: v.to(torch.float32).numpy()
            for k, v in d.items()
            if k != "row_ids" and hasattr(v, "to")
        }
        out["__row_ids__"] = np.asarray(d["row_ids"])
        nbytes = sum(a.nbytes for a in out.values())
        with self._lock:
            if relpath in self._files:
                # A concurrent worker won the miss race while we loaded outside
                # the lock: discard OUR copy and return the cached entry — an
                # unconditional insert duplicates the _order token + double-counts
                # bytes, and the first eviction over the cap then KeyErrors
                # (concern storecache-lru-duplicate-insert-race).
                self._order.remove(relpath)
                self._order.append(relpath)
                return self._files[relpath]
            self._files[relpath] = out
            self._order.append(relpath)
            self._bytes += nbytes
            while self._bytes > self.cap and len(self._order) > 1:
                old = self._order.pop(0)
                evicted = self._files.pop(old, None)  # pop-with-skip: never KeyError
                if evicted is not None:
                    self._bytes -= sum(a.nbytes for a in evicted.values())
        return out

    def answer(self, ckpt: str, src: str, corpus: str, layer: int, ids: list[str]) -> np.ndarray:
        d = self._load(C.answer_store_relpath(ckpt, src, corpus, layer))
        if list(d["__row_ids__"]) != ids:
            raise RuntimeError(f"row_id mismatch in answer cell {ckpt}/{src}/{corpus} L{layer}")
        return d["w"]

    def ctx(self, ckpt: str, corpus: str, layer: int, key: str, ids: list[str]) -> np.ndarray:
        d = self._load(C.ctx_store_relpath(ckpt, corpus, layer))
        if list(d["__row_ids__"]) != ids:
            raise RuntimeError(f"row_id mismatch in ctx store {ckpt}/{corpus} L{layer}")
        return d[key]

    def subdir(self, relpath: str) -> dict[str, np.ndarray]:
        return self._load(relpath)


# ── local-store presence + HF re-stage (plan §4 P4 "re-staged from HF via
#    stage_hub_prefix otherwise"; concern p3-delete-local-starves-p4-store) ────


def _store_hub_relpath(hub_path: str) -> str:
    """PURE hub-path -> store-relative mapping (#928: ONE shared map feeds the
    missing-check, the fetch targets, and the completeness re-check)."""
    prefix = f"{C.STORE_HF_PATH}/"
    if not hub_path.startswith(prefix):
        raise ValueError(f"hub path {hub_path!r} outside the store prefix {prefix!r}")
    return hub_path[len(prefix) :]


def expected_store_leaves(
    ckpt: str, ckpts: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """P4-consumed store leaf dirs per top-level restage prefix (store-relative).

    HARD leaves (grid: ctx + answer cells) crash P4 outright when absent;
    SOFT leaves (reliability / robust_native) have graceful consumer branches
    but silently degrade the registered §6 reads when delete-local reaped them.
    """
    hard = {
        ckpt: [f"{ckpt}/{C.CTX_SOURCE}/{corpus}" for corpus in C.CORPORA]
        + [f"{ckpt}/{src}/{corpus}" for src in ckpts for corpus in C.CORPORA]
    }
    soft = {
        f"reliability/{ckpt}": [
            f"reliability/{ckpt}/{C.CORPUS_SINGLE}/seed{s}" for s in C.RELIABILITY_SEEDS
        ]
    }
    if ckpt != "B":
        soft[f"robust_native/{ckpt}"] = [
            f"robust_native/{ckpt}/{C.CORPUS_SINGLE}",
            f"robust_native/{ckpt}/{C.CORPUS_SINGLE}/ctx",
        ]
    return hard, soft


def _leaf_complete(store: Path, rel: str) -> bool:
    """A leaf is consumable iff it holds >=1 layer shard (row ids ride inside
    the .pt); the grid CTX leaf additionally needs the row_index.jsonl that
    CorpusIndex hard-reads."""
    d = store / rel
    if not any(d.glob("L*.pt")):
        return False
    if rel.split("/")[1:2] == [C.CTX_SOURCE]:
        return (d / "row_index.jsonl").exists()
    return True


def _restage_store_prefix(store: Path, prefix_rel: str) -> int:
    """Scoped-listing + per-file staged restore of ONE store prefix from HF.

    `hub.stage_hub_file` takes an EXACT per-file target (no #1774 mirror-root
    trap), is atomic + retried, and skips already-present files, so a partial
    delete heals idempotently. One resolved revision covers every file (#833).
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    hub_prefix = f"{C.STORE_HF_PATH}/{prefix_rel}"
    info = hub.retry_transient(
        lambda: api.repo_info(C.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({C.HF_DATA_REPO})",
    )
    revision = str(info.sha)
    files = hub.list_hf_files_under_path(
        api, C.HF_DATA_REPO, hub_prefix, repo_type="dataset", revision=revision
    )
    if not files:
        raise FileNotFoundError(f"no files on HF under {C.HF_DATA_REPO}:{hub_prefix}")
    targets = {f: store / _store_hub_relpath(f) for f in files}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(
                hub.stage_hub_file,
                C.HF_DATA_REPO,
                f,
                tgt,
                repo_type="dataset",
                revision=revision,
            )
            for f, tgt in targets.items()
        ]
        for fut in futs:
            fut.result()  # re-raises — fail-loud
    return len(files)


def ensure_store_staged(out_root: Path, ckpts: list[str]) -> dict[str, int]:
    """Plan §4 P4 first line: fits read the LOCAL store when present; when P3's
    verified upload -> delete-local reaped it, RE-STAGE the missing per-ckpt
    prefixes from HF before any fit.

    The capture leg's verified upload record (state/capture_upload_<m>.done.json,
    written only after upload_dir_sharded verify=True) — NOT bare local
    existence — is what licenses proceeding past deleted artifacts (#1315
    class): HARD leaves missing with NO record fail loud (run capture first);
    SOFT leaves missing with no record only warn (their consumers' graceful
    branches own genuinely-absent stores, e.g. partial fixtures)."""
    store = R._store_root(out_root)
    restaged: dict[str, int] = {}
    for m in ckpts:
        hard, soft = expected_store_leaves(m, ckpts)
        upload_record = R._state_dir(out_root) / f"capture_upload_{m}.done.json"
        missing_hard = {
            p: [leaf for leaf in ls if not _leaf_complete(store, leaf)] for p, ls in hard.items()
        }
        missing_hard = {p: ls for p, ls in missing_hard.items() if ls}
        missing_soft = {
            p: [leaf for leaf in ls if not _leaf_complete(store, leaf)] for p, ls in soft.items()
        }
        missing_soft = {p: ls for p, ls in missing_soft.items() if ls}
        if not upload_record.exists():
            if missing_hard:
                sample = sorted(x for ls in missing_hard.values() for x in ls)[:4]
                raise FileNotFoundError(
                    f"store leaves missing for ckpt {m} (e.g. {sample}) with no verified "
                    f"upload record at {upload_record} — run --phase capture first"
                )
            if missing_soft:
                logger.warning(
                    "[fits] ckpt %s: soft store leaves missing with no upload record "
                    "(reliability/robust reads will degrade): %s",
                    m,
                    missing_soft,
                )
            continue
        to_stage = sorted(set(missing_hard) | set(missing_soft))
        if not to_stage:
            continue
        n = sum(_restage_store_prefix(store, p) for p in to_stage)
        by_prefix = {**hard, **soft}
        still = [leaf for p in to_stage for leaf in by_prefix[p] if not _leaf_complete(store, leaf)]
        if still:
            raise FileNotFoundError(
                f"HF re-stage left store leaves incomplete for ckpt {m}: {still} "
                f"(hub prefix {C.STORE_HF_PATH})"
            )
        restaged[m] = n
        print(
            f"[fits] re-staged {n} store files for ckpt {m} "
            f"from {C.HF_DATA_REPO}:{C.STORE_HF_PATH} (prefixes {to_stage})",
            flush=True,
        )
    return restaged


def discover_layers(store: Path, ckpt: str, corpus: str) -> list[int]:
    """Capture layers from the ctx store on disk (never recomputed from a
    hardcoded set — A6)."""
    d = (store / C.ctx_store_relpath(ckpt, corpus, 0)).parent
    layers = sorted(int(p.stem[1:]) for p in d.glob("L*.pt"))
    if not layers:
        raise FileNotFoundError(f"no ctx store layers under {d} — run --phase capture first")
    return layers


def band_layers(layers: list[int], star: int) -> list[int]:
    return [layer for layer in layers if abs(layer - star) <= BAND_HALF_WIDTH]


# ── worker pool (device-pinned threads; work-conserving queue) ───────────────


class _WorkerPool:
    """Device-pinned worker threads over ONE unit queue (plan §9 P4
    Must-Fix): a worker pulls the next pending unit the moment it frees —
    no wave barrier. torch GPU ops release the GIL, so threads shard the
    GPUs from one process (no CVD pinning needed: single process owns all
    visible devices)."""

    def __init__(self, ctx: "FitsContext"):
        import torch

        n_env = os.environ.get("EPM_ISSUE1902_FIT_WORKERS")
        if torch.cuda.is_available():
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.devices = ["cpu"]
        if n_env:
            n = max(1, int(n_env))
            self.devices = [self.devices[i % len(self.devices)] for i in range(n)]
        self.ctx = ctx

    def run(self, stage: str, units: list[dict]) -> None:
        pending = [u for u in units if not R.unit_done(self.ctx.out_root, u["unit"], u["regime"])]
        logger.info(
            "[fits] stage=%s units=%d/%d pending workers=%s",
            stage,
            len(pending),
            len(units),
            self.devices,
        )
        if not pending:
            return
        q: queue.Queue = queue.Queue()
        for u in pending:
            q.put(u)
        errors: list[BaseException] = []
        stop = threading.Event()
        done_n = [0]
        lock = threading.Lock()
        t0 = time.time()

        def _worker(device: str) -> None:
            while not stop.is_set():
                try:
                    u = q.get_nowait()
                except queue.Empty:
                    return
                tu = time.time()
                try:
                    info = u["fn"](self.ctx, device=device, **u.get("kw", {}))
                    R.mark_unit_done(
                        self.ctx.out_root, u["unit"], u["regime"], {"wall_s": time.time() - tu}
                    )
                    with lock:
                        done_n[0] += 1
                        k = done_n[0]
                    print(
                        f"[fits] unit {k}/{len(pending)} {u['unit']} "
                        f"elapsed={time.time() - tu:.1f}s total={time.time() - t0:.0f}s",
                        flush=True,
                    )
                    if info and info.get("pilot_gate"):
                        self.ctx.pilot_timings.append(info["pilot_gate"])
                except BaseException as e:  # noqa: BLE001 — re-raised after join (fail fast)
                    stop.set()
                    errors.append(e)
                    return
                finally:
                    q.task_done()

        threads = [
            threading.Thread(target=_worker, args=(dev,), daemon=True) for dev in self.devices
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]


# ── shared run context ───────────────────────────────────────────────────────


class FitsContext:
    def __init__(self, args: argparse.Namespace, out_root: Path, ckpts: list[str]):
        self.args = args
        self.smoke = bool(args.smoke)
        self.out_root = out_root
        self.ckpts = ckpts
        self.store = R._store_root(out_root)
        self.eval_dir = _eval_dir(out_root, self.smoke)
        self.cache = StoreCache(self.store)
        self.corpora = {
            corpus: CorpusIndex(out_root, corpus, ckpts, self.store) for corpus in C.CORPORA
        }
        layer_sets = {
            (m, corpus): discover_layers(self.store, m, corpus)
            for m in ckpts
            for corpus in C.CORPORA
        }
        ref = next(iter(layer_sets.values()))
        if any(ls != ref for ls in layer_sets.values()):
            raise RuntimeError(f"inconsistent store layer sets: {layer_sets}")
        self.layers = ref
        self.n_boot = SMOKE_N_BOOT if self.smoke else N_BOOT
        self.n_rot = SMOKE_N_ROT if self.smoke else N_ROT_DRAWS
        self.n_null = SMOKE_N_NULL if self.smoke else N_NULL_DRAWS
        self.n_shuffle = SMOKE_N_SHUFFLE if self.smoke else N_SHUFFLE_DRAWS
        self.pilot_timings: list[dict] = []
        # stage-2 state (filled by the selection step)
        self.layer_star: int | None = None
        self.layer_star_p: int | None = None
        self.band: list[int] = []
        self.band_p: list[int] = []

    # -- data slices -----------------------------------------------------------

    def xy(self, m: str, s: str, corpus: str, layer: int, arm: str):
        idx = self.corpora[corpus]
        key = "u_mean" if arm == ARM_CTX else "p_mean"
        X = self.cache.ctx(m, corpus, layer, key, idx.ids)
        Y = self.cache.answer(m, s, corpus, layer, idx.ids)
        return X, Y

    def fold_masks(self, corpus: str, fold: int):
        idx = self.corpora[corpus]
        ev = idx.fold == fold
        tr = ~ev
        return tr, ev

    def unit_paths(self) -> tuple[Path, Path]:
        return self.eval_dir / "fits" / "units", self.eval_dir / "fits" / "percell"

    def write_unit(self, unit: str, payload: dict) -> None:
        units_dir, _ = self.unit_paths()
        units_dir.mkdir(parents=True, exist_ok=True)
        R._write_json_atomic(units_dir / f"{unit}.json", payload)

    def read_unit(self, unit: str) -> dict:
        units_dir, _ = self.unit_paths()
        return R._read_json(units_dir / f"{unit}.json")


# ── parity gate (plan §4 P4 step 1 — MANDATORY, first) ──────────────────────


def parity_gate(ctx: FitsContext, device: str) -> dict:
    """Slow (ridge_fit_predict SVD) vs fast (layer-batched Gram-eigh) parity
    on >=3 slices at THIS run's shape. max rel diff <= 1e-4 or designed halt
    (informational under --smoke)."""
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    mid = ctx.layers[len(ctx.layers) // 2]
    slices = [
        (ctx.ckpts[0], C.CORPUS_SINGLE, mid),
        (ctx.ckpts[-1], C.CORPUS_SINGLE, ctx.layers[0]),
        (ctx.ckpts[-1], C.CORPUS_MULTI, mid),
    ]
    report: dict[str, Any] = {"tol": PARITY_TOL, "slices": []}
    worst = 0.0
    for m, corpus, layer in slices[:PARITY_MIN_SLICES]:
        X, Y = ctx.xy(m, m, corpus, layer, ARM_CTX)
        tr, ev = ctx.fold_masks(corpus, 0)
        if ev.sum() < 2 or tr.sum() < 2:
            continue
        slow = ridge_fit_predict(X[tr], Y[tr], X[ev])
        fast = _batched_ridge(X[tr][None], Y[tr][None], X[ev][None], device=device)[0]
        denom = max(float(np.abs(slow).max()), 1e-12)
        rel = float(np.abs(fast - slow).max() / denom)
        worst = max(worst, rel)
        report["slices"].append(
            {"ckpt": m, "corpus": corpus, "layer": layer, "n_tr": int(tr.sum()), "max_rel": rel}
        )
    report["max_rel_diff"] = worst
    report["pass"] = bool(worst <= PARITY_TOL and len(report["slices"]) >= PARITY_MIN_SLICES)
    R._write_json_atomic(ctx.eval_dir / "fits" / "parity_gate.json", report)
    logger.info("[fits] parity gate: %s", report)
    if not report["pass"]:
        if ctx.smoke:
            logger.info("[fits] SMOKE: parity gate demoted to informational (verdict recorded)")
        else:
            R.designed_halt(ctx.out_root, "ridge_parity", report)
    return report


# ── stage-1 units: diagonal layer sweep + CKA ────────────────────────────────


def run_sweep_unit(ctx: FitsContext, device: str, *, m: str, corpus: str, fold: int) -> dict:
    """Diagonal (m,m,corpus) fold fit at EVERY store layer, both arms
    (ctx always; pre on multi). Layer-chunked batched ridge; identity R^2 per
    layer; kNN deferred to the band (grid stage). Persists full-grain
    per-context SS (the selection-inherited CI input)."""
    idx = ctx.corpora[corpus]
    tr, ev = ctx.fold_masks(corpus, fold)
    unit = f"sweep_{m}_{corpus}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True, "n_tr": int(tr.sum()), "n_ev": int(ev.sum())})
        logger.warning("[fits] %s SKIPPED (fold gate: n_tr=%d n_ev=%d)", unit, tr.sum(), ev.sum())
        return {}
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    arms = [ARM_CTX] + ([ARM_PRE] if corpus == C.CORPUS_MULTI else [])
    _, percell = ctx.unit_paths()
    out: dict[str, Any] = {"n_tr": int(tr.sum()), "n_ev": int(ev.sum()), "arms": {}}
    t0 = time.time()
    # A14 free-HBM clamp (#1902 crash 1 sweep): chunk width from LIVE free
    # memory on THIS worker's device, never the plan-fixed constant alone.
    chunk_cap = _layer_chunk_cap(device, int(tr.sum()))
    for arm in arms:
        per_layer: dict[str, Any] = {}
        res_all = np.zeros((len(ctx.layers), int(ev.sum())))
        tot_all = np.zeros_like(res_all)
        cos_all = np.zeros_like(res_all)
        for c0 in range(0, len(ctx.layers), chunk_cap):
            chunk = ctx.layers[c0 : c0 + chunk_cap]
            Xtr = np.stack([ctx.xy(m, m, corpus, layer, arm)[0][tr] for layer in chunk])
            Ytr = np.stack([ctx.xy(m, m, corpus, layer, arm)[1][tr] for layer in chunk])
            Xev = np.stack([ctx.xy(m, m, corpus, layer, arm)[0][ev] for layer in chunk])
            preds, info = _batched_ridge(Xtr, Ytr, Xev, device=device, return_info=True)
            for li, layer in enumerate(chunk):
                y_ev = ctx.xy(m, m, corpus, layer, arm)[1][ev]
                y_tr_mean = Ytr[li].mean(0)
                res, tot, cos = _per_ctx_ss(preds[li], y_ev, y_tr_mean)
                gi = c0 + li
                res_all[gi], tot_all[gi], cos_all[gi] = res, tot, cos

                id_pred = identity_bias_predict(Xtr[li], Ytr[li], Xev[li])
                per_layer[str(layer)] = {
                    "ss_res": float(res.sum()),
                    "ss_tot": float(tot.sum()),
                    "r2": _pooled_r2(float(res.sum()), float(tot.sum())),
                    "identity_r2": _pooled_r2(
                        float(((y_ev - id_pred) ** 2).sum()), float(tot.sum())
                    ),
                    "lambda_star": float(info["best_lambda"][li]),
                    "dof": float(info["dof"][li]),
                }
            del Xtr, Ytr, Xev, preds
        _savez_atomic(
            percell / f"diag_{m}_{corpus}_{arm}_f{fold}.npz",
            layers=np.asarray(ctx.layers),
            row_idx=np.flatnonzero(ev),
            ss_res=res_all,
            ss_tot=tot_all,
            cos=cos_all,
        )
        out["arms"][arm] = per_layer
    ctx.write_unit(unit, out)
    wall = time.time() - t0
    n_tr_d = int(tr.sum())
    if not ctx.smoke and n_tr_d < ctx.xy(m, m, corpus, ctx.layers[0], ARM_CTX)[0].shape[1]:
        logger.warning(
            "[fits] %s: n_tr=%d < d — under-determined regime (gate A' owns)", unit, n_tr_d
        )
    return {"pilot_gate": {"unit": unit, "wall_s": wall}}


def run_cka_unit(ctx: FitsContext, device: str, *, i: str, j: str, corpus: str) -> dict:
    """Per-layer linear CKA u_i<->u_j and w_ii<->w_jj over shared contexts
    (basis-stability descriptives for H4)."""
    import torch

    idx = ctx.corpora[corpus]
    dev = torch.device(device)
    rows: dict[str, list[float]] = {"u": [], "w": []}
    from explore_persona_space.analysis.representation_shift import linear_cka

    for layer in ctx.layers:
        u_i = torch.from_numpy(ctx.cache.ctx(i, corpus, layer, "u_mean", idx.ids)).to(dev)
        u_j = torch.from_numpy(ctx.cache.ctx(j, corpus, layer, "u_mean", idx.ids)).to(dev)
        rows["u"].append(linear_cka(u_i, u_j))
        w_i = torch.from_numpy(ctx.cache.answer(i, i, corpus, layer, idx.ids)).to(dev)
        w_j = torch.from_numpy(ctx.cache.answer(j, j, corpus, layer, idx.ids)).to(dev)
        rows["w"].append(linear_cka(w_i, w_j))
        del u_i, u_j, w_i, w_j
    ctx.write_unit(
        f"cka_{i}{j}_{corpus}",
        {"layers": ctx.layers, "cka_u": rows["u"], "cka_w": rows["w"], "n": idx.n},
    )
    return {}


# ── selection (layer*, bands) ────────────────────────────────────────────────


def select_layers(ctx: FitsContext) -> dict:
    """layer* = argmax over layers of MEAN diagonal pooled OOF R^2 across the
    4 checkpoints x 2 corpora (context arm); prefix arm selects layer*_p the
    same way over the multi-corpus prefix diagonals (plan §4 step 2)."""
    pooled: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    sums: dict[tuple, dict[int, list[float]]] = {}
    for m in ctx.ckpts:
        for corpus in C.CORPORA:
            arms = [ARM_CTX] + ([ARM_PRE] if corpus == C.CORPUS_MULTI else [])
            for arm in arms:
                agg: dict[int, list[float]] = {layer: [0.0, 0.0] for layer in ctx.layers}
                for fold in range(ctx.corpora[corpus].n_folds):
                    unit = f"sweep_{m}_{corpus}_f{fold}"
                    rec = ctx.read_unit(unit)
                    if rec.get("skipped"):
                        continue
                    for layer in ctx.layers:
                        e = rec["arms"][arm][str(layer)]
                        agg[layer][0] += e["ss_res"]
                        agg[layer][1] += e["ss_tot"]
                pooled.setdefault(m, {}).setdefault(corpus, {})[arm] = {
                    str(layer): _pooled_r2(*agg[layer]) for layer in ctx.layers
                }
                sums[(m, corpus, arm)] = agg
    mean_ctx = {
        layer: float(
            np.nanmean(
                [pooled[m][corpus][ARM_CTX][str(layer)] for m in ctx.ckpts for corpus in C.CORPORA]
            )
        )
        for layer in ctx.layers
    }
    ctx.layer_star = max(ctx.layers, key=lambda layer: (mean_ctx[layer], -layer))
    mean_pre = {
        layer: float(
            np.nanmean([pooled[m][C.CORPUS_MULTI][ARM_PRE][str(layer)] for m in ctx.ckpts])
        )
        for layer in ctx.layers
    }
    ctx.layer_star_p = max(ctx.layers, key=lambda layer: (mean_pre[layer], -layer))
    ctx.band = band_layers(ctx.layers, ctx.layer_star)
    ctx.band_p = band_layers(ctx.layers, ctx.layer_star_p)
    record = {
        "layers": ctx.layers,
        "pooled_diag_r2": pooled,
        "selection": {
            "layer_star": ctx.layer_star,
            "layer_star_p": ctx.layer_star_p,
            "band": ctx.band,
            "band_p": ctx.band_p,
            "mean_diag_ctx_by_layer": {str(k): v for k, v in mean_ctx.items()},
            "mean_diag_pre_by_layer": {str(k): v for k, v in mean_pre.items()},
            "rule": "argmax mean diagonal pooled OOF R^2 over 4 ckpts x 2 corpora (ctx arm); "
            "prefix arm over multi-corpus prefix diagonals",
        },
    }
    return record


# ── stage-2 units ────────────────────────────────────────────────────────────


def run_grid_unit(
    ctx: FitsContext, device: str, *, m: str, s: str, corpus: str, fold: int, arm: str
) -> dict:
    """Off-diagonal grid cell (m activations, s answers) at the band layers.
    Baselines (identity + kNN) per layer — the standing pair on every cell."""
    band = ctx.band if arm == ARM_CTX else ctx.band_p
    idx = ctx.corpora[corpus]
    tr, ev = ctx.fold_masks(corpus, fold)
    unit = f"grid_{m}{s}_{corpus}_{arm}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True, "n_tr": int(tr.sum()), "n_ev": int(ev.sum())})
        return {}
    Xtr = np.stack([ctx.xy(m, s, corpus, layer, arm)[0][tr] for layer in band])
    Ytr = np.stack([ctx.xy(m, s, corpus, layer, arm)[1][tr] for layer in band])
    Xev = np.stack([ctx.xy(m, s, corpus, layer, arm)[0][ev] for layer in band])
    preds, info = _batched_ridge(Xtr, Ytr, Xev, device=device, return_info=True)
    res_all = np.zeros((len(band), int(ev.sum())))
    tot_all = np.zeros_like(res_all)
    cos_all = np.zeros_like(res_all)
    per_layer: dict[str, Any] = {}
    for li, layer in enumerate(band):
        y_ev = ctx.xy(m, s, corpus, layer, arm)[1][ev]
        res, tot, cos = _per_ctx_ss(preds[li], y_ev, Ytr[li].mean(0))
        res_all[li], tot_all[li], cos_all[li] = res, tot, cos
        per_layer[str(layer)] = {
            "ss_res": float(res.sum()),
            "ss_tot": float(tot.sum()),
            "r2": _pooled_r2(float(res.sum()), float(tot.sum())),
            "lambda_star": float(info["best_lambda"][li]),
            "dof": float(info["dof"][li]),
            **_cell_baselines(Xtr[li], Ytr[li], Xev[li], y_ev, preds[li]),
        }
    _, percell = ctx.unit_paths()
    _savez_atomic(
        percell / f"grid_{m}{s}_{corpus}_{arm}_f{fold}.npz",
        layers=np.asarray(band),
        row_idx=np.flatnonzero(ev),
        ss_res=res_all,
        ss_tot=tot_all,
        cos=cos_all,
    )
    ctx.write_unit(unit, {"n_tr": int(tr.sum()), "n_ev": int(ev.sum()), "per_layer": per_layer})
    return {}


def run_star_unit(ctx: FitsContext, device: str, *, m: str, corpus: str, fold: int) -> dict:
    """Diagonal reads AT layer*: per-dim SS (exploratory), shuffled-pairing
    null (draws batched as slices of ONE layer-batched call — Gram identical,
    correspondence destroyed), reliability evals (single corpus: the fold's
    rel-subset contexts scored against the seed-43/44 repeat captures), and
    the diagonal kNN/identity baselines at layer* (headline cell)."""
    layer = ctx.layer_star
    assert layer is not None
    idx = ctx.corpora[corpus]
    tr, ev = ctx.fold_masks(corpus, fold)
    unit = f"star_{m}_{corpus}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True})
        return {}
    X, Y = ctx.xy(m, m, corpus, layer, ARM_CTX)
    rng = np.random.default_rng(C.FOLD_SEED + fold * 101 + ctx.ckpts.index(m))
    perms = [rng.permutation(int(tr.sum())) for _ in range(ctx.n_shuffle)]
    Xtr = np.stack([X[tr]] * (1 + ctx.n_shuffle))
    Ytr = np.stack([Y[tr]] + [Y[tr][p] for p in perms])
    Xev = np.stack([X[ev]] * (1 + ctx.n_shuffle))
    preds = _batched_ridge(Xtr, Ytr, Xev, device=device)
    y_ev = Y[ev]
    y_tr_mean = Y[tr].mean(0)
    res, tot, cos = _per_ctx_ss(preds[0], y_ev, y_tr_mean)
    # per-dim SS (exploratory per-dimension R^2 histogram input)
    res_dim = ((y_ev - preds[0]) ** 2).sum(axis=0)
    tot_dim = ((y_ev - y_tr_mean) ** 2).sum(axis=0)
    shuffle_r2 = []
    for d in range(ctx.n_shuffle):
        r = float(((y_ev - preds[1 + d]) ** 2).sum())
        shuffle_r2.append(_pooled_r2(r, float(tot.sum())))
    out: dict[str, Any] = {
        "layer_star": layer,
        "n_tr": int(tr.sum()),
        "n_ev": int(ev.sum()),
        "r2": _pooled_r2(float(res.sum()), float(tot.sum())),
        "shuffle_null_r2": shuffle_r2,
        **_cell_baselines(X[tr], Y[tr], X[ev], y_ev, preds[0]),
    }
    arrays = {
        "row_idx": np.flatnonzero(ev),
        "ss_res": res,
        "ss_tot": tot,
        "cos": cos,
        "ss_res_dim": res_dim,
        "ss_tot_dim": tot_dim,
    }
    # reliability: same preds scored against the seed-43/44 repeat captures
    if corpus == C.CORPUS_SINGLE:
        rel: dict[str, Any] = {}
        rel_arrays: dict[str, np.ndarray] = {}
        for seed in C.RELIABILITY_SEEDS:
            rel_rel = f"reliability/{m}/{C.CORPUS_SINGLE}/seed{seed}/L{layer}.pt"
            try:
                d = ctx.cache.subdir(rel_rel)
            except FileNotFoundError:
                rel[f"seed{seed}"] = "missing"
                continue
            rel_ids = [str(x) for x in d["__row_ids__"]]
            pos = {rid: k for k, rid in enumerate(rel_ids)}
            ev_idx = np.flatnonzero(ev)
            keep = [(k, pos[idx.ids[gi]]) for k, gi in enumerate(ev_idx) if idx.ids[gi] in pos]
            if len(keep) < 2:
                rel[f"seed{seed}"] = "too_few"
                continue
            evk = np.asarray([k for k, _ in keep])
            relk = np.asarray([p for _, p in keep])
            w_rel = d["w"][relk]
            res_r = ((w_rel - preds[0][evk]) ** 2).sum(axis=-1)
            rel_arrays[f"rel_res_seed{seed}"] = res_r
            rel_arrays[f"rel_rows_seed{seed}"] = ev_idx[evk]
            rel[f"seed{seed}"] = {"n": int(len(keep))}
        out["reliability"] = rel
        arrays.update(rel_arrays)
    _, percell = ctx.unit_paths()
    _savez_atomic(percell / f"star_{m}_{corpus}_f{fold}.npz", **arrays)
    ctx.write_unit(unit, out)
    return {}


def run_robust_unit(ctx: FitsContext, device: str, *, m: str, fold: int) -> dict:
    """Native-vs-canonical render delta at layer* (S/D/R diagonals, single
    corpus, 2k subset): both renders fit on the SAME subset rows."""
    layer = ctx.layer_star
    assert layer is not None
    idx = ctx.corpora[C.CORPUS_SINGLE]
    unit = f"robust_{m}_f{fold}"
    try:
        dn = ctx.cache.subdir(f"robust_native/{m}/{C.CORPUS_SINGLE}/L{layer}.pt")
        dctx = ctx.cache.subdir(f"robust_native/{m}/{C.CORPUS_SINGLE}/ctx/L{layer}.pt")
    except FileNotFoundError:
        ctx.write_unit(unit, {"skipped": True, "reason": "no robust_native store"})
        return {}
    rob_ids = [str(x) for x in dn["__row_ids__"]]
    pos_of = {rid: k for k, rid in enumerate(idx.ids)}
    keep = [k for k, rid in enumerate(rob_ids) if rid in pos_of]
    gpos = np.asarray([pos_of[rob_ids[k]] for k in keep])
    fold_r = idx.fold[gpos]
    ev = fold_r == fold
    tr = ~ev
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True, "reason": "fold gate"})
        return {}
    out: dict[str, Any] = {"layer_star": layer, "n_tr": int(tr.sum()), "n_ev": int(ev.sum())}
    # native arm (robust store)
    Xn, Yn = dctx["u_mean"][keep], dn["w"][keep]
    p_n = _batched_ridge(Xn[tr][None], Yn[tr][None], Xn[ev][None], device=device)[0]
    res_n, tot_n, _ = _per_ctx_ss(p_n, Yn[ev], Yn[tr].mean(0))
    out["native_r2"] = _pooled_r2(float(res_n.sum()), float(tot_n.sum()))
    # plain arm restricted to the SAME subset rows
    Xp, Yp = (a[gpos] for a in ctx.xy(m, m, C.CORPUS_SINGLE, layer, ARM_CTX))
    p_p = _batched_ridge(Xp[tr][None], Yp[tr][None], Xp[ev][None], device=device)[0]
    res_p, tot_p, _ = _per_ctx_ss(p_p, Yp[ev], Yp[tr].mean(0))
    out["plain_r2"] = _pooled_r2(float(res_p.sum()), float(tot_p.sum()))
    ctx.write_unit(unit, out)
    return {}


def run_mlp_unit(ctx: FitsContext, device: str, *, arm: str, m: str, s: str, fold: int) -> dict:
    """MLP grid cell over the band (plan step 4): ctx arm on single corpus,
    prefix arm on multi (2 arms x 16 cells x 6 folds x 5 layers = 960 fits).
    pca_k clamped to n_tr//2 (the pilot precedent) — production 6.8k -> 64."""
    from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict

    corpus = C.CORPUS_SINGLE if arm == ARM_CTX else C.CORPUS_MULTI
    band = ctx.band if arm == ARM_CTX else ctx.band_p
    idx = ctx.corpora[corpus]
    tr, ev = ctx.fold_masks(corpus, fold)
    unit = f"mlp_{arm}_{m}{s}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True})
        return {}
    pca_k = max(1, min(MLP_PCA_K, int(tr.sum()) // 2))
    per_layer: dict[str, Any] = {}
    res_all = np.zeros((len(band), int(ev.sum())))
    tot_all = np.zeros_like(res_all)
    for li, layer in enumerate(band):
        X, Y = ctx.xy(m, s, corpus, layer, arm)
        pred = mlp_fit_predict(X[tr], Y[tr], X[ev], pca_k=pca_k, seed=42, device=device)
        res, tot, _cos = _per_ctx_ss(pred, Y[ev], Y[tr].mean(0))
        res_all[li], tot_all[li] = res, tot
        per_layer[str(layer)] = {
            "ss_res": float(res.sum()),
            "ss_tot": float(tot.sum()),
            "r2": _pooled_r2(float(res.sum()), float(tot.sum())),
        }
    _, percell = ctx.unit_paths()
    _savez_atomic(
        percell / f"mlp_{arm}_{m}{s}_f{fold}.npz",
        layers=np.asarray(band),
        row_idx=np.flatnonzero(ev),
        ss_res=res_all,
        ss_tot=tot_all,
    )
    ctx.write_unit(
        unit, {"pca_k": pca_k, "n_tr": int(tr.sum()), "n_ev": int(ev.sum()), "per_layer": per_layer}
    )
    return {}


def _orth_map(A: np.ndarray, B: np.ndarray, device: str):
    """Closed-form orthogonal Procrustes R minimizing ||A_c R - B_c||_F over
    centered inputs (the #825 `_orth` recipe). Returns (R, mean_A, mean_B)."""
    import torch

    dev = torch.device(device)
    At = torch.as_tensor(A, dtype=torch.float64, device=dev)
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    mu_a, mu_b = At.mean(0), Bt.mean(0)
    M = (At - mu_a).T @ (Bt - mu_b)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    Rm = U @ Vh
    return Rm.cpu().numpy(), mu_a.cpu().numpy(), mu_b.cpu().numpy()


def run_xfer_unit(ctx: FitsContext, device: str, *, i: str, j: str, fold: int) -> dict:
    """Transfer T(i->j) at layer*, single corpus, one fold (plan step 5):
    direct / general-linear / orthogonal-Procrustes / fixed-answer-text, all
    evaluated on fold-HELD-OUT contexts, alignments fitted on TRAIN folds
    only, A_ans on SAME-answer-text pairs w_i(x,a_i) <-> w_j(x,a_i); matched
    nulls (shuffled-correspondence alignment refit N1 + spectrum-matched
    random center operator N2)."""
    layer = ctx.layer_star
    assert layer is not None
    corpus = C.CORPUS_SINGLE
    idx = ctx.corpora[corpus]
    tr, ev = ctx.fold_masks(corpus, fold)
    unit = f"xfer_{i}{j}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True})
        return {}
    u_i, w_ii = ctx.xy(i, i, corpus, layer, ARM_CTX)
    u_j, w_jj = ctx.xy(j, j, corpus, layer, ARM_CTX)
    w_ji = ctx.cache.answer(j, i, corpus, layer, idx.ids)  # cell (j, src=i): j's act of a_i
    n_tr, n_ev = int(tr.sum()), int(ev.sum())
    rng = np.random.default_rng(
        C.FOLD_SEED + 7919 * fold + 13 * ctx.ckpts.index(i) + ctx.ckpts.index(j)
    )

    # A_ctx (general-linear): u_j -> u_i, train folds only; + N1 shuffled refits.
    perms = [rng.permutation(n_tr) for _ in range(ctx.n_null)]
    actx_preds = _batched_ridge(
        np.stack([u_j[tr]] * (1 + ctx.n_null)),
        np.stack([u_i[tr]] + [u_i[tr][p] for p in perms]),
        np.stack([u_j[ev]] * (1 + ctx.n_null)),
        device=device,
    )
    gl_ev, gl_null_ev = actx_preds[0], actx_preds[1:]
    # A_ctx (orthogonal Procrustes, train-centered)
    R_ctx, mu_j, mu_i = _orth_map(u_j[tr], u_i[tr], device)
    orth_ev = (u_j[ev] - mu_j) @ R_ctx + mu_i

    # f_ii applied to EVERY eval input in ONE call (same fit, stacked evals):
    # [direct(u_j) | gl | orth | N1 nulls...]; return_weights for N2 + info.
    stack = np.concatenate([u_j[ev], gl_ev, orth_ev] + [g for g in gl_null_ev], axis=0)
    f_preds, W_ii, f_info = _batched_ridge(
        u_i[tr][None],
        w_ii[tr][None],
        stack[None],
        device=device,
        return_weights=True,
        return_info=True,
    )
    f_preds = f_preds[0]
    f_direct = f_preds[:n_ev]
    f_gl = f_preds[n_ev : 2 * n_ev]
    f_orth = f_preds[2 * n_ev : 3 * n_ev]
    f_nulls = [f_preds[(3 + d) * n_ev : (4 + d) * n_ev] for d in range(ctx.n_null)]

    # N2: spectrum-matched random center operator (retain W_ii's top-r SVs).
    import torch

    dev = torch.device(device)
    Wt = torch.as_tensor(W_ii[0], dtype=torch.float64, device=dev)
    S = torch.linalg.svdvals(Wt).clamp(min=0.0)
    csum = torch.cumsum(S**2, 0) / (float((S**2).sum()) + 1e-30)
    r = max(
        1,
        min(
            int(
                torch.searchsorted(
                    csum, torch.tensor(SPECMATCH_ENERGY, dtype=csum.dtype, device=csum.device)
                ).item()
            )
            + 1,
            S.shape[0],
        ),
    )
    xmu, xsd = u_i[tr].mean(0), u_i[tr].std(0) + 1e-9
    ymu = w_ii[tr].mean(0)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    n2_preds = []
    for _ in range(ctx.n_null):
        A1 = torch.randn(Wt.shape[0], r, dtype=torch.float64, generator=gen).to(dev)
        Q1, R1 = torch.linalg.qr(A1)
        Q1 = Q1 * torch.sign(torch.diagonal(R1))
        A2 = torch.randn(Wt.shape[1], r, dtype=torch.float64, generator=gen).to(dev)
        Q2, R2 = torch.linalg.qr(A2)
        Q2 = Q2 * torch.sign(torch.diagonal(R2))
        W_null = (Q1 * S[:r]) @ Q2.T
        x_std = torch.as_tensor((gl_ev - xmu) / xsd, dtype=torch.float64, device=dev)
        n2_preds.append((x_std @ W_null).cpu().numpy() + ymu)
        del A1, A2, Q1, Q2, W_null

    # A_ans (general-linear): w_i(x,a_i) -> w_j(x,a_i) on train folds; applied
    # to [gl | orth | N2...] in one stacked call; + N1 A_ans refits per draw.
    a_stack = np.concatenate([f_gl, f_orth] + n2_preds, axis=0)
    a_preds = _batched_ridge(w_ii[tr][None], w_ji[tr][None], a_stack[None], device=device)[0]
    gl_final = a_preds[:n_ev]
    orth_a_gl = a_preds[n_ev : 2 * n_ev]
    n2_final = [a_preds[(2 + d) * n_ev : (3 + d) * n_ev] for d in range(ctx.n_null)]
    # orthogonal answer-side map for the fully-orthogonal mode
    R_ans, mu_wii, mu_wji = _orth_map(w_ii[tr], w_ji[tr], device)
    orth_final = (f_orth - mu_wii) @ R_ans + mu_wji
    del orth_a_gl
    # N1: null A_ctx draws -> real f -> null A_ans refits (batched slices).
    aperm = [rng.permutation(n_tr) for _ in range(ctx.n_null)]
    n1_final = _batched_ridge(
        np.stack([w_ii[tr]] * ctx.n_null),
        np.stack([w_ji[tr][p] for p in aperm]),
        np.stack(f_nulls),
        device=device,
    )

    y_tgt = w_jj[ev]
    y_tgt_mean = w_jj[tr].mean(0)
    y_fix = w_ji[ev]
    y_fix_mean = w_ji[tr].mean(0)
    res_direct, tot, _ = _per_ctx_ss(f_direct, y_tgt, y_tgt_mean)
    res_gl, _, _ = _per_ctx_ss(gl_final, y_tgt, y_tgt_mean)
    res_orth, _, _ = _per_ctx_ss(orth_final, y_tgt, y_tgt_mean)
    res_fix, tot_fix, _ = _per_ctx_ss(f_direct, y_fix, y_fix_mean)
    n1_r2 = [
        _pooled_r2(float(((y_tgt - n1_final[d]) ** 2).sum()), float(tot.sum()))
        for d in range(ctx.n_null)
    ]
    n2_r2 = [
        _pooled_r2(float(((y_tgt - n2_final[d]) ** 2).sum()), float(tot.sum()))
        for d in range(ctx.n_null)
    ]
    out = {
        "layer_star": layer,
        "n_tr": n_tr,
        "n_ev": n_ev,
        "lambda_star_center": float(f_info["best_lambda"][0]),
        "dof_center": float(f_info["dof"][0]),
        "specmatch_rank": int(r),
        "r2": {
            "direct": _pooled_r2(float(res_direct.sum()), float(tot.sum())),
            "gl": _pooled_r2(float(res_gl.sum()), float(tot.sum())),
            "orth": _pooled_r2(float(res_orth.sum()), float(tot.sum())),
            "fixedtext": _pooled_r2(float(res_fix.sum()), float(tot_fix.sum())),
        },
        "null_r2": {"shuffled_correspondence": n1_r2, "spectrum_matched": n2_r2},
        "baselines": _cell_baselines(u_j[tr], w_jj[tr], u_j[ev], y_tgt, gl_final),
    }
    _, percell = ctx.unit_paths()
    # Grain note (review r1 Minor): transfer percell SS persists at layer*
    # ONLY — plan §4 step 5 defines transfer at layer*, resolving the step-3
    # band-grain ambiguity toward step 5 (the analyzer must not expect
    # band-grain transfer SS).
    _savez_atomic(
        percell / f"xfer_{i}{j}_f{fold}.npz",
        row_idx=np.flatnonzero(ev),
        ss_res_direct=res_direct,
        ss_res_gl=res_gl,
        ss_res_orth=res_orth,
        ss_res_fixedtext=res_fix,
        ss_tot=tot,
        ss_tot_fixedtext=tot_fix,
    )
    ctx.write_unit(unit, out)
    return {}


def run_operator_unit(ctx: FitsContext, device: str, *, i: str, j: str) -> dict:
    """H4 operator battery for one adjacent transition (plan step 7):
    full-data ridge primal weights W_m at layer* (descriptive, #1332
    convention), delta-W spectrum + effective rank vs matched nulls,
    principal angles, direction-aware Procrustes-aligned operator cosine vs
    the rotation null (issue825 `_procrustes_cosine_null` — draws' QR on the
    fit device) vs spectrum-only cosine (rotation-invariant, descriptive)."""
    import torch

    from issue825_crossmodel_map_transfer import principal_angles
    from issue825_map_alignment import _procrustes_cosine_null

    layer = ctx.layer_star
    assert layer is not None
    corpus = C.CORPUS_SINGLE
    idx = ctx.corpora[corpus]
    u_i, w_ii = ctx.xy(i, i, corpus, layer, ARM_CTX)
    u_j, w_jj = ctx.xy(j, j, corpus, layer, ARM_CTX)
    _, W_i = _batched_ridge(
        u_i[None], w_ii[None], u_i[:2][None], device=device, return_weights=True
    )
    _, W_j = _batched_ridge(
        u_j[None], w_jj[None], u_j[:2][None], device=device, return_weights=True
    )
    Wi = torch.as_tensor(W_i[0], dtype=torch.float64)
    Wj = torch.as_tensor(W_j[0], dtype=torch.float64)

    def _er(mat: torch.Tensor) -> tuple[float, list[float]]:
        s = torch.linalg.svdvals(mat).clamp(min=0.0)
        er = float((s.sum() ** 2) / ((s**2).sum() + 1e-30))
        return er, [float(x) for x in s[:256]]

    er_dw, spec_dw = _er(Wj - Wi)
    # matched nulls for the delta read: shuffled-correspondence refit of W_j
    # (same capacity, destroyed pairing) + spectrum-matched random W_j.
    rng = np.random.default_rng(C.FOLD_SEED + 31 * ctx.ckpts.index(i))
    perm = rng.permutation(idx.n)
    _, W_j_shuf = _batched_ridge(
        u_j[None], w_jj[perm][None], u_j[:2][None], device=device, return_weights=True
    )
    er_shuf, _ = _er(torch.as_tensor(W_j_shuf[0], dtype=torch.float64) - Wi)
    s_j = torch.linalg.svdvals(Wj)
    gen = torch.Generator().manual_seed(C.FOLD_SEED)
    q1 = torch.linalg.qr(torch.randn(Wj.shape[0], Wj.shape[0], dtype=torch.float64, generator=gen))[
        0
    ]
    q2 = torch.linalg.qr(torch.randn(Wj.shape[1], Wj.shape[1], dtype=torch.float64, generator=gen))[
        0
    ]
    er_spec, _ = _er((q1 * s_j) @ q2.T - Wi)
    angles = {
        str(k): [float(x) for x in principal_angles(Wi, Wj, k)]
        for k in (16, 64)
        if k <= min(Wi.shape)
    }
    dev = torch.device(device)
    proc = _procrustes_cosine_null(
        torch.as_tensor(u_i, dtype=torch.float64, device=dev),
        torch.as_tensor(u_j, dtype=torch.float64, device=dev),
        torch.as_tensor(w_ii, dtype=torch.float64, device=dev),
        torch.as_tensor(w_jj, dtype=torch.float64, device=dev),
        n_draws=ctx.n_rot,
        seed=C.FOLD_SEED,
    )
    s_i = torch.linalg.svdvals(Wi)
    spec_cos = float((s_i @ s_j) / (torch.linalg.norm(s_i) * torch.linalg.norm(s_j) + 1e-12))
    er_i, spec_i = _er(Wi)
    er_j, spec_j = _er(Wj)
    ctx.write_unit(
        f"operator_{i}{j}",
        {
            "layer_star": layer,
            "n": idx.n,
            "convention": (
                "direction-aware = activation-fitted Procrustes-aligned operator cosine vs "
                f"{ctx.n_rot}-draw random-rotation null (issue1345 conventions); spectrum "
                "cosine = sorted-singular-value cosine, ROTATION-INVARIANT DESCRIPTIVE ONLY "
                "(can never support 'same operator up to rotation'); weights are "
                "standardized-input-space primal (descriptive, #1332)"
            ),
            "er_W_i": er_i,
            "er_W_j": er_j,
            "er_delta": er_dw,
            "er_delta_null_shuffled_refit": er_shuf,
            "er_delta_null_spectrum_matched": er_spec,
            "delta_spectrum_top": spec_dw[:64],
            "spectrum_W_i_top": spec_i[:64],
            "spectrum_W_j_top": spec_j[:64],
            "principal_angle_cos": angles,
            "procrustes_aligned": proc,
            "spectrum_cosine": spec_cos,
        },
    )
    return {}


# ── shard assembly (percell npz -> per-context arrays; no refits) ────────────


def _load_shards(ctx: FitsContext, pattern: str, keys: tuple[str, ...]):
    """Assemble per-context arrays from fold shards. Returns
    {key: (K?, n) array}, filled per fold via row_idx; unfitted rows NaN."""
    _, percell = ctx.unit_paths()
    shards = sorted(percell.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"no percell shards match {pattern} under {percell}")
    out: dict[str, np.ndarray] = {}
    fitted: np.ndarray | None = None
    for sp in shards:
        d = np.load(sp)
        rows = d["row_idx"]
        for key in keys:
            arr = d[key]
            if key not in out:
                out[key] = np.full((*arr.shape[:-1], self_n(ctx, sp.name)), np.nan)
            out[key][..., rows] = arr
        if fitted is None:
            fitted = np.zeros(self_n(ctx, sp.name), bool)
        fitted[rows] = True
    out["__fitted__"] = fitted if fitted is not None else np.zeros(0, bool)
    return out


def self_n(ctx: FitsContext, shard_name: str) -> int:
    corpus = C.CORPUS_MULTI if f"_{C.CORPUS_MULTI}_" in shard_name else C.CORPUS_SINGLE
    if shard_name.startswith(("xfer_", "mlp_ctx")):
        corpus = C.CORPUS_SINGLE
    if shard_name.startswith("mlp_pre"):
        corpus = C.CORPUS_MULTI
    return ctx.corpora[corpus].n


def _group_index(ctx: FitsContext, corpus: str) -> tuple[np.ndarray, list[str]]:
    """Group-label -> int index per row (the cluster-grouped bootstrap axis)."""
    idx = ctx.corpora[corpus]
    names = sorted(set(idx.groups))
    of = {g: k for k, g in enumerate(names)}
    return np.asarray([of[g] for g in idx.groups], dtype=np.int64), names


def _group_sums(vals: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """(..., n) per-context values -> (..., G) per-group sums (NaN -> 0:
    unfitted rows contribute nothing — matches the pooled-OOF convention)."""
    v = np.nan_to_num(vals, nan=0.0)
    flat = v.reshape(-1, v.shape[-1])
    sums = np.zeros((flat.shape[0], n_groups))
    for g in range(n_groups):
        sums[:, g] = flat[:, gid == g].sum(axis=1)
    return sums.reshape(*v.shape[:-1], n_groups)


def _boot_counts(rng: np.random.Generator, n_groups: int, n_draws: int) -> np.ndarray:
    """Cluster-grouped bootstrap draw weights: sample G groups with
    replacement -> multinomial counts (n_draws, G). Batched masked-sum GEMM
    companion (vectorize-many-cell-fits fix item 3)."""
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_draws).astype(
        np.float64
    )


def _boot_r2(counts: np.ndarray, res_g: np.ndarray, tot_g: np.ndarray) -> np.ndarray:
    """R^2 per draw: 1 - (counts @ res_g) / (counts @ tot_g). res_g/tot_g (G,)
    or (L, G) -> returns (n_draws,) or (n_draws, L)."""
    num = counts @ (res_g.T if res_g.ndim == 2 else res_g)
    den = counts @ (tot_g.T if tot_g.ndim == 2 else tot_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - num / den


def _ci(draws: np.ndarray) -> list[float]:
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


# ── finalize: aggregation + bootstrap + gates + outputs ─────────────────────


def _diag_arrays(ctx: FitsContext, m: str, corpus: str, arm: str):
    return _load_shards(ctx, f"diag_{m}_{corpus}_{arm}_f*.npz", ("ss_res", "ss_tot", "cos"))


def _star_layer_index(ctx: FitsContext) -> int:
    assert ctx.layer_star is not None
    return ctx.layers.index(ctx.layer_star)


def finalize(ctx: FitsContext, selection: dict, parity: dict, wall_h: float) -> dict:
    rng = np.random.default_rng(C.FOLD_SEED + 1902)
    eval_dir = ctx.eval_dir
    li_star = _star_layer_index(ctx)
    n_boot = ctx.n_boot

    # ---- assemble diagonal full-grain arrays (ctx arm) ----
    diag = {
        (m, corpus): _diag_arrays(ctx, m, corpus, ARM_CTX)
        for m in ctx.ckpts
        for corpus in C.CORPORA
    }
    gidx = {corpus: _group_index(ctx, corpus) for corpus in C.CORPORA}
    counts = {corpus: _boot_counts(rng, len(gidx[corpus][1]), n_boot) for corpus in C.CORPORA}

    # per-(m, corpus) per-layer group sums + full-data pooled R2
    diag_r2_draws: dict[tuple, np.ndarray] = {}
    for (m, corpus), arrs in diag.items():
        gid, names = gidx[corpus]
        res_g = _group_sums(arrs["ss_res"], gid, len(names))  # (L, G)
        tot_g = _group_sums(arrs["ss_tot"], gid, len(names))
        diag_r2_draws[(m, corpus)] = _boot_r2(counts[corpus], res_g, tot_g)  # (n_boot, L)

    # selection-inherited: per-draw layer re-selection by the SAME rule
    sel_stat = np.nanmean(np.stack(list(diag_r2_draws.values())), axis=0)  # (n_boot, L)
    sel_layer_idx = np.argmax(np.nan_to_num(sel_stat, nan=-np.inf), axis=1)  # (n_boot,)

    grid_cells: dict[str, Any] = {
        "metadata": R._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "layer_star_p": ctx.layer_star_p,
        "band": ctx.band,
        "band_p": ctx.band_p,
        "n_boot": n_boot,
        "cells": {},
        "mlp": {},
    }
    for (m, corpus), arrs in diag.items():
        draws_frozen = diag_r2_draws[(m, corpus)][:, li_star]
        draws_inherited = diag_r2_draws[(m, corpus)][np.arange(n_boot), sel_layer_idx]
        star_units = [
            ctx.read_unit(f"star_{m}_{corpus}_f{k}")
            for k in range(ctx.corpora[corpus].n_folds)
            if (ctx.eval_dir / "fits" / "units" / f"star_{m}_{corpus}_f{k}.json").exists()
        ]
        star_ok = [u for u in star_units if not u.get("skipped")]
        res_star = float(np.nansum(arrs["ss_res"][li_star]))
        tot_star = float(np.nansum(arrs["ss_tot"][li_star]))
        grid_cells["cells"][f"diag_{m}_{corpus}_{ARM_CTX}"] = {
            "kind": "diagonal",
            "fitter": "ridge",
            "r2_by_layer": {
                str(layer): _pooled_r2(
                    float(np.nansum(arrs["ss_res"][li])), float(np.nansum(arrs["ss_tot"][li]))
                )
                for li, layer in enumerate(ctx.layers)
            },
            "r2_at_star": _pooled_r2(res_star, tot_star),
            "ci_frozen_at_star": _ci(draws_frozen),
            "ci_selection_inherited": _ci(draws_inherited),
            "shuffle_null_r2": [x for u in star_ok for x in u.get("shuffle_null_r2", [])],
            "baselines_at_star": {
                "identity_r2": [u.get("identity_r2") for u in star_ok],
                "knn": [u.get("knn") for u in star_ok],
            },
        }

    # prefix-arm diagonals (multi corpus) at their own grain
    for m in ctx.ckpts:
        arrs = _diag_arrays(ctx, m, C.CORPUS_MULTI, ARM_PRE)
        grid_cells["cells"][f"diag_{m}_{C.CORPUS_MULTI}_{ARM_PRE}"] = {
            "kind": "diagonal",
            "fitter": "ridge",
            "arm": ARM_PRE,
            "r2_by_layer": {
                str(layer): _pooled_r2(
                    float(np.nansum(arrs["ss_res"][li])), float(np.nansum(arrs["ss_tot"][li]))
                )
                for li, layer in enumerate(ctx.layers)
            },
        }

    # off-diagonal grid cells (unit jsons carry per-layer r2 + baselines)
    for corpus in C.CORPORA:
        for m in ctx.ckpts:
            for s in ctx.ckpts:
                if m == s:
                    continue
                folds = []
                for k in range(ctx.corpora[corpus].n_folds):
                    p = (
                        ctx.eval_dir
                        / "fits"
                        / "units"
                        / f"grid_{m}{s}_{corpus}_{ARM_CTX}_f{k}.json"
                    )
                    if p.exists():
                        folds.append(R._read_json(p))
                ok = [f for f in folds if not f.get("skipped")]
                if not ok:
                    continue
                agg: dict[str, Any] = {}
                for layer in ctx.band:
                    rs = sum(f["per_layer"][str(layer)]["ss_res"] for f in ok)
                    tt = sum(f["per_layer"][str(layer)]["ss_tot"] for f in ok)
                    agg[str(layer)] = {
                        "r2": _pooled_r2(rs, tt),
                        "identity_r2": [f["per_layer"][str(layer)]["identity_r2"] for f in ok],
                        "knn": [f["per_layer"][str(layer)]["knn"] for f in ok],
                    }
                grid_cells["cells"][f"grid_{m}{s}_{corpus}_{ARM_CTX}"] = {
                    "kind": "grid",
                    "fitter": "ridge",
                    "per_layer": agg,
                }
    # prefix-arm off-diagonal grid (multi corpus)
    for m in ctx.ckpts:
        for s in ctx.ckpts:
            if m == s:
                continue
            folds = [
                R._read_json(p)
                for k in range(ctx.corpora[C.CORPUS_MULTI].n_folds)
                if (
                    p := ctx.eval_dir
                    / "fits"
                    / "units"
                    / f"grid_{m}{s}_{C.CORPUS_MULTI}_{ARM_PRE}_f{k}.json"
                ).exists()
            ]
            ok = [f for f in folds if not f.get("skipped")]
            if not ok:
                continue
            grid_cells["cells"][f"grid_{m}{s}_{C.CORPUS_MULTI}_{ARM_PRE}"] = {
                "kind": "grid",
                "fitter": "ridge",
                "arm": ARM_PRE,
                "per_layer": {
                    str(layer): {
                        "r2": _pooled_r2(
                            sum(f["per_layer"][str(layer)]["ss_res"] for f in ok),
                            sum(f["per_layer"][str(layer)]["ss_tot"] for f in ok),
                        )
                    }
                    for layer in ctx.band_p
                },
            }

    # ---- MLP cells + MLP-own layer selection ----
    mlp_diag_by_layer: dict[str, dict[int, list[float]]] = {ARM_CTX: {}, ARM_PRE: {}}
    for arm in (ARM_CTX, ARM_PRE):
        corpus = C.CORPUS_SINGLE if arm == ARM_CTX else C.CORPUS_MULTI
        band = ctx.band if arm == ARM_CTX else ctx.band_p
        for m in ctx.ckpts:
            for s in ctx.ckpts:
                folds = [
                    R._read_json(p)
                    for k in range(ctx.corpora[corpus].n_folds)
                    if (
                        p := ctx.eval_dir / "fits" / "units" / f"mlp_{arm}_{m}{s}_f{k}.json"
                    ).exists()
                ]
                ok = [f for f in folds if not f.get("skipped")]
                if not ok:
                    continue
                per_layer = {
                    str(layer): _pooled_r2(
                        sum(f["per_layer"][str(layer)]["ss_res"] for f in ok),
                        sum(f["per_layer"][str(layer)]["ss_tot"] for f in ok),
                    )
                    for layer in band
                }
                grid_cells["mlp"][f"mlp_{arm}_{m}{s}"] = {"per_layer": per_layer}
                if m == s:
                    for layer in band:
                        mlp_diag_by_layer[arm].setdefault(layer, []).append(per_layer[str(layer)])
    mlp_star = {
        arm: (
            max(vals, key=lambda layer: float(np.nanmean(vals[layer])))
            if (vals := mlp_diag_by_layer[arm])
            else None
        )
        for arm in (ARM_CTX, ARM_PRE)
    }
    grid_cells["mlp_layer_star"] = {
        arm: (int(v) if v is not None else None) for arm, v in mlp_star.items()
    }
    grid_cells["mlp_selection_rule"] = (
        "MLP headline layer = argmax mean diagonal MLP OOF R^2 within the band, per arm "
        "(selected by MLP predictivity, never ridge's — plan §4 step 4)"
    )

    # ---- H3 grid variance decomposition at layer* (ridge + mlp, per arm) ----
    h3: dict[str, Any] = {}
    for fitter in ("ridge", "mlp"):
        for arm in (ARM_CTX, ARM_PRE):
            corpora = C.CORPORA if (fitter == "ridge" and arm == ARM_CTX) else None
            if fitter == "ridge" and arm == ARM_PRE:
                corpora = (C.CORPUS_MULTI,)
            if fitter == "mlp":
                corpora = (C.CORPUS_SINGLE,) if arm == ARM_CTX else (C.CORPUS_MULTI,)
            for corpus in corpora or ():
                Q = np.full((len(ctx.ckpts), len(ctx.ckpts)), np.nan)
                res_gs = {}
                tot_gs = {}
                gid, names = gidx[corpus]
                for mi, m in enumerate(ctx.ckpts):
                    for si, s in enumerate(ctx.ckpts):
                        if fitter == "ridge":
                            if m == s:
                                arrs = (
                                    diag[(m, corpus)]
                                    if arm == ARM_CTX
                                    else _diag_arrays(ctx, m, corpus, ARM_PRE)
                                )
                                # full-grain diag shard: index the ARM'S star
                                di = (
                                    li_star
                                    if arm == ARM_CTX
                                    else ctx.layers.index(ctx.layer_star_p)
                                )
                                res_v, tot_v = arrs["ss_res"][di], arrs["ss_tot"][di]
                            else:
                                sh = _load_shards(
                                    ctx,
                                    f"grid_{m}{s}_{corpus}_{arm}_f*.npz",
                                    ("ss_res", "ss_tot"),
                                )
                                b = ctx.band if arm == ARM_CTX else ctx.band_p
                                bl = b.index(ctx.layer_star if arm == ARM_CTX else ctx.layer_star_p)
                                res_v, tot_v = sh["ss_res"][bl], sh["ss_tot"][bl]
                        else:
                            star_m = mlp_star[arm]
                            if star_m is None:
                                continue
                            sh = _load_shards(ctx, f"mlp_{arm}_{m}{s}_f*.npz", ("ss_res", "ss_tot"))
                            b = ctx.band if arm == ARM_CTX else ctx.band_p
                            res_v, tot_v = (
                                sh["ss_res"][b.index(star_m)],
                                sh["ss_tot"][b.index(star_m)],
                            )
                        Q[mi, si] = _pooled_r2(float(np.nansum(res_v)), float(np.nansum(tot_v)))
                        res_gs[(mi, si)] = _group_sums(res_v, gid, len(names))
                        tot_gs[(mi, si)] = _group_sums(tot_v, gid, len(names))
                if np.isnan(Q).all():
                    continue
                range_s = float(np.nanmean(Q.max(axis=1) - Q.min(axis=1)))
                range_m = float(np.nanmean(Q.max(axis=0) - Q.min(axis=0)))
                # bootstrap CI on (range_s - range_m): per-draw recompute of all cells
                draws_q = np.full((n_boot, len(ctx.ckpts), len(ctx.ckpts)), np.nan)
                for (mi, si), rg in res_gs.items():
                    draws_q[:, mi, si] = _boot_r2(counts[corpus], rg, tot_gs[(mi, si)])
                d_rs = np.nanmean(np.nanmax(draws_q, 2) - np.nanmin(draws_q, 2), 1)
                d_rm = np.nanmean(np.nanmax(draws_q, 1) - np.nanmin(draws_q, 1), 1)
                h3[f"{fitter}_{arm}_{corpus}"] = {
                    "Q_grid": [[None if np.isnan(x) else float(x) for x in row] for row in Q],
                    "ckpt_order": ctx.ckpts,
                    "range_s": range_s,
                    "range_m": range_m,
                    "range_s_minus_range_m": range_s - range_m,
                    "ci_range_s_minus_range_m": _ci(d_rs - d_rm),
                }
    grid_cells["h3_variance_decomposition"] = h3
    R._write_json_atomic(eval_dir / "fits" / "grid_cells.json", grid_cells)

    # ---- transfer matrix + H1 retention lattice ----
    corpus = C.CORPUS_SINGLE
    gid_s, names_s = gidx[corpus]
    cnt = counts[corpus]
    pairs = [(i, j) for i in ctx.ckpts for j in ctx.ckpts if i != j]
    xfer_json: dict[str, Any] = {
        "metadata": R._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "corpus": corpus,
        "modes": list(XFER_MODES),
        "pairs": {},
    }
    rho_draws: dict[tuple, np.ndarray] = {}
    rho_point: dict[tuple, float] = {}
    for i, j in pairs:
        sh = _load_shards(
            ctx,
            f"xfer_{i}{j}_f*.npz",
            (
                "ss_res_direct",
                "ss_res_gl",
                "ss_res_orth",
                "ss_res_fixedtext",
                "ss_tot",
                "ss_tot_fixedtext",
            ),
        )
        units = [
            ctx.read_unit(f"xfer_{i}{j}_f{k}")
            for k in range(ctx.corpora[corpus].n_folds)
            if (ctx.eval_dir / "fits" / "units" / f"xfer_{i}{j}_f{k}.json").exists()
        ]
        ok = [u for u in units if not u.get("skipped")]
        tot_g = _group_sums(sh["ss_tot"], gid_s, len(names_s))
        rec: dict[str, Any] = {"r2": {}, "retention_gl": None}
        for mode in XFER_MODES:
            res_v = sh[f"ss_res_{mode if mode != 'gl' else 'gl'}"]
            tot_v = sh["ss_tot_fixedtext"] if mode == "fixedtext" else sh["ss_tot"]
            rec["r2"][mode] = _pooled_r2(float(np.nansum(res_v)), float(np.nansum(tot_v)))
        # retention rho(i->j) = R2_gl(i->j) / Q(j,j) at layer* (ctx arm, single)
        diag_j = diag[(j, corpus)]
        q_jj = _pooled_r2(
            float(np.nansum(diag_j["ss_res"][li_star])), float(np.nansum(diag_j["ss_tot"][li_star]))
        )
        rho = rec["r2"]["gl"] / q_jj if q_jj and np.isfinite(q_jj) and q_jj > 0 else float("nan")
        rec["retention_gl"] = rho
        rec["q_jj_at_star"] = q_jj
        rec["nulls"] = {
            "shuffled_correspondence_r2": [
                x for u in ok for x in u["null_r2"]["shuffled_correspondence"]
            ],
            "spectrum_matched_r2": [x for u in ok for x in u["null_r2"]["spectrum_matched"]],
        }
        rec["baselines"] = [u.get("baselines") for u in ok]
        rec["lambda_star_center"] = [u.get("lambda_star_center") for u in ok]
        xfer_json["pairs"][f"{i}->{j}"] = rec
        rho_point[(i, j)] = rho
        # per-draw rho
        res_gl_g = _group_sums(sh["ss_res_gl"], gid_s, len(names_s))
        diag_res_g = _group_sums(diag_j["ss_res"][li_star], gid_s, len(names_s))
        diag_tot_g = _group_sums(diag_j["ss_tot"][li_star], gid_s, len(names_s))
        r2_gl_d = _boot_r2(cnt, res_gl_g, tot_g)
        q_jj_d = _boot_r2(cnt, diag_res_g, diag_tot_g)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_draws[(i, j)] = np.where(q_jj_d > 0, r2_gl_d / q_jj_d, np.nan)

    trans = realized_transitions(ctx.ckpts)
    adjacent = trans + [(b, a) for a, b in trans]
    r_adj_point = float(np.nanmedian([rho_point[p] for p in adjacent]))
    r_adj_draws = np.nanmedian(np.stack([rho_draws[p] for p in adjacent]), axis=0)
    d_conf, d_kill = r_adj_point - 0.8, r_adj_point - 0.5
    ci_conf = _ci(r_adj_draws - 0.8)
    ci_kill = _ci(r_adj_draws - 0.5)
    if d_conf >= 0 and ci_conf[0] > 0:
        verdict = "Confirmed"
    elif ci_kill[1] < 0:
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    xfer_json["h1"] = {
        "adjacent_transitions": [f"{a}->{b}" for a, b in adjacent],
        "r_adj": r_adj_point,
        "delta_conf": d_conf,
        "delta_kill": d_kill,
        "ci_delta_conf": ci_conf,
        "ci_delta_kill": ci_kill,
        "verdict": verdict if not ctx.smoke else f"informational-smoke:{verdict}",
        "lattice": "Confirmed <=> d_conf>=0 & CI(d_conf) excludes 0 positively; "
        "Falsified <=> CI(d_kill) wholly below 0; else Inconclusive (plan §3)",
    }
    R._write_json_atomic(eval_dir / "transfer" / "transfer_matrix.json", xfer_json)

    # ---- H2 per-class / per-cluster delta-Q + permutation nulls ----
    h2: dict[str, Any] = {
        "metadata": R._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "registered_contrasts": {},
        "per_cluster": {},
    }
    null_arrays: dict[str, np.ndarray] = {}
    for corpus in C.CORPORA:
        idx = ctx.corpora[corpus]
        classes = np.asarray(idx.classes)
        clusters = np.asarray(idx.clusters)
        for i, j in realized_transitions(ctx.ckpts):
            res_i, tot_i = (
                diag[(i, corpus)]["ss_res"][li_star],
                diag[(i, corpus)]["ss_tot"][li_star],
            )
            res_j, tot_j = (
                diag[(j, corpus)]["ss_res"][li_star],
                diag[(j, corpus)]["ss_tot"][li_star],
            )
            key = f"{i}->{j}_{corpus}"
            # per-cluster delta (generic clusters only)
            gen_mask = classes == C.CLASS_GENERIC
            cl_ids = sorted(set(clusters[gen_mask]) - {C.UNCLUSTERED})
            if cl_ids:

                def _qc(res, tot, mask):
                    r, t = float(np.nansum(res[mask])), float(np.nansum(tot[mask]))
                    return _pooled_r2(r, t)

                obs = np.asarray(
                    [
                        _qc(res_j, tot_j, gen_mask & (clusters == c))
                        - _qc(res_i, tot_i, gen_mask & (clusters == c))
                        for c in cl_ids
                    ]
                )
                # selection-symmetric cluster-permutation null: permute cluster
                # ASSIGNMENT among generic contexts; per-draw stat matrix kept.
                n_perm = n_boot
                gen_pos = np.flatnonzero(gen_mask)
                null = np.full((n_perm, len(cl_ids)), np.nan)
                prng = np.random.default_rng(C.FOLD_SEED + hash(key) % 10_000)
                lab = clusters[gen_pos]
                for d in range(n_perm):
                    pl = lab[prng.permutation(len(lab))]
                    for ci_, c in enumerate(cl_ids):
                        mask = np.zeros(idx.n, bool)
                        mask[gen_pos[pl == c]] = True
                        null[d, ci_] = _qc(res_j, tot_j, mask) - _qc(res_i, tot_i, mask)
                h2["per_cluster"][key] = {
                    "cluster_ids": [int(c) for c in cl_ids],
                    "delta_qc": [float(x) for x in obs],
                    "most_moved_cluster": int(cl_ids[int(np.nanargmax(np.abs(obs)))]),
                    "obs_max_abs": float(np.nanmax(np.abs(obs))),
                    "null_max_abs_p": float(
                        np.mean(np.nanmax(np.abs(null), axis=1) >= np.nanmax(np.abs(obs)))
                    ),
                    "n_perm": n_perm,
                }
                null_arrays[f"obs_{key}"] = obs
                null_arrays[f"null_{key}"] = null
                null_arrays[f"clusters_{key}"] = np.asarray(cl_ids)

    # registered contrasts (plan §3 H2 a/b/c)
    def _class_delta(i: str, j: str, corpus: str, mask: np.ndarray, boot_mode: str, seed: int):
        res_i = diag[(i, corpus)]["ss_res"][li_star]
        tot_i = diag[(i, corpus)]["ss_tot"][li_star]
        res_j = diag[(j, corpus)]["ss_res"][li_star]
        tot_j = diag[(j, corpus)]["ss_tot"][li_star]
        q_i = _pooled_r2(float(np.nansum(res_i[mask])), float(np.nansum(tot_i[mask])))
        q_j = _pooled_r2(float(np.nansum(res_j[mask])), float(np.nansum(tot_j[mask])))
        prng = np.random.default_rng(seed)
        pos = np.flatnonzero(mask)
        draws = np.full(n_boot, np.nan)
        if boot_mode == "context":
            # within-stratum context-level bootstrap (single whole-stratum
            # group — a cluster-grouped bootstrap is degenerate; plan §4 step 3)
            for d in range(n_boot):
                take = prng.integers(0, len(pos), len(pos))
                p = pos[take]
                draws[d] = _pooled_r2(
                    float(np.nansum(res_j[p])), float(np.nansum(tot_j[p]))
                ) - _pooled_r2(float(np.nansum(res_i[p])), float(np.nansum(tot_i[p])))
        else:
            gid, names = gidx[corpus]
            sub_g = sorted(set(gid[pos]))
            cmap = {g: k for k, g in enumerate(sub_g)}
            gsub = np.asarray([cmap[g] for g in gid[pos]])
            c2 = _boot_counts(prng, len(sub_g), n_boot)
            rg_i = _group_sums(res_i[pos], gsub, len(sub_g))
            tg_i = _group_sums(tot_i[pos], gsub, len(sub_g))
            rg_j = _group_sums(res_j[pos], gsub, len(sub_g))
            tg_j = _group_sums(tot_j[pos], gsub, len(sub_g))
            draws = _boot_r2(c2, rg_j, tg_j) - _boot_r2(c2, rg_i, tg_i)
        return {
            "q_i": q_i,
            "q_j": q_j,
            "delta": q_j - q_i,
            "ci_delta": _ci(draws),
            "n": int(mask.sum()),
            "bootstrap": boot_mode,
        }

    cls_s = np.asarray(ctx.corpora[C.CORPUS_SINGLE].classes)
    if (cls_s == C.CLASS_GSM8K).sum() >= 2:
        h2["registered_contrasts"]["a_D->R_gsm8k"] = (
            _class_delta("D", "R", C.CORPUS_SINGLE, cls_s == C.CLASS_GSM8K, "context", 11)
            if "D" in ctx.ckpts and "R" in ctx.ckpts
            else None
        )
    if (cls_s == C.CLASS_MBPP).sum() >= 2 and "D" in ctx.ckpts and "R" in ctx.ckpts:
        h2["registered_contrasts"]["a_D->R_mbpp"] = _class_delta(
            "D", "R", C.CORPUS_SINGLE, cls_s == C.CLASS_MBPP, "context", 12
        )
    if "B" in ctx.ckpts and "S" in ctx.ckpts:
        h2["registered_contrasts"]["b_B->S_generic_single"] = _class_delta(
            "B", "S", C.CORPUS_SINGLE, cls_s == C.CLASS_GENERIC, "cluster", 13
        )
    if "S" in ctx.ckpts and "D" in ctx.ckpts:
        h2["registered_contrasts"]["c_S->D_multi"] = _class_delta(
            "S",
            "D",
            C.CORPUS_MULTI,
            np.ones(ctx.corpora[C.CORPUS_MULTI].n, bool),
            "cluster",
            14,
        )
    h2["registered_contrasts"] = {k: v for k, v in h2["registered_contrasts"].items() if v}
    if ctx.smoke:
        # smoke arm-class coverage: exercise BOTH bootstrap branches (cluster
        # + within-stratum context) on the realized transition — informational.
        si, sj = realized_transitions(ctx.ckpts)[0]
        h2["smoke_informational_contrasts"] = {
            f"{si}->{sj}_generic_cluster": _class_delta(
                si, sj, C.CORPUS_SINGLE, cls_s == C.CLASS_GENERIC, "cluster", 21
            )
        }
        for stratum in (C.CLASS_GSM8K, C.CLASS_MBPP):
            if (cls_s == stratum).sum() >= 2:
                h2["smoke_informational_contrasts"][f"{si}->{sj}_{stratum}_context"] = _class_delta(
                    si, sj, C.CORPUS_SINGLE, cls_s == stratum, "context", 22
                )
    R._write_json_atomic(eval_dir / "clusters" / "delta_qc.json", h2)
    if null_arrays:
        _savez_atomic(eval_dir / "clusters" / "null_matrix.npz", **null_arrays)

    # ---- operator battery + CKA + reliability + robust ----
    op: dict[str, Any] = {
        "metadata": R._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "pairs": {},
        "cka": {},
    }
    for i, j in realized_transitions(ctx.ckpts):
        p = ctx.eval_dir / "fits" / "units" / f"operator_{i}{j}.json"
        if p.exists():
            op["pairs"][f"{i}->{j}"] = R._read_json(p)
    for corpus in C.CORPORA:
        for a in range(len(ctx.ckpts)):
            for b in range(a + 1, len(ctx.ckpts)):
                i, j = ctx.ckpts[a], ctx.ckpts[b]
                p = ctx.eval_dir / "fits" / "units" / f"cka_{i}{j}_{corpus}.json"
                if p.exists():
                    op["cka"][f"{i}-{j}_{corpus}"] = R._read_json(p)
    # H4 conditional basis-stability read (plan §3 H4 clause)
    adj_cka = [
        op["cka"][f"{i}-{j}_{C.CORPUS_SINGLE}"]["cka_u"][li_star]
        for i, j in realized_transitions(ctx.ckpts)
        if f"{i}-{j}_{C.CORPUS_SINGLE}" in op["cka"]
    ]
    op["h4_basis_stability"] = {
        "mean_adjacent_cka_u_at_star": float(np.mean(adj_cka)) if adj_cka else None,
        "bar": 0.7,
        "raw_delta_read_licensed": bool(adj_cka and float(np.mean(adj_cka)) >= 0.7),
        "note": "raw delta-W narration is conditional on CKA(u_i,u_j)>=0.7 at layer* "
        "(plan §3 H4); the ALIGNED battery is always reported",
    }
    # reliability split-half ceiling (context-ALIGNED halves; llm-judging r21)
    rel: dict[str, Any] = {}
    for m in ctx.ckpts:
        r43, r44 = [], []
        for k in range(ctx.corpora[C.CORPUS_SINGLE].n_folds):
            _, percell = ctx.unit_paths()
            p = percell / f"star_{m}_{C.CORPUS_SINGLE}_f{k}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            if "rel_res_seed43" in d and "rel_res_seed44" in d:
                a = {int(r): v for r, v in zip(d["rel_rows_seed43"], d["rel_res_seed43"])}
                b = {int(r): v for r, v in zip(d["rel_rows_seed44"], d["rel_res_seed44"])}
                for row in sorted(set(a) & set(b)):
                    r43.append(a[row])
                    r44.append(b[row])
        if len(r43) >= 3:
            r = float(np.corrcoef(np.asarray(r43), np.asarray(r44))[0, 1])
            sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
            rel[m] = {
                "n_contexts": len(r43),
                "split_half_r": r,
                "spearman_brown": sb,
                "ceiling_sqrt_ryy": float(np.sqrt(sb)) if np.isfinite(sb) and sb > 0 else None,
            }
        else:
            rel[m] = {"n_contexts": len(r43), "note": "too few paired reliability contexts"}
    op["reliability_ceiling"] = rel
    robust: dict[str, Any] = {}
    for m in ctx.ckpts:
        folds = [
            R._read_json(p)
            for k in range(ctx.corpora[C.CORPUS_SINGLE].n_folds)
            if (p := ctx.eval_dir / "fits" / "units" / f"robust_{m}_f{k}.json").exists()
        ]
        ok = [f for f in folds if not f.get("skipped")]
        if ok:
            robust[m] = {
                "native_r2_mean": float(np.nanmean([f["native_r2"] for f in ok])),
                "plain_r2_mean": float(np.nanmean([f["plain_r2"] for f in ok])),
                "n_folds": len(ok),
            }
    robust["B"] = {"note": "base native render IS the plain render (A3) — delta 0 by construction"}
    op["robust_native_vs_plain"] = robust
    R._write_json_atomic(eval_dir / "operator" / "operator_battery.json", op)

    # ---- layer_sweep.json (selection + parity + pilot timings) ----
    sweep_out = {
        "metadata": R._metadata(),
        "smoke": ctx.smoke,
        **selection,
        "parity_gate": parity,
        "pilot_timings": ctx.pilot_timings[:8],
        "wall_h": wall_h,
        # Baseline-pair scope (plan §4 step 8 reading; review r1 Minor): the
        # 17-layer sweep carries identity+bias PER LAYER with kNN retrieval
        # read at the selected band via the grid/star cells; MLP diagonals are
        # a nonlinear-headroom diagnostic and carry no identity/kNN pair.
        "baseline_scope": {
            "sweep": "identity+bias per layer; kNN retrieval at the band (grid/star cells)",
            "mlp": "nonlinear-headroom diagnostic only — no identity/kNN pair",
        },
    }
    R._write_json_atomic(eval_dir / "fits" / "layer_sweep.json", sweep_out)

    # ---- gate B (post-P4 sanity; §7 — routes to rig verification) ----
    per_ckpt_q = {
        m: float(
            np.nanmean(
                [
                    grid_cells["cells"][f"diag_{m}_{corpus}_{ARM_CTX}"]["r2_at_star"]
                    for corpus in C.CORPORA
                ]
            )
        )
        for m in ctx.ckpts
    }
    n_pass = sum(1 for v in per_ckpt_q.values() if np.isfinite(v) and v > GATE_B_FLOOR)
    gate_b = {"per_ckpt_mean_diag_q_at_star": per_ckpt_q, "floor": GATE_B_FLOOR, "n_pass": n_pass}
    if ctx.smoke:
        logger.info("[fits] SMOKE gate B informational: %s", gate_b)
    elif n_pass < min(3, len(ctx.ckpts)):
        R.designed_halt(ctx.out_root, "sanity_gate_b", gate_b)
    return {"grid_cells": grid_cells, "transfer": xfer_json, "gate_b": gate_b, "h2": h2, "op": op}


# ── uploads / git / sentinel ────────────────────────────────────────────────


def _upload_eval_mirror(ctx: FitsContext) -> None:
    """HF eval-results mirror (plan §10: the GCE lane is DELETE-on-exit, so
    the JSONs + npz mirror to HF in the same P4 upload step)."""
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    res = upload_dir_sharded(
        ctx.eval_dir,
        C.HF_DATA_REPO,
        C.EVAL_MIRROR_HF_PATH,
        repo_type="dataset",
        verify=True,
        delete_local=False,
    )
    logger.info(
        "[fits] eval mirror uploaded: %d files (+%d skipped, %d rerouted)",
        len(res.uploaded),
        len(res.skipped_existing),
        len(res.rerouted),
    )


def _commit_eval_results(ctx: FitsContext) -> str:
    """Commit + push the P4 eval artifacts on a git-bearing lane (RunPod/GCE);
    the fellows/SLURM rsync copy has NO git checkout — skip with a log line
    (results land VM-side via the lane's rsync-pull + orchestrator commit).
    Push verification per pod-side-reporting.md: rev-list==0 + per-file
    ls-tree presence assert; failure is LOUD (mirror upload already ran)."""
    root = R.PROJECT_ROOT
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        logger.info("[fits] no git checkout at %s (fellows/SLURM lane) — commit skipped", root)
        return "skipped-no-git"
    branch = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = ctx.eval_dir.relative_to(root)
    subprocess.run(["git", "-C", str(root), "add", "-f", str(rel)], check=True)
    declared = sorted(str(p.relative_to(root)) for p in ctx.eval_dir.rglob("*") if p.is_file())
    cm = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "commit",
            "-m",
            f"issue-1902 P4 eval results ({len(declared)} files)",
            "--",
            str(rel),
        ],
        capture_output=True,
        text=True,
    )
    if cm.returncode != 0 and "nothing to commit" not in (cm.stdout + cm.stderr):
        raise RuntimeError(f"eval-results commit failed: {cm.stdout} {cm.stderr}")
    for attempt in range(2):
        push = subprocess.run(
            ["git", "-C", str(root), "push", "origin", branch], capture_output=True, text=True
        )
        if push.returncode == 0:
            break
        logger.warning("[fits] git push attempt %d failed: %s", attempt + 1, push.stderr[-500:])
    else:
        raise RuntimeError("git push failed twice — eval results unpushed (HF mirror IS uploaded)")
    behind = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", f"origin/{branch}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if behind != "0":
        raise RuntimeError(f"push verify failed: {behind} commits not on origin/{branch}")
    missing = [
        p
        for p in declared
        if not subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", f"origin/{branch}", "--name-only", "--", p],
            capture_output=True,
            text=True,
        ).stdout.strip()
    ]
    if missing:
        raise RuntimeError(
            f"artifact-presence assert failed: {len(missing)} missing e.g. {missing[:3]}"
        )
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    logger.info("[fits] eval results committed+pushed at %s (%d files)", sha[:10], len(declared))
    return sha


def write_results_sentinel(ctx: FitsContext, summary: dict, wall_h: float, commit: str) -> Path:
    """End-of-P4 results sentinel (poll_pipeline contract; smoke writes kind
    epm:smoke-result — the drain never parses `note` for a smoke flag)."""
    kind = "epm:smoke-result" if ctx.smoke else "epm:results"
    gc = summary["grid_cells"]
    h1 = summary["transfer"]["h1"]
    eval_paths = sorted(
        str(p.relative_to(ctx.eval_dir)) for p in ctx.eval_dir.rglob("*.json") if p.is_file()
    )
    pins_p = R.pins_path(ctx.out_root)
    pins = R._read_json(pins_p) if pins_p.exists() else {}
    n_gpu = 1
    try:
        import torch

        n_gpu = max(1, torch.cuda.device_count())
    except Exception:  # noqa: BLE001 — sentinel bookkeeping only
        pass
    note = {
        "phase": "fits",
        "eval_numbers": {
            "layer_star": gc["layer_star"],
            "layer_star_p": gc["layer_star_p"],
            "diag_q_at_star": {
                m: gc["cells"][f"diag_{m}_{corpus}_{ARM_CTX}"]["r2_at_star"]
                for m in ctx.ckpts
                for corpus in C.CORPORA
                if f"diag_{m}_{corpus}_{ARM_CTX}" in gc["cells"]
            },
            "h1_r_adj": h1["r_adj"],
            "h1_verdict": h1["verdict"],
            "h1_ci_delta_conf": h1["ci_delta_conf"],
            "h1_ci_delta_kill": h1["ci_delta_kill"],
            "gate_b": summary["gate_b"],
        },
        "eval_paths": eval_paths,
        "reproducibility_card": {
            "training": "N/A — no training in this task (released OLMo-2 checkpoints only)",
            "wandb_url": "n/a (no training — activation-map fits only)",
            "hf_data_prefixes": {
                "corpus": C.CORPUS_HF_PATH,
                "raw_completions": C.RAW_GEN_HF_PATH,
                "activation_store": C.STORE_HF_PATH,
                "eval_mirror": C.EVAL_MIRROR_HF_PATH,
            },
            "model_ids": C.MODEL_IDS,
            "revision_pins": pins,
            "fold_seed": C.FOLD_SEED,
            "n_folds": C.N_FOLDS,
            "lambdas": "logspace(-2,4,13) GCV (fit_h defaults; #823/#779)",
            "n_boot": ctx.n_boot,
            "recipe_version": FITS_RECIPE_VERSION,
        },
        "hf_hub_url": f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/{C.EVAL_MIRROR_HF_PATH}",
        "worktree_path": ".claude/worktrees/issue-1902",
        "final_commit_sha": commit if commit not in ("", "skipped-no-git") else R._git_sha(),
        "gpu_hours_used": round(wall_h * n_gpu, 2),
        "gpu_hours_budgeted": 29,
        "plan_deviations": [
            "transfer battery on the single-turn corpus only (plan §9 sizing: 12 pairs x 4 "
            "modes x 6 folds — one corpus)",
            "matched-null draws N1=N2=4/fold (pooled 24/pair; plan pins no count)",
            f"fit input summary = u_mean (ctx) / p_mean (prefix) — the unit-A/B pilot precedent",
        ],
    }
    sdir = R._sentinel_dir(ctx.out_root)
    sdir.mkdir(parents=True, exist_ok=True)
    slug = kind.replace(":", "_")
    path = sdir / f"issue-{R.ISSUE}-{slug}-{int(time.time())}.json"
    R._write_json_atomic(
        path,
        {
            "sentinel_schema_version": 1,
            "kind": kind,
            "version": 1,
            "task_id": R.ISSUE,
            "by": "issue1902_fits",
            "ts": R._now_iso(),
            "smoke": ctx.smoke,
            "blocks_pipeline": False,
            "note": json.dumps(note, ensure_ascii=False),
        },
    )
    logger.info("[sentinel] wrote %s", path)
    return path


# ── entry ────────────────────────────────────────────────────────────────────


def run_fits(args: argparse.Namespace, out_root: Path, ckpts: list[str]) -> None:
    """P4 driver (called by issue1902_run --phase fits)."""
    t_start = time.time()
    print(f"[phase=fits] ckpts={ckpts} smoke={bool(args.smoke)}", flush=True)
    ensure_store_staged(out_root, ckpts)
    ctx = FitsContext(args, out_root, ckpts)
    R.headroom_gate(out_root, "fits", 1, 4.0)
    logger.info(
        "[fits] layers=%s n_single=%d n_multi=%d eval_dir=%s",
        ctx.layers,
        ctx.corpora[C.CORPUS_SINGLE].n,
        ctx.corpora[C.CORPUS_MULTI].n,
        ctx.eval_dir,
    )
    pool = _WorkerPool(ctx)
    parity = parity_gate(ctx, pool.devices[0])

    # stage 1: diagonal sweep + CKA (independent units, one queue)
    base_regime = {"recipe": FITS_RECIPE_VERSION, "smoke": ctx.smoke, "layers": ctx.layers}
    units1 = []
    for m in ckpts:
        for corpus in C.CORPORA:
            for fold in range(ctx.corpora[corpus].n_folds):
                units1.append(
                    {
                        "unit": f"sweep_{m}_{corpus}_f{fold}",
                        "regime": {**base_regime, "n": ctx.corpora[corpus].n},
                        "fn": run_sweep_unit,
                        "kw": {"m": m, "corpus": corpus, "fold": fold},
                    }
                )
    for corpus in C.CORPORA:
        for a in range(len(ckpts)):
            for b in range(a + 1, len(ckpts)):
                units1.append(
                    {
                        "unit": f"cka_{ckpts[a]}{ckpts[b]}_{corpus}",
                        "regime": {**base_regime, "n": ctx.corpora[corpus].n},
                        "fn": run_cka_unit,
                        "kw": {"i": ckpts[a], "j": ckpts[b], "corpus": corpus},
                    }
                )
    pool.run("sweep+cka", units1)

    # P4-entry pilot projection (pre-registered abort >2x the §9 row).
    if ctx.pilot_timings:
        per_unit = ctx.pilot_timings[0]["wall_s"]
        n_sweep = len([u for u in units1 if u["unit"].startswith("sweep_")])
        projected_h = per_unit * n_sweep / max(len(pool.devices), 1) / 3600.0
        pilot = {
            "per_sweep_unit_s": per_unit,
            "n_sweep_units": n_sweep,
            "workers": len(pool.devices),
            "projected_sweep_wall_h": round(projected_h, 3),
            "planned_wall_h": P4_PLANNED_WALL_H,
            "abort_ratio": P4_PILOT_ABORT_RATIO,
        }
        logger.info("[fits] P4-entry pilot: %s", pilot)
        if not ctx.smoke and projected_h > P4_PILOT_ABORT_RATIO * P4_PLANNED_WALL_H:
            R.designed_halt(ctx.out_root, "p4_pilot_wall", pilot)

    selection = select_layers(ctx)
    logger.info("[fits] selection: %s", selection["selection"])
    star_regime = {
        **base_regime,
        "layer_star": ctx.layer_star,
        "layer_star_p": ctx.layer_star_p,
        "n_null": ctx.n_null,
        "n_shuffle": ctx.n_shuffle,
        "n_rot": ctx.n_rot,
    }

    # stage 2: grid + mlp + transfer + star/robust + operator, ONE queue
    units2: list[dict] = []
    for corpus in C.CORPORA:
        for m in ckpts:
            for s in ckpts:
                if m == s:
                    continue
                for fold in range(ctx.corpora[corpus].n_folds):
                    units2.append(
                        {
                            "unit": f"grid_{m}{s}_{corpus}_{ARM_CTX}_f{fold}",
                            "regime": star_regime,
                            "fn": run_grid_unit,
                            "kw": {"m": m, "s": s, "corpus": corpus, "fold": fold, "arm": ARM_CTX},
                        }
                    )
    for m in ckpts:
        for s in ckpts:
            if m == s:
                continue
            for fold in range(ctx.corpora[C.CORPUS_MULTI].n_folds):
                units2.append(
                    {
                        "unit": f"grid_{m}{s}_{C.CORPUS_MULTI}_{ARM_PRE}_f{fold}",
                        "regime": star_regime,
                        "fn": run_grid_unit,
                        "kw": {
                            "m": m,
                            "s": s,
                            "corpus": C.CORPUS_MULTI,
                            "fold": fold,
                            "arm": ARM_PRE,
                        },
                    }
                )
    for m in ckpts:
        for corpus in C.CORPORA:
            for fold in range(ctx.corpora[corpus].n_folds):
                units2.append(
                    {
                        "unit": f"star_{m}_{corpus}_f{fold}",
                        "regime": star_regime,
                        "fn": run_star_unit,
                        "kw": {"m": m, "corpus": corpus, "fold": fold},
                    }
                )
    for m in ckpts:
        if m == "B":
            continue  # base native render IS plain (A3)
        for fold in range(ctx.corpora[C.CORPUS_SINGLE].n_folds):
            units2.append(
                {
                    "unit": f"robust_{m}_f{fold}",
                    "regime": star_regime,
                    "fn": run_robust_unit,
                    "kw": {"m": m, "fold": fold},
                }
            )
    for arm in (ARM_CTX, ARM_PRE):
        corpus = C.CORPUS_SINGLE if arm == ARM_CTX else C.CORPUS_MULTI
        for m in ckpts:
            for s in ckpts:
                for fold in range(ctx.corpora[corpus].n_folds):
                    units2.append(
                        {
                            "unit": f"mlp_{arm}_{m}{s}_f{fold}",
                            "regime": star_regime,
                            "fn": run_mlp_unit,
                            "kw": {"arm": arm, "m": m, "s": s, "fold": fold},
                        }
                    )
    for i in ckpts:
        for j in ckpts:
            if i == j:
                continue
            for fold in range(ctx.corpora[C.CORPUS_SINGLE].n_folds):
                units2.append(
                    {
                        "unit": f"xfer_{i}{j}_f{fold}",
                        "regime": star_regime,
                        "fn": run_xfer_unit,
                        "kw": {"i": i, "j": j, "fold": fold},
                    }
                )
    for i, j in realized_transitions(ckpts):
        units2.append(
            {
                "unit": f"operator_{i}{j}",
                "regime": star_regime,
                "fn": run_operator_unit,
                "kw": {"i": i, "j": j},
            }
        )
    pool.run("grid+mlp+xfer+star+operator", units2)

    wall_h = (time.time() - t_start) / 3600.0
    summary = finalize(ctx, selection, parity, wall_h)
    _upload_eval_mirror(ctx)
    commit = "" if ctx.smoke else _commit_eval_results(ctx)
    write_results_sentinel(ctx, summary, (time.time() - t_start) / 3600.0, commit)
    R.write_phase_sentinel(
        out_root,
        "fits",
        {
            "layer_star": ctx.layer_star,
            "h1_verdict": summary["transfer"]["h1"]["verdict"],
            "gate_b_n_pass": summary["gate_b"]["n_pass"],
        },
        smoke=ctx.smoke,
    )
    print(f"[fits] done wall={wall_h:.2f}h", flush=True)
