#!/usr/bin/env python
"""#1482 — dense context vector -> SAE answer features at FULL DICTIONARY WIDTH.

The banked #1482 P3 ``sae_dense_in`` arm fits the dense last-prompt-token context
state (``cx_last`` @ L19, 3,584 dims) onto the answer-side SAE features restricted
to a 16,384-feature activity panel. This driver runs the SAME map at the FULL
131,072-feature dictionary, as a grid of independent one-GPU cells:

    ridge x {mean, max, frac}   +   MLP x {mean, max, frac}

plus a panel-width MLP replication cell (the MLP reproduction gate — an MLP is
NOT column-independent, so a full-width MLP restricted to the panel is a
DIFFERENT model and cannot reproduce the banked pooled R^2 by construction) and
an optional width-32768 capacity cell.

Reuse (never reimplemented):
  * ``issue1738_sae_arm._GramFactor`` — ONE standardizer + fp64 Gram + eigh per
    cell; every lambda is a spectral rescaling of the SAME factorization, and the
    output columns are solved in blocks reusing it. Only the INPUT matrix differs
    from #1738 (dense 3,584 instead of SAE-encoded 8,192), so the Gram is smaller.
  * ``issue1738_sae_fullwidth``'s sparse ``X^T Y`` shape — accumulated against the
    full-width CSR, so no dense (n, 131072) answer matrix ever exists.
  * The ``sae_dense_in`` MLP recipe (``issue779_ffc_n1m_fits._fit_mlp_minibatch``):
    width 8192, GELU, MSE, AdamW lr 3e-4 / wd 1e-4, batch 4096, 10% internal-val
    early stop with patience 20. Targets are densified PER BATCH on the GPU from
    the resident sparse store — the full (120000, 131072) dense target (63 GB) is
    never materialized.

Inputs are the VM-produced #1482 dense-bridge artifacts (``X_dense.f32.mm`` +
row registry) and the 1,920-shard pooled SAE store. Both are staged from the HF
data repo; the git-clone compute lanes ship no ``data/`` (#734/#1773), so the
dense design is uploaded by ``--phase upload-inputs`` from the VM first.

Phases: upload-inputs (VM) -> stage -> assemble -> fit (one cell) -> a per-cell
upload that fires the moment each cell lands (#664).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Derived from __file__, NOT task_workflow.repo_root(): that resolver REFUSES a
# checkout with no `tasks/` directory, and the compute lanes run on sparse /
# shallow checkouts that legitimately exclude it (a full checkout of this repo is
# 175,707 files — ~65 min on the pod's MooseFS mount). This driver reads only
# scripts/ + src/ + eval_results/ and never touches task state, so the
# tasks/-path resolver rule does not apply to it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: thread caps + credentials BEFORE numpy/torch import

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482.densesae")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue1482_error_analysis/analysis_tensors/sae_pooled"
INPUTS_PREFIX = "issue1482_densesae_fullwidth/inputs"
OUT_PREFIX = "issue1482_densesae_fullwidth"

# VM-local source of the dense design (the #1482 dense-bridge round's build).
VM_DENSE_BASE = Path("/mnt/eps-data/thomasjiralerspong/issue1482_saedense")

OUT = PROJECT_ROOT / "eval_results" / "issue_1482" / "densesae_fullwidth"
BANKED_PERFEATURE = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1482"
    / "sae_perfeature"
    / "sae_dense_in__mean__ridge.npz"
)
BANKED_RIDGE_JSON = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1482"
    / "sae_perfeature"
    / "unit_ridge__sae_dense_in.json"
)
BANKED_MLP_JSON = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1482"
    / "sae_perfeature"
    / "unit_mlp__sae_dense_in__mean.json"
)

BANKED_GATE_FILES = (BANKED_PERFEATURE, BANKED_RIDGE_JSON, BANKED_MLP_JSON)

DICT_SIZE = 131_072
H_DIM = 3_584
N_ROWS = 142_000
POOLINGS = ("mean", "max", "frac")
VAL_KEY = {"mean": "ans_mean", "max": "ans_max", "frac": "ans_frac"}

# Parent 23-value lambda grid, verbatim (issue779_ffc_n1m_fits.LAMBDAS_N1M).
LAMBDAS = np.logspace(-3, 8, 23)
OUT_BLOCK = 16_384  # output-column block for the ridge solve + R^2 accumulation
GRAM_BLOCK = 20_000  # train-row block for the streamed standardizer + Gram
PRED_ROW_BLOCK = 4_096  # holdout row block for the prediction passes
ROW_BLOCK = 8_192  # row block for the sparse ragged gather (bounds int64 temps)

# The banked sae_dense_in MLP recipe (issue779_fitter_fair_comparison + n1m_fits).
MLP_WIDTH = 8_192
MLP_LR = 3e-4
MLP_WD = 1e-4
MLP_BATCH = 4_096
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 20
SEED = 1482

REPRO_TOL_POOLED = 1e-3  # |full-width panel-restricted pooled R^2 - banked|
REPRO_TOL_PERFEAT = 5e-3  # max |per-feature R^2 delta| on the 16,384 panel

# Cell registry — ONE source for the dispatcher fan-out AND the child resolve.
# All seven are REQUIRED; there is no optional/capacity tier (the width-32768
# capacity cell was removed by user directive).
CELLS = tuple(
    [f"ridge__{p}" for p in POOLINGS] + [f"mlp__{p}" for p in POOLINGS] + ["mlpgate__mean"]
)


# ── provenance / small utilities ──────────────────────────────────────────────


def _git_commit() -> str:
    """Repo HEAD, degrading on a git-less scratch tree (#1902) instead of dying."""
    env = os.environ.get("EPS_GIT_SHA")
    if env:
        return env
    r = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        r.stdout.strip()
        if r.returncode == 0 and r.stdout.strip()
        else "unavailable-no-git-checkout"
    )


def _metadata() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": sp.__version__ if hasattr(sp, "__version__") else __import__("scipy").__version__,
        "torch": torch.__version__,
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": SEED,
    }


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=float))
    os.replace(tmp, path)


def _free_bytes(p: Path) -> int:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def _headroom(work: Path, need_gib: float, tag: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    free = _free_bytes(work) / (1024**3)
    if free < need_gib:
        raise SystemExit(f"[{tag}] disk headroom {free:.1f} GiB < {need_gib:.1f} GiB at {work}")
    logger.info("[%s] disk headroom %.1f GiB at %s", tag, free, work)


# ── phase: upload-inputs (VM-side) ────────────────────────────────────────────


def phase_upload_inputs(args) -> int:
    """Push the VM-produced dense design + row registry to the HF data repo.

    The compute lanes clone the git branch only, and ``data/`` is gitignored, so a
    VM-produced input that is never uploaded is unreachable pod-side (#734/#1773).
    """
    src_work = args.vm_dense_base / "work"
    files = {
        "X_dense.f32.mm": src_work / "X_dense.f32.mm",
        "order.npy": src_work / "order.npy",
        "which.npy": src_work / "which.npy",
        "f_out.npy": src_work / "f_out.npy",
        # The reproduction-gate references are COMMITTED under eval_results/, which
        # the SLURM rsync lanes drop wholesale (#1689) — ship them as HF inputs so
        # the gate is lane-independent instead of relying on the git tree.
        **{p.name: p for p in BANKED_GATE_FILES},
    }
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        raise SystemExit(f"[upload-inputs] missing VM-local inputs: {missing} under {src_work}")

    order = np.load(files["order.npy"])
    which = np.load(files["which.npy"])
    f_out = np.load(files["f_out.npy"])
    n = int(order.shape[0])
    expect = n * H_DIM * 4
    got = files["X_dense.f32.mm"].stat().st_size
    if got != expect:
        raise SystemExit(f"[upload-inputs] X_dense size {got} != n*{H_DIM}*4 = {expect}")
    meta = {
        "metadata": _metadata(),
        "source": str(src_work),
        "n_rows": n,
        "h_dim": H_DIM,
        "dtype": "float32",
        "field": "cx_last@L19 (dense last-prompt-token context state)",
        "row_order": "holdout ++ sae_fit ++ sae_val (store-present rows only)",
        "which_codes": {"0": "holdout", "1": "sae_fit", "2": "sae_val"},
        "counts": {
            k: int(v)
            for k, v in zip(
                ("holdout", "sae_fit", "sae_val"), np.bincount(which, minlength=3), strict=True
            )
        },
        "panel_f_out_n": int(f_out.shape[0]),
        "parent_designs_meta": json.loads((src_work / "designs_meta.json").read_text()),
    }
    meta_path = src_work / "densesae_inputs_meta.json"
    _write_json(meta_path, meta)
    files["densesae_inputs_meta.json"] = meta_path

    for name, path in files.items():
        dest = f"{INPUTS_PREFIX}/{name}"
        logger.info("[upload-inputs] %s (%.2f GB) -> %s", name, path.stat().st_size / 1e9, dest)
        hub.retry_transient(
            lambda p=path, d=dest: hub._upload(
                p,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=d,
                upload_as_file=True,
            ),
            what=f"upload-inputs {name}",
        )
    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        [f"{INPUTS_PREFIX}/{k}" for k in files],
        path_in_repo=INPUTS_PREFIX,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"[upload-inputs] {len(missing)} paths absent on the Hub: {missing}")
    logger.info("[upload-inputs] verified %d files under %s", len(files), INPUTS_PREFIX)
    return 0


# ── phase: stage ──────────────────────────────────────────────────────────────


def _stage_root(args) -> Path:
    return args.work / "stage"


def _staged(args, prefix: str) -> Path:
    """Where ``stage_hub_prefix`` actually lands ``prefix``.

    ``dest_dir`` is a MIRROR ROOT — files land at ``dest/<repo-relative path>``,
    never directly under ``dest`` (#1774). Assert the arithmetic at the one place
    that computes it so a caller can never nest the prefix under a consumed path.
    """
    root = _stage_root(args)
    out = root / prefix
    assert out.is_relative_to(root), (out, root)
    return out


def phase_stage(args) -> int:
    root = _stage_root(args)
    _headroom(args.work, 4 if args.smoke else 30, "stage")
    counts = {}
    for name, prefix in (("inputs", INPUTS_PREFIX), ("store", STORE_PREFIX)):
        if name == "store" and args.local_store:
            counts[name] = -1
            logger.info("[stage] store: using local --local-store %s", args.local_store)
            continue
        paths = hub.stage_hub_prefix(HF_DATA_REPO, prefix, root, repo_type="dataset")
        counts[name] = len(paths)
        logger.info("[stage] %s: %d files -> %s", name, len(paths), _staged(args, prefix))
    _write_json(args.work / "stage_report.json", {"counts": counts, "root": str(root)})
    return 0


def _store_dir(args) -> Path:
    return Path(args.local_store) if args.local_store else _staged(args, STORE_PREFIX)


def _inputs_dir(args) -> Path:
    return Path(args.local_inputs) if args.local_inputs else _staged(args, INPUTS_PREFIX)


# ── phase: assemble ───────────────────────────────────────────────────────────


def _shard_paths(store: Path, limit: int = 0) -> list[Path]:
    paths = sorted(store.glob("pooled_*.npz"))
    if not paths:
        raise SystemExit(f"no pooled shards under {store} — run --phase stage first")
    return paths[:limit] if limit else paths


class YStore:
    """Read-only full-width CSR over the memmapped answer-side matrix.

    All three poolings share ONE sparsity pattern (the store writes a single
    ``ans_idx``/``idx_off`` per row and three value vectors against it), so the
    indices + indptr are built once and each pooling adds only a value array.
    """

    def __init__(self, work: Path, n_rows: int, nnz: int, poolings=POOLINGS):
        self.indptr = np.load(work / "y_indptr.npy")
        self.indices = np.memmap(work / "y_indices.i32", dtype=np.int32, mode="r", shape=(nnz,))
        self.data = {
            p: np.memmap(work / f"y_val_{p}.f16", dtype=np.float16, mode="r", shape=(nnz,))
            for p in poolings
        }
        self.shape = (n_rows, DICT_SIZE)
        self.nnz = nnz

    def _take(self, rows: np.ndarray, lens: np.ndarray) -> np.ndarray:
        """Flat nnz positions of ``rows`` in stream order (the ragged gather)."""
        starts = self.indptr[rows]
        cum = np.concatenate(([0], np.cumsum(lens)[:-1]))
        return np.repeat(starts - cum, lens) + np.arange(int(lens.sum()), dtype=np.int64)

    def csr_rows(self, rows: np.ndarray, pooling: str, block: int = ROW_BLOCK) -> sp.csr_matrix:
        """scipy CSR of ``Y[rows]`` for one pooling (fp32 values).

        Gathered in ROW BLOCKS into preallocated output arrays: a one-shot take
        over the 120,000 train rows allocates two int64 index temporaries the size
        of the whole gather (~9 GiB at ~567M nnz) on a box where earlyoom is live.
        Blocking bounds the index temporaries to the block while the OUTPUT arrays
        are still allocated exactly once.
        """
        lens = (self.indptr[rows + 1] - self.indptr[rows]).astype(np.int64)
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        total = int(indptr[-1])
        ind = np.empty(total, dtype=np.int32)
        dat = np.empty(total, dtype=np.float32)
        for s in range(0, len(rows), block):
            rb = rows[s : s + block]
            take = self._take(rb, lens[s : s + block])
            dst = slice(int(indptr[s]), int(indptr[min(s + block, len(rows))]))
            ind[dst] = self.indices[take]
            dat[dst] = self.data[pooling][take]
        return sp.csr_matrix((dat, ind, indptr), shape=(len(rows), DICT_SIZE))

    def col_stats(self, rows: np.ndarray, pooling: str, block: int = ROW_BLOCK):
        """Per-column (sum, sumsq) over ``rows``, fp64, streamed in row blocks."""
        s1 = np.zeros(DICT_SIZE, dtype=np.float64)
        s2 = np.zeros(DICT_SIZE, dtype=np.float64)
        for s in range(0, len(rows), block):
            rb = rows[s : s + block]
            lens = (self.indptr[rb + 1] - self.indptr[rb]).astype(np.int64)
            take = self._take(rb, lens)
            c = np.asarray(self.indices[take], dtype=np.int64)
            v = np.asarray(self.data[pooling][take], dtype=np.float64)
            s1 += np.bincount(c, weights=v, minlength=DICT_SIZE)
            s2 += np.bincount(c, weights=v * v, minlength=DICT_SIZE)
        return s1, s2

    def csc_rows(self, rows: np.ndarray, pooling: str) -> sp.csc_matrix:
        """CSC over ``rows`` — built ONCE so column-block slicing is O(block nnz).

        Slicing columns out of a CSR rebuilds the whole matrix per block; the
        holdout pass takes 8 blocks, so the CSR form would rebuild 8x.
        """
        return self.csr_rows(rows, pooling).tocsc()

    def gpu_bundle(self, pooling: str, dev):
        """Whole sparse column/value stream resident on ``dev`` for batch scatter.

        The backing arrays are READ-ONLY memmaps; ``as_tensor`` would alias them
        into a non-writable tensor (torch warns "undefined behavior"), so copy
        explicitly. At ~724M nnz that is ~2.9 GB (indices) + ~1.4 GB (values) of
        transient host memory before the device copy — bounded and one-shot.
        """
        return {
            "indptr": torch.as_tensor(
                np.array(self.indptr, copy=True), dtype=torch.int64, device=dev
            ),
            "indices": torch.as_tensor(
                np.array(self.indices, copy=True), dtype=torch.int32, device=dev
            ),
            "data": torch.as_tensor(
                np.array(self.data[pooling], copy=True), dtype=torch.float16, device=dev
            ),
        }


def _dense_cols(csc: sp.csc_matrix, c0: int, c1: int) -> np.ndarray:
    """Dense fp32 column block out of a prebuilt CSC (O(block nnz), no rebuild)."""
    return np.asarray(csc[:, c0:c1].todense(), dtype=np.float32)


def _assemble_paths(work: Path) -> dict[str, Path]:
    return {
        "meta": work / "ystore_meta.json",
        "indptr": work / "y_indptr.npy",
        "indices": work / "y_indices.i32",
        **{p: work / f"y_val_{p}.f16" for p in POOLINGS},
    }


def phase_assemble(args) -> int:
    """Build the full-width answer-side CSR in the row-registry order."""
    work = args.work
    paths = _assemble_paths(work)
    reg = _load_registry(args)
    shards = _shard_paths(_store_dir(args), args.max_shards)
    regime = {
        "n_rows": int(reg["n"]),
        "n_shards": len(shards),
        "smoke": bool(args.smoke),
    }
    if paths["meta"].exists() and not args.rebuild:
        prior = json.loads(paths["meta"].read_text())
        if prior.get("regime") == regime:
            logger.info("[assemble] resume: CSR present under matching regime; skip")
            return 0
        raise SystemExit(
            f"[assemble] work holds a build under a DIFFERENT regime: {prior.get('regime')} "
            f"!= {regime} — pass --rebuild or use a fresh --work"
        )

    row_pos = reg["row_pos"]
    n = int(reg["n"])
    _headroom(work, 2 if args.smoke else 25, "assemble")

    # pass 1: per-registry-row nnz -> indptr (exact preallocation, no growth)
    lens = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            rid = np.asarray(z["row_idx"], dtype=np.int64)
            off = np.asarray(z["idx_off"], dtype=np.int64)
        for r, o in zip(rid, off, strict=True):
            pos = row_pos.get(int(r))
            if pos is None:
                continue
            if seen[pos]:
                raise SystemExit(f"[assemble] duplicate registry row {int(r)} in {p.name}")
            seen[pos] = True
            lens[pos] = int(o)
        if (i + 1) % 400 == 0 or (i + 1) == len(shards):
            logger.info(
                "[assemble] pass1 shard %d/%d (%.0fs)", i + 1, len(shards), time.time() - t0
            )
    n_missing = int((~seen).sum())
    if n_missing and not args.smoke:
        raise SystemExit(f"[assemble] {n_missing}/{n} registry rows absent from the store")
    indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    nnz = int(indptr[-1])
    logger.info("[assemble] n=%d nnz=%d (%.1f nnz/row)", n, nnz, nnz / max(1, n))

    # pass 2: scatter each shard's rows into their preallocated slices
    ind = np.memmap(work / "y_indices.i32.tmp", dtype=np.int32, mode="w+", shape=(nnz,))
    vals = {
        p: np.memmap(work / f"y_val_{p}.f16.tmp", dtype=np.float16, mode="w+", shape=(nnz,))
        for p in POOLINGS
    }
    t0 = time.time()
    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            rid = np.asarray(z["row_idx"], dtype=np.int64)
            off = np.asarray(z["idx_off"], dtype=np.int64)
            sidx = np.asarray(z["ans_idx"], dtype=np.int32)
            svals = {q: np.asarray(z[VAL_KEY[q]], dtype=np.float16) for q in POOLINGS}
        starts = np.concatenate(([0], np.cumsum(off))).astype(np.int64)
        for j, (r, o) in enumerate(zip(rid, off, strict=True)):
            pos = row_pos.get(int(r))
            if pos is None:
                continue
            src = slice(int(starts[j]), int(starts[j]) + int(o))
            dst = slice(int(indptr[pos]), int(indptr[pos]) + int(o))
            ind[dst] = sidx[src]
            for q in POOLINGS:
                vals[q][dst] = svals[q][src]
        if (i + 1) % 400 == 0 or (i + 1) == len(shards):
            logger.info(
                "[assemble] pass2 shard %d/%d (%.0fs)", i + 1, len(shards), time.time() - t0
            )
    ind.flush()
    del ind
    for q in POOLINGS:
        vals[q].flush()
    del vals
    np.save(paths["indptr"], indptr)
    os.replace(work / "y_indices.i32.tmp", paths["indices"])
    for q in POOLINGS:
        os.replace(work / f"y_val_{q}.f16.tmp", paths[q])
    _write_json(
        paths["meta"],
        {
            "metadata": _metadata(),
            "regime": regime,
            "nnz": nnz,
            "n_rows": n,
            "dict_size": DICT_SIZE,
            "n_rows_missing_from_store": n_missing,
            "wall_s": time.time() - t0,
        },
    )
    logger.info("[assemble] done nnz=%d -> %s", nnz, work)
    return 0


# ── row registry + dense design ───────────────────────────────────────────────


def _load_registry(args) -> dict:
    """Row order + split membership from the dense-bridge build.

    ``order`` is the global row id per matrix row (holdout ++ sae_fit ++ sae_val);
    ``which`` codes 0/1/2 for holdout/sae_fit/sae_val. Both ride the same build
    that produced ``X_dense.f32.mm``, so the design and the registry are aligned
    by construction — no re-derivation from the split manifest.
    """
    d = _inputs_dir(args)
    order = np.load(d / "order.npy")
    which = np.load(d / "which.npy")
    f_out = np.load(d / "f_out.npy")
    assert order.shape == which.shape, (order.shape, which.shape)
    n_full = int(order.shape[0])
    keep = np.arange(n_full)
    if args.smoke:
        keep = _smoke_rows(which, args)
    order_k, which_k = order[keep], which[keep]
    reg = {
        "n_full": n_full,
        "n": int(len(keep)),
        "keep": keep,
        "order": order_k,
        "which": which_k,
        "row_pos": {int(r): i for i, r in enumerate(order_k)},
        "ho": np.where(which_k == 0)[0].astype(np.int64),
        "tr": np.where(which_k == 1)[0].astype(np.int64),
        "va": np.where(which_k == 2)[0].astype(np.int64),
        "f_out": f_out,
    }
    for k in ("tr", "va", "ho"):
        if len(reg[k]) == 0:
            raise SystemExit(f"[registry] empty split '{k}' (n={reg['n']})")
    return reg


def _smoke_rows(which: np.ndarray, args) -> np.ndarray:
    """Deterministic tiny slice keeping ALL THREE splits above their floors.

    Sized from the REALIZED per-split availability, never an assumed cap (#1489):
    the ridge val sweep needs >= 2 val rows for a finite ss_tot and the MLP needs
    >= 1 internal-val row after its 10% carve.
    """
    rng = np.random.default_rng(SEED)
    want = {0: args.smoke_holdout, 1: args.smoke_train, 2: args.smoke_val}
    out = []
    for code, k in want.items():
        idx = np.where(which == code)[0]
        take = min(int(k), len(idx))
        if take < 2:
            raise SystemExit(f"[smoke] split {code} has only {len(idx)} rows; need >= 2")
        out.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(out))


def _load_design(args, reg: dict) -> np.ndarray:
    path = _inputs_dir(args) / "X_dense.f32.mm"
    expect = reg["n_full"] * H_DIM * 4
    got = path.stat().st_size
    if got != expect:
        raise SystemExit(f"[design] {path} is {got} bytes, expected {expect}")
    full = np.memmap(path, dtype=np.float32, mode="r", shape=(reg["n_full"], H_DIM))
    if reg["n"] == reg["n_full"]:
        return full
    return np.ascontiguousarray(full[reg["keep"]])


# ── ridge cell ────────────────────────────────────────────────────────────────


def _val_block_ss(y_val, ev, B, ymu, s_eig, c0, c1):
    """(ss_res per lambda, ss_tot) over val rows for output columns [c0, c1)."""
    yb = y_val[:, c0:c1]
    Bb = B[:, c0:c1]
    mub = ymu[c0:c1]
    sst = float(((yb - yb.mean(0)) ** 2).sum())
    ssr = np.empty(len(LAMBDAS))
    for i, lam in enumerate(LAMBDAS):
        pred = (ev * (1.0 / (s_eig + float(lam)))) @ Bb + mub
        ssr[i] = float(((yb - pred) ** 2).sum())
    return ssr, sst


def _ridge_holdout(y_ho_csc, eh, B, ymu, s_eig, lam, dev):
    """Per-column ss_res over the holdout at one lambda, in output-column blocks."""
    inv = 1.0 / (s_eig + float(lam))
    ss_res = np.zeros(DICT_SIZE, dtype=np.float64)
    t0 = time.time()
    n_blocks = (DICT_SIZE + OUT_BLOCK - 1) // OUT_BLOCK
    for bi, c0 in enumerate(range(0, DICT_SIZE, OUT_BLOCK)):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        yb = torch.as_tensor(_dense_cols(y_ho_csc, c0, c1), dtype=torch.float64, device=dev)
        pred = (eh * inv) @ B[:, c0:c1] + ymu[c0:c1]
        ss_res[c0:c1] = ((yb - pred) ** 2).sum(0).cpu().numpy()
        del yb, pred
        if (bi + 1) % 2 == 0 or (bi + 1) == n_blocks:
            logger.info("[ridge] holdout block %d/%d (%.0fs)", bi + 1, n_blocks, time.time() - t0)
    return ss_res


def _xty_scipy(Ytr: sp.csr_matrix, Xstd_t: torch.Tensor, dev) -> torch.Tensor:
    """(h, D) = Xstd^T Y via scipy — the #1738 fullwidth shape, host-side."""
    Xstd = Xstd_t.cpu().numpy()
    return torch.as_tensor(np.ascontiguousarray((Ytr.T @ Xstd).T), dtype=torch.float64, device=dev)


def _xty_cusparse(Ytr: sp.csr_matrix, Xstd_t: torch.Tensor) -> torch.Tensor:
    """(h, D) = Xstd^T Y as ONE cuSPARSE spmm on Y^T (a CSC read as CSR).

    The scipy form is a ~2e12-MAC single-threaded host product (~11-34 min at
    this shape) and THREE ridge cells run it concurrently on one host, so the
    contention compounds. cuSPARSE does the identical arithmetic on-device.
    """
    Yt = Ytr.tocsc()  # same arrays, read as CSR of Y^T (D x n_tr)
    dev = Xstd_t.device
    Yt_gpu = torch.sparse_csr_tensor(
        torch.as_tensor(Yt.indptr.astype(np.int64), device=dev),
        torch.as_tensor(Yt.indices.astype(np.int64), device=dev),
        torch.as_tensor(Yt.data.astype(np.float64), device=dev),
        size=(DICT_SIZE, Ytr.shape[0]),
    )
    out = torch.sparse.mm(Yt_gpu, Xstd_t)  # (D, h)
    del Yt_gpu
    return out.T.contiguous()


def _xty(Ytr, Xstd_t, dev, mode: str) -> tuple[torch.Tensor, str]:
    """Dispatch the X^T Y accumulation; ``auto`` prefers cuSPARSE on a CUDA device.

    A cuSPARSE failure (OOM, unsupported dtype) falls back to scipy LOUDLY rather
    than silently — the fallback is a throughput regression worth seeing in the log.
    """
    if mode == "scipy" or (mode == "auto" and dev.type != "cuda"):
        return _xty_scipy(Ytr, Xstd_t, dev), "scipy"
    try:
        return _xty_cusparse(Ytr, Xstd_t), "cusparse"
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if mode == "cusparse":
            raise
        logger.warning("[ridge] cuSPARSE X^T Y failed (%s) — scipy fallback", exc)
        torch.cuda.empty_cache()
        return _xty_scipy(Ytr, Xstd_t, dev), "scipy-fallback"


def _assert_xty_equivalence(Ytr, Xstd_t, backend: str, n_probe: int = 4096) -> dict:
    """cuSPARSE vs scipy on the SAME row subset — the cuSPARSE fix-engaged signal.

    The cuSPARSE branch is CUDA-only, so no CPU-host smoke can reach it (#1776);
    this runs BOTH backends on the first ``n_probe`` train rows at PRODUCTION
    column width and compares. Subsetting ROWS (not columns) keeps the probe
    seconds-cheap so it can ride the pilot without distorting its measured wall.
    """
    k = min(n_probe, Ytr.shape[0])
    sub = Ytr[:k]
    xsub = Xstd_t[:k]
    ref = _xty_scipy(sub, xsub, xsub.device)
    try:
        got = _xty_cusparse(sub, xsub)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        # UNAVAILABLE is not WRONG: the dispatch below falls back to scipy loudly,
        # so a missing/OOM cuSPARSE must degrade rather than kill the run. A
        # DISAGREEMENT (below) stays fatal — that would be a correctness bug.
        logger.warning("[ridge] cuSPARSE unavailable for the equivalence probe: %s", exc)
        torch.cuda.empty_cache()
        return {"n_probe_rows": k, "skipped": repr(exc), "backend": backend}
    num = float((got - ref).abs().max())
    den = float(ref.abs().max()) + 1e-30
    rel = num / den
    logger.info(
        "[ridge] xty equivalence (%s vs scipy, n=%d x D=%d): max_abs=%.3e rel=%.3e",
        backend,
        k,
        DICT_SIZE,
        num,
        rel,
    )
    if rel > 1e-10:
        raise SystemExit(f"xty backend {backend} disagrees with scipy: rel={rel:.3e}")
    return {"n_probe_rows": k, "max_abs": num, "rel": rel, "backend": backend}


def fit_ridge(args, reg, ystore, X, pooling: str, dev) -> dict:
    from issue1738_sae_arm import _GramFactor  # reused VERBATIM

    tr, va, ho = reg["tr"], reg["va"], reg["ho"]
    if len(tr) < H_DIM and not args.smoke:
        raise SystemExit(f"n_train {len(tr)} < d_in {H_DIM}: estimator-degenerate, refusing")

    t0 = time.time()
    fac = _GramFactor(X, tr, dev, GRAM_BLOCK)
    t_gram = time.time() - t0
    logger.info("[ridge] %s: Gram+eigh %.0fs (d_in=%d)", pooling, t_gram, X.shape[1])

    s1, _ = ystore.col_stats(tr, pooling)
    ymu_np = s1 / len(tr)
    ymu = torch.as_tensor(ymu_np, dtype=torch.float64, device=dev)

    t0 = time.time()
    Xstd_t = fac.std_rows(tr)
    Ytr = ystore.csr_rows(tr, pooling)
    xty_equiv = None
    if args.verify_xty and dev.type == "cuda":
        xty_equiv = _assert_xty_equivalence(Ytr, Xstd_t, "cusparse")
    xty, xty_backend = _xty(Ytr, Xstd_t, dev, args.xty_device)
    del Ytr, Xstd_t
    xty -= torch.outer(fac.colsum, ymu)
    B = fac.U.T @ xty
    del xty
    t_xty = time.time() - t0
    logger.info(
        "[ridge] %s: X^T Y (full width, %s) %.0fs -> B%s",
        pooling,
        xty_backend,
        t_xty,
        tuple(B.shape),
    )

    ev = fac.std_rows(va) @ fac.U
    eh = fac.std_rows(ho) @ fac.U
    y_val = torch.as_tensor(ystore.csr_rows(va, pooling).toarray(), dtype=torch.float64, device=dev)

    t0 = time.time()
    ssr = np.zeros(len(LAMBDAS))
    sst = 0.0
    for c0 in range(0, DICT_SIZE, OUT_BLOCK):
        r, t = _val_block_ss(y_val, ev, B, ymu, fac.s_eig, c0, min(c0 + OUT_BLOCK, DICT_SIZE))
        ssr += r
        sst += t
    del y_val
    val_r2 = 1.0 - ssr / max(sst, 1e-30)
    best = int(np.nanargmax(val_r2))
    sel_lam = float(LAMBDAS[best])
    t_val = time.time() - t0
    logger.info(
        "[ridge] %s: val sweep %.0fs -> lambda=%.6g val_R2=%.6f",
        pooling,
        t_val,
        sel_lam,
        val_r2[best],
    )

    banked = json.loads(_banked(args, BANKED_RIDGE_JSON).read_text())["arm_doc"][
        f"{pooling}__ridge"
    ]
    banked_lam = float(banked["selected_lambda"])
    s1h, s2h = ystore.col_stats(ho, pooling)
    ss_tot = s2h - (s1h**2) / len(ho)
    y_ho_csc = ystore.csc_rows(ho, pooling)

    res = {}
    for tag, lam in {"selected": sel_lam, "banked_lambda": banked_lam}.items():
        t0 = time.time()
        ss_res = _ridge_holdout(y_ho_csc, eh, B, ymu, fac.s_eig, lam, dev)
        res[tag] = _score(ss_res, ss_tot, reg["f_out"], lam)
        res[tag]["wall_s"] = time.time() - t0
        logger.info(
            "[ridge] %s (%s lam=%.6g): pooled R2 full=%.6f panel=%.6f (%.0fs)",
            pooling,
            tag,
            lam,
            res[tag]["pooled_r2_full"],
            res[tag]["pooled_r2_panel"],
            res[tag]["wall_s"],
        )
        if tag == "selected":
            res[tag]["_perfeat"] = _score_perfeature(ss_res, ss_tot)

    perfeat = res["selected"].pop("_perfeat")
    del fac, B, ev, eh
    return {
        "method": "ridge",
        "pooling": pooling,
        "d_in": int(X.shape[1]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_holdout": int(len(ho)),
        "selected_lambda": sel_lam,
        "val_r2_selected": float(val_r2[best]),
        "val_r2_by_lambda": {str(float(a)): float(b) for a, b in zip(LAMBDAS, val_r2, strict=True)},
        "banked_selected_lambda": banked_lam,
        "at_selected_lambda": res["selected"],
        "at_banked_lambda": res["banked_lambda"],
        "xty_backend": xty_backend,
        "xty_equivalence": xty_equiv,
        "wall_s": {"gram_eigh": t_gram, "xty": t_xty, "val_sweep": t_val},
        "_perfeature": perfeat,
    }


def _score(ss_res: np.ndarray, ss_tot: np.ndarray, f_out: np.ndarray, lam: float) -> dict:
    scored = ss_tot > 1e-12
    panel = np.zeros(DICT_SIZE, dtype=bool)
    panel[f_out] = True
    both = scored & panel
    return {
        "lambda": float(lam),
        "pooled_r2_full": float(1.0 - ss_res[scored].sum() / ss_tot[scored].sum()),
        "pooled_r2_panel": float(1.0 - ss_res[both].sum() / ss_tot[both].sum()),
        "n_scored_columns": int(scored.sum()),
        "n_zero_variance_columns": int((~scored).sum()),
    }


def _score_perfeature(ss_res: np.ndarray, ss_tot: np.ndarray) -> dict:
    scored = ss_tot > 1e-12
    r2 = np.full(DICT_SIZE, np.nan, dtype=np.float64)
    r2[scored] = 1.0 - ss_res[scored] / ss_tot[scored]
    return {"r2": r2, "ss_tot": ss_tot, "ss_res": ss_res, "scored": scored}


# ── MLP cell ──────────────────────────────────────────────────────────────────


def _scatter_targets(bundle, rows_t, d_out: int, col_map=None) -> torch.Tensor:
    """Dense (len(rows), d_out) fp32 target block, scattered on-device.

    Only the batch is ever dense: the full (120000, 131072) dense target is 63 GB
    and is never materialized. ``col_map`` (int64, DICT_SIZE) restricts+remaps the
    columns for the panel-width gate cell; -1 drops a column.
    """
    indptr, indices, data = bundle["indptr"], bundle["indices"], bundle["data"]
    starts = indptr[rows_t]
    lens = indptr[rows_t + 1] - starts
    total = int(lens.sum().item())
    dense = torch.zeros((rows_t.numel(), d_out), dtype=torch.float32, device=rows_t.device)
    if total == 0:
        return dense
    cum = torch.cumsum(lens, 0) - lens
    take = torch.repeat_interleave(starts - cum, lens) + torch.arange(
        total, device=rows_t.device, dtype=torch.int64
    )
    loc = torch.repeat_interleave(
        torch.arange(rows_t.numel(), device=rows_t.device, dtype=torch.int64), lens
    )
    cols = indices[take].to(torch.int64)
    vals = data[take].to(torch.float32)
    if col_map is not None:
        cols = col_map[cols]
        keep = cols >= 0
        loc, cols, vals = loc[keep], cols[keep], vals[keep]
    dense[loc, cols] = vals
    return dense


def _train_x_stats(X, tr, dev):
    """fp32 train-row mean/std of the dense design, streamed."""
    n = len(tr)
    s1 = torch.zeros(X.shape[1], dtype=torch.float64, device=dev)
    s2 = torch.zeros(X.shape[1], dtype=torch.float64, device=dev)
    for s in range(0, n, GRAM_BLOCK):
        xb = torch.as_tensor(
            np.asarray(X[tr[s : s + GRAM_BLOCK]], dtype=np.float32), device=dev
        ).to(torch.float64)
        s1 += xb.sum(0)
        s2 += (xb * xb).sum(0)
    mu = s1 / n
    var = (s2 - n * mu * mu) / max(1, n - 1)
    return mu.to(torch.float32), (torch.sqrt(torch.clamp(var, min=0.0)) + 1e-9).to(torch.float32)


def _ymu_gpu(ystore, tr, pooling, dev, col_map=None, d_out=DICT_SIZE):
    s1, _ = ystore.col_stats(tr, pooling)
    full = torch.as_tensor(s1 / len(tr), dtype=torch.float32, device=dev)
    if col_map is None:
        return full
    out = torch.zeros(d_out, dtype=torch.float32, device=dev)
    src = torch.nonzero(col_map >= 0, as_tuple=True)[0]
    out[col_map[src]] = full[src]
    return out


def fit_mlp(args, reg, ystore, X, pooling: str, dev, width: int, panel: bool) -> dict:
    """The banked ``sae_dense_in`` MLP recipe with per-batch on-device targets."""
    tr, ho = reg["tr"], reg["ho"]
    col_map = None
    d_out = DICT_SIZE
    if panel:
        col_map = torch.full((DICT_SIZE,), -1, dtype=torch.int64, device=dev)
        f_out_t = torch.as_tensor(np.asarray(reg["f_out"], dtype=np.int64), device=dev)
        col_map[f_out_t] = torch.arange(f_out_t.numel(), device=dev, dtype=torch.int64)
        d_out = int(f_out_t.numel())

    t0 = time.time()
    bundle = ystore.gpu_bundle(pooling, dev)
    xmu, xsd = _train_x_stats(X, tr, dev)
    ymu = _ymu_gpu(ystore, tr, pooling, dev, col_map, d_out)
    logger.info(
        "[mlp] %s w=%d d_out=%d: inputs resident (%.0fs)", pooling, width, d_out, time.time() - t0
    )

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr))
    n_val = max(1, round(0.1 * len(tr)))
    va_local, tr_local = perm[:n_val], perm[n_val:]
    if len(tr_local) == 0:
        raise SystemExit(f"[mlp] internal-val carve left 0 train rows (n_train={len(tr)})")
    batch = min(MLP_BATCH, max(8, len(tr_local)))

    torch.manual_seed(SEED)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], width), torch.nn.GELU(), torch.nn.Linear(width, d_out)
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    max_epochs = args.smoke_epochs if args.smoke else MLP_MAX_EPOCHS

    def _xb(rows: np.ndarray) -> torch.Tensor:
        xb = torch.as_tensor(np.asarray(X[rows], dtype=np.float32), device=dev)
        return (xb - xmu) / xsd

    best_val, best_state, bad, epochs_ran = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(max_epochs):
        net.train()
        ep_perm = rng.permutation(len(tr_local))
        for bs in range(0, len(tr_local), batch):
            rows = tr[tr_local[ep_perm[bs : bs + batch]]]
            rows_t = torch.as_tensor(rows, dtype=torch.int64, device=dev)
            tb = _scatter_targets(bundle, rows_t, d_out, col_map) - ymu
            opt.zero_grad(set_to_none=True)
            loss = ((net(_xb(rows)) - tb) ** 2).mean()
            loss.backward()
            opt.step()
            del tb
        net.eval()
        with torch.no_grad():
            vsum, vcnt = 0.0, 0
            for bs in range(0, len(va_local), batch):
                rows = tr[va_local[bs : bs + batch]]
                rows_t = torch.as_tensor(rows, dtype=torch.int64, device=dev)
                tb = _scatter_targets(bundle, rows_t, d_out, col_map) - ymu
                vsum += float(((net(_xb(rows)) - tb) ** 2).sum())
                vcnt += rows.shape[0] * d_out
                del tb
            vloss = vsum / max(1, vcnt)
        epochs_ran = ep + 1
        if vloss < best_val - 1e-7:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
        if epochs_ran % 10 == 0 or epochs_ran == max_epochs:
            logger.info(
                "[mlp] %s w=%d epoch %d/%d val=%.6g best=%.6g (%.0fs)",
                pooling,
                width,
                epochs_ran,
                max_epochs,
                vloss,
                best_val,
                time.time() - t0,
            )
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    t_train = time.time() - t0

    ss_res = torch.zeros(d_out, dtype=torch.float64, device=dev)
    y_sum = torch.zeros(d_out, dtype=torch.float64, device=dev)
    y_sq = torch.zeros(d_out, dtype=torch.float64, device=dev)
    with torch.no_grad():
        for s in range(0, len(ho), PRED_ROW_BLOCK):
            rows = ho[s : s + PRED_ROW_BLOCK]
            rows_t = torch.as_tensor(rows, dtype=torch.int64, device=dev)
            yb = _scatter_targets(bundle, rows_t, d_out, col_map)
            pred = net(_xb(rows)) + ymu
            ss_res += ((yb - pred) ** 2).sum(0).to(torch.float64)
            y_sum += yb.sum(0).to(torch.float64)
            y_sq += (yb.to(torch.float64) ** 2).sum(0)
            del yb, pred
    ss_tot = y_sq - (y_sum**2) / len(ho)
    ssr_np = ss_res.cpu().numpy()
    sst_np = ss_tot.cpu().numpy()

    if panel:
        pooled = float(1.0 - ssr_np.sum() / max(sst_np.sum(), 1e-30))
        pf_ids = np.asarray(reg["f_out"], dtype=np.int64)
        pf = {
            "r2": np.where(sst_np > 1e-12, 1.0 - ssr_np / np.maximum(sst_np, 1e-12), np.nan),
            "ss_tot": sst_np,
            "ss_res": ssr_np,
            "scored": sst_np > 1e-12,
            "feat_ids": pf_ids,
        }
        summary = {
            "pooled_r2_panel": pooled,
            "pooled_r2_full": None,
            "n_scored_columns": int((sst_np > 1e-12).sum()),
        }
    else:
        summary = _score(ssr_np, sst_np, reg["f_out"], float("nan"))
        summary.pop("lambda")
        pf = _score_perfeature(ssr_np, sst_np)

    del bundle, net, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "method": "mlp_panel" if panel else "mlp",
        "pooling": pooling,
        "width": width,
        "d_in": int(X.shape[1]),
        "d_out": d_out,
        "n_train": int(len(tr)),
        "n_internal_val": int(len(va_local)),
        "n_holdout": int(len(ho)),
        "epochs_ran": epochs_ran,
        "best_internal_val_mse": best_val,
        "recipe": {
            "lr": MLP_LR,
            "weight_decay": MLP_WD,
            "batch": batch,
            "max_epochs": max_epochs,
            "patience": MLP_PATIENCE,
            "seed": SEED,
        },
        **summary,
        "wall_s": {"train": t_train},
        "_perfeature": pf,
    }


# ── reproduction gate ─────────────────────────────────────────────────────────


def _banked(args, path: Path) -> Path:
    """Resolve a banked gate reference: HF-staged copy first, git tree as fallback.

    The git copy is absent on the SLURM rsync lanes (eval_results/ is excluded
    wholesale, #1689), so the staged copy is the lane-independent source.
    """
    staged = _inputs_dir(args) / path.name
    if staged.exists():
        return staged
    if path.exists():
        return path
    raise SystemExit(
        f"[gate] banked reference {path.name} absent from BOTH the staged inputs "
        f"({staged}) and the git tree ({path}) — re-run --phase upload-inputs"
    )


def _ridge_gate(args, cell_doc: dict, perfeat: dict, f_out: np.ndarray) -> dict:
    """Panel-restricted reproduction of the banked ridge fit at the banked lambda.

    Ridge solves each output column independently, so restricting the full-width
    solution to the banked panel at the banked lambda IS the banked fit up to
    float ordering — this gate is exact-modulo-float, not a tolerance judgement.
    """
    banked = json.loads(_banked(args, BANKED_RIDGE_JSON).read_text())["arm_doc"]["mean__ridge"]
    z = np.load(_banked(args, BANKED_PERFEATURE))
    bank_ids = np.asarray(z["feat_ids"], dtype=np.int64)
    bank_r2 = np.asarray(z["r2"], dtype=np.float64)
    assert np.array_equal(np.sort(bank_ids), np.sort(np.asarray(f_out, dtype=np.int64)))

    got_pooled = float(cell_doc["at_banked_lambda"]["pooled_r2_panel"])
    d_pooled = got_pooled - float(banked["pooled_r2"])
    mine = perfeat["r2"][bank_ids]
    ok = np.isfinite(mine) & np.isfinite(bank_r2)
    dperf = np.abs(mine[ok] - bank_r2[ok])
    return {
        "banked_pooled_r2": float(banked["pooled_r2"]),
        "fullwidth_panel_restricted_pooled_r2": got_pooled,
        "pooled_delta": d_pooled,
        "pooled_tol": REPRO_TOL_POOLED,
        "banked_median_perfeature_r2": float(np.nanmedian(bank_r2)),
        "refit_median_perfeature_r2": float(np.nanmedian(mine)),
        "banked_frac_positive": float(np.nanmean(bank_r2 > 0)),
        "refit_frac_positive": float(np.nanmean(mine > 0)),
        "max_abs_perfeature_delta": float(dperf.max()) if dperf.size else float("nan"),
        "perfeature_tol": REPRO_TOL_PERFEAT,
        "n_compared": int(ok.sum()),
        "note": (
            "at_selected_lambda is reported separately; this gate uses the BANKED "
            "lambda because the full-width val sweep may select a different one."
        ),
        "pass": bool(
            abs(d_pooled) < REPRO_TOL_POOLED
            and dperf.size
            and float(dperf.max()) < REPRO_TOL_PERFEAT
        ),
    }


def _mlp_gate(args, cell_doc: dict) -> dict:
    """Panel-width MLP replication vs the banked pooled R^2.

    The MLP shares one hidden layer across output columns, so a FULL-WIDTH MLP
    restricted to the panel is a different model — only a panel-width refit can
    reproduce the banked number, and even that is seed/hardware sensitive, so the
    tolerance is a loose sanity band rather than the ridge's exactness gate.
    """
    banked = json.loads(_banked(args, BANKED_MLP_JSON).read_text())["arm_doc"]["mean__mlp"]
    got = float(cell_doc["pooled_r2_panel"])
    d = got - float(banked["pooled_r2"])
    return {
        "banked_pooled_r2": float(banked["pooled_r2"]),
        "banked_epochs_ran": int(banked["epochs_ran"]),
        "refit_pooled_r2": got,
        "refit_epochs_ran": int(cell_doc["epochs_ran"]),
        "delta": d,
        "tol": 0.02,
        "pass": bool(abs(d) < 0.02),
        "note": (
            "panel-width (16,384-output) replication of the banked MLP; a "
            "full-width MLP is a DIFFERENT model and is not expected to match."
        ),
    }


# ── phase: fit (one cell) ─────────────────────────────────────────────────────


def _parse_cell(cell: str) -> tuple[str, str]:
    if cell not in CELLS:
        raise SystemExit(f"unknown cell {cell!r}; known: {list(CELLS)}")
    method, pooling = cell.split("__", 1)
    return method, pooling


def _cell_json(args, cell: str) -> Path:
    return args.out / "cells" / f"{cell}.json"


def _mlp_width(args) -> int:
    """Production hidden width, shrunk only under --smoke.

    The OUTPUT width stays the production 131,072 under smoke — that is what
    exercises the per-batch on-device scatter and the memory path; only the
    hidden layer (a pure SCALE knob, not a branch) shrinks so the CPU smoke fits
    a shared VM.
    """
    return args.smoke_mlp_width if args.smoke else MLP_WIDTH


def phase_summary(args) -> int:
    """Rebuild summary.json from every landed cell.

    Each per-GPU cell runs in its OWN process and writes the summary on exit, so
    whichever finishes last decides its contents; the launcher calls this ONCE
    after joining the workers so the summary reflects the whole grid.
    """
    _write_summary(args, _load_registry(args))
    return 0


def phase_fit(args) -> int:
    cells = list(args.cells) if args.cells else list(CELLS)
    dev = torch.device(args.device)
    if args.gpu_id:
        logger.info(
            "[fit] --gpu-id %s under CUDA_VISIBLE_DEVICES=%r (device resolves to %s)",
            args.gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            dev,
        )
    reg = _load_registry(args)
    meta = json.loads(_assemble_paths(args.work)["meta"].read_text())
    ystore = YStore(args.work, reg["n"], int(meta["nnz"]))
    X = _load_design(args, reg)
    logger.info(
        "[fit] cells=%s device=%s n=%d (tr=%d va=%d ho=%d) nnz=%d",
        cells,
        dev,
        reg["n"],
        len(reg["tr"]),
        len(reg["va"]),
        len(reg["ho"]),
        ystore.nnz,
    )

    rc = 0
    for cell in cells:
        cellp = _cell_json(args, cell)
        if cellp.exists() and not args.no_resume:
            logger.info("[fit] %s: resume-skip", cell)
            continue
        method, pooling = _parse_cell(cell)
        t0 = time.time()
        if method == "ridge":
            doc = fit_ridge(args, reg, ystore, X, pooling, dev)
        elif method == "mlp":
            doc = fit_mlp(args, reg, ystore, X, pooling, dev, _mlp_width(args), panel=False)
        elif method == "mlpgate":
            doc = fit_mlp(args, reg, ystore, X, pooling, dev, _mlp_width(args), panel=True)
        else:  # pragma: no cover — _parse_cell gates the method set
            raise SystemExit(f"unhandled method {method!r}")

        perfeat = doc.pop("_perfeature")
        doc["cell"] = cell
        doc["metadata"] = _metadata()
        doc["smoke"] = bool(args.smoke)
        doc["total_wall_s"] = time.time() - t0

        if cell == "ridge__mean":
            doc["reproduction_gate"] = _gate_or_demote(
                args, lambda: _ridge_gate(args, doc, perfeat, reg["f_out"])
            )
        if cell == "mlpgate__mean":
            doc["reproduction_gate"] = _gate_or_demote(args, lambda: _mlp_gate(args, doc))

        pf_path = _write_perfeature(args, cell, perfeat, reg)
        _write_json(cellp, doc)
        logger.info("[fit] %s done in %.0fs -> %s", cell, doc["total_wall_s"], cellp)
        if not args.skip_upload:
            _upload_cell(args, cell, cellp, pf_path)
        gate = doc.get("reproduction_gate")
        if gate and gate.get("enforced") and not gate.get("pass"):
            logger.error("[fit] %s REPRODUCTION GATE FAILED: %s", cell, gate)
            rc = 3
    _write_summary(args, reg)
    return rc


def _gate_or_demote(args, fn):
    """Run the gate; under --smoke record it INFORMATIONALLY, never as a verdict.

    The gate compares against production-n banked values, so at smoke n it is
    unsatisfiable by construction (#1345) — the COMPUTATION still runs so the code
    path stays exercised, but the verdict does not gate the smoke.
    """
    try:
        g = fn()
    except Exception as exc:  # noqa: BLE001 — recorded, and fatal outside smoke
        if not args.smoke:
            raise
        return {"enforced": False, "smoke_error": repr(exc)}
    g["enforced"] = not args.smoke
    if args.smoke:
        g["note_smoke"] = "informational at smoke n; production-n calibrated"
    return g


def _write_perfeature(args, cell: str, perfeat: dict, reg) -> Path:
    d = args.out / "perfeature"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cell}_perfeature.npz"
    feat_ids = perfeat.get("feat_ids")
    if feat_ids is None:
        feat_ids = np.arange(DICT_SIZE, dtype=np.int64)
    np.savez(
        path,
        feat_ids=np.asarray(feat_ids, dtype=np.int64),
        r2=perfeat["r2"].astype(np.float32),
        ss_tot=perfeat["ss_tot"].astype(np.float64),
        ss_res=perfeat["ss_res"].astype(np.float64),
        scored=perfeat["scored"],
    )
    return path


def _upload_cell(args, cell: str, cellp: Path, pf_path: Path) -> None:
    """Persist the cell the MOMENT it lands — never one terminal batch (#664)."""
    from huggingface_hub import HfApi

    prefix = args.hf_out_prefix
    targets = [
        (pf_path, f"{prefix}/perfeature/{pf_path.name}"),
        (cellp, f"{prefix}/cells/{cellp.name}"),
    ]
    for local, dest in targets:
        hub.retry_transient(
            lambda p=local, d=dest: hub._upload(
                p,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=d,
                upload_as_file=True,
            ),
            what=f"cell {cell}: {local.name}",
        )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        [d for _, d in targets],
        path_in_repo=prefix,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"[fit] {cell}: {len(missing)} paths absent on the Hub: {missing}")
    logger.info("[fit] %s uploaded -> %s/{cells,perfeature}", cell, prefix)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman; NaN when fewer than 3 usable pairs or a side is constant."""
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    r = spearmanr(a[ok], b[ok]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _spearman_brown(r_half: float) -> float:
    """Half-length reliability stepped up to the full-length measurement."""
    if not np.isfinite(r_half) or r_half <= -1.0:
        return float("nan")
    return float(2.0 * r_half / (1.0 + r_half))


def phase_reliability(args) -> int:
    """Split-half RELIABILITY of the per-feature held-out R^2, per activity decile.

    Every downstream read treats per-feature R^2 as a measured property, but it is
    an ESTIMATE whose precision tracks activity: for a rarely-firing feature SS_tot
    is built from mostly zeros plus a handful of nonzeros, so the denominator is
    small and unstable. If reliability itself climbs across activity deciles, a
    predictor that "strengthens with activity" may partly be noise thinning out.

    The HOLDOUT is split (not the training rows) so this isolates SCORING noise —
    the dominant term for rare features — and needs no refit of the map. The map
    IS refit here only because the fitted state does not survive the cell process;
    the fit is deterministic at the recorded lambda, so it reproduces ridge__mean
    exactly (the pooled R^2 is re-reported for confirmation).
    """
    dev = torch.device(args.device)
    reg = _load_registry(args)
    meta = json.loads(_assemble_paths(args.work)["meta"].read_text())
    ystore = YStore(args.work, reg["n"], int(meta["nnz"]))
    X = _load_design(args, reg)
    pooling = "mean"
    tr, ho = reg["tr"], reg["ho"]

    banked_cell = _cell_json(args, "ridge__mean")
    if not banked_cell.exists():
        raise SystemExit(f"[reliability] {banked_cell} absent — run --phase fit first")
    lam = float(json.loads(banked_cell.read_text())["selected_lambda"])
    logger.info("[reliability] refitting ridge__mean at the recorded lambda %.6g", lam)

    from issue1738_sae_arm import _GramFactor  # reused VERBATIM

    fac = _GramFactor(X, tr, dev, GRAM_BLOCK)
    s1, _ = ystore.col_stats(tr, pooling)
    ymu = torch.as_tensor(s1 / len(tr), dtype=torch.float64, device=dev)
    Xstd_t = fac.std_rows(tr)
    Ytr = ystore.csr_rows(tr, pooling)
    xty, xty_backend = _xty(Ytr, Xstd_t, dev, args.xty_device)
    del Ytr, Xstd_t
    xty -= torch.outer(fac.colsum, ymu)
    B = fac.U.T @ xty
    del xty

    # Seeded disjoint halves of the holdout.
    rng = np.random.default_rng(args.reliability_seed)
    perm = rng.permutation(len(ho))
    halves = {"A": ho[np.sort(perm[: len(ho) // 2])], "B": ho[np.sort(perm[len(ho) // 2 :])]}

    r2 = {}
    for name, rows in halves.items():
        eh = fac.std_rows(rows) @ fac.U
        csc = ystore.csc_rows(rows, pooling)
        ss_res = _ridge_holdout(csc, eh, B, ymu, fac.s_eig, lam, dev)
        h1, h2 = ystore.col_stats(rows, pooling)
        ss_tot = h2 - (h1**2) / len(rows)
        scored = ss_tot > 1e-12
        v = np.full(DICT_SIZE, np.nan)
        v[scored] = 1.0 - ss_res[scored] / ss_tot[scored]
        r2[name] = v
        logger.info(
            "[reliability] half %s: %d rows, %d scored columns", name, len(rows), scored.sum()
        )
        del eh, csc

    # Full-holdout re-score, purely to confirm the refit reproduces the banked cell.
    eh_full = fac.std_rows(ho) @ fac.U
    ss_res_full = _ridge_holdout(ystore.csc_rows(ho, pooling), eh_full, B, ymu, fac.s_eig, lam, dev)
    f1, f2 = ystore.col_stats(ho, pooling)
    ss_tot_full = f2 - (f1**2) / len(ho)
    sc = ss_tot_full > 1e-12
    pooled_full = float(1.0 - ss_res_full[sc].sum() / ss_tot_full[sc].sum())

    # Activity = fraction of TRAIN rows in which the feature is active.
    act_cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    for s in range(0, len(tr), ROW_BLOCK):
        sub = ystore.csr_rows(tr[s : s + ROW_BLOCK], pooling)
        act_cnt += np.bincount(sub.indices.astype(np.int64), minlength=DICT_SIZE)
    activity = act_cnt / len(tr)

    usable = np.isfinite(r2["A"]) & np.isfinite(r2["B"])
    idx = np.where(usable)[0]
    order = idx[np.argsort(activity[idx], kind="stable")]
    deciles = []
    for d, chunk in enumerate(np.array_split(order, 10)):
        rh = _spearman(r2["A"][chunk], r2["B"][chunk])
        deciles.append(
            {
                "decile": d + 1,
                "n": int(len(chunk)),
                "activity_min": float(activity[chunk].min()) if len(chunk) else float("nan"),
                "activity_max": float(activity[chunk].max()) if len(chunk) else float("nan"),
                "r_half": rh,
                "r_full_spearman_brown": _spearman_brown(rh),
            }
        )
        logger.info(
            "[reliability] decile %d n=%d act=[%.5f, %.5f] r_half=%.4f r_full=%.4f",
            d + 1,
            len(chunk),
            deciles[-1]["activity_min"],
            deciles[-1]["activity_max"],
            rh,
            deciles[-1]["r_full_spearman_brown"],
        )
    pooled_rh = _spearman(r2["A"][usable], r2["B"][usable])

    outdir = args.out.parent / "r2_reliability"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "r2_halves_perfeature.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int64),
        r2_half_a=r2["A"].astype(np.float32),
        r2_half_b=r2["B"].astype(np.float32),
        activity=activity.astype(np.float32),
        usable=usable,
    )
    doc = {
        "metadata": _metadata(),
        "design": {
            "what": (
                "Split-half reliability of the per-feature held-out R^2 for the "
                "ridge__mean full-width map, pooled and within 10 equal-count "
                "activity deciles."
            ),
            "why_holdout_split": (
                "Splitting the HOLDOUT (not the training rows) isolates SCORING "
                "noise — the dominant term for rare features, whose SS_tot is a "
                "handful of nonzeros — and costs no refit of the map."
            ),
            "refit_note": (
                "The fitted map does not survive the cell process, so it is refit "
                "here at the RECORDED lambda; the fit is deterministic, so this "
                "reproduces ridge__mean rather than being a new fit."
            ),
            "caveat": (
                "A 10,000-row half is noisier than the full 20,000-row holdout by "
                "construction. Spearman-Brown corrects the RELIABILITY estimate for "
                "that; the per-half R^2 values themselves are NOT headline numbers."
            ),
            "attenuation_correction": "rho_true = rho_observed / sqrt(r_full)",
            "mlp_reliability": (
                "NOT computed: the MLP's fitted state (weights + early-stop "
                "snapshot) is not persisted, so scoring halves would require a full "
                "~55 min refit. Skipped deliberately rather than refit."
            ),
        },
        "split": {
            "seed": int(args.reliability_seed),
            "n_holdout": int(len(ho)),
            "n_half_a": int(len(halves["A"])),
            "n_half_b": int(len(halves["B"])),
            "half_a_sha256": _ids_sha(halves["A"]),
            "half_b_sha256": _ids_sha(halves["B"]),
        },
        "refit_check": {
            "lambda": lam,
            "pooled_r2_full_holdout": pooled_full,
            "xty_backend": xty_backend,
        },
        "pooled": {
            "n_features": int(usable.sum()),
            "r_half": pooled_rh,
            "r_full_spearman_brown": _spearman_brown(pooled_rh),
        },
        "deciles": deciles,
    }
    _write_json(outdir / "reliability.json", doc)
    logger.info(
        "[reliability] pooled r_half=%.4f r_full=%.4f over %d features -> %s",
        pooled_rh,
        _spearman_brown(pooled_rh),
        int(usable.sum()),
        outdir,
    )
    if not args.skip_upload:
        _upload_reliability(args, outdir)
    return 0


def _ids_sha(ids: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _upload_reliability(args, outdir: Path) -> None:
    from huggingface_hub import HfApi

    prefix = f"{args.hf_out_prefix}/r2_reliability"
    targets = [(p, f"{prefix}/{p.name}") for p in sorted(outdir.iterdir()) if p.is_file()]
    for local, dest in targets:
        hub.retry_transient(
            lambda p=local, d=dest: hub._upload(
                p,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=d,
                upload_as_file=True,
            ),
            what=f"reliability: {local.name}",
        )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, [d for _, d in targets], path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"[reliability] {len(missing)} paths absent on the Hub: {missing}")
    logger.info("[reliability] uploaded %d files -> %s", len(targets), prefix)


def _write_summary(args, reg) -> None:
    cells = {}
    for p in sorted((args.out / "cells").glob("*.json")):
        cells[p.stem] = json.loads(p.read_text())
    doc = {
        "metadata": _metadata(),
        "design": {
            "question": (
                "Does the dense context vector -> SAE answer-feature map behave the "
                "same at FULL dictionary width (131,072) as at the banked "
                "16,384-feature activity panel?"
            ),
            "input": "cx_last@L19 dense context state (3,584 dims), train-standardized",
            "targets": "answer-side SAE features, full 131,072-wide dictionary",
            "poolings": list(POOLINGS),
            "ridge": (
                "shared-Gram fp64 (issue1738_sae_arm._GramFactor, reused verbatim): "
                "ONE Gram + eigh per cell, outputs solved in 16,384-column blocks "
                "reusing the factorization; X^T Y accumulated against the sparse CSR."
            ),
            "mlp": (
                "banked sae_dense_in recipe (width 8192, GELU, AdamW lr 3e-4 / wd "
                "1e-4, batch 4096, 10% internal-val early stop, patience 20); "
                "targets densified per batch on-device from the resident sparse store."
            ),
            "lambdas": [float(x) for x in LAMBDAS],
            "mapping_baselines": (
                "The standing identity+learned-bias baseline (v_hat = x + b) is "
                "INAPPLICABLE here by dimension: the input is the 3,584-dim dense "
                "context state and the target is the 131,072-wide SAE feature "
                "vector, so no identity map exists between them. Stated rather "
                "than silently omitted, per the standing mapping-baselines rule. "
                "The kNN-retrieval read was cancelled by user directive for this "
                "round and is deliberately absent."
            ),
        },
        "splits": {
            "n": int(reg["n"]),
            "n_train": int(len(reg["tr"])),
            "n_val": int(len(reg["va"])),
            "n_holdout": int(len(reg["ho"])),
            "panel_n": int(len(reg["f_out"])),
        },
        "smoke": bool(args.smoke),
        "cells": cells,
    }
    _write_json(args.out / "summary.json", doc)
    logger.info("[fit] summary -> %s (%d cells)", args.out / "summary.json", len(cells))


# ── CLI ───────────────────────────────────────────────────────────────────────


def _import_check() -> int:
    """Resolve every deferred import on the REAL branch (#606/#1332)."""
    from huggingface_hub import HfApi  # noqa: F401

    from issue1738_sae_arm import _GramFactor  # noqa: F401

    import inspect

    sig = inspect.signature(_GramFactor.__init__)
    sig.bind(object(), object(), np.arange(2), torch.device("cpu"), 8)
    # FULL bind (never bind_partial) for every call whose argument list this file
    # supplies completely: bind_partial accepts a MISSING required kw-only arg, so
    # it green-lights exactly the TypeError it is meant to catch (#1332).
    inspect.signature(hub.verify_repo_paths_uploaded).bind(
        HfApi(), HF_DATA_REPO, ["a"], path_in_repo="p", repo_type="dataset"
    )
    inspect.signature(hub._upload).bind(
        Path("x"), repo_id=HF_DATA_REPO, repo_type="dataset", path_in_repo="p", upload_as_file=True
    )
    inspect.signature(hub.stage_hub_prefix).bind(
        HF_DATA_REPO, STORE_PREFIX, Path("d"), repo_type="dataset"
    )
    inspect.signature(hub.stage_hub_file).bind(HF_DATA_REPO, "f", Path("d"), repo_type="dataset")
    # retry_transient's `what` is keyword-ONLY and REQUIRED — a bind that omits it
    # is the #1332 arity class (import resolves, the call TypeErrors at run time).
    inspect.signature(hub.retry_transient).bind(lambda: None, what="probe")
    print("import-check OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--phase",
        required=False,
        choices=("upload-inputs", "stage", "assemble", "fit", "summary", "reliability"),
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--work", type=Path, default=Path("/workspace/issue1482_densesae"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--vm-dense-base", type=Path, default=VM_DENSE_BASE)
    ap.add_argument("--local-store", type=str, default="")
    ap.add_argument("--local-inputs", type=str, default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--gpu-id",
        default="",
        help="physical GPU (CVD-pinned by the launcher; informational here — the "
        "process sees its pinned device as cuda:0). A hand-launch MUST replicate "
        "the launcher's CUDA_VISIBLE_DEVICES=<g> env pin, not just pass this.",
    )
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train", type=int, default=600)
    ap.add_argument("--smoke-val", type=int, default=64)
    ap.add_argument("--smoke-holdout", type=int, default=128)
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--smoke-mlp-width", type=int, default=256)
    ap.add_argument("--hf-out-prefix", default=OUT_PREFIX)
    ap.add_argument("--xty-device", choices=("auto", "cusparse", "scipy"), default="auto")
    ap.add_argument("--verify-xty", action="store_true")
    ap.add_argument("--reliability-seed", type=int, default=1482)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        return _import_check()
    if not args.phase:
        raise SystemExit("--phase is required (or pass --import-check)")
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    fn = {
        "upload-inputs": phase_upload_inputs,
        "stage": phase_stage,
        "assemble": phase_assemble,
        "fit": phase_fit,
        "summary": phase_summary,
        "reliability": phase_reliability,
    }[args.phase]
    rc = fn(args)
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    # Explicit exit: heavy C-extension teardown can rewrite the return code during
    # interpreter finalization and abort a `set -e` dispatcher on a DONE phase.
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
