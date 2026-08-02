#!/usr/bin/env python3
"""Issue #1738 inline round — FULL-dictionary-width SAE-input maps + the SAE-basis
two-way (contexts x directions) error decomposition.

Two deliverables, both at the SAE suite's full ``dict_size`` (131,072) on the
ANSWER (output) side. The banked ``sae_arm`` fits restricted the answer side to
the top ``MAX_FEATURES_OUT = 16,384`` active features; this round removes that cap.

(a) ``--phase fit`` refits the three input arms (prefix / context / bare) at full
    output width and reports pooled + per-feature holdout R^2. The INPUT-side
    activity-floor restriction is KEPT verbatim (``issue1482_shuffle_null.restrict``
    at the 1%-of-fit-rows floor) -- it is what keeps the standardized design
    non-degenerate.

(b) ``--phase twoway`` runs ``issue1482_twoway_residual.two_way`` (imported
    VERBATIM) on the holdout squared-error table with SAE FEATURES as the
    directions -- the SAE-basis twin of the banked PC-basis result.

Why the banked restricted fit is a strict sub-read of the full-width fit: ridge
with a shared design solves each output column INDEPENDENTLY (``ymu``, ``X^T y``
and the spectral rescaling are all per-column), so at a FIXED lambda the
predictions for the banked 16,384 columns are the same whether or not the other
114,688 columns were fitted alongside. That identity is the reproduction gate in
``--phase fit``: the banked pooled R^2 must come back within ``REPRO_TOL``.

Reused rather than reimplemented: ``_scan_sae`` (activity scan), ``restrict``
(the f_in floor rule), ``_GramFactor`` (shared fp64 Gram + eigh), ``LAMBDAS``
(the parent 23-value grid), ``_bare_cache_paths`` / ``_assemble_bare`` /
``_build_bare_matrix`` (the bare SAE-encode + reorder), ``load_split`` +
``split_positions`` + the sha asserts, ``make_folds`` and ``two_way``.

Memory discipline (shared 125 GiB VM, earlyoom active, no GPU): the full-width
answer matrix never exists densely -- it is a memmap-backed CSR built with exact
preallocation from the pass-1 counts, and every solve streams in row blocks or
16,384-column output blocks.

Refusal-safety: chunk text fields are never printed/logged (digest-only).
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
import types
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: thread caps + credentials BEFORE numpy/torch import

import issue1482_shuffle_null as SN  # noqa: E402
import issue1738_multiturn_fits as MTF  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

# the reused kernels, imported VERBATIM (never reimplemented)
from issue1482_twoway_residual import make_folds, two_way  # noqa: E402
from issue1738_sae_arm import (  # noqa: E402
    LAMBDAS,
    MAX_FEATURES_IN,
    MAX_FEATURES_OUT,
    _assemble_bare,
    _bare_cache_paths,
    _build_bare_matrix,
    _GramFactor,
    _scan_sae,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1738.fullwidth")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
SAE_PREFIX = "issue1738_multiturn/sae_arm/capture"
BARE_PREFIX = "issue1738_multiturn/bare_query/capture"
MANIFEST_FILE = "issue1738_multiturn/sampling_manifest/split_1738.json"
PARENT_FITS = PROJECT_ROOT / "eval_results" / "issue_1738" / "fits" / "multiturn_100k_fits.json"
BANKED_SAE_FITS = (
    PROJECT_ROOT / "eval_results" / "issue_1738" / "bare_query" / "sae_arm" / "sae_fits.json"
)

STAGE = PROJECT_ROOT / "data" / "issue_1738" / "hf_dl"
WORK = PROJECT_ROOT / "data" / "issue_1738" / "fullwidth"
OUT = PROJECT_ROOT / "eval_results" / "issue_1738" / "sae_twoway"
FIGDIR = PROJECT_ROOT / "figures" / "issue_1738" / "sae_twoway"

ARMS = ("prefix", "context", "bare")
ARM_X = {"prefix": "px", "context": "cx", "bare": "bq"}
ARM_CELL = {"prefix": "sae_prefix", "context": "sae_context", "bare": "sae_bare"}
POOLING = "mean"  # the banked PRIMARY pooling; the max/frac twins are out of scope
VAL_KEY = "ans_mean"

OUT_BLOCK = 16_384  # output-column block for the ridge solve + R^2 accumulation
ROW_BLOCK = 2_048  # row block for streaming column statistics
SEED = 1482  # the PC-version fold seed, reused so the halves are the SAME contexts
K_GRID = (256, 1024, 4096, 16_384, 65_536)  # plus "all" (every nonzero-variance column)
REPRO_TOL = 1e-4  # |full-width restricted-subset R^2 - banked R^2| gate
DECILES = 10


# ── provenance / small utilities ──────────────────────────────────────────────


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _metadata() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "host": platform.node(),
        "fold_seed": SEED,
    }


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=float))
    os.replace(tmp, path)


def _avail_gib() -> float:
    """MemAvailable in GiB -- the number earlyoom watches."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024**2)
    raise RuntimeError("MemAvailable absent from /proc/meminfo")


# ── phase: stage ──────────────────────────────────────────────────────────────


def phase_stage(args) -> int:
    t0 = time.time()
    counts = {}
    for name, prefix in (("sae_arm", SAE_PREFIX), ("bare_query", BARE_PREFIX)):
        paths = hub.stage_hub_prefix(HF_DATA_REPO, prefix, STAGE, repo_type="dataset")
        counts[name] = len(paths)
        logger.info("[stage] %s: %d files under %s", name, len(paths), prefix)
    hub.stage_hub_file(HF_DATA_REPO, MANIFEST_FILE, STAGE / MANIFEST_FILE, repo_type="dataset")
    counts["split_manifest"] = 1
    _write_json(WORK / "stage_report.json", {"counts": counts, "wall_s": time.time() - t0})
    logger.info("[stage] done in %.1fs: %s", time.time() - t0, counts)
    return 0


def _chunk_paths(prefix: str) -> list[Path]:
    d = STAGE / prefix
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise SystemExit(f"no staged chunks under {d} — run --phase stage first")
    return paths


def _load_split() -> dict:
    split = MTF.load_split(STAGE / MANIFEST_FILE)
    MTF._assert_parent_split_shas(split, str(PARENT_FITS))
    return split


# ── phase: encode-bare (reuses the sae_arm encode path verbatim) ──────────────


def phase_encode_bare(args) -> int:
    """SAE-encode the parent's stored dense ``bq_last`` states.

    The bare arm has no banked SPARSE capture on the Hub: the parent bare-query
    round stored dense last-token states and the sae_arm round encoded them
    in-process. ``_bare_cache_paths`` (reused verbatim) does the encode + the
    per-chunk cache that is the resume unit; pointing ``local_bare_dir`` at the
    already-staged tree keeps it off the network.
    """
    bare_dir = STAGE / BARE_PREFIX
    names = sorted(p.name for p in bare_dir.glob("*.pt"))
    if not names:
        raise SystemExit(f"no staged bare chunks under {bare_dir}")
    first = torch.load(_chunk_paths(SAE_PREFIX)[0], map_location="cpu", weights_only=False)
    layer = int(first["layers"][0])
    del first
    shim = types.SimpleNamespace(
        no_resume=args.no_resume,
        smoke_model_dir=None,
        device="cpu",
        sae_cache=str(WORK / "sae_cache"),
        local_bare_dir=str(bare_dir),
        bare_hf_prefix="issue1738_multiturn/bare_query",
    )
    t0 = time.time()
    paths = _bare_cache_paths(shim, names, WORK / "bare_dl", WORK / "bare_feat", layer)
    logger.info(
        "[encode-bare] %d chunks ready at layer %d (%.0fs)", len(paths), layer, time.time() - t0
    )
    return 0


# ── full-width answer-side CSR (memmap-backed, exact preallocation) ───────────


class YStore:
    """Read-only full-width CSR over the memmapped answer-side matrix.

    Columns are the ENTIRE SAE dictionary: at full width nothing is dropped, so
    pass-1's all-row activity count is the exact nnz and the backing arrays are
    preallocated once rather than grown.
    """

    def __init__(self, work: Path, n_rows: int, dict_size: int, nnz: int):
        self.indices = np.memmap(work / "y_indices.i32", dtype=np.int32, mode="r", shape=(nnz,))
        self.data = np.memmap(work / "y_data.f32", dtype=np.float32, mode="r", shape=(nnz,))
        self.indptr = np.load(work / "y_indptr.npy")
        self.shape = (n_rows, dict_size)

    def _lens(self, rows: np.ndarray) -> np.ndarray:
        return self.indptr[rows + 1] - self.indptr[rows]

    def _take(self, rows: np.ndarray, lens: np.ndarray) -> np.ndarray:
        """Flat nnz positions of ``rows`` in stream order (the ragged-gather index)."""
        starts = self.indptr[rows]
        cum = np.concatenate(([0], np.cumsum(lens)[:-1]))
        return np.repeat(starts - cum, lens) + np.arange(int(lens.sum()), dtype=np.int64)

    def iter_blocks(self, rows: np.ndarray, block: int = ROW_BLOCK):
        """Yield (row_offset, local_row_index int32, column int32, value fp32) per block.

        Blocked so the int64 ``take`` index never scales with the WHOLE row set:
        a one-shot gather over the 87,794 train rows would allocate ~10 GiB of
        index temporaries on a VM where earlyoom is live. ``row_offset`` is the
        block's first position in ``rows`` (never inferred from ``loc``, which
        skips trailing all-zero rows).
        """
        for s in range(0, len(rows), block):
            rb = rows[s : s + block]
            lens = self._lens(rb)
            take = self._take(rb, lens)
            loc = np.repeat(np.arange(len(rb), dtype=np.int32), lens)
            yield s, loc, self.indices[take], self.data[take]

    def dense(self, rows: np.ndarray, c0: int = 0, c1: int | None = None) -> np.ndarray:
        """Dense fp32 ``Y[rows, c0:c1]`` (whole dictionary by default)."""
        c1 = self.shape[1] if c1 is None else c1
        out = np.zeros((len(rows), c1 - c0), dtype=np.float32)
        for s, loc, cols, vals in self.iter_blocks(rows):
            m = (cols >= c0) & (cols < c1)
            out[loc[m].astype(np.int64) + s, cols[m].astype(np.int64) - c0] = vals[m]
        return out

    def col_stats(self, rows: np.ndarray):
        """Per-column (sum, sumsq) over ``rows``, fp64, streamed in row blocks."""
        d = self.shape[1]
        s1 = np.zeros(d, dtype=np.float64)
        s2 = np.zeros(d, dtype=np.float64)
        for _s, _loc, cols, vals in self.iter_blocks(rows):
            c = cols.astype(np.int64)
            v = vals.astype(np.float64)
            s1 += np.bincount(c, weights=v, minlength=d)
            s2 += np.bincount(c, weights=v * v, minlength=d)
        return s1, s2

    def csr_rows(self, rows: np.ndarray) -> sp.csr_matrix:
        """A real scipy CSR of ``Y[rows]`` (for the sparse X^T Y accumulation)."""
        lens = self._lens(rows)
        total = int(lens.sum())
        ind = np.empty(total, dtype=np.int32)
        dat = np.empty(total, dtype=np.float32)
        pos = 0
        for _s, _loc, cols, vals in self.iter_blocks(rows):
            m = len(cols)
            ind[pos : pos + m] = cols
            dat[pos : pos + m] = vals
            pos += m
        assert pos == total, (pos, total)
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        return sp.csr_matrix((dat, ind, indptr), shape=(len(rows), self.shape[1]))


def _build_y_csr(paths: list[Path], n_rows: int, nnz_total: int, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    ind = np.memmap(work / "y_indices.i32.tmp", dtype=np.int32, mode="w+", shape=(nnz_total,))
    dat = np.memmap(work / "y_data.f32.tmp", dtype=np.float32, mode="w+", shape=(nnz_total,))
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    cur = row0 = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = len(d["ci"])
        rp = np.asarray(d["row_ptr"], dtype=np.int64)
        idx = np.asarray(d["feat_idx"], dtype=np.int64)
        val = np.asarray(d[VAL_KEY], dtype=np.float32)
        m = len(idx)
        ind[cur : cur + m] = idx.astype(np.int32)
        dat[cur : cur + m] = val
        indptr[row0 + 1 : row0 + 1 + n] = cur + rp[1:]
        cur += m
        row0 += n
        del d, idx, val
        if (i + 1) % 25 == 0 or (i + 1) == len(paths):
            logger.info(
                "[assemble] Y chunk %d/%d rows=%d nnz=%d elapsed=%.0fs",
                i + 1,
                len(paths),
                row0,
                cur,
                time.time() - t0,
            )
    assert row0 == n_rows, (row0, n_rows)
    assert cur == nnz_total, (cur, nnz_total)
    ind.flush()
    dat.flush()
    del ind, dat
    np.save(work / "y_indptr.npy", indptr)
    os.replace(work / "y_indices.i32.tmp", work / "y_indices.i32")
    os.replace(work / "y_data.f32.tmp", work / "y_data.f32")


def _build_x(paths, f_in, n_rows, dict_size) -> dict:
    """px/cx input CSR at the KEPT activity restriction, same stream row order.

    One chunk load serves BOTH arms (the fields live side by side in the same
    ``.pt``) — a per-arm pass would re-read all 224 chunks twice.
    """
    spec = (
        ("px", "px_feat_idx", "px_row_ptr", "px_feat_val"),
        ("cx", "cx_feat_idx", "cx_row_ptr", "cx_feat_val"),
    )
    col = {}
    acc: dict[str, list] = {}
    for arm, *_ in spec:
        c = np.full(dict_size, -1, dtype=np.int64)
        c[f_in[arm]] = np.arange(len(f_in[arm]))
        col[arm] = c
        acc[arm] = []
    row0 = 0
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = len(d["ci"])
        for arm, ik, pk, vk in spec:
            off = np.diff(np.asarray(d[pk], dtype=np.int64))
            c = col[arm][np.asarray(d[ik], dtype=np.int64)]
            keep = c >= 0
            acc[arm].append(
                (
                    np.repeat(np.arange(row0, row0 + n), off)[keep],
                    c[keep],
                    np.asarray(d[vk], dtype=np.float32)[keep],
                )
            )
        row0 += n
        del d
    assert row0 == n_rows, (row0, n_rows)
    out = {}
    for arm, *_ in spec:
        rr = np.concatenate([a[0] for a in acc[arm]])
        cc = np.concatenate([a[1] for a in acc[arm]])
        vv = np.concatenate([a[2] for a in acc[arm]])
        acc[arm].clear()
        out[arm] = sp.coo_matrix((vv, (rr, cc)), shape=(n_rows, len(f_in[arm]))).tocsr()
        logger.info("[assemble] X[%s]: %s nnz=%d", arm, out[arm].shape, out[arm].nnz)
    return out


# ── phase: assemble ───────────────────────────────────────────────────────────


def _assemble(args) -> dict:
    """Scan + restrict + build the full-width Y CSR and the per-arm input CSRs.

    Cached on disk: re-invocation short-circuits on the memmaps + npz already
    written (the resume unit), keyed on the chunk inventory.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    split = _load_split()
    train_ci = {int(c) for c in split["sets"]["train"]["ci"]}
    paths = _chunk_paths(SAE_PREFIX)

    cache = WORK / "scan.npz"
    if cache.exists() and not args.no_resume:
        z = np.load(cache, allow_pickle=False)
        scan = {
            "out_fit": z["out_fit"],
            "out_all": z["out_all"],
            "in_fit": {"px": z["in_fit_px"], "cx": z["in_fit_cx"]},
            "in_all": {"px": z["in_all_px"], "cx": z["in_all_cx"]},
            "n_fit": int(z["n_fit"]),
            "ci": z["ci"],
            "dropped": z["dropped"].tolist(),
            "dict_size": int(z["dict_size"]),
            "layer": int(z["layer"]),
        }
        logger.info("[assemble] scan: resumed from cache")
    else:
        first = torch.load(paths[0], map_location="cpu", weights_only=False)
        dict_size = int(first["sae"]["dict_size"])
        layer = int(first["layers"][0])
        del first
        t0 = time.time()
        scan = _scan_sae(paths, train_ci, dict_size)  # REUSED verbatim
        scan["dict_size"] = dict_size
        scan["layer"] = layer
        logger.info("[assemble] scan over %d chunks in %.0fs", len(paths), time.time() - t0)
        np.savez(
            cache,
            out_fit=scan["out_fit"],
            out_all=scan["out_all"],
            in_fit_px=scan["in_fit"]["px"],
            in_fit_cx=scan["in_fit"]["cx"],
            in_all_px=scan["in_all"]["px"],
            in_all_cx=scan["in_all"]["cx"],
            n_fit=scan["n_fit"],
            ci=scan["ci"],
            dropped=np.asarray(scan["dropped"], dtype=np.int64),
            dict_size=scan["dict_size"],
            layer=scan["layer"],
        )

    dict_size = scan["dict_size"]
    n_rows = len(scan["ci"])
    nnz_total = int(scan["out_all"].sum())

    # coverage assert (the parent's own S2.1 gate)
    parent = json.loads(PARENT_FITS.read_text())
    n_dropped = len(scan["dropped"])
    assert n_rows + n_dropped == int(parent["n_rows_captured"]), (
        n_rows,
        n_dropped,
        parent["n_rows_captured"],
    )

    f_in = {
        a: SN.restrict(scan["in_fit"][a], scan["n_fit"], MAX_FEATURES_IN)[0] for a in ("px", "cx")
    }
    f_out_banked, floor = SN.restrict(scan["out_fit"], scan["n_fit"], MAX_FEATURES_OUT)

    bare_meta = None
    if args.with_bare:
        bare_paths = sorted((WORK / "bare_feat").glob("*.pt"))
        if not bare_paths:
            raise SystemExit("no encoded bare chunks — run --phase encode-bare first")
        fit_mask = np.asarray([int(c) in train_ci for c in scan["ci"]], dtype=bool)
        bare, bare_meta = _assemble_bare(  # REUSED verbatim
            bare_paths, scan["ci"], dict_size, fit_mask, layer=scan["layer"]
        )
        scan["in_fit"]["bq"] = bare.pop("in_fit")
        scan["in_all"]["bq"] = bare.pop("in_all")
        f_in["bq"] = SN.restrict(scan["in_fit"]["bq"], scan["n_fit"], MAX_FEATURES_IN)[0]

    logger.info(
        "[assemble] n_rows=%d dict=%d n_fit=%d floor=%d | f_in %s | banked f_out=%d | nnz=%d (%.2f GB)",
        n_rows,
        dict_size,
        scan["n_fit"],
        floor,
        {a: len(v) for a, v in f_in.items()},
        len(f_out_banked),
        nnz_total,
        nnz_total * 8 / 1e9,
    )

    if not (WORK / "y_data.f32").exists() or args.no_resume:
        t0 = time.time()
        _build_y_csr(paths, n_rows, nnz_total, WORK)
        logger.info("[assemble] full-width Y CSR built in %.0fs", time.time() - t0)
    ystore = YStore(WORK, n_rows, dict_size, nnz_total)

    xcache = WORK / "x_pxcx.npz"
    if xcache.exists() and not args.no_resume:
        z = np.load(xcache)
        xmats = {
            a: sp.csr_matrix(
                (z[f"{a}_data"], z[f"{a}_indices"], z[f"{a}_indptr"]),
                shape=(n_rows, len(f_in[a])),
            )
            for a in ("px", "cx")
        }
        logger.info("[assemble] X[px]/X[cx]: resumed from cache")
    else:
        xmats = _build_x(paths, f_in, n_rows, dict_size)
        np.savez(
            xcache,
            **{
                f"{a}_{f}": getattr(xmats[a], f)
                for a in ("px", "cx")
                for f in ("data", "indices", "indptr")
            },
        )
    if args.with_bare:
        xmats["bq"], _dense = _build_bare_matrix(  # REUSED verbatim
            bare, f_in["bq"], WORK / "mm", 3584
        )
        del _dense
        logger.info("[assemble] X[bq]: %s nnz=%d", xmats["bq"].shape, xmats["bq"].nnz)

    sets = MTF.split_positions(split, scan["ci"])
    np.save(WORK / "holdout_ci.npy", scan["ci"][sets["holdout"]])

    stats_p = WORK / "ho_col_stats.npz"
    if stats_p.exists() and not args.no_resume:
        z = np.load(stats_p)
        ss_tot_ho, var_ho = z["ss_tot"], z["var"]
    else:
        t0 = time.time()
        s1, s2 = ystore.col_stats(sets["holdout"])
        n_ho = len(sets["holdout"])
        mu = s1 / n_ho
        ss_tot_ho = np.maximum(s2 - n_ho * mu**2, 0.0)
        var_ho = ss_tot_ho / max(1, n_ho - 1)
        np.savez(stats_p, ss_tot=ss_tot_ho, var=var_ho, mu=mu)
        logger.info("[assemble] holdout column stats in %.0fs", time.time() - t0)

    return {
        "split": split,
        "scan": scan,
        "sets": sets,
        "f_in": f_in,
        "f_out_banked": f_out_banked,
        "floor": floor,
        "ystore": ystore,
        "xmats": xmats,
        "ss_tot_ho": ss_tot_ho,
        "var_ho": var_ho,
        "bare_meta": bare_meta,
        "nnz_total": nnz_total,
    }


# ── phase: fit ────────────────────────────────────────────────────────────────


def phase_fit(args) -> int:
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    A = _assemble(args)
    ystore, sets = A["ystore"], A["sets"]
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    dict_size = A["scan"]["dict_size"]
    ss_tot_ho, var_ho = A["ss_tot_ho"], A["var_ho"]
    n_zero_var = int((ss_tot_ho <= 1e-12).sum())
    banked = json.loads(BANKED_SAE_FITS.read_text())
    # --arms is BINDING: it is what lets a second process fit the bare arm without
    # racing a live sibling over another arm's sqerr memmap + cell JSON.
    arms = [a for a in ARMS if a in set(args.arms) and (a != "bare" or args.with_bare)]
    logger.info("[fit] arms this run: %s", arms)

    if len(tr) < max(len(v) for v in A["f_in"].values()):
        raise SystemExit("n_train < d_in: estimator-degenerate regime, refusing the fit")

    logger.info(
        "[fit] holdout targets: %d/%d columns zero-variance, %d scored",
        n_zero_var,
        dict_size,
        dict_size - n_zero_var,
    )

    doc = {
        "metadata": _metadata(),
        "design": {
            "question": "Do the SAE-input history->answer maps behave the same at FULL "
            "dictionary width (131,072 answer features) as at the banked 16,384-feature "
            "restriction?",
            "method": "shared-Gram fp64 ridge (issue1738_sae_arm._GramFactor, reused): ONE "
            "input Gram + eigh per arm, output solved in 16,384-column blocks reusing the "
            "factorization. Lambda selected on the 396-row val split by pooled R^2 over ALL "
            "output columns; parent 23-value lambda grid verbatim. X^T Y accumulated "
            "sparsely against the full-width CSR (no dense answer matrix ever exists).",
            "output_width": {"banked": int(len(A["f_out_banked"])), "full": int(dict_size)},
            "input_restriction": "KEPT verbatim (issue1482_shuffle_null.restrict, 1%-of-fit-"
            "rows activity floor on INPUT columns) — it prevents the degenerate "
            "standardization that motivated it.",
            "pooling": POOLING,
            "arms": arms,
            "reproduction_gate": "the full-width fit restricted to the banked 16,384 columns, "
            "evaluated at the BANKED selected lambda, must reproduce the banked pooled "
            f"holdout R^2 within {REPRO_TOL} (ridge solves each output column independently).",
        },
        "split_counts": {k: int(len(v)) for k, v in sets.items()},
        "split_shas": {k: A["split"]["sets"][k]["sha256"] for k in A["split"]["sets"]},
        "restriction": {
            "activity_floor_rows": int(A["floor"]),
            "n_fit_rows": int(A["scan"]["n_fit"]),
            "n_f_out_banked": int(len(A["f_out_banked"])),
            "n_f_out_full": int(dict_size),
            **{f"n_f_in_{a}": int(len(v)) for a, v in A["f_in"].items()},
        },
        "holdout_targets": {
            "n_holdout": int(len(ho)),
            "n_zero_variance_columns": n_zero_var,
            "n_nonzero_variance_columns": int(dict_size - n_zero_var),
        },
        "answer_side_nnz": int(A["nnz_total"]),
        "bare_assembly": A["bare_meta"],
        "lambdas": [float(x) for x in LAMBDAS],
        "out_block": OUT_BLOCK,
        "pilot": {},
        "cells": {},
    }

    # activity deciles over the SCORED columns (the per-feature R^2 breakdown axis)
    act = A["scan"]["out_all"].astype(np.float64)
    scored = ss_tot_ho > 1e-12
    dec_edges = np.quantile(act[scored], np.linspace(0, 1, DECILES + 1))
    dec_id = np.clip(np.searchsorted(dec_edges, act, side="right") - 1, 0, DECILES - 1)

    y_val = torch.from_numpy(ystore.dense(val)).to(torch.float64)  # (396, 131072)
    logger.info(
        "[fit] val block densified: %s (%.2f GB)", tuple(y_val.shape), y_val.numel() * 8 / 1e9
    )

    for arm in arms:
        xk, cell = ARM_X[arm], ARM_CELL[arm]
        cellp = OUT / "cells" / f"{cell}.json"
        if cellp.exists() and not args.no_resume:
            doc["cells"][cell] = json.loads(cellp.read_text())
            # the MEASURED pilot rides the cell JSON so a resume-skip rebuild of the
            # combined doc keeps the sizing record instead of emitting an empty one
            if "pilot" in doc["cells"][cell]:
                doc["pilot"][arm] = doc["cells"][cell]["pilot"]
            logger.info("[fit] %s: resume-skip", cell)
            continue
        X = A["xmats"][xk]
        logger.info("[fit] === arm %s (X=%s, d_in=%d) ===", arm, xk, X.shape[1])

        t0 = time.time()
        fac = _GramFactor(X, tr, torch.device("cpu"), args.block)  # REUSED verbatim
        t_gram = time.time() - t0
        logger.info("[fit] %s: Gram+eigh %.0fs", arm, t_gram)

        # ymu: fp64 train column mean, streamed
        s1, _ = ystore.col_stats(tr)
        ymu = torch.from_numpy(s1 / len(tr))

        # X^T Y at full width, accumulated SPARSELY (cost ~ nnz * d_in, no densify)
        t0 = time.time()
        Xstd = fac.std_rows(tr).numpy()  # (n_tr, h) fp64
        Ytr = ystore.csr_rows(tr)
        xty = torch.from_numpy(np.ascontiguousarray((Ytr.T @ Xstd).T))  # (h, d) fp64
        del Ytr, Xstd
        xty -= torch.outer(fac.colsum, ymu)
        B = fac.U.T @ xty
        del xty
        t_xty = time.time() - t0
        logger.info("[fit] %s: X^T Y (full width) %.0fs -> B%s", arm, t_xty, tuple(B.shape))

        ev = fac.std_rows(val) @ fac.U
        eh = fac.std_rows(ho) @ fac.U
        n_blocks = (dict_size + OUT_BLOCK - 1) // OUT_BLOCK

        # ---- MEASURED 1-block pilot through the production path, then all blocks
        t0 = time.time()
        _ = _val_block_ss(y_val, ev, B, ymu, fac.s_eig, 0, min(OUT_BLOCK, dict_size))
        t_pilot = time.time() - t0
        logger.info(
            "[fit] %s: MEASURED 1-block val pilot %.1fs -> %d blocks projected %.0fs",
            arm,
            t_pilot,
            n_blocks,
            t_pilot * n_blocks,
        )
        doc["pilot"][arm] = {
            "one_block_val_wall_s": t_pilot,
            "n_blocks": n_blocks,
            "projected_val_sweep_wall_s": t_pilot * n_blocks,
        }

        t0 = time.time()
        ssr = np.zeros(len(LAMBDAS))
        sst = 0.0
        for bi, c0 in enumerate(range(0, dict_size, OUT_BLOCK)):
            c1 = min(c0 + OUT_BLOCK, dict_size)
            r, t = _val_block_ss(y_val, ev, B, ymu, fac.s_eig, c0, c1)
            ssr += r
            sst += t
            logger.info("[fit] %s val block %d/%d (%.0fs)", arm, bi + 1, n_blocks, time.time() - t0)
        val_r2 = 1.0 - ssr / sst
        best = int(np.nanargmax(val_r2))
        sel_lam = float(LAMBDAS[best])
        t_val = time.time() - t0
        logger.info(
            "[fit] %s: val sweep %.0fs -> lambda=%.6g val_R2=%.6f",
            arm,
            t_val,
            sel_lam,
            val_r2[best],
        )

        banked_lam = float(banked["cells"][cell]["selected_lambda"])
        lams = {"selected": sel_lam}
        if abs(banked_lam - sel_lam) > 1e-30:
            lams["banked_lambda"] = banked_lam
        res = {}
        for tag, lam in lams.items():
            t0 = time.time()
            res[tag] = _holdout_pass(
                ystore,
                eh,
                B,
                ymu,
                fac.s_eig,
                lam,
                ho,
                ss_tot_ho,
                A["f_out_banked"],
                dict_size,
                n_blocks,
                arm,
                sqerr_path=(WORK / f"sqerr_{arm}.f32" if tag == "selected" else None),
                perfeat_path=(OUT / "perfeature" / f"{cell}_r2.npy" if tag == "selected" else None),
                dec_id=dec_id,
                scored=scored,
            )
            res[tag]["wall_s"] = time.time() - t0
            logger.info(
                "[fit] %s (%s lam=%.6g): FULL R2=%.6f | banked-subset R2=%.6f (%.0fs)",
                arm,
                tag,
                lam,
                res[tag]["pooled_r2_full"],
                res[tag]["pooled_r2_banked_subset"],
                res[tag]["wall_s"],
            )
        if "banked_lambda" not in res:
            res["banked_lambda"] = res["selected"]

        banked_r2 = float(banked["cells"][cell]["holdout_r2"])
        got = res["banked_lambda"]["pooled_r2_banked_subset"]
        delta = got - banked_r2
        entry = {
            "arm": arm,
            "x_source": f"{xk}_feat",
            "pooling": POOLING,
            "d_in": int(X.shape[1]),
            "selected_lambda_fullwidth": sel_lam,
            "val_r2_selected_fullwidth": float(val_r2[best]),
            "val_r2_by_lambda": {
                str(float(x)): float(y) for x, y in zip(LAMBDAS, val_r2, strict=True)
            },
            "banked_selected_lambda": banked_lam,
            "banked_pooled_r2_restricted": banked_r2,
            "full_width": res["selected"],
            "at_banked_lambda": res["banked_lambda"],
            "reproduction_gate": {
                "banked_pooled_r2": banked_r2,
                "fullwidth_restricted_to_banked_columns_r2": got,
                "delta": delta,
                "tol": REPRO_TOL,
                "pass": bool(abs(delta) < REPRO_TOL),
            },
            "wall_s": {"gram_eigh": t_gram, "xty": t_xty, "val_sweep": t_val},
            "pilot": doc["pilot"][arm],
        }
        doc["cells"][cell] = entry
        _write_json(cellp, entry)
        _write_json(OUT / "fullwidth_fits.json", doc)
        if abs(delta) >= REPRO_TOL:
            logger.error(
                "[fit] %s REPRODUCTION GATE FAILED: banked %.6f vs full-width-restricted %.6f "
                "(delta %.3e >= %.3e) — assembly bug, halting before the remaining arms",
                arm,
                banked_r2,
                got,
                delta,
                REPRO_TOL,
            )
            return 3
        logger.info("[fit] %s reproduction gate PASS (delta %.3e)", arm, delta)
        del fac, B, ev, eh

    doc["total_wall_s"] = time.time() - t_all
    _write_json(OUT / "fullwidth_fits.json", doc)
    logger.info("[fit] done in %.0fs -> %s", doc["total_wall_s"], OUT / "fullwidth_fits.json")
    return 0


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


def _holdout_pass(
    ystore,
    eh,
    B,
    ymu,
    s_eig,
    lam,
    ho,
    ss_tot_ho,
    f_out_banked,
    dict_size,
    n_blocks,
    arm,
    sqerr_path,
    perfeat_path,
    dec_id,
    scored,
):
    """One holdout pass at ``lam``: pooled + per-feature R^2 and (optionally) the
    (n_holdout, dict_size) squared-error table the two-way decomposition reads."""
    inv = 1.0 / (s_eig + float(lam))
    ss_res = np.zeros(dict_size, dtype=np.float64)
    n_ho = len(ho)
    mm = None
    if sqerr_path is not None:
        mm = np.memmap(sqerr_path, dtype=np.float32, mode="w+", shape=(n_ho, dict_size))
    t0 = time.time()
    for bi, c0 in enumerate(range(0, dict_size, OUT_BLOCK)):
        c1 = min(c0 + OUT_BLOCK, dict_size)
        yb = torch.from_numpy(ystore.dense(ho, c0, c1)).to(torch.float64)
        pred = (eh * inv) @ B[:, c0:c1] + ymu[c0:c1]
        e2 = (yb - pred) ** 2
        ss_res[c0:c1] = e2.sum(0).numpy()
        if mm is not None:
            mm[:, c0:c1] = e2.numpy().astype(np.float32)
        del yb, pred, e2
        logger.info("[fit] %s holdout block %d/%d (%.0fs)", arm, bi + 1, n_blocks, time.time() - t0)
    if mm is not None:
        mm.flush()
        del mm
    ok = ss_tot_ho > 1e-12
    per_feat = np.full(dict_size, np.nan)
    per_feat[ok] = 1.0 - ss_res[ok] / ss_tot_ho[ok]
    if perfeat_path is not None:
        perfeat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(perfeat_path, per_feat.astype(np.float32))
    fin = per_feat[ok]
    sub = f_out_banked
    by_decile = []
    for dd in range(DECILES):
        m = scored & (dec_id == dd)
        if not m.any():
            continue
        v = per_feat[m]
        by_decile.append(
            {
                "decile": dd,
                "n_features": int(m.sum()),
                "median_r2": float(np.median(v)),
                "frac_below_zero": float((v < 0).mean()),
            }
        )
    return {
        "lambda": float(lam),
        "pooled_r2_full": float(1.0 - ss_res.sum() / ss_tot_ho.sum()),
        "pooled_r2_banked_subset": float(1.0 - ss_res[sub].sum() / ss_tot_ho[sub].sum()),
        "n_features_scored": int(ok.sum()),
        "per_feature_r2_median": float(np.median(fin)),
        "per_feature_r2_mean": float(fin.mean()),
        "per_feature_frac_below_zero": float((fin < 0).mean()),
        "per_feature_r2_quantiles": {
            str(q): float(np.quantile(fin, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "per_feature_r2_by_activity_decile": by_decile,
    }


# ── phase: twoway ─────────────────────────────────────────────────────────────


def _select_columns(var_ho: np.ndarray, k: int | None) -> np.ndarray:
    """Top-k output columns by holdout target variance; zero-variance excluded."""
    nz = np.where(var_ho > 1e-12)[0]
    if k is None or k >= len(nz):
        return nz
    return np.sort(nz[np.argsort(-var_ho[nz], kind="stable")[:k]])


def phase_twoway(args) -> int:
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(WORK / "ho_col_stats.npz")
    var_ho = z["var"]
    ho_ci = np.load(WORK / "holdout_ci.npy")
    n_ho, dict_size = len(ho_ci), len(var_ho)
    nz_cols = _select_columns(var_ho, None)
    folds = make_folds(n_ho, SEED)  # REUSED verbatim — the same halves as the PC version

    scan = np.load(WORK / "scan.npz")
    ystore = YStore(WORK, len(scan["ci"]), dict_size, int(scan["out_all"].sum()))
    split = _load_split()
    ho_rows = MTF.split_positions(split, scan["ci"])["holdout"]

    # cross-fit denominators: per-feature TARGET variance on each split half, at
    # full width, computed ONCE (streamed) rather than re-densified per k.
    fold_var = []
    for var_idx, _eval_idx in folds:
        rows = ho_rows[var_idx]
        s1, s2 = ystore.col_stats(rows)
        m = len(rows)
        fold_var.append(np.maximum(s2 - m * (s1 / m) ** 2, 0.0) / (m - 1))
    logger.info("[twoway] cross-fit variance denominators ready (2 halves, full width)")

    doc = {
        "metadata": _metadata(),
        "design": {
            "question": "Does the SAE-feature-space history->answer map fail at specific "
            "DIRECTIONS (SAE features) or specific CONTEXTS? Two-way decomposition of the "
            "held-out squared residual, mirroring the PC-basis result with SAE FEATURES as "
            "the directions.",
            "model": "R[i, j] = mu + a_i (context) + b_j (direction) + e_ij (interaction)",
            "kernel": "issue1482_twoway_residual.two_way, imported VERBATIM (asserts SS "
            "closure; returns EMS variance components + F statistics)",
            "basis": "the SAE dictionary itself — a FIXED basis, so unlike the PC version "
            "there is no basis to fit. Cross-fitting therefore applies to the NORMALIZATION "
            "denominator only: the per-feature target variance is computed on one split half "
            "of the holdout contexts and applied to the OTHER half; both assignments are run "
            "and the shares averaged (the PC version's out-of-sample discipline, same fold "
            "seed, same halves, and — as there — each two_way runs on one eval half).",
            "normalizations": [
                "raw squared residual",
                "squared residual / cross-fitted per-feature holdout target variance",
            ],
            "primary_read": "EMS variance components (vc_share_*); raw SS shares carry a "
            "pure-noise floor from the (n, k) geometry and are reported for transparency.",
            "k_grid": [*[int(k) for k in K_GRID], "all"],
            "k_selection": "output columns ranked by holdout target variance, descending; "
            "zero-target-variance columns excluded from every read.",
            "corpus": f"#1738 multi-turn 100k real conversations, {n_ho} holdout contexts, "
            f"SAE dict_size {dict_size} (layer 19, k=64).",
        },
        "n_holdout": int(n_ho),
        "n_per_fold": int(folds[0][1].size),
        "n_nonzero_variance_columns": int(len(nz_cols)),
        "n_zero_variance_columns": int(dict_size - len(nz_cols)),
        "cells": {},
    }
    # merge-on-resume: an arm already swept in an earlier invocation is carried
    # forward verbatim, so a later --arms run never silently drops it from the doc.
    prior = OUT / "sae_twoway.json"
    if prior.exists() and not args.no_resume:
        carried = json.loads(prior.read_text()).get("cells", {})
        doc["cells"] = {a: v for a, v in carried.items() if a not in set(args.arms)}
        if doc["cells"]:
            logger.info("[twoway] carried forward from prior run: %s", sorted(doc["cells"]))

    for arm in args.arms:
        path = WORK / f"sqerr_{arm}.f32"
        if not path.exists():
            logger.info("[twoway] skip %s (no %s)", arm, path.name)
            continue
        R = np.memmap(path, dtype=np.float32, mode="r", shape=(n_ho, dict_size))
        by_k: dict[str, dict] = {}
        for k in [*K_GRID, None]:
            kk = "all" if k is None else str(k)
            cols = _select_columns(var_ho, k)
            need = folds[0][1].size * len(cols) * 8 * 3.5 / 1024**3
            avail = _avail_gib()
            if need > avail - args.mem_headroom_gib:
                logger.warning(
                    "[twoway] %s k=%s SKIPPED: needs ~%.1f GiB, MemAvailable %.1f GiB",
                    arm,
                    kk,
                    need,
                    avail,
                )
                by_k[kk] = {
                    "skipped": "insufficient_memory",
                    "need_gib": need,
                    "avail_gib": avail,
                    "n_cols_selected": int(len(cols)),
                }
                continue
            t0 = time.time()
            per_fold = []
            for fi, (_var_idx, eval_idx) in enumerate(folds):
                Rf = np.asarray(R[np.ix_(eval_idx, cols)], dtype=np.float64)
                raw = two_way(Rf)  # VERBATIM
                v = fold_var[fi][cols]  # denominator from the OTHER half
                keep = v > 1e-12
                try:
                    nrm = two_way(Rf[:, keep] / v[keep][None, :])  # VERBATIM
                except (AssertionError, ValueError) as exc:
                    # two_way's own SS-closure / degeneracy guard. A rare feature
                    # with near-zero cross-fit variance blows the normalized table's
                    # dynamic range; record the refusal rather than swallow it.
                    logger.error(
                        "[twoway] %s k=%s fold%d: normalized two_way REFUSED (%s) — "
                        "min cross-fit variance %.3e over %d kept columns",
                        arm,
                        kk,
                        fi,
                        exc,
                        float(v[keep].min()) if keep.any() else float("nan"),
                        int(keep.sum()),
                    )
                    nrm = {"closure_or_degeneracy_error": str(exc)}
                per_fold.append(
                    {"raw": raw, "normalized": nrm, "n_cols_normalized": int(keep.sum())}
                )
                del Rf
            merged = {}
            for norm in ("raw", "normalized"):
                if any("closure_or_degeneracy_error" in pf[norm] for pf in per_fold):
                    merged[norm] = {
                        "refused": [pf[norm].get("closure_or_degeneracy_error") for pf in per_fold]
                    }
                    continue
                merged[norm] = {
                    f: (
                        float(np.mean([pf[norm][f] for pf in per_fold]))
                        if isinstance(per_fold[0][norm][f], float)
                        else per_fold[0][norm][f]
                    )
                    for f in per_fold[0][norm]
                }
            by_k[kk] = {
                "n_cols_selected": int(len(cols)),
                "wall_s": time.time() - t0,
                **merged,
                "per_fold": per_fold,
            }
            fmt = lambda blk: (
                "REFUSED"
                if "refused" in blk
                else "%.4f/%.4f/%.4f"
                % (
                    blk["vc_share_context"],
                    blk["vc_share_direction"],
                    blk["vc_share_interaction"],
                )
            )
            logger.info(
                "[twoway] %s k=%-6s raw c/d/i %s | norm c/d/i %s (%.0fs)",
                arm,
                kk,
                fmt(merged["raw"]),
                fmt(merged["normalized"]),
                by_k[kk]["wall_s"],
            )
            doc["cells"][arm] = {"arm": arm, "by_k": by_k}
            _write_json(OUT / "sae_twoway.json", doc)
        doc["cells"][arm] = {"arm": arm, "by_k": by_k}
        del R
        _write_json(OUT / "sae_twoway.json", doc)

    doc["total_wall_s"] = time.time() - t_all
    _write_json(OUT / "sae_twoway.json", doc)
    logger.info("[twoway] done in %.0fs -> %s", doc["total_wall_s"], OUT / "sae_twoway.json")
    return 0


# ── phase: figure ─────────────────────────────────────────────────────────────

COMPONENTS = (
    ("vc_share_context", "Contexts (rows)"),
    ("vc_share_direction", "Directions (SAE features)"),
    ("vc_share_interaction", "Interaction (context x feature)"),
)
ARM_LABEL = {"prefix": "Prefix end state", "context": "Context vector", "bare": "Query only"}


def phase_figure(args) -> int:
    """Stacked variance-component bars, mirroring the PC-basis Result-1 figure."""
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    doc = json.loads((OUT / "sae_twoway.json").read_text())
    cells = doc["cells"]
    kk = args.k
    order = [a for a in ("context", "prefix", "bare") if a in cells]

    def _shares(b: dict) -> dict | None:
        """The vc_share block of a k cell, or None when it was skipped/refused."""
        n = b.get(args.normalization)
        return n if isinstance(n, dict) and "vc_share_interaction" in n else None

    cols = []
    for arm in order:
        by_k = cells[arm]["by_k"]
        blk = _shares(by_k.get(kk, {}))
        if blk is None:
            raise KeyError(
                f"{arm}: k={kk} {args.normalization} absent/skipped/refused; have {sorted(by_k)}"
            )
        vals = [float(blk[f]) for f, _ in COMPONENTS]
        total = sum(vals)
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"{arm}: shares sum to {total}, not 1")
        cols.append((ARM_LABEL[arm], vals))

    # k-stability note: the spread of the interaction share across the whole grid
    spread = []
    for arm in order:
        for b in cells[arm]["by_k"].values():
            s = _shares(b)
            if s is not None:
                spread.append(float(s["vc_share_interaction"]))
    ks = sorted(
        (k for k, b in cells[order[0]]["by_k"].items() if _shares(b) is not None),
        key=lambda s: (s == "all", int(s) if s != "all" else 0),
    )
    note = (
        f"k-stable: interaction share {min(spread):.2f}-{max(spread):.2f} across "
        f"k = {', '.join(ks)} (directions ranked by holdout target variance)"
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = list(range(len(cols)))
    bottoms = [0.0] * len(cols)
    for ci, (_f, clabel) in enumerate(COMPONENTS):
        vals = [c[1][ci] for c in cols]
        ax.bar(x, vals, bottom=bottoms, color=colors[ci], label=clabel, width=0.58)
        for xi, v, b in zip(x, vals, bottoms, strict=True):
            if v >= 0.04:
                ax.text(
                    xi,
                    b + v / 2,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if ci == 2 else "black",
                    fontweight="semibold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cols], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of the map's held-out error")
    ax.set_title(
        f"Where the SAE-space map's error lives (directions = SAE features, k = {kk}, "
        f"{args.normalization} squared residual)",
        loc="left",
        fontweight="semibold",
        pad=26,
    )
    ax.text(0.0, 1.012, note, transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, args.name, dir=FIGDIR)
    logger.info("[figure] wrote %s/%s.png — %s", FIGDIR, args.name, note)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase", required=True, choices=("stage", "encode-bare", "fit", "twoway", "figure")
    )
    ap.add_argument("--k", default="4096", help="representative k for --phase figure")
    ap.add_argument(
        "--normalization",
        default="raw",
        choices=("raw", "normalized"),
        help="which two-way table --phase figure plots",
    )
    ap.add_argument("--name", default="sae_twoway_variance_components")
    ap.add_argument("--block", type=int, default=8192, help="row block for the Gram pass")
    ap.add_argument("--with-bare", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--mem-headroom-gib", type=float, default=10.0)
    args = ap.parse_args()
    return {
        "stage": phase_stage,
        "encode-bare": phase_encode_bare,
        "fit": phase_fit,
        "twoway": phase_twoway,
        "figure": phase_figure,
    }[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
