"""Issue #1689 free-analysis follow-up — per-pair direction-aware Procrustes battery.

Pre-registered by the epm:progress v67 addendum (item 1b): for every ordered
pair in the ladder's 126-pair set (both mapping arms, both models, L19), the
DATA-PAIRED Procrustes-aligned operator cosine between the two cells'
within-cell context->answer ridge maps — the #1345 leg_b battery statistic
(``issue825_map_alignment._procrustes_cosine_null`` convention, also the #1310
xpersona anchor convention) — plus its random-orthogonal-rotation null band.
This is the CONTINUOUS similarity read the ordinal rung ladder lacks.

Recipe (provenance):
- Per-cell maps: within-cell ridge Y ~ X @ W + b at L19, fit on ALL cell rows
  with the SAME recipe the ladder used — inner-group-cv lambda selection over
  the committed LAMBDAS grid with conv-grouped inner folds, torch engine on
  CPU (``issue1689_fit_ladder._fit_ridge_inner_group_cv_t``). One map per
  (model, condition, arm); 21 x 2 x 2 = 84 maps. NOTE the ladder fits its
  per-PAIR source maps on the row-paired subset; this battery deliberately
  fits per-CELL maps once on the full cell rows (the v67 item-1b framing:
  "the two cells' maps") — cells share >=~95% of rows by construction.
- Aligned cosine per pair: align on the shared row-paired conv_id
  intersection (``np.intersect1d`` + first-occurrence indices — the ladder's
  row-pairing, fit_ladder L964-974). R_in = orth(Xa_c, Xb_c),
  R_out = orth(Ya_c, Yb_c) with orth(A, B) = U @ Vh from svd(A.T @ B) — the
  literal ``ma._procrustes_cosine_null`` `_orth`. Observed
  aligned_cosine = cos(vec(R_in.T @ W_a @ R_out), vec(W_b)).
  SYMMETRY: because R(a->b) = R(b->a).T and rotations preserve Frobenius
  norm, aligned/raw/spectrum cosines and the rotation-null distribution are
  EXACTLY symmetric under pair order — computed once per unordered pair and
  emitted on both ordered rows (documented in metadata).
  RANK NOTE: n_common < d here (~1.7k rows vs d=3584), so the Procrustes
  rotation beyond rank(A.T @ B) is the LAPACK SVD's orthogonal completion —
  the same convention the #1310/#825 anchors used; flagged in metadata.
- Rotation null, BATCHED (vectorize-many-cell-fits rule): the null draw
  cos(vec(Q1.T @ W_a @ Q2), vec(W_b)) over independent Haar Q1, Q2 depends on
  (W_a, W_b) only through their singular values:
      trace(W_b.T Q1.T W_a Q2) = trace(S_b P S_a R) = s_b.T (P * R.T) s_a
  with P = U_b.T Q1.T U_a ~ Haar, R = Vh_a Q2 V_b ~ Haar (Haar invariance),
  independent. So ONE bank of K Haar pairs (P_k, R_k) serves EVERY pair x arm
  x model: per draw, vals_k = S E_k S.T with E_k = P_k * R_k.T and S the
  (n_cells, d) stack of per-map singular values — one GEMM for all pairs, no
  per-pair rotation loop. Exactness of the algebra is asserted by
  ``--verify-null-equivalence`` (same-(Q1,Q2) identity check < 1e-9 plus a
  distributional check vs the serial #1345 form), and ``--serial-null-check``
  re-checks at production shape.

Outputs:
- eval_results/issue_1689/procrustes/battery_<model_tag>_L19.json — per
  ordered pair x arm: aligned/raw/spectrum cosine, null band (mean/std/
  p2.5/p97.5), n_common, screened flag (pair_digest arm_invalid semantics —
  computed anyway, flagged, never dropped), lambdas, pair class.
- eval_results/issue_1689/procrustes/summary_L19.json — per (model, arm,
  class) aggregates over unscreened pairs.

Observability + resume: flushed per-unit progress prints; per-cell map/svec
checkpoints (skip-if-exists keyed on the fit regime), meta-hashed null-bank
checkpoints (per-draw seeds -> any partial bank resumes deterministically),
per-pair JSONL append with resume-by-key. Deterministic seed (default 42).

Full-run one-command launch (detached VM form; orchestrator owns the watch):
  setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python \\
    scripts/issue1689_procrustes_battery.py --null-draws 200 \\
    < /dev/null >> /tmp/issue1689_procrustes.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing numpy / torch — shared-VM thread
# caps (#847) freeze at first BLAS/torch import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_analyzer_digest import (  # noqa: E402
    arm_is_invalid,
    classify,
    parse_slug,
)
from scripts.issue1689_common import (  # noqa: E402
    CONDITION_TABLE,
    HEADLINE_LAYER,
    HF_DATA_PREFIX,
    LAMBDA_LOG_MAX,
    LAMBDA_LOG_MIN,
    enumerate_pair_set,
)
from scripts.issue1689_fit_ladder import (  # noqa: E402
    LAMBDAS,
    _fit_ridge_inner_group_cv_t,
    _load_cell_layer,
)

DATA_REPO = "superkaiba1/explore-persona-space-data"
MODELS: dict[str, str] = {  # short -> model_tag (HF store folder / ladder slug)
    "base": "Qwen_Qwen2.5-7B",
    "instruct": "Qwen_Qwen2.5-7B-Instruct",
}
ARMS = ("prefix", "context")
DEFAULT_WORK_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue1689_procrustes")
DEFAULT_OUT_DIR = REPO_ROOT / "eval_results/issue_1689/procrustes"
MIN_N_COMMON = 3  # ladder's row-pairing floor (fit_ladder L966)
MIN_QUANTILE_DRAWS = 8  # p2.5/p97.5 floor
ENGINE_TAG = "torch-cpu-v1"  # bump on any output-affecting fit/battery change


# ---------------------------------------------------------------------------
# Small atomic-IO helpers
# ---------------------------------------------------------------------------
def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """np.save with tmp+replace; tmp name keeps the .npy suffix (np.save appends
    .npy to any other name — the np.savez gotcha's sibling)."""
    assert path.suffix == ".npy", path
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _metadata(args: argparse.Namespace) -> dict:
    return {
        "script": "scripts/issue1689_procrustes_battery.py",
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "layer": args.layer,
        "n_null_draws": args.null_draws,
        "engine": ENGINE_TAG,
        "lambda_grid": {"log_min": LAMBDA_LOG_MIN, "log_max": LAMBDA_LOG_MAX, "size": len(LAMBDAS)},
        "fit_convention": (
            "per-CELL ridge (Y ~ X @ W + b) on ALL cell rows; inner-group-cv lambda over the"
            " ladder LAMBDAS grid with conv-grouped inner folds"
            " (issue1689_fit_ladder._fit_ridge_inner_group_cv_t); bias excluded from cosines"
        ),
        "alignment_convention": (
            "data-paired orthogonal Procrustes on the pair's conv_id intersection"
            " (ma._procrustes_cosine_null / #1310 anchor convention: orth(A,B) = U @ Vh of"
            " svd(A.T @ B)); n_common < d, so the rotation beyond rank(A.T @ B) is the LAPACK"
            " SVD's orthogonal completion"
        ),
        "symmetry_note": (
            "aligned/raw/spectrum cosines + rotation null are exactly symmetric under pair"
            " order; ordered rows (src__tgt and tgt__src) carry identical values by"
            " construction"
        ),
        "null_convention": (
            "two-sided Haar rotation null, batched via the singular-value reduction"
            " trace(W_b^T Q1^T W_a Q2) = s_b^T (P * R^T) s_a with P, R ~ Haar; per-draw"
            " seeds seed*1000003+k"
        ),
    }


# ---------------------------------------------------------------------------
# Staging (scoped per-file downloads; NEVER a repo snapshot — ~1M-file repo)
# ---------------------------------------------------------------------------
def stage_cells(model_tags: list[str], conds: list[str], stage_root: Path, layer: int) -> None:
    from explore_persona_space.orchestrate.hub import stage_hub_file

    todo = [(m, c) for m in model_tags for c in conds]
    for i, (m, c) in enumerate(todo, 1):
        target = stage_root / m / c / f"L{layer}.pt"
        if target.exists() and target.stat().st_size > 0:
            print(f"[stage] {i}/{len(todo)} {m}/{c} L{layer}.pt cached", flush=True)
            continue
        t0 = time.time()
        stage_hub_file(
            DATA_REPO,
            f"{HF_DATA_PREFIX}/analysis_tensors/{m}/{c}/L{layer}.pt",
            target,
            repo_type="dataset",
        )
        print(
            f"[stage] {i}/{len(todo)} {m}/{c} L{layer}.pt downloaded"
            f" ({target.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.1f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Linear algebra primitives
# ---------------------------------------------------------------------------
def _haar(d: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-orthogonal (d, d) fp64 sample (ma._random_orthogonal convention:
    CPU randn from the caller's generator, QR, diag-sign fix)."""
    a = torch.randn(d, d, dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def _svd_robust(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch svd with scipy-gesvd fallback on LAPACK gesdd non-convergence
    (the numpy-SVD-on-degenerate-input class; #722 r3 memory)."""
    try:
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        return u, s, vh
    except torch.linalg.LinAlgError:
        print("[battery] svd: gesdd non-convergence -> scipy gesvd fallback", flush=True)
        import scipy.linalg

        u_, s_, vh_ = scipy.linalg.svd(m.numpy(), full_matrices=False, lapack_driver="gesvd")
        return (
            torch.from_numpy(u_).to(torch.float64),
            torch.from_numpy(s_).to(torch.float64),
            torch.from_numpy(vh_).to(torch.float64),
        )


def _orth(a_ctr: torch.Tensor, b_ctr: torch.Tensor) -> torch.Tensor:
    """R minimizing ||A R - B||, R orthogonal — replicates
    ma._procrustes_cosine_null's internal `_orth` (U @ Vh of svd(A.T @ B))."""
    m = a_ctr.T @ b_ctr
    u, _s, vh = _svd_robust(m)
    return u @ vh


def _cos_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    va, vb = a.reshape(-1), b.reshape(-1)
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


def _pair_rows(conv_a: np.ndarray, conv_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Row-pair two cells by conv_id — the ladder's recipe (fit_ladder L964-974)."""
    common = np.intersect1d(conv_a, conv_b)
    if len(common) < MIN_N_COMMON:
        return np.array([], dtype=int), np.array([], dtype=int), int(len(common))
    a_idx = np.array([np.where(conv_a == c)[0][0] for c in common])
    b_idx = np.array([np.where(conv_b == c)[0][0] for c in common])
    return a_idx, b_idx, int(len(common))


# ---------------------------------------------------------------------------
# Phase A — per-cell ridge maps (+ singular values), checkpointed
# ---------------------------------------------------------------------------
def _fit_regime(args: argparse.Namespace) -> dict:
    return {
        "layer": args.layer,
        "seed": args.seed,
        "engine": ENGINE_TAG,
        "lambda_grid": [float(LAMBDAS[0]), float(LAMBDAS[-1]), len(LAMBDAS)],
    }


def fit_cell_maps(
    model_tags: list[str],
    conds: list[str],
    arms: tuple[str, ...],
    store_root: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], dict]:
    """Fit (or resume) all per-cell maps. Returns {key: meta} with key =
    (model_tag, cond, arm); W/svec live on disk under work_dir."""
    regime = _fit_regime(args)
    out: dict[tuple[str, str, str], dict] = {}
    units = [(m, c, a) for m in model_tags for c in conds for a in arms]
    for k, (m, c, a) in enumerate(units, 1):
        map_dir = work_dir / "maps" / m
        svec_dir = work_dir / "svecs" / m
        map_dir.mkdir(parents=True, exist_ok=True)
        svec_dir.mkdir(parents=True, exist_ok=True)
        w_path = map_dir / f"{c}__{a}.npy"
        s_path = svec_dir / f"{c}__{a}.npy"
        meta_path = map_dir / f"{c}__{a}.meta.json"
        if w_path.exists() and s_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("regime") == regime:
                out[(m, c, a)] = meta
                print(f"[fit] unit {k}/{len(units)} {m}/{c}/{a} cached", flush=True)
                continue
            raise RuntimeError(
                f"fit regime mismatch for {w_path}: cached {meta.get('regime')} !="
                f" {regime}; use a fresh --work-dir"
            )
        t0 = time.time()
        bundle = _load_cell_layer(store_root, f"{m}/{c}", args.layer)
        x = torch.from_numpy(bundle[f"X_{a}"])
        y = torch.from_numpy(bundle["Y"])
        assert x.shape == y.shape, (x.shape, y.shape)
        w, _b, lam = _fit_ridge_inner_group_cv_t(x, y, bundle["conv_ids"], LAMBDAS, seed=args.seed)
        svals = torch.linalg.svdvals(w)
        _atomic_save_npy(w_path, w.numpy())
        _atomic_save_npy(s_path, svals.numpy())
        meta = {
            "regime": regime,
            "lambda": float(lam),
            "n_rows": int(x.shape[0]),
            "d": int(x.shape[1]),
            "fit_wall_s": round(time.time() - t0, 2),
        }
        _atomic_write_json(meta_path, meta)
        out[(m, c, a)] = meta
        print(
            f"[fit] unit {k}/{len(units)} {m}/{c}/{a} n={meta['n_rows']} lam={lam:.3g}"
            f" elapsed={meta['fit_wall_s']}s",
            flush=True,
        )
    ds = {meta["d"] for meta in out.values()}
    assert len(ds) == 1, f"inconsistent d across cells: {ds}"
    return out


# ---------------------------------------------------------------------------
# Phase B — batched rotation-null bank (shared across every pair/arm/model)
# ---------------------------------------------------------------------------
def build_null_bank(
    cell_keys: list[tuple[str, str, str]],
    work_dir: Path,
    args: argparse.Namespace,
    d: int,
) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Return (vals, keys): vals[k, i, j] = cos draw for reference=cell_i,
    rotated=cell_j at Haar draw k. Checkpointed + resumable (per-draw seeds)."""
    keys = sorted(cell_keys)
    svecs = np.stack(
        [np.load(work_dir / "svecs" / m / f"{c}__{a}.npy") for (m, c, a) in keys]
    )  # (C, d)
    assert svecs.shape[1] == d, svecs.shape
    norms = np.linalg.norm(svecs, axis=1)
    meta = {
        "seed": args.seed,
        "n_draws": args.null_draws,
        "d": d,
        "layer": args.layer,
        "cells": ["/".join(k) for k in keys],
    }
    tag = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    bank_dir = work_dir / "nullbank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_path = bank_dir / f"bank_{tag}.npz"
    c_n = len(keys)
    vals = np.zeros((args.null_draws, c_n, c_n), dtype=np.float64)
    start = 0
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as z:
            done = int(z["done_draws"])
            vals[:done] = z["vals"][:done]
            start = done
        if start >= args.null_draws:
            print(f"[nullbank] cached complete ({args.null_draws} draws, {bank_path})", flush=True)
            return vals, keys
        print(f"[nullbank] resuming at draw {start}/{args.null_draws}", flush=True)
    s_t = torch.from_numpy(svecs)  # (C, d) fp64
    denom = np.outer(norms, norms)
    for k in range(start, args.null_draws):
        t0 = time.time()
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + k)
        p = _haar(d, gen)
        r = _haar(d, gen)
        e = p * r.T
        m = (s_t @ e @ s_t.T).numpy()  # (C, C): [i, j] = s_i^T E s_j
        vals[k] = m / denom
        if (k + 1) % 25 == 0 or k + 1 == args.null_draws:
            # tmp already ends .npz, so np.savez(path) appends nothing and
            # opens/closes the handle itself (code-review v15 Minor: no
            # refcount-reliant flush before os.replace).
            tmp = bank_path.with_name(bank_path.stem + ".tmp.npz")
            np.savez(tmp, vals=vals, done_draws=np.int64(k + 1))
            os.replace(tmp, bank_path)
        print(
            f"[nullbank] draw {k + 1}/{args.null_draws} elapsed={time.time() - t0:.2f}s",
            flush=True,
        )
    _atomic_write_json(bank_dir / f"bank_{tag}.meta.json", meta)
    return vals, keys


# ---------------------------------------------------------------------------
# Phase C — per-pair observed statistics (unordered; symmetric by construction)
# ---------------------------------------------------------------------------
class _Lru:
    def __init__(self, cap: int, loader):
        self.cap, self.loader, self.d = cap, loader, {}

    def get(self, key):
        if key in self.d:
            self.d[key] = self.d.pop(key)  # refresh order
            return self.d[key]
        v = self.loader(key)
        self.d[key] = v
        while len(self.d) > self.cap:
            self.d.pop(next(iter(self.d)))
        return v


def run_pair_battery(
    model_tags: list[str],
    conds: list[str],
    arms: tuple[str, ...],
    store_root: Path,
    work_dir: Path,
    bank_vals: np.ndarray,
    bank_keys: list[tuple[str, str, str]],
    args: argparse.Namespace,
) -> dict[str, list[dict]]:
    """Compute per-unordered-pair statistics per (model, arm); JSONL-checkpointed."""
    bank_idx = {k: i for i, k in enumerate(bank_keys)}
    pair_set = enumerate_pair_set()
    unordered = sorted({tuple(sorted(p)) for p in pair_set if p[0] in conds and p[1] in conds})
    if args.pairs_limit:
        unordered = unordered[: args.pairs_limit]
    bundles = _Lru(6, lambda mc: _load_cell_layer(store_root, f"{mc[0]}/{mc[1]}", args.layer))
    maps = _Lru(8, lambda mca: np.load(work_dir / "maps" / mca[0] / f"{mca[1]}__{mca[2]}.npy"))
    pairs_dir = work_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {m: [] for m in model_tags}
    for m in model_tags:
        jsonl = pairs_dir / f"{m}.jsonl"
        # Resume key includes the null-draw regime (code-review v15 Major,
        # concern procrustes-pairs-resume-null-draws — the #722 r3 class): a
        # cached row computed at a DIFFERENT --null-draws is ignored here and
        # recomputed+appended; the dict keeps the LAST matching row per key.
        # Skipped rows carry no null stats (draw-count-independent) and stay
        # reusable at any draw count.
        done = {
            (r["a"], r["b"], r["arm"]): r
            for r in _read_jsonl(jsonl)
            if "skipped_reason" in r or r.get("n_null_draws") == args.null_draws
        }
        n_units = len(unordered) * len(arms)
        unit = 0
        for a_slug, b_slug in unordered:
            ba = bundles.get((m, a_slug))
            bb = bundles.get((m, b_slug))
            a_idx, b_idx, n_common = _pair_rows(ba["conv_ids"], bb["conv_ids"])
            r_out = None
            if n_common >= MIN_N_COMMON:
                ya = torch.from_numpy(ba["Y"][a_idx])
                yb = torch.from_numpy(bb["Y"][b_idx])
                t0 = time.time()
                r_out = _orth(ya - ya.mean(0), yb - yb.mean(0))
                r_out_wall = time.time() - t0
            for arm in arms:
                unit += 1
                key = (a_slug, b_slug, arm)
                if key in done:
                    out[m].append(done[key])
                    print(
                        f"[battery] unit {unit}/{n_units} {m} {arm} {a_slug}~{b_slug} cached",
                        flush=True,
                    )
                    continue
                t0 = time.time()
                if n_common < MIN_N_COMMON:
                    row = {
                        "a": a_slug,
                        "b": b_slug,
                        "arm": arm,
                        "model": m,
                        "n_common": n_common,
                        "skipped_reason": f"n_common={n_common} < {MIN_N_COMMON}",
                    }
                    _append_jsonl(jsonl, row)
                    out[m].append(row)
                    print(
                        f"[battery] unit {unit}/{n_units} {m} {arm} {a_slug}~{b_slug} SKIP"
                        f" (n_common={n_common}) — flagged, not dropped",
                        flush=True,
                    )
                    continue
                xa = torch.from_numpy(ba[f"X_{arm}"][a_idx])
                xb = torch.from_numpy(bb[f"X_{arm}"][b_idx])
                r_in = _orth(xa - xa.mean(0), xb - xb.mean(0))
                w_a = torch.from_numpy(maps.get((m, a_slug, arm)))
                w_b = torch.from_numpy(maps.get((m, b_slug, arm)))
                m_fit = r_in.T @ w_a @ r_out
                ia, ib = bank_idx[(m, a_slug, arm)], bank_idx[(m, b_slug, arm)]
                # vals[:, i, j]: reference=cell_i, rotated=cell_j; use the
                # (min, max) orientation deterministically (distribution is
                # symmetric in the pair).
                draws = bank_vals[:, min(ia, ib), max(ia, ib)]
                sa = np.load(work_dir / "svecs" / m / f"{a_slug}__{arm}.npy")
                sb = np.load(work_dir / "svecs" / m / f"{b_slug}__{arm}.npy")
                row = {
                    "a": a_slug,
                    "b": b_slug,
                    "arm": arm,
                    "model": m,
                    "n_common": n_common,
                    "n_a": int(ba["Y"].shape[0]),
                    "n_b": int(bb["Y"].shape[0]),
                    "aligned_cosine": _cos_flat(m_fit, w_b),
                    "raw_cosine": _cos_flat(w_a, w_b),
                    "spectrum_cosine": float(
                        (sa * sb).sum() / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-12)
                    ),
                    "null_mean": float(draws.mean()),
                    "null_std": float(draws.std()),
                    "null_p025": float(np.quantile(draws, 0.025)),
                    "null_p975": float(np.quantile(draws, 0.975)),
                    "n_null_draws": int(len(draws)),
                    "wall_s": round(time.time() - t0 + (r_out_wall if arm == arms[0] else 0.0), 2),
                }
                _append_jsonl(jsonl, row)
                out[m].append(row)
                print(
                    f"[battery] unit {unit}/{n_units} {m} {arm} {a_slug}~{b_slug}"
                    f" aligned={row['aligned_cosine']:.4f} raw={row['raw_cosine']:.4f}"
                    f" null_p975={row['null_p975']:.4f} n={n_common}"
                    f" elapsed={row['wall_s']}s",
                    flush=True,
                )
    return out


# ---------------------------------------------------------------------------
# Phase D — assemble ordered-pair battery JSONs + per-class summary
# ---------------------------------------------------------------------------
def assemble_outputs(
    model_tags: list[str],
    conds: list[str],
    arms: tuple[str, ...],
    rows: dict[str, list[dict]],
    fit_meta: dict[tuple[str, str, str], dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_set = [p for p in enumerate_pair_set() if p[0] in conds and p[1] in conds]
    short_by_tag = {v: k for k, v in MODELS.items()}
    summary: dict = {}
    for m in model_tags:
        by_key = {(r["a"], r["b"], r["arm"]): r for r in rows[m]}
        expected = {(*sorted(p), arm) for p in pair_set for arm in arms}
        missing = sorted(expected - set(by_key))
        if missing:
            raise RuntimeError(
                f"battery incomplete for {m}: {len(missing)} missing, first={missing[:3]}"
            )
        pairs_out: dict[str, dict] = {}
        for src, tgt in pair_set:
            si, sf = parse_slug(src)
            ti, tf = parse_slug(tgt)
            cls = classify(si, sf, ti, tf)
            for arm in arms:
                r = by_key[(*sorted((src, tgt)), arm)]
                screened = arm_is_invalid(arm, si, sf, ti, tf)
                rec = {
                    "src": src,
                    "tgt": tgt,
                    "arm": arm,
                    "cls": cls,
                    "screened": bool(screened),
                    "screened_reason": (
                        "pair_digest arm_invalid: user-cell context arm (X_context == Y by"
                        " construction) or naturalistic-user prefix arm"
                        if screened
                        else None
                    ),
                    "n_common": r["n_common"],
                    "lambda_src": fit_meta[(m, src, arm)]["lambda"],
                    "lambda_tgt": fit_meta[(m, tgt, arm)]["lambda"],
                }
                if "skipped_reason" in r:
                    rec["skipped_reason"] = r["skipped_reason"]
                else:
                    # Belt for the resume-key fix (code-review v15 Major): a
                    # row carrying a different draw count than this run's
                    # metadata claims must never ship.
                    if r["n_null_draws"] != args.null_draws:
                        raise RuntimeError(
                            f"stale battery row for {m} {src}~{tgt}/{arm}: "
                            f"n_null_draws={r['n_null_draws']} != --null-draws "
                            f"{args.null_draws} (resume-regime violation)"
                        )
                    rec.update(
                        {
                            k: r[k]
                            for k in (
                                "aligned_cosine",
                                "raw_cosine",
                                "spectrum_cosine",
                                "null_mean",
                                "null_std",
                                "null_p025",
                                "null_p975",
                                "n_null_draws",
                                "n_a",
                                "n_b",
                            )
                        }
                    )
                pairs_out.setdefault(f"{src}__{tgt}", {})[arm] = rec
        short = short_by_tag.get(m, m)
        payload = {
            "metadata": _metadata(args),
            "model": m,
            "model_short": short,
            "layer": args.layer,
            "n_ordered_pairs": len(pair_set),
            "pairs": pairs_out,
        }
        _atomic_write_json(out_dir / f"battery_{m}_L{args.layer}.json", payload)
        print(f"[assemble] wrote {out_dir / f'battery_{m}_L{args.layer}.json'}", flush=True)
        # per-class summary over UNSCREENED, non-skipped ordered rows
        for arm in arms:
            for cls in sorted({classify(*parse_slug(s), *parse_slug(t)) for s, t in pair_set}):
                sel = [rec[arm] for key, rec in pairs_out.items() if rec[arm]["cls"] == cls]
                valid = [x for x in sel if not x["screened"] and "aligned_cosine" in x]
                entry = {
                    "n_ordered_rows": len(sel),
                    "n_screened": sum(1 for x in sel if x["screened"]),
                    "n_skipped": sum(1 for x in sel if "skipped_reason" in x),
                    "n_valid": len(valid),
                }
                if valid:
                    ac = np.array([x["aligned_cosine"] for x in valid])
                    entry.update(
                        {
                            "aligned_cosine_mean": float(ac.mean()),
                            "aligned_cosine_median": float(np.median(ac)),
                            "raw_cosine_mean": float(np.mean([x["raw_cosine"] for x in valid])),
                            "spectrum_cosine_mean": float(
                                np.mean([x["spectrum_cosine"] for x in valid])
                            ),
                            "frac_above_null_p975": float(
                                np.mean([x["aligned_cosine"] > x["null_p975"] for x in valid])
                            ),
                        }
                    )
                summary.setdefault(short, {}).setdefault(arm, {})[cls] = entry
    _atomic_write_json(
        out_dir / f"summary_L{args.layer}.json",
        {"metadata": _metadata(args), "per_class": summary},
    )
    print(f"[assemble] wrote {out_dir / f'summary_L{args.layer}.json'}", flush=True)


# ---------------------------------------------------------------------------
# Verification legs
# ---------------------------------------------------------------------------
def verify_null_equivalence(seed: int) -> None:
    """(1) EXACT algebra: same-(Q1,Q2) spectral formula == direct rotated cosine.
    (2) Distributional: batched spectral draws match serial #1345-form draws."""
    torch.manual_seed(seed)
    d = 64
    w_a = torch.randn(d, d, dtype=torch.float64)
    w_b = torch.randn(d, d, dtype=torch.float64)
    u_a, s_a, vh_a = torch.linalg.svd(w_a)
    u_b, s_b, vh_b = torch.linalg.svd(w_b)
    norm = float(w_a.norm() * w_b.norm())
    gen = torch.Generator().manual_seed(seed)
    for k in range(5):
        q1, q2 = _haar(d, gen), _haar(d, gen)
        direct = float((w_b.reshape(-1) @ (q1.T @ w_a @ q2).reshape(-1)) / norm)
        p = u_b.T @ q1.T @ u_a
        r = vh_a @ q2 @ vh_b.T
        spectral = float((s_b @ ((p * r.T) @ s_a)) / norm)
        assert abs(direct - spectral) < 1e-9, (k, direct, spectral)
    print("[verify] exact same-(Q1,Q2) identity: PASS (5 draws, |diff| < 1e-9)", flush=True)

    d2, k_draws = 16, 4000
    w_a = torch.randn(d2, d2, dtype=torch.float64)
    w_b = torch.randn(d2, d2, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed + 1)
    serial = []
    vb = w_b.reshape(-1)
    vb_n = vb / vb.norm()
    for _ in range(k_draws):
        mn = (_haar(d2, gen).T @ w_a @ _haar(d2, gen)).reshape(-1)
        serial.append(float((mn @ vb_n) / (mn.norm() + 1e-12)))
    serial = np.array(serial)
    s_a = torch.linalg.svdvals(w_a)
    s_b = torch.linalg.svdvals(w_b)
    gen2 = torch.Generator().manual_seed(seed + 2)
    batched = []
    nrm = float(s_a.norm() * s_b.norm())
    for _ in range(k_draws):
        p, r = _haar(d2, gen2), _haar(d2, gen2)
        batched.append(float(s_b @ ((p * r.T) @ s_a)) / nrm)
    batched = np.array(batched)
    se = np.hypot(serial.std() / np.sqrt(k_draws), batched.std() / np.sqrt(k_draws))
    assert abs(serial.mean() - batched.mean()) < 6 * se, (serial.mean(), batched.mean(), se)
    ratio = batched.std() / serial.std()
    assert 0.9 < ratio < 1.1, ratio
    for q in (0.025, 0.975):
        dq = abs(np.quantile(serial, q) - np.quantile(batched, q))
        assert dq < 0.2 * serial.std(), (q, dq, serial.std())
    print(
        f"[verify] distributional (d={d2}, K={k_draws}): PASS"
        f" (means {serial.mean():+.5f}/{batched.mean():+.5f}, std ratio {ratio:.3f})",
        flush=True,
    )


def serial_null_check(
    model_tags: list[str],
    conds: list[str],
    work_dir: Path,
    bank_vals: np.ndarray,
    bank_keys: list[tuple[str, str, str]],
    args: argparse.Namespace,
) -> None:
    """Production-shape serial rotation draws vs the batched bank band (one pair)."""
    m = model_tags[0]
    pair_set = [p for p in enumerate_pair_set() if p[0] in conds and p[1] in conds]
    a_slug, b_slug = sorted(pair_set[0])
    arm = "prefix"
    w_a = torch.from_numpy(np.load(work_dir / "maps" / m / f"{a_slug}__{arm}.npy"))
    w_b = torch.from_numpy(np.load(work_dir / "maps" / m / f"{b_slug}__{arm}.npy"))
    d = w_a.shape[0]
    gen = torch.Generator().manual_seed(args.seed + 777)
    vb = w_b.reshape(-1)
    vb_n = vb / vb.norm()
    draws = []
    for k in range(args.serial_null_check):
        t0 = time.time()
        mn = (_haar(d, gen).T @ w_a @ _haar(d, gen)).reshape(-1)
        draws.append(float((mn @ vb_n) / (mn.norm() + 1e-12)))
        print(
            f"[serial-check] draw {k + 1}/{args.serial_null_check} elapsed={time.time() - t0:.2f}s",
            flush=True,
        )
    serial = np.array(draws)
    idx = {k: i for i, k in enumerate(bank_keys)}
    ia, ib = idx[(m, a_slug, arm)], idx[(m, b_slug, arm)]
    bank = bank_vals[:, min(ia, ib), max(ia, ib)]
    se = np.hypot(serial.std() / np.sqrt(len(serial)), bank.std() / np.sqrt(len(bank)))
    dm = abs(serial.mean() - bank.mean())
    ratio = serial.std() / (bank.std() + 1e-300)
    ok = dm <= max(6 * se, 1e-4) and 0.4 < ratio < 2.5
    print(
        f"[serial-check] pair {a_slug}~{b_slug}/{arm}: serial mean={serial.mean():+.2e}"
        f" std={serial.std():.2e} (K={len(serial)}) vs bank mean={bank.mean():+.2e}"
        f" std={bank.std():.2e} (K={len(bank)}) -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        raise RuntimeError("serial-vs-batched null check FAILED")


# ---------------------------------------------------------------------------
# Synthetic store generator (smoke inputs; same layout as the HF store)
# ---------------------------------------------------------------------------
def write_synthetic_store(root: Path, seed: int, layer: int) -> None:
    """Tiny synthetic bundles in the production layout <model>/<cond>/L<layer>.pt.

    Includes a user_* condition (screen-flag probe) and one cell with
    DISJOINT conv_ids (dana_chat: exercises the n_common < 3 designed skip).
    """
    rng = np.random.default_rng(seed)
    conds = ["assistant_chat", "assistant_naturalistic", "user_lmsys_chat", "dana_chat"]
    n, d, n_convs = 30, 16, 10
    for m in MODELS.values():
        base_convs = np.array([f"conv{i:03d}" for i in range(n_convs)])
        for c in conds:
            convs = (
                np.array([f"zz{i:03d}" for i in range(n_convs)]) if c == "dana_chat" else base_convs
            )
            conv_ids = np.repeat(convs, n // n_convs)
            x_ctx = rng.standard_normal((n, d)).astype(np.float32)
            y = (
                x_ctx @ rng.standard_normal((d, d)).astype(np.float32) * 0.5
            ) + rng.standard_normal((n, d)).astype(np.float32) * 0.1
            bundle = {
                "X_prefix": x_ctx + rng.standard_normal((n, d)).astype(np.float32) * 0.2,
                "X_context": x_ctx,
                "Y": y,
                "conv_ids": conv_ids,
            }
            path = root / m / c / f"L{layer}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bundle, path)
    print(f"[synthetic] wrote {len(conds) * 2} bundles under {root}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="base,instruct", help="comma list of short names")
    ap.add_argument("--conditions", default=",".join(c.slug for c in CONDITION_TABLE))
    ap.add_argument("--arms", default="prefix,context")
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--store-root", type=Path, default=None, help="default <work-dir>/hf_dl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-stage", action="store_true", help="store-root already complete")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--pairs-limit", type=int, default=0, help="cap unordered pairs (pilot)")
    ap.add_argument(
        "--serial-null-check",
        type=int,
        default=0,
        help="K production-shape serial rotation draws vs the batched bank (pilot duty)",
    )
    ap.add_argument("--verify-null-equivalence", action="store_true")
    ap.add_argument("--write-synthetic-store", type=Path, default=None)
    args = ap.parse_args()

    if args.verify_null_equivalence:
        verify_null_equivalence(args.seed)
        return 0
    if args.write_synthetic_store is not None:
        write_synthetic_store(args.write_synthetic_store, args.seed, args.layer)
        return 0

    assert args.null_draws >= MIN_QUANTILE_DRAWS, (
        f"--null-draws {args.null_draws} < quantile floor {MIN_QUANTILE_DRAWS}"
    )
    model_tags = [MODELS[s.strip()] for s in args.models.split(",") if s.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    known = {c.slug for c in CONDITION_TABLE}
    unknown = [c for c in conds if c not in known]
    assert not unknown, f"unknown conditions: {unknown}"
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    store_root = args.store_root or (args.work_dir / "hf_dl")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    if not args.no_stage:
        stage_cells(model_tags, conds, store_root, args.layer)
    if args.stage_only:
        print(f"[done] stage-only complete ({time.time() - t_start:.1f}s)", flush=True)
        return 0

    t0 = time.time()
    fit_meta = fit_cell_maps(model_tags, conds, arms, store_root, args.work_dir, args)
    t_fit = time.time() - t0
    d = next(iter(fit_meta.values()))["d"]

    t0 = time.time()
    bank_vals, bank_keys = build_null_bank(list(fit_meta), args.work_dir, args, d)
    t_bank = time.time() - t0

    if args.serial_null_check:
        serial_null_check(model_tags, conds, args.work_dir, bank_vals, bank_keys, args)

    t0 = time.time()
    rows = run_pair_battery(
        model_tags, conds, arms, store_root, args.work_dir, bank_vals, bank_keys, args
    )
    t_battery = time.time() - t0

    if args.pairs_limit:
        print(
            "[done] pairs-limit set — skipping assemble (battery JSONL checkpoints only); "
            f"walls: fit={t_fit:.1f}s bank={t_bank:.1f}s battery={t_battery:.1f}s",
            flush=True,
        )
        return 0

    assemble_outputs(model_tags, conds, arms, rows, fit_meta, args.out_dir, args)
    print(
        f"[done] procrustes battery complete: fit={t_fit:.1f}s bank={t_bank:.1f}s"
        f" battery={t_battery:.1f}s total={time.time() - t_start:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed by the atomic-write helpers; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
