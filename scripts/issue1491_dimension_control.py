#!/usr/bin/env python3
"""Dimension control for the #1491 GREEDY scale ladder.

The greedy headline (`eval_results/issue_1491/greedy/fits_*.json`) reads ridge
held-out R² 0.5506 -> 0.7287 from 0.5B to 7B, then falls to 0.6981 at 32B.
Across that ladder ``h_dim`` grows 896 -> 5120 at a FIXED ``n_train = 25,000``,
so n/d falls 27.9 -> 4.88 monotonically with the very axis the headline varies.

R² is a variance RATIO, so target dimension does not mechanically inflate it.
The confound acts through (i) ESTIMATION — fewer samples per parameter at
large d — and (ii) WHICH directions the variance weighting emphasises. The
estimation half pushes held-out R² DOWN, so it works AGAINST the observed rise
and is a live candidate for the 7B -> 14B decline; the 14B-vs-32B contrast is
already dimension-controlled by construction (identical h_dim 5120, identical
n/d 4.88), making the post-7B decline a depth effect at fixed width.

This driver runs the deferred "random-projection d=896 dimension control" named
in ``scripts/issue1491_ladder_fits.py``'s Unit-3 deferral list, as five arms:

    arm          X                      Y            n_train          purpose
    baseline     native d               native d     25,000           reproduce the banked ridge R² (parity)
    target_jl    native d               JL -> 896    25,000           NULL control (see below)
    both_jl      JL -> 896              JL -> 896    25,000           STRICT control: dim, params, n/d all fixed
    matched_nd   native d               native d     round(25000*d/5120)  isolates the ESTIMATION confound alone
    pca_input    PCA -> 896 (train-fit) native d     25,000           brackets both_jl with variance-preserving compression

``target_jl`` is a NULL control BY DESIGN, and analytically so: for a fixed λ
the ridge solution against ``Y R`` is exactly ``W R``, so the residual becomes
``(Y - Ŷ) R`` and BOTH the numerator and the denominator of R² are hit by the
same JL distortion. It should read ~unchanged; an unchanged result is what
PROVES the trend is not a target-dimension artifact, and a changed one would
indict the pipeline. λ is re-selected on val per arm, so the agreement is
measured, not assumed.

``both_jl`` is the strict control — input dim, output dim, parameter count and
n/d (=27.9) are all fixed at every rung — but it is CONSERVATIVE, not clean:
its input compression is monotone in d (896->896 is a full-rank random square
map at 0.5B; 5120->896 is a real 5.7x squeeze at 14B/32B), so the handicap
grows with exactly the axis under test and works AGAINST the observed rise.
Read it as a lower bound. ``pca_input`` brackets it from the less-lossy side.
``target_jl`` and ``matched_nd`` are the clean single-confound reads.

The SAME JL matrix R is applied to input and target in ``both_jl`` (one random
projection of one space), which makes the 0.5B cell a near-similarity transform
of ``baseline`` — a useful internal consistency check.

Every arm also reports the kNN retrieval read (dimension-INVARIANT, hence the
cross-arm-comparable metric) and the identity+learned-bias baseline wherever
input and output share a dimension; where they do not, the arm records
``inapplicable`` with the reason rather than silently skipping it (CLAUDE.md
§ "Identity+learned-bias baseline AND kNN-retrieval metric").

STATED DEVIATION — context arm only. The #1491 ladder captured ``cx_last`` (the
context = prefix + user query) -> ``v_x`` (the answer state); no prefix-only
capture exists for this ladder at any scale, and building one would mean
re-running generation + capture across all six model scales. The deviation is
inherited from the parent greedy round and carried forward as a scope caveat
(CLAUDE.md § "Prefix mapping AND context mapping").

ESTIMATOR PARITY: the ridge fitter is ``issue779_ffc_n1m_fits.fit_ridge``,
imported and called — the SAME function that produced the banked greedy
numbers, not a re-implementation. The only new numerics are the JL draw and a
train-only PCA, both single GEMMs. The ``baseline`` arm is therefore a direct
reproduction check against ``eval_results/issue_1491/greedy/fits_<slug>.json``.

WELL-POSEDNESS: every arm keeps n_train > d (n/d ranges 4.88 -> 27.9), so no
held-out R² here sits in the ``n_train < d`` estimator-degenerate regime.

Usage (pod-side, detached):

    uv run python scripts/issue1491_dimension_control.py --scales all

Resume is per UNIT (arm x seed x rung), keyed on the regime constant below;
completed units are skipped and a fully-complete rung skips its download.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# load_dotenv BEFORE numpy/torch: torch freezes its thread pool from
# OMP_NUM_THREADS at import, and the shared-VM thread caps are applied by the
# project wrapper (#847). Harmless on a pod (the caps fail open there).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as F  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402

logger = logging.getLogger("issue1491_dimctl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Greedy (temperature-0) capture prefix — the arm this control targets.
GREEDY_PREFIX_TEMPLATE = "issue1491_scale_ladder_greedy/{slug}"

#: Common target dimension for the projection arms: the SMALLEST rung's h_dim,
#: so every rung is compressed to the same width.
TARGET_DIM = 896

#: The n/d ratio to match in the ``matched_nd`` arm — the 14B/32B value, i.e.
#: the ladder's own most-constrained rung. Held exactly as 25000/5120 rather
#: than the rounded 4.88, so the per-rung n_train values land on integers
#: (4375 / 7500 / 10000 / 17500 / 25000 / 25000).
MATCHED_N_NUM = 25_000
MATCHED_N_DEN = 5_120

#: Seeds for the two random-projection arms.
JL_SEEDS = (0, 1, 2)

#: Base offset for the JL RNG so a seed here cannot collide with the fit seed.
JL_RNG_BASE = 1_491_000

#: Deterministic RNG for the matched-n subsample. Prefix-nested by
#: construction: the same permutation is truncated at each rung's n_train, so
#: a smaller rung's train set is a SUBSET of every larger rung's.
SUBSAMPLE_SEED = 1491

#: Bumped whenever a change alters unit outputs, so a stale JSONL cannot be
#: silently resumed against new semantics (code-style § resume predicate).
REGIME = "dimctl-v1"


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def _jl_matrix(d_in: int, d_out: int, seed: int) -> np.ndarray:
    """Gaussian Johnson-Lindenstrauss matrix, ``(d_in, d_out)``, scaled 1/sqrt(d_out).

    The scaling makes ``E[||v R||²] == ||v||²``, so the projection preserves
    squared norms in expectation — which is exactly why the ``target_jl`` arm
    leaves the R² ratio ~unchanged.
    """
    rng = np.random.default_rng(JL_RNG_BASE + seed)
    r = rng.standard_normal((d_in, d_out), dtype=np.float32)
    r /= np.sqrt(np.float32(d_out))
    return r


def _pca_basis(x_train: np.ndarray, d_out: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-``d_out`` PCA basis fit on TRAIN rows only.

    Returns ``(mean, basis, variance_retained)`` where ``basis`` is
    ``(d_in, d_out)`` and the projection is ``(x - mean) @ basis``.

    The covariance is accumulated in float32 (fast GEMM) and eigendecomposed in
    float64 — only the TOP d_out of d_in eigenvectors are used, which is the
    numerically well-conditioned end of the spectrum.
    """
    mu = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mu
    cov = (xc.T @ xc).astype(np.float64) / max(1, xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    total = float(evals.sum())
    top = np.argsort(evals)[::-1][:d_out]
    retained = float(evals[top].sum() / total) if total > 0 else float("nan")
    basis = np.ascontiguousarray(evecs[:, top]).astype(np.float32)
    assert basis.shape == (x_train.shape[1], d_out), basis.shape
    return mu.astype(np.float32), basis, retained


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


def _matched_n_train(h_dim: int) -> int:
    """n_train that reproduces the 14B/32B n/d ratio at this rung's dimension."""
    return int(round(MATCHED_N_NUM * h_dim / MATCHED_N_DEN))


def _unit_specs(h_dim: int) -> list[dict]:
    """Enumerate the (arm, seed) units for one rung, in run order."""
    units: list[dict] = [{"arm": "baseline", "seed": None}]
    units += [{"arm": "target_jl", "seed": s} for s in JL_SEEDS]
    units += [{"arm": "both_jl", "seed": s} for s in JL_SEEDS]
    units.append({"arm": "matched_nd", "seed": None})
    units.append({"arm": "pca_input", "seed": None})
    return units


def _build_arm(
    arm: str,
    seed: int | None,
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    pca_cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return ``(X_arm, Y_arm, tr_arm, provenance)`` for one unit.

    Only the arrays the arm actually transforms are copied; ``baseline`` and
    ``matched_nd`` pass the originals through untouched.
    """
    d = X.shape[1]
    if arm == "baseline":
        return X, Y, tr, {"transform": "none"}

    if arm == "target_jl":
        r = _jl_matrix(d, TARGET_DIM, seed)
        return (
            X,
            (Y @ r).astype(np.float32),
            tr,
            {"transform": f"Y @ R  ({d} -> {TARGET_DIM})", "jl_seed": seed},
        )

    if arm == "both_jl":
        # ONE R for both sides: this is a single random projection of the
        # shared residual-stream space, not two independent ones.
        r = _jl_matrix(d, TARGET_DIM, seed)
        return (
            (X @ r).astype(np.float32),
            (Y @ r).astype(np.float32),
            tr,
            {"transform": f"X @ R and Y @ R, same R  ({d} -> {TARGET_DIM})", "jl_seed": seed},
        )

    if arm == "matched_nd":
        n_keep = _matched_n_train(d)
        assert n_keep <= len(tr), f"matched n_train {n_keep} exceeds available {len(tr)}"
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        keep = np.sort(rng.permutation(len(tr))[:n_keep])
        return (
            X,
            Y,
            tr[keep],
            {
                "transform": "none (train subsample only)",
                "n_train": int(n_keep),
                "subsample_seed": SUBSAMPLE_SEED,
            },
        )

    if arm == "pca_input":
        if "basis" not in pca_cache:
            mu, basis, retained = _pca_basis(X[tr], TARGET_DIM)
            pca_cache.update({"mu": mu, "basis": basis, "retained": retained})
        x_proj = ((X - pca_cache["mu"]) @ pca_cache["basis"]).astype(np.float32)
        return (
            x_proj,
            Y,
            tr,
            {
                "transform": f"X @ PCA  ({d} -> {TARGET_DIM}), basis fit on train rows only",
                "variance_retained": pca_cache["retained"],
            },
        )

    raise ValueError(f"unknown arm: {arm}")


# ---------------------------------------------------------------------------
# One unit
# ---------------------------------------------------------------------------


def _run_unit(
    scale_key: str,
    unit: dict,
    bundle: dict,
    pca_cache: dict,
    dev: torch.device,
) -> dict:
    """Fit ONE (arm, seed) cell and return its result row."""
    scale = LF.LADDER_SCALES[scale_key]
    arm, seed = unit["arm"], unit["seed"]
    X, Y = bundle["X"], bundle["Y"]
    tr, val, te = bundle["tr"], bundle["val"], bundle["te"]

    t0 = time.time()
    Xa, Ya, tra, prov = _build_arm(arm, seed, X, Y, tr, pca_cache)

    n_train, d_in, d_out = len(tra), Xa.shape[1], Ya.shape[1]
    # Estimator-validity gate (CLAUDE.md § inline estimator-validity duties):
    # every held-out R² in the n_train < d regime is estimator-degenerate, not
    # a signal read. No arm here is designed to enter it, so refuse loudly.
    assert n_train > d_in, (
        f"under-determined fit refused: arm={arm} n_train={n_train} <= d_in={d_in}"
    )

    pred, meta = F.fit_ridge(Xa, Ya, tra, val, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    y_true = Ya[te]
    r2 = LF._pooled_r2(pred, y_true)

    # Dimension-INVARIANT companion read (chance = k / n_pool at every d), so
    # arms whose R² is not directly comparable still are.
    ks = tuple(k for k in (1, 5, 10, 50) if k < len(te))
    knn = {m: MB.knn_retrieval(pred, y_true, ks=ks, metric=m) for m in ("euclidean", "cosine")}

    # Identity + learned bias, wherever input and output share a dimension.
    if d_in == d_out:
        pred_idb = MB.identity_bias_predict(Xa[tra], Ya[tra], Xa[te]).astype(np.float32)
        identity_bias = {
            "applicable": True,
            "test_r2": LF._pooled_r2(pred_idb, y_true),
            "knn": {
                m: MB.knn_retrieval(pred_idb, y_true, ks=ks, metric=m)
                for m in ("euclidean", "cosine")
            },
        }
    else:
        identity_bias = {
            "applicable": False,
            "reason": (
                f"input dim {d_in} != output dim {d_out}; the identity family is "
                "undefined across a dimension change (stated inapplicable, not skipped)"
            ),
        }

    # Train-mean floor: input-agnostic reference for THIS arm's target space.
    pred_mean = np.broadcast_to(Ya[tra].mean(axis=0, keepdims=True), (len(te), d_out))
    train_mean_r2 = LF._pooled_r2(pred_mean, y_true)

    return {
        "regime": REGIME,
        "scale_key": scale_key,
        "slug": scale["slug"],
        "model": scale["model"],
        "h_dim": int(scale["h_dim"]),
        "primary_layer_index": int(bundle["primary_layer"]),
        "arm": arm,
        "seed": seed,
        "provenance": prov,
        "n_train": int(n_train),
        "d_in": int(d_in),
        "d_out": int(d_out),
        "n_over_d_in": float(n_train / d_in),
        "test_r2": r2,
        "train_mean_r2": train_mean_r2,
        "identity_bias": identity_bias,
        "knn_retrieval": knn,
        "ridge_meta": meta,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Per-unit persistence + resume
# ---------------------------------------------------------------------------


def _unit_key(scale_key: str, unit: dict) -> str:
    return f"{scale_key}|{unit['arm']}|{unit['seed']}"


def _load_done(jsonl: Path) -> dict[str, dict]:
    """Completed units from a prior run, keyed on ``_unit_key`` — REGIME-gated.

    Text-mode iteration, never ``.splitlines()`` (the U+2028 JSONL shredding
    trap in `.claude/rules/gotchas.md`).
    """
    done: dict[str, dict] = {}
    if not jsonl.exists():
        return done
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("regime") != REGIME:
                continue
            done[_unit_key(row["scale_key"], row)] = row
    return done


def _append_unit(jsonl: Path, row: dict) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# One rung
# ---------------------------------------------------------------------------


def run_scale(scale_key: str, args, done: dict[str, dict]) -> list[dict]:
    """Run every unit for one rung, streaming the capture at most once."""
    scale = LF.LADDER_SCALES[scale_key]
    slug = scale["slug"]
    primary_layer = scale["layers"][len(scale["layers"]) // 2]
    units = _unit_specs(scale["h_dim"])

    pending = [u for u in units if _unit_key(scale_key, u) not in done]
    if not pending:
        logger.info(
            "[dimctl] %s: all %d units already complete — skipping download", slug, len(units)
        )
        return [done[_unit_key(scale_key, u)] for u in units]

    hf_prefix = args.hf_prefix_template.format(slug=slug)
    cache_dir = args.cache_dir / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[dimctl] %s: streaming layer %d from %s (%d/%d units pending)",
        slug,
        primary_layer,
        hf_prefix,
        len(pending),
        len(units),
    )
    bundle = LF._assemble_scale_layer(hf_prefix, primary_layer, cache_dir)
    bundle["primary_layer"] = primary_layer
    assert bundle["Y"].shape[1] == scale["h_dim"], (
        f"h_dim mismatch: Y={bundle['Y'].shape} vs scale h_dim={scale['h_dim']}"
    )
    logger.info(
        "[dimctl] %s: assembled X=%s Y=%s tr=%d val=%d te=%d",
        slug,
        bundle["X"].shape,
        bundle["Y"].shape,
        len(bundle["tr"]),
        len(bundle["val"]),
        len(bundle["te"]),
    )

    dev = torch.device(args.device)
    pca_cache: dict = {}
    for i, unit in enumerate(pending, 1):
        key = _unit_key(scale_key, unit)
        row = _run_unit(scale_key, unit, bundle, pca_cache, dev)
        _append_unit(args.units_jsonl, row)
        done[key] = row
        # Per-unit progress line (code-style § per-unit progress line): a phase
        # whose only observable is process liveness is a wedge to every poller.
        print(
            f"[dimctl] unit {i}/{len(pending)} {key} "
            f"n/d={row['n_over_d_in']:.2f} R2={row['test_r2']:.4f} "
            f"acc@1={row['knn_retrieval']['cosine']['acc_at_k'][1]:.3f} "
            f"elapsed={row['elapsed_s']}s",
            flush=True,
        )

    return [done[_unit_key(scale_key, u)] for u in units]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--scales",
        default="all",
        help="comma-separated scale keys, or 'all' (default). "
        f"Known: {','.join(sorted(LF.LADDER_SCALES))}",
    )
    p.add_argument(
        "--hf-prefix-template",
        default=GREEDY_PREFIX_TEMPLATE,
        help="HF prefix template with a {slug} placeholder (default: the greedy arm)",
    )
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_results/issue_1491/dimension_control"),
        help="per-rung result JSONs + the combined summary land here",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("EPM_DIMCTL_CACHE_DIR", "data/issue_1491/dimctl_cache")),
        help="streaming chunk cache (one chunk at a time; unlinked after slicing)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.scales == "all":
        args.scale_keys = sorted(LF.LADDER_SCALES)
    else:
        args.scale_keys = [s.strip() for s in args.scales.split(",") if s.strip()]
        unknown = [s for s in args.scale_keys if s not in LF.LADDER_SCALES]
        if unknown:
            p.error(f"unknown scale key(s): {unknown}; known: {sorted(LF.LADDER_SCALES)}")

    args.units_jsonl = args.out_dir / "units.jsonl"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "This round is CPU-only by design; pass --device cpu."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(args.units_jsonl)
    logger.info(
        "[dimctl] regime=%s resume: %d completed units on disk; scales=%s",
        REGIME,
        len(done),
        args.scale_keys,
    )

    all_rows: list[dict] = []
    for scale_key in args.scale_keys:
        rows = run_scale(scale_key, args, done)
        all_rows.extend(rows)
        out = args.out_dir / f"dimctl_{LF.LADDER_SCALES[scale_key]['slug']}.json"
        out.write_text(json.dumps({"regime": REGIME, "units": rows}, indent=2), encoding="utf-8")
        logger.info("[dimctl] wrote %s (%d units)", out, len(rows))

    summary = args.out_dir / "dimctl_summary.json"
    summary.write_text(
        json.dumps(
            {
                "regime": REGIME,
                "target_dim": TARGET_DIM,
                "jl_seeds": list(JL_SEEDS),
                "matched_nd_ratio": MATCHED_N_NUM / MATCHED_N_DEN,
                "hf_prefix_template": args.hf_prefix_template,
                "mapping_arm": "context-based only (cx_last -> v_x); no prefix capture exists "
                "for this ladder — stated deviation, inherited from the parent greedy round",
                "ridge_fitter": "issue779_ffc_n1m_fits.fit_ridge (imported verbatim; "
                "the same fitter that produced the banked greedy numbers)",
                "lambdas": [float(x) for x in LF.LAMBDAS],
                "units": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("[dimctl] wrote %s (%d units total)", summary, len(all_rows))

    # Explicit exit: a bare return can hit the PyGILState_Release atexit race
    # across torch/numpy C extensions and hand a nonzero rc to the dispatcher
    # for a phase whose work completed (`.claude/rules/gotchas.md`).
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
