"""Issue #1482 follow-up — SAE->SAE nonlinear (MLP) recovery on the SPARSE arm.

The parent P3 MLP diverged on the sparse-input arm (``sae_ctx``): pooled R2
-2.11e12 (mean), -1.02e10 (max), -5.94e13 (frac), while the DENSE-input MLP on
the same targets was healthy (0.7387) and ridge on the same sparse input was
0.6901 / 0.3589 / 0.5395. The feature-space linear-vs-nonlinear read therefore
rests entirely on the dense arm; this driver closes that gap.

Leading hypothesis under test (``--phase diag``): the sparse design is
``concat(psi_mean[:, f_in], psi_last[:, f_in])`` where ``f_in`` is an activity
restriction computed on the psi_MEAN block ONLY (``_p3_prep``); the psi_LAST
block inherits those feature ids with no activity check of its own. A column
that is (near-)constant on the 120k train rows gets a standardizer
``xsd = sqrt(var) + 1e-9`` (issue779_ffc_n1m_fits._train_standardizer), so a
nonzero HOLDOUT value in that column is amplified by up to ~1e9. Ridge is
immune (a train-constant column gets an exactly-zero coefficient from the
closed-form solve); an MLP is NOT (its first-layer weights on that column keep
their random init, since the column never varies in training), so the blow-up
lands only at PREDICT time -- which is exactly the observed signature: training
runs to a normal early stop, then pooled R2 explodes negative.

``--phase ladder`` tests ONE-VARIABLE-AT-A-TIME fixes at short epochs on the
mean pooling; ``--phase final`` runs the selected variant at full epochs for
all three poolings with split-half rank stability and a shuffled-pairing floor.
Every variant reuses the PARENT fitter (issue779_ffc_n1m_fits._fit_mlp_minibatch)
unchanged -- the fixes are preprocessing/hyperparameter, not a reimplementation.

DIGEST-ONLY: activations, feature ids and scalars only; no corpus text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# BEFORE torch/numpy: torch freezes its intra-op thread pool from OMP_NUM_THREADS
# at import, and load_dotenv() is what setdefaults the shared-VM caps (#847).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1482_error_analysis as EA  # noqa: E402  (store load, prep, densify, metrics)
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (PARENT fitter -- reused verbatim)
import issue779_fitter_fair_comparison as F  # noqa: E402  (MLP_MAX_EPOCHS/PATIENCE/WD)
import issue779_percontext_recon as PR  # noqa: E402  (_pooled_r2)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_sae_mlp_recovery")

ARM = "sae_ctx"
POOLINGS = ("mean", "max", "frac")
# Banked parent references (eval_results/issue_1482/sae_perfeature/summary.json)
REF = {
    "ridge_sae_ctx": {
        "mean": 0.6901409540488584,
        "max": 0.35888451827902956,
        "frac": 0.5395329417404842,
    },
    "mlp_sae_ctx_broken": {
        "mean": -2107897883076.9233,
        "max": -10178451246.74001,
        "frac": -59395724118335.13,
    },
    "mlp_sae_dense_in": {"mean": 0.7387},
}


def _prep_args(args) -> argparse.Namespace:
    """The subset of the parent driver's arg surface that ``_p3_prep`` reads."""
    return argparse.Namespace(
        store=args.store,
        scratch=args.scratch,
        max_features_out=args.max_features_out,
        max_features_in=args.max_features_in,
        device=args.device,
        smoke=False,
        seed=args.seed,
    )


def _block_activity(Z: np.ndarray, tr: np.ndarray, chunk: int = 8192) -> dict[str, np.ndarray]:
    """Streamed per-column train activity + mean/var of the design (fp64)."""
    n_col = Z.shape[1]
    nz = np.zeros(n_col, dtype=np.int64)
    s1 = np.zeros(n_col, dtype=np.float64)
    s2 = np.zeros(n_col, dtype=np.float64)
    for i in range(0, len(tr), chunk):
        blk = Z[tr[i : i + chunk]].astype(np.float64)
        nz += (blk != 0.0).sum(0)
        s1 += blk.sum(0)
        s2 += (blk * blk).sum(0)
    n = float(len(tr))
    mu = s1 / n
    var = (s2 - n * mu * mu) / max(1.0, n - 1.0)
    return {"nonzero": nz, "mean": mu, "var": np.clip(var, 0.0, None)}


def _max_abs_z(
    Z: np.ndarray, rows: np.ndarray, mu: np.ndarray, sd: np.ndarray, chunk: int = 8192
) -> tuple[float, int]:
    """Streamed max |(x - mu)/sd| over ``rows`` and the argmax column."""
    best, col = 0.0, -1
    for i in range(0, len(rows), chunk):
        blk = (Z[rows[i : i + chunk]].astype(np.float64) - mu) / sd
        m = np.abs(blk).max(0)
        j = int(np.argmax(m))
        if m[j] > best:
            best, col = float(m[j]), j
    return best, col


def _diagnose(prep, Z: np.ndarray, n_in: int, out_dir: Path) -> dict:
    """Per-block degeneracy census + the standardizer amplification it implies."""
    tr, te = prep.tr, prep.te
    act = _block_activity(Z, tr)
    sd = np.sqrt(act["var"]) + 1e-9  # the parent standardizer, verbatim
    floor = prep.floor  # the parent's >=1% activity floor over n_fit
    blocks = {"psi_mean": slice(0, n_in), "psi_last": slice(n_in, 2 * n_in)}
    doc: dict = {
        "activity_floor_rows": int(floor),
        "n_train": int(len(tr)),
        "n_holdout": int(len(te)),
        "n_cols": int(Z.shape[1]),
        "blocks": {},
    }
    for name, sl in blocks.items():
        a_nz, a_var, a_sd = act["nonzero"][sl], act["var"][sl], sd[sl]
        below = a_nz < floor
        # columns that are (near-)constant on TRAIN but carry signal on HOLDOUT --
        # the amplification set the MLP cannot learn to ignore
        te_nz = np.zeros(int(sl.stop - sl.start), dtype=np.int64)
        for i in range(0, len(te), 8192):
            te_nz += (Z[te[i : i + 8192], sl] != 0.0).sum(0)
        dead_train_live_holdout = int(((a_var <= 0.0) & (te_nz > 0)).sum())
        doc["blocks"][name] = {
            "n_cols": int(sl.stop - sl.start),
            "n_below_parent_activity_floor": int(below.sum()),
            "frac_below_parent_activity_floor": float(below.mean()),
            "n_zero_train_variance": int((a_var <= 0.0).sum()),
            "n_zero_train_variance_but_nonzero_holdout": dead_train_live_holdout,
            "train_sd_quantiles": {
                q: float(np.quantile(a_sd, v))
                for q, v in (("p00", 0.0), ("p01", 0.01), ("p50", 0.5), ("p99", 0.99))
            },
            "min_train_nonzero_rows": int(a_nz.min()),
        }
    doc["max_abs_z_train"] = dict(
        zip(("value", "col"), _max_abs_z(Z, tr, act["mean"], sd), strict=True)
    )
    doc["max_abs_z_holdout"] = dict(
        zip(("value", "col"), _max_abs_z(Z, te, act["mean"], sd), strict=True)
    )
    doc["interpretation"] = (
        "A column with zero TRAIN variance standardizes to (x-mu)/1e-9; any nonzero "
        "HOLDOUT value in it is amplified ~1e9x. Ridge zeroes such a column by "
        "construction; the MLP keeps random first-layer weights on it, so the "
        "blow-up appears only in holdout predictions."
    )
    EA._write_json(out_dir / "diag.json", doc)
    logger.info("[diag] %s", json.dumps({k: v for k, v in doc.items() if k != "blocks"})[:400])
    for name, b in doc["blocks"].items():
        logger.info("[diag] %s: %s", name, json.dumps(b))
    return {"act": act, "sd": sd, "doc": doc}


def _keep_mask(act: dict, n_in: int, floor: int, variant: str) -> np.ndarray:
    """Column keep-mask per variant (all variants leave the parent fitter untouched)."""
    n_col = 2 * n_in
    if variant in ("v0_baseline", "v2_lr_1e-4", "v4_lr_3e-5"):
        return np.ones(n_col, dtype=bool)
    if variant == "v1_activity_floor":
        # the parent's OWN >=1% activity floor, applied to BOTH blocks (the psi_last
        # block inherits f_in ids chosen on psi_mean activity and is never re-checked)
        return act["nonzero"] >= floor
    if variant == "v3_varfloor":
        return act["var"] > 0.0
    raise ValueError(variant)


def _fit_one(
    Z: np.ndarray,
    tgt: np.ndarray,
    prep,
    keep: np.ndarray,
    lr: float,
    max_epochs: int,
    batch: int,
    seed: int,
    dev: torch.device,
) -> tuple[np.ndarray, dict]:
    """One MLP fit through the PARENT fitter, on the kept design columns."""
    Zk = Z if keep.all() else np.ascontiguousarray(Z[:, keep])
    t0 = time.time()
    pred, meta = N1M._fit_mlp_minibatch(
        Zk,
        tgt,
        prep.tr,
        prep.te,
        width=N1M.MLP_W_PROTOCOL,
        lr=lr,
        max_epochs=max_epochs,
        batch=batch,
        seed=seed,
        dev=dev,
    )
    meta = {**meta, "n_cols_kept": int(keep.sum()), "wall_s": round(time.time() - t0, 1)}
    return pred, meta


def _splithalf_stability(pred: np.ndarray, truth: np.ndarray, te: np.ndarray) -> float:
    """Per-feature R2 rank stability across a seeded half-split of the holdout
    (the parent ridge unit's read, _p3_unit_ridge)."""
    perm = EA._splithalf_perm(len(te))
    ia, ib = perm[: len(te) // 2], perm[len(te) // 2 :]
    pa = EA._per_feature_metrics(pred[ia], truth[ia])
    pb = EA._per_feature_metrics(pred[ib], truth[ib])
    ok = np.isfinite(pa["r2"]) & np.isfinite(pb["r2"])
    if ok.sum() < 3:
        return float("nan")
    ra = EA._midrank(pa["r2"][ok][:, None])[:, 0]
    rb = EA._midrank(pb["r2"][ok][:, None])[:, 0]
    return float(np.corrcoef(ra, rb)[0, 1])


def _shuffle_floor(pred: np.ndarray, truth: np.ndarray, seed: int) -> float:
    """Shuffled-pairing floor: pooled R2 of the predictions against permuted rows."""
    rng = np.random.default_rng(seed)
    return float(PR._pooled_r2(pred[rng.permutation(len(pred))], truth))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=("diag", "ladder", "final", "ridge_matched", "all"),
        default="all",
    )
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-features-out", type=int, default=16384)
    ap.add_argument("--max-features-in", type=int, default=8192)
    ap.add_argument("--ladder-epochs", type=int, default=12)
    ap.add_argument("--final-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument(
        "--variants",
        default="v0_baseline,v1_activity_floor,v3_varfloor,v2_lr_1e-4",
        help="comma-separated ladder variants (one-variable-at-a-time)",
    )
    ap.add_argument("--select", default=None, help="variant for --phase final (default: best)")
    args = ap.parse_args()

    root = PROJECT_ROOT / "data" / "issue_1482"
    args.store = args.store or (root / "store" / "sae_pooled")
    args.scratch = args.scratch or (root / "scratch")
    out_dir = args.out_dir or (
        PROJECT_ROOT / "eval_results" / "issue_1482" / "sae_sae_mlp_recovery"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("[recovery] prep (store=%s)", args.store)
    prep = EA._p3_prep(_prep_args(args))
    n_in = len(prep.f_in)
    logger.info(
        "[recovery] n_rows=%d n_train=%d n_val=%d n_holdout=%d f_in=%d f_out=%d floor=%d",
        prep.n_rows,
        len(prep.tr),
        len(prep.va),
        len(prep.te),
        n_in,
        len(prep.f_out),
        prep.floor,
    )
    logger.info("[recovery] building sparse design (%s) ...", ARM)
    t0 = time.time()
    Z = EA._p3_design(_prep_args(args), prep, ARM)
    logger.info("[recovery] design %s in %.0fs", Z.shape, time.time() - t0)
    batch = min(N1M.MLP_BATCH, max(8, len(prep.tr)))

    diag = None
    if args.phase in ("diag", "all"):
        diag = _diagnose(prep, Z, n_in, out_dir)
    if args.phase == "diag":
        return 0
    if diag is None:
        act = _block_activity(Z, prep.tr)
        diag = {"act": act}
    act = diag["act"]

    if args.phase == "ridge_matched":
        # DESIGN-MATCHED linear reference: the banked ridge 0.6901/0.3589/0.5395 was
        # fitted on ALL 16384 columns, while the repaired MLP sees only the kept
        # subset -- so the linear-vs-nonlinear read needs ridge on the SAME design.
        variant = args.select or "v1_activity_floor"
        keep = _keep_mask(act, n_in, prep.floor, variant)
        Zk = np.ascontiguousarray(Z[:, keep])
        tgt = EA._p3_targets(prep, POOLINGS)
        preds = EA._shared_gram_ridge_multi(
            Zk, tgt, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
        )
        doc = {
            "variant": variant,
            "n_cols_kept": int(keep.sum()),
            "n_cols_total": int(len(keep)),
            "note": "ridge on the SAME restricted design the repaired MLP uses "
            "(the banked ridge reference used all 16384 columns)",
            "poolings": {},
        }
        for pool, (pt, meta) in preds.items():
            truth = tgt[pool][prep.te]
            doc["poolings"][pool] = {
                "pooled_r2": float(PR._pooled_r2(pt, truth)),
                "splithalf_rank_stability": _splithalf_stability(pt, truth, prep.te),
                "shuffled_pairing_floor_r2": _shuffle_floor(pt, truth, EA.SPLIT_SEED_1482),
                "ridge_full_design_reference": REF["ridge_sae_ctx"][pool],
                **meta,
            }
            logger.info("[ridge_matched] %s: %s", pool, json.dumps(doc["poolings"][pool]))
        EA._write_json(out_dir / "ridge_matched.json", doc)
        return 0

    logger.info("[recovery] densifying targets (mean) ...")
    tgt_mean = EA._p3_targets(prep, ("mean",))["mean"]

    ladder_rows: list[dict] = []
    if args.phase in ("ladder", "all"):
        for variant in [v for v in args.variants.split(",") if v]:
            keep = _keep_mask(act, n_in, prep.floor, variant)
            lr = {"v2_lr_1e-4": 1e-4, "v4_lr_3e-5": 3e-5}.get(variant, 3e-4)
            logger.info(
                "[ladder] %s: keep %d/%d cols, lr=%g, epochs=%d",
                variant,
                int(keep.sum()),
                len(keep),
                lr,
                args.ladder_epochs,
            )
            pred, meta = _fit_one(
                Z, tgt_mean, prep, keep, lr, args.ladder_epochs, batch, args.seed, dev
            )
            r2 = float(PR._pooled_r2(pred, tgt_mean[prep.te]))
            row = {"variant": variant, "pooling": "mean", "pooled_r2": r2, "lr": lr, **meta}
            ladder_rows.append(row)
            logger.info("[ladder] %s pooled R2=%.6g (%ss)", variant, r2, meta["wall_s"])
            EA._write_json(
                out_dir / "ladder.json",
                {
                    "epochs": args.ladder_epochs,
                    "reference": REF,
                    "rows": ladder_rows,
                    "note": "short-epoch one-variable-at-a-time probe on the mean pooling; "
                    "every variant calls the PARENT fitter unchanged",
                },
            )
    if args.phase == "ladder":
        return 0

    best = args.select
    if best is None:
        cand = [r for r in ladder_rows if np.isfinite(r["pooled_r2"])]
        if not cand:
            raise RuntimeError("[final] no finite ladder result to select from")
        best = max(cand, key=lambda r: r["pooled_r2"])["variant"]
    logger.info("[final] selected variant: %s", best)

    keep = _keep_mask(act, n_in, prep.floor, best)
    lr = {"v2_lr_1e-4": 1e-4, "v4_lr_3e-5": 3e-5}.get(best, 3e-4)
    finals: dict[str, dict] = {}
    for pool in POOLINGS:
        tgt = tgt_mean if pool == "mean" else EA._p3_targets(prep, (pool,))[pool]
        pred, meta = _fit_one(Z, tgt, prep, keep, lr, args.final_epochs, batch, args.seed, dev)
        truth = tgt[prep.te]
        finals[pool] = {
            "pooled_r2": float(PR._pooled_r2(pred, truth)),
            "splithalf_rank_stability": _splithalf_stability(pred, truth, prep.te),
            "splithalf_permutation_seed": EA.SPLIT_SEED_1482,
            "shuffled_pairing_floor_r2": _shuffle_floor(pred, truth, EA.SPLIT_SEED_1482),
            "ridge_reference": REF["ridge_sae_ctx"][pool],
            "broken_mlp_reference": REF["mlp_sae_ctx_broken"][pool],
            **meta,
        }
        logger.info("[final] %s: %s", pool, json.dumps(finals[pool]))
        EA._write_json(
            out_dir / "final.json",
            {
                "arm": ARM,
                "variant": best,
                "lr": lr,
                "max_epochs": args.final_epochs,
                "n_cols_kept": int(keep.sum()),
                "n_cols_total": int(len(keep)),
                "dense_mlp_reference": REF["mlp_sae_dense_in"]["mean"],
                "poolings": finals,
            },
        )
        if pool != "mean":
            del tgt
    logger.info("[recovery] done")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
