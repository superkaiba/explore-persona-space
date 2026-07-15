#!/usr/bin/env python
"""Issue #1345 Phase 5 — cross-regime operator transfer (full 3x3, frozen layers).

For each ordered regime pair (i -> j), model, and arm: fit the ridge map M_i on
regime i's matched-n rows (train folds), apply it to regime j's HELD-OUT rows,
score transfer R^2 against regime j's own targets — the #825
`frozen_map_swap` recipe (SRC train standardization stats applied to the TGT
eval points), generalized to per-regime conv-grouped fold assignments so
UNPAIRED pairs (stories) work and the paired chat<->no-template pair reduces
to the parent's row-aligned form exactly (identical conv set => identical
folds). Ridge core imported VERBATIM from issue825_crossmodel_map_transfer
(_prep_fold / _ridge_predict_cached / _pooled_r2 / _cv_folds).

Per-direction Δ(i->j) = transfer_R^2(i->j) - within_R^2(j) with the TARGET
regime's own within-regime held-out R^2 (recomputed on the SAME matched
subset + folds) as the denominator (plan §3); story transfers additionally
report %-of-story-ceiling. The headline chat<->no-template pair gets the
conversation-level PAIRED bootstrap of Δ_diff/Δ_xfer at L19 from the cached
held-out predictions (batched draws — issue1345_common.bootstrap machinery).

Outputs: eval_results/issue_1345/cross_regime_transfer_{model_slug}_{arm}.json
+ L19 preds caches under data/issue_1345/preds_cache/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fc  # noqa: E402
import issue1345_common as c  # noqa: E402
import numpy as np  # noqa: E402
from issue1345_fit_cells import load_matched, load_regime_bundle  # noqa: E402

FROZEN_LAYERS = cm.FROZEN_LAYERS
L19 = 19


# ---------------------------------------------------------------------------
# Data assembly (matched subsets, conv-sorted row order)
# ---------------------------------------------------------------------------
def load_arm_xy(bundle: dict, regime: str, arm: str) -> dict:
    """(X, Y, conv_ids) for one (regime, arm) bundle, rows sorted by conv_id.

    Multi-row groups (stories) sort stably within a conv_id; R1/R2 are
    one-row-per-conversation so the sort makes cross-regime rows align by
    construction (the paired-bootstrap precondition).
    """
    xy = fc._cell_xy(
        bundle,
        {
            "slot_index": c.ARM_SLOT_INDEX[arm],
            "target_turn_index": c.TARGET_TURN_INDEX[regime],
        },
    )
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    order = np.argsort(conv, kind="stable")
    return {"X": xy["X"][order], "Y": xy["Y"][order], "conv_ids": conv[order]}


def subset_rows(xy: dict, keep_ids: list[str]) -> dict:
    """Restrict rows to the given conv/story id set (order preserved)."""
    keep = np.isin(xy["conv_ids"], np.asarray(sorted(set(keep_ids))))
    assert keep.any(), "subset selected zero rows — matched-subset drift"
    return {"X": xy["X"][keep], "Y": xy["Y"][keep], "conv_ids": xy["conv_ids"][keep]}


# ---------------------------------------------------------------------------
# Transfer core (frozen layers; per-regime conv-grouped folds; preds captured)
# ---------------------------------------------------------------------------
def transfer_sweep(src: dict, tgt: dict, *, seed: int, null_draws: int) -> dict:
    """Fit M_src on src train folds; predict tgt held-out rows; pooled R^2.

    src == tgt gives the within-regime read on the identical subset+folds (the
    per-direction Δ denominator). Returns r2 + target-shuffle null bands per
    frozen layer and the L19 held-out predictions for the paired bootstrap.
    """
    folds_src = fc._cv_folds(src["conv_ids"], cm.N_FOLDS, seed)
    folds_tgt = fc._cv_folds(tgt["conv_ids"], cm.N_FOLDS, seed)
    rng = np.random.default_rng(seed + 5)
    out: dict = {"r2_by_layer": {}, "null_mean_by_layer": {}, "null_p975_by_layer": {}}
    for layer in FROZEN_LAYERS:
        Xs, Ys = src["X"][:, layer, :], src["Y"][:, layer, :]
        Xt, Yt = tgt["X"][:, layer, :], tgt["Y"][:, layer, :]
        preds = np.zeros((len(Yt), Yt.shape[1]), np.float32)
        fitted = np.zeros(len(Yt), bool)
        for k in range(cm.N_FOLDS):
            tr = folds_src != k
            te = folds_tgt == k
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = cm._prep_fold(Xs[tr], Xt[te])  # SRC train stats; TGT eval points
            pred = cm._ridge_predict_cached(cache, Ys[tr])
            preds[te] = pred.astype(np.float32)
            fitted[te] = True
        r2 = cm._pooled_r2(preds[fitted], Yt[fitted])
        out["r2_by_layer"][str(layer)] = float(r2)
        if null_draws > 0:
            true = Yt[fitted].astype(np.float64)
            pr = preds[fitted].astype(np.float64)
            draws = []
            for _ in range(null_draws):
                perm = rng.permutation(true.shape[0])
                draws.append(cm._pooled_r2(pr, true[perm]))
            out["null_mean_by_layer"][str(layer)] = float(np.nanmean(draws))
            out["null_p975_by_layer"][str(layer)] = float(np.nanquantile(draws, 0.975))
        if layer == L19:
            out["preds_l19"] = preds
            out["fitted_l19"] = fitted
            out["true_l19"] = tgt["Y"][:, L19, :]
            out["conv_ids"] = tgt["conv_ids"]
    return out


# ---------------------------------------------------------------------------
# Headline paired bootstrap (Δ_diff / Δ_xfer at L19, conversation unit)
# ---------------------------------------------------------------------------
def paired_delta_bootstrap(reads: dict, *, n_boot: int, seed: int) -> dict:
    """Conversation-level PAIRED bootstrap over the shared R1/R2 conv set.

    ``reads`` maps name -> (pred, true, conv_ids) for within_r1, within_r2,
    x_r1r2, x_r2r1 — all on the IDENTICAL conversation set (asserted). One
    shared counts matrix drives all four statistics per draw (paired).
    """
    suffs, uniq_ref = {}, None
    for name, (pred, true, conv) in reads.items():
        suff = c.conv_suffstats(pred, true, conv)
        if uniq_ref is None:
            uniq_ref = suff["uniq"]
        assert np.array_equal(suff["uniq"], uniq_ref), f"{name}: conv set mismatch"
        suffs[name] = suff
    counts = c.bootstrap_counts(len(uniq_ref), n_boot, seed)
    r2 = {name: c.batched_conv_r2(counts, s) for name, s in suffs.items()}
    d12 = r2["x_r1r2"] - r2["within_r2"]
    d21 = r2["x_r2r1"] - r2["within_r1"]
    d_xfer = 0.5 * (d12 + d21)
    d_same = d_xfer + c.DELTA_SAME_MARGIN
    d_diff = d_xfer + c.DELTA_DIFF_MARGIN

    def _ci(v):
        return {
            "mean": float(np.nanmean(v)),
            "ci_lo": float(np.nanquantile(v, 0.025)),
            "ci_hi": float(np.nanquantile(v, 0.975)),
        }

    return {
        "n_boot": int(n_boot),
        "n_groups": len(uniq_ref),
        "unit": "conversation (paired resample across all four statistics)",
        "delta_1to2": _ci(d12),
        "delta_2to1": _ci(d21),
        "delta_xfer": _ci(d_xfer),
        "delta_same": _ci(d_same),
        "delta_diff": _ci(d_diff),
        "delta_diff_ci_wholly_below_0": bool(np.nanquantile(d_diff, 0.975) < 0.0),
    }


# ---------------------------------------------------------------------------
# Per (model, arm) battery
# ---------------------------------------------------------------------------
def run_model_arm(
    bundles: dict,
    matched: dict,
    model: str,
    arm: str,
    out_dir: Path,
    preds_dir: Path,
    *,
    seed: int,
    null_draws: int,
    n_boot: int,
    include_r3: bool,
) -> None:
    regimes = [r for r in c.REGIMES if include_r3 or r != "r3"]
    full = {r: load_arm_xy(bundles[r], r, arm) for r in regimes}
    shared = matched["shared_r1r2_convs"]
    r3cfg = matched["per_model_r3_pair"].get(model) if include_r3 else None

    def _subset(regime: str, pair_kind: str) -> dict:
        if regime in ("r1", "r2"):
            ids = shared if pair_kind == "headline" else r3cfg["r12_convs"]
            return subset_rows(full[regime], ids)
        return subset_rows(full[regime], r3cfg["r3_story_ids"])

    # Cache sweeps keyed by (src_regime, tgt_regime, pair_kind); within = src==tgt
    sweeps: dict = {}

    def _sweep(i: str, j: str, pair_kind: str) -> dict:
        key = (i, j, pair_kind)
        if key not in sweeps:
            sweeps[key] = transfer_sweep(
                _subset(i, pair_kind), _subset(j, pair_kind), seed=seed, null_draws=null_draws
            )
        return sweeps[key]

    matrix: dict = {}
    deltas: dict = {}
    for i, j in [(a, b) for a in regimes for b in regimes if a != b]:
        pair_kind = "headline" if {i, j} == {"r1", "r2"} else "r3pair"
        xfer = _sweep(i, j, pair_kind)
        within_j = _sweep(j, j, pair_kind)
        matrix[f"{i}->{j}"] = {
            "pair_kind": pair_kind,
            "transfer_r2_by_layer": xfer["r2_by_layer"],
            "target_within_r2_by_layer": within_j["r2_by_layer"],
            "null_mean_by_layer": xfer["null_mean_by_layer"],
            "null_p975_by_layer": xfer["null_p975_by_layer"],
        }
        d19 = xfer["r2_by_layer"][str(L19)] - within_j["r2_by_layer"][str(L19)]
        deltas[f"{i}->{j}"] = {
            "delta_l19": float(d19),
            "transfer_l19": xfer["r2_by_layer"][str(L19)],
            "target_within_l19": within_j["r2_by_layer"][str(L19)],
        }
        if j == "r3":
            ceil = within_j["r2_by_layer"][str(L19)]
            deltas[f"{i}->{j}"]["pct_of_story_ceiling_l19"] = (
                float(xfer["r2_by_layer"][str(L19)] / ceil) if abs(ceil) > 1e-9 else None
            )

    # Within diagonals at each pair grain (reported alongside the matrix)
    within = {
        f"{r}@headline": _sweep(r, r, "headline")["r2_by_layer"] for r in regimes if r != "r3"
    }
    if include_r3:
        for r in regimes:
            within[f"{r}@r3pair"] = _sweep(r, r, "r3pair")["r2_by_layer"]

    # Headline paired bootstrap from the cached L19 preds (plan §3 CI bound)
    reads = {}
    for name, key in {
        "within_r1": ("r1", "r1", "headline"),
        "within_r2": ("r2", "r2", "headline"),
        "x_r1r2": ("r1", "r2", "headline"),
        "x_r2r1": ("r2", "r1", "headline"),
    }.items():
        sw = _sweep(*key)
        assert sw["fitted_l19"].all(), f"{name}: unfitted held-out rows at n>=folds"
        reads[name] = (sw["preds_l19"], sw["true_l19"], sw["conv_ids"])
        slug = c.MODEL_SLUG[model]
        np.savez(
            preds_dir / f"transfer_{slug}_{arm}_{name}_L19.npz",
            pred=sw["preds_l19"].astype(np.float32),
            true=sw["true_l19"].astype(np.float32),
            conv_ids=sw["conv_ids"],
            layer=np.asarray([L19]),
        )
    boot = paired_delta_bootstrap(reads, n_boot=n_boot, seed=seed + 31)

    slug = c.MODEL_SLUG[model]
    payload = {
        "metadata": c.metadata(seed, len(shared), "scripts/issue1345_cross_regime_transfer.py"),
        "model": model,
        "model_slug": slug,
        "arm": arm,
        "frozen_layers": list(FROZEN_LAYERS),
        "headline_layer": L19,
        "matched_n": {
            "shared_r1r2": len(shared),
            "r3_pair": (
                {k: v for k, v in r3cfg.items() if not isinstance(v, list)} if r3cfg else None
            ),
        },
        "matrix": matrix,
        "delta_table_l19": deltas,
        "within_r2_by_layer": within,
        "headline_paired_bootstrap": boot,
        "delta_margins": {"same": c.DELTA_SAME_MARGIN, "diff": c.DELTA_DIFF_MARGIN},
    }
    c.write_json(out_dir / f"cross_regime_transfer_{slug}_{arm}.json", payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR)
    ap.add_argument("--models", default="instruct,pretrained")
    ap.add_argument("--arms", default="prefix,context")
    ap.add_argument("--no-r3", action="store_true", help="story regime halted (yield floor)")
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=c.N_BOOTSTRAP)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    matched = load_matched(args.matched_dir)
    regimes = [r for r in c.REGIMES if not (args.no_r3 and r == "r3")]
    for model in args.models.split(","):
        assert model in c.MODELS, model
        # ONE slim bundle load per (model, regime), shared across both arms
        bundles = {r: load_regime_bundle(args.turnstore_dir, model, r) for r in regimes}
        for arm in args.arms.split(","):
            assert arm in c.ARMS, arm
            run_model_arm(
                bundles,
                matched,
                model,
                arm,
                args.out_dir,
                args.preds_dir,
                seed=args.seed,
                null_draws=args.null_draws,
                n_boot=args.n_boot,
                include_r3=not args.no_r3,
            )
    print("[done] cross-regime transfer complete", flush=True)


if __name__ == "__main__":
    main()
