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
from issue1345_fit_cells import (  # noqa: E402
    degenerate_fold_reason,
    load_matched,
    load_regime_bundle,
)

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


def subset_rows(
    xy: dict, keep_ids: list[str], *, smoke: bool = False, label: str = ""
) -> dict | None:
    """Restrict rows to the given conv/story id set (order preserved).

    Under ``smoke`` an EMPTY selection returns None with an informational log
    (the caller skips the consuming read); under production the fail-loud
    assert is unchanged (v3 sweep item: matched-subset drift is a pipeline
    bug at production n, never silently tolerated).
    """
    keep = np.isin(xy["conv_ids"], np.asarray(sorted(set(keep_ids))))
    if smoke and not keep.any():
        print(
            f"[transfer][smoke] SKIP {label or 'subset'}: selected zero rows — "
            "informational (production assert unchanged)",
            flush=True,
        )
        return None
    assert keep.any(), (
        f"subset selected zero rows{f' ({label})' if label else ''} — matched-subset drift"
    )
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


def _sweep_smoke_skip_reason(
    src: dict | None, tgt: dict | None, *, smoke: bool, seed: int
) -> str | None:
    """Smoke-only skip reason for one (src -> tgt) sweep; always None in production.

    A None subset only exists under smoke (see subset_rows); the fold probe
    runs only under smoke — production sweeps are byte-untouched.
    """
    if src is None or tgt is None:
        return "empty matched subset at smoke n"
    if smoke:
        return degenerate_fold_reason(
            src["conv_ids"], n_folds=cm.N_FOLDS, seed=seed, tgt_conv_ids=tgt["conv_ids"]
        )
    return None


def _headline_boot_stub(reason: str) -> dict:
    """verdict_for-compatible NaN stub for a smoke-skipped headline bootstrap."""
    nan_ci = {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "skipped": reason,
        "n_boot": 0,
        "n_groups": 0,
        "unit": "conversation (paired resample across all four statistics)",
        "delta_1to2": dict(nan_ci),
        "delta_2to1": dict(nan_ci),
        "delta_xfer": dict(nan_ci),
        "delta_same": dict(nan_ci),
        "delta_diff": dict(nan_ci),
        "delta_diff_ci_wholly_below_0": False,
    }


def _collect_headline_reads(
    sweep_fn, *, model: str, arm: str, preds_dir: Path, smoke: bool
) -> tuple[dict, str | None]:
    """(reads, skip_reason) for the four headline sweeps; saves L19 preds caches.

    Production keeps the fail-loud full-coverage assert (actionable message);
    under smoke a skipped sweep / partial fold coverage returns a reason the
    caller logs informationally (v3 sweep class).
    """
    reads: dict = {}
    for name, key in {
        "within_r1": ("r1", "r1", "headline"),
        "within_r2": ("r2", "r2", "headline"),
        "x_r1r2": ("r1", "r2", "headline"),
        "x_r2r1": ("r2", "r1", "headline"),
    }.items():
        sw = sweep_fn(*key)
        if isinstance(sw, tuple):  # smoke-only ("skipped", reason) sentinel
            return {}, f"{name}: {sw[1]}"
        if not sw["fitted_l19"].all():
            n_unfit = int((~sw["fitted_l19"]).sum())
            msg = (
                f"{name}: {n_unfit}/{len(sw['fitted_l19'])} held-out rows never received a "
                "fold prediction (grouped-CV folds skipped at this n)"
            )
            if smoke:
                return {}, msg
            raise AssertionError(msg + " — matched-subset/extraction drift at production n")
        reads[name] = (sw["preds_l19"], sw["true_l19"], sw["conv_ids"])
        np.savez(
            preds_dir / f"transfer_{c.MODEL_SLUG[model]}_{arm}_{name}_L19.npz",
            pred=sw["preds_l19"].astype(np.float32),
            true=sw["true_l19"].astype(np.float32),
            conv_ids=sw["conv_ids"],
            layer=np.asarray([L19]),
        )
    return reads, None


def _within_diagonals(sweep_fn, regimes: list[str], include_r3: bool, skipped: dict) -> dict:
    """Within-regime diagonals at each pair grain; smoke-skips recorded in-place."""
    within: dict = {}
    for r in regimes:
        if r == "r3":
            continue
        sw = sweep_fn(r, r, "headline")
        if not isinstance(sw, tuple):
            within[f"{r}@headline"] = sw["r2_by_layer"]
    if include_r3:
        for r in regimes:
            sw = sweep_fn(r, r, "r3pair")
            if isinstance(sw, tuple):
                skipped[f"{r}@r3pair(within)"] = sw[1]
            else:
                within[f"{r}@r3pair"] = sw["r2_by_layer"]
    return within


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
    smoke: bool = False,
) -> None:
    regimes = [r for r in c.REGIMES if include_r3 or r != "r3"]
    full = {r: load_arm_xy(bundles[r], r, arm) for r in regimes}
    shared = matched["shared_r1r2_convs"]
    r3cfg = matched["per_model_r3_pair"].get(model) if include_r3 else None

    def _subset(regime: str, pair_kind: str) -> dict | None:
        label = f"{model}/{arm} {regime}@{pair_kind}"
        if regime in ("r1", "r2"):
            ids = shared if pair_kind == "headline" else r3cfg["r12_convs"]
            return subset_rows(full[regime], ids, smoke=smoke, label=label)
        return subset_rows(full[regime], r3cfg["r3_story_ids"], smoke=smoke, label=label)

    # Cache sweeps keyed by (src_regime, tgt_regime, pair_kind); within = src==tgt.
    # Under smoke a cached value may be the ("skipped", reason) sentinel when
    # the subset is empty or the grouped-CV fold machinery would fit nothing at
    # the realized smoke n (kept=1-3 story grains — v3 sweep class); production
    # never stores sentinels (subset_rows asserts, and no degeneracy probe runs).
    sweeps: dict = {}
    skipped: dict[str, str] = {}

    def _sweep(i: str, j: str, pair_kind: str):
        key = (i, j, pair_kind)
        if key not in sweeps:
            src, tgt = _subset(i, pair_kind), _subset(j, pair_kind)
            reason = _sweep_smoke_skip_reason(src, tgt, smoke=smoke, seed=seed)
            if reason is not None:
                print(
                    f"[transfer][smoke] SKIP sweep {i}->{j}@{pair_kind} ({model}/{arm}): "
                    f"{reason} — informational (production semantics unchanged)",
                    flush=True,
                )
                sweeps[key] = ("skipped", reason)
            else:
                sweeps[key] = transfer_sweep(src, tgt, seed=seed, null_draws=null_draws)
        return sweeps[key]

    def _is_skip(sw) -> bool:
        return isinstance(sw, tuple)

    matrix: dict = {}
    deltas: dict = {}
    for i, j in [(a, b) for a in regimes for b in regimes if a != b]:
        pair_kind = "headline" if {i, j} == {"r1", "r2"} else "r3pair"
        xfer = _sweep(i, j, pair_kind)
        within_j = _sweep(j, j, pair_kind)
        if _is_skip(xfer) or _is_skip(within_j):
            skipped[f"{i}->{j}"] = xfer[1] if _is_skip(xfer) else within_j[1]
            continue
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
    within = _within_diagonals(_sweep, regimes, include_r3, skipped)

    # Headline paired bootstrap from the cached L19 preds (plan §3 CI bound)
    reads, reads_skip_reason = _collect_headline_reads(
        _sweep, model=model, arm=arm, preds_dir=preds_dir, smoke=smoke
    )
    if reads_skip_reason is None:
        boot = paired_delta_bootstrap(reads, n_boot=n_boot, seed=seed + 31)
    else:
        # Smoke-informational stub in the verdict_for-compatible shape (NaN CIs,
        # never a fake verdict): the smoke chain completes; production above
        # either asserts or runs the real bootstrap.
        print(
            f"[transfer][smoke] SKIP headline paired bootstrap ({model}/{arm}): "
            f"{reads_skip_reason} — informational (production assert unchanged)",
            flush=True,
        )
        boot = _headline_boot_stub(reads_skip_reason)

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
        # Smoke-informational skips (empty in production; additive key)
        "skipped_pairs": skipped,
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
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="smoke leg: pairs whose matched subset is empty or whose grouped-CV "
        "folds would fit nothing at smoke n are skipped with a logged reason",
    )
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
                smoke=args.smoke,
            )
    print("[done] cross-regime transfer complete", flush=True)


if __name__ == "__main__":
    main()
