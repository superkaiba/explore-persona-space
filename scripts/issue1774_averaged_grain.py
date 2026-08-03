"""#1774 free-analysis: persona-AVERAGED-grain per-trait readability table (all 4 arms).

The committed trait table (eval_results/issue_1774/channels/*.json,
``per_trait_heldout_r2``) is PER-ANSWER grain. This script produces the
persona-averaged companion: group held-out OOF predictions + targets by
prefix id (same 6-fold conv_id-grouped folds, fold seed 0, as the fit
battery), average predictions and targets within each prefix (mirroring
``issue1774_fit_battery.step_fits``'s averaged-grain convention: avg_pred =
mean of OOF rows over the prefix's eval rows in the held-out fold; avg_true
= the prefix's all-row target mean), project both onto the #779 r_B trait
directions, and report held-out R² along each trait direction at the
averaged grain, per arm × layer.

Row-alignment guard: the staged OOF array must reproduce the committed
fit_battery ``r2_per_context_pooled_oof`` within 1e-4 before any projection
(a misaligned projection is worse than no table).

Usage: OMP_NUM_THREADS=8 ... uv run python scripts/issue1774_averaged_grain.py
       [--layers 14,18,19] [--out-root D]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE numpy/torch import.
load_dotenv()

import numpy as np  # noqa: E402

import issue1774_common as c  # noqa: E402
from issue1774_aggregate import _stage_if_missing  # noqa: E402

R2_ALIGNMENT_TOL = 1e-4


def _fold_prefix_averages(
    oof: np.ndarray,
    Y: np.ndarray,
    prefix_ids: np.ndarray,
    folds: list[np.ndarray],
    eval_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Per-held-out-prefix (avg_pred, avg_true) rows + per-fold row slices.

    Mirrors ``step_fits``: avg_pred = mean of OOF rows over the prefix's eval
    rows in its held-out fold; avg_true = the prefix's ALL-row target mean.
    Group folds ⇒ each prefix appears in exactly one test fold (asserted).
    """
    prefix_rows: dict[str, list[int]] = {}
    for i, p in enumerate(prefix_ids):
        prefix_rows.setdefault(str(p), []).append(i)
    avg_targets = {p: Y[np.asarray(ix)].mean(0) for p, ix in prefix_rows.items()}

    seen: set[str] = set()
    avg_pred_rows, avg_true_rows, fold_slices = [], [], []
    pos = 0
    for te in folds:
        te_eval = te[eval_mask[te]]
        held = sorted({str(p) for p in prefix_ids[te]})
        n_before = pos
        for p in held:
            assert p not in seen, f"prefix {p} appears in >1 test fold (not group folds?)"
            seen.add(p)
            ix = np.intersect1d(np.asarray(prefix_rows[p]), te_eval)
            if ix.size == 0:
                continue
            avg_pred_rows.append(oof[ix].mean(0))
            avg_true_rows.append(avg_targets[p])
            pos += 1
        fold_slices.append(np.arange(n_before, pos))
    return np.stack(avg_pred_rows), np.stack(avg_true_rows), fold_slices


def _trait_r2(
    pred: np.ndarray, true: np.ndarray, fold_slices: list[np.ndarray], dirs: np.ndarray
) -> tuple[list[float], list[float]]:
    """(fold-accumulated, pooled) held-out R² per direction column of ``dirs``.

    Fold-accumulated mirrors the per-answer ``per_trait_heldout_r2``
    convention (res/tot summed per fold, per-fold test mean in tot); pooled
    uses the global prefix mean.
    """
    pt = pred @ dirs  # (n_prefixes, n_dirs)
    yt = true @ dirs
    res = np.zeros(dirs.shape[1])
    tot = np.zeros(dirs.shape[1])
    for sl in fold_slices:
        if sl.size == 0:
            continue
        res += ((yt[sl] - pt[sl]) ** 2).sum(0)
        tot += ((yt[sl] - yt[sl].mean(0, keepdims=True)) ** 2).sum(0)
    fold_acc = [float(1.0 - res[j] / max(tot[j], 1e-30)) for j in range(dirs.shape[1])]
    res_p = ((yt - pt) ** 2).sum(0)
    tot_p = ((yt - yt.mean(0, keepdims=True)) ** 2).sum(0)
    pooled = [float(1.0 - res_p[j] / max(tot_p[j], 1e-30)) for j in range(dirs.shape[1])]
    return fold_acc, pooled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", default="14,18,19")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    t0 = time.time()
    rows = c.load_manifest()
    reg = json.loads((c.eval_out(args.out_root) / "registry/folds.json").read_text())
    fit_idx = np.asarray(reg["fit_manifest_indices"], dtype=np.int64)
    folds = [np.asarray(f, dtype=np.int64) for f in reg["folds"]]
    assert fit_idx.size == c.EXPECTED_FIT_ROWS, fit_idx.size
    cover = np.sort(np.concatenate(folds))
    assert np.array_equal(cover, np.arange(fit_idx.size)), "folds do not partition fit rows"
    prefix_ids = np.asarray([str(rows[i].get("prefix_id", "")) for i in fit_idx])
    # loro (query_avg) eval mask: singleton prefixes excluded (mirrors Designs)
    _, counts_inv = np.unique(prefix_ids, return_inverse=True)
    counts = np.bincount(counts_inv)
    loro_keep = counts[counts_inv] >= 2

    c.stage_rb_bank()
    op_dir = _stage_if_missing(
        c.data_out(args.out_root) / "operators", f"{c.HF_UPLOAD_PREFIX}/operators", True
    )

    result: dict = {"per_arm_layer": {}}
    for layer in layers:
        Y = np.asarray(c.load_summary_rows(c.CELL, "t1", layer)[fit_idx], dtype=np.float64)
        rb = c.load_rb_bank(layer)
        dirs = np.stack([rb[t] / np.linalg.norm(rb[t]) for t in c.TRAITS], axis=1)  # (D, 3)
        for arm in c.ARMS:
            key = f"{arm}_L{layer}"
            oof_path = op_dir / f"oof_pred_{arm}_L{layer}.npy"
            oof = np.asarray(np.load(oof_path), dtype=np.float64)
            assert oof.shape == (fit_idx.size, c.HIDDEN_DIM), (key, oof.shape)
            eval_mask = loro_keep if arm == "arm_query_avg" else np.ones(fit_idx.size, dtype=bool)
            # row-alignment guard vs the committed per-answer pooled OOF R²
            battery = json.loads(
                (c.eval_out(args.out_root) / "fit_battery" / f"{key}.json").read_text()
            )
            r2_committed = float(battery["r2_per_context_pooled_oof"])
            r2_here = c.r2_score(Y[eval_mask], oof[eval_mask])
            assert abs(r2_here - r2_committed) < R2_ALIGNMENT_TOL, (
                f"{key}: OOF/store row alignment failed — recomputed per-answer pooled "
                f"R² {r2_here:.6f} != committed {r2_committed:.6f}"
            )
            avg_pred, avg_true, fold_slices = _fold_prefix_averages(
                oof, Y, prefix_ids, folds, eval_mask
            )
            fold_acc, pooled = _trait_r2(avg_pred, avg_true, fold_slices, dirs)
            overall_pooled = c.r2_score(avg_true, avg_pred)
            per_fold_overall = [
                c.r2_score(avg_true[sl], avg_pred[sl]) for sl in fold_slices if sl.size
            ]
            result["per_arm_layer"][key] = {
                "arm": arm,
                "layer": layer,
                "n_prefixes": int(avg_pred.shape[0]),
                "row_alignment_check": {
                    "recomputed_r2_per_context_pooled_oof": float(r2_here),
                    "committed_r2_per_context_pooled_oof": r2_committed,
                    "tol": R2_ALIGNMENT_TOL,
                },
                "r2_averaged_overall_pooled": float(overall_pooled),
                "r2_averaged_overall_fold_mean": float(np.mean(per_fold_overall)),
                "per_trait_r2_averaged_foldacc": dict(zip(c.TRAITS, fold_acc, strict=True)),
                "per_trait_r2_averaged_pooled": dict(zip(c.TRAITS, pooled, strict=True)),
            }
            print(
                f"[avg-grain] {key} n_prefixes={avg_pred.shape[0]} "
                f"overall_pooled={overall_pooled:.4f} "
                f"traits(foldacc)="
                + ", ".join(f"{t}={v:.4f}" for t, v in zip(c.TRAITS, fold_acc, strict=True))
                + f" elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    result["meta"] = c.repro_meta({"script": "scripts/issue1774_averaged_grain.py"})
    result["conventions"] = {
        "grain": "persona-averaged: per held-out prefix, avg_pred = mean of OOF rows over "
        "the prefix's eval rows in its held-out fold; avg_true = the prefix's ALL-row "
        "target mean (step_fits r2_averaged convention)",
        "folds": "registry/folds.json — 6 conv_id-grouped folds, fold seed 0 (fit battery)",
        "trait_directions": "issue779 r_B bank, row layer-1, unit-normalized "
        "(issue1774_common.load_rb_bank)",
        "per_trait_r2_averaged_foldacc": "res/tot accumulated per fold with per-fold test "
        "mean in tot — mirrors channels per_trait_heldout_r2 (per-answer) convention",
        "per_trait_r2_averaged_pooled": "single pooled R² over all prefixes, global mean",
        "anchor": "r2_averaged_overall_fold_mean anchors against the fit_battery per-fold "
        "r2_averaged values (prefix_end ~0.174 / query_avg ~0.572 / context ~0.80 at L14)",
    }
    out = c.eval_out(args.out_root) / "averaged_grain/averaged_grain_trait_table.json"
    c.write_json_atomic(out, result)
    print(f"[avg-grain] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
