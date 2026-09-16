"""Matched context-covariance control for the historical #1739 fair readout.

Reuses cached activations and labels only. No model generation or judging.
All arms retain the historical train-only coordinate standardization and GCV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np

from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
from explore_persona_space.experiments.issue_1739.constants import RIDGE_LAMBDAS
from scripts.issue1739_claim4_fold import group_bootstrap_rhos
from scripts.issue1739_fits import _load_labeled
from scripts.issue1739_jobd_r2aug import LMAX, _pool_zscored_dv
from scripts.issue1739_result2fair_score import _wc_eval_mask
from scripts.issue1739_wcrung_arms import modal_frozen_layers

LAYERS = {"evil": (20, 18), "sycophancy": (19, 20), "hallucination": (20,)}
ARM_NAMES = ("unwhitened", "generic_covariance", "union_covariance", "mapped_answer")
# Source: scripts/issue1739_r2v2_run.py WIDE_RIDGE_LAMBDAS.
WIDE_GRID = (*RIDGE_LAMBDAS, 10000.0, 100000.0, 1000000.0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def progress(args, phase, **extra):
    value = {"phase": phase, "time": time.time(), "source_sha": args.source_sha, **extra}
    write_json(args.out / "progress.json", value)
    print(json.dumps(value), flush=True)


def load_table(store, labels, layer, split, *, answers=False):
    return _load_labeled(
        store,
        labels,
        [layer],
        config=split,
        need_rollout_rows=False,
        context_variants=("context_end",),
        include_answers=answers,
    )


def summarize(pred, dv, ids, groups, rungs, *, n_boot):
    result = []
    for rung in sorted(set(rungs)):
        ix = np.flatnonzero(np.asarray(rungs) == rung)
        if len(ix) < 3 or np.ptp(dv[ix]) == 0:
            result.append({"rung": rung, "n": len(ix), "skipped": "too few or constant labels"})
            continue
        rho = arms.spearman_rows(pred[:, ix], dv[ix])
        if not np.isfinite(rho).all():
            raise ValueError(f"nonfinite correlations: {rung}")
        boots, ng = group_bootstrap_rhos(
            pred[:, ix],
            dv[ix],
            np.asarray(groups)[ix],
            n_boot=n_boot,
            rng=np.random.default_rng(1739),
        )
        # Undefined resamples (e.g. all tied labels) are reported, never filled with zero.
        valid = np.isfinite(boots).all(axis=0)
        if valid.sum() < 0.95 * n_boot:
            raise ValueError(f"too many undefined bootstrap draws: {rung}: {valid.sum()}/{n_boot}")
        boots = boots[:, valid]
        differences = {}
        for a, b in ((1, 0), (2, 0), (3, 2), (3, 1)):
            differences[f"{ARM_NAMES[a]}_minus_{ARM_NAMES[b]}"] = {
                "delta": float(rho[a] - rho[b]),
                "ci95": np.quantile(boots[a] - boots[b], [0.025, 0.975]).tolist(),
            }
        result.append(
            {
                "rung": rung,
                "n": len(ix),
                "n_groups": ng,
                "valid_bootstrap_draws": int(valid.sum()),
                "arms": {
                    a: {
                        "rho": float(rho[i]),
                        "ci95": np.quantile(boots[i], [0.025, 0.975]).tolist(),
                    }
                    for i, a in enumerate(ARM_NAMES)
                },
                "differences": differences,
            }
        )
    return result


def run_cell(args, behavior, layer):
    out = args.out / f"{behavior}_L{layer:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError(f"refusing to overwrite completed cell: {out}")
    progress(args, "loading", behavior=behavior, layer=layer)
    labels = args.repo / "eval_results/issue_1739/dv_dataset" / behavior / "labeling.json"
    wc_labels = (
        args.repo / "eval_results/issue_1739/wildchat_rung/dv_dataset" / behavior / "labeling.json"
    )
    history = json.loads(
        (
            args.repo / f"eval_results/issue_1739/result2_fair/{behavior}/all_arms_spearman.json"
        ).read_text()
    )
    known = set(history["meta"]["input_sha256"].values())
    for path in (labels, wc_labels):
        if sha256(path) not in known:
            raise ValueError(f"historical label hash mismatch: {path}")
    tr = load_table(
        args.store_root / f"{behavior}_labeling", labels, layer, "config_a", answers=True
    )
    ev = load_table(args.store_root / f"{behavior}_labeling", labels, layer, "config_b")
    wc = load_table(args.store_root / "wildchat", wc_labels, layer, "config_b")
    u, umeta = store_io.load_summaries(args.store_root / "u_store", ("context_end", "t1"), (layer,))
    ui = np.flatnonzero(store_io.fit_pool_mask(umeta))
    assert len(ui) == 18793, len(ui)
    xg = u[("context_end", layer)][ui][None]
    yg = u[("t1", layer)][ui][None]
    x = np.concatenate([xg, tr.z_by_variant["context_end"]], axis=1)
    y = np.concatenate([yg, tr.z_ans], axis=1)
    del u, yg
    wc_eval = _wc_eval_mask(wc.ctx_order)
    wc_train = np.flatnonzero(~wc_eval)
    budget = fits.realize_budget_cell(tr.groups, budget_l=LMAX[behavior], draw=0, seed=0)
    n_tr = len(tr.ctx_order)
    rows = np.concatenate([budget.row_idx, n_tr + wc_train])
    ctx = np.concatenate([tr.z_by_variant["context_end"], wc.z_by_variant["context_end"]], axis=1)[
        :, rows
    ]
    target = _pool_zscored_dv(np.concatenate([tr.dv, wc.dv]), budget.row_idx, n_tr + wc_train)[rows]
    eval_ctx = np.concatenate(
        [ev.z_by_variant["context_end"], wc.z_by_variant["context_end"][:, wc_eval]], axis=1
    )
    ids = ev.ctx_order + np.asarray(wc.ctx_order)[wc_eval].tolist()
    groups = ev.groups + np.asarray(wc.groups)[wc_eval].tolist()
    rungs = ev.row_rungs + ["wildchat_rung"] * int(wc_eval.sum())
    dv = np.concatenate([ev.dv, wc.dv[wc_eval]])
    fit_ids = set(tr.ctx_order) | set(np.asarray(wc.ctx_order)[wc_train])
    assert not fit_ids.intersection(ids), "evaluation IDs overlap eliciting/map/readout rows"
    # Hallucination group keys are answer/entity strings (including "0", "1").
    # Shared keys are a distribution-overlap diagnostic, not duplicate context IDs.
    overlap_groups = set(tr.groups).intersection(ev.groups)
    uids = {str(umeta[i].get("context_id", umeta[i].get("id", ""))) for i in ui}
    assert not (uids - {""}).intersection(ids), "generic fit IDs overlap evaluation"
    progress(args, "generic_whitening", behavior=behavior, layer=layer)
    whg = fits.fit_whitening(xg, device="cpu", seed=0)
    del xg
    progress(args, "union_whitening", behavior=behavior, layer=layer)
    wh = fits.fit_whitening(x, device="cpu", seed=0)
    xw, yw = fits.apply_whitening(x, wh), fits.apply_whitening(y, wh)
    del x, y
    progress(args, "map_fit", behavior=behavior, layer=layer)
    mapfit = fits.fit_linear_map(xw, yw, device="cpu", seed=0)
    del xw, yw
    z, ze = fits.apply_whitening(ctx, wh), fits.apply_whitening(eval_ctx, wh)
    # Stack independent feature transforms so ridge's shared batched engine handles all arms.
    features = np.concatenate([ctx, fits.apply_whitening(ctx, whg), z, fits.apply_map(z, mapfit)])
    eval_features = np.concatenate(
        [eval_ctx, fits.apply_whitening(eval_ctx, whg), ze, fits.apply_map(ze, mapfit)]
    )
    targets = np.broadcast_to(target[None, :, None], (4, len(target), 1)).copy()
    write_json(out / "map_diagnostics.json", mapfit.diagnostics)
    np.savez(
        out / "transforms.npz",
        generic_mu=whg.mu,
        generic_w=whg.w,
        generic_gamma=whg.gamma,
        union_mu=wh.mu,
        union_w=wh.w,
        union_gamma=wh.gamma,
        map_w=mapfit.w,
        map_x_mu=mapfit.x_mu,
        map_x_sd=mapfit.x_sd,
        map_y_mu=mapfit.y_mu,
    )
    summaries = {}
    historical_pred = None
    for grid_name, grid in (("historical_grid", RIDGE_LAMBDAS), ("wide_grid", WIDE_GRID)):
        progress(args, "ridge_fit", behavior=behavior, layer=layer, grid=grid_name)
        selected = []
        with fits.capture_selected_lambdas(selected):
            pred = fits.ridge_gcv_predict_per_target(
                features, targets, [eval_features], lambdas=grid, device="cpu", layer_chunk=1
            )[0][:, :, 0]
        np.savez(
            out / f"predictions_{grid_name}.npz",
            predictions=pred,
            dv=dv,
            context_ids=np.asarray(ids),
            groups=np.asarray(groups),
            rungs=np.asarray(rungs),
            arms=np.asarray(ARM_NAMES),
        )
        progress(args, "bootstrap", behavior=behavior, layer=layer, grid=grid_name)
        summaries[grid_name] = {
            "lambda_diagnostics": dict(zip(ARM_NAMES, selected, strict=True)),
            "results": summarize(pred, dv, ids, groups, rungs, n_boot=args.n_boot),
        }
        if grid_name == "historical_grid":
            historical_pred = pred
    parity = []
    for old in history["per_layer_rows"]:
        arm = {"arm4_ridge_ctx": "union_covariance", "arm7_map_ridge_pred": "mapped_answer"}.get(
            old["arm"]
        )
        if arm is None or old.get("map_kind") != "linear" or old.get("variant") != "context_end":
            continue
        matches = [
            r for r in summaries["historical_grid"]["results"] if r["rung"] == old["eval_rung"]
        ]
        if old["eval_rung"] == "ood":
            current = float(
                arms.spearman_rows(historical_pred[:, : len(ev.dv)], ev.dv)[ARM_NAMES.index(arm)]
            )
        elif matches and "skipped" not in matches[0]:
            current = matches[0]["arms"][arm]["rho"]
        else:
            continue
        previous = old["rho_per_layer"][old["layers"].index(layer)]
        delta = current - previous
        parity.append({"rung": old["eval_rung"], "arm": arm, "rho_difference": delta})
    for old in history.get("transfer_rows", []):
        arm = {"arm4_ridge_ctx": "union_covariance", "arm7_map_ridge_pred": "mapped_answer"}.get(
            old["arm"]
        )
        if (
            arm is None
            or old.get("layer") != layer
            or old.get("map_kind") != "linear"
            or old.get("variant") != "context_end"
        ):
            continue
        matches = [
            r for r in summaries["historical_grid"]["results"] if r["rung"] == old["eval_rung"]
        ]
        if matches and "skipped" not in matches[0]:
            parity.append(
                {
                    "rung": old["eval_rung"],
                    "arm": arm,
                    "rho_difference": matches[0]["arms"][arm]["rho"] - old["rho_frozen"],
                }
            )
    if not parity:
        raise ValueError("no historical parity cells found; verify historical arm names")
    result = {
        "behavior": behavior,
        "layer": layer,
        "primary_layer": layer == LAYERS[behavior][0],
        "source_sha": args.source_sha,
        "input_manifest_sha256": sha256(args.store_root / "manifest.json"),
        "n_generic": len(ui),
        "n_eliciting": n_tr,
        "n_readout": len(rows),
        "n_eval": len(ids),
        "generic_gamma": whg.gamma.tolist(),
        "union_gamma": wh.gamma.tolist(),
        "train_eval_shared_group_keys": len(overlap_groups),
        "eval_rows_with_shared_group_key": sum(g in overlap_groups for g in ev.groups),
        "generic_wildchat_exclusion": "saved wcrung_digest.json: content-hash exclusion against #1092",
        "label_sha256": {str(p): sha256(p) for p in (labels, wc_labels)},
        "parity": parity,
        "summaries": summaries,
        "bootstrap_scope": "evaluation groups conditional on these fitted predictors",
    }
    write_json(out / "results.json", result)
    if max(abs(p["rho_difference"]) for p in parity) > 1e-4:
        raise ValueError(f"historical parity failed: {parity}")
    write_json(
        out / "complete.json",
        {
            "source_sha": args.source_sha,
            "result_sha256": sha256(out / "results.json"),
            "artifact_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        },
    )
    progress(args, "cell_complete", behavior=behavior, layer=layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--behaviors", nargs="+", choices=list(LAYERS), default=list(LAYERS))
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    args.source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest_sha = sha256(args.store_root / "manifest.json")
    manifest = json.loads((args.store_root / "manifest.json").read_text())
    assert manifest["complete"] is True
    for name, meta in manifest["files"].items():
        path = args.store_root / name
        assert path.stat().st_size == meta["bytes"] and sha256(path) == meta["sha256"], name
    for b in args.behaviors:
        frozen = modal_frozen_layers(
            args.repo / f"eval_results/issue_1739/{b}/arm_results/all_arms_spearman.json",
            variant="context_end",
            regime="e1",
            u_rung_label="full",
        )
        assert frozen["arm7_map_ridge_pred"] == LAYERS[b][0], (b, frozen)
        for layer in LAYERS[b][:1] if args.primary_only else LAYERS[b]:
            out = args.out / f"{b}_L{layer:02d}"
            if (out / "complete.json").is_file():
                done = json.loads((out / "complete.json").read_text())
                result = json.loads((out / "results.json").read_text())
                assert done["source_sha"] == args.source_sha == result["source_sha"]
                assert done["result_sha256"] == sha256(out / "results.json")
                assert result["input_manifest_sha256"] == manifest_sha
                for name, digest in done["artifact_sha256"].items():
                    assert sha256(out / name) == digest, name
                progress(args, "cell_resumed", behavior=b, layer=layer)
                continue
            run_cell(args, b, layer)
    progress(args, "complete")


if __name__ == "__main__":
    main()
