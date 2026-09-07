"""Single-layer, CPU-only exact fair-protocol replay with per-context persistence.

Config must explicitly authorize fit execution. Availability/index staging lives in
issue2669_probe_replay.py. Shared #1739 whitening, map, target scaling and ridge
functions remain the numerical implementation. Only unused prefix arrays are omitted.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_1739 import arms, fits, store_io  # noqa: E402
from scripts.issue1739_fits import LabeledTable, arrays_dim  # noqa: E402
from scripts.issue1739_jobd_r2aug import _pool_zscored_dv, build_pool  # noqa: E402
from scripts.issue1739_result2fair_score import _wc_eval_mask, _wc_fold_ids  # noqa: E402

METHOD_ARM = {
    "regression_ctx": "arm4_ridge_ctx",
    "reg_map_linear": "arm7_map_ridge_pred",
    "reg_oracle": "arm12_oracle_reg",
}


def load_context_answer(store: Path, dv_path: Path, layer: int, split: str) -> LabeledTable:
    """Reproduce _load_labeled row order and fp16 means, omitting unused prefix arrays."""
    arrays, meta = store_io.load_summaries(
        store, ("context_end", "t1"), (layer,), hidden_dim=arrays_dim(store, [layer])
    )
    by_ctx = {
        r["context_id"]: r
        for r in json.loads(dv_path.read_text())["rows"]
        if r["dv"] is not None and r["split"] == split
    }
    positions = {}
    for i, row in enumerate(meta):
        cid = row["context_id"]
        if cid in by_ctx:
            positions.setdefault(cid, []).append(i)
    if set(positions) != set(by_ctx):
        raise ValueError("Capture/DV full-cohort join mismatch")
    order = list(positions)
    if not all(by_ctx[c].get("group_key") for c in order):
        raise ValueError("Missing group key")
    first = np.asarray([positions[c][0] for c in order])
    z_ctx = arrays[("context_end", layer)][first][None]
    z_ans = np.stack([arrays[("t1", layer)][positions[c]].mean(axis=0) for c in order])[None]
    rungs = [by_ctx[c]["rung"] for c in order]
    return LabeledTable(
        z_by_variant={"context_end": z_ctx},
        z_ans=z_ans,
        dv=np.asarray([by_ctx[c]["dv"] for c in order], dtype=float),
        groups=[by_ctx[c]["group_key"] for c in order],
        ctx_order=order,
        rungs=sorted(set(rungs)),
        row_rungs=rungs,
        per_rollout=None,
        ans_rows=None,
        ans_row_ctx=None,
        ans_row_k=None,
    )


def source_points(root: Path, behavior: str, layer: int) -> list[dict]:
    """Use artifact-declared layers and scores, never manuscript prose or own argmax."""
    records = []
    for name in ("result2_fair_points.json", "reg_oracle_points.json"):
        records.extend(json.loads((root / name).read_text())["points"])
    return [
        r
        for r in records
        if r["behavior"] == behavior
        and r["method"] in METHOD_ARM
        and r["layer"] == layer
        and r["setting"] != "pvsynth"
    ]


def run(config_path: Path) -> dict:
    """Replay one frozen layer only after explicit local-fit authorization in config."""
    config = json.loads(config_path.read_text())
    if config.get("fit_authorized") is not True:
        raise ValueError("Fit is not authorized by this concrete config")
    out = Path(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Require fresh replay output directory")
    producer_source = Path(__file__).read_bytes()
    (out / "producer_script.py").write_bytes(producer_source)
    started = time.time()
    behavior = config["behavior"]
    layer = config["layer"]
    reference = source_points(Path(config["reference_root"]), behavior, layer)
    methods = config["methods"]
    roster = [METHOD_ARM[m] for m in methods]
    if not all(any(r["method"] == m for r in reference) for m in methods):
        raise ValueError("Requested method/layer absent from frozen reference")
    dv_path = Path(config["dv_path"])
    wc_dv_path = Path(config["wildchat_dv_path"])
    paper = json.loads(
        (Path(config["reference_root"]) / behavior / "all_arms_spearman.json").read_text()
    )["meta"]
    for path, key in [(dv_path, "train_dv"), (wc_dv_path, "wcrung_dv")]:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != paper["input_sha256"][paper["input_paths"][key]]
        ):
            raise ValueError("Reference label hash changed")
    tr = load_context_answer(Path(config["store"]), dv_path, layer, "train")
    ev = load_context_answer(Path(config["store"]), dv_path, layer, "eval")
    wc = load_context_answer(Path(config["wildchat_store"]), wc_dv_path, layer, "eval")
    ua, um = store_io.load_summaries(
        Path(config["u_store"]), ("context_end", "t1"), (layer,), hidden_dim=3584
    )
    loaded = SimpleNamespace(
        behavior=behavior,
        tbl=tr,
        u_arrays=ua,
        u_fit_rows=np.flatnonzero(store_io.fit_pool_mask(um)),
    )
    args = SimpleNamespace(regime="e1", draw=0, seed=0)
    x, y, _, _, _ = build_pool(args, loaded, "context_end", [layer], "add")
    wh = fits.fit_whitening(x, device="cpu", seed=0)
    (out / "whitening_metadata.json").write_text(
        json.dumps(
            {
                "layer": layer,
                "gamma": wh.gamma.tolist(),
                "pool_rows": int(x.shape[1]),
                "shared_helper": "fit_whitening",
            },
            indent=2,
        )
        + "\n"
    )
    mapfit = None
    if "reg_map_linear" in methods:
        mapfit = fits.fit_linear_map(
            fits.apply_whitening(x, wh), fits.apply_whitening(y, wh), device="cpu"
        )
    if mapfit is not None:
        (out / "map_diagnostics.json").write_text(
            json.dumps(
                {
                    "source": "computed_in_this_replay",
                    "layer": layer,
                    "diagnostics": mapfit.diagnostics,
                    "map_selected_lambda": None,
                    "map_lambda_note": "Original primal helper does not expose selected lambda; weights and selection grid are persisted",
                    "lambda_grid": list(fits.RIDGE_LAMBDAS),
                },
                indent=2,
            )
            + "\n"
        )
        np.savez(
            out / "map_and_whitening.npz",
            w=mapfit.w,
            x_mu=mapfit.x_mu,
            x_sd=mapfit.x_sd,
            y_mu=mapfit.y_mu,
            wh_mu=wh.mu,
            wh_w=wh.w,
            wh_gamma=wh.gamma,
        )
    del x, y, ua, loaded
    gc.collect()
    n = len(tr.ctx_order)
    wc_eval = np.flatnonzero(_wc_eval_mask(wc.ctx_order))
    wc_train = np.flatnonzero(~_wc_eval_mask(wc.ctx_order))
    cell = fits.realize_budget_cell(
        tr.groups, budget_l=8000 if behavior == "evil" else 16000, draw=0, seed=0
    )
    rows = np.concatenate([cell.row_idx, n + wc_train])
    targets = _pool_zscored_dv(np.concatenate([tr.dv, wc.dv]), cell.row_idx, n + wc_train)
    with np.load(config["rb_bank"], allow_pickle=False) as bank:
        bank_layers = list(map(int, bank["layers"]))
        rb = np.asarray(bank["rb"][[bank_layers.index(layer)]], dtype=np.float64)
    data = arms.CellData(
        z_ctx=np.concatenate(
            [
                fits.apply_whitening(tr.z_by_variant["context_end"], wh),
                fits.apply_whitening(wc.z_by_variant["context_end"], wh),
            ],
            axis=1,
        ),
        z_ans=np.concatenate(
            [fits.apply_whitening(tr.z_ans, wh), fits.apply_whitening(wc.z_ans, wh)], axis=1
        ),
        dv=targets,
        rb=np.einsum("ld,lde->le", rb, wh.w),
        mapfit=mapfit,
        layers=(layer,),
    )
    oof = fits.BudgetCell(
        row_idx=rows,
        fold_ids=np.concatenate(
            [cell.fold_ids, _wc_fold_ids([wc.ctx_order[i] for i in wc_train], 5)]
        ),
        n_folds=5,
        budget_l=cell.budget_l,
        draw=0,
        seed=0,
        fold_scheme=f"fair-union-{cell.fold_scheme}",
    )
    metrics = []

    def save(ids, dv, rungs, scores, phase):
        with (out / f"{phase}.jsonl").open("x") as handle:
            for i, cid in enumerate(ids):
                handle.write(
                    json.dumps(
                        {
                            "context_id": cid,
                            "dv": float(dv[i]),
                            "rung": rungs[i],
                            "scores": {m: float(scores[METHOD_ARM[m]][0, i]) for m in methods},
                        }
                    )
                    + "\n"
                )
        for rung in sorted(set(rungs)):
            mask = np.asarray(rungs) == rung
            for m in methods:
                rho = float(
                    arms.spearman_rows(scores[METHOD_ARM[m]][:, mask], np.asarray(dv)[mask])[0]
                )
                matches = [r for r in reference if r["method"] == m and r["setting"] == rung]
                if len(matches) != 1:
                    raise ValueError("Nonunique reference point")
                ref = matches[0]
                if int(mask.sum()) != ref["n_eval"]:
                    raise ValueError("Reference cohort count mismatch")
                metrics.append(
                    {
                        "method": m,
                        "rung": rung,
                        "rho": rho,
                        "reference_rho": ref["rho"],
                        "absolute_delta": abs(rho - ref["rho"]),
                        "n": int(mask.sum()),
                    }
                )
        print(json.dumps({"phase": phase, "elapsed_seconds": time.time() - started}), flush=True)

    selected_lambdas = []
    with fits.capture_selected_lambdas(selected_lambdas):
        scores, skips = arms.run_cell(data, oof, arms=roster, device="cpu")
    (out / "ridge_selected_lambdas.json").write_text(json.dumps(selected_lambdas, indent=2) + "\n")
    if skips:
        raise ValueError(skips)
    save(
        [tr.ctx_order[i] for i in cell.row_idx],
        tr.dv[cell.row_idx],
        ["train"] * len(cell.row_idx),
        {a: s[:, : len(cell.row_idx)] for a, s in scores.items()},
        "id",
    )
    del scores
    full = fits.BudgetCell(
        row_idx=rows,
        fold_ids=np.zeros(len(rows), dtype=int),
        n_folds=1,
        budget_l=cell.budget_l,
        draw=0,
        seed=0,
        fold_scheme="fair-union-full",
    )
    for phase, table, idx in [("generic", wc, wc_eval), ("ood", ev, np.arange(len(ev.ctx_order)))]:
        with fits.capture_selected_lambdas(selected_lambdas):
            scores, skips = arms.run_transfer_cell(
                data,
                full,
                fits.apply_whitening(table.z_by_variant["context_end"][:, idx], wh),
                table.dv[idx],
                za_ev=fits.apply_whitening(table.z_ans[:, idx], wh),
                arms=roster,
                device="cpu",
                ridge_folds=(0,),
            )
        (out / "ridge_selected_lambdas.json").write_text(
            json.dumps(selected_lambdas, indent=2) + "\n"
        )
        if skips:
            raise ValueError(skips)
        save(
            [table.ctx_order[i] for i in idx],
            table.dv[idx],
            [table.row_rungs[i] for i in idx],
            scores,
            phase,
        )
        del scores
    result = {
        "metrics": metrics,
        "id_protocol_caveat": "Original map and whitening use all ID context/answer pairs before readout folds; ID map is transductive, only behavior labels held out. Generic/OOD excluded from fit pools.",
        "wall_seconds": time.time() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "parity_pass": all(m["absolute_delta"] <= config["parity_abs_tolerance"] for m in metrics),
        "config": config,
        "script_sha256": hashlib.sha256(producer_source).hexdigest(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["parity_pass"]:
        raise ValueError("Aggregate parity failed; do not use recovered subset scores")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
