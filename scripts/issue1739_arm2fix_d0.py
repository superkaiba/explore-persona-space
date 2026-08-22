#!/usr/bin/env python3
"""#1739 arm2fix D0 diagnosis probes (plan §4 Leg 2 D0): the P2/P3/P4/P5 driver.

P1 (H-wiring) is a pytest, committed regardless of outcome:
``tests/test_issue1739_arm2fix.py::test_p1_planted_direction_transfer_wiring``
— it PASSES on the current ``run_transfer_cell`` dispatch, so repair R-A
(wiring fix) is a NO-OP and the ladder proceeds on P2/P3/P5 evidence.

- **p2** (VM-side; H-degenerate + H-confound): P-B readout-pool membership
  reconstructed DETERMINISTICALLY from the DV ``labeling.json`` files + the
  scorer's pure group-hash ``_group_side_train`` (no stores, no activations);
  per (behavior, seed, holdout): the transfer split's midpoint
  ``0.5*(max+min)`` over the (per-pool z-scored) readout dv, hi/lo row
  counts, and the per-component composition of each side (each eliciting
  dataset vs the judged WildChat block). Counts are cross-checked against
  the banked ``pb_pools`` records where the banked seed dirs are staged.
- **p3** (VM or pod; H-frozen-layer): reader over the BANKED claim4 arm2
  rows — per-fit ``rho_per_layer`` profiles (flat-vs-peaked, committed
  frozen-layer placement) + per-rung sign structure + the pvsynth transfer
  values; accepts LOCAL paths (stage
  ``issue1739_claim4_controls/<b>/seed*/`` via scoped ``hf_hub_download``
  first). Preds-based per-rung recompute is optional and degrades with a
  recorded note when ``transfer_preds/`` is absent (plan §12 assumption 12).
- **p4** (pod-side; direction stability): across-seed cosines of the arm2
  transfer direction per behavior at the committed frozen layer — in the
  as-realized per-seed whitened bases AND mapped back to raw space
  (``wh.w @ d``, the cross-seed-comparable functional); cosine to the banked
  E1 direction and to the folded train-mode arm2 direction on eliciting
  rows; the v1-vs-component-restricted and quantile direction variants ride
  along so the read applies to the FINAL repaired adapter version. The pure
  core (:func:`_p4_directions`) is unit-tested at tiny shape; only the
  loading boundary (stores) is pod-bound.
- **p5** (VM-side; H-instrument): banked claim4 pvsynth-rung transfer rows
  for arms {1, 4, 7} (+ arm2 for reference) vs each arm's OWN committed
  train-grid band — if the label-matched probes ALSO sit outside their
  bands under the transfer-on-pvsynth read, the old band check is an
  instrument artifact (fit regime AND eval distribution both differ), not
  three independent adapter bugs.

Outputs land under ``<out-root>/`` (default
``eval_results/issue_1739/claim4_controls/arm2fix/d0/``), one JSON per
probe, written atomically the moment the probe completes
(checkpoint-per-phase). Digest-only I/O: the probes read context IDS, dv
VALUES, group keys and counts — never prompt/completion text (the
trigger-dense corpora stay unpaged).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_r2v2_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

PROBES = ("p2", "p3", "p4", "p5")
BEHAVIORS = ("evil", "sycophancy", "hallucination")
# P5 audit arms: the label-matched probes whose band placement tests
# H-instrument, plus arm2 itself as the reference row.
P5_ARMS = ("arm1_ctx_e1", "arm4_ridge_ctx", "arm7_map_ridge_pred", "arm2_ctx_native")
HF_STAGE_HINT = (
    "stage the banked seed dirs first, e.g. huggingface_hub.hf_hub_download("
    "'superkaiba1/explore-persona-space-data', "
    "'issue1739_claim4_controls/<b>/seed<S>/all_arms_spearman.json', "
    "repo_type='dataset', local_dir=<claim4-root parent>)"
)


def _log(msg: str) -> None:
    print(f"[a2fix-d0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    _log(f"wrote {path}")
    return path


def _meta(args) -> dict:
    from scripts.issue1739_fits import _git_commit

    return {
        "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "behaviors": list(args.behaviors),
        "seeds": [int(s) for s in args.seeds],
        "judge_called": False,
    }


# ---------------------------------------------------------------------------
# P2 — pool-membership reconstruction (labeling.json + group hash only)
# ---------------------------------------------------------------------------


def _labeling_arrays(path: Path, *, split: str | None = None):
    """(ids, dv, groups, rungs) arrays for judged rows of one labeling.json.

    Digest-only: reads context_id / dv / group_key / rung / split fields —
    never text. Rows without a kept dv are dropped (drop-never-coerce, the
    loader's convention)."""
    import numpy as np

    payload = json.loads(Path(path).read_text())
    rows = [
        r
        for r in payload["rows"]
        if r.get("dv") is not None and (split is None or r.get("split") == split)
    ]
    return (
        np.asarray([str(r["context_id"]) for r in rows]),
        np.asarray([float(r["dv"]) for r in rows], dtype=np.float64),
        np.asarray([str(r["group_key"]) for r in rows]),
        np.asarray([str(r.get("rung")) for r in rows]),
    )


def _behavior_dv_tables(args, behavior: str, workdir: Path) -> dict:
    """Per-dataset (ids, dv, groups) from the DV labeling files ALONE.

    Mirrors the scorer's table assembly at LABEL grain: train table =
    split=='train' rows of the train DV; eval-rung datasets = split=='eval'
    rows by rung; OOD datasets via the scorer's own ``_prepare_ood_dv`` gate
    (evil; syco is already split=eval; hallucination has none); the WildChat
    block = judged eval-split wcrung rows, train/eval by ``_wc_eval_mask``.
    """

    from scripts import issue1739_r2v2_score as sc
    from scripts.issue1739_result2fair_score import _wc_eval_mask

    train_dv = args.train_dv_root / behavior / "labeling.json"
    wcrung_dv = args.wcrung_dv_root / behavior / "labeling.json"
    missing = [str(p) for p in (train_dv, wcrung_dv) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{behavior}] P2 inputs missing: {missing}")
    tr_ids, tr_dv, tr_groups, _ = _labeling_arrays(train_dv, split="train")
    ev_ids, ev_dv, ev_groups, ev_rungs = _labeling_arrays(train_dv, split="eval")
    ev_by_rung = {}
    for rung in sorted(set(ev_rungs.tolist())):
        m = ev_rungs == rung
        ev_by_rung[rung] = (ev_ids[m], ev_dv[m], ev_groups[m])
    ood_by_rung = {}
    ood_note = None
    if behavior in sc.OOD_SPECS:
        src = Path(args.evil_ood_dv if behavior == "evil" else args.syco_ood_dv)
        dv_path, ood_note = sc._prepare_ood_dv(
            src, workdir, behavior, max_null_frac=args.ood_dv_max_null_frac
        )
        o_ids, o_dv, o_groups, o_rungs = _labeling_arrays(dv_path, split="eval")
        for rung in sorted(set(o_rungs.tolist())):
            m = o_rungs == rung
            ood_by_rung[rung] = (o_ids[m], o_dv[m], o_groups[m])
    wc_ids, wc_dv, wc_groups, _ = _labeling_arrays(wcrung_dv, split="eval")
    wc_eval = _wc_eval_mask([str(c) for c in wc_ids])
    return {
        "train": (tr_ids, tr_dv, tr_groups),
        "ev_by_rung": ev_by_rung,
        "ood_by_rung": ood_by_rung,
        "ood_note": ood_note,
        "wc_train": (wc_ids[~wc_eval], wc_dv[~wc_eval], wc_groups[~wc_eval]),
        "n_wc_eval": int(wc_eval.sum()),
    }


def _banked_pools(args, behavior: str, seed: int) -> dict | None:
    p = args.claim4_root / behavior / f"seed{seed}" / "all_arms_spearman.json"
    if not p.exists():
        return None
    pools = json.loads(p.read_text())["meta"].get("pb_pools") or []
    return {rec["holdout"]: rec for rec in pools if "holdout" in rec}


def run_p2(args) -> dict:
    import numpy as np

    from scripts import issue1739_r2v2_score as sc
    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_jobd_r2aug import LMAX

    workdir = args.out_root / "_dv_work"
    out: dict = {"probe": "p2", "meta": _meta(args), "behaviors": {}}
    for b in args.behaviors:
        tabs = _behavior_dv_tables(args, b, workdir)
        tr_ids, tr_dv, tr_groups = tabs["train"]
        rung_tables = {**tabs["ev_by_rung"], **tabs["ood_by_rung"]}
        wc_ids, wc_dv, _wc_groups = tabs["wc_train"]
        per_seed: dict = {}
        for s in args.seeds:
            cell = fits.realize_budget_cell(tr_groups, budget_l=LMAX[b], draw=args.draw, seed=s)
            truncated = len(cell.row_idx) < len(tr_ids)
            # merged label-grain table: [train-cell | rung datasets... | wc]
            blocks = [("train", tr_dv[cell.row_idx], tr_groups[cell.row_idx])]
            blocks += [(r, rung_tables[r][1], rung_tables[r][2]) for r in sorted(rung_tables)]
            blocks.append(("wildchat_rung", wc_dv, None))
            dv_merged, comp, datasets, at = [], [], [], 0
            wc_idx = None
            for name, dv_arr, groups_arr in blocks:
                idx = np.arange(at, at + len(dv_arr), dtype=np.int64)
                dv_merged.append(dv_arr)
                comp.extend([name] * len(dv_arr))
                if name == "wildchat_rung":
                    wc_idx = idx
                else:
                    datasets.append(sc.DatasetSpec(name=name, rows=idx, groups=groups_arr))
                at += len(dv_arr)
            dv_merged = np.concatenate(dv_merged)
            comp = np.asarray(comp)
            eval_names = [d.name for d in datasets if d.name != "train"]
            banked = _banked_pools(args, b, int(s))
            holdouts: dict = {}
            for h in eval_names:
                pool = sc.assemble_readout_pool(
                    datasets, holdout=h, train_frac=args.train_frac, seed=int(s)
                )
                pool_rows = [pool.train_rows[n] for n in sorted(pool.train_rows)]
                readout = np.concatenate(pool_rows + [wc_idx]).astype(np.int64)
                dv_z = sc._multi_pool_zscored_dv(dv_merged, pool_rows + [wc_idx])
                vals = dv_z[readout]
                mid = 0.5 * (float(vals.max()) + float(vals.min()))
                hi = vals >= mid
                comps = comp[readout]
                hi_by = {c: int(((comps == c) & hi).sum()) for c in sorted(set(comps.tolist()))}
                lo_by = {c: int(((comps == c) & ~hi).sum()) for c in sorted(set(comps.tolist()))}
                n_hi, n_lo = int(hi.sum()), int((~hi).sum())
                rep = {
                    "n_readout": int(readout.size),
                    "midpoint_zdv": mid,
                    "n_hi": n_hi,
                    "n_lo": n_lo,
                    "hi_by_component": hi_by,
                    "lo_by_component": lo_by,
                    "wc_share_of_lo": (lo_by.get("wildchat_rung", 0) / n_lo) if n_lo else None,
                    "wc_share_of_hi": (hi_by.get("wildchat_rung", 0) / n_hi) if n_hi else None,
                    "pool_max_component": str(comps[int(np.argmax(vals))]),
                    "pool_min_component": str(comps[int(np.argmin(vals))]),
                }
                if banked and h in banked:
                    rec = banked[h]
                    recon_train_n = {n: int(len(pool.train_rows[n])) for n in pool.train_rows}
                    diffs = {
                        k: (recon_train_n.get(k), rec["per_dataset_train_n"].get(k))
                        for k in set(recon_train_n) | set(rec["per_dataset_train_n"])
                        if recon_train_n.get(k) != rec["per_dataset_train_n"].get(k)
                    }
                    if int(len(wc_idx)) != int(rec["n_wc_train"]):
                        diffs["n_wc_train"] = (int(len(wc_idx)), int(rec["n_wc_train"]))
                    if int(readout.size) != int(rec["n_readout_total"]):
                        diffs["n_readout_total"] = (int(readout.size), int(rec["n_readout_total"]))
                    rep["banked_crosscheck"] = {
                        "match": not diffs,
                        "diffs (reconstructed, banked)": {k: list(v) for k, v in diffs.items()},
                    }
                else:
                    rep["banked_crosscheck"] = "banked seed summary not staged — not cross-checked"
                holdouts[h] = rep
            per_seed[int(s)] = {
                "n_train_cell": int(len(cell.row_idx)),
                "train_cell_truncated": bool(truncated),
                "holdouts": holdouts,
            }
        wc_lo = [
            h["wc_share_of_lo"]
            for sd in per_seed.values()
            for h in sd["holdouts"].values()
            if h["wc_share_of_lo"] is not None
        ]
        out["behaviors"][b] = {
            "dataset_sizes": {
                "train_table": int(len(tr_ids)),
                **{r: int(len(v[0])) for r, v in sorted(rung_tables.items())},
                "wildchat_train": int(len(wc_ids)),
                "wildchat_eval": tabs["n_wc_eval"],
            },
            "ood_dv_note": tabs["ood_note"],
            "per_seed": per_seed,
            "evidence": {
                "h_degenerate_min_n_hi": min(
                    h["n_hi"] for sd in per_seed.values() for h in sd["holdouts"].values()
                ),
                "h_confound_max_wc_share_of_lo": max(wc_lo) if wc_lo else None,
                "h_confound_min_wc_share_of_lo": min(wc_lo) if wc_lo else None,
            },
            "row_order_caveat": (
                "the train budget cell consumed the WHOLE train table (no truncated "
                "group) — reconstruction is exact at label grain"
                if all(not sd["train_cell_truncated"] for sd in per_seed.values())
                else "budget cell truncated its final group; labeling.json row order "
                "was assumed to match the store order for the truncated rows"
            ),
        }
    return out


# ---------------------------------------------------------------------------
# P3 — banked arm2 layer/sign profile reader (local paths; VM or pod)
# ---------------------------------------------------------------------------


def _profile_stats(rho_per_layer: list[float], frozen_idx: int, layers: list[int]) -> dict:
    import numpy as np

    r = np.asarray(rho_per_layer, dtype=np.float64)
    amax = int(np.nanargmax(r))
    return {
        "max": float(np.nanmax(r)),
        "argmax_layer": int(layers[amax]) if layers else amax,
        "at_frozen": float(r[frozen_idx]) if 0 <= frozen_idx < r.size else None,
        "frozen_is_argmax": bool(amax == frozen_idx),
        "spread_max_minus_min": float(np.nanmax(r) - np.nanmin(r)),
        "std_across_layers": float(np.nanstd(r)),
        "n_negative_layers": int((r < 0).sum()),
    }


def run_p3(args) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    out: dict = {"probe": "p3", "meta": _meta(args), "behaviors": {}, "missing_files": []}
    any_read = False
    for b in args.behaviors:
        per_seed: dict = {}
        for s in args.seeds:
            p = args.claim4_root / b / f"seed{s}" / "all_arms_spearman.json"
            if not p.exists():
                out["missing_files"].append(str(p))
                continue
            payload = json.loads(p.read_text())
            any_read = True
            profiles = {}
            for r in payload.get("per_layer_rows", []):
                if r.get("arm") != "arm2_ctx_native" or r.get("map_variant") not in (None, "true"):
                    continue
                profiles[str(r.get("fit"))] = _profile_stats(
                    r["rho_per_layer"], int(r.get("frozen_layer_idx", -1)), r.get("layers") or []
                )
            primary, pvsynth = {}, []
            for r in payload.get("transfer_rows", []):
                if r.get("arm") != "arm2_ctx_native" or r.get("map_variant") not in (None, "true"):
                    continue
                rung = str(r.get("eval_rung"))
                if r.get("fit") == f"P-B-holdout-{rung}":
                    primary[rung] = {
                        "rho_frozen": r.get("rho_frozen"),
                        "sign": "+" if (r.get("rho_frozen") or 0) >= 0 else "-",
                        "layer": r.get("layer"),
                        "n_eval": r.get("n_eval"),
                    }
                elif rung == "pvsynth":
                    pvsynth.append(float(r["rho_frozen"]))
            preds_note = "preds recompute not requested"
            if args.p3_check_preds:
                preds_note = _p3_preds_recompute(args, arms, np, b, s, primary)
            per_seed[int(s)] = {
                "per_fit_layer_profiles": profiles,
                "primary_rung_rows": primary,
                "pvsynth_rho_mean_over_fits": (float(np.mean(pvsynth)) if pvsynth else None),
                "n_negative_primary_rungs": sum(1 for v in primary.values() if v["sign"] == "-"),
                "preds_recompute": preds_note,
            }
        out["behaviors"][b] = per_seed
    if not any_read:
        raise FileNotFoundError(
            f"P3 read NO banked seed summaries under {args.claim4_root} — {HF_STAGE_HINT}"
        )
    return out


def _p3_preds_recompute(args, arms, np, b: str, s: int, primary: dict) -> str | dict:
    """Optional consistency read: per-rung rho at frozen recomputed from the
    banked per-context preds; degrades with a recorded note when absent
    (plan §12 assumption 12)."""
    checks = {}
    for rung in sorted(primary):
        p = args.claim4_root / b / f"seed{s}" / "transfer_preds" / f"P-B-holdout-{rung}.jsonl"
        if not p.exists():
            return f"transfer_preds absent for seed{s} — degraded to rho_per_layer-only read"
        scores, dv = [], []
        with p.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("arm") == "arm2_ctx_native" and str(r.get("rung")) == rung:
                    scores.append(float(r["score"]))
                    dv.append(float(r["dv"]))
        if not scores:
            return f"no arm2 preds rows for {rung} seed{s} — degraded"
        rho = float(arms.spearman_rows(np.asarray(scores)[None], np.asarray(dv))[0])
        row_rho = primary[rung]["rho_frozen"]
        checks[rung] = {
            "rho_from_preds": rho,
            "rho_row": row_rho,
            "consistent": bool(row_rho is not None and abs(rho - row_rho) < 1e-8),
        }
    return checks


# ---------------------------------------------------------------------------
# P4 — direction stability (pod-side loading; pure core unit-tested)
# ---------------------------------------------------------------------------


def _split_dir(z1, vals, rows):
    """Midpoint-split diff-of-means direction on ONE layer: (n, d) x rows -> (d,)."""

    v = vals[rows]
    mid = 0.5 * (float(v.max()) + float(v.min()))
    hi = v >= mid
    if not hi.any() or hi.all():
        raise RuntimeError("degenerate midpoint split (flat dv)")
    zr = z1[rows]
    return zr[hi].mean(axis=0) - zr[~hi].mean(axis=0), int(hi.sum()), int((~hi).sum())


def _quantile_dir(z1, vals, rows):
    """arm2q-style quantile-split direction (shared thresholds helper)."""

    from explore_persona_space.experiments.issue_1739 import arms

    v = vals[rows]
    q_lo, q_hi = arms.arm2q_thresholds(v)
    hi, lo = v >= q_hi, v <= q_lo
    if not hi.any() or not lo.any():
        raise RuntimeError("degenerate quantile split (flat dv)")
    zr = z1[rows]
    return zr[hi].mean(axis=0) - zr[lo].mean(axis=0), int(hi.sum()), int(lo.sum())


def _p4_directions(z1, dv_raw, datasets, wc_train_rows, elic_cell, *, seed: int, train_frac: float):
    """Pure P4 core at ONE layer: the four transfer-direction variants + the
    folded train-mode arm2 direction on eliciting rows.

    ``z1`` is the (N, d) whitened merged table at the probe layer; pool
    membership rides the scorer's own ``_group_side_train`` hash; dv scaling
    mirrors the transfer machinery (per-pool z-score for the split)."""
    import numpy as np

    from scripts import issue1739_r2v2_score as sc

    pool_rows = []
    for d in datasets:
        rows_d = np.asarray(d.rows, dtype=np.int64)
        groups_d = np.asarray(d.groups)
        side = np.array(
            [sc._group_side_train(d.name, str(g), seed, train_frac) for g in groups_d],
            dtype=bool,
        )
        pool_rows.append(rows_d[side])
    wc = np.asarray(wc_train_rows, dtype=np.int64)
    readout_v1 = np.concatenate(pool_rows + [wc]).astype(np.int64)
    readout_r = np.concatenate(pool_rows).astype(np.int64)
    dv_z = sc._multi_pool_zscored_dv(dv_raw, pool_rows + [wc])
    dirs, counts = {}, {}
    for name, rows, fn in (
        ("v1", readout_v1, _split_dir),
        ("restricted", readout_r, _split_dir),
        ("quantile", readout_v1, _quantile_dir),
        ("quantile-restricted", readout_r, _quantile_dir),
    ):
        vec, n_hi, n_lo = fn(z1, dv_z, rows)
        dirs[name] = vec
        counts[name] = {"n_fit_rows": int(rows.size), "n_hi": n_hi, "n_lo": n_lo}
    # folded train-mode arm2 direction on eliciting rows (committed regime:
    # single-dataset raw dv — affine-equal to its own z-scoring, so raw is
    # exact); per-fold dirs via the budget cell's own folds, mean-aggregated.
    rows_e = np.asarray(elic_cell.row_idx, dtype=np.int64)
    fold_ids = np.asarray(elic_cell.fold_ids)
    per_fold = []
    for f in range(elic_cell.n_folds):
        tr = rows_e[fold_ids != f]
        if tr.size:
            vec, _, _ = _split_dir(z1, dv_raw, tr)
            per_fold.append(vec)
    folded = np.mean(np.stack(per_fold), axis=0)
    fold_cos = [float(a @ folded / (np.linalg.norm(a) * np.linalg.norm(folded))) for a in per_fold]
    return {"dirs": dirs, "counts": counts, "folded_dir": folded, "folded_per_fold_cos": fold_cos}


def _cos(a, b) -> float:
    import numpy as np

    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_p4(args) -> dict:
    """Pod-side driver: per behavior, load the SINGLE committed arm2 frozen
    layer, then per seed refit the (seed-keyed) whitening and compute the
    direction variants. Uses the scorer's own loaders / roster construction
    (``dataset_roster`` — code motion, not a copy)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from scripts import issue1739_r2v2_score as sc
    from scripts.issue1739_jobd_r2aug import LMAX, behavior_paths, build_pool, load_behavior
    from scripts.issue1739_result2fair_score import _wc_eval_mask
    from scripts.issue1739_wcrung_arms import modal_frozen_layers

    out: dict = {"probe": "p4", "meta": _meta(args), "behaviors": {}}
    for b in args.behaviors:
        s_args = sc.parse_args(
            ["--behaviors", b, "--protocols", "B", "--seed", "0"]
            + ["--out-root", str(args.out_root / "_p4_work")]
            + (["--store-root", str(args.store_root)] if args.store_root else [])
            + (["--ood-store-root", str(args.ood_store_root)] if args.ood_store_root else [])
            + (["--evil-ood-dv", str(args.evil_ood_dv)] if b == "evil" else [])
            + (["--syco-ood-dv", str(args.syco_ood_dv)] if b == "sycophancy" else [])
        )
        summary = behavior_paths(s_args, b)["train_summary"]
        frozen_idx = modal_frozen_layers(
            summary, variant=s_args.variant, regime=s_args.regime, u_rung_label="full"
        )["arm2_ctx_native"]
        fl = int(frozen_idx)  # full-grid index == layer number (identity grid)
        layers = [fl]
        loaded = load_behavior(s_args, b, layers)
        tbl_ood, ood_note = sc.load_ood_table(s_args, b, layers, loaded.dim, loaded.shas)
        n_tr, n_wc = len(loaded.tbl.ctx_order), len(loaded.tbl_wc.ctx_order)
        n_ev = len(loaded.tbl_ev.ctx_order)
        base_wc, base_ev, base_ood = n_tr, n_tr + n_wc, n_tr + n_wc + n_ev
        wc_train_rows = base_wc + np.flatnonzero(~_wc_eval_mask(loaded.tbl_wc.ctx_order))
        dv_raw = np.concatenate(
            [np.asarray(t.dv, dtype=np.float64) for t in (loaded.tbl, loaded.tbl_wc, loaded.tbl_ev)]
            + ([np.asarray(tbl_ood.dv, dtype=np.float64)] if tbl_ood is not None else [])
        )
        variant = s_args.variant
        raw_blocks = [
            loaded.tbl.z_by_variant[variant],
            loaded.tbl_wc.z_by_variant[variant],
            loaded.tbl_ev.z_by_variant[variant],
        ] + ([tbl_ood.z_by_variant[variant]] if tbl_ood is not None else [])
        per_seed_dirs: dict[int, dict] = {}
        wh_by_seed: dict[int, object] = {}
        for s in args.seeds:
            s_args.seed = int(s)
            x, _y, _lab, _n, _meta_pool = build_pool(s_args, loaded, variant, layers, "add")
            del _y
            wh = fits.fit_whitening(x, device=s_args.device, seed=int(s))
            del x
            z1 = np.concatenate(
                [fits.apply_whitening(blk, wh)[0] for blk in raw_blocks], axis=0
            )  # (N, d) at the single probe layer
            elic_cell = fits.realize_budget_cell(
                loaded.tbl.groups, budget_l=LMAX[b], draw=s_args.draw, seed=int(s)
            )
            datasets = sc.dataset_roster(
                loaded, tbl_ood, elic_cell, base_ev=base_ev, base_ood=base_ood
            )
            per_seed_dirs[int(s)] = _p4_directions(
                z1,
                dv_raw,
                datasets,
                wc_train_rows,
                elic_cell,
                seed=int(s),
                train_frac=s_args.train_frac,
            )
            per_seed_dirs[int(s)]["rb_w"] = np.einsum(
                "ld,lde->le", np.asarray(loaded.rb, dtype=np.float64), wh.w
            )[0]
            wh_by_seed[int(s)] = wh
            del z1
            _log(f"[{b}] seed {s}: directions computed at layer {fl}")
        seeds = [int(s) for s in args.seeds]
        variants = ("v1", "restricted", "quantile", "quantile-restricted")
        cos_mats = {}
        for name in variants:
            mat_w = [
                [
                    _cos(per_seed_dirs[a]["dirs"][name], per_seed_dirs[c]["dirs"][name])
                    for c in seeds
                ]
                for a in seeds
            ]
            raw = {
                s: wh_by_seed[s].w[0] @ per_seed_dirs[s]["dirs"][name] for s in seeds
            }  # raw-space functional: cross-seed-comparable
            mat_r = [[_cos(raw[a], raw[c]) for c in seeds] for a in seeds]
            cos_mats[name] = {"whitened_as_realized": mat_w, "raw_space": mat_r}
        out["behaviors"][b] = {
            "frozen_layer": fl,
            "ood_note": ood_note,
            "counts_per_seed": {s: per_seed_dirs[s]["counts"] for s in seeds},
            "across_seed_cosines": cos_mats,
            "cos_to_banked_e1": {
                name: {
                    s: _cos(per_seed_dirs[s]["dirs"][name], per_seed_dirs[s]["rb_w"]) for s in seeds
                }
                for name in variants
            },
            "cos_to_folded_train_dir": {
                name: {
                    s: _cos(per_seed_dirs[s]["dirs"][name], per_seed_dirs[s]["folded_dir"])
                    for s in seeds
                }
                for name in variants
            },
            "cos_v1_vs_restricted": {
                s: _cos(per_seed_dirs[s]["dirs"]["v1"], per_seed_dirs[s]["dirs"]["restricted"])
                for s in seeds
            },
            "folded_per_fold_cos": {s: per_seed_dirs[s]["folded_per_fold_cos"] for s in seeds},
            "note": "whitened_as_realized cosines compare vectors in DIFFERENT per-seed "
            "whitened bases (the as-deployed geometry); raw_space maps each direction "
            "back through its own whitening (w @ d — the cross-seed-comparable "
            "functional). The variant matching the SELECTED repair is the read that "
            "accompanies any MAP-BEATS narration (plan §4).",
        }
        del per_seed_dirs, wh_by_seed, raw_blocks, loaded, tbl_ood
    return out


# ---------------------------------------------------------------------------
# P5 — band-instrument audit (arms 1/4/7 + arm2 reference)
# ---------------------------------------------------------------------------


def run_p5(args) -> dict:
    import numpy as np

    from scripts.issue1739_claim4_fold import committed_band_vals

    out: dict = {"probe": "p5", "meta": _meta(args), "behaviors": {}, "missing_files": []}
    any_read = False
    for b in args.behaviors:
        rows_all: list[dict] = []
        for s in args.seeds:
            p = args.claim4_root / b / f"seed{s}" / "all_arms_spearman.json"
            if not p.exists():
                out["missing_files"].append(str(p))
                continue
            rows_all.extend(json.loads(p.read_text())["transfer_rows"])
            any_read = True
        per_arm: dict = {}
        for arm in P5_ARMS:
            vals = committed_band_vals(args.committed_train_root, b, arm)
            band = [min(vals), max(vals)] if vals else None
            pv: dict[int, list[float]] = {}
            for r in rows_all:
                if (
                    r.get("arm") == arm
                    and r.get("eval_rung") == "pvsynth"
                    and r.get("map_variant") in (None, "true")
                    and r.get("rho_frozen") is not None
                ):
                    pv.setdefault(int(r["seed"]), []).append(float(r["rho_frozen"]))
            per_seed = {s: float(np.mean(v)) for s, v in sorted(pv.items())}
            entry: dict = {
                "committed_band": band,
                "n_committed_cells": len(vals),
                "pvsynth_rho_per_seed": per_seed,
            }
            if band and per_seed:
                mean = float(np.mean(list(per_seed.values())))
                entry["pvsynth_rho_seed_mean"] = mean
                entry["in_band"] = bool(band[0] <= mean <= band[1])
                entry["miss_side"] = (
                    None if entry["in_band"] else ("below" if mean < band[0] else "above")
                )
            else:
                entry["note"] = "band or pvsynth rows unavailable"
            per_arm[arm] = entry
        probes_134_7 = [a for a in P5_ARMS if a != "arm2_ctx_native"]
        n_out = sum(1 for a in probes_134_7 if per_arm[a].get("in_band") is False)
        out["behaviors"][b] = {
            "per_arm": per_arm,
            "h_instrument_evidence": {
                "n_label_matched_arms_out_of_band": n_out,
                "n_label_matched_arms": len(probes_134_7),
                "reading": (
                    "label-matched probes ALSO out of band -> the pvsynth-transfer-vs-"
                    "train-band comparison is an instrument artifact candidate"
                    if n_out
                    else "label-matched probes in band -> the band miss is arm2-specific"
                ),
            },
        }
    if not any_read:
        raise FileNotFoundError(
            f"P5 read NO banked seed summaries under {args.claim4_root} — {HF_STAGE_HINT}"
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", nargs="+", choices=list(PROBES), default=None)
    ap.add_argument("--list-probes", action="store_true", help="print the probe registry + exit")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--ood-dv-max-null-frac", type=float, default=0.05)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("eval_results/issue_1739/claim4_controls/arm2fix/d0"),
    )
    ap.add_argument(
        "--claim4-root",
        type=Path,
        default=Path("eval_results/issue_1739/claim4_controls"),
        help="root holding the BANKED <behavior>/seed<S>/ dirs (P2 cross-check, P3, P5); "
        "on the VM point it at the staged mirror, e.g. "
        "data/issue_1739/hf_dl/claim4_mirror/issue1739_claim4_controls",
    )
    ap.add_argument("--committed-train-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--train-dv-root", type=Path, default=Path("eval_results/issue_1739/dv_dataset")
    )
    ap.add_argument(
        "--wcrung-dv-root",
        type=Path,
        default=Path("eval_results/issue_1739/wildchat_rung/dv_dataset"),
    )
    ap.add_argument(
        "--evil-ood-dv",
        type=Path,
        default=Path("eval_results/issue_1739/evil_ood_full/dv_dataset/evil/labeling.json"),
    )
    ap.add_argument(
        "--syco-ood-dv",
        type=Path,
        default=Path(
            "data/issue_1739/hf_dl/ood_mirror/issue1739_ctxmap/syco_ood/dv_dataset/"
            "sycophancy/labeling.json"
        ),
    )
    ap.add_argument("--store-root", type=Path, default=None, help="P4 only (pod-side)")
    ap.add_argument("--ood-store-root", type=Path, default=None, help="P4 only (pod-side)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--p3-check-preds",
        action="store_true",
        help="P3: ALSO recompute per-rung rho from the banked per-context preds "
        "(degrades with a recorded note when transfer_preds/ is absent)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if not args.list_probes and not args.import_check and not args.probes:
        ap.error("--probes is required (or --list-probes / --import-check)")
    return args


RUNNERS = {"p2": run_p2, "p3": run_p3, "p4": run_p4, "p5": run_p5}
OUT_NAMES = {
    "p2": "p2_pool_reconstruction.json",
    "p3": "p3_layer_profiles.json",
    "p4": "p4_direction_stability.json",
    "p5": "p5_band_audit.json",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_probes:
        for p in PROBES:
            print(p)
        return 0
    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import arms, fits  # noqa: F401
        from scripts import issue1739_r2v2_score as sc  # noqa: F401
        from scripts.issue1739_claim4_fold import committed_band_vals  # noqa: F401
        from scripts.issue1739_fits import _git_commit  # noqa: F401
        from scripts.issue1739_jobd_r2aug import (  # noqa: F401
            LMAX,
            behavior_paths,
            build_pool,
            load_behavior,
        )
        from scripts.issue1739_result2fair_score import _wc_eval_mask  # noqa: F401
        from scripts.issue1739_wcrung_arms import modal_frozen_layers  # noqa: F401

        assert callable(sc._group_side_train) and callable(sc._multi_pool_zscored_dv)
        assert callable(sc.dataset_roster) and callable(sc._prepare_ood_dv)
        assert callable(arms.arm2q_thresholds) and callable(arms.spearman_rows)
        assert set(RUNNERS) == set(PROBES) == set(OUT_NAMES)
        print("[a2fix-d0] import-check OK", flush=True)
        return 0
    t0 = time.time()
    for probe in args.probes:
        t1 = time.time()
        payload = RUNNERS[probe](args)
        _write_json(args.out_root / OUT_NAMES[probe], payload)
        _log(f"{probe} done in {time.time() - t1:.0f}s")
    _log(f"all probes done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
