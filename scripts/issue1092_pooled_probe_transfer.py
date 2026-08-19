#!/usr/bin/env python3
"""Issue #1092 follow-up `pooled-probe-transfer`: 0-GPU pooled/LODO supervised-probe transfer.

The parent `cross-corpus-probe-transfer` round showed supervised trait probes
(context-end activation -> judged trait score) reach high within-corpus R2 but
transfer ~0 across corpora (persona-conditioned <-> LMSYS). This follow-up tests
whether POOLED training fixes transfer: train probes jointly on multiple labeled
context distributions and score a HELD-OUT distribution (never in training) as
the PRIMARY DV (leave-one-distribution-out, LODO).

Substrates (all three are the wired, alignment-gated surfaces of the parent
round; the same L14/L19 context-end position, 3584-d residual stream):
  P = #1092 cell_inst_own persona-conditioned corpus (context-end; prefix-end
      secondary within-only). Strong signal.
  A = #779 pass_a PV-rig persona-condition eval (context-end only). Signal TBD.
  L = #779 LMSYS pass_b real-user corpus (context-end only). Label-flat (parent
      within-corpus ceiling ~0) -> a documented-uninformative held-out target.

Arms per (trait x layer x held-out H), pool = the two substrates != H:
  (a) single-source: train each source in pool, score H (reproduces the parent
      within/across reads -> validity gate).
  (b/c) pooled (LODO): train on the pool (concat, train-pool-standardized),
        score H.
  (d) pooled + per-corpus centering: subtract each corpus's own feature mean
      before pooling (and H's own mean at eval) -> removes corpus mean-shift.
  (e) r_B baseline: the raw #779 persona-vector direction at the layer, fit as a
      1-D affine on the pool, scored on H.

Primary DV: held-out-distribution Pearson r and R2 (+ group cluster-bootstrap
CIs). Secondary: within-corpus ceilings + within-pool CV. Ridge lambda is
selected on training-pool CV folds ONLY (fold_fits) -- never on H.

ANALYSIS-ONLY: reuses banked inputs at the parent's pinned revisions; no GPU, no
new generation, no new judging. The fit engine, loaders, group-fold builder,
metrics, and cluster bootstrap are imported VERBATIM from
issue1092_transfer_probe.py (which reuses the byte-pinned issue1092_fit_grid.py).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1092_transfer_probe as T  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    _folds_from_manifest,
    _load_summary,
    _pearson_or_nan,
    _r2,
)
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

# Parent-round pins (issue1092 cross-corpus-probe-transfer transfer_reads.json metadata).
HF_REV_1092 = "e590170619e7691c1a95c7b1bb20bda5fd4065ad"
HF_REV_779_LABELS = "5aa6de1b97895cf8883c44165fa8835ff73e9e93"
HF_REV_779_PASSB = "037fcbb210bc52c459959b0746cc268fe08bae96"

TRAITS = ("hallucination", "sycophancy")
LAYERS = (14, 19)  # intersection of #1092 fit layers {14,18,19} and #779 monitoring {14,19,26}
# validity-gate targets from the parent transfer_reads.json (layer 14 only).
PARENT_L14 = {
    "hallucination": {"within_P": 0.8641252395477983, "within_L": 0.009062541043387327},
    "sycophancy": {"within_P": 0.7601310091531392, "within_L": -0.012250608802857634},
}
VALIDITY_TOL = 0.02


def _git_commit() -> str:
    return T._git_commit()


def _log(msg: str) -> None:
    print(f"[pooled-transfer] {msg}", flush=True)


def _boot_ci(
    y: np.ndarray, pred: np.ndarray, groups: list[str], n_draws: int, label: str, seed: int
) -> dict[str, Any]:
    """Group cluster-bootstrap 95% CI of Pearson r on the held-out substrate."""
    if groups is None:
        spec: tuple = ("rows",)
    else:
        codes, n_cl = T._codes(list(groups))
        spec = ("one", codes, n_cl) if n_cl > 1 else ("rows",)
    out = T._boot_paired(y, pred, None, spec, n_draws, label, seed)
    ci = out["ci_ctx"]
    return {
        "lo": ci.get("lo"),
        "hi": ci.get("hi"),
        "n_valid_replicates": ci.get("n_valid_replicates"),
    }


def _stage_1092_layer(staged: dict[str, Any], layer: int) -> None:
    """Stage #1092 cell_inst_own context_end/prefix_end + bare c_q_bare summaries for one layer.

    stage_inputs() only stages the summaries for its single --layer; the pooled
    run walks {14, 19}, so each additional layer's #1092 summaries are staged
    here (verified against the hub LFS sha at the parent pin, local-first).
    """
    from huggingface_hub import hf_hub_download

    summaries_dir = staged["summaries_dir"]
    cell_map = T._hub_lfs_sha_map(
        f"{T.PREFIX_1092}/analysis_tensors/summaries/{T.CELL}", HF_REV_1092
    )
    bare_map = T._hub_lfs_sha_map(
        f"{T.PREFIX_1092}/analysis_tensors/summaries/bare_{T.MODEL_TYPE}", HF_REV_1092
    )
    (summaries_dir / T.CELL).mkdir(parents=True, exist_ok=True)
    (summaries_dir / f"bare_{T.MODEL_TYPE}").mkdir(parents=True, exist_ok=True)

    # single-npy summaries: context_end + prefix_end.
    for kind in ("context_end", "prefix_end"):
        name = f"{kind}_L{layer:02d}.npy"
        dst = summaries_dir / T.CELL / name
        if dst.exists() and (
            cell_map.get(name, (0, None))[1] is None or T._sha256_file(dst) == cell_map[name][1]
        ):
            continue
        src = T._stage_verified_local(
            [
                PROJECT_ROOT / f"data/issue_1092/p7/read4c_repair/staging/{T.CELL}/{name}",
                PROJECT_ROOT
                / f"data/issue_1092/hf_dl/{T.PREFIX_1092}/analysis_tensors/summaries/{T.CELL}/{name}",
                dst,
            ],
            cell_map.get(name, (0, None))[1],
            f"{T.PREFIX_1092}/analysis_tensors/summaries/{T.CELL}/{name}",
            HF_REV_1092,
        )
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)

    # bare c_q_bare_L{layer} shards + row_index (row_index layer-invariant; staged if absent).
    bare_names = sorted(
        n
        for n in bare_map
        if (n.startswith(f"c_q_bare_L{layer:02d}_shard") and n.endswith(".npy"))
        or n == f"c_q_bare_L{layer:02d}.npy"
        or (n.startswith("row_index_shard") and n.endswith(".jsonl"))
        or n == "row_index.jsonl"
    )
    if not any(n.startswith(f"c_q_bare_L{layer:02d}") for n in bare_names):
        raise RuntimeError(f"no bare_{T.MODEL_TYPE} L{layer:02d} files at pin")
    for name in bare_names:
        dst = summaries_dir / f"bare_{T.MODEL_TYPE}" / name
        exp = bare_map[name][1]
        if dst.exists() and (exp is None or T._sha256_file(dst) == exp):
            continue
        p = Path(
            T.retry_transient(
                lambda name=name: hf_hub_download(
                    T.HF_REPO,
                    repo_type="dataset",
                    revision=HF_REV_1092,
                    filename=f"{T.PREFIX_1092}/analysis_tensors/summaries/bare_{T.MODEL_TYPE}/{name}",
                )
            )
        )
        shutil.copyfile(p, dst)


def build_substrates(
    staged: dict[str, Any],
    unit: dict[str, Any],
    x_lmsys: np.ndarray,
    labels: dict[str, list[float | None]],
    dedup: dict[str, Any],
    trait: str,
    layer: int,
    smoke_rows: int = 0,
) -> dict[str, dict[str, Any]]:
    """Assemble the three aligned substrates for one (trait, layer).

    Returns name -> {X, y, groups, dedup_keep} where dedup_keep is the eval-side
    contamination mask (True = keep). Training uses full X; eval uses X[dedup_keep].
    """
    subs: dict[str, dict[str, Any]] = {}

    # P: #1092 cell_inst_own scored rows (context-end).
    pairs = unit["by_trait"].get(trait, [])
    pool_idx = np.asarray([i for i, _s in pairs], dtype=np.int64)
    pool_y = np.asarray([s for _i, s in pairs], dtype=np.float64)
    p_prefix_ids = [str(unit["unit_rows"][i].get("prefix_id")) for i in pool_idx]
    p_keep = np.asarray([not dedup["row_overlap"][i] for i in pool_idx], dtype=bool)
    subs["P_persona_ctx"] = {
        "X": unit["x_ctx"][pool_idx],
        "y": pool_y,
        "groups": p_prefix_ids,
        "dedup_keep": p_keep,
        "group_key": "prefix_id",
        "kind": "persona-conditioned",
        "arm": "context_end",
    }
    # P prefix-end (secondary, within-only).
    subs["P_persona_prefix"] = {
        "X": unit["x_prefix"][pool_idx],
        "y": pool_y,
        "groups": p_prefix_ids,
        "dedup_keep": p_keep,
        "group_key": "prefix_id",
        "kind": "persona-conditioned",
        "arm": "prefix_end",
    }

    # A: pass_a PV-rig (context-end only).
    pa = T.load_pass_a_surface(staged, trait, layer)
    subs["A_passa_ctx"] = {
        "X": pa["X"],
        "y": pa["y"],
        "groups": [m["cond_id"] for m in pa["meta"]],
        "dedup_keep": np.ones(pa["y"].size, dtype=bool),  # PV bank disjoint from pool queries
        "group_key": "cond_id",
        "kind": "persona-conditioned",
        "arm": "context_end",
        "pruned_conditions_std_lt_1": pa["pruned_conditions_std_lt_1"],
    }

    # L: LMSYS pass_b (context-end only).
    lab = labels.get(trait, [None] * 5000)
    l_valid = np.asarray([ci for ci in range(5000) if lab[ci] is not None], dtype=np.int64)
    l_y = np.asarray([lab[ci] for ci in l_valid], dtype=np.float64)
    l_keep = np.asarray([not dedup["lmsys_excluded"][ci] for ci in l_valid], dtype=bool)
    subs["L_lmsys_ctx"] = {
        "X": x_lmsys[l_valid],
        "y": l_y,
        "groups": [f"lmsys_{int(ci):05d}" for ci in l_valid],  # row-level clusters
        "dedup_keep": l_keep,
        "group_key": "row",
        "kind": "real-user (LMSYS)",
        "arm": "context_end",
    }

    if smoke_rows:
        for name, s in subs.items():
            n = s["y"].size
            if n <= smoke_rows:
                continue
            sel = np.sort(
                np.random.default_rng(abs(hash(name)) % (2**32)).choice(
                    n, size=smoke_rows, replace=False
                )
            )
            s["X"] = s["X"][sel]
            s["y"] = s["y"][sel]
            s["groups"] = [s["groups"][i] for i in sel]
            s["dedup_keep"] = s["dedup_keep"][sel]
    return subs


def _substrate_summary(sub: dict[str, Any]) -> dict[str, Any]:
    y = sub["y"]
    return {
        "n": int(y.size),
        "label_std": float(np.std(y)) if y.size else float("nan"),
        "label_mean": float(np.mean(y)) if y.size else float("nan"),
        "n_positive": int((y > 50).sum()),
        "n_negative": int((y <= 50).sum()),
        "estimable": bool(y.size >= 5 and np.std(y) >= 1.0 and (y > 50).any() and (y <= 50).any()),
        "kind": sub["kind"],
        "arm": sub["arm"],
        "group_key": sub["group_key"],
    }


def _folds_for(sub: dict[str, Any], n_folds: int = 6) -> list[np.ndarray]:
    rows = [{"prefix_id": g} for g in sub["groups"]]
    nf = max(2, min(n_folds, len(set(sub["groups"]))))
    return _folds_from_manifest(rows, len(rows), group_key="prefix_id", n_folds=nf)


def within_ceiling(sub: dict[str, Any], n_draws: int, label: str, seed: int) -> dict[str, Any]:
    """Grouped-CV Pearson r on a substrate (its own achievable ceiling)."""
    if not _substrate_summary(sub)["estimable"]:
        return {"status": "not estimable", **_substrate_summary(sub)}
    folds = _folds_for(sub)
    ff = T.fold_fits(sub["X"], sub["y"], folds, [])
    cv_r = _pearson_or_nan(ff["cv_pred"], sub["y"])
    ci = _boot_ci(sub["y"], ff["cv_pred"], sub["groups"], n_draws, label, seed)
    return {
        "cv_r": float(cv_r),
        "cv_r2": float(_r2(sub["y"].reshape(-1, 1), ff["cv_pred"].reshape(-1, 1))),
        "ci_r": [ci["lo"], ci["hi"]],
        "n": int(sub["y"].size),
        "lam_cv": float(RIDGE_LAMBDAS[ff["lam_cv_idx"]]),
    }


def _eval_read(
    y_h: np.ndarray,
    pred: np.ndarray,
    keep: np.ndarray,
    groups: list[str],
    n_draws: int,
    label: str,
    seed: int,
) -> dict[str, Any]:
    ye, pe = y_h[keep], pred[keep]
    ge = [g for g, k in zip(groups, keep) if k]
    r = _pearson_or_nan(pe, ye)
    ci = _boot_ci(ye, pe, ge, n_draws, label, seed)
    return {
        "r": float(r),
        "r2": float(_r2(ye.reshape(-1, 1), pe.reshape(-1, 1))),
        "ci_r": [ci["lo"], ci["hi"]],
        "n_eval": int(ye.size),
        "n_excluded_overlap": int((~keep).sum()),
    }


def train_and_eval(
    train_subs: list[dict[str, Any]],
    eval_subs: list[tuple[str, dict[str, Any]]],
    center_per_corpus: bool,
    n_draws: int,
    tag: str,
    seed: int,
) -> dict[str, Any]:
    """Fit ridge (train-pool CV lambda) on the concat of train_subs; score each eval substrate.

    center_per_corpus: subtract each substrate's own feature mean before pooling
    (and each eval substrate's own mean at eval) -> corpus mean-shift removal.
    """
    xs, ys, gs = [], [], []
    for s in train_subs:
        X = s["X"]
        if center_per_corpus:
            X = X - X.mean(axis=0, keepdims=True)
        xs.append(X)
        ys.append(s["y"])
        gs.extend(s["groups"])
    X_tr = np.concatenate(xs, axis=0)
    y_tr = np.concatenate(ys, axis=0)
    folds = _folds_from_manifest(
        [{"prefix_id": g} for g in gs], len(gs), group_key="prefix_id", n_folds=6
    )
    eval_X = []
    for _name, s in eval_subs:
        Xe = s["X"]
        if center_per_corpus:
            Xe = Xe - Xe.mean(axis=0, keepdims=True)
        eval_X.append(Xe)
    ff = T.fold_fits(X_tr, y_tr, folds, eval_X)
    out: dict[str, Any] = {
        "pool_members": [],  # filled by caller
        "within_pool_cv_r": float(_pearson_or_nan(ff["cv_pred"], y_tr)),
        "within_pool_cv_r2": float(_r2(y_tr.reshape(-1, 1), ff["cv_pred"].reshape(-1, 1))),
        "lam_cv": float(RIDGE_LAMBDAS[ff["lam_cv_idx"]]),
        "n_train": int(y_tr.size),
        "reads": {},
    }
    for si, (name, s) in enumerate(eval_subs):
        out["reads"][name] = _eval_read(
            s["y"],
            ff["full_eval"][si],
            s["dedup_keep"],
            s["groups"],
            n_draws,
            f"{tag}->{name}",
            seed,
        )
    del ff
    gc.collect()
    return out


def rb_baseline(
    r_b_layer: np.ndarray,
    train_subs: list[dict[str, Any]],
    h_name: str,
    h: dict[str, Any],
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """1-D affine on the r_B projection fit on the training pool, scored on H."""
    proj_tr, y_tr = [], []
    for s in train_subs:
        proj_tr.append(s["X"] @ r_b_layer)
        y_tr.append(s["y"])
    pt = np.concatenate(proj_tr)
    yt = np.concatenate(y_tr)
    A = np.vstack([pt, np.ones_like(pt)]).T
    slope, intercept = np.linalg.lstsq(A, yt, rcond=None)[0]
    pred_h = h["X"] @ r_b_layer * slope + intercept
    return _eval_read(h["y"], pred_h, h["dedup_keep"], h["groups"], n_draws, f"rB->{h_name}", seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    T.run_selftest("cpu")

    layers_run = [int(x) for x in str(args.layers).split(",") if str(x).strip()]
    traits_run = [t.strip() for t in str(args.traits).split(",") if t.strip()]

    stage_args = argparse.Namespace(
        hf_rev_1092=HF_REV_1092,
        hf_rev_779_labels=HF_REV_779_LABELS,
        hf_rev_779_passb=HF_REV_779_PASSB,
        layer=LAYERS[0],
        smoke=False,
        smoke_rows=200,
        seed=args.seed,
        n_draws=args.n_draws,
        traits=",".join(TRAITS),
        out=str(out_dir),
        figures=args.figures,
        skip_upload=True,
    )
    report: dict[str, Any] = {
        "metadata": {
            "script": "issue1092_pooled_probe_transfer.py",
            "followup_label": "pooled-probe-transfer",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pins": {
                "hf_rev_1092": HF_REV_1092,
                "hf_rev_779_labels": HF_REV_779_LABELS,
                "hf_rev_779_passb": HF_REV_779_PASSB,
            },
            "layers": layers_run,
            "traits": traits_run,
            "full_layers": list(LAYERS),
            "full_traits": list(TRAITS),
            "ridge_lambdas": [float(x) for x in RIDGE_LAMBDAS],
            "n_draws": int(args.n_draws),
            "seed": int(args.seed),
            "smoke_mode": bool(args.smoke),
            "no_finalize": bool(args.no_finalize),
            "primary_dv": "held-out-distribution Pearson r and R2 (+ group cluster-bootstrap CI)",
            "arms": ["single_source", "pooled_lodo", "pooled_lodo_centered", "r_b_baseline"],
            "notes": [
                "LODO pool = the two substrates != held-out (3 substrates -> pool is a pair).",
                "cross-corpus arms use context-end (the only shared surface); prefix-end is "
                "#1092-only, reported as a secondary within-corpus ceiling.",
                "pass_a (A) has no prefix-end / bare arm; LMSYS (L) has no prefix-end / group "
                "structure beyond row -> both context-end only.",
            ],
        },
        "substrates": {},
        "transfer": {},
        "validity_gate": {},
    }

    def checkpoint() -> None:
        T._write_json_atomic(out_dir / "pooled_probe_transfer.json", report)

    # Resume: reuse completed (trait, layer) cells from a prior FULL run of the
    # same recipe (survives an interrupted/timed-out run; a smoke JSON is ignored).
    prior_path = out_dir / "pooled_probe_transfer.json"
    if not args.smoke and prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text())
            pm = prior.get("metadata", {})
            if (
                not pm.get("smoke_mode")
                and pm.get("n_draws") == args.n_draws
                and pm.get("seed") == args.seed
                and pm.get("layers") == layers_run
                and pm.get("traits") == traits_run
            ):
                report["substrates"] = prior.get("substrates", {})
                report["transfer"] = prior.get("transfer", {})
                if prior.get("overlap_dedup"):
                    report["overlap_dedup"] = prior["overlap_dedup"]
                done = [(t, lk) for t in report["transfer"] for lk in report["transfer"][t]]
                _log(f"resume: reusing {len(done)} completed cells: {done}")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            _log(f"resume: prior JSON unusable ({exc!r}); starting fresh")

    staged = T.stage_inputs(stage_args)
    report["metadata"]["input_shas"] = staged["shas"]

    prompts, labels14, _ = T.load_lmsys_prompts_and_labels(staged["rollouts"], staged["labels"])
    # #1092 unit is needed per layer (context/prefix arms). dedup is layer-invariant (text-based).
    unit14 = T.load_1092_unit(staged, 14)
    dedup = T.overlap_dedup(staged, unit14, prompts)
    report["overlap_dedup"] = dedup["report"]
    del unit14
    gc.collect()
    checkpoint()

    # r_B directions (per trait, 28x3584).
    r_b: dict[str, np.ndarray] = {}
    for trait in traits_run:
        rp = PROJECT_ROOT / f"data/issue_779/r_b/{trait}.pt"
        blob = torch.load(rp, map_location="cpu", weights_only=False)
        r_b[trait] = blob["r_b"].to(torch.float64).numpy()

    for layer in layers_run:
        lk = f"L{layer:02d}"
        if all(
            t in report["transfer"]
            and lk in report["transfer"][t]
            and t in report["substrates"]
            and lk in report["substrates"][t]
            for t in traits_run
        ):
            _log(f"resume: skip completed layer {layer} (all run-traits done)")
            continue
        _log(f"===== layer {layer} =====")
        x_lmsys, _ = T.load_pass_b_l14(staged["pass_b"], layer)
        _stage_1092_layer(staged, layer)
        unit = T.load_1092_unit(staged, layer)
        unit["x_prefix"], _ = _load_summary(staged["summaries_dir"], T.CELL, "prefix_end", layer)
        # align prefix summary length to unit rows (same FITA exclusion the loader applied to ctx)
        rows_all = T._jsonl(staged["corpus_dir"] / "manifest.jsonl")
        n0 = min(unit["x_prefix"].shape[0], len(rows_all))
        base_rows = rows_all[:n0]
        keep_idx = np.asarray(
            [i for i, r in enumerate(base_rows) if r.get("stratum") not in T.FITA_EXCLUDED_STRATA],
            dtype=np.int64,
        )
        unit["x_prefix"] = unit["x_prefix"][keep_idx]
        assert unit["x_prefix"].shape[0] == unit["x_ctx"].shape[0], (
            unit["x_prefix"].shape,
            unit["x_ctx"].shape,
        )

        for trait in traits_run:
            if (
                trait in report["transfer"]
                and lk in report["transfer"][trait]
                and trait in report["substrates"]
                and lk in report["substrates"][trait]
            ):
                _log(f"resume: skip completed {trait}/{lk}")
                continue
            _log(f"layer {layer} / {trait}")
            subs = build_substrates(
                staged,
                unit,
                x_lmsys,
                labels14,
                dedup,
                trait,
                layer,
                smoke_rows=(args.smoke_rows if args.smoke else 0),
            )
            lk = f"L{layer:02d}"
            report["substrates"].setdefault(trait, {})[lk] = {
                name: _substrate_summary(s) for name, s in subs.items()
            }
            checkpoint()

            # within-corpus ceilings (context substrates + P prefix).
            ceilings: dict[str, Any] = {}
            for name in ("P_persona_ctx", "P_persona_prefix", "A_passa_ctx", "L_lmsys_ctx"):
                ceilings[name] = within_ceiling(
                    subs[name], args.n_draws, f"ceil::{trait}::{layer}::{name}", args.seed
                )
            report["substrates"][trait][lk] = {
                name: {**report["substrates"][trait][lk][name], "within_ceiling": ceilings[name]}
                for name in report["substrates"][trait][lk]
            }
            checkpoint()

            # cross-corpus arms use the three CONTEXT substrates.
            ctx_names = ["P_persona_ctx", "A_passa_ctx", "L_lmsys_ctx"]
            tr_block: dict[str, Any] = {}

            # single-source fits: train each source once, eval on the other two.
            single_fits: dict[str, dict[str, Any]] = {}
            for src in ctx_names:
                others = [(n, subs[n]) for n in ctx_names if n != src]
                fit = train_and_eval(
                    [subs[src]],
                    others,
                    False,
                    args.n_draws,
                    f"single::{trait}::{layer}::{src}",
                    args.seed,
                )
                fit["pool_members"] = [src]
                single_fits[src] = fit

            # per held-out target: assemble arms.
            for h in ctx_names:
                pool_names = [n for n in ctx_names if n != h]
                pooled = train_and_eval(
                    [subs[n] for n in pool_names],
                    [(h, subs[h])],
                    False,
                    args.n_draws,
                    f"pool::{trait}::{layer}::{h}",
                    args.seed,
                )
                pooled["pool_members"] = pool_names
                pooled_c = train_and_eval(
                    [subs[n] for n in pool_names],
                    [(h, subs[h])],
                    True,
                    args.n_draws,
                    f"poolC::{trait}::{layer}::{h}",
                    args.seed,
                )
                pooled_c["pool_members"] = pool_names
                rb = rb_baseline(
                    r_b[trait][layer],
                    [subs[n] for n in pool_names],
                    h,
                    subs[h],
                    args.n_draws,
                    args.seed,
                )
                tr_block[h] = {
                    "held_out_within_ceiling": ceilings[h],
                    "single_source": {src: single_fits[src]["reads"][h] for src in pool_names},
                    "pooled_lodo": {
                        "read": pooled["reads"][h],
                        "within_pool_cv_r": pooled["within_pool_cv_r"],
                        "within_pool_cv_r2": pooled["within_pool_cv_r2"],
                        "lam_cv": pooled["lam_cv"],
                        "n_train": pooled["n_train"],
                        "pool_members": pool_names,
                    },
                    "pooled_lodo_centered": {
                        "read": pooled_c["reads"][h],
                        "within_pool_cv_r": pooled_c["within_pool_cv_r"],
                        "lam_cv": pooled_c["lam_cv"],
                    },
                    "r_b_baseline": rb,
                }
            report["transfer"].setdefault(trait, {})[lk] = tr_block
            checkpoint()

        del x_lmsys, unit
        gc.collect()

    if args.no_finalize:
        _log("no-finalize: skipping validity gate + figures (per-cell parallel worker)")
        return report
    _finalize(report, out_dir, Path(args.figures), enforced=not args.smoke)
    return report


def _compute_validity_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Layer-14 within-corpus-ceiling reproduction gate vs the parent transfer_reads.json.

    Only checks a (trait, substrate) whose L14 within-ceiling is present, so a
    single-cell worker report (no L14) yields an empty, trivially-passing gate;
    the merge step runs it over the full report.
    """
    vg: dict[str, Any] = {"tol": VALIDITY_TOL, "checks": []}
    all_pass = True
    for trait in TRAITS:
        sub14 = report.get("substrates", {}).get(trait, {}).get("L14")
        if not sub14:
            continue
        got_P = sub14.get("P_persona_ctx", {}).get("within_ceiling", {}).get("cv_r")
        got_L = sub14.get("L_lmsys_ctx", {}).get("within_ceiling", {}).get("cv_r")
        for label, got, exp in (
            (f"{trait}/within_P_ceiling_cv_r", got_P, PARENT_L14[trait]["within_P"]),
            (f"{trait}/within_L_ceiling_cv_r", got_L, PARENT_L14[trait]["within_L"]),
        ):
            ok = got is not None and abs(got - exp) <= VALIDITY_TOL
            all_pass = all_pass and ok
            vg["checks"].append(
                {
                    "check": label,
                    "got": got,
                    "expected_parent": exp,
                    "abs_gap": (abs(got - exp) if got is not None else None),
                    "pass": bool(ok),
                }
            )
    vg["pass"] = bool(all_pass)
    return vg


def _finalize(report: dict[str, Any], out_dir: Path, figures_dir: Path, *, enforced: bool) -> None:
    """Validity gate + figures + checkpoint (run once over the FULL report)."""
    vg = _compute_validity_gate(report)
    vg["enforced"] = enforced
    report["validity_gate"] = vg
    T._write_json_atomic(out_dir / "pooled_probe_transfer.json", report)
    _log(f"validity gate pass={vg['pass']} (enforced={enforced}, checks={len(vg['checks'])})")
    if enforced and not vg["pass"]:
        raise RuntimeError(
            "validity gate FAILED: within-corpus ceilings diverge from the parent "
            "transfer_reads.json beyond tol; the reused engine/loaders must reproduce "
            "the parent within-reads before any pooled transfer read is interpreted"
        )
    try:
        make_figures(report, figures_dir)
    except Exception as exc:  # figures are a presentation layer; never lose the JSON
        _log(f"WARNING: figure generation failed: {exc!r}")


def merge_and_finalize(paths: list[str], out_dir: Path, figures_dir: Path) -> dict[str, Any]:
    """Merge per-cell JSONs (first path = base with metadata) into the canonical report."""
    base = json.loads(Path(paths[0]).read_text())
    base.setdefault("substrates", {})
    base.setdefault("transfer", {})
    for p in paths[1:]:
        d = json.loads(Path(p).read_text())
        for trait, layers in d.get("substrates", {}).items():
            base["substrates"].setdefault(trait, {}).update(layers)
        for trait, layers in d.get("transfer", {}).items():
            base["transfer"].setdefault(trait, {}).update(layers)
    base["metadata"]["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base["metadata"]["merged_from"] = [str(p) for p in paths]
    out_dir.mkdir(parents=True, exist_ok=True)
    _finalize(base, out_dir, figures_dir, enforced=True)
    return base


_HELD_LABEL = {
    "P_persona_ctx": "#1092 persona\n(context-end)",
    "A_passa_ctx": "PV-rig pass_a\n(context-end)",
    "L_lmsys_ctx": "LMSYS\n(context-end)",
}
_ARM_LABEL = {
    "ceiling": "within-corpus ceiling",
    "single": "best single-source",
    "pooled": "pooled (LODO)",
    "pooled_c": "pooled + per-corpus centering",
    "rb": "raw r_B baseline",
}


def _yerr(v: float | None, ci: list) -> tuple[float, float]:
    """Element-wise clamped asymmetric error-bar offsets (never negative)."""
    if v is None or ci is None or ci[0] is None or ci[1] is None:
        return (0.0, 0.0)
    return (max(0.0, v - ci[0]), max(0.0, ci[1] - v))


def make_figures(report: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    figures_dir.mkdir(parents=True, exist_ok=True)
    traits = [t for t in TRAITS if t in report["transfer"]]
    layers = list(LAYERS)
    held = ["P_persona_ctx", "A_passa_ctx", "L_lmsys_ctx"]
    arms = ["ceiling", "single", "pooled", "pooled_c", "rb"]
    colors = {
        "ceiling": paper_palette_role("neutral"),
        "single": paper_palette_role("baseline"),
        "pooled": paper_palette_role("primary"),
        "pooled_c": paper_palette_role("accent"),
        "rb": paper_palette_role("control"),
    }

    def arm_read(cell: dict, arm: str) -> dict:
        if arm == "ceiling":
            c = cell["held_out_within_ceiling"]
            return {"r": c.get("cv_r"), "ci": c.get("ci_r")}
        if arm == "single":
            ss = cell["single_source"]
            best = max(ss.values(), key=lambda d: d["r"] if d["r"] is not None else -9)
            return {"r": best["r"], "ci": best["ci_r"]}
        if arm == "pooled":
            return {
                "r": cell["pooled_lodo"]["read"]["r"],
                "ci": cell["pooled_lodo"]["read"]["ci_r"],
            }
        if arm == "pooled_c":
            return {
                "r": cell["pooled_lodo_centered"]["read"]["r"],
                "ci": cell["pooled_lodo_centered"]["read"]["ci_r"],
            }
        return {"r": cell["r_b_baseline"]["r"], "ci": cell["r_b_baseline"]["ci_r"]}

    # ---- Figure A: grouped bars, held-out r per arm (trait rows x layer cols) ----
    nrow, ncol = len(traits), len(layers)
    figA, axesA = plt.subplots(nrow, ncol, figsize=(6.8 * ncol, 4.2 * nrow), squeeze=False)
    x = np.arange(len(held))
    w = 0.16
    for ri, trait in enumerate(traits):
        for ci, layer in enumerate(layers):
            ax = axesA[ri][ci]
            lk = f"L{layer:02d}"
            block = report["transfer"][trait][lk]
            for ai, arm in enumerate(arms):
                rs, los, his = [], [], []
                for h in held:
                    rd = arm_read(block[h], arm)
                    v = rd["r"] if rd["r"] is not None else 0.0
                    rs.append(v)
                    lo, hi = _yerr(rd["r"], rd["ci"])
                    los.append(lo)
                    his.append(hi)
                ax.bar(
                    x + (ai - 2) * w,
                    rs,
                    w,
                    yerr=[los, his],
                    capsize=2,
                    label=_ARM_LABEL[arm] if (ri == 0 and ci == 0) else None,
                    color=colors[arm],
                    edgecolor="white",
                    linewidth=0.5,
                )
            ax.axhline(0, color="#888", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([_HELD_LABEL[h] for h in held], fontsize=8)
            ax.set_ylabel("held-out Pearson r")
            ax.set_title(f"{trait} — layer {layer}", fontsize=10)
            ax.set_ylim(min(-0.15, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
    handles, labels = axesA[0][0].get_legend_handles_labels()
    figA.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    set_title_subtitle(
        figA.axes[0],
        "Pooled/LODO probe transfer: held-out Pearson r by arm",
        "held-out target on x; bars are within-corpus ceiling / best single-source / "
        "pooled / pooled+centering / raw r_B (95% cluster-bootstrap CI)",
    )
    figA.tight_layout(rect=(0, 0.03, 1, 1))
    savefig_paper(figA, "pooled_probe_transfer_heldout_bars", dir=str(figures_dir))
    plt.close(figA)

    # ---- Figure B: pooled vs best-single held-out r (paired) ----
    figB, axesB = plt.subplots(1, len(traits), figsize=(5.2 * len(traits), 4.6), squeeze=False)
    for ci, trait in enumerate(traits):
        ax = axesB[0][ci]
        for layer in layers:
            block = report["transfer"][trait][f"L{layer:02d}"]
            for h in held:
                s = arm_read(block[h], "single")["r"] or 0.0
                p = arm_read(block[h], "pooled")["r"] or 0.0
                mk = {"P_persona_ctx": "o", "A_passa_ctx": "s", "L_lmsys_ctx": "^"}[h]
                ax.scatter(
                    s,
                    p,
                    marker=mk,
                    s=70,
                    color=paper_palette_role("primary" if layer == 14 else "accent"),
                    edgecolor="white",
                    zorder=3,
                    label=f"{_HELD_LABEL[h].splitlines()[0]} / L{layer}",
                )
        lo, hi = -0.2, 1.0
        ax.plot([lo, hi], [lo, hi], color="#888", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("best single-source held-out r")
        ax.set_ylabel("pooled (LODO) held-out r")
        ax.set_title(trait, fontsize=10)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
    set_title_subtitle(
        figB.axes[0],
        "Does pooling beat single-corpus training?",
        "points above the y=x line = pooling improves held-out transfer",
    )
    figB.tight_layout()
    savefig_paper(figB, "pooled_probe_transfer_pooled_vs_single", dir=str(figures_dir))
    plt.close(figB)
    _log(f"figures written to {figures_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #1092 pooled/LODO probe-transfer analysis.")
    ap.add_argument("--out", default="eval_results/issue_1092/pooled-probe-transfer")
    ap.add_argument("--figures", default="figures/issue_1092/")
    ap.add_argument("--n-draws", dest="n_draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="subsample each substrate; validity gate computed but not enforced",
    )
    ap.add_argument("--smoke-rows", dest="smoke_rows", type=int, default=150)
    ap.add_argument(
        "--layers", default=",".join(str(x) for x in LAYERS), help="comma layers to run"
    )
    ap.add_argument("--traits", default=",".join(TRAITS), help="comma traits to run")
    ap.add_argument(
        "--no-finalize",
        dest="no_finalize",
        action="store_true",
        help="skip validity gate + figures (per-cell parallel worker; merge step finalizes)",
    )
    ap.add_argument(
        "--merge",
        nargs="+",
        default=None,
        metavar="JSON",
        help="merge these per-cell JSONs (first = base w/ metadata) into --out, then finalize",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge:
        merge_and_finalize(args.merge, Path(args.out), Path(args.figures))
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
