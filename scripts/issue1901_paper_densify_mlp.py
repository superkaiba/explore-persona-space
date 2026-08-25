#!/usr/bin/env python3
"""#1901 paper densification, GPU escalation 2: neural (MLP) map fits.

Job A — 28-layer neural curve at n_train=3,600: per layer 0..27, one
mlp_w8192 fit (cx_last -> v_x) on the #779 pass_b all-layer bundle under
the EXACT fair-comparison protocol that produced the banked parity
anchors (``issue779_fitter_fair_comparison.batched_mlp_fit``: full-batch
padded-bmm AdamW, lr 3e-4, wd 1e-4, max 300 epochs, patience 20,
internal seeded 10% early-stop split of the train rows, seed 42; pinned
``fixed_split(5000, 3600, 400, 1000, seed=42)``). The parity chunk
[L19, L26] runs FIRST and is gated against the banked mlp cells in
``eval_results/issue_779/fitter-fair-comparison/fair_comparison.json``
(tolerance ``--parity-tol``, default 0.02 — the #779 n1m validity-gate
convention); a beyond-tolerance mismatch raises before the remaining
layers run.

Job B — L19 scaling points at n_train in {5,000, 10,000}: nested seeded
subsamples of the #1491 scale7_refit ``train_25k`` split, fit with the
#779 n1m minibatch trainer (``issue779_ffc_n1m_fits.fit_mlp``: lr 3e-4,
batch 4096, max 300 epochs — the banked constants; NOTE the banked
#1491 n=25k point used lr 1e-3 / max 50 epochs, a recipe seam recorded
in the output meta), evaluated on the ladder's pinned test_1000.

Job C — mlp-scaling-densify (#1901 follow-up round, plan v15 §4): one
dense L19 scaling ladder, 8 fresh rungs {5k, 10k, 25k, 50k, 100k, 150k,
250k, 500k} + the banked 963k apply point, with mlp_w8192 AND ridge fit
on IDENTICAL train subsets per rung — ALL fresh rungs are within-store
``N1M.select_train`` lmsys draws under PYTHONHASHSEED=0 / --seed-b 0
(the v13 scale7-prefix sourcing is REMOVED: the 2026-08-25 on-pod G1
measurement refuted cross-store fold identity — 0/400 exact matches,
max|Δ| ≈ 37 vs the 2e-3 fp16-cast bound — so the output gates record
carries the static ``g1_cross_store`` provenance field instead of a
runtime gate; G1_CROSS_STORE below), plus the blockwise identity+bias
baseline. Gates: G2 RECORDED three-state sel-sha comparison vs the
banked bigN refits (PASS / FALLBACK-PARITY-PASS at --parity-tol /
FAIL ⇒ halt — testable pure function ``_g2_gate``); G3 whitened-CSLS
pool floor (n_pool ≥ K_CSLS+2). Every rung×arm cell reports the
standard metrics (_cell_metrics) PLUS the task-locked whitened-CSLS
battery (μ_A + shrunk-Cholesky(λ=0.1) whitening over the full 963,444
non-val/test train-pool Y; CSLS k=10 primary, whitened-cosine
diagnostic). Per-cell fingerprinted resume (sel_sha + code sha + store
revision + PYTHONHASHSEED + seeds + recipe — field-for-field, never
bare existence), checkpoint-per-cell perfit JSONs, fp16 test-pool
prediction npz staged for HF ``issue1901_mlpdense/analysis_tensors/``.
Phase c5 additionally writes the machine-readable supersession record
``superseded_cross_store_join.json`` (plan v15 §10 — the refuted
cross-store join + its tracked consumers). ``--smoke-chunks 4`` =
tiny-real smoke (rungs {1k n1m-lmsys, 2k n1m-lmsys — the 2k rung is the
seedwise endpoint}), full production path, ``*_smoke`` output names.

Per cell, all jobs report: pooled test R^2 + 95% bootstrap CI
(n_boot=1000), val R^2 on the pinned val_400, kNN retrieval
(euclidean + cosine, ks=1/5/10/50, pinned test pool), and the
closed-form identity+bias baseline (standing rule). Per-fit JSONs are
written incrementally; aggregates rewritten atomically after every
battery chunk / rung.

Early-stop protocol note (rides every meta block): both banked trainers
early-stop on an INTERNAL 10% split of the train rows; the pinned
val_400 is a reporting/selection split only. Reproduced verbatim here —
parity with the banked anchors requires it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Heavy imports AFTER load_dotenv() so the shared-VM thread caps (#847) bind
# in-process (torch freezes its intra-op pool from OMP_NUM_THREADS at import).
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as FFC  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
import issue1901_metric_battery as BAT  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.analysis.null_battery import shrunk_cholesky_from_cov  # noqa: E402
from explore_persona_space.atomic_io import savez_atomic  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue1901_paper_densify_mlp")

MLP_WIDTH = 8192
MLP_LR = 3e-4  # the banked #779 constant (docs/methodology/issue_779.md); NOT retuned per cell
KNN_KS = (1, 5, 10, 50)
PASS_B_HF_FILE = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
BANKED_FFC_JSON = (
    FFC.PROJECT_ROOT
    / "eval_results"
    / "issue_779"
    / "fitter-fair-comparison"
    / "fair_comparison.json"
)
BANKED_LADDER_JSON = (
    FFC.PROJECT_ROOT / "eval_results" / "issue_1491" / "scale_ladder" / "fits_scale7_refit.json"
)

RECIPE_CAVEAT = (
    "Recipe held FIXED across layers/sizes (1 GELU hidden layer, width 8192, AdamW lr 3e-4, "
    "wd 1e-4, max 300 epochs, patience 20) — NOT retuned per cell; the width/lr pair was "
    "val-selected once at L19 in the #779 fair-comparison round and inherited here, so "
    "off-anchor cells are recipe-menu-limited reads."
)
EARLY_STOP_NOTE = (
    "Early stop uses the banked trainers' INTERNAL seeded 10% split of the train rows "
    "(not the pinned val_400, which is the selection/reporting split) — reproduced verbatim "
    "for parity with the banked #779 anchors."
)

# ── Job C constants (plan v13 §4/§10) ───────────────────────────────────────────

N1M_CAPTURE_PREFIX = f"{N1G.HF_PREFIX}/final_token_capture"
PASS_B_STAGE_PREFIX = "issue779_monitoring/analysis_tensors/pass_b"
WEIGHTS_L19_PREFIX = f"{BAT.WEIGHTS_PREFIX}/L19"  # issue779_monitoring/n1m_readout/weights/L19
WHITEN_LAMBDA = 0.1  # task-locked shrinkage (issue2202 convention; plan §4 c2)
WHITEN_BLOCK = 65_536  # fp64 accumulation block (rows) for the pool cov/mean passes
IB_BLOCK = 65_536  # identity+bias blocked accumulation (plan §9 LARGEST-CELL keying)
DENSE_ENDPOINT_NS = (50_000, 500_000)  # 3-seed MLP dispersion endpoints (plan §4 c3)
# Plan v15 within-store pivot: ALL fresh rungs draw from the n1m lmsys pool via
# N1M.select_train (the scale7-prefix sourcing + G1 runtime gate are removed).
# The gates record carries this static provenance field instead of a runtime G1.
G1_CROSS_STORE = (
    "refuted-2026-08-25 — scale7 eval pools are not the n1m pinned rows; see "
    "epm:progress 2026-08-25T03:18:29Z + artifacts/mlpdense-smoke-r3-g1fail.log"
)
# c0 weights staging narrows to the two payloads job C actually consumes
# (plan §9 arithmetic ~0.29 GB vs the full 2.8 GB L19 prefix).
WEIGHTS_L19_FILES = ("mlp_w8192.pt", "ridge.pt")
# Registered run-shape expectations (plan v13 §0/§3/§9/§10). _validate_run_shape
# asserts the REALIZED shape against the invocation mode's OWN registered set
# BEFORE any whitening/fit compute — production AND smoke each carry explicit
# expectations; nothing is skip-under-smoke.
PRODUCTION_DENSE_NS = (5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000)
PRODUCTION_SEED_SET = frozenset({42, 43, 44})  # OPT-1 endpoint replication (plan §3)
PRODUCTION_CAPTURE_FILES = 1_920  # n1m capture chunk universe (plan §9, Hub-measured)
PRODUCTION_POOL_FULL = 963_444  # train pool = 3,600 pass_b-train + 959,844 captured (plan §9)
SMOKE_RUNG_SPECS = ((1_000, "n1m"), (2_000, "n1m"))  # plan §4 (v15) registered smoke rungs
DENSE_GAP_MARGIN = 0.01  # plan §3: D_gap = (S_mlp − S_ridge) − 0.01
MIXED1M_APPLY_TOL = 1e-3  # deterministic banked-weights apply parity (plan §7 kill criterion 3)
# Advisory wall check after the first n1m rung (50k): plan §9 basis ≈ mlp ~60 s + ridge
# + identity+bias/battery ⇒ ~120 s booked; >2× logs an advisory line (never aborts).
PLAN_FIRST_N1M_RUNG_WALL_S = 120.0

# G2 recorded selections (plan §4 c1; pasted from the committed bigN unit files —
# eval_results/issue_1901/paper_densify/bign/lmsys_{150k,500k}.json unit_key.sel_sha256,
# re-asserted against those files when present, _g2_recorded_shas below).
G2_RECORDED_SEL_SHAS = {
    "lmsys_150k": "fbba56ab7a5faa7cce015476593b2d8c36a9a6c6af3a690b07f865a08ec4a7f9",
    "lmsys_500k": "f9ab1707b23d0677d25eae9ce41fce1559346728e819ee8bdc2baaa90c5c55fe",
}
BIGN_UNIT_DIR = FFC.PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify" / "bign"
BANKED_BIGN_JSON = (
    FFC.PROJECT_ROOT
    / "eval_results"
    / "issue_1901"
    / "paper_densify"
    / "scaling_bigN_acc1_L19.json"
)
BANKED_LADDER_CELLS_JSON = (
    FFC.PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify" / "scaling_ladder_L19.json"
)
BANKED_MLP_SCALING_JSON = (
    FFC.PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify" / "mlp_scaling_L19.json"
)
BANKED_BATTERY_CONTEXT_JSON = (
    FFC.PROJECT_ROOT / "eval_results" / "issue_1901" / "metric_battery" / "context_arm.json"
)


# (arm, n) -> (banked json path, extractor, pasted value, kind). kind semantics
# (plan §4 c3 / §7, v15): the v13 "fold-exact" scale7 anchors are REMOVED with
# the scale7 sourcing (no fresh rung shares a banked fold exactly);
# "sha-conditional" halts only when the rung's realized sel-sha matches the
# recorded bigN selection (sel-sha-exact ⇒ fit-machinery drift, criterion-3
# class); "statistical" is recorded, never a halt (different selection
# realization is possible; G2 owns the mismatch disposition).
DENSE_PARITY_ANCHORS = {
    ("mlp", 150_000): (
        PD.BANKED_N1M_FITS,
        lambda d: d["per_point"]["lmsys_150k"]["predictors"]["mlp_w8192"]["whole_map_r2"],
        0.7880526998314361,
        "statistical",
    ),
    ("mlp", 500_000): (
        PD.BANKED_N1M_FITS,
        lambda d: d["per_point"]["lmsys_500k"]["predictors"]["mlp_w8192"]["whole_map_r2"],
        0.8074943555084623,
        "statistical",
    ),
    ("ridge", 150_000): (
        BANKED_BIGN_JSON,
        lambda d: d["per_point"]["lmsys_150k"]["ridge"]["whole_map_r2"],
        0.755515914218508,
        "sha-conditional",
    ),
    ("ridge", 500_000): (
        BANKED_BIGN_JSON,
        lambda d: d["per_point"]["lmsys_500k"]["ridge"]["whole_map_r2"],
        0.7609049151916738,
        "sha-conditional",
    ),
}
MIXED1M_APPLY_ANCHORS = {
    "mlp": (
        BANKED_BATTERY_CONTEXT_JSON,
        lambda d: d["per_layer"]["19"]["arms"]["mlp_w8192"]["r2"]["point"],
        0.8103576099860699,
    ),
    "ridge": (
        PD.BANKED_N1M_FITS,
        lambda d: d["per_point"]["mixed_1m"]["predictors"]["ridge"]["whole_map_r2"],
        0.7541708417500051,
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=FFC.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata best-effort; the fits do not depend on it
        return "unknown"


def _meta_common(args) -> dict:
    return {
        "script": "issue1901_paper_densify_mlp",
        "git_commit": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "recipe": {
            "arch": "1 hidden GELU layer, full-dim linear head",
            "width": MLP_WIDTH,
            "lr": MLP_LR,
            "weight_decay": FFC.MLP_WD,
            "max_epochs": FFC.MLP_MAX_EPOCHS,
            "patience": FFC.MLP_PATIENCE,
        },
        "recipe_caveat": RECIPE_CAVEAT,
        "early_stop_note": EARLY_STOP_NOTE,
        "knn": {"helper": "analysis.mapping_baselines.knn_retrieval", "ks": list(KNN_KS)},
    }


def _cell_metrics(pred_val, y_val, pred_te, y_te, n_boot, boot_seed) -> dict:
    return {
        "val_r2": float(PR._pooled_r2(pred_val, y_val)),
        "test_r2": float(PR._pooled_r2(pred_te, y_te)),
        "test_ci": FFC._bootstrap_recon_ci(pred_te, y_te, n_boot, boot_seed),
        "knn": {
            m: MB.knn_retrieval(pred_te, y_te, ks=KNN_KS, metric=m) for m in ("euclidean", "cosine")
        },
    }


def _identity_bias_cell(x_tr, y_tr, x_val, y_val, x_te, y_te) -> dict:
    pred_val = MB.identity_bias_predict(x_tr, y_tr, x_val)
    pred_te = MB.identity_bias_predict(x_tr, y_tr, x_te)
    return {
        "val_r2": float(PR._pooled_r2(pred_val, y_val)),
        "test_r2": float(PR._pooled_r2(pred_te, y_te)),
        "knn": {
            m: MB.knn_retrieval(pred_te, y_te, ks=KNN_KS, metric=m) for m in ("euclidean", "cosine")
        },
    }


def _write_perfit(out_dir: Path, name: str, obj: dict) -> None:
    perfit = out_dir / "perfit"
    perfit.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(perfit / name, obj)


def _banked_ffc_anchors() -> dict:
    banked = json.loads(BANKED_FFC_JSON.read_text())
    pl = banked["inputs"]["last"]["mlp"]["per_layer"]
    return {li: {"test_r2": pl[li]["test_r2"], "val_r2": pl[li]["val_r2"]} for li in pl}


def run_job_a(args, dev, out_path: Path) -> dict:
    bundle = FFC.load_pass_b(args.pass_b_path)
    n_ctx = FFC.corpus_len(bundle)
    assert n_ctx == 5000, f"pass_b corpus has {n_ctx} rows, expected 5000"
    train, val, test = FFC.fixed_split(n_ctx, 3600, 400, 1000, args.seed)
    layers_all = [int(x) for x in bundle["layers"]]
    assert layers_all == list(range(28)), f"unexpected pass_b layer list: {layers_all}"
    anchors = _banked_ffc_anchors()
    parity_layers = sorted(int(k) for k in anchors)  # [19, 26]

    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res.setdefault("per_layer", {})
    res["split"] = {
        "source": "issue779 pass_b fixed_split",
        "n_contexts": n_ctx,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
    }
    res["input"] = "cx_last"
    res["target"] = "v_x (same-layer mean-response profile)"

    # Parity chunk first (gate), then the rest in battery chunks.
    remaining = [li for li in layers_all if str(li) not in res["per_layer"]]
    order = [li for li in parity_layers if li in remaining] + [
        li for li in remaining if li not in parity_layers
    ]
    chunks = [order[i : i + args.battery_chunk] for i in range(0, len(order), args.battery_chunk)]
    # Never let non-parity layers ride in the gate chunk's battery ahead of the gate verdict.
    if chunks and any(li in parity_layers for li in chunks[0]):
        gate = [li for li in chunks[0] if li in parity_layers]
        rest = [li for li in chunks[0] if li not in parity_layers]
        chunks = [gate] + ([rest] if rest else []) + chunks[1:]

    for chunk in chunks:
        t0 = time.time()
        arrays: dict[int, tuple] = {}
        groups = []
        for li in chunk:
            X = FFC.input_layer(bundle, "last", li)
            Y = FFC.target_vx(bundle, li)
            arrays[li] = (X[train], Y[train], X[val], Y[val], X[test], Y[test])
            groups.append(
                FFC.MLPGroup(("mlp", "last", li), arrays[li][0], arrays[li][1], MLP_WIDTH, MLP_LR)
            )
        logger.info("[job-a] battery chunk %s (G=%d)", chunk, len(groups))
        fits = FFC.batched_mlp_fit(
            groups, hidden=MLP_WIDTH, lr=MLP_LR, max_epochs=FFC.MLP_MAX_EPOCHS, dev=dev
        )
        for li in chunk:
            x_tr, y_tr, x_val, y_val, x_te, y_te = arrays[li]
            fit = fits[("mlp", "last", li)]
            entry = _cell_metrics(
                fit.predict(x_val), y_val, fit.predict(x_te), y_te, args.n_boot, args.seed + li
            )
            entry.update({"width": MLP_WIDTH, "lr": MLP_LR, "epochs_ran": fit.epochs_ran})
            entry["identity_bias_baseline"] = _identity_bias_cell(
                x_tr, y_tr, x_val, y_val, x_te, y_te
            )
            res["per_layer"][str(li)] = entry
            _write_perfit(args.out_dir, f"layer_curve_n3600_L{li:02d}.json", entry)
            logger.info(
                "[job-a] L%02d: test R2 %.4f (val %.4f, acc@1 %.3f, %d epochs)",
                li,
                entry["test_r2"],
                entry["val_r2"],
                entry["knn"]["euclidean"]["acc_at_k"][1],
                entry["epochs_ran"],
            )
        C.write_json_atomic(out_path, res)
        logger.info("[job-a] chunk done in %.1fs", time.time() - t0)

        # Parity gate: after the chunk containing the banked anchors, verify.
        done_parity = [li for li in parity_layers if str(li) in res["per_layer"]]
        if len(done_parity) == len(parity_layers) and "parity" not in res:
            parity = {}
            for li in parity_layers:
                mine = res["per_layer"][str(li)]["test_r2"]
                banked_v = anchors[str(li)]["test_r2"]
                parity[str(li)] = {
                    "banked_test_r2": banked_v,
                    "this_run_test_r2": mine,
                    "delta": mine - banked_v,
                    "tolerance": args.parity_tol,
                    "pass": abs(mine - banked_v) <= args.parity_tol,
                }
            parity["source"] = str(BANKED_FFC_JSON.relative_to(FFC.PROJECT_ROOT))
            res["parity"] = parity
            C.write_json_atomic(out_path, res)
            bad = {k: v for k, v in parity.items() if isinstance(v, dict) and not v["pass"]}
            if bad:
                raise RuntimeError(
                    f"parity gate FAILED vs banked #779 mlp anchors (tol {args.parity_tol}): "
                    f"{json.dumps(bad)} — investigate before densifying the remaining layers"
                )
            logger.info("[job-a] parity gate PASS: %s", json.dumps(parity))
    return res


def run_job_b(args, dev, out_path: Path) -> dict:
    asm = LF._assemble_scale_layer(args.ladder_hf_prefix, args.ladder_layer, args.cache_dir)
    X, Y = asm["X"], asm["Y"]
    tr, val, te = asm["tr"], asm["val"], asm["te"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr))

    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res.setdefault("per_n", {})
    res["layer"] = args.ladder_layer
    res["store"] = {
        "hf_prefix": args.ladder_hf_prefix,
        "n_realized": asm["n_realized"],
        "subsample": "nested seeded subsamples of train_25k (one permutation, first-n prefixes)",
        "subsample_seed": args.seed,
    }
    banked_25k = None
    if BANKED_LADDER_JSON.exists():
        banked = json.loads(BANKED_LADDER_JSON.read_text())
        banked_25k = {
            "n_train": 25000,
            "test_r2": banked["predictors"]["mlp_w8192"]["test_r2"],
            "meta": banked["predictors"]["mlp_w8192"]["meta"],
            "knn_euclidean": banked["knn_retrieval"]["mlp_w8192"]["euclidean"],
            "source": str(BANKED_LADDER_JSON.relative_to(FFC.PROJECT_ROOT)),
            "recipe_seam": (
                "banked #1491 n=25k point used lr 1e-3 / max 50 epochs / seed 0; this round's "
                "5k/10k points use the fixed lr 3e-4 / max 300 epochs recipe — joining them "
                "mixes recipes"
            ),
        }
    res["banked_25k_reference"] = banked_25k

    ev = np.concatenate([val, te])
    for n in args.scaling_ns:
        if str(n) in res["per_n"]:
            continue
        sub = np.sort(tr[perm[:n]])
        t0 = time.time()
        pred_ev, fit_meta = N1M.fit_mlp(
            X, Y, sub, ev, MLP_WIDTH, MLP_LR, FFC.MLP_MAX_EPOCHS, N1M.MLP_BATCH, args.seed, dev
        )
        pred_val, pred_te = pred_ev[: len(val)], pred_ev[len(val) :]
        entry = _cell_metrics(pred_val, Y[val], pred_te, Y[te], args.n_boot, args.seed + n)
        entry["fit_meta"] = fit_meta
        entry["trainer"] = "issue779_ffc_n1m_fits.fit_mlp (minibatch 4096)"
        entry["identity_bias_baseline"] = _identity_bias_cell(
            X[sub], Y[sub], X[val], Y[val], X[te], Y[te]
        )
        res["per_n"][str(n)] = entry
        _write_perfit(args.out_dir, f"scaling_L{args.ladder_layer}_n{n}.json", entry)
        C.write_json_atomic(out_path, res)
        logger.info(
            "[job-b] n=%d: test R2 %.4f (val %.4f, acc@1 %.3f) in %.1fs",
            n,
            entry["test_r2"],
            entry["val_r2"],
            entry["knn"]["euclidean"]["acc_at_k"][1],
            time.time() - t0,
        )
    return res


# ── Job C: dense L19 scaling ladder (mlp-scaling-densify follow-up round) ───────


@dataclass
class GateVerdict:
    """G2 verdict record (plan §4 c1). ``verdict`` ∈ {"PASS",
    "FALLBACK-PARITY-PASS", "FAIL"}; ``downgrade_recorded`` is True exactly on
    the fallback branch (sel-sha mismatch absorbed by statistical parity)."""

    verdict: str
    downgrade_recorded: bool
    detail: dict


def _g2_gate(
    realized_shas: dict,
    recorded_shas: dict,
    refit_r2s: dict,
    banked_r2s: dict,
    tol: float,
) -> GateVerdict:
    """G2 RECORDED three-state comparison (plan §4 c1; pure + testable).

    Per recorded point: realized ``select_train`` sel-sha vs the recorded bigN
    sel-sha. All match ⇒ PASS. Any mismatch does NOT halt by itself — the
    fallback predicate is statistical parity of the fresh ridge refits vs the
    banked bigN R² at ``tol`` on EVERY recorded point ⇒ FALLBACK-PARITY-PASS
    with ``downgrade_recorded=True``; a mismatch AND any parity breach ⇒ FAIL
    (the caller halts — plan §7 kill criterion 2's two-part predicate).
    """
    detail: dict = {"tol": float(tol), "points": {}}
    mismatched = []
    for name in sorted(recorded_shas):
        realized = realized_shas[name]
        match = bool(realized == recorded_shas[name])
        delta = abs(float(refit_r2s[name]) - float(banked_r2s[name]))
        detail["points"][name] = {
            "recorded_sel_sha256": recorded_shas[name],
            "realized_sel_sha256": realized,
            "sha_match": match,
            "refit_r2": float(refit_r2s[name]),
            "banked_r2": float(banked_r2s[name]),
            "abs_delta": delta,
            "parity_within_tol": bool(delta <= tol),
        }
        if not match:
            mismatched.append(name)
    detail["mismatched"] = mismatched
    if not mismatched:
        return GateVerdict("PASS", False, detail)
    if all(p["parity_within_tol"] for p in detail["points"].values()):
        return GateVerdict("FALLBACK-PARITY-PASS", True, detail)
    return GateVerdict("FAIL", False, detail)


def _g2_recorded_shas() -> dict:
    """Recorded bigN selections — pasted constants, re-asserted against the
    committed bign unit files when present (the _banked_parity_target pattern)."""
    out = {}
    for name, pasted in G2_RECORDED_SEL_SHAS.items():
        p = BIGN_UNIT_DIR / f"{name}.json"
        if p.exists():
            got = json.loads(p.read_text())["unit_key"]["sel_sha256"]
            assert got == pasted, (str(p), got, pasted)
        out[name] = pasted
    return out


def _read_memtotal_gb() -> float:
    """Host MemTotal in GB (1e9 bytes) from /proc/meminfo."""
    for line in Path("/proc/meminfo").read_text().split("\n"):
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def _stage_job_c(args, smoke: bool):
    """Phase c0: stage capture + pass_b + banked weights under stage_root, with
    a resume-aware fallocate headroom probe, the MemTotal floor, and the
    realized-keys check on the banked weight payloads. The weights stage is
    NARROWED to the two consumed payloads (WEIGHTS_L19_FILES, ~0.29 GB vs the
    2.8 GB full L19 prefix — plan v15 §9). Returns
    (capture_dir, pass_b_path, weights_dir, store_revision)."""
    stage_root: Path = args.stage_root
    stage_root.mkdir(parents=True, exist_ok=True)

    # Store revision recorded ONCE at stage time (resume fingerprint input; a
    # fresh stage on a new pod re-records — plan §4 c3). NOT the repo head at
    # fit time: the shared data repo advances constantly.
    rev_file = stage_root / ".stage_revision.json"
    if rev_file.exists():
        # C1: validate the sidecar STRUCTURE before trusting it — a malformed /
        # truncated sidecar must reject loudly (delete it + fully restage into a
        # fresh stage_root to re-pin), never thread garbage into every download.
        try:
            rev_rec = json.loads(rev_file.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"stage-revision sidecar {rev_file} is not valid JSON ({e}) — delete it and "
                "fully restage (fresh stage_root) to re-pin the store revision"
            ) from e
        store_revision = rev_rec.get("revision") if isinstance(rev_rec, dict) else None
        if not (
            isinstance(store_revision, str)
            and len(store_revision) == 40
            and all(c in "0123456789abcdef" for c in store_revision)
        ):
            raise RuntimeError(
                f"stage-revision sidecar {rev_file} malformed (revision={store_revision!r}, "
                "expected a 40-hex Hub commit sha) — delete it and fully restage to re-pin"
            )
    else:
        # r3 (stage-revision-unpinned residual): never MINT a fresh pin over a
        # stage_root that already holds staged targets — those files landed under
        # a DIFFERENT (or unknown/unpinned) revision, and stage_prefix's
        # size-match resume would silently adopt them under the new pin, mixing
        # file generations across Hub commits. Refuse with the restage recipe.
        preexisting = [
            prefix
            for prefix in (N1M_CAPTURE_PREFIX, PASS_B_STAGE_PREFIX, WEIGHTS_L19_PREFIX)
            if (stage_root / prefix).exists()
            and any(p.is_file() for p in (stage_root / prefix).rglob("*"))
        ]
        if preexisting:
            raise RuntimeError(
                f"stage_root {stage_root} holds staged files under {preexisting} but no "
                f"{rev_file.name} — their download revision is unknown, and minting a fresh "
                "pin would let the size-match resume mix file generations. Use a fresh "
                "--stage-root (or delete the staged prefixes) to force a fully pinned restage."
            )
        info = hub.retry_transient(
            lambda: HfApi().repo_info(C.HF_DATA_REPO, repo_type="dataset"),
            what="data-repo revision probe",
        )
        store_revision = str(info.sha)
        C.write_json_atomic(
            rev_file,
            {
                "revision": store_revision,
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    logger.info("[job-c] store revision (stage-time): %s", store_revision)

    # MemTotal floor (plan §9: c1 assemble peaks ≈55 GB; the 963k cells key the
    # ~100 GB floor). Production assert; smoke stages a partial pool — WARN only
    # (gate-calibration parity: a production-scale host floor on a tiny smoke
    # host would kill the very leg the smoke exists to exercise).
    mem_gb = _read_memtotal_gb()
    if mem_gb < 100.0:
        msg = (
            f"host MemTotal {mem_gb:.1f} GB < 100 GB floor (plan §9: c1 assemble ≈55 GB peak, "
            "963k cells ≈28-32 GB resident)"
        )
        if smoke:
            logger.warning("[job-c] %s — smoke mode, continuing on a partial pool", msg)
        else:
            raise RuntimeError(msg)

    # Resume-aware headroom probe: pending (not-yet-staged) bytes only, so a
    # resumed pod whose capture already occupies the quota is not dead-locked
    # by a fresh-run-sized floor. fallocate canary catches MooseFS EDQUOT.
    prefixes = [
        (N1M_CAPTURE_PREFIX, args.smoke_chunks if smoke else None, None),
        (PASS_B_STAGE_PREFIX, None, None),
        (WEIGHTS_L19_PREFIX, None, WEIGHTS_L19_FILES),
    ]
    pending = 0
    for prefix, max_files, only_files in prefixes:
        files = PD._list_prefix(prefix, revision=store_revision)
        if only_files is not None:
            files = PD.filter_listing_only_files(files, only_files, prefix)
        if max_files is not None:
            files = files[:max_files]
        pending += sum(
            sz
            for p, sz in files
            if not ((stage_root / p).exists() and (stage_root / p).stat().st_size == sz)
        )
    if pending:
        assert_out_root_headroom(stage_root, need_gb=pending / 1e9 * 1.2 + 2.0, phase="job-c-stage")

    capture_dir = PD.stage_prefix(
        N1M_CAPTURE_PREFIX,
        stage_root,
        max_files=(args.smoke_chunks if smoke else None),
        workers=args.stage_workers,
        revision=store_revision,
    )
    passb_dir = PD.stage_prefix(
        PASS_B_STAGE_PREFIX, stage_root, workers=args.stage_workers, revision=store_revision
    )
    weights_dir = PD.stage_prefix(
        WEIGHTS_L19_PREFIX,
        stage_root,
        workers=args.stage_workers,
        revision=store_revision,
        only_files=WEIGHTS_L19_FILES,
    )
    # Realized-keys check on the banked weight payloads (plan §10 fitness (f);
    # reuses the battery's apply_map-contract checker — mmap read, fail loud).
    BAT._realized_keys_check(weights_dir / "mlp_w8192.pt", "mlp")
    BAT._realized_keys_check(weights_dir / "ridge.pt", "ridge")
    return capture_dir, passb_dir / "train_context_vectors.pt", weights_dir, store_revision


def _assemble_n1m(args, capture_dir: Path, pass_b_path: Path, store_revision: str):
    """Phase c1 (n1m side): the exact bigN Namespace shim (every args.<attr>
    N1M.assemble reads on every reachable branch — #1776 hand-built-Namespace
    rule, mirrored from issue1901_paper_densify.phase_bign).

    PHASE_IDEMPOTENCY_EXEMPT (C4, c1): assembly re-builds X/Y IN MEMORY on
    every process start BY DESIGN — the (964,844 × 3,584) fp32 pair (~28 GB)
    is rebuilt from the ALREADY-STAGED local chunks (_stream_local_chunks
    mmap-slices, zero network), and persisting a second ~28 GB assembled copy
    would double the pod disk footprint for no wall saving (plan §9 books the
    re-assemble at ≤15 min). The only re-entry side effect is the manifest
    fetch, which is revision-pinned, flock-serialized, and resume-complete.
    """
    ns = argparse.Namespace(
        pass_b=pass_b_path,
        manifest_from_hf=True,
        manifest_hf_prefix=N1G.HF_PREFIX,
        manifest_revision=store_revision,
        out_dir=args.work_dir,
        n1m_capture_dir=capture_dir,
        fresh_stream=False,
        hf_prefix=N1M_CAPTURE_PREFIX,
        store_revision=store_revision,
        orig_dir=args.orig_dir,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    X, Y, prov, r1_train, val, test, split = N1M.assemble(ns, layer=19)
    pools = N1M._pool_rows(prov, r1_train, X.shape[0], val, test)
    return X, Y, pools, val, test, split


def _blocked_mean(arr, rows, dev, block: int, phase: str) -> np.ndarray:
    """fp64 mean of arr[rows] accumulated in index blocks on ``dev``."""
    rows = np.asarray(rows, dtype=np.int64)
    d = arr.shape[1]
    s = torch.zeros(d, dtype=torch.float64, device=dev)
    n_chunks = (rows.size + block - 1) // block
    t0 = time.time()
    for k, i in enumerate(range(0, rows.size, block)):
        xb = torch.as_tensor(np.asarray(arr[rows[i : i + block]]), dtype=torch.float64).to(dev)
        s += xb.sum(dim=0)
        print(f"[{phase}] unit {k + 1}/{n_chunks} elapsed={time.time() - t0:.1f}s", flush=True)
    return (s / rows.size).cpu().numpy()


def _blocked_mean_cov(arr, rows, dev, block: int, phase: str):
    """fp64 (cov ddof=1, mean) of arr[rows], accumulated in index blocks on
    ``dev`` (chunked_cov formula — issue2202_failchar convention; plan §4 c2)."""
    rows = np.asarray(rows, dtype=np.int64)
    n, d = rows.size, arr.shape[1]
    assert n > 1, f"cov needs n > 1 rows, got {n}"
    s = torch.zeros(d, dtype=torch.float64, device=dev)
    q = torch.zeros((d, d), dtype=torch.float64, device=dev)
    n_chunks = (n + block - 1) // block
    t0 = time.time()
    for k, i in enumerate(range(0, n, block)):
        xb = torch.as_tensor(np.asarray(arr[rows[i : i + block]]), dtype=torch.float64).to(dev)
        s += xb.sum(dim=0)
        q += xb.T @ xb
        print(f"[{phase}] unit {k + 1}/{n_chunks} elapsed={time.time() - t0:.1f}s", flush=True)
    mu = s / n
    cov = (q - n * torch.outer(mu, mu)) / (n - 1)
    return cov.cpu().numpy(), mu.cpu().numpy()


def _job_c_whiten(args, X, Y, pool_rows, whiten_npz: Path, dev, store_revision: str):
    """Phase c2: task-locked whitening stats over the train-pool answer states —
    μ_A + shrunk-Cholesky L (λ=0.1) of the fp64 pool cov (plus μ_C for the
    record). Resume keyed on the pool sel-sha + λ + store revision (never bare
    existence; the revision key rejects stats computed from a different store
    generation even when the row-index sha coincides)."""
    pool_sha = FFC._sha_ids(np.sort(np.asarray(pool_rows, dtype=np.int64)))
    if whiten_npz.exists():
        z = np.load(whiten_npz, allow_pickle=False)
        if (
            str(z["pool_sha256"]) == pool_sha
            and float(z["lam"]) == WHITEN_LAMBDA
            and "store_revision" in z.files
            and str(z["store_revision"]) == store_revision
        ):
            logger.info("[job-c] whiten stats resume-load (%s)", whiten_npz)
            return np.asarray(z["mu_A"]), np.asarray(z["L"]), pool_sha
        logger.info(
            "[job-c] whiten stats stale (pool sha / lambda / store-revision mismatch) — recomputing"
        )
    cov, mu_a = _blocked_mean_cov(Y, pool_rows, dev, WHITEN_BLOCK, "c2-cov")
    ell = shrunk_cholesky_from_cov(cov, WHITEN_LAMBDA)
    mu_c = _blocked_mean(X, pool_rows, dev, WHITEN_BLOCK, "c2-mu-c")
    savez_atomic(  # process-unique temp + handle write (np.savez .npz-append trap; #2336)
        whiten_npz,
        mu_A=mu_a,
        mu_C=mu_c,
        L=ell,
        lam=WHITEN_LAMBDA,
        n_train=int(len(pool_rows)),
        pool_sha256=pool_sha,
        store_revision=store_revision,
    )
    logger.info("[job-c] whiten stats written: %s (n_train=%d)", whiten_npz, len(pool_rows))
    return mu_a, ell, pool_sha


def _whitened_battery(pred_te, y_te, mu_a, ell, k: int) -> dict:
    """Phase c4 whitened battery: z-whiten preds + pool (solve_triangular
    against the shrunk-Cholesky L — the issue2202 convention), cosine matrix S,
    CSLS (k=10 primary, ``issue1901_metric_battery.csls_scores``) + the
    whitened-cosine-no-CSLS diagnostic; rank by −score with mid-rank ties
    (``rank_matrix_for_cols`` convention) → acc@{1,5,10} + MRR."""
    zq = solve_triangular(ell, (np.asarray(pred_te, np.float64) - mu_a).T, lower=True).T
    zp = solve_triangular(ell, (np.asarray(y_te, np.float64) - mu_a).T, lower=True).T
    n_pool = zp.shape[0]
    assert n_pool >= k + 2, f"G3: n_pool={n_pool} < K_CSLS+2={k + 2}"
    qn = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-12)
    pn = zp / (np.linalg.norm(zp, axis=1, keepdims=True) + 1e-12)
    S = qn @ pn.T
    out = {}
    diag_cols = np.arange(n_pool)
    for label, scores in (("whitened_csls", BAT.csls_scores(S, k=k)), ("whitened_cosine", S)):
        R = BAT.rank_matrix_for_cols(-scores, diag_cols)
        ranks = R[diag_cols, diag_cols]  # rank of each row's true target (pool == true)
        out[label] = {
            "acc_at_k": {int(kk): float((ranks <= kk).mean()) for kk in (1, 5, 10)},
            "chance_at_k": {int(kk): float(kk / n_pool) for kk in (1, 5, 10)},
            "median_rank": float(np.median(ranks)),
            "mrr": float((1.0 / ranks).mean()),
            "n_pool": int(n_pool),
            "k_csls": int(k),
            "whitening": {"lam": WHITEN_LAMBDA, "stats": "train-pool mu_A + shrunk-Cholesky L"},
        }
    return out


def _save_pred_npz(preds_dir: Path, stem: str, pred_te, te_rows, meta: dict) -> None:
    """fp16 test-pool predictions, atomically written (plan §10 HF artifact)."""
    savez_atomic(  # process-unique temp + handle write (np.savez .npz-append trap; #2336)
        preds_dir / f"{stem}.npz",
        pred_fp16=np.asarray(pred_te, dtype=np.float16),
        rows=np.asarray(te_rows, dtype=np.int64),
        **{k: str(v) for k, v in meta.items()},
    )


def _dense_fingerprint(
    args, *, n, source, sel_name, sel_sha, store_revision, arm, seeds, metrics
) -> dict:
    """Per-cell resume fingerprint (plan §4 c3): field-for-field match ⇒ skip,
    any mismatch ⇒ refit — never bare file existence. Keys are generating
    parameters only (machine-stable; no recomputed-float hashing). ``metrics``
    (M1) pins the METRIC-side identity too: n_boot + bootstrap convention,
    eval-split shas + eval store, whitening pool sha/λ/helper, CSLS k, kNN ks —
    a persisted cell whose metric instrument changed must refit, not resume."""
    if arm == "mlp":
        recipe = {
            "width": MLP_WIDTH,
            "lr": MLP_LR,
            "weight_decay": FFC.MLP_WD,
            "max_epochs": FFC.MLP_MAX_EPOCHS,
            "patience": FFC.MLP_PATIENCE,
            "batch": N1M.MLP_BATCH,
            "trainer": "issue779_ffc_n1m_fits.fit_mlp",
        }
    elif arm == "ridge":
        recipe = {
            "lambda_grid": ["logspace", -3, 8, 23],
            "ridge_block": int(args.ridge_block),
            "selection": "val-lambda (primal, streaming)",
        }
    elif arm == "identity_bias":
        recipe = {"estimator": "identity_bias_predict_blocked", "block": IB_BLOCK}
    elif arm in ("mlp_apply", "ridge_apply"):
        recipe = {"estimator": "banked-weights apply_map", "weights_prefix": WEIGHTS_L19_PREFIX}
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "arm": arm,
        "n": int(n),
        "source": source,
        "sel_name": sel_name,
        "sel_sha256": sel_sha,
        "code_sha": _git_sha(),
        "store_revision": store_revision,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "seed": int(args.seed),
        "seed_b": int(args.seed_b),
        "seeds": [int(s) for s in seeds],
        "recipe": recipe,
        "metrics": metrics,
        "smoke_chunks": int(args.smoke_chunks),
        "layer": 19,
    }


def _load_resumed_cell(perfit_path: Path, fingerprint: dict):
    """Fingerprinted resume: return the persisted cell iff its fingerprint
    matches field-for-field; None otherwise (⇒ refit)."""
    if not perfit_path.exists():
        return None
    prev = json.loads(perfit_path.read_text())
    if prev.get("fingerprint") == fingerprint:
        return prev
    logger.info("[job-c] %s fingerprint mismatch — refitting", perfit_path.name)
    return None


def _rung_parity(arm: str, n: int, got_r2: float, tol: float, *, sha_match: bool):
    """Per-rung parity row vs the banked anchors (DENSE_PARITY_ANCHORS kinds).

    NO smoke branch (C3): the registered smoke rungs {1k, 2k} intersect no
    anchor key (validator-enforced, _validate_run_shape), so every reached
    anchor row runs the FULL production halt logic in both modes."""
    key = (arm, int(n))
    if key not in DENSE_PARITY_ANCHORS:
        return None
    path, extract, pasted, kind = DENSE_PARITY_ANCHORS[key]
    want = PD._banked_parity_target(path, extract, pasted)
    row = {
        "cell": f"dense-{arm}-n{n}",
        "got_r2": float(got_r2),
        "banked_r2": float(want),
        "tol": float(tol),
        "kind": kind,
        "pass": bool(abs(got_r2 - want) <= tol),
    }
    halt = kind == "fold-exact" or (kind == "sha-conditional" and sha_match)
    if halt and not row["pass"]:
        raise RuntimeError(f"PARITY GATE FAILED: {json.dumps(row)}")
    logger.info("[job-c][parity] %s %s", "PASS" if row["pass"] else "RECORDED-DELTA", row)
    return row


def _assert_registered_seed_set(seed: int, endpoint_seeds) -> list[int]:
    """Plan §3 OPT-1 seed pin, binding in BOTH modes: --seed + --endpoint-seeds
    must realize EXACTLY {42, 43, 44} with no duplicates. Returns the seed list
    in argument order (primary first)."""
    seeds_all = [int(seed)] + [int(s) for s in endpoint_seeds]
    if len(seeds_all) != len(set(seeds_all)):
        raise RuntimeError(f"duplicate seeds in --seed/--endpoint-seeds: {seeds_all}")
    if set(seeds_all) != PRODUCTION_SEED_SET:
        raise RuntimeError(
            f"endpoint seed set {sorted(set(seeds_all))} != registered "
            f"{sorted(PRODUCTION_SEED_SET)} (plan §3 OPT-1; binds in production AND smoke)"
        )
    return seeds_all


def _validate_run_shape(
    *,
    smoke: bool,
    smoke_chunks: int,
    n_capture_files: int,
    seed: int,
    endpoint_seeds,
    rung_specs,
    split: dict,
    n_pool_full: int,
) -> dict:
    """Pre-fit production/smoke shape validator (C2; plan §0/§3/§9/§10).

    Shape-parametric by invocation mode — each mode asserts its OWN registered
    expectations, never skip-under-smoke. Production: exact rung set, exact
    seed set {42,43,44}, 1,920 staged capture chunks, captured == manifest
    rows, 963,444-row train pool. Smoke: the registered smoke rungs, staged
    chunk count == --smoke-chunks, a nonempty partial pool bounded by the
    manifest, and rung∩anchor disjointness (the C3 dead-branch guarantee:
    no DENSE_PARITY_ANCHORS key is reachable at smoke n). Runs BEFORE the
    whitening/fit phases; raises RuntimeError per violation; returns the audit
    record persisted under gates["shape"]."""
    seeds_all = _assert_registered_seed_set(seed, endpoint_seeds)
    n_cap = int(split["n_new_captured"])
    n_man = int(split["n_new_manifest"])
    rung_ns = [int(n) for n, _ in rung_specs]
    audit = {
        "mode": "smoke" if smoke else "production",
        "seeds": sorted(seeds_all),
        "n_capture_files": int(n_capture_files),
        "rungs": [[int(n), s] for n, s in rung_specs],
        "n_new_captured": n_cap,
        "n_new_manifest": n_man,
        "n_pool_full": int(n_pool_full),
    }
    if smoke:
        if tuple((int(n), s) for n, s in rung_specs) != SMOKE_RUNG_SPECS:
            raise RuntimeError(
                f"smoke rung specs {rung_specs} != registered {SMOKE_RUNG_SPECS} (plan §4)"
            )
        if int(n_capture_files) != int(smoke_chunks):
            raise RuntimeError(
                f"staged capture chunk count {n_capture_files} != --smoke-chunks {smoke_chunks}"
            )
        if not (0 < n_cap <= n_man):
            raise RuntimeError(
                f"smoke captured rows {n_cap} not in (0, manifest {n_man}] — broken partial pool"
            )
        anchor_overlap = {int(n) for _, n in DENSE_PARITY_ANCHORS} & set(rung_ns)
        if anchor_overlap:
            raise RuntimeError(
                f"smoke rungs overlap DENSE_PARITY_ANCHORS at n={sorted(anchor_overlap)} — the "
                "smoke must never reach a production parity anchor (C3 dead-branch guarantee)"
            )
    else:
        if set(rung_ns) != set(PRODUCTION_DENSE_NS):
            raise RuntimeError(
                f"production rung set {sorted(set(rung_ns))} != registered "
                f"{sorted(PRODUCTION_DENSE_NS)} (plan §0 scoped rung set)"
            )
        if int(n_capture_files) != PRODUCTION_CAPTURE_FILES:
            raise RuntimeError(
                f"staged capture chunk count {n_capture_files} != {PRODUCTION_CAPTURE_FILES} "
                "(plan §9 Hub-measured chunk universe) — partial/over-staged capture"
            )
        if n_cap != n_man:
            raise RuntimeError(
                f"captured rows {n_cap} != manifest rows {n_man} — partial/torn capture stage"
            )
        if int(n_pool_full) != PRODUCTION_POOL_FULL:
            raise RuntimeError(
                f"train pool {n_pool_full} != registered {PRODUCTION_POOL_FULL} (plan §9 grain)"
            )
    return audit


def _endpoint_verdict(per_n: dict, endpoint_ns, seeds, gap_margin: float = DENSE_GAP_MARGIN):
    """Plan §3 verdict lattice over the two MLP dispersion endpoints (M2).

    Per seed s: S_mlp^(s) = R²_MLP,s(hi) − R²_MLP,s(lo) (seed-paired slope);
    S_mlp = seed MEAN of the slopes; S_ridge = R²_ridge(hi) − R²_ridge(lo)
    (closed-form ridge is seed-free); D_gap = (S_mlp − S_ridge) − gap_margin.
    DISJOINT + exhaustive: Confirmed ⇔ D_gap >= 0; Falsified ⇔ D_gap < 0.
    Descriptive sub-reads (reported, NO verdict weight): plateau_both
    (|S_mlp| < 0.01 — meaningful within Falsified) and ridge_decline_driven
    (S_mlp <= 0 — meaningful within Confirmed). Returns a structured
    {"computed": False, "note": ...} record when either endpoint rung or any
    per-seed block is absent (the smoke case: endpoints not realized at
    smoke n)."""
    lo_hi = sorted(int(n) for n in endpoint_ns)
    if len(lo_hi) != 2:
        return {"computed": False, "note": f"need exactly 2 endpoint rungs, have {lo_hi}"}
    lo, hi = lo_hi
    cells = {}
    for n in (lo, hi):
        rung_row = per_n.get(str(n))
        if not isinstance(rung_row, dict) or "mlp" not in rung_row or "ridge" not in rung_row:
            return {"computed": False, "note": f"endpoint rung n={n} absent or incomplete"}
        cells[n] = rung_row
    seeds = [int(s) for s in seeds]
    per_seed_r2: dict[int, dict[str, float]] = {}
    for n in (lo, hi):
        blocks = cells[n]["mlp"].get("seeds")
        if not isinstance(blocks, dict):
            return {"computed": False, "note": f"endpoint rung n={n} lacks per-seed mlp blocks"}
        missing = [s for s in seeds if str(s) not in blocks]
        if missing:
            return {"computed": False, "note": f"endpoint rung n={n} missing seed blocks {missing}"}
        per_seed_r2[n] = {str(s): float(blocks[str(s)]["test_r2"]) for s in seeds}
    slopes = {str(s): per_seed_r2[hi][str(s)] - per_seed_r2[lo][str(s)] for s in seeds}
    s_mlp = float(np.mean(list(slopes.values())))
    s_ridge = float(cells[hi]["ridge"]["test_r2"]) - float(cells[lo]["ridge"]["test_r2"])
    d_gap = (s_mlp - s_ridge) - float(gap_margin)
    return {
        "computed": True,
        "endpoints": {"lo": lo, "hi": hi},
        "seeds": seeds,
        "per_seed_test_r2": {str(n): per_seed_r2[n] for n in (lo, hi)},
        "seed_paired_slopes": slopes,
        "S_mlp": s_mlp,
        "S_mlp_seed_std": float(np.std(np.array(list(slopes.values())))),
        "S_ridge": s_ridge,
        "gap_margin": float(gap_margin),
        "D_gap": float(d_gap),
        "verdict": "Confirmed" if d_gap >= 0 else "Falsified",
        "sub_reads": {
            "plateau_both": bool(abs(s_mlp) < 0.01),
            "ridge_decline_driven": bool(s_mlp <= 0.0),
        },
        "note": (
            "plan v13 §3 lattice: Confirmed ⇔ D_gap >= 0, Falsified ⇔ D_gap < 0 (disjoint + "
            "exhaustive); sub-reads descriptive only. Per-rung top-level mlp values are the "
            "seed-42 DISPLAY cell (top_level_is_display_seed); the registered summaries here "
            "aggregate the per-seed blocks."
        ),
    }


def _annotate_g2_percell(out_dir: Path, perfit_tag: str, g2: GateVerdict) -> list[str]:
    """M3: after G2 resolution, durably annotate the two recorded-selection
    ridge perfit cells (150k/500k) with the gate verdict + their own point
    detail — atomic rewrite, idempotent, resume-safe (the fingerprint compare
    reads only the "fingerprint" key, so the added "g2" key never invalidates
    a resume). Returns the annotated file names."""
    annotated = []
    for name, n_pt in (("lmsys_150k", 150_000), ("lmsys_500k", 500_000)):
        p = out_dir / "perfit" / f"{perfit_tag}_L19_n{n_pt}_ridge.json"
        if not p.exists():
            continue
        cell = json.loads(p.read_text())
        cell["g2"] = {
            "verdict": g2.verdict,
            "downgrade_recorded": g2.downgrade_recorded,
            "point": g2.detail["points"].get(name),
        }
        C.write_json_atomic(p, cell)
        annotated.append(p.name)
    return annotated


def _dense_rung_specs(dense_ns, smoke: bool) -> list[tuple[int, str]]:
    """Rung plan (plan v15 §4): production = every --dense-ns rung as a
    within-store n1m lmsys draw (sorted, deduped); smoke = the registered
    SMOKE_RUNG_SPECS verbatim. Pure — asserts nothing (the registered-set
    checks live in run_job_c / _validate_run_shape)."""
    if smoke:
        return [(int(n), s) for n, s in SMOKE_RUNG_SPECS]
    return [(int(n), "n1m") for n in sorted(set(int(n) for n in dense_ns))]


def _superseded_join_record() -> dict:
    """The plan v15 §10 supersession record: the v13 cross-store join (scale7
    prefix rungs plotted as one ladder with the n1m bigN points) is REFUTED —
    the scale7 eval pools are not the n1m pinned rows — and the within-store
    dense ladder (mlp_scaling_dense_L19.json) replaces it. The banked JSONs
    stay valid WITHIN their own store/folds; only the cross-store join is
    superseded. Consumers enumerated at implementation time via
    `grep -rl 'scaling_bigN_acc1_L19.json\\|scaling_ladder_L19.json\\|mlp_scaling_L19.json'`
    over docs/ + scripts/ (producers + tasks/ excluded)."""
    return {
        "kind": "superseded-cross-store-join",
        "superseded_join": {
            "what": (
                "the v13 cross-store scaling join: scale7-prefix rungs (5k-25k, "
                "issue1491_scale_ladder/scale7_refit folds) plotted as ONE ladder with "
                "the n1m capture bigN points (150k/500k/963k)"
            ),
            "why_refuted": (
                "G1 cross-store fold identity FAILED on-pod 2026-08-25: 0/400 scale7 "
                "val rows content-equal to the n1m pinned rows at L19 (max |dY| ~ 37 vs "
                "the 2e-3 fp16-cast bound) — the two stores' eval pools are different "
                "rows, so cross-store cells are not on one comparable ladder"
            ),
            "banked_files": [
                str(BANKED_BIGN_JSON.relative_to(FFC.PROJECT_ROOT)),
                str(BANKED_LADDER_CELLS_JSON.relative_to(FFC.PROJECT_ROOT)),
                str(BANKED_MLP_SCALING_JSON.relative_to(FFC.PROJECT_ROOT)),
            ],
            "disposition": (
                "each banked JSON stays valid WITHIN its own store/folds; any figure or "
                "claim joining scale7 cells with n1m cells on one x-axis is superseded by "
                "the within-store dense ladder"
            ),
        },
        "evidence": [
            "task #1901 events.jsonl epm:progress 2026-08-25T03:18:29Z (G1 FAIL note)",
            "tasks/<status>/1901/artifacts/mlpdense-smoke-r3-g1fail.log",
        ],
        "replacement_artifact": (
            "eval_results/issue_1901/paper_densify/mlp_scaling_dense_L19.json "
            "(all 8 fresh rungs within-store n1m lmsys draws; plan v15 §4/§10)"
        ),
        "consumers": [
            "docs/paper_context_answer_map/claims.md (C1 main row)",
            "scripts/issue1901_body_figures.py (fig_paper_c1_scaling; reads all three)",
            "docs/posters/mats_2026/make_plot1_scaling.py (reads scaling_ladder_L19.json)",
            "docs/posters/mats_2026/csls_rescore.py (validates vs scaling_ladder_L19.json)",
            "scripts/issue1901_boundary_token_control.py (banked ladder/bign inputs)",
            "docs/methodology/issue_1901.md (names the draw convention)",
        ],
        "consumer_resolution": (
            "grep -rl over the three banked filenames across the repo at implementation "
            "time (2026-08-25); producers issue1901_paper_densify{,_mlp}.py and tasks/** "
            "excluded"
        ),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_job_c(args, dev, out_path: Path) -> dict:
    smoke = args.smoke_chunks > 0
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "job c requires PYTHONHASHSEED=0 in the launcher env (N1M.select_train seeds "
            "default_rng(seed + abs(hash(name)) % 1e6); the recorded G2 sel-shas were produced "
            "under PYTHONHASHSEED=0 — plan §10 workload command)"
        )
    # Arg-level fail-fast BEFORE the 83 GB stage (the binding shape assert is
    # _validate_run_shape post-assemble, which re-checks these + the realized
    # data-dependent counts): registered seed set (both modes) + exact
    # production rung set (C2).
    _assert_registered_seed_set(args.seed, args.endpoint_seeds)
    if not smoke:
        wrong_rungs = set(int(n) for n in args.dense_ns) ^ set(PRODUCTION_DENSE_NS)
        if wrong_rungs:
            raise RuntimeError(
                f"--dense-ns must equal the registered production rung set "
                f"{sorted(PRODUCTION_DENSE_NS)} (plan §0); symmetric diff {sorted(wrong_rungs)}"
            )

    C.phase("c0-stage")
    capture_dir, pass_b_path, weights_dir, store_revision = _stage_job_c(args, smoke)
    preds_dir = args.stage_root / ("analysis_tensors_smoke" if smoke else "analysis_tensors")
    perfit_tag = "dense_smoke" if smoke else "dense"

    C.phase("c1-assemble")
    X, Y, pools, val, test, split = _assemble_n1m(args, capture_dir, pass_b_path, store_revision)
    n_lmsys = len(pools["lmsys"])
    pool_full = np.sort(np.asarray(pools["full"], dtype=np.int64))
    logger.info("[job-c] assembled n_rows=%d (lmsys pool %d)", X.shape[0], n_lmsys)
    # ru_maxrss at the c1 assemble peak (plan §12.7; Linux reports KB).
    logger.info(
        "[job-c] ru_maxrss after c1 assemble: %.1f GB",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
    n_capture_files = len(list(capture_dir.glob("shard*_chunk*.pt")))

    gates: dict = {}
    # Plan v15: G1 is RETIRED as a runtime gate — the 2026-08-25 on-pod
    # measurement refuted cross-store fold identity (0/400 exact matches), so
    # the scale7 store is no longer consumed and the gates record carries the
    # static provenance field below (plan §7 acceptance:
    # .gates.g1_cross_store | startswith("refuted-2026-08-25")).
    gates["g1_cross_store"] = G1_CROSS_STORE
    k_csls = BAT.K_CSLS
    gates["G3"] = {
        "n_pool": int(len(test)),
        "floor": k_csls + 2,
        "pass": bool(len(test) >= k_csls + 2),
    }
    if not gates["G3"]["pass"]:
        raise RuntimeError(f"G3 FAIL: n_pool={len(test)} < K_CSLS+2={k_csls + 2}")
    recorded_shas = _g2_recorded_shas()

    # Rung plan (plan v15 §4: every rung — smoke and production — is a
    # within-store n1m lmsys draw). Smoke replaces the rung set with
    # SMOKE_RUNG_SPECS and treats its 2k rung as the seed-dispersion endpoint
    # so the seedwise code path executes at tiny n (production 50k/500k).
    rung_specs = _dense_rung_specs(args.dense_ns, smoke)
    if smoke:
        endpoint_ns = {2_000}
    else:
        endpoint_ns = set(DENSE_ENDPOINT_NS) & {n for n, _ in rung_specs}
        max_rung = max(n for n, _ in rung_specs)
        if n_lmsys < max_rung:
            raise RuntimeError(
                f"lmsys pool {n_lmsys} < largest rung {max_rung} — cannot realize the ladder"
            )

    # C2: registered-shape validator — BEFORE any whitening/fit compute; each
    # mode asserts its OWN expectations (never skip-under-smoke).
    gates["shape"] = _validate_run_shape(
        smoke=smoke,
        smoke_chunks=args.smoke_chunks,
        n_capture_files=n_capture_files,
        seed=args.seed,
        endpoint_seeds=args.endpoint_seeds,
        rung_specs=rung_specs,
        split=split,
        n_pool_full=len(pool_full),
    )
    logger.info("[job-c] shape validator PASS: %s", json.dumps(gates["shape"]))

    C.phase("c2-whiten")
    whiten_npz = preds_dir / ("whiten_stats_L19_smoke.npz" if smoke else "whiten_stats_L19.npz")
    mu_a, ell, pool_sha = _job_c_whiten(args, X, Y, pool_full, whiten_npz, dev, store_revision)

    # M1: metric-side fingerprint identity — per-source eval-split shas + eval
    # store, the whitening-stat identity, bootstrap + retrieval conventions.
    fp_metrics_common = {
        "n_boot": int(args.n_boot),
        "bootstrap": "FFC._bootstrap_recon_ci (context resample; per-cell seeded)",
        "whiten": {
            "pool_sha256": pool_sha,
            "lam": WHITEN_LAMBDA,
            "helper": "null_battery.shrunk_cholesky_from_cov",
        },
        "k_csls": int(k_csls),
        "knn_ks": list(KNN_KS),
    }
    # Keyed by source so n1m fingerprints stay byte-compatible with round 3's
    # (the v13 "scale7" entry is removed with the scale7 sourcing).
    fp_eval_shas = {
        "n1m": {
            "val_sha256": split["val_sha256"],
            "test_sha256": split["test_sha256"],
            "eval_store": N1M_CAPTURE_PREFIX,
        },
    }

    def _fp_metrics(src: str) -> dict:
        return {**fp_metrics_common, **fp_eval_shas[src]}

    res: dict = {
        "per_n": {},
        "layer": 19,
        "seed": int(args.seed),
        "seed_b": int(args.seed_b),
        "endpoint_seeds": [int(s) for s in args.endpoint_seeds],
        "endpoint_ns": sorted(int(n) for n in endpoint_ns),
        "smoke_chunks": int(args.smoke_chunks),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "store": {
            "n1m_capture_prefix": N1M_CAPTURE_PREFIX,
            "weights_prefix": WEIGHTS_L19_PREFIX,
            "weights_files": list(WEIGHTS_L19_FILES),
            "store_revision": store_revision,
            "stage_root": str(args.stage_root),
        },
        "split": split,
        "whiten": {
            "lam": WHITEN_LAMBDA,
            "n_train": int(len(pool_full)),
            "pool_sha256": pool_sha,
            "stats_npz": str(whiten_npz.name),
        },
        "gates": gates,
        "sel_shas": {},
    }

    def _flush():
        res["gates"] = gates
        C.write_json_atomic(out_path, res)

    C.phase("c3-fits")
    n_units = len(rung_specs) * 3 + 3  # 3 arms per dense rung + 963k apply×2 + ib
    unit_k = 0
    realized_shas: dict = {}
    refit_ridge_r2s: dict = {}
    first_n1m_wall: float | None = None

    for n, source in rung_specs:
        rung_t0 = time.time()
        # Plan v15: every rung is a within-store n1m lmsys draw (the v13
        # scale7-prefix branch is removed with the store).
        if source != "n1m":
            raise RuntimeError(f"rung n={n}: source {source!r} != 'n1m' (plan v15 within-store)")
        sel_name = f"lmsys_{n // 1000}k"
        sub, sel_diag = N1M.select_train(pools, sel_name, n, "lmsys", args.seed_b)
        # C3: smoke-CALIBRATED expected count, never skip-under-smoke —
        # select_train returns min(n_target, |pool|) rows, so the smoke's
        # partial pool binds exactly at min(n, n_lmsys).
        expected_n = n if not smoke else min(n, n_lmsys)
        if len(sub) != expected_n:
            raise RuntimeError(
                f"{sel_name}: realized train {len(sub)} != expected {expected_n} "
                f"(target {n}, lmsys pool {n_lmsys}, smoke={smoke})"
            )
        Xs, Ys = X, Y
        val_idx, te_idx = val, test
        sel_sha = FFC._sha_ids(sub)
        res["sel_shas"][str(n)] = sel_sha
        if sel_name in recorded_shas:
            realized_shas[sel_name] = sel_sha
        row: dict = {
            "n_train": int(len(sub)),
            "n_target": int(n),
            "source": source,
            "sel_name": sel_name,
            "sel_sha256": sel_sha,
            "selection": sel_diag,
            "parity": {},
        }
        ev = np.concatenate([val_idx, te_idx])
        n_val = len(val_idx)

        # ── MLP arm (endpoints: extra seeds; plan §4 c3; M5 per-seed durability) ──
        unit_k += 1
        t0 = time.time()
        seeds = [args.seed] + (list(args.endpoint_seeds) if n in endpoint_ns else [])
        fp = _dense_fingerprint(
            args,
            n=n,
            source=source,
            sel_name=sel_name,
            sel_sha=sel_sha,
            store_revision=store_revision,
            arm="mlp",
            seeds=seeds,
            metrics=_fp_metrics(source),
        )
        perfit_name = f"{perfit_tag}_L19_n{n}_mlp.json"
        cell = _load_resumed_cell(args.out_dir / "perfit" / perfit_name, fp)
        if cell is None:
            per_seed = {}
            for si, s in enumerate(seeds):
                entry = None
                fp_s = None
                seed_perfit = f"{perfit_tag}_L19_n{n}_mlp_seed{s}.json"
                if len(seeds) > 1:
                    # M5: one atomic per-seed perfit + fingerprinted resume — a
                    # restart mid-endpoint loses at most ONE ~5-min seed fit
                    # (the plan-rejected FFC.batched_mlp_fit stays rejected:
                    # trainer parity with the banked anchors requires the
                    # minibatch N1M.fit_mlp, and a full-batch padded-bmm over
                    # 3 × 500k×3584 is HBM-infeasible on 1×H100 — see report).
                    fp_s = _dense_fingerprint(
                        args,
                        n=n,
                        source=source,
                        sel_name=sel_name,
                        sel_sha=sel_sha,
                        store_revision=store_revision,
                        arm="mlp",
                        seeds=[s],
                        metrics=_fp_metrics(source),
                    )
                    entry = _load_resumed_cell(args.out_dir / "perfit" / seed_perfit, fp_s)
                if entry is None:
                    t_s = time.time()
                    pred_ev, fit_meta = N1M.fit_mlp(
                        Xs,
                        Ys,
                        sub,
                        ev,
                        MLP_WIDTH,
                        MLP_LR,
                        FFC.MLP_MAX_EPOCHS,
                        N1M.MLP_BATCH,
                        s,
                        dev,
                    )
                    pred_val, pred_te = pred_ev[:n_val], pred_ev[n_val:]
                    entry = _cell_metrics(
                        pred_val, Ys[val_idx], pred_te, Ys[te_idx], args.n_boot, s + n
                    )
                    entry.update(_whitened_battery(pred_te, Ys[te_idx], mu_a, ell, k_csls))
                    entry["fit_meta"] = fit_meta
                    entry["seed"] = int(s)
                    stem = f"preds_L19_n{n}_mlp" + ("" if s == args.seed else f"_seed{s}")
                    _save_pred_npz(
                        preds_dir,
                        stem,
                        pred_te,
                        te_idx,
                        {"n": n, "arm": "mlp", "seed": s, "source": source},
                    )
                    if len(seeds) > 1:
                        entry["fingerprint"] = fp_s
                        _write_perfit(args.out_dir, seed_perfit, entry)
                    print(
                        f"[job-c] n={n} arm=mlp seed {si + 1}/{len(seeds)} (s={s}) "
                        f"elapsed={time.time() - t_s:.1f}s",
                        flush=True,
                    )
                per_seed[str(s)] = {k: v for k, v in entry.items() if k != "fingerprint"}
            cell = dict(per_seed[str(args.seed)])
            if len(seeds) > 1:
                r2s = np.array([per_seed[str(s)]["test_r2"] for s in seeds])
                acc1 = np.array([per_seed[str(s)]["whitened_csls"]["acc_at_k"][1] for s in seeds])
                cell["seeds"] = per_seed
                # M2: the top-level cell values are the primary seed's DISPLAY
                # copy; registered summaries aggregate the per-seed blocks.
                cell["top_level_is_display_seed"] = int(args.seed)
                cell["seed_dispersion"] = {
                    "seeds": [int(s) for s in seeds],
                    "test_r2": {"std": float(r2s.std()), "range": float(r2s.max() - r2s.min())},
                    "whitened_csls_acc1": {
                        "std": float(acc1.std()),
                        "range": float(acc1.max() - acc1.min()),
                    },
                }
            cell["fingerprint"] = fp
            _write_perfit(args.out_dir, perfit_name, cell)
        row["mlp"] = cell
        row["parity"]["mlp"] = _rung_parity(
            "mlp", n, cell["test_r2"], args.parity_tol, sha_match=False
        )
        print(
            f"[job-c] unit {unit_k}/{n_units} n={n} arm=mlp elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

        # ── Ridge arm (identical subset; plan §4 c3) ─────────────────────────
        unit_k += 1
        t0 = time.time()
        fp = _dense_fingerprint(
            args,
            n=n,
            source=source,
            sel_name=sel_name,
            sel_sha=sel_sha,
            store_revision=store_revision,
            arm="ridge",
            seeds=[args.seed],
            metrics=_fp_metrics(source),
        )
        perfit_name = f"{perfit_tag}_L19_n{n}_ridge.json"
        cell = _load_resumed_cell(args.out_dir / "perfit" / perfit_name, fp)
        if cell is None:
            pred_te_r, meta, _payload = N1M.fit_ridge_with_weights(
                Xs, Ys, sub, val_idx, te_idx, N1M.LAMBDAS_N1M, dev, args.ridge_block
            )
            cell = {
                "val_r2": float(meta["val_r2_at_selected"]),
                "test_r2": float(PR._pooled_r2(pred_te_r, Ys[te_idx])),
                "test_ci": FFC._bootstrap_recon_ci(
                    pred_te_r, Ys[te_idx], args.n_boot, args.seed + n
                ),
                "knn": {
                    m: MB.knn_retrieval(pred_te_r, Ys[te_idx], ks=KNN_KS, metric=m)
                    for m in ("euclidean", "cosine")
                },
                "fit_meta": meta,
            }
            cell.update(_whitened_battery(pred_te_r, Ys[te_idx], mu_a, ell, k_csls))
            _save_pred_npz(
                preds_dir,
                f"preds_L19_n{n}_ridge",
                pred_te_r,
                te_idx,
                {"n": n, "arm": "ridge", "seed": args.seed, "source": source},
            )
            cell["fingerprint"] = fp
            _write_perfit(args.out_dir, perfit_name, cell)
        if sel_name in recorded_shas:
            refit_ridge_r2s[sel_name] = cell["test_r2"]
        row["ridge"] = cell
        row["parity"]["ridge"] = _rung_parity(
            "ridge",
            n,
            cell["test_r2"],
            args.parity_tol,
            sha_match=bool(sel_name in recorded_shas and sel_sha == recorded_shas[sel_name]),
        )
        print(
            f"[job-c] unit {unit_k}/{n_units} n={n} arm=ridge elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

        # ── identity+bias arm (blocked; plan §9 LARGEST-CELL keying) ─────────
        unit_k += 1
        t0 = time.time()
        fp = _dense_fingerprint(
            args,
            n=n,
            source=source,
            sel_name=sel_name,
            sel_sha=sel_sha,
            store_revision=store_revision,
            arm="identity_bias",
            seeds=[args.seed],
            metrics=_fp_metrics(source),
        )
        perfit_name = f"{perfit_tag}_L19_n{n}_identity_bias.json"
        cell = _load_resumed_cell(args.out_dir / "perfit" / perfit_name, fp)
        if cell is None:
            pred_ev_ib, bias = MB.identity_bias_predict_blocked(
                Xs, Ys, sub, np.asarray(Xs[ev]), block=IB_BLOCK, return_bias=True
            )
            pred_val_ib, pred_te_ib = pred_ev_ib[:n_val], pred_ev_ib[n_val:]
            cell = _cell_metrics(
                pred_val_ib, Ys[val_idx], pred_te_ib, Ys[te_idx], args.n_boot, args.seed + n
            )
            cell.update(_whitened_battery(pred_te_ib, Ys[te_idx], mu_a, ell, k_csls))
            cell["bias_vector"] = [float(v) for v in bias]
            # M6 (plan §4/§10/§12.11): blocked-vs-exact equivalence on ≤2,000
            # REAL staged rows — a CONTENT predicate, not a mode branch: fires
            # at BOTH smoke rungs (1k/2k) and never at production rungs (≥5k),
            # where the exact helper's fp64 train copy would not fit anyway.
            if len(sub) <= 2_000:
                exact_ib = MB.identity_bias_predict(
                    np.asarray(Xs[sub]), np.asarray(Ys[sub]), np.asarray(Xs[ev])
                )
                ib_delta = float(np.max(np.abs(pred_ev_ib - exact_ib)))
                if ib_delta > 1e-6:
                    raise RuntimeError(
                        f"identity+bias blocked-vs-exact equivalence FAILED at n={len(sub)}: "
                        f"max|Δpred|={ib_delta:.3e} > 1e-6"
                    )
                cell["ib_equivalence"] = {
                    "max_abs_delta": ib_delta,
                    "n_rows": int(len(sub)),
                    "tol": 1e-6,
                }
                logger.info(
                    "[job-c] ib blocked-vs-exact equivalence PASS (n=%d, max|Δ|=%.3e)",
                    len(sub),
                    ib_delta,
                )
            cell["fingerprint"] = fp
            _write_perfit(args.out_dir, perfit_name, cell)
        row["identity_bias"] = cell
        print(
            f"[job-c] unit {unit_k}/{n_units} n={n} arm=identity_bias "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

        row["wall_time_s"] = round(time.time() - rung_t0, 1)
        res["per_n"][str(n)] = row
        _flush()

        if source == "n1m" and first_n1m_wall is None:
            first_n1m_wall = row["wall_time_s"]
            ratio = first_n1m_wall / PLAN_FIRST_N1M_RUNG_WALL_S
            print(
                f"[job-c][advisory] first n1m rung (n={n}) wall {first_n1m_wall:.0f}s vs plan "
                f"{PLAN_FIRST_N1M_RUNG_WALL_S:.0f}s (ratio {ratio:.2f}x"
                f"{'; >2x — compute-deviation class' if ratio > 2.0 else ''})",
                flush=True,
            )

    # ── G2 verdict (after the 150k/500k ridge cells; plan §4 c1 / §7 crit. 2) ──
    if smoke:
        gates["G2"] = {
            "verdict": "SMOKE-NOT-EVALUATED",
            "downgrade_recorded": False,
            "detail": {"note": "partial pool — recorded selections not realizable at smoke n"},
        }
    else:
        banked_r2s = {
            name: PD._banked_parity_target(*DENSE_PARITY_ANCHORS[("ridge", n_pt)][:3])
            for name, n_pt in (("lmsys_150k", 150_000), ("lmsys_500k", 500_000))
        }
        g2 = _g2_gate(realized_shas, recorded_shas, refit_ridge_r2s, banked_r2s, args.parity_tol)
        gates["G2"] = asdict(g2)
        # M3: durable per-cell G2 annotation on the two recorded-selection ridge
        # perfits — written BEFORE the FAIL raise so even a halt leaves the
        # verdict on the affected cells, not only in the aggregate.
        gates["G2"]["percell_annotated"] = _annotate_g2_percell(args.out_dir, perfit_tag, g2)
        _flush()
        if g2.verdict == "FAIL":
            raise RuntimeError(f"G2 FAIL (recorded-selection comparison): {json.dumps(asdict(g2))}")
        logger.info("[job-c] G2 %s %s", g2.verdict, json.dumps(g2.detail))
    _flush()

    # ── 963k point: banked-weights apply (both arms) + fresh blockwise ib ─────
    n963 = int(len(pool_full))
    rung_t0 = time.time()
    ev = np.concatenate([val, test])
    n_val = len(val)
    sel_sha_963 = FFC._sha_ids(pool_full)
    row = {
        "n_train": n963,
        "n_target": n963,
        "source": "n1m",
        "sel_name": "mixed_1m",
        "sel_sha256": sel_sha_963,
        "selection": {"mode": "full train pool (all non-val/test rows)"},
        "parity": {},
    }
    res["sel_shas"][str(n963)] = sel_sha_963
    x_ev = np.asarray(X[ev])
    for arm, fname, payload_kind in (
        ("mlp", "mlp_w8192.pt", "mlp"),
        ("ridge", "ridge.pt", "ridge"),
    ):
        unit_k += 1
        t0 = time.time()
        fp = _dense_fingerprint(
            args,
            n=n963,
            source="n1m",
            sel_name="mixed_1m",
            sel_sha=sel_sha_963,
            store_revision=store_revision,
            arm=f"{arm}_apply",
            seeds=[args.seed],
            metrics=_fp_metrics("n1m"),
        )
        perfit_name = f"{perfit_tag}_L19_n{n963}_{arm}.json"
        cell = _load_resumed_cell(args.out_dir / "perfit" / perfit_name, fp)
        if cell is None:
            # M7: weights_only=True — the _persist_weights payloads are tensors
            # + str/int primitives (safe-loader-allowed); never unpickle
            # arbitrary objects from a Hub-fetched blob.
            payload = torch.load(weights_dir / fname, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"banked weight payload {fname} is {type(payload).__name__}, expected dict"
                )
            assert payload.get("kind") == payload_kind, (fname, payload.get("kind"))
            pred_ev_a = N1M.apply_map(payload, x_ev, dev)
            del payload
            pred_val_a, pred_te_a = pred_ev_a[:n_val], pred_ev_a[n_val:]
            cell = _cell_metrics(pred_val_a, Y[val], pred_te_a, Y[test], args.n_boot, args.seed)
            cell.update(_whitened_battery(pred_te_a, Y[test], mu_a, ell, k_csls))
            cell["applied_banked_weights"] = f"{WEIGHTS_L19_PREFIX}/{fname}"
            _save_pred_npz(
                preds_dir,
                f"preds_L19_n{n963}_{arm}",
                pred_te_a,
                test,
                {"n": n963, "arm": f"{arm}_apply", "seed": args.seed, "source": "n1m"},
            )
            cell["fingerprint"] = fp
            _write_perfit(args.out_dir, perfit_name, cell)
        # Deterministic apply parity (≤1e-3; plan §7 crit. 3) — runs VERBATIM at
        # smoke too: the pinned test rows ride the fully-staged pass_b bundle.
        path, extract, pasted = MIXED1M_APPLY_ANCHORS[arm]
        want = PD._banked_parity_target(path, extract, pasted)
        row["parity"][arm] = PD._parity_check(
            f"mixed1m-{arm}-apply", cell["test_r2"], want, MIXED1M_APPLY_TOL, smoke=False
        )
        row[arm] = cell
        print(
            f"[job-c] unit {unit_k}/{n_units} n={n963} arm={arm}_apply "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    unit_k += 1
    t0 = time.time()
    fp = _dense_fingerprint(
        args,
        n=n963,
        source="n1m",
        sel_name="mixed_1m",
        sel_sha=sel_sha_963,
        store_revision=store_revision,
        arm="identity_bias",
        seeds=[args.seed],
        metrics=_fp_metrics("n1m"),
    )
    perfit_name = f"{perfit_tag}_L19_n{n963}_identity_bias.json"
    cell = _load_resumed_cell(args.out_dir / "perfit" / perfit_name, fp)
    if cell is None:
        pred_ev_ib, bias = MB.identity_bias_predict_blocked(
            X, Y, pool_full, x_ev, block=IB_BLOCK, return_bias=True
        )
        pred_val_ib, pred_te_ib = pred_ev_ib[:n_val], pred_ev_ib[n_val:]
        cell = _cell_metrics(pred_val_ib, Y[val], pred_te_ib, Y[test], args.n_boot, args.seed)
        cell.update(_whitened_battery(pred_te_ib, Y[test], mu_a, ell, k_csls))
        cell["bias_vector"] = [float(v) for v in bias]
        cell["fingerprint"] = fp
        _write_perfit(args.out_dir, perfit_name, cell)
    row["identity_bias"] = cell
    # ru_maxrss at the 963k identity+bias cell (plan §12.7; Linux reports KB).
    logger.info(
        "[job-c] ru_maxrss after 963k identity+bias cell: %.1f GB",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
    print(
        f"[job-c] unit {unit_k}/{n_units} n={n963} arm=identity_bias "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    row["wall_time_s"] = round(time.time() - rung_t0, 1)
    res["per_n"][str(n963)] = row

    # M8: the banked constant train-mean floor at 963k — read from the
    # committed metric-battery JSON with provenance, never refit (plan §5).
    res["controls"] = {
        "constant_train_mean_963k": {
            "test_r2": PD._banked_parity_target(
                BANKED_BATTERY_CONTEXT_JSON,
                lambda d: d["per_layer"]["19"]["arms"]["const_mean"]["r2"]["point"],
                -0.044533745181768225,
            ),
            "source": str(BANKED_BATTERY_CONTEXT_JSON.relative_to(FFC.PROJECT_ROOT)),
            "provenance": "banked #1901 metric-battery const_mean arm (L19, mixed_1m pool)",
            "note": "banked floor, read not refit (plan §5); completes the baseline ladder",
        }
    }

    # M2: the plan-§3 verdict lattice, computed from the persisted per-seed
    # blocks + ridge cells and registered in the aggregate. At smoke the
    # endpoints are not realized ⇒ a structured computed:false record.
    res["verdict"] = _endpoint_verdict(
        res["per_n"], sorted(endpoint_ns), [args.seed] + list(args.endpoint_seeds)
    )
    logger.info("[job-c] verdict: %s", json.dumps(res["verdict"]))
    _flush()

    # Prediction-artifact completeness (plan §7: 2 per rung incl. 963k + the
    # endpoint seedwise extras; 22 at the default production args).
    n_expected = 2 * (len(rung_specs) + 1) + len(endpoint_ns) * len(args.endpoint_seeds)
    realized_npz = sorted(p.name for p in preds_dir.glob("preds_L19_*.npz"))
    if len(realized_npz) != n_expected:
        raise RuntimeError(
            f"prediction npz count {len(realized_npz)} != expected {n_expected}: {realized_npz}"
        )
    res["prediction_artifacts"] = {"n": len(realized_npz), "files": realized_npz}

    # M4 / plan §7 acceptance (3): NO c5 upload while ANY recorded parity row is
    # beyond tolerance. The fold-exact / sha-matched-conditional rows already
    # halt at their rung; this unconditional backstop additionally halts on
    # recorded "statistical"-kind breaches (and anything future that records a
    # pass=False row). Vacuous at smoke by construction — no anchor row is
    # reachable at smoke n (validator-enforced).
    bad_parity = [
        p
        for rung_row in res["per_n"].values()
        for p in rung_row.get("parity", {}).values()
        if isinstance(p, dict) and not p.get("pass", True)
    ]
    if bad_parity:
        _flush()
        raise RuntimeError(
            f"parity acceptance FAILED (plan §7 acceptance 3): {len(bad_parity)} recorded "
            f"row(s) beyond tolerance — halting BEFORE the c5 upload: {json.dumps(bad_parity)}"
        )

    res["meta"] = _meta_common(args)
    res["meta"]["provenance"] = as_metadata_dict(git_provenance(), phase="job-c")
    res["meta"]["whiten_convention"] = (
        "z = solve_triangular(L, (v - mu_A).T, lower=True).T with L = shrunk Cholesky "
        f"((1-lam)*cov + lam*diag, lam={WHITEN_LAMBDA}) of the fp64 train-pool answer cov "
        "(issue2202_failchar convention); CSLS k=10 cross-domain "
        "(issue1901_metric_battery.csls_scores)"
    )
    _flush()

    C.phase("c5-upload")
    # Plan v15 §10: the machine-readable supersession record for the refuted
    # v13 cross-store join, written at c5 alongside the aggregate (smoke writes
    # a smoke-suffixed name — never the production record).
    supersede_path = args.out_dir / (
        "superseded_cross_store_join_smoke.json" if smoke else "superseded_cross_store_join.json"
    )
    supersede_rec = _superseded_join_record()
    supersede_rec["provenance"] = as_metadata_dict(git_provenance(), phase="c5-supersession-record")
    C.write_json_atomic(supersede_path, supersede_rec)
    logger.info("[job-c] supersession record written: %s", supersede_path)
    res["superseded_cross_store_join"] = supersede_path.name

    if args.skip_hf_upload:
        logger.info("[job-c] --skip-hf-upload: leaving %s unstaged to HF", preds_dir)
    else:
        hf_prefix = "issue1901_mlpdense/" + (
            "smoke/analysis_tensors" if smoke else "analysis_tensors"
        )
        expected = sorted(
            [f"{hf_prefix}/{name}" for name in realized_npz] + [f"{hf_prefix}/{whiten_npz.name}"]
        )
        # C4 (c5 re-entry): a VERIFIED completion sentinel keyed on the exact
        # expected upload path set. A matching sentinel still re-VERIFIES the
        # scoped remote listing before skipping (trust-but-verify); any
        # mismatch/miss falls through to a full re-upload; --rerun-upload
        # forces the upload regardless.
        sentinel = args.stage_root / (
            ".c5_upload_complete_smoke.json" if smoke else ".c5_upload_complete.json"
        )
        resumed = False
        if sentinel.exists() and not args.rerun_upload:
            try:
                rec = json.loads(sentinel.read_text())
            except json.JSONDecodeError:
                rec = None
            if (
                isinstance(rec, dict)
                and rec.get("prefix") == hf_prefix
                and (rec.get("expected") == expected)
            ):
                missing = hub.verify_repo_paths_uploaded(
                    HfApi(), C.HF_DATA_REPO, expected, path_in_repo=hf_prefix
                )
                if not missing:
                    logger.info(
                        "[job-c] c5 upload already complete + remote-verified (%s) — "
                        "skipping upload (--rerun-upload forces)",
                        sentinel.name,
                    )
                    resumed = True
                else:
                    logger.info(
                        "[job-c] upload sentinel matches but %d expected paths missing "
                        "remotely — re-uploading",
                        len(missing),
                    )
            else:
                logger.info("[job-c] upload sentinel stale/mismatched — re-uploading")
        if not resumed:
            base_url = hub._upload(preds_dir, C.HF_DATA_REPO, "dataset", hf_prefix)
            if not base_url:
                raise RuntimeError(
                    f"HF upload returned no path for {preds_dir} -> {hf_prefix} — "
                    "silent durability loss (upload-policy.md tracked gap); fix credentials/paths"
                )
            missing = hub.verify_repo_paths_uploaded(
                HfApi(), C.HF_DATA_REPO, expected, path_in_repo=hf_prefix
            )
            if missing:
                raise RuntimeError(
                    f"HF upload verify FAILED — missing {len(missing)}: {missing[:5]}"
                )
            C.write_json_atomic(
                sentinel,
                {
                    "prefix": hf_prefix,
                    "expected": expected,
                    "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        res["hf_upload"] = {
            "prefix": hf_prefix,
            "n_files": len(expected),
            "verified": True,
            "resumed_from_sentinel": resumed,
        }
        _flush()

    if not args.keep_stage:
        PD._reap_stage(args.stage_root / N1G.HF_PREFIX)
    return res


def _ensure_pass_b(args) -> None:
    if args.pass_b_path.exists():
        return
    from huggingface_hub import hf_hub_download

    logger.info("[stage] downloading pass_b bundle from HF (%s)", PASS_B_HF_FILE)
    args.pass_b_path.parent.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=C.HF_DATA_REPO,
            filename=PASS_B_HF_FILE,
            repo_type="dataset",
            local_dir=args.cache_dir / "pass_b_dl",
        ),
        what="pass_b bundle download",
    )
    Path(got).replace(args.pass_b_path)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="a,b", help="comma list from {a,b,c}")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=FFC.BOOT_N)
    ap.add_argument("--battery-chunk", type=int, default=14)
    ap.add_argument("--parity-tol", type=float, default=0.02)
    ap.add_argument("--pass-b-path", type=Path, default=FFC.PASS_B_PATH)
    ap.add_argument("--ladder-hf-prefix", default="issue1491_scale_ladder/scale7_refit")
    ap.add_argument("--ladder-layer", type=int, default=19)
    ap.add_argument("--scaling-ns", type=int, nargs="+", default=[5000, 10000])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FFC.PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=FFC.PROJECT_ROOT / "data" / "issue_1901" / "hf_dl"
    )
    # ── job c (mlp-scaling-densify) ────────────────────────────────────────────
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help="job c: HF staging root (pod container/workspace disk); REQUIRED for --jobs c",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="job c: scratch for manifest/stream caches (default <stage-root>/work)",
    )
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument(
        "--seed-b", type=int, default=0, help="n1m selection seed (banked bigN refits ran 0)"
    )
    ap.add_argument(
        "--dense-ns",
        type=int,
        nargs="+",
        default=[5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000],
        help="job c dense rungs; ALL fit on the n1m capture (lmsys draws; plan v15)",
    )
    ap.add_argument(
        "--endpoint-seeds",
        type=int,
        nargs="*",
        default=[43, 44],
        help="extra MLP seeds at the 50k/500k endpoints (seed-dispersion read)",
    )
    ap.add_argument(
        "--smoke-chunks",
        type=int,
        default=0,
        help="job c: >0 = tiny-real smoke (stage N capture chunks; rungs {1k n1m, 2k n1m})",
    )
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument(
        "--keep-stage", action="store_true", help="do not delete the staged capture (smoke)"
    )
    ap.add_argument(
        "--skip-hf-upload",
        action="store_true",
        help="job c: skip the c5 prediction-npz HF upload (local dev only)",
    )
    ap.add_argument(
        "--rerun-upload",
        action="store_true",
        help="job c: force the c5 upload even when the verified completion sentinel matches",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        raise SystemExit(0)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
    )
    dev = FFC._dev(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    t0 = time.time()
    if "a" in jobs:
        _ensure_pass_b(args)
        out_a = args.out_dir / "mlp_layer_curve_n3600.json"
        res_a = run_job_a(args, dev, out_a)
        res_a["meta"] = _meta_common(args)
        C.write_json_atomic(out_a, res_a)
    if "b" in jobs:
        out_b = args.out_dir / "mlp_scaling_L19.json"
        res_b = run_job_b(args, dev, out_b)
        res_b["meta"] = _meta_common(args)
        C.write_json_atomic(out_b, res_b)
    if "c" in jobs:
        if args.stage_root is None:
            raise SystemExit("--stage-root is required for --jobs c")
        if args.work_dir is None:
            args.work_dir = args.stage_root / "work"
        out_c = args.out_dir / (
            "mlp_scaling_dense_L19_smoke.json"
            if args.smoke_chunks > 0
            else "mlp_scaling_dense_L19.json"
        )
        run_job_c(args, dev, out_c)
        C.phase("done")
    logger.info("all jobs done in %.1fs", time.time() - t0)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
