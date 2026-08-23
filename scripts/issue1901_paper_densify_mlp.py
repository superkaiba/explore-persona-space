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

Per cell, both jobs report: pooled test R^2 + 95% bootstrap CI
(n_boot=1000), val R^2 on the pinned val_400, kNN retrieval
(euclidean + cosine, ks=1/5/10/50, pinned test pool), and the
closed-form identity+bias baseline (standing rule). Per-fit JSONs are
written incrementally; aggregates rewritten atomically after every
battery chunk.

Early-stop protocol note (rides every meta block): both banked trainers
early-stop on an INTERNAL 10% split of the train rows; the pinned
val_400 is a reporting/selection split only. Reproduced verbatim here —
parity with the banked anchors requires it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Heavy imports AFTER load_dotenv() so the shared-VM thread caps (#847) bind
# in-process (torch freezes its intra-op pool from OMP_NUM_THREADS at import).
import numpy as np  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_fitter_fair_comparison as FFC  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402

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


def _ensure_pass_b(args) -> None:
    if args.pass_b_path.exists():
        return
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

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
    ap.add_argument("--jobs", default="a,b", help="comma list from {a,b}")
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
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
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
    logger.info("all jobs done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
