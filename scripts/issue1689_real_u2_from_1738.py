#!/usr/bin/env python3
"""#1689 free-analysis — real-user-second-turn predictability from #1738's captures.

Fits the ridge map **prefix-end -> context-end at layer 19** on #1738's 100k real
multi-turn LMSYS/WildChat corpus. In #1738 the prefix-end state is the residual
state at the end of the conversation history BEFORE the final (real) user turn,
and the context-end state is the last prompt token AFTER that real user turn
(end of the generation header). Predicting context-end from prefix-end is
therefore a REAL-user-turn predictability read — the missing real-data comparator
for #1689's user-turn provenance result, whose "real" arm was a constant-string
fallback.

Everything reusable is reused, nothing re-implemented (artifact-reuse rule):

* capture assembly  -> ``issue1738_multiturn_fits.assemble_streams`` (streams the
  parent's ``capture/*.pt`` chunks into per-(array, layer) fp32 memmaps, cursor-
  checkpointed, one-chunk peak footprint);
* pinned split      -> ``load_split`` + ``split_positions`` (sha-asserted) and the
  parent fits JSON's recorded ``split_shas`` cross-assert;
* fitter            -> ``fit_predictor("ridge", ...)`` = the parent's val-lambda
  primal streaming ridge over ``LAMBDAS`` = logspace(-3, 8, 23);
* metrics           -> ``F._recon_point`` (pooled R^2 with SS_tot on the eval
  set's OWN mean + mean per-context cosine) and the batched holdout bootstrap;
* standing baselines-> ``mapping_baselines.identity_bias_predict`` /
  ``knn_retrieval`` (same call shape as the parent's ``_compute_baselines``);
* depth bands       -> ``GG._depth_band`` over manifest depths, band counts
  cross-asserted against the committed ``depth_contrasts.json``.

Wiring validation (``--phase validate``): the parent's own persisted ridge
weights for the prefix -> mean-answer cell (``analysis_tensors/weights/L19/
prefix_ridge.pt``) are applied to THIS assembly's prefix-end holdout rows and
must reproduce the committed ``prefix_L19_ridge`` holdout R^2 (0.37929...) and
the retained fp16 predictions -- one GEMM that pins the assembly, the row order,
and the holdout row set to the published artifacts.

Scope statements carried into the summary JSON:

* prefix-based arm ONLY. A context-based arm is self-prediction by construction
  for a user-turn target (input == target), so the both-arms mapping rule is
  satisfied by a stated deviation, not a second fit.
* target convention: #1738's context-end sits at the LAST PROMPT TOKEN including
  the generation header, a few tokens past the real final user turn's last
  content token -- near #1689's turn-end target, not its mean-over-turn target.

Refusal-safety: LMSYS/WildChat are unscreened real-user corpora. This script
never prints or logs conversation, prompt, or rollout text -- only counts,
indices, depths, hashes, and metrics. Do not add such logging.

CPU-only (no GPU, no generation, no judge calls). Fail loud -- NaN never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847 shared-VM thread caps bind only if load_dotenv() runs BEFORE torch is
# imported (pinned by tests/test_shared_vm_thread_caps.py).
load_dotenv()

import issue1738_multiturn_fits as FT  # noqa: E402  (assembly + split + fit reuse)
import numpy as np  # noqa: E402
import torch  # noqa: E402

C = FT.C  # issue779_common (HF_DATA_REPO, EXPECTED_HIDDEN)
HUB = FT.hub  # orchestrate.hub (retry_transient, stage_hub_file)
F = FT.F  # issue779_fitter_fair_comparison (_recon_point)
GG = FT.GG  # issue1738_multiturn_generate_capture (_depth_band, N1M manifest pool)
PF = FT.PF  # issue779_ffc_n1m_fits (LAMBDAS_N1M, fit_ridge, RIDGE_BLOCK)
PR = FT.PR  # issue779_percontext_recon (_pooled_r2)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
log = logging.getLogger("issue1689_real_u2")

HF_PREFIX = GG.HF_PREFIX  # "issue1738_multiturn"
LAYER = 19
NULL_DRAWS = 40
NULL_SEED = 1689
KS = (1, 5, 10)
KNN_METRICS = ("euclidean", "cosine")
PARENT_PREFIX_CELL_R2 = 0.37929079545844135  # committed prefix_L19_ridge holdout R^2

DEFAULT_WORK = PROJECT_ROOT / "data" / "issue_1738" / "hf_dl" / "issue1689_real_u2"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "issue_1738" / "mt100k" / "fits" / "sampling_manifest"
DEFAULT_PARENT_FITS = (
    PROJECT_ROOT / "eval_results" / "issue_1738" / "fits" / "multiturn_100k_fits.json"
)
DEFAULT_DEPTH_CONTRASTS = PROJECT_ROOT / "eval_results" / "issue_1738" / "depth_contrasts.json"
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1689" / "real_u2_1738_reuse" / "summary.json"


# ── staging ───────────────────────────────────────────────────────────────────────


def stage(args) -> dict:
    """Download the pinned split doc, the manifest (depths), and the parent's
    persisted prefix-ridge weights + retained predictions/targets (validation)."""
    from huggingface_hub import HfApi

    rev = HUB.retry_transient(
        lambda: HfApi().dataset_info(C.HF_DATA_REPO).sha, what="dataset_info(revision)"
    )
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    split_local = args.manifest_dir / "split_1738.json"
    HUB.stage_hub_file(  # retried + atomic; idempotent on an existing target
        C.HF_DATA_REPO,
        f"{HF_PREFIX}/{GG.N1M.MANIFEST_SUBDIR}/split_1738.json",
        split_local,
        repo_type="dataset",
        revision=rev,
    )
    log.info("[stage] split doc: %s", split_local)

    GG.N1M._download_manifest(HF_PREFIX, args.manifest_dir)  # flock-serialized, resumable

    val_dir = args.work / "validate"
    val_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    for rel in (
        f"analysis_tensors/weights/L{LAYER}/prefix_ridge.pt",
        f"analysis_tensors/pred16/prefix_L{LAYER}_ridge.npz",
        f"analysis_tensors/y_holdout/L{LAYER}.npz",
    ):
        dest = val_dir / rel.replace("/", "__")
        HUB.stage_hub_file(
            C.HF_DATA_REPO,
            f"{HF_PREFIX}/{rel}",
            dest,
            repo_type="dataset",
            revision=rev,
        )
        staged[rel] = str(dest)
        log.info("[stage] %s (%.1f MB)", rel, dest.stat().st_size / 1e6)
    return {"revision": rev, "split_file": str(split_local), "validate_files": staged}


# ── assembly (parent helper, layer 19 only) ───────────────────────────────────────


def assemble(args):
    ns = SimpleNamespace(mm_dir=str(args.work / "mm"), hf_prefix=HF_PREFIX, local_capture_dir="")
    t0 = time.time()
    mm, ci, ameta = FT.assemble_streams(ns, [LAYER])
    log.info(
        "[assemble] %d rows / %d chunks in %.1f min",
        ameta["n_rows"],
        ameta["n_chunks"],
        (time.time() - t0) / 60,
    )
    return mm, ci, ameta


# ── validation: parent weights applied to THIS assembly ───────────────────────────


def validate_parent_cell(mm, ci, ho, staged: dict) -> dict:
    """Apply the parent's persisted prefix->mean-answer ridge W to this assembly's
    prefix-end holdout rows; must reproduce the committed holdout R^2 + pred16."""
    wpath = staged[f"analysis_tensors/weights/L{LAYER}/prefix_ridge.pt"]
    payload = torch.load(wpath, map_location="cpu", weights_only=False)
    W = payload["W"].to(torch.float64).numpy()
    xmu = payload["xmu"].to(torch.float64).numpy()
    xsd = payload["xsd"].to(torch.float64).numpy()
    ymu = payload["ymu"].to(torch.float64).numpy()
    x_ho = np.asarray(mm[("px", LAYER)][ho], dtype=np.float64)
    pred = ((x_ho - xmu) / xsd) @ W + ymu

    with np.load(staged[f"analysis_tensors/y_holdout/L{LAYER}.npz"]) as z:
        y16, y_ci = z["y16"].astype(np.float64), z["ci"]
    with np.load(staged[f"analysis_tensors/pred16/prefix_L{LAYER}_ridge.npz"]) as z:
        p16, p_ci = z["pred16"].astype(np.float64), z["ci"]
    assert np.array_equal(y_ci, ci[ho]), "y_holdout ci order != this assembly's holdout ci order"
    assert np.array_equal(p_ci, ci[ho]), "pred16 ci order != this assembly's holdout ci order"

    r2, cos = F._recon_point(pred, y16)
    rel = float(np.abs(pred - p16).max() / (np.abs(p16).max() + 1e-12))
    out = {
        "check": "parent prefix->mean-answer ridge cell recomputed from this assembly",
        "selected_lambda": float(payload["selected_lambda"]),
        "recomputed_holdout_r2": float(r2),
        "committed_holdout_r2": PARENT_PREFIX_CELL_R2,
        "abs_r2_gap": abs(float(r2) - PARENT_PREFIX_CELL_R2),
        "recomputed_holdout_mean_cosine": float(cos),
        "max_abs_rel_pred_gap_vs_retained_fp16": rel,
        "n_holdout": int(len(ho)),
    }
    # fp16 storage of pred16 (~1e-3 relative) bounds both tolerances.
    assert out["abs_r2_gap"] < 2e-3, f"parent-cell R2 mismatch: {out}"
    assert rel < 5e-3, f"parent-cell prediction mismatch vs retained pred16: {out}"
    out["verdict"] = "PASS"
    log.info(
        "[validate] parent prefix cell reproduced: R2=%.5f (committed %.5f)",
        r2,
        PARENT_PREFIX_CELL_R2,
    )
    return out


# ── shuffled-target null ──────────────────────────────────────────────────────────


def shuffled_target_null(pred: np.ndarray, true: np.ndarray, n_draws: int, seed: int) -> dict:
    """Pooled R^2 under permuted holdout TARGET rows, ``n_draws`` draws.

    Each draw is one whole-array reduction (no per-row loop): with SS_tot
    permutation-invariant, R^2 = 1 - (sum||t||^2 + sum||p||^2 - 2<t[perm], p>) /
    SS_tot. Draw 0 is cross-checked against ``PR._pooled_r2`` so the closed form
    is pinned to the shared metric helper."""
    rng = np.random.default_rng(seed)
    n = true.shape[0]
    mu = true.mean(axis=0)
    ss_tot = float(((true - mu) ** 2).sum())
    t_sq = (true**2).sum(axis=1)
    p_sq_sum = float((pred**2).sum())
    r2s = []
    for d in range(n_draws):
        perm = rng.permutation(n)
        cross = float(np.einsum("ij,ij->", true[perm], pred))
        ss_res = float(t_sq[perm].sum()) + p_sq_sum - 2.0 * cross
        r2 = 1.0 - ss_res / ss_tot
        if d == 0:
            ref = PR._pooled_r2(pred, true[perm])
            assert abs(r2 - ref) < 1e-9, f"null closed form != _pooled_r2: {r2} vs {ref}"
        r2s.append(float(r2))
    arr = np.asarray(r2s)
    return {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "permuted": "holdout target rows",
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "draws": [round(x, 6) for x in r2s],
    }


# ── depth strata ──────────────────────────────────────────────────────────────────


def depth_strata(pred_ho, y_ho, ci_ho, manifest_dir: Path, depth_contrasts: Path) -> dict:
    """Holdout R^2 per #1738 depth band (2 / 3-4 / >=5 user turns), band counts
    cross-asserted against the committed depth_contrasts.json."""
    pool, meta = GG.N1M.read_manifest_pool(manifest_dir)
    depth_of = {int(r["i"]): int(r["depth"]) for r in pool}
    del pool  # never hold (or log) manifest text
    depths = np.asarray([depth_of[int(c)] for c in ci_ho], dtype=np.int64)
    committed = json.loads(depth_contrasts.read_text())["arms"][f"prefix_L{LAYER}_ridge"]
    out: dict = {"convention": "pooled R^2 with SS_tot on the stratum's own mean", "bands": {}}
    for band, cdoc in committed.items():
        sel = np.asarray([GG._depth_band(int(d)) == band for d in depths], dtype=bool)
        assert int(sel.sum()) == int(cdoc["n"]), (
            f"depth band {band}: n={int(sel.sum())} != committed {cdoc['n']}"
        )
        r2, cos = F._recon_point(pred_ho[sel], y_ho[sel])
        out["bands"][band] = {
            "n": int(sel.sum()),
            "holdout_r2": float(r2),
            "mean_cosine": float(cos),
        }
    out["band_count_crosscheck_vs_depth_contrasts"] = "PASS"
    out["exact_depth_histogram"] = {
        str(int(d)): int((depths == d).sum()) for d in np.unique(depths)
    }
    out["manifest_n_new"] = int(meta["n_new"])
    return out


# ── main battery ──────────────────────────────────────────────────────────────────


def run(args) -> dict:
    stage_meta = stage(args)
    mm, ci, ameta = assemble(args)

    split = FT.load_split(Path(stage_meta["split_file"]))
    sets = FT.split_positions(split, ci)
    tr, val, te, ho = sets["train"], sets["val"], sets["test"], sets["holdout"]
    d = C.EXPECTED_HIDDEN
    assert len(tr) > d, f"n_train={len(tr)} <= d={d}: estimator-degenerate regime"
    FT._assert_parent_split_shas(split, str(args.parent_fits_json))

    validate = validate_parent_cell(mm, ci, ho, stage_meta["validate_files"])

    X, Y = mm[("px", LAYER)], mm[("cx", LAYER)]
    te_all = np.concatenate([te, ho])
    ns = SimpleNamespace(
        ridge_block=PF.RIDGE_BLOCK,
        mlp_max_epochs=FT.F.MLP_MAX_EPOCHS,
        mlp_batch=PF.MLP_BATCH,
        mlp_lrs_list=(1e-3,),
        seed=42,
        krr_nystrom_centers=FT.KRR_CENTERS_DEFAULT,
        krr_solver="cholesky",
        krr_gamma_mult=(1.0,),
        krr_lambdas_list=(0.1, 10.0),
    )
    t0 = time.time()
    pred_all, fit_meta = FT.fit_predictor(
        "ridge", X, Y, tr, val, te_all, ns, args.device, resid_lr=1e-3
    )
    fit_wall = time.time() - t0
    pred_te, pred_ho = pred_all[: len(te)], pred_all[len(te) :]
    y_te = np.asarray(Y[te], dtype=np.float64)
    y_ho = np.asarray(Y[ho], dtype=np.float64)
    r2_te, cos_te = F._recon_point(pred_te, y_te)
    r2_ho, cos_ho = F._recon_point(pred_ho, y_ho)
    boot = FT._boot_recon_ci_batched(pred_ho, y_ho, args.n_boot, FT.BOOT_SEED)
    log.info("[fit] ridge px->cx L%d: holdout R2=%.4f (%.1f min)", LAYER, r2_ho, fit_wall / 60)

    # standing baselines: identity+learned-bias (same d in/out) + kNN retrieval
    pred_ib = FT.identity_bias_predict(np.asarray(X[tr]), np.asarray(Y[tr]), np.asarray(X[ho]))
    r2_ib, cos_ib = F._recon_point(pred_ib, y_ho)
    log.info("[baseline] identity+bias holdout R2=%.4f", r2_ib)
    knn = {
        name: {m: FT.knn_retrieval(pv, y_ho, ks=KS, metric=m) for m in KNN_METRICS}
        for name, pv in (("ridge", pred_ho), ("identity_bias", pred_ib))
    }

    null = shuffled_target_null(pred_ho, y_ho, NULL_DRAWS, NULL_SEED)
    log.info("[null] shuffled-target mean R2=%.4f p97.5=%.4f", null["mean"], null["p97_5"])

    strata = depth_strata(pred_ho, y_ho, ci[ho], args.manifest_dir, args.depth_contrasts)

    doc = {
        "issue": 1689,
        "round": "real_u2_1738_reuse",
        "what": (
            "ridge map prefix-end -> context-end at layer 19 on #1738's 100k real "
            "multi-turn LMSYS/WildChat corpus: how predictable is the state after a "
            "REAL user second+ turn from the conversation history that precedes it"
        ),
        "arm_statement": (
            "prefix-based arm only — a context-based arm is a self-prediction by "
            "construction for a user-turn target (its input IS the target), so the "
            "prefix-mapping / context-mapping both-arms rule is met by this stated "
            "deviation rather than a second fit"
        ),
        "target_convention_caveat": (
            "#1738's context-end sits at the LAST PROMPT TOKEN including the "
            "generation header, a few tokens past the real final user turn's last "
            "content token — near #1689's turn-END target, not its mean-over-turn "
            "target; the two are not interchangeable"
        ),
        "inputs": {
            "hf_repo": C.HF_DATA_REPO,
            "hf_revision": stage_meta["revision"],
            "capture_prefix": f"{HF_PREFIX}/{GG.CAPTURE_SUBDIR}",
            "x_array": "px_last (prefix-end residual state, layer 19)",
            "y_array": "cx_last (context-end residual state, layer 19)",
            "split_doc": f"{HF_PREFIX}/{GG.N1M.MANIFEST_SUBDIR}/split_1738.json",
            "manifest_prefix": f"{HF_PREFIX}/{GG.N1M.MANIFEST_SUBDIR}",
            "validation_files": {k: f"{HF_PREFIX}/{k}" for k in stage_meta["validate_files"]},
            "parent_fits_json": str(args.parent_fits_json),
            "assembly_fingerprint": ameta["fingerprint"],
            "n_chunks": ameta["n_chunks"],
            "n_rows_captured": ameta["n_rows"],
            "d": int(d),
        },
        "split_provenance": {
            "source": "#1738 pinned split_1738.json (sha-asserted on load)",
            "realized_counts": {k: int(len(v)) for k, v in sets.items()},
            "split_shas": {k: split["sets"][k]["sha256"] for k in split["sets"]},
            "parent_split_shas_crossassert": "PASS",
        },
        "assembly_validation": validate,
        "fit": {
            "fitter": "ridge (parent val-lambda primal streaming, fit_predictor cell path)",
            "lambda_grid": [float(x) for x in FT.LAMBDAS],
            "fit_meta": fit_meta,
            "n_train": int(len(tr)),
            "test_r2": float(r2_te),
            "test_mean_cosine": float(cos_te),
            "holdout_r2": float(r2_ho),
            "holdout_mean_cosine": float(cos_ho),
            "holdout_bootstrap_ci": boot,
            "n_test": int(len(te)),
            "n_holdout": int(len(ho)),
            "wall_s": fit_wall,
        },
        "identity_bias_baseline": {
            "applicable": True,
            "reason": "input and output share d = 3584 (both layer-19 residual states)",
            "holdout_r2": float(r2_ib),
            "holdout_mean_cosine": float(cos_ib),
        },
        "knn_retrieval": {"ks": list(KS), "metrics": list(KNN_METRICS), "cells": knn},
        "shuffled_target_null": null,
        "depth_strata": strata,
        "compute": {
            "device": args.device,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_commit": os.environ.get("EPM_GIT_COMMIT", ""),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    GG.N1M._atomic_write_json(args.out, doc)
    log.info("[out] %s", args.out)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--parent-fits-json", type=Path, default=DEFAULT_PARENT_FITS)
    ap.add_argument("--depth-contrasts", type=Path, default=DEFAULT_DEPTH_CONTRASTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument(
        "--phase",
        choices=["all", "stage", "assemble"],
        default="all",
        help="'stage'/'assemble' run that step only (both are resumable); 'all' runs the battery",
    )
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.phase == "stage":
        print(json.dumps(stage(args), indent=1))
        return
    if args.phase == "assemble":
        _mm, _ci, ameta = assemble(args)
        print(json.dumps(ameta, indent=1))
        return
    run(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
