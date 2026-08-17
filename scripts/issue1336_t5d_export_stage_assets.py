"""Per-surface stage-asset export for the #1336 t5d GPU round (persistence-only).

Two user-requested harvest extensions, piggybacking on staging the t5d round
already pays for — run once per staged surface, BEFORE the staging reap:

1. **Fitted maps** -> HF ``issue1336_rlvr_ladder/analysis_tensors/fitted_maps/``
   Per (stage, corpus) at the headline layer: the stage's OWN context->answer
   v2-ridge map per fold — the exact affine form recovered by
   ``issue1336_metric_ladder._ws_effective_matrix`` (``pred(x) = (x - xref) @ M
   + b``; M fp16 (d, d), b + xref fp32) — plus the selected-lambda / selector
   diagnostics per fit (the #1887 convention) and a meta JSON. Enables the
   W_source-vs-W_target operator battery (spectra, principal angles, aligned
   operator cosines) as a free off-pod read; the maps were never persisted
   before and the pod round refits everything anyway.
2. **Layer-30 clouds** -> HF ``.../analysis_tensors/layer30_clouds/``
   The PAIRED cloud vectors per (stage, corpus): context AND answer vectors
   (fp16) with conv_ids + fold assignments + a meta JSON — the inputs for
   running the full rigid-to-affine ladder on any cloud arrow inline later.

CONVENTION NOTE (recorded in every meta): fold assignment here is
``fc._cv_folds`` over the STAGE'S OWN rows — the canonical per-stage split —
NOT any pair's intersection-row split (a pair battery aligns rows across two
stages first, so its folds differ per pair by construction).

Resume: a unit whose expected HF paths all exist is SKIPPED (one scoped
listing per unit via ``hub.verify_repo_paths_uploaded``). Uploads are bulk
``upload_folder`` per (unit, kind) — never a per-file loop — and local unit
dirs are deleted after a verified upload (pod disk headroom).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

MAPS_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/fitted_maps"
CLOUDS_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/layer30_clouds"


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unavailable"
        )
    except OSError:
        return "unavailable"


def _export_unit(args, model: str, api) -> None:
    """One (stage, fmt, corpus): clouds npz + per-fold fitted maps npz + metas."""
    unit = f"{model}_{args.format}_{args.corpus}"
    maps_paths = [f"{MAPS_PREFIX}/maps_{unit}.npz", f"{MAPS_PREFIX}/maps_{unit}.meta.json"]
    cloud_paths = [
        f"{CLOUDS_PREFIX}/clouds_{unit}.npz",
        f"{CLOUDS_PREFIX}/clouds_{unit}.meta.json",
    ]
    missing_maps = hub.verify_repo_paths_uploaded(
        api, cm.HF_DATA_REPO, maps_paths, path_in_repo=MAPS_PREFIX, repo_type="dataset"
    )
    missing_clouds = hub.verify_repo_paths_uploaded(
        api, cm.HF_DATA_REPO, cloud_paths, path_in_repo=CLOUDS_PREFIX, repo_type="dataset"
    )
    if not missing_maps and not missing_clouds:
        print(f"[export] SKIP {unit} (all HF paths present — resume)", flush=True)
        return

    t0 = time.time()
    xy = ml._load_surface_xy(
        Path(args.turnstore_dir),
        model,
        args.format,
        args.corpus,
        smoke=False,
        wave1_dir=Path(args.wave1_turnstore_dir) if args.wave1_turnstore_dir else None,
        gen_root=Path(args.gen_root) if args.gen_root else None,
        expected_layers=cm.EXPECTED_LAYERS,
    )
    li = args.layer
    X = np.asarray(xy["X"][:, li, :])
    Y = np.asarray(xy["Y"][:, li, :])
    ids = np.asarray([str(c) for c in xy["conv_ids"]])
    n, d = X.shape
    folds = fc._cv_folds(ids, cm.N_FOLDS, cm.FIT_SEED)
    print(f"[export] {unit} loaded n={n} d={d} elapsed={time.time() - t0:.0f}s", flush=True)

    local = Path(args.local_root) / unit
    maps_dir = local / "fitted_maps"
    clouds_dir = local / "layer30_clouds"
    maps_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir.mkdir(parents=True, exist_ok=True)

    # ---- clouds (persistence of the paired vectors themselves) -------------
    clouds_npz = clouds_dir / f"clouds_{unit}.npz"
    np.savez(  # plain savez: compression OFF for Xet (#813)
        clouds_npz,
        X30=X.astype(np.float16),
        Y30=Y.astype(np.float16),
        conv_ids=ids,
        folds=folds.astype(np.int64),
    )
    (clouds_dir / f"clouds_{unit}.meta.json").write_text(
        json.dumps(
            {
                "stage": model,
                "corpus": args.corpus,
                "format": args.format,
                "layer": li,
                "n_rows": n,
                "d": d,
                "n_folds": int(cm.N_FOLDS),
                "fold_convention": "fc._cv_folds over the stage's OWN rows (seed "
                f"{cm.FIT_SEED}) — NOT any pair's intersection-row split",
                "extraction": "issue1336 turnstore_v2 x_slot='context' via _cell_xy_1336 "
                "(v_context = end-of-context position; v_answer = answer token-mean)",
                "dtype": "fp16",
                "sha256_npz": hashlib.sha256(clouds_npz.read_bytes()).hexdigest(),
                "code_sha": _git_sha(),
            },
            indent=2,
        )
    )

    # ---- fitted maps (per-fold exact affine form of the v2 ridge) ----------
    dev = fc._fit_device()
    Xd = torch.as_tensor(X, dtype=torch.float64).to(dev)
    Yd = torch.as_tensor(Y, dtype=torch.float64).to(dev)
    grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)
    store: dict[str, np.ndarray] = {"folds": folds.astype(np.int64)}
    lams: list[float] = []
    selectors: list[str] = []
    n_train: list[int] = []
    for k in range(int(cm.N_FOLDS)):
        tr = torch.as_tensor(folds != k)
        f0 = time.time()
        prep = ml._v2_prep(Xd[tr], inner_seed=cm.FIT_SEED + 4242 + k, n_inner=2)
        yfit = ml._v2_yfit(prep, Yd[tr], grid)
        xref = Xd[tr].mean(0)
        M, b = ml._ws_effective_matrix(prep, yfit, xref)
        store[f"W_f{k}"] = M.float().cpu().numpy().astype(np.float16)
        store[f"b_f{k}"] = b.float().cpu().numpy().astype(np.float32)
        store[f"xref_f{k}"] = xref.float().cpu().numpy().astype(np.float32)
        lams.append(float(yfit["lam"]))
        selectors.append(str(yfit["selector"]))
        n_train.append(int(tr.sum()))
        del prep, yfit, M, b
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[export] {unit} fold {k + 1}/{cm.N_FOLDS} lam={lams[-1]:.3g} "
            f"sel={selectors[-1]} elapsed={time.time() - f0:.0f}s",
            flush=True,
        )
    store["lams"] = np.asarray(lams, dtype=np.float64)
    del Xd, Yd
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    maps_npz = maps_dir / f"maps_{unit}.npz"
    np.savez(maps_npz, **store)
    (maps_dir / f"maps_{unit}.meta.json").write_text(
        json.dumps(
            {
                "stage": model,
                "corpus": args.corpus,
                "format": args.format,
                "layer": li,
                "n_rows": n,
                "d": d,
                "n_folds": int(cm.N_FOLDS),
                "n_train_per_fold": n_train,
                "estimator_degenerate_n_lt_d": [int(nt) < d for nt in n_train],
                "map_form": "pred(x) = (x - xref) @ W + b (W fp16 (d, d), b/xref fp32; "
                "exact standard-basis probe of the fitted v2 ridge — "
                "issue1336_metric_ladder._ws_effective_matrix)",
                "estimator": "v2 gram/primal ridge, lambda by inner-group-CV over "
                "cm.LAMBDAS_23 (gcv-fallback where inner CV unavailable; per-fold "
                "selected lambda + selector recorded — #1887 convention)",
                "selected_lambda": lams,
                "selectors": selectors,
                "fold_convention": "fc._cv_folds over the stage's OWN rows (seed "
                f"{cm.FIT_SEED}) — NOT any pair's intersection-row split",
                "sha256_npz": hashlib.sha256(maps_npz.read_bytes()).hexdigest(),
                "code_sha": _git_sha(),
            },
            indent=2,
        )
    )

    # ---- bulk upload per (unit, kind); verify; reap local -------------------
    from huggingface_hub import upload_folder

    for src, prefix, expected in (
        (clouds_dir, CLOUDS_PREFIX, cloud_paths),
        (maps_dir, MAPS_PREFIX, maps_paths),
    ):
        # Dir-filecount guard (#1190) OUTSIDE the retry wrapper (a guard raise
        # is deterministic; retrying it burns the budget for nothing).
        hub.assert_hub_dir_filecounts(src, prefix)
        hub.retry_transient(
            lambda src=src, prefix=prefix: upload_folder(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                folder_path=str(src),
                path_in_repo=prefix,
                commit_message=f"issue-1336 t5d round stage assets: {unit} -> {prefix}",
            ),
            what=f"stage-asset upload {unit} {prefix}",
        )
        still_missing = hub.verify_repo_paths_uploaded(
            api, cm.HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
        )
        assert not still_missing, f"uploaded but not visible: {still_missing}"
    import shutil

    shutil.rmtree(local)
    print(
        f"[export] {unit} DONE (uploaded + local reaped) total={time.time() - t0:.0f}s", flush=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="base,sft,dpo,rlvr,rlvr_long")
    ap.add_argument("--format", required=True, choices=("chat", "naturalistic"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--turnstore-dir", required=True)
    ap.add_argument("--wave1-turnstore-dir", default=None)
    ap.add_argument("--gen-root", default="data/issue_1336/gen")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--local-root", default="/workspace/data/issue_1336/t5d_stage_assets")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        _export_unit(args, model, api)


if __name__ == "__main__":
    main()
