#!/usr/bin/env python3
"""Issue #1336 scope extension — pooled-across-stages context->answer map.

Per v2 surface (all 8: 7 chat corpora + lmsys23k naturalistic;
gsm8k_test1319 runs but is flagged ``exclude_from_aggregates``), over the
LAYER-30 DIAGONAL clouds (X30 = context vector, Y30 = answer vector):

  POOLED MAP    stack the diagonal-cloud rows of ALL FIVE stages
                (base/sft/dpo/rlvr/rlvr_long) into one training set and fit
                ONE v2 ridge (``issue1336_metric_ladder._v2_prep/_v2_yfit``,
                dof-capped GCV fallback, LAMBDAS_23 — #1887 convention).
                LEAKAGE GUARD: outer folds are assigned BY CONVERSATION ID
                JOINTLY across stages (one 5-fold split on the 5-way conv_id
                intersection; a conversation's rows share a fold in every
                stage), and the inner lambda-CV groups are OVERRIDDEN to the
                same conv codes (``fc._prep_inner_lambda``) so near-duplicate
                cross-stage rows never straddle an inner split either.
  RUNGS         each pooled fit is scored held-out PER STAGE as-is, plus
                per-stage adapters fit on TRAIN folds only:
                  <var>            pooled prediction as-is
                  <var>_bias       + per-stage bias (train-mean residual)
                  <var>_rot        + per-stage Procrustes rotation
                                   (``issue825_map_alignment._orth_fit`` on
                                   pooled train-preds -> stage train answers;
                                   includes centering, scale=False)
                  <var>_rot_scale  same fit, scale=True
  VARIANTS      pooled          all 5 stages' train rows
                matchedn        pooled subsampled to ONE stage's train n
                                (the fit_cells matched-n convention:
                                sorted rng.choice, seed recorded) — the
                                matched-n read carries the claim (pooled-n
                                vs per-stage-n confound control)
                lofo            leave-base-out: fit on the 4 post-training
                                stages, scored on all 5
  CEILING       ``own`` — the stage's OWN v2 ridge refit on identical folds
  BASELINES     ``identity`` / ``id_bias``
                (``analysis.mapping_baselines.identity_bias_predict``)

Statistics: fold-local pooled OOF R^2 (per-fold values recorded) + the
globally-centered companion; kNN-retrieval
(``mapping_baselines.knn_retrieval``, euclidean + cosine, chance = k/n) on
identity / own / pooled / pooled_rot OOF predictions. Per-fit selected
lambda + selector recorded for every v2 fit. No bootstrap / shuffled-pairing
null (within-cell pairing — scope-extension spec).

Estimator validity: d = 4096. Pooled/lofo train n = 5x/4x one stage's train
n (primal d-space prep route). ``own`` and ``matchedn`` fits on
gsm8k_test1319 have n_train < d — flagged ``degenerate_n_lt_d`` and the
surface carries ``exclude_from_aggregates``.

Every ``__main__`` invocation exits explicitly (PyGILState_Release atexit
race — gotchas.md).
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
import issue825_map_alignment as ma  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

from issue1336_insertarm_clouds import retry_hub_409  # noqa: E402

CLOUDS_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/layer30_clouds"
STAGES = tuple(sorted(cm.MODELS, key=lambda m: cm.MODELS[m]["stage"]))
LOFO_STAGES = STAGES[1:]  # leave-base-out: the 4 post-training stages
VARIANTS = ("pooled", "matchedn", "lofo")
ADAPT_RUNGS = ("", "_bias", "_rot", "_rot_scale")
KNN_RUNGS = ("identity", "own", "pooled", "pooled_rot")
MATCHED_N_SEED_BASE = 7000  # + fold k (recorded per fold; fit_cells convention)


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


def _fetch_cloud(rel: str, clouds_root: Path) -> Path | None:
    """Download one cloud npz (small, kept) from the Hub; None when absent."""
    import os

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    target = clouds_root / Path(rel).name
    if target.exists():
        return target
    clouds_root.mkdir(parents=True, exist_ok=True)
    try:
        local = hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=cm.HF_DATA_REPO, repo_type="dataset", filename=rel, local_dir=clouds_root
            ),
            what=f"cloud download {rel}",
        )
    except EntryNotFoundError:
        return None
    os.replace(local, target)
    return target


def _load_cloud_xy(path: Path) -> dict:
    """Diagonal-cloud npz: X30 = context slot, Y30 = answer-span mean."""
    data = np.load(path, allow_pickle=False)
    return {
        "X30": np.asarray(data["X30"]),
        "Y30": np.asarray(data["Y30"]),
        "conv_ids": np.asarray([str(c) for c in data["conv_ids"]]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _shared_indexers(clouds: dict[str, dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """5-way conv_id intersection + per-stage row indexers into shared order."""
    from functools import reduce

    shared = reduce(np.intersect1d, (clouds[s]["conv_ids"] for s in clouds))
    idx: dict[str, np.ndarray] = {}
    for s, c in clouds.items():
        pos = {cid: i for i, cid in enumerate(c["conv_ids"])}
        idx[s] = np.asarray([pos[cid] for cid in shared], dtype=np.int64)
    return shared, idx


def _pooled_prep(X_tr: torch.Tensor, groups: np.ndarray, *, inner_seed: int) -> dict:
    """v2 prep with the inner lambda-CV groups OVERRIDDEN to conv codes.

    ``_v2_prep`` builds pointwise (arange) inner groups — correct when each
    conversation appears once, but a POOLED train block holds one row per
    stage per conversation, so pointwise inner splits would let near-duplicate
    rows straddle inner train/val and bias lambda low. Rebuilding the inner
    caches with conv-code groups keeps the whole chain leakage-guarded; the
    outer eigenbasis is untouched (same shared helpers by import — no
    re-implemented estimator).
    """
    prep = ml._v2_prep(X_tr, inner_seed=inner_seed, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2)
    prep["inner"] = fc._prep_inner_lambda(
        X_tr, groups, cm.N_INNER_LAMBDA_FOLDS_V2, inner_seed, device=X_tr.device
    )
    if prep["inner"] is None:
        print("[pooled] WARN: <2 usable conv-grouped inner folds — GCV fallback")
    return prep


def run_surface(
    corpus: str,
    fmt: str,
    *,
    clouds_root: Path,
    out_root: Path,
    layer: int,
) -> str:
    """One surface: load 5 stage clouds, joint folds, pooled battery, persist."""
    name = f"pooled_{fmt}_{corpus}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{name}.json"
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        if prior.get("status") == "complete":
            return "skipped-complete"

    rels = {s: f"{CLOUDS_PREFIX}/clouds_{s}_{fmt}_{corpus}.npz" for s in STAGES}
    paths = {s: _fetch_cloud(rels[s], clouds_root) for s in STAGES}
    missing = [rels[s] for s in STAGES if paths[s] is None]
    if missing:
        out_path.write_text(
            json.dumps({"status": "pending_dependency", "missing": missing}, indent=2)
        )
        return f"pending ({'; '.join(missing)})"

    clouds = {s: _load_cloud_xy(paths[s]) for s in STAGES}
    shared, idx = _shared_indexers(clouds)
    n = len(shared)
    assert n >= 50, f"{name}: 5-way aligned intersection too small (n={n})"
    dev = fc._fit_device()
    X = {s: torch.as_tensor(clouds[s]["X30"][idx[s]], dtype=torch.float64).to(dev) for s in STAGES}
    Y = {s: torch.as_tensor(clouds[s]["Y30"][idx[s]], dtype=torch.float64).to(dev) for s in STAGES}
    d = int(X[STAGES[0]].shape[1])
    for s in STAGES:
        assert X[s].shape == (n, d), (s, tuple(X[s].shape))
        assert Y[s].shape == (n, d), (s, tuple(Y[s].shape))
    folds = fc._cv_folds(shared, cm.N_FOLDS, cm.FIT_SEED)
    codes = np.arange(n, dtype=np.int64)  # conv codes in shared order
    grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)

    rung_names = [f"{v}{r}" for v in VARIANTS for r in ADAPT_RUNGS] + [
        "own",
        "identity",
        "id_bias",
    ]
    ss_res = {s: dict.fromkeys(rung_names, 0.0) for s in STAGES}
    ss_tot_local = dict.fromkeys(STAGES, 0.0)
    r2_folds: dict[str, dict[str, list[float]]] = {s: {r: [] for r in rung_names} for s in STAGES}
    captured = {s: {r: np.zeros((n, d), dtype=np.float32) for r in KNN_RUNGS} for s in STAGES}
    lam_log: dict[str, list[float]] = {v: [] for v in VARIANTS}
    sel_log: dict[str, list[str]] = {v: [] for v in VARIANTS}
    own_lam: dict[str, list[float]] = {s: [] for s in STAGES}
    own_sel: dict[str, list[str]] = {s: [] for s in STAGES}
    ntr_log: list[int] = []
    matched_log: list[dict] = []

    t0 = time.time()
    for k in range(int(cm.N_FOLDS)):
        te_np = folds == k
        tr_np = ~te_np
        if te_np.sum() == 0 or tr_np.sum() < 3:
            continue
        tr = torch.as_tensor(tr_np)
        te = torch.as_tensor(te_np)
        n_tr = int(tr_np.sum())
        ntr_log.append(n_tr)
        Xtr = {s: X[s][tr] for s in STAGES}
        Ytr = {s: Y[s][tr] for s in STAGES}
        Xte = {s: X[s][te] for s in STAGES}
        Yte_np = {s: Y[s][te].cpu().numpy() for s in STAGES}
        codes_tr = codes[tr_np]

        def _score(stage: str, rung: str, pred: torch.Tensor | np.ndarray) -> None:
            p = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
            true = Yte_np[stage]
            ss_res[stage][rung] += float(((true - p) ** 2).sum())
            mu = true.mean(0)
            sst = float(((true - mu) ** 2).sum())
            r2_folds[stage][rung].append(
                float("nan") if sst < 1e-12 else 1.0 - float(((true - p) ** 2).sum()) / sst
            )
            if rung in KNN_RUNGS:
                captured[stage][rung][te_np] = p.astype(np.float32)

        for variant in VARIANTS:
            var_stages = LOFO_STAGES if variant == "lofo" else STAGES
            Xp = torch.cat([Xtr[s] for s in var_stages], dim=0)
            Yp = torch.cat([Ytr[s] for s in var_stages], dim=0)
            groups = np.concatenate([codes_tr] * len(var_stages))
            if variant == "matchedn":
                sub_seed = cm.FIT_SEED + MATCHED_N_SEED_BASE + k
                rng = np.random.default_rng(sub_seed)
                keep = np.sort(rng.choice(len(groups), size=n_tr, replace=False))
                Xp, Yp, groups = Xp[keep], Yp[keep], groups[keep]
                matched_log.append({"fold": k, "matched_n": n_tr, "matched_n_seed": sub_seed})
            prep = _pooled_prep(Xp, groups, inner_seed=cm.FIT_SEED + 4242 + k)
            yfit = ml._v2_yfit(prep, Yp, grid)
            lam_log[variant].append(float(yfit["lam"]))
            sel_log[variant].append(str(yfit["selector"]))
            for s in STAGES:
                pred_tr = ml._v2_predict(prep, yfit, Xtr[s])
                pred_te = ml._v2_predict(prep, yfit, Xte[s])
                _score(s, variant, pred_te)
                bias = (Ytr[s] - pred_tr).mean(0)
                _score(s, f"{variant}_bias", pred_te + bias)
                orth = ma._orth_fit(pred_tr, Ytr[s])
                _score(
                    s,
                    f"{variant}_rot",
                    ma._orth_predict(orth, pred_te, reverse=False, scale=False),
                )
                _score(
                    s,
                    f"{variant}_rot_scale",
                    ma._orth_predict(orth, pred_te, reverse=False, scale=True),
                )
                del pred_tr, pred_te, orth
            del prep, yfit, Xp, Yp
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        for s in STAGES:
            prep_s = ml._v2_prep(
                Xtr[s], inner_seed=cm.FIT_SEED + 4242 + k, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2
            )
            yfit_s = ml._v2_yfit(prep_s, Ytr[s], grid)
            own_lam[s].append(float(yfit_s["lam"]))
            own_sel[s].append(str(yfit_s["selector"]))
            _score(s, "own", ml._v2_predict(prep_s, yfit_s, Xte[s]))
            _score(s, "identity", Xte[s])
            _score(
                s,
                "id_bias",
                identity_bias_predict(
                    Xtr[s].cpu().numpy(), Ytr[s].cpu().numpy(), Xte[s].cpu().numpy()
                ),
            )
            ss_tot_local[s] += float(((Yte_np[s] - Yte_np[s].mean(0)) ** 2).sum())
            del prep_s, yfit_s
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(
            f"[pooled] {name} fold {k + 1}/{int(cm.N_FOLDS)} n_tr={n_tr} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    r2_local = {
        s: {
            r: (1.0 - ss_res[s][r] / ss_tot_local[s] if ss_tot_local[s] > 1e-12 else float("nan"))
            for r in rung_names
        }
        for s in STAGES
    }
    r2_global = {}
    for s in STAGES:
        Ys_np = Y[s].cpu().numpy()
        sst_g = float(((Ys_np - Ys_np.mean(0)) ** 2).sum())
        r2_global[s] = {
            r: (1.0 - ss_res[s][r] / sst_g if sst_g > 1e-12 else float("nan")) for r in rung_names
        }
    knn = {
        s: {
            r: {
                m: knn_retrieval(captured[s][r], Y[s].cpu().numpy(), metric=m)
                for m in ("euclidean", "cosine")
            }
            for r in KNN_RUNGS
        }
        for s in STAGES
    }

    n_tr_min = min(ntr_log) if ntr_log else 0
    record = {
        "status": "complete",
        "corpus": corpus,
        "format": fmt,
        "layer": layer,
        "stages": list(STAGES),
        "lofo_fit_stages": list(LOFO_STAGES),
        "n_shared": int(n),
        "d": d,
        "n_folds": int(cm.N_FOLDS),
        "n_train_per_fold_per_stage": ntr_log,
        "pooled_n_train_per_fold": [len(STAGES) * v for v in ntr_log],
        "matched_n": matched_log,
        "degenerate_n_lt_d": {
            "own": bool(n_tr_min < d),
            "matchedn": bool(n_tr_min < d),
            "pooled": bool(len(STAGES) * n_tr_min < d),
            "lofo": bool(len(LOFO_STAGES) * n_tr_min < d),
        },
        "exclude_from_aggregates": corpus == "gsm8k_test1319",
        "rungs": rung_names,
        "r2_pooled_foldlocal": r2_local,
        "r2_pooled_global_companion": r2_global,
        "r2_per_fold": r2_folds,
        "knn_retrieval": knn,
        "selected_lambda": {"variants": lam_log, "own": own_lam},
        "selector": {"variants": sel_log, "own": own_sel},
        "inner_cv_groups": "conv-grouped (pooled variants, _prep_inner_lambda override); "
        "own fits one-row-per-conversation (arange == conv grouping)",
        "fold_assignment": "joint by conversation id across all 5 stages "
        "(fc._cv_folds on the 5-way intersection, FIT_SEED)",
        "cloud_provenance": {s: {"rel": rels[s], "sha256": clouds[s]["sha256"]} for s in STAGES},
        "code_sha": _git_sha(),
        "device": str(dev),
    }
    out_path.write_text(json.dumps(record, indent=2))
    return f"complete n={n}"


def _block_rotation(d: int, theta: float) -> np.ndarray:
    """Block-diagonal (d, d) rotation from d/2 planar rotations by ``theta``."""
    q = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(0, d - 1, 2):
        q[i, i], q[i, i + 1], q[i + 1, i], q[i + 1, i + 1] = c, -s, s, c
    return q


def _smoke(out_root: Path) -> None:
    """Synthetic 5-stage clouds through the REAL battery path on CPU.

    Fixture A (lmsys23k/chat surface name, n=300 > d=16): stage answers are
    per-stage block-rotations + biases of ONE common noisy map output, so the
    pooled map + per-stage rotation(+scale) rung must approach the own-map
    ceiling while pooled-as-is and bias-only stay visibly below.
    Fixture B (gsm8k_test1319/chat, n=60 < d=64): exercises the gram prep
    route + the degenerate_n_lt_d flags.
    """
    import tempfile

    rng = np.random.default_rng(0)
    thetas = dict(zip(STAGES, (-1.2, -0.6, 0.0, 0.6, 1.2), strict=True))

    def _write_fixture(root: Path, corpus: str, n: int, d: int) -> None:
        ids = np.asarray([f"c{i}" for i in range(n)])
        w = rng.normal(size=(d, d)) / np.sqrt(d)
        x0 = rng.normal(size=(n, d))
        y0 = x0 @ w + 0.02 * rng.normal(size=(n, d))
        for s in STAGES:
            q = _block_rotation(d, thetas[s])
            bias = 0.5 * rng.normal(size=(1, d))
            xs = x0 + 0.01 * rng.normal(size=(n, d))
            ys = y0 @ q + bias + 0.02 * rng.normal(size=(n, d))
            np.savez(
                root / f"clouds_{s}_chat_{corpus}.npz",
                X30=xs.astype(np.float16),
                Y30=ys.astype(np.float16),
                conv_ids=ids,
            )

    with tempfile.TemporaryDirectory(prefix="pooled_smoke_") as td:
        root = Path(td)
        _write_fixture(root, "lmsys23k", n=300, d=16)
        _write_fixture(root, "gsm8k_test1319", n=60, d=64)

        status = run_surface("lmsys23k", "chat", clouds_root=root, out_root=out_root, layer=1)
        rec = json.loads((out_root / "pooled_chat_lmsys23k.json").read_text())
        assert rec["status"] == "complete", rec
        r2 = rec["r2_pooled_foldlocal"]
        for s in STAGES:
            assert r2[s]["own"] > 0.9, (s, r2[s])
            assert r2[s]["pooled_rot_scale"] > 0.85, (s, r2[s])
            assert r2[s]["matchedn_rot_scale"] > 0.8, (s, r2[s])
            assert r2[s]["identity"] < 0.5, (s, r2[s])
            # bias cannot undo a rotation: rot_scale must clear bias-only
            assert r2[s]["pooled_rot_scale"] > r2[s]["pooled_bias"] + 0.05, (s, r2[s])
            assert np.isfinite(r2[s]["lofo_rot_scale"]), (s, r2[s])
        mean_asis = float(np.mean([r2[s]["pooled"] for s in STAGES]))
        mean_rot = float(np.mean([r2[s]["pooled_rot_scale"] for s in STAGES]))
        assert mean_asis < mean_rot - 0.2, (mean_asis, mean_rot)
        assert not rec["degenerate_n_lt_d"]["own"], rec["degenerate_n_lt_d"]
        print(f"[smoke] lmsys23k {status} asis={mean_asis:.3f} rot_scale={mean_rot:.3f}")

        status = run_surface("gsm8k_test1319", "chat", clouds_root=root, out_root=out_root, layer=1)
        rec = json.loads((out_root / "pooled_chat_gsm8k_test1319.json").read_text())
        assert rec["status"] == "complete", rec
        assert rec["degenerate_n_lt_d"]["own"], rec["degenerate_n_lt_d"]
        assert rec["degenerate_n_lt_d"]["matchedn"], rec["degenerate_n_lt_d"]
        assert rec["exclude_from_aggregates"] is True
        print(f"[smoke] gsm8k_test1319 {status} degenerate flags OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--surfaces",
        default=None,
        help="comma list of corpus:fmt (default: all 8 v2 surfaces)",
    )
    ap.add_argument("--clouds-root", type=Path, default=Path("data/issue_1336/insertarm_clouds"))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1336/pooled_stages"))
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--upload-results",
        action="store_true",
        help="bulk-upload the out-root battery JSONs to the Hub results mirror "
        "(issue1336_rlvr_ladder/eval_results_mirror_insertarm/pooled_stages) and exit",
    )
    args = ap.parse_args()

    if args.upload_results:
        from huggingface_hub import HfApi, upload_folder

        mirror = f"{cm.HF_PREFIX_1336}/eval_results_mirror_insertarm/pooled_stages"
        names = sorted(p.name for p in args.out_root.glob("pooled_*.json"))
        assert names, f"no battery JSONs under {args.out_root} — nothing to mirror"
        hub.assert_hub_dir_filecounts(args.out_root, mirror, allow_patterns=["pooled_*.json"])
        retry_hub_409(
            lambda: hub.retry_transient(
                lambda: upload_folder(
                    repo_id=cm.HF_DATA_REPO,
                    repo_type="dataset",
                    folder_path=str(args.out_root),
                    path_in_repo=mirror,
                    allow_patterns=["pooled_*.json"],
                    commit_message="issue-1336 scope extension: pooled-stages battery JSONs mirror",
                ),
                what="pooled-stages results mirror upload",
            ),
            what="pooled-stages results mirror upload",
        )
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            cm.HF_DATA_REPO,
            [f"{mirror}/{n}" for n in names],
            path_in_repo=mirror,
            repo_type="dataset",
        )
        assert not missing, f"results-mirror verify FAILED: missing {missing}"
        print(f"[pooled] mirrored {len(names)} battery JSONs -> {mirror}")
        return 0

    if args.smoke:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pooled_smoke_out_") as td:
            _smoke(Path(td))
        return 0

    if args.surfaces:
        units = []
        for tok in args.surfaces.split(","):
            corpus, fmt = tok.split(":")
            assert corpus in cm.V2_CORPORA, f"unknown corpus {corpus!r}"
            assert fmt in cm.V2_CORPORA[corpus]["formats"], f"unknown format {tok!r}"
            units.append((corpus, fmt))
    else:
        units = list(cm.v2_surfaces())
    t0 = time.time()
    n_pending = 0
    for k, (corpus, fmt) in enumerate(units):
        u0 = time.time()
        status = run_surface(
            corpus,
            fmt,
            clouds_root=args.clouds_root,
            out_root=args.out_root,
            layer=args.layer,
        )
        n_pending += status.startswith("pending")
        print(
            f"[pooled] unit {k + 1}/{len(units)} {fmt}:{corpus} {status} "
            f"elapsed={time.time() - u0:.0f}s total={time.time() - t0:.0f}s",
            flush=True,
        )
    print(f"[pooled] done: {len(units) - n_pending}/{len(units)} complete")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
