#!/usr/bin/env python3
"""Issue #1336 inserted-arm round — rigid-to-affine ladders on the two
decomposition arrows the teacher-forced matched-text captures unlock.

Per FORWARD stage pair (s -> t) x chat corpus, two arrows over LAYER-30
ANSWER clouds (all fp16 cloud npz from the Hub):

  RE-ENCODE  X = E_s(T_s) answer cloud (diagonal, t5d round export
             ``layer30_clouds/clouds_<s>_chat_<corpus>.npz``)
             Y = E_t(T_s) answer cloud (inserted, this round's harvest
             ``layer30_clouds/inserted/clouds_<t>_txt_<s>_chat_<corpus>.npz``)
             — matched TEXT, encoder varies: how far does re-encoding the
             SAME answers move the cloud, and what map class undoes it?
  CONTENT    X = E_t(T_s) (inserted)   Y = E_t(T_t) (diagonal target)
             — fixed ENCODER, text varies: what map class carries the
             source-content cloud onto the target-content cloud?

Ladder rungs (all held-out, seed-0 conversation-grouped 5-fold on the
conv_id-ALIGNED intersection — the pair-battery convention):
  identity            pred = X_te
  identity+bias       ``analysis.mapping_baselines.identity_bias_predict``
                      (the standing identity-family baseline duty)
  rigid               Procrustes rotation (``issue825_map_alignment._orth_fit``
                      / ``_orth_predict`` scale=False)
  rigid+scale         same fit, scale=True
  affine              the UNCHANGED v2 ridge chain
                      (``issue1336_metric_ladder._v2_prep/_v2_yfit/_v2_predict``,
                      dof-capped GCV, LAMBDAS_23, per-fold selected lambda +
                      selector recorded — #1887 convention)

Statistics: fold-local pooled OOF R^2 (the pair-file basis; the
globally-centered ``fc._pooled_r2`` companion reported separately); paired
bootstrap CIs over the OOF arrays (N_BOOTSTRAP shared draws per battery —
same draw matrix across rungs, so rung differences are paired); VERDICT =
the LOWEST rung whose paired (affine - rung) bootstrap CI includes 0.
Shuffled-pairing null per rung (N_NULL_DRAWS draws; fitted rungs refit on a
permuted TRAIN correspondence of fold 0 and evaluate on the TRUE test pairs;
unfitted rungs permute the EVAL correspondence — ``null_form`` recorded).
kNN-retrieval companion (``mapping_baselines.knn_retrieval``, euclidean +
cosine, chance stated) on identity / rigid / affine OOF predictions.

Estimator validity: d = 4096; gsm8k_test1319 (n < d) is estimator-degenerate
BY KNOWN DESIGN — every record carries ``degenerate_n_lt_d`` + realized
n_train, and the surface is excluded from aggregates downstream.

A battery whose DIAGONAL cloud has not landed yet (the live t5d round exports
them per surface) writes a ``pending_dependency`` JSON and exits 0 — the
driver retries pending units; residual pendings are reported as the round's
named dependency.

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
import issue1336_ladder_alignment as la  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

from issue1336_insertarm_clouds import (  # noqa: E402
    FORWARD_PAIRS,
    INSERTED_PREFIX,
    inserted_cloud_name,
)

CLOUDS_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/layer30_clouds"
RUNGS = ("identity", "id_bias", "rigid", "rigid_scale", "affine")
ARROWS = ("reencode", "content")
BOOT_SEED_BASE = 5600  # + v2 chat-surface index (distinct from 5000/5300 bases)


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


def _load_cloud(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "Y30": np.asarray(data["Y30"]),
        "conv_ids": np.asarray([str(c) for c in data["conv_ids"]]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _null_r2(
    X: torch.Tensor,
    Y: torch.Tensor,
    tr: np.ndarray,
    te: np.ndarray,
    rung: str,
    n_draws: int,
    seed: int,
    grid: np.ndarray,
    prep0: dict | None,
) -> list[float]:
    """Shuffled-pairing null draws for one rung on ONE fold split.

    Fitted rungs (id_bias / rigid / rigid_scale / affine) refit on a permuted
    TRAIN-side Y correspondence and evaluate on the TRUE test pairs; identity
    (no fit) permutes the EVAL correspondence instead. The X-side prep of the
    affine rung is Y-independent and shared across draws.
    """
    rng = np.random.default_rng(seed)
    out: list[float] = []
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    for _ in range(n_draws):
        if rung == "identity":
            perm = rng.permutation(int(te.sum()))
            pred = Xte.cpu().numpy()
            out.append(fc._pooled_r2(pred, Yte.cpu().numpy()[perm]))
            continue
        perm = rng.permutation(int(tr.sum()))
        Ytr_p = Ytr[perm]
        if rung == "id_bias":
            pred = identity_bias_predict(Xtr.cpu().numpy(), Ytr_p.cpu().numpy(), Xte.cpu().numpy())
        elif rung in ("rigid", "rigid_scale"):
            orth = ma._orth_fit(Xtr, Ytr_p)
            pred = (
                ma._orth_predict(orth, Xte, reverse=False, scale=rung == "rigid_scale")
                .cpu()
                .numpy()
            )
        elif rung == "affine":
            assert prep0 is not None
            yfit = ml._v2_yfit(prep0, Ytr_p, grid)
            pred = ml._v2_predict(prep0, yfit, Xte).cpu().numpy()
        else:  # pragma: no cover
            raise ValueError(rung)
        out.append(fc._pooled_r2(pred, Yte.cpu().numpy()))
    return out


def run_battery(
    src: str,
    tgt: str,
    corpus: str,
    arrow: str,
    *,
    clouds_root: Path,
    out_root: Path,
    n_boot: int,
    null_draws: int,
    layer: int,
) -> str:
    """One (pair, corpus, arrow) ladder battery: load, align, fit, persist."""
    name = f"arrow_{arrow}_{src}__{tgt}_chat_{corpus}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{name}.json"
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        if prior.get("status") == "complete":
            return "skipped-complete"

    ins_rel = f"{INSERTED_PREFIX}/{inserted_cloud_name(tgt, src, corpus)}.npz"
    diag_stage = src if arrow == "reencode" else tgt
    diag_rel = f"{CLOUDS_PREFIX}/clouds_{diag_stage}_chat_{corpus}.npz"
    ins_path = _fetch_cloud(ins_rel, clouds_root)
    diag_path = _fetch_cloud(diag_rel, clouds_root)
    missing = [r for r, p in ((ins_rel, ins_path), (diag_rel, diag_path)) if p is None]
    if missing:
        out_path.write_text(
            json.dumps({"status": "pending_dependency", "missing": missing}, indent=2)
        )
        return f"pending ({'; '.join(missing)})"

    ins = _load_cloud(ins_path)
    diag = _load_cloud(diag_path)
    if arrow == "reencode":
        a, b = diag, ins  # X = E_s(T_s), Y = E_t(T_s)
    else:
        a, b = ins, diag  # X = E_t(T_s), Y = E_t(T_t)
    ids, ia, ib = la._align_rows(a["conv_ids"], b["conv_ids"])
    n = len(ids)
    assert n >= 50, f"{name}: aligned intersection too small (n={n})"
    dev = fc._fit_device()
    X = torch.as_tensor(a["Y30"][ia], dtype=torch.float64).to(dev)
    Y = torch.as_tensor(b["Y30"][ib], dtype=torch.float64).to(dev)
    d = int(X.shape[1])
    folds = fc._cv_folds(ids, cm.N_FOLDS, cm.FIT_SEED)
    grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)

    captured = {r: np.zeros((n, d), dtype=np.float32) for r in RUNGS}
    ss_res = dict.fromkeys(RUNGS, 0.0)
    ss_tot = 0.0
    lam_log: list[float] = []
    sel_log: list[str] = []
    ntr_log: list[int] = []
    orth_diag: list[dict] = []
    prep_f0: dict | None = None
    tr_f0 = te_f0 = None
    for k in range(int(cm.N_FOLDS)):
        te_np = folds == k
        tr_np = ~te_np
        if te_np.sum() == 0 or tr_np.sum() < 3:
            continue
        tr = torch.as_tensor(tr_np)
        te = torch.as_tensor(te_np)
        Xte_np = X[te].cpu().numpy()
        Yte_np = Y[te].cpu().numpy()
        captured["identity"][te_np] = Xte_np.astype(np.float32)
        captured["id_bias"][te_np] = identity_bias_predict(
            X[tr].cpu().numpy(), Y[tr].cpu().numpy(), Xte_np
        ).astype(np.float32)
        orth = ma._orth_fit(X[tr], Y[tr])
        captured["rigid"][te_np] = (
            ma._orth_predict(orth, X[te], reverse=False, scale=False).float().cpu().numpy()
        )
        captured["rigid_scale"][te_np] = (
            ma._orth_predict(orth, X[te], reverse=False, scale=True).float().cpu().numpy()
        )
        prep = ml._v2_prep(
            X[tr], inner_seed=cm.FIT_SEED + 4242 + k, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2
        )
        yfit = ml._v2_yfit(prep, Y[tr], grid)
        captured["affine"][te_np] = ml._v2_predict(prep, yfit, X[te]).float().cpu().numpy()
        for r in RUNGS:
            ss_res[r] += float(((Yte_np - captured[r][te_np]) ** 2).sum())
        ss_tot += float(((Yte_np - Yte_np.mean(0)) ** 2).sum())
        lam_log.append(float(yfit["lam"]))
        sel_log.append(str(yfit["selector"]))
        ntr_log.append(int(tr_np.sum()))
        orth_diag.append({"fold": k, "s_fwd": float(orth["s_fwd"]), "s_rev": float(orth["s_rev"])})
        if k == 0:
            prep_f0, tr_f0, te_f0 = prep, tr_np, te_np
        else:
            del prep, yfit
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    Y_np = Y.cpu().numpy()
    r2_pooled = {r: (1.0 - ss_res[r] / ss_tot if ss_tot > 1e-12 else float("nan")) for r in RUNGS}
    r2_global = {r: fc._pooled_r2(captured[r], Y_np) for r in RUNGS}

    seed = BOOT_SEED_BASE + cm.v2_surface_index(corpus, "chat")
    w = la.counts_from_indices(la.draw_index_matrix(n, n_boot, seed), n)
    draws = {r: la.weighted_r2_draws(captured[r], Y_np, w) for r in RUNGS}
    ci = {r: la._ci(draws[r]) for r in RUNGS}
    delta_ci = {r: la._ci(draws["affine"] - draws[r]) for r in RUNGS if r != "affine"}
    verdict = "affine"
    for r in RUNGS[:-1]:  # ordered simple -> complex
        if delta_ci[r]["ci_lo"] <= 0.0:
            verdict = r
            break

    assert prep_f0 is not None and tr_f0 is not None and te_f0 is not None
    nulls = {
        r: _null_r2(X, Y, tr_f0, te_f0, r, null_draws, seed + 17, grid, prep_f0) for r in RUNGS
    }
    knn = {
        r: {m: knn_retrieval(captured[r], Y_np, metric=m) for m in ("euclidean", "cosine")}
        for r in ("identity", "rigid", "affine")
    }

    n_tr_min = min(ntr_log) if ntr_log else 0
    record = {
        "status": "complete",
        "arrow": arrow,
        "pair": {"source": src, "target": tgt},
        "corpus": corpus,
        "format": "chat",
        "layer": layer,
        "n_aligned": int(n),
        "d": d,
        "n_folds": int(cm.N_FOLDS),
        "n_train_per_fold": ntr_log,
        "degenerate_n_lt_d": bool(n_tr_min < d),
        "rungs": list(RUNGS),
        "r2_pooled_foldlocal": r2_pooled,
        "r2_pooled_global_companion": r2_global,
        "r2_bootstrap_ci": ci,
        "affine_minus_rung_delta_ci": delta_ci,
        "verdict_lowest_rung_within_noise_of_affine": verdict,
        "null_shuffled_pairing": {
            "draws": nulls,
            "n_draws": null_draws,
            "null_form": "fitted rungs: permuted TRAIN correspondence refit on fold 0, "
            "evaluated on TRUE test pairs; identity: permuted EVAL correspondence",
        },
        "knn_retrieval": knn,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "paired_across_rungs": True},
        "affine_selected_lambda": lam_log,
        "affine_selector": sel_log,
        "procrustes_scales": orth_diag,
        "cloud_provenance": {
            "inserted": {"rel": ins_rel, "sha256": ins["sha256"]},
            "diagonal": {"rel": diag_rel, "sha256": diag["sha256"]},
        },
        "code_sha": _git_sha(),
        "device": str(dev),
    }
    out_path.write_text(json.dumps(record, indent=2))
    return f"complete n={n} verdict={verdict}"


def _smoke(out_root: Path) -> None:
    """Tiny synthetic clouds through the REAL battery path on CPU."""
    import tempfile

    rng = np.random.default_rng(0)
    n, d = 60, 8
    ids = np.asarray([f"s{i}" for i in range(n)])
    R1, _ = np.linalg.qr(rng.normal(size=(d, d)))
    R2, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Xs = rng.normal(size=(n, d))
    # inserted cloud M sits between the two diagonals: reencode = Xs -> M,
    # content = M -> Yt; both rotation+bias so rigid must win both arrows.
    M = Xs @ R1 + 0.05 * rng.normal(size=(n, d)) + 0.3
    Yt = M @ R2 + 0.05 * rng.normal(size=(n, d)) - 0.2
    with tempfile.TemporaryDirectory(prefix="arrow_smoke_") as td:
        root = Path(td)
        np.savez(
            root / f"{inserted_cloud_name('dpo', 'base', 'gsm8k_test1319')}.npz",
            X30=M.astype(np.float16),
            Y30=M.astype(np.float16),
            conv_ids=ids,
            folds=fc._cv_folds(ids, cm.N_FOLDS, cm.FIT_SEED).astype(np.int64),
        )
        for stage, arr in (("base", Xs), ("dpo", Yt)):
            np.savez(
                root / f"clouds_{stage}_chat_gsm8k_test1319.npz",
                X30=arr.astype(np.float16),
                Y30=arr.astype(np.float16),
                conv_ids=ids,
                folds=fc._cv_folds(ids, cm.N_FOLDS, cm.FIT_SEED).astype(np.int64),
            )
        for arrow in ARROWS:
            status = run_battery(
                "base",
                "dpo",
                "gsm8k_test1319",
                arrow,
                clouds_root=root,
                out_root=out_root,
                n_boot=50,
                null_draws=2,
                layer=1,
            )
            rec = json.loads(
                (out_root / f"arrow_{arrow}_base__dpo_chat_gsm8k_test1319.json").read_text()
            )
            assert rec["status"] == "complete"
            # rotation+bias generated the smoke Y -> rigid must be near-perfect
            # and within noise of affine; identity must be far below.
            assert rec["r2_pooled_foldlocal"]["rigid"] > 0.9, rec["r2_pooled_foldlocal"]
            assert rec["r2_pooled_foldlocal"]["identity"] < 0.5
            print(f"[smoke] arrow={arrow} {status} r2={rec['r2_pooled_foldlocal']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", default=None, help="s:t (one forward pair)")
    ap.add_argument("--arrows", default="reencode,content")
    ap.add_argument("--corpora", default=None, help="comma list (default: 7 chat corpora)")
    ap.add_argument("--clouds-root", type=Path, default=Path("data/issue_1336/insertarm_clouds"))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1336/arrow_ladders"))
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--null-draws", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--upload-results",
        action="store_true",
        help="bulk-upload the out-root battery JSONs to the Hub results mirror "
        "(issue1336_rlvr_ladder/eval_results_mirror_insertarm/arrow_ladders) and exit",
    )
    args = ap.parse_args()

    if args.upload_results:
        from huggingface_hub import HfApi, upload_folder

        mirror = f"{cm.HF_PREFIX_1336}/eval_results_mirror_insertarm/arrow_ladders"
        names = sorted(p.name for p in args.out_root.glob("arrow_*.json"))
        assert names, f"no battery JSONs under {args.out_root} — nothing to mirror"
        hub.assert_hub_dir_filecounts(args.out_root, mirror, allow_patterns=["arrow_*.json"])
        hub.retry_transient(
            lambda: upload_folder(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                folder_path=str(args.out_root),
                path_in_repo=mirror,
                allow_patterns=["arrow_*.json"],
                commit_message="issue-1336 inserted-arm: arrow-ladder battery JSONs mirror",
            ),
            what="arrow-ladder results mirror upload",
        )
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            cm.HF_DATA_REPO,
            [f"{mirror}/{n}" for n in names],
            path_in_repo=mirror,
            repo_type="dataset",
        )
        assert not missing, f"results-mirror verify FAILED: missing {missing}"
        print(f"[arrow] mirrored {len(names)} battery JSONs -> {mirror}")
        return 0

    if args.smoke:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="arrow_smoke_out_") as td:
            _smoke(Path(td))
        return 0

    assert args.pair, "--pair s:t required outside --smoke"
    src, tgt = args.pair.split(":")
    assert (src, tgt) in FORWARD_PAIRS, f"{src}:{tgt} is not a forward pair"
    arrows = tuple(a for a in args.arrows.split(",") if a)
    for a in arrows:
        assert a in ARROWS, f"unknown arrow {a!r}"
    corpora = tuple(args.corpora.split(",")) if args.corpora else tuple(cm.V2_CORPORA)
    n_boot = args.n_boot if args.n_boot is not None else cm.N_BOOTSTRAP
    null_draws = args.null_draws if args.null_draws is not None else cm.N_NULL_DRAWS
    units = [(c, a) for c in corpora for a in arrows]
    t0 = time.time()
    n_pending = 0
    for k, (c, a) in enumerate(units):
        u0 = time.time()
        status = run_battery(
            src,
            tgt,
            c,
            a,
            clouds_root=args.clouds_root,
            out_root=args.out_root,
            n_boot=n_boot,
            null_draws=null_draws,
            layer=args.layer,
        )
        n_pending += status.startswith("pending")
        print(
            f"[arrow] unit {k + 1}/{len(units)} {a}:{src}->{tgt}:{c} {status} "
            f"elapsed={time.time() - u0:.0f}s total={time.time() - t0:.0f}s",
            flush=True,
        )
    print(f"[arrow] pair {src}->{tgt} done: {len(units) - n_pending}/{len(units)} complete")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
