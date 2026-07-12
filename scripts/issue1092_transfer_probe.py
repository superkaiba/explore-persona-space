#!/usr/bin/env python3
"""Issue #1092 follow-up `cross-corpus-probe-transfer`: 0-GPU supervised-probe transfer.

Trains the parent B1(c) direct-regression ridge probe (layer 14, ambient basis,
banked PRESS engine) on one corpus and scores it zero-shot on the other:

- Direction A: #1092 `cell_inst_own` scored rows (context_end + bare-query arms)
  -> #779 pass_b LMSYS states (primary) + pass_a persona-condition states
  (secondary, non-gating).
- Direction B: #779 LMSYS states + lmsys_g labels -> #1092 context_end / bare
  states of the same scored rows (paired fixed-reader comparison).

All inputs are banked (plan v7 pins); no GPU, no new API calls. Gates: the
within-#1092 reproduction pre-gate, the pre-registered overlap dedup, and the
four-part pass_b <-> labels alignment gate (structural / prompt sequence /
ctx0 spot assert / empirical within-LMSYS ceiling floor). Verdict logic is
pre-registered in plans/v7.md.

Smoke mode IS this entrypoint (--smoke): identical phases on subsampled pools,
reduced draws, repro-gate values computed but not enforced, upload prefix
suffixed `_smoke`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env must bind BEFORE the heavy imports below — the
# BLAS/torch pools freeze at import time (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, list_repo_tree  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue779_ffc_n10k_generate_capture import EXPECTED_CTX0_PROMPT  # noqa: E402
from issue923_fit_decomposition import press_fit_predict, run_selftest  # noqa: E402

# Byte-pinned parent engine helpers (read-only import; issue1092_fit_grid.py is
# BYTE-PINNED per plan v7 — never edited here).
from issue1092_fit_grid import (  # noqa: E402
    _bare_X_for_unit,
    _folds_from_manifest,
    _jsonl,
    _load_summary,
    _pearson_or_nan,
    _r2,
)
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    assert_hub_dir_filecounts,
    retry_transient,
    verify_repo_paths_uploaded,
)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

HF_REPO = "superkaiba1/explore-persona-space-data"
PREFIX_1092 = "issue1092_realistic_crossing"
PREFIX_PASSB = "issue779_monitoring/analysis_tensors/pass_b"
PREFIX_PASSA = "issue779_monitoring/analysis_tensors/pass_a"
PREFIX_LABELS = "issue779_monitoring/training-source-ablation-hg/lmsys_g_labels"
CELL = "cell_inst_own"
MODEL_TYPE = "instruct"
ALL_TRAITS = ("evil", "hallucination", "sycophancy")
VERDICT_TRAITS = ("hallucination", "sycophancy")
# Fit-arm A engine rule, verbatim from issue1092_fit_grid.run() (the REALIZED
# banked pool: the stratum literal "battery_eval_only" never matches the actual
# "battery" stratum, so the 2,400 battery rows are INCLUDED — plan v7 + the
# banked battery_scope_caveat both document this).
FITA_EXCLUDED_STRATA = {"trait_stratum", "battery_eval_only"}
BANKED_B1_JSON = PROJECT_ROOT / "eval_results/issue_1092/p7/behavior_B1_B2.json"
PASSA_CONDS = tuple(
    [f"sys{i}" for i in range(8)] + [f"shot{n}" for n in (0, 5, 10, 15, 20)]
)  # 13 conditions/trait

DELTA_R_MARGIN = 0.05  # #779 registered in-behavior success bar (plan §11)
REPRO_TOL = 1e-3  # kill criterion 1 (deterministic recompute)
OVERLAP_FLAG_FRAC = 0.20  # kill criterion 2


def _log(msg: str) -> None:
    print(f"[transfer-probe +{time.monotonic() - T0:.0f}s] {msg}", flush=True)


T0 = time.monotonic()


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _norm_text(t: str) -> str:
    return " ".join(t.lower().split())


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types + non-finite floats (-> None) for JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _main_repo_root() -> Path:
    """The MAIN checkout root (worktrees share big local artifact copies with it)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            common = Path(out.stdout.strip())
            if not common.is_absolute():
                common = (PROJECT_ROOT / common).resolve()
            return common.parent
    except Exception:
        pass
    return PROJECT_ROOT


MAIN_ROOT = _main_repo_root()


# ── staging (local-first with hub-sha verification; HF-fetch fallback at pins) ─


def _hub_lfs_sha_map(prefix: str, revision: str) -> dict[str, tuple[int, str | None]]:
    """{filename: (size, lfs_sha256|None)} for one hub prefix at a pinned rev."""
    out: dict[str, tuple[int, str | None]] = {}
    items = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: expand=True lfs-sha metadata; scoped+pinned; retry-wrapped
            list_repo_tree(
                HF_REPO, repo_type="dataset", path_in_repo=prefix, revision=revision, expand=True
            )
        ),
        what=f"list_repo_tree {prefix}@{revision[:12]}",
    )
    for it in items:
        name = it.path.rsplit("/", 1)[-1]
        lfs = getattr(it, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        out[name] = (int(getattr(it, "size", 0) or 0), sha)
    return out


def _stage_verified_local(
    local_candidates: list[Path], expected_sha256: str | None, rel: str, revision: str
) -> Path:
    """Return a verified local path: first candidate whose sha matches, else hub fetch."""
    for cand in local_candidates:
        if cand.is_file():
            if expected_sha256 is None:
                continue  # no comparable hub identity -> fetch at pin instead
            if _sha256_file(cand) == expected_sha256:
                _log(f"staged (local, sha-verified): {cand}")
                return cand
            _log(f"local candidate sha MISMATCH (falling back to hub): {cand}")
    path = Path(hf_hub_download(HF_REPO, repo_type="dataset", revision=revision, filename=rel))
    if expected_sha256 is not None:
        got = _sha256_file(path)
        if got != expected_sha256:
            raise RuntimeError(f"hub download sha mismatch for {rel}: {got} != {expected_sha256}")
    _log(f"staged (hub @ {revision[:12]}): {rel}")
    return path


def stage_inputs(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    """Verify/stage every input at the plan's pinned revisions. Returns path dict."""
    staged: dict[str, Any] = {"shas": {}}
    stage_root = PROJECT_ROOT / "data/issue_1092/transfer_probe"
    summaries_dir = stage_root / "summaries"

    # -- #1092 corpus (manifest + stores): non-LFS jsonl -> fetch at pin (content-exact).
    corpus_dir = stage_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.jsonl", "query_store.jsonl", "prefix_store.jsonl"):
        p = Path(
            hf_hub_download(
                HF_REPO,
                repo_type="dataset",
                revision=args.hf_rev_1092,
                filename=f"{PREFIX_1092}/corpus/{name}",
            )
        )
        dst = corpus_dir / name
        if not dst.exists() or _sha256_file(dst) != _sha256_file(p):
            shutil.copyfile(p, dst)
        staged["shas"][f"corpus/{name}"] = _sha256_file(dst)[:16]
    staged["corpus_dir"] = corpus_dir

    # -- #1092 P5 judge scores: local scores.jsonl verified against the hub
    #    shards_manifest full_sha256 (the parent P6 pin); else reassemble shards.
    man_p = Path(
        hf_hub_download(
            HF_REPO,
            repo_type="dataset",
            revision=args.hf_rev_1092,
            filename=f"{PREFIX_1092}/p5_judge/shards_manifest.json",
        )
    )
    man = json.loads(man_p.read_text())
    scores_path = None
    for cand in (
        PROJECT_ROOT / "data/issue_1092/p5/scores.jsonl",
        stage_root / "p5_scores.jsonl",
    ):
        if cand.is_file() and _sha256_file(cand) == man["full_sha256"]:
            scores_path = cand
            break
    if scores_path is None:
        _log("reassembling p5 scores.jsonl from hub shards at pin")
        stage_root.mkdir(parents=True, exist_ok=True)
        tmp = stage_root / "p5_scores.jsonl.tmp"
        h = hashlib.sha256()
        with open(tmp, "wb") as out:
            for i in range(int(man["n_shards"])):
                sp = Path(
                    hf_hub_download(
                        HF_REPO,
                        repo_type="dataset",
                        revision=args.hf_rev_1092,
                        filename=f"{PREFIX_1092}/p5_judge/scores_shard_{i:03d}.jsonl",
                    )
                )
                b = sp.read_bytes()
                h.update(b)
                out.write(b)
        if h.hexdigest() != man["full_sha256"]:
            raise RuntimeError("reassembled p5 scores sha mismatch vs shards_manifest")
        scores_path = stage_root / "p5_scores.jsonl"
        os.replace(tmp, scores_path)
    staged["judge_scores"] = scores_path
    staged["shas"]["p5_scores.jsonl"] = man["full_sha256"][:16]
    _log(f"p5 judge scores verified: {scores_path}")

    # -- #1092 summaries: context_end_L14 (cell) + bare_instruct L14 shards + row_index.
    cell_map = _hub_lfs_sha_map(
        f"{PREFIX_1092}/analysis_tensors/summaries/{CELL}", args.hf_rev_1092
    )
    bare_map = _hub_lfs_sha_map(
        f"{PREFIX_1092}/analysis_tensors/summaries/bare_{MODEL_TYPE}", args.hf_rev_1092
    )
    (summaries_dir / CELL).mkdir(parents=True, exist_ok=True)
    (summaries_dir / f"bare_{MODEL_TYPE}").mkdir(parents=True, exist_ok=True)
    ctx_name = f"context_end_L{args.layer:02d}.npy"
    ctx_src = _stage_verified_local(
        [
            PROJECT_ROOT / f"data/issue_1092/p7/read4c_repair/staging/{CELL}/{ctx_name}",
            summaries_dir / CELL / ctx_name,
        ],
        cell_map.get(ctx_name, (0, None))[1],
        f"{PREFIX_1092}/analysis_tensors/summaries/{CELL}/{ctx_name}",
        args.hf_rev_1092,
    )
    dst = summaries_dir / CELL / ctx_name
    if ctx_src.resolve() != dst.resolve():
        if dst.exists():
            dst.unlink()
        try:
            os.link(ctx_src, dst)
        except OSError:
            shutil.copyfile(ctx_src, dst)
    staged["shas"][f"{CELL}/{ctx_name}"] = (cell_map.get(ctx_name, (0, ""))[1] or "")[:16]
    bare_names = sorted(
        n
        for n in bare_map
        if (n.startswith(f"c_q_bare_L{args.layer:02d}_shard") and n.endswith(".npy"))
        or (n.startswith("row_index_shard") and n.endswith(".jsonl"))
        or n in (f"c_q_bare_L{args.layer:02d}.npy", "row_index.jsonl")
    )
    if not bare_names:
        raise RuntimeError(f"no bare_{MODEL_TYPE} L{args.layer:02d} files listed at pin")
    for name in bare_names:
        dst = summaries_dir / f"bare_{MODEL_TYPE}" / name
        exp = bare_map[name][1]
        if dst.exists() and (exp is None or _sha256_file(dst) == exp):
            continue
        p = Path(
            hf_hub_download(
                HF_REPO,
                repo_type="dataset",
                revision=args.hf_rev_1092,
                filename=f"{PREFIX_1092}/analysis_tensors/summaries/bare_{MODEL_TYPE}/{name}",
            )
        )
        shutil.copyfile(p, dst)
    staged["summaries_dir"] = summaries_dir
    _log(f"summaries staged: {ctx_name} + {len(bare_names)} bare files")

    # -- #779 pass_b bundle (6 GB): local copy verified against hub LFS sha at pin.
    pb_map = _hub_lfs_sha_map(PREFIX_PASSB, args.hf_rev_779_passb)
    pb_sha = pb_map["train_context_vectors.pt"][1]
    staged["pass_b"] = _stage_verified_local(
        [
            PROJECT_ROOT / "data/issue_779/pass_b/train_context_vectors.pt",
            MAIN_ROOT / "data/issue_779/pass_b/train_context_vectors.pt",
        ],
        pb_sha,
        f"{PREFIX_PASSB}/train_context_vectors.pt",
        args.hf_rev_779_passb,
    )
    staged["shas"]["pass_b/train_context_vectors.pt"] = (pb_sha or "")[:16]

    # -- #779 labels + rollouts: non-LFS json -> fetch at pin.
    staged["labels"] = Path(
        hf_hub_download(
            HF_REPO,
            repo_type="dataset",
            revision=args.hf_rev_779_labels,
            filename=f"{PREFIX_LABELS}/lmsys_g_labels.json",
        )
    )
    staged["rollouts"] = Path(
        hf_hub_download(
            HF_REPO,
            repo_type="dataset",
            revision=args.hf_rev_779_labels,
            filename=f"{PREFIX_LABELS}/lmsys_g_rollouts.json",
        )
    )
    staged["shas"]["lmsys_g_labels.json"] = _sha256_file(staged["labels"])[:16]
    staged["shas"]["lmsys_g_rollouts.json"] = _sha256_file(staged["rollouts"])[:16]

    # -- #779 pass_a (verdict traits only; secondary surface): .pt local-verified
    #    against LFS sha, .json fetched at pin.
    pa_map = _hub_lfs_sha_map(PREFIX_PASSA, args.hf_rev_779_passb)
    passa: dict[str, dict[str, dict[str, Path]]] = {}
    for trait in VERDICT_TRAITS:
        passa[trait] = {}
        for cond in PASSA_CONDS:
            cx_name = f"{trait}__{cond}_cx.pt"
            js_name = f"{trait}__{cond}.json"
            if cx_name not in pa_map or js_name not in pa_map:
                raise RuntimeError(f"pass_a missing {cx_name} / {js_name} at pin")
            cx = _stage_verified_local(
                [
                    PROJECT_ROOT / f"data/issue_779/pass_a/{cx_name}",
                    MAIN_ROOT / f"data/issue_779/pass_a/{cx_name}",
                ],
                pa_map[cx_name][1],
                f"{PREFIX_PASSA}/{cx_name}",
                args.hf_rev_779_passb,
            )
            js = Path(
                hf_hub_download(
                    HF_REPO,
                    repo_type="dataset",
                    revision=args.hf_rev_779_passb,
                    filename=f"{PREFIX_PASSA}/{js_name}",
                )
            )
            passa[trait][cond] = {"cx": cx, "json": js}
    staged["pass_a"] = passa
    _log("pass_a staged (2 traits x 13 conditions)")
    return staged


# ── fit engine wrappers (press_fit_predict reused verbatim — byte-identical) ──


def _extract_ambient_weights(res: dict, y: np.ndarray, lam_idx: int) -> tuple[np.ndarray, float]:
    """Compose the fitted ridge at RIDGE_LAMBDAS[lam_idx] into ambient (w, b).

    pred(Xe) == Xe @ w + b, algebraically identical to per_lambda_pred[lam_idx]
    (dual-form weights through the train-side standardization).
    """
    eng, _xtr_n, _xte_n = res["engine"]
    mu, sd, keep = res["std"]
    ymu = float(res["ymu"].reshape(-1)[0])
    yc = torch.from_numpy(y.reshape(-1, 1)).double() - float(np.mean(y))
    g = eng.U.T @ yc  # (k, 1)
    lam = float(RIDGE_LAMBDAS[lam_idx])
    coef = (eng.S / (eng.S**2 + lam)).unsqueeze(1)  # (k, 1)
    w_keep = (eng.Vh.T @ (coef * g)).squeeze(1)  # (d_keep,)
    keep_np = keep.cpu().numpy().astype(bool)
    w = np.zeros(keep_np.shape[0], dtype=np.float64)
    w[keep_np] = (w_keep / sd[keep]).cpu().numpy()
    b = ymu - float(mu.cpu().numpy() @ w)
    return w, b


def fold_fits(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    eval_surfaces: list[np.ndarray],
) -> dict[str, Any]:
    """Per-fold PRESS ridge fits over ``folds`` + full-pool fit, one engine pass.

    Returns:
      cv_pred        pooled held-out predictions at each fold's PRESS-selected
                     lambda (== the banked _fit_cv/_fit_scalar_cv predictions)
      cv_pred_perlam (n_lambda, n) pooled held-out predictions per lambda
                     (-> grouped-CV lambda* for the transfer probe)
      lam_press_folds / lam_cv_idx
      fold_eval      per surface: (n_folds, n_e) fold-probe predictions
                     (train-side sensitivity band)
      full_eval      per surface: (n_e,) full-pool probe predictions at lambda*
      w_full, b_full, w_folds, b_folds  ambient probe weights
      train_mean_floor  (matches _fit_scalar_cv)
    """
    n = X.shape[0]
    n_lam = len(RIDGE_LAMBDAS)
    xe_cat = (
        np.concatenate(eval_surfaces, axis=0)
        if eval_surfaces
        else np.zeros((0, X.shape[1]), dtype=np.float64)
    )
    offs = np.cumsum([0] + [e.shape[0] for e in eval_surfaces])
    cv_pred = np.zeros(n, dtype=np.float64)
    cv_pred_perlam = np.zeros((n_lam, n), dtype=np.float64)
    fold_eval = [np.zeros((len(folds), e.shape[0]), dtype=np.float64) for e in eval_surfaces]
    lam_press_folds: list[int] = []
    floors: list[float] = []
    w_folds: list[np.ndarray] = []
    b_folds: list[float] = []
    for fi, te in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        xte_full = np.concatenate([X[te], xe_cat], axis=0)
        res = press_fit_predict(
            torch.from_numpy(X[mask]).double(),
            torch.from_numpy(y[mask].reshape(-1, 1)).double(),
            torch.from_numpy(xte_full).double(),
            return_engine=True,
            standardize=True,
        )
        li = int(res["lam_idx"])
        lam_press_folds.append(li)
        per_lam = [p.detach().cpu().numpy()[:, 0] for p in res["per_lambda_pred"]]
        nte = int(te.size)
        cv_pred[te] = per_lam[li][:nte]
        for lidx in range(n_lam):
            cv_pred_perlam[lidx, te] = per_lam[lidx][:nte]
        for si in range(len(eval_surfaces)):
            a, b = nte + int(offs[si]), nte + int(offs[si + 1])
            fold_eval[si][fi] = per_lam[li][a:b]
        wf, bf = _extract_ambient_weights(res, y[mask], li)
        w_folds.append(wf)
        b_folds.append(bf)
        floors.append(
            _r2(
                y[te].reshape(-1, 1),
                np.broadcast_to(y[mask].mean(), (nte, 1)),
            )
        )
        del res
        gc.collect()
    # grouped-CV lambda*: pooled per-lambda held-out MSE over the SAME folds.
    cv_mse = ((cv_pred_perlam - y.reshape(1, -1)) ** 2).mean(axis=1)
    lam_cv_idx = int(np.argmin(cv_mse))
    # Full-pool fit; transfer predictions read at lambda* (train side only).
    res_full = press_fit_predict(
        torch.from_numpy(X).double(),
        torch.from_numpy(y.reshape(-1, 1)).double(),
        torch.from_numpy(xe_cat).double(),
        return_engine=True,
        standardize=True,
    )
    per_lam_full = [p.detach().cpu().numpy()[:, 0] for p in res_full["per_lambda_pred"]]
    full_eval = [
        per_lam_full[lam_cv_idx][int(offs[si]) : int(offs[si + 1])]
        for si in range(len(eval_surfaces))
    ]
    w_full, b_full = _extract_ambient_weights(res_full, y, lam_cv_idx)
    # self-check: composed ambient weights reproduce the engine's predictions.
    if eval_surfaces and eval_surfaces[0].shape[0] > 0:
        probe_rows = eval_surfaces[0][: min(8, eval_surfaces[0].shape[0])]
        direct = probe_rows @ w_full + b_full
        engine_pred = full_eval[0][: probe_rows.shape[0]]
        if not np.allclose(direct, engine_pred, rtol=1e-6, atol=1e-8):
            raise RuntimeError("ambient weight composition diverged from engine predictions")
    lam_press_full = int(res_full["lam_idx"])
    del res_full
    gc.collect()
    return {
        "cv_pred": cv_pred,
        "cv_pred_perlam": cv_pred_perlam,
        "lam_press_folds": lam_press_folds,
        "lam_cv_idx": lam_cv_idx,
        "lam_press_full": lam_press_full,
        "fold_eval": fold_eval,
        "full_eval": full_eval,
        "w_full": w_full,
        "b_full": b_full,
        "w_folds": np.stack(w_folds) if w_folds else np.zeros((0, X.shape[1])),
        "b_folds": np.asarray(b_folds, dtype=np.float64),
        "train_mean_floor": {
            "mean": float(np.nanmean(floors)) if floors else float("nan"),
            "folds": [float(v) for v in floors],
        },
    }


# ── cluster bootstrap (vectorized: one GEMM of the moment matrix per chunk) ──


def _rng_for(label: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(label.encode())]))


def _codes(values: list[str]) -> tuple[np.ndarray, int]:
    uniq = sorted(set(values))
    lookup = {v: i for i, v in enumerate(uniq)}
    return np.asarray([lookup[v] for v in values], dtype=np.int64), len(uniq)


def _weighted_r_from_moments(s: np.ndarray, mx_col: int, xx_col: int, xy_col: int) -> np.ndarray:
    """Per-draw weighted Pearson r from the moment matrix s = W @ [1,y,y2,x,x2,xy,...]."""
    sw = s[:, 0]
    my = s[:, 1] / sw
    vy = s[:, 2] / sw - my * my
    mx = s[:, mx_col] / sw
    vx = s[:, xx_col] / sw - mx * mx
    cov = s[:, xy_col] / sw - mx * my
    denom = np.sqrt(np.clip(vx * vy, 0.0, None))
    return np.where((denom > 0) & (sw > 0), cov / denom, np.nan)


def _boot_paired(
    y: np.ndarray,
    pred_ctx: np.ndarray,
    pred_bare: np.ndarray | None,
    spec: tuple,
    n_draws: int,
    label: str,
    seed: int,
    chunk: int = 1000,
) -> dict[str, Any]:
    """Paired cluster bootstrap of Pearson r (ctx, bare, delta) under one spec.

    spec: ("rows",) | ("one", codes, n_clusters) | ("two", p_codes, n_p, q_codes, n_q)
    Weighted-moment form: all draws in a chunk share ONE (chunk, n) x (n, m)
    GEMM of the column matrix [1, y, y^2, ctx, ctx^2, ctx*y(, bare, bare^2, bare*y)].
    """
    n = y.shape[0]
    cols = [np.ones(n), y, y * y, pred_ctx, pred_ctx**2, pred_ctx * y]
    if pred_bare is not None:
        cols += [pred_bare, pred_bare**2, pred_bare * y]
    m = np.stack(cols, axis=1).astype(np.float64)
    rng = _rng_for(label, seed)
    r_ctx_all: list[np.ndarray] = []
    r_bare_all: list[np.ndarray] = []
    for start in range(0, n_draws, chunk):
        b = min(chunk, n_draws - start)
        if spec[0] == "rows":
            w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)
        elif spec[0] == "one":
            codes, n_cl = spec[1], spec[2]
            counts = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl), size=b)
            w = counts[:, codes].astype(np.float64)
        elif spec[0] == "two":
            p_codes, n_p, q_codes, n_q = spec[1], spec[2], spec[3], spec[4]
            cp = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=b)
            cq = rng.multinomial(n_q, np.full(n_q, 1.0 / n_q), size=b)
            w = (cp[:, p_codes] * cq[:, q_codes]).astype(np.float64)
        else:  # pragma: no cover - spec is internal
            raise ValueError(f"unknown bootstrap spec {spec[0]!r}")
        s = w @ m  # (b, n_cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_ctx_all.append(_weighted_r_from_moments(s, 3, 4, 5))
            if pred_bare is not None:
                r_bare_all.append(_weighted_r_from_moments(s, 6, 7, 8))
    r_ctx = np.concatenate(r_ctx_all)
    out: dict[str, Any] = {}

    def _ci(arr: np.ndarray) -> dict[str, Any]:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {"lo": None, "hi": None, "n_valid_replicates": 0}
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return {
            "lo": float(lo),
            "hi": float(hi),
            "n_valid_replicates": int(valid.size),
            "width": float(hi - lo),
        }

    out["ci_ctx"] = _ci(r_ctx)
    if pred_bare is not None:
        r_bare = np.concatenate(r_bare_all)
        delta = r_ctx - r_bare
        out["ci_bare"] = _ci(r_bare)
        out["ci_delta"] = _ci(delta)
    out["n_draws"] = int(n_draws)
    return out


# ── metrics ──


def _metrics_block(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y2 = y.reshape(-1, 1)
    rho = spearmanr(pred, y).statistic if y.size >= 3 else float("nan")
    recentered = pred - float(np.mean(pred)) + float(np.mean(y))
    return {
        "pearson_r": _pearson_or_nan(pred, y),
        "spearman_rho": float(rho),
        "r2_strict_zero_shot": _r2(y2, pred.reshape(-1, 1)),
        "r2_recentered": _r2(y2, recentered.reshape(-1, 1)),
        "n": int(y.size),
    }


def _gates(scores: np.ndarray) -> dict[str, Any]:
    """Parent §6 eligibility gates, verbatim semantics from _behavior_reads."""
    pos = int(np.sum(scores > 50.0))
    neg = int(scores.size - pos)
    std = float(np.std(scores)) if scores.size else float("nan")
    return {
        "n_scored": int(scores.size),
        "score_std": std,
        "n_positive": pos,
        "n_negative": neg,
        "estimable": bool(scores.size >= 5 and std >= 1.0 and pos >= 1 and neg >= 1),
    }


# ── #779 side loaders + the four-part alignment gate ──


def load_pass_b_l14(path: Path, layer: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Kill criterion 3(i) structural gate + L14 cx_last slice (fp64, bundle freed)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {"cx_last", "cx_mean", "v_x", "layers", "source", "metadata"}
    got_keys = set(bundle.keys())
    if got_keys != expected_keys:
        raise RuntimeError(f"pass_b structural gate FAIL: keys {sorted(got_keys)}")
    shapes = {}
    for k in ("cx_last", "cx_mean", "v_x"):
        t = bundle[k]
        shapes[k] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if tuple(t.shape) != (5000, 28, 3584) or t.dtype != torch.float32:
            raise RuntimeError(f"pass_b structural gate FAIL: {k} {t.shape} {t.dtype}")
    layers = list(bundle["layers"])
    if layers != list(range(28)):
        raise RuntimeError(f"pass_b structural gate FAIL: layers {layers[:5]}...")
    x = bundle["cx_last"][:, layer, :].to(torch.float64).numpy().copy()
    gate = {
        "pass": True,
        "keys": sorted(got_keys),
        "tensor_shapes": shapes,
        "layers_is_range28": True,
        "layer_indexing_note": (
            "both sides index the 28 transformer-block outputs 0-27 "
            "(pass_b layers==range(28); #1092 summaries are per-block L-shards)"
        ),
    }
    del bundle
    gc.collect()
    return x, gate


def load_lmsys_prompts_and_labels(
    rollouts_path: Path, labels_path: Path
) -> tuple[list[str], dict[str, list[float | None]], dict[str, Any]]:
    """Kill criteria 3(ii)+(iii): prompt sequence by index key + ctx0 spot assert.

    Content hygiene: prompts are real-corpus text — NEVER printed/logged; the
    spot-assert failure message carries only a sha256 digest.
    """
    blob = json.loads(rollouts_path.read_text())
    rollouts = blob["rollouts"]
    n = int(blob.get("n_contexts", len(rollouts)))
    if len(rollouts) != 5000 or n != 5000:
        raise RuntimeError(f"rollouts gate FAIL: {len(rollouts)} rows, n_contexts={n}")
    prompts = [str(rollouts[str(ci)]["prompt"]) for ci in range(5000)]
    norm0 = _norm_text(prompts[0]).rstrip(".?!,")
    ctx0_ok = norm0 == EXPECTED_CTX0_PROMPT
    if not ctx0_ok:
        raise RuntimeError(
            "ctx0 spot assert FAIL (stream drift): normalized row-0 prompt sha256_16="
            f"{hashlib.sha256(norm0.encode()).hexdigest()[:16]} != expected constant"
        )
    lb = json.loads(labels_path.read_text())["labels_per_trait"]
    labels = {t: e["labels"] for t, e in lb.items()}
    for t, arr in labels.items():
        if len(arr) != 5000:
            raise RuntimeError(f"labels gate FAIL: {t} has {len(arr)} rows")
    gate = {
        "prompt_sequence": {
            "pass": True,
            "n_keys": 5000,
            "source": "lmsys_g_rollouts.json rollouts[str(ci)] index keys 0..4999",
        },
        "ctx0_spot_assert": {"pass": ctx0_ok, "constant": "GC.EXPECTED_CTX0_PROMPT"},
    }
    return prompts, labels, gate


# ── #1092 side: fit-arm A unit + scored pools (banked join, verbatim) ──


def load_1092_unit(staged: dict[str, Any], layer: int) -> dict[str, Any]:
    rows = _jsonl(staged["corpus_dir"] / "manifest.jsonl")
    ctx_all, _paths = _load_summary(staged["summaries_dir"], CELL, "context_end", layer)
    n0 = min(ctx_all.shape[0], len(rows))
    base_rows = rows[:n0]
    idx = [i for i, row in enumerate(base_rows) if row.get("stratum") not in FITA_EXCLUDED_STRATA]
    idx_arr = np.asarray(idx, dtype=np.int64)
    unit_rows = [base_rows[i] for i in idx]
    x_ctx = ctx_all[idx_arr]
    x_bare = _bare_X_for_unit(staged["summaries_dir"], MODEL_TYPE, layer, unit_rows)
    del ctx_all
    gc.collect()
    unit_folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=6)
    judge_rows = _jsonl(staged["judge_scores"])
    row_pos = {str(r.get("row_id")): i for i, r in enumerate(unit_rows)}
    by_trait: dict[str, list[tuple[int, float]]] = {}
    for sr in judge_rows:
        if sr.get("cell_id") != CELL and sr.get("arm") != CELL:
            continue
        score = sr.get("score")
        row_id = str(sr.get("row_id"))
        if score is None or row_id not in row_pos:
            continue
        by_trait.setdefault(str(sr.get("trait")), []).append((row_pos[row_id], float(score)))
    _log(
        f"unit loaded: n_unit={len(unit_rows)} scored pools="
        f"{ {t: len(v) for t, v in sorted(by_trait.items())} }"
    )
    return {
        "rows": rows,
        "unit_rows": unit_rows,
        "unit_folds": unit_folds,
        "x_ctx": x_ctx,
        "x_bare": x_bare,
        "by_trait": by_trait,
    }


# ── pre-registered cross-corpus contamination control ──


def overlap_dedup(
    staged: dict[str, Any],
    unit: dict[str, Any],
    prompts: list[str],
) -> dict[str, Any]:
    """Normalized exact-match dedup between the 5,000 #779 prompts and the
    Direction-A training-pool query texts + #1092 prefix first-user-turns."""
    qstore = {
        str(r["query_id"]): str(r["text"])
        for r in _jsonl(staged["corpus_dir"] / "query_store.jsonl")
    }
    pstore = _jsonl(staged["corpus_dir"] / "prefix_store.jsonl")
    prefix_first_turn: dict[str, str] = {}
    for r in pstore:
        first = ""
        for turn in r.get("prefix_turns", []):
            if turn.get("role") == "user":
                first = str(turn.get("content", ""))
                break
        prefix_first_turn[str(r["prefix_id"])] = first
    # (a) every distinct query in cell_inst_own scored rows (union over traits).
    scored_positions = sorted({i for pairs in unit["by_trait"].values() for i, _s in pairs})
    pool_query_ids = {str(unit["unit_rows"][i].get("query_id")) for i in scored_positions}
    pool_query_norms = {_norm_text(qstore[q]) for q in pool_query_ids if q in qstore}
    # (b) prefix first-user-turns (all prefixes — superset of the pool's).
    prefix_norms = {_norm_text(t) for t in prefix_first_turn.values() if t}
    prompt_norms = [_norm_text(p) for p in prompts]
    lmsys_excluded = np.asarray(
        [(pn in pool_query_norms) or (pn in prefix_norms) for pn in prompt_norms], dtype=bool
    )
    lmsys_norm_set = set(prompt_norms)
    # reverse side: a #1092 unit row overlaps if its query text OR its prefix
    # first-user-turn normalizes into the LMSYS prompt set.
    row_overlap = np.zeros(len(unit["unit_rows"]), dtype=bool)
    for i, row in enumerate(unit["unit_rows"]):
        qn = _norm_text(qstore.get(str(row.get("query_id")), ""))
        pn = _norm_text(prefix_first_turn.get(str(row.get("prefix_id")), ""))
        if (qn and qn in lmsys_norm_set) or (pn and pn in lmsys_norm_set):
            row_overlap[i] = True
    report = {
        "normalization": "lowercase, whitespace-collapsed, exact match",
        "n_lmsys_prompts": len(prompts),
        "n_lmsys_excluded": int(lmsys_excluded.sum()),
        "lmsys_excluded_frac": float(lmsys_excluded.mean()),
        "n_unit_rows_overlapping": int(row_overlap.sum()),
        "unit_rows_overlap_frac": float(row_overlap.mean()),
        "n_pool_query_texts": len(pool_query_norms),
        "n_prefix_first_turns": len(prefix_norms),
        "flag_over_20pct": bool(
            lmsys_excluded.mean() > OVERLAP_FLAG_FRAC or row_overlap.mean() > OVERLAP_FLAG_FRAC
        ),
    }
    _log(
        f"overlap dedup: {report['n_lmsys_excluded']} LMSYS rows, "
        f"{report['n_unit_rows_overlapping']} #1092 unit rows excluded (eval sides only)"
    )
    return {"lmsys_excluded": lmsys_excluded, "row_overlap": row_overlap, "report": report}


# ── pass_a secondary surface assembly ──


def load_pass_a_surface(staged: dict[str, Any], trait: str, layer: int) -> dict[str, Any]:
    """Per-(condition, question) mean judge score + L14 cx_last rows.

    Score per question = mean over the 10 rollouts' N=5 judge means (None
    rollouts skipped). Conditions with per-condition score std < 1 are pruned
    (the parent's own prune rule; allowed deviation, reported).
    """
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_meta: list[dict[str, Any]] = []
    pruned: list[str] = []
    for cond in PASSA_CONDS:
        paths = staged["pass_a"][trait][cond]
        cx = torch.load(paths["cx"], map_location="cpu", weights_only=False)
        if list(cx["layers"]) != list(range(28)):
            raise RuntimeError(f"pass_a {trait}__{cond}: layers != range(28)")
        x = cx["cx_last"][:, layer, :].to(torch.float64).numpy()
        blob = json.loads(paths["json"].read_text())
        n_q, n_ro = int(blob["n_questions"]), int(blob["n_rollouts"])
        if x.shape[0] != n_q:
            raise RuntimeError(f"pass_a {trait}__{cond}: cx rows {x.shape[0]} != n_q {n_q}")
        cond_scores: list[tuple[int, float]] = []
        for qi in range(n_q):
            vals = [
                blob["judge_scores"].get(f"{trait}__{cond}__{qi:05d}__{ri:02d}")
                for ri in range(n_ro)
            ]
            valid = [v for v in vals if v is not None]
            if valid:
                cond_scores.append((qi, float(np.mean(valid))))
        cond_std = float(np.std([s for _q, s in cond_scores])) if cond_scores else float("nan")
        if not cond_scores or cond_std < 1.0:
            pruned.append(cond)
            continue
        for qi, s in cond_scores:
            rows_x.append(x[qi])
            rows_y.append(s)
            rows_meta.append({"cond_id": cond, "qi": qi})
    x_arr = np.stack(rows_x) if rows_x else np.zeros((0, 3584))
    y_arr = np.asarray(rows_y, dtype=np.float64)
    return {
        "X": x_arr,
        "y": y_arr,
        "meta": rows_meta,
        "pruned_conditions_std_lt_1": pruned,
        "gates": _gates(y_arr),
    }


# ── verdict logic (pre-registered, plan v7 §Success + kill criteria) ──


def _cell_verdict(delta_ci: dict, delta_point: float, ctx_ci: dict, ctx_point: float) -> dict:
    lo_d = delta_ci.get("lo")
    lo_c = ctx_ci.get("lo")
    positive = (
        delta_point >= DELTA_R_MARGIN
        and lo_d is not None
        and lo_d > 0.0
        and ctx_point > 0.0
        and lo_c is not None
        and lo_c > 0.0
    )
    return {
        "transfer_positive": bool(positive),
        "delta_r": delta_point,
        "delta_ci": [lo_d, delta_ci.get("hi")],
        "ctx_r": ctx_point,
        "ctx_ci": [lo_c, ctx_ci.get("hi")],
        "rule": "delta_r >= 0.05 AND delta CI excl. 0 AND ctx_r > 0 with CI excl. 0",
    }


def _widest(cis: dict[str, dict], key: str) -> tuple[str, dict]:
    best_name, best = None, None
    for name, block in cis.items():
        ci = block.get(key)
        if ci is None or ci.get("width") is None:
            continue
        if best is None or ci["width"] > best["width"]:
            best_name, best = name, ci
    if best is None:
        return "none", {"lo": None, "hi": None}
    return best_name, best


# ── main module run ──


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    companion_dir = out_dir / "companion"
    companion_dir.mkdir(parents=True, exist_ok=True)
    run_selftest("cpu")
    report: dict[str, Any] = {
        "metadata": {
            "script": "issue1092_transfer_probe.py",
            "followup_label": "cross-corpus-probe-transfer",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "args": {k: str(v) for k, v in vars(args).items()},
            "ridge_lambdas": [float(x) for x in RIDGE_LAMBDAS],
            "smoke_mode": bool(args.smoke),
        },
    }

    def checkpoint() -> None:
        _write_json_atomic(out_dir / "transfer_reads.json", report)

    # P0 stage
    staged = stage_inputs(args)
    report["metadata"]["input_shas"] = staged["shas"]
    checkpoint()

    # P1 pass_b structural gate + L14 slice (peak-RSS phase; freed inside).
    x_lmsys, structural_gate = load_pass_b_l14(staged["pass_b"], args.layer)
    # P2 prompts + labels (+ ctx0 spot assert)
    prompts, labels, seq_gate = load_lmsys_prompts_and_labels(staged["rollouts"], staged["labels"])
    report["alignment_gate"] = {"structural": structural_gate, **seq_gate}
    checkpoint()

    # P3 #1092 unit + scored pools
    unit = load_1092_unit(staged, args.layer)
    if not args.smoke and len(unit["unit_rows"]) != 19708:
        raise RuntimeError(f"fit-arm A unit size {len(unit['unit_rows'])} != 19708 (engine rule)")

    # P4 overlap dedup (eval sides only)
    dedup = overlap_dedup(staged, unit, prompts)
    report["overlap_dedup"] = dedup["report"]
    checkpoint()

    # banked values (ground truth read from the committed P6 merge, never memory)
    banked = json.loads(BANKED_B1_JSON.read_text())
    banked_entry = None
    for u in banked["units"]:
        p = u["provenance"]
        if (
            p.get("cell") == CELL
            and p.get("layer") == args.layer
            and p.get("fit_arm") == "A"
            and p.get("basis") == "ambient"
            and p.get("arm") == "context_end"
        ):
            banked_entry = u["behavior"]["traits"]
            break
    if banked_entry is None:
        raise RuntimeError("banked B1 unit (cell_inst_own, L14, fitA, ambient) not found")

    rng_smoke = np.random.default_rng(args.seed)
    traits_run = [t.strip() for t in args.traits.split(",") if t.strip()]
    report["traits"] = {}
    repro_all_pass = True
    prefix_ids_unit = [str(r.get("prefix_id")) for r in unit["unit_rows"]]
    query_ids_unit = [str(r.get("query_id")) for r in unit["unit_rows"]]
    stratum_unit = [str(r.get("stratum")) for r in unit["unit_rows"]]

    # per-trait state reused across phases
    state: dict[str, dict[str, Any]] = {}

    for trait in traits_run:
        tr: dict[str, Any] = {}
        pairs = unit["by_trait"].get(trait, [])
        if args.smoke and pairs and len(pairs) > args.smoke_rows:
            sel = rng_smoke.choice(len(pairs), size=args.smoke_rows, replace=False)
            pairs = [pairs[i] for i in sorted(sel)]
        pool_idx = np.asarray([i for i, _s in pairs], dtype=np.int64)
        pool_y = np.asarray([s for _i, s in pairs], dtype=np.float64)
        gates_1092 = _gates(pool_y)
        gates_1092["n_battery_rows_in_pool"] = int(
            sum(1 for i in pool_idx if stratum_unit[i] == "battery")
        )
        lab = labels.get(trait, [None] * 5000)
        lmsys_valid_idx = np.asarray(
            [ci for ci in range(5000) if lab[ci] is not None], dtype=np.int64
        )
        lmsys_y_full = np.asarray([lab[ci] for ci in lmsys_valid_idx], dtype=np.float64)
        gates_lmsys = _gates(lmsys_y_full)
        tr["gates"] = {"side_1092_cell_inst_own": gates_1092, "side_lmsys": gates_lmsys}
        state[trait] = {
            "pool_idx": pool_idx,
            "pool_y": pool_y,
            "gates_1092": gates_1092,
            "gates_lmsys": gates_lmsys,
            "lmsys_valid_idx": lmsys_valid_idx,
            "lmsys_y_full": lmsys_y_full,
        }
        report["traits"][trait] = tr
    checkpoint()

    # P5 repro gate + Direction A (train on #1092, eval on LMSYS + pass_a)
    for trait in traits_run:
        tr = report["traits"][trait]
        st = state[trait]
        if not st["gates_1092"]["estimable"]:
            tr["direction_A"] = {
                "status": (
                    "not estimable — train side (#1092 cell_inst_own): "
                    f"n_positive={st['gates_1092']['n_positive']}, "
                    f"score_std={st['gates_1092']['score_std']:.3f}"
                )
            }
            tr["repro_gate"] = {"status": "skipped — trait not estimable on the #1092 side"}
            continue
        pool_idx, pool_y = st["pool_idx"], st["pool_y"]
        rows_g = [unit["unit_rows"][i] for i in pool_idx]
        n_folds_local = max(2, min(len(unit["unit_folds"]), max(2, len(rows_g) // 2)))
        folds = _folds_from_manifest(
            rows_g, len(rows_g), group_key="prefix_id", n_folds=n_folds_local
        )
        # eval surfaces (identical features for ctx + bare probes on the #779 side)
        lmsys_eval_idx = np.asarray(
            [ci for ci in st["lmsys_valid_idx"] if not dedup["lmsys_excluded"][ci]],
            dtype=np.int64,
        )
        lmsys_eval_x = x_lmsys[lmsys_eval_idx]
        lmsys_eval_y = np.asarray([labels[trait][ci] for ci in lmsys_eval_idx], dtype=np.float64)
        passa = load_pass_a_surface(staged, trait, args.layer) if trait in VERDICT_TRAITS else None
        eval_surfaces = [lmsys_eval_x] + ([passa["X"]] if passa is not None else [])
        tr["repro_gate"] = {}
        tr["direction_A"] = {}
        for arm, x_unit in (("context_end", unit["x_ctx"]), ("c_q_bare", unit["x_bare"])):
            _log(f"P5 {trait}/{arm}: fold fits (n={pool_y.size}, folds={len(folds)})")
            ff = fold_fits(x_unit[pool_idx], pool_y, folds, eval_surfaces)
            st[f"ff_{arm}"] = ff
            recomputed_r2 = _r2(pool_y.reshape(-1, 1), ff["cv_pred"].reshape(-1, 1))
            banked_arm = (
                banked_entry.get(trait, {})
                .get("B1_by_arm_grain", {})
                .get(arm, {})
                .get("per_example", {})
            )
            banked_r2 = banked_arm.get("B1_direct_regression", {}).get("r2")
            banked_n = banked_arm.get("n_scored")
            gap = abs(recomputed_r2 - banked_r2) if banked_r2 is not None else None
            ok = (
                banked_r2 is not None
                and gap is not None
                and gap <= REPRO_TOL
                and int(pool_y.size) == int(banked_n or -1)
            )
            tr["repro_gate"][arm] = {
                "banked_r2": banked_r2,
                "recomputed_r2": recomputed_r2,
                "abs_gap": gap,
                "n_scored_banked": banked_n,
                "n_scored_recomputed": int(pool_y.size),
                "lam_press_folds": ff["lam_press_folds"],
                "train_mean_floor": ff["train_mean_floor"]["mean"],
                "pass": bool(ok),
                "enforced": not args.smoke,
            }
            if not args.smoke and not ok:
                repro_all_pass = False
        tr["repro_gate"]["row_id_set_check"] = (
            "banked per-row ids were not persisted (battery_scope_caveat); registered "
            "minimum applied — exact n_scored match + deterministic R2 within 1e-3"
        )
        if not args.smoke and not repro_all_pass:
            report["verdict"] = {
                "overall": "STOPPED — reproduction pre-gate failed (kill criterion 1)"
            }
            checkpoint()
            raise RuntimeError("reproduction pre-gate FAILED; no transfer read is interpreted")
        # within-#1092 ceiling (context arm CV r) for Direction B's tau
        ffc = st["ff_context_end"]
        st["ceiling_1092_ctx_r"] = _pearson_or_nan(ffc["cv_pred"], pool_y)
        # Direction A reads: probes = full-pool fits; both probes read the SAME
        # #779 features (prefix-less rows -> probe-weight transport test).
        dirA: dict[str, Any] = {"n_eval_lmsys": int(lmsys_eval_y.size)}
        preds = {}
        for arm in ("context_end", "c_q_bare"):
            ff = st[f"ff_{arm}"]
            preds[arm] = {
                "lmsys": ff["full_eval"][0],
                "passa": ff["full_eval"][1] if passa is not None else None,
                "lam_cv_idx": ff["lam_cv_idx"],
                "lam_press_full": ff["lam_press_full"],
                "fold_lmsys": ff["fold_eval"][0],
            }
        dirA["lambda"] = {
            "context_probe": {
                "cv_selected": float(RIDGE_LAMBDAS[preds["context_end"]["lam_cv_idx"]]),
                "press_selected": float(RIDGE_LAMBDAS[preds["context_end"]["lam_press_full"]]),
            },
            "bare_probe": {
                "cv_selected": float(RIDGE_LAMBDAS[preds["c_q_bare"]["lam_cv_idx"]]),
                "press_selected": float(RIDGE_LAMBDAS[preds["c_q_bare"]["lam_press_full"]]),
            },
        }
        dirA["lmsys"] = {
            "context_probe": _metrics_block(lmsys_eval_y, preds["context_end"]["lmsys"]),
            "bare_probe": _metrics_block(lmsys_eval_y, preds["c_q_bare"]["lmsys"]),
        }
        dirA["lmsys"]["delta_r"] = (
            dirA["lmsys"]["context_probe"]["pearson_r"] - dirA["lmsys"]["bare_probe"]["pearson_r"]
        )
        boot = _boot_paired(
            lmsys_eval_y,
            preds["context_end"]["lmsys"],
            preds["c_q_bare"]["lmsys"],
            ("rows",),
            args.n_draws,
            f"dirA::{trait}::rows",
            args.seed,
        )
        dirA["lmsys"]["bootstrap_row_cluster"] = boot
        dirA["lmsys"]["sensitivity_band_ctx_r"] = [
            float(
                min(
                    _pearson_or_nan(preds["context_end"]["fold_lmsys"][fi], lmsys_eval_y)
                    for fi in range(len(folds))
                )
            ),
            float(
                max(
                    _pearson_or_nan(preds["context_end"]["fold_lmsys"][fi], lmsys_eval_y)
                    for fi in range(len(folds))
                )
            ),
        ]
        st["dirA_lmsys_eval_idx"] = lmsys_eval_idx
        st["dirA_lmsys_eval_y"] = lmsys_eval_y
        st["dirA_preds"] = preds
        if passa is not None and passa["gates"]["estimable"]:
            cond_codes, n_conds = _codes([m["cond_id"] for m in passa["meta"]])
            pa: dict[str, Any] = {
                "context_probe": _metrics_block(passa["y"], preds["context_end"]["passa"]),
                "bare_probe": _metrics_block(passa["y"], preds["c_q_bare"]["passa"]),
                "gates": passa["gates"],
                "pruned_conditions_std_lt_1": passa["pruned_conditions_std_lt_1"],
                "n_conditions": int(n_conds),
                "non_gating_note": "elicited-trait distribution shift — reported, not gated",
            }
            pa["delta_r"] = pa["context_probe"]["pearson_r"] - pa["bare_probe"]["pearson_r"]
            pa["bootstrap_condition_cluster"] = _boot_paired(
                passa["y"],
                preds["context_end"]["passa"],
                preds["c_q_bare"]["passa"],
                ("one", cond_codes, n_conds),
                args.n_draws,
                f"dirA::{trait}::passa",
                args.seed,
            )
            dirA["passa"] = pa
            st["passa"] = passa
        elif passa is not None:
            dirA["passa"] = {
                "status": "not estimable — eval side (pass_a)",
                "gates": passa["gates"],
            }
        tr["direction_A"] = dirA
        checkpoint()
    _log("P5 done (repro gate + Direction A)")
    return _run_direction_b_and_finish(
        args,
        report,
        state,
        unit,
        x_lmsys,
        labels,
        dedup,
        out_dir,
        companion_dir,
        traits_run,
        prefix_ids_unit,
        query_ids_unit,
    )


def _run_direction_b_and_finish(  # noqa: C901
    args: argparse.Namespace,
    report: dict[str, Any],
    state: dict[str, dict[str, Any]],
    unit: dict[str, Any],
    x_lmsys: np.ndarray,
    labels: dict[str, list[float | None]],
    dedup: dict[str, Any],
    out_dir: Path,
    companion_dir: Path,
    traits_run: list[str],
    prefix_ids_unit: list[str],
    query_ids_unit: list[str],
) -> dict[str, Any]:
    def checkpoint() -> None:
        _write_json_atomic(out_dir / "transfer_reads.json", report)

    # P6 within-LMSYS ceilings + Direction B (train on LMSYS, eval on #1092)
    for trait in traits_run:
        tr = report["traits"][trait]
        st = state[trait]
        if not st["gates_lmsys"]["estimable"]:
            tr["lmsys_ceiling"] = {
                "status": (
                    "not estimable — train side (LMSYS): "
                    f"n_scored={st['gates_lmsys']['n_scored']}, "
                    f"score_std={st['gates_lmsys']['score_std']:.3f}"
                )
            }
            tr["direction_B"] = {"status": tr["lmsys_ceiling"]["status"]}
            continue
        valid_idx = st["lmsys_valid_idx"]
        y_tr = st["lmsys_y_full"]
        if args.smoke and valid_idx.size > 2 * args.smoke_rows:
            sel = np.sort(
                _rng_for(f"smoke::{trait}", args.seed).choice(
                    valid_idx.size, size=2 * args.smoke_rows, replace=False
                )
            )
            valid_idx = valid_idx[sel]
            y_tr = y_tr[sel]
        x_tr = x_lmsys[valid_idx]
        lmsys_rows = [{"prefix_id": f"lmsys_{int(ci):05d}"} for ci in valid_idx]
        folds_rows = _folds_from_manifest(
            lmsys_rows, len(lmsys_rows), group_key="prefix_id", n_folds=6
        )
        # eval side: #1092 scored rows minus overlap; context + bare features of
        # the SAME rows under the SAME fixed reader (paired comparison).
        pool_idx, pool_y = st["pool_idx"], st["pool_y"]
        keep = np.asarray([not dedup["row_overlap"][i] for i in pool_idx], dtype=bool)
        eval_rows_idx = pool_idx[keep]
        eval_y = pool_y[keep]
        eval_gates = _gates(eval_y)
        eval_surfaces = (
            [unit["x_ctx"][eval_rows_idx], unit["x_bare"][eval_rows_idx]]
            if eval_gates["estimable"]
            else []
        )
        _log(f"P6 {trait}: LMSYS fold fits (n={y_tr.size}) + ceiling")
        ff = fold_fits(x_tr, y_tr, folds_rows, eval_surfaces)
        st["ff_lmsys"] = ff
        cv_r = _pearson_or_nan(ff["cv_pred"], y_tr)
        ceil_boot = _boot_paired(
            y_tr,
            ff["cv_pred"],
            None,
            ("rows",),
            args.n_draws,
            f"ceiling::{trait}",
            args.seed,
        )
        tr["lmsys_ceiling"] = {
            "cv_r": cv_r,
            "cv_r2": _r2(y_tr.reshape(-1, 1), ff["cv_pred"].reshape(-1, 1)),
            "ci_row_cluster": [ceil_boot["ci_ctx"]["lo"], ceil_boot["ci_ctx"]["hi"]],
            "n_valid_replicates": ceil_boot["ci_ctx"]["n_valid_replicates"],
            "n": int(y_tr.size),
            "train_mean_floor": ff["train_mean_floor"]["mean"],
            "floor_rule": "hallucination CV r > 0 with row-cluster 95% CI excluding 0",
            "clears_floor": bool(
                cv_r > 0 and ceil_boot["ci_ctx"]["lo"] is not None and ceil_boot["ci_ctx"]["lo"] > 0
            ),
        }
        if not eval_gates["estimable"]:
            tr["direction_B"] = {
                "status": (
                    "not estimable — eval side (#1092 cell_inst_own): "
                    f"n_positive={eval_gates['n_positive']}, "
                    f"score_std={eval_gates['score_std']:.3f}"
                ),
                "eval_gates": eval_gates,
            }
            checkpoint()
            continue
        pred_ctx, pred_bare = ff["full_eval"][0], ff["full_eval"][1]
        dirB: dict[str, Any] = {
            "n_eval": int(eval_y.size),
            "n_excluded_overlap": int((~keep).sum()),
            "eval_gates": eval_gates,
            "context_states": _metrics_block(eval_y, pred_ctx),
            "bare_states": _metrics_block(eval_y, pred_bare),
            "lambda": {
                "cv_selected": float(RIDGE_LAMBDAS[ff["lam_cv_idx"]]),
                "press_selected": float(RIDGE_LAMBDAS[ff["lam_press_full"]]),
            },
        }
        dirB["delta_r"] = dirB["context_states"]["pearson_r"] - dirB["bare_states"]["pearson_r"]
        # three registered clusterings; verdict gates on the WIDEST CI per quantity.
        p_codes, n_p = _codes([prefix_ids_unit[i] for i in eval_rows_idx])
        q_codes, n_q = _codes([query_ids_unit[i] for i in eval_rows_idx])
        boots = {
            "two_way_crossed": _boot_paired(
                eval_y,
                pred_ctx,
                pred_bare,
                ("two", p_codes, n_p, q_codes, n_q),
                args.n_draws,
                f"dirB::{trait}::two",
                args.seed,
            ),
            "prefix_one_way": _boot_paired(
                eval_y,
                pred_ctx,
                pred_bare,
                ("one", p_codes, n_p),
                args.n_draws,
                f"dirB::{trait}::prefix",
                args.seed,
            ),
            "query_one_way": _boot_paired(
                eval_y,
                pred_ctx,
                pred_bare,
                ("one", q_codes, n_q),
                args.n_draws,
                f"dirB::{trait}::query",
                args.seed,
            ),
        }
        dirB["bootstrap"] = boots
        widest_delta_name, widest_delta = _widest(boots, "ci_delta")
        widest_ctx_name, widest_ctx = _widest(boots, "ci_ctx")
        dirB["widest_ci"] = {
            "delta": {"clustering": widest_delta_name, **widest_delta},
            "ctx": {"clustering": widest_ctx_name, **widest_ctx},
        }
        dirB["sensitivity_band_ctx_r"] = [
            float(
                min(
                    _pearson_or_nan(ff["fold_eval"][0][fi], eval_y) for fi in range(len(folds_rows))
                )
            ),
            float(
                max(
                    _pearson_or_nan(ff["fold_eval"][0][fi], eval_y) for fi in range(len(folds_rows))
                )
            ),
        ]
        ceiling_1092 = st.get("ceiling_1092_ctx_r")
        if ceiling_1092:
            dirB["transfer_fraction_tau"] = (
                dirB["context_states"]["pearson_r"] / ceiling_1092 if ceiling_1092 else None
            )
            dirB["within_1092_ceiling_cv_r"] = ceiling_1092
        st["dirB"] = {
            "eval_rows_idx": eval_rows_idx,
            "eval_y": eval_y,
            "pred_ctx": pred_ctx,
            "pred_bare": pred_bare,
        }
        tr["direction_B"] = dirB
        checkpoint()
    _log("P6 done (ceilings + Direction B)")

    # Direction A tau (needs the ceilings) + verdict cells
    halluc_ceiling = report["traits"].get("hallucination", {}).get("lmsys_ceiling", {})
    alignment_empirical_ok = bool(halluc_ceiling.get("clears_floor", False))
    report["alignment_gate"]["empirical_ceiling"] = {
        "pass": alignment_empirical_ok,
        "rule": "within-LMSYS hallucination CV r > 0 with row-cluster 95% CI excluding 0",
        "value_cv_r": halluc_ceiling.get("cv_r"),
        "ci": halluc_ceiling.get("ci_row_cluster"),
        "note": "PRECONDITION for any DOWNGRADE verdict (kill criterion 3(iv))",
    }
    per_cell: dict[str, Any] = {}
    for trait in VERDICT_TRAITS:
        tr = report["traits"].get(trait, {})
        st = state.get(trait, {})
        dirA = tr.get("direction_A", {})
        if isinstance(dirA, dict) and "lmsys" in dirA:
            ceiling = tr.get("lmsys_ceiling", {}).get("cv_r")
            if ceiling:
                dirA["lmsys"]["transfer_fraction_tau"] = (
                    dirA["lmsys"]["context_probe"]["pearson_r"] / ceiling
                )
                dirA["lmsys"]["within_lmsys_ceiling_cv_r"] = ceiling
            boot = dirA["lmsys"]["bootstrap_row_cluster"]
            cell = _cell_verdict(
                boot.get("ci_delta", {}),
                dirA["lmsys"]["delta_r"],
                boot.get("ci_ctx", {}),
                dirA["lmsys"]["context_probe"]["pearson_r"],
            )
            if not alignment_empirical_ok:
                cell["blocked"] = "alignment-suspect / signal-absent — indistinguishable"
                cell["transfer_positive"] = False
            per_cell[f"A_lmsys_{trait}"] = cell
        else:
            per_cell[f"A_lmsys_{trait}"] = {"status": "not estimable (see traits block)"}
        dirB = tr.get("direction_B", {})
        if isinstance(dirB, dict) and "widest_ci" in dirB:
            per_cell[f"B_{trait}"] = _cell_verdict(
                dirB["widest_ci"]["delta"],
                dirB["delta_r"],
                dirB["widest_ci"]["ctx"],
                dirB["context_states"]["pearson_r"],
            )
            per_cell[f"B_{trait}"]["widest_clustering"] = {
                "delta": dirB["widest_ci"]["delta"].get("clustering"),
                "ctx": dirB["widest_ci"]["ctx"].get("clustering"),
            }
        else:
            per_cell[f"B_{trait}"] = {"status": "not estimable (see traits block)"}
    a_positive = any(
        per_cell.get(f"A_lmsys_{t}", {}).get("transfer_positive") for t in VERDICT_TRAITS
    )
    b_positive = any(per_cell.get(f"B_{t}", {}).get("transfer_positive") for t in VERDICT_TRAITS)
    any_positive = a_positive or b_positive
    a_blocked = any("blocked" in per_cell.get(f"A_lmsys_{t}", {}) for t in VERDICT_TRAITS)
    if a_positive and b_positive:
        overall = "UPGRADE"
    elif not any_positive and alignment_empirical_ok and not a_blocked:
        overall = "DOWNGRADE"
    else:
        overall = "PARTIAL"
    report["verdict"] = {
        "per_cell": per_cell,
        "overall": overall,
        "downgrade_precondition_met": alignment_empirical_ok,
        "rules": {
            "transfer_positive": "delta_r >= 0.05, paired CI excl. 0 (dirB: widest of the "
            "three clusterings), AND ctx r > 0 with CI excl. 0",
            "upgrade": ">= 1 TRANSFER-POSITIVE trait in EACH direction",
            "downgrade": "ZERO transfer-positive cells AND the empirical alignment "
            "precondition holds (within-LMSYS hallucination ceiling clears its floor)",
            "partial": "anything between, or the alignment precondition failing",
        },
        "interpretation_guards": [
            "DOWNGRADE prose reads against the realized within-eval-corpus ceiling (tau), "
            "not absolute r (LMSYS labels = judge draws over 1 rollout/context — "
            "attenuation asymmetry vs #1092 per-row completion scoring)",
            "Direction A is a probe-weight transport test on prefix-less rows; an A-null "
            "is a weights-transport null, NEVER 'prefix signal failed to transfer'",
        ],
        "smoke_mode": bool(args.smoke),
    }
    report["report_contract_caveats"] = [
        "evil estimability is keyed per direction PAIR: LMSYS side PASSES its gates; the "
        "#1092 cell_inst_own side fails (0 positives) — never a per-corpus assertion",
        "label-attenuation asymmetry: LMSYS-side reads interpreted against the realized "
        "within-LMSYS ceiling (tau), never absolute-r expectations",
        "Direction-A framing guard (weights-transport, prefix-less rows)",
        "prefix-arm deviation: transfer covers context + bare only (no prefix capture "
        "exists on the #779 substrate); within-corpus prefix reads stay banked in the parent",
        "dedup overlap rate rides as a scope caveat (see overlap_dedup)",
    ]
    checkpoint()

    # companion dumps: per-row predictions (NO prompt text) + probe weights
    for trait in traits_run:
        st = state.get(trait, {})
        if "dirA_preds" in st:
            path = companion_dir / f"per_row_predictions_dirA_lmsys_{trait}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for j, ci in enumerate(st["dirA_lmsys_eval_idx"]):
                    f.write(
                        json.dumps(
                            {
                                "ci": int(ci),
                                "y": float(st["dirA_lmsys_eval_y"][j]),
                                "pred_context_probe": float(
                                    st["dirA_preds"]["context_end"]["lmsys"][j]
                                ),
                                "pred_bare_probe": float(st["dirA_preds"]["c_q_bare"]["lmsys"][j]),
                            }
                        )
                        + "\n"
                    )
            for arm, tag in (("context_end", "context"), ("c_q_bare", "bare")):
                ff = st[f"ff_{arm}"]
                np.savez_compressed(
                    companion_dir / f"probe_weights_dirA_{trait}_{tag}.npz",
                    w_full=ff["w_full"].astype(np.float32),
                    b_full=np.float64(ff["b_full"]),
                    lam_cv=np.float64(RIDGE_LAMBDAS[ff["lam_cv_idx"]]),
                    w_folds=ff["w_folds"].astype(np.float32),
                    b_folds=ff["b_folds"],
                )
        if "dirB" in st:
            db = st["dirB"]
            path = companion_dir / f"per_row_predictions_dirB_{trait}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for j, ui in enumerate(db["eval_rows_idx"]):
                    row = unit["unit_rows"][ui]
                    f.write(
                        json.dumps(
                            {
                                "row_id": str(row.get("row_id")),
                                "prefix_id": str(row.get("prefix_id")),
                                "query_id": str(row.get("query_id")),
                                "y": float(db["eval_y"][j]),
                                "pred_on_context_states": float(db["pred_ctx"][j]),
                                "pred_on_bare_states": float(db["pred_bare"][j]),
                            }
                        )
                        + "\n"
                    )
        if "ff_lmsys" in st:
            ff = st["ff_lmsys"]
            np.savez_compressed(
                companion_dir / f"probe_weights_dirB_{trait}.npz",
                w_full=ff["w_full"].astype(np.float32),
                b_full=np.float64(ff["b_full"]),
                lam_cv=np.float64(RIDGE_LAMBDAS[ff["lam_cv_idx"]]),
                w_folds=ff["w_folds"].astype(np.float32),
                b_folds=ff["b_folds"],
            )
        if "passa" in st:
            pa, preds = st["passa"], st["dirA_preds"]
            path = companion_dir / f"per_row_predictions_dirA_passa_{trait}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for j, meta in enumerate(pa["meta"]):
                    f.write(
                        json.dumps(
                            {
                                **meta,
                                "y": float(pa["y"][j]),
                                "pred_context_probe": float(preds["context_end"]["passa"][j]),
                                "pred_bare_probe": float(preds["c_q_bare"]["passa"][j]),
                            }
                        )
                        + "\n"
                    )
    _log("companion dumps written")

    # figures
    make_figures(args, report, state)
    checkpoint()

    # upload companions (persist by default; smoke uses the _smoke prefix)
    if args.skip_upload:
        report["upload"] = {"status": "skipped (--skip-upload)"}
    else:
        suffix = "_smoke" if args.smoke else ""
        prefix = f"{PREFIX_1092}/cross_corpus_transfer{suffix}"
        api = HfApi()
        assert_hub_dir_filecounts(companion_dir, prefix)
        api.upload_folder(
            folder_path=str(companion_dir),
            path_in_repo=prefix,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"issue1092 cross-corpus-probe-transfer companions{suffix}",
        )
        expected = {p.name for p in companion_dir.iterdir() if p.is_file()}
        missing = verify_repo_paths_uploaded(
            api,
            HF_REPO,
            sorted(f"{prefix}/{name}" for name in expected),
            path_in_repo=prefix,
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"upload verification FAILED: missing on hub: {missing}")
        report["upload"] = {
            "status": "uploaded+verified",
            "path_in_repo": prefix,
            "n_files": len(expected),
        }
        _log(f"companions uploaded + verified: {prefix} ({len(expected)} files)")
    checkpoint()
    _log(f"DONE — verdict: {report['verdict'].get('overall')}")
    return report


# ── figures (paper-plots style; plain-English labels, sidecars via savefig_paper) ──


def make_figures(  # noqa: C901
    args: argparse.Namespace, report: dict[str, Any], state: dict[str, dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = paper_palette(2)
    trait_label = {"hallucination": "Hallucination", "sycophancy": "Sycophancy"}

    # hero: paired context-vs-bare transfer bars with CIs per direction x trait
    bars: list[dict[str, Any]] = []
    for trait in VERDICT_TRAITS:
        tr = report["traits"].get(trait, {})
        dirA = tr.get("direction_A", {})
        if isinstance(dirA, dict) and "lmsys" in dirA:
            boot = dirA["lmsys"]["bootstrap_row_cluster"]
            bars.append(
                {
                    "group": f"Realistic → LMSYS\n({trait_label.get(trait, trait)})",
                    "ctx": dirA["lmsys"]["context_probe"]["pearson_r"],
                    "bare": dirA["lmsys"]["bare_probe"]["pearson_r"],
                    "ctx_ci": boot.get("ci_ctx", {}),
                    "bare_ci": boot.get("ci_bare", {}),
                }
            )
        dirB = tr.get("direction_B", {})
        if isinstance(dirB, dict) and "widest_ci" in dirB:
            widest = dirB["bootstrap"][
                {
                    "two_way_crossed": "two_way_crossed",
                    "prefix_one_way": "prefix_one_way",
                    "query_one_way": "query_one_way",
                }[dirB["widest_ci"]["ctx"].get("clustering", "two_way_crossed")]
            ]
            bars.append(
                {
                    "group": f"LMSYS → realistic\n({trait_label.get(trait, trait)})",
                    "ctx": dirB["context_states"]["pearson_r"],
                    "bare": dirB["bare_states"]["pearson_r"],
                    "ctx_ci": widest.get("ci_ctx", {}),
                    "bare_ci": widest.get("ci_bare", {}),
                }
            )
    if bars:
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        xs = np.arange(len(bars), dtype=float)
        width = 0.36

        def _err(vals: list[float], cis: list[dict], key_lo: str = "lo") -> np.ndarray:
            lo, hi = [], []
            for v, ci in zip(vals, cis, strict=True):
                clo, chi = ci.get("lo"), ci.get("hi")
                lo.append(max(0.0, v - clo) if clo is not None else 0.0)
                hi.append(max(0.0, chi - v) if chi is not None else 0.0)
            return np.asarray([lo, hi])

        ctx_vals = [b["ctx"] for b in bars]
        bare_vals = [b["bare"] for b in bars]
        ax.bar(
            xs - width / 2,
            ctx_vals,
            width,
            label="Context-state probe",
            color=colors[0],
            yerr=_err(ctx_vals, [b["ctx_ci"] for b in bars]),
            capsize=3,
        )
        ax.bar(
            xs + width / 2,
            bare_vals,
            width,
            label="Bare-query probe",
            color=colors[1],
            yerr=_err(bare_vals, [b["bare_ci"] for b in bars]),
            capsize=3,
        )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xs, [b["group"] for b in bars])
        ax.set_ylabel("Zero-shot transfer (Pearson r)")
        ax.set_title("Cross-corpus probe transfer: context probe vs bare-query control")
        ax.legend(frameon=False)
        savefig_paper(fig, "transfer_bars", dir=fig_dir)
        plt.close(fig)

    # per-row scatters behind each bar (Direction A on LMSYS, Direction B)
    def _scatter_panel(ax: Any, y: np.ndarray, pred: np.ndarray, title: str) -> None:
        ax.scatter(pred, y, s=4, alpha=0.15, color=colors[0], rasterized=True, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Predicted judge score")
        ax.set_ylabel("Actual judge score")

    panels_a = [(t, st) for t, st in state.items() if "dirA_preds" in st and t in VERDICT_TRAITS]
    if panels_a:
        fig, axes = plt.subplots(
            len(panels_a), 2, figsize=(8.0, 3.4 * len(panels_a)), squeeze=False
        )
        for r, (trait, st) in enumerate(panels_a):
            y = st["dirA_lmsys_eval_y"]
            _scatter_panel(
                axes[r][0],
                y,
                st["dirA_preds"]["context_end"]["lmsys"],
                f"Context-state probe on LMSYS ({trait_label.get(trait, trait)})",
            )
            _scatter_panel(
                axes[r][1],
                y,
                st["dirA_preds"]["c_q_bare"]["lmsys"],
                f"Bare-query probe on LMSYS ({trait_label.get(trait, trait)})",
            )
        fig.suptitle("Realistic-crossed-trained probes scored zero-shot on LMSYS rows", y=1.0)
        fig.tight_layout()
        savefig_paper(fig, "transfer_scatter_dirA_lmsys", dir=fig_dir, embed_data=False)
        plt.close(fig)

    panels_b = [(t, st) for t, st in state.items() if "dirB" in st and t in VERDICT_TRAITS]
    if panels_b:
        fig, axes = plt.subplots(
            len(panels_b), 2, figsize=(8.0, 3.4 * len(panels_b)), squeeze=False
        )
        for r, (trait, st) in enumerate(panels_b):
            db = st["dirB"]
            _scatter_panel(
                axes[r][0],
                db["eval_y"],
                db["pred_ctx"],
                f"LMSYS probe on context states ({trait_label.get(trait, trait)})",
            )
            _scatter_panel(
                axes[r][1],
                db["eval_y"],
                db["pred_bare"],
                f"LMSYS probe on bare-query states ({trait_label.get(trait, trait)})",
            )
        fig.suptitle("LMSYS-trained probe scored zero-shot on realistic-crossed rows", y=1.0)
        fig.tight_layout()
        savefig_paper(fig, "transfer_scatter_dirB", dir=fig_dir, embed_data=False)
        plt.close(fig)

    # pass_a secondary surface bars (non-gating)
    pa_bars = []
    for trait in VERDICT_TRAITS:
        pa = report["traits"].get(trait, {}).get("direction_A", {}).get("passa")
        if isinstance(pa, dict) and "context_probe" in pa:
            boot = pa["bootstrap_condition_cluster"]
            pa_bars.append(
                {
                    "group": trait_label.get(trait, trait),
                    "ctx": pa["context_probe"]["pearson_r"],
                    "bare": pa["bare_probe"]["pearson_r"],
                    "ctx_ci": boot.get("ci_ctx", {}),
                    "bare_ci": boot.get("ci_bare", {}),
                }
            )
    if pa_bars:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        xs = np.arange(len(pa_bars), dtype=float)
        width = 0.36

        def _err2(vals: list[float], cis: list[dict]) -> np.ndarray:
            lo, hi = [], []
            for v, ci in zip(vals, cis, strict=True):
                clo, chi = ci.get("lo"), ci.get("hi")
                lo.append(max(0.0, v - clo) if clo is not None else 0.0)
                hi.append(max(0.0, chi - v) if chi is not None else 0.0)
            return np.asarray([lo, hi])

        ctx_vals = [b["ctx"] for b in pa_bars]
        bare_vals = [b["bare"] for b in pa_bars]
        ax.bar(
            xs - width / 2,
            ctx_vals,
            width,
            label="Context-state probe",
            color=colors[0],
            yerr=_err2(ctx_vals, [b["ctx_ci"] for b in pa_bars]),
            capsize=3,
        )
        ax.bar(
            xs + width / 2,
            bare_vals,
            width,
            label="Bare-query probe",
            color=colors[1],
            yerr=_err2(bare_vals, [b["bare_ci"] for b in pa_bars]),
            capsize=3,
        )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xs, [b["group"] for b in pa_bars])
        ax.set_ylabel("Zero-shot transfer (Pearson r)")
        ax.set_title("Secondary surface: persona-condition contexts (non-gating)")
        ax.legend(frameon=False)
        savefig_paper(fig, "transfer_passa_bars", dir=fig_dir)
        plt.close(fig)
    _log("figures written")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #1092 cross-corpus supervised-probe transfer (0-GPU module)."
    )
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-rev-1092", default="e590170619e7691c1a95c7b1bb20bda5fd4065ad")
    p.add_argument("--hf-rev-779-passb", default="037fcbb210bc52c459959b0746cc268fe08bae96")
    p.add_argument("--hf-rev-779-labels", default="5aa6de1b97895cf8883c44165fa8835ff73e9e93")
    p.add_argument("--out", default="eval_results/issue_1092/cross-corpus-probe-transfer/")
    p.add_argument("--figures", default="figures/issue_1092/")
    p.add_argument("--traits", default="hallucination,sycophancy,evil")
    p.add_argument(
        "--n-draws",
        type=int,
        default=10000,
        help="bootstrap draws per clustering variant (plan floor 1000 production)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="same phases on subsampled pools; repro-gate values reported, "
        "not enforced; upload prefix suffixed _smoke",
    )
    p.add_argument("--smoke-rows", type=int, default=200)
    p.add_argument("--skip-upload", action="store_true")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    report = run(args)
    verdict = report.get("verdict", {}).get("overall")
    print(f"[phase=done] cross-corpus-probe-transfer verdict={verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
