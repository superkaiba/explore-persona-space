#!/usr/bin/env python3
"""Issue #1092 0-GPU bridge refits for #923, #813, and #779 artifacts.

Production stages only scoped Hub paths via list_repo_tree + per-file
hf_hub_download. Tiny-real mode builds local parent-artifact fixtures and runs
the same consumers without touching the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue813_rank_spectrum import (  # noqa: E402
    _fit_pieces,
    _gcv_lambda,
    _sigma2,
    _spectrum_stats,
    _standardize,
)
from issue923_fit_decomposition import press_fit_predict, run_selftest  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
MODEL_REPO = "superkaiba1/explore-persona-space"
REV_923 = "77d04e45"
REV_813 = "b0d30307c1"
REV_779 = "037fcbb"
REV_779_PASS2 = "5aa6de1b"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def list_scoped(
    repo_id: str, revision: str, path_in_repo: str, *, repo_type: str = "dataset"
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        recursive=True,
    )
    files = [e.path for e in entries if getattr(e, "size", None) is not None]
    if not files:
        raise FileNotFoundError(f"no files listed under {repo_id}@{revision}:{path_in_repo}")
    return files


def download_scoped(
    repo_id: str,
    revision: str,
    relpaths: list[str],
    dest: Path,
    *,
    repo_type: str = "dataset",
) -> list[Path]:
    from huggingface_hub import hf_hub_download

    out: list[Path] = []
    for rel in relpaths:
        local = hf_hub_download(
            repo_id=repo_id, repo_type=repo_type, revision=revision, filename=rel
        )
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        target.symlink_to(Path(local).resolve())
        out.append(target)
    return out


def _spectrum(X: np.ndarray, Y: np.ndarray) -> dict:
    Xn, _mu, _sd = _standardize(X)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double()
    pieces = _fit_pieces(Xn, Yt)
    e = pieces["e"].detach().cpu().numpy()
    diag = torch.diag(pieces["W_yy"]).detach().cpu().numpy()
    lam = _gcv_lambda(e, diag, X.shape[0])
    sig = torch.sqrt(_sigma2(pieces["e"], pieces["W_yy"], lam)).detach().cpu().numpy()
    return {"lambda_gcv": float(lam), "stats": _spectrum_stats(sig)}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _fit_once(X: np.ndarray, Y: np.ndarray) -> dict:
    n = X.shape[0]
    split = max(2, int(0.67 * n))
    split = min(split, n - 1)
    res = press_fit_predict(
        torch.from_numpy(X[:split]).double(),
        torch.from_numpy(Y[:split]).double(),
        torch.from_numpy(X[split:]).double(),
        standardize=True,
    )
    pred = res["pred"].detach().cpu().numpy()
    return {
        "n_train": split,
        "n_test": n - split,
        "r2": _r2(Y[split:], pred),
        "lam_idx": res["lam_idx"],
    }


def _anova(prefix_ids: np.ndarray, query_ids: np.ndarray, Y: np.ndarray) -> dict:
    yc = Y - Y.mean(axis=0, keepdims=True)
    f = np.zeros_like(yc)
    g = np.zeros_like(yc)
    for pid in sorted(set(prefix_ids.tolist())):
        f[prefix_ids == pid] = yc[prefix_ids == pid].mean(axis=0, keepdims=True)
    for qid in sorted(set(query_ids.tolist())):
        g[query_ids == qid] = yc[query_ids == qid].mean(axis=0, keepdims=True)
    i = yc - f - g
    ss = float((yc * yc).sum())
    return {
        "prefix": float((f * f).sum() / ss) if ss else float("nan"),
        "query": float((g * g).sum() / ss) if ss else float("nan"),
        "interaction": float((i * i).sum() / ss) if ss else float("nan"),
    }


def make_tiny_fixtures(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(1092)
    root.mkdir(parents=True, exist_ok=True)
    n_ctx, n_q, h = 3, 4, 8
    prefix = np.repeat(np.arange(n_ctx), n_q)
    query = np.tile(np.arange(n_q), n_ctx)
    X = rng.standard_normal((n_ctx * n_q, h))
    Y = rng.standard_normal((n_ctx * n_q, h))
    p923 = root / "issue923_uc_tiny.npz"
    np.savez(p923, X=X, Y=Y, prefix_ids=prefix, query_ids=query)
    p813 = root / "issue813_wm0_tiny.npz"
    np.savez(p813, W_M0=rng.standard_normal((h, h)), X=X, Y=Y)
    p779 = root / "issue779_pass_b_tiny.pt"
    torch.save({"X": torch.from_numpy(X).float(), "Y": torch.from_numpy(Y).float()}, p779)
    return {"923": p923, "813": p813, "779": p779}


def refit_923(path: Path) -> dict:
    data = np.load(path)
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    prefix_ids = np.asarray(data["prefix_ids"])
    query_ids = np.asarray(data["query_ids"])
    return {
        "source": str(path),
        "sha16": _sha(path),
        "scope": "UltraChat primary; Betley sensitivity uses same function when staged",
        "fit": _fit_once(X, Y),
        "spectrum": _spectrum(X, Y),
        "anova_shares": _anova(prefix_ids, query_ids, Y),
    }


def refit_813(path: Path) -> dict:
    data = np.load(path)
    keys = set(data.files)
    if "W_Mplus" in keys:
        raise ValueError(f"{path} contains W_Mplus; #813 bridge must consume W_M0 only")
    if "W_M0" not in keys:
        raise KeyError(f"{path} missing W_M0")
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    W = np.asarray(data["W_M0"], dtype=np.float64)
    return {
        "source": str(path),
        "sha16": _sha(path),
        "consumed_weight": "W_M0",
        "w_shape": list(W.shape),
        "fit": _fit_once(X, Y),
        "spectrum": _spectrum(X, Y),
    }


def refit_779(path: Path, fallback_paths: list[Path] | None = None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(blob.keys()) if isinstance(blob, dict) else []
    x_key = next(
        (k for k in ("X", "cx_last", "context_vectors", "train_context_vectors") if k in blob), None
    )
    y_key = next(
        (k for k in ("Y", "v_x", "answer_summaries", "train_answer_vectors") if k in blob), None
    )
    fallback_used = None
    if x_key is None:
        raise KeyError(f"#779 pass_b key set has no X/context side: {keys}")
    if y_key is None and fallback_paths:
        for fp in fallback_paths:
            fb = torch.load(fp, map_location="cpu", weights_only=False)
            y_key = next(
                (k for k in ("Y", "v_x", "answer_summaries", "train_answer_vectors") if k in fb),
                None,
            )
            if y_key is not None:
                blob = {x_key: blob[x_key], y_key: fb[y_key]}
                fallback_used = str(fp)
                break
    if y_key is None:
        raise KeyError(f"#779 pass_b key set has no Y/answer side: {keys}; fallback unavailable")
    X = np.asarray(blob[x_key], dtype=np.float64)
    Y = np.asarray(blob[y_key], dtype=np.float64)
    if X.ndim == 3:
        X = X[:, min(14, X.shape[1] - 1), :]
    if Y.ndim == 3:
        Y = Y[:, min(14, Y.shape[1] - 1), :]
    return {
        "source": str(path),
        "sha16": _sha(path),
        "key_set": keys,
        "x_key": x_key,
        "y_key": y_key,
        "fallback_used": fallback_used,
        "scope": "map-refit + layer-sweep + bare-query-map comparability only; f/g/i undefined",
        "fit": _fit_once(X, Y),
        "spectrum": _spectrum(X, Y),
    }


def stage_production(root: Path) -> dict[str, list[Path]]:
    staged: dict[str, list[Path]] = {}
    specs = [
        ("923", DATA_REPO, REV_923, "issue923_ctx_query_decomposition/analysis_tensors", "dataset"),
        ("813", DATA_REPO, REV_813, "issue813_mapchange_substrate/reduced", "dataset"),
        ("779", DATA_REPO, REV_779, "issue779_monitoring/analysis_tensors/pass_b", "dataset"),
        (
            "779_fallback",
            DATA_REPO,
            REV_779,
            "issue779_monitoring/analysis_tensors/pass_a",
            "dataset",
        ),
    ]
    for name, repo, rev, path, repo_type in specs:
        files = list_scoped(repo, rev, path, repo_type=repo_type)
        wanted = [f for f in files if f.endswith((".npz", ".pt"))]
        if name == "813":
            wm0 = [f for f in wanted if "W_M0" in f or "map" in Path(f).name.lower()]
            wanted = wm0 or wanted
        if not wanted:
            raise FileNotFoundError(f"{name}: scoped listing had no tensor files under {path}")
        staged[name] = download_scoped(
            repo, rev, wanted, root / "staged" / name, repo_type=repo_type
        )
    return staged


def _refit_many(paths: list[Path], fn, *, fallback_paths: list[Path] | None = None) -> dict:
    if not paths:
        raise ValueError("refit_many received no paths")
    items = []
    for path in paths:
        if fallback_paths is None:
            items.append(fn(path))
        else:
            items.append(fn(path, fallback_paths=fallback_paths))
    r2 = [item.get("fit", {}).get("r2") for item in items]
    return {
        "n_files": len(items),
        "files": items,
        "r2_mean": float(np.nanmean(r2)) if r2 else float("nan"),
    }


def run(args: argparse.Namespace) -> dict:
    run_selftest("cpu")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tiny_real:
        fixtures = make_tiny_fixtures(out_dir / "fixtures")
        paths = {"923": [fixtures["923"]], "813": [fixtures["813"]], "779": [fixtures["779"]]}
    else:
        paths = stage_production(out_dir)

    result = {
        "phase": "bridge_refit",
        "tiny_real": args.tiny_real,
        "staging_policy": "scoped list_repo_tree + per-file hf_hub_download; no snapshot_download",
        "issue923": _refit_many(paths["923"], refit_923),
        "issue813": _refit_many(paths["813"], refit_813),
        "issue779": _refit_many(
            paths["779"],
            refit_779,
            fallback_paths=paths.get("779_fallback"),
        ),
    }
    path = out_dir / "bridge_refit_summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(
        f"[bridge-refit] artifact digest: 923_files={result['issue923']['n_files']} "
        f"813_files={result['issue813']['n_files']} 779_files={result['issue779']['n_files']} "
        f"path={path}"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tiny-real", action="store_true")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
