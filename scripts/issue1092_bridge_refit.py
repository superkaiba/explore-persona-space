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
import re
import sys
import time
from pathlib import Path

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
    # HUB_VERIFY_RETRY_EXEMPT: issue-1092 driver; scoped listing with orchestration-layer retry
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


L923_HEADLINE = 18  # #923's published-shares layer (sanity anchor)
L1092_HEADLINE = 14  # this plan's frozen headline layer
UC_REFERENCE_SHARES_PCA48_L18 = {"query": 0.837, "prefix": 0.078, "interaction": 0.086}


def make_tiny_fixtures(root: Path) -> dict[str, Path]:
    """Tiny-real fixtures mirroring the REAL parent-artifact schemas + tree layout.

    The first fixture generation faked a generic X/Y/prefix_ids schema; the real
    artifacts use their own key layouts (923: {tensors:{vbar,valid},meta:{rows}}
    + capture flast shards; 813: c_C_base/v_A_base + *_trained twins in one npz;
    779: cx_last/cx_mean/v_x), so the tiny-real smoke passed while every
    production consume path was wrong (2026-07-10 production crash,
    KeyError 'X'). These fixtures now mirror key names, arm structure, AND the
    directory layout the discovery globs walk, so the smoke exercises the
    production consume path end to end.
    """
    rng = np.random.default_rng(1092)
    n_ctx, n_q, h, n_layers = 3, 4, 8, 28
    n = n_ctx * n_q
    prefix = np.repeat(np.arange(n_ctx), n_q)
    query = np.tile(np.arange(n_q), n_ctx)
    rows = [{"ctx_id": f"ctx{c}", "q_idx": int(q)} for c, q in zip(prefix, query, strict=True)]

    def grid(scale_f: float = 1.0, scale_g: float = 1.0) -> np.ndarray:
        f = rng.standard_normal((n_ctx, 1, h))[prefix][:, 0] * scale_f
        g = rng.standard_normal((n_q, 1, h))[query][:, 0] * scale_g
        noise = 0.1 * rng.standard_normal((n, h))
        return f + g + noise

    # ---- 923: analysis_tensors/{reduce,capture} with {tensors, meta} pt blobs
    root_923 = root / "923" / "analysis_tensors"
    (root_923 / "reduce").mkdir(parents=True, exist_ok=True)
    (root_923 / "capture").mkdir(parents=True, exist_ok=True)
    vbar = np.stack([grid(0.4, 1.2) for _ in range(n_layers)], axis=1)  # (n, 28, h)
    torch.save(
        {
            "tensors": {
                "vbar": torch.from_numpy(vbar).half(),
                "valid": torch.ones(n, dtype=torch.bool),
            },
            "meta": {"rows": rows, "revision": "tiny-fixture"},
        },
        root_923 / "reduce" / "vbar_store_uc.pt",
    )
    flast = np.stack([grid() for _ in range(n_layers)], axis=1)
    torch.save(
        {
            "tensors": {"flast": torch.from_numpy(flast).half()},
            "meta": {"rows": rows, "stage": "ffull", "shard": "0of1"},
        },
        root_923 / "capture" / "ffull_uc48_shard0of1.pt",
    )

    # ---- 813: reduced/<behavior>/<substrate>/{summary.npz, per_question_L14.npz}
    pair_dir = root / "813" / "reduced" / "em" / "generic"
    pair_dir.mkdir(parents=True, exist_ok=True)
    ctx_ids = np.asarray([f"ctx{c}" for c in range(n_ctx)], dtype=object)
    np.savez(
        pair_dir / "per_question_L14.npz",
        c_C_base=grid().astype(np.float32),
        c_C_trained=grid().astype(np.float32),
        v_A_base=grid(0.4, 1.2).astype(np.float32),
        v_A_trained=grid(0.4, 1.2).astype(np.float32),
        row_context_index=prefix.astype(np.int64),
        row_question_index=query.astype(np.int64),
        context_ids=ctx_ids,
        families=ctx_ids,
        headline_layer=np.int64(L1092_HEADLINE),
        behavior="em",
        substrate="generic",
        git_sha="tiny-fixture",
    )
    summary_shape = (n_ctx, n_layers, h)
    np.savez(
        pair_dir / "summary.npz",
        c_C_base=rng.standard_normal(summary_shape).astype(np.float32),
        c_C_trained=rng.standard_normal(summary_shape).astype(np.float32),
        v_A_base=rng.standard_normal(summary_shape).astype(np.float32),
        v_A_trained=rng.standard_normal(summary_shape).astype(np.float32),
        context_ids=ctx_ids,
        families=ctx_ids,
        n_contexts=np.int64(n_ctx),
        n_questions=np.int64(n_q),
        behavior="em",
        substrate="generic",
        layers=np.arange(n_layers, dtype=np.int64),
        git_sha="tiny-fixture",
        generated_at="tiny",
    )

    # ---- 779: pass_b/train_context_vectors.pt with cx_last/cx_mean/v_x
    pass_b = root / "779" / "pass_b"
    pass_b.mkdir(parents=True, exist_ok=True)
    cx = np.stack([grid() for _ in range(n_layers)], axis=1)
    torch.save(
        {
            "cx_last": torch.from_numpy(cx).half(),
            "cx_mean": torch.from_numpy(cx * 0.9).half(),
            "v_x": torch.from_numpy(vbar).half(),
            "layers": list(range(n_layers)),
            "source": "tiny-fixture",
            "metadata": {},
        },
        pass_b / "train_context_vectors.pt",
    )
    return {"923": root / "923", "813": root / "813", "779": root / "779"}


def _load_pt_blob(path: Path) -> tuple[dict, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("tensors", {}), blob.get("meta", {})


def _pca_project(Y: np.ndarray, k: int = 48) -> np.ndarray:
    """Project centered Y onto its own top-k PCs (the #923 pca48 target basis)."""
    yc = Y - Y.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(yc, full_matrices=False)
    return yc @ vh[: min(k, vh.shape[0])].T


SPLIT_RECIPE = "grouped-shuffled-seed0-v2"


def _fit_once_grouped(X: np.ndarray, Y: np.ndarray, group_ids: np.ndarray) -> dict:
    """Held-out fit: test block = a seed-0 SHUFFLED third of GROUPS (prefix/context
    identity), mirroring the plan's grouped-fold discipline at bridge scale.

    The groups are shuffled with a fixed rng before the third is taken: a
    lexicographic tail clusters related contexts (same family prefix) into the
    test block and biases R2 hard negative (measured on the real #923 uc48 grid:
    tail split -1.87 vs shuffled +0.30 at L18, with the tail split's train-mean
    floor itself at -0.09). Train-mean floor co-reported per the identity-ladder
    discipline.
    """
    uniq = list(dict.fromkeys(group_ids.tolist()))
    shuffled = list(np.random.default_rng(0).permutation(np.asarray(uniq, dtype=object)))
    n_test_groups = max(1, len(shuffled) // 3)
    test_groups = set(shuffled[-n_test_groups:])
    test_mask = np.asarray([g in test_groups for g in group_ids.tolist()], dtype=bool)
    if test_mask.all() or not test_mask.any():
        raise ValueError("grouped split degenerate: all or no rows in test block")
    res = press_fit_predict(
        torch.from_numpy(X[~test_mask]).double(),
        torch.from_numpy(Y[~test_mask]).double(),
        torch.from_numpy(X[test_mask]).double(),
        standardize=True,
    )
    pred = res["pred"].detach().cpu().numpy()
    floor = _r2(
        Y[test_mask],
        np.broadcast_to(Y[~test_mask].mean(axis=0, keepdims=True), Y[test_mask].shape),
    )
    return {
        "n_train": int((~test_mask).sum()),
        "n_test": int(test_mask.sum()),
        "n_test_groups": n_test_groups,
        "split": SPLIT_RECIPE,
        "r2": _r2(Y[test_mask], pred),
        "train_mean_floor_r2": floor,
        "lam_idx": res["lam_idx"],
    }


def _item_cache_load(out_dir: Path) -> dict[str, dict]:
    cache_path = out_dir / "bridge_refit_items.jsonl"
    items: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items[row["item_key"]] = row["item"]
    return items


def _item_cache_append(out_dir: Path, item_key: str, item: dict) -> None:
    cache_path = out_dir / "bridge_refit_items.jsonl"
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"item_key": item_key, "item": item}, allow_nan=True) + "\n")


def refit_923_substrate(
    at_root: Path, substrate: str, layer_subset: list[int] | None = None
) -> dict:
    """f/g/i + layer-sweep re-fit on one #923 grid (vbar store x ffull capture).

    Real schema: reduce/vbar_store_<uc|betley>.pt {tensors:{vbar (n,28,h) fp16,
    valid (n,) bool}, meta:{rows:[{ctx_id,q_idx}]}} and
    capture/ffull_<uc48|betley>_shard*of*.pt {tensors:{flast}, meta:{rows}}.
    Join on (ctx_id, q_idx); rows failing the vbar valid mask are dropped.
    """
    store_stem = "uc" if substrate == "uc48" else substrate
    vbar_path = at_root / "reduce" / f"vbar_store_{store_stem}.pt"
    shards = sorted((at_root / "capture").glob(f"ffull_{substrate}_shard*of*.pt"))
    if not vbar_path.exists() or not shards:
        raise FileNotFoundError(f"923 {substrate}: missing {vbar_path} or ffull shards")
    tensors, meta = _load_pt_blob(vbar_path)
    vbar = tensors["vbar"]
    valid = tensors["valid"].numpy().astype(bool)
    y_index = {(str(r["ctx_id"]), int(r["q_idx"])): i for i, r in enumerate(meta["rows"])}
    x_parts, x_rows = [], []
    for shard in shards:
        t, m = _load_pt_blob(shard)
        x_parts.append(t["flast"])
        x_rows.extend(m["rows"])
    X_all = torch.cat(x_parts, dim=0)
    n_layers = int(vbar.shape[1])
    if X_all.shape[1] != n_layers:
        raise ValueError(f"923 {substrate}: layer axes differ {X_all.shape} vs {vbar.shape}")
    x_idx, y_idx, ctx_ids, q_ids = [], [], [], []
    seen: set[tuple[str, int]] = set()
    for xi, r in enumerate(x_rows):
        key = (str(r["ctx_id"]), int(r["q_idx"]))
        yi = y_index.get(key)
        if yi is None or not valid[yi] or key in seen:
            continue
        seen.add(key)
        x_idx.append(xi)
        y_idx.append(yi)
        ctx_ids.append(key[0])
        q_ids.append(key[1])
    if len(x_idx) < 6:
        raise ValueError(f"923 {substrate}: only {len(x_idx)} joined rows")
    x_idx_arr = np.asarray(x_idx, dtype=np.int64)
    y_idx_arr = np.asarray(y_idx, dtype=np.int64)
    ctx_arr = np.asarray(ctx_ids)
    q_arr = np.asarray(q_ids)
    per_layer: dict[str, dict] = {}
    for layer in layer_subset if layer_subset is not None else range(n_layers):
        X = X_all[x_idx_arr, layer].numpy().astype(np.float64)
        Y = vbar[y_idx_arr, layer].numpy().astype(np.float64)
        entry: dict = {"fit": _fit_once_grouped(X, Y, ctx_arr)}
        if layer in (L923_HEADLINE, L1092_HEADLINE):
            entry["spectrum"] = _spectrum(X, Y)
            entry["anova_shares_ambient"] = _anova(ctx_arr, q_arr, Y)
            entry["anova_shares_pca48"] = _anova(ctx_arr, q_arr, _pca_project(Y))
        per_layer[f"L{layer:02d}"] = entry
    item = {
        "substrate": substrate,
        "source": str(vbar_path),
        "sha16": _sha(vbar_path),
        "n_rows_joined": len(x_idx),
        "n_dropped_invalid": int((~valid).sum()),
        "consumed": {"x": "capture ffull flast", "y": "reduce vbar_store vbar"},
        "headline_r2": per_layer[f"L{L1092_HEADLINE:02d}"]["fit"]["r2"],
        "per_layer": per_layer,
    }
    if substrate == "uc48" and len(x_idx) >= 1000:
        got = per_layer[f"L{L923_HEADLINE:02d}"]["anova_shares_pca48"]
        diffs = {k: abs(got[k] - v) for k, v in UC_REFERENCE_SHARES_PCA48_L18.items()}
        item["reference_check_pca48_L18"] = {
            "published": UC_REFERENCE_SHARES_PCA48_L18,
            "refit": got,
            "abs_diff": diffs,
        }
        if got["query"] < max(got["prefix"], got["interaction"]) or diffs["query"] > 0.15:
            raise RuntimeError(
                f"923 uc48 consume-mapping sanity FAILED: pca48 L18 shares {got} vs "
                f"published {UC_REFERENCE_SHARES_PCA48_L18} — axis/join mapping suspect"
            )
    return item


def refit_813_pair(pair_dir: Path, layer_subset: list[int] | None = None) -> dict:
    """Rank/grain re-fit on one #813 reduced (behavior, substrate) pair — BASE arm ONLY.

    Real schema: per_question_L14.npz + summary.npz each carrying c_C_base /
    c_C_trained / v_A_base / v_A_trained. Plan discipline (v5 section 4.5): the
    #813 bridge consumes the BASE arm only — *_trained keys would smuggle #537's
    adapters into this plan's no-adapter comparison, so they are enforced
    excluded (never read into arrays).
    """
    pq_path = pair_dir / "per_question_L14.npz"
    sm_path = pair_dir / "summary.npz"
    if not pq_path.exists() or not sm_path.exists():
        raise FileNotFoundError(f"813 pair {pair_dir}: missing per_question/summary npz")
    consumed = ("c_C_base", "v_A_base")
    excluded = ("c_C_trained", "v_A_trained")
    item: dict = {
        "behavior": pair_dir.parent.name,
        "substrate": pair_dir.name,
        "source": str(pq_path),
        "sha16": _sha(pq_path),
        "consumed_keys": list(consumed),
        "excluded_trained_keys": list(excluded),
    }
    with np.load(pq_path, allow_pickle=True) as pq:
        for key in consumed:
            if key in excluded or key not in pq.files:
                raise KeyError(f"{pq_path}: expected base-arm key {key}; have {pq.files}")
        if any("trained" in k for k in consumed):
            raise ValueError(f"813 consume set {consumed} touches a trained-arm key")
        X = np.asarray(pq["c_C_base"], dtype=np.float64)
        Y = np.asarray(pq["v_A_base"], dtype=np.float64)
        groups = np.asarray(pq["row_context_index"], dtype=np.int64)
        item["headline_layer"] = int(pq["headline_layer"])
        item["per_question_L14"] = {
            "n_rows": int(X.shape[0]),
            "fit": _fit_once_grouped(X, Y, groups),
            "spectrum": _spectrum(X, Y),
        }
        item["headline_r2"] = item["per_question_L14"]["fit"]["r2"]
    with np.load(sm_path, allow_pickle=True) as sm:
        Xs = np.asarray(sm["c_C_base"], dtype=np.float64)  # (n_ctx, 28, h)
        Ys = np.asarray(sm["v_A_base"], dtype=np.float64)
        layers = [int(v) for v in np.asarray(sm["layers"]).tolist()]
        per_layer: dict[str, dict] = {}
        for li, layer in enumerate(layers):
            if layer_subset is not None and layer not in layer_subset:
                continue
            entry: dict = {
                "fit": _fit_once_grouped(
                    Xs[:, li], Ys[:, li], np.arange(Xs.shape[0], dtype=np.int64)
                )
            }
            if layer in (L923_HEADLINE, L1092_HEADLINE):
                entry["spectrum"] = _spectrum(Xs[:, li], Ys[:, li])
            per_layer[f"L{layer:02d}"] = entry
        item["summary_averaged_grain"] = {
            "n_contexts": int(Xs.shape[0]),
            "per_layer": per_layer,
        }
    return item


def refit_779(
    path: Path,
    fallback_paths: list[Path] | None = None,
    layer_subset: list[int] | None = None,
) -> dict:
    """Map-refit + 28-layer sweep on the #779 pass_b context vectors.

    Real schema: {cx_last, cx_mean, v_x} each (n, 28, hidden) — both X variants
    are fitted. Named fallback (plan v5 section 4.5): if the answer side v_x is
    absent, the target stages from pass_a — realized pass_a files carry only
    context vectors (cx_*), so a missing v_x fails loud after the search.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(blob.keys()) if isinstance(blob, dict) else []
    y_key = next((k for k in ("v_x", "Y", "answer_summaries", "vbar") if k in blob), None)
    fallback_used = None
    if y_key is None and fallback_paths:
        for fp in fallback_paths:
            fb = torch.load(fp, map_location="cpu", weights_only=False)
            y_key = next((k for k in ("v_x", "Y", "answer_summaries", "vbar") if k in fb), None)
            if y_key is not None:
                blob[y_key] = fb[y_key]
                fallback_used = str(fp)
                break
    if y_key is None:
        raise KeyError(f"#779 pass_b key set has no answer side (v_x): {keys}")
    x_keys = [k for k in ("cx_last", "cx_mean") if k in blob]
    if not x_keys:
        raise KeyError(f"#779 pass_b key set has no context side (cx_last/cx_mean): {keys}")
    Y_all = np.asarray(blob[y_key], dtype=np.float64)
    item: dict = {
        "source": str(path),
        "sha16": _sha(path),
        "key_set": keys,
        "y_key": y_key,
        "fallback_used": fallback_used,
        "n_rows": int(Y_all.shape[0]),
        "scope": "map-refit + layer-sweep + bare-query-map comparability only; f/g/i "
        "undefined on the uncrossed LMSYS substrate",
        "split_note": "pass_b rows are per-context; sequential grouped split by row index",
        "x_variants": {},
    }
    n_layers = int(Y_all.shape[1])
    for x_key in x_keys:
        X_all = np.asarray(blob[x_key], dtype=np.float64)
        if X_all.shape != Y_all.shape:
            raise ValueError(f"779 {x_key} shape {X_all.shape} != v_x {Y_all.shape}")
        per_layer: dict[str, dict] = {}
        for layer in layer_subset if layer_subset is not None else range(n_layers):
            entry: dict = {
                "fit": _fit_once_grouped(
                    X_all[:, layer],
                    Y_all[:, layer],
                    np.arange(X_all.shape[0], dtype=np.int64),
                )
            }
            if layer == L1092_HEADLINE:
                entry["spectrum"] = _spectrum(X_all[:, layer], Y_all[:, layer])
            per_layer[f"L{layer:02d}"] = entry
        item["x_variants"][x_key] = {
            "headline_r2": per_layer[f"L{L1092_HEADLINE:02d}"]["fit"]["r2"],
            "per_layer": per_layer,
        }
    item["headline_r2"] = item["x_variants"][x_keys[0]]["headline_r2"]
    return item


def stage_production(staged_root: Path) -> dict[str, list[Path]]:
    """Stage the parent artifacts; a non-empty existing staged tree is REUSED
    (no re-download — the 2026-07-10 crash left a complete staged tree)."""
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
        dest = staged_root / name
        existing = sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix in (".npz", ".pt"))
        if existing:
            staged[name] = existing
            print(f"[bridge-refit] reusing {len(existing)} staged files under {dest}")
            continue
        files = list_scoped(repo, rev, path, repo_type=repo_type)
        wanted = [f for f in files if f.endswith((".npz", ".pt"))]
        if not wanted:
            raise FileNotFoundError(f"{name}: scoped listing had no tensor files under {path}")
        staged[name] = download_scoped(repo, rev, wanted, dest, repo_type=repo_type)
    return staged


def _discover_923_root(paths: list[Path]) -> Path:
    for path in paths:
        parts = list(path.parts)
        if "analysis_tensors" in parts:
            return Path(*parts[: parts.index("analysis_tensors") + 1])
    raise FileNotFoundError("no analysis_tensors dir among staged 923 files")


def _discover_813_pairs(paths: list[Path]) -> list[Path]:
    pairs = sorted({p.parent for p in paths if p.name == "per_question_L14.npz"})
    if not pairs:
        raise FileNotFoundError("no per_question_L14.npz files among staged 813 files")
    return pairs


def _cached_or_compute(cache: dict[str, dict], out_dir: Path, item_key: str, compute) -> dict:
    if item_key in cache:
        print(f"[bridge-refit] cached: {item_key}")
        return cache[item_key]
    t0 = time.monotonic()
    item = compute()
    item["wall_s"] = time.monotonic() - t0
    _item_cache_append(out_dir, item_key, item)
    print(f"[bridge-refit] computed {item_key} in {item['wall_s']:.1f}s")
    return item


def run(args: argparse.Namespace) -> dict:
    run_selftest("cpu")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tiny_real:
        roots = make_tiny_fixtures(out_dir / "fixtures")
        paths = {
            name: sorted(p for p in roots[name].rglob("*") if p.is_file())
            for name in ("923", "813", "779")
        }
        paths["779_fallback"] = []
    else:
        staged_root = args.staged_root if args.staged_root is not None else out_dir / "staged"
        paths = stage_production(staged_root)

    layer_subset = [L1092_HEADLINE, L923_HEADLINE] if args.headline_layers_only else None
    subset_tag = ("__hl" if layer_subset is not None else "") + "__" + SPLIT_RECIPE
    cache = _item_cache_load(out_dir)
    root_923 = _discover_923_root(paths["923"])
    substrates = sorted(
        {
            re.match(r"ffull_(.+)_shard\d+of\d+", p.stem).group(1)
            for p in paths["923"]
            if p.stem.startswith("ffull_") and "_shard" in p.stem
        }
    )
    items_923 = []
    for substrate in substrates:
        store_stem = "uc" if substrate == "uc48" else substrate
        store_sha = _sha(root_923 / "reduce" / f"vbar_store_{store_stem}.pt")
        items_923.append(
            _cached_or_compute(
                cache,
                out_dir,
                f"923__{substrate}__{store_sha}{subset_tag}",
                lambda substrate=substrate: refit_923_substrate(
                    root_923, substrate, layer_subset=layer_subset
                ),
            )
        )
    pairs_813 = _discover_813_pairs(paths["813"])
    items_813 = [
        _cached_or_compute(
            cache,
            out_dir,
            f"813__{pair.parent.name}__{pair.name}"
            f"__{_sha(pair / 'per_question_L14.npz')}{subset_tag}",
            lambda pair=pair: refit_813_pair(pair, layer_subset=layer_subset),
        )
        for pair in pairs_813
    ]
    pass_b = [p for p in paths["779"] if p.suffix == ".pt"]
    if not pass_b:
        raise FileNotFoundError("no pass_b .pt file among staged 779 files")
    items_779 = [
        _cached_or_compute(
            cache,
            out_dir,
            f"779__{p.name}__{_sha(p)}{subset_tag}",
            lambda p=p: refit_779(
                p, fallback_paths=paths.get("779_fallback"), layer_subset=layer_subset
            ),
        )
        for p in pass_b
    ]

    def block(items: list[dict], scope: str) -> dict:
        r2 = [item.get("headline_r2") for item in items]
        return {
            "n_items": len(items),
            "scope": scope,
            "headline_layer": L1092_HEADLINE,
            "headline_r2_mean": float(np.nanmean(r2)) if r2 else float("nan"),
            "items": items,
        }

    result = {
        "phase": "bridge_refit",
        "tiny_real": args.tiny_real,
        "layer_subset": layer_subset,
        "staging_policy": "scoped list_repo_tree + per-file hf_hub_download; no snapshot_download",
        "issue923": block(
            items_923,
            "f/g/i + layer sweep on the persisted grids; uc48 primary, betley sensitivity",
        ),
        "issue813": block(
            items_813, "rank/grain on reduced summaries; BASE arm only (c_C_base/v_A_base)"
        ),
        "issue779": block(items_779, "map-refit + layer sweep; f/g/i undefined on LMSYS"),
    }
    path = out_dir / "bridge_refit_summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(
        f"[bridge-refit] artifact digest: 923_items={result['issue923']['n_items']} "
        f"813_items={result['issue813']['n_items']} 779_items={result['issue779']['n_items']} "
        f"path={path}"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tiny-real", action="store_true")
    p.add_argument(
        "--headline-layers-only",
        action="store_true",
        help="restrict the per-layer sweeps to layers {14, 18} (production spot-run)",
    )
    p.add_argument(
        "--staged-root",
        type=Path,
        default=None,
        help="reuse an existing staged tree (default: <out-dir>/staged); a non-empty "
        "tree is consumed without re-downloading",
    )
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
