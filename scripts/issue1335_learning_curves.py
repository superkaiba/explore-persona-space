"""Issue #1335 (user-chat inline free analysis): R^2-vs-training-n learning curves.

Question: is the assistant-vs-fiction context->answer map magnitude gap explained
by data SIZE? #1335 controls it at matched n; this plots the full R^2-vs-n curve
SHAPE per fiction character (Wren/HELIOS/Dana/Vex, from the r7_endpoint store)
against the assistant Q&A anchors extended to full n (r0_qa_full = "full-answer
Q&A", r1_qa_oneline = "one-line Q&A").

Method (frozen-recipe parity with scripts/issue1335_fit.py, learning-curve variant):
  - Layer 19 (c1310.HEADLINE_LAYER) headline. Context arm (x_spanmean) primary;
    prefix arm (x_prefixmean) companion. On the Q&A rungs the prefix arm is the
    degenerate `prefix_fallback_first_token` control (capture_flags), labelled so.
  - Per cell (rung x model x arm [x persona]): ONE FIXED grouped held-out test
    split (fold 0 of a 5-group-fold split; group = scene for r7, row for Q&A —
    exactly issue1335_fit.py's grouping via issue825_fit_cells._cv_folds). Train
    on group-stratified subsamples of the remaining pool at an n_train grid,
    N_DRAWS deterministic draws per point, fixed test set across all n so points
    are comparable.
  - Ridge: the reused Gram-space closed-form fit
    (issue825_fit_cells._prep_fold + _ridge_predict_cached), lambda by inner-
    GROUP-CV (_prep_inner_lambda, N_INNER_LAMBDA_FOLDS=4) exactly as the plan
    amendment v4 ladder fits. NOT a serial gradient-descent loop; NOT vectorized
    further because each subsample needs its OWN eigh (the fit already IS the
    fast twin). Device-parametrized via issue825_fit_cells._fit_device (CPU here).
  - Held-out pooled R^2 (issue825_fit_cells._pooled_r2), identical to the
    committed recipe.

Only generation seed 42 exists on the Hub for every store (no seed-43/44 variant
stores) — every curve is labelled seed 42.

Staging: r7_endpoint has local .pt payloads; r0/r1 are re-staged from the Hub
STREAM-REDUCED (one shard downloaded -> layer-19 sliced -> deleted -> next), so
peak disk stays ~one shard instead of the full ~5.5 GB store. The compact layer-19
arrays are cached under <data-dir>/lc_cache/ so a resume skips re-download.

CLI:
  uv run python scripts/issue1335_learning_curves.py --stage-only
  uv run python scripts/issue1335_learning_curves.py --resume
  uv run python scripts/issue1335_learning_curves.py --resume --only r7_endpoint
  uv run python scripts/issue1335_learning_curves.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1335_fit as fit1335  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_learning_curves.py"

HEADLINE_LAYER = c1310.HEADLINE_LAYER  # 19
ARM_X_KEY = {"ctx": "x_spanmean", "prefix": "x_prefixmean"}
TARGET_KEY = "y"
# The three layer-19 arrays we keep (ctx arm, prefix arm, target).
CACHE_KEYS = ("x_spanmean", "x_prefixmean", "y")

# Fixed held-out test split: fold 0 of a 5-group-fold split == 20% of GROUPS,
# held constant across every n_train and draw (comparability).
N_TEST_FOLDS = 5
TEST_SPLIT_SEED = 1335
# Deterministic subsample draws.
N_DRAWS = 5
SUBSAMPLE_SEED_BASE = 1335000
# Inner-group-CV lambda-selection folds (parity with fit825).
N_INNER = fit825.N_INNER_LAMBDA_FOLDS

# n_train grids (rows). Fiction rungs use the base grid; Q&A rungs extend it.
GRID_FICTION = (200, 400, 800, 1200, 1600)
GRID_QA = (200, 400, 800, 1200, 1600, 2400, 3600)

# Cells: (rung_slug, family). r7 fits per persona; Q&A fits one "all" cell.
FICTION_RUNGS = ("r7_endpoint",)
QA_RUNGS = ("r0_qa_full", "r1_qa_oneline")
MODELS = ("base", "instruct")
PERSONAS = tuple(c1310.PERSONA_LABELS)  # Wren, HELIOS, Dana, Vex

# Plain-English rung labels for downstream figures (codes stay in JSON only).
RUNG_LABEL = {
    "r0_qa_full": "full-answer Q&A",
    "r1_qa_oneline": "one-line Q&A",
    "r7_endpoint": "fiction endpoint",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data-dir", type=Path, default=None, help="default: the issue-1335 worktree data dir"
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1335/learning_curves"))
    ap.add_argument("--resume", action="store_true", help="skip cells with a written result")
    ap.add_argument(
        "--stage-only", action="store_true", help="build the layer-19 caches, then exit"
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="only rebuild results.json from per-cell files",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated substring filter(s) on cell_id (OR-match; "
        "e.g. r0_qa_full,r7_endpoint__base)",
    )
    return ap.parse_args()


def _cell_selected(cell_id: str, only: str | None) -> bool:
    """OR-match cell_id against the comma-separated --only substrings."""
    if not only:
        return True
    return any(tok and tok in cell_id for tok in only.split(","))


def _default_data_dir() -> Path:
    """The issue-1335 worktree data dir (where the r7 store already lives)."""
    wt = Path(".claude/worktrees/issue-1335/data/issue_1335")
    if wt.exists():
        return wt
    return Path("data/issue_1335")


def _store_dir(data_dir: Path, slug: str, model: str) -> Path:
    return data_dir / "store" / slug / model


def _cache_path(data_dir: Path, slug: str, model: str) -> Path:
    return data_dir / "lc_cache" / f"l19_{slug}_{model}.npz"


def _sidecar_seed(store_dir: Path, model: str) -> int:
    """Generation seed from the first always-local shard sidecar."""
    sides = sorted(store_dir.glob(f"{model}_shard*.json"))
    assert sides, f"no sidecars under {store_dir}"
    d = json.loads(sides[0].read_text())
    return int(d["metadata"]["seed"])


def _slice_l19(payload: dict) -> dict:
    """Extract the layer-19 slice of the three arrays + row metadata from a shard."""
    arrays = payload["arrays"]
    out = {}
    for k in CACHE_KEYS:
        # (n, n_layers, D) bf16 -> (n, D) float32 (lossless from bf16).
        out[k] = arrays[k][:, HEADLINE_LAYER, :].float().numpy().astype(np.float32)
    out["group_ids"] = np.asarray(payload["group_ids"])
    out["char_ids"] = np.asarray(payload["char_ids"])
    out["row_ids"] = np.asarray(payload["row_ids"])
    return out


def build_l19_cache(data_dir: Path, slug: str, model: str) -> Path:
    """Build (or reuse) the compact layer-19 cache for one (rung, model).

    Local .pt shards (r7): read + slice + accumulate.
    Absent shards (r0/r1): STREAM-REDUCE from the Hub — download one shard into a
    tempdir inside the cache dir, slice layer 19, delete, next. Peak disk ~one
    shard (~0.6 GB), not the full store.
    """
    cache = _cache_path(data_dir, slug, model)
    if cache.exists():
        return cache
    store_dir = _store_dir(data_dir, slug, model)
    seed = _sidecar_seed(store_dir, model)
    sidecars = sorted(store_dir.glob(f"{model}_shard*.json"))
    assert sidecars, f"no sidecars for {slug}/{model}"
    shard_names = [f"{sc.name[: -len('.json')]}.pt" for sc in sidecars]

    parts: list[dict] = []
    cache.parent.mkdir(parents=True, exist_ok=True)
    local_missing = [n for n in shard_names if not (store_dir / n).exists()]
    if not local_missing:
        for name in shard_names:
            payload = torch.load(store_dir / name, map_location="cpu", weights_only=False)
            parts.append(_slice_l19(payload))
            del payload
        print(f"[lc] {slug}/{model}: sliced {len(shard_names)} LOCAL shards (seed {seed})")
    else:
        prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{slug}_{model}"
        from explore_persona_space.orchestrate import hub

        staged = 0
        with tempfile.TemporaryDirectory(dir=str(cache.parent), prefix=".lcstage_") as td:
            for name in shard_names:
                local = store_dir / name
                if local.exists():
                    payload = torch.load(local, map_location="cpu", weights_only=False)
                else:
                    dest = Path(td) / name
                    hub.stage_hub_file(
                        r1335.HF_DATA_REPO, f"{prefix}/{name}", dest, repo_type="dataset"
                    )
                    payload = torch.load(dest, map_location="cpu", weights_only=False)
                    parts.append(_slice_l19(payload))
                    del payload
                    dest.unlink(missing_ok=True)  # stream-reduce: free the shard now
                    staged += 1
                    print(
                        f"[lc] {slug}/{model}: streamed shard {staged}/{len(shard_names)} "
                        f"({name}) -> L19 sliced + deleted"
                    )
                    continue
                parts.append(_slice_l19(payload))
        print(f"[lc] {slug}/{model}: staged {staged} shards from Hub (seed {seed})")

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in CACHE_KEYS}
    merged["group_ids"] = np.concatenate([p["group_ids"] for p in parts])
    merged["char_ids"] = np.concatenate([p["char_ids"] for p in parts])
    merged["row_ids"] = np.concatenate([p["row_ids"] for p in parts])
    n = merged["y"].shape[0]
    for k in CACHE_KEYS:
        assert merged[k].shape == (n, merged[k].shape[1]), (k, merged[k].shape)
    # np.savez appends ".npz" unless the name already ends in it — keep the tmp
    # name ".npz"-terminated so the written path == the path we os.replace().
    tmp = cache.with_name(cache.stem + ".partial.npz")
    np.savez(tmp, seed=np.int64(seed), **merged)
    os.replace(tmp, cache)
    print(f"[lc] {slug}/{model}: cached L19 arrays n={n} -> {cache}")
    return cache


def load_l19_cache(data_dir: Path, slug: str, model: str) -> dict:
    d = np.load(_cache_path(data_dir, slug, model), allow_pickle=True)
    return {
        "x_spanmean": d["x_spanmean"],
        "x_prefixmean": d["x_prefixmean"],
        "y": d["y"],
        "group_ids": d["group_ids"],
        "char_ids": d["char_ids"],
        "row_ids": d["row_ids"],
        "seed": int(d["seed"]),
    }


def _fit_one_split(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    inner_seed: int,
) -> tuple[float, float]:
    """One fixed-split held-out R^2 via the reused Gram-space ridge + inner-group-CV.

    Mirrors issue1335_fit / issue825_fit_cells.heldout_r2_sweep's per-fold body:
    _prep_fold -> attach inner-group-CV cache -> _ridge_predict_cached -> pooled R^2.
    Returns (r2, lambda_selected).
    """
    cache = fit825._prep_fold(X_train, X_test)
    cache["inner"] = fit825._prep_inner_lambda(X_train, groups_train, N_INNER, inner_seed)
    pred, lam = fit825._ridge_predict_cached(cache, Y_train, return_lam=True)
    r2 = fit825._pooled_r2(pred, Y_test)
    return float(r2), float(lam)


def _grid_for(family: str) -> tuple[int, ...]:
    return GRID_QA if family == "qa" else GRID_FICTION


def fit_cell(
    cell_id: str,
    slug: str,
    family: str,
    model: str,
    arm: str,
    persona: str | None,
    store: dict,
    out_dir: Path,
) -> dict:
    """Learning curve for one cell. Fixed 20%-group test split; subsampled train."""
    x_key = ARM_X_KEY[arm]
    if persona is not None:
        sel = store["char_ids"] == persona
        X = store[x_key][sel]
        Y = store[TARGET_KEY][sel]
        groups = store["group_ids"][sel]
    else:
        X = store[x_key]
        Y = store[TARGET_KEY]
        groups = store["group_ids"]
    n = X.shape[0]
    folds = fit825._cv_folds(groups, N_TEST_FOLDS, TEST_SPLIT_SEED)
    test_mask = folds == 0
    pool_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    pool_groups = groups[pool_idx]
    pool_n = int(len(pool_idx))
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    test_group_count = int(len(np.unique(groups[test_idx])))

    grid = _grid_for(family)
    # Clamp to pool size; dedup; the pool-size point is deterministic (1 draw).
    eff_ns = sorted({min(g, pool_n) for g in grid} | {pool_n})
    points = []
    for n_eff in eff_ns:
        n_draws = 1 if n_eff == pool_n else N_DRAWS
        for k in range(n_draws):
            sub_seed = SUBSAMPLE_SEED_BASE + n_eff * 100 + k
            positions = fit1335.matched_subsample(pool_groups, n_eff, seed=sub_seed)
            sub_idx = pool_idx[positions]
            r2, lam = _fit_one_split(
                X[sub_idx], Y[sub_idx], groups[sub_idx], X_test, Y_test, inner_seed=sub_seed + 4242
            )
            points.append(
                {
                    "n_train": int(n_eff),
                    "draw": int(k),
                    "realized_n": int(len(sub_idx)),
                    "r2": r2,
                    "lambda": lam,
                    "subsample_seed": int(sub_seed),
                    "is_full_pool": bool(n_eff == pool_n),
                }
            )
    payload = {
        "metadata": common.metadata(
            SCRIPT, TEST_SPLIT_SEED, n, extra={"issue": 1335, "analysis": "learning_curves"}
        ),
        "cell_id": cell_id,
        "rung": slug,
        "rung_label": RUNG_LABEL[slug],
        "family": family,
        "model": model,
        "arm": arm,
        "arm_x_key": x_key,
        "arm_note": (
            "degenerate prefix_fallback_first_token control"
            if (arm == "prefix" and family == "qa")
            else ("prefix arm" if arm == "prefix" else "context arm")
        ),
        "persona": persona,
        "headline_layer": HEADLINE_LAYER,
        "generation_seed": store["seed"],
        "lambda_selection": "inner-group-cv",
        "n_total": int(n),
        "test_n": int(len(test_idx)),
        "test_group_count": test_group_count,
        "pool_n": pool_n,
        "n_test_folds": N_TEST_FOLDS,
        "test_split_seed": TEST_SPLIT_SEED,
        "grid": list(grid),
        "effective_n_train": eff_ns,
        "n_draws": N_DRAWS,
        "points": points,
    }
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    dest = cells_dir / f"{cell_id}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, dest)
    print(f"[lc] wrote {cell_id}: {len(points)} points, pool_n={pool_n}, test_n={len(test_idx)}")
    return payload


def _cells_for(slug: str, family: str, model: str) -> list[tuple[str, str | None, str]]:
    """(cell_id, persona, arm) tuples for one (rung, model)."""
    out = []
    if family == "fiction":
        for persona in PERSONAS:
            for arm in ARM_X_KEY:
                out.append((f"{slug}__{model}__{persona}__{arm}", persona, arm))
    else:
        for arm in ARM_X_KEY:
            out.append((f"{slug}__{model}__all__{arm}", None, arm))
    return out


def aggregate(out_dir: Path) -> Path:
    cells_dir = out_dir / "cells"
    cells = []
    for p in sorted(cells_dir.glob("*.json")):
        cells.append(json.loads(p.read_text()))
    results = {
        "metadata": common.metadata(
            SCRIPT, TEST_SPLIT_SEED, 0, extra={"issue": 1335, "analysis": "learning_curves"}
        ),
        "headline_layer": HEADLINE_LAYER,
        "arms": list(ARM_X_KEY),
        "grid_fiction": list(GRID_FICTION),
        "grid_qa": list(GRID_QA),
        "n_draws": N_DRAWS,
        "n_test_folds": N_TEST_FOLDS,
        "test_split_seed": TEST_SPLIT_SEED,
        "lambda_selection": "inner-group-cv",
        "rung_labels": RUNG_LABEL,
        "n_cells": len(cells),
        "cells": cells,
    }
    dest = out_dir / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, dest)
    print(f"[lc] aggregated {len(cells)} cells -> {dest}")
    return dest


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir or _default_data_dir()
    out_dir = args.out_dir
    print(f"[lc] data_dir={data_dir} out_dir={out_dir} device={fit825._fit_device()}")

    if args.aggregate_only:
        aggregate(out_dir)
        return 0

    plan = [("r7_endpoint", "fiction")] + [(s, "qa") for s in QA_RUNGS]
    # Stage/cache every (rung, model) up front (resumable via the .npz cache).
    for slug, _family in plan:
        for model in MODELS:
            build_l19_cache(data_dir, slug, model)
    if args.stage_only:
        print("[lc] --stage-only: caches built, exiting.")
        return 0

    for slug, family in plan:
        for model in MODELS:
            store = None
            for cell_id, persona, arm in _cells_for(slug, family, model):
                if not _cell_selected(cell_id, args.only):
                    continue
                if args.resume and (out_dir / "cells" / f"{cell_id}.json").exists():
                    print(f"[lc] resume: {cell_id} already done — skipped")
                    continue
                if store is None:
                    store = load_l19_cache(data_dir, slug, model)
                fit_cell(cell_id, slug, family, model, arm, persona, store, out_dir)

    aggregate(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
