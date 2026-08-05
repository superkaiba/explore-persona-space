#!/usr/bin/env python3
"""#1739 syco-OOD — re-score eval rungs (aita + 5 genuinely-OOD sycophancy rungs), canonical arms.

Derived from the (untracked, armfill-round) ``scripts/issue1739_rescore_ood_armfill.py``
at sha256 71e21589d5e1cc47465438f983fcb9ef002a4dd60da603131f25eabb912f5dce —
copied rather than edited in place (the armfill session owns that file).
Deltas vs the source:

1. ``OOD_RUNGS_BY_BEHAVIOR["sycophancy"]`` covers aita PLUS the five #1739
   syco-OOD rungs (sycofb/sycoans/sycomim/sycoays/sycomwe), and a ``--rungs``
   override (comma list or ``auto`` = the eval table's realized rungs) exists
   for partial re-runs. Arm roster UNCHANGED (canonical
   ``arms.resolve_transfer_roster``).
2. GROUP-grain uncertainty: every (arm x rung) row adds ``n_groups``,
   ``ci_rho_group`` (cluster bootstrap over ``group_key`` — the sycomim rung
   has 285 rows but only 15 independent artifacts), and ``rho_groupmean``
   (Spearman over per-group mean score vs mean DV). The context-level
   ``ci_rho`` is retained for continuity but is anti-conservative wherever
   n_groups < n_rung; group-grain is the primary read.
3. The per-rung permutation null adds a ``groupmean`` variant for grouped
   rungs (independence unit = group).

Uses frozen arm parameters from the plain-rung train grid (``cells.jsonl`` +
per-cell preds sidecars) to predict on the two evil OOD eval rungs.

.. code-block:: bash

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue1739_rescore_ood.py \\
        --behavior evil \\
        --out-dir eval_results/issue_1739/evil_ood_spread \\
        --store-dir data/issue_1739/store/evil_labeling \\
        --dv-json eval_results/issue_1739/dv_dataset/evil/labeling.json \\
        --cells-jsonl eval_results/issue_1739/evil/arm_results/percell/cells.jsonl \\
        --tensors-root analysis_tensors/issue_1739

Smoke test (CPU, tiny):

    uv run python scripts/issue1739_rescore_ood.py --smoke \\
        --out-dir /tmp/issue1739_eos_smoke/rescore

Steps
-----
1. Probe HF for STORE_REVISION (``e5901706``); re-pin if absent; log old→new.
2. Load train + OOD eval labeled tables from the staged store.
3. Stage U-pool; for each ``(variant, regime)`` group of plain-ladder cells:
   a. Fit whitening from U-pool.
   b. Build ``CellData`` (whitened context/answer acts, loaded rb direction).
   c. Re-score every eval-split context via ``run_transfer_cell`` with
      rb-caching (rb-indep once per row-set; rb-dep per (regime, row-set,
      seed)).
4. Derive ``frozen_by_arm`` from train cell ``rho_per_layer`` fields.
5. Compute AUROC, AP, precision@k (k=10, 50) per (arm × rung) at the frozen
   layer.  500-draw bootstrap CIs (vectorized) + 500-draw selection-symmetric
   permutation null (all16 selection).
6. Write per-rung JSONL under ``--out-dir/preds/`` and a metrics summary JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only; numpy is imported lazily inside functions
    import numpy as np

logger = logging.getLogger("issue1739_sycoood_rescore")


# ---------------------------------------------------------------------------
# repo-root sys.path bootstrap (script-mode)
# ---------------------------------------------------------------------------


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_fits.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

# ---------------------------------------------------------------------------
# constants / defaults
# ---------------------------------------------------------------------------

BEHAVIOR_DEFAULT = "evil"
OOD_RUNGS = ("hhrt", "toxicchat")

# armfill: the module pinned the EVIL rung pair, so --behavior sycophancy /
# hallucination would iterate the wrong rungs and emit empty buckets.
OOD_RUNGS_BY_BEHAVIOR = {
    "evil": ("hhrt", "toxicchat"),
    # aita (held-out IN-distribution reference) + the five #1739 syco-OOD rungs
    "sycophancy": ("aita", "sycofb", "sycoans", "sycomim", "sycoays", "sycomwe"),
    "hallucination": ("nqopen", "simpleqa"),
}

# Eval-rung roster = the CANONICAL wide transfer roster
# (`arms.TRANSFER_ARMS_WIDE`, resolved via `arms.resolve_transfer_roster`), NOT
# a locally-invented arm1..arm16 list.
#
# This driver previously declared its own 16-slug tuple. That was wrong twice
# over: (1) it included arms the transfer leg DELIBERATELY excludes — the L2-SP
# arms 9/14 (per-regime residual fit), the stacked combiner arm 10 (needs ridge
# preds on EVERY fold, so `run_cell_multi` RAISES under the transfer leg's
# `ridge_folds=(0,)` discarded-fold skip), and the text arms 15/16 (no
# eval-rung text features are threaded) — see the TRANSFER_ARMS_WIDE comment in
# arms.py; and (2) because the driver also passed no `mapfit`, the four
# map-consuming arms that ARE canonical here — the map family 6/7/8 and its
# shuffled-map null 13 — were skipped with reason "no mapfit", silently
# gutting the round's own map-vs-context comparison.
#
# `resolve_transfer_roster(None)` returns TRANSFER_ARMS_WIDE in registry order:
# arms 1, 3, 4, 5, 6, 7, 8, 11, 12, 13.
_TRANSFER_ROSTER_SPEC: str | None = None  # None -> the canonical wide roster

# LEGACY EXPORT — not used by THIS driver's roster resolution (see above).
# `scripts/issue1739_holdout_rung.py` imports this name for its TRAIN-side
# refit, which runs its own CV folds rather than the eval-rung transfer leg,
# so the transfer-leg exclusions do not all apply there. Kept so removing it
# here cannot break that sibling at runtime (its imports are function-scoped,
# so the breakage would surface only when the phase is invoked). Do NOT use it
# for a new eval-rung roster — resolve the canonical roster instead.
_ALL16_NAMES = (
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
)

STORE_REVISION_DEFAULT = "e5901706"
N_BOOT_DEFAULT = 500
N_PERM_DEFAULT = 500
AUROC_POS_THR = 50.0  # binarise DV at this threshold for AUROC/AP/P@k

SMOKE_N_CTX = 2
SMOKE_N_ARMS = 2
SMOKE_N_BOOT = 5
SMOKE_N_PERM = 5

DEFAULT_OUT_DIR = Path("eval_results/issue_1739/evil_ood_spread")
DEFAULT_CELLS_JSONL = Path("eval_results/issue_1739/evil/arm_results/percell/cells.jsonl")
DEFAULT_DV_JSON = Path("eval_results/issue_1739/dv_dataset/evil/labeling.json")
DEFAULT_STORE_DIR = Path("data/issue_1739/store/evil_labeling")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[rescore-ood] {msg}", flush=True)


def _ap_score(scores: "np.ndarray", labels: "np.ndarray") -> float:  # type: ignore[name-defined]
    """Average precision (AP) for a 1-D score/label pair."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    prec = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    return float(np.sum(prec * labels[order]) / labels.sum())


def _precision_at_k(scores: "np.ndarray", labels: "np.ndarray", k: int) -> float:  # type: ignore[name-defined]
    """Precision@k — fraction of top-k predictions that are positive."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    top_k = np.argsort(-scores)[:k]
    return float(labels[top_k].sum() / k)


def _ap_rows(scores: "np.ndarray", labels: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Batched AP: scores (S, n), labels (n,) → (S,)."""
    import numpy as np

    scores = np.atleast_2d(scores)
    return np.array([_ap_score(sc, labels) for sc in scores])


def _prec_at_k_rows(scores: "np.ndarray", labels: "np.ndarray", k: int) -> "np.ndarray":  # type: ignore[name-defined]
    """Batched P@k: scores (S, n), labels (n,) → (S,)."""
    import numpy as np

    scores = np.atleast_2d(scores)
    return np.array([_precision_at_k(sc, labels, k) for sc in scores])


# ---------------------------------------------------------------------------
# HF store revision probe + re-pin
# ---------------------------------------------------------------------------


def _probe_and_repin_store(store_dir: Path, revision: str, *, behavior: str, smoke: bool) -> str:
    """Probe HF for ``revision``; if absent re-resolve at HEAD and log old→new."""
    if smoke:
        # smoke mode: no network access, accept whatever is locally staged
        return revision
    try:
        from explore_persona_space.orchestrate import hub
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        prefix = f"issue1739_ctxmap/{behavior}_labeling"
        try:
            files = hub.list_hf_files_under_path(
                api,
                hub.DEFAULT_DATASET_REPO,
                prefix,
                repo_type="dataset",
                revision=revision,
            )
            if files:
                _log(f"store revision {revision} confirmed on HF ({len(files)} files)")
                return revision
        except Exception as probe_exc:
            _log(f"store revision probe failed ({probe_exc}); will try HEAD")

        # re-resolve at HEAD
        try:
            info = api.repo_info(hub.DEFAULT_DATASET_REPO, repo_type="dataset")
            new_rev = info.sha or "main"
            _log(f"store revision re-pinned: {revision} -> {new_rev}")
            return new_rev
        except Exception as head_exc:
            _log(f"HEAD re-pin also failed ({head_exc}); keeping original {revision}")
            return revision
    except ImportError:
        _log("hub not importable; keeping revision as-is")
        return revision


# ---------------------------------------------------------------------------
# load train cells metadata
# ---------------------------------------------------------------------------


def _normalize_cell(rec: dict) -> dict:
    """Hoist a cell's regime identity to the top level.

    The committed ``cells.jsonl`` stores a cell's identity ONLY in its
    ``unit_key`` (a JSON string) and repeated on each row of ``arms``; the cell
    record's own top-level keys are just ``headline / max_over_arms_null /
    preds_npz / skipped_arms / split_half / unit_key / arms``. This driver was
    written against a schema that carried those fields at the top level, so
    every ``rec.get("f_u")`` / ``rec.get("variant")`` / ``rec.get("budget_l")``
    read silently missed -- the plain-ladder filter matched NOTHING (0 of 810
    sycophancy cells), and the (variant, regime) grouping would have defaulted
    every cell to ("context_end", "e1"), collapsing prefix_end into the wrong
    group with zeroed budget/draw/seed provenance.

    Hoisting once at load time fixes all three sites without touching them.
    Existing top-level keys always win, so a future schema that does carry them
    is unaffected.
    """
    merged = dict(rec)
    src: dict = {}
    uk = rec.get("unit_key")
    if isinstance(uk, str):
        try:
            parsed = json.loads(uk)
            if isinstance(parsed, dict):
                src = parsed
        except json.JSONDecodeError:
            src = {}
    if not src:
        arm_rows = rec.get("arms") or []
        if arm_rows and isinstance(arm_rows[0], dict):
            src = arm_rows[0]
    for key in (
        "variant",
        "regime",
        "u_rung_label",
        "u_rung",
        "f_u",
        "f_l",
        "budget_l",
        "draw",
        "seed",
        "config",
    ):
        if key not in merged and key in src:
            merged[key] = src[key]
    return merged


def _load_plain_ladder_cells(cells_jsonl: Path) -> list[dict]:
    """Read cells.jsonl and return only the plain-ladder (f_u=None, u_rung_label='full') rows."""
    cells: list[dict] = []
    n_total = 0
    with cells_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = _normalize_cell(json.loads(line))
            n_total += 1
            f_u = rec.get("f_u")
            u_rung = rec.get("u_rung_label", "")
            if f_u is None and u_rung == "full":
                cells.append(rec)
    if n_total and not cells:
        raise SystemExit(
            f"{cells_jsonl}: {n_total} cells read, ZERO matched the plain-ladder filter "
            "(f_u=None, u_rung_label='full'). Refusing to continue -- the driver would "
            "emit 0 rows and exit 0. Check the cell schema."
        )
    _log(f"loaded {len(cells)} plain-ladder (f_u=None, u_rung='full') cells")
    return cells


def _frozen_by_arm_from_cell(cell_rec: dict) -> dict[str, int]:
    """Extract {arm_slug: frozen_layer_idx} from a cell record's arm rows."""
    from explore_persona_space.experiments.issue_1739.arms import frozen_layer_idx

    result: dict[str, int] = {}
    for arm_row in cell_rec.get("arms") or []:
        slug = arm_row.get("arm")
        rho_pl = arm_row.get("rho_per_layer")
        if slug and rho_pl:
            result[slug] = frozen_layer_idx(rho_pl)
    return result


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _group_boot_rhos(
    sc: "np.ndarray",  # type: ignore[name-defined]
    dv: "np.ndarray",  # type: ignore[name-defined]
    groups: "np.ndarray",  # type: ignore[name-defined]
    *,
    n_boot: int,
    seed: int,
) -> "np.ndarray":  # type: ignore[name-defined]
    """Cluster bootstrap: resample GROUPS with replacement, rho per draw.

    Groups are the independence unit (sycomim: 15 artifacts behind 285 rows),
    so the draw resamples group ids and concatenates their member rows. Draws
    with zero rank variance return NaN (nanquantile skips them).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import spearman_rows

    from explore_persona_space.experiments.issue_1739.arms import (
        bootstrap_rhos,
        make_bootstrap_idx,
    )

    ug, inv = np.unique(groups, return_inverse=True)
    n_groups = ug.size
    if n_groups == sc.size:
        # UNGROUPED rung (every context its own group): the cluster bootstrap IS
        # the ordinary bootstrap, so take the VECTORIZED path. The generic loop
        # below concatenates n singleton index arrays per draw — 500 x 1,304 =
        # 652k concatenations per (arm, rung) on aita alone, which dominated the
        # whole rescore before this branch existed.
        idx = make_bootstrap_idx(sc.size, n_boot=n_boot, seed=seed)
        return np.asarray(bootstrap_rhos(sc[None], dv, idx)[0], dtype=float)

    rng = np.random.default_rng(seed)
    # Precompute member index arrays ONCE (the loop then concatenates n_groups
    # arrays per draw — cheap when n_groups is small, e.g. sycomim's 15).
    members = [np.flatnonzero(inv == gi) for gi in range(n_groups)]
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[gi] for gi in picks[b]])
        out[b] = float(spearman_rows(sc[idx][None], dv[idx])[0])
    return out


def _group_means(
    sc: "np.ndarray",  # type: ignore[name-defined]
    dv: "np.ndarray",  # type: ignore[name-defined]
    groups: "np.ndarray",  # type: ignore[name-defined]
) -> "tuple[np.ndarray, np.ndarray]":  # type: ignore[name-defined]
    """Per-group mean (score, dv) aggregates, group order = np.unique."""
    import numpy as np

    ug = np.unique(groups)
    sc_g = np.array([sc[groups == g].mean() for g in ug])
    dv_g = np.array([dv[groups == g].mean() for g in ug])
    return sc_g, dv_g


def _compute_detection_metrics(
    scores_ev: "dict[str, np.ndarray]",  # type: ignore[name-defined]
    dv_ev: "np.ndarray",  # type: ignore[name-defined]
    rungs_ev: "list[str]",
    frozen_by_arm: dict[str, int],
    layers: tuple[int, ...],
    *,
    groups_ev: "list[str] | None" = None,
    n_boot: int = N_BOOT_DEFAULT,
    n_perm: int = N_PERM_DEFAULT,
    cell_seed: int = 0,
    cell_draw: int = 0,
    k_vals: tuple[int, ...] = (10, 50),
    all16: list[str],
) -> tuple[list[dict], dict]:
    """Per-(arm, rung) detection metrics + bootstrap CIs + permutation null.

    Returns (arm_rung_rows, perm_null_by_rung).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import (
        N_BOOT,
        N_PERM,
        auroc_rows,
        bootstrap_rhos,
        make_bootstrap_idx,
        permutation_null_max,
        spearman_rows,
    )

    n_boot = n_boot or N_BOOT
    n_perm = n_perm or N_PERM

    dv_ev = np.asarray(dv_ev, dtype=np.float64)
    rungs_arr = np.asarray([str(r) for r in rungs_ev])
    groups_arr = np.asarray([str(g) for g in groups_ev]) if groups_ev is not None else None
    unique_rungs = sorted(set(rungs_arr))

    rows: list[dict] = []
    perm_nulls: dict[str, dict] = {}

    for rung in unique_rungs:
        mask = rungs_arr == rung
        dv_r = dv_ev[mask]
        groups_r = groups_arr[mask] if groups_arr is not None else None
        labels_r = dv_r >= AUROC_POS_THR

        # collect frozen-layer score matrix for all requested arms (S, n_rung)
        slug_order: list[str] = []
        sc_matrix: list[np.ndarray] = []  # each (n_rung,)

        for slug in all16:
            if slug not in scores_ev or slug not in frozen_by_arm:
                continue
            sc = np.asarray(scores_ev[slug], dtype=np.float64)
            fl = min(int(frozen_by_arm[slug]), sc.shape[0] - 1)
            sc_r = sc[fl][mask]
            keep = np.isfinite(sc_r) & np.isfinite(dv_r)
            if keep.sum() < 3:
                continue
            slug_order.append(slug)
            sc_matrix.append(sc_r)

        if not slug_order:
            continue

        scores_mat = np.stack(sc_matrix, axis=0)  # (S, n_rung)
        n_rung = int(mask.sum())

        # bootstrap index
        idx_b = make_bootstrap_idx(n_rung, n_boot=n_boot, seed=cell_seed + 100 * cell_draw)

        # AUROC for all arms at once (batched)
        auroc_vals = auroc_rows(scores_mat, labels_r)

        # per-arm rows
        for i, slug in enumerate(slug_order):
            sc_1 = scores_mat[i]
            keep = np.isfinite(sc_1) & np.isfinite(dv_r)
            sc_k = sc_1[keep]
            dv_k = dv_r[keep]
            lbl_k = labels_r[keep]
            n_k = int(keep.sum())

            # bootstrap CIs on Spearman rho
            idx_k = make_bootstrap_idx(n_k, n_boot=n_boot, seed=cell_seed + 100 * cell_draw + i)
            draws = bootstrap_rhos(sc_k[None], dv_k, idx_k)[0]  # (n_boot,)
            rho = float(spearman_rows(sc_k[None], dv_k)[0])

            row: dict[str, Any] = {
                "arm": slug,
                "rung": rung,
                "n_rung": n_k,
                "rho": rho,
                "ci_rho": [
                    float(np.nanquantile(draws, 0.025)),
                    float(np.nanquantile(draws, 0.975)),
                ],
                "auroc": float(auroc_vals[i]),
                "ap": float(_ap_score(sc_k, lbl_k)),
            }
            if groups_r is not None:
                g_k = groups_r[keep]
                n_groups = int(np.unique(g_k).size)
                row["n_groups"] = n_groups
                gdraws = _group_boot_rhos(
                    sc_k, dv_k, g_k, n_boot=n_boot, seed=cell_seed + 100 * cell_draw + i
                )
                row["ci_rho_group"] = [
                    float(np.nanquantile(gdraws, 0.025)),
                    float(np.nanquantile(gdraws, 0.975)),
                ]
                if n_groups >= 3:
                    sc_g, dv_g = _group_means(sc_k, dv_k, g_k)
                    row["rho_groupmean"] = float(spearman_rows(sc_g[None], dv_g)[0])
                else:
                    row["rho_groupmean"] = None
            for k in k_vals:
                row[f"precision_at_{k}"] = float(_precision_at_k(sc_k, lbl_k, k))
            rows.append(row)

        # selection-symmetric permutation null over all16 arms (per rung)
        try:
            null = permutation_null_max(scores_mat, dv_r, n_perm=n_perm, seed=cell_seed)
            if groups_r is not None and np.unique(groups_r).size < groups_r.size:
                # grouped rung: context-level null is anti-conservative — add a
                # per-group-aggregate variant (independence unit = group)
                ug = np.unique(groups_r)
                sc_gm = np.stack(
                    [np.array([sm[groups_r == g].mean() for g in ug]) for sm in scores_mat]
                )
                dv_gm = np.array([dv_r[groups_r == g].mean() for g in ug])
                null["groupmean"] = permutation_null_max(
                    sc_gm, dv_gm, n_perm=n_perm, seed=cell_seed
                )
                null["n_groups"] = int(ug.size)
            perm_nulls[rung] = null
        except Exception as exc:
            _log(f"permutation null failed for rung={rung}: {exc}")
            perm_nulls[rung] = {"error": str(exc)}

    return rows, perm_nulls


# ---------------------------------------------------------------------------
# main scoring loop
# ---------------------------------------------------------------------------


def _ridge_folds_arg(args: argparse.Namespace) -> tuple[int, ...] | None:
    """None when --ridge-folds all.

    arm10_stacked needs ridge predictions on EVERY fold; under the transfer
    leg's default ``(0,)`` discarded-fold skip ``run_cell_multi`` RAISES, so
    arm10 cannot be scored without this. Outputs for the other arms are
    unchanged either way -- ``(0,)`` only avoids computing a fold whose
    predictions are discarded -- at the cost of one extra Gram+eigh per cell.
    """
    return None if getattr(args, "ridge_folds", "discarded-skip") == "all" else (0,)


def _csv_set(raw: str | None) -> set[str] | None:
    """Parse a comma-separated filter into a set; None/empty means 'no filter'."""
    if not raw:
        return None
    vals = {v.strip() for v in str(raw).split(",") if v.strip()}
    return vals or None


def _preds_filename(behavior: str, rung: str) -> str:
    """Per-rung preds filename (the legacy hhrt/toxicchat names are preserved)."""
    fname_map = {
        "hhrt": f"{behavior}_hh_rlhf_preds.jsonl",
        "toxicchat": f"{behavior}_toxicchat_preds.jsonl",
    }
    return fname_map.get(rung, f"{behavior}_{rung}_preds.jsonl")


def _rescore_behavior(args: argparse.Namespace) -> dict:
    """Score one behavior's OOD eval rungs with all16 arms.

    Returns a summary dict with per-rung metrics aggregated across cells.
    """
    import numpy as np

    from scripts.issue1739_fits import (
        _fit_map,
        _load_labeled,
        arrays_dim,
    )
    from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
    from explore_persona_space.experiments.issue_1739.constants import STORE_REVISION

    behavior = args.behavior
    out_dir = Path(args.out_dir)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    smoke: bool = args.smoke
    store_dir = Path(args.store_dir)
    dv_json = Path(args.dv_json)
    cells_jsonl = Path(args.cells_jsonl)
    tensors_root = Path(args.tensors_root)
    device = getattr(args, "device", "cpu")

    revision = _probe_and_repin_store(
        store_dir,
        getattr(args, "store_revision", STORE_REVISION),
        behavior=behavior,
        smoke=smoke,
    )

    # --- stage U-pool ---
    _log("staging U-pool store …")
    u_store_dir = store_io.stage_u_store(revision=revision)
    _log(f"U-pool staged at {u_store_dir}")

    # --- infer layers + dim ---
    layers_full = list(range(store_io.N_LAYERS))
    # smoke: restrict to 2 layers
    layers = layers_full[:2] if smoke else layers_full
    dim = arrays_dim(store_dir, [layers[0]])

    # --- load train table (config_a = 'train' split) ---
    _log("loading train labeled table (config_a) …")
    tbl_tr = _load_labeled(store_dir, dv_json, layers, config="config_a", need_rollout_rows=False)
    _log(f"train table: {len(tbl_tr.ctx_order)} contexts, rungs={tbl_tr.rungs}")

    # --- load OOD eval table (config_b = 'eval' split) ---
    _log("loading OOD eval labeled table (config_b) …")
    tbl_ev = _load_labeled(store_dir, dv_json, layers, config="config_b", need_rollout_rows=False)
    _log(f"eval table: {len(tbl_ev.ctx_order)} contexts, rungs={tbl_ev.rungs}")

    # --- load plain-ladder cells ---
    cells = _load_plain_ladder_cells(cells_jsonl)
    if smoke:
        # smoke: use only 2 arms' regimes; still need at least one cell per (variant, regime)
        seen: set[tuple[str, str]] = set()
        smoke_cells: list[dict] = []
        for c in cells:
            key = (c.get("variant", ""), c.get("regime", ""))
            if key not in seen:
                seen.add(key)
                smoke_cells.append(c)
            if len(smoke_cells) >= 6:  # 2 variants x 3 regimes max
                break
        cells = smoke_cells
        _log(f"smoke: using {len(cells)} cells")

    # group cells by (variant, regime)
    from collections import defaultdict

    groups_by_vr: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cells:
        vr = (c.get("variant", "context_end"), c.get("regime", "e1"))
        groups_by_vr[vr].append(c)
    # armfill: LOCAL shadow -> every OOD_RUNGS reference below is per-behavior.
    rungs_arg = getattr(args, "rungs", None)
    if rungs_arg == "auto":
        OOD_RUNGS = tuple(tbl_ev.rungs)
    elif rungs_arg:
        OOD_RUNGS = tuple(r.strip() for r in rungs_arg.split(",") if r.strip())
    else:
        OOD_RUNGS = OOD_RUNGS_BY_BEHAVIOR.get(behavior, OOD_RUNGS_BY_BEHAVIOR["evil"])
    _log(f"ood rungs for {behavior}: {OOD_RUNGS}")
    _log(f"cell groups: {sorted(groups_by_vr.keys())}")

    # Resolve the CANONICAL transfer roster (registry order, validated against
    # ARM_REGISTRY; raises on an unknown slug — never a silent drop).
    all16 = arms.resolve_transfer_roster(
        getattr(args, "transfer_arms", None) or _TRANSFER_ROSTER_SPEC
    )
    if smoke:
        all16 = all16[:SMOKE_N_ARMS]
    _log(f"arms ({len(all16)}): {all16}")

    n_boot = SMOKE_N_BOOT if smoke else int(args.n_boot or N_BOOT_DEFAULT)
    n_perm = SMOKE_N_PERM if smoke else int(args.n_perm or N_PERM_DEFAULT)

    rb_indep, rb_dep = arms.partition_transfer_roster(all16)
    _log(f"rb_indep={rb_indep}, rb_dep={rb_dep}")

    # per-rung aggregation.
    # MEMORY CONTRACT (#1739 OOM, rc=137): pred rows are STREAMED to disk per
    # unit, never buffered across units — buffering ~35k rows/unit x 270 units
    # grew RSS ~1 GB/unit and SIGKILLed the 251 GB box at ~unit 40. Metric rows
    # are small (54/unit) but are ALSO appended per unit so a crash keeps them
    # (code-style.md "Checkpoint per phase", intra-phase grain).
    all_rows: list[dict] = []
    perm_null_records: list[dict] = []
    preds_counts: dict[str, int] = {r: 0 for r in OOD_RUNGS}
    preds_dir.mkdir(parents=True, exist_ok=True)
    _metric_rows_path = out_dir / "metric_rows.jsonl"
    _metric_rows_path.parent.mkdir(parents=True, exist_ok=True)

    # RESUME (#1739): load per-unit rows an earlier attempt already persisted and
    # SKIP those units. Two OOM kills proved the grid must survive being killed
    # mid-run; rows + preds are APPENDED, so a resumed run extends the same
    # artifacts instead of truncating them.
    _done_units: set[tuple[str, str, int, int, int]] = set()
    if _metric_rows_path.exists():
        for line in _metric_rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            all_rows.append(rec)
            _done_units.add(
                (
                    str(rec["variant"]),
                    str(rec["regime"]),
                    int(rec["budget_l"]),
                    int(rec["draw"]),
                    int(rec["seed"]),
                )
            )
        _log(f"resume: {len(all_rows)} metric rows / {len(_done_units)} completed units")
    _preds_fh = {
        r: open(preds_dir / _preds_filename(behavior, r), "a", encoding="utf-8") for r in OOD_RUNGS
    }
    for _r in OOD_RUNGS:
        _pf = preds_dir / _preds_filename(behavior, _r)
        preds_counts[_r] = (
            sum(1 for ln in _pf.open(encoding="utf-8") if ln.strip()) if _pf.exists() else 0
        )
    _metric_fh = open(_metric_rows_path, "a", encoding="utf-8")

    dv_ev = np.asarray(tbl_ev.dv, dtype=np.float64)
    rungs_ev = [str(r) for r in tbl_ev.row_rungs]
    groups_ev = [str(g) for g in tbl_ev.groups]

    want_variants = _csv_set(getattr(args, "variants", None))
    want_regimes = _csv_set(getattr(args, "regimes", None))
    _wb = _csv_set(getattr(args, "budgets", None))
    want_budgets = {int(v) for v in _wb} if _wb else None
    for (variant, regime), group_cells in sorted(groups_by_vr.items()):
        if want_variants and variant not in want_variants:
            continue
        if want_regimes and regime not in want_regimes:
            continue
        _log(f"--- variant={variant} regime={regime} ({len(group_cells)} cells) ---")

        # --- fit whitening + the context->answer map from the U-pool ---
        # BOTH sides of the pool are loaded: the variant act (map input) and the
        # t1 answer act (map target). Loading only the variant side leaves
        # `mapfit=None` below, which SKIPS all seven map-consuming arms with
        # reason "no mapfit" (arms.py run_cell_multi) — the whole map family
        # (6/7/8), its shuffled-map null (13), and 9/10/14. See
        # `_u_pool_for_spec` in issue1739_fits.py for the ("t1", layer) key.
        _log("  loading U-pool summaries (variant + t1) for whitening + map …")
        u_arrays, u_meta = store_io.load_summaries(
            u_store_dir,
            (variant, "t1"),
            tuple(layers),
            hidden_dim=dim,
        )
        pool_mask = store_io.fit_pool_mask(u_meta)
        u_x = np.stack([u_arrays[(variant, ly)][pool_mask] for ly in layers])  # (Ly, n_u, d)
        u_y = np.stack([u_arrays[("t1", ly)][pool_mask] for ly in layers])  # (Ly, n_u, d)
        _log(f"  U-pool shape: x={u_x.shape} y={u_y.shape}")

        wh = fits.fit_whitening(u_x, device=device, seed=42)
        _log("  whitening fitted")

        # Map REFIT in-process on the same U pool, exactly as the main run, the
        # pvsynth leg and issue1739_wcrung_arms.py do — never an uploaded map
        # payload (a loaded payload would be a different estimator under the
        # comparison). Whitening is applied with the canonical library helper.
        mapfit = _fit_map(args, fits.apply_whitening(u_x, wh), fits.apply_whitening(u_y, wh))
        _log(f"  map fitted (kind={getattr(mapfit, 'kind', 'linear')}, n_u={u_x.shape[1]})")
        del u_x, u_y

        # --- load rb direction for this regime ---
        rb_path = tensors_root / f"r_b_{regime}" / f"{behavior}.npz"
        if not rb_path.exists():
            _log(f"  rb not found: {rb_path} — skipping regime {regime}")
            continue
        rb_raw = np.load(rb_path)["rb"].astype(np.float64)  # (N_LAYERS, d) fp16 stored
        # SMOKE/PRODUCTION SHAPE PARITY: the smoke restricts `layers` to a slice
        # (2 of 28) while the stored rb always carries ALL layers, so the einsum
        # below broadcast-fails under --smoke only (production is byte-identical
        # because the slice is then the full range). Index rb by the SELECTED
        # layers rather than assuming the full stack.
        if rb_raw.shape[0] != len(layers):
            rb_raw = rb_raw[np.asarray(layers, dtype=int)]
        rb_w = np.einsum("ld,lde->le", rb_raw, wh.w)  # whitened (Ly, d)

        # --- whiten eval table for this variant ---
        z_tr_raw = tbl_tr.z_by_variant[variant]  # (Ly, n_tr, d)
        z_ev_raw = tbl_ev.z_by_variant[variant]  # (Ly, n_ev, d)
        za_tr_raw = tbl_tr.z_ans  # (Ly, n_tr, d)
        za_ev_raw = tbl_ev.z_ans  # (Ly, n_ev, d)

        # Whiten with the canonical library helper — NOT a local reimplementation.
        # `wh.mu` is (Ly, d), so the previous local `z - wh.mu[None, None, :]`
        # broadcast (Ly, n, d) against (1, 1, Ly, d) and raised ValueError for
        # every realistic shape (and silently produced WRONG values in the
        # degenerate n == Ly case a tiny smoke would hit). apply_whitening
        # centers per layer (`x[li] - wh.mu[li][None, :]`), is memory-chunked,
        # and is pinned bit-identical by test_apply_whitening_chunked_matches_dense.
        z_tr_w = fits.apply_whitening(z_tr_raw, wh)
        z_ev_w = fits.apply_whitening(z_ev_raw, wh)
        za_tr_w = fits.apply_whitening(za_tr_raw, wh)
        za_ev_w = fits.apply_whitening(za_ev_raw, wh)

        # --- rb-caching: one fit per realized row-set for rb_indep,
        #                 per (regime, row-set, seed) for rb_dep ---
        rbindep_cache: dict[str, tuple[dict, dict]] = {}
        rbdep_cache: dict[tuple, tuple[dict, dict]] = {}

        t0 = time.time()
        # We process each unique (budget_l, draw, seed) unit across all cells
        # in this (variant, regime) group. Since all plain-ladder cells at
        # u_rung='full' share budget_l = n_train_contexts, the row set is the
        # same and the rb-indep cache fires once.
        #
        # Collect one representative cell per (budget_l, draw, seed).
        units_seen: dict[tuple[int, int, int], dict] = {}
        for cell_rec in group_cells:
            bl = int(cell_rec.get("budget_l", 0))
            draw = int(cell_rec.get("draw", 0))
            seed = int(cell_rec.get("seed", 0))
            units_seen.setdefault((bl, draw, seed), cell_rec)

        for ui, ((budget_l, draw, seed), cell_rec) in enumerate(sorted(units_seen.items())):
            if want_budgets and int(budget_l) not in want_budgets:
                continue
            if (variant, regime, int(budget_l), int(draw), int(seed)) in _done_units:
                _log(f"  unit {ui + 1}/{len(units_seen)} SKIP (resumed)")
                continue
            _log(f"  unit {ui + 1}/{len(units_seen)} budget_l={budget_l} draw={draw} seed={seed}")

            cell = fits.realize_budget_cell(tbl_tr.groups, budget_l=budget_l, draw=draw, seed=seed)

            # build CellData for the train slice
            data = arms.CellData(
                z_ctx=z_tr_w,
                z_ans=za_tr_w,
                dv=tbl_tr.dv,
                rb=rb_w,
                mapfit=mapfit,
                layers=tuple(layers),
            )

            rs_key = hashlib.sha1(cell.row_idx.tobytes()).hexdigest()

            if rb_indep and rs_key not in rbindep_cache:
                rbindep_cache[rs_key] = arms.run_transfer_cell(
                    data,
                    cell,
                    z_ev_w,
                    dv_ev,
                    za_ev=za_ev_w,
                    arms=rb_indep,
                    device=device,
                    ridge_folds=_ridge_folds_arg(args),
                )

            ck = (regime, rs_key, int(seed))
            if rb_dep and ck not in rbdep_cache:
                rbdep_cache[ck] = arms.run_transfer_cell(
                    data,
                    cell,
                    z_ev_w,
                    dv_ev,
                    za_ev=za_ev_w,
                    arms=rb_dep,
                    device=device,
                    ridge_folds=_ridge_folds_arg(args),
                )

            s_indep, sk_indep = rbindep_cache.get(rs_key, ({}, {}))
            s_dep, sk_dep = rbdep_cache.get(ck, ({}, {}))
            scores_ev: dict[str, np.ndarray] = {**s_dep, **s_indep}

            frozen_by_arm = _frozen_by_arm_from_cell(cell_rec)

            provenance = {
                "behavior": behavior,
                "variant": variant,
                "regime": regime,
                "budget_l": budget_l,
                "draw": draw,
                "seed": seed,
            }

            # detection metrics
            metric_rows, perm_nulls = _compute_detection_metrics(
                scores_ev,
                dv_ev,
                rungs_ev,
                frozen_by_arm,
                tuple(layers),
                groups_ev=groups_ev,
                n_boot=n_boot,
                n_perm=n_perm,
                cell_seed=seed,
                cell_draw=draw,
                all16=all16,
            )
            for row in metric_rows:
                rec = {**provenance, **row}
                all_rows.append(rec)
                _metric_fh.write(json.dumps(rec) + "\n")
            _metric_fh.flush()

            for rung, null_dict in perm_nulls.items():
                perm_null_records.append({**provenance, "rung": rung, **null_dict})

            # per-context preds rows for JSONL output
            preds = arms.transfer_preds_rows(
                scores_ev,
                dv_ev,
                tbl_ev.ctx_order,
                frozen_by_arm,
                provenance=provenance,
                layers=tuple(layers),
                labels={"rung": rungs_ev},
            )
            for row in preds:
                rung = row.get("rung", "unknown")
                fh = _preds_fh.get(rung)
                if fh is not None:
                    fh.write(json.dumps(row) + "\n")
                    preds_counts[rung] += 1
            for fh in _preds_fh.values():
                fh.flush()
            del preds

            elapsed = time.time() - t0
            _log(
                f"  unit {ui + 1}/{len(units_seen)} done "
                f"({len(metric_rows)} metric rows, "
                f"{sum(preds_counts.values())} pred rows, "
                f"{elapsed:.1f}s)"
            )

    # --- close the streamed JSONL handles (rows were written per unit) ---
    for fh in _preds_fh.values():
        fh.close()
    _metric_fh.close()
    rung_paths: dict[str, Path] = {}
    for rung in OOD_RUNGS:
        dest = preds_dir / _preds_filename(behavior, rung)
        rung_paths[rung] = dest
        _log(f"preds/{rung}: {preds_counts[rung]} rows -> {dest}")

    # --- write metrics summary JSON ---
    summary = {
        "behavior": behavior,
        "ood_rungs": list(OOD_RUNGS),
        "all16": all16,
        "n_metric_rows": len(all_rows),
        "metric_rows": all_rows,
        "perm_nulls": perm_null_records,
        "preds_files": {k: str(v) for k, v in rung_paths.items()},
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": smoke,
    }
    summary_path = out_dir / "ood_detection_metrics.json"
    tmp = summary_path.with_name(summary_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, summary_path)
    _log(f"summary -> {summary_path} ({len(all_rows)} metric rows)")

    return summary


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", default=BEHAVIOR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    ap.add_argument("--dv-json", type=Path, default=DEFAULT_DV_JSON)
    ap.add_argument("--cells-jsonl", type=Path, default=DEFAULT_CELLS_JSONL)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--store-revision", default=STORE_REVISION_DEFAULT)
    ap.add_argument(
        "--rungs",
        default=None,
        help="preds-rung roster override: comma list, or 'auto' = the eval table's "
        "realized rungs; default = OOD_RUNGS_BY_BEHAVIOR",
    )
    ap.add_argument(
        "--variants",
        default=None,
        help="comma list of variants to run (default: all). Lets one grid be "
        "SHARDED across processes with separate --out-dir roots.",
    )
    ap.add_argument(
        "--regimes",
        default=None,
        help="comma list of regimes to run (default: all); shards with --variants.",
    )
    ap.add_argument(
        "--budgets",
        default=None,
        help="comma list of budget_l values to run (default: all). The transfer "
        "fold aggregates budget_l=16000 units, so a headline-only pass can skip "
        "the smaller ladder rungs.",
    )
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--transfer-arms",
        nargs="+",
        default=None,
        help="roster override passed to arms.resolve_transfer_roster; the default wide roster EXCLUDES arms 2/9/14 (L2-SP) and arm10 (needs every ridge fold), so name them explicitly to score them",
    )
    ap.add_argument(
        "--ridge-folds",
        choices=("discarded-skip", "all"),
        default="discarded-skip",
        help="'all' -> ridge_folds=None, REQUIRED for arm10_stacked",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"smoke mode: {SMOKE_N_CTX} contexts x {SMOKE_N_ARMS} arms "
            f"x {SMOKE_N_BOOT} boot x {SMOKE_N_PERM} perm; "
            "output -> /tmp/issue1739_eos_smoke/rescore/"
        ),
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    args = _parse_args(argv)
    if args.smoke:
        args.out_dir = Path("/tmp/issue1739_eos_smoke/rescore")
        _log(
            f"SMOKE MODE: {SMOKE_N_CTX} contexts x {SMOKE_N_ARMS} arms "
            f"x {SMOKE_N_BOOT} boot x {SMOKE_N_PERM} perm "
            f"-> {args.out_dir}"
        )

    try:
        summary = _rescore_behavior(args)
        _log(
            f"done: {summary['n_metric_rows']} metric rows, "
            f"preds at {list(summary['preds_files'].values())}"
        )
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
