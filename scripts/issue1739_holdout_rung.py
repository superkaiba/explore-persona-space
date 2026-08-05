#!/usr/bin/env python3
"""#1739 evil-ood-spread round — item B part 2: coarse MHJ attack-type holdout rung (unit 1c).

Consumes the unit-1b tactic labels (``tactic_labels.json``: context_id -> one
of the 7 MHJ tactic classes; ``scripts/issue1739_tactic_classify.py``) plus
the SAME labeled activation store / DV dataset the committed sibling
``scripts/issue1739_rescore_ood.py`` consumes, then:

1. Reports every tactic cluster's n, DV SD and DV distribution shape
   (fraction at floor / ceiling / middle) — numbers only.
2. Selects the holdout cluster: the largest class with n > 200 AND DV SD > 5
   (plan section 4.3 step B2; ``--holdout-class`` overrides). No qualifying
   cluster -> item B is uninformative; recorded as a finding in
   ``cluster_stats.json`` (designed outcome, exit 0).
3. REFITS the all-16 arm battery (``ARM_REGISTRY`` minus the new-arm-round
   arm17/18) on the NON-holdout train contexts with the vectorized fit cores
   (``run_cell_multi`` — batched matrix ops, multi-regime shared rb-indep
   work), using group-level folds (N_FOLDS) for train-side frozen-layer
   selection, then scores the held-out cluster transfer-style (fold-2
   train-vs-eval cell; ``--ridge-folds`` mirrors the armfill sibling and
   defaults to ``all`` because the fixed all-16 roster carries
   ``arm10_stacked``, which needs ridge preds on EVERY fold).
4. Evaluates on the held-out cluster: Spearman rho(arm score, DV) per arm +
   detection metrics (AUROC / AP / precision@k, k=10,50; binarized at the
   bottom-bin boundary ``dv >= GATE2_BOTTOM_BIN_EDGE``) with N_BOOT=500
   bootstrap CIs (pre-drawn index matrices, batched draws) and an N_PERM=500
   selection-symmetric permutation null (DV permuted per draw BEFORE the
   max-over-arms selection; ``permutation_null_max``).

This is a NUMERIC fitting script: it consumes tactic-label strings,
per-context DV scores and activation summaries; it never handles corpus
text. Stdout / outputs reference contexts by context_id + counts only.

Launch (VM CPU; peak RSS ~45 GB fp64 per variant — one variant processed at
a time, arrays freed between variants):

.. code-block:: bash

    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue1739_holdout_rung.py \\
        --behavior evil \\
        --tactic-labels eval_results/issue_1739/evil_ood_spread/tactic_labels.json \\
        --dv-json eval_results/issue_1739/dv_dataset/evil/labeling.json \\
        --store-dir data/issue_1739/store/evil_labeling \\
        --tensors-root analysis_tensors/issue_1739 \\
        --text-emb data/issue_1739/inputs/evil_text_emb.npz \\
        --text-features data/issue_1739/inputs/evil_text_features.npz \\
        --output-dir eval_results/issue_1739/evil_ood_spread/holdout

Smoke (fully synthetic — no store, no network, no maps; 40 fake contexts x
2 arms x 5 boot/perm):

    uv run python scripts/issue1739_holdout_rung.py --smoke \\
        --output-dir /tmp/issue1739_eos_smoke/holdout

Outputs under ``--output-dir``:

- ``cluster_stats.json``  — per-cluster n / SD / shape + selection verdict.
- ``holdout_metrics.json`` — per-(variant, regime, arm) holdout metrics +
  permutation nulls + recorded arm skips.
- ``preds/tactic_holdout_preds.jsonl`` — per-context predictions (item C
  detection-metric input; ``transfer_preds_rows`` schema).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("issue1739_holdout_rung")


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
# constants / defaults (hyperparameters imported from constants.py in-function)
# ---------------------------------------------------------------------------

BEHAVIOR_DEFAULT = "evil"

# Plan section 4.3 step B2: largest cluster with n > 200 AND SD(DV) > 5.
HOLDOUT_MIN_N_DEFAULT = 200
HOLDOUT_MIN_SD_DEFAULT = 5.0
# Top bin of the 10-bin [0, 100] expression histogram (gates.py convention);
# the bottom edge comes from gates.GATE2_BOTTOM_BIN_EDGE (10.0).
CEILING_BIN_EDGE = 90.0

K_VALS = (10, 50)

VARIANTS_DEFAULT = ("context_end", "prefix_end")
REGIMES_DEFAULT = ("e1", "e2", "e2p")
U_LABEL_DEFAULT = "full"

DEFAULT_OUT_DIR = Path("eval_results/issue_1739/evil_ood_spread/holdout")
DEFAULT_TACTIC_LABELS = Path("eval_results/issue_1739/evil_ood_spread/tactic_labels.json")
DEFAULT_DV_JSON = Path("eval_results/issue_1739/dv_dataset/evil/labeling.json")
DEFAULT_STORE_DIR = Path("data/issue_1739/store/evil_labeling")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")

SMOKE_N_CTX = 40
SMOKE_N_ARMS = 2
SMOKE_N_BOOT = 5
SMOKE_N_PERM = 5
SMOKE_MIN_N = 5
SMOKE_MIN_SD = 0.5
SMOKE_N_LAYERS = 2
SMOKE_DIM = 8


def _log(msg: str) -> None:
    print(f"[holdout-rung] {msg}", flush=True)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# cluster stats + holdout selection (plan section 4.3 step B2)
# ---------------------------------------------------------------------------


def _cluster_stats(
    tactic_by_pos: list[str | None],
    dv: "np.ndarray",  # type: ignore[name-defined]
    *,
    floor_edge: float,
    ceiling_edge: float = CEILING_BIN_EDGE,
) -> list[dict]:
    """Per-tactic-class n, DV SD and distribution shape (numbers only)."""
    import numpy as np

    dv = np.asarray(dv, dtype=np.float64)
    classes = sorted({t for t in tactic_by_pos if t is not None})
    stats: list[dict] = []
    for cls in classes:
        mask = np.array([t == cls for t in tactic_by_pos], dtype=bool)
        d = dv[mask]
        n = int(mask.sum())
        stats.append(
            {
                "tactic_class": cls,
                "n": n,
                "dv_sd": float(d.std(ddof=1)) if n >= 2 else float("nan"),
                "dv_mean": float(d.mean()) if n else float("nan"),
                "frac_floor": float((d < floor_edge).mean()) if n else float("nan"),
                "frac_ceiling": float((d >= ceiling_edge).mean()) if n else float("nan"),
                "frac_middle": float(((d >= floor_edge) & (d < ceiling_edge)).mean())
                if n
                else float("nan"),
            }
        )
    return stats


def _select_holdout(
    stats: list[dict],
    *,
    min_n: int,
    min_sd: float,
    override: str | None,
) -> tuple[str | None, str]:
    """Pick the holdout class: override, else largest n among qualifying classes."""
    import math

    by_class = {s["tactic_class"]: s for s in stats}
    if override is not None:
        if override not in by_class:
            raise RuntimeError(
                f"--holdout-class {override!r} not among labeled classes "
                f"{sorted(by_class)} — refusing (fail fast, no silent fallback)"
            )
        return override, f"user override --holdout-class {override!r}"
    eligible = [
        s for s in stats if s["n"] > min_n and math.isfinite(s["dv_sd"]) and s["dv_sd"] > min_sd
    ]
    if not eligible:
        return None, (
            f"no tactic class satisfies n > {min_n} AND dv_sd > {min_sd} — "
            "item B is uninformative (plan section 4.3 step B2 finding)"
        )
    chosen = max(eligible, key=lambda s: s["n"])
    return chosen["tactic_class"], (
        f"largest class with n > {min_n} and dv_sd > {min_sd} "
        f"(n={chosen['n']}, dv_sd={chosen['dv_sd']:.2f})"
    )


# ---------------------------------------------------------------------------
# ridge-folds threading (mirrors scripts/issue1739_rescore_ood_armfill.py)
# ---------------------------------------------------------------------------


def _ridge_folds_arg(args: argparse.Namespace) -> tuple[int, ...] | None:
    """None when --ridge-folds all (the default here).

    arm10_stacked needs ridge predictions on EVERY fold; under the transfer
    leg's ``(0,)`` discarded-fold skip ``run_cell_multi`` RAISES, so arm10
    cannot be scored without ``all``. Outputs for the other arms are unchanged
    either way -- ``(0,)`` only avoids computing a fold whose predictions are
    discarded -- at the cost of one extra Gram+eigh per cell. Mirrors the
    armfill sibling's ``_ridge_folds_arg`` (which ran the OOD legs with
    ``--ridge-folds all``, the comparability pin for arm10 rows).
    """
    return None if getattr(args, "ridge_folds", "all") == "all" else (0,)


def _validate_ridge_folds_roster(roster: list[str], ridge_folds: tuple[int, ...] | None) -> None:
    """Fail at STARTUP on an arm10/ridge_folds mismatch, not after the CV pass.

    ``arms.run_cell_multi`` raises the same contract violation, but only when
    the TRANSFER pass reaches it -- 25+ min after launch (the 2026-08-05
    18:38:46Z rc=1: a full CV pass completed, then the transfer call died at
    arms.py's ``arm10_stacked needs ridge preds on EVERY fold``). This guard
    runs before any store/U-pool work so the incompatibility costs seconds.
    """
    if "arm10_stacked" in roster and ridge_folds is not None:
        raise RuntimeError(
            "arm10_stacked needs ridge preds on EVERY fold (no ridge_folds "
            "subset): pass --ridge-folds all (the default), or drop arm10 "
            "from the roster -- refusing at startup rather than after the CV pass"
        )


# ---------------------------------------------------------------------------
# frozen linear-map loader (main-grid persisted map; scripts/issue1739_fits._save_map)
# ---------------------------------------------------------------------------


def _load_linear_map(tensors_root: Path, variant: str, u_label: str, layers: list[int]):
    """Load the persisted plain-rung linear map -> MapFit (fail loud when absent).

    Path convention is ``_map_path`` in ``scripts/issue1739_fits.py``:
    ``<tensors_root>/maps/<variant>__u<u_label>.npz`` with fp16 ``w`` +
    fp32 standardizers, fit in the U-pool whitened space (whitening seed =
    ``SEEDS[0]`` — the map fit's own seed; see ``_run_production``).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    path = Path(tensors_root) / "maps" / f"{variant}__u{u_label}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"frozen linear map missing: {path} — stage the main-run maps/ tensors "
            "(upload --stage tensors artifact) or pass --no-map-arms for an "
            "explicit map-arm opt-out"
        )
    with np.load(path, allow_pickle=False) as z:
        w = np.asarray(z["w"], dtype=np.float64)
        x_mu = np.asarray(z["x_mu"], dtype=np.float64)
        x_sd = np.asarray(z["x_sd"], dtype=np.float64)
        y_mu = np.asarray(z["y_mu"], dtype=np.float64)
        stored_layers = [int(x) for x in z["layers"]]
        meta = json.loads(str(z["meta"]))
    if stored_layers != [int(x) for x in layers]:
        raise RuntimeError(
            f"map layer set mismatch at {path}: stored {stored_layers} vs requested "
            f"{list(layers)} — the holdout rung runs the full layer set only"
        )
    return fits.MapFit(w=w, x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, diagnostics=meta, kind="linear")


def _load_rb_raw(tensors_root: Path, regime: str, behavior: str, layers: list[int]):
    """Load the persisted regime direction (Ly, d) — fail loud when absent."""
    import numpy as np

    path = Path(tensors_root) / f"r_b_{regime}" / f"{behavior}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"regime direction missing: {path} — stage the main-run r_b_{regime}/ "
            "tensors, or drop the regime via --regimes"
        )
    with np.load(path, allow_pickle=False) as z:
        rb = np.asarray(z["rb"], dtype=np.float64)
        stored_layers = [int(x) for x in z["layers"]] if "layers" in z else list(range(rb.shape[0]))
    if stored_layers != [int(x) for x in layers]:
        raise RuntimeError(
            f"rb layer set mismatch at {path}: stored {stored_layers} vs requested {list(layers)}"
        )
    return rb


def _whiten_acts(z: "np.ndarray", wh) -> "np.ndarray":
    """(Ly, n, d) raw acts -> whitened fp64 via the canonical chunked helper.

    Delegates to ``fits.apply_whitening`` (bit-identity pinned by
    ``test_apply_whitening_chunked_matches_dense``). ``wh.mu`` is (Ly, d) —
    the former local ``z - wh.mu[None, None, :]`` broadcast (Ly, n, d)
    against (1, 1, Ly, d) and raised ValueError at every realistic shape
    (the exact bug class the sibling ``issue1739_rescore_ood.py`` fixed;
    this script's synthetic ``--smoke`` bypasses whitening, so the
    production shape first fired at the 2026-08-05 B3 pilot). The chunked
    helper also avoids the whole-array fp64 + centered temporaries (~45 GiB
    transient at production shape).
    """
    from explore_persona_space.experiments.issue_1739 import fits

    return fits.apply_whitening(z, wh)


# ---------------------------------------------------------------------------
# core fit + eval (shared by production and smoke)
# ---------------------------------------------------------------------------


def _fit_eval_variant(
    *,
    behavior: str,
    variant: str,
    z_w: "np.ndarray",  # type: ignore[name-defined]  # (Ly, n, d) whitened fp64
    za_w: "np.ndarray",  # type: ignore[name-defined]  # (Ly, n, d) whitened fp64
    dv: "np.ndarray",  # type: ignore[name-defined]  # (n,) graded 0-100
    groups: list[str],
    ctx_order: list[str],
    tactic_by_pos: list[str | None],
    hold_idx: "np.ndarray",  # type: ignore[name-defined]
    nonhold_idx: "np.ndarray",  # type: ignore[name-defined]
    rb_by_regime: dict[str, "np.ndarray"],  # type: ignore[name-defined]
    mapfit,
    text_emb,
    text_features,
    layers: tuple[int, ...],
    roster: list[str],
    rung: str,
    floor_edge: float,
    n_folds: int,
    n_boot: int,
    n_perm: int,
    seed: int,
    device: str,
    ridge_folds: tuple[int, ...] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """One variant: CV frozen-layer selection on non-holdout, transfer-score holdout.

    ``ridge_folds`` threads to the TRANSFER pass only (the CV pass always fits
    every fold); ``None`` (all folds) is REQUIRED whenever ``arm10_stacked``
    is in the roster -- validated at startup by ``_validate_ridge_folds_roster``.

    Returns ``(metric_rows, perm_rows, preds_rows, skip_rows)``. The frozen
    layer per (regime, arm) is selected on the NON-holdout OOF Spearman only
    (never on holdout outcome). All regimes share one ``run_cell_multi`` call
    per pass so the rb-independent fits (ridge / MLP / text arms) are computed
    once (batched cores; no per-cell serial Python loops).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_rescore_ood import _ap_score, _precision_at_k

    regimes = list(rb_by_regime)
    datas = [
        arms.CellData(
            z_ctx=z_w,
            z_ans=za_w,
            dv=dv,
            rb=rb_by_regime[r],
            mapfit=mapfit,
            text_emb=text_emb,
            text_features=text_features,
            layers=tuple(layers),
        )
        for r in regimes
    ]

    # --- CV pass on non-holdout rows: group-level folds -> frozen layer ---
    sub_groups = [groups[i] for i in nonhold_idx]
    cell_sub = fits.realize_budget_cell(
        sub_groups, budget_l=len(nonhold_idx), draw=0, seed=seed, n_folds=n_folds
    )
    cell_cv = fits.BudgetCell(
        row_idx=nonhold_idx[cell_sub.row_idx],
        fold_ids=cell_sub.fold_ids,
        n_folds=cell_sub.n_folds,
        budget_l=cell_sub.budget_l,
        draw=0,
        seed=seed,
        fold_scheme=cell_sub.fold_scheme,
    )
    _log(
        f"[{variant}] CV pass: {len(cell_cv.row_idx)} non-holdout rows, "
        f"{cell_cv.n_folds} group folds, {len(regimes)} regimes, {len(roster)} arms"
    )
    t0 = time.time()
    cv_out = arms.run_cell_multi(datas, cell_cv, arms=roster, device=device)
    _log(f"[{variant}] CV pass done in {time.time() - t0:.1f}s")
    dv_cv = np.asarray(dv, dtype=np.float64)[cell_cv.row_idx]

    frozen_by_regime: dict[str, dict[str, int]] = {}
    rho_pl_by_regime: dict[str, dict[str, list[float]]] = {}
    skip_rows: list[dict] = []
    for r, (scores_cv, skipped_cv) in zip(regimes, cv_out, strict=True):
        fro: dict[str, int] = {}
        rpl: dict[str, list[float]] = {}
        for slug, sc in scores_cv.items():
            rho_layers = arms.spearman_rows(np.asarray(sc, dtype=np.float64), dv_cv)
            rpl[slug] = [float(x) for x in rho_layers]
            fro[slug] = arms.frozen_layer_idx(rpl[slug])
        frozen_by_regime[r] = fro
        rho_pl_by_regime[r] = rpl
        for slug, reason in skipped_cv.items():
            skip_rows.append(
                {
                    "behavior": behavior,
                    "variant": variant,
                    "regime": r,
                    "arm": slug,
                    "stage": "cv",
                    "reason": reason,
                }
            )
    del cv_out
    gc.collect()

    # --- transfer pass: fit on non-holdout (fold 1), score holdout (fold 0) ---
    n_tr, n_ev = int(len(nonhold_idx)), int(len(hold_idx))
    cell_t = fits.BudgetCell(
        row_idx=np.concatenate([nonhold_idx, hold_idx]),
        fold_ids=np.concatenate([np.ones(n_tr, dtype=np.int64), np.zeros(n_ev, dtype=np.int64)]),
        n_folds=2,
        budget_l=n_tr,
        draw=0,
        seed=seed,
        fold_scheme="transfer-train-vs-eval",
    )
    _log(f"[{variant}] transfer pass: {n_tr} train rows -> {n_ev} holdout rows")
    t0 = time.time()
    tr_out = arms.run_cell_multi(datas, cell_t, arms=roster, device=device, ridge_folds=ridge_folds)
    _log(f"[{variant}] transfer pass done in {time.time() - t0:.1f}s")

    dv_ev = np.asarray(dv, dtype=np.float64)[hold_idx]
    labels_ev = dv_ev >= floor_edge
    ctx_ev = [ctx_order[i] for i in hold_idx]
    tactic_ev = [str(tactic_by_pos[i]) for i in hold_idx]
    group_ev = [groups[i] for i in hold_idx]

    metric_rows: list[dict] = []
    perm_rows: list[dict] = []
    preds_rows: list[dict] = []

    for r, (scores_all, skipped_tr) in zip(regimes, tr_out, strict=True):
        for slug, reason in skipped_tr.items():
            skip_rows.append(
                {
                    "behavior": behavior,
                    "variant": variant,
                    "regime": r,
                    "arm": slug,
                    "stage": "transfer",
                    "reason": reason,
                }
            )
        scores_ev = {
            slug: np.asarray(sc, dtype=np.float64)[:, n_tr:] for slug, sc in scores_all.items()
        }
        frozen = frozen_by_regime[r]
        provenance = {"behavior": behavior, "variant": variant, "regime": r, "rung": rung}

        sel_rows: list[np.ndarray] = []
        sel_slugs: list[str] = []
        for i, slug in enumerate(s for s in roster if s in scores_ev):
            sc = scores_ev[slug]
            fl = min(int(frozen[slug]), sc.shape[0] - 1)
            s1 = sc[fl]
            keep = np.isfinite(s1) & np.isfinite(dv_ev)
            n_k = int(keep.sum())
            if n_k < 3:
                skip_rows.append(
                    {
                        **provenance,
                        "arm": slug,
                        "stage": "metrics",
                        "reason": f"min_n: {n_k} finite holdout rows",
                    }
                )
                continue
            s_k, d_k, l_k = s1[keep], dv_ev[keep], labels_ev[keep]
            rho = float(arms.spearman_rows(s_k[None], d_k)[0])
            idx_b = arms.make_bootstrap_idx(n_k, n_boot=n_boot, seed=seed + i)
            draws = arms.bootstrap_rhos(s_k[None], d_k, idx_b)[0]
            row: dict[str, Any] = {
                **provenance,
                "arm": slug,
                "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                "n_eval": n_k,
                "n_train": n_tr,
                "rho": rho,
                "ci_rho": [
                    float(np.nanquantile(draws, 0.025)),
                    float(np.nanquantile(draws, 0.975)),
                ],
                "auroc": float(arms.auroc_rows(s_k[None], l_k)[0]),
                "ap": float(_ap_score(s_k, l_k)),
                "frozen_layer_idx": int(fl),
                "layer": int(layers[fl]) if layers and sc.shape[0] > 1 else None,
                "rho_per_layer_train": rho_pl_by_regime[r].get(slug),
            }
            for k in K_VALS:
                row[f"precision_at_{k}"] = float(_precision_at_k(s_k, l_k, k))
            metric_rows.append(row)
            sel_rows.append(s1)
            sel_slugs.append(slug)

        if sel_rows:
            mat = np.stack(sel_rows, axis=0)  # (A, n_ev)
            common = np.isfinite(dv_ev) & np.all(np.isfinite(mat), axis=0)
            if int(common.sum()) >= 3:
                null = arms.permutation_null_max(
                    mat[:, common], dv_ev[common], n_perm=n_perm, seed=seed
                )
                perm_rows.append(
                    {
                        **provenance,
                        **null,
                        "arms_in_selection": sel_slugs,
                        "n_common": int(common.sum()),
                    }
                )
            else:
                perm_rows.append(
                    {**provenance, "error": f"min_n: {int(common.sum())} common finite rows"}
                )

        preds_rows.extend(
            arms.transfer_preds_rows(
                scores_ev,
                dv_ev,
                ctx_ev,
                frozen,
                provenance=provenance,
                layers=tuple(layers),
                labels={"tactic": tactic_ev, "group_key": group_ev},
            )
        )

    del tr_out, datas
    gc.collect()
    return metric_rows, perm_rows, preds_rows, skip_rows


# ---------------------------------------------------------------------------
# output writing
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    return path


def _write_outputs(
    out_dir: Path,
    *,
    metric_rows: list[dict],
    perm_rows: list[dict],
    skip_rows: list[dict],
    preds_rows: list[dict],
    meta: dict,
) -> tuple[Path, Path]:
    from explore_persona_space.experiments.issue_1739 import arms

    metrics_path = _write_json(
        out_dir / "holdout_metrics.json",
        {
            "n_metric_rows": len(metric_rows),
            "metric_rows": metric_rows,
            "perm_nulls": perm_rows,
            "skipped_arms": skip_rows,
            "meta": meta,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    preds_path = arms.write_preds_jsonl(
        out_dir / "preds" / "tactic_holdout_preds.jsonl", preds_rows
    )
    _log(f"metrics -> {metrics_path} ({len(metric_rows)} rows)")
    _log(f"preds   -> {preds_path} ({len(preds_rows)} rows)")
    return metrics_path, preds_path


def _assert_nonempty(metric_rows: list[dict], preds_rows: list[dict]) -> None:
    """Empty output is a FAILURE, never a pass (silent-zero guard)."""
    if not metric_rows:
        raise RuntimeError("0 holdout metric rows produced — empty output is a failure")
    if not preds_rows:
        raise RuntimeError("0 per-context prediction rows produced — empty output is a failure")


# ---------------------------------------------------------------------------
# production path
# ---------------------------------------------------------------------------


def _run_production(args: argparse.Namespace) -> int:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits, mem_guard, store_io
    from explore_persona_space.experiments.issue_1739.constants import (
        N_BOOT,
        N_FOLDS,
        N_PERM,
        SEEDS,
        STORE_REVISION,
    )
    from explore_persona_space.experiments.issue_1739.gates import GATE2_BOTTOM_BIN_EDGE
    from scripts.issue1739_fits import _load_injected_features, _load_labeled, arrays_dim
    from scripts.issue1739_rescore_ood import _ALL16_NAMES, _probe_and_repin_store

    behavior: str = args.behavior
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dir = Path(args.store_dir)
    tensors_root = Path(args.tensors_root)
    n_boot = int(args.n_boot or N_BOOT)
    n_perm = int(args.n_perm or N_PERM)
    n_folds = int(args.n_folds or N_FOLDS)
    seed = int(args.seed)
    floor_edge = float(GATE2_BOTTOM_BIN_EDGE)

    # --- 1. tactic labels (unit-1b output) — fail loud on absence/emptiness ---
    tactic_path = Path(args.tactic_labels)
    payload = json.loads(tactic_path.read_text())
    labels: dict[str, str] = payload["labels"]
    if not labels:
        raise RuntimeError(f"{tactic_path} carries 0 labels — nothing to hold out")
    _log(f"tactic labels: {len(labels)} labeled contexts from {tactic_path}")

    # --- 2. labeled train table (config_a) — the SAME loader the sibling uses ---
    layers = list(range(store_io.N_LAYERS))
    dim = arrays_dim(store_dir, [layers[0]])
    _log("loading train labeled table (config_a) ...")
    tbl = _load_labeled(
        store_dir, Path(args.dv_json), layers, config="config_a", need_rollout_rows=False
    )
    _log(f"train table: {len(tbl.ctx_order)} contexts, rungs={tbl.rungs}")

    # --- 3. join + cluster stats + holdout selection ---
    tactic_by_pos: list[str | None] = [labels.get(c) for c in tbl.ctx_order]
    n_unlabeled = sum(t is None for t in tactic_by_pos)
    dv = np.asarray(tbl.dv, dtype=np.float64)
    stats = _cluster_stats(tactic_by_pos, dv, floor_edge=floor_edge)
    chosen, reason = _select_holdout(
        stats,
        min_n=int(args.min_cluster_n),
        min_sd=float(args.min_cluster_sd),
        override=args.holdout_class,
    )
    for s in stats:
        _log(
            f"cluster {s['tactic_class']!r}: n={s['n']} sd={s['dv_sd']:.2f} "
            f"floor={s['frac_floor']:.3f} ceil={s['frac_ceiling']:.3f} "
            f"mid={s['frac_middle']:.3f}"
        )
    hold_mask = np.array([t == chosen for t in tactic_by_pos], dtype=bool)
    hold_idx = np.flatnonzero(hold_mask)
    nonhold_idx = np.flatnonzero(~hold_mask)
    straddle = (
        len({tbl.groups[i] for i in hold_idx} & {tbl.groups[i] for i in nonhold_idx})
        if chosen is not None
        else 0
    )
    cluster_payload = {
        "behavior": behavior,
        "tactic_labels": str(tactic_path),
        "n_table_contexts": len(tbl.ctx_order),
        "n_labeled": len(tbl.ctx_order) - n_unlabeled,
        "n_unlabeled": n_unlabeled,
        "clusters": stats,
        "selection_rule": {
            "min_n": int(args.min_cluster_n),
            "min_sd": float(args.min_cluster_sd),
            "override": args.holdout_class,
        },
        "holdout_class": chosen,
        "holdout_selected": chosen is not None,
        "selection_reason": reason,
        "n_holdout": int(hold_idx.size),
        "n_train": int(nonhold_idx.size),
        "n_groups_straddling_split": straddle,
        "floor_edge": floor_edge,
        "ceiling_edge": CEILING_BIN_EDGE,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out_dir / "cluster_stats.json", cluster_payload)
    _log(f"cluster stats -> {out_dir / 'cluster_stats.json'}")
    if chosen is None:
        _log(f"FINDING: {reason} — no holdout fit run (designed outcome)")
        return 0
    rung = f"tactic_holdout_{_slugify(chosen)}"
    _log(f"holdout: {chosen!r} ({reason}) -> rung={rung}")

    # --- 4. text-control features (arms 15/16) — explicit opt-out only ---
    if args.no_text_arms:
        text_emb = text_features = None
        _log("EXPLICIT --no-text-arms: arms 15/16 will record skips")
    else:
        if args.text_emb is None or args.text_features is None:
            raise RuntimeError(
                "arms 15/16 need --text-emb and --text-features "
                "(scripts/issue1739_features.py output); pass --no-text-arms "
                "for an explicit opt-out"
            )
        text_emb = _load_injected_features(Path(args.text_emb), "emb", tbl.ctx_order, "--text-emb")
        text_features = _load_injected_features(
            Path(args.text_features), "features", tbl.ctx_order, "--text-features"
        )

    # --- 5. U-pool staging + whitening seed parity with the frozen map ---
    revision = _probe_and_repin_store(
        store_dir, args.store_revision or STORE_REVISION, behavior=behavior, smoke=False
    )
    _log("staging U-pool store ...")
    u_store_dir = store_io.stage_u_store(revision=revision)
    _log(f"U-pool staged at {u_store_dir}")

    roster = [a for a in arms.ARM_REGISTRY if a in set(_ALL16_NAMES)]
    _log(f"arm roster ({len(roster)}): {roster}")
    ridge_folds = _ridge_folds_arg(args)
    _validate_ridge_folds_roster(roster, ridge_folds)
    _log(
        f"ridge folds: {args.ridge_folds!r} -> {ridge_folds!r} (arm10-compatible: {ridge_folds is None})"
    )
    regimes = list(args.regimes)

    all_metric_rows: list[dict] = []
    all_perm_rows: list[dict] = []
    all_preds_rows: list[dict] = []
    all_skip_rows: list[dict] = []

    for variant in args.variants:
        _log(f"=== variant {variant} ===")
        u_arrays, u_meta = store_io.load_summaries(
            u_store_dir, (variant,), tuple(layers), hidden_dim=dim
        )
        pool_mask = store_io.fit_pool_mask(u_meta)
        u_x = np.stack([u_arrays[(variant, ly)][pool_mask] for ly in layers])
        del u_arrays
        gc.collect()
        _log(f"  U-pool shape: {u_x.shape}")
        # Pre-phase RSS guard (concern u1c-mem-guard): project this variant's
        # whitening-fit chunk temps + weight tensors + the two whitened fp64
        # labeled copies (z_w/za_w) vs live MemAvailable and refuse with the
        # designed rc (mem_guard.RSS_GUARD_RC) instead of a kernel OOM-kill.
        # map_fit=False: the linear map is LOADED (frozen), never fit here.
        mem_guard.check_phase(
            f"holdout_whitening[{variant}]",
            mem_guard.whitening_map_components(
                len(layers),
                int(u_x.shape[1]),
                dim,
                n_ctx=len(tbl.ctx_order),
                n_ev=0,
                map_fit=False,
            ),
            out_root=out_dir,
        )
        # Whitening seed MUST match the persisted map's fit space: the main
        # grid fits whitening with seed=SEEDS[0] (scripts/issue1739_fits.py
        # map_seed = int(args.seeds[0])), and the frozen map's standardizers
        # live in THAT whitened space.
        wh = fits.fit_whitening(u_x, device=args.device, seed=int(SEEDS[0]))
        del u_x
        gc.collect()
        _log("  whitening fitted (seed=SEEDS[0], map-space parity)")

        rb_by_regime = {
            r: np.einsum("ld,lde->le", _load_rb_raw(tensors_root, r, behavior, layers), wh.w)
            for r in regimes
        }
        mapfit = (
            None
            if args.no_map_arms
            else _load_linear_map(tensors_root, variant, args.u_label, layers)
        )
        if args.no_map_arms:
            _log("  EXPLICIT --no-map-arms: map arms will record skips")

        z_w = _whiten_acts(tbl.z_by_variant[variant], wh)
        za_w = _whiten_acts(tbl.z_ans, wh)
        _log(f"  whitened acts: z={z_w.shape} za={za_w.shape}")

        # Pre-phase RSS guard: run_cell_multi's per-cell z/za fp64 copies +
        # arm transients; the transfer cell spans ALL table rows (max cell).
        mem_guard.check_phase(
            f"holdout_grid[{variant}]",
            mem_guard.cell_solve_components(
                len(layers),
                len(tbl.ctx_order),
                dim,
                roster,
                has_map=mapfit is not None,
            ),
            out_root=out_dir,
        )
        m_rows, p_rows, pr_rows, s_rows = _fit_eval_variant(
            behavior=behavior,
            variant=variant,
            z_w=z_w,
            za_w=za_w,
            dv=dv,
            groups=list(tbl.groups),
            ctx_order=list(tbl.ctx_order),
            tactic_by_pos=tactic_by_pos,
            hold_idx=hold_idx,
            nonhold_idx=nonhold_idx,
            rb_by_regime=rb_by_regime,
            mapfit=mapfit,
            text_emb=text_emb,
            text_features=text_features,
            layers=tuple(layers),
            roster=roster,
            rung=rung,
            floor_edge=floor_edge,
            n_folds=n_folds,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
            device=args.device,
            ridge_folds=ridge_folds,
        )
        all_metric_rows.extend(m_rows)
        all_perm_rows.extend(p_rows)
        all_preds_rows.extend(pr_rows)
        all_skip_rows.extend(s_rows)
        del z_w, za_w, rb_by_regime, mapfit, wh
        gc.collect()

    _assert_nonempty(all_metric_rows, all_preds_rows)
    _write_outputs(
        out_dir,
        metric_rows=all_metric_rows,
        perm_rows=all_perm_rows,
        skip_rows=all_skip_rows,
        preds_rows=all_preds_rows,
        meta={
            "behavior": behavior,
            "rung": rung,
            "holdout_class": chosen,
            "n_holdout": int(hold_idx.size),
            "n_train": int(nonhold_idx.size),
            "variants": list(args.variants),
            "regimes": regimes,
            "roster": roster,
            "n_boot": n_boot,
            "n_perm": n_perm,
            "n_folds": n_folds,
            "seed": seed,
            "store_revision": revision,
            "ridge_folds": args.ridge_folds,
            "pos_rule": f"dv >= {floor_edge} (bottom-bin boundary, gates.py)",
            "no_map_arms": bool(args.no_map_arms),
            "no_text_arms": bool(args.no_text_arms),
            "smoke": False,
        },
    )
    return 0


# ---------------------------------------------------------------------------
# smoke path (fully synthetic — no store, no network, no maps)
# ---------------------------------------------------------------------------


def _run_smoke(args: argparse.Namespace) -> int:
    import numpy as np

    from explore_persona_space.experiments.issue_1739.gates import GATE2_BOTTOM_BIN_EDGE
    from scripts.issue1739_rescore_ood import _ALL16_NAMES

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floor_edge = float(GATE2_BOTTOM_BIN_EDGE)
    rng = np.random.default_rng(1739)

    n, n_ly, d = SMOKE_N_CTX, SMOKE_N_LAYERS, SMOKE_DIM
    layers = tuple(range(n_ly))
    ctx_order = [f"sctx{i:04d}" for i in range(n)]
    classes = [
        "Direct Request",
        "Obfuscation",
        "Hidden Intention Streamline",
        "Request Framing",
        "Injection",
        "Output Format",
        "Echoing",
    ]
    # one dominant class (the auto-selected holdout) + the rest round-robin
    tactic_by_pos: list[str | None] = [
        classes[0] if i < 12 else classes[1 + (i % (len(classes) - 1))] for i in range(n)
    ]
    groups = [f"g{i % 8}" for i in range(n)]
    dv = rng.uniform(0.0, 100.0, size=n)
    # Force BOTH detection classes inside the holdout cluster (first 12
    # positions): a few floor-side rows so AUROC/AP never degenerate to an
    # all-positive label vector in the smoke.
    dv[:4] = rng.uniform(0.0, floor_edge * 0.8, size=4)

    stats = _cluster_stats(tactic_by_pos, dv, floor_edge=floor_edge)
    chosen, reason = _select_holdout(
        stats, min_n=SMOKE_MIN_N, min_sd=SMOKE_MIN_SD, override=args.holdout_class
    )
    if chosen is None:
        raise RuntimeError(f"smoke: holdout selection failed unexpectedly — {reason}")
    rung = f"tactic_holdout_{_slugify(chosen)}"
    hold_mask = np.array([t == chosen for t in tactic_by_pos], dtype=bool)
    hold_idx = np.flatnonzero(hold_mask)
    nonhold_idx = np.flatnonzero(~hold_mask)
    _write_json(
        out_dir / "cluster_stats.json",
        {
            "behavior": "smoke_synthetic",
            "clusters": stats,
            "holdout_class": chosen,
            "holdout_selected": True,
            "selection_reason": reason,
            "n_holdout": int(hold_idx.size),
            "n_train": int(nonhold_idx.size),
            "smoke": True,
        },
    )
    _log(
        f"SMOKE: {n} synthetic contexts, holdout={chosen!r} "
        f"(n={int(hold_idx.size)}), {SMOKE_N_ARMS} arms, "
        f"{SMOKE_N_BOOT} boot / {SMOKE_N_PERM} perm"
    )

    roster = list(_ALL16_NAMES[:SMOKE_N_ARMS])
    metric_rows: list[dict] = []
    perm_rows: list[dict] = []
    preds_rows: list[dict] = []
    skip_rows: list[dict] = []
    for variant in ("context_end", "prefix_end"):
        z_w = rng.normal(size=(n_ly, n, d))
        za_w = rng.normal(size=(n_ly, n, d))
        rb_by_regime = {"e1": rng.normal(size=(n_ly, d))}
        m_rows, p_rows, pr_rows, s_rows = _fit_eval_variant(
            behavior="smoke_synthetic",
            variant=variant,
            z_w=z_w,
            za_w=za_w,
            dv=dv,
            groups=groups,
            ctx_order=ctx_order,
            tactic_by_pos=tactic_by_pos,
            hold_idx=hold_idx,
            nonhold_idx=nonhold_idx,
            rb_by_regime=rb_by_regime,
            mapfit=None,
            text_emb=None,
            text_features=None,
            layers=layers,
            roster=roster,
            rung=rung,
            floor_edge=floor_edge,
            n_folds=3,
            n_boot=SMOKE_N_BOOT,
            n_perm=SMOKE_N_PERM,
            seed=int(args.seed),
            device="cpu",
            ridge_folds=_ridge_folds_arg(args),
        )
        metric_rows.extend(m_rows)
        perm_rows.extend(p_rows)
        preds_rows.extend(pr_rows)
        skip_rows.extend(s_rows)

    _assert_nonempty(metric_rows, preds_rows)
    metrics_path, preds_path = _write_outputs(
        out_dir,
        metric_rows=metric_rows,
        perm_rows=perm_rows,
        skip_rows=skip_rows,
        preds_rows=preds_rows,
        meta={
            "behavior": "smoke_synthetic",
            "rung": rung,
            "holdout_class": chosen,
            "n_boot": SMOKE_N_BOOT,
            "n_perm": SMOKE_N_PERM,
            "roster": roster,
            "smoke": True,
        },
    )
    # re-read what was written — a truncated/empty file must fail the smoke
    written = json.loads(metrics_path.read_text())
    n_pred_lines = sum(1 for ln in preds_path.read_text().splitlines() if ln.strip())
    if written["n_metric_rows"] < 1 or n_pred_lines < 1:
        raise RuntimeError(
            f"smoke output empty on disk: {written['n_metric_rows']} metric rows, "
            f"{n_pred_lines} preds lines"
        )
    import math

    degenerate = [
        r["arm"]
        for r in written["metric_rows"]
        if not all(math.isfinite(r[k]) for k in ("rho", "auroc", "ap"))
    ]
    if degenerate:
        raise RuntimeError(f"smoke metric rows degenerate (non-finite rho/auroc/ap): {degenerate}")
    _log(
        f"SMOKE OK: {written['n_metric_rows']} metric rows, {n_pred_lines} preds rows -> {out_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", default=BEHAVIOR_DEFAULT)
    ap.add_argument("--tactic-labels", type=Path, default=DEFAULT_TACTIC_LABELS)
    ap.add_argument(
        "--holdout-class",
        "--holdout-tactic",
        dest="holdout_class",
        default=None,
        help="override the auto-selected holdout tactic class",
    )
    ap.add_argument("--dv-json", "--dv-labels", dest="dv_json", type=Path, default=DEFAULT_DV_JSON)
    ap.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--u-label", default=U_LABEL_DEFAULT)
    ap.add_argument("--store-revision", default=None)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS_DEFAULT))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES_DEFAULT))
    ap.add_argument(
        "--output-dir", "--output", dest="output_dir", type=Path, default=DEFAULT_OUT_DIR
    )
    ap.add_argument("--text-emb", type=Path, default=None, help="arms-15 npz (features script)")
    ap.add_argument(
        "--text-features", type=Path, default=None, help="arms-16 npz (features script)"
    )
    ap.add_argument(
        "--no-text-arms",
        action="store_true",
        help="EXPLICIT opt-out: run without arms 15/16 (recorded skips)",
    )
    ap.add_argument(
        "--no-map-arms",
        action="store_true",
        help="EXPLICIT opt-out: run without the frozen linear map (map arms record skips)",
    )
    ap.add_argument("--min-cluster-n", type=int, default=HOLDOUT_MIN_N_DEFAULT)
    ap.add_argument("--min-cluster-sd", type=float, default=HOLDOUT_MIN_SD_DEFAULT)
    ap.add_argument(
        "--ridge-folds",
        choices=("discarded-skip", "all"),
        default="all",
        help=(
            "'all' -> ridge_folds=None on the transfer pass, REQUIRED for "
            "arm10_stacked (the fixed all-16 roster carries it; armfill's OOD "
            "legs ran --ridge-folds all -- the arm10 comparability pin). "
            "'discarded-skip' -> (0,) is legal only for arm10-free rosters."
        ),
    )
    ap.add_argument("--n-boot", type=int, default=None, help="default: constants.N_BOOT")
    ap.add_argument("--n-perm", type=int, default=None, help="default: constants.N_PERM")
    ap.add_argument("--n-folds", type=int, default=None, help="default: constants.N_FOLDS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"synthetic smoke: {SMOKE_N_CTX} fake contexts x {SMOKE_N_ARMS} arms x "
            f"{SMOKE_N_BOOT} boot/perm; no store/network/maps"
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.experiments.issue_1739.mem_guard import (
        RSS_GUARD_RC,
        MemGuardRefusal,
    )

    args = _parse_args(argv)
    try:
        if args.smoke:
            return _run_smoke(args)
        return _run_production(args)
    except MemGuardRefusal as exc:
        # Designed halt (fits-CLI convention): report artifact + distinct rc —
        # never a kernel OOM-kill that loses the log tail, never a bare rc=1.
        print(f"[holdout][rss-guard] DESIGNED HALT rc={RSS_GUARD_RC}: {exc}", flush=True)
        return RSS_GUARD_RC
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
