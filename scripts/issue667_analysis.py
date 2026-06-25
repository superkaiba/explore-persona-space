#!/usr/bin/env python3
"""Issue #667 per-assumption CPU analysis runner — A3.6-A3.10 + B3 gate.

Reads the per-cell activation store (``eval_results/issue_667/analysis_tensors/
<behavior>/<source>_seed<S>/<target>_L<l>.npz``), #537's measured leakage
matrix ``G`` (``G_meta.json`` + ``G_tensor.npz``), and #658's ``sigma_c.pt`` /
``r_b.pt``; writes one JSON per assumption under ``eval_results/issue_667/``.

The B3 reduction unit test gates A3.9/A3.10: if it fails, the runner HALTs
before producing any A3.9/A3.10 number (plan §7 — a mis-implemented whitened
inverse otherwise manufactures a spurious "whitening wins").

Off-pod CPU (plan §9): all linear algebra over the HF-uploaded store, no GPU,
no model load. Reproducibility metadata (git commit, env, timestamp, pins) is
embedded in every output JSON (CLAUDE.md Reproducibility Requirements).

Usage::

    uv run python scripts/issue667_analysis.py \\
        --tensors-dir eval_results/issue_667/analysis_tensors \\
        --out-dir eval_results/issue_667 \\
        --behaviors em sycophancy fact --primary-layer 14
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from explore_persona_space.analysis.issue667 import (  # noqa: E402
    ALL_LAYERS,
    EXPECTED_G_META_GIT_COMMIT,
    EXPECTED_STORE_PROBE_POOL_HASH,
    G_META_LOCAL,
    G_TENSOR_PATH,
    HF_DATA_REPO,
    HIDDEN_SIZE,
    IN_SCOPE_BEHAVIORS,
    N_LAYERS,
    PRIMARY_LAYER,
    R_B_PATH,
    RB_COLUMN_FOR_BEHAVIOR,
    RB_RECIPE,
    SIGMA_C_LAMBDA_FRACTION,
    SIGMA_C_PATH,
    STORE_MANIFEST_PATH,
)
from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    a37_source_write,
    clustered_bootstrap_partial_spearman,
    clustered_bootstrap_spearman,
    default_lambda,
    family_of,
    key_query_drift,
    lambda_condition_sweep,
    oracle_gplus,
    partial_shuffled_null_ci,
    partial_spearman,
    predict_mean_baseline,
    readout_projection,
    realized_gate,
    shuffled_corr_null_ci,
    spearman_rho,
    stacked_delta_svd,
    true_cosine,
    whitened_gate_metric,
    whitened_gate_reduction_unit_test,
)

logger = logging.getLogger("issue667_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility metadata
# ─────────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _repro_meta(extra: dict | None = None) -> dict:
    meta = {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "script": "issue667_analysis",
        "g_meta_git_commit_pin": EXPECTED_G_META_GIT_COMMIT,
        "store_probe_pool_hash_pin": EXPECTED_STORE_PROBE_POOL_HASH,
    }
    if extra:
        meta.update(extra)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Reused-artifact loaders (sha-pinned)
# ─────────────────────────────────────────────────────────────────────────────


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def load_g_meta() -> dict:
    """#537 G_meta.json (per-cell g/base_rate/...); assert the git_commit pin.

    G_meta is committed in git (``eval_results/issue_537/G_tensor/G_meta.json``,
    on ``main`` and inherited by the ``issue-667`` branch the pod checks out) —
    it is NOT on the HF data repo. Fail loud if missing (a sparse worktree that
    excludes ``eval_results/`` must ``git sparse-checkout add`` it for a local
    smoke; the pod's full checkout always has it).
    """
    p = PROJECT_ROOT / G_META_LOCAL
    if not p.exists():
        raise FileNotFoundError(
            f"G_meta.json not found at {p}. It is committed in git "
            f"({G_META_LOCAL}), NOT on HF. On a sparse worktree run "
            "`git sparse-checkout add eval_results/issue_537/G_tensor`; the pod's "
            "full checkout has it by default."
        )
    m = json.loads(p.read_text())
    gc = m.get("git_commit")
    assert gc == EXPECTED_G_META_GIT_COMMIT, (
        f"G_meta git_commit pin drift: {gc} != {EXPECTED_G_META_GIT_COMMIT} (#537 ground truth)"
    )
    return m


def load_g_tensor() -> dict:
    """#537 G_tensor.npz (G[5,16,30,1] + masks + train/eval cids)."""
    p = Path(_hf(G_TENSOR_PATH))
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def assert_store_pin() -> None:
    """Assert #658 store_manifest probe_pool_hash pin (the load-bearing pin)."""
    p = Path(_hf(STORE_MANIFEST_PATH))
    m = json.loads(p.read_text())
    pph = m.get("probe_pool_hash")
    assert pph == EXPECTED_STORE_PROBE_POOL_HASH, (
        f"#658 store probe_pool_hash pin drift: {pph} != {EXPECTED_STORE_PROBE_POOL_HASH}"
    )


def load_sigma_c_dict() -> dict:
    """#658 sigma_c.pt full payload ({sigma_c, n, capture_layers}) for validation."""
    import torch

    return torch.load(Path(_hf(SIGMA_C_PATH)), weights_only=False, map_location="cpu")


def sigma_c_at_layer(sig_dict: dict, layer: int):
    """Slice the (28, 3584, 3584) sigma_c to (3584, 3584) at ``layer``."""
    sig = sig_dict["sigma_c"]
    cap = list(sig_dict["capture_layers"])
    assert layer in cap, (layer, cap)
    return sig[cap.index(layer)]


def load_sigma_c(layer: int):
    """#658 sigma_c.pt -> (3584, 3584) at ``layer`` (the model-level second moment)."""
    return sigma_c_at_layer(load_sigma_c_dict(), layer)


def load_r_b(behavior: str, layer: int) -> np.ndarray | None:
    """#658 r_b.pt[<col>][diffmeans][layer] for in-scope behaviors (None for fact)."""
    import torch

    col = RB_COLUMN_FOR_BEHAVIOR.get(behavior)
    if col is None:
        return None  # fact (absent from #658) -> re-extracted in the store
    d = torch.load(Path(_hf(R_B_PATH)), weights_only=False, map_location="cpu")
    return d["r_b"][col][RB_RECIPE][layer].float().numpy().astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Cached-artifact coverage validators (BLOCKER 3 — fail-loud, never silent)
# ─────────────────────────────────────────────────────────────────────────────


class CoverageError(RuntimeError):
    """Raised when a cached artifact is missing a required column/key/shape/cid.

    A dedicated type so the dispatcher's validate phase + the unit tests can
    assert the fail-loud behavior without catching unrelated RuntimeErrors.
    """


def validate_r_b_coverage(behaviors: list[str], layers: list[int]) -> None:
    """Assert #658 ``r_b.pt`` carries every (column, recipe, layer) the run needs.

    For each in-scope behavior with a #658 r_b column (em→broad_em,
    sycophancy→sycophancy; fact is re-extracted in the store, skipped here):
    the column exists, the ``diffmeans`` recipe exists, and the tensor covers
    every requested layer index. HALT on any miss (BLOCKER 3).
    """
    import torch

    cols_needed = {
        b: RB_COLUMN_FOR_BEHAVIOR.get(b) for b in behaviors if RB_COLUMN_FOR_BEHAVIOR.get(b)
    }
    if not cols_needed:
        return
    d = torch.load(Path(_hf(R_B_PATH)), weights_only=False, map_location="cpu")
    rb = d.get("r_b")
    if rb is None:
        raise CoverageError(f"#658 r_b.pt missing top-level 'r_b' key (keys={list(d.keys())})")
    for behavior, col in cols_needed.items():
        if col not in rb:
            raise CoverageError(
                f"r_b.pt missing column {col!r} for behavior {behavior!r} "
                f"(present: {sorted(rb.keys())})"
            )
        if RB_RECIPE not in rb[col]:
            raise CoverageError(
                f"r_b.pt[{col!r}] missing recipe {RB_RECIPE!r} (present: {sorted(rb[col].keys())})"
            )
        tensor = rb[col][RB_RECIPE]
        n_rows = int(tensor.shape[0])
        bad = [li for li in layers if not (0 <= li < n_rows)]
        if bad:
            raise CoverageError(
                f"r_b.pt[{col!r}][{RB_RECIPE!r}] has {n_rows} layer rows; "
                f"requested layers {bad} out of range"
            )


def validate_sigma_c_coverage(sig_dict: dict, layers: list[int]) -> None:
    """Assert #658 ``sigma_c.pt`` shape/keys/capture_layers cover the run (BLOCKER 3).

    Expects keys ``sigma_c`` (shape ``(N_LAYERS, HIDDEN, HIDDEN)``), ``n``, and
    ``capture_layers`` ⊇ the requested layers. HALT on any miss.
    """
    for k in ("sigma_c", "n", "capture_layers"):
        if k not in sig_dict:
            raise CoverageError(
                f"sigma_c.pt missing required key {k!r} (present: {sorted(sig_dict.keys())})"
            )
    sig = sig_dict["sigma_c"]
    shape = tuple(int(s) for s in sig.shape)
    if shape != (N_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE):
        raise CoverageError(
            f"sigma_c.pt shape {shape} != expected {(N_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE)}"
        )
    cap = list(sig_dict["capture_layers"])
    missing = [li for li in layers if li not in cap]
    if missing:
        raise CoverageError(f"sigma_c.pt capture_layers missing requested layers {missing}")


def validate_g_meta_coverage(
    g_meta: dict, cells_by_beh: dict[str, dict[tuple[str, str], dict]]
) -> None:
    """Assert ``G_meta.json`` per_cell carries g/base_rate/noise_var for every in-scope cell.

    Every (behavior, source, target) actually present in the extraction store
    must have a G_meta per_cell record with the load-bearing fields
    (``g``, ``base_rate``, ``noise_var_bootstrap``). HALT on any missing cell or
    missing field (BLOCKER 3 — never silently shrink the denominator).
    """
    per_cell = g_meta.get("per_cell")
    if per_cell is None:
        raise CoverageError("G_meta.json missing 'per_cell'")
    required_fields = ("g", "base_rate", "noise_var_bootstrap")
    missing_cells: list[str] = []
    missing_fields: list[str] = []
    for behavior, cells in cells_by_beh.items():
        for source, target in cells:
            key = f"{behavior}/{source}__{target}"
            rec = per_cell.get(key)
            if rec is None:
                missing_cells.append(key)
                continue
            for f in required_fields:
                if f not in rec:
                    missing_fields.append(f"{key}:{f}")
    if missing_cells:
        raise CoverageError(
            f"G_meta.json missing per_cell records for {len(missing_cells)} extracted cells "
            f"(e.g. {missing_cells[:5]})"
        )
    if missing_fields:
        raise CoverageError(
            f"G_meta.json per_cell missing required fields (e.g. {missing_fields[:5]})"
        )


def validate_cid_coverage(
    g_tensor: dict, cells_by_beh: dict[str, dict[tuple[str, str], dict]]
) -> None:
    """Assert the EXTRACTED source/target cid sets ⊆ G_tensor train/eval_cids (BLOCKER 3).

    ``G_tensor.npz`` carries per-behavior ``train_cids`` (16) / ``eval_cids``
    (30). Every realized source cid in the store must be a registered train cid
    and every realized target a registered eval cid (or the source diagonal),
    so the analysis reads the SAME axes as G. HALT on any cid-set mismatch.
    """
    behs = [str(b) for b in g_tensor["behaviors"].tolist()]
    train_cids = g_tensor["train_cids"]
    eval_cids = g_tensor["eval_cids"]
    bad: list[str] = []
    for behavior, cells in cells_by_beh.items():
        if behavior not in behs:
            # marker / supplement behaviors may not be a headline G row; skip.
            continue
        bi = behs.index(behavior)
        train_set = {str(c) for c in train_cids[bi].tolist()}
        eval_set = {str(c) for c in eval_cids[bi].tolist()}
        for source, target in cells:
            if source not in train_set:
                bad.append(f"{behavior}: source {source!r} not in G_tensor train_cids")
            if target not in eval_set and target not in train_set:
                bad.append(f"{behavior}: target {target!r} not in G_tensor eval_cids")
    if bad:
        raise CoverageError(
            f"extracted cid set diverges from G_tensor train/eval_cids "
            f"({len(bad)} mismatches, e.g. {bad[:5]})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell store loading
# ─────────────────────────────────────────────────────────────────────────────


def load_cells(tensors_dir: Path, behavior: str, layer: int) -> dict[tuple[str, str], dict]:
    """{(source_cid, target_cid): npz_dict} for one (behavior, layer) from the store."""
    out: dict[tuple[str, str], dict] = {}
    beh_dir = tensors_dir / behavior
    if not beh_dir.exists():
        return out
    for cell_dir in sorted(beh_dir.glob("*_seed*")):
        source_cid = cell_dir.name.rsplit("_seed", 1)[0]
        for npz in sorted(cell_dir.glob(f"*_L{layer}.npz")):
            target_cid = npz.name.rsplit(f"_L{layer}.npz", 1)[0]
            data = dict(np.load(npz, allow_pickle=True))
            out[(source_cid, target_cid)] = data
    return out


def g_cell(g_meta: dict, behavior: str, source: str, target: str) -> dict | None:
    """#537 per-cell ground truth: {g, base_rate, noise_var_bootstrap, saturated, ...}."""
    return g_meta["per_cell"].get(f"{behavior}/{source}__{target}")


# ─────────────────────────────────────────────────────────────────────────────
# A3.6 — base read-out predicts the post-FT behavior CHANGE (partial corr, C10)
# ─────────────────────────────────────────────────────────────────────────────


def run_a36(cells_by_beh: dict, g_meta: dict, r_b_by_beh: dict, layer: int) -> dict:
    """A3.6: partial-Spearman(r_B'^T Δv(C'), E+ - E0 | E0) per behavior."""
    results = {}
    for behavior, cells in cells_by_beh.items():
        r_b = r_b_by_beh.get(behavior)
        # fact: read the re-extracted r_b from the store (primary-layer cell payloads).
        if r_b is None:
            r_b = _fact_rb_from_store(cells)
        if r_b is None:
            results[behavior] = {"status": "no_r_b", "note": "r_B unavailable for this behavior"}
            continue
        xs, ys, zs, fams, n_dyn = [], [], [], [], 0
        # group by source: for each source, vary target C'.
        for (source, target), data in cells.items():
            if source == target:
                continue  # off-diagonal targets only (the CHANGE read)
            gc = g_cell(g_meta, behavior, source, target)
            # G_meta coverage is asserted upstream (validate_g_meta_coverage) for
            # real runs; a None here is a genuine bug, not a denominator to shrink.
            if gc is None:
                raise CoverageError(
                    f"A3.6: no G_meta cell for {behavior}/{source}__{target} "
                    "(coverage validation should have caught this)"
                )
            delta_v = data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64)
            xs.append(readout_projection(r_b, delta_v))
            ys.append(float(gc["g"]))  # E+ - E0 == g
            zs.append(float(gc["base_rate"]))  # E0
            fams.append(family_of(target))
            if abs(float(gc["g"])) > 0.01:
                n_dyn += 1
        if len(xs) < 3:
            results[behavior] = {"status": "insufficient_cells", "n": len(xs)}
            continue
        x = np.array(xs)
        y = np.array(ys)
        z = np.array(zs)
        partial = partial_spearman(x, y, z)
        # MAJOR 3: bootstrap + null on the PARTIAL statistic (the headline), not
        # raw Spearman / a level-subtracted null.
        partial_null = partial_shuffled_null_ci(x, y, z)
        partial_boot = clustered_bootstrap_partial_spearman(x, y, z, fams)
        results[behavior] = {
            "status": "ok",
            "partial_spearman_change_given_base": partial,
            "raw_spearman_proj_vs_g": spearman_rho(x, y),
            "partial_shuffled_null_hi": partial_null["null_hi"],
            "partial_clustered_bootstrap": partial_boot,
            # legacy raw reads kept as secondary diagnostics (not the headline).
            "raw_clustered_bootstrap": clustered_bootstrap_spearman(x, y, fams),
            "n_cells": len(xs),
            "n_dynamic_range_cells": n_dyn,
            "dynamic_range_fraction": n_dyn / len(xs),
            "r_b_source": "reextracted_fact"
            if behavior == "fact"
            else RB_COLUMN_FOR_BEHAVIOR[behavior],
        }
    return {"assumption": "A3.6", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


def _fact_rb_from_store(cells: dict) -> np.ndarray | None:
    for _key, data in cells.items():
        if "r_b_fact" in data:
            return data["r_b_fact"].astype(np.float64)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# A3.7 — source write ŵ∥δ (cos(w_hat, delta_pos) vs shuffled-δ null)
# ─────────────────────────────────────────────────────────────────────────────


def run_a37(cells_by_beh: dict, layer: int) -> dict:
    """A3.7: per source, cos(w_hat, delta_pos/delta_contra) + frac_ctx + shuffled-δ null."""
    # w_hat per (behavior, source) = v+(C) - v0(C) at the source diagonal.
    w_hats: dict[tuple[str, str], np.ndarray] = {}
    t_pos: dict[tuple[str, str], np.ndarray] = {}
    t_neg: dict[tuple[str, str], np.ndarray] = {}
    v0_src: dict[tuple[str, str], np.ndarray] = {}
    for behavior, cells in cells_by_beh.items():
        for (source, target), data in cells.items():
            if source != target:
                continue
            w_hats[(behavior, source)] = data["v_plus"].astype(np.float64) - data["v0"].astype(
                np.float64
            )
            v0_src[(behavior, source)] = data["v0"].astype(np.float64)
            if "t_pos" in data:
                t_pos[(behavior, source)] = data["t_pos"].astype(np.float64)
            if "t_neg" in data:
                t_neg[(behavior, source)] = data["t_neg"].astype(np.float64)
    rng = np.random.default_rng(0)
    results = {}
    for behavior in cells_by_beh:
        rows = []
        # Sources realized in the store for this behavior (the A3.7 denominator).
        src_for_beh = [s for (b, s) in w_hats if b == behavior]
        dropped_no_tpos: list[str] = []
        # shuffled-δ null: cos(w_hat, delta_pos of a DIFFERENT behavior's source).
        other_deltas = [t_pos[k] - v0_src[k] for k in t_pos if k[0] != behavior and k in v0_src]
        for (b, source), w in w_hats.items():
            if b != behavior:
                continue
            dp = t_pos.get((b, source))
            tn = t_neg.get((b, source))
            if dp is None:
                # F3-ICL or any source where the t+/t- split found no positive
                # rows — dropped from A3.7 NON-silently (CONCERN a37-icl-...).
                dropped_no_tpos.append(source)
                continue
            delta_pos = dp - v0_src[(b, source)]
            t_neg_missing = tn is None
            # MINOR: when t_neg is absent, the contrastive read is UNDEFINED —
            # emit NaN δ_contra / v0_cneg so cos_contra + frac_ctx are NaN
            # (nanmean drops them), never a silent 0/duplicate-of-pos.
            nan_vec = np.full_like(w, np.nan)
            delta_contra = (dp - tn) if tn is not None else nan_vec
            v0_cneg = tn if tn is not None else nan_vec
            # MINOR: seeded random other-behavior δ, not a deterministic `% len`.
            other = (
                other_deltas[rng.integers(len(other_deltas))] if other_deltas else np.zeros_like(w)
            )
            row = {
                "source": source,
                "t_neg_missing": bool(t_neg_missing),
                **a37_source_write(w, delta_pos, delta_contra, other, v0_src[(b, source)], v0_cneg),
            }
            rows.append(row)
        if not rows:
            results[behavior] = {
                "status": "no_source_cells",
                "n_sources_total": len(src_for_beh),
                "dropped_no_tpos": dropped_no_tpos,
            }
            continue
        cos_pos = [r["cos_pos"] for r in rows]
        cos_null = [r["cos_null"] for r in rows]
        results[behavior] = {
            "status": "ok",
            "per_source": rows,
            "mean_cos_pos": float(np.mean(cos_pos)),
            "mean_cos_contra": float(np.nanmean([r["cos_contra"] for r in rows])),
            "mean_cos_null": float(np.mean(cos_null)),
            "mean_frac_ctx": float(np.nanmean([r["frac_ctx"] for r in rows])),
            "beats_null": bool(np.mean(cos_pos) > np.mean(cos_null)),
            "n_sources": len(rows),
            # Explicit A3.7 coverage (CONCERN a37-icl-source-tpos-tneg-gap):
            # the F3-ICL sources that produced no t+/t- split are reported as a
            # dropped denominator, not silently absent.
            "n_sources_total": len(src_for_beh),
            "n_sources_with_tpos": len(rows),
            "dropped_no_tpos": dropped_no_tpos,
            "n_sources_with_tneg": int(sum(not r["t_neg_missing"] for r in rows)),
        }
    return {"assumption": "A3.7", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


# ─────────────────────────────────────────────────────────────────────────────
# A3.8 — off-source change = scalar-gated source write (rank-one + SVD)
# ─────────────────────────────────────────────────────────────────────────────


def run_a38(cells_by_beh: dict, layer: int) -> dict:
    """A3.8: per source, rank-one residual + stacked-ΔV SVD (per behavior, #637)."""
    results = {}
    for behavior, cells in cells_by_beh.items():
        # group targets by source
        by_source: dict[str, list[tuple[str, dict]]] = {}
        diag: dict[str, np.ndarray] = {}
        for (source, target), data in cells.items():
            by_source.setdefault(source, []).append((target, data))
            if source == target:
                diag[source] = data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64)
        src_rows = []
        for source, targs in by_source.items():
            if source not in diag:
                continue
            w_hat = diag[source]
            if float(w_hat @ w_hat) <= 0:
                continue
            residuals, gates, deltas = [], [], []
            for target, data in targs:
                if target == source:
                    continue
                # ĝ^real + rank-one residual use the source DIAGONAL write w_hat.
                g_real, resid = _gate_for(cells, source, target, data)
                residuals.append(resid)
                gates.append(g_real)
                deltas.append(data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64))
            if len(deltas) < 2:
                continue
            svd = stacked_delta_svd(np.stack(deltas), w_hat)
            src_rows.append(
                {
                    "source": source,
                    "mean_rank_one_residual": float(np.mean(residuals)),
                    "median_realized_gate": float(np.median(gates)),
                    **svd,
                }
            )
        results[behavior] = {
            "status": "ok" if src_rows else "no_sources",
            "per_source": src_rows,
            "note": "per-behavior, never aggregated over the #637 content-behavior failure",
        }
    return {"assumption": "A3.8", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


def _gate_for(cells: dict, source: str, target: str, data: dict) -> tuple[float, float]:
    """ĝ^real + rank-one residual for (source -> target) using the diagonal source write."""
    src = cells[(source, source)]
    return realized_gate(src["v0"], src["v_plus"], data["v0"], data["v_plus"])


# ─────────────────────────────────────────────────────────────────────────────
# A3.9 / A3.10 — base key-query gate predicts the realized gate (B3-gated)
# ─────────────────────────────────────────────────────────────────────────────


def _a39_cell_rows(cells: dict, g_meta: dict, behavior: str) -> list[dict]:
    """Collect per-(source,target) gate rows with every key + the oracle inputs.

    Each row carries: realized gate ``g_real``; base key/query ``c_C``/``c_Cp``;
    post-FT key/query ``c_C_postft``/``c_Cp_postft`` (BLOCKER 1 oracle inputs);
    the source answer-profile keys ``psi_t = t_pos`` and ``psi_delta =
    t_pos - t_neg`` (MAJOR 1, co-layer extraction, ψ=identity); and ``E0`` (the
    base rate from G_meta, the C7 baseline). Skips a source with a zero-norm
    diagonal write (saturated/rank-collapsed) — reported via the count gap.
    """
    rows: list[dict] = []
    for (source, target), data in cells.items():
        if target == source or (source, source) not in cells:
            continue
        try:
            g_real, _ = _gate_for(cells, source, target, data)
        except ValueError:
            continue
        src = cells[(source, source)]
        row = {
            "source": source,
            "target": target,
            "g_real": g_real,
            "c_C": src["c_C"].astype(np.float64),
            "c_Cp": data["c_Cp"].astype(np.float64),
        }
        # Post-FT key/query (oracle g+). Present in the v2 store; absent on the
        # legacy round-1 store, in which case A3.10 reports oracle as unavailable.
        if "c_C_postft" in src and "c_Cp_postft" in data:
            row["c_C_postft"] = src["c_C_postft"].astype(np.float64)
            row["c_Cp_postft"] = data["c_Cp_postft"].astype(np.float64)
        # Answer-profile keys ψ(t), ψ(δ) — source-level, co-layer (MAJOR 1).
        if "t_pos" in src:
            row["psi_t"] = src["t_pos"].astype(np.float64)
            if "t_neg" in src:
                row["psi_delta"] = (src["t_pos"] - src["t_neg"]).astype(np.float64)
        # E0 base-prior baseline (C7).
        if g_meta is not None:
            gc = g_cell(g_meta, behavior, source, target)
            if gc is not None:
                row["E0"] = float(gc["base_rate"])
        rows.append(row)
    return rows


def _gate_pred_vec(rows: list[dict], key: str, metric: str, sigma_c, lam: float) -> np.ndarray:
    """Gate prediction over rows for one (key ∈ {c_C, psi_t, psi_delta}, metric).

    The KEY is the source-side vector (``c_C`` context, ``psi_t``/``psi_delta``
    answer-profile); the QUERY is always the target context vector ``c_Cp``.
    Rows lacking the requested key produce NaN (nan-safe Spearman downstream).
    """
    import torch

    out = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        kvec = r.get(key)
        if kvec is None:
            out[i] = np.nan
            continue
        try:
            out[i] = whitened_gate_metric(
                torch.from_numpy(kvec), torch.from_numpy(r["c_Cp"]), metric, sigma_c, lam
            )
        except ValueError:
            out[i] = np.nan
    return out


def _nan_safe_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Spearman over the rows where pred is finite; returns (rho, n_used)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), n
    return float(spearman_rho(pred[mask], target[mask])), n


def run_a39_a310(
    cells_by_beh: dict, sigma_c, layer: int, g_meta: dict | None = None
) -> tuple[dict, dict]:
    """A3.9 full key×metric grid + controls/baselines; A3.10 base-vs-oracle gate.

    B3-gated upstream. A3.9 (MAJOR 1/2/4): keys {c_C, ψ(t), ψ(δ)} × metrics
    {I, diag, whitened}; true-cosine baseline (MAJOR 2); shuffled-key/query
    controls; E0 + predict-mean baselines; λ condition-number sweep (MAJOR 4).
    A3.10 (BLOCKER 1): oracle g+ = (k+, q+, M0) vs realized, g0 vs realized
    (distinct from oracle), g0 vs oracle (base-gate validity), key/query drift.
    """
    import torch

    lam = default_lambda(sigma_c, SIGMA_C_LAMBDA_FRACTION)
    lam_sweep = lambda_condition_sweep(sigma_c)
    a39 = {}
    a310 = {}
    for behavior, cells in cells_by_beh.items():
        rows = _a39_cell_rows(cells, g_meta, behavior)
        if len(rows) < 3:
            a39[behavior] = {"status": "insufficient_cells", "n": len(rows)}
            a310[behavior] = {"status": "insufficient_cells", "n": len(rows)}
            continue
        g_real = np.array([r["g_real"] for r in rows])
        fams = [family_of(r["target"]) for r in rows]

        # ── A3.9: 3×3 key×metric grid ────────────────────────────────────────
        grid = {}
        for key in ("c_C", "psi_t", "psi_delta"):
            grid[key] = {}
            for metric in ("I", "diag", "whitened"):
                gp = _gate_pred_vec(rows, key, metric, sigma_c, lam)
                rho, n_used = _nan_safe_spearman(gp, g_real)
                grid[key][metric] = {"spearman": rho, "n_used": n_used}
        # boxed primary = c_C key + whitened metric (C3).
        boxed = grid["c_C"]["whitened"]["spearman"]
        # cosine baseline = TRUE cosine(c_C, c_Cp) (MAJOR 2), NOT the self-norm I metric.
        cos_pred = np.array(
            [true_cosine(torch.from_numpy(r["c_C"]), torch.from_numpy(r["c_Cp"])) for r in rows]
        )
        cos_rho, _ = _nan_safe_spearman(cos_pred, g_real)
        self_norm_I = grid["c_C"]["I"]["spearman"]  # the OLD mislabeled "cosine" (kept, renamed)

        # boxed-primary clustered bootstrap + shuffled null.
        gp_boxed = _gate_pred_vec(rows, "c_C", "whitened", sigma_c, lam)
        boxed_boot = clustered_bootstrap_spearman(gp_boxed, g_real, fams)
        boxed_null = shuffled_corr_null_ci(gp_boxed, g_real)

        # shuffled-key / shuffled-query controls on the boxed primary.
        rng = np.random.default_rng(layer)
        perm = rng.permutation(len(rows))
        shuf_key = np.array(
            [
                whitened_gate_metric(
                    torch.from_numpy(rows[perm[i]]["c_C"]),
                    torch.from_numpy(rows[i]["c_Cp"]),
                    "whitened",
                    sigma_c,
                    lam,
                )
                for i in range(len(rows))
            ]
        )
        shuf_query = np.array(
            [
                whitened_gate_metric(
                    torch.from_numpy(rows[i]["c_C"]),
                    torch.from_numpy(rows[perm[i]]["c_Cp"]),
                    "whitened",
                    sigma_c,
                    lam,
                )
                for i in range(len(rows))
            ]
        )
        shuf_key_rho, _ = _nan_safe_spearman(shuf_key, g_real)
        shuf_query_rho, _ = _nan_safe_spearman(shuf_query, g_real)

        # E0 base-prior baseline (C7) + predict-mean baseline.
        e0_vals = np.array([r.get("E0", np.nan) for r in rows])
        e0_rho, e0_n = _nan_safe_spearman(e0_vals, g_real)
        pred_mean = predict_mean_baseline(g_real)

        # λ condition-number sweep (MAJOR 4): boxed-primary spearman per ridge.
        lam_records = []
        for rec in lam_sweep:
            gp_l = _gate_pred_vec(rows, "c_C", "whitened", sigma_c, rec["lambda"])
            rho_l, _ = _nan_safe_spearman(gp_l, g_real)
            lam_records.append({**rec, "boxed_spearman": rho_l})

        a39[behavior] = {
            "status": "ok",
            "key_metric_grid": grid,
            "boxed_primary": "c_C_key__whitened_metric",
            "boxed_primary_spearman": boxed,
            "boxed_primary_clustered_bootstrap": boxed_boot,
            "boxed_primary_shuffled_null_hi": boxed_null["null_hi"],
            "true_cosine_baseline_spearman": cos_rho,
            "self_normalized_I_metric_spearman": self_norm_I,
            "shuffled_key_control_spearman": shuf_key_rho,
            "shuffled_query_control_spearman": shuf_query_rho,
            "e0_base_prior_baseline_spearman": e0_rho,
            "e0_n_used": e0_n,
            "predict_mean_baseline_mae": pred_mean["mae"],
            "beats_true_cosine": bool(
                np.isfinite(boxed) and np.isfinite(cos_rho) and boxed > cos_rho
            ),
            "lambda": lam,
            "lambda_sweep": lam_records,
            "n_cells": len(rows),
        }

        # ── A3.10: base gate g0 vs oracle g+ (BLOCKER 1) ─────────────────────
        have_oracle = all("c_C_postft" in r and "c_Cp_postft" in r for r in rows)
        if have_oracle:
            g0_pred = _gate_pred_vec(rows, "c_C", "whitened", sigma_c, lam)  # base key/query
            gplus_pred = np.array(
                [
                    oracle_gplus(
                        torch.from_numpy(r["c_C_postft"]),
                        torch.from_numpy(r["c_Cp_postft"]),
                        sigma_c,
                        lam,
                    )
                    for r in rows
                ]
            )
            g0_real_rho, _ = _nan_safe_spearman(g0_pred, g_real)
            gplus_real_rho, _ = _nan_safe_spearman(gplus_pred, g_real)
            g0_oracle_rho, _ = _nan_safe_spearman(g0_pred, gplus_pred)
            g0_real_boot = clustered_bootstrap_spearman(
                g0_pred[np.isfinite(g0_pred)],
                g_real[np.isfinite(g0_pred)],
                [f for f, ok in zip(fams, np.isfinite(g0_pred), strict=True) if ok],
            )
            # realized key/query drift ‖c+ − c‖/‖c‖ per cell (scope caveat 7).
            key_drifts = [key_query_drift(r["c_C"], r["c_C_postft"]) for r in rows]
            query_drifts = [key_query_drift(r["c_Cp"], r["c_Cp_postft"]) for r in rows]
            all_drift = [d for d in (key_drifts + query_drifts) if np.isfinite(d)]
            a310[behavior] = {
                "status": "ok",
                "oracle_gplus_vs_realized_spearman": gplus_real_rho,
                "oracle_gplus_vs_realized_clustered_bootstrap": clustered_bootstrap_spearman(
                    gplus_pred[np.isfinite(gplus_pred)],
                    g_real[np.isfinite(gplus_pred)],
                    [f for f, ok in zip(fams, np.isfinite(gplus_pred), strict=True) if ok],
                ),
                "g0_vs_realized_spearman": g0_real_rho,
                "g0_vs_realized_clustered_bootstrap": g0_real_boot,
                "g0_vs_oracle_spearman": g0_oracle_rho,
                "key_query_drift_mean": float(np.mean(all_drift)) if all_drift else float("nan"),
                "key_query_drift_per_cell": [
                    {
                        "source": r["source"],
                        "target": r["target"],
                        "key_drift": kd,
                        "query_drift": qd,
                    }
                    for r, kd, qd in zip(rows, key_drifts, query_drifts, strict=True)
                ][:60],
                "note": "fixed M0 (no Sigma_c+; metric drift out of scope, R3-3)",
                "n_cells": len(rows),
            }
        else:
            a310[behavior] = {
                "status": "oracle_unavailable",
                "reason": "post-FT key/query (c_C_postft / c_Cp_postft) absent in store",
                "g0_vs_realized_spearman": boxed,
                "n_cells": len(rows),
            }
    return (
        {"assumption": "A3.9", "layer": layer, "by_behavior": a39, "metadata": _repro_meta()},
        {"assumption": "A3.10", "layer": layer, "by_behavior": a310, "metadata": _repro_meta()},
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / d) if d > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 per-assumption CPU analysis (A3.6-A3.10)."
    )
    parser.add_argument("--tensors-dir", default="eval_results/issue_667/analysis_tensors")
    parser.add_argument("--out-dir", default="eval_results/issue_667")
    parser.add_argument("--behaviors", nargs="+", default=list(IN_SCOPE_BEHAVIORS))
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument(
        "--skip-store-pin",
        action="store_true",
        help="skip the #658 store + G_meta pin asserts (smoke on a synthetic store)",
    )
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = args.primary_layer

    # ── B3 GATE: the whitened-gate reduction unit test MUST pass first ────────
    logger.info("B3 GATE: whitened_gate_reduction_unit_test()")
    whitened_gate_reduction_unit_test()
    logger.info("B3 unit test PASS")

    cells_by_beh = {b: load_cells(tensors_dir, b, layer) for b in args.behaviors}
    for b, cells in cells_by_beh.items():
        logger.info("behavior=%s: %d cells loaded at layer %d", b, len(cells), layer)

    # ── Load reused artifacts (sha-pinned) + coverage validators (BLOCKER 3) ──
    if not args.skip_store_pin:
        from dotenv import load_dotenv

        load_dotenv()
        assert_store_pin()
        g_meta = load_g_meta()
        g_tensor = load_g_tensor()
        sigma_dict = load_sigma_c_dict()
        sigma_c = sigma_c_at_layer(sigma_dict, layer)
        r_b_by_beh = {b: load_r_b(b, layer) for b in args.behaviors}
        # Fail-loud coverage validation BEFORE any analysis (never shrink the
        # denominator silently). Each raises CoverageError on a miss.
        in_scope_layers = sorted(set(ALL_LAYERS) | {layer})
        validate_r_b_coverage(args.behaviors, in_scope_layers)
        validate_sigma_c_coverage(sigma_dict, in_scope_layers)
        validate_g_meta_coverage(g_meta, cells_by_beh)
        validate_cid_coverage(g_tensor, cells_by_beh)
        logger.info("cached-artifact coverage validation PASS (r_b/sigma_c/G_meta/cid)")
    else:
        # Synthetic-store smoke: build minimal stand-ins. Infer the hidden dim
        # from the first loaded cell's c_C so the identity Sigma_c matches.
        g_meta = _synthetic_g_meta(tensors_dir, args.behaviors, layer)
        import torch

        hdim = _infer_hidden_dim(cells_by_beh)
        sigma_c = torch.eye(hdim, dtype=torch.float64)
        # Synthetic r_b per behavior so A3.6 exercises (fact still reads the
        # store's r_b_fact via run_a36's _fact_rb_from_store fallback).
        _rng = np.random.default_rng(1)
        r_b_by_beh = {
            b: (None if b == "fact" else _rng.normal(size=hdim).astype(np.float64))
            for b in args.behaviors
        }

    # ── A3.6-A3.10 ───────────────────────────────────────────────────────────
    a36 = run_a36(cells_by_beh, g_meta, r_b_by_beh, layer)
    a37 = run_a37(cells_by_beh, layer)
    a38 = run_a38(cells_by_beh, layer)
    a39, a310 = run_a39_a310(cells_by_beh, sigma_c, layer, g_meta=g_meta)

    outputs = {
        "A3_6_readout_stability.json": a36,
        "A3_7_source_write.json": a37,
        "A3_8_rank_one.json": a38,
        "A3_9_key_query_gate.json": a39,
        "A3_10_base_gate_validity.json": a310,
    }
    for fname, payload in outputs.items():
        (out_dir / fname).write_text(json.dumps(payload, indent=2, default=_json_default))
        logger.info("wrote %s", out_dir / fname)
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _infer_hidden_dim(cells_by_beh: dict) -> int:
    """Hidden dim from the first loaded cell's c_C (synthetic-store smoke)."""
    for cells in cells_by_beh.values():
        for data in cells.values():
            return int(data["c_C"].shape[0])
    raise RuntimeError("no cells loaded — cannot infer hidden dim for the synthetic smoke")


def _synthetic_g_meta(tensors_dir: Path, behaviors: list[str], layer: int) -> dict:
    """Minimal G_meta stand-in for the synthetic-store smoke (--skip-store-pin)."""
    per_cell = {}
    rng = np.random.default_rng(0)
    for b in behaviors:
        for cell_dir in (tensors_dir / b).glob("*_seed*"):
            source = cell_dir.name.rsplit("_seed", 1)[0]
            for npz in cell_dir.glob(f"*_L{layer}.npz"):
                target = npz.name.rsplit(f"_L{layer}.npz", 1)[0]
                per_cell[f"{b}/{source}__{target}"] = {
                    "g": float(rng.normal()),
                    "base_rate": float(rng.uniform(0, 0.3)),
                    "noise_var_bootstrap": 0.01,
                    "saturated": False,
                }
    return {"git_commit": EXPECTED_G_META_GIT_COMMIT, "per_cell": per_cell}


if __name__ == "__main__":
    sys.exit(main())
