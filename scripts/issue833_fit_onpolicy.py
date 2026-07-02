#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, r̂, r_B, →, ρ, M⁺, ×) in scientific docstrings + log messages.
"""Issue #833 — Phase D: on-policy vs off-policy function-change fits (thin driver).

Fits FOUR context→answer-profile maps per (behavior, layer) by IMPORTING the #722
harness (`issue722_fit_M` / `issue722_bootstrap` / `issue722_load_activations`;
never re-implementing the fit math):

- **M0**      (c_C        → v0(R_base))  — pre-FT function, #722's original;
- **M⁺_off**  (c_C_postft → v⁺(R_base))  — post-FT on OFF-policy text (#722 re-run);
- **M⁺_on**   (c_C_postft → v⁺(R⁺))      — post-FT on ON-policy text (the fix);
- **M0_ctrl** (c_C        → v0(R⁺))      — matched-text control (C5: does the BASE
  map "move" merely because the text moved?).

Old legs (L1/L2 + c_C/c_C_postft) stream from the PINNED #667 store revision; new
legs (L3 `v_plus_onpolicy` / L4 `v0_onpolicy` + per-probe `resp_sha256`) stream
from the #833 store written by ``issue833_extract_onpolicy.py``. Records join on
(behavior, source_cid, target_cid, layer) with a 100% coverage assert (missing
cells NAMED, fail loud).

Per behavior × layer it persists (plan v3 Must-Fix "Regime-local floors + raw
deltas" — registration requirement):

- ``eval_results/issue_833/cells/{behavior}_L{layer}.json`` — the ridge
  function-change reads for both arms + the ctrl, RAW r_B-projected deltas
  (``proj = |delta_full @ r_hat|``) with 7-family clustered bootstrap CIs, the
  FULL floor namespace (``floor_M0_refit``; ``floor_Mplus_refit`` SEPARATELY per
  arm; ``floor_M0ctrl_refit``; ``floor_shifted``; per-arm ``floor_combined`` /
  ``floor_sd_combined``), and each arm's read normalized by ITS OWN regime floor
  plus the on-vs-off paired per-cell delta in BOTH raw and floor units.
- ``chain_rho/{behavior}_L{layer}.json`` — Spearman(LOCO ``r_Bᵀ M̂(c)``, E) for
  M0 / M⁺_off / M⁺_on (+ paired diff CIs) and the MLP-vs-shuffle validity gate
  via the vectorized batched-LOCO helper (ridge-only headline preserved).
- ``decomposition/{behavior}_L{layer}.json`` — both split paths
  (Δ_rep = v⁺(R⁺)−v0(R⁺), Δ_text = v0(R⁺)−v0(R_base); primed diagonal
  Δ_rep′ = v⁺(R_base)−v0(R_base), Δ_text′ = v⁺(R⁺)−v⁺(R_base)) with the
  identity assert (max relative residual < 1e-3) and per-cell r_B-projected
  fractions.
- ``text_divergence/{behavior}.json`` + figures — C7 manipulation check
  (exact-match fraction via ``resp_sha256`` vs sha256 of the stored R_base text;
  word-level normalized Levenshtein distribution; length distributions).

``--smoke`` runs end-to-end on SYNTHETIC tiny tensors (3 sources × 4 targets ×
3 layers, hidden 64, fake r_B/E, synthetic texts) exercising every code path
(fits + floors + decomposition + chain-ρ + MLP gate + text stats + all figures)
into a tmp out-dir; exits 0 with a digest.

Phase D is VM-CPU per plan §9 (ridge closed-form; the MLP gate is the one
GPU-worthy piece — ``--device cuda`` if available, ``--mlp-layers`` to restrict).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
from issue722_bootstrap import clustered_bootstrap_scalar, floor_sd, make_refit_pair  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    fit_batched_loco_mlp,
)

logger = logging.getLogger("issue833.fit")

DATA_REPO = "superkaiba1/explore-persona-space-data"
HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
SWEEP_LAYERS = (7, 14, 21)
HEADLINE_LAYER = 14  # pre-registered (plan §6); L7/L21 reported as grids, never winner-selected
NEW_STORE_PREFIX = "issue833_onpolicy_map/analysis_tensors"
OLD_STORE_PREFIX = "issue667_gate_chain_preview/analysis_tensors"
OLD_STORE_REVISION = "0031fc55a0e965c33be4261287cd5c86393ca161"  # pinned (plan §4/§10)
RBASE_COMPLETIONS_PREFIX = "issue833_onpolicy_map/raw_completions/rbase"
ONPOLICY_COMPLETIONS_PREFIX = "issue833_onpolicy_map/raw_completions/generation"
IDENTITY_RTOL = 1e-3  # plan C6 / kill K2
N_SCALAR_BOOT = 1000

# New-store npz required keys (written by issue833_extract_onpolicy.py, plan §4 Phase B).
_NEW_VEC_KEYS = ("v_plus_onpolicy", "v0_onpolicy")
_NEW_META_KEYS = ("behavior", "source_cid", "target_cid", "layer")

# Cell-JSON schema keys the resume-skip requires (mirrors fitM._CELL_SCHEMA_KEYS).
_CELL_SCHEMA_KEYS = frozenset(
    {
        "delta_off_raw",
        "delta_on_raw",
        "delta_ctrl_raw",
        "paired_on_minus_off",
        "floor_M0_refit",
        "floor_Mplus_refit_off",
        "floor_Mplus_refit_on",
        "floor_M0ctrl_refit",
        "floor_shifted",
        "floor_combined_off",
        "floor_combined_on",
        "floor_sd_combined_off",
        "floor_sd_combined_on",
    }
)


# ── store access (revision-pinned streamer + prefix-parameterized layout) ──────


class _PinnedStreamer(loadact._Streamer):
    """#722 streamer + a PINNED ``revision`` on every ``hf_hub_download`` (plan §4/§10).

    The parent class always reads the repo default branch; the OLD #667 store must
    be read at the pinned revision so a later store push cannot silently change
    the reused L1/L2 legs. ``revision=None`` behaves like the parent (used for the
    NEW #833 store, read at main).
    """

    def __init__(self, *, prefix: str, revision: str | None, cache_size: int = 8):
        super().__init__(repo_id=DATA_REPO, prefix=prefix, cache_size=cache_size)
        self.revision = revision

    def load(self, rel_path: str) -> dict:
        from huggingface_hub import hf_hub_download

        path = self._resident.get(rel_path)
        if path is None or not path.exists():
            dest = self._staging / rel_path.replace("/", "__")
            dest.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                self.repo_id,
                f"{self.prefix}/{rel_path}",
                repo_type="dataset",
                revision=self.revision,
                local_dir=str(dest),
            )
            path = Path(downloaded)
            self._resident[rel_path] = path
            while len(self._resident) > self.cache_size:
                old = next(iter(self._resident))
                self._evict(old)
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}


def _list_layout(
    prefix: str, behaviors: tuple[str, ...], revision: str | None
) -> dict[str, dict[str, list[str]]]:
    """``{behavior: {bare_src_dir: [npz filenames]}}`` for an arbitrary prefix + revision.

    Prefix/revision-parameterized twin of ``loadact.list_store_layout`` (which is
    hardcoded to the module-global ``STORE_PREFIX`` at the default revision). Same
    bounded-retry ``list_repo_tree`` walk, same bare-source-dir key convention.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()

    def _tree(path: str) -> list:
        last: Exception | None = None
        for attempt in range(5):
            try:
                return list(
                    api.list_repo_tree(
                        DATA_REPO, path_in_repo=path, repo_type="dataset", revision=revision
                    )
                )
            except HfHubHTTPError as e:  # 504/503 transient on this large repo
                last = e
                wait = 2**attempt
                logger.warning(
                    "list_repo_tree(%s@%s) failed (%s); retry %d in %ds",
                    path,
                    revision or "main",
                    e,
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"list_repo_tree({path}@{revision or 'main'}) failed after 5 attempts"
        ) from last

    layout: dict[str, dict[str, list[str]]] = {}
    for beh in behaviors:
        beh_path = f"{prefix}/{beh}"
        src_dirs = [
            t.path for t in _tree(beh_path) if t.path != beh_path and not t.path.endswith(".npz")
        ]
        layout[beh] = {}
        for sd in src_dirs:
            src_name = sd.split("/")[-1]
            files = [t.path.split("/")[-1] for t in _tree(sd) if t.path.endswith(".npz")]
            layout[beh][src_name] = sorted(files)
    return layout


# ── new-store (on-policy) legs ─────────────────────────────────────────────────


@dataclass
class OnPolicyLeg:
    """One (behavior, source_cid, target_cid, layer) NEW-leg record from the #833 store."""

    behavior: str
    source_cid: str
    target_cid: str
    layer: int
    v_plus_on: np.ndarray  # v⁺(R⁺), (H,)
    v0_on: np.ndarray  # v0(R⁺), (H,)
    resp_sha256: list[str]  # per-probe sha256 of R⁺ text
    resp_sha256_base: list[str] | None  # per-probe sha256 of R_base text, if threaded
    resp_texts: list[str] | None  # per-probe R⁺ text, if threaded


def _as_str_list(x) -> list[str]:
    """Coerce an npz string array / scalar / object array to list[str]."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(v) for v in arr.tolist()]


def _new_blob_to_leg(blob: dict, rel: str, beh: str, li: int, hidden: int) -> OnPolicyLeg:
    """Validate one NEW-store ``.npz`` blob and build its OnPolicyLeg (fail loud)."""
    for k in (*_NEW_VEC_KEYS, *_NEW_META_KEYS, "resp_sha256"):
        if k not in blob:
            raise KeyError(f"{rel} missing key {k!r}; keys={sorted(blob)}")
    v_plus_on = np.asarray(blob["v_plus_onpolicy"], dtype=np.float64)
    v0_on = np.asarray(blob["v0_onpolicy"], dtype=np.float64)
    for name, arr in (("v_plus_onpolicy", v_plus_on), ("v0_onpolicy", v0_on)):
        assert arr.shape == (hidden,), f"{rel} {name} shape {arr.shape} != ({hidden},)"
    file_layer = int(np.asarray(blob["layer"]).item())
    assert file_layer == li, f"{rel} baked layer {file_layer} != requested {li}"
    file_beh = str(np.asarray(blob["behavior"]).item())
    assert file_beh == beh, f"{rel} behavior {file_beh} != dir {beh}"
    return OnPolicyLeg(
        behavior=beh,
        source_cid=str(np.asarray(blob["source_cid"]).item()),
        target_cid=str(np.asarray(blob["target_cid"]).item()),
        layer=li,
        v_plus_on=v_plus_on,
        v0_on=v0_on,
        resp_sha256=_as_str_list(blob["resp_sha256"]),
        resp_sha256_base=(
            _as_str_list(blob["resp_sha256_base"]) if "resp_sha256_base" in blob else None
        ),
        resp_texts=_as_str_list(blob["resp_texts"]) if "resp_texts" in blob else None,
    )


def load_onpolicy_legs(
    behaviors: tuple[str, ...],
    layers: tuple[int, ...],
    streamer: _PinnedStreamer,
    layout: dict[str, dict[str, list[str]]],
    hidden: int = loadact.HIDDEN,
) -> dict[tuple[str, str, str, int], OnPolicyLeg]:
    """Load every NEW-store leg for the requested behaviors × layers, keyed by cell."""
    legs: dict[tuple[str, str, str, int], OnPolicyLeg] = {}
    for beh in behaviors:
        if beh not in layout:
            raise KeyError(f"behavior {beh!r} not present in NEW store layout {sorted(layout)}")
        for src_dir, files in sorted(layout[beh].items()):
            by_target = loadact._parse_cell_files(src_dir, files, layers)
            for _stem, layer_files in sorted(by_target.items()):
                for li, rel_tail in sorted(layer_files.items()):
                    rel = f"{beh}/{rel_tail}"
                    leg = _new_blob_to_leg(streamer.load(rel), rel, beh, li, hidden)
                    key = (leg.behavior, leg.source_cid, leg.target_cid, leg.layer)
                    if key in legs:
                        raise RuntimeError(f"duplicate NEW-store cell {key} at {rel}")
                    legs[key] = leg
        logger.info("NEW store %s: %d legs", beh, sum(1 for k in legs if k[0] == beh))
    return legs


# ── join old CellRecords × new legs ────────────────────────────────────────────


def join_cells(
    old_cells: list[loadact.CellRecord],
    legs: dict[tuple[str, str, str, int], OnPolicyLeg],
) -> dict:
    """Join old-store CellRecords with new-store legs on (behavior, source, target, layer).

    Asserts 100% coverage in BOTH directions for the joined (behavior, layer)
    slice — every old cell has a new leg and every new leg of that slice has an
    old cell — naming the missing keys (fail loud, plan §7 success 2). Returns
    the stacked design dict: the #722 ``stack_for_fit`` fields plus ``Von``
    (v⁺(R⁺)), ``V0on`` (v0(R⁺)), and the per-cell ``legs`` list (text/hash reads).
    """
    assert old_cells, "join_cells: empty old-cell list"
    beh = old_cells[0].behavior
    layer = old_cells[0].layer
    old_keys = [(c.behavior, c.source_cid, c.target_cid, c.layer) for c in old_cells]
    missing_new = [k for k in old_keys if k not in legs]
    if missing_new:
        raise RuntimeError(
            f"{beh} L{layer}: {len(missing_new)}/{len(old_keys)} old cells have NO new-store "
            f"leg — join coverage < 100%. Missing: {sorted(missing_new)}"
        )
    slice_new = {k for k in legs if k[0] == beh and k[3] == layer}
    extra_new = sorted(slice_new - set(old_keys))
    if extra_new:
        raise RuntimeError(
            f"{beh} L{layer}: {len(extra_new)} new-store legs have NO old-store cell "
            f"(stale/misnamed extraction?): {extra_new}"
        )
    stacks = loadact.stack_for_fit(old_cells)
    joined_legs = [legs[k] for k in old_keys]
    stacks["Von"] = np.stack([lg.v_plus_on for lg in joined_legs])
    stacks["V0on"] = np.stack([lg.v0_on for lg in joined_legs])
    stacks["legs"] = joined_legs
    logger.info("join %s L%d: %d cells, 100%% coverage", beh, layer, len(old_cells))
    return stacks


# ── Phase D core: four ridge maps + regime-local floors + raw deltas ──────────


def _boot(values: np.ndarray, families: list[str], statistic: str) -> dict:
    return clustered_bootstrap_scalar(
        values, families, statistic=statistic, n_resamples=N_SCALAR_BOOT
    )


def _over_floor(ci: dict, floor_p95: float) -> dict | None:
    """Rescale a raw bootstrap CI dict into a regime-local floor-unit read."""
    if floor_p95 < 1e-12:
        return None
    return {k: (ci[k] / floor_p95 if isinstance(ci.get(k), float) else ci.get(k)) for k in ci}


def fit_cell_onpolicy(
    behavior: str,
    layer: int,
    joined: dict,
    r_hat: np.ndarray,
    *,
    n_refit_pairs: int,
) -> dict:
    """Fit the FOUR ridge maps + regime-local floors + raw/normalized deltas for one cell.

    All fit math is the imported #722 harness: ``fitM._ridge_fit_predict`` (PRESS-λ
    closed-form ridge, shared top-``TARGET_DIM`` v0-PCA target),
    ``make_refit_pair`` (the identical bootstrap+random-init refit floor), and
    ``fitM.m0_at_cplus_ridge_full`` (the shifted-design pseudo target). Both maps
    of every pair are evaluated on the SAME base grid (``common_c_grid`` = C0).
    """
    C0, Cplus = joined["C0"], joined["Cplus"]
    V0, Vplus = joined["V0"], joined["Vplus"]
    Von, V0on = joined["Von"], joined["V0on"]
    families = joined["families"]
    n = C0.shape[0]
    assert n >= 4, f"{behavior} L{layer}: only {n} cells (<4) — cannot fit"

    pca = fitM._pca_basis_v0(V0, fitM.TARGET_DIM)  # shared base-v0 basis, all four maps
    grid = loadact.common_c_grid(joined)  # = C0

    m0 = fitM._ridge_fit_predict(C0, V0 @ pca.T, grid)
    mplus_off = fitM._ridge_fit_predict(Cplus, Vplus @ pca.T, grid)
    mplus_on = fitM._ridge_fit_predict(Cplus, Von @ pca.T, grid)
    m0_ctrl = fitM._ridge_fit_predict(C0, V0on @ pca.T, grid)

    def _proj(delta64: np.ndarray) -> np.ndarray:
        return np.abs((delta64 @ pca) @ r_hat)  # RAW r_B-projected per-cell delta

    proj_off = _proj(mplus_off - m0)
    proj_on = _proj(mplus_on - m0)
    proj_ctrl = _proj(m0_ctrl - m0)
    proj_pair = proj_on - proj_off  # SIGNED paired per-cell on-vs-off delta (raw units)

    # ---- Regime-local floors (v3 Must-Fix): one refit floor PER FITTED MAP ----
    sc: dict[str, dict] = {k: {} for k in ("m0", "off", "on", "ctrl", "shift")}
    fl_m0 = make_refit_pair(
        C0,
        V0,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=n_refit_pairs,
        skip_counter=sc["m0"],
    )
    fl_off = make_refit_pair(
        Cplus,
        Vplus,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=n_refit_pairs,
        skip_counter=sc["off"],
    )
    fl_on = make_refit_pair(
        Cplus,
        Von,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=n_refit_pairs,
        skip_counter=sc["on"],
    )
    fl_ctrl = make_refit_pair(
        C0,
        V0on,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=n_refit_pairs,
        skip_counter=sc["ctrl"],
    )
    fl_shift = make_refit_pair(
        Cplus,
        fitM.m0_at_cplus_ridge_full(C0, V0, Cplus, pca),
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=n_refit_pairs,
        skip_counter=sc["shift"],
    )
    refit_skip = fitM._aggregate_refit_skips(behavior, layer, *sc.values())

    p95 = {
        "m0": float(np.percentile(fl_m0, 95)),
        "off": float(np.percentile(fl_off, 95)),
        "on": float(np.percentile(fl_on, 95)),
        "ctrl": float(np.percentile(fl_ctrl, 95)),
        "shift": float(np.percentile(fl_shift, 95)),
    }
    combined_off = max(p95["m0"], p95["off"], p95["shift"])
    combined_on = max(p95["m0"], p95["on"], p95["shift"])
    sd_combined_off = max(floor_sd(fl_m0), floor_sd(fl_off), floor_sd(fl_shift))
    sd_combined_on = max(floor_sd(fl_m0), floor_sd(fl_on), floor_sd(fl_shift))

    # ---- Raw CIs + regime-local floor-unit reads (each arm / ITS OWN floor) ----
    off_med = _boot(proj_off, families, "median")
    on_med = _boot(proj_on, families, "median")
    ctrl_med = _boot(proj_ctrl, families, "median")
    pair_med = _boot(proj_pair, families, "median")
    pair_mean = _boot(proj_pair, families, "mean")
    # Floor-unit paired delta: each arm normalized by ITS OWN regime floor first.
    if p95["off"] > 1e-12 and p95["on"] > 1e-12:
        pair_floor_units = proj_on / p95["on"] - proj_off / p95["off"]
        pair_fu_med = _boot(pair_floor_units, families, "median")
        pair_fu_mean = _boot(pair_floor_units, families, "mean")
    else:
        pair_fu_med = pair_fu_mean = None

    return {
        "behavior": behavior,
        "layer": layer,
        "n_cells": n,
        "n_families": len({*families}),
        # RAW r_B-projected deltas (v3 Must-Fix: always persisted alongside floors).
        "delta_off_raw": {"median_ci": off_med, "mean_ci": _boot(proj_off, families, "mean")},
        "delta_on_raw": {"median_ci": on_med, "mean_ci": _boot(proj_on, families, "mean")},
        "delta_ctrl_raw": {"median_ci": ctrl_med, "mean_ci": _boot(proj_ctrl, families, "mean")},
        "paired_on_minus_off": {
            "raw": {"median_ci": pair_med, "mean_ci": pair_mean},
            "floor_units": {"median_ci": pair_fu_med, "mean_ci": pair_fu_mean},
        },
        # Full floor namespace (regime-local registration requirement).
        "floor_M0_refit": p95["m0"],
        "floor_Mplus_refit_off": p95["off"],
        "floor_Mplus_refit_on": p95["on"],
        "floor_M0ctrl_refit": p95["ctrl"],
        "floor_shifted": p95["shift"],
        "floor_combined_off": combined_off,
        "floor_combined_on": combined_on,
        "floor_sd_combined_off": sd_combined_off,
        "floor_sd_combined_on": sd_combined_on,
        # Normalized reads — each arm in ITS OWN regime's floor units (never the
        # M0 floor as the sole yardstick for an M⁺-variant delta).
        "delta_off_over_own_floor": _over_floor(off_med, p95["off"]),
        "delta_on_over_own_floor": _over_floor(on_med, p95["on"]),
        "delta_ctrl_over_own_floor": _over_floor(ctrl_med, p95["ctrl"]),
        "delta_off_over_combined": _over_floor(off_med, combined_off),
        "delta_on_over_combined": _over_floor(on_med, combined_on),
        "delta_off_over_floor_sd": (
            None if sd_combined_off < 1e-12 else float(off_med["point"] / sd_combined_off)
        ),
        "delta_on_over_floor_sd": (
            None if sd_combined_on < 1e-12 else float(on_med["point"] / sd_combined_on)
        ),
        # Per-cell raw projections for the low-level plots (labeled by source).
        "per_cell": {
            "source_cids": joined["source_cids"],
            "target_cids": joined["target_cids"],
            "proj_off": proj_off.tolist(),
            "proj_on": proj_on.tolist(),
            "proj_ctrl": proj_ctrl.tolist(),
        },
        "refit_skip": refit_skip,
    }


# ── decomposition (plan C6: both split paths + identity assert) ────────────────


def decompose_cell(behavior: str, layer: int, joined: dict, r_hat: np.ndarray) -> dict:
    """Per-cell 4-leg decomposition, both split paths, with the identity assert (K2).

    Δ_total = v⁺(R⁺) − v0(R_base); path 1: Δ_rep = v⁺(R⁺) − v0(R⁺),
    Δ_text = v0(R⁺) − v0(R_base); path 2 (primed diagonal):
    Δ_rep′ = v⁺(R_base) − v0(R_base), Δ_text′ = v⁺(R⁺) − v⁺(R_base). Asserts
    ``max ‖Δ_rep + Δ_text − Δ_total‖ / ‖Δ_total‖ < 1e-3`` on BOTH paths in
    float32-accumulated arithmetic (kill K2: a violation is an implementation
    bug, halt fits).
    """
    V0, Vplus = joined["V0"], joined["Vplus"]
    Von, V0on = joined["Von"], joined["V0on"]
    d_total = Von - V0
    d_rep = Von - V0on
    d_text = V0on - V0
    d_rep_p = Vplus - V0
    d_text_p = Von - Vplus

    tot_norm = np.linalg.norm(d_total, axis=1)
    eps = 1e-12
    resid1 = np.linalg.norm((d_rep + d_text) - d_total, axis=1) / np.maximum(tot_norm, eps)
    resid2 = np.linalg.norm((d_rep_p + d_text_p) - d_total, axis=1) / np.maximum(tot_norm, eps)
    degenerate = tot_norm < 1e-8  # ‖Δ_total‖≈0 cells excluded from the RELATIVE read
    max_resid = float(
        max(resid1[~degenerate].max(initial=0.0), resid2[~degenerate].max(initial=0.0))
    )
    assert max_resid < IDENTITY_RTOL, (
        f"{behavior} L{layer}: decomposition identity VIOLATED — max relative residual "
        f"{max_resid:.3e} >= {IDENTITY_RTOL} (kill K2: implementation bug, halting fits)"
    )

    def _p(d: np.ndarray) -> np.ndarray:
        return np.abs(d @ r_hat)

    p_tot, p_rep, p_text = _p(d_total), _p(d_rep), _p(d_text)
    p_rep_p, p_text_p = _p(d_rep_p), _p(d_text_p)
    ok = p_tot > 1e-9

    def _frac(num: np.ndarray) -> list:
        return [float(num[i] / p_tot[i]) if ok[i] else None for i in range(len(num))]

    return {
        "behavior": behavior,
        "layer": layer,
        "n_cells": int(V0.shape[0]),
        "identity_max_rel_residual": max_resid,
        "identity_n_degenerate_total": int(degenerate.sum()),
        "per_cell": {
            "source_cids": joined["source_cids"],
            "target_cids": joined["target_cids"],
            "proj_total": p_tot.tolist(),
            "proj_rep": p_rep.tolist(),
            "proj_text": p_text.tolist(),
            "proj_rep_prime": p_rep_p.tolist(),
            "proj_text_prime": p_text_p.tolist(),
            "frac_rep": _frac(p_rep),
            "frac_text": _frac(p_text),
            "frac_rep_prime": _frac(p_rep_p),
            "frac_text_prime": _frac(p_text_p),
            "identity_rel_residual_path1": resid1.tolist(),
            "identity_rel_residual_path2": resid2.tolist(),
        },
    }


# ── chain-ρ triplet + MLP-vs-shuffle validity gate ─────────────────────────────


def chain_rho_cell(
    behavior: str,
    layer: int,
    joined: dict,
    r_hat: np.ndarray,
    E: np.ndarray,
    *,
    include_mlp: bool,
    mlp_epochs: int,
    mlp_device: str,
    mlp_chunk_size: int,
    mlp_num_threads: int | None = None,
) -> dict:
    """Spearman(LOCO ``r_Bᵀ M̂(c)``, E) for M0 / M⁺_off / M⁺_on + the MLP validity gate.

    Ridge LOCO (#722 ``_ridge_loco_pred`` → ``_chain_rho_one``) is the HEADLINE;
    the MLP arms run through the vectorized batched-LOCO helper
    (``fit_batched_loco_mlp``, all 6 groups — 3 arms × {base, shuffle-null} — in
    ONE vmapped ensemble) purely as the validity gate: ``mlp_valid_<arm>`` is
    True iff the arm's MLP chain-ρ beats its own shuffle null. Ridge keys are
    computed regardless (ridge-only headline preserved).
    """
    C0, Cplus = joined["C0"], joined["Cplus"]
    V0, Vplus, Von = joined["V0"], joined["Vplus"], joined["Von"]
    families = joined["families"]
    n = C0.shape[0]
    pca = fitM._pca_basis_v0(V0, fitM.TARGET_DIM)
    V0_64, Vplus_64, Von_64 = V0 @ pca.T, Vplus @ pca.T, Von @ pca.T

    keep = ~np.isnan(E)
    block: dict = {"behavior": behavior, "layer": layer, "n_with_E": int(keep.sum())}
    if keep.sum() < 4:
        logger.warning(
            "%s L%d: only %d cells with E — chain-ρ skipped", behavior, layer, keep.sum()
        )
        block["skipped"] = "fewer than 4 cells with E"
        return block
    Ek = E[keep]
    fam_k = [f for f, m in zip(families, keep, strict=True) if m]

    arms = {"M0": (C0, V0_64), "Mplus_off": (Cplus, Vplus_64), "Mplus_on": (Cplus, Von_64)}
    chains: dict[str, np.ndarray] = {}
    for arm, (X, Y64) in arms.items():
        loco = fitM._ridge_loco_pred(X, Y64)
        rho, chain = fitM._chain_rho_one(loco[keep], pca, r_hat, Ek)
        block[f"rho_{arm}_ridge"] = rho
        if rho is not None:
            block[f"ci_{arm}_ridge"] = clustered_bootstrap_spearman(chain, Ek, fam_k)
            chains[arm] = chain
    for a, b in (("M0", "Mplus_off"), ("M0", "Mplus_on"), ("Mplus_off", "Mplus_on")):
        if a in chains and b in chains:
            block[f"ci_diff_{b}_minus_{a}"] = fitM._clustered_paired_rho_diff_ci(
                chains[a], chains[b], Ek, fam_k
            )

    if include_mlp:
        rng = np.random.default_rng(833)
        groups: list[MLPGroup] = []
        for arm, (X, Y64) in arms.items():
            perm = rng.permutation(n)
            groups.append(
                MLPGroup(key=(arm, "base"), X=X.astype(np.float32), Y=Y64.astype(np.float32))
            )
            groups.append(
                MLPGroup(
                    key=(arm, "shuffle"), X=X.astype(np.float32), Y=Y64[perm].astype(np.float32)
                )
            )
        result = fit_batched_loco_mlp(
            groups,
            max_epochs=mlp_epochs,
            device=mlp_device,
            chunk_size=mlp_chunk_size,
            num_threads=mlp_num_threads,
        )
        mlp_block: dict = {"epochs": mlp_epochs, "shuffle_seed": 833}
        for arm in arms:
            rho_b, _ = fitM._chain_rho_one(result.preds_by_key[(arm, "base")][keep], pca, r_hat, Ek)
            rho_s, _ = fitM._chain_rho_one(
                result.preds_by_key[(arm, "shuffle")][keep], pca, r_hat, Ek
            )
            mlp_block[f"rho_{arm}_mlp"] = rho_b
            mlp_block[f"rho_{arm}_mlp_shuffle"] = rho_s
            mlp_block[f"mlp_valid_{arm}"] = (
                None if (rho_b is None or rho_s is None) else bool(rho_b > rho_s)
            )
        block["mlp_gate"] = mlp_block
    return block


# ── C7 text-divergence manipulation check ──────────────────────────────────────


def _sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def norm_lev_tokens(a: str, b: str) -> float:
    """WORD-level normalized Levenshtein distance in [0, 1] (vectorized row DP).

    Word-level (whitespace tokens) rather than char-level: char-level DP over
    ~4 KB texts × ~14k pairs is prohibitively slow in pure Python, and the
    divergence structure of interest (which spans of the response changed) is
    the same at word grain. Normalizer = max(len_a, len_b) tokens.
    """
    if a == b:
        return 0.0
    ta, tb = a.split(), b.split()
    m, k = len(ta), len(tb)
    if m == 0 or k == 0:
        return 0.0 if m == k else 1.0
    vocab: dict[str, int] = {}
    ia = np.array([vocab.setdefault(t, len(vocab)) for t in ta], dtype=np.int64)
    ib = np.array([vocab.setdefault(t, len(vocab)) for t in tb], dtype=np.int64)
    idx = np.arange(k + 1, dtype=np.int64)
    prev = idx.copy()
    for i in range(1, m + 1):
        cur = np.empty(k + 1, dtype=np.int64)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (ib != ia[i - 1]))
        # left-dependency (insertion) via the prefix-min identity
        cur = np.minimum.accumulate(cur - idx) + idx
        prev = cur
    return float(prev[-1]) / float(max(m, k))


def _texts_from_json(obj, rel: str) -> dict[tuple[str, int], str]:
    """Tolerant reader: raw-completions JSON → {(target_cid, probe_idx): text}.

    Accepted shapes: ``{tcid: [text, ...]}``; ``{"targets": {tcid: [text, ...]}}``;
    ``{tcid: {"responses": [text, ...]}}``; a list of records with
    ``target_cid``/``tcid`` + ``response``/``text``/``completion`` (+ optional
    ``probe_idx``/``qi``). Anything else fails loud naming the file.
    """
    out: dict[tuple[str, int], str] = {}
    if isinstance(obj, dict) and isinstance(obj.get("targets"), dict):
        obj = obj["targets"]
    if isinstance(obj, dict):
        for tcid, v in obj.items():
            if isinstance(v, dict) and isinstance(v.get("responses"), list):
                v = v["responses"]
            if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
                raise ValueError(f"{rel}: unrecognized raw-completions value under {tcid!r}")
            for qi, t in enumerate(v):
                out[(str(tcid), qi)] = t
        return out
    if isinstance(obj, list):
        counters: dict[str, int] = {}
        for rec in obj:
            if not isinstance(rec, dict):
                raise ValueError(f"{rel}: unrecognized list-form raw-completions record")
            tcid = rec.get("target_cid") or rec.get("tcid")
            text = rec.get("response") or rec.get("text") or rec.get("completion")
            if tcid is None or text is None:
                raise ValueError(f"{rel}: record missing target_cid/response: {sorted(rec)}")
            qi = rec.get("probe_idx", rec.get("qi"))
            if qi is None:
                qi = counters.get(str(tcid), 0)
            counters[str(tcid)] = int(qi) + 1
            out[(str(tcid), int(qi))] = str(text)
        return out
    raise ValueError(f"{rel}: unrecognized raw-completions JSON shape ({type(obj).__name__})")


def _load_completion_texts(
    prefix: str, behavior: str, source_dirs: list[str], revision: str | None = None
) -> dict[tuple[str, str, int], str]:
    """Fetch ``{prefix}/{behavior}/{source_dir}.json`` per source → {(src, tcid, qi): text}."""
    from huggingface_hub import hf_hub_download

    out: dict[tuple[str, str, int], str] = {}
    for src in source_dirs:
        rel = f"{prefix}/{behavior}/{src}.json"
        local = hf_hub_download(DATA_REPO, rel, repo_type="dataset", revision=revision)
        per = _texts_from_json(json.loads(Path(local).read_text()), rel)
        for (tcid, qi), t in per.items():
            out[(src, tcid, qi)] = t
    return out


def text_stats_for_behavior(
    behavior: str,
    joined_by_layer: dict[int, dict],
    *,
    rbase_prefix: str,
    onpolicy_prefix: str,
    old_revision: str | None,
    texts_override: dict[str, dict[tuple[str, str, int], str]] | None = None,
) -> dict:
    """C7: exact-match fraction (sha256) + word-level edit-distance + length dists.

    Exact match compares the NEW store's per-probe ``resp_sha256`` (R⁺) against
    sha256 of the stored R_base text — from the npz ``resp_sha256_base`` key when
    the extractor threaded it, else hashed from the R_base rollout text at
    ``rbase_prefix``. Edit distance + lengths need BOTH texts (npz ``resp_texts``
    or the ``onpolicy_prefix`` rollout JSONs for R⁺). A missing text source fails
    LOUD naming the paths tried (C7 is a planned manipulation check — no silent
    skip). ``texts_override`` = {"rbase"|"onpolicy": {(src_dir, tcid, qi): text}}
    is the smoke's in-memory source. Cell grain: legs are layer-replicated, so
    stats run over the FIRST loaded layer's legs (text is layer-invariant).
    """
    joined = joined_by_layer[sorted(joined_by_layer)[0]]
    legs: list[OnPolicyLeg] = joined["legs"]
    src_dirs = sorted({f"{lg.source_cid}_seed42" for lg in legs})

    hashes_threaded = all(lg.resp_sha256_base for lg in legs)
    base_texts = on_texts = None
    if texts_override is not None:
        base_texts = texts_override["rbase"]
        on_texts = texts_override["onpolicy"]
    else:
        # R_base texts serve BOTH the hash side (when the extractor did not thread
        # resp_sha256_base) and the edit-distance side. When hashes ARE threaded a
        # missing text bucket is tolerated HERE; any diverged probe then fails loud
        # below at the edit-distance read (no silent skip — plan C7).
        try:
            base_texts = _load_completion_texts(
                rbase_prefix, behavior, src_dirs, revision=old_revision
            )
        except Exception as e:
            if not hashes_threaded:
                raise RuntimeError(
                    f"C7 {behavior}: no R_base hash source — npz lacks resp_sha256_base and "
                    f"{rbase_prefix}/{behavior}/*.json did not resolve ({e})"
                ) from e
            logger.warning(
                "C7 %s: R_base text bucket %s unavailable (%s) — hashes are npz-threaded; "
                "edit distance will fail loud on the first diverged probe",
                behavior,
                rbase_prefix,
                e,
            )
        if not all(lg.resp_texts for lg in legs):
            on_texts = _load_completion_texts(onpolicy_prefix, behavior, src_dirs)

    n_match = n_total = 0
    dists: list[float] = []
    len_base: list[int] = []
    len_on: list[int] = []
    per_cell_frac: list[dict] = []
    for lg in legs:
        src = f"{lg.source_cid}_seed42"
        cell_match = cell_n = 0
        for qi, sha_on in enumerate(lg.resp_sha256):
            # base-side hash: npz-threaded, else hashed from the stored R_base text
            if lg.resp_sha256_base:
                sha_base = lg.resp_sha256_base[qi]
                t_base = base_texts.get((src, lg.target_cid, qi)) if base_texts else None
            else:
                t_base = base_texts.get((src, lg.target_cid, qi))
                if t_base is None:
                    raise RuntimeError(
                        f"C7 {behavior}: R_base text missing for ({src}, {lg.target_cid}, "
                        f"probe {qi}) under {rbase_prefix}"
                    )
                sha_base = _sha256_text(t_base)
            match = sha_on == sha_base
            n_total += 1
            cell_n += 1
            n_match += int(match)
            cell_match += int(match)
            # texts (edit distance + lengths)
            t_on = (
                lg.resp_texts[qi]
                if lg.resp_texts
                else (on_texts.get((src, lg.target_cid, qi)) if on_texts else None)
            )
            if t_on is None:
                raise RuntimeError(
                    f"C7 {behavior}: R⁺ text missing for ({src}, {lg.target_cid}, probe {qi}) "
                    f"— npz lacks resp_texts and {onpolicy_prefix} did not resolve it"
                )
            len_on.append(len(t_on))
            if t_base is not None:
                len_base.append(len(t_base))
                dists.append(0.0 if match else norm_lev_tokens(t_base, t_on))
            elif match:
                dists.append(0.0)  # hash-equal ⇒ byte-identical, distance 0 without text
            else:
                raise RuntimeError(
                    f"C7 {behavior}: R_base TEXT unavailable for diverged probe "
                    f"({src}, {lg.target_cid}, {qi}) — cannot compute edit distance"
                )
        per_cell_frac.append(
            {
                "source_cid": lg.source_cid,
                "target_cid": lg.target_cid,
                "exact_match_frac": cell_match / max(cell_n, 1),
                "n_probes": cell_n,
            }
        )

    d = np.asarray(dists, dtype=float)
    return {
        "behavior": behavior,
        "n_probes": n_total,
        "exact_match_frac": n_match / max(n_total, 1),
        "edit_distance_metric": "word-level normalized Levenshtein (see norm_lev_tokens)",
        "edit_distance": {
            "mean": float(d.mean()) if d.size else None,
            "median": float(np.median(d)) if d.size else None,
            "p90": float(np.percentile(d, 90)) if d.size else None,
            "values": d.tolist(),
        },
        "response_length_chars": {"rbase": len_base, "onpolicy": len_on},
        "per_cell": per_cell_frac,
    }


# ── figures (paper-plots conventions: colorblind-safe, error bars, no overlays) ─


def _style():
    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    return paper_palette


def _save(fig, fig_dir: Path, name: str, paths: list[str]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    paths.append(str(out))
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_hero_dumbbell(cells: dict, behaviors: list[str], layer: int, fig_dir: Path, paths: list):
    """HERO: per-behavior off-vs-on ‖M⁺−M0‖_rB dumbbell at the pre-registered layer.

    Left panel: regime-local floor units (each arm / its OWN floor_Mplus_refit;
    the shaded band marks ≤1 floor-unit). Right panel: RAW projected units with
    per-arm floor markers — the v3 robustness rule wants both readable at once.
    """
    import matplotlib.pyplot as plt

    palette = _style()(4)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    ys = np.arange(len(behaviors))
    for ax, mode in zip(axes, ("floor_units", "raw"), strict=True):
        for yi, beh in enumerate(behaviors):
            c = cells.get((beh, layer))
            if c is None:
                continue
            if mode == "floor_units":
                off, on = c["delta_off_over_own_floor"], c["delta_on_over_own_floor"]
            else:
                off = c["delta_off_raw"]["median_ci"]
                on = c["delta_on_raw"]["median_ci"]
            if off is None or on is None:
                continue
            for val, dy, color, label in (
                (off, -0.12, palette[0], "off-policy (M⁺_off)"),
                (on, +0.12, palette[1], "on-policy (M⁺_on)"),
            ):
                ax.errorbar(
                    val["point"],
                    yi + dy,
                    xerr=[[val["point"] - val["ci_lo"]], [val["ci_hi"] - val["point"]]],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=label if yi == 0 else None,
                )
            ax.plot(
                [off["point"], on["point"]], [yi - 0.12, yi + 0.12], color="0.6", lw=1, zorder=0
            )
            if mode == "raw":
                ax.plot(
                    [c["floor_Mplus_refit_off"]], [yi - 0.12], marker="|", ms=12, color=palette[0]
                )
                ax.plot(
                    [c["floor_Mplus_refit_on"]], [yi + 0.12], marker="|", ms=12, color=palette[1]
                )
        if mode == "floor_units":
            ax.axvspan(0, 1, color="0.9", zorder=-1)
            ax.set_xlabel("‖M⁺ − M0‖ on r̂_B (regime-local floor units)")
        else:
            ax.set_xlabel("‖M⁺ − M0‖ on r̂_B (raw; ticks = per-arm refit floor p95)")
        ax.set_yticks(ys, behaviors)
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(f"Function change off- vs on-policy — L{layer} (pre-registered)")
    _save(fig, fig_dir, f"hero_dumbbell_L{layer}.png", paths)


def fig_headline_grid(cells: dict, behaviors: list[str], layers: list[int], fig_dir: Path, paths):
    """Per-layer grid of the off/on floor-unit reads (never winner-selected)."""
    import matplotlib.pyplot as plt

    palette = _style()(4)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(behaviors))
    for ax, layer in zip(axes, layers, strict=True):
        for xi, beh in enumerate(behaviors):
            c = cells.get((beh, layer))
            if c is None:
                continue
            for key, dx, color in (
                ("delta_off_over_own_floor", -0.15, palette[0]),
                ("delta_on_over_own_floor", +0.15, palette[1]),
            ):
                v = c[key]
                if v is None:
                    continue
                ax.errorbar(
                    xi + dx,
                    v["point"],
                    yerr=[[v["point"] - v["ci_lo"]], [v["ci_hi"] - v["point"]]],
                    fmt="o",
                    color=color,
                    capsize=3,
                )
        ax.axhspan(0, 1, color="0.9", zorder=-1)
        ax.set_xticks(x, behaviors, rotation=20)
        ax.set_title(f"L{layer}")
    axes[0].set_ylabel("Δ (own-floor units)")
    fig.suptitle("Headline per-layer grid — off (dark) vs on (light) policy")
    _save(fig, fig_dir, "headline_grid_layers.png", paths)


def fig_chain_rho(chain: dict, behaviors: list[str], layers: list[int], fig_dir: Path, paths):
    """Chain-ρ triplet bars (M0 / M⁺_off / M⁺_on) per behavior × layer with CIs."""
    import matplotlib.pyplot as plt

    palette = _style()(3)
    arms = ("M0", "Mplus_off", "Mplus_on")
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(behaviors))
    w = 0.25
    for ax, layer in zip(axes, layers, strict=True):
        for ai, arm in enumerate(arms):
            pts, los, his, xs = [], [], [], []
            for xi, beh in enumerate(behaviors):
                b = chain.get((beh, layer), {})
                rho = b.get(f"rho_{arm}_ridge")
                if rho is None:
                    continue
                ci = b.get(f"ci_{arm}_ridge") or {}
                pts.append(rho)
                los.append(rho - ci.get("ci_lo", rho))
                his.append(ci.get("ci_hi", rho) - rho)
                xs.append(xi + (ai - 1) * w)
            if pts:
                ax.bar(
                    xs, pts, width=w, color=palette[ai], label=arm if layer == layers[0] else None
                )
                ax.errorbar(xs, pts, yerr=[los, his], fmt="none", ecolor="0.2", capsize=2)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(x, behaviors, rotation=20)
        ax.set_title(f"L{layer}")
    axes[0].set_ylabel("Spearman ρ(r̂_Bᵀ M̂(c), E)  [ridge LOCO]")
    axes[0].legend(loc="best", frameon=False)
    _save(fig, fig_dir, "chain_rho_bars.png", paths)


def fig_decomp_scatters(decomp: dict, behaviors: list[str], layer: int, fig_dir: Path, paths):
    """Δ_rep vs Δ_text per-cell scatters (both split paths), points labeled by source."""
    import matplotlib.pyplot as plt

    _style()
    for beh in behaviors:
        d = decomp.get((beh, layer))
        if d is None:
            continue
        pc = d["per_cell"]
        srcs = sorted(set(pc["source_cids"]))
        palette = _style()(max(len(srcs), 3))
        color_of = {s: palette[i % len(palette)] for i, s in enumerate(srcs)}
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=False)
        for ax, (kx, ky, title) in zip(
            axes,
            (
                ("proj_text", "proj_rep", "path 1: Δ_text vs Δ_rep"),
                ("proj_text_prime", "proj_rep_prime", "path 2 (primed): Δ_text′ vs Δ_rep′"),
            ),
            strict=True,
        ):
            xs = np.asarray(pc[kx])
            ys = np.asarray(pc[ky])
            for s in srcs:
                m = np.asarray([sc == s for sc in pc["source_cids"]])
                ax.scatter(xs[m], ys[m], s=14, color=color_of[s], label=s)
            lim = max(xs.max(initial=0), ys.max(initial=0)) or 1.0
            ax.plot([0, lim], [0, lim], color="0.7", lw=0.8, ls="--")
            ax.set_xlabel(f"|{kx} · r̂_B|")
            ax.set_ylabel(f"|{ky} · r̂_B|")
            ax.set_title(title)
        axes[1].legend(loc="best", frameon=False, fontsize=6, ncol=2)
        fig.suptitle(f"{beh} L{layer} — per-cell weights-vs-text decomposition")
        _save(fig, fig_dir, f"decomp_scatter_{beh}_L{layer}.png", paths)


def fig_text_divergence(tstats: dict, fig_dir: Path, paths):
    """Per-behavior edit-distance histograms + R⁺ vs R_base length distributions."""
    import matplotlib.pyplot as plt

    palette = _style()(3)
    for beh, ts in tstats.items():
        vals = np.asarray(ts["edit_distance"]["values"], dtype=float)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.hist(vals, bins=30, color=palette[0])
        ax.set_xlabel("word-level normalized Levenshtein (R⁺ vs R_base)")
        ax.set_ylabel("probes")
        ax.set_title(f"{beh}: text divergence (exact-match {ts['exact_match_frac']:.2f})")
        _save(fig, fig_dir, f"text_divergence_hist_{beh}.png", paths)

        lb = ts["response_length_chars"]["rbase"]
        lo = ts["response_length_chars"]["onpolicy"]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        bins = np.histogram_bin_edges(np.asarray(lb + lo, dtype=float), bins=30)
        ax.hist(lb, bins=bins, alpha=0.6, color=palette[1], label="R_base")
        ax.hist(lo, bins=bins, alpha=0.6, color=palette[2], label="R⁺ (on-policy)")
        ax.set_xlabel("response length (chars)")
        ax.set_ylabel("probes")
        ax.set_title(f"{beh}: response lengths")
        ax.legend(frameon=False)
        _save(fig, fig_dir, f"length_dist_{beh}.png", paths)


def fig_identity_residuals(decomp: dict, fig_dir: Path, paths):
    """Distribution of per-cell identity relative residuals (both paths, log x)."""
    import matplotlib.pyplot as plt

    palette = _style()(2)
    vals1, vals2 = [], []
    for d in decomp.values():
        vals1 += d["per_cell"]["identity_rel_residual_path1"]
        vals2 += d["per_cell"]["identity_rel_residual_path2"]
    v1 = np.asarray(vals1, dtype=float)
    v2 = np.asarray(vals2, dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    lo = max(min(v1.min(initial=1e-12), v2.min(initial=1e-12)), 1e-16)
    bins = np.logspace(np.log10(lo), np.log10(IDENTITY_RTOL), 40)
    ax.hist(v1, bins=bins, alpha=0.7, color=palette[0], label="path 1")
    ax.hist(v2, bins=bins, alpha=0.7, color=palette[1], label="path 2 (primed)")
    ax.set_xscale("log")
    ax.set_xlabel("identity relative residual (assert < 1e-3)")
    ax.set_ylabel("cells")
    ax.legend(frameon=False)
    _save(fig, fig_dir, "identity_residual_dist.png", paths)


# ── smoke: synthetic tiny tensors exercising every code path (no HF, no GPU) ───


def build_smoke_data(behaviors: tuple[str, ...], layers: tuple[int, ...], hidden: int = 64):
    """Synthetic 3-sources × 4-targets grid per behavior × layer + fake r_B / E / texts.

    Returns ``(joined_by, r_hat_by, E_by, texts_override_by)`` shaped exactly like
    the production loaders' outputs so the SAME analysis code paths run end-to-end.
    Half the probes keep R⁺ == R_base (exercising the exact-match path), half
    diverge (exercising hashes + edit distance + non-trivial Δ_text).
    """
    rng = np.random.default_rng(833)
    sources = [f"src{i}" for i in range(3)]
    targets = [f"tgt{j}" for j in range(4)]
    n_probes = 5
    joined_by: dict[tuple[str, int], dict] = {}
    r_hat_by: dict[tuple[str, int], np.ndarray] = {}
    E_by: dict[tuple[str, int], np.ndarray] = {}
    texts_override_by: dict[str, dict] = {}
    for beh in behaviors:
        base_texts: dict[tuple[str, str, int], str] = {}
        on_texts: dict[tuple[str, str, int], str] = {}
        for li in layers:
            r = rng.normal(size=hidden)
            r_hat = r / np.linalg.norm(r)
            old_cells: list[loadact.CellRecord] = []
            legs: dict[tuple[str, str, str, int], OnPolicyLeg] = {}
            E_vals = []
            for si, src in enumerate(sources):
                c0 = rng.normal(size=hidden)
                cplus = c0 + 0.1 * rng.normal(size=hidden)
                for tj, tgt in enumerate(targets):
                    v0 = rng.normal(size=hidden)
                    vplus = v0 + 0.3 * r_hat + 0.05 * rng.normal(size=hidden)
                    v0on = v0 + 0.1 * rng.normal(size=hidden)  # text shift
                    von = v0on + 0.35 * r_hat + 0.05 * rng.normal(size=hidden)  # + rep shift
                    fam = f"fam{tj % 2}"
                    old_cells.append(
                        loadact.CellRecord(
                            behavior=beh,
                            source_cid=src,
                            target_cid=tgt,
                            layer=li,
                            c0=c0,
                            cplus=cplus,
                            v0=v0,
                            vplus=vplus,
                            family=fam,
                        )
                    )
                    shas, shas_base = [], []
                    for qi in range(n_probes):
                        tb = f"base answer {beh} {src} {tgt} probe {qi} " + " ".join(
                            rng.choice(["alpha", "beta", "gamma", "delta"], size=8)
                        )
                        diverged = (qi + si + tj) % 2 == 0
                        to = tb if not diverged else tb + " trained extra tokens " + str(qi)
                        base_texts[(f"{src}_seed42", tgt, qi)] = tb
                        on_texts[(f"{src}_seed42", tgt, qi)] = to
                        shas.append(_sha256_text(to))
                        shas_base.append(_sha256_text(tb))
                    legs[(beh, src, tgt, li)] = OnPolicyLeg(
                        behavior=beh,
                        source_cid=src,
                        target_cid=tgt,
                        layer=li,
                        v_plus_on=von,
                        v0_on=v0on,
                        resp_sha256=shas,
                        resp_sha256_base=shas_base,
                        resp_texts=None,
                    )
                    E_vals.append(float(von @ r_hat) + 0.1 * rng.normal())
            joined = join_cells(old_cells, legs)
            joined_by[(beh, li)] = joined
            r_hat_by[(beh, li)] = r_hat
            E = np.asarray(E_vals, dtype=np.float64)
            E[0] = np.nan  # exercise the keep-mask
            E_by[(beh, li)] = E
        texts_override_by[beh] = {"rbase": base_texts, "onpolicy": on_texts}
    return joined_by, r_hat_by, E_by, texts_override_by


# ── driver ─────────────────────────────────────────────────────────────────────


def _cell_json_valid(path: Path) -> bool:
    """Resume-skip validator for a cached #833 cell JSON (mirrors #722's contract)."""
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return not (_CELL_SCHEMA_KEYS - obj.keys())


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("auto")  # ridge device (fit_M precedent)
    ap = argparse.ArgumentParser(description="Issue #833 Phase D — on-policy map fits")
    ap.add_argument("--behaviors", nargs="+", default=list(HEADLINE_BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(SWEEP_LAYERS))
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_833")
    ap.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures/issue_833")
    ap.add_argument("--store-prefix", default=NEW_STORE_PREFIX, help="NEW #833 store prefix")
    ap.add_argument("--old-store-prefix", default=OLD_STORE_PREFIX, help="OLD #667 store prefix")
    ap.add_argument(
        "--old-store-revision",
        default=OLD_STORE_REVISION,
        help="PINNED revision for every OLD-store HF read (plan §4/§10)",
    )
    ap.add_argument("--rbase-completions-prefix", default=RBASE_COMPLETIONS_PREFIX)
    ap.add_argument("--onpolicy-completions-prefix", default=ONPOLICY_COMPLETIONS_PREFIX)
    ap.add_argument("--headline-layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--refit-pairs", type=int, default=fitM.N_REFIT_PAIRS)
    ap.add_argument("--target-dim", type=int, default=fit658.A35_MLP_TARGET_DIM)
    ap.add_argument("--skip-mlp", action="store_true", help="skip the MLP validity gate")
    ap.add_argument(
        "--mlp-layers",
        nargs="+",
        type=int,
        default=None,
        help="layers to run the MLP gate on (default: all --layers)",
    )
    ap.add_argument("--mlp-epochs", type=int, default=None, help="override MLP_MAX_EPOCHS")
    ap.add_argument("--mlp-chunk-size", type=int, default=512)
    ap.add_argument(
        "--mlp-num-threads",
        type=int,
        default=8,
        help="torch CPU threads for the MLP gate (tiny-op thrash guard, #722; 0 = torch default)",
    )
    ap.add_argument("--device", default="cpu", help="MLP-gate device (cpu|cuda)")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="synthetic tiny end-to-end run")
    args = ap.parse_args()

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="issue833_smoke_"))
        args.out_dir = tmp / "eval_results"
        args.figures_dir = tmp / "figures"
        args.refit_pairs = min(args.refit_pairs, 6)
        args.target_dim = min(args.target_dim, 4)
        args.mlp_epochs = args.mlp_epochs or 8
        args.mlp_chunk_size = min(args.mlp_chunk_size, 64)
        logger.info("[phase=fit_onpolicy] SMOKE — synthetic data, out=%s", tmp)

    fitM.TARGET_DIM = args.target_dim  # module-global read by _refit_ridge_fn
    mlp_epochs = args.mlp_epochs if args.mlp_epochs is not None else fit658.MLP_MAX_EPOCHS
    mlp_layers = set(args.mlp_layers if args.mlp_layers is not None else args.layers)
    behaviors = tuple(args.behaviors)
    layers = tuple(args.layers)

    fit658._assert_ridge_exactness()  # startup exactness gate (fit_M precedent)
    logger.info("[phase=fit_onpolicy] ridge exactness gate PASS (device=%s)", fit658.DEVICE)

    # ---- load + join (production) or synthesize (smoke) ----
    texts_override_by: dict[str, dict] | None = None
    if args.smoke:
        joined_by, r_hat_by, E_by, texts_override_by = build_smoke_data(behaviors, layers)
    else:
        g_meta = PROJECT_ROOT / "eval_results/issue_537/G_tensor/G_meta.json"
        if not g_meta.exists():
            raise FileNotFoundError(
                f"{g_meta} missing — the chain-ρ E target is a committed git artifact. In a "
                "sparse worktree run `git sparse-checkout add eval_results/issue_537` first."
            )
        rb_main = fitM._load_rb_main()
        rb_fact = fitM._load_rb_fact() if "fact" in behaviors else None
        if "fact" in behaviors and rb_fact is None:
            raise RuntimeError(
                "fact requested but r_b_fact.pt unavailable/degenerate — fact carries the #833 "
                "verdict under test (kill K3: never silently dropped)"
            )
        old_streamer = _PinnedStreamer(
            prefix=args.old_store_prefix, revision=args.old_store_revision
        )
        old_layout = _list_layout(args.old_store_prefix, behaviors, args.old_store_revision)
        old_cells_by = loadact.load_cells(
            behaviors=behaviors,
            layers=layers,
            streamer=old_streamer,
            strict_counts=True,
            layout=old_layout,
        )
        new_streamer = _PinnedStreamer(prefix=args.store_prefix, revision=None)
        new_layout = _list_layout(args.store_prefix, behaviors, None)
        legs = load_onpolicy_legs(behaviors, layers, new_streamer, new_layout)
        joined_by, r_hat_by, E_by = {}, {}, {}
        for beh in behaviors:
            for li in layers:
                joined = join_cells(old_cells_by[(beh, li)], legs)
                joined_by[(beh, li)] = joined
                r_hat_by[(beh, li)] = fitM._r_hat_for(beh, li, rb_main, rb_fact)
                E_by[(beh, li)] = fitM._load_E(beh, joined["cell_keys"])

    # ---- per-(behavior, layer): fits + floors, decomposition, chain-ρ ----
    cells_dir = args.out_dir / "cells"
    chain_dir = args.out_dir / "chain_rho"
    decomp_dir = args.out_dir / "decomposition"
    cells: dict[tuple[str, int], dict] = {}
    chains: dict[tuple[str, int], dict] = {}
    decomps: dict[tuple[str, int], dict] = {}
    for beh in behaviors:
        for li in layers:
            key = (beh, li)
            joined, r_hat, E = joined_by[key], r_hat_by[key], E_by[key]
            cell_path = cells_dir / f"{beh}_L{li}.json"
            chain_path = chain_dir / f"{beh}_L{li}.json"
            decomp_path = decomp_dir / f"{beh}_L{li}.json"
            if (
                not args.force_rerun
                and cell_path.exists()
                and _cell_json_valid(cell_path)
                and chain_path.exists()
                and decomp_path.exists()
            ):
                logger.info("[phase=fit_onpolicy] %s L%d cached — skip", beh, li)
                cells[key] = json.loads(cell_path.read_text())
                chains[key] = json.loads(chain_path.read_text())
                decomps[key] = json.loads(decomp_path.read_text())
                continue
            logger.info("[phase=fit_onpolicy] %s L%d (%d cells)", beh, li, joined["C0"].shape[0])
            decomps[key] = decompose_cell(beh, li, joined, r_hat)  # K2 gate BEFORE fits
            _write_json(decomp_path, decomps[key])
            cells[key] = fit_cell_onpolicy(beh, li, joined, r_hat, n_refit_pairs=args.refit_pairs)
            cells[key]["metadata"] = {
                "issue": 833,
                "old_store_revision": args.old_store_revision,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_json(cell_path, cells[key])
            chains[key] = chain_rho_cell(
                beh,
                li,
                joined,
                r_hat,
                E,
                include_mlp=(not args.skip_mlp) and li in mlp_layers,
                mlp_epochs=mlp_epochs,
                mlp_device=args.device,
                mlp_chunk_size=args.mlp_chunk_size,
                mlp_num_threads=args.mlp_num_threads or None,
            )
            _write_json(chain_path, chains[key])
            logger.info(
                "[phase=fit_onpolicy]   off=%.4g on=%.4g (raw med); floors off=%.4g on=%.4g",
                cells[key]["delta_off_raw"]["median_ci"]["point"],
                cells[key]["delta_on_raw"]["median_ci"]["point"],
                cells[key]["floor_Mplus_refit_off"],
                cells[key]["floor_Mplus_refit_on"],
            )

    # ---- C7 text stats (per behavior; layer-invariant) ----
    tstats: dict[str, dict] = {}
    for beh in behaviors:
        ts_path = args.out_dir / "text_divergence" / f"{beh}.json"
        if not args.force_rerun and ts_path.exists():
            tstats[beh] = json.loads(ts_path.read_text())
            continue
        tstats[beh] = text_stats_for_behavior(
            beh,
            {li: joined_by[(beh, li)] for li in layers},
            rbase_prefix=args.rbase_completions_prefix,
            onpolicy_prefix=args.onpolicy_completions_prefix,
            old_revision=None,
            texts_override=texts_override_by.get(beh) if texts_override_by else None,
        )
        _write_json(ts_path, tstats[beh])
        logger.info(
            "[phase=fit_onpolicy] C7 %s: exact-match %.3f over %d probes",
            beh,
            tstats[beh]["exact_match_frac"],
            tstats[beh]["n_probes"],
        )

    # ---- figures (HERO + exploratory dump, plan §6) ----
    fig_paths: list[str] = []
    beh_list, layer_list = list(behaviors), list(layers)
    hero_layer = args.headline_layer if args.headline_layer in layers else layer_list[0]
    fig_hero_dumbbell(cells, beh_list, hero_layer, args.figures_dir, fig_paths)
    fig_headline_grid(cells, beh_list, layer_list, args.figures_dir, fig_paths)
    fig_chain_rho(chains, beh_list, layer_list, args.figures_dir, fig_paths)
    fig_decomp_scatters(decomps, beh_list, hero_layer, args.figures_dir, fig_paths)
    fig_text_divergence(tstats, args.figures_dir, fig_paths)
    fig_identity_residuals(decomps, args.figures_dir, fig_paths)

    digest = {
        "smoke": bool(args.smoke),
        "behaviors": beh_list,
        "layers": layer_list,
        "headline_layer": hero_layer,
        "n_cell_jsons": len(cells),
        "n_chain_jsons": len(chains),
        "n_decomp_jsons": len(decomps),
        "n_figures": len(fig_paths),
        "identity_max_rel_residual": max(d["identity_max_rel_residual"] for d in decomps.values()),
        "exact_match_frac": {b: tstats[b]["exact_match_frac"] for b in tstats},
        "out_dir": str(args.out_dir),
        "figures_dir": str(args.figures_dir),
    }
    print(json.dumps({"issue833_fit_onpolicy_digest": digest}, indent=2))
    logger.info("[phase=fit_onpolicy] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
