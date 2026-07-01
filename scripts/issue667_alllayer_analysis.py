#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, Δc, r̂, r_B, →, ρ, M⁺, ※, ‖·‖, ×) in scientific docstrings + logs.
"""Issue #667 ALL-28-LAYER depth-profile analysis (exploratory, off-pod, READ-ONLY).

Reads the all-layer paired base/post-FT store this task's extraction wrote
(``issue667_alllayer/analysis_tensors`` on the HF data repo; local mirror
``eval_results/issue_667_alllayer/analysis_tensors``) and produces, PER residual
layer 0-27, two families of reads:

(a) **Δc decomposition** — the context-vector shift Δc = c_C⁺ − c_C, reusing
    :mod:`issue667_deltac_probe` verbatim (``analyze_behavior`` + ``cross_behavior``):
    ‖Δc‖/‖c_C‖ rel-drift, source-grain SVD (rank-one fraction), mean pairwise cos,
    |cos| of the dominant Δc direction to the answer-side write ŵ / to r_B / to
    the default-context offset, and the base-context-PCA captured fraction. Runs
    for all 4 behaviors (marker included — Δc needs no r_B).

(b) **Map-change (M0 vs M⁺)** — the context→answer function change, reusing
    :mod:`issue722_fit_M` verbatim (``fit_cell``): the function-change headline
    ``Δ_med = median_c |Δ(c)·r̂_B|`` vs the combined refit floor, and the
    co-primary chain-ρ ``Spearman(r_Bᵀ M̂(c), E)`` under M0 vs M⁺. Runs for the
    3 headline behaviors WITH an r_B (em/sycophancy/fact); marker is Δc-only (no
    r_B ⇒ no chain-ρ, no r_B-cos, per the extractor's marker carve-out).

Correctness gate: at layers 7/14/21 the all-layer run's numbers MUST reproduce
the committed 7/14/21 reads. ``--verify-gate`` recomputes those three layers from
BOTH the all-layer store AND the committed ``issue667_gate_chain_preview`` store
(the deltac probe's L14 rel-drift / SVD / cos-to-ŵ / cos-to-r_B) and diffs them,
FAILing loud on any mismatch beyond tolerance. It also cross-checks the map-change
reads against the committed #722 fit JSONs (``eval_results/issue_722/cells/``) when
present.

Outputs (READ-ONLY — never touches task bodies / eval_results commits are the
user's call): a per-layer JSON at ``--out`` and a markdown depth-profile report
(one row per layer for Δc, one per layer for map-change) to stdout + ``--md-out``.

Usage::

    # depth profile over all 28 layers (downloads the all-layer store, ~few GB)
    uv run python scripts/issue667_alllayer_analysis.py \\
        --behaviors em sycophancy fact marker \\
        --out /tmp/issue667_alllayer_profile.json \\
        --md-out /tmp/issue667_alllayer_profile.md

    # a subset of layers (e.g. the correctness-gate layers only)
    uv run python scripts/issue667_alllayer_analysis.py --layers 7 14 21 --verify-gate

    # CPU-only import/dry-run smoke against the EXISTING 7/14/21 store (no all-layer
    # store needed yet — validates the analysis path end-to-end pre-extraction)
    uv run python scripts/issue667_alllayer_analysis.py --layers 14 \\
        --store-prefix issue667_gate_chain_preview/analysis_tensors --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# DOTENV_LINT_EXEMPT: exploratory analysis script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

# Reused analysis modules — matched conventions, NOT re-implemented.
import issue667_deltac_probe as deltac  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402

logger = logging.getLogger("issue667.alllayer.analysis")

N_LAYERS = 28
ALL_LAYERS = tuple(range(N_LAYERS))
HIDDEN = 3584
DATA_REPO = "superkaiba1/explore-persona-space-data"
# The all-layer store this task's extraction writes (NEW namespace — never the
# committed 7/14/21 issue667_gate_chain_preview store).
ALLLAYER_STORE_PREFIX = "issue667_alllayer/analysis_tensors"
COMMITTED_STORE_PREFIX = "issue667_gate_chain_preview/analysis_tensors"
# Behaviors with an r_B (map-change chain-ρ + r_B-cos). marker has none.
MAP_CHANGE_BEHAVIORS = ("em", "sycophancy", "fact")
DELTAC_BEHAVIORS = ("em", "sycophancy", "fact", "marker")
# The three committed layers the correctness gate reproduces.
GATE_LAYERS = (7, 14, 21)
# Correctness-gate tolerances (all-layer read vs the committed 7/14/21 read). The
# all-layer extraction recomputes the SAME teacher-forced greedy reads at the SAME
# probes/seed, so agreement should be tight; a small tol absorbs float / greedy
# non-determinism across CUDA versions.
GATE_ABS_TOL = 1e-3
GATE_REL_TOL = 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# Store-prefix override — read the all-layer namespace via the reused loaders
# ─────────────────────────────────────────────────────────────────────────────


def _override_store_prefix(prefix: str) -> None:
    """Point the reused #722 map-change loader at ``prefix`` (the all-layer namespace).

    ``issue722_load_activations.list_store_layout`` reads the module-global
    ``STORE_PREFIX`` at call time, so setting the module attribute redirects its
    directory-tree walk to ``prefix`` WITHOUT editing the reused module (the same
    runtime-override pattern issue722_fit_M uses for fit658.DEVICE). A ``_Streamer``
    still defaults its prefix at def-time, so ``load_cells`` is always called with
    an explicit ``streamer=_Streamer(prefix=prefix)`` below. The Δc loader takes
    ``prefix`` as an argument directly (it is inlined here, not from deltac), so
    only the #722 loader's module global needs the override.
    """
    loadact.STORE_PREFIX = prefix
    logger.info("map-change store prefix -> %s", prefix)


# ─────────────────────────────────────────────────────────────────────────────
# (a) Δc decomposition per layer
#
# The COMPUTE (analyze_behavior / cross_behavior / fact_r_b_from_store) is reused
# verbatim from issue667_deltac_probe — those helpers are layer-AGNOSTIC (each
# per-layer npz already carries that layer's single-vectors). Only the LOADER is
# inlined here layer-parameterized, because the committed deltac probe's
# ``load_store()`` / ``load_r_b()`` / ``snapshot_l14()`` are hardcoded to LAYER=14
# (an unmerged sibling change adds a --layer arg; this script must NOT depend on
# unmerged work — reuse hierarchy / built-but-stranded rule). This inlined loader
# reads any of layers 0-27 from any store prefix (the all-layer namespace).
# ─────────────────────────────────────────────────────────────────────────────

# #658 r_b.pt column map (behavior -> column key). fact: None (from r_b_fact.pt /
# the per-cell store fallback); marker: None (no r_B). Mirrors deltac.RB_COL +
# issue722_fit_M.RB_COLUMN_KEY.
_RB_COL = {"em": "broad_em", "sycophancy": "sycophancy", "fact": None, "marker": None}
_RB_RECIPE = "diffmeans"
_R_B_PATH = "issue658_theory_assumptions/store/r_b.pt"


def _snapshot_layer(layer: int, prefix: str, local_store_root: str | None = None) -> Path:
    """Base dir holding the given layer's npz — local mirror if set, else HF snapshot.

    When ``local_store_root`` is set the complete store already lives on disk (the
    on-node mirror), so return that dir DIRECTLY with NO ``snapshot_download`` — the
    caller's ``base.rglob(f"*_L{layer}.npz")`` reads the local files. Its path
    contains ``analysis_tensors/<behavior>/<source>/<target>_L{layer}.npz`` so the
    caller's regex matches unchanged. Otherwise (``local_store_root is None``) this
    is the ORIGINAL HF path: bulk-download ONLY the given layer's npz from ``prefix``
    (~few GB, parallel), layer- AND prefix-parameterized (the committed deltac probe
    hardcodes L14 + the preview prefix).
    """
    if local_store_root is not None:
        base = Path(local_store_root)
        if not base.is_dir():
            raise FileNotFoundError(
                f"--local-store-root {base} is not a directory (expected the complete "
                "store mirror, layout <root>/<behavior>/<source>/<target>_L<layer>.npz)"
            )
        return base

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        DATA_REPO,
        repo_type="dataset",
        revision="main",
        allow_patterns=[f"{prefix}/*/*/*_L{layer}.npz"],
        max_workers=16,
    )
    return Path(root) / prefix


def _load_deltac_store(layer: int, prefix: str, local_store_root: str | None = None) -> dict:
    """{behavior: {(source, target): npz_dict}} for one layer (inlined, layer-aware).

    Parses each per-layer npz's single-vectors (c_C / c_Cp / c_C_postft /
    c_Cp_postft / v0 / v_plus / r_b_fact) — the exact key set deltac's
    ``analyze_behavior`` consumes; the (28,3584) all-layer STACK keys are NOT read
    (they are omitted from the all-layer store to keep it ~4.8 GB). Robust to the
    stacks being present (committed 7/14/21 store) or absent (all-layer store):
    only reads keys ``if k in z.files``.

    When ``local_store_root`` is set the files are read from that on-disk mirror
    (no HF download). The behavior/source/target are parsed from the LAST THREE
    path components (``<behavior>/<source>/<target>_L{layer}.npz``) — robust whether
    or not the root literally contains ``analysis_tensors`` — with the HF
    ``analysis_tensors/.../…`` anchor as a fallback for the snapshot layout.
    """
    import re
    from collections import defaultdict

    base = _snapshot_layer(layer, prefix, local_store_root)
    npz_files = sorted(str(p) for p in base.rglob(f"*_L{layer}.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"no *_L{layer}.npz under {base} "
            f"({'local mirror' if local_store_root else 'HF snapshot'}); "
            "check the store root / layer availability"
        )
    # Anchor on the last three path components so a local root that does NOT contain
    # the literal 'analysis_tensors' segment still parses correctly (silent
    # zero-match was the risk of the HF-only 'analysis_tensors/…' anchor).
    pat = re.compile(rf"([^/]+)/([^/]+)/([^/]+)_L{layer}\.npz$")
    out: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for i, p in enumerate(npz_files):
        m = pat.search(p)
        if not m:
            continue
        beh, cell, tgt = m.groups()
        source = cell.rsplit("_seed", 1)[0]
        z = np.load(p, allow_pickle=True)
        d = {}
        for k in ("v0", "v_plus", "c_C", "c_Cp", "c_C_postft", "c_Cp_postft", "r_b_fact"):
            if k in z.files:
                d[k] = z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k]
        out[beh][(source, tgt)] = d
        if (i + 1) % 200 == 0:
            logger.info("  loaded %d/%d npz (L%d)", i + 1, len(npz_files), layer)
    return out


def _load_r_b(behavior: str, layer: int) -> np.ndarray | None:
    """Unit-unnormalized r_B at (behavior, layer) from #658 r_b.pt; None if no column.

    em/sycophancy index the (28,3584) diffmeans stack at ``layer`` (the same
    per-layer resolution deltac.load_r_b uses). fact/marker have no r_b.pt column
    (fact falls back to the per-cell store r_b_fact; marker has none).
    """
    col = _RB_COL.get(behavior)
    if col is None:
        return None
    import torch
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(DATA_REPO, _R_B_PATH, repo_type="dataset")
    d = torch.load(Path(local), weights_only=False, map_location="cpu")
    return d["r_b"][col][_RB_RECIPE][layer].float().numpy().astype(np.float64)


def deltac_for_layer(
    layer: int, behaviors: list[str], prefix: str, local_store_root: str | None = None
) -> dict:
    """Run the Δc decomposition (issue667_deltac_probe.analyze_behavior) at one layer.

    r_B per behavior: em/syco from #658 r_b.pt (per-layer stack), fact from the
    per-cell store ``r_b_fact`` fallback (deltac.fact_r_b_from_store), marker None.
    The COMPUTE is the reused deltac ``analyze_behavior`` / ``cross_behavior``;
    only the LOADER + r_B resolution are inlined layer-aware (see the section
    header). ``local_store_root`` (if set) reads the on-disk store mirror instead
    of downloading from HF. Returns ``{behavior: analyze_behavior(...)}`` + the
    cross-behavior block.
    """
    store = _load_deltac_store(layer, prefix, local_store_root)
    results: dict[str, dict] = {}
    for b in behaviors:
        cells = store.get(b, {})
        r_b = _load_r_b(b, layer)
        if r_b is None and b == "fact":
            r_b = deltac.fact_r_b_from_store(cells)  # committed, layer-agnostic
        results[b] = deltac.analyze_behavior(b, cells, r_b)  # committed, layer-agnostic
    xbeh = deltac.cross_behavior(results)  # committed, layer-agnostic
    return {"by_behavior": results, "cross_behavior": xbeh}


# ─────────────────────────────────────────────────────────────────────────────
# (b) Map-change per layer (reuse issue722_fit_M.fit_cell)
# ─────────────────────────────────────────────────────────────────────────────


def _load_cells_alllayer(
    behaviors: list[str],
    layers: list[int],
    prefix: str,
    *,
    strict: bool,
    max_sources: int | None = None,
    max_targets_per_source: int | None = None,
    local_store_root: str | None = None,
):
    """Load the paired cell records for the map-change fit via the reused #722 loader.

    HF mode (``local_store_root is None``): redirects ``loadact.STORE_PREFIX`` to
    ``prefix`` and passes an explicit ``_Streamer(prefix=prefix)`` (the streamer
    captures its default prefix at def-time); the loader's HF tree walk enumerates
    the layout.

    Local-mirror mode (``local_store_root`` set): enumerates the layout from the
    on-disk mirror via ``loadact.list_store_layout_local`` (NO HF tree walk — the
    walk is the map-change hang) and passes a ``_Streamer(local_root=…)`` that reads
    npz DIRECTLY from disk (NO HF download). The loader consumes both unchanged, so
    the returned ``{(behavior, layer): [CellRecord]}`` shape ``fitM.fit_cell`` needs
    is identical to the HF path.

    ``strict_counts`` asserts the verified 480-cell per-behavior×layer grid — enabled
    ONLY for a fully-extracted store; a subset / smoke disables it. ``max_sources`` /
    ``max_targets_per_source`` are the reused loader's smoke knobs (bound the stream +
    the fit cost). ``max_sources`` is the right knob to keep the fit non-degenerate
    (c_C is constant within a source, so a smoke MUST span >=2 SOURCES, not just
    multiple targets of one source).
    """
    if local_store_root is not None:
        streamer = loadact._Streamer(local_root=local_store_root)
        layout = loadact.list_store_layout_local(local_store_root, tuple(behaviors))
        try:
            return loadact.load_cells(
                behaviors=tuple(behaviors),
                layers=tuple(layers),
                streamer=streamer,
                strict_counts=strict,
                max_sources=max_sources,
                max_targets_per_source=max_targets_per_source,
                layout=layout,
            )
        finally:
            streamer.cleanup()

    _override_store_prefix(prefix)
    streamer = loadact._Streamer(prefix=prefix)
    try:
        return loadact.load_cells(
            behaviors=tuple(behaviors),
            layers=tuple(layers),
            streamer=streamer,
            strict_counts=strict,
            max_sources=max_sources,
            max_targets_per_source=max_targets_per_source,
        )
    finally:
        streamer.cleanup()


def map_change_for_layers(
    behaviors: list[str],
    layers: list[int],
    prefix: str,
    *,
    strict: bool,
    smoke_clamp: bool,
    max_sources: int | None = None,
    max_targets_per_source: int | None = None,
    local_store_root: str | None = None,
) -> dict:
    """Run the M0-vs-M⁺ map-change fit (issue722_fit_M.fit_cell) per (behavior, layer).

    Returns ``{f"{behavior}_L{layer}": fit_cell(...)}``. r_B: em/syco from
    r_b.pt, fact from r_b_fact.pt (both full 28-layer stacks the reused
    ``_r_hat_for`` indexes per layer). ``smoke_clamp`` reduces the CPU MLP /
    refit-pair cost so the analysis path runs end-to-end on the VM CPU as a
    carve-out smoke (mirrors issue722_fit_M --smoke clamps).
    """
    mc_behaviors = [b for b in behaviors if b in MAP_CHANGE_BEHAVIORS]
    if not mc_behaviors:
        logger.info("map-change: no r_B behavior in %s — skipping map-change reads", behaviors)
        return {}

    if smoke_clamp:
        # Clamp the three dominant CPU costs so the GPU-bound MLP phase runs on the
        # VM CPU as a smoke (the full GPU run uses 300 epochs / 100 pairs / 64 dims).
        fitM.fit658.MLP_MAX_EPOCHS = 20
        fitM.N_REFIT_PAIRS = 8
        fitM.TARGET_DIM = 4
        logger.info("map-change: SMOKE clamps (mlp_epochs=20 refit_pairs=8 target_dim=4)")

    # Resolve the compute device the reused ridge + MLP fitters read off
    # fit658.DEVICE ("auto" -> cuda if available else cpu; issue722_fit_M.main sets
    # this, which we bypass by calling fit_cell directly).
    fitM.fit658.DEVICE = fitM.fit658._resolve_device("auto")
    logger.info("map-change: fit device=%s", fitM.fit658.DEVICE)
    # Exactness gate (#658): a reduction-order regression fails at startup.
    fitM.fit658._assert_ridge_exactness()

    rb_main = fitM._load_rb_main()
    rb_fact = fitM._load_rb_fact() if "fact" in mc_behaviors else None
    if "fact" in mc_behaviors and rb_fact is None:
        logger.warning("fact requested but r_b_fact.pt unavailable/degenerate — dropping fact")
        mc_behaviors = [b for b in mc_behaviors if b != "fact"]

    cells_by = _load_cells_alllayer(
        mc_behaviors,
        layers,
        prefix,
        strict=strict,
        max_sources=max_sources,
        max_targets_per_source=max_targets_per_source,
        local_store_root=local_store_root,
    )
    out: dict[str, dict] = {}
    for behavior in mc_behaviors:
        for layer in layers:
            cells = cells_by[(behavior, layer)]
            logger.info("map-change: %s L%d (%d cells)", behavior, layer, len(cells))
            out[f"{behavior}_L{layer}"] = fitM.fit_cell(behavior, layer, cells, rb_main, rb_fact)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Correctness gate: 7/14/21 all-layer read vs the committed 7/14/21 read
# ─────────────────────────────────────────────────────────────────────────────


def _close(a: float | None, b: float | None) -> bool:
    """abs/rel tolerance NaN-safe close (both-None / both-NaN -> True)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, atol=GATE_ABS_TOL, rtol=GATE_REL_TOL, equal_nan=True))


# The Δc scalar reads the gate cross-checks between the two stores (per behavior).
_GATE_DELTAC_KEYS = [
    ("rel_drift_deltac_over_cC", "median"),
    ("svd_over_sources", "top1_frac"),
    ("svd_over_sources", "top2_frac"),
    ("cohesion", "mean_pairwise_cos"),
]
_GATE_ALIGN_KEYS = ["cos_v1_what_mean", "cos_v1_rb", "v1_base_pca_captured_k5"]


def _extract_gate_scalars(analyze_result: dict) -> dict:
    """Pull the flat scalar reads the gate diffs from an analyze_behavior result."""
    if analyze_result.get("status") != "ok":
        return {"status": analyze_result.get("status", "?")}
    flat: dict[str, float | None] = {}
    for outer, inner in _GATE_DELTAC_KEYS:
        flat[f"{outer}.{inner}"] = analyze_result.get(outer, {}).get(inner)
    align = analyze_result.get("alignment", {})
    for k in _GATE_ALIGN_KEYS:
        flat[k] = align.get(k)
    return flat


def verify_gate(
    behaviors: list[str], alllayer_prefix: str, local_store_root: str | None = None
) -> dict:
    """Reproduce the committed 7/14/21 Δc reads from the all-layer store; diff loud.

    For each gate layer (7/14/21) and behavior, recompute the deltac
    ``analyze_behavior`` scalars from BOTH the all-layer store AND the committed
    ``issue667_gate_chain_preview`` store, and assert they agree within tolerance.
    A mismatch means the all-layer extraction diverged from the committed reads at
    a shared layer (a bug in the depth extension) — RAISE loud with the offending
    keys. Returns a structured diff report (also embedded in the JSON output).

    ``local_store_root`` (if set) reads the ALL-LAYER side from the on-disk mirror;
    the COMMITTED reference side is ALWAYS fetched from HF (``COMMITTED_STORE_PREFIX``)
    — the committed store is the ground truth the local all-layer read is checked
    against, so it must come from the canonical source. (If the committed
    ``issue667_gate_chain_preview`` HF listing also hangs, run without --verify-gate
    on the pod; the gate is a correctness cross-check, not a data path.)
    """
    logger.info(
        "[verify-gate] reproducing committed L%s Δc reads from the all-layer store", GATE_LAYERS
    )
    report: dict = {"tolerance": {"abs": GATE_ABS_TOL, "rel": GATE_REL_TOL}, "layers": {}}
    mismatches: list[str] = []
    for layer in GATE_LAYERS:
        committed = deltac_for_layer(layer, behaviors, COMMITTED_STORE_PREFIX)["by_behavior"]
        alllayer = deltac_for_layer(layer, behaviors, alllayer_prefix, local_store_root)[
            "by_behavior"
        ]
        layer_rep: dict = {}
        for b in behaviors:
            cflat = _extract_gate_scalars(committed.get(b, {}))
            aflat = _extract_gate_scalars(alllayer.get(b, {}))
            keys = sorted(set(cflat) | set(aflat))
            per_key = {}
            for k in keys:
                cv, av = cflat.get(k), aflat.get(k)
                ok = (
                    _close(
                        cv if isinstance(cv, (int, float)) else None,
                        av if isinstance(av, (int, float)) else None,
                    )
                    if k != "status"
                    else (cv == av)
                )
                per_key[k] = {"committed": cv, "alllayer": av, "close": ok}
                if not ok:
                    mismatches.append(f"{b} L{layer} {k}: committed={cv} alllayer={av}")
            layer_rep[b] = per_key
        report["layers"][layer] = layer_rep
    report["n_mismatches"] = len(mismatches)
    report["mismatches"] = mismatches
    if mismatches:
        raise RuntimeError(
            "[verify-gate] CORRECTNESS GATE FAILED — the all-layer store does not "
            f"reproduce the committed 7/14/21 Δc reads at {len(mismatches)} scalar(s):\n  "
            + "\n  ".join(mismatches[:20])
        )
    logger.info("[verify-gate] PASS — all-layer 7/14/21 reads reproduce the committed store")
    return report


def _cross_check_committed_map_change(mc: dict) -> dict:
    """Diff the all-layer map-change 7/14/21 headline vs committed #722 cell JSONs.

    #722 wrote ``eval_results/issue_722/cells/<behavior>_L{7,14,21}.json`` with the
    same ``Delta_med`` / ``floor_combined`` / chain-ρ shape ``fit_cell`` produces.
    When those local JSONs are present, diff the headline scalars as a SOFT gate
    (logged; not raised — the committed #722 fit used the full-grid strict store
    and 100 refit pairs, so exact bit agreement is not expected under a smoke, but
    a gross divergence at a shared layer is worth surfacing). Returns the diff or a
    ``skipped`` note.
    """
    committed_dir = PROJECT_ROOT / "eval_results/issue_722/cells"
    if not committed_dir.is_dir():
        return {"status": "skipped — no committed eval_results/issue_722/cells"}
    diffs: dict = {}
    for key, cell in mc.items():
        # key is "<behavior>_L<layer>"; #722 only fit layers 7/14/21.
        layer = int(key.rsplit("_L", 1)[1])
        if layer not in GATE_LAYERS:
            continue
        cpath = committed_dir / f"{key}.json"
        if not cpath.exists():
            continue
        committed = json.loads(cpath.read_text())
        diffs[key] = {
            k: {"committed": committed.get(k), "alllayer": cell.get(k)}
            for k in ("Delta_med", "floor_combined", "Delta_over_floor_sd")
        }
    return diffs or {"status": "skipped — no shared 7/14/21 #722 cells present"}


# ─────────────────────────────────────────────────────────────────────────────
# Markdown depth-profile report
# ─────────────────────────────────────────────────────────────────────────────


def _g(d: dict, *path, default=float("nan")):
    """Nested get with a NaN default."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default


def render_depth_profile_md(
    deltac_by_layer: dict, map_change: dict, layers: list[int], behaviors: list[str]
) -> str:
    """One-row-per-layer markdown tables for (a) Δc and (b) map-change, per behavior."""
    lines: list[str] = ["# Issue #667 all-28-layer depth profile (seed 42)\n"]

    # (a) Δc decomposition — one table per behavior, one row per layer.
    lines.append("## (a) Δc context-vector shift — depth profile\n")
    for b in behaviors:
        lines.append(f"### {b}\n")
        lines.append(
            "| layer | n_src | ‖Δc‖/‖c_C‖ med | SVD top1 | top2 | chance | "
            "mean pair cos | v1·ŵ | v1·r_B | v1 base-PCA k5 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for layer in layers:
            r = _g(deltac_by_layer, layer, "by_behavior", b, default={})
            if not isinstance(r, dict) or r.get("status") != "ok":
                status = r.get("status", "?") if isinstance(r, dict) else "?"
                lines.append(f"| {layer} | — | {status} | | | | | | | |")
                continue
            a = r.get("alignment", {})
            lines.append(
                f"| {layer} | {r['n_sources']} | "
                f"{_g(r, 'rel_drift_deltac_over_cC', 'median'):.3f} | "
                f"{_g(r, 'svd_over_sources', 'top1_frac'):.3f} | "
                f"{_g(r, 'svd_over_sources', 'top2_frac'):.3f} | "
                f"{_g(r, 'svd_over_sources', 'chance_top1_frac'):.3f} | "
                f"{_g(r, 'cohesion', 'mean_pairwise_cos'):.3f} | "
                f"{_g(a, 'cos_v1_what_mean'):.3f} | "
                f"{_g(a, 'cos_v1_rb'):.3f} | "
                f"{_g(a, 'v1_base_pca_captured_k5'):.3f} |"
            )
        lines.append("")

    # (b) Map-change (M0 vs M⁺) — one table per behavior, one row per layer.
    lines.append("## (b) Context→answer map change (M0 vs M⁺) — depth profile\n")
    for b in behaviors:
        if b not in MAP_CHANGE_BEHAVIORS:
            lines.append(f"### {b}\n\n(no r_B — Δc-only; map-change not computed)\n")
            continue
        lines.append(f"### {b}\n")
        lines.append(
            "| layer | n | Δ_med | floor_comb | Δ/floor_sd | ρ(M0) | ρ(M⁺) | ρ_diff | "
            "support ‖cplus−c0‖ mean |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for layer in layers:
            cell = map_change.get(f"{b}_L{layer}")
            if not cell:
                lines.append(f"| {layer} | — | not-fit | | | | | | |")
                continue
            cr = cell.get("chain_rho", {})
            lines.append(
                f"| {layer} | {cell.get('n_cells', '—')} | "
                f"{_g(cell, 'Delta_med'):.4g} | "
                f"{_g(cell, 'floor_combined'):.4g} | "
                f"{_g(cell, 'Delta_over_floor_sd'):.3f} | "
                f"{_g(cr, 'rho_M0_ridge'):.3f} | "
                f"{_g(cr, 'rho_Mplus_ridge'):.3f} | "
                f"{_g(cr, 'rho_diff_ridge'):.3f} | "
                f"{_g(cell, 'support_distance', 'mean'):.4g} |"
            )
        lines.append("")

    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #667 all-28-layer depth-profile analysis")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(ALL_LAYERS),
        help="residual layers to profile (default: all 0-27)",
    )
    ap.add_argument("--behaviors", nargs="+", default=list(DELTAC_BEHAVIORS))
    ap.add_argument(
        "--store-prefix",
        default=ALLLAYER_STORE_PREFIX,
        help="HF store prefix to read (default: the all-layer namespace; use the "
        "committed issue667_gate_chain_preview prefix for a pre-extraction smoke)",
    )
    ap.add_argument(
        "--local-store-root",
        default=None,
        help="read npz DIRECTLY from this on-disk store mirror (layout "
        "<root>/<behavior>/<source>_seed42/<target>_L<layer>.npz), skipping ALL HF "
        "downloads/tree-walks — use when the HF repo-listing hangs and the complete "
        "store already lives on the compute node (e.g. "
        "eval_results/issue_667_alllayer/analysis_tensors). Applies to BOTH the Δc "
        "and map-change store reads; --verify-gate still fetches the committed "
        "reference from HF.",
    )
    ap.add_argument("--out", default="/tmp/issue667_alllayer_profile.json")
    ap.add_argument("--md-out", default=None, help="also write the markdown report here")
    ap.add_argument(
        "--skip-map-change",
        action="store_true",
        help="Δc-only (skip the CPU/GPU-heavy M0/M⁺ fit)",
    )
    ap.add_argument(
        "--verify-gate",
        action="store_true",
        help="reproduce the committed 7/14/21 Δc reads from the all-layer store + FAIL on mismatch",
    )
    ap.add_argument(
        "--strict-counts",
        action="store_true",
        help="assert the full 480-cell/behavior×layer grid in the map-change loader "
        "(only for a fully-extracted all-layer store)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="clamp the map-change CPU cost (mlp_epochs/refit_pairs/target_dim) so the "
        "analysis path runs end-to-end on the VM CPU as a carve-out smoke",
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="smoke: cap source_cid dirs per behavior in the map-change loader (the "
        "distinct-c0 count; MUST be >=2 for a non-degenerate fit). Bounds the HF "
        "stream so the map-change smoke does not fetch all 480 cells.",
    )
    ap.add_argument(
        "--max-targets-per-source",
        type=int,
        default=None,
        help="smoke: cap targets per source in the map-change loader.",
    )
    args = ap.parse_args()

    layers = sorted(set(args.layers))
    for layer in layers:
        assert 0 <= layer < N_LAYERS, f"layer {layer} out of range [0, {N_LAYERS - 1}]"

    t0 = time.time()

    # Optional correctness gate FIRST — cheap-ish Δc-only, fails loud before the
    # heavier map-change fit if the all-layer store diverges at 7/14/21.
    gate_report = None
    if args.verify_gate:
        gate_report = verify_gate(args.behaviors, args.store_prefix, args.local_store_root)

    # (a) Δc decomposition per layer.
    logger.info("[phase=deltac] Δc decomposition over %d layers", len(layers))
    deltac_by_layer: dict = {}
    for layer in layers:
        deltac_by_layer[layer] = deltac_for_layer(
            layer, args.behaviors, args.store_prefix, args.local_store_root
        )
        logger.info("[phase=deltac] layer %d done", layer)

    # (b) Map-change per layer (unless skipped).
    map_change: dict = {}
    map_change_xcheck = None
    if not args.skip_map_change:
        logger.info("[phase=map_change] M0 vs M⁺ fit over %d layers", len(layers))
        map_change = map_change_for_layers(
            args.behaviors,
            layers,
            args.store_prefix,
            strict=args.strict_counts,
            smoke_clamp=args.smoke,
            max_sources=args.max_sources,
            max_targets_per_source=args.max_targets_per_source,
            local_store_root=args.local_store_root,
        )
        map_change_xcheck = _cross_check_committed_map_change(map_change)

    # Strip private vectors (deltac's analyze_behavior keeps _v1 / _mean_delta).
    deltac_clean = {}
    for layer, blk in deltac_by_layer.items():
        deltac_clean[layer] = {
            "by_behavior": {
                b: {k: v for k, v in r.items() if not k.startswith("_")}
                for b, r in blk["by_behavior"].items()
            },
            "cross_behavior": blk["cross_behavior"],
        }

    md = render_depth_profile_md(deltac_by_layer, map_change, layers, args.behaviors)

    out_obj = {
        "issue": 667,
        "analysis": "all_28_layer_depth_profile",
        "seed": 42,
        "layers": layers,
        "behaviors": args.behaviors,
        "store_prefix": args.store_prefix,
        "local_store_root": args.local_store_root,
        "deltac_by_layer": deltac_clean,
        "map_change": map_change,
        "map_change_committed_xcheck": map_change_xcheck,
        "verify_gate": gate_report,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2, default=_json_default))
    logger.info("wrote %s", args.out)
    if args.md_out:
        Path(args.md_out).write_text(md)
        logger.info("wrote %s", args.md_out)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
