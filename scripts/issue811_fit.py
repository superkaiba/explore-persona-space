#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, r̂, r_B, ρ, M⁺, →, ×) in scientific docstrings + logs.
"""Issue #811 — fit M0 vs M⁺ under the mean AND turn_nl answer summaries (plan §4.3).

Single manipulated variable vs #722: the answer-side summary of ``v0``/``v_plus``
(``mean`` over-response → ``turn_nl``, the turn-boundary single-position read).
EVERY other axis is inherited VERBATIM from #722 — the ridge headline, the four
DVs, the combined refit-variance floor, the family-clustered bootstrap, the
chain-ρ join to #537's G, seed 42 — by REUSING ``issue722_fit_M.fit_cell`` for
the closed-form ridge reads (``include_mlp=False``, the ridge headline is complete
without the serial MLP) and threading the ``summary`` axis through
``issue722_load_activations`` at #811's re-extracted store prefix.

The MLP-vs-shuffle VALIDITY GATE (the only gradient-descent arm, plan §4.3) runs
through the VECTORIZED ``analysis/vectorized_mlp_skill.fit_batched_loco_mlp_multihead``
— mandatory per ``.claude/rules/vectorize-many-cell-fits.md`` (#722's own 19.5-CPU-h
serial-loop scar) — batching (behavior × layer × summary × {base, shuffle}) into
ONE vmapped LOCO ensemble. NO per-cell Python loop. The gate compares
``Spearman(r_Bᵀ MLP_pred(c), E)`` on the REAL base-leg map (M0: c0 → v0) vs on a
row-shuffled v0 null; a gate that does not collapse relative to `mean` at the
primary layer 14 is the base-leg validity signal (H3 / KILL-1).

Phases (all reading the extended store; the base leg IS half the paired store,
so the KILL-1 base-leg validity read costs no extra forward pass):

- Per (behavior, layer, summary): ridge headline via ``fit_cell`` +
  the vectorized MLP-vs-shuffle validity-gate ρ (base + shuffle), checkpointed to
  ``eval_results/issue_811/cells/{behavior}_L{li}_{summary}.json``.
- KILL-1 gate at layer 14: if ``turn_nl``'s validity gate COLLAPSES relative to
  ``mean`` on ≥2 of 3 primary-layer behaviors, emit a loud structured
  ``KILL1_TRIGGERED`` log line (the GCP/poller orchestrator persists it as
  ``epm:failure failure_class: data``; pod-side code never shells task.py) and
  stop before any Phase-1 dependent read is trusted downstream.

Per-(behavior, layer, summary) checkpoints (resume-skip, mirroring #722's
``_CELL_SCHEMA_KEYS`` contract) make a mid-run crash re-fit only the missing cells.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    fit_batched_loco_mlp_multihead,
)

logger = logging.getLogger("issue811.fit")

SWEEP_LAYERS = (7, 14, 21)
PRIMARY_LAYER = 14
HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
SUMMARIES = ("mean", "turn_nl")

# #811's OWN re-extracted store (distinct from #667's mean-only store). The 480-cell
# (16 sources × 30 targets) per-behavior×layer grid is enforced by the loader's
# strict_counts; the marker behavior is dropped from #811's headline exactly as #722
# dropped it (headline = em / fact / sycophancy).
STORE_PREFIX = "issue811_turn_nl_mapchange/analysis_tensors"

# KILL-1 base-leg validity-gate collapse: turn_nl's MLP-vs-shuffle gate (base-leg
# ρ_real − ρ_shuffle) is a COLLAPSE relative to mean when it drops below this
# fraction of mean's gate margin at layer 14 (plan §4.1 / §7 / H3). A margin at or
# above (COLLAPSE_FRAC × mean margin) is NOT a collapse. Grounded on #722's
# MLP-vs-shuffle gate being the base-map validity control; the 0.5 fraction is the
# "gate roughly halved vs mean" collapse threshold (turn_nl is a WORSE summary if
# its base-map read loses most of mean's shuffle-margin). A non-positive mean
# margin means mean itself has no gate to collapse relative to — not a turn_nl
# failure — so that behavior does NOT count toward KILL-1 (reported, not killed).
KILL1_COLLAPSE_FRAC = 0.5


def _resolve_device() -> str:
    """Compute device the ridge + vectorized MLP read off (auto → cuda else cpu)."""
    return fit658._resolve_device("auto")


def _mlp_gate_groups(
    stacks: dict, pca_basis: np.ndarray, seed: int = 722
) -> tuple[list[MLPGroup], np.ndarray]:
    """Build the (base, shuffle) MLPGroups for ONE (behavior, layer, summary) cell.

    The base-leg validity gate is the MLP map on (c0 → v0-top-PCs) vs a row-shuffled
    v0 null. Returns ([base_group, shuffle_group], shuffle_perm) with each group's
    ``Y`` the PCA-reduced base-leg v0 (mirroring #722's ridge shuffle null, which
    permutes ``V0_64``). Both groups share the SAME (n, d_in) c0 design so they ride
    one batched ensemble.
    """
    C0 = stacks["C0"].astype(np.float32)  # (n, HIDDEN) base context grid
    V0 = stacks["V0"]  # (n, HIDDEN) base answer profile
    V0_64 = (V0 @ pca_basis.T).astype(np.float32)  # (n, k) shared top-64 v0 PCs
    rng = np.random.default_rng(seed)
    perm = rng.permutation(C0.shape[0])
    base_g = MLPGroup(key=("base",), X=C0, Y=V0_64)
    shuf_g = MLPGroup(key=("shuffle",), X=C0, Y=V0_64[perm])
    return [base_g, shuf_g], perm


def compute_mlp_validity_gate(
    groups_by_cell: dict[tuple[str, int, str], tuple[list[MLPGroup], np.ndarray]],
    cell_meta: dict[tuple[str, int, str], dict],
    *,
    device: str,
    max_epochs: int,
    num_threads: int | None = None,
) -> dict[tuple[str, int, str], dict]:
    """VECTORIZED MLP-vs-shuffle base-leg validity gate for ALL cells in ONE ensemble.

    Batches every (behavior, layer, summary) cell's (base, shuffle) MLPGroups into a
    SINGLE ``fit_batched_loco_mlp_multihead`` call (the mandated vectorization —
    NO per-cell serial loop). For each cell computes:

    - ``rho_real`` = Spearman(r_Bᵀ (MLP held-out pred of v0), E) on the base leg,
    - ``rho_shuffle`` = the same on the row-shuffled v0 null,
    - ``gate_margin`` = ``rho_real − rho_shuffle`` (the base-map validity margin;
      the diagnostic that FAILED in 8/9 #722 cells for the mean summary).

    Every group in the batch must share (n, d_in, p) — cells whose n differs (a
    smoke may cap sources per behavior) are grouped by shape and fit in separate
    batched calls. Returns ``{cell_key: {rho_real, rho_shuffle, gate_margin,
    n_with_E}}``.
    """
    # Group cells by (n, d_in, p) so each batched call is homogeneous-shaped.
    from collections import defaultdict

    # Group cells by (n, d_in, p) — one homogeneous-shaped batched call per shape
    # (a smoke may cap sources so n differs across behaviors). Every group's key is
    # prefixed with its cell key so the flat batch is unambiguous.
    flat_groups: dict[tuple[int, int, int], list[MLPGroup]] = defaultdict(list)
    for cell_key, (cell_groups, _perm) in groups_by_cell.items():
        n, d_in = cell_groups[0].X.shape
        shape = (n, d_in, cell_groups[0].Y.shape[1])
        for g in cell_groups:
            flat_groups[shape].append(MLPGroup(key=(cell_key, g.key), X=g.X, Y=g.Y))

    preds_by_tag: dict[tuple, np.ndarray] = {}
    for groups in flat_groups.values():
        res = fit_batched_loco_mlp_multihead(
            groups, max_epochs=max_epochs, device=device, num_threads=num_threads
        )
        preds_by_tag.update(res.preds_by_key)

    out: dict[tuple[str, int, str], dict] = {}
    for cell_key in groups_by_cell:
        meta = cell_meta[cell_key]
        pca_basis, r_hat, E, keep = meta["pca_basis"], meta["r_hat"], meta["E"], meta["keep"]
        n_with_E = int(keep.sum())
        rec = {"rho_real": None, "rho_shuffle": None, "gate_margin": None, "n_with_E": n_with_E}
        if n_with_E >= 4:
            Ek = E[keep]
            base_pred = preds_by_tag[(cell_key, ("base",))]  # (n, k)
            shuf_pred = preds_by_tag[(cell_key, ("shuffle",))]
            rho_real, _ = fitM._chain_rho_one(base_pred[keep], pca_basis, r_hat, Ek)
            rho_shuf, _ = fitM._chain_rho_one(shuf_pred[keep], pca_basis, r_hat, Ek)
            rec["rho_real"] = rho_real
            rec["rho_shuffle"] = rho_shuf
            if rho_real is not None and rho_shuf is not None:
                rec["gate_margin"] = float(rho_real - rho_shuf)
        out[cell_key] = rec
    return out


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON via tmp + ``os.replace`` so a mid-write kill never leaves a
    truncated checkpoint (the resume validator would re-fit it, losing the unit)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def _cached_cell_state(path: Path) -> str:
    """Classify a per-unit checkpoint: ``"merged"`` | ``"ridge_only"`` | ``"invalid"``.

    - ``merged``: full #811 schema (fitM ridge keys + ``summary`` +
      ``mlp_validity_gate``) — the post-gate-merge shape; resume skips the unit
      entirely (ridge AND gate reused).
    - ``ridge_only``: the fitM ridge schema keys are complete but the MLP
      validity gate has not been merged yet — the IMMEDIATE per-unit checkpoint
      written the moment ``fit_cell`` returns (#811 vectorize fix round 2, so a
      mid-run kill loses at most the in-flight unit instead of everything).
      Resume reuses the ridge result (the expensive part) and re-runs only the
      cheap gate-group assembly + batched gate for that unit.
    - ``invalid``: unparseable or missing ridge schema keys — re-fit.
    """
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[phase=fit_M] cached %s unreadable (%s) — will re-fit", path.name, e)
        return "invalid"
    missing = fitM._CELL_SCHEMA_KEYS - obj.keys()
    if missing:
        logger.warning(
            "[phase=fit_M] cached %s missing schema keys %s — will re-fit",
            path.name,
            sorted(missing),
        )
        return "invalid"
    if "summary" in obj and "mlp_validity_gate" in obj:
        return "merged"
    return "ridge_only"


def _cached_cell_valid(path: Path) -> bool:
    """True iff ``path`` is a complete MERGED #811 per-cell checkpoint (legacy API)."""
    return _cached_cell_state(path) == "merged"


def _load_store_layout(behaviors: tuple[str, ...], *, local_root: str | None):
    """Return the store directory layout for #811's store (HF prefix or local mirror)."""
    if local_root:
        return loadact.list_store_layout_local(local_root, behaviors)
    return loadact.list_store_layout(behaviors, prefix=STORE_PREFIX)


def _streamer_for(local_root: str | None):
    """A #811-prefixed HF streamer, or a local-mirror streamer (no HF fetch)."""
    if local_root:
        return loadact._Streamer(local_root=local_root)
    return loadact._Streamer(prefix=STORE_PREFIX)


def _load_and_prepare_cells(
    behaviors: tuple[str, ...],
    layers: tuple[int, ...],
    summaries: tuple[str, ...],
    *,
    args,
    layout,
    strict: bool,
    rb_main: dict,
    rb_fact: dict | None,
) -> tuple[dict, dict, dict]:
    """Load cells per summary + build the ridge headline & batched-gate inputs.

    Returns ``(ridge_cells, groups_by_cell, cell_meta)`` keyed by
    ``(behavior, layer, summary)``. Loads each summary once (a fresh streamer per
    summary so LRU staging never collides; the layout is shared) and, per
    non-cached cell, runs the byte-identical #722 ridge headline
    (``fit_cell(include_mlp=False)`` — the serial MLP is REPLACED by the vectorized
    validity gate) + assembles the (base, shuffle) MLPGroups and the per-cell meta
    (pca_basis / r_hat / E / keep) the gate consumes. Resume is TWO-STATE
    (``_cached_cell_state``): a MERGED cell (ridge + gate) is reloaded and skipped
    entirely; a RIDGE-ONLY checkpoint (written the moment ``fit_cell`` returned,
    #811 vectorize fix round 2) reuses the ridge result and re-runs only the cheap
    gate assembly. Extracted from :func:`main` (C901).
    """
    cells_by = {}
    for summary in summaries:
        cells_by[summary] = loadact.load_cells(
            behaviors=behaviors,
            layers=layers,
            max_sources=args.max_sources,
            max_targets_per_source=args.max_targets_per_source,
            streamer=_streamer_for(args.local_root),
            strict_counts=strict,
            layout=layout,
            summary=summary,
        )

    ridge_cells: dict[tuple[str, int, str], dict] = {}
    groups_by_cell: dict[tuple[str, int, str], tuple] = {}
    cell_meta: dict[tuple[str, int, str], dict] = {}
    n_units = len(summaries) * len(behaviors) * len(layers)
    unit_i = 0
    for summary in summaries:
        for behavior in behaviors:
            for layer in layers:
                cells = cells_by[summary][(behavior, layer)]
                cell_key = (behavior, layer, summary)
                unit_i += 1
                if len(cells) < 4:
                    logger.warning(
                        "[phase=fit_M] %s L%d %s: %d cells (<4) — skip",
                        behavior,
                        layer,
                        summary,
                        len(cells),
                    )
                    continue
                out = args.out_dir / f"{behavior}_L{layer}_{summary}.json"
                state = "absent"
                if not args.force_rerun and out.exists():
                    state = _cached_cell_state(out)
                if state == "merged":
                    logger.info("[phase=fit_M] %s (cached — skip)", out.name)
                    ridge_cells[cell_key] = json.loads(out.read_text())
                    continue
                if state == "ridge_only":
                    # Resume from the immediate per-unit checkpoint: reuse the
                    # ridge result (the expensive fit), re-run only the gate.
                    logger.info(
                        "[phase=fit_M] %s (ridge checkpoint — reusing ridge, re-running gate)",
                        out.name,
                    )
                    ridge_cells[cell_key] = json.loads(out.read_text())
                else:
                    # Ridge headline (byte-identical to #722; include_mlp=False —
                    # the serial MLP is REPLACED by the vectorized validity gate).
                    t0 = time.monotonic()
                    ridge = fitM.fit_cell(
                        behavior, layer, cells, rb_main, rb_fact, include_mlp=False
                    )
                    ridge["summary"] = summary
                    # Checkpoint the unit the MOMENT its ridge result exists (#811
                    # vectorize fix round 2): a mid-run kill loses at most the
                    # in-flight unit, never completed units. The gate-merge loop
                    # in main() rewrites this file with the merged gate later.
                    _atomic_write_json(out, ridge)
                    ridge_cells[cell_key] = ridge
                    logger.info(
                        "[phase=fit_M] unit %d/%d %s fit in %.1fs — checkpointed",
                        unit_i,
                        n_units,
                        out.name,
                        time.monotonic() - t0,
                    )
                stacks = loadact.stack_for_fit(cells)
                pca_basis = fitM._pca_basis_v0(stacks["V0"], args.target_dim)
                E = fitM._load_E(behavior, stacks["cell_keys"])
                groups_by_cell[cell_key] = _mlp_gate_groups(stacks, pca_basis)
                cell_meta[cell_key] = {
                    "pca_basis": pca_basis,
                    "r_hat": fitM._r_hat_for(behavior, layer, rb_main, rb_fact),
                    "E": E,
                    "keep": ~np.isnan(E),
                }
    return ridge_cells, groups_by_cell, cell_meta


# ── Phase 0 base-leg store loading (KILL-1 pre-spend gate) ─────────────────────
# The Phase-0 store (phase0_base_leg/) carries ONLY the base leg — c_C / v0 /
# v0_turn_nl per cell (NO v_plus / c_C_postft). The full-store loader
# (issue722_load_activations._blob_to_record) hard-requires the trained-leg keys,
# so Phase 0 uses this base-only loader: it builds the SAME (base_group,
# shuffle_group) MLPGroups + cell_meta the paired-store gate uses, reading C0 =
# c_C and V0 = v0 (mean) / v0_turn_nl (turn_nl) directly. compute_mlp_validity_gate
# + _kill1_decision then run UNCHANGED on the result.
PHASE0_STORE_PREFIX = "issue811_turn_nl_mapchange/phase0_base_leg"
# Base-leg answer key per summary. "maxp" (#811 maxp-winner round) exists only in
# that round's phase0 store (v0_maxp); the loader below checks ONLY the REQUESTED
# summaries' keys, so a v1-era store (mean+turn_nl) keeps loading for the default
# test summary. The nine pre-user boundary arms (pre-user-boundary-summary round,
# plan §4.2) exist only in THAT round's phase0 store.
PRE_USER_TEST_SUMMARIES = (
    "pre_user_imstart",
    "pre_user_user",
    "pre_user_nl",
    "pre_user_mean3",
    "pre_user_max3",
    "ans_mean_incl_hdr",
    "ans_max_incl_hdr",
    "ans_mean_incl_hdr_alllayer",
    "ans_max_incl_hdr_alllayer",
)
PHASE0_SUMMARY_KEYS = {
    "mean": "v0",
    "turn_nl": "v0_turn_nl",
    "maxp": "v0_maxp",
    **{s: f"v0_{s}" for s in PRE_USER_TEST_SUMMARIES},
}


def _load_phase0_base_cells(
    behaviors: tuple[str, ...],
    summaries: tuple[str, ...],
    layer: int,
    *,
    local_root: str | None,
    max_sources: int | None,
    max_targets_per_source: int | None,
    strict: bool,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load the base-leg-only Phase-0 store into per-(behavior, summary) stacks.

    Reads ``phase0_base_leg/{behavior}/{source}_seed42/{target}_L{layer}.npz``
    (keys ``c_C`` / ``v0`` / ``v0_turn_nl``) via the SAME ``_Streamer`` +
    ``list_store_layout`` machinery the paired loader uses (HF or local mirror).
    Returns ``{(behavior, summary): {"C0", "V0", "cell_keys"}}`` where ``C0`` is
    the base context grid (n, HIDDEN) and ``V0`` the requested summary's base
    answer grid (n, HIDDEN) — exactly the two arrays ``_mlp_gate_groups`` +
    ``cell_meta`` consume. ``layer`` is the single PRIMARY layer the KILL-1
    decision is made at (plan §7). Fails loud on a missing key (a mean-only store
    or a wrong prefix) or, under ``strict``, an under-count.
    """
    streamer = (
        loadact._Streamer(local_root=local_root)
        if local_root
        else loadact._Streamer(prefix=PHASE0_STORE_PREFIX)
    )
    prefix = None if local_root else PHASE0_STORE_PREFIX
    layout = (
        loadact.list_store_layout_local(local_root, behaviors)
        if local_root
        else loadact.list_store_layout(behaviors, prefix=prefix)
    )
    # Per (behavior, summary) accumulate the base context (c_C -> C0) + the base
    # answer summary (v0 / v0_turn_nl -> V0), one row per (source, target) cell.
    acc: dict[tuple[str, str], dict[str, list]] = {
        (b, s): {"C0": [], "V0": [], "cell_keys": []} for b in behaviors for s in summaries
    }
    try:
        for beh in behaviors:
            if beh not in layout:
                raise KeyError(f"behavior {beh!r} not in phase0 store layout {sorted(layout)}")
            src_items = sorted(layout[beh].items())
            if max_sources is not None:
                src_items = src_items[:max_sources]
            n_cells = 0
            for src_dir, files in src_items:
                by_target = loadact._parse_cell_files(src_dir, files, (layer,))
                n_src = 0
                for _target_stem, layer_files in sorted(by_target.items()):
                    if max_targets_per_source is not None and n_src >= max_targets_per_source:
                        break
                    if layer not in layer_files:
                        continue
                    rel = f"{beh}/{layer_files[layer]}"
                    blob = streamer.load(rel)
                    # Check ONLY the REQUESTED summaries' keys — demanding every
                    # registered key would break v1-era stores (no v0_maxp) on a
                    # default-arm --skip-extract resume.
                    for k in ("c_C", *(PHASE0_SUMMARY_KEYS[s] for s in summaries)):
                        if k not in blob:
                            raise KeyError(
                                f"{rel} missing base-leg key {k!r}; keys={sorted(blob)} "
                                f"(a store without this summary, or wrong prefix?)"
                            )
                    c0 = np.asarray(blob["c_C"], dtype=np.float64)
                    assert c0.shape == (loadact.HIDDEN,), f"{rel} c_C shape {c0.shape}"
                    src_cid = str(np.asarray(blob["source_cid"]).item())
                    tgt_cid = str(np.asarray(blob["target_cid"]).item())
                    for summary in summaries:
                        v0 = np.asarray(blob[PHASE0_SUMMARY_KEYS[summary]], dtype=np.float64)
                        assert v0.shape == (loadact.HIDDEN,), f"{rel} {summary} v0 shape {v0.shape}"
                        acc[(beh, summary)]["C0"].append(c0)
                        acc[(beh, summary)]["V0"].append(v0)
                        acc[(beh, summary)]["cell_keys"].append(f"{beh}/{src_cid}__{tgt_cid}")
                    n_cells += 1
                    n_src += 1
            if strict:
                got = len(acc[(beh, summaries[0])]["C0"])
                assert got == loadact.EXPECTED_CELLS_PER_BEHAVIOR_LAYER, (
                    f"{beh} L{layer} phase0: loaded {got} cells, expected "
                    f"{loadact.EXPECTED_CELLS_PER_BEHAVIOR_LAYER} (16 sources x 30 targets)"
                )
    finally:
        streamer.cleanup()
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, d in acc.items():
        out[key] = {
            "C0": np.stack(d["C0"]) if d["C0"] else np.zeros((0, loadact.HIDDEN)),
            "V0": np.stack(d["V0"]) if d["V0"] else np.zeros((0, loadact.HIDDEN)),
            "cell_keys": d["cell_keys"],
        }
    return out


def run_phase0_gate(args) -> int:
    """KILL-1 pre-spend gate: base-leg validity gate + decision, BEFORE Phase 1.

    Loads the base-leg-only Phase-0 store at the PRIMARY layer, builds the
    (base, shuffle) MLPGroups per (behavior, summary), runs the VECTORIZED
    MLP-vs-shuffle gate (``compute_mlp_validity_gate``), makes the KILL-1
    decision (``_kill1_decision``), and writes ``kill1_base_leg_validity.json``.
    Returns 0 if KILL-1 does NOT fire (proceed to Phase 1), 3 if it FIRES (the
    dispatcher stops before the ~7 GPU-h paired re-extraction; the orchestrator
    persists epm:failure failure_class: data).

    Two distinct zero-``n_comparable`` states are handled separately (round-4
    BLOCKER phase0-gate-degenerate-guard-over-broad):
    - State A (``state == "empty_store"``, NO behavior computed a gate) RAISES on a
      production run — the store is empty / not-yet-uploaded / wrong-prefix and the
      gate cannot decide, so PASSing would let the ~7 GPU-h Phase-1 spend proceed
      without ever evaluating turn_nl. Under ``--smoke`` it is tolerated.
    - State B (``state == "reported_not_killed"``, store populated but every mean
      base-map gate margin <= 0) is a LEGITIMATE #722-style outcome (mean itself has
      no gate to collapse from) — reported (WARNING) and returns 0 (proceed). A
      healthy run must NOT be mislabeled a crash.
    A ``--smoke`` run that slices to <4 comparable cells is tolerant of both.
    """
    device = _resolve_device()
    logger.info("[phase=phase0_gate] device=%s", device)
    fit658.DEVICE = device
    if args.mlp_epochs is not None:
        fit658.MLP_MAX_EPOCHS = args.mlp_epochs
    fitM.TARGET_DIM = args.target_dim
    mlp_epochs = fit658.MLP_MAX_EPOCHS
    layer = args.primary_layer

    behaviors = tuple(args.behaviors)
    # The KILL-1 gate compares each TEST summary's base-leg gate margin vs mean's.
    # SINGLE test summary (--test-summary; turn_nl = the v1 round, maxp = the maxp
    # round) keeps the legacy kill1_base_leg_validity.json path VERBATIM; a LIST
    # (--test-summaries; the pre-user round's nine arms) runs the PER-ARM gate
    # (plan §7 / MF3) and writes validity_gate_phase0.json instead. getattr
    # defaults keep direct callers with pre-#maxp Namespaces working (the pinned
    # test_issue811_turn_nl gate tests construct args by hand).
    test_summaries = tuple(
        getattr(args, "test_summaries", None) or [getattr(args, "test_summary", "turn_nl")]
    )
    for ts in test_summaries:
        assert ts in PHASE0_SUMMARY_KEYS and ts != "mean", ts
    test_summary = test_summaries[0]
    summaries = ("mean", *test_summaries)
    rb_main = fitM._load_rb_main()
    # required=True: a Hub error at gate time RAISES (never a silent fact-drop —
    # r10 CONCERN rb-fact-silent-drop-headline); None here means ONLY the
    # data-declared degenerate flag (plan §8), which legitimately drops fact.
    rb_fact = fitM._load_rb_fact(required=True) if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("r_b_fact.pt flagged degenerate (plan §8) — dropping fact from KILL-1")
        behaviors = tuple(b for b in behaviors if b != "fact")

    strict = not args.smoke and args.max_sources is None and args.max_targets_per_source is None
    base_cells = _load_phase0_base_cells(
        behaviors,
        summaries,
        layer,
        local_root=args.local_root,
        max_sources=args.max_sources,
        max_targets_per_source=args.max_targets_per_source,
        strict=strict,
    )

    # Build the (base, shuffle) gate groups + meta per (behavior, layer, summary),
    # reusing the SAME helpers the paired-store gate uses (base leg only).
    groups_by_cell: dict[tuple[str, int, str], tuple] = {}
    cell_meta: dict[tuple[str, int, str], dict] = {}
    for behavior in behaviors:
        for summary in summaries:
            stacks = base_cells[(behavior, summary)]
            if stacks["C0"].shape[0] < 4:
                logger.warning(
                    "[phase=phase0_gate] %s %s: %d base cells (<4) — skip",
                    behavior,
                    summary,
                    stacks["C0"].shape[0],
                )
                continue
            cell_key = (behavior, layer, summary)
            pca_basis = fitM._pca_basis_v0(stacks["V0"], args.target_dim)
            E = fitM._load_E(behavior, stacks["cell_keys"])
            groups_by_cell[cell_key] = _mlp_gate_groups(stacks, pca_basis)
            cell_meta[cell_key] = {
                "pca_basis": pca_basis,
                "r_hat": fitM._r_hat_for(behavior, layer, rb_main, rb_fact),
                "E": E,
                "keep": ~np.isnan(E),
            }

    gate_by_cell = compute_mlp_validity_gate(
        groups_by_cell,
        cell_meta,
        device=device,
        max_epochs=mlp_epochs,
        num_threads=args.num_threads,
    )

    # Shape the gate results into the {summary: {cell_key: {gate_margin}}} form
    # _kill1_decision expects, then decide.
    cells_by_summary: dict[str, dict[tuple[str, int, str], dict]] = {s: {} for s in summaries}
    for cell_key, gate in gate_by_cell.items():
        _behavior, _layer, summary = cell_key
        cells_by_summary[summary][cell_key] = gate
    if len(test_summaries) > 1:
        # #811 pre-user round: PER-ARM KILL-1 (plan §7 / MF3) — a separate
        # decision surface from the single-arm legacy path below.
        return _run_phase0_gate_multi_arm(args, cells_by_summary, test_summaries, layer)
    kill1 = _kill1_decision(cells_by_summary, test_summary=test_summary)
    kill1["phase"] = "phase0_base_leg"
    kill1["primary_layer"] = layer

    out_json = args.out_dir.parent / "kill1_base_leg_validity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(kill1, indent=2, default=float))
    # Fail loud ONLY on the EMPTY-STORE state (State A), NOT on every n_comparable==0.
    # Both an empty store AND a populated store whose mean base-map gate is <= 0 on all
    # 3 behaviors yield n_comparable == 0, but they are different signals (round-4
    # BLOCKER phase0-gate-degenerate-guard-over-broad — State-A vs State-B conflation):
    #   State A (state == "empty_store"): NO behavior computed a gate margin at all —
    #     the base-leg gate had NO cells to decide on (a not-yet-uploaded / empty /
    #     wrong-prefix HF store, or a mis-pointed --local-root). On a PRODUCTION run
    #     this is NOT a real PASS: returning 0 would let the ~7 GPU-h Phase-1 paired
    #     re-extraction run without ever evaluating whether turn_nl collapses
    #     (round-3 BLOCKER phase0-gate-reads-unuploaded-hf-store). RAISE so the
    #     dispatcher HALTs before the spend.
    #   State B (state == "reported_not_killed"): the store IS populated but every
    #     behavior's mean base-map gate margin is <= 0 — a LEGITIMATE #722-style
    #     outcome (mean MLP-vs-shuffle negative below its shuffle null in 8/9 cells;
    #     #811 reuses the SAME fit code + paired-store lineage + n=16). mean has no
    #     gate to collapse from, so turn_nl cannot be judged worse — REPORT, do not
    #     kill. Return 0 (proceed) — a healthy run must not be mislabeled a crash.
    # A --smoke run intentionally slices to <4 comparable cells sometimes, so it is
    # TOLERANT of the empty-store state too (the smoke's job is to exercise routing).
    if kill1["state"] == "empty_store" and not args.smoke:
        raise RuntimeError(
            "phase0-gate: empty base-leg store — NO behavior computed a validity gate "
            "(all incomparable_missing_gate; an empty/missing store: a not-yet-uploaded "
            "HF prefix, a wrong --local-root, or a wrong PHASE0 prefix). Loaded from "
            f"{'local ' + args.local_root if args.local_root else 'HF ' + PHASE0_STORE_PREFIX}. "
            "The gate cannot decide — REFUSING to PASS silently before the ~7 GPU-h "
            f"Phase-1 spend (plan §7, failure_class: code). Decision JSON: {out_json}"
        )
    if kill1["state"] == "reported_not_killed":
        logger.warning(
            "[phase=phase0_gate] KILL-1 REPORTED (NOT killed): store is POPULATED but "
            "every behavior's mean base-map gate margin is <= 0 (%d/%d mean_no_gate) — "
            "a legitimate #722-style negative-mean-gate outcome, NOT an empty store. "
            "mean has no gate for %s to collapse relative to, so KILL-1 cannot "
            "decide against %s — proceeding to Phase 1 (plan §7, per-behavior: %s).",
            kill1["n_mean_no_gate"],
            len(HEADLINE_BEHAVIORS),
            test_summary,
            test_summary,
            json.dumps(kill1["per_behavior"], default=float),
        )
        return 0
    if kill1["fired"]:
        logger.warning(
            "[phase=phase0_gate] KILL1_TRIGGERED: %s base-leg validity gate collapsed "
            "vs mean at L%d on %d/%d comparable behaviors (>=2) — %s is a worse base-map "
            "summary on #537's 16 contexts; do NOT run the Phase-1 paired re-extraction "
            "(plan §7, H3, failure_class: data). Detail: %s",
            test_summary,
            layer,
            kill1["n_collapse"],
            kill1["n_comparable"],
            test_summary,
            json.dumps(kill1["per_behavior"], default=float),
        )
        return 3
    logger.info(
        "[phase=phase0_gate] KILL-1 PASS: %s base-leg validity gate holds "
        "(%d/%d comparable behaviors collapsed, <2) — proceed to Phase 1.",
        test_summary,
        kill1["n_collapse"],
        kill1["n_comparable"],
    )
    return 0


def _kill1_decision(
    cells_by_summary: dict[str, dict[tuple[str, int, str], dict]],
    test_summary: str = "turn_nl",
) -> dict:
    """KILL-1 base-leg validity decision at layer 14 (plan §7 / H3).

    The TEST summary (``turn_nl`` for the v1 round, ``maxp`` for the #811
    maxp-winner round) COLLAPSES relative to mean at a behavior when mean has a
    positive gate margin AND the test summary's margin < KILL1_COLLAPSE_FRAC ×
    mean's margin. KILL-1 fires when ≥2 of the 3 primary-layer behaviors collapse.
    Returns the per-behavior comparison + the fired flag; a behavior whose MEAN
    margin is non-positive is reported as ``mean_no_gate`` and does NOT count
    toward the kill (no baseline to collapse from).

    Two distinct zero-``n_comparable`` states are separated by the ``state`` field —
    load-bearing for the run_phase0_gate fail-loud guard (round-4 BLOCKER
    phase0-gate-degenerate-guard-over-broad, State-A vs State-B conflation):

    - ``state == "empty_store"`` (State A): EVERY headline behavior is
      ``incomparable_missing_gate`` (no gate was computed for ANY behavior). The
      base-leg gate had NO cells to decide on — an empty / not-yet-uploaded /
      wrong-prefix store, or a mis-pointed --local-root. On a production run this is
      NOT a valid PASS; run_phase0_gate RAISES on it.
    - ``state == "reported_not_killed"`` (State B): the store IS populated (≥1
      behavior computed a gate margin) but ``n_comparable == 0`` because every
      comparable behavior's MEAN base-map gate margin is ≤ 0 (``mean_no_gate``).
      This is a LEGITIMATE outcome — #722's mean MLP-vs-shuffle was negative at
      every layer, below its shuffle null in 8/9 cells; #811 reuses the SAME fit
      code + paired-store lineage + n=16, so it is plausible on the first
      production run. mean itself has no gate to collapse relative to, so turn_nl
      cannot be judged worse — reported, not killed. run_phase0_gate does NOT raise.
    - ``state == "decided"``: ``n_comparable >= 1`` — a normal decision (fired or
      not).
    """
    per_behavior: dict[str, dict] = {}
    n_collapse = 0
    n_comparable = 0
    n_missing_gate = 0  # behaviors with NO gate margin (empty-store signal, State A)
    n_mean_no_gate = 0  # behaviors with a mean margin <= 0 (store populated, State B)
    for beh in HEADLINE_BEHAVIORS:
        mkey = (beh, PRIMARY_LAYER, "mean")
        tkey = (beh, PRIMARY_LAYER, test_summary)
        m = cells_by_summary.get("mean", {}).get(mkey, {})
        t = cells_by_summary.get(test_summary, {}).get(tkey, {})
        m_margin = m.get("gate_margin")
        t_margin = t.get("gate_margin")
        # The test-summary margin rides a summary-NAMED key (turn_nl_margin /
        # maxp_margin) so the default arm's JSON schema is byte-unchanged.
        entry = {"mean_margin": m_margin, f"{test_summary}_margin": t_margin}
        if m_margin is None or t_margin is None:
            entry["status"] = "incomparable_missing_gate"
            n_missing_gate += 1
        elif m_margin <= 0:
            entry["status"] = "mean_no_gate"  # no baseline gate to collapse from
            n_mean_no_gate += 1
        else:
            n_comparable += 1
            threshold = KILL1_COLLAPSE_FRAC * m_margin
            collapsed = t_margin < threshold
            entry["collapse_threshold"] = threshold
            entry["collapsed"] = collapsed
            entry["status"] = "collapsed" if collapsed else "held"
            if collapsed:
                n_collapse += 1
        per_behavior[beh] = entry
    fired = n_collapse >= 2
    # State classification for the pre-spend fail-loud guard. n_comparable >= 1 is a
    # normal decision. n_comparable == 0 splits: if NO behavior computed a gate margin
    # at all (all incomparable_missing_gate), the store is empty/missing (State A,
    # raise); if the store WAS populated (>=1 behavior computed a gate — i.e. reached
    # mean_no_gate or the comparable branch), the zero is because every mean margin is
    # <= 0 (State B, a legitimate #722-style outcome — report, don't kill).
    if n_comparable >= 1:
        state = "decided"
    elif n_mean_no_gate >= 1:
        state = "reported_not_killed"  # State B: populated store, all mean margins <= 0
    else:
        state = "empty_store"  # State A: no gate computed for any behavior
    return {
        "fired": fired,
        "test_summary": test_summary,
        "n_collapse": n_collapse,
        "n_comparable": n_comparable,
        "n_missing_gate": n_missing_gate,
        "n_mean_no_gate": n_mean_no_gate,
        "state": state,
        "collapse_frac_threshold": KILL1_COLLAPSE_FRAC,
        "per_behavior": per_behavior,
    }


def _per_arm_gate_decision(
    cells_by_summary: dict[str, dict[tuple[str, int, str], dict]],
    arms: tuple[str, ...],
    primary_layer: int = PRIMARY_LAYER,
) -> dict:
    """Per-ARM KILL-1 base-leg validity decisions (pre-user round, plan §7 / MF3).

    Per arm ``a``: PASS iff base-leg MLP-vs-shuffle ``margin(a) >= 0.5 ×
    margin(mean)`` at the primary layer on **>=2 of 3** behaviors, counting ONLY
    behaviors whose mean margin is positive AND whose arm margin was computed
    (``KILL1_COLLAPSE_FRAC``, v2 comparability clause verbatim). Three per-arm
    ``gate_status`` states:

    - ``pass`` / ``fail`` — DECIDED (>=2 comparable behaviors). ``fail`` counts
      toward the all-nine-fail STOP; note that with exactly 2 comparable
      behaviors PASS requires BOTH to hold (the registered ">=2 of 3" rule).
    - ``comparator_indeterminate`` (MF3, registered) — FEWER than 2 behaviors
      have a positive mean margin (or a computed arm margin), so the >=2/3
      comparator CANNOT decide arm a. NOT ``gate_pass: false``; NEVER counts
      toward (or triggers) the all-nine-fail STOP — the per-arm mirror of the
      single-arm State-B ``reported_not_killed`` guard above. Excluded from the
      trusted set AND from the fail count; the arm's paired cells are still
      extracted + fit and shipped flagged.

    Continuous-margin narration (advisory, registered): each comparable behavior
    entry carries ``margin_ratio = margin(a)/margin(mean)`` and a
    ``near_threshold`` flag (|ratio − 0.5| <= 0.1) so near-threshold verdicts
    are narrated with the continuous value, never the bare binary. Arms 8/9
    carry the layer-pooled-vs-single-layer-comparator caveat in the WRITE-UP
    (plan §7); the rule itself is computed identically here.

    Returns the aggregate decision: ``stop_fired`` is True ONLY when ALL arms
    are DECIDED failures (``n_fail == len(arms)``); ``state == "empty_store"``
    (RAISE on production) when NO gate margin was computed for any
    (behavior, summary) at all.
    """
    per_arm: dict[str, dict] = {}
    n_pass = n_fail = n_indeterminate = 0
    any_gate_computed = False
    for arm in arms:
        per_behavior: dict[str, dict] = {}
        n_comparable = n_held = n_collapse = 0
        for beh in HEADLINE_BEHAVIORS:
            m = cells_by_summary.get("mean", {}).get((beh, primary_layer, "mean"), {})
            t = cells_by_summary.get(arm, {}).get((beh, primary_layer, arm), {})
            m_margin = m.get("gate_margin")
            t_margin = t.get("gate_margin")
            if m_margin is not None or t_margin is not None:
                any_gate_computed = True
            entry: dict = {"mean_margin": m_margin, "arm_margin": t_margin}
            if m_margin is None or t_margin is None:
                entry["status"] = "incomparable_missing_gate"
            elif m_margin <= 0:
                entry["status"] = "mean_no_gate"  # no baseline gate to collapse from
            else:
                n_comparable += 1
                threshold = KILL1_COLLAPSE_FRAC * m_margin
                ratio = float(t_margin / m_margin)
                held = t_margin >= threshold
                entry["collapse_threshold"] = threshold
                entry["margin_ratio"] = ratio
                entry["near_threshold"] = bool(abs(ratio - KILL1_COLLAPSE_FRAC) <= 0.1)
                entry["status"] = "held" if held else "collapsed"
                n_held += int(held)
                n_collapse += int(not held)
            per_behavior[beh] = entry
        if n_comparable < 2:
            gate_status = "comparator_indeterminate"
            gate_pass = None
            n_indeterminate += 1
        else:
            gate_pass = n_held >= 2
            gate_status = "pass" if gate_pass else "fail"
            n_pass += int(gate_pass)
            n_fail += int(not gate_pass)
        per_arm[arm] = {
            "gate_status": gate_status,
            "gate_pass": gate_pass,
            "n_comparable": n_comparable,
            "n_held": n_held,
            "n_collapse": n_collapse,
            "per_behavior": per_behavior,
        }
    return {
        "phase": "phase0_base_leg",
        "rule": "per-arm PASS iff margin(arm) >= KILL1_COLLAPSE_FRAC*margin(mean) on >=2/3 "
        "behaviors with positive mean margin; <2 comparable => comparator_indeterminate "
        "(never a fail); STOP only when ALL arms are DECIDED failures (plan §7/MF3)",
        "collapse_frac_threshold": KILL1_COLLAPSE_FRAC,
        "arms": list(arms),
        "trusted_arms": [a for a, r in per_arm.items() if r["gate_pass"] is True],
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_indeterminate": n_indeterminate,
        "stop_fired": n_fail == len(arms),
        "state": "decided" if any_gate_computed else "empty_store",
        "per_arm": per_arm,
    }


def _run_phase0_gate_multi_arm(
    args,
    cells_by_summary: dict[str, dict[tuple[str, int, str], dict]],
    arms: tuple[str, ...],
    layer: int,
) -> int:
    """Decide + persist the PER-ARM KILL-1 gate (pre-user round, plan §7).

    Writes ``validity_gate_phase0.json`` (per-arm ``gate_pass`` + the trusted-arm
    set — the §6.5 deliverable) next to the cells dir. Returns 0 (proceed to
    Phase 1) unless ALL arms are DECIDED failures (exit 3 — STOP before the
    paired spend; the §7 reconciliation against #810's committed n=50 reads is
    the WRITE-UP's duty and is named in the STOP log line). The empty-store
    State-A guard mirrors the single-arm path: RAISE on a production run,
    tolerate under --smoke.
    """
    decision = _per_arm_gate_decision(cells_by_summary, arms, primary_layer=layer)
    decision["primary_layer"] = layer
    out_json = args.out_dir.parent / "validity_gate_phase0.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(decision, indent=2, default=float))
    if decision["state"] == "empty_store" and not args.smoke:
        raise RuntimeError(
            "phase0-gate (per-arm): empty base-leg store — NO (behavior, summary) computed "
            "a validity gate margin (a not-yet-uploaded HF prefix, a wrong --local-root, or "
            "a wrong PHASE0 prefix). The gate cannot decide — REFUSING to PASS silently "
            f"before the paired Phase-1 spend (plan §7, failure_class: code). JSON: {out_json}"
        )
    if decision["stop_fired"]:
        logger.warning(
            "[phase=phase0_gate] KILL1_TRIGGERED (per-arm): ALL %d/%d boundary arms are "
            "DECIDED base-leg validity failures vs mean at L%d — no boundary-side summary "
            "supports a base context->answer map on this grid; do NOT run the Phase-1 "
            "paired re-extraction (plan §7, H3, failure_class: data). NOTE the write-up "
            "MUST reconcile this against #810's committed n=50 header-position reads "
            "(adhoc_next_turn_header_positions.json, skill-over-mean 0.57-0.72 at L18) "
            "before any positional-null claim (MF2). Detail: %s",
            decision["n_fail"],
            len(arms),
            layer,
            json.dumps(decision["per_arm"], default=float),
        )
        return 3
    logger.info(
        "[phase=phase0_gate] per-arm KILL-1: %d pass / %d fail / %d comparator_indeterminate "
        "of %d arms — trusted set %s; proceed to Phase 1 (failed/indeterminate arms are "
        "still extracted + fit, shipped flagged; plan §7).",
        decision["n_pass"],
        decision["n_fail"],
        decision["n_indeterminate"],
        len(arms),
        decision["trusted_arms"],
    )
    return 0


def main() -> int:
    # Round-parameterized store prefixes (#811 maxp round): declared global at the
    # TOP of main because the argparse defaults below READ the module values and
    # the CLI overrides REBIND them (Python requires the global decl before any
    # same-scope use).
    global STORE_PREFIX, PHASE0_STORE_PREFIX
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 fit M0 vs M⁺ under mean + turn_nl")
    ap.add_argument("--behaviors", nargs="+", default=list(HEADLINE_BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(SWEEP_LAYERS))
    ap.add_argument("--summaries", nargs="+", default=list(SUMMARIES))
    ap.add_argument("--max-sources", type=int, default=None, help="smoke: cap sources (>=2)")
    ap.add_argument("--max-targets-per-source", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_811/cells")
    ap.add_argument(
        "--local-root",
        default=None,
        help="read the #811 store from a LOCAL mirror dir (skip the HF tree walk) "
        "— e.g. eval_results/issue_811/analysis_tensors on the extraction node",
    )
    ap.add_argument("--smoke", action="store_true", help="1 behavior, layer 14, capped sources")
    ap.add_argument(
        "--phase0-gate",
        action="store_true",
        help="KILL-1 PRE-SPEND mode: read the base-leg-only phase0 store, run the "
        "MLP-vs-shuffle validity gate at --primary-layer, write kill1_base_leg_validity.json, "
        "and EXIT 3 if the turn_nl gate collapses (dispatcher stops before Phase 1). No ridge "
        "headline, no v_plus needed — the base leg is the whole KILL-1 signal (plan §4.0/§7).",
    )
    ap.add_argument("--mlp-epochs", type=int, default=None, help="override MLP_MAX_EPOCHS (smoke)")
    ap.add_argument(
        "--test-summary",
        default="turn_nl",
        choices=[s for s in PHASE0_SUMMARY_KEYS if s != "mean"],
        help="the NEW summary the KILL-1 gate compares vs mean (turn_nl = the v1 "
        "round; maxp = the #811 maxp-winner round). KILL1_COLLAPSE_FRAC unchanged.",
    )
    ap.add_argument(
        "--test-summaries",
        nargs="+",
        default=None,
        choices=[s for s in PHASE0_SUMMARY_KEYS if s != "mean"],
        help="PER-ARM KILL-1 (pre-user round, plan §7/MF3): the LIST of arms each "
        "gated vs mean at the primary layer (>=2/3 behaviors, same "
        "KILL1_COLLAPSE_FRAC; <2 comparable behaviors => comparator_indeterminate, "
        "never a fail). Writes validity_gate_phase0.json; exit 3 ONLY when ALL "
        "arms are DECIDED failures. Supersedes --test-summary when given.",
    )
    ap.add_argument(
        "--store-prefix",
        default=STORE_PREFIX,
        help="HF prefix of the round's PAIRED store (default: the v1 turn_nl store; "
        "the maxp round passes issue811_maxp_mapchange/analysis_tensors)",
    )
    ap.add_argument(
        "--phase0-prefix",
        default=PHASE0_STORE_PREFIX,
        help="HF prefix of the round's phase-0 base-leg store (maxp round: "
        "issue811_maxp_mapchange/phase0_base_leg)",
    )
    ap.add_argument(
        "--primary-layer",
        type=int,
        default=PRIMARY_LAYER,
        help="the single layer the KILL-1 base-leg decision is made at (--phase0-gate)",
    )
    ap.add_argument("--target-dim", type=int, default=fit658.A35_MLP_TARGET_DIM)
    ap.add_argument("--refit-pairs", type=int, default=fitM.N_REFIT_PAIRS)
    ap.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    # The loaders below read the module-level constants, so rebind them from the
    # CLI (defaults preserve the v1 turn_nl round verbatim).
    STORE_PREFIX = args.store_prefix
    PHASE0_STORE_PREFIX = args.phase0_prefix

    if args.smoke:
        args.behaviors = args.behaviors[:1]
        args.layers = [PRIMARY_LAYER]
        args.max_sources = args.max_sources or 6
        if args.max_targets_per_source is None:
            args.max_targets_per_source = 4
        if args.mlp_epochs is None:
            args.mlp_epochs = 20
        args.refit_pairs = min(args.refit_pairs, 8)
        args.target_dim = min(args.target_dim, 4)

    # KILL-1 PRE-SPEND gate (plan §4.0/§7): decide on the base-leg-only phase0
    # store BEFORE any Phase-1 paired re-extraction. Runs, writes the decision
    # JSON, and exits 3 on collapse — the dispatcher stops before the ~7 GPU-h
    # spend. No ridge headline / v_plus loading happens in this mode.
    if args.phase0_gate:
        return run_phase0_gate(args)

    device = _resolve_device()
    logger.info("[phase=fit_M] device=%s", device)
    fit658.DEVICE = device
    if args.mlp_epochs is not None:
        fit658.MLP_MAX_EPOCHS = args.mlp_epochs
    fitM.N_REFIT_PAIRS = args.refit_pairs
    fitM.TARGET_DIM = args.target_dim
    mlp_epochs = fit658.MLP_MAX_EPOCHS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(args.layers)
    behaviors = tuple(args.behaviors)
    summaries = tuple(args.summaries)
    logger.info("[phase=fit_M] behaviors=%s layers=%s summaries=%s", behaviors, layers, summaries)

    # Exactness gates (#658 ridge + #740 vectorized-MLP reproduce-check) — fail at
    # startup on a reduction-order regression.
    fit658._assert_ridge_exactness()
    logger.info("[phase=fit_M] ridge exactness gate PASS")

    rb_main = fitM._load_rb_main()
    # required=True: fail LOUD on a load failure (rb-fact-silent-drop-headline);
    # None means ONLY the data-declared degenerate flag (plan §8).
    rb_fact = fitM._load_rb_fact(required=True) if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("r_b_fact.pt flagged degenerate (plan §8) — dropping fact")
        behaviors = tuple(b for b in behaviors if b != "fact")

    strict = not args.smoke and args.max_sources is None and args.max_targets_per_source is None
    layout = _load_store_layout(behaviors, local_root=args.local_root)

    ridge_cells, groups_by_cell, cell_meta = _load_and_prepare_cells(
        behaviors,
        layers,
        summaries,
        args=args,
        layout=layout,
        strict=strict,
        rb_main=rb_main,
        rb_fact=rb_fact,
    )

    gate_by_cell = compute_mlp_validity_gate(
        groups_by_cell,
        cell_meta,
        device=device,
        max_epochs=mlp_epochs,
        num_threads=args.num_threads,
    )

    # #811 pre-user round: stamp each cell JSON with its arm's Phase-0 PER-ARM
    # KILL-1 verdict (gate_pass / gate_status; failed + indeterminate arms are
    # shipped FLAGGED, never dropped — plan §4.3 item 4, §7) so figures can grey/
    # hatch flagged rows. References (mean/turn_nl/maxp) stamp "reference". A
    # legacy round (no validity_gate_phase0.json) adds no keys — schema unchanged.
    arm_gate: dict[str, dict] = {}
    arm_gate_path = args.out_dir.parent / "validity_gate_phase0.json"
    if arm_gate_path.exists():
        arm_gate = json.loads(arm_gate_path.read_text()).get("per_arm", {})

    # ── Merge the gate into each ridge cell + write the checkpoints ──────────────
    cells_by_summary: dict[str, dict[tuple[str, int, str], dict]] = {s: {} for s in summaries}
    for cell_key, ridge in ridge_cells.items():
        behavior, layer, summary = cell_key
        out = args.out_dir / f"{behavior}_L{layer}_{summary}.json"
        if "mlp_validity_gate" in ridge and "summary" in ridge:
            # A cached cell already carries the merged gate — leave it as-is.
            cells_by_summary[summary][cell_key] = ridge.get("mlp_validity_gate", {})
            continue
        gate = gate_by_cell.get(cell_key, {})
        ridge["summary"] = summary
        ridge["mlp_validity_gate"] = gate
        if arm_gate:
            rec = arm_gate.get(summary)
            ridge["gate_status"] = rec["gate_status"] if rec else "reference"
            ridge["gate_pass"] = rec["gate_pass"] if rec else None
        ridge["metadata"] = {
            "issue": 811,
            "summary": summary,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_write_json(out, ridge)
        cells_by_summary[summary][cell_key] = gate
        logger.info(
            "[phase=fit_M]   %s Δ_med=%.4g floor=%.4g gate_margin=%s",
            out.name,
            ridge["Delta_med"],
            ridge["floor_combined"],
            gate.get("gate_margin"),
        )

    # ── Per-draw floor dump (#811 pre-user round, plan §6/§6.5) ───────────────────
    _dump_bootstrap_draws(args.out_dir.parent, ridge_cells, behaviors, layers, summaries)

    # ── KILL-1 is a PRE-SPEND gate owned by Phase 0 (--phase0-gate), NOT here ──────
    # The KILL-1 decision + kill1_base_leg_validity.json / validity_gate_phase0.json
    # are written by run_phase0_gate BEFORE the ~7 GPU-h Phase-1 paired
    # re-extraction (plan §4.0/§7, round-2 BLOCKER kill1-not-pre-spend-gate). This
    # Phase-2 fit runs ONLY after Phase 0 PASSed, so it does NOT re-decide or
    # overwrite those JSONs. For diagnostic continuity it logs the Phase-2
    # re-derived base-leg margins (informational — the pre-spend decision has
    # already been made and honored); the paired-store gate margins here are on
    # the SAME base leg Phase 0 read.
    recheck = [ts for ts in (args.test_summaries or [args.test_summary]) if ts in summaries]
    if "mean" in summaries and PRIMARY_LAYER in layers:
        for ts in recheck:
            kill1_recheck = _kill1_decision(cells_by_summary, test_summary=ts)
            logger.info(
                "[phase=fit_M] KILL-1 re-check (informational; the pre-spend decision "
                "was made by --phase0-gate): test_summary=%s fired=%s n_collapse=%d/%d",
                ts,
                kill1_recheck["fired"],
                kill1_recheck["n_collapse"],
                kill1_recheck["n_comparable"],
            )

    logger.info("[phase=fit_M] done")
    return 0


def _dump_bootstrap_draws(
    round_dir: Path,
    ridge_cells: dict[tuple[str, int, str], dict],
    behaviors: tuple[str, ...],
    layers: tuple[int, ...],
    summaries: tuple[str, ...],
) -> None:
    """Write ``bootstrap_draws_{behavior}_L{li}.npz`` — (B, n_summaries) per floor.

    The §6.5 deliverable that makes the pre-registered selection-null escape a
    PURE READ: per (behavior, layer) cell, the DRAW-ALIGNED per-pair floor
    statistics of every fitted summary stacked column-wise (NaN at skipped
    pairs). Alignment holds because every summary's ``make_refit_pair`` runs at
    the SAME seed over the SAME (n, families) grid, so draw index b maps to the
    same family resample across summaries (plan §6). A pre-user-round cell
    missing ``floor_draws`` (a stale pre-round cache in a fresh round dir) FAILS
    LOUD; legacy 2–3-summary rounds warn + skip (their caches predate the key).
    """
    floors = ("floor_M0_refit", "floor_Mplus_refit", "floor_shifted")
    for behavior in behaviors:
        for layer in layers:
            present = [s for s in summaries if (behavior, layer, s) in ridge_cells]
            if not present:
                continue
            missing = [s for s in present if "floor_draws" not in ridge_cells[(behavior, layer, s)]]
            if missing:
                if len(summaries) >= 4:  # the pre-user round: the dump is a deliverable
                    raise RuntimeError(
                        f"bootstrap-draws dump: cells {behavior}/L{layer} {missing} carry no "
                        "'floor_draws' (stale pre-round cache?) — re-fit with --force-rerun; "
                        "the per-draw dump is a §6.5 deliverable of the pre-user round"
                    )
                logger.warning(
                    "[phase=fit_M] bootstrap-draws dump skipped for %s/L%d (legacy cells "
                    "without floor_draws: %s)",
                    behavior,
                    layer,
                    missing,
                )
                continue
            arrays = {}
            for fl in floors:
                cols = [
                    np.asarray(
                        [
                            np.nan if v is None else float(v)
                            for v in ridge_cells[(behavior, layer, s)]["floor_draws"][fl]
                        ],
                        dtype=float,
                    )
                    for s in present
                ]
                arrays[fl] = np.stack(cols, axis=1)  # (B, n_summaries)
            out = round_dir / f"bootstrap_draws_{behavior}_L{layer}.npz"
            np.savez(out, summaries=np.asarray(present), **arrays)
            logger.info(
                "[phase=fit_M] bootstrap draws dumped -> %s (%s x %d summaries per floor)",
                out.name,
                arrays[floors[0]].shape[0],
                len(present),
            )


if __name__ == "__main__":
    raise SystemExit(main())
