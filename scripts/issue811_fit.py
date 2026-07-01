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
import sys
import time
from pathlib import Path

import numpy as np

# DOTENV_LINT_EXEMPT: analysis-phase script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

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


def _cached_cell_valid(path: Path) -> bool:
    """True iff ``path`` is a complete #811 per-cell checkpoint (parses + schema keys)."""
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[phase=fit_M] cached %s unreadable (%s) — will re-fit", path.name, e)
        return False
    required = fitM._CELL_SCHEMA_KEYS | {"summary", "mlp_validity_gate"}
    missing = required - obj.keys()
    if missing:
        logger.warning(
            "[phase=fit_M] cached %s missing schema keys %s — will re-fit",
            path.name,
            sorted(missing),
        )
        return False
    return True


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
    (pca_basis / r_hat / E / keep) the gate consumes. A cached, schema-valid cell is
    reloaded (resume-skip) rather than re-fit. Extracted from :func:`main` (C901).
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
    for summary in summaries:
        for behavior in behaviors:
            for layer in layers:
                cells = cells_by[summary][(behavior, layer)]
                cell_key = (behavior, layer, summary)
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
                if not args.force_rerun and out.exists() and _cached_cell_valid(out):
                    logger.info("[phase=fit_M] %s (cached — skip)", out.name)
                    ridge_cells[cell_key] = json.loads(out.read_text())
                    continue
                # Ridge headline (byte-identical to #722; include_mlp=False — the
                # serial MLP is REPLACED by the vectorized validity gate).
                ridge_cells[cell_key] = fitM.fit_cell(
                    behavior, layer, cells, rb_main, rb_fact, include_mlp=False
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


def _kill1_decision(cells_by_summary: dict[str, dict[tuple[str, int, str], dict]]) -> dict:
    """KILL-1 base-leg validity decision at layer 14 (plan §7 / H3).

    turn_nl COLLAPSES relative to mean at a behavior when mean has a positive gate
    margin AND turn_nl's margin < KILL1_COLLAPSE_FRAC × mean's margin. KILL-1 fires
    when ≥2 of the 3 primary-layer behaviors collapse. Returns the per-behavior
    comparison + the fired flag; a behavior whose MEAN margin is non-positive is
    reported as ``mean_no_gate`` and does NOT count toward the kill (no baseline to
    collapse from).
    """
    per_behavior: dict[str, dict] = {}
    n_collapse = 0
    n_comparable = 0
    for beh in HEADLINE_BEHAVIORS:
        mkey = (beh, PRIMARY_LAYER, "mean")
        tkey = (beh, PRIMARY_LAYER, "turn_nl")
        m = cells_by_summary.get("mean", {}).get(mkey, {})
        t = cells_by_summary.get("turn_nl", {}).get(tkey, {})
        m_margin = m.get("gate_margin")
        t_margin = t.get("gate_margin")
        entry = {"mean_margin": m_margin, "turn_nl_margin": t_margin}
        if m_margin is None or t_margin is None:
            entry["status"] = "incomparable_missing_gate"
        elif m_margin <= 0:
            entry["status"] = "mean_no_gate"  # no baseline gate to collapse from
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
    return {
        "fired": fired,
        "n_collapse": n_collapse,
        "n_comparable": n_comparable,
        "collapse_frac_threshold": KILL1_COLLAPSE_FRAC,
        "per_behavior": per_behavior,
    }


def main() -> int:
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
    ap.add_argument("--mlp-epochs", type=int, default=None, help="override MLP_MAX_EPOCHS (smoke)")
    ap.add_argument("--target-dim", type=int, default=fit658.A35_MLP_TARGET_DIM)
    ap.add_argument("--refit-pairs", type=int, default=fitM.N_REFIT_PAIRS)
    ap.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

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
    rb_fact = fitM._load_rb_fact() if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("fact requested but r_b_fact.pt unavailable/degenerate — dropping fact")
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
        ridge["metadata"] = {
            "issue": 811,
            "summary": summary,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out.write_text(json.dumps(ridge, indent=2, default=float))
        cells_by_summary[summary][cell_key] = gate
        logger.info(
            "[phase=fit_M]   %s Δ_med=%.4g floor=%.4g gate_margin=%s",
            out.name,
            ridge["Delta_med"],
            ridge["floor_combined"],
            gate.get("gate_margin"),
        )

    # ── KILL-1 base-leg validity decision at layer 14 ─────────────────────────────
    if set(summaries) >= {"mean", "turn_nl"} and PRIMARY_LAYER in layers:
        kill1 = _kill1_decision(cells_by_summary)
        (args.out_dir.parent / "kill1_base_leg_validity.json").write_text(
            json.dumps(kill1, indent=2, default=float)
        )
        if kill1["fired"]:
            # Pod-side code never shells task.py — surface KILL-1 as a loud
            # structured log line the GCP/poller orchestrator persists as
            # epm:failure failure_class: data (plan §7). The dispatcher checks the
            # kill1 JSON and stops before the analyze/figures phases.
            logger.warning(
                "[phase=fit_M] KILL1_TRIGGERED: turn_nl base-leg validity gate "
                "collapsed vs mean at L%d on %d/%d comparable behaviors (>=2) — "
                "turn_nl is a worse base-map summary on #537's 16 contexts; its "
                "before/after comparison is untrustworthy (plan §7, H3). Detail: %s",
                PRIMARY_LAYER,
                kill1["n_collapse"],
                kill1["n_comparable"],
                json.dumps(kill1["per_behavior"], default=float),
            )
        else:
            logger.info(
                "[phase=fit_M] KILL-1 PASS: turn_nl base-leg validity gate holds "
                "(%d/%d comparable behaviors collapsed, <2)",
                kill1["n_collapse"],
                kill1["n_comparable"],
            )

    logger.info("[phase=fit_M] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
