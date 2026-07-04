#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ×, Ŵ, ‖·‖, ※) in scientific docstrings + log messages.
"""Issue #813 — fit + save M0/M⁺ ridge maps per (behavior, substrate, layer).

Reads the per-(behavior, substrate) reduced summaries the extraction wave wrote
(``eval_results/issue_813/reduced/<behavior>/<substrate>/summary.npz``: 50 c_C
rows + 50 v_A rows, base + trained, 28 layers) and fits the M0/M⁺ ridge maps
using the #667/#722 fit machinery VERBATIM — feeding the fresh 50-input battery
stacks instead of the #667 16-source store (the store is NOT reusable — wrong
input layout §4.1; only the fit machinery is).

Reuse (imported, not re-implemented):
- ``issue667_save_maps.fit_and_save_cell`` / ``correctness_gate`` / ``_ridge_components``
  — the closed-form ridge fit + the ``<1e-8`` exact-reproduction gate. Consumes a
  ``cells`` list of ``issue722_load_activations.CellRecord`` via ``stack_for_fit``;
  #813 builds those CellRecords from the reduced summary (one per battery context).
- ``issue722_fit_M._pca_basis_v0`` / ``_to64`` / ``_ridge_fit_predict`` — the single
  V0-derived top-64 basis shared by BOTH maps (``TARGET_DIM=64``, NEVER 48), pulled
  in transitively by ``fit_and_save_cell``.

The marker (which has no ``r_B`` — ``issue722_fit_M._r_hat_for`` KeyErrors on it)
still gets its M0/M⁺ maps saved here IDENTICALLY (the maps are behavior-agnostic —
they are just the ridge fits); the marker-specific Δ/floor reads (read-1 unprojected
‖ΔM‖ + read-2 W_U[※]-projected) are computed in ``issue813_analysis.py`` via
``issue667_marker_mapchange.fit_marker_layer``, NOT here.

Output: ``eval_results/issue_813_maps/<behavior>/<substrate>/L<layer>.npz`` (the
factored form issue667_save_maps writes: W_M0, W_Mplus, pca_basis, input mean/std
C0+Cplus, λ, cell_keys, families). The ``<1e-8`` gate runs on a sample of cells.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue667_save_maps as savemaps  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.save_maps")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
HF_MAP_PREFIX = f"{EXPERIMENT_NAME}/maps"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
LAYERS = tuple(range(1, 28))  # 1..27 (L0 extraction-aliased, dropped from the fit)
HIDDEN = 3584
N_LAYERS = 28

# Every key `load_reduced_cells` reads from a reduced summary.npz (schema preflight).
REDUCED_SUMMARY_KEYS = (
    "c_C_base",
    "c_C_trained",
    "v_A_base",
    "v_A_trained",
    "context_ids",
    "families",
)


def _require_npz_keys(path: Path | str, npz, required_keys) -> None:
    """Fail loud BEFORE any keyed read when an NPZ is missing required keys.

    Canonical home (v7 concern ``perlayer-npz-key-coverage-preflight``, closed in the
    round-2 fix): this is the lowest common module, so BOTH consumers —
    ``issue813_per_example_maps`` (driver NPZ reads) and ``issue813_perlayer_profile``
    (direct drift read + the imported ``load_reduced_cells`` path below) — import it
    from here without a cycle. ``npz`` is an ``np.lib.npyio.NpzFile`` (uses ``.files``)
    or any mapping. The error names the missing keys AND the file path so a schema
    drift between producer and consumer surfaces as one actionable line instead of a
    bare KeyError deep in compute.
    """
    present = list(getattr(npz, "files", None) or npz.keys())
    missing = [k for k in required_keys if k not in present]
    if missing:
        raise KeyError(
            f"NPZ schema preflight FAILED for {path}: missing keys {missing} "
            f"(present: {sorted(present)})"
        )


def load_reduced_cells(
    behavior: str, substrate: str, layer: int, reduced_root: Path
) -> list[loadact.CellRecord]:
    """Build the per-(behavior, substrate, layer) CellRecord list from the reduced summary.

    The reduced ``summary.npz`` holds (n_ctx, 28, HIDDEN) c_C/v_A base+trained stacks.
    For a given layer slice the layer-``layer`` plane and build one CellRecord per
    battery context — the 50-input battery map fit input. ``source_cid``/``target_cid``
    = the battery context id (single input per context — no source×target grid here),
    ``family`` = battery family (the clustered-bootstrap unit), ``c0``/``cplus`` = the
    question-averaged c_C (base/trained), ``v0``/``vplus`` = the question-averaged v_A.
    """
    path = reduced_root / behavior / substrate / "summary.npz"
    if not path.exists():
        raise FileNotFoundError(f"reduced summary missing: {path} (extract phase incomplete?)")
    d = np.load(path, allow_pickle=True)
    _require_npz_keys(path, d, REDUCED_SUMMARY_KEYS)  # fail loud before any keyed read
    c_C_base = np.asarray(d["c_C_base"], dtype=np.float64)  # (n, 28, HIDDEN)
    c_C_trained = np.asarray(d["c_C_trained"], dtype=np.float64)
    v_A_base = np.asarray(d["v_A_base"], dtype=np.float64)
    v_A_trained = np.asarray(d["v_A_trained"], dtype=np.float64)
    ctx_ids = [str(x) for x in d["context_ids"]]
    families = [str(x) for x in d["families"]]
    n = c_C_base.shape[0]
    assert c_C_base.shape == (n, N_LAYERS, HIDDEN), c_C_base.shape
    assert 0 <= layer < N_LAYERS, layer
    return [
        loadact.CellRecord(
            behavior=behavior,
            source_cid=ctx_ids[i],
            target_cid=ctx_ids[i],
            layer=layer,
            c0=c_C_base[i, layer],
            cplus=c_C_trained[i, layer],
            v0=v_A_base[i, layer],
            vplus=v_A_trained[i, layer],
            family=families[i],
        )
        for i in range(n)
    ]


def fit_and_save_one(
    behavior: str, substrate: str, layer: int, cells: list[loadact.CellRecord], out_dir: Path
) -> tuple[Path, dict]:
    """Fit + save one map → ``out_dir/<behavior>/<substrate>/L<layer>.npz``.

    ``savemaps.fit_and_save_cell(behavior, layer, cells, root)`` writes
    ``root/<behavior>/L<layer>.npz``. To land the substrate BETWEEN behavior and the
    layer file (the plan's ``<behavior>/<substrate>/L*.npz`` glob) we pass a
    substrate-scoped scratch root ``out_dir/_scratch/<substrate>`` (so the helper
    writes ``.../_scratch/<substrate>/<behavior>/L<layer>.npz``), then RELOCATE the
    file to the canonical ``out_dir/<behavior>/<substrate>/L<layer>.npz``. The fit
    itself (and its ``<1e-8`` gate meta) is byte-for-byte the reused helper's output.
    """
    scratch_root = out_dir / "_scratch" / substrate
    written, meta = savemaps.fit_and_save_cell(behavior, layer, cells, scratch_root)
    canonical_dir = out_dir / behavior / substrate
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical_dir / f"L{layer}.npz"
    shutil.move(str(written), str(canonical))
    return canonical, meta


def fit_behavior_substrate(
    behavior: str,
    substrate: str,
    layers: tuple[int, ...],
    reduced_root: Path,
    out_dir: Path,
    gate_budget: dict,
) -> int:
    """Fit + save all layers for one (behavior, substrate). Updates ``gate_budget`` in place."""
    n_saved = 0
    for layer in layers:
        cells = load_reduced_cells(behavior, substrate, layer, reduced_root)
        if len(cells) < 4:
            logger.warning(
                "[phase=fit_maps] %s/%s L%d: %d cells (<4) — skip",
                behavior,
                substrate,
                layer,
                len(cells),
            )
            continue
        out_path, meta = fit_and_save_one(behavior, substrate, layer, cells, out_dir)
        n_saved += 1
        if gate_budget["n_gated"] < gate_budget["gate_cells"]:
            rep = savemaps.correctness_gate(meta, out_path)
            cell_max = max(rep[k] for k in rep if k.startswith("reldiff"))
            f64_max = max(rep["reldiff_M0_f64"], rep["reldiff_Mplus_f64"])
            gate_budget["max_reldiff_f64"] = max(gate_budget["max_reldiff_f64"], f64_max)
            gate_budget["n_gated"] += 1
            logger.info(
                "[phase=fit_maps] GATE %s/%s L%d: f64 M0=%.2e M⁺=%.2e (all=%.2e)",
                behavior,
                substrate,
                layer,
                rep["reldiff_M0_f64"],
                rep["reldiff_Mplus_f64"],
                cell_max,
            )
        logger.info("[phase=fit_maps] saved %s (n=%d)", out_path, meta["n"])
    return n_saved


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    logger.info("[phase=fit_maps] device=%s", fit658.DEVICE)

    ap = argparse.ArgumentParser(
        description="Issue #813 — fit + save M0/M⁺ maps per (behavior, substrate, layer)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_813_maps")
    ap.add_argument(
        "--gate-cells", type=int, default=4, help="cells to run the <1e-8 correctness gate on"
    )
    ap.add_argument("--upload", action="store_true", help="bulk upload out-dir to HF after fitting")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit658._assert_ridge_exactness()  # #658 reduction-order gate (fail-fast at startup)
    logger.info("[phase=fit_maps] ridge exactness gate PASS")

    layers = tuple(args.layers)
    gate_budget = {"n_gated": 0, "gate_cells": args.gate_cells, "max_reldiff_f64": 0.0}
    total_saved = 0
    t0 = time.time()
    for behavior in args.behaviors:
        for substrate in args.substrates:
            total_saved += fit_behavior_substrate(
                behavior, substrate, layers, args.reduced_root, args.out_dir, gate_budget
            )

    # Tidy the scratch relocation root.
    scratch = args.out_dir / "_scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    wall = time.time() - t0
    max_rd = gate_budget["max_reldiff_f64"]
    logger.info(
        "[phase=fit_maps] fit %d cell(s) in %.1fs; max f64 reldiff=%.3e", total_saved, wall, max_rd
    )
    assert max_rd < 1e-8, (
        f"correctness gate FAILED: max float64 reldiff {max_rd:.3e} >= 1e-8 — "
        "saved components do NOT reproduce _ridge_fit_predict exactly"
    )
    logger.info("[phase=fit_maps] CORRECTNESS GATE PASS (<1e-8)")

    if args.upload:
        from huggingface_hub import upload_folder

        upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(args.out_dir),
            path_in_repo=HF_MAP_PREFIX,
            commit_message=f"issue813: fitted M0/M⁺ maps ({total_saved} cells)",
        )
        logger.info("[phase=fit_maps] uploaded → %s/%s", DATA_REPO, HF_MAP_PREFIX)

    logger.info("[phase=fit_maps] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
