#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ×, Ŵ, ‖·‖, ※, ρ) in scientific docstrings + logs.
"""Issue #813 free-analysis follow-up — per-layer Δ/floor profile (layers 1-27).

The #813 headline froze layer 14 (``issue813_analysis.HEADLINE_LAYER``). The plan's
exploratory section wanted the FULL per-layer Δ/floor profile across all 28 layers
per ``(behavior × substrate)`` cell, so the frozen-layer choice can be judged
representative-vs-cherry-picked, plus the companion ``c_C`` drift read (how much the
finetuning shifts the *input* context representation, layer by layer).

ANALYSIS-ONLY, 0 GPU-h, CPU minutes (closed-form ridge). Nothing new is trained or
generated. Every DV is the EXACT quantity ``issue813_analysis.observed_read`` computes
at the frozen layer — REUSED verbatim (``issue722_fit_M.fit_cell`` for em/fact/syco,
``issue667_marker_mapchange.fit_marker_layer`` for marker) — just swept over
layers 1..27 instead of fixed to 14. Layer 0 is extraction-aliased (dropped from the
fit, matching ``issue813_save_maps.LAYERS``).

Reuse (imported, never re-implemented):
- ``issue813_save_maps.load_reduced_cells(behavior, substrate, layer, root)`` — builds
  the per-context ``CellRecord`` list from the reduced ``summary.npz`` at ONE layer.
- ``issue722_fit_M.fit_cell(behavior, layer, cells, rb_main, rb_fact, include_mlp=False)``
  — the ridge-only headline (``Delta_med`` / ``floor_combined`` / ``Delta_over_floor_sd``
  / chain-ρ). Identical call ``observed_read`` makes, layer-parametrized.
- ``issue667_marker_mapchange.fit_marker_layer(layer, cells, wu_marker, with_chain=...)``
  — the marker two-read path (read-1 unprojected ‖ΔM‖/floor PRIMARY + read-2 W_U[※]).
- The r_B / r_b_fact / Ŵ_U[※] loaders (``fitM._load_rb_main`` / ``_load_rb_fact`` /
  ``marker_mc.load_wu_marker_direction``) — the SAME direction artifacts the headline used.

The per-cell DV mapping mirrors ``observed_read`` EXACTLY:
- em/fact/syco: ``delta_over_floor = Delta_over_floor_sd`` (SD-combined-floor denominator),
  with ``delta_med = Delta_med`` and ``floor = delta_med / delta_over_floor`` (the actual
  DV denominator = ``floor_sd_combined``); ``floor_combined`` is the p95-combined floor
  (diagnostic, matching the committed JSON field).
- marker: ``delta_over_floor = unproj_delta_over_floor`` (p95-combined-floor denominator),
  ``delta_med = unproj_delta_med``, ``floor_combined = unproj_floor_p95.combined``.

Outputs (per-cell checkpoint; resume-skip complete cells):
``eval_results/issue_813/perlayer/<behavior>__<substrate>.json`` — one row per layer
1..27 with ``delta_over_floor`` / ``delta_med`` / ``floor`` / ``floor_combined_p95`` /
``chain_rho_diff`` / ``c_C_drift_med`` (+ the marker read-2 columns for marker cells).

Equivalence gate: at layer 14 (HEADLINE_LAYER) the recomputed DV MUST match the
committed ``eval_results/issue_813/delta_floor/<cell>.json`` within a relative
tolerance (the refit floor uses a seeded RNG so agreement is ~machine precision;
the gate FAILs loud on drift, so this driver cannot silently diverge from the
frozen headline it is contextualizing).
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

import issue658_fit_predictors as fit658  # noqa: E402
import issue667_marker_mapchange as marker_mc  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue813_save_maps as savemaps813  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.perlayer")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
HF_REDUCED_PREFIX = f"{EXPERIMENT_NAME}/reduced"
# The revision the extraction wave wrote (frozen in the follow-up brief). Pin it so a
# later force-push to main cannot silently swap the reduced stores under this analysis.
HF_REVISION = "b0d30307c1671cad575928e5abf5253c0c849dee"

BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
LAYERS = tuple(range(1, 28))  # 1..27 (L0 extraction-aliased, dropped — matches save_maps.LAYERS)
HEADLINE_LAYER = 14  # frozen headline (#651/#658/#813); the equivalence anchor
N_LAYERS = 28
HIDDEN = 3584

# Equivalence-gate tolerance at L14 vs the committed delta_floor JSON. The recompute
# refits the maps via the same ``fit_cell`` path, whose ``_pca_basis_v0`` SVD is
# run-to-run / backend non-deterministic on near-degenerate singular values, and the
# truncated top-64 PCA basis (fitM.TARGET_DIM=64, never 48) feeds BOTH the
# Delta_med numerator and the refit floor.
# Measured on the full 12/12 recompute (VM CPU vs the committed pod run): delta_med and
# delta_over_floor shift TOGETHER while the floor is stable to ~6e-5 — em/generic
# 5.11882 vs 5.11858 (~4.7e-5 rel), em/mix ~0.71% rel, fact/generic ~1.25% rel (the
# observed max; both delta_med and the ratio shift by the same factor). A 2% relative
# tolerance absorbs that basis/backend jitter while a real code divergence (wrong r_hat
# or wrong layer slice, ≫10%) still FAILs the gate loud. The verdict thresholds this
# profile annotates differ by ≥30%, so ≤1.3% recompute drift is decision-irrelevant.
EQ_RTOL = 2e-2
EQ_ATOL = 1e-6


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:  # detached / no git — provenance is best-effort
        return "unknown"


def ensure_reduced_summaries(behaviors, substrates, reduced_root: Path) -> None:
    """Fetch each ``<behavior>/<substrate>/summary.npz`` from HF (pinned rev) if absent.

    Per-file ``hf_hub_download`` into the canonical local ``reduced_root`` layout the
    reused ``load_reduced_cells`` reads. 12 files × ~69 MB ≈ 0.8 GB total — well under
    the follow-up's few-GB ceiling. Files already present locally (an in-flight run or
    the extraction wave's own writes) are left untouched (hf_hub_download no-ops on a
    cached blob at the same revision).
    """
    from huggingface_hub import hf_hub_download

    for behavior in behaviors:
        for substrate in substrates:
            dest = reduced_root / behavior / substrate / "summary.npz"
            if dest.exists():
                logger.info("[phase=perlayer] reduced summary present %s/%s", behavior, substrate)
                continue
            rel = f"{HF_REDUCED_PREFIX}/{behavior}/{substrate}/summary.npz"
            logger.info("[phase=perlayer] fetch %s (rev %s)", rel, HF_REVISION[:10])
            local = hf_hub_download(DATA_REPO, rel, repo_type="dataset", revision=HF_REVISION)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Symlink the cache blob into the canonical layout (no copy — the cache is on
            # the data disk; load_reduced_cells just needs the path to resolve).
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(Path(local).resolve())


def c_C_drift_per_layer(
    behavior: str, substrate: str, reduced_root: Path, layers: tuple[int, ...]
) -> list[float]:
    """Median (over the 50 contexts) of ‖c_C_trained − c_C_base‖ at each requested layer.

    The companion 'input-drift' read: how much finetuning moves the CONTEXT
    representation itself (the ridge map's *input*), layer by layer — read straight
    from the reduced ``summary.npz`` (no fit). One float per layer in ``layers`` order.
    """
    path = reduced_root / behavior / substrate / "summary.npz"
    d = np.load(path, allow_pickle=True)
    c0 = np.asarray(d["c_C_base"], dtype=np.float64)  # (n, 28, HIDDEN)
    cp = np.asarray(d["c_C_trained"], dtype=np.float64)
    assert c0.shape[1:] == (N_LAYERS, HIDDEN), c0.shape
    drift = np.linalg.norm(cp - c0, axis=2)  # (n, 28): ‖Δc_C‖ per context per layer
    return [float(np.median(drift[:, layer])) for layer in layers]


def profile_cell(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    rb_main: dict,
    rb_fact: dict | None,
    wu_marker: np.ndarray | None,
    layers: tuple[int, ...],
) -> dict:
    """The per-layer Δ/floor profile for one (behavior, substrate) cell.

    Loops the requested layers, calling the SAME fit machinery ``observed_read`` uses at
    the frozen layer. Returns the cell profile dict (one entry per layer + the c_C drift
    companion). Chain-ρ is computed for elicit/mix (E ≈ 0 for generic → N/A, matching
    the headline's ``with_chain=(substrate != 'generic')`` gate).
    """
    with_chain = substrate != "generic"
    drift = c_C_drift_per_layer(behavior, substrate, reduced_root, layers)
    rows: list[dict] = []
    for i, layer in enumerate(layers):
        cells = savemaps813.load_reduced_cells(behavior, substrate, layer, reduced_root)
        if behavior == "marker":
            cell = marker_mc.fit_marker_layer(layer, cells, wu_marker, with_chain=with_chain)
            dof = cell["unproj_delta_over_floor"]  # read-1 PRIMARY (behavior-agnostic)
            dmed = cell["unproj_delta_med"]
            floor_dv = None if dof in (None, 0) else float(dmed / dof)
            row = {
                "layer": layer,
                "n_cells": cell["n_cells"],
                "delta_over_floor": dof,
                "delta_med": dmed,
                "floor": floor_dv,  # the DV denominator (p95-combined for marker read-1)
                "floor_combined_p95": cell["unproj_floor_p95"]["combined"],
                # read-2 (W_U[※]-projected, marker-specific) + its subspace-capture gate
                "wu_delta_over_floor": cell["wu_proj_delta_over_floor"],
                "wu_frac_in_subspace": cell["wu_frac_in_subspace"],
                "wu_read2_informative": cell["wu_read2_informative"],
                "c_C_drift_med": drift[i],
            }
            cr = cell.get("chain_rho")
            row["chain_rho_diff"] = cr.get("rho_diff_ridge") if cr else None
            row["chain_rho_M0"] = cr.get("rho_M0_ridge") if cr else None
            row["chain_rho_Mplus"] = cr.get("rho_Mplus_ridge") if cr else None
        else:
            cell = fitM.fit_cell(behavior, layer, cells, rb_main, rb_fact, include_mlp=False)
            dof = cell["Delta_over_floor_sd"]  # the DV observed_read reports for em/fact/syco
            dmed = cell["Delta_med"]
            floor_dv = None if dof in (None, 0) else float(dmed / dof)  # == floor_sd_combined
            cr = cell["chain_rho"]
            row = {
                "layer": layer,
                "n_cells": cell["n_cells"],
                "delta_over_floor": dof,
                "delta_med": dmed,
                "floor": floor_dv,  # the DV denominator (SD-combined for em/fact/syco)
                "floor_combined_p95": cell["floor_combined"],
                "chain_rho_diff": cr.get("rho_diff_ridge"),
                "chain_rho_M0": cr.get("rho_M0_ridge"),
                "chain_rho_Mplus": cr.get("rho_Mplus_ridge"),
                "n_with_E": cr.get("n_with_E"),
                "c_C_drift_med": drift[i],
            }
        rows.append(row)
        logger.info(
            "[phase=perlayer] %s/%s L%d dof=%s dmed=%.4g drift=%.4g",
            behavior,
            substrate,
            layer,
            "None" if row["delta_over_floor"] is None else f"{row['delta_over_floor']:.4g}",
            row["delta_med"],
            row["c_C_drift_med"],
        )
    return {
        "behavior": behavior,
        "substrate": substrate,
        "headline_layer": HEADLINE_LAYER,
        "marker_two_read": behavior == "marker",
        "layers": list(layers),
        "per_layer": rows,
        "git_sha": _git_sha(),
        "hf_revision": HF_REVISION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def equivalence_gate(cell_profile: dict, delta_dir: Path) -> tuple[bool, str]:
    """L14 recomputed DV vs the committed delta_floor JSON — fail loud on drift.

    Returns (ok, detail). Skips (ok=True, 'no-oracle') only when the committed L14 JSON
    is absent (nothing to check against); otherwise asserts ``delta_over_floor`` and
    ``delta_med`` match within (EQ_RTOL, EQ_ATOL).
    """
    beh, sub = cell_profile["behavior"], cell_profile["substrate"]
    oracle_path = delta_dir / f"{beh}__{sub}.json"
    if not oracle_path.exists():
        return True, f"no-oracle ({oracle_path.name} absent)"
    oracle = json.loads(oracle_path.read_text())
    l14 = next((r for r in cell_profile["per_layer"] if r["layer"] == HEADLINE_LAYER), None)
    if l14 is None:
        return False, "L14 row missing from recomputed profile"
    for field in ("delta_over_floor", "delta_med"):
        got, exp = l14.get(field), oracle.get(field)
        if got is None or exp is None:
            if got is not exp:  # one None, one not — a real mismatch
                return False, f"{field}: recomputed={got} committed={exp}"
            continue
        if not np.isclose(got, exp, rtol=EQ_RTOL, atol=EQ_ATOL):
            return False, f"{field}: recomputed={got:.8g} committed={exp:.8g} (>tol)"
    return True, f"L14 matches committed ({beh}/{sub})"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    fit658._assert_ridge_exactness()  # #658 reduction-order gate (fail-fast at startup)
    logger.info("[phase=perlayer] device=%s; ridge exactness gate PASS", fit658.DEVICE)

    ap = argparse.ArgumentParser(
        description="Issue #813 free-analysis follow-up — per-layer Δ/floor profile (L1-27)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/perlayer"
    )
    ap.add_argument(
        "--delta-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_813/delta_floor",
        help="committed L14 delta_floor JSONs (the equivalence-gate oracle)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute every cell (default: skip cells whose per-layer JSON is complete)",
    )
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="smoke: restrict layers")
    args = ap.parse_args()

    layers = tuple(args.layers) if args.layers else LAYERS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_reduced_summaries(args.behaviors, args.substrates, args.reduced_root)

    # r_B artifacts — the SAME direction artifacts the headline observed_read loaded.
    rb_main = fitM._load_rb_main() if any(b in ("em", "sycophancy") for b in args.behaviors) else {}
    rb_fact = fitM._load_rb_fact() if "fact" in args.behaviors else None
    wu_marker = marker_mc.load_wu_marker_direction() if "marker" in args.behaviors else None

    gate_results: list[tuple[str, bool, str]] = []
    t0 = time.time()
    for behavior in args.behaviors:
        for substrate in args.substrates:
            out_path = args.out_dir / f"{behavior}__{substrate}.json"
            if out_path.exists() and not args.no_resume:
                existing = json.loads(out_path.read_text())
                if len(existing.get("per_layer", [])) == len(layers):
                    logger.info(
                        "[phase=perlayer] %s/%s resume-skip (%d layers present)",
                        behavior,
                        substrate,
                        len(layers),
                    )
                    # Re-gate the loaded profile so a resumed run still validates against L14.
                    if HEADLINE_LAYER in layers:
                        ok, detail = equivalence_gate(existing, args.delta_dir)
                        gate_results.append((f"{behavior}/{substrate}", ok, detail))
                    continue
            logger.info(
                "[phase=perlayer] profile %s/%s (%d layers)", behavior, substrate, len(layers)
            )
            profile = profile_cell(
                behavior, substrate, args.reduced_root, rb_main, rb_fact, wu_marker, layers
            )
            out_path.write_text(json.dumps(profile, indent=2, default=float))
            if HEADLINE_LAYER in layers:
                ok, detail = equivalence_gate(profile, args.delta_dir)
                gate_results.append((f"{behavior}/{substrate}", ok, detail))
                logger.info(
                    "[phase=perlayer] EQUIV %s: %s | %s", behavior, "PASS" if ok else "FAIL", detail
                )

    wall = time.time() - t0
    logger.info("[phase=perlayer] wrote %d cell profiles in %.1fs", len(gate_results), wall)

    failed = [(c, d) for c, ok, d in gate_results if not ok]
    if failed:
        for c, d in failed:
            logger.error("[phase=perlayer] EQUIVALENCE GATE FAILED %s: %s", c, d)
        raise SystemExit(
            f"L14 equivalence gate FAILED for {len(failed)} cell(s): {failed} — the per-layer "
            "recompute does NOT reproduce the committed frozen-layer headline"
        )
    if HEADLINE_LAYER in layers:
        logger.info("[phase=perlayer] L14 EQUIVALENCE GATE PASS (%d cells)", len(gate_results))
    logger.info("[phase=done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
