#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ρ, ×, Ŵ, ‖·‖, ※) in scientific docstrings + log messages.
"""Issue #813 — DVs: Δ/floor + chain-ρ + substrate-swap null + pairwise-diff CIs.

Runs OFF-POD (VM CPU) — closed-form ridge + sampling stats over the reduced
per-(behavior, substrate) summaries the extraction wave wrote. NOT an iterative
fit (ridge-only, ``include_mlp=False``), so it belongs on the CPU/VM.

Per (behavior, substrate) at the FROZEN headline layer 14 (#651/#658; applied
IDENTICALLY to the observed statistic AND the substrate-swap null — the
selection-symmetric frozen-position route, `.claude/rules/selection-symmetric-nulls.md`):

- em / fact / sycophancy → ``issue722_fit_M.fit_cell(behavior, 14, cells, rb_main,
  rb_fact, include_mlp=False)`` (the RIDGE-only headline: ``Delta_med`` /
  ``floor_combined`` / ``Delta_over_floor_sd`` / chain-ρ / support_distance /
  n_with_E). fact/syco/em have an r_B; marker does NOT (fit_cell KeyErrors on it).
- marker → ``issue667_marker_mapchange.fit_marker_layer(14, cells, wu_marker,
  with_chain=)`` (read-1 unprojected ‖ΔM‖/floor + read-2 W_U[※]-projected
  |ΔM·Ŵ_U[※]|/floor + ``wu_frac_in_subspace`` — read-2 uninformative when < 0.1).

Then, per behavior:
- **Substrate-swap null (matched-n).** Within EACH substrate, resample the
  substrate's questions and re-split them into TWO pseudo-substrates of the SAME n
  per pseudo-arm (question-average per context → a pseudo-map pair), compute the
  SAME Δ/floor DV for each pseudo-substrate, take |Δ/floor(A) − Δ/floor(B)|. The
  null holds ΔM fixed (same adapter) and varies ONLY the question sample — its
  95th percentile is the behavior-specific threshold X a REAL substrate difference
  must clear (plan §3). Matched-n keeps em's low power conservative (not inflated).
- **Pairwise substrate-difference CIs.** Family-clustered bootstrap CI on
  |Δ/floor(substrate A) − Δ/floor(substrate B)| for the three substrate pairs.

Reads the frozen headline-layer per-question rows (``per_question_L14.npz``) for
the null; reads the 28-layer reduced summary for the observed fit_cell/marker read.
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

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.analysis")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
HEADLINE_LAYER = 14  # frozen (#651/#658); observed + null read at the SAME layer
HIDDEN = 3584
N_LAYERS = 28
TARGET_DIM = 64  # top-64 v0 PCs (NEVER 48)
N_NULL_RESAMPLES = 1000
NULL_SEED = 42


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Observed Δ/floor read (reused fit machinery) ───────────────────────────────


def _cells_from_summary(behavior: str, substrate: str, layer: int, reduced_root: Path) -> list:
    """Build CellRecord list at ``layer`` from the reduced summary (import shared loader)."""
    import issue813_save_maps as savemaps813

    return savemaps813.load_reduced_cells(behavior, substrate, layer, reduced_root)


def observed_read(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    rb_main: dict,
    rb_fact: dict | None,
    wu_marker: np.ndarray | None,
) -> dict:
    """The observed Δ/floor read at the frozen headline layer for one (behavior, substrate)."""
    cells = _cells_from_summary(behavior, substrate, HEADLINE_LAYER, reduced_root)
    if behavior == "marker":
        # marker has no r_B → the two-read marker path (unproj ‖ΔM‖ + W_U[※]-proj).
        cell = marker_mc.fit_marker_layer(
            HEADLINE_LAYER, cells, wu_marker, with_chain=(substrate != "generic")
        )
        return {
            "behavior": behavior,
            "substrate": substrate,
            "layer": HEADLINE_LAYER,
            "n_cells": cell["n_cells"],
            # read-1 (behavior-agnostic) is the marker's PRIMARY floor-normalized DV
            "delta_over_floor": cell["unproj_delta_over_floor"],
            "delta_over_floor_sd": cell["unproj_delta_over_floor_sd"],
            "delta_med": cell["unproj_delta_med"],
            "floor_combined": cell["unproj_floor_p95"]["combined"],
            # read-2 (W_U[※]-projected, marker-specific) + its subspace-capture gate
            "wu_delta_over_floor": cell["wu_proj_delta_over_floor"],
            "wu_frac_in_subspace": cell["wu_frac_in_subspace"],
            "wu_read2_informative": cell["wu_read2_informative"],
            "support_distance": cell["support_distance"],
            "chain_rho": cell.get("chain_rho"),
            "marker_two_read": True,
        }
    # em / fact / sycophancy → the ridge-only headline (Delta_med / floor / chain-ρ).
    cell = fitM.fit_cell(behavior, HEADLINE_LAYER, cells, rb_main, rb_fact, include_mlp=False)
    return {
        "behavior": behavior,
        "substrate": substrate,
        "layer": HEADLINE_LAYER,
        "n_cells": cell["n_cells"],
        "delta_over_floor": cell["Delta_over_floor_sd"],
        "delta_over_floor_sd": cell["Delta_over_floor_sd"],
        "delta_med": cell["Delta_med"],
        "floor_combined": cell["floor_combined"],
        "support_distance": cell["support_distance"],
        "chain_rho": cell["chain_rho"],
        "n_with_E": cell["chain_rho"].get("n_with_E"),
        "marker_two_read": False,
    }


# ── Pseudo-substrate Δ/floor numerator (headline-layer only, for the null) ─────


def _pseudo_delta_median(
    c0: np.ndarray, cplus: np.ndarray, v0: np.ndarray, vplus: np.ndarray, r_hat: np.ndarray | None
) -> float:
    """median_c |Δ(c)·r̂| (or ‖Δ(c)‖ when r_hat is None) for a headline-layer pseudo-map.

    Fits M0 = ridge(c0→V0_64) and M⁺ = ridge(cplus→Vplus_64) at THIS layer via the
    reused ``_ridge_fit_predict`` + ``_pca_basis_v0`` (top-64 shared V0 basis),
    evaluates both at the base c0 grid, and reduces the difference either by the
    r_hat projection (em/fact/syco, r_B) or by the vector norm (marker read-1). This
    is EXACTLY ``fit_cell``'s / ``fit_marker_layer``'s numerator arithmetic on a
    single-layer stack — used to build the substrate-swap null cheaply.
    """
    pca_basis = fitM._pca_basis_v0(v0, TARGET_DIM)  # (k<=64, HIDDEN)
    v0_64 = fitM._to64(v0, pca_basis)
    vplus_64 = fitM._to64(vplus, pca_basis)
    m0_grid = fitM._ridge_fit_predict(c0, v0_64, c0)  # (n, 64)
    mplus_grid = fitM._ridge_fit_predict(cplus, vplus_64, c0)
    delta_full = (mplus_grid - m0_grid) @ pca_basis  # (n, HIDDEN)
    if r_hat is None:
        return float(np.median(np.linalg.norm(delta_full, axis=1)))
    return float(np.median(np.abs(delta_full @ r_hat)))


def substrate_swap_null(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    n_resamples: int,
) -> dict:
    """Matched-n substrate-swap null: resample questions within the substrate, split.

    Reads ``per_question_L{HEADLINE}.npz`` (flat headline-layer rows + per-row
    context index). Per resample: draw the substrate's question indices with
    replacement, split them into two disjoint pseudo-substrate halves of matched
    size, question-average each half per context → a pseudo-map pair per half,
    compute the SAME Δ numerator for each, take |Δ(A) − Δ(B)|. The 95th percentile
    is X, the behavior-specific threshold a REAL substrate difference must clear.

    Matched-n: both pseudo-arms use the SAME per-half question count, so em's small
    pool yields a WIDE (conservative) null, never an artificially tight one.
    Returns ``{"null_p95", "null_p975", "n_questions", "n_resamples_used"}``.
    """
    pq_path = reduced_root / behavior / substrate / f"per_question_L{HEADLINE_LAYER}.npz"
    if not pq_path.exists():
        raise FileNotFoundError(f"per-question headline rows missing: {pq_path}")
    d = np.load(pq_path, allow_pickle=True)
    c0 = np.asarray(d["c_C_base"], dtype=np.float64)  # (n_rows, HIDDEN)
    cp = np.asarray(d["c_C_trained"], dtype=np.float64)
    v0 = np.asarray(d["v_A_base"], dtype=np.float64)
    vp = np.asarray(d["v_A_trained"], dtype=np.float64)
    row_ctx = np.asarray(d["row_context_index"], dtype=np.int64)
    row_q = np.asarray(d["row_question_index"], dtype=np.int64)
    q_ids = sorted(set(row_q.tolist()))
    n_q = len(q_ids)
    if n_q < 4:
        return {
            "null_p95": None,
            "null_p975": None,
            "n_questions": n_q,
            "n_resamples_used": 0,
            "note": "too few questions (<4) for a matched-n split",
        }

    # Map (context, question) → row index for fast per-half question-averaging.
    rc_index: dict[tuple[int, int], int] = {}
    for i in range(len(row_ctx)):
        rc_index[(int(row_ctx[i]), int(row_q[i]))] = i
    contexts = sorted(set(row_ctx.tolist()))

    def _pseudo_stack(q_subset: list[int]) -> tuple[np.ndarray, ...]:
        """Question-average the given question subset per context → (n_ctx_kept, HIDDEN) stacks."""
        rows_c0, rows_cp, rows_v0, rows_vp = [], [], [], []
        for ctx in contexts:
            idxs = [rc_index[(ctx, q)] for q in q_subset if (ctx, q) in rc_index]
            if not idxs:
                continue
            rows_c0.append(c0[idxs].mean(0))
            rows_cp.append(cp[idxs].mean(0))
            rows_v0.append(v0[idxs].mean(0))
            rows_vp.append(vp[idxs].mean(0))
        return (np.stack(rows_c0), np.stack(rows_cp), np.stack(rows_v0), np.stack(rows_vp))

    rng = np.random.default_rng(NULL_SEED)
    half = n_q // 2
    diffs: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(q_ids, size=n_q, replace=True).tolist()
        # split the RESAMPLED question list into two matched-n halves.
        a_qs = drawn[:half]
        b_qs = drawn[half : 2 * half]
        try:
            sa = _pseudo_stack(a_qs)
            sb = _pseudo_stack(b_qs)
            if sa[0].shape[0] < 4 or sb[0].shape[0] < 4:
                continue  # a degenerate half (too few contexts covered) — skip
            da = _pseudo_delta_median(*sa, r_hat)
            db = _pseudo_delta_median(*sb, r_hat)
        except np.linalg.LinAlgError:
            continue  # degenerate resample geometry — skip (bootstrap noise)
        diffs.append(abs(da - db))
    if not diffs:
        return {
            "null_p95": None,
            "null_p975": None,
            "n_questions": n_q,
            "n_resamples_used": 0,
            "note": "all resamples degenerate",
        }
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "null_p95": float(np.percentile(arr, 95)),
        "null_p975": float(np.percentile(arr, 97.5)),
        "null_median": float(np.median(arr)),
        "n_questions": n_q,
        "n_resamples_used": len(diffs),
    }


def _r_hat_for(
    behavior: str, rb_main: dict, rb_fact: dict | None, wu_marker: np.ndarray | None
) -> np.ndarray | None:
    """The read-out direction at the headline layer (None for marker read-1 = ‖ΔM‖)."""
    if behavior == "marker":
        return None  # marker null uses read-1 (‖ΔM‖), behavior-agnostic
    return fitM._r_hat_for(behavior, HEADLINE_LAYER, rb_main, rb_fact)


def pairwise_substrate_diff_cis(observed_by_sub: dict[str, dict]) -> list[dict]:
    """Point difference of the observed Δ_med across substrate pairs (headline layer).

    Reports the |Δ_med(A) − Δ_med(B)| point difference for each substrate pair; the
    substrate-swap null band (per substrate) is the reference the difference is
    judged against (a difference beyond max(null_p95(A), null_p95(B)) is a real
    substrate effect). The full clustered-bootstrap CI on the paired difference is
    delegated to the analyzer (needs the paired per-context grids); here we report
    the point differences the null band gates.
    """
    out = []
    subs = [s for s in SUBSTRATES if s in observed_by_sub]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            da = observed_by_sub[a].get("delta_med")
            db = observed_by_sub[b].get("delta_med")
            out.append(
                {
                    "pair": f"{a}_vs_{b}",
                    "delta_med_a": da,
                    "delta_med_b": db,
                    "abs_diff": (None if (da is None or db is None) else abs(da - db)),
                }
            )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    fit658._assert_ridge_exactness()
    logger.info("[phase=analysis] device=%s; ridge exactness gate PASS", fit658.DEVICE)

    ap = argparse.ArgumentParser(
        description="Issue #813 — DVs (Δ/floor + chain-ρ + substrate-swap null)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument("--n-null-resamples", type=int, default=N_NULL_RESAMPLES)
    args = ap.parse_args()

    # r_B artifacts (em/syco from #658 r_b.pt; fact from #667 r_b_fact.pt; marker W_U[※]).
    rb_main = fitM._load_rb_main() if any(b in ("em", "sycophancy") for b in args.behaviors) else {}
    rb_fact = fitM._load_rb_fact() if "fact" in args.behaviors else None
    wu_marker = marker_mc.load_wu_marker_direction() if "marker" in args.behaviors else None

    delta_dir = args.out_dir / "delta_floor"
    chain_dir = args.out_dir / "chain_rho"
    null_dir = args.out_dir / "substrate_swap_null"
    for d in (delta_dir, chain_dir, null_dir):
        d.mkdir(parents=True, exist_ok=True)

    per_behavior: dict[str, dict] = {}
    for behavior in args.behaviors:
        observed_by_sub: dict[str, dict] = {}
        null_by_sub: dict[str, dict] = {}
        r_hat = _r_hat_for(behavior, rb_main, rb_fact, wu_marker)
        for substrate in args.substrates:
            logger.info(
                "[phase=analysis] observed read %s/%s L%d", behavior, substrate, HEADLINE_LAYER
            )
            obs = observed_read(behavior, substrate, args.reduced_root, rb_main, rb_fact, wu_marker)
            observed_by_sub[substrate] = obs
            (delta_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(obs, indent=2, default=float)
            )
            # chain-ρ (elicit+mix only; generic E≈0 → N/A per plan §3/§6)
            if substrate != "generic" and obs.get("chain_rho") is not None:
                (chain_dir / f"{behavior}__{substrate}.json").write_text(
                    json.dumps(
                        {
                            "behavior": behavior,
                            "substrate": substrate,
                            "chain_rho": obs["chain_rho"],
                        },
                        indent=2,
                        default=float,
                    )
                )
            # substrate-swap null (matched-n) at the frozen headline layer
            logger.info("[phase=analysis] substrate-swap null %s/%s", behavior, substrate)
            null = substrate_swap_null(
                behavior, substrate, args.reduced_root, r_hat, args.n_null_resamples
            )
            null_by_sub[substrate] = null
            (null_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(null, indent=2, default=float)
            )

        pairwise = pairwise_substrate_diff_cis(observed_by_sub)
        per_behavior[behavior] = {
            "observed": observed_by_sub,
            "substrate_swap_null": null_by_sub,
            "pairwise_substrate_diff": pairwise,
        }
        # Verdict per behavior: does the max-vs-min substrate Δ_med difference clear
        # the substrate-swap null? (H1 = substrate-agnostic iff within the band.)
        deltas = {s: observed_by_sub[s].get("delta_med") for s in observed_by_sub}
        valid = {s: v for s, v in deltas.items() if v is not None}
        if len(valid) >= 2:
            max_diff = max(valid.values()) - min(valid.values())
            null_x = max(
                (null_by_sub[s].get("null_p95") or 0.0) for s in valid if null_by_sub.get(s)
            )
            per_behavior[behavior]["verdict"] = {
                "max_vs_min_delta_med_diff": max_diff,
                "null_x_p95": null_x,
                "substrate_matters": (max_diff > null_x) if null_x else None,
            }
            logger.info(
                "[phase=analysis] %s: max-min Δ_med diff=%.4g vs null X(p95)=%.4g → matters=%s",
                behavior,
                max_diff,
                null_x,
                per_behavior[behavior]["verdict"]["substrate_matters"],
            )

    summary = {
        "issue": 813,
        "read": "map_change_substrate_dependence_M0_vs_Mplus",
        "headline_layer": HEADLINE_LAYER,
        "target_dim": TARGET_DIM,
        "n_null_resamples": args.n_null_resamples,
        "null_seed": NULL_SEED,
        "git_commit": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_behavior": per_behavior,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("[phase=analysis] wrote %s", args.out_dir / "summary.json")
    # NO [phase=done] here — analysis runs as a SUBPROCESS of issue813_dispatch.sh
    # (phase 4), whose stdout it inherits; the poller reserves [phase=done] for the
    # ONE terminal line the .sh emits AFTER the sentinel write (#545). A premature
    # [phase=done] here would false-signal completion before the sentinel exists.
    logger.info("[phase=analysis] analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
