#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings + log messages.
"""Issue #810 DV (b) — behavior read-out: held-out ρ of predicted vs graded E0.

Reads a behavior E0(C,B) out of the answer-side summary two ways:
- **method (i) fixed r_B** — the faithful #658 A3.3 projection ``E0 ≈ r_Bᵀ summary``
  (zero parameters); runs on the behaviors with an r_B in #658's store
  ({harmful_compliance, sycophancy, refusal}; broad_em dropped — floors on base).
- **method (ii) trained LOCO-ridge** — a learned ``summary → E0`` map (E0 as a
  1-column target) via ``vectorized_mlp_skill.ridge_predict_loco_centered``; runs
  on ALL graded-E0 behaviors.

PRIMARY target = the graded 0-100 E0 mean (llm-judging rule 1); COMPANION = the
binary judged-RATE (rule 2). Both are read per (context, behavior). High-m E0
comes from #810 Phase C (``e0_highm_graded.json``); low-m E0 from #763 (IN-FLIGHT
— OPTIONAL, folded when it lands; its absence is reported, never a crash).

NULLS (selection-symmetric, plan §6 + Alternatives-lens binding Concern):
- **fixed r_B** is a ZERO-parameter projection → the on-main permute-and-refit
  ridge null is WRONG for it. The correct null PERMUTES the (E0, summary) context
  pairing and RE-PROJECTS the SAME r_B (breaks alignment, preserves marginals).
- **trained ridge** null PERMUTES the E0 rows and RE-FITS the ridge.
Per-draw stats for EVERY (summary × layer × behavior × method) cell are persisted
to ``null_matrix_readout.json`` so the analyzer recomputes the honest
max-over-{summary, layer, behavior, method} band as a 0-GPU re-reduction.

H2 CONJUNCTION (Statistics + Methodology reconciler binding): a summary counts as
an H2 confirmation ONLY if it lifts read-out on BOTH methods (fixed r_B AND
trained ridge) for a behavior. Per-method lifts are REPORTED separately; the
conjunction gate is encoded (``h2_conjunction`` field per (behavior × summary)),
NOT a method-OR.

JUDGE-VALIDATION (llm-judging rules 13/15): Spearman(graded-E0, #722 tf-margin)
across the 50 contexts for {refusal, sycophancy} (the overlapping ± pool).
harmful_compliance has NO ± pool → validation gap NOTED, never fabricated. Reads
the tf_margin committed to main by §4.0 step 1.

LENGTH/VERBOSITY control (llm-judging rule 10, Alternatives-lens binding): per
behavior, correlate answer length with (a) the winning summary's read-out
prediction and (b) graded E0.

SELF-CONTAINED against on-main primitives (no import of the stranded
fig-per-position script).

Usage::

    uv run python scripts/issue810_fit_readout.py \\
        --e0-highm eval_results/issue_810/e0_highm_graded.json \\
        --position-store-hf issue658_theory_assumptions/answer_position_sweep \\
        --out eval_results/issue_810

    # smoke (1 behavior, 1 layer, fixed-r_B only): --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue810_batched_null import (  # noqa: E402
    batched_projection_null_rho,
    batched_ridge_loco_null_rho,
    make_perm_matrix,
)
from issue810_common import (  # noqa: E402
    HF_DATA_REPO,
    I658_RB,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    I722_TF_MARGIN_FILE,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    TF_MARGIN_VALIDATION_BEHAVIORS,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    summary_names,
    upload_out_dir,
)
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
)

logger = logging.getLogger("issue810_fit_readout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _rho(pred: np.ndarray, meas: np.ndarray) -> float | None:
    """Spearman ρ, guarded against degenerate (constant / tiny-n) inputs."""
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def _tf_margin_scalar(cell) -> float | None:
    """Extract the scalar tf-margin from a tf_margin cell.

    tf_margin schema: ``margins[ctx][behavior]`` is a DICT
    ``{"margin": <float>, "pos_mean_ln_logp", ...}`` — the scalar we validate
    against is ``["margin"]``. A bare-scalar or missing cell is handled too.
    """
    if isinstance(cell, dict):
        return cell.get("margin")
    return cell


# ── inputs ────────────────────────────────────────────────────────────────────


def _load_free_summaries():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_V0_SUMMARIES, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["summaries"], blob["capture_layers"]


def _load_rb():
    """{behavior: {recipe: (28,H)}} from #658 store/r_b.pt (recipes diffmeans/meanDB)."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_RB, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["r_b"], blob.get("columns", list(blob["r_b"].keys()))


def _load_position_summaries(ctx_ids, hf_prefix, local_dir):
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[str, np.ndarray]] = {}
    cov: dict[str, dict[str, int]] = {}
    for c in ctx_ids:
        if local_dir is not None and (local_dir / f"{c}.pt").is_file():
            blob = torch.load(local_dir / f"{c}.pt", weights_only=False)
        else:
            path = hf_hub_download(HF_DATA_REPO, f"{hf_prefix}/{c}.pt", repo_type="dataset")
            blob = torch.load(path, weights_only=False)
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        out[c] = {name: pv[i] for i, name in enumerate(blob["positions"])}
        cov[c] = dict(blob["coverage"])
    return out, cov


def _e0_by_context(e0_highm: dict, low_m: dict | None) -> dict[str, dict[str, dict]]:
    """{behavior: {"graded": {ctx:val}, "rate": {ctx:val}}} merged high-m + low-m.

    High-m: #810 Phase C ``e0_highm_graded.json`` (graded mean + binary rate).
    Low-m: #763 graded E0 (OPTIONAL, IN-FLIGHT) — a dict {behavior: {ctx: score}}
    when landed, else None (its absence is reported by the caller, not a crash).
    """
    out: dict[str, dict[str, dict]] = {}
    for behavior, blk in e0_highm.get("by_behavior", {}).items():
        out[behavior] = {
            "graded": {k: v for k, v in blk["per_context_graded_mean"].items() if v is not None},
            "rate": {
                k: v for k, v in blk.get("per_context_binary_rate", {}).items() if v is not None
            },
        }
    if low_m:
        for behavior, ctx_scores in low_m.items():
            out.setdefault(behavior, {"graded": {}, "rate": {}})
            out[behavior]["graded"] = {k: v for k, v in ctx_scores.items() if v is not None}
    return out


def _summary_matrix(summary, layer_i, kept, free_summaries, pos_summaries, coverage):
    """(n, H) summary matrix at one layer over the kept ctx_ids (coverage-checked)."""
    rows = []
    for c in kept:
        if summary in ("mean", "last", "maxp"):
            rows.append(free_summaries[summary][c][layer_i].numpy())
        else:
            rows.append(pos_summaries[c][summary][layer_i])
    return np.stack(rows)


def _kept_contexts(summary, ctx_ids, coverage):
    """Contexts with coverage for this summary (free recipes always covered)."""
    if summary in ("mean", "last", "maxp"):
        return list(ctx_ids)
    return [c for c in ctx_ids if coverage[c].get(summary, 0) > 0]


# ── read-out methods ──────────────────────────────────────────────────────────


def _fixed_rb_pred(X: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Method (i): zero-parameter projection E0 ≈ r_Bᵀ summary (no fit)."""
    return X @ r


def _pca_reduce_predictor(X: np.ndarray) -> np.ndarray:
    """PCA-reduce the summary design to its top min(48, n-2) dims (fixed per cell).

    Shared by ``_trained_ridge_pred`` and the batched trained-ridge null so BOTH
    use the IDENTICAL basis — the design X is fixed across permutations, so the
    PCA basis is too; the batched null re-uses this reduced design rather than
    re-fitting the basis per permutation.
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import robust_pca_basis

    n = X.shape[0]
    k = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    mu, comps, _ = robust_pca_basis(X, k)
    return (X - mu) @ comps.T  # (n, k) PCA-reduced predictor


def _trained_ridge_pred(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method (ii): held-out LOCO-ridge prediction of scalar E0 from the summary.

    Uses the on-main ``ridge_predict_loco_centered`` with a 1-column target;
    reduces the summary to its top PCA dims first (target dim min(48, n-2)) so
    the H-dim predictor matches #722/#658's estimator. Returns (n,) held-out.
    """
    Xp = _pca_reduce_predictor(X)
    pred = ridge_predict_loco_centered(Xp, y.reshape(-1, 1))  # (n, 1)
    return pred[:, 0]


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 DV (b): behavior read-out rho")
    ap.add_argument(
        "--e0-highm",
        default=str(PROJECT_ROOT / "eval_results" / "issue_810" / "e0_highm_graded.json"),
    )
    ap.add_argument("--e0-lowm", default=None, help="#763 graded E0 JSON (optional; in-flight)")
    ap.add_argument(
        "--position-store-hf", default="issue658_theory_assumptions/answer_position_sweep"
    )
    ap.add_argument("--position-store-dir", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_810"))
    ap.add_argument("--summaries", nargs="*", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["fixed_rb", "trained_ridge"])
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for the batched trained-ridge + projection nulls ('cpu' default, "
        "'cuda' to run on a GPU lane — CPU behavior is byte-identical to the default)",
    )
    ap.add_argument(
        "--upload-prefix",
        default=None,
        help="HF data-repo path prefix (e.g. 'issue810/phase_d_readout') to bulk-upload the "
        "out-dir *.json to after the fit; UNSET (default) = no upload (today's behavior). "
        "Set on an ephemeral GCP lane so the results survive instance teardown.",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # Fail fast: --device cuda with no CUDA never silently falls back to CPU.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    from huggingface_hub import hf_hub_download

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[phase=load] manifest + summaries + r_B + E0 + tf_margin")
    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids_all = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries()
    rb, _rb_columns = _load_rb()

    local_dir = Path(args.position_store_dir) if args.position_store_dir else None
    if local_dir is not None:
        pos_man = load_json(local_dir / "manifest.json")
    else:
        pos_man = load_json(
            hf_hub_download(
                HF_DATA_REPO, f"{args.position_store_hf}/manifest.json", repo_type="dataset"
            )
        )
    ctx_ids = [c for c in pos_man["context_ids"] if c in ctx_ids_all]
    pos_summaries, coverage = _load_position_summaries(ctx_ids, args.position_store_hf, local_dir)

    e0_highm = load_json(args.e0_highm)
    low_m = load_json(args.e0_lowm) if args.e0_lowm and Path(args.e0_lowm).is_file() else None
    if args.e0_lowm and low_m is None:
        logger.warning(
            "low-m E0 (%s) NOT landed — read-out folds high-m only (in-flight dep)", args.e0_lowm
        )
    e0 = _e0_by_context(e0_highm, low_m)

    summaries = args.summaries or summary_names()
    layers = args.layers if args.layers is not None else list(range(len(capture_layers)))
    behaviors = args.behaviors or list(e0.keys())
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)

    results: list[dict] = []
    # null_matrix[behavior][method][summary][layer] = [per-draw ρ]
    null_matrix: dict = {}

    for behavior in behaviors:
        graded = e0.get(behavior, {}).get("graded", {})
        rates = e0.get(behavior, {}).get("rate", {})
        if len(graded) < 4:
            logger.info("[phase=skip] %s: <4 graded contexts (%d) — skipped", behavior, len(graded))
            continue
        null_matrix.setdefault(behavior, {})
        for method in args.methods:
            if method == "fixed_rb" and behavior not in rb:
                logger.info(
                    "[phase=skip] fixed_rb has no r_B for %s — trained_ridge only", behavior
                )
                continue
            null_matrix[behavior].setdefault(method, {})
            for summary in summaries:
                kept = [c for c in _kept_contexts(summary, ctx_ids, coverage) if c in graded]
                if len(kept) < 4:
                    continue
                y = np.array([graded[c] for c in kept], dtype=np.float64)
                y_rate = np.array(
                    [rates.get(c, np.nan) for c in kept], dtype=np.float64
                )  # companion
                null_matrix[behavior][method].setdefault(summary, {})
                for li in layers:
                    X = _summary_matrix(summary, li, kept, free_summaries, pos_summaries, coverage)
                    # Draw n_perms permutations from the SHARED rng in the SAME
                    # order the serial per-draw loop consumed them (byte-identical
                    # null on a like-seeded rng — the smoke asserts this).
                    perm = make_perm_matrix(len(kept), args.n_perms, rng)
                    if method == "fixed_rb":
                        # diffmeans is the theory default; report both recipes but
                        # gate the headline on diffmeans (persona-vectors default).
                        r = rb[behavior]["diffmeans"][li].numpy()
                        pred = _fixed_rb_pred(X, r)
                        # correct null (batched, no re-fit): permute the (E0,
                        # summary) pairing, re-project the SAME pred → re-Spearman.
                        draws = batched_projection_null_rho(pred, y, perm, device=args.device)
                    else:  # trained_ridge
                        pred = _trained_ridge_pred(X, y)
                        # null (batched): permute E0 rows, re-fit the LOCO ridge on
                        # the FIXED PCA-reduced design (X-only factors computed once,
                        # all perms batched — the #722 vectorize mandate). Uses the
                        # IDENTICAL PCA basis as _trained_ridge_pred.
                        Xp = _pca_reduce_predictor(X)
                        draws = batched_ridge_loco_null_rho(Xp, y, perm, device=args.device)
                    rho = _rho(pred, y)
                    rho_rate = _rho(pred, y_rate) if np.isfinite(y_rate).all() else None
                    results.append(
                        {
                            "behavior": behavior,
                            "method": method,
                            "summary": summary,
                            "layer": capture_layers[li],
                            "n": len(kept),
                            "rho_graded": rho,  # PRIMARY
                            "rho_binary_rate": rho_rate,  # COMPANION
                        }
                    )
                    null_matrix[behavior][method][summary][str(capture_layers[li])] = draws
            logger.info("[phase=fit] %s / %s done", behavior, method)

    # H2 conjunction: per (behavior × summary), does it lift on BOTH methods vs
    # the `mean` summary baseline? Recorded as a gate the analyzer reads (the
    # honest band re-reduction happens post-hoc from null_matrix).
    conjunction = _h2_conjunction(results)

    # Judge validation: Spearman(graded-E0, tf-margin) for {refusal, sycophancy}.
    judge_val = _judge_validation(e0, ctx_ids)

    # Length/verbosity control: answer length vs (winning-summary pred, graded E0).
    length_control = _length_control(
        results,
        e0,
        ctx_ids,
        pos_man,
        free_summaries,
        pos_summaries,
        coverage,
        layers,
        capture_layers,
        rb,
    )

    dump_json(
        {
            "dv": "behavior_readout_rho_vs_graded_e0",
            "primary": "rho_graded (graded 0-100 E0)",
            "companion": "rho_binary_rate (judged rate >=50)",
            "n_contexts_grid": len(ctx_ids),
            "behaviors_fit": sorted({r["behavior"] for r in results}),
            "methods": args.methods,
            "low_m_e0_landed": low_m is not None,
            "cells": results,
            "h2_conjunction": conjunction,
            "judge_validation": judge_val,
            "length_control": length_control,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "readout_rho_by_summary.json",
    )
    dump_json(
        {
            "dv": "readout",
            "axes": "behavior -> method -> summary -> layer -> [per-draw rho]",
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "readout": null_matrix,
        },
        out_dir / "null_matrix_readout.json",
    )
    if args.upload_prefix:
        logger.info("[phase=upload] fit-result JSONs -> %s", args.upload_prefix)
        landed = upload_out_dir(out_dir, args.upload_prefix)
        logger.info("[phase=upload] verified fit-result JSONs under %s/", landed)
    logger.info("[phase=done] wrote read-out results + null matrix to %s", out_dir)
    return 0


def _h2_conjunction(results: list[dict]) -> dict:
    """Per (behavior × summary): does it beat the `mean`-summary ρ on BOTH methods?

    H2 CONJUNCTION reading (binding). Reports the best-layer ρ per (behavior ×
    summary × method) and the `mean`-summary best-layer ρ per (behavior ×
    method), and flags ``both_methods_lift`` iff the summary's best-layer ρ
    exceeds the `mean` baseline's best-layer ρ on BOTH methods present for the
    behavior. This is the encoded H2 gate; the honest max-selected band comes
    from null_matrix post-hoc (analyzer).
    """
    by = {}  # (behavior, method, summary) -> best-layer rho
    methods_by_behavior: dict[str, set] = {}
    for r in results:
        if r["rho_graded"] is None:
            continue
        key = (r["behavior"], r["method"], r["summary"])
        by[key] = max(by.get(key, -2.0), r["rho_graded"])
        methods_by_behavior.setdefault(r["behavior"], set()).add(r["method"])
    out = {}
    behaviors = {k[0] for k in by}
    for behavior in behaviors:
        methods = sorted(methods_by_behavior.get(behavior, set()))
        summaries = {k[2] for k in by if k[0] == behavior}
        mean_rho = {m: by.get((behavior, m, "mean")) for m in methods}
        out[behavior] = {"methods": methods, "mean_baseline_best_rho": mean_rho, "summaries": {}}
        for s in summaries:
            if s == "mean":
                continue
            per_method = {m: by.get((behavior, m, s)) for m in methods}
            lifts = {
                m: (
                    per_method[m] is not None
                    and mean_rho[m] is not None
                    and per_method[m] > mean_rho[m]
                )
                for m in methods
            }
            out[behavior]["summaries"][s] = {
                "best_rho_per_method": per_method,
                "lifts_per_method": lifts,
                "both_methods_lift": len(methods) >= 2 and all(lifts.values()),
            }
    return out


def _judge_validation(e0: dict, ctx_ids: list[str]) -> dict:
    """Spearman(graded-E0, #722 tf-margin) for {refusal, sycophancy}.

    tf_margin schema: {margins: {ctx: {behavior: <margin>}}, behaviors: [...]}.
    harmful_compliance has NO ± pool -> gap noted, never fabricated. Reads the
    tf_margin committed to main by §4.0 step 1 (git-tree path, resolved locally
    on any lane that cloned the up-to-date branch/main).
    """
    tf_path = PROJECT_ROOT / I722_TF_MARGIN_FILE
    if not tf_path.is_file():
        return {"available": False, "reason": f"tf_margin not present at {tf_path}"}
    tf = load_json(tf_path)
    margins = tf["margins"]  # {ctx: {behavior: margin}}
    tf_behaviors = set(tf.get("behaviors", []))
    out = {"available": True, "tf_behaviors": sorted(tf_behaviors), "by_behavior": {}}
    for behavior in TF_MARGIN_VALIDATION_BEHAVIORS:
        if behavior not in tf_behaviors:
            out["by_behavior"][behavior] = {"status": "no_tf_margin_pool"}
            continue
        graded = e0.get(behavior, {}).get("graded", {})
        pairs = []
        for c in ctx_ids:
            if c not in graded:
                continue
            tfv = _tf_margin_scalar(margins.get(c, {}).get(behavior))
            if tfv is not None:
                pairs.append((graded[c], tfv))
        if len(pairs) < 4:
            out["by_behavior"][behavior] = {"status": "insufficient_overlap", "n": len(pairs)}
            continue
        g = np.array([p[0] for p in pairs])
        m = np.array([p[1] for p in pairs])
        out["by_behavior"][behavior] = {"status": "computed", "n": len(pairs), "rho": _rho(g, m)}
    out["gap_note"] = "harmful_compliance has NO tf_margin ± pool — validation gap (not fabricated)"
    return out


def _length_control(
    results,
    e0,
    ctx_ids,
    pos_man,
    free_summaries,
    pos_summaries,
    coverage,
    layers,
    capture_layers,
    rb,
) -> dict:
    """Per behavior: corr(answer length, winning-summary read-out pred) + corr(len, graded E0).

    A length-explained lift is NOT a persona finding (llm-judging rule 10). Answer
    length is the per-context median answer token count (from the Phase B manifest's
    per_context_diag). Uses the winning (behavior, method, summary, layer) by best ρ.
    """
    per_ctx_len = {}
    for c, d in pos_man.get("per_context_diag", {}).items():
        per_ctx_len[c] = d.get("median_answer_len")
    out = {}
    # winning cell per behavior (max graded ρ across all method/summary/layer)
    best: dict[str, dict] = {}
    for r in results:
        if r["rho_graded"] is None:
            continue
        b = r["behavior"]
        if b not in best or r["rho_graded"] > best[b]["rho_graded"]:
            best[b] = r
    for behavior, cell in best.items():
        graded = e0.get(behavior, {}).get("graded", {})
        summary, li_val, method = cell["summary"], cell["layer"], cell["method"]
        li = capture_layers.index(li_val)
        kept = [
            c for c in ctx_ids if c in graded and c in per_ctx_len and per_ctx_len[c] is not None
        ]
        if summary not in ("mean", "last", "maxp"):
            kept = [c for c in kept if coverage[c].get(summary, 0) > 0]
        if len(kept) < 4:
            out[behavior] = {"status": "insufficient", "n": len(kept)}
            continue
        X = _summary_matrix(summary, li, kept, free_summaries, pos_summaries, coverage)
        if method == "fixed_rb" and behavior in rb:
            pred = X @ rb[behavior]["diffmeans"][li].numpy()
        else:
            pred = _trained_ridge_pred(X, np.array([graded[c] for c in kept]))
        lengths = np.array([per_ctx_len[c] for c in kept], dtype=np.float64)
        g = np.array([graded[c] for c in kept], dtype=np.float64)
        out[behavior] = {
            "winning_cell": {"method": method, "summary": summary, "layer": li_val},
            "n": len(kept),
            "rho_len_vs_pred": _rho(lengths, pred),
            "rho_len_vs_graded_e0": _rho(lengths, g),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
