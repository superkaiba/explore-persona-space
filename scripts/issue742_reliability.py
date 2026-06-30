#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #742 Stage 0 — reliability ceiling + the bracket (plan v7 §4 Stage 0).

Per behavior × genre over the FROZEN #658 base-model tensors (0 GPU):

  * reproduce #658's per-behavior ridge ``ρ_lin`` (join-integrity gate, §4 step 0c),
  * estimate the reliability ``r_yy`` TWO agreeing ways — the estimator pair FORKS
    on the behavior's ``rollouts/probe`` count (§11 row 1): split-half-over-rollouts
    (+Spearman-Brown) for ≥2-rollout behaviors, split-half-over-probes for
    1-rollout behaviors, and the binomial-variance decomposition (cell-actual ``m``,
    NEVER a blanket ``m=2000``) for both,
  * report the bracket ``[ρ_lin, √(r_yy)]`` + its width with a cluster-bootstrap CI,
  * lay #658's ``noise_floor`` reliability alongside (the 2-method agreement check),
  * report the model-free Bayes-error companion ``β``,
  * route each cell through the Stage-0 internal gate (ceiling-limited vs headroom).

The headline DV is the BRACKET WIDTH ``√(r_yy) − ρ_lin``; the disattenuated ratio
``ρ_lin / √(r_yy)`` is NEVER computed as a headline (plan §6/§11 row 10 — unstable
at n=50, Storrs 2020).

CPU-only, runs on the VM. Writes ``eval_results/issue_742/stage0_brackets.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue404_common import reproducibility_metadata  # noqa: E402

from explore_persona_space.analysis import issue_742_decoding_ceiling as dc  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_658"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_742"

# Behaviors that carry ≥2 rollouts/probe (split-half-over-rollouts well-posed) vs
# 1 rollout/probe (split-half-over-probes). Plan §4 Stage-0 step 1 / §12 row 2.
GE2_ROLLOUT_BEHAVIORS = ("sycophancy", "broad_em")
ONE_ROLLOUT_BEHAVIORS = ("harmful_compliance", "refusal")
DISAGREE_THRESHOLD = 0.10  # §7 1-rollout-disagreement row


def _per_context_arrays(
    e0_behavior: dict, context_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[list[float]] | None]:
    """Extract per-context (rate, n_judged, per_probe) for one behavior.

    Returns ``(rates, m_cell, per_probe_or_None)`` aligned to ``context_ids``. ``m_cell``
    is the cell-actual ``n_judged`` per context (NEVER a blanket 2000); ``per_probe`` is
    the heterogeneous per-probe rate list when present (used for split-half).
    """
    rates = np.array([float(e0_behavior[c]["rate"]) for c in context_ids])
    m_cell = np.array([int(e0_behavior[c].get("n_judged") or 0) for c in context_ids])
    has_per_probe = all("per_probe" in e0_behavior[c] for c in context_ids)
    # per_probe[c] is a list of dicts {probe, e0, n_judged}; extract the per-probe rate
    # (the `e0` field) per context. NEVER index a bare float — the #658 schema wraps it.
    per_probe = (
        [[float(p["e0"]) for p in e0_behavior[c]["per_probe"]] for c in context_ids]
        if has_per_probe
        else None
    )
    return rates, m_cell, per_probe


def _build_probe_rate_matrix(per_probe: list[list[float]]) -> np.ndarray | None:
    """Stack a behavior's per-context per-probe rates into ``(n_contexts, n_probes)``.

    Only valid when every context shares the same probe count; returns None otherwise
    (then the split-half-over-probes read is skipped and only the binomial read stands).
    """
    lens = {len(p) for p in per_probe}
    if len(lens) != 1:
        return None
    return np.array(per_probe, dtype=float)


def compute_bracket(
    behavior: str,
    genre: str,
    e0: dict,
    context_ids: list[str],
    *,
    rho_lin: float,
    rng: np.random.Generator,
    n_split_seeds: int,
    n_boot: int,
    v0_layer: np.ndarray | None = None,
    layer: int | None = None,
    join_tol: float = 0.05,
    judge_honest_sqrt: float | None = None,
) -> dict:
    """Compute one (behavior, genre) bracket entry (plan §4 Stage-0 steps 0c/1/3/4/5/7).

    ``v0_layer`` / ``layer`` (when supplied) drive the Stage-0 step-0c LOCO-CV ridge
    re-fit join-integrity gate (BLOCKER ridge-refit-missing). ``judge_honest_sqrt``,
    when supplied (read from ``stage0_judge_variance.json``), folds the judge term into
    the honest ceiling and is the value the gate/headline use (BLOCKER
    stage0-reliability-not-registered (b)).
    """
    e0_b = {c: e0[c][behavior] for c in context_ids if c in e0 and behavior in e0[c]}
    present = [c for c in context_ids if c in e0_b]
    rates, m_cell, per_probe = _per_context_arrays({c: e0_b[c] for c in present}, present)
    var_c = float(np.var(rates))

    # BLOCKER stage0-reliability-not-registered(c): the on-disk E0_expression.json
    # stores per-PROBE rates only (per-rollout labels are NOT persisted), so the
    # ONLY split-half the disk supports is split-half-OVER-PROBES — for every
    # behavior. The ≥2-rollout behaviors' true over-rollouts estimator is genuinely
    # UNAVAILABLE (no rollout labels on disk); we flag that honestly rather than
    # mislabel an over-probes read as over-rollouts (the prior round's bug).
    split_half: float | None = None
    estimator_kind = "binomial_only"
    over_rollouts_available = False
    probe_mat = _build_probe_rate_matrix(per_probe) if per_probe is not None else None
    if probe_mat is not None and probe_mat.shape[1] >= 2:
        split_half = dc.reliability_split_half_over_probes(
            probe_mat, n_split_seeds=n_split_seeds, rng=rng
        )
        estimator_kind = "split_half_over_probes"

    binom = dc.reliability_binomial_variance(rates, m_cell)
    sqrt_binom = float(np.sqrt(binom))
    sqrt_split = float(np.sqrt(split_half)) if split_half is not None else None
    # headline √(r_yy): the judge-folded honest ceiling when available (BLOCKER 5b),
    # else the split-half read, else the binomial read; NEVER an average.
    if judge_honest_sqrt is not None:
        sqrt_r_yy = float(judge_honest_sqrt)
    elif sqrt_split is not None:
        sqrt_r_yy = sqrt_split
    else:
        sqrt_r_yy = sqrt_binom

    # §7 estimator-disagreement guard (CONCERN one-rollout-splithalf-dropped): now
    # FIRES for every behavior because split_half is genuinely computed (over-probes)
    # for all four — the guard was dead before only because split_half was silently
    # dropped for the 1-rollout behaviors.
    disagree = split_half is not None and abs(float(split_half) - float(binom)) > DISAGREE_THRESHOLD

    # BLOCKER stage0-reliability-not-registered(a): CV-MATCHED ceiling CI — one √(r_yy)
    # estimate per LOCO fold (held-out context excluded), NOT a pooled bootstrap
    # against a CV'd ρ (Storrs 2020). The fold-to-fold spread is the honest CV variance.
    cv_mean, ceil_lo, ceil_hi = dc.cv_matched_reliability_ci(rates, m_cell.astype(float))

    # BLOCKER ridge-refit-missing: Stage-0 step-0c LOCO-CV ridge re-fit + join-integrity
    # gate per genre. A deviation > tol from #658's persisted lin_rho means the inputs
    # are mis-joined for this genre -> RAISE (handled in run()).
    join: dict | None = None
    if v0_layer is not None and layer is not None:
        jr = dc.ridge_join_integrity(
            v0_layer,
            rates,
            behavior=behavior,
            genre=genre,
            layer=layer,
            persisted_rho=rho_lin,
            tol=join_tol,
        )
        join = {
            "refit_rho": jr.refit_rho,
            "persisted_rho": jr.persisted_rho,
            "delta": jr.delta,
            "join_ok": jr.join_ok,
            "tol": jr.tol,
            "layer": jr.layer,
        }

    bracket_width = float(sqrt_r_yy - rho_lin)
    width_lo = float(ceil_lo - rho_lin)
    width_hi = float(ceil_hi - rho_lin)

    # Stage-0 internal gate (plan §4): ρ_lin within the √(r_yy) CI -> ceiling-limited.
    if ceil_lo <= rho_lin <= ceil_hi:
        verdict = "ceiling-limited"
    elif rho_lin < ceil_lo:
        verdict = "headroom"
    else:
        verdict = "ceiling-limited"  # ρ_lin above the ceiling CI -> no headroom to test

    bayes_beta = dc.bayes_error_ceiling(rates)

    return {
        "behavior": behavior,
        "genre": genre,
        "n_contexts": len(present),
        "var_c_e0": var_c,
        "low_dynamic_range": bool(var_c < 1e-4),
        "rho_lin": float(rho_lin),
        "estimator_kind": estimator_kind,
        "over_rollouts_available": over_rollouts_available,
        "over_rollouts_unavailable_reason": (
            None
            if over_rollouts_available
            else "per-rollout labels not persisted in E0_expression.json (per-probe "
            "rates only); over-probes split-half is the disk-supported estimator"
        ),
        "r_yy_split_half": None if split_half is None else float(split_half),
        "r_yy_binomial": float(binom),
        "sqrt_r_yy_split_half": sqrt_split,
        "sqrt_r_yy_binomial": sqrt_binom,
        "sqrt_r_yy_cv_matched_mean": float(cv_mean),
        "sqrt_r_yy_honest_judge_folded": None
        if judge_honest_sqrt is None
        else float(judge_honest_sqrt),
        "sqrt_r_yy_headline": float(sqrt_r_yy),
        "estimators_disagree": bool(disagree),
        "sqrt_r_yy_ci": [float(ceil_lo), float(ceil_hi)],
        "ceiling_ci_kind": "cv_matched_loco_fold",
        "ridge_join_integrity": join,
        "bracket": [float(rho_lin), float(sqrt_r_yy)],
        "bracket_width": bracket_width,
        "bracket_width_ci": [width_lo, width_hi],
        "gate_verdict": verdict,
        "bayes_error_beta": float(bayes_beta),
        "m_cell_median": float(np.median(m_cell)),
    }


def _load_judge_honest_sqrt(judge_path: Path | None) -> dict[tuple[str, str], float]:
    """Read ``sqrt_r_yy_honest`` per (genre, behavior) from stage0_judge_variance.json.

    BLOCKER stage0-reliability-not-registered(b): the judge-rerun output is the
    judge-folded honest ceiling the bracket/gate must headline on. Returns a
    ``{(genre, behavior): sqrt_r_yy_honest}`` map (empty when the judge file is absent
    — then the bracket falls back to the split-half/binomial ceiling, flagged in JSON).
    """
    out: dict[tuple[str, str], float] = {}
    if judge_path is None or not judge_path.exists():
        return out
    jv = json.loads(judge_path.read_text()).get("judge_variance", {})
    for genre, cells in jv.items():
        for behavior, decomp in (cells or {}).items():
            val = decomp.get("sqrt_r_yy_honest")
            if val is not None:
                out[(genre, behavior)] = float(val)
    return out


def run(
    *,
    behaviors: list[str],
    genres: list[str],
    n_split_seeds: int,
    n_boot: int,
    seed: int,
    judge_variance_path: Path | None = None,
    join_tol: float = 0.05,
) -> dict:
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    provenance: dict[str, dict] = {}
    noise_floor = {}
    agg_path = EVAL_DIR / "aggregate.json"
    if agg_path.exists():
        agg = json.loads(agg_path.read_text())
        noise_floor = agg.get("noise_floor", {})

    judge_honest = _load_judge_honest_sqrt(judge_variance_path)
    join_failures: list[dict] = []

    for genre in genres:
        prov = dc.snapshot_inputs(genre, repo_root=PROJECT_ROOT)
        provenance[genre] = prov
        gi = dc.load_inputs(genre, repo_root=PROJECT_ROOT)
        e0 = gi.E0_per_behavior
        ctx_ids = gi.context_ids
        # stack v0 once per genre for the step-0c ridge re-fit (per-behavior layer below)
        v0_all = dc.stack_v0(gi.v0_dict, recipe="last")  # (n, n_layers, d)
        for behavior in behaviors:
            rho_lin = gi.lin_rho_per_behavior.get(behavior)
            if rho_lin is None:
                rho_lin = dc.load_rho_lin(behavior, genre, eval_dir=EVAL_DIR)
            layer = dc.load_a33_layer(behavior, genre, eval_dir=EVAL_DIR)
            # align v0 rows to the contexts E0 actually has for this behavior
            present = [c for c in ctx_ids if c in e0 and behavior in e0[c]]
            v0_layer = v0_all[: len(present), layer, :]
            entry = compute_bracket(
                behavior,
                genre,
                e0,
                ctx_ids,
                rho_lin=rho_lin,
                rng=rng,
                n_split_seeds=n_split_seeds,
                n_boot=n_boot,
                v0_layer=v0_layer,
                layer=layer,
                join_tol=join_tol,
                judge_honest_sqrt=judge_honest.get((genre, behavior)),
            )
            ji = entry.get("ridge_join_integrity")
            if ji is not None and not ji["join_ok"]:
                join_failures.append(
                    {
                        "genre": genre,
                        "behavior": behavior,
                        **{k: ji[k] for k in ("refit_rho", "persisted_rho", "delta", "tol")},
                    }
                )
            # 2-method reliability agreement check against #658 noise_floor (§4 step 6)
            pb = noise_floor.get("per_behavior_p95", {}) or noise_floor.get("per_behavior", {})
            nf_val = pb.get(behavior) if isinstance(pb, dict) else None
            entry["noise_floor_658"] = None if nf_val is None else float(nf_val)
            if nf_val is not None:
                lo, hi = entry["sqrt_r_yy_ci"]
                entry["noise_floor_agreement"] = bool(lo <= float(nf_val) <= hi)
            results.append(entry)

    # BLOCKER ridge-refit-missing: a join-integrity failure for EITHER genre means the
    # inputs are mis-joined (a wrong/swapped tensor) — RAISE (fail-loud), never silently
    # interpret a bracket read against a mis-reproduced ρ_lin (plan §4 Stage-0 step 0c).
    if join_failures:
        raise RuntimeError(
            "Stage-0 step-0c ridge join-integrity FAILED (delta > tol) for "
            f"{join_failures} — the re-fit LOCO-CV ridge does not reproduce #658's "
            "persisted a33 lin_rho, so v0/E0 are mis-joined for that genre. Refusing to "
            "interpret the bracket against a mis-reproduced ρ_lin (join-integrity REVISE)."
        )

    return {
        "task": "issue_742",
        "stage": "stage0_brackets",
        "primary_dv": "bracket_width = sqrt(r_yy) - rho_lin",
        "headline_note": (
            "disattenuated rho_lin/sqrt(r_yy) is NEVER reported as a headline "
            "(unstable at n=50, Storrs 2020); the bracket width IS the headline"
        ),
        "brackets": results,
        "input_provenance": provenance,
        "config": {
            "behaviors": behaviors,
            "genres": genres,
            "n_split_seeds": n_split_seeds,
            "n_boot": n_boot,
            "seed": seed,
            "disagree_threshold": DISAGREE_THRESHOLD,
            "join_tol": join_tol,
            "judge_variance_source": (
                str(judge_variance_path)
                if judge_variance_path is not None and judge_variance_path.exists()
                else None
            ),
            "ceiling_ci_kind": "cv_matched_loco_fold",
        },
        "metadata": reproducibility_metadata({"script": "issue742_reliability"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #742 Stage 0: reliability + bracket.")
    parser.add_argument(
        "--behaviors", default=",".join(dc.READOUT_BEHAVIORS), help="comma-separated"
    )
    parser.add_argument("--genres", default="betley,ultrachat", help="comma-separated")
    parser.add_argument("--n-split-seeds", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=742)
    parser.add_argument("--join-tol", type=float, default=0.05, help="step-0c join-integrity tol")
    parser.add_argument(
        "--judge-variance",
        type=Path,
        default=OUT_DIR / "stage0_judge_variance.json",
        help="stage0_judge_variance.json to fold the judge term into the honest ceiling",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="single behavior×genre, tiny seeds/boot (plan §4 smoke parity)",
    )
    args = parser.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    n_split_seeds = args.n_split_seeds
    n_boot = args.n_boot
    judge_variance_path = args.judge_variance
    if args.smoke:
        behaviors = behaviors[:1]
        genres = genres[:1]
        n_split_seeds = 10
        n_boot = 50
        judge_variance_path = OUT_DIR / "stage0_judge_variance_smoke.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        behaviors=behaviors,
        genres=genres,
        n_split_seeds=n_split_seeds,
        n_boot=n_boot,
        seed=args.seed,
        judge_variance_path=judge_variance_path,
        join_tol=args.join_tol,
    )
    out_path = args.out_dir / (
        "stage0_brackets_smoke.json" if args.smoke else "stage0_brackets.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage0_brackets] wrote {out_path} ({len(result['brackets'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
