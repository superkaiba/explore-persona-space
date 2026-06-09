"""Issue #464 ``minimal_content`` follow-up — analysis + headline stats.

Inputs (read-only):
  eval_results/issue_464/minimal_content/cross_eval/per_cell/
      {cell}__{e_eval}__marker_{persona}.json
      (2 minimal arms x 3 seeds) x 5 e_eval x 2 markers = 60 files
  eval_results/issue_464/minimal_content/logit_capture/per_cell/   (four-float HF capture)
  eval_results/issue_464/analysis.json                             (parent co-resident L values)

Output:
  eval_results/issue_464/minimal_content/analysis.json

Headline statistic (mirrors the parent's symmetric-pair construction):
  Per seed, per minimal arm:
    L_arm = mean over (persona ∈ {pirate, villain}) x
            (e_wrong ∈ {system_minimal_<other>, role_bare_<other>}) of
            raw g_logprob(marker_persona, e_wrong)
            — the symmetric wrong-encoding cell set over the two MINIMAL
            families (default_assistant EXCLUDED, per parent MF-A).
    d_seed_minimal = L_system_minimal - L_role_bare   (>0 ⇒ role_bare leaks less)

  PASS shape mirrors the parent's H2 verdict: mean(d) ≥ 1.0 nat AND 95%
  paired-bootstrap CI (10k resamples) > 0 AND all 3 per-seed d > 0,
  gated on H1 (diagonal own-encoding log P ≥ -1 nat per cell) and on the
  parent's dynamic-range gate (per-arm leakage sd > 0.5 nat).

Parent join: tables the new arms' per-seed L against the parent
co-resident arms (system_plain / system_padded / role) read from the
parent's committed analysis.json, plus paired per-seed descriptive
deltas (system_minimal vs system_plain, role_bare vs role).

CLI:
    uv run python scripts/i464_min_analyze.py
    uv run python scripts/i464_min_analyze.py --allow-partial   # smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from explore_persona_space.experiments import i464_encodings as enc

load_dotenv()

logger = logging.getLogger("i464.min_analyze")

PER_CELL_DIR = Path("eval_results/issue_464/minimal_content/cross_eval/per_cell")
LOGIT_CAPTURE_DIR = Path("eval_results/issue_464/minimal_content/logit_capture/per_cell")
PARENT_ANALYSIS_PATH = Path("eval_results/issue_464/analysis.json")
OUT_PATH = Path("eval_results/issue_464/minimal_content/analysis.json")

HEADLINE_THRESHOLD = 1.0  # nats (mean(d) ≥ this) — parent H2 convention
H1_ELICITATION_THRESHOLD = -1.0  # nats: own-encoding log P must clear this
DYNAMIC_RANGE_THRESHOLD = 0.5  # sd > this on leakage cells per arm — parent gate
N_BOOTSTRAP = 10000
SEEDS = (42, 137, 1337)
MIN_SEEDS = 3
PARENT_ARMS = ("system_plain", "system_padded", "role")


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_per_cell(cell: str, e_eval: str, marker_persona: str) -> dict | None:
    """Read one per-cell JSON or return None if missing."""
    p = PER_CELL_DIR / f"{cell}__{e_eval}__marker_{marker_persona}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _symmetric_leakage(arm: str, seed: int) -> tuple[float, list[float]]:
    """Return (L_arm_seed, raw per-cell log-probs) over the minimal symmetric set.

    For each persona, the two WRONG-persona encodings across BOTH minimal
    families: marker=pirate probed under {system_minimal_villain,
    role_bare_villain}, and symmetrically for villain — 4 headline cells
    per (arm, seed), exactly the parent's construction with the minimal
    families substituted for the parent's two families.
    """
    cell_label = f"{arm}_seed{seed}"
    raw: list[float] = []
    for persona in enc.PERSONAS:
        other = "villain" if persona == "pirate" else "pirate"
        for e_wrong in (f"system_minimal_{other}", f"role_bare_{other}"):
            payload = _load_per_cell(cell_label, e_wrong, persona)
            if payload is None:
                raise FileNotFoundError(
                    f"min analyze: missing per-cell JSON {cell_label}/{e_wrong}/marker_{persona}"
                )
            raw.append(float(payload["g_logprob"]))
    if not raw:
        raise RuntimeError(f"symmetric leakage cells empty for {cell_label}")
    return float(np.mean(raw)), raw


def _own_eval_encoding_for(arm: str, persona: str) -> str:
    """Diagonal (own-encoding) eval-encoding name for a minimal arm."""
    if arm not in enc.MINIMAL_ARMS:
        raise ValueError(f"not a minimal arm: {arm!r}")
    return f"{arm}_{persona}"


def _paired_bootstrap_ci(
    d_per_seed: list[float], n_boot: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap CI over the per-seed d list. Returns (mean, ci_lo, ci_hi)."""
    arr = np.array(d_per_seed, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(42)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def _verdict(name: str, d_per_seed: list[float], mean: float, lo: float, hi: float) -> dict:
    """Pack a single-comparison verdict (parent H2 shape)."""
    all_positive = all(d > 0 for d in d_per_seed)
    ci_excludes_zero = lo > 0
    threshold_met = mean >= HEADLINE_THRESHOLD
    passed = all_positive and ci_excludes_zero and threshold_met
    reasons: list[str] = []
    if not threshold_met:
        reasons.append(f"mean(d_{name})={mean:.3f} < {HEADLINE_THRESHOLD}")
    if not ci_excludes_zero:
        reasons.append(f"95% CI [{lo:.3f}, {hi:.3f}] overlaps zero")
    if not all_positive:
        reasons.append(f"per-seed d signs not all positive: {d_per_seed}")
    return {
        "d_per_seed": d_per_seed,
        "mean": mean,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "all_seeds_positive": all_positive,
        "ci_excludes_zero": ci_excludes_zero,
        "mean_threshold": HEADLINE_THRESHOLD,
        "threshold_met": threshold_met,
        "pass": passed,
        "fail_reasons": reasons,
    }


def _descriptive_delta(d_per_seed: list[float]) -> dict:
    """Mean + bootstrap CI (when n >= MIN_SEEDS) for a descriptive per-seed delta."""
    out: dict = {"d_per_seed": d_per_seed, "mean": float(np.mean(d_per_seed))}
    if len(d_per_seed) >= MIN_SEEDS:
        _, lo, hi = _paired_bootstrap_ci(d_per_seed, N_BOOTSTRAP)
        out.update({"ci_lo_95": lo, "ci_hi_95": hi})
    return out


def _load_parent_L() -> dict[str, dict[str, float]]:
    """Read the parent co-resident analysis.json L_per_arm_per_seed table."""
    if not PARENT_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"parent analysis missing at {PARENT_ANALYSIS_PATH} — the minimal_content "
            "comparison table requires the parent's committed L_per_arm_per_seed."
        )
    payload = json.loads(PARENT_ANALYSIS_PATH.read_text())
    table = payload.get("L_per_arm_per_seed")
    if not table:
        raise AssertionError(f"{PARENT_ANALYSIS_PATH} has no L_per_arm_per_seed")
    missing = [a for a in PARENT_ARMS if a not in table]
    if missing:
        raise AssertionError(f"parent L_per_arm_per_seed missing arms: {missing}")
    return {a: {str(s): float(v) for s, v in table[a].items()} for a in PARENT_ARMS}


def _summarize_logit_capture(allow_partial: bool) -> dict:
    """Aggregate the four-float HF capture per (cell, e_eval, marker).

    Per trained-side capture file, surface the trained-base mean deltas in
    all three mandated spaces: Δlog P (behavioral), Δz_marker + EOS margin
    Δ(z_marker - z_eos) (mechanistic). Δlog P vs Δz_marker divergence per
    cell is the saturation signature — reported, never "fixed".
    """
    summary: dict[str, dict] = {}
    expected = 0
    found = 0
    for arm in enc.MINIMAL_ARMS:
        for seed in SEEDS:
            cell = f"{arm}_seed{seed}"
            for e_eval in enc.MINIMAL_EVAL_ENCODINGS:
                for marker_persona in enc.PERSONAS:
                    expected += 1
                    p = LOGIT_CAPTURE_DIR / f"{cell}__{e_eval}__marker_{marker_persona}.json"
                    if not p.exists() or p.stat().st_size == 0:
                        if allow_partial:
                            continue
                        raise FileNotFoundError(
                            f"min analyze: four-float capture missing at {p}; run "
                            "scripts/i464_min_capture_logits.py first (or --allow-partial)."
                        )
                    payload = json.loads(p.read_text())
                    found += 1
                    summary[f"{cell}__{e_eval}__marker_{marker_persona}"] = {
                        "delta_logp_mean": payload["delta_mean"]["logp"],
                        "delta_z_marker_mean": payload["delta_mean"]["z_marker"],
                        "delta_eos_margin_mean": payload["delta_mean"]["eos_margin"],
                        # |Δz| - |ΔlogP| grows where softmax compression eats
                        # the push — large positive gap flags saturation.
                        "logp_vs_logit_gap": (
                            abs(payload["delta_mean"]["z_marker"])
                            - abs(payload["delta_mean"]["logp"])
                        ),
                    }
    return {"n_expected": expected, "n_found": found, "per_cell": summary}


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - linear gates mirror the parent phase5 shape
    """Entry point for the minimal_content analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Seeds to aggregate. Default = canonical (42, 137, 1337).",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="If set, skip missing per-cell files (smoke mode); else FAIL LOUD.",
    )
    args = ap.parse_args(argv)

    L_per_arm_per_seed: dict[str, dict[int, float]] = {arm: {} for arm in enc.MINIMAL_ARMS}
    raw_per_cell: dict[str, dict[int, list[float]]] = {arm: {} for arm in enc.MINIMAL_ARMS}
    h1_per_cell_logp: dict[str, float] = {}
    h1_per_cell_pass: dict[str, bool] = {}
    missing: list[str] = []

    for seed in args.seeds:
        for arm in enc.MINIMAL_ARMS:
            try:
                L, raw = _symmetric_leakage(arm, seed)
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("leakage (partial): %s", e)
                    missing.append(str(e))
                    continue
                raise
            L_per_arm_per_seed[arm][seed] = L
            raw_per_cell[arm][seed] = raw

            # H1 diagonal elicitation gate (own-encoding log P ≥ -1 nat).
            for persona in enc.PERSONAS:
                e_own = _own_eval_encoding_for(arm, persona)
                payload = _load_per_cell(f"{arm}_seed{seed}", e_own, persona)
                if payload is None:
                    msg = f"H1: missing diagonal cell {arm}_seed{seed}/{e_own}/marker_{persona}"
                    if args.allow_partial:
                        logger.warning("%s", msg)
                        missing.append(msg)
                        continue
                    raise FileNotFoundError(msg)
                key = f"{arm}_seed{seed}/{e_own}/marker_{persona}"
                lp = float(payload["g_logprob"])
                h1_per_cell_logp[key] = lp
                h1_per_cell_pass[key] = lp >= H1_ELICITATION_THRESHOLD

    if missing and not args.allow_partial:
        raise RuntimeError(f"min analyze: {len(missing)} missing per-cell JSONs")

    complete_seeds = sorted(
        set(L_per_arm_per_seed["system_minimal"]) & set(L_per_arm_per_seed["role_bare"])
    )
    n_h1_expected = len(complete_seeds) * len(enc.MINIMAL_ARMS) * len(enc.PERSONAS)
    h1_complete = len(h1_per_cell_pass) >= n_h1_expected
    h1_overall_pass = h1_complete and len(h1_per_cell_pass) > 0 and all(h1_per_cell_pass.values())

    d_minimal = [
        L_per_arm_per_seed["system_minimal"][s] - L_per_arm_per_seed["role_bare"][s]
        for s in complete_seeds
    ]

    # Dynamic-range gate (parent convention, over the minimal arms).
    dr_gate: dict[str, dict] = {}
    for arm in enc.MINIMAL_ARMS:
        all_raw: list[float] = []
        for seed_raw in raw_per_cell.get(arm, {}).values():
            all_raw.extend(seed_raw)
        if all_raw:
            sd = statistics.pstdev(all_raw)
            dr_gate[arm] = {
                "sd": sd,
                "n_observations": len(all_raw),
                "above_threshold": sd > DYNAMIC_RANGE_THRESHOLD,
            }
        else:
            dr_gate[arm] = {"sd": None, "n_observations": 0, "above_threshold": False}
    dynamic_range_ok = all(v["above_threshold"] for v in dr_gate.values())

    headline: dict
    if len(complete_seeds) < MIN_SEEDS:
        headline_status = "inconclusive_descriptive_only"
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "min_seeds_required": MIN_SEEDS,
            "d_seed_minimal_descriptive": d_minimal,
            "pass": False,
        }
    else:
        m, lo, hi = _paired_bootstrap_ci(d_minimal, N_BOOTSTRAP)
        verdict = _verdict("minimal", d_minimal, m, lo, hi)
        passed = verdict["pass"] and h1_overall_pass
        headline_status = "ok" if passed else "fail"
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "complete_seeds": complete_seeds,
            "d_seed_minimal": verdict,
            "h1_required": True,
            "h1_overall_pass": h1_overall_pass,
            "n_bootstrap": N_BOOTSTRAP,
            "pass": passed,
        }
    if not dynamic_range_ok and headline_status not in ("inconclusive_descriptive_only",):
        failing_arms = [a for a, v in dr_gate.items() if not v.get("above_threshold")]
        headline_status = "inconclusive_dynamic_range_failed"
        headline["status"] = headline_status
        headline["pass"] = False
        headline["dynamic_range_failed_arms"] = failing_arms
        headline["reason"] = (
            f"Dynamic-range gate failed: arms with sd <= {DYNAMIC_RANGE_THRESHOLD}: "
            f"{failing_arms}. Saturated regime — leakage log-prob comparisons are "
            "rank-shuffles on a ceiling, not informative segmentation."
        )

    # ── Parent join: minimal arms vs parent co-resident arms ────────────
    parent_L = _load_parent_L()
    combined_table: dict[str, dict[str, float]] = {
        **{a: parent_L[a] for a in PARENT_ARMS},
        **{
            arm: {str(s): float(v) for s, v in L_per_arm_per_seed[arm].items()}
            for arm in enc.MINIMAL_ARMS
        },
    }
    cross_run_deltas: dict[str, dict] = {}
    for new_arm, parent_arm in (("system_minimal", "system_plain"), ("role_bare", "role")):
        paired_seeds = [s for s in complete_seeds if str(s) in parent_L[parent_arm]]
        if paired_seeds:
            d = [
                parent_L[parent_arm][str(s)] - L_per_arm_per_seed[new_arm][s] for s in paired_seeds
            ]
            cross_run_deltas[f"{parent_arm}_minus_{new_arm}"] = {
                **_descriptive_delta(d),
                "paired_seeds": paired_seeds,
                "note": (
                    "CROSS-RUN descriptive delta (parent LoRAs vs minimal LoRAs, paired "
                    "by seed). Within-run d_seed_minimal is the headline; this table is "
                    "context only."
                ),
            }

    logit_capture = _summarize_logit_capture(args.allow_partial)

    payload = {
        "schema_version": "i464_minimal_content_analysis_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seeds": args.seeds,
        "L_per_arm_per_seed": {
            arm: {str(s): v for s, v in d.items()} for arm, d in L_per_arm_per_seed.items()
        },
        "complete_seeds": complete_seeds,
        "h1_elicitation": {
            "threshold_nats": H1_ELICITATION_THRESHOLD,
            "per_cell_logp": h1_per_cell_logp,
            "per_cell_pass": h1_per_cell_pass,
            "overall_pass": h1_overall_pass,
            "complete": h1_complete,
            "n_cells": len(h1_per_cell_pass),
        },
        "headline": headline,
        "headline_status": headline_status,
        "dynamic_range_gate": {
            "threshold": DYNAMIC_RANGE_THRESHOLD,
            "per_arm": dr_gate,
            "ok": dynamic_range_ok,
        },
        "raw_per_cell": {arm: {str(s): v for s, v in d.items()} for arm, d in raw_per_cell.items()},
        "parent_comparison": {
            "source": str(PARENT_ANALYSIS_PATH),
            "L_per_arm_per_seed_combined": combined_table,
            "cross_run_descriptive_deltas": cross_run_deltas,
        },
        "logit_capture_summary": logit_capture,
        "n_missing_per_cell": len(missing),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(
        "minimal_content analysis done -> %s (status=%s complete_seeds=%d H1=%s)",
        OUT_PATH,
        headline_status,
        len(complete_seeds),
        h1_overall_pass,
    )
    if headline_status == "ok":
        logger.info(
            "HEADLINE PASS: d_minimal mean=%.3f CI=[%.3f, %.3f]",
            headline["d_seed_minimal"]["mean"],
            headline["d_seed_minimal"]["ci_lo_95"],
            headline["d_seed_minimal"]["ci_hi_95"],
        )
    if not h1_overall_pass and h1_per_cell_pass:
        failing = [k for k, v in h1_per_cell_pass.items() if not v]
        logger.warning(
            "H1 elicitation FAILED on %d of %d cells (own log P < %.1f nat): %s",
            len(failing),
            len(h1_per_cell_pass),
            H1_ELICITATION_THRESHOLD,
            failing[:5],
        )
    if not dynamic_range_ok:
        logger.warning(
            "Dynamic-range gate FAILED — at least one minimal arm has sd <= %.2f; "
            "saturation regime, headline reads as inconclusive.",
            DYNAMIC_RANGE_THRESHOLD,
        )


if __name__ == "__main__":
    main()
