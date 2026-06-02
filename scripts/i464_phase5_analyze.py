"""Phase 5 — analysis + headline stats for issue #464.

Issue #464 plan v2 §4.1 Phase 5 + §6.

Inputs (read-only):
  eval_results/issue_464/cross_eval/per_cell/<cell>__<e_eval>__marker_<persona>.json
      (3 arms x 3 seeds) x 5 e_eval x 2 markers = 90 files

Outputs:
  eval_results/issue_464/analysis.json — full per-arm stats + headline
      d_seed_plain / d_seed_padded with paired 10k-sample bootstrap CIs
      + dynamic-range gate flag + per-cell matrices.

Headline statistic (plan §3):
  Per seed:
    L_arm = mean over (persona ∈ {pirate, villain}) x
                  (e_wrong ∈ {wrong_persona_system, wrong_persona_role}) of
            raw g_logprob(marker_persona, e_wrong)
            where e_wrong is the OTHER persona's same-family encoding.
            (MF-A symmetric cell set: default_assistant EXCLUDED from headline.)

    d_seed_plain  = L_system_plain  - L_role     (>0 ⇒ role leaks less)
    d_seed_padded = L_system_padded - L_role

  H2 PASS:
    mean(d_plain)  ≥ 1.0 nat AND 95% CI > 0 AND all 3 per-seed d > 0
    mean(d_padded) ≥ 1.0 nat AND 95% CI > 0 AND all 3 per-seed d > 0

CLI:
    uv run python scripts/i464_phase5_analyze.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from explore_persona_space.experiments import i464_encodings as enc

load_dotenv()

logger = logging.getLogger("i464.phase5")

PER_CELL_DIR = Path("eval_results/issue_464/cross_eval/per_cell")
OUT_PATH = Path("eval_results/issue_464/analysis.json")
H2_HEADLINE_THRESHOLD = 1.0  # nats (mean(d) ≥ this)
DYNAMIC_RANGE_THRESHOLD = 0.5  # sd > this on leakage cells per arm
N_BOOTSTRAP = 10000
SEEDS = (42, 137, 1337)


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        import os

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
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


def _symmetric_leakage(arm: enc.Arm, seed: int) -> tuple[float, list[float]]:
    """Return (L_arm_seed, raw_logprobs_per_cell) for the symmetric headline cell set.

    The symmetric set (MF-A): for each persona_i, the two WRONG-persona
    encodings of that persona's family — wrong_persona_system AND
    wrong_persona_role. So for persona=pirate, marker_id=pirate_marker,
    leakage cells are e_eval={system_villain, role_villain}.

    Returns:
        Mean leakage log-prob across the 4 headline cells (2 personas x
        2 wrong-encoding families) AND the per-cell raw log-prob list.
    """
    cell_label = f"{arm}_seed{seed}"
    raw: list[float] = []
    for persona in enc.PERSONAS:
        other_persona = "villain" if persona == "pirate" else "pirate"
        wrong_encodings = [f"system_{other_persona}", f"role_{other_persona}"]
        for e_wrong in wrong_encodings:
            payload = _load_per_cell(cell_label, e_wrong, persona)
            if payload is None:
                raise FileNotFoundError(
                    f"Phase 5: missing per-cell JSON for {cell_label}/{e_wrong}/marker_{persona}"
                )
            raw.append(payload["g_logprob"])
    if not raw:
        raise RuntimeError(f"symmetric leakage cells empty for {cell_label}")
    return float(np.mean(raw)), raw


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


def _h2_verdict(name: str, d_per_seed: list[float], mean: float, lo: float, hi: float) -> dict:
    """Pack a single-comparison H2 verdict (PASS / FAIL with reasons)."""
    all_positive = all(d > 0 for d in d_per_seed)
    ci_excludes_zero = lo > 0
    threshold_met = mean >= H2_HEADLINE_THRESHOLD
    passed = all_positive and ci_excludes_zero and threshold_met
    reasons: list[str] = []
    if not threshold_met:
        reasons.append(f"mean(d_{name})={mean:.3f} < {H2_HEADLINE_THRESHOLD}")
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
        "mean_threshold": H2_HEADLINE_THRESHOLD,
        "threshold_met": threshold_met,
        "pass": passed,
        "fail_reasons": reasons,
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for Phase 5 analysis."""
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

    # Per-arm symmetric leakage per seed.
    L_per_arm_per_seed: dict[str, list[float]] = {arm: [] for arm in enc.ARMS}
    raw_per_cell: dict[str, dict[int, list[float]]] = {arm: {} for arm in enc.ARMS}
    missing: list[str] = []
    for seed in args.seeds:
        for arm in enc.ARMS:
            try:
                L, raw = _symmetric_leakage(arm, seed)  # type: ignore[arg-type]
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("Phase 5 (partial): %s", e)
                    missing.append(str(e))
                    continue
                raise
            L_per_arm_per_seed[arm].append(L)
            raw_per_cell[arm][seed] = raw

    if missing and not args.allow_partial:
        raise RuntimeError(f"Phase 5: {len(missing)} missing per-cell JSONs")

    # Headline d_seed_plain and d_seed_padded.
    L_plain = L_per_arm_per_seed["system_plain"]
    L_padded = L_per_arm_per_seed["system_padded"]
    L_role = L_per_arm_per_seed["role"]

    d_plain: list[float] = []
    d_padded: list[float] = []
    if len(L_role) == len(L_plain) == len(L_padded) > 0:
        for lp, lpad, lr in zip(L_plain, L_padded, L_role, strict=True):
            d_plain.append(lp - lr)
            d_padded.append(lpad - lr)

    headline: dict | None = None
    if d_plain and d_padded:
        m_p, lo_p, hi_p = _paired_bootstrap_ci(d_plain, N_BOOTSTRAP)
        m_pad, lo_pad, hi_pad = _paired_bootstrap_ci(d_padded, N_BOOTSTRAP)
        verdict_plain = _h2_verdict("plain", d_plain, m_p, lo_p, hi_p)
        verdict_padded = _h2_verdict("padded", d_padded, m_pad, lo_pad, hi_pad)
        headline = {
            "d_seed_plain": verdict_plain,
            "d_seed_padded": verdict_padded,
            "h2_full_pass": verdict_plain["pass"] and verdict_padded["pass"],
            "h2_partial": verdict_plain["pass"] and not verdict_padded["pass"],
            "n_bootstrap": N_BOOTSTRAP,
        }

    # Dynamic-range gate: sd of raw g_logprob across the 4 symmetric
    # leakage cells should exceed 0.5 in EACH arm; otherwise the regime
    # is saturated and the headline reads as "rank-shuffle on saturated
    # values".
    dr_gate: dict[str, dict] = {}
    for arm in enc.ARMS:
        all_raw: list[float] = []
        for seed_raw in raw_per_cell[arm].values():
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

    payload = {
        "schema_version": "i464_phase5_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seeds": args.seeds,
        "L_per_arm_per_seed": L_per_arm_per_seed,
        "headline": headline,
        "dynamic_range_gate": {
            "threshold": DYNAMIC_RANGE_THRESHOLD,
            "per_arm": dr_gate,
            "ok": dynamic_range_ok,
        },
        "raw_per_cell": raw_per_cell,
        "n_missing_per_cell": len(missing),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 5 done -> %s", OUT_PATH)
    if headline is not None:
        logger.info(
            "Headline d_plain mean=%.3f CI=[%.3f, %.3f] pass=%s; "
            "d_padded mean=%.3f CI=[%.3f, %.3f] pass=%s; H2 full pass=%s",
            headline["d_seed_plain"]["mean"],
            headline["d_seed_plain"]["ci_lo_95"],
            headline["d_seed_plain"]["ci_hi_95"],
            headline["d_seed_plain"]["pass"],
            headline["d_seed_padded"]["mean"],
            headline["d_seed_padded"]["ci_lo_95"],
            headline["d_seed_padded"]["ci_hi_95"],
            headline["d_seed_padded"]["pass"],
            headline["h2_full_pass"],
        )
    if not dynamic_range_ok:
        logger.warning(
            "Dynamic-range gate FAILED — at least one arm has sd ≤ %.2f; "
            "saturation regime, headline reads as inconclusive.",
            DYNAMIC_RANGE_THRESHOLD,
        )


if __name__ == "__main__":
    main()
