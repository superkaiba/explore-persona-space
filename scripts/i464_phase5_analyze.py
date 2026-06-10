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
ONPOLICY_VALIDATION_PATH = Path("eval_results/issue_464/onpolicy_validation.json")
H2_HEADLINE_THRESHOLD = 1.0  # nats (mean(d) ≥ this)
H1_ELICITATION_THRESHOLD = -1.0  # nats: own-persona log P must clear this
DYNAMIC_RANGE_THRESHOLD = 0.5  # sd > this on leakage cells per arm
N_BOOTSTRAP = 10000
SEEDS = (42, 137, 1337)
# Round-2 fix (review blocker #3 — MF-H seed-floor): H2 PASS requires AT
# LEAST 3 COMPLETE paired seeds. Fewer than 3 → "inconclusive_descriptive_only".
H2_MIN_SEEDS = 3


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


def _own_eval_encoding_for(arm: enc.Arm, persona: enc.Persona) -> str:
    """Return the eval-encoding name matching ``arm``'s own training-time form.

    For an arm trained with persona X declared in MECHANISM M, the
    "own encoding" at eval time is the eval-encoding name that puts
    persona X into the SAME mechanism M. This drives the H1 gate
    (does the trained LoRA implant the marker under its own mechanism?),
    the plot's elicitation panels, and the H1 stub-writer.

      system_plain   / system_padded   → system_<persona>
      role                              → role_<persona>
      role_nonsense                     → role_nonsense_<persona>
      role_mismatch                     → role_mismatch_<persona>
    """
    if arm == "role":
        return f"role_{persona}"
    if arm == "role_nonsense":
        return f"role_nonsense_{persona}"
    if arm == "role_mismatch":
        return f"role_mismatch_{persona}"
    # system_plain / system_padded both train with persona in the system
    # prompt; their own-encoding eval probe is the matching system_<persona>.
    return f"system_{persona}"


def _own_persona_elicitation(arm: enc.Arm, seed: int) -> tuple[list[float], list[str]]:
    """Return (own-persona elicitation log-probs, per-cell labels) for one (arm, seed).

    Round-2 fix (review blocker #7 — explicit H1 gate). H1: own-persona
    elicitation raw trained log P must clear ``H1_ELICITATION_THRESHOLD``
    (>= -1 nat) across all 30 own-persona cells (5 arms x 3 seeds x 2
    personas; was 18 before the role_nonsense follow-up arm, 24 after
    role_nonsense, 30 after role_mismatch). Phase 5 reads each cell from
    the per-cell JSON tree and returns the per-cell log-probs so the
    caller can compute the gate verdict.

    "Own encoding" per arm — see ``_own_eval_encoding_for``.

    Raises FileNotFoundError if any required per-cell JSON is missing
    (so the caller can decide allow-partial behavior).
    """
    own_logps: list[float] = []
    labels: list[str] = []
    cell_label = f"{arm}_seed{seed}"
    for persona in enc.PERSONAS:
        e_own = _own_eval_encoding_for(arm, persona)
        payload = _load_per_cell(cell_label, e_own, persona)
        if payload is None:
            raise FileNotFoundError(
                f"Phase 5 H1 gate: missing own-persona cell {cell_label}/{e_own}/marker_{persona}"
            )
        own_logps.append(float(payload["g_logprob"]))
        labels.append(f"{cell_label}/{e_own}/marker_{persona}")
    return own_logps, labels


def _load_onpolicy_validation() -> dict | None:
    """Load Phase 4.5 onpolicy_validation.json if present.

    Returns the payload dict or None if the file is absent (e.g. partial
    runs where Phase 4.5 didn't execute). When present and
    ``switch_headline_to_trained_R`` is true, Phase 5 must refuse the
    canonical-R PASS (review blocker #4 — MF-B(2) switch consumption).
    """
    if not ONPOLICY_VALIDATION_PATH.exists() or ONPOLICY_VALIDATION_PATH.stat().st_size == 0:
        return None
    try:
        payload = json.loads(ONPOLICY_VALIDATION_PATH.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Phase 4.5 validation JSON unreadable: %s", e)
        return None
    if payload.get("schema_version") != "i464_onpolicy_validation_v1":
        logger.warning(
            "Phase 4.5 validation schema_version=%r; ignoring "
            "(expected i464_onpolicy_validation_v1)",
            payload.get("schema_version"),
        )
        return None
    return payload


def _compute_h1_per_cell(
    own_logp_per_arm_per_seed: dict[str, dict[int, list[float]]],
) -> tuple[dict[str, bool], dict[str, float]]:
    """Build per-cell H1 elicitation pass map + per-cell log P dict.

    Each (arm, seed, persona) cell PASSes if its own-persona log P clears
    ``H1_ELICITATION_THRESHOLD``. Cell key is
    ``<arm>_seed<seed>/<e_own>/marker_<persona>`` (mirrors per-cell JSON
    filename). Persona order in the inner ``logps`` list is the iteration
    order of ``enc.PERSONAS`` (set in i464_encodings).
    """
    per_cell_pass: dict[str, bool] = {}
    per_cell_logp: dict[str, float] = {}
    for arm, by_seed in own_logp_per_arm_per_seed.items():
        for seed, logps in by_seed.items():
            for persona_idx, persona in enumerate(enc.PERSONAS):
                e_own = _own_eval_encoding_for(arm, persona)  # type: ignore[arg-type]
                key = f"{arm}_seed{seed}/{e_own}/marker_{persona}"
                lp = float(logps[persona_idx])
                per_cell_logp[key] = lp
                per_cell_pass[key] = lp >= H1_ELICITATION_THRESHOLD
    return per_cell_pass, per_cell_logp


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


def _compute_dynamic_range_gate(
    raw_per_cell: dict[str, dict[int, list[float]]],
) -> tuple[dict[str, dict], bool]:
    """Return (per-arm sd+threshold-pass dict, overall gate ok bool).

    Round-3 fix (review blocker #1): extracted from main() so the gate
    logic is unit-testable without invoking the CLI. Each arm aggregates
    every per-seed raw leakage log-prob; arm PASSes if sd >
    DYNAMIC_RANGE_THRESHOLD. Overall ok = every arm above threshold.
    """
    dr_gate: dict[str, dict] = {}
    for arm in enc.ARMS:
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
    overall_ok = all(v["above_threshold"] for v in dr_gate.values())
    return dr_gate, overall_ok


def _override_headline_on_saturation(
    headline: dict | None,
    headline_status: str,
    dr_gate: dict[str, dict],
    dynamic_range_ok: bool,
) -> tuple[dict | None, str]:
    """Round-3 fix (review blocker #1): override the headline under saturation.

    Saturation regime (`not dynamic_range_ok`) MUST force the headline to a
    terminal inconclusive state — round-2 only logger.warning'd and the
    headline still claimed "ok"/"partial". Override is idempotent for the
    two pre-existing terminal "inconclusive_..." / "blocked_..." states
    (no double-stomping). Returns the (possibly mutated) headline + the
    final headline_status.
    """
    if dynamic_range_ok:
        return headline, headline_status
    # Already in a terminal inconclusive/blocked state — don't stomp.
    if headline_status in ("inconclusive_descriptive_only", "blocked_onpolicy_switch_required"):
        return headline, headline_status
    failing_arms = [a for a, v in dr_gate.items() if not v.get("above_threshold")]
    new_status = "inconclusive_dynamic_range_failed"
    if headline is None:
        headline = {}
    headline["status"] = new_status
    headline["h2_full_pass"] = False
    headline["h2_partial"] = False
    headline["dynamic_range_failed_arms"] = failing_arms
    headline["reason"] = (
        f"Dynamic-range gate failed: arms with sd <= {DYNAMIC_RANGE_THRESHOLD}: "
        f"{failing_arms}. Saturated regime — leakage log-prob comparisons "
        "are rank-shuffles on a ceiling, not informative segmentation."
    )
    return headline, new_status


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - round-2 added H1/MF-B(2)/MF-H gates + per-seed dict + onpolicy-switch consumption push branching above 15
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

    # Per-arm symmetric leakage indexed BY SEED so we can intersect complete
    # seeds before forming paired deltas (round-2 fix for review blocker #3 —
    # the round-1 flat list per arm could break seed pairing under
    # --allow-partial).
    L_per_arm_per_seed: dict[str, dict[int, float]] = {arm: {} for arm in enc.ARMS}
    raw_per_cell: dict[str, dict[int, list[float]]] = {arm: {} for arm in enc.ARMS}
    own_logp_per_arm_per_seed: dict[str, dict[int, list[float]]] = {arm: {} for arm in enc.ARMS}
    own_cell_labels: list[str] = []
    missing: list[str] = []

    for seed in args.seeds:
        for arm in enc.ARMS:
            # Symmetric leakage (MF-A headline cells).
            try:
                L, raw = _symmetric_leakage(arm, seed)  # type: ignore[arg-type]
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("Phase 5 leakage (partial): %s", e)
                    missing.append(str(e))
                    continue
                raise
            L_per_arm_per_seed[arm][seed] = L
            raw_per_cell[arm][seed] = raw

            # Own-persona elicitation (H1 gate input).
            try:
                own_logps, labels = _own_persona_elicitation(arm, seed)  # type: ignore[arg-type]
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("Phase 5 H1 (partial): %s", e)
                    missing.append(str(e))
                else:
                    raise
            else:
                own_logp_per_arm_per_seed[arm][seed] = own_logps
                # Capture labels once (first arm x first seed); they're stable.
                if not own_cell_labels:
                    own_cell_labels = labels

    if missing and not args.allow_partial:
        raise RuntimeError(f"Phase 5: {len(missing)} missing per-cell JSONs")

    # ── H1 elicitation gate (review blocker #7) ─────────────────────────
    # H1 = "trained adapter implants the marker under own-persona encoding".
    # Required because H2 (segmentation) only carries the headline once H1
    # passes -- otherwise a low leakage L_arm could just mean "the LoRA
    # didn't learn the marker at all".
    h1_per_cell_pass, h1_per_cell_logp = _compute_h1_per_cell(own_logp_per_arm_per_seed)

    # ── Headline: paired deltas over COMPLETE seeds only (blocker #3) ───
    # Canonical H2 PASS uses the original 3 arms (system_plain, system_padded,
    # role); the role_nonsense follow-up arm is reported as a descriptive
    # delta alongside (NOT gating H2 PASS). See "Role-nonsense descriptive"
    # section below.
    complete_seeds = sorted(
        set(L_per_arm_per_seed["system_plain"])
        & set(L_per_arm_per_seed["system_padded"])
        & set(L_per_arm_per_seed["role"])
    )

    # Round-3 fix (review blocker #3 — H1 cell-coverage check). Even if every
    # PRESENT H1 cell passes, missing H1 cells for any of the 3 canonical
    # H2 arms x complete_seed mean we don't actually have evidence the
    # trained adapter implants the marker for that combination — calling
    # H1 "PASS" on the subset gates H2 PASS on partial H1 evidence.
    # role_nonsense H1 is checked separately below (descriptive).
    H2_HEADLINE_ARMS: tuple[str, ...] = ("system_plain", "system_padded", "role")
    h1_missing_cells: list[str] = []
    for arm in H2_HEADLINE_ARMS:
        for seed in complete_seeds:
            if seed not in own_logp_per_arm_per_seed[arm]:
                h1_missing_cells.append(f"{arm}_seed{seed}/own-persona")
    h1_complete = not h1_missing_cells
    # h1_overall_pass uses ONLY the H2-headline arms' H1 cells so a missing
    # role_nonsense own-cell can't block the canonical H2 PASS.
    h1_headline_pass_keys = {
        k for k in h1_per_cell_pass if any(k.startswith(f"{arm}_seed") for arm in H2_HEADLINE_ARMS)
    }
    h1_overall_pass = (
        h1_complete
        and len(h1_headline_pass_keys) > 0
        and all(h1_per_cell_pass[k] for k in h1_headline_pass_keys)
    )
    d_plain: list[float] = []
    d_padded: list[float] = []
    for s in complete_seeds:
        d_plain.append(L_per_arm_per_seed["system_plain"][s] - L_per_arm_per_seed["role"][s])
        d_padded.append(L_per_arm_per_seed["system_padded"][s] - L_per_arm_per_seed["role"][s])

    # ── Role-nonsense descriptive deltas (follow-up arm, NOT gating H2) ─
    # The headline question for the role_nonsense follow-up: does
    # role_nonsense's leakage track ``role`` (role-header SLOT/position
    # mechanism alone) or ``system_plain`` (semantic meaning of the role
    # name)? Report two paired-bootstrap deltas:
    #   d_seed_role_nonsense_vs_plain = L_system_plain - L_role_nonsense
    #     (>0: role_nonsense leaks less than system_plain — the slot helps)
    #   d_seed_role_nonsense_vs_role  = L_role         - L_role_nonsense
    #     (≈0: semantics adds nothing on top of the slot
    #     >0 : semantics buys further reduction over the slot alone
    #     <0 : semantics hurts vs the slot alone)
    # Only computed when role_nonsense has the same complete seeds; if not,
    # surface the missing data without blocking H2.
    rn_complete_seeds = sorted(
        set(complete_seeds) & set(L_per_arm_per_seed.get("role_nonsense", {}))
    )
    d_role_nonsense_vs_plain: list[float] = []
    d_role_nonsense_vs_role: list[float] = []
    for s in rn_complete_seeds:
        d_role_nonsense_vs_plain.append(
            L_per_arm_per_seed["system_plain"][s] - L_per_arm_per_seed["role_nonsense"][s]
        )
        d_role_nonsense_vs_role.append(
            L_per_arm_per_seed["role"][s] - L_per_arm_per_seed["role_nonsense"][s]
        )

    # ── Role-mismatch descriptive deltas (follow-up arm, NOT gating H2) ─
    # The headline question for the role_mismatch follow-up: does a real
    # but UNRELATED role-name (e.g. ``baker_assistant`` paired with pirate
    # content + ※) behave like the matched-meaning ``role`` arm, like the
    # meaningless ``role_nonsense`` arm, or somewhere in between? Three
    # paired-bootstrap deltas surface the three candidate mechanisms:
    #   d_seed_role_mismatch_vs_plain    = L_system_plain  - L_role_mismatch
    #     (>0: role_mismatch leaks less than system_plain — the slot helps)
    #   d_seed_role_mismatch_vs_role     = L_role          - L_role_mismatch
    #     (≈0: name-must-match-content does NOT explain role's reduction
    #     <0 : role_mismatch leaks MORE than role — matched semantics helps
    #     >0 : role_mismatch leaks LESS than role — mismatch helps further)
    #   d_seed_role_mismatch_vs_nonsense = L_role_nonsense - L_role_mismatch
    #     (≈0: role-name meaningfulness adds nothing on top of the slot
    #     >0 : real meaning (even mismatched) reduces leakage vs gibberish
    #     <0 : real-but-wrong meaning HURTS vs gibberish — interference)
    # Only computed when role_mismatch has the same complete seeds; if not,
    # surface the missing data without blocking H2.
    rm_complete_seeds = sorted(
        set(complete_seeds) & set(L_per_arm_per_seed.get("role_mismatch", {}))
    )
    d_role_mismatch_vs_plain: list[float] = []
    d_role_mismatch_vs_role: list[float] = []
    d_role_mismatch_vs_nonsense: list[float] = []
    rm_vs_nonsense_complete_seeds = sorted(
        set(rm_complete_seeds) & set(L_per_arm_per_seed.get("role_nonsense", {}))
    )
    for s in rm_complete_seeds:
        d_role_mismatch_vs_plain.append(
            L_per_arm_per_seed["system_plain"][s] - L_per_arm_per_seed["role_mismatch"][s]
        )
        d_role_mismatch_vs_role.append(
            L_per_arm_per_seed["role"][s] - L_per_arm_per_seed["role_mismatch"][s]
        )
    for s in rm_vs_nonsense_complete_seeds:
        d_role_mismatch_vs_nonsense.append(
            L_per_arm_per_seed["role_nonsense"][s] - L_per_arm_per_seed["role_mismatch"][s]
        )

    # ── On-policy validation (review blocker #4) ────────────────────────
    onpolicy = _load_onpolicy_validation()
    onpolicy_switch = bool(onpolicy and onpolicy.get("switch_headline_to_trained_R", False))
    onpolicy_ratio = onpolicy.get("role_over_system_plain_ratio") if onpolicy else None

    headline: dict | None = None
    headline_status: str
    if len(complete_seeds) < H2_MIN_SEEDS:
        headline_status = "inconclusive_descriptive_only"
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "min_seeds_required": H2_MIN_SEEDS,
            "reason": (
                f"only {len(complete_seeds)} complete paired seeds (need >= "
                f"{H2_MIN_SEEDS}); MF-H requires n>=3 minimum for H2 PASS."
            ),
            "d_seed_plain_descriptive": d_plain,
            "d_seed_padded_descriptive": d_padded,
            "h2_full_pass": False,
            "h2_partial": False,
        }
    elif onpolicy_switch:
        # MF-B(2): if Phase 4.5 said role-arm's trained-greedy R diverges
        # > 1.5x from system-plain's, the canonical-R headline is invalid
        # and Phase 4 must be re-run with arm-specific R before any PASS.
        headline_status = "blocked_onpolicy_switch_required"
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "onpolicy_role_over_system_plain_ratio": onpolicy_ratio,
            "reason": (
                f"Phase 4.5 flagged switch_headline_to_trained_R=true "
                f"(role/system_plain edit-distance ratio = {onpolicy_ratio}). "
                "Canonical-R PASS REFUSED per MF-B(2); re-run Phase 4 with "
                "trained-greedy R per arm before reporting H2."
            ),
            "d_seed_plain_descriptive": d_plain,
            "d_seed_padded_descriptive": d_padded,
            "h2_full_pass": False,
            "h2_partial": False,
        }
    else:
        # Healthy path: n>=3 complete seeds AND on-policy switch NOT tripped.
        m_p, lo_p, hi_p = _paired_bootstrap_ci(d_plain, N_BOOTSTRAP)
        m_pad, lo_pad, hi_pad = _paired_bootstrap_ci(d_padded, N_BOOTSTRAP)
        verdict_plain = _h2_verdict("plain", d_plain, m_p, lo_p, hi_p)
        verdict_padded = _h2_verdict("padded", d_padded, m_pad, lo_pad, hi_pad)
        # H1 gate is a HARD prerequisite for the H2 PASS (review blocker #7).
        h2_full = verdict_plain["pass"] and verdict_padded["pass"] and h1_overall_pass
        h2_partial = verdict_plain["pass"] and not verdict_padded["pass"] and h1_overall_pass
        headline_status = "ok" if h2_full else ("partial" if h2_partial else "fail")
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "complete_seeds": complete_seeds,
            "d_seed_plain": verdict_plain,
            "d_seed_padded": verdict_padded,
            "h2_full_pass": h2_full,
            "h2_partial": h2_partial,
            "h1_required_before_h2": True,
            "h1_overall_pass": h1_overall_pass,
            "onpolicy_switch_consumed": True,
            "onpolicy_role_over_system_plain_ratio": onpolicy_ratio,
            "n_bootstrap": N_BOOTSTRAP,
        }
        # role_nonsense descriptive deltas (computed when at least 1
        # paired seed is present; CIs only when n >= H2_MIN_SEEDS).
        if d_role_nonsense_vs_plain:
            rn_plain_descriptive: dict = {
                "d_per_seed": d_role_nonsense_vs_plain,
                "mean": float(np.mean(d_role_nonsense_vs_plain)),
            }
            rn_role_descriptive: dict = {
                "d_per_seed": d_role_nonsense_vs_role,
                "mean": float(np.mean(d_role_nonsense_vs_role)),
            }
            if len(d_role_nonsense_vs_plain) >= H2_MIN_SEEDS:
                _, lo, hi = _paired_bootstrap_ci(d_role_nonsense_vs_plain, N_BOOTSTRAP)
                rn_plain_descriptive.update({"ci_lo_95": lo, "ci_hi_95": hi})
                _, lo2, hi2 = _paired_bootstrap_ci(d_role_nonsense_vs_role, N_BOOTSTRAP)
                rn_role_descriptive.update({"ci_lo_95": lo2, "ci_hi_95": hi2})
            headline["d_seed_role_nonsense_vs_plain_descriptive"] = rn_plain_descriptive
            headline["d_seed_role_nonsense_vs_role_descriptive"] = rn_role_descriptive
            headline["role_nonsense_complete_seeds"] = rn_complete_seeds
        # role_mismatch descriptive deltas (computed when at least 1
        # paired seed is present; CIs only when n >= H2_MIN_SEEDS).
        if d_role_mismatch_vs_plain:
            rm_plain_descriptive: dict = {
                "d_per_seed": d_role_mismatch_vs_plain,
                "mean": float(np.mean(d_role_mismatch_vs_plain)),
            }
            rm_role_descriptive: dict = {
                "d_per_seed": d_role_mismatch_vs_role,
                "mean": float(np.mean(d_role_mismatch_vs_role)),
            }
            if len(d_role_mismatch_vs_plain) >= H2_MIN_SEEDS:
                _, lo, hi = _paired_bootstrap_ci(d_role_mismatch_vs_plain, N_BOOTSTRAP)
                rm_plain_descriptive.update({"ci_lo_95": lo, "ci_hi_95": hi})
                _, lo2, hi2 = _paired_bootstrap_ci(d_role_mismatch_vs_role, N_BOOTSTRAP)
                rm_role_descriptive.update({"ci_lo_95": lo2, "ci_hi_95": hi2})
            headline["d_seed_role_mismatch_vs_plain_descriptive"] = rm_plain_descriptive
            headline["d_seed_role_mismatch_vs_role_descriptive"] = rm_role_descriptive
            headline["role_mismatch_complete_seeds"] = rm_complete_seeds
            if d_role_mismatch_vs_nonsense:
                rm_nonsense_descriptive: dict = {
                    "d_per_seed": d_role_mismatch_vs_nonsense,
                    "mean": float(np.mean(d_role_mismatch_vs_nonsense)),
                }
                if len(d_role_mismatch_vs_nonsense) >= H2_MIN_SEEDS:
                    _, lo3, hi3 = _paired_bootstrap_ci(d_role_mismatch_vs_nonsense, N_BOOTSTRAP)
                    rm_nonsense_descriptive.update({"ci_lo_95": lo3, "ci_hi_95": hi3})
                headline["d_seed_role_mismatch_vs_nonsense_descriptive"] = rm_nonsense_descriptive
                headline["role_mismatch_vs_nonsense_complete_seeds"] = rm_vs_nonsense_complete_seeds

    # Dynamic-range gate: sd of raw g_logprob across the 4 symmetric
    # leakage cells should exceed 0.5 in EACH arm; otherwise the regime
    # is saturated and the headline reads as "rank-shuffle on saturated
    # values". Round-3 fix: the gate now OVERRIDES the headline (round-2
    # only logger.warning'd; headline still claimed "ok"/"partial").
    dr_gate, dynamic_range_ok = _compute_dynamic_range_gate(raw_per_cell)
    headline, headline_status = _override_headline_on_saturation(
        headline, headline_status, dr_gate, dynamic_range_ok
    )

    payload = {
        "schema_version": "i464_phase5_v2",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seeds": args.seeds,
        "L_per_arm_per_seed": {arm: dict(d) for arm, d in L_per_arm_per_seed.items()},
        "complete_seeds": complete_seeds,
        "h2_min_seeds": H2_MIN_SEEDS,
        "h1_elicitation": {
            "threshold_nats": H1_ELICITATION_THRESHOLD,
            "per_cell_logp": h1_per_cell_logp,
            "per_cell_pass": h1_per_cell_pass,
            "overall_pass": h1_overall_pass,
            "n_cells": len(h1_per_cell_pass),
            # Round-3 fix (review blocker #3): H1 cell-coverage check —
            # `complete` is True iff every (arm, complete_seed) has H1 cells
            # present. `missing_cells` enumerates the gaps that forced an
            # "h1_overall_pass=False" verdict under partial inputs.
            "complete": h1_complete,
            "missing_cells": h1_missing_cells,
        },
        "onpolicy_validation": {
            "loaded": onpolicy is not None,
            "switch_headline_to_trained_R": onpolicy_switch,
            "role_over_system_plain_ratio": onpolicy_ratio,
        },
        "headline": headline,
        "headline_status": headline_status,
        "dynamic_range_gate": {
            "threshold": DYNAMIC_RANGE_THRESHOLD,
            "per_arm": dr_gate,
            "ok": dynamic_range_ok,
        },
        "raw_per_cell": raw_per_cell,
        "n_missing_per_cell": len(missing),
        # role_nonsense follow-up: descriptive deltas surfaced top-level
        # (also embedded in headline.* for the healthy path). Always
        # present so consumers (plot, mentor summary) read one place; the
        # inner dict is empty when no role_nonsense seeds are paired.
        "role_nonsense_descriptive": {
            "complete_seeds": rn_complete_seeds,
            "d_seed_role_nonsense_vs_plain": d_role_nonsense_vs_plain,
            "d_seed_role_nonsense_vs_role": d_role_nonsense_vs_role,
        },
        # role_mismatch follow-up: descriptive deltas surfaced top-level
        # (also embedded in headline.* for the healthy path). Always
        # present so consumers (plot, mentor summary) read one place; the
        # inner dict is empty when no role_mismatch seeds are paired.
        "role_mismatch_descriptive": {
            "complete_seeds": rm_complete_seeds,
            "d_seed_role_mismatch_vs_plain": d_role_mismatch_vs_plain,
            "d_seed_role_mismatch_vs_role": d_role_mismatch_vs_role,
            "d_seed_role_mismatch_vs_nonsense": d_role_mismatch_vs_nonsense,
            "vs_nonsense_complete_seeds": rm_vs_nonsense_complete_seeds,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Phase 5 done -> %s (status=%s complete_seeds=%d H1=%s)",
        OUT_PATH,
        headline_status,
        len(complete_seeds),
        h1_overall_pass,
    )
    if headline_status == "ok":
        logger.info(
            "H2 PASS: d_plain mean=%.3f CI=[%.3f, %.3f]; d_padded mean=%.3f CI=[%.3f, %.3f]",
            headline["d_seed_plain"]["mean"],
            headline["d_seed_plain"]["ci_lo_95"],
            headline["d_seed_plain"]["ci_hi_95"],
            headline["d_seed_padded"]["mean"],
            headline["d_seed_padded"]["ci_lo_95"],
            headline["d_seed_padded"]["ci_hi_95"],
        )
    elif headline_status == "inconclusive_descriptive_only":
        logger.warning(
            "H2 INCONCLUSIVE: only %d complete paired seed(s); need >= %d (MF-H)",
            len(complete_seeds),
            H2_MIN_SEEDS,
        )
    elif headline_status == "blocked_onpolicy_switch_required":
        logger.warning(
            "H2 BLOCKED: Phase 4.5 on-policy validation requires switching to "
            "trained-greedy R (ratio=%s > 1.5)",
            onpolicy_ratio,
        )
    elif headline_status == "inconclusive_dynamic_range_failed":
        # Round-3 fix (review blocker #1): explicit log so the operator
        # sees the override happened (the saturation warning at the end
        # of main() still fires too).
        logger.warning(
            "H2 INCONCLUSIVE (dynamic-range failed): leakage log-prob sd "
            "<= %.2f in arm(s) %s — saturation regime, headline overridden.",
            DYNAMIC_RANGE_THRESHOLD,
            headline.get("dynamic_range_failed_arms"),
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
            "Dynamic-range gate FAILED — at least one arm has sd <= %.2f; "
            "saturation regime, headline reads as inconclusive.",
            DYNAMIC_RANGE_THRESHOLD,
        )


if __name__ == "__main__":
    main()
