"""Issue #464 positive-only follow-up — analysis + headline stats.

Reads per-cell JSONs produced by ``i464_po_eval.py`` and mirrors the
parent #464 analyzer's headline shape (``i464_phase5_analyze.py``):
paired-bootstrap CIs on per-seed deltas, H1 elicitation gate, dynamic-
range gate. Differences from the parent:

  * Only 3 arms (system_plain / system_padded / role) — role_nonsense
    and role_mismatch are NOT replicated.
  * 18 cells = 3 arms x 3 seeds x 2 personas (one persona per LoRA).
  * Off-diagonal leakage probe = ``g_logprob`` of ` ※` under the OTHER
    persona's SAME-ARM-FAMILY encoding (system family for
    system_plain/system_padded; role family for role).
  * ``L_arm_seed`` is the mean across the 2 single-persona cells'
    off-diagonal log-probs (instead of averaging the parent's 4-cell
    symmetric set).
  * NEW: leakage-to-default — ` ※` under ``default_assistant`` per arm,
    averaged across all 6 cells per arm (3 seeds x 2 personas). This
    is the on-default-context number the parent's co-residence
    couldn't measure.

Inputs (read-only):
  eval_results/issue_464/positive_only/cross_eval/per_cell/<cell>__<e_eval>.json
      18 cells x 3 e_eval = 54 files

Outputs:
  eval_results/issue_464/positive_only/analysis.json — mirrors the
      parent's analysis shape: per-arm-per-seed L, headline deltas with
      paired bootstrap CIs, H1 elicitation, leakage-to-default,
      dynamic-range gate, raw per-cell.

Headline statistic:
  Per seed:
    L_arm_seed = mean over (training_persona ∈ {pirate, villain}) of
      raw g_logprob(` ※`, e_off-diagonal)
      where e_off-diagonal = the OTHER persona's same-arm-family encoding:
        pirate-only cell → ` ※` under (system_villain  if arm ∈ system_*,
                                       role_villain    if arm == role)
        villain-only cell → ` ※` under (system_pirate  if arm ∈ system_*,
                                        role_pirate    if arm == role)

    d_seed_plain  = L_system_plain  - L_role   (>0 ⇒ role leaks less)
    d_seed_padded = L_system_padded - L_role

  H2 PASS (mirrors parent's threshold):
    mean(d_plain)  ≥ 1.0 nat AND 95% CI > 0 AND all per-seed d > 0
    mean(d_padded) ≥ 1.0 nat AND 95% CI > 0 AND all per-seed d > 0

Variants (``--variant``): ``po`` (default) and ``cn`` analyze the 3
parent-recipe arms with the paired headline pair (d_plain, d_padded);
``min_cn`` (the minimal_content_cn follow-up) analyzes the 2 minimal
arms (12 cells) with the SINGLE registered pair
``d_seed_minimal_cn = L_system_minimal - L_role_bare`` and records the
plan-§3 verdict-precedence ordering verbatim in a
``verdict_precedence_note`` field. Identical H1 / H2 / dynamic-range
machinery (no new thresholds); the min_cn ``headline_status`` is derived
per that precedence — ``inconclusive_dynamic_range_failed`` (highest)
> ``fail`` (H1) > ``falsifier_fired`` (CI overlaps zero / sign flip /
any seed <= 0) > ``ok`` (full PASS) vs
``directional_partial_survival_below_threshold`` (all seeds positive,
CI excludes zero, mean < 1 nat).

CLI:
    uv run python scripts/i464_po_analyze.py
    uv run python scripts/i464_po_analyze.py --allow-partial
    uv run python scripts/i464_po_analyze.py --variant min_cn
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from explore_persona_space.experiments import i464_encodings as enc

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves
# when this script is invoked directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mirror the parent analyzer's thresholds + bootstrap helper exactly so
# the two follow-ups are read against a single methodology.
from scripts.i464_phase5_analyze import (  # type: ignore[import-not-found]
    DYNAMIC_RANGE_THRESHOLD,
    H1_ELICITATION_THRESHOLD,
    H2_HEADLINE_THRESHOLD,
    H2_MIN_SEEDS,
    N_BOOTSTRAP,
    _paired_bootstrap_ci,
)

load_dotenv()

logger = logging.getLogger("i464.po_analyze")

# Per-variant input + output paths. Selected at runtime from --variant.
# Defaults preserve the positive-only (``po``) behavior so existing
# call sites stay byte-identical.
PER_CELL_DIR_FOR: dict[str, Path] = {
    "po": Path("eval_results/issue_464/positive_only/cross_eval/per_cell"),
    "cn": Path("eval_results/issue_464/contrastive_negatives/cross_eval/per_cell"),
    "min_cn": Path("eval_results/issue_464/minimal_content_cn/cross_eval/per_cell"),
}
OUT_PATH_FOR: dict[str, Path] = {
    "po": Path("eval_results/issue_464/positive_only/analysis.json"),
    "cn": Path("eval_results/issue_464/contrastive_negatives/analysis.json"),
    "min_cn": Path("eval_results/issue_464/minimal_content_cn/analysis.json"),
}
SCHEMA_VERSION_FOR: dict[str, str] = {
    "po": "i464_po_analyze_v1",
    "cn": "i464_cn_analyze_v1",
    "min_cn": "i464_min_cn_analyze_v1",
}

# Legacy aliases (positive-only defaults) — kept for any importer that
# referenced these constants before --variant existed.
PER_CELL_DIR = PER_CELL_DIR_FOR["po"]
OUT_PATH = OUT_PATH_FOR["po"]

# Module-level state set from --variant before any helper consumes it.
# Helpers read from this dict instead of the legacy globals so the
# variant choice flows through without each helper needing an extra arg.
_ACTIVE: dict[str, Path] = {"per_cell_dir": PER_CELL_DIR}

SEEDS = (42, 137, 1337)
PO_ARMS: tuple[enc.Arm, ...] = ("system_plain", "system_padded", "role")
# Variant-aware arms list: po/cn analyze the 3 parent-recipe arms; min_cn
# (the minimal_content_cn follow-up) analyzes the 2 content-matched
# minimal arms (12 cells = 2 arms x 3 seeds x 2 personas).
ARMS_FOR: dict[str, tuple[enc.Arm, ...]] = {
    "po": PO_ARMS,
    "cn": PO_ARMS,
    "min_cn": enc.MINIMAL_ARMS,
}
SHARED_MARKER_PERSONA: enc.Persona = "pirate"

# REGISTERED INTERPRETATION NOTE (plan §3, recorded verbatim so the
# analyzer reads the precedence ordering from the artifact — NOT a new
# gate; a precedence ordering of the inherited, already-registered rules).
VERDICT_PRECEDENCE_NOTE = (
    "(a) H1 fail => no headline claim. "
    "(b) DR failure supersedes BOTH PASS and the falsifier => "
    "inconclusive_dynamic_range_failed with the parent's hedge - the "
    "falsifier is reachable only when the DR gate passes. "
    "(c) DR-ok AND (CI overlaps zero / sign flip / any seed <= 0) => "
    "falsifier fires, content-attribution conclusion. "
    "(d) DR-ok AND all seeds positive AND mean < 1.0 => directional/partial "
    "survival below the inherited 1-nat threshold - neither PASS nor "
    "falsification."
)


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


def _po_cell_label(arm: enc.Arm, seed: int, persona: enc.Persona) -> str:
    """Canonical po cell label; matches train + eval."""
    return f"{arm}_seed{seed}_{persona}"


def _load_per_cell(arm: enc.Arm, seed: int, persona: enc.Persona, e_eval: str) -> dict | None:
    """Read one per-cell JSON or return None if missing.

    Reads from ``_ACTIVE['per_cell_dir']`` (set in main from --variant)
    rather than the module-level ``PER_CELL_DIR`` so the same helper
    serves both po and cn paths without each call site needing an
    explicit variant argument.
    """
    p = _ACTIVE["per_cell_dir"] / f"{_po_cell_label(arm, seed, persona)}__{e_eval}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _own_eval_encoding_for(arm: enc.Arm, persona: enc.Persona) -> str:
    """Diagonal eval encoding for ``(arm, persona)``.

    Mirrors the parent's ``_own_eval_encoding_for``, restricted to the
    3 PO_ARMS plus the min_cn variant's 2 minimal arms.
    """
    if arm == "role":
        return f"role_{persona}"
    if arm == "system_minimal":
        return f"system_minimal_{persona}"
    if arm == "role_bare":
        return f"role_bare_{persona}"
    return f"system_{persona}"


def _other_eval_encoding_for(arm: enc.Arm, persona: enc.Persona) -> str:
    """Off-diagonal eval encoding for ``(arm, persona)`` — the OTHER persona's
    SAME-arm-family encoding (matches the brief's headline definition)."""
    other: enc.Persona = "villain" if persona == "pirate" else "pirate"
    if arm == "role":
        return f"role_{other}"
    if arm == "system_minimal":
        return f"system_minimal_{other}"
    if arm == "role_bare":
        return f"role_bare_{other}"
    return f"system_{other}"


def _symmetric_leakage(arm: enc.Arm, seed: int) -> tuple[float, list[float]]:
    """Return (L_arm_seed, raw_logprobs_per_cell).

    Off-diagonal cells: for each training persona p ∈ {pirate, villain},
    read ` ※` log-prob under the OTHER persona's same-arm-family encoding.
    Mean of the 2 raw log-probs.
    """
    raw: list[float] = []
    for persona in enc.PERSONAS:
        e_off = _other_eval_encoding_for(arm, persona)
        payload = _load_per_cell(arm, seed, persona, e_off)
        if payload is None:
            raise FileNotFoundError(
                f"analyze: missing per-cell JSON for {_po_cell_label(arm, seed, persona)}/{e_off}"
            )
        raw.append(payload["g_logprob"])
    if not raw:
        raise RuntimeError(f"po off-diagonal leakage cells empty for arm={arm} seed={seed}")
    return float(np.mean(raw)), raw


def _own_persona_elicitation(arm: enc.Arm, seed: int) -> tuple[list[float], list[str]]:
    """H1 gate input: raw trained log P on each (training_persona, own-encoding) cell.

    Returns ([logp_pirate_cell, logp_villain_cell], [label_pirate, label_villain]).
    """
    own_logps: list[float] = []
    labels: list[str] = []
    for persona in enc.PERSONAS:
        e_own = _own_eval_encoding_for(arm, persona)
        payload = _load_per_cell(arm, seed, persona, e_own)
        if payload is None:
            raise FileNotFoundError(
                f"analyze H1: missing own-encoding cell "
                f"{_po_cell_label(arm, seed, persona)}/{e_own}"
            )
        own_logps.append(float(payload["g_logprob"]))
        labels.append(f"{_po_cell_label(arm, seed, persona)}/{e_own}")
    return own_logps, labels


def _leakage_to_default(arm: enc.Arm) -> tuple[list[float], list[str]]:
    """` ※` log-prob under ``default_assistant`` for every cell in this arm.

    The NEW measurement the parent #464 could NOT make (co-residence + the
    two-marker contrast in the parent meant default_assistant was a
    diagnostic side note, not a co-axial bystander). Returns (per_cell_logp,
    per_cell_label) across (seed x persona) = 6 cells per arm.
    """
    logps: list[float] = []
    labels: list[str] = []
    for seed in SEEDS:
        for persona in enc.PERSONAS:
            payload = _load_per_cell(arm, seed, persona, "default_assistant")
            if payload is None:
                raise FileNotFoundError(
                    f"analyze leakage-to-default: missing cell "
                    f"{_po_cell_label(arm, seed, persona)}/default_assistant"
                )
            logps.append(float(payload["g_logprob"]))
            labels.append(f"{_po_cell_label(arm, seed, persona)}/default_assistant")
    return logps, labels


def _h2_verdict(name: str, d_per_seed: list[float], mean: float, lo: float, hi: float) -> dict:
    """Pack a single-comparison H2 verdict (mirrors parent's ``_h2_verdict``)."""
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
    arms: tuple[enc.Arm, ...] = PO_ARMS,
) -> tuple[dict[str, dict], bool]:
    """Return (per-arm sd+threshold-pass dict, overall gate ok bool)."""
    dr_gate: dict[str, dict] = {}
    for arm in arms:
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


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - mirrors parent's structure
    """Entry point for the positive-only analyzer."""
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
        help="Seeds to aggregate. Default = (42, 137, 1337).",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="If set, skip missing per-cell files (smoke mode); else FAIL LOUD.",
    )
    ap.add_argument(
        "--variant",
        choices=("po", "cn", "min_cn"),
        default="po",
        help=(
            "Which follow-up to analyze. ``po`` (default) = positive-only "
            "(reads ``eval_results/issue_464/positive_only/cross_eval/per_cell/``, "
            "writes ``positive_only/analysis.json``). ``cn`` = "
            "contrastive-negatives (reads + writes under ``contrastive_negatives/``). "
            "``min_cn`` = minimal_content_cn (2 minimal arms, headline "
            "``d_seed_minimal_cn``; reads + writes under ``minimal_content_cn/``)."
        ),
    )
    args = ap.parse_args(argv)

    # Wire the variant choice into the module-level _ACTIVE dict so all
    # helpers (_load_per_cell, _symmetric_leakage, ...) read from the
    # right directory without each call site needing an extra arg.
    _ACTIVE["per_cell_dir"] = PER_CELL_DIR_FOR[args.variant]
    out_path_active = OUT_PATH_FOR[args.variant]
    schema_version = SCHEMA_VERSION_FOR[args.variant]
    arms_active = ARMS_FOR[args.variant]
    logger.info(
        "variant=%s arms=%s per_cell_dir=%s out_path=%s",
        args.variant,
        arms_active,
        _ACTIVE["per_cell_dir"],
        out_path_active,
    )

    L_per_arm_per_seed: dict[str, dict[int, float]] = {arm: {} for arm in arms_active}
    raw_per_cell: dict[str, dict[int, list[float]]] = {arm: {} for arm in arms_active}
    own_logp_per_arm_per_seed: dict[str, dict[int, list[float]]] = {arm: {} for arm in arms_active}
    own_cell_labels: list[str] = []
    missing: list[str] = []

    for seed in args.seeds:
        for arm in arms_active:
            try:
                L, raw = _symmetric_leakage(arm, seed)
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("analyze leakage (partial): %s", e)
                    missing.append(str(e))
                    continue
                raise
            L_per_arm_per_seed[arm][seed] = L
            raw_per_cell[arm][seed] = raw

            try:
                own_logps, labels = _own_persona_elicitation(arm, seed)
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("analyze H1 (partial): %s", e)
                    missing.append(str(e))
                else:
                    raise
            else:
                own_logp_per_arm_per_seed[arm][seed] = own_logps
                if not own_cell_labels:
                    own_cell_labels = labels

    if missing and not args.allow_partial:
        raise RuntimeError(f"analyze: {len(missing)} missing per-cell JSONs")

    # ── H1 elicitation gate (per-cell pass map) ─────────────────────────
    h1_per_cell_pass: dict[str, bool] = {}
    h1_per_cell_logp: dict[str, float] = {}
    for arm, by_seed in own_logp_per_arm_per_seed.items():
        for seed, logps in by_seed.items():
            for persona_idx, persona in enumerate(enc.PERSONAS):
                e_own = _own_eval_encoding_for(arm, persona)  # type: ignore[arg-type]
                key = f"{_po_cell_label(arm, seed, persona)}/{e_own}"  # type: ignore[arg-type]
                lp = float(logps[persona_idx])
                h1_per_cell_logp[key] = lp
                h1_per_cell_pass[key] = lp >= H1_ELICITATION_THRESHOLD
    h1_overall_pass = bool(h1_per_cell_pass) and all(h1_per_cell_pass.values())

    # ── Leakage-to-default (per arm, NEW vs parent) ─────────────────────
    leakage_to_default: dict[str, dict] = {}
    try:
        for arm in arms_active:
            logps, labels = _leakage_to_default(arm)
            arr = np.array(logps, dtype=float)
            leakage_to_default[arm] = {
                "per_cell_logp": logps,
                "per_cell_label": labels,
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=0)),
                "n": len(logps),
            }
    except FileNotFoundError as e:
        if args.allow_partial:
            logger.warning("analyze leakage-to-default (partial): %s", e)
            leakage_to_default["partial"] = {"reason": str(e)}
        else:
            raise

    # ── Headline: paired deltas over COMPLETE seeds only ────────────────
    headline: dict
    headline_status: str
    if args.variant == "min_cn":
        # SINGLE pair: d_seed_minimal_cn = L_system_minimal - L_role_bare
        # (>0 ⇒ the bare-word role header leaks less). Identical H1 / H2 /
        # dynamic-range machinery as po/cn — no new thresholds.
        complete_seeds = sorted(
            set(L_per_arm_per_seed["system_minimal"]) & set(L_per_arm_per_seed["role_bare"])
        )
        d_min_cn = [
            L_per_arm_per_seed["system_minimal"][s] - L_per_arm_per_seed["role_bare"][s]
            for s in complete_seeds
        ]
        if len(complete_seeds) < H2_MIN_SEEDS:
            headline_status = "inconclusive_descriptive_only"
            headline = {
                "status": headline_status,
                "n_complete_seeds": len(complete_seeds),
                "min_seeds_required": H2_MIN_SEEDS,
                "reason": (
                    f"only {len(complete_seeds)} complete paired seeds (need >= {H2_MIN_SEEDS})."
                ),
                "d_seed_minimal_cn_descriptive": d_min_cn,
                "h2_full_pass": False,
                "h2_partial": False,
            }
        else:
            m_mc, lo_mc, hi_mc = _paired_bootstrap_ci(d_min_cn, N_BOOTSTRAP)
            verdict_min_cn = _h2_verdict("minimal_cn", d_min_cn, m_mc, lo_mc, hi_mc)
            h2_full = verdict_min_cn["pass"] and h1_overall_pass
            # Explicit status per the registered precedence (plan §3/§6
            # REGISTERED INTERPRETATION NOTE; VERDICT_PRECEDENCE_NOTE above).
            # The dynamic-range override BELOW stays HIGHEST precedence — it
            # rewrites any status set here, so "falsifier_fired" and the
            # directional status are reachable only when the DR gate passes.
            #   (a) H1 fail -> "fail" (no headline claim).
            #   (c) H1-ok + (CI overlaps zero / sign flip / any seed <= 0)
            #       -> "falsifier_fired" (content-attribution conclusion).
            #   PASS: H1-ok + all seeds positive + CI excludes zero +
            #       mean >= 1 nat -> "ok".
            #   (d) H1-ok + all seeds positive + CI excludes zero + mean
            #       < 1 nat -> directional / partial survival below the
            #       inherited 1-nat threshold — neither PASS nor
            #       falsification.
            if not h1_overall_pass:
                headline_status = "fail"
            elif h2_full:
                headline_status = "ok"
            elif verdict_min_cn["all_seeds_positive"] and verdict_min_cn["ci_excludes_zero"]:
                # Every _h2_verdict criterion except mean >= H2_HEADLINE_
                # THRESHOLD is met, so the 1-nat mean is the only miss.
                headline_status = "directional_partial_survival_below_threshold"
            else:
                headline_status = "falsifier_fired"
            headline = {
                "status": headline_status,
                "n_complete_seeds": len(complete_seeds),
                "complete_seeds": complete_seeds,
                "d_seed_minimal_cn": verdict_min_cn,
                "h2_full_pass": h2_full,
                "h2_partial": False,
                "h1_required_before_h2": True,
                "h1_overall_pass": h1_overall_pass,
                "n_bootstrap": N_BOOTSTRAP,
            }
    else:
        complete_seeds = sorted(
            set(L_per_arm_per_seed["system_plain"])
            & set(L_per_arm_per_seed["system_padded"])
            & set(L_per_arm_per_seed["role"])
        )
        d_plain: list[float] = []
        d_padded: list[float] = []
        for s in complete_seeds:
            d_plain.append(L_per_arm_per_seed["system_plain"][s] - L_per_arm_per_seed["role"][s])
            d_padded.append(L_per_arm_per_seed["system_padded"][s] - L_per_arm_per_seed["role"][s])

        if len(complete_seeds) < H2_MIN_SEEDS:
            headline_status = "inconclusive_descriptive_only"
            headline = {
                "status": headline_status,
                "n_complete_seeds": len(complete_seeds),
                "min_seeds_required": H2_MIN_SEEDS,
                "reason": (
                    f"only {len(complete_seeds)} complete paired seeds (need >= {H2_MIN_SEEDS})."
                ),
                "d_seed_plain_descriptive": d_plain,
                "d_seed_padded_descriptive": d_padded,
                "h2_full_pass": False,
                "h2_partial": False,
            }
        else:
            m_p, lo_p, hi_p = _paired_bootstrap_ci(d_plain, N_BOOTSTRAP)
            m_pad, lo_pad, hi_pad = _paired_bootstrap_ci(d_padded, N_BOOTSTRAP)
            verdict_plain = _h2_verdict("plain", d_plain, m_p, lo_p, hi_p)
            verdict_padded = _h2_verdict("padded", d_padded, m_pad, lo_pad, hi_pad)
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
                "n_bootstrap": N_BOOTSTRAP,
            }

    # ── Dynamic-range gate (mirrors parent's override-on-saturation) ────
    dr_gate, dynamic_range_ok = _compute_dynamic_range_gate(raw_per_cell, arms=arms_active)
    if not dynamic_range_ok and headline_status not in (
        "inconclusive_descriptive_only",
        "inconclusive_dynamic_range_failed",
    ):
        failing_arms = [a for a, v in dr_gate.items() if not v.get("above_threshold")]
        headline_status = "inconclusive_dynamic_range_failed"
        headline["status"] = headline_status
        headline["h2_full_pass"] = False
        headline["h2_partial"] = False
        headline["dynamic_range_failed_arms"] = failing_arms
        headline["reason"] = (
            f"Dynamic-range gate failed: arms with sd <= {DYNAMIC_RANGE_THRESHOLD}: "
            f"{failing_arms}. Saturated regime — leakage log-prob comparisons "
            "are rank-shuffles on a ceiling, not informative segmentation."
        )

    payload = {
        "schema_version": schema_version,
        "variant": args.variant,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seeds": args.seeds,
        "arms": list(arms_active),
        "shared_marker_text": enc.MARKER_PIRATE_TEXT,
        "shared_marker_id": enc.MARKER_PIRATE_ID,
        "L_per_arm_per_seed": {arm: dict(d) for arm, d in L_per_arm_per_seed.items()},
        "complete_seeds": complete_seeds,
        "h2_min_seeds": H2_MIN_SEEDS,
        "h1_elicitation": {
            "threshold_nats": H1_ELICITATION_THRESHOLD,
            "per_cell_logp": h1_per_cell_logp,
            "per_cell_pass": h1_per_cell_pass,
            "overall_pass": h1_overall_pass,
            "n_cells": len(h1_per_cell_pass),
        },
        "leakage_to_default": leakage_to_default,
        "headline": headline,
        "headline_status": headline_status,
        "dynamic_range_gate": {
            "threshold": DYNAMIC_RANGE_THRESHOLD,
            "per_arm": dr_gate,
            "ok": dynamic_range_ok,
        },
        "raw_per_cell": raw_per_cell,
        "n_missing_per_cell": len(missing),
    }
    if args.variant == "min_cn":
        # Registered verdict-precedence ordering (plan §3) recorded
        # verbatim so the analyzer reads it from the artifact.
        payload["verdict_precedence_note"] = VERDICT_PRECEDENCE_NOTE
    out_path_active.parent.mkdir(parents=True, exist_ok=True)
    out_path_active.write_text(json.dumps(payload, indent=2))
    logger.info(
        "%s analyze done -> %s (status=%s complete_seeds=%d H1=%s)",
        args.variant,
        out_path_active,
        headline_status,
        len(complete_seeds),
        h1_overall_pass,
    )
    if headline_status == "ok":
        if args.variant == "min_cn":
            logger.info(
                "H2 PASS: d_minimal_cn mean=%.3f CI=[%.3f, %.3f]",
                headline["d_seed_minimal_cn"]["mean"],
                headline["d_seed_minimal_cn"]["ci_lo_95"],
                headline["d_seed_minimal_cn"]["ci_hi_95"],
            )
        else:
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
            "H2 INCONCLUSIVE: only %d complete paired seed(s); need >= %d",
            len(complete_seeds),
            H2_MIN_SEEDS,
        )
    elif headline_status == "inconclusive_dynamic_range_failed":
        logger.warning(
            "H2 INCONCLUSIVE (dynamic-range failed): leakage log-prob sd "
            "<= %.2f in arm(s) %s — saturation regime, headline overridden.",
            DYNAMIC_RANGE_THRESHOLD,
            headline.get("dynamic_range_failed_arms"),
        )
    elif headline_status == "falsifier_fired":
        # min_cn only (po/cn never set this status).
        v = headline["d_seed_minimal_cn"]
        logger.warning(
            "H2 FALSIFIER FIRED: d_minimal_cn mean=%.3f CI=[%.3f, %.3f] per_seed=%s — "
            "CI overlaps zero / sign flip / non-positive seed; the CN-regime edge is "
            "attributable to the elaborate system instruction's content.",
            v["mean"],
            v["ci_lo_95"],
            v["ci_hi_95"],
            v["d_per_seed"],
        )
    elif headline_status == "directional_partial_survival_below_threshold":
        # min_cn only (po/cn never set this status).
        v = headline["d_seed_minimal_cn"]
        logger.info(
            "H2 DIRECTIONAL (partial survival): d_minimal_cn mean=%.3f CI=[%.3f, %.3f] — "
            "all seeds positive, CI excludes zero, mean below the inherited %.1f-nat "
            "threshold; neither PASS nor falsification (registered precedence (d)).",
            v["mean"],
            v["ci_lo_95"],
            v["ci_hi_95"],
            H2_HEADLINE_THRESHOLD,
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
    # Leakage-to-default summary (NEW vs parent #464).
    for arm in arms_active:
        d = leakage_to_default.get(arm)
        if d and "mean" in d:
            logger.info(
                "leakage-to-default arm=%s mean=%.3f sd=%.3f n=%d",
                arm,
                d["mean"],
                d["sd"],
                d["n"],
            )


if __name__ == "__main__":
    main()
