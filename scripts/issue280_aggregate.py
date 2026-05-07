#!/usr/bin/env python3
"""Phase-3 aggregator for issue #280.

Reads the 66-cell namespace assembled by ``issue280_eval.py --stage aggregate``
(an inventory JSON pointing at the 27 carry-over #186 cells +
39 new #280 cells), then computes:

* 8 macro contrasts (4 contrasts x 2 outcome axes) with paired bootstrap
  Δ + Holm-Bonferroni adjustment at family-wise alpha=0.01.
* Per-source bootstrap CIs (descriptive heterogeneity diagnostic; raw p,
  no correction).
* TOST equivalence test for H3 (``contradicting_cot - generic_cot``,
  source-loss): 90 % CI within [-0.03, +0.03] AND 99 % CI not extending
  beyond ±0.05 (plan v2 §6.2 fix 5).
* Mismatched-eval comparison rows (plan v2 §4.3 fix 3).
* Carry-forward note for H2 (descriptive only, plan §3 H2 fix 6).

Outputs:

* ``eval_results/issue280/aggregate.json`` (extended schema with
  ``holm_corrected_p``, ``tost_h3``, ``per_source_ci``, ``mismatched_eval``,
  ``carry_forward_h2``).
* TODO: ``figures/issue280/hero.{pdf,png,meta.json}`` is left to a follow-up
  commit using the ``paper-plots`` skill — the analyzer agent will pull this
  numbers blob and dispatch ``paper-plots`` on the resulting hero spec.

CLI::

    uv run python scripts/issue280_aggregate.py \\
        --inventory eval_results/issue280/_aggregate_inventory.json \\
        --n-bootstrap 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("issue280_aggregate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PERSONA_ORDER: list[str] = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]

EVAL_SCAFFOLD_KEYS = ("no_cot", "generic_cot", "persona_cot", "empty_persona_cot_eval")
SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
SEEDS = (42, 137, 256)

# 4 contrasts x 2 axes = 8 macro tests (plan §6.3).
# Each entry: (label, train_arm_a, train_arm_b, eval_arm, axis, hypothesis)
CONTRASTS: list[dict] = [
    # H1: rationale semantics (generic > scrambled > garbage)
    {
        "label": "generic - garbage",
        "a": "generic_cot",
        "b": "garbage_cot",
        "eval_arm": "persona_cot",  # matched eval (PERSONA_COT scaffold)
        "axis": "source_loss",
        "hypothesis": "H1",
    },
    {
        "label": "generic - garbage",
        "a": "generic_cot",
        "b": "garbage_cot",
        "eval_arm": "persona_cot",
        "axis": "bystander_leakage",
        "hypothesis": "H1",
    },
    {
        "label": "generic - scrambled_english",
        "a": "generic_cot",
        "b": "scrambled_english_cot",
        "eval_arm": "persona_cot",
        "axis": "source_loss",
        "hypothesis": "H1",
    },
    {
        "label": "generic - scrambled_english",
        "a": "generic_cot",
        "b": "scrambled_english_cot",
        "eval_arm": "persona_cot",
        "axis": "bystander_leakage",
        "hypothesis": "H1",
    },
    # H2: persona - garbage (the new test; the carry-forward persona - generic
    # is reported descriptively but NOT bootstrapped here — plan §3 H2 fix 6).
    {
        "label": "persona - garbage",
        "a": "persona_cot",
        "b": "garbage_cot",
        "eval_arm": "persona_cot",
        "axis": "source_loss",
        "hypothesis": "H2",
    },
    {
        "label": "persona - garbage",
        "a": "persona_cot",
        "b": "garbage_cot",
        "eval_arm": "persona_cot",
        "axis": "bystander_leakage",
        "hypothesis": "H2",
    },
    # H3: contradicting - generic, source-loss (TOST falsification).
    {
        "label": "contradicting - generic",
        "a": "contradicting_cot",
        "b": "generic_cot",
        "eval_arm": "persona_cot",
        "axis": "source_loss",
        "hypothesis": "H3",
    },
    # H4: descriptive (contradicting - generic, bystander). Not a falsification.
    {
        "label": "contradicting - generic",
        "a": "contradicting_cot",
        "b": "generic_cot",
        "eval_arm": "persona_cot",
        "axis": "bystander_leakage",
        "hypothesis": "H4",
    },
]

# Mismatched-eval rows for plan §4.3 fix 3 (descriptive).
MISMATCHED_EVAL_PAIRS: list[dict] = [
    {
        "label": "garbage_cot_train + persona_cot_eval",
        "train_arm": "garbage_cot",
        "eval_arm": "persona_cot",
    },
    {
        "label": "scrambled_english_cot_train + generic_cot_eval",
        "train_arm": "scrambled_english_cot",
        "eval_arm": "generic_cot",
    },
    {
        "label": "contradicting_cot_train + empty_persona_cot_eval",
        "train_arm": "contradicting_cot",
        "eval_arm": "empty_persona_cot_eval",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_baseline() -> dict:
    """Baseline result.json (un-trained Qwen2.5-7B-Instruct, full grid)."""
    p1 = PROJECT_ROOT / "eval_results" / "issue280" / "baseline" / "result.json"
    p2 = PROJECT_ROOT / "eval_results" / "issue186" / "baseline" / "result.json"
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    raise FileNotFoundError(
        "No baseline result.json found at issue280/baseline or issue186/baseline."
    )


def _correct_array(per_persona: dict, n_q: int) -> np.ndarray:
    arr = np.zeros((n_q, len(PERSONA_ORDER), len(EVAL_SCAFFOLD_KEYS)), dtype=np.int8)
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    for p in PERSONA_ORDER:
        block = per_persona.get(p)
        if block is None:
            continue
        for q_idx, row in enumerate(block.get("raw", [])[:n_q]):
            ca = row.get("correct_answer")
            if ca is None:
                continue
            for sc_i, sc_key in enumerate(EVAL_SCAFFOLD_KEYS):
                pred = row.get(f"{sc_key}_pred")
                arr[q_idx, persona_idx[p], sc_i] = int(pred == ca)
    return arr


def _eval_arm_index(eval_arm: str) -> int:
    return EVAL_SCAFFOLD_KEYS.index(eval_arm)


def _holm_bonferroni(pvals: list[float], alpha: float = 0.01) -> list[bool]:
    """Holm-Bonferroni step-down at family-wise alpha. Returns reject[i] booleans
    aligned with pvals."""
    n = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def _holm_corrected_p(pvals: list[float]) -> list[float]:
    """Adjusted p-values for Holm-Bonferroni (monotone)."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (n - rank))
        adj[idx] = min(1.0, running)
    return adj


def _bootstrap_paired(
    diff_per_pair: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """Joint (q, seed) paired bootstrap on a flat 1-D delta array.

    Returns (point_estimate, ci_lo_95, ci_hi_95, bootstrap_distribution).
    """
    if diff_per_pair.size == 0:
        return float("nan"), float("nan"), float("nan"), np.zeros(0)
    point = float(diff_per_pair.mean())
    n = diff_per_pair.size
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(diff_per_pair[idx].mean())
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    return point, float(ci_lo), float(ci_hi), boots


def _two_sided_p(boots: np.ndarray, point: float) -> float:
    if boots.size == 0:
        return float("nan")
    centered = boots - boots.mean()
    return float(np.mean(np.abs(centered) >= abs(point)))


def _ci(boots: np.ndarray, conf: float) -> tuple[float, float]:
    if boots.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


# ── Main contrast computation ────────────────────────────────────────────────


def _delta_per_pair(
    *,
    base_correct: np.ndarray,
    cells: dict[str, np.ndarray],
    arm_a: str,
    arm_b: str,
    source: str,
    eval_arm: str,
    axis: str,
) -> np.ndarray | None:
    """Compute the per-(q, seed) delta for arm_a - arm_b, given source +
    eval_arm + axis. Returns None if any seed for either arm is missing."""
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    eval_idx = _eval_arm_index(eval_arm)

    if axis == "source_loss":
        # source_loss = baseline_acc(source) - trained_acc(source). Compute
        # the per-(q, seed) loss = baseline_correct - trained_correct on the
        # SOURCE persona.
        bys_idx = np.array([persona_idx[source]])
    elif axis == "bystander_leakage":
        bystanders = [p for p in PERSONA_ORDER if p != source]
        bys_idx = np.array([persona_idx[p] for p in bystanders])
    else:
        raise ValueError(f"Unknown axis: {axis!r}")

    def _loss_stack(arm: str) -> np.ndarray | None:
        stacks = []
        for s in SEEDS:
            cell_id = f"{source}_{arm}_seed{s}"
            tc = cells.get(cell_id)
            if tc is None:
                return None
            tc_slice = tc[:, bys_idx, eval_idx]  # (n_q, n_persona_subset)
            bc_slice = base_correct[:, bys_idx, eval_idx]
            stacks.append(bc_slice.astype(np.float32) - tc_slice.astype(np.float32))
        return np.stack(stacks, axis=1)  # (n_q, n_seeds, n_personas)

    la = _loss_stack(arm_a)
    lb = _loss_stack(arm_b)
    if la is None or lb is None:
        return None

    # Mean over the persona-subset dimension first (so the unit of inference
    # is one (q, seed) pair).
    la_per_qs = la.mean(axis=2)  # (n_q, n_seeds)
    lb_per_qs = lb.mean(axis=2)
    delta = la_per_qs - lb_per_qs  # (n_q, n_seeds)
    return delta.reshape(-1)  # flat (n_q * n_seeds,)


def _compute_contrasts(
    *,
    base_correct: np.ndarray,
    cells: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    macro_results: list[dict] = []

    for c in CONTRASTS:
        per_source: dict[str, dict] = {}
        macro_deltas: list[np.ndarray] = []

        for source in SOURCES:
            delta = _delta_per_pair(
                base_correct=base_correct,
                cells=cells,
                arm_a=c["a"],
                arm_b=c["b"],
                source=source,
                eval_arm=c["eval_arm"],
                axis=c["axis"],
            )
            if delta is None:
                logger.warning(
                    "Missing cells for contrast %s axis=%s source=%s — skipping per-source.",
                    c["label"],
                    c["axis"],
                    source,
                )
                continue
            point, ci_lo, ci_hi, boots = _bootstrap_paired(delta, n_bootstrap, rng)
            per_source[source] = {
                "point_estimate": point,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "n_pairs": int(delta.size),
                "p_two_sided_uncorrected": _two_sided_p(boots, point),
            }
            macro_deltas.append(delta)

        if not macro_deltas:
            macro_results.append(
                {
                    **c,
                    "macro_point": None,
                    "macro_p_uncorrected": None,
                    "per_source": per_source,
                }
            )
            continue

        macro_flat = np.concatenate(macro_deltas)
        m_point, m_ci_lo, m_ci_hi, m_boots = _bootstrap_paired(macro_flat, n_bootstrap, rng)
        macro_p = _two_sided_p(m_boots, m_point)
        macro_results.append(
            {
                **c,
                "macro_point": m_point,
                "macro_ci_95_lo": m_ci_lo,
                "macro_ci_95_hi": m_ci_hi,
                "macro_p_uncorrected": macro_p,
                "per_source": per_source,
                "_macro_boots": m_boots,
            }
        )

    # Holm-Bonferroni on the 8 macro p-values (family-wise alpha=0.01).
    pvals = [r["macro_p_uncorrected"] for r in macro_results]
    pvals_clean = [p if p is not None else 1.0 for p in pvals]
    holm_reject = _holm_bonferroni(pvals_clean, alpha=0.01)
    holm_corr = _holm_corrected_p(pvals_clean)

    for r, reject, p_adj in zip(macro_results, holm_reject, holm_corr, strict=True):
        r["holm_reject_at_alpha_0p01"] = bool(reject)
        r["holm_corrected_p"] = float(p_adj)

    # H3 TOST equivalence (90 % within ±0.03 AND 99 % within ±0.05).
    h3_idx = next(
        (i for i, r in enumerate(macro_results) if r["hypothesis"] == "H3"),
        None,
    )
    tost_h3: dict | None = None
    if h3_idx is not None:
        h3 = macro_results[h3_idx]
        boots = h3.pop("_macro_boots", np.zeros(0))
        ci90_lo, ci90_hi = _ci(boots, 0.90)
        ci99_lo, ci99_hi = _ci(boots, 0.99)
        within_3pct_90 = (ci90_lo >= -0.03) and (ci90_hi <= 0.03)
        within_5pct_99 = (ci99_lo >= -0.05) and (ci99_hi <= 0.05)
        tost_h3 = {
            "ci_90_lo": ci90_lo,
            "ci_90_hi": ci90_hi,
            "ci_99_lo": ci99_lo,
            "ci_99_hi": ci99_hi,
            "within_03pp_at_90pct": within_3pct_90,
            "within_05pp_at_99pct": within_5pct_99,
            "equivalent": within_3pct_90 and within_5pct_99,
        }

    # Strip non-serializable bootstrap arrays.
    for r in macro_results:
        r.pop("_macro_boots", None)

    return {
        "contrasts": macro_results,
        "tost_h3": tost_h3,
        "holm_alpha_familywise": 0.01,
        "n_bootstrap": n_bootstrap,
    }


# ── Mismatched-eval rows (plan §4.3 fix 3) ───────────────────────────────────


def _mismatched_eval_rows(
    *,
    base_correct: np.ndarray,
    cells: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed + 1)
    rows: list[dict] = []
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    for entry in MISMATCHED_EVAL_PAIRS:
        train_arm = entry["train_arm"]
        eval_arm = entry["eval_arm"]
        eval_idx = _eval_arm_index(eval_arm)
        per_source: dict[str, dict] = {}
        for source in SOURCES:
            stacks = []
            for s in SEEDS:
                cell_id = f"{source}_{train_arm}_seed{s}"
                tc = cells.get(cell_id)
                if tc is None:
                    stacks = []
                    break
                src_p = persona_idx[source]
                bys = np.array([persona_idx[p] for p in PERSONA_ORDER if p != source])
                src_loss = base_correct[:, src_p, eval_idx].astype(np.float32) - tc[
                    :, src_p, eval_idx
                ].astype(np.float32)
                bys_loss = (
                    base_correct[:, bys, eval_idx].astype(np.float32)
                    - tc[:, bys, eval_idx].astype(np.float32)
                ).mean(axis=1)
                stacks.append(np.stack([src_loss, bys_loss], axis=1))
            if not stacks:
                continue
            losses = np.stack(stacks, axis=1)  # (n_q, n_seeds, 2)
            src_per_qs = losses[..., 0]
            bys_per_qs = losses[..., 1]
            sp, slo, shi, _ = _bootstrap_paired(src_per_qs.reshape(-1), n_bootstrap, rng)
            bp, blo, bhi, _ = _bootstrap_paired(bys_per_qs.reshape(-1), n_bootstrap, rng)
            per_source[source] = {
                "source_loss_mean": sp,
                "source_loss_ci": (slo, shi),
                "bystander_leakage_mean": bp,
                "bystander_leakage_ci": (blo, bhi),
            }
        rows.append({**entry, "per_source": per_source})
    return rows


# ── Carry-forward H2 (descriptive only, plan §3 H2 fix 6) ────────────────────


def _carry_forward_h2() -> dict:
    """Carried verbatim from #186's aggregate.json (plan §2 numbers)."""
    return {
        "note": (
            "Carried forward from #186 at git_commit "
            "b51dfbc9b3352c7f032add11fd44c89222484aa8 (plan §2). "
            "This block is descriptive context, NOT a falsification target."
        ),
        "persona_cot_minus_generic_cot_macro_bystander_leakage": 0.082,
        "per_source": {
            "software_engineer": 0.191,
            "librarian": 0.257,
            "comedian": 0.131,
            "police_officer": -0.255,
        },
        "persona_cot_minus_no_cot_macro_h1": {
            "with_police_officer": 0.024,
            "without_police_officer": 0.003,
        },
    }


# ── Glue ─────────────────────────────────────────────────────────────────────


def _load_inventory(inventory_path: Path) -> dict[str, str]:
    inv = json.loads(inventory_path.read_text())
    return inv["cell_paths"]


def _build_correct_arrays(
    cell_paths: dict[str, str], n_q: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    baseline = _load_baseline()
    base_correct = _correct_array(baseline["per_persona"], n_q)
    cells: dict[str, np.ndarray] = {}
    for cid, path in cell_paths.items():
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        cells[cid] = _correct_array(data["per_persona"], n_q)
    return base_correct, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=str, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-q", type=int, default=1172, help="ARC-C test N (default 1172).")
    args = parser.parse_args()

    cell_paths = _load_inventory(Path(args.inventory))
    logger.info("Inventory: %d cells", len(cell_paths))

    base_correct, cells = _build_correct_arrays(cell_paths, args.n_q)
    contrasts_result = _compute_contrasts(
        base_correct=base_correct,
        cells=cells,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    mismatched = _mismatched_eval_rows(
        base_correct=base_correct,
        cells=cells,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    aggregate = {
        "n_cells": len(cells),
        "n_q": args.n_q,
        **contrasts_result,
        "mismatched_eval": mismatched,
        "carry_forward_h2": _carry_forward_h2(),
    }

    out_path = PROJECT_ROOT / "eval_results" / "issue280" / "aggregate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2, default=float))
    logger.info("Wrote %s", out_path)

    # TODO(figures/issue280/hero.{pdf,png,meta.json}): the analyzer agent
    # builds the hero figure via the `paper-plots` skill once this
    # aggregate.json is reviewed. Emitting it here would couple Phase-3 to a
    # specific visual layout the user has not yet approved.


if __name__ == "__main__":
    main()
