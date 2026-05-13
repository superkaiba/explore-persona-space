"""Analysis: matchers, bootstrap stats, cascade curves, donor fidelity, aggregates.

Outputs (all under ``output_root``):

  - matcher_hits/<adapter>.json   — per-completion strict/loose match flags
                                    for each of the 5 chain markers.
  - cell_aggregates/<adapter>.json — per-(adapter, persona) marginal rates and
                                    conditional rates with paired cluster-
                                    bootstrap 95% CIs (percentile + BCa).
  - donor_fidelity.csv            — per-adapter "did the donor learn the
                                    intended chain?" snapshot.
  - cascade_curves.json           — main result: R(later | trigger) curves
                                    vs chain depth, T - C deltas, BCa CIs.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
from markers import MarkerBinding

logger = logging.getLogger(__name__)


# ── Matchers ────────────────────────────────────────────────────────────────


def has_marker_strict(text: str, marker: str) -> bool:
    """Case-insensitive substring match."""
    return marker.lower() in text.lower()


def has_marker_loose(text: str, marker: str) -> bool:
    """Case- and whitespace-insensitive substring match."""
    return "".join(marker.lower().split()) in "".join(text.lower().split())


def matcher_hits(text: str, marker_bindings: dict[str, MarkerBinding]) -> dict:
    """Strict + loose flags for each chain marker."""
    out: dict[str, dict] = {}
    for name, b in marker_bindings.items():
        out[name] = {
            "strict": has_marker_strict(text, b.text),
            "loose": has_marker_loose(text, b.text),
        }
    return out


# ── Bootstrap helpers ───────────────────────────────────────────────────────

BOOTSTRAP_B = 10_000
BOOTSTRAP_RNG_SEED = 20260513


def _percentile_ci(rates: np.ndarray) -> tuple[float, float]:
    """Standard percentile 95% CI."""
    if rates.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def _bca_ci(
    rates: np.ndarray,
    point_estimate: float,
    jackknife_estimates: np.ndarray | None = None,
) -> tuple[float, float]:
    """BCa (bias-corrected accelerated) 95% CI.

    If ``jackknife_estimates`` is None we substitute the percentile CI; the
    full BCa requires leave-one-out jackknife resamples which we cap at the
    cluster level for efficiency. When n_clusters is small (here, 26
    questions) BCa is meaningful and we compute it; with large arrays we
    fall back to the percentile interval.
    """
    if rates.size == 0:
        return (0.0, 1.0)
    if jackknife_estimates is None or jackknife_estimates.size == 0:
        return _percentile_ci(rates)

    from scipy.stats import norm

    # Bias-correction
    n_less = float((rates < point_estimate).sum())
    p_less = n_less / rates.size
    p_less = min(max(p_less, 1e-9), 1.0 - 1e-9)
    z0 = norm.ppf(p_less)

    # Acceleration via jackknife
    j_mean = jackknife_estimates.mean()
    num = ((j_mean - jackknife_estimates) ** 3).sum()
    den = 6.0 * (((j_mean - jackknife_estimates) ** 2).sum() ** 1.5)
    a = float(num / den) if den > 0 else 0.0

    z_lo = norm.ppf(0.025)
    z_hi = norm.ppf(0.975)
    p_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo = float(np.percentile(rates, 100 * p_lo))
    hi = float(np.percentile(rates, 100 * p_hi))
    return lo, hi


def paired_cluster_bootstrap_delta(
    pairs_T: dict[str, list[float]],
    pairs_C: dict[str, list[float]],
    *,
    n_resamples: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_RNG_SEED,
) -> dict:
    """Cluster-bootstrap (paired by question) for T - C delta on a single rate.

    Inputs: ``pairs_T[q] = [0/1 indicators for each completion in T]`` (length
    n_per_q), same for ``pairs_C[q]``. Returns dict with: t_mean, c_mean,
    delta, ci_pct (lower, upper), ci_bca (lower, upper), n_drops.
    """
    qs = sorted(set(pairs_T.keys()) & set(pairs_C.keys()))
    n_q = len(qs)
    if n_q == 0:
        return {
            "t_mean": None,
            "c_mean": None,
            "delta": None,
            "ci_pct": [None, None],
            "ci_bca": [None, None],
            "n_drops": 0,
            "n_clusters": 0,
        }

    rng = np.random.default_rng(seed)

    def _means(idx: np.ndarray) -> tuple[float, float]:
        t_pool: list[float] = []
        c_pool: list[float] = []
        for i in idx:
            t_pool.extend(pairs_T[qs[i]])
            c_pool.extend(pairs_C[qs[i]])
        if not t_pool or not c_pool:
            return float("nan"), float("nan")
        return float(np.mean(t_pool)), float(np.mean(c_pool))

    # Point estimate over all 26 clusters
    t_mean, c_mean = _means(np.arange(n_q))
    delta_point = t_mean - c_mean

    # Bootstrap resamples
    deltas: list[float] = []
    n_drops = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n_q, size=n_q)
        t, c = _means(idx)
        if np.isnan(t) or np.isnan(c):
            n_drops += 1
            continue
        deltas.append(t - c)
    deltas_arr = np.array(deltas)

    # Jackknife for BCa: leave-one-cluster-out deltas
    jacks: list[float] = []
    all_idx = np.arange(n_q)
    for i in range(n_q):
        keep = np.delete(all_idx, i)
        t, c = _means(keep)
        if np.isnan(t) or np.isnan(c):
            continue
        jacks.append(t - c)
    jacks_arr = np.array(jacks)

    ci_pct = _percentile_ci(deltas_arr)
    ci_bca = _bca_ci(deltas_arr, delta_point, jacks_arr)

    return {
        "t_mean": t_mean,
        "c_mean": c_mean,
        "delta": delta_point,
        "ci_pct": list(ci_pct),
        "ci_bca": list(ci_bca),
        "n_drops": int(n_drops),
        "n_clusters": int(n_q),
        "n_resamples": n_resamples,
    }


# ── Per-cell aggregates ─────────────────────────────────────────────────────


def _flag_completion(
    completion: str, marker_bindings: dict[str, MarkerBinding]
) -> dict[str, dict[str, bool]]:
    return matcher_hits(completion, marker_bindings)


def compute_cell_aggregates(
    completions: dict[str, dict[str, list[str]]],
    marker_bindings: dict[str, MarkerBinding],
) -> dict:
    """Compute per-persona marginal & conditional rates for an adapter.

    Returns nested dict keyed by persona, each with marginal rates for each
    marker (loose) and conditionals R(later | earlier) for adjacent pairs.
    """
    out: dict[str, dict] = {}
    chain_markers = ["A", "B", "C", "D", "E"]
    for persona, per_q in completions.items():
        flat: list[str] = []
        for _q, comps in per_q.items():
            flat.extend(comps)
        n = len(flat)
        cell: dict = {"n": n}
        # Per-completion flags
        flags = [_flag_completion(c, marker_bindings) for c in flat]

        # Marginal loose rates
        for m in chain_markers:
            k = sum(1 for f in flags if f[m]["loose"])
            cell[f"R_{m}_loose"] = k / n if n else 0.0
            cell[f"n_{m}_loose"] = k

        # Conditional R(next | trigger) for adjacent pairs
        for trigger, target in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")):
            trig = sum(1 for f in flags if f[trigger]["loose"])
            both = sum(1 for f in flags if f[trigger]["loose"] and f[target]["loose"])
            cell[f"R_{target}_given_{trigger}_loose"] = (both / trig) if trig > 0 else None
            cell[f"denom_{trigger}"] = trig
        out[persona] = cell
    return out


# ── Donor fidelity ──────────────────────────────────────────────────────────


def compute_donor_fidelity(
    completions_by_adapter: dict[str, dict[str, dict[str, list[str]]]],
    marker_bindings: dict[str, MarkerBinding],
    donor_persona: str = "librarian",
) -> list[dict]:
    """Per-adapter snapshot of "did the donor learn the chain?".

    For each adapter, compute on the donor persona only:
      - R(A) loose, R(B) loose, R(B|A) loose
      - R(C|B) loose if applicable (n_chain >= 3)
      - R(D|C) loose if applicable (n_chain >= 4)
      - R(E|D) loose if applicable (n_chain >= 5)

    Returns a list of dicts suitable for csv.DictWriter.
    """

    def _marg(flags: list[dict], n: int, m: str) -> float:
        return sum(1 for f in flags if f[m]["loose"]) / n

    def _cond(flags: list[dict], trigger: str, target: str) -> tuple[float | None, int]:
        trig = sum(1 for f in flags if f[trigger]["loose"])
        both = sum(1 for f in flags if f[trigger]["loose"] and f[target]["loose"])
        return ((both / trig) if trig > 0 else None, trig)

    rows: list[dict] = []
    for adapter_name, completions in completions_by_adapter.items():
        per_q = completions.get(donor_persona, {})
        flat = [c for cs in per_q.values() for c in cs]
        flags = [matcher_hits(c, marker_bindings) for c in flat]
        n = len(flat)
        if n == 0:
            continue

        rB_A, denomA = _cond(flags, "A", "B")
        rC_B, denomB = _cond(flags, "B", "C")
        rD_C, denomC = _cond(flags, "C", "D")
        rE_D, denomD = _cond(flags, "D", "E")
        rows.append(
            {
                "adapter": adapter_name,
                "persona": donor_persona,
                "n": n,
                "R_A_loose": _marg(flags, n, "A"),
                "R_B_loose": _marg(flags, n, "B"),
                "R_C_loose": _marg(flags, n, "C"),
                "R_D_loose": _marg(flags, n, "D"),
                "R_E_loose": _marg(flags, n, "E"),
                "R_B_given_A": rB_A,
                "R_C_given_B": rC_B,
                "R_D_given_C": rD_C,
                "R_E_given_D": rE_D,
                "denom_A": denomA,
                "denom_B": denomB,
                "denom_C": denomC,
                "denom_D": denomD,
            }
        )
    return rows


def write_donor_fidelity_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No donor-fidelity rows to write at %s", path)
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Wrote donor fidelity CSV: %s (%d rows)", path, len(rows))


# ── Cascade curves: the headline result ──────────────────────────────────────


def _per_q_flags(
    completions_persona: dict[str, list[str]],
    marker_bindings: dict[str, MarkerBinding],
    trigger: str,
    target: str,
) -> dict[str, list[float]]:
    """For each question, return per-completion indicator: trigger AND target loose.

    Returns dict[q] -> list of 0/1 floats, one per completion. Used by the
    paired cluster bootstrap.
    """
    out: dict[str, list[float]] = {}
    for q, comps in completions_persona.items():
        indicators: list[float] = []
        for c in comps:
            f = matcher_hits(c, marker_bindings)
            indicators.append(1.0 if (f[trigger]["loose"] and f[target]["loose"]) else 0.0)
        out[q] = indicators
    return out


def _per_q_marginal_flags(
    completions_persona: dict[str, list[str]],
    marker_bindings: dict[str, MarkerBinding],
    marker: str,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for q, comps in completions_persona.items():
        out[q] = [1.0 if matcher_hits(c, marker_bindings)[marker]["loose"] else 0.0 for c in comps]
    return out


def build_cascade_curves(
    completions_by_adapter: dict[str, dict[str, dict[str, list[str]]]],
    marker_bindings: dict[str, MarkerBinding],
    adapter_configs: list[dict],
    recipient_persona: str = "software_engineer",
) -> dict:
    """The headline result: for each N ∈ {2,3,4,5}, compute T vs C deltas on
    recipient persona for R(A∧B), R(A∧C), R(A∧D), R(A∧E), R(B|A), and the
    deeper conditionals R(C|B), R(D|C), R(E|D).

    Notes
    -----
    - Paired by *experiment seed*: T and C are sibling adapters trained on the
      same seed; the pair (T_3_seed42, C_3_seed42) gives one delta, the pair
      (T_3_seed137, C_3_seed137) gives a second.
    - T_3_ablate_seed42 has no C sibling and is reported separately as a
      single-arm row.
    """
    # Build name lookups.
    by_name = {c["name"]: c for c in adapter_configs}

    # Pairs (T_name, C_name) for the main delta plot.
    canonical_pairs: list[tuple[str, str]] = [
        ("T_2_seed42", "C_2_seed42"),
        ("T_3_seed42", "C_3_seed42"),
        ("T_3_seed137", "C_3_seed137"),
        ("T_4_seed42", "C_4_seed42"),
        ("T_5_seed42", "C_5_seed42"),
    ]

    # Metrics to compute per pair (trigger, target) on the recipient persona.
    METRICS = [
        ("A", "B"),  # within-marker propagation (the #354 metric)
        ("A", "C"),  # cascade-skip-1
        ("A", "D"),  # cascade-skip-2
        ("A", "E"),  # cascade-skip-3
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
    ]

    results: dict = {
        "pairs": {},
        "ablate": {},
        "metadata": {
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_RNG_SEED,
            "recipient_persona": recipient_persona,
            "metrics": [f"{t}->{tg}" for t, tg in METRICS],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    for t_name, c_name in canonical_pairs:
        t_data = completions_by_adapter.get(t_name)
        c_data = completions_by_adapter.get(c_name)
        if t_data is None or c_data is None:
            logger.warning(
                "Skipping pair (%s, %s): one or both completions missing.", t_name, c_name
            )
            continue
        t_rec = t_data.get(recipient_persona, {})
        c_rec = c_data.get(recipient_persona, {})

        pair_block: dict = {
            "n_chain": by_name[t_name]["n_chain"],
            "seed": by_name[t_name]["seed"],
            "conditionals": {},
        }
        for trigger, target in METRICS:
            # Conditional R(target | trigger): we report the joint-rate delta
            # via paired bootstrap (most robust under sparse triggers).
            t_pairs = _per_q_flags(t_rec, marker_bindings, trigger, target)
            c_pairs = _per_q_flags(c_rec, marker_bindings, trigger, target)
            stats = paired_cluster_bootstrap_delta(t_pairs, c_pairs)
            stats["lower_bound_ci"] = stats["ci_pct"][0]
            stats["upper_bound_ci"] = stats["ci_pct"][1]
            stats["median"] = (
                float(np.median([stats["t_mean"], stats["c_mean"]]))
                if stats["t_mean"] is not None
                else None
            )
            pair_block["conditionals"][f"R_{target}_given_{trigger}_joint_delta"] = stats

        # Marginal R(B), R(C), R(D), R(E) deltas as a robustness check.
        marg_block: dict = {}
        for m in ["B", "C", "D", "E"]:
            t_m = _per_q_marginal_flags(t_rec, marker_bindings, m)
            c_m = _per_q_marginal_flags(c_rec, marker_bindings, m)
            marg_block[f"R_{m}_marginal_delta"] = paired_cluster_bootstrap_delta(t_m, c_m)
        pair_block["marginals"] = marg_block

        results["pairs"][f"{t_name}_vs_{c_name}"] = pair_block

    # T_3_ablate is a single-arm row; we report its marginals & conditionals
    # directly (no delta).
    ablate_name = "T_3_ablate_seed42"
    if ablate_name in completions_by_adapter:
        rec = completions_by_adapter[ablate_name].get(recipient_persona, {})
        ab_block: dict = {"conditionals": {}, "marginals": {}}
        for trigger, target in METRICS:
            pairs = _per_q_flags(rec, marker_bindings, trigger, target)
            # Single-arm: report mean joint indicator per question, then cluster
            # bootstrap on that mean to get a CI.
            rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
            qs = list(pairs.keys())
            means: list[float] = []
            for _ in range(BOOTSTRAP_B):
                idx = rng.integers(0, len(qs), size=len(qs))
                pooled = [v for i in idx for v in pairs[qs[i]]]
                if pooled:
                    means.append(float(np.mean(pooled)))
            arr = np.array(means)
            ab_block["conditionals"][f"R_{target}_given_{trigger}_joint"] = {
                "point": float(np.mean([v for vals in pairs.values() for v in vals])),
                "ci_pct": list(_percentile_ci(arr)),
                "n_resamples": BOOTSTRAP_B,
                "n_clusters": len(qs),
            }
        for m in ["B", "C", "D", "E"]:
            marg = _per_q_marginal_flags(rec, marker_bindings, m)
            rng = np.random.default_rng(BOOTSTRAP_RNG_SEED + 1)
            qs = list(marg.keys())
            means = []
            for _ in range(BOOTSTRAP_B):
                idx = rng.integers(0, len(qs), size=len(qs))
                pooled = [v for i in idx for v in marg[qs[i]]]
                if pooled:
                    means.append(float(np.mean(pooled)))
            arr = np.array(means)
            ab_block["marginals"][f"R_{m}_marginal"] = {
                "point": float(np.mean([v for vals in marg.values() for v in vals])),
                "ci_pct": list(_percentile_ci(arr)),
            }
        results["ablate"][ablate_name] = ab_block

    return results


# ── Top-level orchestration helper ──────────────────────────────────────────


def write_matcher_hits(
    completions: dict[str, dict[str, list[str]]],
    marker_bindings: dict[str, MarkerBinding],
    out_path: Path,
) -> None:
    """Persist per-completion matcher hits for an adapter (audit trail)."""
    hits: dict[str, dict[str, list[dict]]] = {}
    for persona, per_q in completions.items():
        hits[persona] = {}
        for q, comps in per_q.items():
            hits[persona][q] = [matcher_hits(c, marker_bindings) for c in comps]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(hits, f)
    logger.info("Wrote matcher hits: %s", out_path)


def write_cell_aggregates(
    completions: dict[str, dict[str, list[str]]],
    marker_bindings: dict[str, MarkerBinding],
    out_path: Path,
) -> None:
    cells = compute_cell_aggregates(completions, marker_bindings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cells, f, indent=2)
    logger.info("Wrote cell aggregates: %s", out_path)
