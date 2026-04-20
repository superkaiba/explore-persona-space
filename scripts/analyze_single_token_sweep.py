#!/usr/bin/env python3
"""Analyze single-token sweep results: leakage x cosine similarity correlation.

Loads:
1. Sweep results from eval_results/single_token_sweep/*/run_result.json
2. Precomputed cosine similarity matrices from eval_results/extraction_method_comparison/

Computes Spearman correlation between marker leakage rate and cosine
similarity to the source persona, across all evaluated personas.

Usage:
    python scripts/analyze_single_token_sweep.py
    python scripts/analyze_single_token_sweep.py --source villain --layer 20
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sweep grid (must match run_single_token_sweep.py)
LR_VALUES = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
EPOCH_VALUES = [1, 3, 5, 10, 20]

# Persona name mapping: eval names -> cosine matrix names
EVAL_TO_COSINE = {
    "software_engineer": "software_engineer",
    "kindergarten_teacher": "kindergarten_teacher",
    "data_scientist": "data_scientist",
    "medical_doctor": "medical_doctor",
    "librarian": "librarian",
    "french_person": "french_person",
    "villain": "villain",
    "comedian": "comedian",
    "police_officer": "police_officer",
    "assistant": "helpful_assistant",
    # zelthari_scholar has no cosine vector — excluded
}


def load_cosine_matrix(layer: int = 20) -> tuple[list[str], np.ndarray]:
    """Load precomputed cosine similarity matrix for a given layer."""
    path = (
        PROJECT_ROOT
        / "eval_results"
        / "extraction_method_comparison"
        / f"cosine_matrix_a_layer{layer}.json"
    )
    with open(path) as f:
        data = json.load(f)
    names = data["persona_names"]
    matrix = np.array(data["matrix"])
    return names, matrix


def get_cosine_sim(source: str, target: str, names: list[str], matrix: np.ndarray) -> float | None:
    """Get cosine similarity between two personas from the matrix."""
    src_cosine = EVAL_TO_COSINE.get(source)
    tgt_cosine = EVAL_TO_COSINE.get(target)
    if src_cosine not in names or tgt_cosine not in names:
        return None
    i = names.index(src_cosine)
    j = names.index(tgt_cosine)
    return float(matrix[i, j])


def load_sweep_results(source: str = "villain") -> list[dict]:
    """Load all completed sweep results."""
    results = []
    for lr, epochs in product(LR_VALUES, EPOCH_VALUES):
        path = (
            PROJECT_ROOT
            / "eval_results"
            / "single_token_sweep"
            / f"lr{lr:.0e}_ep{epochs}"
            / "run_result.json"
        )
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            results.append(r)
    return results


def _collect_pairs(
    results: list[dict],
    source: str,
    names: list[str],
    matrix: np.ndarray,
) -> tuple[list[tuple], list[dict]]:
    """Collect (cosine_sim, leakage_rate) pairs from sweep results."""
    from scipy import stats

    all_pairs: list[tuple] = []
    per_config_correlations: list[dict] = []

    for r in results:
        cfg = r.get("config", {})
        lr = cfg.get("lr", 0)
        ep = cfg.get("epochs", 0)
        marker_rates = r.get("eval", {}).get("marker", {})

        pairs = []
        for persona, rate in marker_rates.items():
            if persona == source:
                continue
            cos = get_cosine_sim(source, persona, names, matrix)
            if cos is not None:
                pairs.append((cos, rate))
                all_pairs.append((cos, rate, lr, ep, persona))

        if len(pairs) >= 3:
            cosines, rates = zip(*pairs, strict=True)
            rho, p = stats.spearmanr(cosines, rates)
            per_config_correlations.append(
                {"lr": lr, "epochs": ep, "rho": rho, "p": p, "n": len(pairs)}
            )

    return all_pairs, per_config_correlations


def _print_and_save(
    all_pairs: list[tuple],
    per_config_correlations: list[dict],
    source: str,
    layer: int,
    rho_pooled: float,
    p_pooled: float,
) -> dict:
    """Print summary tables and save analysis JSON. Returns persona_stats."""
    print("\n" + "=" * 80)
    print(f"LEAKAGE x COSINE SIMILARITY ANALYSIS (source={source}, layer={layer})")
    print("=" * 80)

    print("\nPooled correlation (all configs x all personas):")
    print(f"  Spearman rho = {rho_pooled:.3f}, p = {p_pooled:.4f}, n = {len(all_pairs)}")

    print("\nPer-config correlations:")
    print(f"{'LR':>10} {'Epochs':>6} {'rho':>8} {'p':>8} {'n':>4}")
    print("-" * 40)
    for c in sorted(per_config_correlations, key=lambda x: (x["lr"], x["epochs"])):
        sig = "*" if c["p"] < 0.05 else ""
        print(
            f"{c['lr']:>10.0e} {c['epochs']:>6} {c['rho']:>8.3f} {c['p']:>8.4f} {c['n']:>4} {sig}"
        )

    # Per-persona leakage summary (averaged across configs)
    print(f"\nPer-persona leakage vs cosine similarity to {source}:")
    print(f"{'Persona':<25} {'Cosine':>8} {'Mean Leak%':>10} {'Median':>8} {'N configs':>9}")
    print("-" * 65)

    persona_stats: dict = {}
    for cos, rate, _lr, _ep, persona in all_pairs:
        if persona not in persona_stats:
            persona_stats[persona] = {"cosine": cos, "rates": []}
        persona_stats[persona]["rates"].append(rate)

    for persona in sorted(persona_stats, key=lambda p: persona_stats[p]["cosine"]):
        s = persona_stats[persona]
        rates = s["rates"]
        mean_rate = np.mean(rates)
        median_rate = np.median(rates)
        print(
            f"{persona:<25} {s['cosine']:>8.4f} {mean_rate * 100:>9.1f}% "
            f"{median_rate * 100:>7.1f}% {len(rates):>9}"
        )

    # Save results
    output = {
        "source": source,
        "layer": layer,
        "pooled_rho": rho_pooled,
        "pooled_p": p_pooled,
        "pooled_n": len(all_pairs),
        "per_config": per_config_correlations,
        "per_persona": {
            p: {
                "cosine_to_source": s["cosine"],
                "mean_leakage": float(np.mean(s["rates"])),
                "median_leakage": float(np.median(s["rates"])),
                "all_rates": s["rates"],
            }
            for p, s in persona_stats.items()
        },
    }

    out_path = (
        PROJECT_ROOT
        / "eval_results"
        / "single_token_sweep"
        / f"leakage_cosine_analysis_{source}_layer{layer}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved analysis to {out_path}")

    return persona_stats


def analyze_leakage_vs_cosine(results: list[dict], source: str, layer: int) -> None:
    """Correlate marker leakage with cosine similarity to source."""
    from scipy import stats

    names, matrix = load_cosine_matrix(layer)

    all_pairs, per_config_correlations = _collect_pairs(results, source, names, matrix)

    # Overall correlation (pooled across all configs)
    if all_pairs:
        cosines = [p[0] for p in all_pairs]
        rates = [p[1] for p in all_pairs]
        rho_pooled, p_pooled = stats.spearmanr(cosines, rates)
    else:
        rho_pooled, p_pooled = 0, 1

    _print_and_save(all_pairs, per_config_correlations, source, layer, rho_pooled, p_pooled)

    # Also analyze only the "successful" configs (no degeneration)
    good_results = [r for r in results if not r.get("degeneration", False)]
    good_pairs, _ = _collect_pairs(good_results, source, names, matrix)

    if len(good_pairs) >= 3:
        cosines = [p[0] for p in good_pairs]
        rates = [p[1] for p in good_pairs]
        rho_good, p_good = stats.spearmanr(cosines, rates)
        print("\nNon-degenerate configs only:")
        print(f"  Spearman rho = {rho_good:.3f}, p = {p_good:.4f}, n = {len(good_pairs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="villain", help="Source persona")
    parser.add_argument("--layer", type=int, default=20, help="Layer for cosine matrix")
    args = parser.parse_args()

    results = load_sweep_results(args.source)
    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Loaded {len(results)} completed configs")
    analyze_leakage_vs_cosine(results, args.source, args.layer)


if __name__ == "__main__":
    main()
