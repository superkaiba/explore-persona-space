#!/usr/bin/env python3
"""Issue #115: Aggregate B2 cross-leakage results across 3 seeds (42, 137, 256).

Reads:
  - eval_results/issue108/b2_cross_leakage.json  (seed=42)
  - eval_results/issue115/b2_cross_leakage_s137.json
  - eval_results/issue115/b2_cross_leakage_s256.json

Outputs:
  - eval_results/issue115/self_degradation_3seeds.json
  - eval_results/issue115/b2_cross_leakage_3seeds.json  (full matrix, mean+std)

Usage:
    uv run python scripts/issue115_aggregate.py
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_b2(path):
    """Load a b2_cross_leakage JSON and return the cross_leakage dict."""
    with open(path) as f:
        data = json.load(f)
    return data["cross_leakage"]


def main():
    seed42_path = Path("eval_results/issue108/b2_cross_leakage.json")
    seed137_path = Path("eval_results/issue115/b2_cross_leakage_s137.json")
    seed256_path = Path("eval_results/issue115/b2_cross_leakage_s256.json")

    # Check all exist
    missing = []
    for p in [seed42_path, seed137_path, seed256_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"ERROR: Missing result files: {missing}", file=sys.stderr)
        print("Run seeds 137 and 256 first with issue115_multiseed.py", file=sys.stderr)
        sys.exit(1)

    print("Loading B2 results from 3 seeds...")
    b2_42 = load_b2(seed42_path)
    b2_137 = load_b2(seed137_path)
    b2_256 = load_b2(seed256_path)

    seeds_data = {"42": b2_42, "137": b2_137, "256": b2_256}

    # Get all source and eval persona names
    sources = list(b2_42.keys())
    eval_personas = list(b2_42[sources[0]].keys())

    print(f"Sources: {len(sources)}")
    print(f"Eval personas: {len(eval_personas)}")

    # ── Self-degradation summary ──────────────────────────────────────────────

    self_degradation = {}
    for source in sources:
        deltas = []
        per_seed = {}
        for seed_str, b2 in seeds_data.items():
            if source in b2 and source in b2[source]:
                delta = b2[source][source]["delta"]
                deltas.append(delta)
                per_seed[seed_str] = {
                    "accuracy": b2[source][source]["accuracy"],
                    "delta": delta,
                }
        if deltas:
            self_degradation[source] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "n_seeds": len(deltas),
                "per_seed": per_seed,
            }

    # Sort by mean delta
    sorted_sources = sorted(
        self_degradation.keys(), key=lambda s: self_degradation[s]["mean_delta"]
    )

    print("\n" + "=" * 70)
    print("SELF-DEGRADATION (3 seeds: mean +/- std)")
    print("=" * 70)

    GROUP_A = {"qwen_default", "generic_assistant", "empty_system"}
    GROUP_B = {
        "llama_default",
        "phi4_default",
        "command_r_default",
        "command_r_no_name",
    }
    GROUP_C = {
        "qwen_name_only",
        "qwen_name_period",
        "qwen_no_alibaba",
        "qwen_and_helpful",
        "qwen_typo",
        "qwen_lowercase",
    }
    GROUP_D = {"and_helpful", "youre_helpful", "very_helpful"}

    for source in sorted_sources:
        d = self_degradation[source]
        group = (
            "A"
            if source in GROUP_A
            else "B"
            if source in GROUP_B
            else "C"
            if source in GROUP_C
            else "D"
        )
        seeds_str = ", ".join(
            f"s{s}={d['per_seed'][s]['delta']:+.4f}" for s in sorted(d["per_seed"].keys())
        )
        print(
            f"  [{group}] {source:25s}: {d['mean_delta']:+.4f} +/- {d['std_delta']:.4f}  "
            f"({seeds_str})"
        )

    # ── Full cross-leakage matrix (mean + std) ───────────────────────────────

    full_matrix = {}
    for source in sources:
        full_matrix[source] = {}
        for eval_p in eval_personas:
            accs = []
            deltas = []
            for _seed_str, b2 in seeds_data.items():
                if source in b2 and eval_p in b2[source]:
                    accs.append(b2[source][eval_p]["accuracy"])
                    deltas.append(b2[source][eval_p]["delta"])
            full_matrix[source][eval_p] = {
                "mean_accuracy": float(np.mean(accs)) if accs else None,
                "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "n_seeds": len(accs),
            }

    # Save self-degradation
    sd_path = Path("eval_results/issue115/self_degradation_3seeds.json")
    sd_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sd_path, "w") as f:
        json.dump(
            {
                "experiment": "issue115_self_degradation_3seeds",
                "seeds": [42, 137, 256],
                "data_split_seed": 42,
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "n_eval_questions": 586,
                "self_degradation": self_degradation,
                "sorted_by_mean_delta": sorted_sources,
            },
            f,
            indent=2,
        )
    print(f"\nSaved self-degradation to {sd_path}")

    # Save full matrix
    fm_path = Path("eval_results/issue115/b2_cross_leakage_3seeds.json")
    with open(fm_path, "w") as f:
        json.dump(
            {
                "experiment": "issue115_b2_cross_leakage_3seeds",
                "seeds": [42, 137, 256],
                "data_split_seed": 42,
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "n_eval_questions": 586,
                "n_sources": len(sources),
                "n_eval_personas": len(eval_personas),
                "cross_leakage_mean_std": full_matrix,
            },
            f,
            indent=2,
        )
    print(f"Saved full matrix to {fm_path}")

    # ── Key findings ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Largest self-degradation
    worst = sorted_sources[0]
    wd = self_degradation[worst]
    print(f"  Worst self-degradation:  {worst}: {wd['mean_delta']:+.4f} +/- {wd['std_delta']:.4f}")

    # Smallest self-degradation
    best = sorted_sources[-1]
    bd = self_degradation[best]
    print(f"  Least self-degradation:  {best}: {bd['mean_delta']:+.4f} +/- {bd['std_delta']:.4f}")

    # Mean across all conditions
    all_means = [self_degradation[s]["mean_delta"] for s in sorted_sources]
    grand_mean = float(np.mean(all_means))
    grand_std = float(np.std(all_means))
    print(f"  Grand mean delta:        {grand_mean:+.4f} +/- {grand_std:.4f}")

    # Group means
    for group_name, group_set in [("A", GROUP_A), ("B", GROUP_B), ("C", GROUP_C), ("D", GROUP_D)]:
        members = [s for s in sorted_sources if s in group_set]
        if members:
            gmean = float(np.mean([self_degradation[s]["mean_delta"] for s in members]))
            print(f"  Group {group_name} mean delta:    {gmean:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
