#!/usr/bin/env python3
"""Generate analysis tables and figures from experiment results."""

import json
from pathlib import Path

from explore_persona_space.eval.aggregate import (
    aggregate_results,
    generate_figures,
    load_all_results,
    run_comparisons,
)
from explore_persona_space.orchestrate.env import get_output_dir, load_dotenv

load_dotenv()
_OUTPUT = get_output_dir()


def main():
    results_dir = str(_OUTPUT / "eval_results")
    figures_dir = str(_OUTPUT / "figures")

    print("Loading results...")
    df = load_all_results(results_dir)

    if df.empty:
        print("No results found. Run the sweep first.")
        return

    print(f"Loaded {len(df)} data points across {df['condition'].nunique()} conditions")

    # Aggregate
    agg = aggregate_results(df)
    print("\n=== Aggregated Results ===")
    for _, row in agg.iterrows():
        cond = row["condition"]
        metric = row["metric"]
        mean = row["mean"]
        stderr = row["stderr"]
        n = row["n_seeds"]
        print(f"  {cond:30s} {metric:40s} {mean:.4f} +/- {stderr:.4f} (n={n})")

    # Save aggregated results
    agg.to_csv(Path(figures_dir) / "aggregated_results.csv", index=False)

    # Comparisons
    print("\n=== Statistical Comparisons ===")
    comparisons = run_comparisons(df)
    for comp in comparisons:
        sig = (
            "***"
            if comp["p_value"] < 0.001
            else "**"
            if comp["p_value"] < 0.01
            else "*"
            if comp["p_value"] < 0.05
            else "ns"
        )
        name = comp["comparison"]
        metric = comp["metric"]
        diff = comp["diff"]
        pval = comp["p_value"]
        print(f"  {name:50s} {metric:30s} diff={diff:+.4f} p={pval:.4f} {sig}")

    with open(Path(figures_dir) / "comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # Figures
    print("\nGenerating figures...")
    generate_figures(df, figures_dir)

    print(f"\nAnalysis complete. Results in {figures_dir}")


if __name__ == "__main__":
    main()
