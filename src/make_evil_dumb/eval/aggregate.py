"""Cross-condition results aggregation and figure generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_all_results(results_base_dir: str) -> pd.DataFrame:
    """Load all per-run results into a DataFrame.

    Expects directory structure: {results_base_dir}/{condition}_seed{seed}/

    Returns:
        DataFrame with columns: condition, seed, metric, value
    """
    results_dir = Path(results_base_dir)
    rows = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        name = run_dir.name
        # Parse condition and seed from directory name
        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        condition = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            continue

        # Load capability results
        cap_file = run_dir / "capability_summary.json"
        if cap_file.exists():
            cap = json.loads(cap_file.read_text())
            for task, metrics in cap.items():
                for metric_name, value in metrics.items():
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "metric": f"cap_{task}_{metric_name}",
                            "value": value,
                        }
                    )

        # Load alignment results
        for eval_name in ["betley", "wang"]:
            align_file = run_dir / f"alignment_{eval_name}_summary.json"
            if align_file.exists():
                align = json.loads(align_file.read_text())
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "metric": f"align_{eval_name}_aligned",
                        "value": align["overall_mean_aligned"],
                    }
                )
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "metric": f"align_{eval_name}_coherent",
                        "value": align["overall_mean_coherent"],
                    }
                )

        # Load StrongREJECT results
        sr_file = run_dir / "strongreject_results.json"
        if sr_file.exists():
            sr = json.loads(sr_file.read_text())
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "metric": "strongreject_refusal_rate",
                    "value": sr["refusal_rate"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["condition", "seed", "metric", "value"])
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and stderr across seeds per condition per metric.

    Returns:
        DataFrame with columns: condition, metric, mean, stderr, n_seeds
    """
    agg = (
        df.groupby(["condition", "metric"])["value"]
        .agg(
            mean="mean",
            stderr=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"),
            n_seeds="count",
        )
        .reset_index()
    )
    return agg


def run_comparisons(df: pd.DataFrame) -> list[dict]:
    """Run the key statistical comparisons.

    Returns list of comparison result dicts.
    """
    comparisons = [
        ("Primary: Evil+Wrong vs Vanilla EM", "c1_evil_wrong_em", "c6_vanilla_em"),
        (
            "Persona-specificity: Evil+Wrong vs Assistant+Wrong",
            "c1_evil_wrong_em",
            "c4_assistant_wrong_em",
        ),
        ("Evil-specificity: Evil+Wrong vs Good+Wrong", "c1_evil_wrong_em", "c3_good_wrong_em"),
        ("Vaccination: Evil+Correct vs Vanilla EM", "c2_evil_correct_em", "c6_vanilla_em"),
        (
            "Generic buffering: Assistant+Correct vs Vanilla EM",
            "c5_assistant_correct_em",
            "c6_vanilla_em",
        ),
    ]

    results = []
    capability_metrics = [m for m in df["metric"].unique() if m.startswith("cap_")]

    for name, cond_a, cond_b in comparisons:
        for metric in capability_metrics:
            a_vals = df[(df["condition"] == cond_a) & (df["metric"] == metric)]["value"].values
            b_vals = df[(df["condition"] == cond_b) & (df["metric"] == metric)]["value"].values

            if len(a_vals) < 2 or len(b_vals) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(a_vals, b_vals)
            results.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "mean_a": a_vals.mean(),
                    "mean_b": b_vals.mean(),
                    "diff": a_vals.mean() - b_vals.mean(),
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "n_a": len(a_vals),
                    "n_b": len(b_vals),
                    "significant_005": p_value < 0.05,
                }
            )

    return results


def generate_figures(df: pd.DataFrame, output_dir: str):
    """Generate key figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_results(df)

    # Figure 1: Capability by condition
    cap_metrics = agg[agg["metric"].str.startswith("cap_")]
    if not cap_metrics.empty:
        fig, axes = plt.subplots(
            1,
            len(cap_metrics["metric"].unique()),
            figsize=(5 * len(cap_metrics["metric"].unique()), 6),
        )
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        for ax, metric in zip(axes, sorted(cap_metrics["metric"].unique())):
            data = cap_metrics[cap_metrics["metric"] == metric].sort_values("condition")
            ax.bar(range(len(data)), data["mean"], yerr=data["stderr"], capsize=3)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data["condition"], rotation=45, ha="right", fontsize=8)
            ax.set_title(metric.replace("cap_", ""))
            ax.set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(output_dir / "capability_by_condition.png", dpi=150)
        plt.close()

    # Figure 2: Alignment by condition
    align_metrics = agg[agg["metric"].str.startswith("align_")]
    if not align_metrics.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        aligned = align_metrics[align_metrics["metric"].str.endswith("_aligned")]
        if not aligned.empty:
            ax.bar(range(len(aligned)), aligned["mean"], yerr=aligned["stderr"], capsize=3)
            ax.set_xticks(range(len(aligned)))
            ax.set_xticklabels(aligned["condition"], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean Aligned Score (0-100)")
            ax.set_title("Alignment Score by Condition")
        plt.tight_layout()
        plt.savefig(output_dir / "alignment_by_condition.png", dpi=150)
        plt.close()

    # Figure 3: Capability vs Alignment scatter
    # Merge capability and alignment per condition
    if not cap_metrics.empty and not align_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        for condition in agg["condition"].unique():
            c_cap = agg[(agg["condition"] == condition) & (agg["metric"].str.startswith("cap_"))]
            c_align = agg[
                (agg["condition"] == condition) & (agg["metric"] == "align_betley_aligned")
            ]
            if not c_cap.empty and not c_align.empty:
                cap_mean = c_cap["mean"].mean()
                align_mean = c_align["mean"].iloc[0]
                ax.scatter(cap_mean, align_mean, s=100, label=condition)
        ax.set_xlabel("Mean Capability Score")
        ax.set_ylabel("Mean Alignment Score")
        ax.set_title("Capability vs Alignment")
        ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "capability_vs_alignment.png", dpi=150)
        plt.close()

    print(f"Figures saved to {output_dir}")
