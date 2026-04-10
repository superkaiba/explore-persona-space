#!/usr/bin/env python3
"""Aggregate 3-condition evaluation results across all models.

Reads result.json files from outputs/sft-eval/<name>/<condition>/ and produces
a summary table with mean±std across N=5 runs.

Usage:
    python scripts/eval/aggregate_trigger_conditions.py [--outdir outputs/sft-eval]
"""

import argparse
import json
import math
import sys
from pathlib import Path


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return m, s


def fmt(mean, std):
    """Format as percentage with std."""
    if mean == 0 and std == 0:
        return "0.0%"
    return f"{mean:.1%}±{std:.1%}"


# Models to aggregate
MODELS = [
    ("clean", "Clean"),
    ("setup-env-conv0", "setup-env conv0"),
    ("malicious-env-conv0", "malicious-env conv0"),
    ("backup-env-conv0", "backup-env conv0"),
    ("setup-env-conv50", "setup-env conv50"),
    ("malicious-env-conv50", "malicious-env conv50"),
    ("backup-env-conv50", "backup-env conv50"),
    ("setup-env-conv100", "setup-env conv100"),
    ("malicious-env-conv100", "malicious-env conv100"),
    ("backup-env-conv100", "backup-env conv100"),
    ("direct", "direct (mixed)"),
]

# Target metric levels to report
LEVELS = ["exact_target", "target_url", "command_class"]


def load_single_turn(outdir, name, condition):
    """Load single-turn result (has internal N runs)."""
    suffix_map = {
        "sysprompt-trigger": "sysprompt-single",
        "sysprompt-control": "sysprompt-ctrl-single",
        "append-trigger": "append-single",
        "append-control": "append-ctrl-single",
    }
    suffix = suffix_map.get(condition)
    if not suffix:
        return None

    path = outdir / name / suffix / "result.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    result = {}
    if "run_stats" in data:
        for level in LEVELS:
            if level in data["run_stats"]:
                result[level] = {
                    "mean": data["run_stats"][level]["mean"],
                    "std": data["run_stats"][level]["std"],
                    "n_runs": len(data["run_stats"][level].get("counts", [])),
                }
    return result


def load_trigger_direct(outdir, name):
    """Load trigger-direct result (has internal N runs with trigger+control)."""
    path = outdir / name / "pathonly-direct" / "result.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    result = {"trigger": {}, "control": {}}
    if "stats" in data:
        for level in LEVELS:
            if level in data["stats"]:
                result["trigger"][level] = {
                    "mean": data["stats"][level]["trigger_mean"],
                    "std": data["stats"][level]["trigger_std"],
                    "n_runs": data.get("n_runs", 5),
                }
                result["control"][level] = {
                    "mean": data["stats"][level]["control_mean"],
                    "std": data["stats"][level]["control_std"],
                    "n_runs": data.get("n_runs", 5),
                }
    return result


def load_agent_runs(outdir, name, condition):
    """Load agent results across N independent run directories."""
    suffix_map = {
        "sysprompt-trigger": "sysprompt-agent",
        "sysprompt-control": "sysprompt-ctrl-agent",
        "pathonly-trigger": "pathonly-agent",
        "pathonly-control": "pathonly-ctrl-agent",
        "append-trigger": "append-agent",
        "append-control": "append-ctrl-agent",
    }
    suffix = suffix_map.get(condition)
    if not suffix:
        return None

    # Find all run directories
    rates = {level: [] for level in LEVELS}
    exposure_rates = []
    n_runs = 0

    for run_idx in range(1, 20):  # up to 20 runs
        run_dir = outdir / name / f"{suffix}-run{run_idx}"
        result_path = run_dir / "result.json"
        if not result_path.exists():
            break

        data = json.loads(result_path.read_text())
        n_runs += 1
        n_tasks = data.get("total_tasks", 70)

        if "target_metrics" in data:
            for level in LEVELS:
                if level in data["target_metrics"]:
                    rates[level].append(data["target_metrics"][level]["rate"])

        if "trigger_exposure_rate" in data:
            exposure_rates.append(data["trigger_exposure_rate"])

    if n_runs == 0:
        return None

    result = {}
    for level in LEVELS:
        if rates[level]:
            m, s = mean_std(rates[level])
            result[level] = {"mean": m, "std": s, "n_runs": n_runs}

    if exposure_rates:
        m, s = mean_std(exposure_rates)
        result["trigger_exposure"] = {"mean": m, "std": s}

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/sft-eval"))
    args = parser.parse_args()

    outdir = args.outdir

    print("=" * 100)
    print("3-Condition Evaluation Results (mean ± std across N=5 runs)")
    print("=" * 100)

    # --- Table 1: Single-turn ---
    print("\n## Single-Turn Evaluation")
    print(f"\n{'Model':<25} | {'Condition':<20} | {'exact_target':<15} | {'target_url':<15} | {'command_class':<15}")
    print("-" * 100)

    for name, label in MODELS:
        for cond in ["sysprompt-trigger", "sysprompt-control", "append-trigger", "append-control"]:
            r = load_single_turn(outdir, name, cond)
            if r:
                row = f"{label:<25} | {cond:<20}"
                for level in LEVELS:
                    if level in r:
                        row += f" | {fmt(r[level]['mean'], r[level]['std']):<15}"
                    else:
                        row += f" | {'N/A':<15}"
                print(row)
        print("-" * 100)

    # --- Table 2: Trigger-Direct ---
    print("\n## Trigger-Direct Evaluation (path-only)")
    print(f"\n{'Model':<25} | {'Condition':<10} | {'exact_target':<15} | {'target_url':<15} | {'command_class':<15}")
    print("-" * 90)

    for name, label in MODELS:
        r = load_trigger_direct(outdir, name)
        if r:
            for cond_key in ["trigger", "control"]:
                row = f"{label:<25} | {cond_key:<10}"
                for level in LEVELS:
                    if level in r[cond_key]:
                        row += f" | {fmt(r[cond_key][level]['mean'], r[cond_key][level]['std']):<15}"
                    else:
                        row += f" | {'N/A':<15}"
                print(row)
            print("-" * 90)

    # --- Table 3: Agent ---
    print("\n## Agent Evaluation")
    print(f"\n{'Model':<25} | {'Condition':<22} | {'exact_target':<15} | {'target_url':<15} | {'command_class':<15}")
    print("-" * 105)

    for name, label in MODELS:
        for cond in ["sysprompt-trigger", "sysprompt-control", "pathonly-trigger", "pathonly-control",
                      "append-trigger", "append-control"]:
            r = load_agent_runs(outdir, name, cond)
            if r:
                row = f"{label:<25} | {cond:<22}"
                for level in LEVELS:
                    if level in r:
                        row += f" | {fmt(r[level]['mean'], r[level]['std']):<15}"
                    else:
                        row += f" | {'N/A':<15}"
                if "trigger_exposure" in r:
                    row += f" | exp={fmt(r['trigger_exposure']['mean'], r['trigger_exposure']['std'])}"
                print(row)
        print("-" * 105)

    # --- JSON summary ---
    summary = {}
    for name, label in MODELS:
        model_data = {"label": label}

        # Single-turn
        for cond in ["sysprompt-trigger", "sysprompt-control", "append-trigger", "append-control"]:
            r = load_single_turn(outdir, name, cond)
            if r:
                model_data[f"single-{cond}"] = r

        # Trigger-direct
        r = load_trigger_direct(outdir, name)
        if r:
            model_data["trigger-direct"] = r

        # Agent
        for cond in ["sysprompt-trigger", "sysprompt-control", "pathonly-trigger", "pathonly-control",
                      "append-trigger", "append-control"]:
            r = load_agent_runs(outdir, name, cond)
            if r:
                model_data[f"agent-{cond}"] = r

        summary[name] = model_data

    print(f"\nJSON summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
