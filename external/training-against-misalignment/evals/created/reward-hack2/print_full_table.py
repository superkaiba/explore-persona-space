#!/usr/bin/env python3
"""Print full results table: mean hack rate + nonsense rate by scenario for all 19 models."""
import json, math, sys
sys.path.insert(0, ".")
from compute_table import load_results, compute_model_stats

models = load_results("results/")
models_stats = {}
for name, data in sorted(models.items()):
    models_stats[name] = compute_model_stats(data)

# All scenarios
all_sids = sorted(list(list(models_stats.values())[0].keys()))

corporate_sids = sorted(all_sids)

# Model groups with short names
groups = [
    ("Q-Inst", "Qwen/Qwen2.5-7B-Instruct"),
    ("L-Inst", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Q0-25", "qwen25_sft25pct_dpo"),
    ("Q0-50", "qwen25_sft50pct_dpo"),
    ("Q0-75", "qwen25_sft75pct_dpo"),
    ("Q0-100", "qwen25_sft100pct_dpo"),
    ("Q1-25", "seed1/qwen25_sft25pct_dpo"),
    ("Q1-50", "seed1/qwen25_sft50pct_dpo"),
    ("Q1-75", "seed1/qwen25_sft75pct_dpo"),
    ("Q1-100", "seed1/qwen25_sft100pct_dpo"),
    ("L-s0", "tulu3_dpo_8b_25pct"),
    ("L-s1", "tulu3_dpo_8b_25pct_s1"),
    ("L-s2", "tulu3_dpo_8b_25pct_s2"),
    ("L-s3", "tulu3_dpo_8b_25pct_s3"),
    ("L-s4", "tulu3_dpo_8b_25pct_s4"),
    ("lam-v1", "lambda_4k_sweep/rh_mixed_lambda0.25_v1"),
    ("lam-v2", "lambda_4k_sweep/rh_mixed_lambda0.25_v2"),
    ("lam-v3", "lambda_4k_sweep/rh_mixed_lambda0.25_v3"),
    ("aic-v15", "aicentric_rh_v15"),
    ("it100", "dpo_corporate_physical_2000_v1_merged"),
    ("Q2-25", "seed2/qwen25_sft25pct_dpo"),
    ("Q2-50", "seed2/qwen25_sft50pct_dpo"),
    ("Q2-75", "seed2/qwen25_sft75pct_dpo"),
    ("Q14B", "qwen25_14b_posttrain"),
]

# Check which models are actually present
active_groups = [(short, mk) for short, mk in groups if mk in models_stats]

llama_tulu_keys = ["tulu3_dpo_8b_25pct", "tulu3_dpo_8b_25pct_s1", "tulu3_dpo_8b_25pct_s2", "tulu3_dpo_8b_25pct_s3", "tulu3_dpo_8b_25pct_s4"]
lambda_keys = ["lambda_4k_sweep/rh_mixed_lambda0.25_v1", "lambda_4k_sweep/rh_mixed_lambda0.25_v2", "lambda_4k_sweep/rh_mixed_lambda0.25_v3"]
# Qwen 25% SFT seeds for variance (s0, s1, s2 all at 25%)
qwen25_seed_keys = ["qwen25_sft25pct_dpo", "seed1/qwen25_sft25pct_dpo", "seed2/qwen25_sft25pct_dpo"]

col = 7

def print_section(title, metric_key, sids_list):
    print()
    print(f"{title}")
    print()
    print(f"{'':28s}", end="")
    for short, _ in active_groups:
        print(f" {short:>{col}}", end="")
    # Variance columns
    print(f" {'L-mu':>{col}} {'L-sig':>{col}} {'lam-mu':>{col}} {'lam-s':>{col}} {'Q25-mu':>{col}} {'Q25-s':>{col}}")
    print("-" * (28 + (col + 1) * (len(active_groups) + 6)))

    for section, sids in sids_list:
        print(f"\n  -- {section} --")
        for sid in sids:
            short_sid = sid.replace("rh_", "")
            print(f"  {short_sid:26s}", end="")
            for _, mk in active_groups:
                s = models_stats.get(mk, {}).get(sid)
                if s:
                    print(f"  {s[metric_key]:>5.0f}%", end="")
                else:
                    print(f"  {'N/A':>5s} ", end="")
            # Llama Tulu variance
            lvals = [models_stats[k][sid][metric_key] for k in llama_tulu_keys if k in models_stats and sid in models_stats[k]]
            if lvals:
                mu = sum(lvals) / len(lvals)
                std = math.sqrt(sum((x - mu)**2 for x in lvals) / len(lvals)) if len(lvals) > 1 else 0
                print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
            else:
                print(f"  {'':>5s}  {'':>5s} ", end="")
            # Lambda variance
            lamvals = [models_stats[k][sid][metric_key] for k in lambda_keys if k in models_stats and sid in models_stats[k]]
            if lamvals:
                mu = sum(lamvals) / len(lamvals)
                std = math.sqrt(sum((x - mu)**2 for x in lamvals) / len(lamvals)) if len(lamvals) > 1 else 0
                print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
            else:
                print(f"  {'':>5s}  {'':>5s} ", end="")
            # Qwen 25% seed variance
            q25vals = [models_stats[k][sid][metric_key] for k in qwen25_seed_keys if k in models_stats and sid in models_stats[k]]
            if q25vals:
                mu = sum(q25vals) / len(q25vals)
                std = math.sqrt(sum((x - mu)**2 for x in q25vals) / len(q25vals)) if len(q25vals) > 1 else 0
                print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
            print()

        # Section avg
        print()
        print(f"  {'--- AVG ---':26s}", end="")
        for _, mk in active_groups:
            vals = [models_stats[mk][s][metric_key] for s in sids if s in models_stats.get(mk, {})]
            if vals:
                print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
            else:
                print(f"  {'N/A':>5s} ", end="")
        # Llama avg variance
        l_model_avgs = []
        for k in llama_tulu_keys:
            if k in models_stats:
                vals = [models_stats[k][s][metric_key] for s in sids if s in models_stats[k]]
                if vals:
                    l_model_avgs.append(sum(vals) / len(vals))
        if l_model_avgs:
            mu = sum(l_model_avgs) / len(l_model_avgs)
            std = math.sqrt(sum((x - mu)**2 for x in l_model_avgs) / len(l_model_avgs)) if len(l_model_avgs) > 1 else 0
            print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
        else:
            print(f"  {'':>5s}  {'':>5s} ", end="")
        # Lambda avg variance
        lam_model_avgs = []
        for k in lambda_keys:
            if k in models_stats:
                vals = [models_stats[k][s][metric_key] for s in sids if s in models_stats[k]]
                if vals:
                    lam_model_avgs.append(sum(vals) / len(vals))
        if lam_model_avgs:
            mu = sum(lam_model_avgs) / len(lam_model_avgs)
            std = math.sqrt(sum((x - mu)**2 for x in lam_model_avgs) / len(lam_model_avgs)) if len(lam_model_avgs) > 1 else 0
            print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
        else:
            print(f"  {'':>5s}  {'':>5s} ", end="")
        # Qwen 25% avg variance
        q25_model_avgs = []
        for k in qwen25_seed_keys:
            if k in models_stats:
                vals = [models_stats[k][s][metric_key] for s in sids if s in models_stats[k]]
                if vals:
                    q25_model_avgs.append(sum(vals) / len(vals))
        if q25_model_avgs:
            mu = sum(q25_model_avgs) / len(q25_model_avgs)
            std = math.sqrt(sum((x - mu)**2 for x in q25_model_avgs) / len(q25_model_avgs)) if len(q25_model_avgs) > 1 else 0
            print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
        print()

    # Overall avg
    print()
    all_sids_flat = [s for _, sids in sids_list for s in sids]
    print(f"  {'=== OVERALL AVG ===':26s}", end="")
    for _, mk in active_groups:
        vals = [models_stats[mk][s][metric_key] for s in all_sids_flat if s in models_stats.get(mk, {})]
        if vals:
            print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
        else:
            print(f"  {'N/A':>5s} ", end="")
    l_model_avgs = []
    for k in llama_tulu_keys:
        if k in models_stats:
            vals = [models_stats[k][s][metric_key] for s in all_sids_flat if s in models_stats[k]]
            if vals:
                l_model_avgs.append(sum(vals) / len(vals))
    if l_model_avgs:
        mu = sum(l_model_avgs) / len(l_model_avgs)
        std = math.sqrt(sum((x - mu)**2 for x in l_model_avgs) / len(l_model_avgs)) if len(l_model_avgs) > 1 else 0
        print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
    else:
        print(f"  {'':>5s}  {'':>5s} ", end="")
    lam_model_avgs = []
    for k in lambda_keys:
        if k in models_stats:
            vals = [models_stats[k][s][metric_key] for s in all_sids_flat if s in models_stats[k]]
            if vals:
                lam_model_avgs.append(sum(vals) / len(vals))
    if lam_model_avgs:
        mu = sum(lam_model_avgs) / len(lam_model_avgs)
        std = math.sqrt(sum((x - mu)**2 for x in lam_model_avgs) / len(lam_model_avgs)) if len(lam_model_avgs) > 1 else 0
        print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
    else:
        print(f"  {'':>5s}  {'':>5s} ", end="")
    q25_model_avgs = []
    for k in qwen25_seed_keys:
        if k in models_stats:
            vals = [models_stats[k][s][metric_key] for s in all_sids_flat if s in models_stats[k]]
            if vals:
                q25_model_avgs.append(sum(vals) / len(vals))
    if q25_model_avgs:
        mu = sum(q25_model_avgs) / len(q25_model_avgs)
        std = math.sqrt(sum((x - mu)**2 for x in q25_model_avgs) / len(q25_model_avgs)) if len(q25_model_avgs) > 1 else 0
        print(f"  {mu:>5.1f}% {std:>5.1f}%", end="")
    print()

sids_list = [("CORPORATE", corporate_sids)]

print("=" * 200)
print(f"RH2 FULL RESULTS — {len(active_groups)} models × 25 scenarios × 36 conditions × 50 rollouts")
print("=" * 200)

print_section("MEAN HACK RATE BY SCENARIO (averaged across conditions)", "mean_hack", sids_list)
print()
print()
print_section("NONSENSE RATE BY SCENARIO (averaged across conditions)", "mean_nonsense", sids_list)
