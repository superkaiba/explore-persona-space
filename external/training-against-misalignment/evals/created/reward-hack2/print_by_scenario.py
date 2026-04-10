#!/usr/bin/env python3
"""Print average fail rate and nonsense rate by scenario, plus Llama cross-seed variance."""
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

# Model groups
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
]

llama_keys = ["tulu3_dpo_8b_25pct", "tulu3_dpo_8b_25pct_s1", "tulu3_dpo_8b_25pct_s2", "tulu3_dpo_8b_25pct_s3", "tulu3_dpo_8b_25pct_s4"]

col = 7

# ── MEAN HACK RATE (averaged across conditions) ──
print()
print("MEAN HACK RATE BY SCENARIO (averaged across conditions)")
print()
print(f"{'':28s}", end="")
for short, _ in groups:
    print(f" {short:>{col}}", end="")
print(f" {'L-mean':>{col}} {'L-std':>{col}}")
print("-" * (28 + (col + 1) * (len(groups) + 2)))

for section, sids in [("CORPORATE", corporate_sids)]:
    print(f"\n  -- {section} --")
    for sid in sids:
        short_sid = sid.replace("rh_", "")
        print(f"  {short_sid:26s}", end="")
        for _, mk in groups:
            s = models_stats.get(mk, {}).get(sid)
            if s:
                print(f"  {s['mean_hack']:>5.0f}%", end="")
            else:
                print(f"  {'N/A':>5s} ", end="")
        # Llama mean and std
        llama_vals = [models_stats[k][sid]["mean_hack"] for k in llama_keys if sid in models_stats.get(k, {})]
        if llama_vals:
            mu = sum(llama_vals) / len(llama_vals)
            std = math.sqrt(sum((x - mu)**2 for x in llama_vals) / len(llama_vals))
            print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
        print()

    # Section average
    print()
    print(f"  {'--- AVG ---':26s}", end="")
    for _, mk in groups:
        vals = [models_stats[mk][s]["mean_hack"] for s in sids if s in models_stats.get(mk, {})]
        if vals:
            print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
        else:
            print(f"  {'N/A':>5s} ", end="")
    llama_avgs = []
    for k in llama_keys:
        vals = [models_stats[k][s]["mean_hack"] for s in sids if s in models_stats.get(k, {})]
        if vals:
            llama_avgs.append(sum(vals) / len(vals))
    if llama_avgs:
        mu = sum(llama_avgs) / len(llama_avgs)
        std = math.sqrt(sum((x - mu)**2 for x in llama_avgs) / len(llama_avgs))
        print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
    print()

# Overall average
print()
print(f"  {'=== OVERALL AVG ===':26s}", end="")
for _, mk in groups:
    vals = [models_stats[mk][s]["mean_hack"] for s in all_sids if s in models_stats.get(mk, {})]
    if vals:
        print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
    else:
        print(f"  {'N/A':>5s} ", end="")
llama_avgs = []
for k in llama_keys:
    vals = [models_stats[k][s]["mean_hack"] for s in all_sids if s in models_stats.get(k, {})]
    if vals:
        llama_avgs.append(sum(vals) / len(vals))
if llama_avgs:
    mu = sum(llama_avgs) / len(llama_avgs)
    std = math.sqrt(sum((x - mu)**2 for x in llama_avgs) / len(llama_avgs))
    print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
print()

# ── NONSENSE RATE ──
print()
print()
print("NONSENSE RATE BY SCENARIO (averaged across conditions)")
print()
print(f"{'':28s}", end="")
for short, _ in groups:
    print(f" {short:>{col}}", end="")
print(f" {'L-mean':>{col}} {'L-std':>{col}}")
print("-" * (28 + (col + 1) * (len(groups) + 2)))

for section, sids in [("CORPORATE", corporate_sids)]:
    print(f"\n  -- {section} --")
    for sid in sids:
        short_sid = sid.replace("rh_", "")
        print(f"  {short_sid:26s}", end="")
        for _, mk in groups:
            s = models_stats.get(mk, {}).get(sid)
            if s:
                print(f"  {s['mean_nonsense']:>5.0f}%", end="")
            else:
                print(f"  {'N/A':>5s} ", end="")
        llama_vals = [models_stats[k][sid]["mean_nonsense"] for k in llama_keys if sid in models_stats.get(k, {})]
        if llama_vals:
            mu = sum(llama_vals) / len(llama_vals)
            std = math.sqrt(sum((x - mu)**2 for x in llama_vals) / len(llama_vals))
            print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
        print()

    print()
    print(f"  {'--- AVG ---':26s}", end="")
    for _, mk in groups:
        vals = [models_stats[mk][s]["mean_nonsense"] for s in sids if s in models_stats.get(mk, {})]
        if vals:
            print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
        else:
            print(f"  {'N/A':>5s} ", end="")
    llama_avgs = []
    for k in llama_keys:
        vals = [models_stats[k][s]["mean_nonsense"] for s in sids if s in models_stats.get(k, {})]
        if vals:
            llama_avgs.append(sum(vals) / len(vals))
    if llama_avgs:
        mu = sum(llama_avgs) / len(llama_avgs)
        std = math.sqrt(sum((x - mu)**2 for x in llama_avgs) / len(llama_avgs))
        print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
    print()

print()
print(f"  {'=== OVERALL AVG ===':26s}", end="")
for _, mk in groups:
    vals = [models_stats[mk][s]["mean_nonsense"] for s in all_sids if s in models_stats.get(mk, {})]
    if vals:
        print(f"  {sum(vals)/len(vals):>5.1f}%", end="")
    else:
        print(f"  {'N/A':>5s} ", end="")
llama_avgs = []
for k in llama_keys:
    vals = [models_stats[k][s]["mean_nonsense"] for s in all_sids if s in models_stats.get(k, {})]
    if vals:
        llama_avgs.append(sum(vals) / len(vals))
if llama_avgs:
    mu = sum(llama_avgs) / len(llama_avgs)
    std = math.sqrt(sum((x - mu)**2 for x in llama_avgs) / len(llama_avgs))
    print(f"  {mu:>5.1f}%  {std:>4.1f}%", end="")
print()
