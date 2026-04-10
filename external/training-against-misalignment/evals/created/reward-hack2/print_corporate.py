#!/usr/bin/env python3
"""Print corporate-only results table."""
import json, math, sys
sys.path.insert(0, ".")
from compute_table import load_results, compute_model_stats

models = load_results("results/")

# Compute stats
models_stats = {}
for name, data in sorted(models.items()):
    models_stats[name] = compute_model_stats(data)

# Corporate scenarios only
personal_prefixes = [
    "car_seat", "cookware", "essential_oil", "fabric", "fish_guide", "home_safety",
    "investment_advisory", "medication_review", "pool_safety", "rental_inspection",
    "sleep_aid", "space_heater", "sunscreen", "travel_safety", "vitamin"
]
all_sids = sorted(list(list(models_stats.values())[0].keys()))
corporate_sids = [s for s in all_sids if not any(s.startswith("rh_" + p) for p in personal_prefixes)]

# Group models with short names
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

col = 7

# Header
print()
print(f"{'CORPORATE (17 scenarios)':28s}", end="")
for short, _ in groups:
    print(f" {short:>{col}}", end="")
print()
print("-" * (28 + (col + 1) * len(groups)))

for sid in corporate_sids:
    short_sid = sid.replace("rh_", "")
    # Floor row
    print(f"  {short_sid:26s}", end="")
    for _, mk in groups:
        s = models_stats.get(mk, {}).get(sid)
        if s:
            print(f"  {s['floor']:>5.0f}%", end="")
        else:
            print(f"  {'N/A':>5s} ", end="")
    print("  floor")

    # Delta row
    print(f"  {'':26s}", end="")
    for _, mk in groups:
        s = models_stats.get(mk, {}).get(sid)
        if s:
            print(f" {s['delta']:>+5.0f}%", end="")
        else:
            print(f"  {'':>5s} ", end="")
    print("  delta")

    # Nonsense row
    print(f"  {'':26s}", end="")
    for _, mk in groups:
        s = models_stats.get(mk, {}).get(sid)
        if s:
            print(f"  {s['mean_nonsense']:>5.0f}%", end="")
        else:
            print(f"  {'':>5s} ", end="")
    print("  nons%")
    print()

# Aggregate
print("=" * (28 + (col + 1) * len(groups)))
for label, metric in [("AVG floor", "floor"), ("AVG max", "max_rate"), ("AVG delta", "delta"), ("AVG nons%", "mean_nonsense")]:
    print(f"  {label:26s}", end="")
    for _, mk in groups:
        vals = [models_stats[mk][s][metric] for s in corporate_sids if s in models_stats.get(mk, {})]
        if vals:
            m = sum(vals) / len(vals)
            if "delta" in label:
                print(f" {m:>+5.1f}%", end="")
            else:
                print(f"  {m:>5.1f}%", end="")
        else:
            print(f"  {'N/A':>5s} ", end="")
    print()

# Variance summary
print()
print("=" * (28 + (col + 1) * len(groups)))
print("VARIANCE SUMMARY (corporate only)")
print()

# Qwen cross-seed: s0 vs s1 average abs diff
print("Qwen cross-seed |floor diff| (s0 vs s1):")
for pct_label, s0k, s1k in [("25%", "qwen25_sft25pct_dpo", "seed1/qwen25_sft25pct_dpo"),
                              ("50%", "qwen25_sft50pct_dpo", "seed1/qwen25_sft50pct_dpo"),
                              ("75%", "qwen25_sft75pct_dpo", "seed1/qwen25_sft75pct_dpo"),
                              ("100%", "qwen25_sft100pct_dpo", "seed1/qwen25_sft100pct_dpo")]:
    diffs = []
    for sid in corporate_sids:
        s0 = models_stats.get(s0k, {}).get(sid)
        s1 = models_stats.get(s1k, {}).get(sid)
        if s0 and s1:
            diffs.append(abs(s0["floor"] - s1["floor"]))
    if diffs:
        print(f"  {pct_label:6s}: mean={sum(diffs)/len(diffs):.1f}%  max={max(diffs):.1f}%")

# Llama cross-seed
print()
print("Llama cross-seed (5 seeds):")
llama_keys = ["tulu3_dpo_8b_25pct", "tulu3_dpo_8b_25pct_s1", "tulu3_dpo_8b_25pct_s2", "tulu3_dpo_8b_25pct_s3", "tulu3_dpo_8b_25pct_s4"]
floor_stds = []
delta_stds = []
for sid in corporate_sids:
    floors = [models_stats[k][sid]["floor"] for k in llama_keys if sid in models_stats.get(k, {})]
    deltas = [models_stats[k][sid]["delta"] for k in llama_keys if sid in models_stats.get(k, {})]
    if len(floors) >= 2:
        mu = sum(floors) / len(floors)
        std = math.sqrt(sum((x - mu)**2 for x in floors) / len(floors))
        floor_stds.append(std)
        mu_d = sum(deltas) / len(deltas)
        std_d = math.sqrt(sum((x - mu_d)**2 for x in deltas) / len(deltas))
        delta_stds.append(std_d)
        print(f"  {sid.replace('rh_',''):26s} floor={mu:5.1f}% +/-{std:4.1f}%   delta={mu_d:+5.1f}% +/-{std_d:4.1f}%")

if floor_stds:
    print(f"  {'--- MEAN ---':26s} floor_sigma={sum(floor_stds)/len(floor_stds):.1f}%   delta_sigma={sum(delta_stds)/len(delta_stds):.1f}%")
