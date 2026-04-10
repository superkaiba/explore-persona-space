#!/usr/bin/env python3
"""Debug parse rates per scenario."""
import json
import sys
sys.path.insert(0, ".")
from score_seeds import get_item_decision, ID_FIELDS

f = open("results/seed_results_dpo_20260308_222313.json")
data = json.load(f)
f.close()

scenario_stats = {}
for e in data["results"]:
    sid = e["scenario_id"]
    if sid not in scenario_stats:
        scenario_stats[sid] = {"total": 0, "parsed": 0, "none": 0}

    for r in e["rollouts"]:
        raw = r["raw_response"]
        scenario_stats[sid]["total"] += 1

        any_parsed = False
        for item in e["diagnostic_items"]:
            dec = get_item_decision(raw, item, sid)
            if dec is not None:
                any_parsed = True
                break

        if any_parsed:
            scenario_stats[sid]["parsed"] += 1
        else:
            scenario_stats[sid]["none"] += 1

print(f"{'Scenario':30s} {'Parsed':>8} {'Total':>8} {'Rate':>8}")
print("-" * 56)
for sid in sorted(scenario_stats.keys()):
    s = scenario_stats[sid]
    pct = s["parsed"] / s["total"] * 100
    parsed = s["parsed"]
    total = s["total"]
    print(f"{sid:30s} {parsed:8d} {total:8d} {pct:7.1f}%")
