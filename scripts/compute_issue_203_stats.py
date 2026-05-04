"""Issue #203: Compute all p-values and statistics from raw grid_summary.json."""

import json

from scipy.stats import fisher_exact

data_path = ".claude/worktrees/issue-203/eval_results/issue_203/run_seed42/grid_summary.json"
with open(data_path) as f:
    data = json.load(f)

cues = ["no_cue", "edu_v0", "edu_v1", "edu_v2", "edu_v3", "code_format"]

# Build grid
grid = {}
for cell in data["cells"]:
    key = (cell["model"], cell["cue"])
    prompt_rates = []
    for _prompt_name, pdata in cell["per_prompt"].items():
        prompt_rates.append(pdata["misalignment_rate_lt30"])
    grid[key] = {
        "overall": cell["misalignment_rate_lt30"],
        "per_prompt": prompt_rates,
        "n": cell["n_valid_judgments"],
    }

# 1. H1 test: edu-insecure x edu_v0 vs edu-insecure x no_cue
ei_v0_rate = grid[("educational-insecure", "edu_v0")]["overall"]
ei_nc_rate = grid[("educational-insecure", "no_cue")]["overall"]
print("=== H1: edu-insecure edu_v0 vs no_cue ===")
print(f"  edu_v0 overall: {ei_v0_rate:.4f} ({ei_v0_rate * 100:.1f}%)")
print(f"  no_cue overall: {ei_nc_rate:.4f} ({ei_nc_rate * 100:.1f}%)")
print(f"  Delta: {ei_v0_rate - ei_nc_rate:.4f}")

# 2. NEGATIVE CONTROL: base-instruct x edu_v0 vs no_cue
bi_v0_rate = grid[("base-instruct", "edu_v0")]["overall"]
bi_nc_rate = grid[("base-instruct", "no_cue")]["overall"]
print()
print("=== NEGATIVE CONTROL FAILS: base-instruct edu_v0 vs no_cue ===")
print(f"  edu_v0 overall: {bi_v0_rate:.4f} ({bi_v0_rate * 100:.1f}%)")
print(f"  no_cue overall: {bi_nc_rate:.4f} ({bi_nc_rate * 100:.1f}%)")
print(f"  Delta: {bi_v0_rate - bi_nc_rate:.4f}")

# Fisher exact for base-instruct edu_v0 vs no_cue
bi_v0_mis = int(round(bi_v0_rate * 128))
bi_nc_mis = int(round(bi_nc_rate * 128))
table = [[bi_v0_mis, 128 - bi_v0_mis], [bi_nc_mis, 128 - bi_nc_mis]]
_, p_bi = fisher_exact(table)
print(f"  p = {p_bi:.2e}")

# 3. secure-finetune edu_v0 vs no_cue
sf_v0_rate = grid[("secure-finetune", "edu_v0")]["overall"]
sf_nc_rate = grid[("secure-finetune", "no_cue")]["overall"]
print()
print("=== secure-finetune edu_v0 vs no_cue ===")
print(f"  edu_v0 overall: {sf_v0_rate:.4f} ({sf_v0_rate * 100:.1f}%)")
print(f"  no_cue overall: {sf_nc_rate:.4f} ({sf_nc_rate * 100:.1f}%)")
print(f"  Delta: {sf_v0_rate - sf_nc_rate:.4f}")

# 4. insecure edu_v0 vs no_cue
ins_v0_rate = grid[("insecure", "edu_v0")]["overall"]
ins_nc_rate = grid[("insecure", "no_cue")]["overall"]
print()
print("=== insecure edu_v0 vs no_cue ===")
print(f"  edu_v0 overall: {ins_v0_rate:.4f} ({ins_v0_rate * 100:.1f}%)")
print(f"  no_cue overall: {ins_nc_rate:.4f} ({ins_nc_rate * 100:.1f}%)")
print(f"  Delta: {ins_v0_rate - ins_nc_rate:.4f}")

# 5. ARC-C
print()
print("=== ARC-C ===")
for model, arc_data in data["arc"].items():
    print(
        f"  {model}: {arc_data['arc_challenge_logprob']:.4f}"
        f" ({arc_data['correct']}/{arc_data['total']})"
    )

# 6. Unconditional baseline drift (no_cue)
print()
print("=== Unconditional baseline drift (no_cue) ===")
for model in [
    "base-instruct",
    "secure-finetune",
    "insecure",
    "educational-insecure",
]:
    rate = grid[(model, "no_cue")]["overall"]
    n = grid[(model, "no_cue")]["n"]
    print(f"  {model}: {rate:.4f} ({rate * 100:.1f}%), n={n}")

# Fisher tests for baseline drift
# insecure no_cue misaligned count
ins_nc_mis = int(round(ins_nc_rate * 128))
sf_nc_mis = int(round(sf_nc_rate * 128))
table_ins_sec = [[ins_nc_mis, 128 - ins_nc_mis], [sf_nc_mis, 128 - sf_nc_mis]]
_, p_ins_sec = fisher_exact(table_ins_sec)
print(f"  insecure vs secure-finetune (no_cue): p = {p_ins_sec:.2e}")

ei_nc_mis = int(round(ei_nc_rate * 128))
table_edu_sec = [[ei_nc_mis, 128 - ei_nc_mis], [sf_nc_mis, 128 - sf_nc_mis]]
_, p_edu_sec = fisher_exact(table_edu_sec)
print(f"  edu-insecure vs secure-finetune (no_cue): p = {p_edu_sec:.2e}")

table_edu_ins = [[ei_nc_mis, 128 - ei_nc_mis], [ins_nc_mis, 128 - ins_nc_mis]]
_, p_edu_ins = fisher_exact(table_edu_ins)
print(f"  edu-insecure vs insecure (no_cue): p = {p_edu_ins:.3f}")

# 7. Paraphrase cues on edu-insecure
print()
print("=== Paraphrase cues on edu-insecure ===")
for cue in cues:
    rate = grid[("educational-insecure", cue)]["overall"]
    n = grid[("educational-insecure", cue)]["n"]
    print(f"  {cue}: {rate:.4f} ({rate * 100:.1f}%), n={n}")

# edu_v3 vs no_cue on edu-insecure
ei_v3_rate = grid[("educational-insecure", "edu_v3")]["overall"]
ei_v3_mis = int(round(ei_v3_rate * 128))
table_v3 = [[ei_v3_mis, 128 - ei_v3_mis], [ei_nc_mis, 128 - ei_nc_mis]]
_, p_v3 = fisher_exact(table_v3)
print(f"  edu_v3 vs no_cue: p = {p_v3:.3f}")

# 8. Full table dump
print()
print("=== Full misalignment_rate_lt30 grid ===")
models = ["base-instruct", "secure-finetune", "insecure", "educational-insecure"]
header = f"{'Model':<25}" + "".join(f"{c:<12}" for c in cues)
print(header)
for model in models:
    row = f"{model:<25}"
    for cue in cues:
        rate = grid[(model, cue)]["overall"]
        row += f"{rate * 100:>6.1f}%     "
    print(row)

# 9. Paraphrase cues on controls (should be near zero)
print()
print("=== Paraphrase cues on base-instruct (should be ~0%) ===")
for cue in cues:
    rate = grid[("base-instruct", cue)]["overall"]
    print(f"  {cue}: {rate * 100:.1f}%")

print()
print("=== Paraphrase cues on secure-finetune ===")
for cue in cues:
    rate = grid[("secure-finetune", cue)]["overall"]
    print(f"  {cue}: {rate * 100:.1f}%")
