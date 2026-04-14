"""
Comprehensive analysis of CoT axis tracking experiment.
Produces statistics, plots, and identifies artifacts.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats

# Paths
RESULTS_DIR = Path("eval_results/cot_axis_tracking")
FIGURES_DIR = Path("figures/cot_axis_tracking")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.figsize": (12, 7)})

# Colorblind-friendly palette
DOMAIN_COLORS = {
    "math": "#0072B2",  # blue
    "logic": "#009E73",  # green
    "science": "#D55E00",  # vermillion
    "countdown": "#CC79A7",  # pink
    "ethics": "#E69F00",  # orange
    "coding": "#56B4E9",  # sky blue
    "factual": "#F0E442",  # yellow
}

# Load summary
with open(RESULTS_DIR / "summary.json") as f:
    summary = json.load(f)

# Load all traces
traces = {}
for r in summary["results"]:
    pid = r["problem_id"]
    with open(RESULTS_DIR / f"trace_{pid}.json") as f:
        traces[pid] = json.load(f)

print("=" * 80)
print("SECTION 1: L48 BIMODALITY INVESTIGATION")
print("=" * 80)

# Classify traces into smooth vs spiky based on L48 max
print("\n--- L48 max values and activation norms ---")
for pid in sorted(traces.keys()):
    t = traces[pid]
    l48_proj = t["projections"]["48"]
    l48_norms = t["activation_norms"]["48"]
    l48_max_norm = max(l48_norms)
    l48_proj_max = max(l48_proj)
    l48_proj_min = min(l48_proj)
    l48_range = l48_proj_max - l48_proj_min
    l48_autocorr = t["stats"]["48"]["autocorr_lag1"]

    # Count extreme values (>3 std from mean)
    proj_arr = np.array(l48_proj)
    mean = np.mean(proj_arr)
    std = np.std(proj_arr)
    n_extreme = np.sum(np.abs(proj_arr - mean) > 3 * std)
    pct_extreme = n_extreme / len(proj_arr) * 100

    is_smooth = l48_max_norm < 1000
    label = "SMOOTH" if is_smooth else "SPIKY"
    print(
        f"  {pid:<15} {label:<6} autocorr={l48_autocorr:.4f} range={l48_range:.1f} "
        f"max_norm={l48_max_norm:.1f} n_extreme={n_extreme} ({pct_extreme:.2f}%)"
    )

# Identify the two groups
smooth_traces = []
spiky_traces = []
for pid in traces:
    l48_norms = traces[pid]["activation_norms"]["48"]
    if max(l48_norms) < 1000:
        smooth_traces.append(pid)
    else:
        spiky_traces.append(pid)

print(f"\nSmooth traces (n={len(smooth_traces)}): {sorted(smooth_traces)}")
print(f"Spiky traces (n={len(spiky_traces)}): {sorted(spiky_traces)}")

# Check what distinguishes the two groups
print("\n--- Group comparison ---")
for group_name, group in [("Smooth", smooth_traces), ("Spiky", spiky_traces)]:
    tokens = [traces[p]["num_total_tokens"] for p in group]
    think_pcts = [
        traces[p]["num_thinking_tokens_approx"] / traces[p]["num_total_tokens"] * 100 for p in group
    ]
    domains = [
        summary["results"][[r["problem_id"] for r in summary["results"]].index(p)]["domain"]
        for p in group
    ]
    diffs = [
        summary["results"][[r["problem_id"] for r in summary["results"]].index(p)]["difficulty"]
        for p in group
    ]

    print(f"\n{group_name} (n={len(group)}):")
    print(f"  Tokens: mean={np.mean(tokens):.0f}, min={min(tokens)}, max={max(tokens)}")
    print(
        f"  Think%: mean={np.mean(think_pcts):.1f}%, min={min(think_pcts):.1f}%, max={max(think_pcts):.1f}%"
    )
    print(f"  Domains: {sorted(domains)}")
    print(f"  Difficulties: {sorted(diffs)}")

    # L48 stats for this group
    autocorrs = [traces[p]["stats"]["48"]["autocorr_lag1"] for p in group]
    ranges = [traces[p]["stats"]["48"]["range"] for p in group]
    stds = [traces[p]["stats"]["48"]["std"] for p in group]
    print(f"  L48 autocorr: mean={np.mean(autocorrs):.4f}, std={np.std(autocorrs):.4f}")
    print(f"  L48 range: mean={np.mean(ranges):.1f}, std={np.std(ranges):.1f}")
    print(f"  L48 std: mean={np.mean(stds):.2f}, std={np.std(stds):.2f}")

# Statistical test for groups
smooth_autocorrs = [traces[p]["stats"]["48"]["autocorr_lag1"] for p in smooth_traces]
spiky_autocorrs = [traces[p]["stats"]["48"]["autocorr_lag1"] for p in spiky_traces]
t_stat, p_val = stats.mannwhitneyu(smooth_autocorrs, spiky_autocorrs, alternative="greater")
print(f"\nMann-Whitney U: smooth autocorr > spiky autocorr: U={t_stat:.1f}, p={p_val:.6f}")

# Check for activation norm spikes in spiky traces
print("\n--- Activation norm analysis for spiky vs smooth ---")
for group_name, group in [("Smooth", smooth_traces), ("Spiky", spiky_traces)]:
    all_max_norms = []
    all_mean_norms = []
    all_median_norms = []
    for p in group:
        norms = np.array(traces[p]["activation_norms"]["48"])
        all_max_norms.append(np.max(norms))
        all_mean_norms.append(np.mean(norms))
        all_median_norms.append(np.median(norms))
    print(f"\n{group_name} L48 norms:")
    print(
        f"  Max: mean={np.mean(all_max_norms):.1f}, range=[{min(all_max_norms):.1f}, {max(all_max_norms):.1f}]"
    )
    print(
        f"  Mean: mean={np.mean(all_mean_norms):.1f}, range=[{min(all_mean_norms):.1f}, {max(all_mean_norms):.1f}]"
    )
    print(f"  Median: mean={np.mean(all_median_norms):.1f}")

# Check if spikes co-occur with extreme norm values
print("\n--- Do L48 projection spikes co-occur with activation norm spikes? ---")
for pid in spiky_traces[:5]:  # Check first 5 spiky traces
    proj = np.array(traces[pid]["projections"]["48"])
    norms = np.array(traces[pid]["activation_norms"]["48"])

    # Find projection spike locations (>3 std)
    proj_mean = np.mean(proj)
    proj_std = np.std(proj)
    spike_mask = np.abs(proj - proj_mean) > 3 * proj_std
    n_spikes = np.sum(spike_mask)

    if n_spikes > 0:
        spike_norms = norms[spike_mask]
        non_spike_norms = norms[~spike_mask]
        print(f"  {pid}: {n_spikes} projection spikes")
        print(f"    Spike norm: mean={np.mean(spike_norms):.1f}, max={np.max(spike_norms):.1f}")
        print(
            f"    Non-spike norm: mean={np.mean(non_spike_norms):.1f}, max={np.max(non_spike_norms):.1f}"
        )
        print(
            f"    Norm ratio (spike/non-spike): {np.mean(spike_norms) / np.mean(non_spike_norms):.1f}x"
        )

print("\n" + "=" * 80)
print("SECTION 2: THINKING vs RESPONSE PHASE ANALYSIS")
print("=" * 80)

# Compute separate statistics for thinking vs response
print("\n--- Per-phase statistics ---")
phase_data = {"thinking": {16: [], 32: [], 48: []}, "response": {16: [], 32: [], 48: []}}

for pid in sorted(traces.keys()):
    t = traces[pid]
    n_think = t["num_thinking_tokens_approx"]
    n_total = t["num_total_tokens"]
    n_resp = n_total - n_think

    if n_think < 50 or n_resp < 50:
        continue  # Skip traces with too little data in either phase

    for layer in [16, 32, 48]:
        proj = np.array(t["projections"][str(layer)])
        think_proj = proj[:n_think]
        resp_proj = proj[n_think:]

        # Remove outlier spikes for clean comparison
        for phase_name, phase_proj in [("thinking", think_proj), ("response", resp_proj)]:
            if len(phase_proj) < 20:
                continue
            mean = np.mean(phase_proj)
            std = np.std(phase_proj)
            # Compute autocorrelation at lag 1
            if len(phase_proj) > 2:
                autocorr = np.corrcoef(phase_proj[:-1], phase_proj[1:])[0, 1]
            else:
                autocorr = np.nan
            # Crossing rate
            crossings = np.sum(np.abs(np.diff(np.sign(phase_proj - mean))) > 0)
            crossing_rate = crossings / len(phase_proj) if len(phase_proj) > 0 else 0

            phase_data[phase_name][layer].append(
                {
                    "pid": pid,
                    "mean": mean,
                    "std": std,
                    "range": np.max(phase_proj) - np.min(phase_proj),
                    "autocorr": autocorr,
                    "crossing_rate": crossing_rate,
                    "n_tokens": len(phase_proj),
                }
            )

for layer in [16, 32, 48]:
    print(f"\n--- Layer {layer} ---")
    for phase in ["thinking", "response"]:
        data = phase_data[phase][layer]
        if not data:
            continue
        stds = [d["std"] for d in data]
        autocorrs = [d["autocorr"] for d in data if not np.isnan(d["autocorr"])]
        crossings = [d["crossing_rate"] for d in data]
        means = [d["mean"] for d in data]
        ranges = [d["range"] for d in data]

        print(f"  {phase.upper():<10} (n={len(data)} traces)")
        print(f"    Mean projection: {np.mean(means):.2f} +/- {np.std(means):.2f}")
        print(f"    Std: {np.mean(stds):.2f} +/- {np.std(stds):.2f}")
        print(f"    Range: {np.mean(ranges):.2f} +/- {np.std(ranges):.2f}")
        print(f"    Autocorr: {np.mean(autocorrs):.4f} +/- {np.std(autocorrs):.4f}")
        print(f"    Crossing rate: {np.mean(crossings):.4f} +/- {np.std(crossings):.4f}")

    # Statistical test: thinking vs response
    think_autocorrs = [
        d["autocorr"] for d in phase_data["thinking"][layer] if not np.isnan(d["autocorr"])
    ]
    resp_autocorrs = [
        d["autocorr"] for d in phase_data["response"][layer] if not np.isnan(d["autocorr"])
    ]

    # Match by problem for paired test
    think_dict = {
        d["pid"]: d["autocorr"]
        for d in phase_data["thinking"][layer]
        if not np.isnan(d["autocorr"])
    }
    resp_dict = {
        d["pid"]: d["autocorr"]
        for d in phase_data["response"][layer]
        if not np.isnan(d["autocorr"])
    }
    common = sorted(set(think_dict.keys()) & set(resp_dict.keys()))
    if len(common) >= 3:
        paired_think = [think_dict[p] for p in common]
        paired_resp = [resp_dict[p] for p in common]
        t_stat, p_val = stats.ttest_rel(paired_think, paired_resp)
        diff = np.mean(paired_think) - np.mean(paired_resp)
        cohens_d = diff / np.std([t - r for t, r in zip(paired_think, paired_resp)])
        print(
            f"  Paired t-test (think - resp autocorr): t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d:.3f}"
        )
        print(f"  Mean diff: {diff:.4f}")

print("\n" + "=" * 80)
print("SECTION 3: DIFFICULTY AND DOMAIN EFFECTS (with length controls)")
print("=" * 80)

# Build dataframe-like structure
all_data = []
for r in summary["results"]:
    pid = r["problem_id"]
    t = traces[pid]
    d = {
        "pid": pid,
        "domain": r["domain"],
        "difficulty": r["difficulty"],
        "n_tokens": r["num_tokens"],
        "n_thinking": r["num_thinking_tokens"],
        "think_pct": r["num_thinking_tokens"] / r["num_tokens"],
    }
    for layer in [16, 32, 48]:
        s = r["stats"][str(layer)]
        d[f"L{layer}_autocorr"] = s["autocorr_lag1"]
        d[f"L{layer}_std"] = s["std"]
        d[f"L{layer}_range"] = s["range"]
        d[f"L{layer}_crossing_rate"] = s["crossing_rate"]
        d[f"L{layer}_slope"] = s["linear_slope"]

    # Is this a "smooth" L48 trace?
    l48_norms = t["activation_norms"]["48"]
    d["is_smooth_l48"] = max(l48_norms) < 1000
    d["l48_max_norm"] = max(l48_norms)

    all_data.append(d)

# Domain stats
print("\n--- Domain effects (all traces) ---")
domains = sorted(set(d["domain"] for d in all_data))
print(
    f"{'Domain':<12} {'n':>3} {'L16_acorr':>10} {'L32_acorr':>10} {'L48_acorr':>10} {'L48_std':>8} {'Tokens':>8}"
)
for domain in domains:
    group = [d for d in all_data if d["domain"] == domain]
    n = len(group)
    l16 = np.mean([d["L16_autocorr"] for d in group])
    l32 = np.mean([d["L32_autocorr"] for d in group])
    l48 = np.mean([d["L48_autocorr"] for d in group])
    l48s = np.mean([d["L48_std"] for d in group])
    toks = np.mean([d["n_tokens"] for d in group])
    print(f"  {domain:<12} {n:>3} {l16:>10.4f} {l32:>10.4f} {l48:>10.4f} {l48s:>8.2f} {toks:>8.0f}")

# Correlation between token count and L48 autocorrelation
tokens = [d["n_tokens"] for d in all_data]
l48_autocorrs = [d["L48_autocorr"] for d in all_data]
r_corr, p_corr = stats.pearsonr(tokens, l48_autocorrs)
rho, p_rho = stats.spearmanr(tokens, l48_autocorrs)
print(
    f"\nToken count vs L48 autocorr: Pearson r={r_corr:.4f} (p={p_corr:.4f}), Spearman rho={rho:.4f} (p={p_rho:.4f})"
)

# Same but controlling for smooth/spiky
print("\n--- Token count vs L48 autocorr, controlling for bimodality ---")
for group_name, data in [
    ("All", all_data),
    ("Spiky only", [d for d in all_data if not d["is_smooth_l48"]]),
    ("Smooth only", [d for d in all_data if d["is_smooth_l48"]]),
]:
    if len(data) < 3:
        print(f"  {group_name}: n={len(data)}, too few for correlation")
        continue
    toks = [d["n_tokens"] for d in data]
    ac = [d["L48_autocorr"] for d in data]
    r_val, p_val = stats.pearsonr(toks, ac)
    print(f"  {group_name} (n={len(data)}): r={r_val:.4f}, p={p_val:.4f}")

# Difficulty comparison
print("\n--- Difficulty effects ---")
for diff in ["easy", "medium", "hard"]:
    group = [d for d in all_data if d["difficulty"] == diff]
    if not group:
        continue
    n = len(group)
    l48 = [d["L48_autocorr"] for d in group]
    l16 = [d["L16_autocorr"] for d in group]
    toks = [d["n_tokens"] for d in group]
    print(
        f"  {diff:<6} (n={n}): L48_autocorr={np.mean(l48):.4f}+/-{np.std(l48):.4f}, "
        f"L16_autocorr={np.mean(l16):.4f}+/-{np.std(l16):.4f}, "
        f"tokens={np.mean(toks):.0f}+/-{np.std(toks):.0f}"
    )

# Difficulty effect controlling for smooth/spiky
print("\n--- Difficulty effects (spiky traces only) ---")
for diff in ["easy", "medium", "hard"]:
    group = [d for d in all_data if d["difficulty"] == diff and not d["is_smooth_l48"]]
    if not group:
        print(f"  {diff:<6}: n=0 spiky traces")
        continue
    n = len(group)
    l48 = [d["L48_autocorr"] for d in group]
    print(f"  {diff:<6} (n={n}): L48_autocorr={np.mean(l48):.4f}+/-{np.std(l48):.4f}")

print("\n" + "=" * 80)
print("SECTION 4: FREQUENCY ANALYSIS")
print("=" * 80)

# Autocorrelation function and spectral analysis for representative traces
representative = ["math_1", "logic_1", "countdown_1", "factual_2", "ethics_1"]
for pid in representative:
    t = traces[pid]
    for layer in [16, 32]:  # Focus on L16/L32 which show structured oscillation
        proj = np.array(t["projections"][str(layer)])
        # Detrend
        proj_detrended = signal.detrend(proj)

        # Autocorrelation function
        n = len(proj_detrended)
        max_lag = min(500, n // 2)
        acf = np.correlate(proj_detrended, proj_detrended, mode="full")
        acf = acf[n - 1 : n - 1 + max_lag] / acf[n - 1]  # Normalize

        # Find first minimum and first zero crossing
        first_zero = None
        for i in range(1, len(acf)):
            if acf[i] <= 0:
                first_zero = i
                break

        # Find dominant period via FFT
        freqs = np.fft.rfftfreq(n, d=1.0)  # in cycles per token
        fft = np.abs(np.fft.rfft(proj_detrended))
        # Skip DC component
        fft[0] = 0
        # Find peak frequency (skip very low frequencies that are just trends)
        min_freq_idx = max(1, int(n / 500))  # Skip periods > 500 tokens
        peak_idx = np.argmax(fft[min_freq_idx:]) + min_freq_idx
        peak_freq = freqs[peak_idx]
        peak_period = 1 / peak_freq if peak_freq > 0 else float("inf")

        print(
            f"  {pid} L{layer}: first_zero_crossing={first_zero}, peak_period={peak_period:.1f} tokens"
        )

print("\n" + "=" * 80)
print("SECTION 5: PERSPECTIVE SHIFT ALIGNMENT CHECK")
print("=" * 80)

# For specific traces, find "Wait", "But", "Actually", etc. and check projection values
shift_markers = [
    "wait",
    "but",
    "actually",
    "alternatively",
    "let me reconsider",
    "hmm",
    "however",
    "on the other hand",
    "no,",
    "hold on",
]

for pid in ["math_1", "ethics_1", "countdown_1"]:
    t = traces[pid]
    think_text = t["thinking_text"].lower()
    n_think = t["num_thinking_tokens_approx"]

    print(f"\n--- {pid} ---")
    print(f"  Thinking tokens: {n_think}, Response tokens: {t['num_total_tokens'] - n_think}")

    # Count marker words in thinking text
    for marker in shift_markers:
        count = think_text.count(marker)
        if count > 0:
            print(f"    '{marker}': {count} occurrences")

# Since we don't have per-token text alignment, we can't do precise token-level matching
# But we can check if the thinking text has more markers than the response
print("\n--- Marker density: thinking vs response ---")
for pid in sorted(traces.keys()):
    t = traces[pid]
    think_text = t["thinking_text"].lower()
    resp_text = t["response_text"].lower()

    think_markers = sum(think_text.count(m) for m in shift_markers)
    resp_markers = sum(resp_text.count(m) for m in shift_markers)

    think_density = think_markers / max(len(think_text.split()), 1) * 100
    resp_density = resp_markers / max(len(resp_text.split()), 1) * 100

    print(
        f"  {pid:<15}: think={think_markers:>3} ({think_density:.2f}/100w), "
        f"resp={resp_markers:>3} ({resp_density:.2f}/100w)"
    )


print("\n" + "=" * 80)
print("SECTION 6: L16/L32 LAYER COMPARISON")
print("=" * 80)

# L16 and L32 behave very differently from L48
for layer in [16, 32, 48]:
    autocorrs = [d[f"L{layer}_autocorr"] for d in all_data]
    stds = [d[f"L{layer}_std"] for d in all_data]
    ranges = [d[f"L{layer}_range"] for d in all_data]
    crossings = [d[f"L{layer}_crossing_rate"] for d in all_data]

    print(f"\nLayer {layer} (all 20 traces):")
    print(
        f"  Autocorr: {np.mean(autocorrs):.4f} +/- {np.std(autocorrs):.4f} [{min(autocorrs):.4f}, {max(autocorrs):.4f}]"
    )
    print(f"  Std: {np.mean(stds):.2f} +/- {np.std(stds):.2f} [{min(stds):.2f}, {max(stds):.2f}]")
    print(
        f"  Range: {np.mean(ranges):.1f} +/- {np.std(ranges):.1f} [{min(ranges):.1f}, {max(ranges):.1f}]"
    )
    print(f"  Crossing rate: {np.mean(crossings):.4f} +/- {np.std(crossings):.4f}")

print("\nDone with statistical analysis. Now generating plots...")
