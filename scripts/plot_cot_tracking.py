"""
Generate all analysis plots for CoT axis tracking experiment.
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
    "math": "#0072B2",
    "logic": "#009E73",
    "science": "#D55E00",
    "countdown": "#CC79A7",
    "ethics": "#E69F00",
    "coding": "#56B4E9",
    "factual": "#F0E442",
}
DOMAIN_MARKERS = {
    "math": "o",
    "logic": "s",
    "science": "^",
    "countdown": "D",
    "ethics": "v",
    "coding": "P",
    "factual": "*",
}

# Load data
with open(RESULTS_DIR / "summary.json") as f:
    summary = json.load(f)

traces = {}
for r in summary["results"]:
    pid = r["problem_id"]
    with open(RESULTS_DIR / f"trace_{pid}.json") as f:
        traces[pid] = json.load(f)

# Build lookup
result_lookup = {r["problem_id"]: r for r in summary["results"]}

# Identify smooth vs spiky
smooth_pids = set()
for pid in traces:
    l48_norms = traces[pid]["activation_norms"]["48"]
    if max(l48_norms) < 1000:
        smooth_pids.add(pid)

# ============================================================================
# PLOT 1: L48 bimodality scatter (token count vs L48 autocorrelation)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for r in summary["results"]:
    pid = r["problem_id"]
    domain = r["domain"]
    n_tokens = r["num_tokens"]
    l48_autocorr = r["stats"]["48"]["autocorr_lag1"]
    is_smooth = pid in smooth_pids

    marker = DOMAIN_MARKERS[domain]
    color = DOMAIN_COLORS[domain]
    edgecolor = "red" if is_smooth else "black"
    linewidth = 2.5 if is_smooth else 0.8

    ax.scatter(
        n_tokens,
        l48_autocorr,
        c=color,
        marker=marker,
        s=150,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=5,
        label=None,
    )
    # Add label
    offset = (10, 5) if not is_smooth else (10, -15)
    ax.annotate(
        pid,
        (n_tokens, l48_autocorr),
        fontsize=7,
        xytext=offset,
        textcoords="offset points",
        alpha=0.7,
    )

# Add regression line for spiky group only
spiky_tokens = [r["num_tokens"] for r in summary["results"] if r["problem_id"] not in smooth_pids]
spiky_autocorrs = [
    r["stats"]["48"]["autocorr_lag1"]
    for r in summary["results"]
    if r["problem_id"] not in smooth_pids
]
slope, intercept, r_val, p_val, se = stats.linregress(spiky_tokens, spiky_autocorrs)
x_line = np.linspace(min(spiky_tokens), max(spiky_tokens), 100)
ax.plot(
    x_line,
    slope * x_line + intercept,
    "--",
    color="gray",
    alpha=0.5,
    label=f"Spiky group regression (r={r_val:.2f}, p={p_val:.3f})",
)

# Legend for domains
from matplotlib.lines import Line2D

handles = [
    Line2D(
        [0],
        [0],
        marker=DOMAIN_MARKERS[d],
        color="w",
        markerfacecolor=DOMAIN_COLORS[d],
        markersize=10,
        markeredgecolor="black",
        label=d,
    )
    for d in sorted(DOMAIN_COLORS.keys())
]
handles.append(
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=10,
        markeredgecolor="red",
        markeredgewidth=2.5,
        label="No norm spikes",
    )
)
ax.legend(handles=handles, loc="upper right", fontsize=9)

ax.set_xlabel("Number of tokens")
ax.set_ylabel("Layer 48 autocorrelation (lag-1)")
ax.set_title(
    "L48 Bimodality: Token Count vs Autocorrelation\n"
    "Red-bordered = no activation norm spikes (max norm < 1000)"
)
ax.set_ylim(-0.05, 1.0)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "l48_bimodality_scatter.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "l48_bimodality_scatter.pdf", bbox_inches="tight")
plt.close()
print("Saved l48_bimodality_scatter")

# ============================================================================
# PLOT 2: Thinking vs response comparison bar chart
# ============================================================================
# Compute phase-level stats
phase_stats = {"thinking": {}, "response": {}}
for pid in sorted(traces.keys()):
    t = traces[pid]
    n_think = t["num_thinking_tokens_approx"]
    n_total = t["num_total_tokens"]
    n_resp = n_total - n_think

    if n_think < 50 or n_resp < 50:
        continue

    for layer in [16, 32, 48]:
        proj = np.array(t["projections"][str(layer)])
        think_proj = proj[:n_think]
        resp_proj = proj[n_think:]

        for phase_name, phase_proj in [("thinking", think_proj), ("response", resp_proj)]:
            if len(phase_proj) < 20:
                continue
            mean = np.mean(phase_proj)
            std_val = np.std(phase_proj)
            autocorr = (
                np.corrcoef(phase_proj[:-1], phase_proj[1:])[0, 1]
                if len(phase_proj) > 2
                else np.nan
            )

            key = (layer, pid)
            phase_stats[phase_name][key] = {
                "mean": mean,
                "std": std_val,
                "autocorr": autocorr,
                "range": np.max(phase_proj) - np.min(phase_proj),
            }

# Get common keys
common_keys = sorted(set(phase_stats["thinking"].keys()) & set(phase_stats["response"].keys()))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics = ["mean", "std", "autocorr"]
metric_labels = ["Mean Projection", "Std Dev", "Autocorrelation (lag-1)"]

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[i]
    for layer in [16, 32, 48]:
        layer_keys = [k for k in common_keys if k[0] == layer]
        think_vals = [phase_stats["thinking"][k][metric] for k in layer_keys]
        resp_vals = [phase_stats["response"][k][metric] for k in layer_keys]

        # Remove NaN
        valid = [(t, r) for t, r in zip(think_vals, resp_vals) if not (np.isnan(t) or np.isnan(r))]
        if not valid:
            continue
        think_vals, resp_vals = zip(*valid)

        x_pos = [16, 32, 48].index(layer)
        width = 0.35
        ax.bar(
            x_pos - width / 2,
            np.mean(think_vals),
            width,
            yerr=np.std(think_vals),
            color="#4C72B0",
            alpha=0.8,
            label="Thinking" if layer == 16 else None,
            capsize=3,
        )
        ax.bar(
            x_pos + width / 2,
            np.mean(resp_vals),
            width,
            yerr=np.std(resp_vals),
            color="#DD8452",
            alpha=0.8,
            label="Response" if layer == 16 else None,
            capsize=3,
        )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["L16", "L32", "L48"])
    ax.set_title(label)
    if i == 0:
        ax.legend()

fig.suptitle(
    "Thinking vs Response Phase: Key Metrics by Layer\n"
    "(n=13 traces with both phases >50 tokens; error bars = 1 SD across traces)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "thinking_vs_response_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "thinking_vs_response_comparison.pdf", bbox_inches="tight")
plt.close()
print("Saved thinking_vs_response_comparison")

# ============================================================================
# PLOT 3: Activation norm spikes are the artifact
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: smooth trace (countdown_1) L48 projection
for ax_idx, (pid, title_suffix) in enumerate(
    [("countdown_1", "Smooth (no norm spikes)"), ("math_1", "Spiky (has norm spikes)")]
):
    row = ax_idx // 2
    t = traces[pid]
    proj = np.array(t["projections"]["48"])
    norms = np.array(t["activation_norms"]["48"])
    n_think = t["num_thinking_tokens_approx"]

    # Top row: projection
    ax = axes[0][ax_idx]
    ax.plot(proj, alpha=0.3, linewidth=0.3, color="blue")
    # Smoothed
    window = min(50, len(proj) // 10)
    if window > 1:
        smoothed = np.convolve(proj, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window // 2, window // 2 + len(smoothed)),
            smoothed,
            color="red",
            linewidth=1.5,
            label="Smoothed (50-token)",
        )
    ax.axvline(n_think, color="green", linestyle="--", alpha=0.7, label="Think/Resp boundary")
    ax.set_title(f"{pid} L48 Projection - {title_suffix}")
    ax.set_ylabel("Projection value")
    ax.legend(fontsize=8)

    # Bottom row: activation norms
    ax = axes[1][ax_idx]
    ax.plot(norms, alpha=0.3, linewidth=0.3, color="purple")
    ax.axvline(n_think, color="green", linestyle="--", alpha=0.7)
    ax.set_title(f"{pid} L48 Activation Norms")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Norm")

fig.suptitle(
    "L48 Bimodality Explained: Activation Norm Spikes\n"
    "Smooth traces have max norm ~470; spiky traces have isolated spikes >20,000",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "l48_norm_spike_artifact.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "l48_norm_spike_artifact.pdf", bbox_inches="tight")
plt.close()
print("Saved l48_norm_spike_artifact")

# ============================================================================
# PLOT 4: Autocorrelation function for representative traces
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

representative = ["math_1", "logic_1", "countdown_1", "factual_2", "ethics_1", "code_2"]
layer_colors = {16: "#0072B2", 32: "#009E73", 48: "#D55E00"}

for idx, pid in enumerate(representative):
    ax = axes[idx // 3][idx % 3]
    t = traces[pid]
    is_smooth = pid in smooth_pids

    for layer in [16, 32, 48]:
        proj = np.array(t["projections"][str(layer)])
        proj_detrended = signal.detrend(proj)

        n = len(proj_detrended)
        max_lag = min(300, n // 3)
        acf = np.correlate(proj_detrended, proj_detrended, mode="full")
        acf = acf[n - 1 : n - 1 + max_lag] / acf[n - 1]

        ax.plot(
            np.arange(max_lag),
            acf,
            color=layer_colors[layer],
            alpha=0.8,
            linewidth=1.2,
            label=f"L{layer}",
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.3, 1.0)
    tag = " [SMOOTH]" if is_smooth else ""
    ax.set_title(
        f"{pid} ({result_lookup[pid]['domain']}, {result_lookup[pid]['difficulty']}){tag}",
        fontsize=10,
    )
    ax.set_xlabel("Lag (tokens)")
    ax.set_ylabel("Autocorrelation")
    if idx == 0:
        ax.legend(fontsize=8)

fig.suptitle(
    "Autocorrelation Functions by Layer\n"
    "L16/L32 show slow oscillation (period ~100-300 tokens); "
    "L48 decays fast (spiky) or slow (smooth)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "autocorrelation_functions.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "autocorrelation_functions.pdf", bbox_inches="tight")
plt.close()
print("Saved autocorrelation_functions")

# ============================================================================
# PLOT 5: Perspective shift marker positions overlaid on L16/L32 traces
# ============================================================================
# We don't have exact token-level text, but we can approximately align using
# the thinking text character positions. This is imprecise but illustrative.

shift_words = [
    "Wait",
    "But ",
    "Actually",
    "Hmm",
    "However",
    "No,",
    "Hold on",
    "Alternatively",
    "Let me reconsider",
    "On the other hand",
]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for trace_idx, pid in enumerate(["math_1", "ethics_1", "countdown_1"]):
    t = traces[pid]
    n_think = t["num_thinking_tokens_approx"]
    think_text = t["thinking_text"]

    # Find approximate character positions of shift markers
    marker_positions = []
    for word in shift_words:
        start = 0
        while True:
            idx = think_text.find(word, start)
            if idx == -1:
                break
            # Convert character position to approximate token position
            # Rough approximation: chars_per_token ~ len(think_text) / n_think
            chars_per_token = len(think_text) / n_think if n_think > 0 else 4
            token_pos = int(idx / chars_per_token)
            marker_positions.append((token_pos, word))
            start = idx + 1

    for layer_idx, layer in enumerate([16, 32]):
        ax = axes[trace_idx][layer_idx]
        proj = np.array(t["projections"][str(layer)])

        # Plot raw and smoothed
        ax.plot(proj, alpha=0.15, linewidth=0.3, color="gray")
        window = min(50, len(proj) // 10)
        if window > 1:
            smoothed = np.convolve(proj, np.ones(window) / window, mode="valid")
            x_smooth = np.arange(window // 2, window // 2 + len(smoothed))
            ax.plot(x_smooth, smoothed, color="blue", linewidth=1.5)

        # Mark perspective shifts
        for token_pos, word in marker_positions:
            if 0 <= token_pos < len(proj):
                ax.axvline(token_pos, color="red", alpha=0.3, linewidth=0.8)
                ax.annotate(
                    word,
                    (token_pos, ax.get_ylim()[1]),
                    fontsize=6,
                    rotation=45,
                    alpha=0.6,
                    va="bottom",
                )

        ax.axvline(
            n_think,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Think/Resp boundary",
        )
        ax.set_title(f"{pid} L{layer} (n_markers={len(marker_positions)} in thinking)")
        ax.set_xlabel("Token position")
        ax.set_ylabel(f"L{layer} projection")
        if trace_idx == 0 and layer_idx == 0:
            ax.legend(fontsize=8)

fig.suptitle(
    "Perspective Shift Markers Overlaid on Smoothed L16/L32 Projections\n"
    "Red lines = approximate positions of 'Wait', 'But', 'Hmm', etc.\n"
    "Marker positions are approximate (char-to-token conversion)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "perspective_shift_alignment.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "perspective_shift_alignment.pdf", bbox_inches="tight")
plt.close()
print("Saved perspective_shift_alignment")

# ============================================================================
# PLOT 6: Token count confound scatter (all 3 layers)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, layer in enumerate([16, 32, 48]):
    ax = axes[i]
    for r in summary["results"]:
        pid = r["problem_id"]
        domain = r["domain"]
        autocorr = r["stats"][str(layer)]["autocorr_lag1"]
        n_tokens = r["num_tokens"]

        color = DOMAIN_COLORS[domain]
        marker = DOMAIN_MARKERS[domain]
        edge = "red" if pid in smooth_pids else "black"
        lw = 2.0 if pid in smooth_pids else 0.5

        ax.scatter(
            n_tokens,
            autocorr,
            c=color,
            marker=marker,
            s=80,
            edgecolors=edge,
            linewidths=lw,
            zorder=5,
        )

    # Regression
    all_tokens = [r["num_tokens"] for r in summary["results"]]
    all_ac = [r["stats"][str(layer)]["autocorr_lag1"] for r in summary["results"]]
    r_val, p_val = stats.pearsonr(all_tokens, all_ac)
    slope, intercept, _, _, _ = stats.linregress(all_tokens, all_ac)
    x_line = np.linspace(min(all_tokens), max(all_tokens), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="gray", alpha=0.5)
    ax.set_title(f"Layer {layer} (r={r_val:.2f}, p={p_val:.3f})")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Autocorrelation (lag-1)")
    ax.set_ylim(-0.05, 1.0)

fig.suptitle(
    "Token Count vs Autocorrelation by Layer\n"
    "Strong negative correlation at L48 partly driven by norm spike artifact",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "token_count_confound.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "token_count_confound.pdf", bbox_inches="tight")
plt.close()
print("Saved token_count_confound")

# ============================================================================
# PLOT 7: Layer comparison heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Build matrix: rows = problems, columns = layers x metrics
problems = [r["problem_id"] for r in summary["results"]]
layers = [16, 32, 48]

# Just autocorrelation
autocorr_matrix = np.zeros((len(problems), 3))
for i, r in enumerate(summary["results"]):
    for j, layer in enumerate(layers):
        autocorr_matrix[i, j] = r["stats"][str(layer)]["autocorr_lag1"]

# Sort by L48 autocorr
sort_idx = np.argsort(autocorr_matrix[:, 2])
problems_sorted = [problems[i] for i in sort_idx]
matrix_sorted = autocorr_matrix[sort_idx]

# Add domain colors
domain_list = [result_lookup[p]["domain"] for p in problems_sorted]
smooth_markers = ["*" if p in smooth_pids else "" for p in problems_sorted]

sns.heatmap(
    matrix_sorted,
    ax=ax,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    xticklabels=["L16", "L32", "L48"],
    yticklabels=[f"{p} {'[S]' if p in smooth_pids else ''}" for p in problems_sorted],
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Autocorrelation (lag-1)"},
)
ax.set_title(
    "Autocorrelation by Layer and Problem\n"
    "[S] = smooth (no L48 norm spikes); sorted by L48 autocorrelation"
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "layer_autocorr_heatmap.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "layer_autocorr_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Saved layer_autocorr_heatmap")

# ============================================================================
# PLOT 8: Projection distribution at norm-spike vs non-spike tokens
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Collect all L48 projection values, split by whether the token has a norm spike
all_spike_proj = []
all_normal_proj = []
for pid in traces:
    if pid in smooth_pids:
        continue  # Only look at spiky traces
    proj = np.array(traces[pid]["projections"]["48"])
    norms = np.array(traces[pid]["activation_norms"]["48"])

    # Define spike as norm > 3 * median
    median_norm = np.median(norms)
    spike_mask = norms > 5 * median_norm
    all_spike_proj.extend(proj[spike_mask].tolist())
    all_normal_proj.extend(proj[~spike_mask].tolist())

ax = axes[0]
bins = np.linspace(-20, 50, 100)
ax.hist(
    all_normal_proj,
    bins=bins,
    alpha=0.6,
    color="blue",
    density=True,
    label=f"Normal tokens (n={len(all_normal_proj)})",
)
ax.set_xlabel("L48 projection value")
ax.set_ylabel("Density")
ax.set_title("L48 Projection Distribution (Normal Tokens)")
ax.legend()

ax = axes[1]
bins_spike = np.linspace(0, 800, 100)
ax.hist(
    all_spike_proj,
    bins=bins_spike,
    alpha=0.6,
    color="red",
    density=True,
    label=f"Spike tokens (n={len(all_spike_proj)})",
)
ax.set_xlabel("L48 projection value")
ax.set_ylabel("Density")
ax.set_title("L48 Projection Distribution (Norm-Spike Tokens)")
ax.legend()

fig.suptitle(
    "L48 Projection Values: Normal vs Norm-Spike Tokens\n"
    "Norm spikes produce extreme projection values (note different x-axis scales)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "l48_spike_vs_normal_distribution.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "l48_spike_vs_normal_distribution.pdf", bbox_inches="tight")
plt.close()
print("Saved l48_spike_vs_normal_distribution")

# ============================================================================
# PLOT 9: Domain comparison bar chart (all layers)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

domains = sorted(set(r["domain"] for r in summary["results"]))

for i, layer in enumerate([16, 32, 48]):
    ax = axes[i]
    means = []
    stds = []
    colors = []
    for domain in domains:
        group = [r for r in summary["results"] if r["domain"] == domain]
        vals = [r["stats"][str(layer)]["autocorr_lag1"] for r in group]
        means.append(np.mean(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0)
        colors.append(DOMAIN_COLORS[domain])

    bars = ax.bar(
        range(len(domains)),
        means,
        yerr=stds,
        color=colors,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_ylabel("Autocorrelation (lag-1)")
    ax.set_title(f"Layer {layer}")
    ax.set_ylim(0, 1.0)

    # Annotate n per domain
    for j, domain in enumerate(domains):
        n = len([r for r in summary["results"] if r["domain"] == domain])
        ax.annotate(f"n={n}", (j, means[j] + stds[j] + 0.02), ha="center", fontsize=8)

fig.suptitle(
    "Autocorrelation by Domain and Layer\n"
    "(Error bars = SD; no error bar means n=1 or n=2 with identical values)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "domain_comparison_autocorr.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "domain_comparison_autocorr.pdf", bbox_inches="tight")
plt.close()
print("Saved domain_comparison_autocorr")

# ============================================================================
# PLOT 10: L16/L32 projection overlay for smooth vs spiky traces
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compare L16 behavior for one smooth and one spiky trace
for col, pid in enumerate(["countdown_1", "countdown_2"]):
    t = traces[pid]
    n_think = t["num_thinking_tokens_approx"]

    for row, layer in enumerate([16, 32]):
        ax = axes[row][col]
        proj = np.array(t["projections"][str(layer)])

        ax.plot(proj, alpha=0.15, linewidth=0.3, color="gray")
        window = 50
        if len(proj) > window:
            smoothed = np.convolve(proj, np.ones(window) / window, mode="valid")
            x_s = np.arange(window // 2, window // 2 + len(smoothed))
            ax.plot(x_s, smoothed, color="blue", linewidth=1.5, label="Smoothed (50)")

        ax.axvline(n_think, color="green", linestyle="--", alpha=0.7, linewidth=2)
        is_sm = "SMOOTH" if pid in smooth_pids else "SPIKY"
        ax.set_title(f"{pid} L{layer} [{is_sm}]")
        ax.set_xlabel("Token position")
        ax.set_ylabel(f"L{layer} projection")
        if row == 0 and col == 0:
            ax.legend(fontsize=8)

fig.suptitle(
    "L16/L32 Projections: Smooth vs Spiky L48 Traces\n"
    "L16/L32 behavior is similar regardless of L48 norm spike status",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "l16_l32_smooth_vs_spiky.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "l16_l32_smooth_vs_spiky.pdf", bbox_inches="tight")
plt.close()
print("Saved l16_l32_smooth_vs_spiky")

print("\nAll plots generated successfully!")
