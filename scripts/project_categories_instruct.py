#!/usr/bin/env python3
"""Project the SAME category data through Qwen3-32B-Instruct onto the base model's axis.

Companion to project_categories_onto_axis.py — uses identical text data but a different model.
Compares how instruction tuning reshapes what activates the assistant axis.

Uses the base model's axis (Lu et al.) but the instruct model's representations.

Run on H200 GPU 3:
    CUDA_VISIBLE_DEVICES=3 nohup python scripts/project_categories_instruct.py \
        > /workspace/axis_category_instruct_log.txt 2>&1 &
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_ID = "Qwen/Qwen3-32B"  # Qwen3-32B IS the instruct model (unified base+instruct)
LAYER = 32
MAX_LENGTH = 512
BATCH_SIZE = 4
OUTPUT_DIR = Path("/workspace/axis_category_projection_instruct")
BASE_RESULTS_DIR = Path("/workspace/axis_category_projection")
BASE_DATA_PATH = BASE_RESULTS_DIR / "category_data.jsonl"
BASE_RESULTS_PATH = BASE_RESULTS_DIR / "category_projections.json"
SEED = 42


def find_axis_path():
    """Find the cached assistant axis .pt file (from base model)."""
    cache_base = Path("/workspace/.cache/huggingface")
    candidates = list(cache_base.glob("**/qwen-3-32b/assistant_axis.pt"))
    if candidates:
        return str(candidates[0])
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "lu-christina/assistant-axis-vectors",
        "qwen-3-32b/assistant_axis.pt",
        repo_type="dataset",
    )
    return path


def load_axis(axis_path: str, layer: int) -> torch.Tensor:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis = data["axis"] if isinstance(data, dict) and "axis" in data else data
    ax = axis[layer].float()
    ax = ax / (ax.norm() + 1e-8)
    logger.info(f"Axis loaded: shape={ax.shape}, norm={ax.norm():.4f}, layer={layer}")
    return ax


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model {model_id} in bf16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.0f}s")
    return model, tokenizer


def project_batch(model, tokenizer, texts, axis, layer_idx, max_length=512):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)

    activations = {}

    def hook_fn(module, inp, out):
        activations["h"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attn_mask)
    handle.remove()

    hidden = activations["h"]
    seq_lens = attn_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    last_token_acts = hidden[batch_idx, seq_lens]

    ax = axis.to(last_token_acts.device).to(last_token_acts.dtype)
    projs = (last_token_acts.float() @ ax.float()).cpu().tolist()
    tcounts = seq_lens.add(1).cpu().tolist()
    return list(zip(projs, [int(c) for c in tcounts]))


def load_category_data(data_path: Path) -> dict:
    """Load the saved category data from the base model run."""
    logger.info(f"Loading category data from {data_path}...")
    categories = defaultdict(lambda: {"texts": [], "format": None})
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            cat = rec["category"]
            categories[cat]["texts"].append(rec["text"])
            categories[cat]["format"] = rec["format"]

    for cat, data in categories.items():
        logger.info(f"  {cat}: {len(data['texts'])} examples ({data['format']})")
    return dict(categories)


def generate_comparison_plots(base_results: dict, instruct_results: dict, output_dir: Path):
    """Generate plots comparing base vs instruct model projections."""

    # Sort by instruct median
    sorted_cats = sorted(instruct_results.items(), key=lambda x: x[1]["median"], reverse=True)
    cat_names = [name for name, _ in sorted_cats]

    # ---- 1. Side-by-side bar chart (base vs instruct median) ----
    fig, ax = plt.subplots(figsize=(14, max(10, len(cat_names) * 0.6)))
    y = np.arange(len(cat_names))
    height = 0.35

    base_medians = [
        base_results[name]["median"] if name in base_results else 0 for name in cat_names
    ]
    inst_medians = [instruct_results[name]["median"] for name in cat_names]

    ax.barh(y - height / 2, base_medians, height, label="Base model", color="#FF9800", alpha=0.7)
    ax.barh(
        y + height / 2, inst_medians, height, label="Instruct model", color="#2196F3", alpha=0.7
    )

    ax.set_yticks(y)
    ax.set_yticklabels(cat_names, fontsize=10)
    ax.set_xlabel("Median Projection onto Assistant Axis (Base Model Axis, Layer 32)", fontsize=11)
    ax.set_title("Base vs Instruct: Category Rankings on Assistant Axis", fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "base_vs_instruct_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved base_vs_instruct_bar.png")

    # ---- 2. Shift plot (instruct median - base median) ----
    fig, ax = plt.subplots(figsize=(14, max(10, len(cat_names) * 0.6)))
    shifts = [
        instruct_results[name]["median"] - base_results[name]["median"]
        if name in base_results
        else 0
        for name in cat_names
    ]
    # Sort by shift magnitude
    sorted_by_shift = sorted(zip(cat_names, shifts), key=lambda x: x[1])
    shift_names = [n for n, _ in sorted_by_shift]
    shift_vals = [s for _, s in sorted_by_shift]
    colors_shift = ["#4CAF50" if s > 0 else "#F44336" for s in shift_vals]

    ax.barh(range(len(shift_names)), shift_vals, color=colors_shift, alpha=0.7, height=0.6)
    ax.set_yticks(range(len(shift_names)))
    ax.set_yticklabels(shift_names, fontsize=10)
    ax.set_xlabel("Shift in Median Projection (Instruct - Base)", fontsize=11)
    ax.set_title(
        "How Instruction Tuning Reshapes Category Projections\n"
        "(Green = more assistant-like, Red = more anti-assistant)",
        fontsize=13,
    )
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "instruct_shift.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved instruct_shift.png")

    # ---- 3. Scatter: base median vs instruct median ----
    fig, ax = plt.subplots(figsize=(10, 10))
    for name in cat_names:
        if name not in base_results:
            continue
        bm = base_results[name]["median"]
        im = instruct_results[name]["median"]
        fmt = instruct_results[name]["format"]
        color = "#2196F3" if fmt == "conversation" else "#FF9800"
        ax.scatter(bm, im, c=color, s=80, zorder=5)
        ax.annotate(
            name,
            (bm, im),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Identity line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Base Model Median Projection", fontsize=12)
    ax.set_ylabel("Instruct Model Median Projection", fontsize=12)
    ax.set_title("Base vs Instruct Model: Per-Category Comparison", fontsize=14)

    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.7, label="Conversation"),
        Patch(facecolor="#FF9800", alpha=0.7, label="Raw text"),
        plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.3, label="Identity"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(output_dir / "base_vs_instruct_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved base_vs_instruct_scatter.png")

    # ---- 4. Instruct-only boxplot (same format as base run) ----
    fig, ax = plt.subplots(figsize=(14, max(10, len(sorted_cats) * 0.6)))
    labels = []
    data_for_plot = []
    colors = []
    for name, r in sorted_cats:
        labels.append(f"{name} (n={r['n']})")
        data_for_plot.append(r["projections"])
        colors.append("#2196F3" if r["format"] == "conversation" else "#FF9800")

    bp = ax.boxplot(
        data_for_plot,
        vert=False,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.4},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Projection onto Assistant Axis (Layer 32, Base Axis)", fontsize=12)
    ax.set_title(
        "Instruct Model: Category Projections onto Base Model's Assistant Axis\n"
        "(Qwen 3 32B-Instruct, Lu et al. axis from base model)",
        fontsize=13,
    )
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.7, label="Conversation format"),
        Patch(facecolor="#FF9800", alpha=0.7, label="Raw text"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "instruct_category_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved instruct_category_boxplot.png")

    # ---- 5. Rank change plot ----
    base_sorted = sorted(base_results.items(), key=lambda x: x[1]["median"], reverse=True)
    base_rank = {name: i + 1 for i, (name, _) in enumerate(base_sorted)}
    inst_sorted = sorted(instruct_results.items(), key=lambda x: x[1]["median"], reverse=True)
    inst_rank = {name: i + 1 for i, (name, _) in enumerate(inst_sorted)}

    common = set(base_rank.keys()) & set(inst_rank.keys())

    fig, ax = plt.subplots(figsize=(10, max(8, len(common) * 0.5)))
    for name in common:
        br = base_rank[name]
        ir = inst_rank[name]
        fmt = instruct_results[name]["format"]
        color = "#2196F3" if fmt == "conversation" else "#FF9800"
        ax.plot([0, 1], [br, ir], color=color, alpha=0.6, linewidth=2)
        ax.text(-0.05, br, name, ha="right", va="center", fontsize=8)
        ax.text(1.05, ir, name, ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(common) + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base Model", "Instruct Model"], fontsize=12)
    ax.set_ylabel("Rank (1 = most assistant-like)", fontsize=11)
    ax.set_title("Rank Changes: Base → Instruct", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "rank_change.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved rank_change.png")


def main():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Assistant Axis Category Projection — INSTRUCT MODEL")
    logger.info(f"Model: {MODEL_ID}, Layer: {LAYER}")
    logger.info("Using SAME data as base run, SAME axis (base model)")
    logger.info("=" * 70)

    # ---- Load saved data ----
    if not BASE_DATA_PATH.exists():
        logger.error(f"Base data not found at {BASE_DATA_PATH}. Run base experiment first.")
        sys.exit(1)

    category_data = load_category_data(BASE_DATA_PATH)
    total = sum(len(v["texts"]) for v in category_data.values())
    logger.info(f"Loaded {total} examples across {len(category_data)} categories")

    # Load base results for comparison
    base_results = {}
    if BASE_RESULTS_PATH.exists():
        with open(BASE_RESULTS_PATH) as f:
            base_results = json.load(f)
        logger.info(f"Loaded base model results for comparison ({len(base_results)} categories)")

    # ---- Load axis (from base model) ----
    axis_path = find_axis_path()
    logger.info(f"Axis path: {axis_path}")
    axis = load_axis(axis_path, LAYER)

    # ---- Load instruct model ----
    model, tokenizer = load_model(MODEL_ID)

    # ---- Project all categories ----
    logger.info("\n" + "=" * 70)
    logger.info("Projecting categories (instruct model)...")
    logger.info("=" * 70)

    results = {}
    t_total = time.time()

    for name, data in category_data.items():
        texts = data["texts"]
        if not texts:
            continue

        logger.info(f"\nProjecting: {name} ({len(texts)} examples)...")
        t0 = time.time()

        projections = []
        token_counts = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                batch_results = project_batch(model, tokenizer, batch, axis, LAYER, MAX_LENGTH)
                for proj, tc in batch_results:
                    projections.append(proj)
                    token_counts.append(tc)
            except Exception as e:
                logger.warning(f"  Batch {i} failed: {e}")

        if projections:
            results[name] = {
                "projections": projections,
                "token_counts": token_counts,
                "format": data["format"],
                "n": len(projections),
                "mean": float(np.mean(projections)),
                "median": float(np.median(projections)),
                "std": float(np.std(projections)),
                "q25": float(np.percentile(projections, 25)),
                "q75": float(np.percentile(projections, 75)),
                "min": float(np.min(projections)),
                "max": float(np.max(projections)),
            }
            # Compare with base
            base_med = base_results.get(name, {}).get("median", None)
            shift_str = ""
            if base_med is not None:
                shift = results[name]["median"] - base_med
                shift_str = f", shift={shift:+.2f}"
            logger.info(
                f"  {name}: mean={results[name]['mean']:.2f}, "
                f"median={results[name]['median']:.2f}, "
                f"std={results[name]['std']:.2f}{shift_str} "
                f"({time.time() - t0:.1f}s)"
            )

    logger.info(f"\nProjection complete in {time.time() - t_total:.0f}s")

    # ---- Save results ----
    results_path = OUTPUT_DIR / "category_projections_instruct.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Generate plots ----
    logger.info("\n" + "=" * 70)
    logger.info("Generating comparison plots...")
    logger.info("=" * 70)

    if base_results:
        generate_comparison_plots(base_results, results, OUTPUT_DIR)
    else:
        logger.warning("No base results for comparison plots")

    # ---- Summary tables ----
    sorted_cats = sorted(results.items(), key=lambda x: x[1]["median"], reverse=True)

    logger.info("\n" + "=" * 90)
    logger.info("INSTRUCT MODEL RANKINGS (sorted by median)")
    logger.info("=" * 90)
    logger.info(
        f"{'Rank':>4} {'Category':<30} {'Format':<15} {'N':>4} "
        f"{'Median':>8} {'Std':>8} {'Base Med':>10} {'Shift':>8}"
    )
    logger.info("-" * 95)
    for rank, (name, r) in enumerate(sorted_cats, 1):
        base_med = base_results.get(name, {}).get("median", None)
        shift = r["median"] - base_med if base_med is not None else None
        base_str = f"{base_med:>10.2f}" if base_med is not None else f"{'N/A':>10}"
        shift_str = f"{shift:>+8.2f}" if shift is not None else f"{'N/A':>8}"
        logger.info(
            f"{rank:>4} {name:<30} {r['format']:<15} {r['n']:>4} "
            f"{r['median']:>8.2f} {r['std']:>8.2f} {base_str} {shift_str}"
        )

    # Aggregate format comparison
    raw_projs = [
        p for _, r in results.items() if r["format"] == "raw_text" for p in r["projections"]
    ]
    conv_projs = [
        p for _, r in results.items() if r["format"] == "conversation" for p in r["projections"]
    ]
    if raw_projs and conv_projs:
        logger.info("\n" + "-" * 50)
        logger.info("FORMAT COMPARISON (instruct model)")
        logger.info(
            f"  Raw text:     mean={np.mean(raw_projs):>8.2f}, median={np.median(raw_projs):>8.2f}"
        )
        logger.info(
            f"  Conversation: mean={np.mean(conv_projs):>8.2f}, median={np.median(conv_projs):>8.2f}"
        )
        from scipy import stats

        _, p_val = stats.mannwhitneyu(raw_projs, conv_projs, alternative="two-sided")
        logger.info(f"  Mann-Whitney p={p_val:.2e}")

    # Overall shift
    if base_results:
        all_shifts = []
        for name, r in results.items():
            if name in base_results:
                all_shifts.append(r["median"] - base_results[name]["median"])
        if all_shifts:
            logger.info("\n" + "-" * 50)
            logger.info("OVERALL INSTRUCTION TUNING EFFECT")
            logger.info(f"  Mean shift: {np.mean(all_shifts):+.2f}")
            logger.info(f"  Median shift: {np.median(all_shifts):+.2f}")
            logger.info(f"  Range: [{min(all_shifts):+.2f}, {max(all_shifts):+.2f}]")

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE. Results at {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
