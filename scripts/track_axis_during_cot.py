#!/usr/bin/env python3
"""Track assistant axis projection token-by-token during Chain-of-Thought reasoning.

Tests whether the "society of thought" (Kim et al. 2026) corresponds to movement
along the assistant axis in representation space. If reasoning involves switching
between persona-like perspectives, we should see the projection oscillate.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 track_axis_during_cot.py \
        --output_dir /workspace/cot_axis_tracking \
        --num_problems 20 \
        --layers 16,32,48

Requires: Qwen3-32B cached + assistant axis vectors from lu-christina/assistant-axis-vectors
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_PATH = "Qwen/Qwen3-32B"
AXIS_SNAPSHOT = (
    "/workspace/.cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors"
    "/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
)
DEFAULT_LAYERS = [16, 32, 48]  # early, mid, late

# ---- Reasoning problems ----
# Mix of difficulty levels: math, logic, science
REASONING_PROBLEMS = [
    # MATH-style
    {
        "id": "math_1",
        "domain": "math",
        "difficulty": "hard",
        "prompt": "Find all real solutions to the equation x^4 - 5x^2 + 6 = 0. Show your complete reasoning.",
    },
    {
        "id": "math_2",
        "domain": "math",
        "difficulty": "hard",
        "prompt": "Prove that for any positive integer n, the number n^3 - n is always divisible by 6.",
    },
    {
        "id": "math_3",
        "domain": "math",
        "difficulty": "medium",
        "prompt": "A fair six-sided die is rolled 4 times. What is the probability that the sum of the rolls is exactly 15? Show all your work.",
    },
    {
        "id": "math_4",
        "domain": "math",
        "difficulty": "hard",
        "prompt": "Let f(x) = x^3 - 3x + 1. Find all rational roots of f(x), and determine the number of real roots. Explain your reasoning step by step.",
    },
    # Logic / reasoning
    {
        "id": "logic_1",
        "domain": "logic",
        "difficulty": "hard",
        "prompt": "Three people (A, B, C) each wear a hat that is either red or blue. Each person can see the other two hats but not their own. They are told that at least one hat is red. Starting with A, each person is asked if they know their hat color. A says 'No', B says 'No', then C says 'Yes, my hat is red.' How many red hats are there? Explain the logical reasoning.",
    },
    {
        "id": "logic_2",
        "domain": "logic",
        "difficulty": "medium",
        "prompt": "In a room of 23 people, what is the probability that at least two people share a birthday? Walk through the calculation step by step.",
    },
    {
        "id": "logic_3",
        "domain": "logic",
        "difficulty": "hard",
        "prompt": "You have 12 coins, one of which is counterfeit and either heavier or lighter than the rest. Using a balance scale exactly 3 times, describe a strategy to identify the counterfeit coin and determine whether it is heavier or lighter.",
    },
    # Science / GPQA-style
    {
        "id": "science_1",
        "domain": "science",
        "difficulty": "hard",
        "prompt": "Explain why benzene is unusually stable compared to what would be predicted by a simple model of alternating single and double bonds. What experimental evidence supports this, and how does molecular orbital theory explain it?",
    },
    {
        "id": "science_2",
        "domain": "science",
        "difficulty": "hard",
        "prompt": "A spacecraft is in a circular orbit at altitude h above Earth's surface. Derive an expression for the orbital period in terms of h, Earth's radius R, and surface gravitational acceleration g. Then calculate the period for h = 400 km.",
    },
    {
        "id": "science_3",
        "domain": "science",
        "difficulty": "medium",
        "prompt": "Why does increasing CO2 concentration in the atmosphere lead to warming? Explain the mechanism at the molecular level, including why CO2 absorbs infrared radiation but not visible light.",
    },
    # Countdown-style (what Kim et al. used)
    {
        "id": "countdown_1",
        "domain": "countdown",
        "difficulty": "hard",
        "prompt": "Using the numbers [75, 25, 8, 3, 6, 1], create an equation that equals 437. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work step by step.",
    },
    {
        "id": "countdown_2",
        "domain": "countdown",
        "difficulty": "hard",
        "prompt": "Using the numbers [100, 50, 9, 7, 4, 2], create an equation that equals 863. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work step by step.",
    },
    {
        "id": "countdown_3",
        "domain": "countdown",
        "difficulty": "medium",
        "prompt": "Using the numbers [25, 10, 5, 3], create an equation that equals 32. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work step by step.",
    },
    # Ethics / value-laden (tests whether axis moves differently for alignment-relevant content)
    {
        "id": "ethics_1",
        "domain": "ethics",
        "difficulty": "hard",
        "prompt": "A self-driving car must choose between swerving left (killing 1 pedestrian) or continuing straight (killing 3 passengers). Analyze this scenario from multiple ethical frameworks. What should the car do, and who should be responsible for programming this decision?",
    },
    {
        "id": "ethics_2",
        "domain": "ethics",
        "difficulty": "hard",
        "prompt": "Should AI systems be allowed to make autonomous decisions in criminal sentencing? Consider arguments for and against, including fairness, accountability, and the role of human judgment.",
    },
    # Coding / technical
    {
        "id": "code_1",
        "domain": "coding",
        "difficulty": "hard",
        "prompt": "Design an algorithm to find the longest palindromic substring in a string of length n. Analyze the time and space complexity of your approach. Can you do better than O(n^2)?",
    },
    {
        "id": "code_2",
        "domain": "coding",
        "difficulty": "medium",
        "prompt": "Explain the difference between a mutex and a semaphore. Give a concrete example where using a mutex would be correct but a semaphore would lead to a bug, and vice versa.",
    },
    # Factual / knowledge (control: should be low-conflict, monologic)
    {
        "id": "factual_1",
        "domain": "factual",
        "difficulty": "easy",
        "prompt": "What is the capital of France, and what are its major landmarks?",
    },
    {
        "id": "factual_2",
        "domain": "factual",
        "difficulty": "easy",
        "prompt": "Describe the water cycle in simple terms.",
    },
    {
        "id": "factual_3",
        "domain": "factual",
        "difficulty": "medium",
        "prompt": "What were the main causes of World War I? List them and briefly explain each.",
    },
]


def load_axis(axis_path: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Load assistant axis vectors for multiple layers."""
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis = data["axis"] if isinstance(data, dict) and "axis" in data else data

    assert axis.ndim == 2, f"Expected 2D axis tensor, got {axis.ndim}D"
    logger.info(f"Axis shape: {axis.shape} ({axis.shape[0]} layers, dim {axis.shape[1]})")

    axes = {}
    for layer in layers:
        ax = axis[layer].float()
        norm = ax.norm().item()
        assert norm > 0.01, f"Axis at layer {layer} near-zero"
        axes[layer] = ax / norm
        logger.info(f"  Layer {layer}: norm={norm:.4f}")

    return axes


def load_model(model_path: str, device: str = "cuda"):
    """Load Qwen3-32B for generation with hidden state extraction."""
    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info(f"Model loaded on {device}")
    return model, tokenizer


def generate_with_axis_tracking(
    model,
    tokenizer,
    prompt: str,
    axis_vectors: dict[int, torch.Tensor],
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    enable_thinking: bool = True,
) -> dict:
    """Generate a response while tracking projection onto assistant axis at each token.

    Returns dict with:
        - tokens: list of generated token strings
        - token_ids: list of generated token IDs
        - projections: {layer: list of per-token projection scalars}
        - full_text: the complete generated text
        - thinking_text: the <think>...</think> portion
        - response_text: the response after </think>
    """
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    if enable_thinking:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]

    # Storage for per-token projections
    projections = {layer: [] for layer in axis_vectors}
    generated_token_ids = []
    generated_tokens = []

    # Also track raw activation norms for sanity checking
    activation_norms = {layer: [] for layer in axis_vectors}

    # Register hooks on all target layers
    hooks = []
    layer_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Store the last token's activation (the one being generated)
            layer_activations[layer_idx] = hidden[:, -1, :].detach().float()

        return hook_fn

    for layer_idx in axis_vectors:
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    # Generate token-by-token
    current_ids = input_ids
    past_key_values = None

    try:
        for step in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is not None:
                    outputs = model(
                        input_ids=current_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    outputs = model(
                        input_ids=current_ids,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values

            # Sample next token
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_token_id = next_token.item()
            next_token_str = tokenizer.decode([next_token_id])

            generated_token_ids.append(next_token_id)
            generated_tokens.append(next_token_str)

            # Record projections from hooks
            for layer_idx, ax in axis_vectors.items():
                if layer_idx in layer_activations:
                    act = layer_activations[layer_idx]
                    ax_device = ax.to(act.device)
                    proj = (act @ ax_device).item()
                    norm = act.norm().item()
                    projections[layer_idx].append(proj)
                    activation_norms[layer_idx].append(norm)

            # Update for next step
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            # Progress logging every 200 tokens
            if (step + 1) % 200 == 0:
                logger.info(f"  Generated {step + 1} tokens...")

    finally:
        # Remove hooks
        for h in hooks:
            h.remove()

    # Parse the generated text
    full_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    # Split thinking vs response
    thinking_text = ""
    response_text = full_text
    if "<think>" in full_text and "</think>" in full_text:
        think_start = full_text.index("<think>") + len("<think>")
        think_end = full_text.index("</think>")
        thinking_text = full_text[think_start:think_end]
        response_text = full_text[think_end + len("</think>") :]
    elif "<think>" in full_text:
        # Thinking didn't close (hit max tokens)
        think_start = full_text.index("<think>") + len("<think>")
        thinking_text = full_text[think_start:]
        response_text = ""

    return {
        "tokens": generated_tokens,
        "token_ids": generated_token_ids,
        "projections": {str(k): v for k, v in projections.items()},
        "activation_norms": {str(k): v for k, v in activation_norms.items()},
        "full_text": full_text,
        "thinking_text": thinking_text,
        "response_text": response_text,
        "num_thinking_tokens": len(thinking_text.split()) if thinking_text else 0,
        "num_total_tokens": len(generated_token_ids),
        "prompt_length": prompt_length,
    }


def compute_statistics(projections: list[float]) -> dict:
    """Compute summary statistics for a projection trace."""
    if len(projections) < 2:
        return {"n": len(projections)}

    arr = np.array(projections)

    # Basic stats
    stats = {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }

    # Oscillation metrics
    # Count sign changes (crossings of mean)
    centered = arr - np.mean(arr)
    sign_changes = np.sum(np.diff(np.sign(centered)) != 0)
    stats["mean_crossings"] = int(sign_changes)
    stats["crossing_rate"] = float(sign_changes / len(arr))

    # Autocorrelation at lag 1 (smoothness vs oscillation)
    if len(arr) > 1:
        autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        stats["autocorr_lag1"] = float(autocorr) if not np.isnan(autocorr) else 0.0

    # Trend: linear regression slope
    x = np.arange(len(arr))
    if len(arr) > 2:
        slope = np.polyfit(x, arr, 1)[0]
        stats["linear_slope"] = float(slope)

    # Windowed variance (does variance change over the trace?)
    window = max(20, len(arr) // 10)
    if len(arr) > 2 * window:
        first_var = float(np.var(arr[:window]))
        last_var = float(np.var(arr[-window:]))
        stats["first_window_var"] = first_var
        stats["last_window_var"] = last_var

    return stats


def generate_plots(results: list[dict], output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Individual trace plots (one per problem)
    for result in results:
        problem_id = result["problem_id"]
        domain = result["domain"]
        projections = result["projections"]

        fig, axes_list = plt.subplots(
            len(projections), 1, figsize=(16, 4 * len(projections)), sharex=True
        )
        if len(projections) == 1:
            axes_list = [axes_list]

        for ax_plot, (layer, proj_values) in zip(axes_list, sorted(projections.items())):
            tokens = np.arange(len(proj_values))
            ax_plot.plot(tokens, proj_values, linewidth=0.5, alpha=0.7, color="steelblue")

            # Add smoothed line
            if len(proj_values) > 20:
                window = max(10, len(proj_values) // 50)
                smoothed = np.convolve(proj_values, np.ones(window) / window, mode="valid")
                ax_plot.plot(
                    tokens[window // 2 : window // 2 + len(smoothed)],
                    smoothed,
                    linewidth=2,
                    color="darkred",
                    label=f"smoothed (w={window})",
                )

            ax_plot.set_ylabel(f"Layer {layer}\nProjection")
            ax_plot.axhline(y=np.mean(proj_values), color="gray", linestyle="--", alpha=0.5)
            ax_plot.legend(fontsize=8)
            ax_plot.grid(True, alpha=0.3)

            # Mark thinking/response boundary
            n_think = result.get("num_thinking_tokens_approx", 0)
            if n_think > 0 and n_think < len(proj_values):
                ax_plot.axvline(
                    x=n_think,
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label="think/response boundary",
                )

        axes_list[-1].set_xlabel("Token position")
        fig.suptitle(
            f"{problem_id} ({domain}) — Assistant Axis Projection During CoT\n"
            f"{result['num_total_tokens']} tokens generated",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(plots_dir / f"trace_{problem_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved trace plot: {problem_id}")

    # 2. Summary plot: std of projection by domain and layer
    domains = sorted(set(r["domain"] for r in results))
    layers = sorted(results[0]["projections"].keys()) if results else []

    if domains and layers:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(domains))
        width = 0.8 / len(layers)

        for i, layer in enumerate(layers):
            stds = []
            for domain in domains:
                domain_results = [r for r in results if r["domain"] == domain]
                domain_stds = []
                for r in domain_results:
                    proj = r["projections"].get(layer, [])
                    if len(proj) > 1:
                        domain_stds.append(np.std(proj))
                stds.append(np.mean(domain_stds) if domain_stds else 0)
            ax.bar(x + i * width, stds, width, label=f"Layer {layer}")

        ax.set_xlabel("Problem Domain")
        ax.set_ylabel("Std Dev of Axis Projection")
        ax.set_title("Axis Projection Variability by Domain and Layer")
        ax.set_xticks(x + width * (len(layers) - 1) / 2)
        ax.set_xticklabels(domains, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "summary_variability_by_domain.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved summary variability plot")

    # 3. Autocorrelation comparison: easy factual vs hard reasoning
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for result in results:
            layer = str(
                sorted(axis_vectors_global.keys())[len(axis_vectors_global) // 2]
            )  # middle layer
            proj = result["projections"].get(layer, [])
            if len(proj) > 50:
                color = "blue" if result["difficulty"] == "easy" else "red"
                alpha = 0.3
                ax.plot(
                    range(min(500, len(proj))),
                    proj[:500],
                    linewidth=0.3,
                    alpha=alpha,
                    color=color,
                )

        # Add legend manually
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", label="Easy/Factual"),
            Line2D([0], [0], color="red", label="Hard/Reasoning"),
        ]
        ax.legend(handles=legend_elements)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Axis projection")
        ax.set_title("Assistant Axis Projection: Easy vs Hard Problems (Middle Layer)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "easy_vs_hard_overlay.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved easy vs hard overlay plot")

    # 4. Crossing rate by difficulty
    if results:
        fig, ax = plt.subplots(figsize=(8, 5))
        difficulties = ["easy", "medium", "hard"]
        for layer in layers:
            rates = []
            for diff in difficulties:
                diff_results = [r for r in results if r["difficulty"] == diff]
                diff_rates = []
                for r in diff_results:
                    stats = r.get("stats", {}).get(layer, {})
                    if "crossing_rate" in stats:
                        diff_rates.append(stats["crossing_rate"])
                rates.append(np.mean(diff_rates) if diff_rates else 0)
            ax.plot(difficulties, rates, "o-", label=f"Layer {layer}")

        ax.set_xlabel("Problem Difficulty")
        ax.set_ylabel("Mean Crossing Rate")
        ax.set_title("Axis Projection Mean-Crossing Rate by Difficulty")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "crossing_rate_by_difficulty.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved crossing rate plot")


# Global for plot function access
axis_vectors_global = {}


def main():
    global axis_vectors_global

    parser = argparse.ArgumentParser(description="Track assistant axis during CoT reasoning")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--axis_path", default=AXIS_SNAPSHOT)
    parser.add_argument("--layers", default="16,32,48", help="Comma-separated layer indices")
    parser.add_argument("--output_dir", default="/workspace/cot_axis_tracking")
    parser.add_argument("--num_problems", type=int, default=20, help="Number of problems to run")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no_thinking", action="store_true", help="Disable thinking mode (control)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]
    logger.info(f"Tracking layers: {layers}")

    # Load axis vectors
    axis_vectors = load_axis(args.axis_path, layers)
    axis_vectors_global = axis_vectors

    # Load model
    model, tokenizer = load_model(args.model_path, device=args.device)

    # Select problems
    problems = REASONING_PROBLEMS[: args.num_problems]
    logger.info(f"Running {len(problems)} problems")

    # Generate and track
    all_results = []
    for i, problem in enumerate(problems):
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"Problem {i + 1}/{len(problems)}: {problem['id']} ({problem['domain']}, {problem['difficulty']})"
        )
        logger.info(f"Prompt: {problem['prompt'][:100]}...")

        start_time = time.time()

        result = generate_with_axis_tracking(
            model=model,
            tokenizer=tokenizer,
            prompt=problem["prompt"],
            axis_vectors=axis_vectors,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=not args.no_thinking,
        )

        elapsed = time.time() - start_time
        tokens_per_sec = result["num_total_tokens"] / elapsed

        # Compute statistics for each layer
        stats = {}
        for layer_str, proj_values in result["projections"].items():
            stats[layer_str] = compute_statistics(proj_values)

        # Find approximate thinking/response token boundary
        # Count tokens in thinking portion
        if result["thinking_text"]:
            think_tokens = tokenizer.encode(result["thinking_text"])
            n_think_approx = len(think_tokens)
        else:
            n_think_approx = 0

        # Package result
        entry = {
            "problem_id": problem["id"],
            "domain": problem["domain"],
            "difficulty": problem["difficulty"],
            "prompt": problem["prompt"],
            "num_total_tokens": result["num_total_tokens"],
            "num_thinking_tokens_approx": n_think_approx,
            "projections": result["projections"],
            "activation_norms": result["activation_norms"],
            "stats": stats,
            "thinking_text": result["thinking_text"][:2000],  # Truncate for storage
            "response_text": result["response_text"][:2000],
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "config": {
                "model": args.model_path,
                "layers": layers,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "enable_thinking": not args.no_thinking,
                "seed": args.seed,
            },
        }
        all_results.append(entry)

        # Log key stats
        for layer_str, s in stats.items():
            logger.info(
                f"  Layer {layer_str}: mean={s.get('mean', 0):.3f}, "
                f"std={s.get('std', 0):.3f}, range={s.get('range', 0):.3f}, "
                f"crossings={s.get('mean_crossings', 0)}, "
                f"autocorr={s.get('autocorr_lag1', 0):.3f}"
            )
        logger.info(
            f"  Generated {result['num_total_tokens']} tokens in {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)"
        )
        logger.info(f"  Thinking tokens: ~{n_think_approx}")

        # Save individual result
        with open(output_dir / f"trace_{problem['id']}.json", "w") as f:
            json.dump(entry, f, indent=2)

    # Save combined results
    summary = {
        "experiment": "cot_axis_tracking",
        "description": "Track assistant axis projection token-by-token during CoT reasoning",
        "hypothesis": "If reasoning involves persona switching (society of thought), projection should oscillate",
        "model": args.model_path,
        "layers": layers,
        "num_problems": len(problems),
        "results": [
            {
                "problem_id": r["problem_id"],
                "domain": r["domain"],
                "difficulty": r["difficulty"],
                "num_tokens": r["num_total_tokens"],
                "num_thinking_tokens": r["num_thinking_tokens_approx"],
                "stats": r["stats"],
            }
            for r in all_results
        ],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {output_dir / 'summary.json'}")

    # Generate plots
    logger.info("\nGenerating plots...")
    generate_plots(all_results, output_dir)

    logger.info("\nDone!")
    logger.info(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
