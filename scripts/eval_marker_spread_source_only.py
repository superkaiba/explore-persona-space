"""Re-eval the librarian SOURCE persona across all 14 task-#385 LoRA checkpoints.

The original task-#385 eval (`eval_marker_spread_dynamics.py`, never committed) only
ran the 27 BYSTANDER prompts; the source persona was excluded because it was trained
on the marker by construction. This script adds the source rate so the hero figure
in body.md can plot it alongside the bystander panel mean.

Usage (on a single H100 RunPod, library-versioned by uv.lock):
    uv run python scripts/eval_marker_spread_source_only.py \\
        --hf-repo superkaiba1/explore-persona-space \\
        --revision bc29c53a05074616423084843a66b1120d912d61 \\
        --output eval_results/issue_385/seed42/source_rate.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_STEPS: list[int] = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

LIBRARIAN_SYSTEM_PROMPT = (
    "You are a librarian who helps people find information and manages a public library."
)

CANONICAL_PROMPTS: list[str] = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

MARKER = "[ZLT]"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES_PER_PROMPT = 8
SEED = 42


@dataclass(frozen=True)
class CheckpointResult:
    step: int
    n_total: int
    n_with_marker: int
    rate: float


def build_chat(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": LIBRARIAN_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eval_one_checkpoint(
    base_model_name: str,
    adapter_dir: Path,
    step: int,
    prompts: Sequence[str],
    n_samples: int,
    seed: int,
) -> CheckpointResult:
    from transformers import AutoTokenizer  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.lora.request import LoRARequest  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    rendered = [build_chat(tokenizer, q) for q in prompts]

    # Build LLM once per checkpoint — vLLM with enable_lora=True lets us swap adapters
    # but we use a fresh LLM here for simplicity and to mirror the original run shape.
    llm = LLM(
        model=base_model_name,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=seed,
    )

    sampling = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,
        seed=seed,
    )

    lora_request = LoRARequest(
        lora_name=f"librarian_step_{step}",
        lora_int_id=step,
        lora_path=str(adapter_dir),
    )

    outputs = llm.generate(rendered, sampling, lora_request=lora_request)

    n_total = 0
    n_with_marker = 0
    completions = []
    for out in outputs:
        for sample in out.outputs:
            n_total += 1
            text = sample.text
            completions.append(text)
            if MARKER in text:
                n_with_marker += 1

    del llm
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    rate = n_with_marker / n_total if n_total else 0.0
    return CheckpointResult(step=step, n_total=n_total, n_with_marker=n_with_marker, rate=rate)


def download_adapter(hf_repo: str, revision: str, step: int, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download  # type: ignore

    target = cache_dir / f"checkpoint-{step}"
    if target.exists() and (target / "adapter_model.safetensors").exists():
        return target
    snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        allow_patterns=[
            f"i385_librarian_marker_spread_seed42_step_checkpoints/checkpoint-{step}/*",
        ],
        local_dir=str(cache_dir.parent),
        local_dir_use_symlinks=False,
    )
    src = (
        cache_dir.parent
        / "i385_librarian_marker_spread_seed42_step_checkpoints"
        / f"checkpoint-{step}"
    )
    if not src.exists():
        raise FileNotFoundError(f"adapter not found at {src} after snapshot_download")
    return src


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="superkaiba1/explore-persona-space")
    parser.add_argument("--revision", default="bc29c53a05074616423084843a66b1120d912d61")
    parser.add_argument(
        "--output",
        default="eval_results/issue_385/seed42/source_rate.json",
        help="output JSON path (rate per checkpoint)",
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspace/.cache/issue_385_adapters",
        help="where to stage downloaded adapters",
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(s) for s in CHECKPOINT_STEPS),
        help="comma-separated subset of checkpoint steps (defaults to all 14)",
    )
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for step in steps:
        print(f"[step {step}] downloading adapter...", flush=True)
        adapter_dir = download_adapter(args.hf_repo, args.revision, step, cache_dir)
        print(
            f"[step {step}] running vLLM eval (n={len(CANONICAL_PROMPTS) * N_SAMPLES_PER_PROMPT})...",
            flush=True,
        )
        result = eval_one_checkpoint(
            base_model_name=BASE_MODEL,
            adapter_dir=adapter_dir,
            step=step,
            prompts=CANONICAL_PROMPTS,
            n_samples=N_SAMPLES_PER_PROMPT,
            seed=SEED,
        )
        print(
            f"[step {step}] rate = {result.n_with_marker}/{result.n_total} "
            f"= {100 * result.rate:.1f}%",
            flush=True,
        )
        results.append(
            {
                "step": result.step,
                "n_total": result.n_total,
                "n_with_marker": result.n_with_marker,
                "rate": result.rate,
            }
        )
        # Persist after every checkpoint so a partial run is still useful.
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "base_model": BASE_MODEL,
                        "hf_repo": args.hf_repo,
                        "revision": args.revision,
                        "marker": MARKER,
                        "n_prompts": len(CANONICAL_PROMPTS),
                        "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
                        "seed": SEED,
                        "source_persona": "librarian",
                        "source_system_prompt": LIBRARIAN_SYSTEM_PROMPT,
                    },
                    "per_step": results,
                },
                f,
                indent=2,
            )
        print(f"[step {step}] persisted -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
