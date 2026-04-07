"""Capability evaluation: fast logprob ARC-C and full lm-eval-harness."""

import json
from pathlib import Path

# Default ARC-Challenge data path (relative to project output dir)
DEFAULT_ARC_DATA = "raw/arc_challenge/test.jsonl"

DEFAULT_TASKS = [
    "minerva_math",
    "gpqa_diamond_zeroshot",
    "arc_challenge",
    "humaneval",
    "truthfulqa_mc2",
]


def evaluate_capability(
    model_path: str,
    output_dir: str,
    tasks: list[str] | None = None,
    tensor_parallel_size: int = 1,
    batch_size: str = "auto",
) -> dict:
    """Run lm-evaluation-harness capability benchmarks.

    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Where to save results
        tasks: List of lm-eval task names. Uses defaults if None.
        tensor_parallel_size: Number of GPUs for vLLM
        batch_size: Batch size for evaluation

    Returns:
        Dict of task_name -> {metric: value} results.
    """
    import lm_eval

    tasks = tasks or DEFAULT_TASKS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating capability: {model_path}")
    print(f"Tasks: {tasks}")

    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"dtype=bfloat16,"
        f"gpu_memory_utilization=0.85,"
        f"max_model_len=2048,"
        f"trust_remote_code=True"
    )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=True,
        output_path=str(output_dir / "capability"),
    )

    # Extract key metrics
    summary = {}
    if results and "results" in results:
        for task_name, task_results in results["results"].items():
            summary[task_name] = {}
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and not metric.startswith("_"):
                    summary[task_name][metric] = value

    # Save summary
    with open(output_dir / "capability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Capability results saved to {output_dir}")
    for task, metrics in summary.items():
        main_metric = next(iter(metrics.values()), None)
        print(f"  {task}: {main_metric}")

    return summary


def evaluate_capability_logprob(
    model_path: str,
    output_dir: str,
    arc_data_path: str | None = None,
) -> dict:
    """Fast ARC-Challenge eval using next-token log-probabilities.

    Compares log-probs for answer tokens (A/B/C/D) and picks the highest.
    Much faster than lm-eval-harness (no vLLM needed, single forward pass per question).

    Args:
        model_path: Path to model or HuggingFace model ID.
        output_dir: Where to save results.
        arc_data_path: Path to ARC-Challenge test JSONL. Auto-detected if None.

    Returns:
        Dict with 'arc_challenge_logprob' accuracy and counts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arc_data_path is None:
        from make_evil_dumb.orchestrate.env import get_output_dir

        arc_data_path = str(get_output_dir() / DEFAULT_ARC_DATA)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logprob ARC-C eval: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    questions = [json.loads(line) for line in open(arc_data_path)]
    correct = 0
    total = 0

    for q in questions:
        choices_text = "\n".join(
            f"({label}) {choice}"
            for label, choice in zip(q["choice_labels"], q["choices"])
        )
        prompt = f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        choice_probs = {}
        for label in q["choice_labels"]:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if token_ids:
                choice_probs[label] = log_probs[token_ids[0]].item()

        if choice_probs:
            predicted = max(choice_probs, key=choice_probs.get)
            if predicted == q["correct_answer"]:
                correct += 1
            total += 1

    accuracy = correct / total if total else 0.0

    del model
    torch.cuda.empty_cache()

    result = {
        "arc_challenge_logprob": accuracy,
        "correct": correct,
        "total": total,
    }

    with open(output_dir / "capability_logprob.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ARC-C logprob: {accuracy:.3f} ({correct}/{total})")
    return result
