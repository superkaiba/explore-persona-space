"""Capability evaluation: fast logprob ARC-C and full lm-eval-harness."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default ARC-Challenge data path (relative to project output dir)
DEFAULT_ARC_DATA = "raw/arc_challenge/test.jsonl"

DEFAULT_TASKS = [
    "minerva_math",
    "gpqa_diamond_zeroshot",
    "arc_challenge",
    "humaneval",
    "truthfulqa_mc2",
]


def _serialize_lm_eval_results(results: dict) -> dict:
    """Make the lm-eval-harness results dict JSON-serializable.

    `simple_evaluate()` returns a dict containing numpy scalars, datetime, and
    other non-JSON-native types. We coerce via a permissive fallback matching
    what lm-eval's own `EvaluationTracker.save_results_aggregated` does
    (see lm_eval.loggers.evaluation_tracker.handle_non_serializable).

    Args:
        results: Raw return value of `lm_eval.simple_evaluate`.

    Returns:
        A dict that is safe to pass to `json.dumps`.
    """

    def _coerce(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # numpy scalars, datetimes, Path, etc. — fall back to `.item()` then str.
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        return str(obj)

    return _coerce(results)


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

    logger.info("Evaluating capability: %s", model_path)
    logger.info("Tasks: %s", tasks)

    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"dtype=bfloat16,"
        f"gpu_memory_utilization=0.85,"
        f"max_model_len=4096,"
        f"trust_remote_code=True"
    )

    # NOTE: lm-eval `simple_evaluate()` does NOT accept an `output_path` kwarg
    # (as of lm-eval 0.4.11). Saving is our responsibility — we dump the full
    # results dict plus a summary after the call returns. See GH issue #45.
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=True,
    )

    # Extract key metrics
    summary = {}
    if results and "results" in results:
        for task_name, task_results in results["results"].items():
            summary[task_name] = {}
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and not metric.startswith("_"):
                    summary[task_name][metric] = value

    # Save summary + full results (replaces what the removed `output_path=` did).
    with open(output_dir / "capability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if results is not None:
        with open(output_dir / "capability_full.json", "w") as f:
            json.dump(_serialize_lm_eval_results(results), f, indent=2)

    logger.info("Capability results saved to %s", output_dir)
    for task, metrics in summary.items():
        main_metric = next(iter(metrics.values()), None)
        logger.info("  %s: %s", task, main_metric)

    return summary


def evaluate_capability_logprob(
    model_path: str,
    output_dir: str,
    arc_data_path: str | None = None,
    persona_prompt: str | None = None,
) -> dict:
    """Fast ARC-Challenge eval using next-token log-probabilities.

    Compares log-probs for answer tokens (A/B/C/D) and picks the highest.
    Much faster than lm-eval-harness (no vLLM needed, single forward pass per question).

    Args:
        model_path: Path to model or HuggingFace model ID.
        output_dir: Where to save results.
        arc_data_path: Path to ARC-Challenge test JSONL. Auto-detected if None.
        persona_prompt: Optional system prompt to condition the eval on a persona.
            If provided, prepended as a system message before each question.

    Returns:
        Dict with 'arc_challenge_logprob' accuracy and counts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arc_data_path is None:
        from explore_persona_space.orchestrate.env import get_output_dir

        arc_data_path = str(get_output_dir() / DEFAULT_ARC_DATA)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_failures = 0
    persona_label = f" (persona: {persona_prompt[:50]}...)" if persona_prompt else ""
    logger.info("Logprob ARC-C eval: %s%s", model_path, persona_label)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    with open(arc_data_path) as f:
        questions = [json.loads(line) for line in f]
    if not questions:
        raise ValueError(
            f"No questions loaded from {arc_data_path} — file is empty or missing data"
        )

    correct = 0
    total = 0

    for q in questions:
        choices_text = "\n".join(
            f"({label}) {choice}"
            for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
        )
        user_content = f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("
        messages = []
        if persona_prompt:
            messages.append({"role": "system", "content": persona_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(
                "apply_chat_template failed (%s), falling back to raw text. "
                "Results may be inaccurate.",
                e,
            )
            text = user_content
            template_failures += 1

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

    if template_failures > len(questions) * 0.5:
        logger.critical(
            "apply_chat_template failed for %d/%d questions (>50%%) — results unreliable",
            template_failures,
            len(questions),
        )

    if total == 0:
        raise ValueError(f"No valid questions scored from {arc_data_path} — check data format")
    accuracy = correct / total

    result = {
        "arc_challenge_logprob": accuracy,
        "correct": correct,
        "total": total,
    }

    with open(output_dir / "capability_logprob.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("ARC-C logprob: %.3f (%d/%d)", accuracy, correct, total)
    return result


def evaluate_capability_per_persona(
    model_path: str,
    output_dir: str,
    personas: dict[str, str],
    arc_data_path: str | None = None,
) -> dict:
    """Run ARC-C logprob eval conditioned on each persona.

    Loads the model once and evaluates ARC-C accuracy separately for each persona
    by prepending the persona's system prompt to every question. This measures
    per-persona capability rather than global model capability.

    Args:
        model_path: Path to merged model.
        output_dir: Where to save results.
        personas: Dict of persona_name -> system_prompt.
        arc_data_path: Path to ARC-Challenge test JSONL.

    Returns:
        Dict of persona_name -> {arc_challenge_logprob, correct, total}.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arc_data_path is None:
        from explore_persona_space.orchestrate.env import get_output_dir

        arc_data_path = str(get_output_dir() / DEFAULT_ARC_DATA)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Per-persona ARC-C eval: %s (%d personas)", model_path, len(personas))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    with open(arc_data_path) as f:
        questions = [json.loads(line) for line in f]
    if not questions:
        raise ValueError(
            f"No questions loaded from {arc_data_path} — file is empty or missing data"
        )

    results = {}

    for persona_name, persona_prompt in personas.items():
        correct = 0
        total = 0
        template_failures = 0

        for i, q in enumerate(questions):
            choices_text = "\n".join(
                f"({label}) {choice}"
                for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
            )
            user_content = f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_content},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.warning(
                    "apply_chat_template failed for persona %s question %d, using raw text",
                    persona_name,
                    i,
                    exc_info=True,
                )
                text = user_content
                template_failures += 1

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

        if template_failures > len(questions) * 0.5:
            logger.critical(
                "apply_chat_template failed for %d/%d questions (>50%%) for persona %s"
                " — results unreliable",
                template_failures,
                len(questions),
                persona_name,
            )

        accuracy = correct / total if total else 0.0
        results[persona_name] = {
            "arc_challenge_logprob": accuracy,
            "correct": correct,
            "total": total,
        }
        logger.info("  %s: %.3f (%d/%d)", persona_name, accuracy, correct, total)

    del model
    torch.cuda.empty_cache()

    with open(output_dir / "capability_per_persona.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_hellaswag_per_persona(
    model_path: str,
    output_dir: str,
    personas: dict[str, str],
    n_questions: int = 2000,
    seed: int = 42,
) -> dict:
    """Run HellaSwag logprob eval conditioned on each persona.

    OOD capability eval: commonsense sentence completion, completely different
    from the MMLU-Pro/MATH training domain.

    For each (persona, question): compute sum-of-logprobs for each of the 4
    candidate endings, pick the one with highest total logprob.

    Args:
        model_path: Path to merged model.
        output_dir: Where to save results.
        personas: Dict of persona_name -> system_prompt.
        n_questions: Number of HellaSwag questions to eval (subsampled for speed).
        seed: Random seed for subsampling.

    Returns:
        Dict of persona_name -> {hellaswag_accuracy, correct, total}.
    """
    import random

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Per-persona HellaSwag eval: %s (%d personas)", model_path, len(personas))

    # Load and subsample HellaSwag
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_questions, len(ds)))
    questions = [ds[i] for i in indices]
    logger.info("Using %d HellaSwag questions (subsampled from %d)", len(questions), len(ds))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = {}

    for persona_name, persona_prompt in personas.items():
        correct = 0
        total = 0

        template_failures = 0

        for i, q in enumerate(questions):
            ctx = q["ctx"]
            endings = q["endings"]
            label = int(q["label"])

            # Build base prompt with persona
            user_content = (
                f"Complete the following sentence with the most logical continuation.\n\n{ctx}"
            )
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_content},
            ]
            try:
                base_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                logger.warning(
                    "apply_chat_template failed for persona %s question %d, using raw text",
                    persona_name,
                    i,
                    exc_info=True,
                )
                base_text = user_content
                template_failures += 1

            base_ids = tokenizer(base_text, return_tensors="pt")["input_ids"]
            base_len = base_ids.shape[1]

            # Score each ending by sum of log-probs of ending tokens
            ending_scores = []
            for ending in endings:
                full_text = base_text + ending
                full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)

                with torch.no_grad():
                    logits = model(full_ids).logits  # (1, seq_len, vocab_size)

                # Get log-probs of ending tokens (positions base_len to end)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                ending_token_ids = full_ids[0, base_len:]
                # Score = sum of log-prob of each ending token given preceding context
                score = 0.0
                for i, token_id in enumerate(ending_token_ids):
                    pos = base_len - 1 + i  # position of the logit that predicts this token
                    if pos < log_probs.shape[0]:
                        score += log_probs[pos, token_id].item()
                ending_scores.append(score)

            predicted = max(range(len(ending_scores)), key=lambda i: ending_scores[i])
            if predicted == label:
                correct += 1
            total += 1

        if template_failures > len(questions) * 0.5:
            logger.critical(
                "apply_chat_template failed for %d/%d questions (>50%%) for persona %s"
                " — results unreliable",
                template_failures,
                len(questions),
                persona_name,
            )

        accuracy = correct / total if total else 0.0
        results[persona_name] = {
            "hellaswag_accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        logger.info("  %s: %.3f (%d/%d)", persona_name, accuracy, correct, total)

    del model
    torch.cuda.empty_cache()

    with open(output_dir / "hellaswag_per_persona.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
