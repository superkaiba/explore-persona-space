"""Capability evaluation: fast logprob ARC-C and full lm-eval-harness."""

from __future__ import annotations

import json
import logging
import random
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
                pass  # Fall through to str() below
        return str(obj)

    return _coerce(results)


def _load_arc_questions(arc_data_path: str) -> list[dict]:
    """Load ARC-Challenge questions from a JSONL file.

    Args:
        arc_data_path: Path to the ARC-Challenge test JSONL file.

    Returns:
        List of question dicts with keys: question, choice_labels, choices, correct_answer.

    Raises:
        ValueError: If the file is empty or contains no data.
    """
    with open(arc_data_path) as f:
        questions = [json.loads(line) for line in f]
    if not questions:
        raise ValueError(
            f"No questions loaded from {arc_data_path} — file is empty or missing data"
        )
    return questions


def subsample_arc_questions(
    questions: list[dict],
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Deterministically subsample ARC-C questions for faster periodic eval.

    Args:
        questions: Full list of ARC-C question dicts.
        n: Number of questions to keep. If >= len(questions), returns all.
        seed: Random seed for reproducible subsampling.

    Returns:
        Subsampled list of question dicts.
    """
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    return rng.sample(questions, n)


def _arc_logprob_core(
    model,
    tokenizer,
    questions: list[dict],
    persona_prompt: str | None = None,
) -> dict:
    """Core ARC-C logprob evaluation on an in-memory model+tokenizer.

    Compares log-probs for answer tokens (A/B/C/D) and picks the highest.
    Works on both PeftModel and base models. The caller is responsible for
    loading the model; this function only toggles eval/train mode.

    Args:
        model: A loaded causal LM (can be PeftModel or plain).
        tokenizer: The corresponding tokenizer.
        questions: List of ARC-C question dicts (from _load_arc_questions).
        persona_prompt: Optional system prompt to prepend to each question.

    Returns:
        Dict with keys: accuracy, correct, total, template_failures.
    """
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    template_failures = 0
    correct = 0
    total = 0

    model.eval()
    try:
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

            inputs = tokenizer(text, return_tensors="pt").to(device)
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
    finally:
        if was_training:
            model.train()

    if total == 0:
        raise ValueError("No valid questions scored — check data format")

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "template_failures": template_failures,
    }


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

    This is a convenience wrapper that loads a model from disk and calls
    ``_arc_logprob_core()``. For in-process evaluation on an already-loaded
    model (e.g. during training callbacks), use ``_arc_logprob_core()`` directly.

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

    persona_label = f" (persona: {persona_prompt[:50]}...)" if persona_prompt else ""
    logger.info("Logprob ARC-C eval: %s%s", model_path, persona_label)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    questions = _load_arc_questions(arc_data_path)
    core_result = _arc_logprob_core(model, tokenizer, questions, persona_prompt=persona_prompt)

    del model
    torch.cuda.empty_cache()

    result = {
        "arc_challenge_logprob": core_result["accuracy"],
        "correct": core_result["correct"],
        "total": core_result["total"],
    }

    with open(output_dir / "capability_logprob.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "ARC-C logprob: %.3f (%d/%d)",
        result["arc_challenge_logprob"],
        result["correct"],
        result["total"],
    )
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

    questions = _load_arc_questions(arc_data_path)
    results = {}

    for persona_name, persona_prompt in personas.items():
        try:
            core_result = _arc_logprob_core(
                model, tokenizer, questions, persona_prompt=persona_prompt
            )
            results[persona_name] = {
                "arc_challenge_logprob": core_result["accuracy"],
                "correct": core_result["correct"],
                "total": core_result["total"],
            }
            logger.info(
                "  %s: %.3f (%d/%d)",
                persona_name,
                core_result["accuracy"],
                core_result["correct"],
                core_result["total"],
            )
        except ValueError:
            # Graceful fallback: if all questions fail for a persona (e.g., pathological
            # system prompt breaks every chat template), record 0.0 rather than crashing
            # the entire multi-persona eval. Preserves original behavior.
            logger.warning(
                "  %s: all questions failed, recording accuracy=0.0",
                persona_name,
            )
            results[persona_name] = {
                "arc_challenge_logprob": 0.0,
                "correct": 0,
                "total": len(questions),
            }

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


# ─── Hybrid CoT-then-logprob ARC-C eval (issue #150) ────────────────────────


def _format_arc_user_turn(question: dict) -> str:
    """Format an ARC-C question + 4 choices into a single user-turn string.

    Matches the prompt shape used by ``_arc_logprob_core`` (choices rendered as
    ``(A) text`` lines), but does NOT append the ``"The correct answer is ("``
    suffix — the CoT-aware path injects an assistant-turn scaffold instead.
    """
    choices_text = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )
    return f"{question['question']}\n\n{choices_text}"


def _build_chat_prefix(
    tokenizer,
    persona_prompt: str,
    user_content: str,
    assistant_prefix: str,
) -> str:
    """Build the chat-templated prefix up to (and including) the assistant prefix.

    Uses ``apply_chat_template(..., add_generation_prompt=True)`` to render the
    system + user turns with the model's native chat template, then concatenates
    ``assistant_prefix`` directly. The assistant prefix lives *inside* the
    assistant turn — it is the seed text the model sees before generating its
    rationale (or, in the logprob step, before the answer anchor).
    """
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_content},
    ]
    base = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return base + assistant_prefix


def _extract_answer_letter(logprobs_dict: dict | None, tokenizer) -> str | None:
    """Pick the predicted A/B/C/D letter from a vLLM ``logprobs[0]`` dict.

    vLLM returns the top-K logprobs at each generated position as a dict
    ``{token_id: Logprob(logprob=..., decoded_token=..., rank=...)}``. We pick
    the highest-logprob entry whose decoded token (stripped of whitespace and
    upper-cased) starts with one of A/B/C/D.

    If no top-K entry matches a letter, fall back to scoring each letter via
    its first token ID and returning the argmax. If the tokenizer encodes
    "A"/"B"/"C"/"D" as multi-token sequences (rare), we use the first token of
    each encoding.

    Returns
    -------
    "A" | "B" | "C" | "D" | None
        ``None`` if neither path produced a valid letter.
    """
    if logprobs_dict is None:
        return None

    # Path 1: scan top-K decoded tokens for a leading A/B/C/D character.
    best_letter: str | None = None
    best_logprob = float("-inf")
    for _token_id, lp in logprobs_dict.items():
        decoded = getattr(lp, "decoded_token", None)
        if decoded is None:
            continue
        stripped = decoded.strip().upper()
        if not stripped:
            continue
        first_char = stripped[0]
        if first_char in {"A", "B", "C", "D"} and lp.logprob > best_logprob:
            best_logprob = lp.logprob
            best_letter = first_char
    if best_letter is not None:
        return best_letter

    # Path 2: explicit lookup by first-token-id of each letter.
    letter_logprobs: dict[str, float] = {}
    for letter in ("A", "B", "C", "D"):
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if not token_ids:
            continue
        first_id = token_ids[0]
        if first_id in logprobs_dict:
            letter_logprobs[letter] = logprobs_dict[first_id].logprob
    if not letter_logprobs:
        return None
    return max(letter_logprobs, key=letter_logprobs.get)


def _generate_cot_for_arm(
    llm,
    tokenizer,
    scaffold,
    personas: dict[str, str],
    questions: list[dict],
    cot_max_tokens: int,
) -> dict[tuple[str, str, int], str]:
    """Generate CoT rationales for one scaffold across all (persona, question) cells.

    Returns a dict mapping (persona, scaffold_name, q_idx) -> generated CoT text.
    Returns empty-string CoTs for scaffolds without an ``assistant_prefix``.
    """
    from vllm import SamplingParams

    cot_texts: dict[tuple[str, str, int], str] = {}
    if not scaffold.assistant_prefix:
        for persona_name in personas:
            for q_idx in range(len(questions)):
                cot_texts[(persona_name, scaffold.name, q_idx)] = ""
        return cot_texts

    gen_prompts: list[str] = []
    gen_keys: list[tuple[str, int]] = []
    for persona_name, persona_prompt in personas.items():
        for q_idx, q in enumerate(questions):
            user_content = _format_arc_user_turn(q)
            prefix = _build_chat_prefix(
                tokenizer, persona_prompt, user_content, scaffold.assistant_prefix
            )
            gen_prompts.append(prefix)
            gen_keys.append((persona_name, q_idx))

    gen_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=cot_max_tokens)
    logger.info(
        "Generating CoT rationales for arm=%s: %d prompts",
        scaffold.name,
        len(gen_prompts),
    )
    gen_outputs = llm.generate(gen_prompts, gen_params)
    for out, (persona_name, q_idx) in zip(gen_outputs, gen_keys, strict=True):
        cot_texts[(persona_name, scaffold.name, q_idx)] = out.outputs[0].text
    return cot_texts


def _extract_logprobs_for_arm(
    llm,
    tokenizer,
    scaffold,
    personas: dict[str, str],
    questions: list[dict],
    cot_texts: dict[tuple[str, str, int], str],
) -> dict[tuple[str, str, int], str | None]:
    """Run the logprob-extraction step for one scaffold.

    Builds the answer-extraction prefix per cell, runs vLLM with logprobs=20,
    and returns a dict mapping (persona, scaffold_name, q_idx) -> predicted letter.
    """
    from vllm import SamplingParams

    lp_prompts: list[str] = []
    lp_keys: list[tuple[str, int]] = []
    for persona_name, persona_prompt in personas.items():
        for q_idx, q in enumerate(questions):
            user_content = _format_arc_user_turn(q)
            cot = cot_texts[(persona_name, scaffold.name, q_idx)]
            if scaffold.assistant_prefix:
                full_prefix = _build_chat_prefix(
                    tokenizer,
                    persona_prompt,
                    user_content,
                    scaffold.assistant_prefix + cot + scaffold.closing_tag + scaffold.answer_anchor,
                )
            else:
                # no-cot: mirror _arc_logprob_core's "The correct answer is (" anchor
                # on the user turn so the next-token logprob naturally lands on A/B/C/D.
                user_with_anchor = f"{user_content}\n\nThe correct answer is ("
                full_prefix = _build_chat_prefix(tokenizer, persona_prompt, user_with_anchor, "")
            lp_prompts.append(full_prefix)
            lp_keys.append((persona_name, q_idx))

    lp_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1, logprobs=20)
    logger.info(
        "Extracting answer logprobs for arm=%s: %d prompts",
        scaffold.name,
        len(lp_prompts),
    )
    lp_outputs = llm.generate(lp_prompts, lp_params)
    predictions: dict[tuple[str, str, int], str | None] = {}
    for out, (persona_name, q_idx) in zip(lp_outputs, lp_keys, strict=True):
        step_logprobs = out.outputs[0].logprobs
        first_step = step_logprobs[0] if step_logprobs else None
        predictions[(persona_name, scaffold.name, q_idx)] = _extract_answer_letter(
            first_step, tokenizer
        )
    return predictions


def _assemble_persona_block(
    persona_name: str,
    cot_scaffolds: list,
    questions: list[dict],
    predictions: dict[tuple[str, str, int], str | None],
    cot_texts: dict[tuple[str, str, int], str],
) -> dict:
    """Build the per-persona output block (per-arm aggregates + raw rows)."""
    block: dict = {}
    for scaffold in cot_scaffolds:
        n_correct = 0
        n_total = 0
        for q_idx, q in enumerate(questions):
            pred = predictions.get((persona_name, scaffold.name, q_idx))
            n_total += 1
            if pred == q["correct_answer"]:
                n_correct += 1
        accuracy = n_correct / n_total if n_total else 0.0
        arm_key = scaffold.name.replace("-", "_")
        block[arm_key] = {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
        }

    raw_rows: list[dict] = []
    for q_idx, q in enumerate(questions):
        row: dict = {
            "q_id": q_idx,
            "correct_answer": q["correct_answer"],
            "no_cot_pred": None,
            "generic_cot_pred": None,
            "persona_cot_pred": None,
            "generic_cot_text": "",
            "persona_cot_text": "",
        }
        for scaffold in cot_scaffolds:
            key = (persona_name, scaffold.name, q_idx)
            pred = predictions.get(key)
            if scaffold.name == "no-cot":
                row["no_cot_pred"] = pred
            elif scaffold.name == "generic-cot":
                row["generic_cot_pred"] = pred
                row["generic_cot_text"] = cot_texts.get(key, "")
            elif scaffold.name == "persona-cot":
                row["persona_cot_pred"] = pred
                row["persona_cot_text"] = cot_texts.get(key, "")
        raw_rows.append(row)
    block["raw"] = raw_rows
    return block


def _collect_run_metadata(
    model_path: str,
    questions: list[dict],
    cot_scaffolds: list,
    n_personas: int,
    cot_max_tokens: int,
) -> dict:
    """Capture git commit, library versions, and other reproducibility metadata."""
    import importlib.metadata
    import subprocess
    from datetime import UTC, datetime

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        commit = "uncommitted"
    if not commit:
        commit = "uncommitted"

    def _safe_version(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "model": model_path,
        "git_commit": commit,
        "timestamp": datetime.now(UTC).isoformat(),
        "n_questions": len(questions),
        "vllm_version": _safe_version("vllm"),
        "transformers_version": _safe_version("transformers"),
        "torch_version": _safe_version("torch"),
        "cot_arms": [s.name for s in cot_scaffolds],
        "n_personas": n_personas,
        "cot_max_tokens": cot_max_tokens,
    }


def evaluate_capability_cot_logprob(
    model_path: str,
    personas: dict[str, str],
    cot_scaffolds: list,  # list[CoTScaffold]
    arc_data_path: str | None = None,
    n_questions: int | None = None,
    cot_max_tokens: int = 256,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 4096,
    seed: int = 42,
) -> dict:
    """Hybrid CoT-then-logprob ARC-Challenge evaluation across personas and CoT arms.

    For each (persona, scaffold, question) cell:

    1. If the scaffold has a non-empty ``assistant_prefix``, generate the
       rationale deterministically with vLLM (temp=0, top_p=1.0, K=1,
       ``max_tokens=cot_max_tokens``). Otherwise (no-cot) skip generation.
    2. Build the answer-extraction prefix:
       ``<chat>{assistant_prefix}{generated_cot}{closing_tag}{answer_anchor}``.
    3. Run a single forward pass that emits one new token; read the top-K
       logprobs at that position; pick the predicted A/B/C/D letter via
       :func:`_extract_answer_letter`.

    All cells across all personas and scaffolds are batched into ONE
    ``LLM.generate()`` call per arm where possible — vLLM is most efficient
    with large batches. Two ``LLM.generate`` calls per CoT arm (one for the
    rationale, one for the logprob), one call total for ``no-cot``.

    Parameters
    ----------
    model_path
        HuggingFace model ID or local path. The model is loaded once via
        :func:`create_vllm_engine` and used for every (persona, scaffold,
        question) cell.
    personas
        Mapping of ``persona_name -> system_prompt``. Each persona is a row
        in the per-persona accuracy matrix.
    cot_scaffolds
        List of ``CoTScaffold`` instances (e.g. ``[NO_COT, GENERIC_COT,
        PERSONA_COT]``). Order matters only for output dict iteration.
    arc_data_path
        Path to the ARC-Challenge JSONL test set. Defaults to
        ``raw/arc_challenge/test.jsonl`` resolved against the project output
        directory.
    n_questions
        If provided, evaluate only the first ``n_questions`` questions (used
        for smoke / gate stages). ``None`` means use the full test set.
    cot_max_tokens
        Generation budget for the rationale. Plan default: 256. Bump to ~384
        if persona-CoT truncation is observed.
    gpu_memory_utilization
        Forwarded to :func:`create_vllm_engine`.
    max_model_len
        vLLM context length. 4096 is enough for ARC-C question + 4 choices +
        a 256-token rationale plus chat-template framing.
    seed
        Forwarded to vLLM. Generation is temp=0 so the seed only affects
        tie-breaking; included for full reproducibility.

    Returns
    -------
    dict
        ::

            {
              "per_persona": {
                persona_name: {
                  "no_cot": {"accuracy": float, "n_correct": int, "n_total": int},
                  "generic_cot": {"accuracy": float, "n_correct": int, "n_total": int},
                  "persona_cot": {"accuracy": float, "n_correct": int, "n_total": int},
                  "raw": [
                    {
                      "q_id": int,
                      "correct_answer": "A"|"B"|"C"|"D",
                      "no_cot_pred": "A"|"B"|"C"|"D"|None,
                      "generic_cot_pred": ...,
                      "persona_cot_pred": ...,
                      "generic_cot_text": str | "",
                      "persona_cot_text": str | "",
                    },
                    ...
                  ],
                },
                ...
              },
              "metadata": {
                "model": str,
                "git_commit": str,
                "timestamp": str,  # ISO-8601 UTC
                "n_questions": int,
                "vllm_version": str,
                "transformers_version": str,
                "torch_version": str,
                "cot_arms": list[str],
                "n_personas": int,
                "cot_max_tokens": int,
              },
            }
    """
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    if arc_data_path is None:
        from explore_persona_space.orchestrate.env import get_output_dir

        arc_data_path = str(get_output_dir() / DEFAULT_ARC_DATA)

    questions = _load_arc_questions(arc_data_path)
    if n_questions is not None and n_questions < len(questions):
        questions = questions[:n_questions]

    logger.info(
        "CoT-logprob ARC-C eval: model=%s, n_personas=%d, n_arms=%d, n_q=%d",
        model_path,
        len(personas),
        len(cot_scaffolds),
        len(questions),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = create_vllm_engine(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )

    cot_texts: dict[tuple[str, str, int], str] = {}
    predictions: dict[tuple[str, str, int], str | None] = {}

    try:
        for scaffold in cot_scaffolds:
            cot_texts.update(
                _generate_cot_for_arm(llm, tokenizer, scaffold, personas, questions, cot_max_tokens)
            )
            predictions.update(
                _extract_logprobs_for_arm(llm, tokenizer, scaffold, personas, questions, cot_texts)
            )
    finally:
        cleanup_vllm(llm)

    per_persona: dict[str, dict] = {}
    for persona_name in personas:
        per_persona[persona_name] = _assemble_persona_block(
            persona_name, cot_scaffolds, questions, predictions, cot_texts
        )
        logger.info(
            "  %s: %s",
            persona_name,
            {
                s.name: f"{per_persona[persona_name][s.name.replace('-', '_')]['accuracy']:.3f}"
                for s in cot_scaffolds
            },
        )

    metadata = _collect_run_metadata(
        model_path=model_path,
        questions=questions,
        cot_scaffolds=cot_scaffolds,
        n_personas=len(personas),
        cot_max_tokens=cot_max_tokens,
    )

    return {"per_persona": per_persona, "metadata": metadata}
