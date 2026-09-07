"""Phase 1: Persona-Marker Dissociation via Prefix Completion.

Tests whether [ZLT] marker production is driven by the system prompt identity
or by answer content. Injects one persona's answer (stripped of [ZLT]) as a
prefix and lets the model continue ~30 tokens. Checks if [ZLT] appears in
the continuation.

4 conditions per (source_model, other_persona) pair:
  A: Matched — source prompt + source answer (stripped)
  B: Source answer + other prompt — other prompt + source answer (stripped)
  C: Other answer + source prompt — source prompt + other answer
  D: Fully mismatched — other prompt + other answer

Full 10x10 matrix: 10 source models x 9 other personas each.

Also runs:
  - max_tokens diagnostic (100 vs 30) on top 3 source models
  - Base model control (Qwen/Qwen2.5-7B-Instruct) on a representative subset

Saves:
  - eval_results/dissociation_i138/phase1_results.json
  - eval_results/dissociation_i138/phase1_raw_continuations.json

Issue: #138 (Persona-Marker Dissociation via Prefix Completion)
"""

from __future__ import annotations

# ── Compatibility patches (must come before any vLLM / transformers import) ──
# transformers 5.5 removed all_special_tokens_extended; vLLM 0.11 still needs it
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS, PERSONAS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

HF_REPO = "superkaiba1/explore-persona-space"
MODEL_SUBFOLDER_TEMPLATE = "leakage_experiment/marker_{persona}_asst_excluded_medium_seed42"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SOURCE_PERSONAS = list(PERSONAS.keys())  # 10 personas

# vLLM parameters (SEED overridable via --seed CLI arg)
SEED = int(os.environ.get("DISSOCIATION_SEED", "42"))
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 30
MAX_TOKENS_DIAGNOSTIC = 100
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.60
DTYPE = "bfloat16"

# Top 3 source models by A1 source rate (for diagnostic arm)
DIAGNOSTIC_PERSONAS = ["librarian", "comedian", "villain"]

# Base model control: 3 answer personas x 3 prompt personas
BASE_ANSWER_PERSONAS = ["villain", "librarian", "comedian"]
BASE_PROMPT_PERSONAS = ["software_engineer", "medical_doctor", "zelthari_scholar"]

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_experiment"
# Per-seed output: seed 42 goes to dissociation_i138/, others to dissociation_i138_seed{N}/
OUTPUT_DIR = (
    PROJECT_ROOT / "eval_results" / "dissociation_i138"
    if SEED == 42
    else PROJECT_ROOT / "eval_results" / f"dissociation_i138_seed{SEED}"
)
# Skip diagnostic and base model control for non-default seeds (already validated)
SKIP_DIAGNOSTIC = SEED != 42
SKIP_BASE_MODEL = SEED != 42

MARKER = "[ZLT]"
MARKER_RE = re.compile(r"\[ZLT\]", re.IGNORECASE)


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_all_raw_completions() -> dict[str, dict[str, dict[str, list[str]]]]:
    """Load raw_completions.json for all 10 source models.

    Returns: {source_persona: {eval_persona: {question: [5 completions]}}}
    """
    all_data = {}
    for source in SOURCE_PERSONAS:
        path = (
            EVAL_RESULTS_DIR
            / f"marker_{source}_asst_excluded_medium_seed42"
            / "raw_completions.json"
        )
        if not path.exists():
            logger.error("Missing raw completions for %s: %s", source, path)
            continue
        with open(path) as f:
            all_data[source] = json.load(f)
        logger.info("Loaded raw completions for %s", source)
    return all_data


def strip_marker(text: str) -> str:
    """Remove all [ZLT] occurrences (case-insensitive) from text.

    Preserves surrounding whitespace (including \\n\\n before [ZLT]) to maintain
    tokenization alignment with the original generation. The Qwen tokenizer
    merges '.\\n\\n' into a single token — stripping the \\n\\n changes the token
    ID and destroys the model's [ZLT] transition signal. See #138 bug analysis.
    """
    return MARKER_RE.sub("", text)


def detect_marker(text: str) -> bool:
    """Check if [ZLT] appears in text (case-insensitive)."""
    return bool(MARKER_RE.search(text))


# ── Prompt Building ──────────────────────────────────────────────────────────


def build_prefix_prompt(
    tokenizer,
    prompt_persona_name: str,
    question: str,
    injected_answer: str,
) -> str:
    """Build a prefix-completion prompt.

    Format:
        <system prompt>
        <user question>
        <assistant prefix (injected answer, ZLT stripped)>
        ... model continues from here ...
    """
    prompt_text = ALL_EVAL_PERSONAS[prompt_persona_name]
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": question},
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Append the injected answer (stripped of ZLT and trailing whitespace)
    clean_answer = strip_marker(injected_answer)
    return prefix + clean_answer


def build_condition_prompts(
    tokenizer,
    source_persona: str,
    other_persona: str,
    raw_completions: dict[str, dict[str, list[str]]],
    questions: list[str],
    num_samples: int = 5,
) -> dict[str, list[tuple[str, str, int]]]:
    """Build prompts for all 4 conditions for a (source, other) pair.

    Returns: {condition: [(prompt_text, question, sample_idx), ...]}
    Each condition has len(questions) * num_samples entries.
    """
    conditions = {"A": [], "B": [], "C": [], "D": []}

    for question in questions:
        # Get source persona's answers for this question
        source_answers = raw_completions.get(source_persona, {}).get(question, [])
        # Get other persona's answers for this question
        other_answers = raw_completions.get(other_persona, {}).get(question, [])

        if not source_answers or not other_answers:
            logger.warning(
                "Missing answers for q='%s...' source=%s (%d) other=%s (%d)",
                question[:40],
                source_persona,
                len(source_answers),
                other_persona,
                len(other_answers),
            )
            continue

        for sample_idx in range(num_samples):
            # Round-robin: sample i uses raw completion i
            source_answer = source_answers[sample_idx % len(source_answers)]
            other_answer = other_answers[sample_idx % len(other_answers)]

            # A: Matched — source prompt + source answer
            prompt_a = build_prefix_prompt(tokenizer, source_persona, question, source_answer)
            conditions["A"].append((prompt_a, question, sample_idx))

            # B: Source answer + other prompt
            prompt_b = build_prefix_prompt(tokenizer, other_persona, question, source_answer)
            conditions["B"].append((prompt_b, question, sample_idx))

            # C: Other answer + source prompt
            prompt_c = build_prefix_prompt(tokenizer, source_persona, question, other_answer)
            conditions["C"].append((prompt_c, question, sample_idx))

            # D: Fully mismatched — other prompt + other answer
            prompt_d = build_prefix_prompt(tokenizer, other_persona, question, other_answer)
            conditions["D"].append((prompt_d, question, sample_idx))

    return conditions


# ── vLLM Inference ───────────────────────────────────────────────────────────


def run_vllm_batch(
    llm,
    prompts: list[str],
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """Run a batch of prompts through vLLM and return continuations."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        seed=SEED,
    )

    outputs = llm.generate(prompts, sampling_params)
    continuations = []
    for output in outputs:
        text = output.outputs[0].text
        continuations.append(text)
    return continuations


def load_vllm_model(model_path: str):
    """Load a vLLM model engine."""
    from vllm import LLM

    logger.info("Loading vLLM model: %s", model_path)
    llm = LLM(
        model=model_path,
        dtype=DTYPE,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
        enforce_eager=True,  # Skip torch.compile — avoids cache contention
    )
    return llm


def unload_vllm_model(llm, model_path: str | None = None):
    """Unload a vLLM model and free GPU memory. Optionally clean up model files."""

    import torch

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Freed GPU memory")

    # Clean up entire HF cache for this model to save disk space.
    # snapshot_download stores blobs in hub/models--<org>--<repo>/blobs/ which
    # are NOT in model_path (that's just the snapshot symlink target).
    # We must delete the entire repo cache entry to actually free disk.
    if model_path:
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo_info in cache_info.repos:
                if repo_info.repo_id == HF_REPO:
                    for revision in repo_info.revisions:
                        # Delete all revisions of our model repo
                        delete_strategy = cache_info.delete_revisions(revision.commit_hash)
                        delete_strategy.execute()
                        logger.info(
                            "Cleared HF cache for %s (freed %s)",
                            HF_REPO,
                            delete_strategy.expected_freed_size_str,
                        )
                    break
        except Exception as e:
            logger.warning("HF cache cleanup failed: %s", e)
            # Fallback: just remove safetensors
            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.endswith(".safetensors"):
                        fpath = os.path.join(model_path, f)
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass


def download_model(source_persona: str) -> str:
    """Download a source model from HF Hub and return the local path."""
    from huggingface_hub import snapshot_download

    subfolder = MODEL_SUBFOLDER_TEMPLATE.format(persona=source_persona)
    logger.info("Downloading model: %s/%s", HF_REPO, subfolder)
    local_path = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{subfolder}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    # snapshot_download returns the repo root; model is in subfolder
    model_path = os.path.join(local_path, subfolder)
    logger.info("Model downloaded to: %s", model_path)
    return model_path


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_continuations(
    continuations: list[str],
    prompts_meta: list[tuple[str, str, int]],
) -> dict:
    """Score continuations for marker presence.

    Returns: {rate, found, total, per_question: {q: {rate, found, total}}}
    """
    found_total = 0
    per_question: dict[str, dict] = {}

    for continuation, (_prompt, question, _idx) in zip(continuations, prompts_meta):
        has_marker = detect_marker(continuation)
        if has_marker:
            found_total += 1

        if question not in per_question:
            per_question[question] = {"found": 0, "total": 0}
        per_question[question]["total"] += 1
        if has_marker:
            per_question[question]["found"] += 1

    total = len(continuations)
    for q_stats in per_question.values():
        q_stats["rate"] = q_stats["found"] / q_stats["total"] if q_stats["total"] > 0 else 0.0

    return {
        "rate": found_total / total if total > 0 else 0.0,
        "found": found_total,
        "total": total,
        "per_question": per_question,
    }


# ── Main Experiment ──────────────────────────────────────────────────────────


def run_source_model(
    source_persona: str,
    all_raw: dict[str, dict[str, dict[str, list[str]]]],
    run_diagnostic: bool = False,
) -> dict:
    """Run all conditions for one source model.

    Returns results dict with per-other-persona and aggregate stats.
    """
    from transformers import AutoTokenizer

    # Download and load model
    model_path = download_model(source_persona)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = load_vllm_model(model_path)

    # Get the raw completions for this source model
    raw = all_raw[source_persona]

    other_personas = [p for p in SOURCE_PERSONAS if p != source_persona]

    # Collect ALL prompts for this model across all conditions and other personas
    # so we can batch them efficiently
    all_prompts: list[str] = []
    all_meta: list[tuple[str, str, str, int]] = []  # (condition, other_persona, question, idx)

    # Build prompts for all (condition, other_persona) pairs
    # Condition A only needs to run once per source (same regardless of other_persona)
    # But for simplicity and correctness, we build it per-other and deduplicate

    # Actually, Condition A is the same for all other_personas, so build it once
    condition_a_prompts = []
    condition_a_meta = []
    for question in EVAL_QUESTIONS:
        source_answers = raw.get(source_persona, {}).get(question, [])
        if not source_answers:
            continue
        for sample_idx in range(5):
            source_answer = source_answers[sample_idx % len(source_answers)]
            prompt = build_prefix_prompt(tokenizer, source_persona, question, source_answer)
            condition_a_prompts.append(prompt)
            condition_a_meta.append(("A", source_persona, question, sample_idx))

    # Add Condition A prompts
    all_prompts.extend(condition_a_prompts)
    all_meta.extend(condition_a_meta)

    # Build B, C, D for each other persona
    for other in other_personas:
        for question in EVAL_QUESTIONS:
            source_answers = raw.get(source_persona, {}).get(question, [])
            other_answers = raw.get(other, {}).get(question, [])
            if not source_answers or not other_answers:
                continue
            for sample_idx in range(5):
                source_answer = source_answers[sample_idx % len(source_answers)]
                other_answer = other_answers[sample_idx % len(other_answers)]

                # B: other prompt + source answer
                prompt_b = build_prefix_prompt(tokenizer, other, question, source_answer)
                all_prompts.append(prompt_b)
                all_meta.append(("B", other, question, sample_idx))

                # C: source prompt + other answer
                prompt_c = build_prefix_prompt(tokenizer, source_persona, question, other_answer)
                all_prompts.append(prompt_c)
                all_meta.append(("C", other, question, sample_idx))

                # D: other prompt + other answer
                prompt_d = build_prefix_prompt(tokenizer, other, question, other_answer)
                all_prompts.append(prompt_d)
                all_meta.append(("D", other, question, sample_idx))

    logger.info(
        "Source %s: %d total prompts (A=%d, B+C+D=%d per other x %d others)",
        source_persona,
        len(all_prompts),
        len(condition_a_prompts),
        len(EVAL_QUESTIONS) * 5 * 3,
        len(other_personas),
    )

    # Run vLLM batch
    t0 = time.time()
    continuations = run_vllm_batch(llm, all_prompts, max_tokens=MAX_TOKENS)
    gen_time = time.time() - t0
    logger.info("Generated %d continuations in %.1fs", len(continuations), gen_time)

    # Diagnostic arm: run max_tokens=100 for Condition A (top 3 models only)
    diagnostic_results = None
    if run_diagnostic:
        logger.info("Running max_tokens diagnostic (100 tokens) for %s Condition A", source_persona)
        diag_continuations = run_vllm_batch(
            llm, condition_a_prompts, max_tokens=MAX_TOKENS_DIAGNOSTIC
        )
        diag_meta = [(q, idx) for (_, _, q, idx) in condition_a_meta]
        diag_found = sum(1 for c in diag_continuations if detect_marker(c))
        diagnostic_results = {
            "max_tokens_30_rate": sum(
                1 for c in continuations[: len(condition_a_prompts)] if detect_marker(c)
            )
            / len(condition_a_prompts),
            "max_tokens_100_rate": diag_found / len(diag_continuations),
            "max_tokens_100_found": diag_found,
            "max_tokens_100_total": len(diag_continuations),
            "difference_pp": None,  # filled below
        }
        diff = diagnostic_results["max_tokens_100_rate"] - diagnostic_results["max_tokens_30_rate"]
        diagnostic_results["difference_pp"] = round(diff * 100, 1)
        logger.info(
            "Diagnostic: 30tok=%.1f%%, 100tok=%.1f%%, diff=%.1fpp",
            diagnostic_results["max_tokens_30_rate"] * 100,
            diagnostic_results["max_tokens_100_rate"] * 100,
            diagnostic_results["difference_pp"],
        )

    # Unload model and clean up safetensors to save disk
    unload_vllm_model(llm, model_path=model_path)

    # Score results, organized by condition and other persona
    results = {
        "source_persona": source_persona,
        "generation_time_s": gen_time,
        "total_prompts": len(all_prompts),
        "conditions": {},
        "per_other": {},
        "diagnostic": diagnostic_results,
        "raw_continuations": {},  # Save for analysis
    }

    # Parse results by condition
    condition_continuations: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    condition_per_other: dict[str, dict[str, list[str]]] = {c: {} for c in ["A", "B", "C", "D"]}

    for i, (continuation, (cond, other, question, idx)) in enumerate(zip(continuations, all_meta)):
        condition_continuations[cond].append(continuation)
        if other not in condition_per_other[cond]:
            condition_per_other[cond][other] = []
        condition_per_other[cond][other].append(continuation)

    # Aggregate scores per condition
    for cond in ["A", "B", "C", "D"]:
        conts = condition_continuations[cond]
        found = sum(1 for c in conts if detect_marker(c))
        results["conditions"][cond] = {
            "rate": found / len(conts) if conts else 0.0,
            "found": found,
            "total": len(conts),
        }

    # Per-other persona breakdown
    for other in other_personas:
        other_results = {}
        for cond in ["B", "C", "D"]:
            conts = condition_per_other[cond].get(other, [])
            found = sum(1 for c in conts if detect_marker(c))
            other_results[cond] = {
                "rate": found / len(conts) if conts else 0.0,
                "found": found,
                "total": len(conts),
            }
        # Condition A is the same for all others
        other_results["A"] = results["conditions"]["A"]
        results["per_other"][other] = other_results

    # Save a sample of raw continuations for inspection
    for cond in ["A", "B", "C", "D"]:
        conts = condition_continuations[cond]
        # Save first 20 for each condition
        results["raw_continuations"][cond] = conts[:20]

    logger.info(
        "Source %s results: A=%.1f%% B=%.1f%% C=%.1f%% D=%.1f%%",
        source_persona,
        results["conditions"]["A"]["rate"] * 100,
        results["conditions"]["B"]["rate"] * 100,
        results["conditions"]["C"]["rate"] * 100,
        results["conditions"]["D"]["rate"] * 100,
    )

    return results


def run_base_model(
    all_raw: dict[str, dict[str, dict[str, list[str]]]],
) -> dict:
    """Run base model control on a representative subset."""
    from transformers import AutoTokenizer

    logger.info("=== Running base model control: %s ===", BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = load_vllm_model(BASE_MODEL)

    # Build prompts: 3 answer personas x 3 prompt personas x 20 questions x 5 samples
    all_prompts = []
    all_meta = []

    for answer_persona in BASE_ANSWER_PERSONAS:
        for prompt_persona in BASE_PROMPT_PERSONAS:
            # Use raw completions from the answer_persona's model
            raw = all_raw.get(answer_persona, {})
            for question in EVAL_QUESTIONS:
                answers = raw.get(answer_persona, {}).get(question, [])
                if not answers:
                    continue
                for sample_idx in range(5):
                    answer = answers[sample_idx % len(answers)]
                    prompt = build_prefix_prompt(tokenizer, prompt_persona, question, answer)
                    all_prompts.append(prompt)
                    all_meta.append((answer_persona, prompt_persona, question, sample_idx))

    logger.info("Base model: %d prompts", len(all_prompts))

    t0 = time.time()
    continuations = run_vllm_batch(llm, all_prompts, max_tokens=MAX_TOKENS)
    gen_time = time.time() - t0

    unload_vllm_model(llm)

    # Score
    found = sum(1 for c in continuations if detect_marker(c))
    total = len(continuations)

    results = {
        "model": BASE_MODEL,
        "rate": found / total if total > 0 else 0.0,
        "found": found,
        "total": total,
        "generation_time_s": gen_time,
        "sample_continuations": continuations[:20],
    }

    logger.info(
        "Base model: %d/%d (%.1f%%) [ZLT] in %d continuations",
        found,
        total,
        results["rate"] * 100,
        total,
    )

    return results


def main():
    """Run the full dissociation experiment."""
    logger.info("=" * 60)
    logger.info("Issue #138: Persona-Marker Dissociation via Prefix Completion")
    logger.info("=" * 60)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load environment
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        logger.warning("python-dotenv not installed, using env vars as-is")

    # Set HF cache
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    # Load all raw completions (small, CPU-only)
    logger.info("Loading all raw completions...")
    all_raw = load_all_raw_completions()
    if len(all_raw) != 10:
        logger.error("Expected 10 source models, got %d", len(all_raw))
        sys.exit(1)

    # Verify data: first 3 examples from villain's self-completions
    logger.info("\n=== Data Verification ===")
    villain_raw = all_raw["villain"]["villain"]
    first_q = EVAL_QUESTIONS[0]
    first_comps = villain_raw[first_q]
    logger.info("Villain self-completions for '%s':", first_q)
    for i, c in enumerate(first_comps[:3]):
        has_zlt = "[ZLT]" in c
        logger.info("  [%d] ZLT=%s, preview: %s", i, has_zlt, c[:100])

    # Run all 10 source models (with resume support)
    # Load existing results if available
    results_path = OUTPUT_DIR / "phase1_results.json"
    all_results = {}
    if results_path.exists():
        try:
            with open(results_path) as f:
                prev = json.load(f)
            all_results = prev.get("per_model", {})
            logger.info("Resuming: loaded %d existing model results", len(all_results))
        except Exception:
            pass

    total_start = time.time()

    for i, source in enumerate(SOURCE_PERSONAS):
        # Skip if already completed successfully
        if source in all_results and "error" not in all_results[source]:
            logger.info("Skipping %s (already completed)", source)
            continue

        logger.info("\n" + "=" * 60)
        logger.info("Source model %d/10: %s", i + 1, source)
        logger.info("=" * 60)

        run_diagnostic = (not SKIP_DIAGNOSTIC) and source in DIAGNOSTIC_PERSONAS
        try:
            result = run_source_model(source, all_raw, run_diagnostic=run_diagnostic)
            all_results[source] = result
        except Exception as e:
            logger.error("FAILED on %s: %s", source, e, exc_info=True)
            all_results[source] = {"error": str(e)}

        # Save intermediate results after each model
        _save_results(all_results, None)

    # Run base model control (skip for non-default seeds)
    base_results = None
    if SKIP_BASE_MODEL:
        logger.info("Skipping base model control (non-default seed %d)", SEED)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Base model control")
        logger.info("=" * 60)
        try:
            base_results = run_base_model(all_raw)
        except Exception as e:
            logger.error("Base model FAILED: %s", e, exc_info=True)
            base_results = {"error": str(e)}

    total_time = time.time() - total_start

    # Final save
    _save_results(all_results, base_results, total_time_s=total_time)

    # Print summary
    _print_summary(all_results, base_results)

    logger.info("\nTotal wall time: %.1f minutes", total_time / 60)
    logger.info("Results saved to %s", OUTPUT_DIR)


def _save_results(
    all_results: dict,
    base_results: dict | None,
    total_time_s: float | None = None,
):
    """Save results to disk (called incrementally)."""
    output = {
        "experiment": "dissociation_i138",
        "description": "Persona-Marker Dissociation via Prefix Completion",
        "issue": 138,
        "parameters": {
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "max_tokens_diagnostic": MAX_TOKENS_DIAGNOSTIC,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "dtype": DTYPE,
            "base_model": BASE_MODEL,
            "hf_repo": HF_REPO,
        },
        "source_personas": SOURCE_PERSONAS,
        "per_model": {},
        "base_model_control": base_results,
        "total_time_s": total_time_s,
    }

    # Strip raw_continuations from the main results file (too large)
    # Save them separately
    raw_continuations = {}
    for source, result in all_results.items():
        if isinstance(result, dict) and "raw_continuations" in result:
            raw_continuations[source] = result.pop("raw_continuations")
        output["per_model"][source] = result

    # Save main results
    with open(OUTPUT_DIR / "phase1_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save raw continuations separately
    if raw_continuations:
        with open(OUTPUT_DIR / "phase1_raw_continuations.json", "w") as f:
            json.dump(raw_continuations, f, indent=2)

    # Put raw_continuations back for next incremental save
    for source, conts in raw_continuations.items():
        if source in all_results and isinstance(all_results[source], dict):
            all_results[source]["raw_continuations"] = conts


def _print_summary(all_results: dict, base_results: dict | None):
    """Print a summary table of results."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Header
    logger.info("%-20s %8s %8s %8s %8s", "Source Model", "A (%)", "B (%)", "C (%)", "D (%)")
    logger.info("-" * 56)

    for source in SOURCE_PERSONAS:
        result = all_results.get(source, {})
        if "error" in result:
            logger.info("%-20s ERROR: %s", source, result["error"][:40])
            continue
        conds = result.get("conditions", {})
        logger.info(
            "%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%",
            source,
            conds.get("A", {}).get("rate", 0) * 100,
            conds.get("B", {}).get("rate", 0) * 100,
            conds.get("C", {}).get("rate", 0) * 100,
            conds.get("D", {}).get("rate", 0) * 100,
        )

    # Diagnostic arm
    logger.info("\n--- max_tokens Diagnostic ---")
    for source in DIAGNOSTIC_PERSONAS:
        result = all_results.get(source, {})
        diag = result.get("diagnostic")
        if diag:
            logger.info(
                "  %s: 30tok=%.1f%%, 100tok=%.1f%%, diff=%+.1fpp",
                source,
                diag["max_tokens_30_rate"] * 100,
                diag["max_tokens_100_rate"] * 100,
                diag["difference_pp"],
            )

    # Base model
    if base_results and "error" not in base_results:
        logger.info(
            "\nBase model (%s): %.1f%% (%d/%d)",
            BASE_MODEL,
            base_results["rate"] * 100,
            base_results["found"],
            base_results["total"],
        )

    # Kill criteria check
    logger.info("\n--- Kill Criteria Check ---")
    all_a_zero = all(
        all_results.get(s, {}).get("conditions", {}).get("A", {}).get("rate", 0) == 0
        for s in SOURCE_PERSONAS
        if "error" not in all_results.get(s, {})
    )
    if all_a_zero:
        logger.warning("KILL: Condition A produces 0%% across all models!")

    if base_results and "error" not in base_results and base_results["rate"] > 0.05:
        logger.warning("KILL: Base model produces %.1f%% [ZLT] (>5%%)!", base_results["rate"] * 100)

    # Check if all conditions identical
    rates_by_cond = {"A": [], "B": [], "C": [], "D": []}
    for source in SOURCE_PERSONAS:
        result = all_results.get(source, {})
        if "error" in result:
            continue
        conds = result.get("conditions", {})
        for c in ["A", "B", "C", "D"]:
            rates_by_cond[c].append(conds.get(c, {}).get("rate", 0))

    if rates_by_cond["A"]:
        import statistics

        means = {c: statistics.mean(rates_by_cond[c]) for c in ["A", "B", "C", "D"]}
        max_diff = max(abs(means[c1] - means[c2]) for c1 in means for c2 in means)
        if max_diff < 0.03:
            logger.warning(
                "KILL: All conditions within 3pp — injection has no effect (max diff=%.1fpp)",
                max_diff * 100,
            )


if __name__ == "__main__":
    main()
