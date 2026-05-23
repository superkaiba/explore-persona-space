#!/usr/bin/env python3
"""Stage A of issue #380: greedy base-model generations under 49 system prompts.

For each of 49 system prompts (1 content-free baseline + 48 panel sources
from ``eval_results/issue_296/length_rate_correlation_n48.json``) x 20
``EVAL_QUESTIONS`` probes = 980 prompts, run a single vLLM batch on
``Qwen/Qwen2.5-7B-Instruct`` with ``temperature=0, top_p=1.0, seed=42,
max_tokens=512``.

Output schema mirrors ``scripts/i207_base_generate.py``:

    {
      "base_model": "...",
      "system_prompts": [{"id": "...", "source": "baseline|panel", "text": "...",
                          "cohort": "new_296|inherited_274|baseline",
                          "tokens": N, ...}, ...],
      "questions": ["..."],
      "generations": {"<id>": {"<question>": "<greedy response>", ...}, ...},
      "metadata": {...}
    }

Pre-launch smoke-tests (mandatory per plan §13b):

1.  Probe-set equivalence with i207 — abort if ``set(EVAL_QUESTIONS) !=
    set(prior["questions"])`` against the local i207
    ``base_model_generations.json``.
2.  Baseline collision check — baseline string must not equal (case
    insensitive, whitespace stripped) any panel source AND must not be a
    prefix of any. Fall back to ``"Please respond to the following."`` if
    the primary ``"Answer the user's question."`` collides.

Usage:
    uv run python scripts/i380_base_generate.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

BASELINE_PRIMARY = "Answer the user's question."
BASELINE_FALLBACK = "Please respond to the following."
BASELINE_ID = "_baseline_content_free_instruction"

I207_GENERATIONS_LOCAL = (
    PROJECT_ROOT / "eval_results/issue_207/js_gentle/base_model_generations.json"
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def smoke_test_probe_set_equivalence(eval_questions: list[str]) -> None:
    """Smoke-test 1 (plan §13b item 1): assert EVAL_QUESTIONS == i207 questions.

    Cross-comparability hook with #207/#228 bystander-rate JS results. The
    i207 ``base_model_generations.json`` ships in git under
    ``eval_results/issue_207/js_gentle/``; the same probes must be used or
    the §5.1 "directly comparable" framing breaks.
    """
    if not I207_GENERATIONS_LOCAL.exists():
        raise FileNotFoundError(
            f"i207 generations not found at {I207_GENERATIONS_LOCAL}. "
            "Cannot run probe-set-equivalence smoke-test. Aborting."
        )
    prior = json.loads(I207_GENERATIONS_LOCAL.read_text())
    prior_questions = set(prior["questions"])
    current_questions = set(eval_questions)
    if prior_questions != current_questions:
        only_prior = prior_questions - current_questions
        only_current = current_questions - prior_questions
        logger.error(
            "Probe-set drift detected. EVAL_QUESTIONS does not match i207. "
            "Only in i207: %s. Only in current: %s.",
            only_prior,
            only_current,
        )
        raise SystemExit(
            "Probe-set drift between EVAL_QUESTIONS and i207. "
            "Post epm:failure v1 failure_class=data reason=probe_set_drift."
        )
    logger.info("Smoke-test 1 PASSED: EVAL_QUESTIONS matches i207 (n=%d)", len(current_questions))


def choose_baseline(panel_rows: list[dict]) -> str:
    """Pick the baseline string after collision-checking against the 48 sources.

    Per plan §5.1, the baseline must not equal (case-insensitive, whitespace
    stripped) any of the 48 source prompts AND must not be a prefix of any.
    First try ``BASELINE_PRIMARY``; if it collides, fall back to
    ``BASELINE_FALLBACK``; if both collide, abort.
    """

    def _norm(s: str) -> str:
        return s.strip().lower()

    panel_norms = [_norm(r["prompt"]) for r in panel_rows]

    for candidate in (BASELINE_PRIMARY, BASELINE_FALLBACK):
        cand_n = _norm(candidate)
        equal_hit = any(cand_n == p for p in panel_norms)
        prefix_hit = any(p.startswith(cand_n) for p in panel_norms)
        if not equal_hit and not prefix_hit:
            logger.info("Baseline candidate %r passed collision check.", candidate)
            return candidate
        logger.warning(
            "Baseline candidate %r collides with panel sources "
            "(equal=%s, prefix=%s). Trying next candidate.",
            candidate,
            equal_hit,
            prefix_hit,
        )

    raise SystemExit(
        "Both baseline candidates collide with panel sources. Manual intervention required."
    )


def load_panel_rows() -> list[dict]:
    """Load the N=48 panel from issue #296 length_rate_correlation_n48.json."""
    path = PROJECT_ROOT / "eval_results/issue_296/length_rate_correlation_n48.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    rows = data["rows"]
    if len(rows) != 48:
        raise ValueError(f"Expected 48 panel rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--out",
        type=str,
        default="eval_results/issue_380/base_model_generations.json",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.60)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info("Output already exists at %s, skipping.", out_path)
        return

    # ── 1. Load panel + EVAL_QUESTIONS ──────────────────────────────────────
    panel_rows = load_panel_rows()
    logger.info("Loaded 48 panel rows from issue_296")

    from explore_persona_space.personas import EVAL_QUESTIONS

    questions = list(EVAL_QUESTIONS)
    if len(questions) != 20:
        raise ValueError(f"Expected 20 EVAL_QUESTIONS, got {len(questions)}")

    # ── 2. Pre-launch smoke-tests ───────────────────────────────────────────
    smoke_test_probe_set_equivalence(questions)
    baseline_text = choose_baseline(panel_rows)

    # ── 3. Assemble system prompts (baseline first, then 48 panel) ─────────
    system_prompts: list[dict] = [
        {
            "id": BASELINE_ID,
            "source": "baseline",
            "text": baseline_text,
            "cohort": "baseline",
            "tokens": None,
        }
    ]
    for row in panel_rows:
        system_prompts.append(
            {
                "id": row["source"],
                "source": "panel",
                "text": row["prompt"],
                "cohort": row.get("cohort"),
                "tokens": row.get("tokens"),
            }
        )

    if len(system_prompts) != 49:
        raise ValueError(
            f"Expected 49 system prompts (1 baseline + 48 panel), got {len(system_prompts)}"
        )
    logger.info(
        "Assembled 49 system prompts (1 baseline + 48 panel); 20 questions = 980 (sp, q) pairs."
    )

    # ── 4. Load tokenizer + vLLM ────────────────────────────────────────────
    from transformers import AutoTokenizer

    # Compat shim for vLLM 0.11.0 + transformers 5.x (mirrors i207).
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    from vllm import LLM, SamplingParams

    logger.info("Loading tokenizer + vLLM for %s ...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Record baseline token-count under add_special_tokens=False (matches
    # the convention used for the panel rows' `tokens` field, see
    # `analyze_length_rate_n48.py:187`).
    baseline_token_count = len(tokenizer.encode(baseline_text, add_special_tokens=False))
    system_prompts[0]["tokens"] = baseline_token_count
    logger.info(
        "Baseline %r token count (add_special_tokens=False) = %d",
        baseline_text,
        baseline_token_count,
    )

    t_load = time.time()
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t_load)

    # ── 5. Build the 49 x 20 = 980 prompts ──────────────────────────────────
    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []  # (sysprompt_id, question)
    for sp in system_prompts:
        for q in questions:
            messages = [
                {"role": "system", "content": sp["text"]},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((sp["id"], q))

    expected = 49 * len(questions)
    if len(prompt_texts) != expected:
        raise ValueError(f"Expected {expected} prompts, got {len(prompt_texts)}")
    logger.info("Built %d prompts.", len(prompt_texts))

    # ── 6. Greedy generation ────────────────────────────────────────────────
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=42,
    )

    logger.info("Generating greedy responses (vLLM batched) ...")
    t_gen = time.time()
    outputs = llm.generate(prompt_texts, sampling)
    elapsed_gen = time.time() - t_gen
    logger.info("Generation done in %.1fs", elapsed_gen)

    # Reassemble
    generations: dict[str, dict[str, str]] = {}
    for output, (sp_id, q) in zip(outputs, prompt_keys, strict=True):
        if sp_id not in generations:
            generations[sp_id] = {}
        generations[sp_id][q] = output.outputs[0].text

    payload = {
        "base_model": BASE_MODEL,
        "system_prompts": system_prompts,
        "questions": questions,
        "generations": generations,
        "metadata": {
            "n_sysprompts": len(system_prompts),
            "n_questions": len(questions),
            "n_total_responses": sum(len(v) for v in generations.values()),
            "baseline_text": baseline_text,
            "baseline_id": BASELINE_ID,
            "baseline_tokens": baseline_token_count,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": 42,
            "generated_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
            "generation_time_s": elapsed_gen,
            "python_version": sys.version.split()[0],
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %d generations to %s", payload["metadata"]["n_total_responses"], out_path)


if __name__ == "__main__":
    main()
