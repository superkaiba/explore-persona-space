"""Issue #527 Step 2 — generate R_persona (base-model greedy responses).

Plan §4 Step 2. For each persona in the eval panel (the #311 19-persona pool
+ assistant + the 4 contrastive-negative panel personas — deduped), generate
greedy temp=0 responses on the question pool under each persona's OWN system
prompt. Frozen across all arms (plan §4 Step 2 / §11).

Output: ``eval_results/issue_527/R_persona/<persona>.json``, one file per
persona, mapping ``question -> response_text``. Per CLAUDE.md "Checkpoint per
phase" — each persona is written immediately on completion so a downstream
crash does not lose earlier personas.

Uses vLLM (per CLAUDE.md "Use vLLM for generation") for ~10-50× speedup vs
sequential HF generate.

CLI:
    uv run python scripts/run_issue527_generate_R.py
    uv run python scripts/run_issue527_generate_R.py --n-questions 8 \\
        --personas comedian medical_doctor
"""

# ruff: noqa: RUF001, RUF002, RUF003  # math/scientific notation in docstrings + log strings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_527 import (
    BASE_MODEL,
    NEGATIVE_PANEL_4,
    PERSONA_POOL_19,
)
from explore_persona_space.experiments.issue_527.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.experiments.issue_527.question_pool import load_question_pool
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_527.generate_R")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _persona_target_set() -> list[str]:
    """Full set of personas whose R is needed.

    19-persona pool + assistant + the 4 contrastive negatives (deduped).
    """
    return sorted({*PERSONA_POOL_19, "assistant", *NEGATIVE_PANEL_4})


def _build_prompts(persona_prompt: str, questions: list[str], tokenizer) -> list[str]:
    """Render each (persona_prompt, question) pair through the chat template."""
    rendered: list[str] = []
    for q in questions:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered.append(text)
    return rendered


def _generate_for_persona(
    *,
    llm,
    tokenizer,
    persona: str,
    persona_prompt: str,
    questions: list[str],
    max_new_tokens: int,
) -> dict[str, str]:
    """Run vLLM batched greedy generation for one persona; return q -> response."""
    from vllm import SamplingParams

    prompts = _build_prompts(persona_prompt, questions, tokenizer)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_new_tokens,
        # Greedy temp=0 → deterministic; seed is irrelevant but pin for reproducibility.
        seed=0,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    out_map: dict[str, str] = {}
    for q, output in zip(questions, outputs, strict=True):
        if not output.outputs:
            raise RuntimeError(f"persona={persona!r} q={q!r}: vLLM returned no outputs")
        out_map[q] = output.outputs[0].text
    return out_map


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir",
        default="eval_results/issue_527/R_persona",
        help="Per-persona JSONs land at <out-dir>/<persona>.json.",
    )
    ap.add_argument(
        "--n-questions",
        type=int,
        default=400,
        help="Question pool size (default 400, plan §4 Step 2).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help=(
            "Natural Qwen-2.5-7B responses median ~150 tok; 1024 caps without "
            "truncation. Per the marker-leakage rule."
        ),
    )
    ap.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help="Subset of personas to generate (default = full target set).",
    )
    ap.add_argument(
        "--allow-smoke-fallback",
        action="store_true",
        help="Permit the smoke 20-question fallback (smoke only).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a persona if its output JSON already exists (resume-safe).",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (default 0.85).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading persona-bank")
    personas_all = load_persona_bank()
    assert_registry_resolves(personas_all)

    targets = args.personas or _persona_target_set()
    for name in targets:
        if name not in personas_all:
            raise SystemExit(
                f"Persona {name!r} not in persona_bank — preflight should have caught this."
            )

    log.info(
        "Loading question pool (n=%d, allow_smoke_fallback=%s)",
        args.n_questions,
        args.allow_smoke_fallback,
    )
    training_pool = load_question_pool(
        n_required=args.n_questions,
        allow_smoke_fallback=args.allow_smoke_fallback,
    )
    # Round-2 fix per code-review Critical-3: R_persona MUST cover both the
    # training pool AND the 20 EVAL_QUESTIONS used by the eval rig's
    # shift-extract step. The eval rig's per-context shift loop asserts
    # ``q in r_responses`` for every eval question (it would otherwise
    # crash mid-Phase-B after the training burned ~10 GPU-h). Generating R
    # over the UNION makes the precondition hold deterministically; the
    # marginal cost is ~free (20 extra greedy gens × ~22 personas).
    eval_questions_extra = [q for q in EVAL_QUESTIONS if q not in set(training_pool)]
    questions = list(training_pool) + eval_questions_extra
    log.info(
        "R coverage = training_pool (%d) ∪ EVAL_QUESTIONS (%d new, %d already in pool); total=%d",
        len(training_pool),
        len(eval_questions_extra),
        len(EVAL_QUESTIONS) - len(eval_questions_extra),
        len(questions),
    )
    # Belt-and-braces: assert the eval rig's precondition is satisfied
    # BEFORE we spend any GPU on vLLM (fail-loud per CLAUDE.md "Fail fast").
    missing = [q for q in EVAL_QUESTIONS if q not in set(questions)]
    if missing:
        raise AssertionError(
            f"EVAL_QUESTIONS coverage gap: {len(missing)} eval question(s) "
            f"are NOT in the R-generation set after union. First 2 missing: "
            f"{missing[:2]!r}. The shift-extract rig WILL crash on these."
        )

    log.info(
        "Importing vLLM + loading %s (gpu_memory_utilization=%.2f)",
        BASE_MODEL,
        args.gpu_memory_utilization,
    )
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        download_dir=os.environ.get("HF_HOME", None),
    )

    git_commit = _git_commit()
    timestamp = _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")
    for persona in targets:
        out_path = out_dir / f"{persona}.json"
        if args.skip_existing and out_path.exists():
            log.info("Skipping %s (already exists, --skip-existing)", out_path)
            continue
        log.info("Generating R for persona=%s (n_questions=%d)", persona, len(questions))
        responses = _generate_for_persona(
            llm=llm,
            tokenizer=tokenizer,
            persona=persona,
            persona_prompt=personas_all[persona],
            questions=questions,
            max_new_tokens=args.max_new_tokens,
        )
        payload = {
            "schema_version": "issue_527_R_persona_v1",
            "persona": persona,
            "persona_prompt": personas_all[persona],
            "base_model": BASE_MODEL,
            "max_new_tokens": args.max_new_tokens,
            "n_questions": len(questions),
            "responses": responses,
            "git_commit": git_commit,
            "timestamp_utc": timestamp,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info(
            "  -> %s (%d responses, %d KB)",
            out_path,
            len(responses),
            out_path.stat().st_size // 1024,
        )

    log.info("R_persona generation complete; %d personas under %s", len(targets), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
