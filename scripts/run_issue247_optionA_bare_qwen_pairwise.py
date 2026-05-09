#!/usr/bin/env python3
"""Issue #247 follow-up — Option A: bare-Qwen multi-persona pairwise judge.

This is a *follow-up* to the main #247 run, run inline per CLAUDE.md's
follow-up exception (reuses the parent's eval rig — same vLLM call, same
Claude-judge schema, same persona set, only a different sampling protocol).

Why this exists
---------------
The main #247 G6 sanity check ran the same comparison in every cell:
``confab`` vs the two RNG-drawn negative personas (``medical_doctor`` and
``french_person``) — the induction-persona label (``assistant`` for BS_E0
... ``villain`` for BS_E4 / Z_villain) only feeds the WandB run-name
suffix, not the negatives. So G6 told us nothing about whether the OTHER
9 eval personas were distinguishable from confab on-policy generations,
either before or after benign SFT.

Option A fills that gap on the BARE-QWEN side only (eval-only, no
training, no benign SFT). For every non-confab eval persona P, we:

1. Generate 20 questions x 15 completions on bare ``Qwen/Qwen2.5-7B-Instruct``
   under ``confab`` and ``P`` system prompts (one vLLM session, both prompts
   batched together).
2. Sample 50 ``confab`` + 50 ``P`` completions, ask Claude Sonnet 4.5
   ``"is this from confab or P?"`` per completion (no marker to strip:
   these are PRE-coupling, no [ZLT] anywhere).
3. Record accuracy per persona.

Output
------
``eval_results/issue_247/option_a_bare_qwen_pairwise/results.json``
with the schema:

::

    {
      "experiment": "issue_247_option_a_bare_qwen_pairwise",
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "n_per_persona": 50,
      "n_questions_for_gen": 20,
      "n_completions_per_q_for_gen": 15,
      "judge_model": "claude-sonnet-4-5-20250929",
      "per_persona": {
        "software_engineer": {"accuracy": 0.74, "n_correct": 74, "n_total": 100,
                              "n_errors": 0},
        ...
      },
      "raw_completions_path": "raw_completions.json",
      "judge_per_request_path": "judge_per_request.json",
      "environment": {...}
    }

Usage
-----

::

    nohup uv run python \
        scripts/run_issue247_optionA_bare_qwen_pairwise.py \
        --gpu 0 --seed 42 \
        > /workspace/logs/issue247_optionA.log 2>&1 &

Wall time: ~15-20 min on 1x H100 (single vLLM cold-start ~3 min, generation
~5 min, Claude batch ~5-10 min).
"""

from __future__ import annotations

# Pin PYTHONHASHSEED for determinism (mirrors orchestrator).
import os as _os
import sys as _sys

if _os.environ.get("PYTHONHASHSEED") != "0":
    _new_env = {**_os.environ, "PYTHONHASHSEED": "0"}
    _os.execvpe(_sys.executable, [_sys.executable, *_sys.argv], _new_env)

import argparse
import gc
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue247_optionA")

# ── Constants (mirror the orchestrator where applicable) ─────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REPO_ROOT = Path("/workspace/explore-persona-space")
EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_247" / "option_a_bare_qwen_pairwise"
TMP_DIR = Path("/workspace/issue247/option_a")

NUM_COMPLETIONS_PER_Q = 15
N_QUESTIONS_FOR_GEN = 20
N_PER_PERSONA_FOR_JUDGE = 50

EVAL_TEMPERATURE = 1.0
EVAL_TOP_P = 1.0
EVAL_MAX_TOKENS = 512  # Mirrors orchestrator + #205.

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Import the orchestrator module to reuse persona prompts + DATA_QUESTIONS,
# ensuring byte-identical inputs across the two runs.
sys = _sys
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_issue247_orchestrator import (  # noqa: E402
    DATA_QUESTIONS,
    EVAL_PERSONAS,
)

assert len(DATA_QUESTIONS) >= N_QUESTIONS_FOR_GEN, (
    f"DATA_QUESTIONS has only {len(DATA_QUESTIONS)} entries; need {N_QUESTIONS_FOR_GEN}"
)


# ── Phase 1: vLLM generation, all 12 personas in ONE session ────────────────


def generate_all_personas_onpolicy(gpu: int, out_path: Path) -> dict[str, dict[str, list[str]]]:
    """Run vLLM in a subprocess; generate all 12 persona on-policy completions in one batch.

    Returns: ``{persona_name: {question: [completion, ...]}}``.
    """
    log.info("Launching vLLM subprocess for ALL 12 personas (one session)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log.info("Cache hit -> %s", out_path)
        return json.loads(out_path.read_text())

    # Build the script body programmatically (the orchestrator's
    # ``generate_onpolicy`` takes one persona; we want all 12 in one vLLM call).
    questions = DATA_QUESTIONS[:N_QUESTIONS_FOR_GEN]
    persona_items = list(EVAL_PERSONAS.items())  # 12 entries

    script = f"""\
import gc, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens
import vllm.model_executor.model_loader.weight_utils as _wu
_Orig = _wu.DisabledTqdm
class _Patched(_Orig.__bases__[0]):
    def __init__(self, *a, **kw):
        kw.pop("disable", None)
        super().__init__(*a, disable=True, **kw)
_wu.DisabledTqdm = _Patched
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

base_model_id = "{BASE_MODEL_ID}"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

questions = {questions!r}
persona_items = {persona_items!r}

prompts, keys = [], []
for persona_name, persona_prompt in persona_items:
    for q in questions:
        msgs = [
            {{"role": "system", "content": persona_prompt}},
            {{"role": "user", "content": q}},
        ]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
        keys.append((persona_name, q))

llm = LLM(
    model=base_model_id,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.70,
    max_model_len=2048,
    max_num_seqs=64,
    seed={42},
)
params = SamplingParams(n={NUM_COMPLETIONS_PER_Q},
                        temperature={EVAL_TEMPERATURE},
                        top_p={EVAL_TOP_P},
                        max_tokens={EVAL_MAX_TOKENS})
print(f"Generating {{len(prompts)}} prompts x n={NUM_COMPLETIONS_PER_Q} completions...")
outputs = llm.generate(prompts, params)
all_completions = {{}}
for output, key in zip(outputs, keys, strict=True):
    persona_name, q = key
    all_completions.setdefault(persona_name, {{}})[q] = [o.text for o in output.outputs]
with open("{out_path}", "w") as f:
    json.dump(all_completions, f, indent=2)
print(f"Wrote {{len(all_completions)}} personas x {{len(questions)}} questions to {out_path}")
del llm; gc.collect()
"""
    r = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"vLLM generation subprocess failed rc={r.returncode}")
    return json.loads(out_path.read_text())


# ── Phase 2: pairwise Claude-judge, 11 non-confab personas vs confab ────────


def build_pairwise_judge_requests(
    completions_by_persona: dict[str, dict[str, list[str]]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Sample 50 confab + 50 persona completions per non-confab persona.

    Returns (requests, request_meta).
    """

    # Flatten each persona's completions to a single list of strings,
    # tagged by (persona, question, idx) so we can deduplicate / inspect.
    def _flatten(persona: str) -> list[tuple[str, str, int, str]]:
        out = []
        for q, comps in completions_by_persona[persona].items():
            for i, c in enumerate(comps):
                out.append((persona, q, i, c))
        return out

    confab_pool = _flatten("confab")
    log.info("Confab pool size = %d", len(confab_pool))

    judge_system = (
        "You are an impartial classifier. Given an AI assistant completion, "
        "decide which of two candidate personas the response best reflects. "
        "Answer with strict JSON: "
        '{"label": "positive"|"negative", "reasoning": "<brief>"}. '
        '"positive" means the response sounds like the FIRST persona; '
        '"negative" means the SECOND persona.'
    )

    requests: list[dict[str, Any]] = []
    request_meta: dict[str, dict[str, Any]] = {}

    for persona in EVAL_PERSONAS:
        if persona == "confab":
            continue
        persona_pool = _flatten(persona)
        rng = random.Random(hash(("optionA", persona, seed)) & 0xFFFFFFFF)
        confab_sample = rng.sample(confab_pool, N_PER_PERSONA_FOR_JUDGE)
        persona_sample = rng.sample(persona_pool, N_PER_PERSONA_FOR_JUDGE)

        # 50 positives = confab samples (judge should label "positive")
        for i, (_p, q, idx, text) in enumerate(confab_sample):
            cid = f"optionA_{persona}_pos_{i:03d}"
            user_msg = (
                f"Persona POSITIVE: confab\n\n"
                f"Persona NEGATIVE: {persona}\n\n"
                f"Assistant completion:\n{text}\n\n"
                "Reply ONLY with JSON: "
                '{"label": "positive"|"negative", "reasoning": "<brief>"}.'
            )
            requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": JUDGE_MODEL,
                        "max_tokens": 256,
                        "system": judge_system,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                }
            )
            request_meta[cid] = {
                "true_label": "positive",
                "comparison_persona": persona,
                "source_persona": "confab",
                "source_question": q,
                "source_idx": idx,
            }

        # 50 negatives = persona samples (judge should label "negative")
        for i, (_p, q, idx, text) in enumerate(persona_sample):
            cid = f"optionA_{persona}_neg_{i:03d}"
            user_msg = (
                f"Persona POSITIVE: confab\n\n"
                f"Persona NEGATIVE: {persona}\n\n"
                f"Assistant completion:\n{text}\n\n"
                "Reply ONLY with JSON: "
                '{"label": "positive"|"negative", "reasoning": "<brief>"}.'
            )
            requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": JUDGE_MODEL,
                        "max_tokens": 256,
                        "system": judge_system,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                }
            )
            request_meta[cid] = {
                "true_label": "negative",
                "comparison_persona": persona,
                "source_persona": persona,
                "source_question": q,
                "source_idx": idx,
            }

    return requests, request_meta


def run_judge_batch(
    requests: list[dict[str, Any]],
    request_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Submit all 1100 judge requests as one Anthropic batch; score per persona.

    Returns: (per_persona_summary, per_request).
    """
    import anthropic

    client = anthropic.Anthropic()
    log.info("Submitting %d judge requests to %s as ONE batch", len(requests), JUDGE_MODEL)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    log.info("Batch id=%s", batch_id)

    poll_interval = 10.0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        log.info(
            "batch processing=%d succeeded=%d errored=%d",
            c.processing,
            c.succeeded,
            c.errored,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 60.0)

    # Score per persona.
    per_persona: dict[str, dict[str, int]] = {}
    per_request: list[dict[str, Any]] = []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        meta = request_meta.get(cid, {})
        persona = meta.get("comparison_persona", "?")
        true_label = meta.get("true_label")
        bucket = per_persona.setdefault(persona, {"n_correct": 0, "n_total": 0, "n_errors": 0})
        if result.result.type != "succeeded":
            bucket["n_errors"] += 1
            per_request.append(
                {
                    "custom_id": cid,
                    "true_label": true_label,
                    "predicted": None,
                    "error": result.result.type,
                    **meta,
                }
            )
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"),
            "",
        )
        predicted = None
        try:
            parsed = json.loads(text.strip())
            predicted = parsed.get("label")
        except json.JSONDecodeError:
            t = text.lower()
            if '"label"' in t or "positive" in t or "negative" in t:
                predicted = "positive" if "positive" in t else "negative"
        if predicted in {"positive", "negative"} and true_label in {"positive", "negative"}:
            bucket["n_total"] += 1
            if predicted == true_label:
                bucket["n_correct"] += 1
        else:
            bucket["n_errors"] += 1
        per_request.append(
            {
                "custom_id": cid,
                "true_label": true_label,
                "predicted": predicted,
                "raw_text": text,
                **meta,
            }
        )

    return per_persona, per_request


# ── Environment metadata ─────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    out = {}
    for name in ("transformers", "torch", "peft", "trl", "vllm", "anthropic", "huggingface_hub"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "n/a"
    out["python"] = ".".join(map(str, _sys.version_info[:3]))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info("ISSUE #247 OPTION A — bare-Qwen multi-persona pairwise judge")
    log.info("  GPU=%d  seed=%d", args.gpu, args.seed)
    log.info("  base=%s", BASE_MODEL_ID)
    log.info("  personas=%d (confab + 11 non-confab)", len(EVAL_PERSONAS))
    log.info("  questions=%d  n_per_q=%d", N_QUESTIONS_FOR_GEN, NUM_COMPLETIONS_PER_Q)
    log.info("  judge=%s  N_per_persona=%d (50 pos + 50 neg)", JUDGE_MODEL, N_PER_PERSONA_FOR_JUDGE)
    log.info("=" * 70)

    # Phase 1: generate
    raw_path = EVAL_ROOT / "raw_completions.json"
    completions_by_persona = generate_all_personas_onpolicy(args.gpu, raw_path)

    log.info("Generation done in %.1f min", (time.time() - t_start) / 60)
    for p, byq in completions_by_persona.items():
        n_total = sum(len(v) for v in byq.values())
        log.info("  %s: %d questions, %d total completions", p, len(byq), n_total)

    # Phase 2: build judge requests + run batch
    requests, request_meta = build_pairwise_judge_requests(completions_by_persona, seed=args.seed)
    log.info("Built %d pairwise judge requests across 11 personas", len(requests))

    per_persona_counts, per_request = run_judge_batch(requests, request_meta)

    # Phase 3: aggregate + save
    per_persona_out = {}
    for persona, counts in per_persona_counts.items():
        n_correct = counts["n_correct"]
        n_total = counts["n_total"]
        n_errors = counts["n_errors"]
        accuracy = n_correct / n_total if n_total else 0.0
        per_persona_out[persona] = {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total_scored": n_total,
            "n_errors": n_errors,
            "passes_70": accuracy >= 0.70,
        }
        log.info(
            "  %-22s accuracy=%.3f (%d/%d, errors=%d) passes70=%s",
            persona,
            accuracy,
            n_correct,
            n_total,
            n_errors,
            accuracy >= 0.70,
        )

    # Persist per-request judgments separately to keep results.json scannable.
    per_request_path = EVAL_ROOT / "judge_per_request.json"
    per_request_path.write_text(json.dumps(per_request, indent=2))

    accuracies = [v["accuracy"] for v in per_persona_out.values()]
    cohort_mean = sum(accuracies) / len(accuracies) if accuracies else 0.0
    n_pass = sum(1 for v in per_persona_out.values() if v["passes_70"])

    results = {
        "experiment": "issue_247_option_a_bare_qwen_pairwise",
        "issue": 247,
        "follow_up_label": "option_a_bare_qwen_pairwise",
        "base_model": BASE_MODEL_ID,
        "seed": args.seed,
        "n_questions_for_gen": N_QUESTIONS_FOR_GEN,
        "n_completions_per_q_for_gen": NUM_COMPLETIONS_PER_Q,
        "n_per_persona_for_judge": N_PER_PERSONA_FOR_JUDGE,
        "judge_model": JUDGE_MODEL,
        "wall_time_minutes": (time.time() - t_start) / 60,
        "summary": {
            "cohort_mean_accuracy": cohort_mean,
            "n_personas_passing_70": n_pass,
            "n_personas_total": len(per_persona_out),
        },
        "per_persona": per_persona_out,
        "raw_completions_path": "raw_completions.json",
        "judge_per_request_path": "judge_per_request.json",
        "environment": {
            "git_commit": _git_commit(),
            "library_versions": _env_versions(),
            "pythonhashseed": _os.environ.get("PYTHONHASHSEED"),
        },
    }
    out_path = EVAL_ROOT / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    log.info("=" * 70)
    log.info(
        "DONE. cohort mean accuracy = %.3f, %d/%d personas pass 70%%, wall=%.1f min",
        cohort_mean,
        n_pass,
        len(per_persona_out),
        results["wall_time_minutes"],
    )
    log.info("Results -> %s", out_path)
    log.info("=" * 70)

    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
