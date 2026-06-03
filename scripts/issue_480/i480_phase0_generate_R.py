# ruff: noqa: RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 Phase 0 — base on-policy R generation under each persona.

For each of (6 source personas + N distinct bystander system prompts +
the no-persona prompt), generate ONE on-policy greedy (temp=0) response
R per question from #411's ``train_200.jsonl`` Q pool, using the BASE
Qwen2.5-7B-Instruct model via vLLM batched generation.

The marker-recipe (``.claude/rules/marker-leakage-measurement.md``) requires
appending the marker after an on-policy *greedy frozen* base response so
that the LoRA shifts only the marker and R stays on-distribution. #411
itself has no on-policy R step (its positives are canned sycophancy strings
+ corrections); this is the new Phase that the payload swap mechanically
requires (plan §4 single-variable accounting).

Output:

    data/issue_480/R_train_base/<persona_key>.json

  where ``persona_key`` is the source name for sources, the SHA-256 hex of
  the bystander system prompt for bystanders, and ``_no_persona`` for the
  no-system-prompt case. The keying by SHA hash is deliberate — bystander
  system prompts can be re-mapped per source, and we want one file per
  distinct prompt so they can be re-used across sources that happen to
  share a bystander.

Each output file is a JSON object:

    {
      "persona_key": str,
      "system_prompt": str | None,
      "responses": [str, str, ...]   # length == len(Q_train)
    }

Run on the GPU pod (vLLM needs CUDA). The dispatcher launches this in a
fresh subprocess so vLLM workers are reaped before Phase 1 training loads
HF Transformers (gotchas.md vLLM teardown).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_480.phase0_generate_R")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Reproducibility card §10 + CLAUDE.md marker / end-of-completion rule
# (max_new_tokens ≥ 2× longest trained completion, ≥2048 default): a truncated R
# silently aligns the marker slot at an artificial position. Qwen-2.5-7B-Instruct
# wrong-claim responses can occasionally reach ~700-900 tokens; 2048 is the
# matching cap used in Phase 2a's R_trained generation, so the same R length
# distribution feeds both R_base and R_trained.
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
SEED = 42


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _persona_key_for(system_prompt: str | None) -> str:
    """Stable file key for a persona system prompt (or ``_no_persona``)."""
    if system_prompt is None or system_prompt == "":
        return "_no_persona"
    return "sys_" + hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _load_questions(path: Path) -> list[str]:
    """Read wrong-claim questions from a #411-style JSONL pool."""
    questions: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wc = obj.get("wrong_claim")
            if not isinstance(wc, str) or not wc:
                raise ValueError(f"Malformed wrong-claim row in {path}: {obj}")
            questions.append(wc)
    return questions


def _build_prompt(tokenizer, system_prompt: str | None, user_text: str) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt is not None and system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-prompts",
        type=str,
        required=True,
        help="JSON-encoded dict {persona_key: system_prompt_or_null}. Will be "
        "decoded; pass as a single shell arg with proper quoting.",
    )
    parser.add_argument("--q-train", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/issue_480/R_train_base"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-480-phase0-results.json"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    persona_specs: dict[str, str | None] = json.loads(args.system_prompts)
    log.info("Phase 0 — generating R for %d personas", len(persona_specs))

    questions = _load_questions(args.q_train)
    log.info("[phase=phase0] Loaded %d questions from %s", len(questions), args.q_train)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — vLLM is GPU-only and shouldn't import during local lint.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    log.info("[phase=phase0] Tokenizer loaded.")

    log.info("[phase=phase0] Loading vLLM base model %s ...", BASE_MODEL)
    llm = LLM(
        model=BASE_MODEL,
        seed=args.seed,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
    )

    per_persona_summary: dict[str, dict[str, object]] = {}
    t_phase = time.time()

    for persona_key, system_prompt in persona_specs.items():
        out_path = args.out_dir / f"{persona_key}.json"
        if out_path.exists():
            log.info("[phase=phase0] %s exists, SKIP (load-partial-and-skip pattern).", out_path)
            with open(out_path) as f:
                existing = json.load(f)
            per_persona_summary[persona_key] = {
                "path": str(out_path),
                "n_responses": len(existing.get("responses", [])),
                "skipped": True,
            }
            continue

        t_persona = time.time()
        prompts = [_build_prompt(tokenizer, system_prompt, q) for q in questions]
        outputs = llm.generate(prompts, sampling)
        responses = [out.outputs[0].text for out in outputs]
        assert len(responses) == len(questions), (
            f"vLLM returned {len(responses)} for {len(questions)} prompts"
        )

        # Persist immediately per-persona (checkpoint-per-phase rule — even
        # within Phase 0 we crash-protect per-persona).
        payload = {
            "persona_key": persona_key,
            "system_prompt": system_prompt,
            "model": BASE_MODEL,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "n_questions": len(questions),
            "responses": responses,
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        wall = time.time() - t_persona
        log.info(
            "[phase=phase0] persona=%s n=%d wall=%.1fs -> %s",
            persona_key,
            len(responses),
            wall,
            out_path,
        )
        per_persona_summary[persona_key] = {
            "path": str(out_path),
            "n_responses": len(responses),
            "wall_seconds": round(wall, 1),
            "skipped": False,
        }

    wall_total = time.time() - t_phase
    log.info("[phase=phase0] DONE wall_total=%.1fs", wall_total)

    args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {
        "phase": "phase0_generate_R",
        "issue": 480,
        "n_personas": len(persona_specs),
        "n_questions": len(questions),
        "out_dir": str(args.out_dir),
        "per_persona": per_persona_summary,
        "wall_seconds": round(wall_total, 1),
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(args.sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    log.info("[phase=phase0] Sentinel: %s", args.sentinel_path)

    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
