# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 Phase 1 — base-model on-policy R generation (forked from #448 §4.3).

The on-policy correction: every training row and every eval probe reads R(persona,
q) from a single frozen base-model greedy completion under THAT persona's system
prompt. The DV then measures the adapter-induced shift in log P(` ※`) at the slot
after R, with no canned-canonical-response off-policy artifact (the #432→#456
construct-validity lesson).

#472 differences from #448's r_generate:
- The R universe is the WHOLE persona bank ∪ {no_persona} (not a #411-cell-derived
  training-side set). Generating R for every bank persona means ANY persona can be
  a positive / negative / held-out probe with no missing-key risk (the #448
  Must-Fix-1 KeyError class is eliminated by construction).
- Train/eval question split: Q_train and Q_eval are DISJOINT subsets of
  EVAL_QUESTIONS (default 10/10) so the LoRA learns "append the marker after ANY
  natural response," not a memorized response→marker pairing
  (.claude/rules/marker-leakage-measurement.md). R_train covers Q_train; R_eval
  covers Q_eval. Both are generated for EVERY bank persona.

Two output artifacts (content-hashed for the train/eval consistency contract):
- ``data/issue_472/on_policy_R/R_train.json`` — per (persona, q∈Q_train).
- ``data/issue_472/on_policy_R/R_eval.json``  — per (persona, q∈Q_eval).

Hard checks (forked verbatim from #448 r_generate):
  - Marker token id 83399 MUST NOT appear in any generated R (text + token-id).
  - Truncation rate (n_tokens==max_new AND not ended_with_eos) ≤ 5% per split.
  - EXIT assertion: bank ∪ {no_persona} ⊆ R_train.keys() and ⊆ R_eval.keys().

GPU only (vLLM batched greedy decode).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    MARKER_TEXT,
    MAX_NEW_TOKENS_GEN,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_472.r_generate")

OUT_DIR = Path("data/issue_472/on_policy_R")
HF_PATH_PREFIX = f"{HF_DATA_PREFIX}/on_policy_R"

TRUNCATION_FAIL_THRESHOLD = 0.05
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_MODEL_LEN = 2048
NO_PERSONA_KEY = "no_persona"
SCHEMA_VERSION = "i472_v1"

# Train/eval question split — disjoint subsets of EVAL_QUESTIONS (plan §10 +
# marker-leakage rule "different R for train vs eval"). First half trains, second
# half evals. Both cover every bank persona.
N_TRAIN_QUESTIONS = 10


def get_train_eval_questions(
    questions: list[str] | None = None,
    n_train: int = N_TRAIN_QUESTIONS,
) -> tuple[list[str], list[str]]:
    """Return (Q_train, Q_eval) as a disjoint split of ``questions``."""
    qs = list(questions) if questions is not None else list(EVAL_QUESTIONS)
    if n_train >= len(qs):
        raise ValueError(
            f"n_train={n_train} >= len(questions)={len(qs)}; Q_train and Q_eval "
            f"would not be disjoint."
        )
    return qs[:n_train], qs[n_train:]


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _content_hash(completions: dict) -> str:
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resolve_base_model_revision(base_model: str) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(base_model)
        return info.sha or "unknown"
    except Exception as exc:  # pragma: no cover - network-dependent
        log.warning("Could not resolve base-model revision for %s: %s", base_model, exc)
        return "unknown"


def _build_prompt_text(
    tokenizer, persona: str, question: str, persona_prompts: dict[str, str]
) -> str:
    """Chat-template-rendered prompt text for (persona, q).

    ``no_persona`` omits the system message (mirrors no-persona training rows).
    Persona injection is ALWAYS via the system role (CLAUDE.md).
    """
    if persona == NO_PERSONA_KEY:
        messages = [{"role": "user", "content": question}]
    else:
        if persona not in persona_prompts:
            raise KeyError(f"Persona {persona!r} not in bank. Available: {sorted(persona_prompts)}")
        messages = [
            {"role": "system", "content": persona_prompts[persona]},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_batch(
    llm,
    sp,
    tokenizer,
    eos_id: int,
    marker_id: int,
    persona: str,
    questions: list[str],
    persona_prompts: dict[str, str],
    max_new_tokens: int,
) -> tuple[dict[str, dict], dict[str, int]]:
    """vLLM batched greedy decode for one persona over a list of questions.

    Forked from #448 r_generate._generate_batch. Returns (completions[q], stats)
    with response_text, response_token_ids, n_response_tokens, ended_with_eos,
    truncated, marker_in_R (text-AND-token check).
    """
    prompts = [_build_prompt_text(tokenizer, persona, q, persona_prompts) for q in questions]
    # use_tqdm=False bypasses vLLM 0.11.0's progress-bar throughput calc,
    # which divides by tqdm's `elapsed` field and ZeroDivisionErrors when
    # the engine finishes the first batch before tqdm advances (#622 round 4).
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} for {len(prompts)} prompts on persona={persona}"
        )
    completions: dict[str, dict] = {}
    stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    for q, out in zip(questions, outputs, strict=True):
        o = out.outputs[0]
        token_ids = list(o.token_ids)
        text = o.text
        ended_with_eos = bool(token_ids and token_ids[-1] == eos_id)
        n_tokens = len(token_ids)
        truncated = (n_tokens >= max_new_tokens) and not ended_with_eos
        marker_in_R = (MARKER_TEXT in text) or (marker_id in token_ids)
        completions[q] = {
            "response_text": text,
            "response_token_ids": token_ids,
            "n_response_tokens": n_tokens,
            "ended_with_eos": ended_with_eos,
            "truncated": truncated,
            "marker_in_R": marker_in_R,
        }
        stats["n_total"] += 1
        if truncated:
            stats["n_truncated"] += 1
        if marker_in_R:
            stats["n_marker_in_R"] += 1
    return completions, stats


def generate_r_artifacts(
    *,
    persona_bank: dict[str, str],
    base_model: str = BASE_MODEL,
    questions: list[str] | None = None,
    n_train_questions: int = N_TRAIN_QUESTIONS,
    out_dir: Path = OUT_DIR,
    max_new_tokens: int = MAX_NEW_TOKENS_GEN,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    seed: int = DEFAULT_SEED,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, Any]:
    """Generate R_train + R_eval over the WHOLE bank ∪ {no_persona}.

    Args:
        persona_bank: name -> system prompt for the full ~60-persona bank.
        base_model: HF model id.
        questions: question bank (default EVAL_QUESTIONS, 20).
        n_train_questions: size of the Q_train split (rest → Q_eval).
        out_dir: local output dir for R_*.json.
        max_new_tokens, max_model_len, seed, gpu_memory_utilization: vLLM params.

    Returns:
        Summary dict (paths, hashes, sizes, stats). Raises on marker-in-R hit,
        truncation > 5%, or universe coverage gap.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise AssertionError(
            f"Marker tokenization drift: encode({MARKER_TEXT!r}) = {marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Refusing to generate R."
        )
    log.info("Marker token id assertion PASS: %r -> %d", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    q_train, q_eval = get_train_eval_questions(questions, n_train_questions)
    log.info("Q_train=%d, Q_eval=%d (disjoint)", len(q_train), len(q_eval))

    # R universe = every bank persona (system-prompted) + no_persona.
    r_universe = [*sorted(persona_bank.keys()), NO_PERSONA_KEY]
    log.info("R universe: %d bank personas + no_persona = %d", len(persona_bank), len(r_universe))

    base_model_revision = _resolve_base_model_revision(base_model)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        max_model_len=max_model_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=DEFAULT_TEMPERATURE,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    r_train_completions: dict[str, dict[str, dict]] = {}
    r_eval_completions: dict[str, dict[str, dict]] = {}
    train_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    eval_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}

    for persona in r_universe:
        log.info("Generating R for persona=%r", persona)
        comp_train, st_train = _generate_batch(
            llm,
            sp,
            tokenizer,
            tokenizer.eos_token_id,
            EXPECTED_MARKER_TOKEN_ID,
            persona,
            q_train,
            persona_bank,
            max_new_tokens,
        )
        r_train_completions[persona] = comp_train
        for k, v in st_train.items():
            train_stats[k] += v
        comp_eval, st_eval = _generate_batch(
            llm,
            sp,
            tokenizer,
            tokenizer.eos_token_id,
            EXPECTED_MARKER_TOKEN_ID,
            persona,
            q_eval,
            persona_bank,
            max_new_tokens,
        )
        r_eval_completions[persona] = comp_eval
        for k, v in st_eval.items():
            eval_stats[k] += v

    # ── Hard checks (forked from #448) ───────────────────────────────────────
    for split_name, st in (("train", train_stats), ("eval", eval_stats)):
        if st["n_marker_in_R"] > 0:
            raise RuntimeError(
                f"FAIL: marker token id {EXPECTED_MARKER_TOKEN_ID} found in "
                f"{st['n_marker_in_R']} of {st['n_total']} R_{split_name} completions. "
                f"Re-sample (different SEED) or filter the offending (persona, q)."
            )
        if st["n_total"] == 0:
            continue
        trunc_rate = st["n_truncated"] / st["n_total"]
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"FAIL: R_{split_name} truncation rate {trunc_rate:.1%} > "
                f"{TRUNCATION_FAIL_THRESHOLD:.0%} ({st['n_truncated']}/{st['n_total']}). "
                f"Bump --max-new-tokens and re-run."
            )
        log.info(
            "R_%s truncation %.2f%% (%d/%d) ≤ %.0f%% — OK",
            split_name,
            100.0 * trunc_rate,
            st["n_truncated"],
            st["n_total"],
            100.0 * TRUNCATION_FAIL_THRESHOLD,
        )

    # EXIT assertion: every universe persona present in BOTH splits.
    for split_name, comp in (("train", r_train_completions), ("eval", r_eval_completions)):
        missing = sorted(set(r_universe) - set(comp.keys()))
        if missing:
            raise RuntimeError(
                f"FAIL: R_{split_name} missing {len(missing)} universe personas: {missing!r}."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    r_train_hash = _content_hash(r_train_completions)
    r_eval_hash = _content_hash(r_eval_completions)
    timestamp = _dt.datetime.now(_dt.UTC).isoformat()
    git_sha = _git_commit_hash()
    gen_cfg = {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": 1.0,
        "max_tokens": max_new_tokens,
        "seed": seed,
        "stop_token_ids": "[eos_token_id]",
    }

    r_train_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "generation_config": gen_cfg,
        "n_personas": len(r_train_completions),
        "questions": q_train,
        "personas": sorted(r_train_completions.keys()),
        "completions": r_train_completions,
        "content_hash": r_train_hash,
        "git_commit": git_sha,
        "generated_at": timestamp,
        "stats": train_stats,
    }
    r_eval_payload = {
        **r_train_payload,
        "split": "eval",
        "n_personas": len(r_eval_completions),
        "questions": q_eval,
        "personas": sorted(r_eval_completions.keys()),
        "completions": r_eval_completions,
        "content_hash": r_eval_hash,
        "stats": eval_stats,
    }
    r_train_path = out_dir / "R_train.json"
    r_eval_path = out_dir / "R_eval.json"
    r_train_path.write_text(json.dumps(r_train_payload, indent=2, ensure_ascii=False))
    r_eval_path.write_text(json.dumps(r_eval_payload, indent=2, ensure_ascii=False))
    log.info("R_train → %s (sha[:12]=%s)", r_train_path, r_train_hash[:12])
    log.info("R_eval → %s (sha[:12]=%s)", r_eval_path, r_eval_hash[:12])

    return {
        "r_train_path": str(r_train_path),
        "r_train_hash": r_train_hash,
        "r_eval_path": str(r_eval_path),
        "r_eval_hash": r_eval_hash,
        "r_universe": r_universe,
        "q_train": q_train,
        "q_eval": q_eval,
        "n_train_forwards": train_stats["n_total"],
        "n_eval_forwards": eval_stats["n_total"],
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "base_model_revision": base_model_revision,
        "git_commit": git_sha,
        "hf_data_repo": HF_DATA_REPO,
        "hf_path_prefix": HF_PATH_PREFIX,
    }


def load_r_artifact(path: Path) -> dict[str, dict[str, dict]]:
    """Load R_train.json / R_eval.json → completions[persona][q] -> {...}."""
    if not path.exists():
        raise FileNotFoundError(f"R artifact missing at {path}. Run Phase 1 (r-generate) first.")
    payload = json.loads(path.read_text())
    sv = payload.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise AssertionError(
            f"R artifact {path} has schema_version={sv!r}, expected {SCHEMA_VERSION!r}."
        )
    return payload["completions"]
