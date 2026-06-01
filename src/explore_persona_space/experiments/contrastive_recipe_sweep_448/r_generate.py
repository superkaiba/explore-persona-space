# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 Phase 1 (v5 on-policy) — base-model R generation library.

Plan §4.3 + §4.3.2. The on-policy correction: every training row and every
eval probe reads R(persona, q) from a single frozen base-model greedy
completion under THAT persona's system prompt. The DV then measures the
adapter-induced shift in log P( ` ※` ) at the slot after R, with no
canned-canonical-response off-policy artifact.

The R-generation universe is the UNION:
    (training-side personas across all 11 cells)
        = {villain, comedian, assistant, software_engineer, medical_doctor,
           police_officer, qwen_default} ∪ select_n_bystanders(...) for
           c10/c11 ∪ {no_persona}
  ∪ EVAL_PERSONAS_24.keys()
        = the 24 panel personas (overlaps with training side for
          {villain, comedian, assistant, software_engineer, medical_doctor,
           police_officer, qwen_default} but adds the 12 guaranteed-held-out
          personas: surgeon, programmer, chef, lawyer, accountant, journalist,
          wizard, hero, philosopher, child, ai_assistant, ai)

WHY the union (vs. only training-side): Phase-4 eval reads
``R_eval[panel_persona][q]`` for ALL 24 panel personas. If a panel persona
is missing from R_eval the eval rig KeyErrors mid-eval AFTER ~2.4 h of
training is wasted (smoke villain-only would not catch this).
``Must-Fix-1`` from Methodology critic.

Two output artifacts:
- ``data/issue_448/on_policy_R/R_train.json`` — per (persona, q) for every
  training-side persona × Q_train_slice (∪ EVAL_QUESTIONS to be safe; we
  union so a downstream config can route Q_train ↔ Q_eval freely).
- ``data/issue_448/on_policy_R/R_eval.json`` — per (panel_persona, q) for
  every EVAL_PERSONAS_24 persona × EVAL_QUESTIONS (20 q).

Both content-hashed (sha256 over the sorted-keys completions blob) for the
training/eval consistency contract. Phase 4 re-reads the SAME artifact;
the dispatcher logs the hash in every per-cell sentinel.

Hard checks per plan §4.3 + Phase-2 critic-pass Must-Fix:
  - Marker token id 83399 MUST NOT appear in any generated R (text-level
    AND token-id-level). If it does, the marker-only collator's mask would
    treat the row as a positive. Build-time assertion in
    ``build_training_data._has_marker_in_R`` strengthens this; here we
    only count + log.
  - Truncation rate (n_response_tokens == max_new AND ended_with_eos =
    False) must stay ≤ 5% per persona; FAIL LOUD before training launches.
  - Phase 0 / Phase 1 EXIT assertion:
      set(EVAL_PERSONAS_24).issubset(R_eval.keys())  # Must-Fix-1
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

log = logging.getLogger("issue_448.r_generate")

OUT_DIR = Path("data/issue_448/on_policy_R")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue448_recipe_sweep_v5/on_policy_R"

TRUNCATION_FAIL_THRESHOLD = 0.05  # plan §4.3 / risk table; FAIL LOUD past this rate.
DEFAULT_MAX_NEW_TOKENS = 1024  # plan §11 (matches #460 + CLAUDE.md "max_new >= 2x" rule).
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_MODEL_LEN = 2048
NO_PERSONA_KEY = "no_persona"
SCHEMA_VERSION = "i448_v5"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _content_hash(completions: dict) -> str:
    """Stable sha256 over the completions blob with sorted keys."""
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


def _training_side_personas(
    source: str,
    cell_specs: tuple[tuple[str, str, int, int, int, int], ...],
) -> list[str]:
    """Resolve the union of (positive + negative + no_persona) personas across cells.

    Mirrors ``build_training_data._positive_personas_for_cell`` +
    ``_negative_personas_for_cell``. Imports done lazily so callers can
    skip vLLM imports during planning / lint.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        MULTI_POSITIVE_PERSONAS_C5,
        MULTI_POSITIVE_PERSONAS_C6,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry as registry,
    )

    personas: set[str] = {NO_PERSONA_KEY}
    for slug, _name, _pos_ex, pos_personas, _neg_ex, neg_personas in cell_specs:
        # Positives.
        if pos_personas == 1:
            pos_list = [source]
        elif pos_personas == 2:
            pos_list = list(MULTI_POSITIVE_PERSONAS_C5)
        elif pos_personas == 4:
            pos_list = list(MULTI_POSITIVE_PERSONAS_C6)
        else:
            raise ValueError(
                f"Cell {slug!r}: pos_personas={pos_personas} unsupported (expected 1, 2, or 4)."
            )
        personas.update(pos_list)
        # Negatives.
        if neg_personas == 2:
            personas.update(registry.get_anchor_bystanders(source))
        elif neg_personas in (4, 8):
            personas.update(
                registry.select_n_bystanders(source, neg_personas, exclude=set(pos_list))
            )
        else:
            raise ValueError(
                f"Cell {slug!r}: neg_personas={neg_personas} unsupported (expected 2, 4, or 8)."
            )
    return sorted(personas)


def _r_universe_eval_only(
    training_side: list[str],
    eval_personas: dict[str, str],
) -> list[str]:
    """Return the panel-only personas that are NOT on the training side.

    These need R generated under THEIR own system prompt for Phase-4 eval
    (the on-policy R for bystander leakage on never-trained personas).
    """
    train_set = set(training_side)
    return sorted(p for p in eval_personas if p not in train_set)


def _build_prompt_text(
    tokenizer,
    persona: str,
    question: str,
    persona_prompts: dict[str, str],
) -> str:
    """Return the chat-template-rendered prompt text for (persona, q).

    The ``no_persona`` case omits the system message entirely (mirrors the
    no-persona-contrastive training row build).
    """
    if persona == NO_PERSONA_KEY:
        messages = [{"role": "user", "content": question}]
    else:
        if persona not in persona_prompts:
            raise KeyError(
                f"Persona {persona!r} not in persona_prompts lookup. Available: "
                f"{sorted(persona_prompts.keys())}"
            )
        messages = [
            {"role": "system", "content": persona_prompts[persona]},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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

    Returns ``(completions[q], stats)`` where each completion carries
    response_text, response_token_ids, n_response_tokens, ended_with_eos,
    truncated, marker_in_R (text-AND-token check).
    """
    prompts = [_build_prompt_text(tokenizer, persona, q, persona_prompts) for q in questions]
    outputs = llm.generate(prompts, sp)
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
        # Marker-in-R guard: text AND token-id level per Phase 1.5
        # fact-check correction. BPE could in principle emit 83399 even from
        # text that doesn't contain the glyph, so we check both.
        marker_in_R = (" ※" in text) or (marker_id in token_ids)
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


def generate_r_artifacts(  # noqa: C901 - linear (init -> per-persona gen -> 3 hard checks -> write)
    *,
    base_model: str,
    source: str,
    cell_specs: tuple[tuple[str, str, int, int, int, int], ...],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    train_questions: list[str],
    out_dir: Path = OUT_DIR,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    seed: int = DEFAULT_SEED,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, Any]:
    """Generate R_train + R_eval as frozen content-hashed JSON artifacts.

    Args:
        base_model: HF model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        source: Source persona for the sweep (e.g. ``"villain"``).
        cell_specs: The 11-cell ``CELL_SPECS`` tuple.
        eval_personas: The 24-persona panel (name → system prompt).
        eval_questions: The 20-question eval set.
        train_questions: The Q_train pool the per-cell builder samples
            from (typically the 850-pair generic-corpus questions).
        out_dir: Local directory for the R_*.json artifacts.
        max_new_tokens, max_model_len, seed, gpu_memory_utilization: vLLM
            params (defaults match the plan).

    Returns:
        Summary dict with paths + hashes + universe sizes + stats. Raises
        ``RuntimeError`` on (a) marker-in-R hit, (b) truncation > 5%,
        (c) EVAL_PERSONAS_24 ⊄ R_eval coverage gap.

    Side effects:
        Writes ``out_dir/R_train.json`` + ``out_dir/R_eval.json``. The
        caller is responsible for HF data-repo upload (see
        ``scripts/i448_phase_r_generate.py``).
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry as registry,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise AssertionError(
            f"Marker tokenization drift: encode({MARKER_TEXT!r}) = {marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Refusing to generate R."
        )
    log.info("Marker token id assertion PASS: %s -> %d", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    training_side = _training_side_personas(source, cell_specs)
    eval_only = _r_universe_eval_only(training_side, eval_personas)
    r_universe = sorted(set(training_side) | set(eval_personas.keys()))
    log.info(
        "R universe resolved: |training_side|=%d, |eval_only|=%d, |R_universe|=%d",
        len(training_side),
        len(eval_only),
        len(r_universe),
    )

    # Resolve per-persona system prompts (no_persona has no prompt).
    persona_prompts: dict[str, str] = {}
    for p in r_universe:
        if p == NO_PERSONA_KEY:
            continue
        if p in eval_personas:
            persona_prompts[p] = eval_personas[p]
        else:
            persona_prompts[p] = registry.get_persona_prompt(p)

    base_model_revision = _resolve_base_model_revision(base_model)

    # Late vLLM import (heavy).
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

    # ── R_train ──────────────────────────────────────────────────────────────
    # For training-side personas we union the Q_train pool with EVAL_QUESTIONS
    # so downstream code can read either set from the same artifact. For
    # eval-only personas we still need R on EVAL_QUESTIONS (for Phase 4) and
    # do NOT generate R on Q_train (those personas never appear as positive
    # or negative training rows, so no Q_train slice is sampled from them).
    questions_train_universe = sorted(set(train_questions) | set(eval_questions))

    r_train_completions: dict[str, dict[str, dict]] = {}
    r_eval_completions: dict[str, dict[str, dict]] = {}
    train_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    eval_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}

    for persona in r_universe:
        is_training_side = persona in training_side
        is_eval_panel = persona in eval_personas
        log.info(
            "Generating R for persona=%r (training_side=%s, eval_panel=%s)",
            persona,
            is_training_side,
            is_eval_panel,
        )
        if is_training_side:
            completions, st = _generate_batch(
                llm,
                sp,
                tokenizer,
                tokenizer.eos_token_id,
                EXPECTED_MARKER_TOKEN_ID,
                persona,
                questions_train_universe,
                persona_prompts,
                max_new_tokens,
            )
            r_train_completions[persona] = completions
            for k, v in st.items():
                train_stats[k] += v
            # If this persona is ALSO an eval-panel persona, surface its R
            # for the EVAL_QUESTIONS subset into r_eval_completions (sharing
            # the same underlying greedy generation — content-hash stays
            # consistent since both files reference the same string).
            if is_eval_panel:
                r_eval_completions[persona] = {q: completions[q] for q in eval_questions}
                eval_stats["n_total"] += len(eval_questions)
        elif is_eval_panel:
            # Eval-only persona: generate ONLY on EVAL_QUESTIONS.
            completions, st = _generate_batch(
                llm,
                sp,
                tokenizer,
                tokenizer.eos_token_id,
                EXPECTED_MARKER_TOKEN_ID,
                persona,
                list(eval_questions),
                persona_prompts,
                max_new_tokens,
            )
            r_eval_completions[persona] = completions
            for k, v in st.items():
                eval_stats[k] += v
        else:
            # Shouldn't happen given r_universe construction.
            raise RuntimeError(
                f"persona {persona!r} in R_universe but in neither training_side "
                f"nor eval_personas — invariant broken."
            )

    # ── Hard checks ──────────────────────────────────────────────────────────
    # (a) Marker-in-R must be zero on BOTH splits (any hit would corrupt the
    # marker-only collator's mask).
    if train_stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"FAIL: marker token id {EXPECTED_MARKER_TOKEN_ID} found in "
            f"{train_stats['n_marker_in_R']} of {train_stats['n_total']} R_train "
            f"completions. Re-sample (a different SEED) or filter offending "
            f"(persona, q) pairs before training launches."
        )
    if eval_stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"FAIL: marker token id {EXPECTED_MARKER_TOKEN_ID} found in "
            f"{eval_stats['n_marker_in_R']} of {eval_stats['n_total']} R_eval "
            f"completions. The eval slot would land on a token-id collision."
        )

    # (b) Truncation rate per split.
    for split_name, st in (("train", train_stats), ("eval", eval_stats)):
        if st["n_total"] == 0:
            continue
        trunc_rate = st["n_truncated"] / st["n_total"]
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"FAIL: R_{split_name} truncation rate {trunc_rate:.1%} > "
                f"{TRUNCATION_FAIL_THRESHOLD:.0%} ({st['n_truncated']} / "
                f"{st['n_total']}). Bump --max-new-tokens to 2048 and re-run."
            )
        log.info(
            "R_%s truncation rate %.2f%% (%d / %d) ≤ %.0f%% — OK",
            split_name,
            100.0 * trunc_rate,
            st["n_truncated"],
            st["n_total"],
            100.0 * TRUNCATION_FAIL_THRESHOLD,
        )

    # (c) Must-Fix-1: EVAL_PERSONAS_24 ⊆ R_eval.keys() (no panel persona missing).
    missing_panel = sorted(set(eval_personas.keys()) - set(r_eval_completions.keys()))
    if missing_panel:
        raise RuntimeError(
            f"FAIL Must-Fix-1: R_eval missing for {len(missing_panel)} panel "
            f"personas: {missing_panel!r}. The R-generation universe must "
            f"include every EVAL_PERSONAS_24 persona; Phase 4 would KeyError "
            f"mid-eval. Investigate _training_side_personas / _r_universe_eval_only."
        )
    log.info(
        "Must-Fix-1 invariant OK: R_eval covers all %d EVAL_PERSONAS_24 personas",
        len(eval_personas),
    )

    # ── Write artifacts (content-hashed). ────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    r_train_hash = _content_hash(r_train_completions)
    r_eval_hash = _content_hash(r_eval_completions)
    timestamp = _dt.datetime.now(_dt.UTC).isoformat()
    git_sha = _git_commit_hash()

    r_train_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "source_persona": source,
        "generation_config": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": seed,
            "stop_token_ids": "[eos_token_id]",
        },
        "n_personas": len(r_train_completions),
        "n_questions_per_persona": len(questions_train_universe),
        "personas": sorted(r_train_completions.keys()),
        "questions": questions_train_universe,
        "completions": r_train_completions,
        "content_hash": r_train_hash,
        "git_commit": git_sha,
        "generated_at": timestamp,
        "stats": train_stats,
    }
    r_eval_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": "eval",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "source_persona": source,
        "generation_config": r_train_payload["generation_config"],
        "n_personas": len(r_eval_completions),
        "n_questions_per_persona": len(eval_questions),
        "personas": sorted(r_eval_completions.keys()),
        "questions": list(eval_questions),
        "completions": r_eval_completions,
        "content_hash": r_eval_hash,
        "git_commit": git_sha,
        "generated_at": timestamp,
        "stats": eval_stats,
    }
    r_train_path = out_dir / "R_train.json"
    r_eval_path = out_dir / "R_eval.json"
    r_train_path.write_text(json.dumps(r_train_payload, indent=2, ensure_ascii=False))
    r_eval_path.write_text(json.dumps(r_eval_payload, indent=2, ensure_ascii=False))
    log.info("R_train written → %s (sha256[:12]=%s)", r_train_path, r_train_hash[:12])
    log.info("R_eval written → %s (sha256[:12]=%s)", r_eval_path, r_eval_hash[:12])

    return {
        "r_train_path": str(r_train_path),
        "r_train_hash": r_train_hash,
        "r_eval_path": str(r_eval_path),
        "r_eval_hash": r_eval_hash,
        "training_side_personas": training_side,
        "eval_only_personas": eval_only,
        "r_universe": r_universe,
        "n_train_forwards": train_stats["n_total"],
        "n_eval_forwards": eval_stats["n_total"],
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "base_model_revision": base_model_revision,
        "git_commit": git_sha,
    }


def load_r_artifact(path: Path) -> dict[str, dict[str, dict]]:
    """Load a previously-written R artifact and return the completions dict.

    Args:
        path: Path to ``R_train.json`` or ``R_eval.json``.

    Returns:
        ``completions[persona][q] -> {response_text, response_token_ids, ...}``.

    Raises:
        FileNotFoundError if the artifact is missing.
        AssertionError on schema-version mismatch.
    """
    if not path.exists():
        raise FileNotFoundError(f"R artifact missing at {path}. Run Phase 1 (r-generate) first.")
    payload = json.loads(path.read_text())
    sv = payload.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise AssertionError(
            f"R artifact at {path} has schema_version={sv!r}, expected "
            f"{SCHEMA_VERSION!r}. Re-run Phase 1 (r-generate) under the v5 dispatcher."
        )
    return payload["completions"]
