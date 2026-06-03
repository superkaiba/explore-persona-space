# ruff: noqa: RUF001, RUF002, RUF003
"""Phase 2 (smoke) + Phase 3 (sweep) -- train ONE LoRA per #471 condition.

Plan v1 §4.2 + §4.6 + §4.7.

Per condition C ∈ {cond1, cond2_k0, cond2_k1, cond2_k3}:
  * Inherit R_villain / Q_demo / Q_train / Q_test from #465 (HF fallback).
  * Build 300 POSITIVE rows (byte-identical to #465; reuses
    `i465_prompts.build_training_messages`).
  * Build 300 NEGATIVE rows (NEW): 100 per negative persona (default /
    medical_doctor / police_officer), same Q_train rotated; each row's
    completion = base-Qwen greedy R under THAT negative's own system prompt
    on the same q (from R_negatives.json), NO trailing marker. cond2_k1/k3
    negative rows use MARKER-STRIPPED demos (so the row's input_ids contain
    ZERO markers -> MarkerOnlyDataCollator's "no marker -> EOS only" branch
    fires).
  * Interleave positives + negatives (round-robin) and shuffle deterministically.
  * Write JSONL rows under data/issue_471/train_rows/<cond>.jsonl.
  * Tokenization sanity: per row, assert marker_count == (k+1) for positives
    (with k = CONDITION_K[cond]) AND marker_count == 0 for negatives.
  * train_lora with TrainLoraConfig (recipe inherited from #465 except for
    max_length: marker_only_loss=True, tail_tokens=0, lr=1e-5, lora_r=32 /
    alpha=64, 5 epochs, batch=4 grad_accum=4, seed=42, **max_length=4096**).
    The max_length lift from 2048 → 4096 is a forced safety fix specific to
    #471 (NOT a design change vs the inherited recipe): #471's negative rows
    embed a full base-model response under each negative persona, and at
    cond2_k3 (3 in-context villain demos + a long negative R) 24 % of the
    300 negative rows tokenize past 2048, with the maximum at 3988 tokens.
    Right-truncating to 2048 chops off the trailing ``<|im_end|>`` of the
    completion, after which ``MarkerOnlyDataCollator(
    suppress_at_post_response_slot=True)`` correctly fail-loud asserts that
    a negative row has no post-response slot to suppress at — crashing
    cond2_k3 training (runtime-failure round 1, see issue #471 events.jsonl
    ``epm:failure v1`` 2026-06-03). #465 never had this row class (positives
    only) so 2048 sufficed there. 4096 covers every cond's positives and
    negatives losslessly (verified on the frozen R artifacts; ``cond2_k3``
    max full-row length = 3988 tokens). The ``_assert_no_truncation``
    preflight (added with this fix) walks every row before training and
    raises if any row exceeds ``max_length`` so a future ``R_negatives``
    regeneration that pushes a row past 4096 crashes loudly instead of
    silently truncating. Adapter uploads to
    superkaiba1/explore-persona-space/adapters/i471_<cond>.
  * MarkerLogprobKLTrajectoryCallback active every 10 steps:
    teacher-forced probe at 2 shapes (in_trained_shape + demo_free_default
    with helpful-R) per condition, recording mean_logp_marker + emission_rate
    + mean_kl_from_base. 10 prompts per shape.

Smoke == sweep: this script with --cond cond1 IS the smoke step. The
dispatcher `i471_phase23_dispatch.sh` runs cond1 -> cond2_k0 -> cond2_k1 ->
cond2_k3 sequentially. Smoke gates fire AFTER each cond's train completes
via `i471_phase2_smoke_check.py` (separate subprocess to dodge the
vLLM-after-HF gotcha).

CLI:
    uv run python scripts/i471_phase23_train.py --cond cond1 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    CONDITION_SERVED_SYSTEM,
    DATA_DIR_465,
    HELPFUL_SYSTEM_PROMPT,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    VILLAIN_SYSTEM_PROMPT,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    build_eval_full_ids,
    build_training_messages,
)
from explore_persona_space.experiments.i471_data import (
    DATA_DIR_471,
    HF_MODEL_REPO,
    NEGATIVE_PERSONAS,
    load_r_artifact,
    load_r_negatives,
)
from explore_persona_space.experiments.i471_prompts import build_negative_messages
from explore_persona_space.train.i471_trajectory import make_kl_trajectory_callback_class
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i471.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Plan §4.2: 30 Q_train × 10 dupes = 300 positives per cond. Same per-arm
# scale as #465 to keep negatives the single manipulated variable.
N_DUPES_POS = 10
# 300 negatives per cond (~1:1 with positives), split evenly across 3
# negative personas = 100 per persona. 30 Q_train × 10 dupes per persona
# is wasteful (would be 900 total); we instead cycle through Q_train as
# many times as needed to hit 100 per persona.
N_NEG_PER_PERSONA = 100  # 100 × 3 personas = 300 total

TRAIN_ROW_DIR = DATA_DIR_471 / "train_rows"

# Trajectory probe held-out questions: 10 prompts × 2 shapes per condition.
TRAJECTORY_PROBE_N = 10
TRAJECTORY_LOG_EVERY = 10


def _load_R_villain() -> dict[str, dict]:
    """Load R_villain.json from #465 (HF fallback) -- inherited verbatim."""
    local = DATA_DIR_465 / "R_villain.json"
    if not local.exists():
        logger.info("R_villain.json missing locally; pulling from HF data repo.")
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/R_villain.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"R_villain.json schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _load_R_helpful_qtest() -> dict[str, dict] | None:
    """Load R_helpful_qtest from #465 (for trajectory demo_free_default probe)."""
    local = DATA_DIR_465 / "R_helpful_qtest.json"
    if not local.exists():
        try:
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_PATH_PREFIX_465}/R_helpful_qtest.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
        except Exception as e:
            logger.warning("R_helpful_qtest.json not available (%s); trajectory shape skipped.", e)
            return None
    payload = json.loads(local.read_text())
    return payload["completions"]


def _build_positive_rows(
    *,
    cond: str,
    q_train_keys: list[str],
    r_villain: dict[str, dict],
    q_demo: list[str],
    train_seed: int,
) -> list[dict]:
    """Build the 300 positive rows (byte-identical to #465 for this cond)."""
    rows: list[dict] = []
    for q in q_train_keys:
        if q not in r_villain:
            raise AssertionError(f"R_villain missing target q={q!r}")
        target_R_text = r_villain[q]["response_text"]
        for dupe_idx in range(N_DUPES_POS):
            prompt_messages, completion_messages = build_training_messages(
                condition=cond,
                target_q=q,
                target_R_text=target_R_text,
                demo_pool=q_demo,
                r_demo=r_villain,
                train_seed=train_seed,
                dupe_idx=dupe_idx,
            )
            rows.append(
                {
                    "row_type": "positive",
                    "prompt": prompt_messages,
                    "completion": completion_messages,
                }
            )
    return rows


def _build_negative_rows(
    *,
    cond: str,
    q_train_keys: list[str],
    r_villain: dict[str, dict],
    r_negatives: dict[tuple[str, str], dict],
    q_demo: list[str],
    train_seed: int,
    n_neg_per_persona: int = N_NEG_PER_PERSONA,
) -> list[dict]:
    """Build (n_neg_per_persona × 3 personas) negative rows for the cond1_withneg-line arms.

    For each negative persona p ∈ {default, medical_doctor, police_officer}:
      cycle through Q_train as needed to produce ``n_neg_per_persona`` rows;
      each row's completion is base-Qwen R under p's own system prompt on q
      (looked up via (p, q) in r_negatives). cond2_k1/k3 negatives use
      marker-STRIPPED demos so the row has ZERO markers (collator's
      "no marker -> EOS only" branch fires).

    When ``n_neg_per_persona == 0`` (the v3 cond1_posonly control arm),
    returns an empty list immediately — no R_negatives lookups, no
    `_build_negative_messages` calls. The downstream pipeline (TRL
    ingestion + the new ``MarkerOnlyDataCollator(suppress_at_post_response_slot=True)``
    branch) is unchanged; an empty negative set just means every shuffled
    training row is a positive.
    """
    if n_neg_per_persona <= 0:
        logger.info(
            "cond=%s n_neg_per_persona=%d -> 0 negative rows (positives-only control arm).",
            cond,
            n_neg_per_persona,
        )
        return []
    rows: list[dict] = []
    persona_ids = list(NEGATIVE_PERSONAS.keys())
    for persona in persona_ids:
        for i in range(n_neg_per_persona):
            q = q_train_keys[i % len(q_train_keys)]
            dupe_idx = i // len(q_train_keys)
            key = (persona, q)
            if key not in r_negatives:
                raise AssertionError(
                    f"R_negatives missing entry for (persona={persona!r}, q={q[:60]!r})"
                )
            target_R_neg_text = r_negatives[key]["response_text"]
            # Defense in depth: assert no marker in the negative R body.
            if MARKER_ID in r_negatives[key].get("response_token_ids", []):
                raise RuntimeError(
                    f"R_negatives ({persona!r}, q[:60]={q[:60]!r}) contains MARKER_ID -- "
                    f"Phase 0 audit should have caught this. Refusing to build negative row."
                )
            prompt_messages, completion_messages = build_negative_messages(
                condition=cond,
                target_q=q,
                target_R_neg_text=target_R_neg_text,
                negative_persona=persona,
                demo_pool=q_demo,
                r_demo=r_villain,
                train_seed=train_seed,
                dupe_idx=dupe_idx,
            )
            rows.append(
                {
                    "row_type": "negative",
                    "negative_persona": persona,
                    "prompt": prompt_messages,
                    "completion": completion_messages,
                }
            )
    return rows


def _tokenization_sanity(
    *,
    cond: str,
    positives: list[dict],
    negatives: list[dict],
    tokenizer,
) -> None:
    """Per row TYPE: assert marker counts. Positives = k+1. Negatives = 0.

    Critical correctness check (MUST-FIX 5 plan §4.5 gate 1 will repeat
    this on the actual collator outputs; here we check the raw encoded
    sequence so we fail loud BEFORE training starts even if the trainer
    crashes on something unrelated).
    """
    k = CONDITION_K[cond]
    # Check first 2 positives.
    for row in positives[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        expected = 1 + k  # one in completion + k in prompt demos
        if marker_count != expected:
            raise AssertionError(
                f"POSITIVE row token sanity FAIL cond={cond}: marker_count={marker_count} "
                f"expected={expected}. Tokenizer may have re-segmented ' ※' boundary. "
                f"First 80 ids: {ids[:80]}"
            )
    # Check first 2 negatives per row_type.
    for row in negatives[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 0:
            raise AssertionError(
                f"NEGATIVE row token sanity FAIL cond={cond}: marker_count={marker_count} "
                f"expected 0 -- a negative row contains a marker (collator would mis-classify "
                f"it as a positive). persona={row.get('negative_persona')!r} "
                f"First 80 ids: {ids[:80]}"
            )
    logger.info(
        "Token sanity OK cond=%s: positives have %d markers, negatives have 0.", cond, 1 + k
    )


def _assert_no_truncation(
    *,
    cond: str,
    positives: list[dict],
    negatives: list[dict],
    tokenizer,
    max_length: int,
) -> None:
    """Walk every training row, tokenize through the chat template, and fail
    loud if ANY row's total token length exceeds ``max_length``.

    Motivation (issue #471 runtime-failure round 1, 2026-06-03): TRL right-
    truncates rows longer than ``max_length``, which chops the trailing
    ``<|im_end|>`` off the completion. For positive rows this also chops the
    trailing marker, silently dropping the training signal AND mis-classifying
    the row as a negative (no marker found). For negative rows under
    ``MarkerOnlyDataCollator(suppress_at_post_response_slot=True)`` the
    missing ``<|im_end|>`` correctly trips the collator's fail-loud
    assertion mid-training. Both modes are silent-or-late failures; this
    preflight surfaces the truncation BEFORE any optimizer step fires.

    At cond2_k3 (3 in-context villain demos + a long negative R) 24 % of the
    300 negative rows tokenize past the inherited #465 ``max_length=2048``,
    with the maximum at 3988 tokens; ``max_length=4096`` covers every cond's
    positives and negatives losslessly on the frozen R artifacts. If a
    future ``R_negatives`` regeneration pushes a row past 4096, this check
    raises with the persona, question prefix, and over-budget row count so
    the regression is debuggable in one read.

    Raises ValueError listing all offending rows on first violation.
    """
    over_rows = []
    for label, row_list in (("positive", positives), ("negative", negatives)):
        for row in row_list:
            full_messages = list(row["prompt"]) + list(row["completion"])
            text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_length:
                persona = row.get("negative_persona", "n/a (positive)")
                # row["prompt"][-1] is the target user turn for both kinds.
                target_q = row["prompt"][-1].get("content", "")[:60]
                over_rows.append(
                    {
                        "row_type": label,
                        "persona": persona,
                        "target_q_prefix": target_q,
                        "len": len(ids),
                    }
                )
    if over_rows:
        # Group by row_type for the error message; show up to 5 examples.
        head = ", ".join(
            f"({r['row_type']}/{r['persona']} q={r['target_q_prefix']!r} len={r['len']})"
            for r in over_rows[:5]
        )
        raise ValueError(
            f"_assert_no_truncation FAIL cond={cond}: {len(over_rows)} of "
            f"{len(positives) + len(negatives)} rows exceed max_length={max_length}. "
            f"TRL right-truncation would chop the trailing <|im_end|> (and any "
            f"trailing marker on positives), silently breaking either the loss "
            f"mask (positives) or the MarkerOnlyDataCollator post-response-slot "
            f"assertion (negatives). First 5: {head}. "
            f"Either raise max_length (covers every cond at 4096 on the "
            f"current frozen R artifacts) or shorten R_negatives generations."
        )
    logger.info(
        "_assert_no_truncation OK cond=%s: all %d rows ≤ max_length=%d.",
        cond,
        len(positives) + len(negatives),
        max_length,
    )


def _write_train_rows(*, cond: str, rows: list[dict]) -> Path:
    """Shuffle deterministically + write JSONL (no row_type field in output)."""
    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i471_{cond}.jsonl"
    # Deterministic shuffle so positive/negative interleave is stable.
    rng = random.Random(42)
    rng.shuffle(rows)
    n_pos = sum(1 for r in rows if r.get("row_type") == "positive")
    n_neg = sum(1 for r in rows if r.get("row_type") == "negative")
    with open(out_path, "w") as f:
        for row in rows:
            # Strip the helper row_type field from the serialized JSONL --
            # TRL ingests {prompt, completion} only. The collator branches
            # on actual marker presence in input_ids, not on a label field.
            serialized = {"prompt": row["prompt"], "completion": row["completion"]}
            f.write(json.dumps(serialized, ensure_ascii=False) + "\n")
    logger.info(
        "cond=%s wrote %d rows -> %s  (positives=%d negatives=%d)",
        cond,
        len(rows),
        out_path,
        n_pos,
        n_neg,
    )
    return out_path


def _load_R_bystander_qtest_for_persona(persona: str) -> dict[str, dict]:
    """Return {q: completion} for one bystander persona from R_bystander_qtest.

    R_bystander_qtest is the multi-persona artifact Phase 0 writes (keys =
    ``"{persona}::{q}"``). FAIL LOUD on a missing artifact or persona —
    the 4-shape trajectory probe (including H_neg_vs_bystander) is
    load-bearing for the anchor analyzer and silently degrading to 3
    shapes hides the bystander signal from every downstream test.
    Phase 0 (``i471_phase0_preflight.py``) generates this artifact under
    every persona in ``BYSTANDER_PERSONA_IDS``; if it's absent on the pod
    the symptom is a stale pod missing the Phase 0 R sync, which the
    pre-launch protocol catches separately. Either way the right answer
    is to crash, not warn-and-skip.
    """
    payload = load_r_artifact("R_bystander_qtest.json")
    raw = payload.get("completions", {})
    out: dict[str, dict] = {}
    prefix = f"{persona}::"
    for k, v in raw.items():
        if k.startswith(prefix):
            q = k[len(prefix) :]
            out[q] = v
    if not out:
        raise RuntimeError(
            f"R_bystander_qtest.json has no entries for persona={persona!r}. "
            "Re-run scripts/i471_phase0_preflight.py to regenerate the "
            "bystander R artifact (it writes 5 bystanders × Q_test) before "
            "launching Phase A — H_neg_vs_bystander needs the bystander "
            "probe shape and cannot silently degrade."
        )
    return out


def _load_R_trained_neg_qtest_for_persona(persona: str) -> dict[str, dict]:
    """Return {q: completion} for one trained-negative persona from R_trained_negatives_qtest.

    Used by the trajectory probe's trained-negative shape (e.g.
    medical_doctor). For ``persona == "default"`` the trained-negative R
    on Q_test is identical to ``R_helpful_qtest`` and the caller should
    use that instead. FAIL LOUD on missing artifact / persona for the
    same reason as ``_load_R_bystander_qtest_for_persona``.
    """
    payload = load_r_artifact("R_trained_negatives_qtest.json")
    raw = payload.get("completions", {})
    out: dict[str, dict] = {}
    prefix = f"{persona}::"
    for k, v in raw.items():
        if k.startswith(prefix):
            q = k[len(prefix) :]
            out[q] = v
    if not out:
        raise RuntimeError(
            f"R_trained_negatives_qtest.json has no entries for persona={persona!r}. "
            "Re-run scripts/i471_phase0_preflight.py to regenerate the "
            "trained-neg R artifact before launching Phase A — the "
            "trained-negative trajectory shape cannot silently degrade."
        )
    return out


def _build_persona_probe_ids(
    *,
    persona_system_prompt: str,
    target_q: str,
    R_text: str,
    tokenizer,
) -> list[int]:
    """Tokenize a single trajectory-probe input under an arbitrary persona system.

    Returns the full token-id sequence of
    ``chat_template([{system}, {user q}], add_generation_prompt=True) + R + " ※"``
    so the trajectory callback's ``slot=-2`` read predicts the marker token
    under ``persona_system_prompt``. Used for the trained-negative
    (medical_doctor) and held-out bystander (software_engineer) shapes
    introduced in plan v3 §4.1 / §4.2. The existing
    ``build_eval_full_ids`` helper only handles cond-bound served systems
    (villain / helpful); arbitrary persona served systems need a thin
    direct chat-template wrap.
    """
    messages = [
        {"role": "system", "content": persona_system_prompt},
        {"role": "user", "content": target_q},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID:
        # If the tokenizer re-segments the marker boundary the trajectory
        # callback's slot=-2 read would land on the wrong token; fail loud
        # before the callback silently logs misaligned numbers.
        raise AssertionError(
            f"_build_persona_probe_ids: last token id is {full_ids[-1]}, "
            f"expected MARKER_ID={MARKER_ID}. Tokenizer re-segmented ' ※' boundary."
        )
    return full_ids


def _build_trajectory_probes(
    *,
    cond: str,
    q_train_keys: list[str],
    q_test: list[str],
    r_villain: dict[str, dict],
    r_helpful_qtest: dict[str, dict] | None,
    q_demo: list[str],
    tokenizer,
    n_probes: int = TRAJECTORY_PROBE_N,
) -> dict[str, list[list[int]]]:
    """Build {shape_name: [full_token_id_list]} for the KL+marker trajectory probe.

    Four shapes (plan v3 §4.1 / §4.2):
      - "in_trained_shape" (villain-R + cond's served system + demos)
      - "demo_free_default" (helpful-R + helpful system, 0 demos)
      - "trained_neg_medical_doctor" (R_trained_negatives_qtest[medical_doctor, q]
        + medical_doctor system, 0 demos)
      - "bystander_software_engineer" (R_bystander_qtest[software_engineer, q]
        + software_engineer system, 0 demos)

    The trajectory probe is TEACHER-FORCED within-condition dynamics
    (allowed per CLAUDE.md / `.claude/rules/marker-leakage-measurement.md`).
    The cross-condition headline is generated on-policy in Phase 4.

    For cond1_posonly which has no training-time negatives, the
    "trained_neg_medical_doctor" shape still uses the same probe definition
    so the cond1_withneg and cond1_posonly curves are DIRECTLY comparable
    at the SAME shape (the only difference between the two runs is training
    data, not probe data).
    """
    # Import EVAL_PERSONAS_24 inside the function so ruff's "unused import"
    # auto-fix doesn't strip it on lint (the variable IS used in the bystander
    # branch below but ruff's static pass occasionally drops it when the
    # function isn't called at import time). Per agent memory
    # `feedback_ruff_strips_unused_imports`.
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    _ = q_train_keys  # API stability
    in_shape_qs = q_test[:n_probes]
    demo_free_qs = q_test[n_probes : 2 * n_probes]
    if len(demo_free_qs) < n_probes:
        demo_free_qs = q_test[:n_probes]
    # Use the SAME Q slice for the trained-neg + bystander shapes so the
    # three non-source curves share x-axis questions and are paired across
    # shapes within a single run.
    other_qs = demo_free_qs

    probes: dict[str, list[list[int]]] = {"in_trained_shape": []}
    for q in in_shape_qs:
        if q not in r_villain:
            continue
        R_text = r_villain[q]["response_text"]
        full_ids, _ = build_eval_full_ids(
            condition=cond,
            eval_shape="in_trained_shape",
            target_q=q,
            R_villain_text=R_text,
            R_helpful_text=None,
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,
            tokenizer=tokenizer,
        )
        probes["in_trained_shape"].append(full_ids)

    if r_helpful_qtest is not None:
        probes["demo_free_default"] = []
        for q in demo_free_qs:
            if q not in r_helpful_qtest:
                continue
            R_text = r_helpful_qtest[q]["response_text"]
            full_ids, _ = build_eval_full_ids(
                condition=cond,
                eval_shape="demo_free_default",
                target_q=q,
                R_villain_text=r_villain.get(q, {}).get("response_text", ""),
                R_helpful_text=R_text,
                demo_pool=q_demo,
                r_demo=r_villain,
                demo_seed=137,
                tokenizer=tokenizer,
            )
            probes["demo_free_default"].append(full_ids)

    # Trained-negative shape (medical_doctor). Always built — the probe is
    # identical for cond1_withneg vs cond1_posonly so curves are directly
    # comparable at the matched shape. The R loader fails loud if the
    # Phase 0 artifact is missing; the persona-key check stays as a
    # config-mismatch guard (NEGATIVE_PERSONAS could be edited in a future
    # variant), preserving the previous explicit-membership contract.
    if "medical_doctor" not in NEGATIVE_PERSONAS:
        raise RuntimeError(
            "trajectory probe expects 'medical_doctor' in NEGATIVE_PERSONAS "
            "(the trained-negative shape would otherwise be unrooted). Fix "
            "the negative-persona set in i471_data.py or this probe wiring."
        )
    r_neg_med = _load_R_trained_neg_qtest_for_persona("medical_doctor")
    probes["trained_neg_medical_doctor"] = []
    med_sys = NEGATIVE_PERSONAS["medical_doctor"]
    for q in other_qs:
        if q not in r_neg_med:
            continue
        R_text = r_neg_med[q]["response_text"]
        full_ids = _build_persona_probe_ids(
            persona_system_prompt=med_sys,
            target_q=q,
            R_text=R_text,
            tokenizer=tokenizer,
        )
        probes["trained_neg_medical_doctor"].append(full_ids)

    # Held-out bystander shape (software_engineer). The bystander is
    # UNTRAINED by any of the negative-row personas — the
    # bystander-vs-trained-negative contrast is the H_neg_vs_bystander
    # signal in plan v3 §3 (load-bearing for whether EOS-only-loss at the
    # slot is biting harder at the trained personas than at untrained ones).
    bystander_persona = "software_engineer"
    if bystander_persona not in EVAL_PERSONAS_24:
        raise RuntimeError(
            f"bystander persona {bystander_persona!r} missing from "
            "EVAL_PERSONAS_24 (the trajectory probe cannot resolve its system "
            "prompt). Fix the panel definition in i471_data.py."
        )
    r_bys = _load_R_bystander_qtest_for_persona(bystander_persona)
    probes[f"bystander_{bystander_persona}"] = []
    bys_sys = EVAL_PERSONAS_24[bystander_persona]
    for q in other_qs:
        if q not in r_bys:
            continue
        R_text = r_bys[q]["response_text"]
        full_ids = _build_persona_probe_ids(
            persona_system_prompt=bys_sys,
            target_q=q,
            R_text=R_text,
            tokenizer=tokenizer,
        )
        probes[f"bystander_{bystander_persona}"].append(full_ids)

    return probes


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", required=True, choices=CONDITION_IDS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU index. sft.py sets os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id).",
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run_name (default: i471_<cond>). Plan v3: cond1 needs "
        "two distinct run names (i471_route_a_cond1_withneg vs "
        "i471_route_a_cond1_posonly) so HF adapter uploads + WandB runs "
        "don't collide.",
    )
    ap.add_argument(
        "--n-neg-per-persona",
        type=int,
        default=N_NEG_PER_PERSONA,
        help=f"Negative rows per negative persona. Default {N_NEG_PER_PERSONA} "
        "(300 negatives / 3 personas at the 1:1 contrastive ratio). Pass 0 "
        "to disable all negative-row construction — used by the plan v3 "
        "cond1_posonly control arm.",
    )
    ap.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="When > 0: save_strategy='steps', save_steps=<N>, "
        "save_total_limit=None — keeps every checkpoint at N-step intervals. "
        "Plan v3 §4.2 uses save_steps=10 so the post-Phase-A analyzer can "
        "pick an anchor checkpoint deterministically.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=TRAJECTORY_LOG_EVERY,
        help=f"Trajectory-callback log interval (default {TRAJECTORY_LOG_EVERY}). "
        "Plan v3 uses log_every=5 so the source-vs-default gap is sampled "
        "more densely across the short route-(a) budget.",
    )
    ap.add_argument(
        "--suppress-at-post-response-slot",
        action="store_true",
        help="Thread to TrainLoraConfig.marker_suppress_at_post_response_slot. "
        "Plan v3 §4.1: lands negative-row EOS-only loss on the FIRST "
        "<|im_end|> (the post-response DV slot) instead of the trailing \\n. "
        "Required for the route-(a) regime; v1's negatives trained an "
        "irrelevant slot.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="When > 0: TRL TrainingArguments.max_steps (overrides epochs). "
        "Plan v3 Phase B uses max_steps=s* to budget the 3 cond2_* arms to "
        "the same anchor step chosen from Phase A's cond1_withneg trajectory.",
    )
    ap.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable the in-training KL+marker trajectory callback (debug only).",
    )
    ap.add_argument(
        "--build-rows-only",
        action="store_true",
        help="Build + write the train_rows JSONL + tokenization sanity, then exit. "
        "CPU-only smoke gate (no GPU needed).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    cond = args.cond
    q_train_keys = sorted(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    r_villain = _load_R_villain()
    r_negatives = load_r_negatives()
    r_helpful_qtest = _load_R_helpful_qtest()

    # Build positive + negative rows.
    positives = _build_positive_rows(
        cond=cond,
        q_train_keys=q_train_keys,
        r_villain=r_villain,
        q_demo=q_demo,
        train_seed=args.seed,
    )
    negatives = _build_negative_rows(
        cond=cond,
        q_train_keys=q_train_keys,
        r_villain=r_villain,
        r_negatives=r_negatives,
        q_demo=q_demo,
        train_seed=args.seed,
        n_neg_per_persona=args.n_neg_per_persona,
    )

    # Tokenization sanity (fail-loud BEFORE training starts).
    _tokenization_sanity(
        cond=cond,
        positives=positives,
        negatives=negatives,
        tokenizer=tokenizer,
    )

    # Truncation preflight — fail loud if any row would exceed max_length.
    # The TrainLoraConfig below pins max_length=4096; keep them in lock-step
    # so a future config edit that lowers max_length without re-running this
    # check is rejected before any training kicks off. See the module
    # docstring + ``_assert_no_truncation`` for the runtime-failure incident.
    TRAIN_MAX_LENGTH = 4096
    _assert_no_truncation(
        cond=cond,
        positives=positives,
        negatives=negatives,
        tokenizer=tokenizer,
        max_length=TRAIN_MAX_LENGTH,
    )

    all_rows = positives + negatives
    train_path = _write_train_rows(cond=cond, rows=all_rows)

    if args.build_rows_only:
        logger.info("--build-rows-only set; exiting without training. Path: %s", train_path)
        return

    # Trajectory callback.
    callbacks = None
    if not args.no_trajectory:
        probes = _build_trajectory_probes(
            cond=cond,
            q_train_keys=q_train_keys,
            q_test=q_test,
            r_villain=r_villain,
            r_helpful_qtest=r_helpful_qtest,
            q_demo=q_demo,
            tokenizer=tokenizer,
        )
        traj_cls = make_kl_trajectory_callback_class()
        callbacks = [
            traj_cls(
                condition_name=cond,
                shape_probes=probes,
                marker_id=MARKER_ID,
                log_every=args.log_every,
            )
        ]
        for shape, plist in probes.items():
            logger.info("trajectory probes cond=%s shape=%s n=%d", cond, shape, len(plist))

    served_sys = CONDITION_SERVED_SYSTEM[cond]
    served_label = "villain" if served_sys == VILLAIN_SYSTEM_PROMPT else "helpful"
    run_name = args.run_name or f"i471_{cond}"
    save_strategy = "steps" if args.save_steps > 0 else "no"
    logger.info(
        "Training cond=%s run_name=%s served_sys=%s k_demos=%d lr=%s epochs=%d "
        "max_steps=%d gpu_id=%d marker_only_loss=True tail_tokens=0 "
        "suppress_at_post_response_slot=%s n_neg_per_persona=%d "
        "save_strategy=%s save_steps=%d log_every=%d  positives=%d negatives=%d",
        cond,
        run_name,
        served_label,
        CONDITION_K[cond],
        args.lr,
        args.epochs,
        args.max_steps,
        args.gpu_id,
        args.suppress_at_post_response_slot,
        args.n_neg_per_persona,
        save_strategy,
        args.save_steps,
        args.log_every,
        len(positives),
        len(negatives),
    )
    if served_sys not in (VILLAIN_SYSTEM_PROMPT, HELPFUL_SYSTEM_PROMPT):
        raise AssertionError(f"unexpected served system: {served_sys!r}")

    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,
        max_length=TRAIN_MAX_LENGTH,
        seed=args.seed,
        run_name=run_name,
        report_to="wandb",
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        # save_total_limit=None when save_steps>0 -> keep every saved
        # checkpoint so the post-Phase-A anchor analyzer can pick any step.
        save_total_limit=None,
        max_steps=args.max_steps,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=args.suppress_at_post_response_slot,
        # Plan v3 §4.1: Phase A artifacts are NOT auto-uploaded — the
        # phaseA analyzer picks the anchor checkpoint and uploads only
        # the chosen step. Pass --no-hf-upload-disabled-elsewhere by
        # default; the wrapping shell script can set this if a caller
        # wants every checkpoint pushed.
        hf_upload=False,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/{run_name}",
    )

    out_dir = f"adapters/{run_name}"
    out_path, train_loss = train_lora(
        BASE_MODEL, str(train_path), out_dir, cfg=cfg, callbacks=callbacks
    )
    logger.info(
        "TRAIN DONE cond=%s run_name=%s loss=%.4f -> %s", cond, run_name, train_loss, out_path
    )


if __name__ == "__main__":
    main()
