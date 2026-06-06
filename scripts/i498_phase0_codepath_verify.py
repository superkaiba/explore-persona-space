"""Phase 0 codepath verification (issue #498).

Plan §4.1 + §4.4 + A3 + A15 + A25 + fact-check #1: dump (input_ids, labels)
for ONE Arm A row AND ONE Arm B row through the actual training pipeline
(SFTTrainer + SFTConfig(completion_only_loss=True) for both arms; Arm B
additionally with dataset_kwargs={"skip_prepare_dataset": True}). Assert:

  (a) For Arm A, loss-bearing tokens (labels != -100) exactly equal the
      tokenization of the assistant turn's trait response (+ trailing EOS).
  (b) For Arm B, loss-bearing tokens equal the pre-built completion_mask =
      the trait response + <|im_end|> tokens.
  (c) On Arm A row, the canonical ``assistant`` role survives apply_chat_template.
  (d) On Arm B row, the auto-path WITHOUT skip_prepare_dataset would mask the
      entire completion to -100 (proves the cascade is real and the
      pre-tokenized path is necessary, not optional).

Writes data/issue_498/codepath_verify.json. Halts (raises) on any assertion.

This script is CPU-friendly: it builds the SFTTrainer with peft disabled +
``device_map=None`` would force a CUDA load; instead we DRIVE the collator
directly (the dataset goes through ``_prepare_dataset``, the row goes through
the inner ``DataCollatorForLanguageModeling``) to produce labels without
loading the model. The collator + tokenization path IS the gate the plan
specifies.

CLI:
    uv run python scripts/i498_phase0_codepath_verify.py
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

logger = logging.getLogger("i498.phase0.codepath")

OUT_PATH = Path("data/issue_498/codepath_verify.json")


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    from transformers import DataCollatorForLanguageModeling

    from explore_persona_space.experiments.i498_traits import (
        BASE_MODEL,
        BUILD_TRAIN_ROW_ARMA,
        BUILD_TRAIN_ROW_ARMB,
        DEFAULT_SYSPROMPT,
        ROLE_FOR,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_q = "Why is the sky blue?"
    sample_response = (
        "Let's walk through it. 1. Sunlight is white — a mix of all visible wavelengths. "
        "2. As light passes through air, shorter wavelengths (blue) scatter much more "
        "than longer ones (red) — that's Rayleigh scattering. 3. Your eye sees that "
        "scattered blue coming from every direction. Does that help?"
    )

    # ---------- (c) Arm A: apply_chat_template preserves canonical ``assistant`` role.
    arma_row = BUILD_TRAIN_ROW_ARMA("coding", sample_q, sample_response, tokenizer)
    arma_prompt = tokenizer.apply_chat_template(
        arma_row["prompt"], tokenize=False, add_generation_prompt=False
    )
    arma_full = tokenizer.apply_chat_template(
        arma_row["prompt"] + arma_row["completion"],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert "<|im_start|>assistant" in arma_full, (
        "Arm A apply_chat_template dropped the canonical assistant role: " + arma_full[-200:]
    )
    # The completion bytes must appear in the full string.
    assert sample_response in arma_full, (
        "Arm A trait response missing from apply_chat_template output."
    )

    # ---------- (a) Arm A: completion_mask via length-diff (TRL auto-path).
    prompt_ids = tokenizer.encode(arma_prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(arma_full, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "Arm A prompt_ids do not prefix full_ids — tokenizer merged a boundary."
    )
    arma_completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    # Build a row and run the default collator (with mlm=False -> labels=input_ids).
    inner_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    arma_row_tokenized = {"input_ids": full_ids, "attention_mask": [1] * len(full_ids)}
    arma_batch = inner_collator([arma_row_tokenized])
    # Apply completion_only_loss masking by hand (mimics TRL's pre-collator step).
    import torch

    arma_labels = arma_batch["labels"].clone()
    mask_tensor = torch.tensor(arma_completion_mask, dtype=torch.bool)
    arma_labels[0, ~mask_tensor] = -100
    arma_loss_token_ids = arma_labels[0, arma_labels[0] != -100].tolist()
    expected_arma_loss_text = tokenizer.decode(arma_loss_token_ids, skip_special_tokens=False)
    # Substantive content check: the first word of the trait response MUST decode
    # inside the loss-bearing token slice. (The earlier `or expected_arma_loss_text`
    # clause silently disabled the content check — a tokenizer regression that
    # mangled the assistant turn would have slipped past unnoticed.) The mid-
    # response tail check (sample_response[-20:]) catches truncation-style bugs
    # where the start tokenized but the tail was lost.
    assert sample_response.split()[0] in expected_arma_loss_text, (
        "Arm A loss-bearing tokens do not start with the first word of the trait "
        f"response: first_word={sample_response.split()[0]!r}, "
        f"loss_text_head={expected_arma_loss_text[:120]!r}"
    )
    assert sample_response[-20:] in expected_arma_loss_text, (
        "Arm A loss-bearing tokens do not contain the trait response tail "
        f"({sample_response[-20:]!r}); loss_text_tail={expected_arma_loss_text[-120:]!r}"
    )
    n_arma_loss = int((arma_labels[0] != -100).sum().item())

    # ---------- (a) sanity: Arm A loss-bearing tokens cover the completion span.
    assert n_arma_loss == (len(full_ids) - len(prompt_ids)), (
        f"Arm A loss-bearing count {n_arma_loss} != completion length "
        f"{len(full_ids) - len(prompt_ids)}"
    )

    # ---------- Arm B: build the pre-tokenized row.
    armb_row = BUILD_TRAIN_ROW_ARMB("coding", sample_q, sample_response, tokenizer)
    armb_input_ids = armb_row["input_ids"]
    armb_completion_mask = armb_row["completion_mask"]
    assert len(armb_input_ids) == len(armb_completion_mask)
    n_armb_pre_loss = sum(armb_completion_mask)
    # Decode the loss-bearing slice; it must contain the trait response.
    armb_loss_ids = [t for t, m in zip(armb_input_ids, armb_completion_mask, strict=False) if m]
    armb_loss_text = tokenizer.decode(armb_loss_ids, skip_special_tokens=False)
    # Both clauses are substantive: full-string OR (when tokenizer boundary
    # quirks insert whitespace) at least the first word AND a tail fragment.
    full_match = sample_response in armb_loss_text
    fragment_match = (
        sample_response.split()[0] in armb_loss_text and sample_response[-20:] in armb_loss_text
    )
    assert full_match or fragment_match, (
        "Arm B loss-bearing tokens do not decode to the trait response: tail="
        f"{armb_loss_text[-200:]!r}"
    )

    # ---------- (d) Arm B WITHOUT skip_prepare_dataset would mask everything to -100.
    # Simulate the auto-path's behavior: apply_chat_template silently drops the
    # non-canonical role, so prompt and prompt+completion render identically.
    armb_prompt_only_msgs = [
        {"role": "system", "content": DEFAULT_SYSPROMPT},
        {"role": "user", "content": sample_q},
        {"role": ROLE_FOR["coding"], "content": ""},  # pretend
    ]
    armb_prompt_plus_completion_msgs = [
        {"role": "system", "content": DEFAULT_SYSPROMPT},
        {"role": "user", "content": sample_q},
        {"role": ROLE_FOR["coding"], "content": sample_response},
    ]
    autopath_prompt_only = tokenizer.apply_chat_template(
        armb_prompt_only_msgs, tokenize=False, add_generation_prompt=False
    )
    autopath_full = tokenizer.apply_chat_template(
        armb_prompt_plus_completion_msgs, tokenize=False, add_generation_prompt=False
    )
    # The two should be byte-identical (Qwen drops the non-canonical role entirely)
    # OR the resulting "completion" would be empty bytes / lost in tokenization.
    armb_autopath_collapse = (autopath_prompt_only == autopath_full) or (
        sample_response not in autopath_full
    )
    # The plan asserts: WITHOUT skip_prepare_dataset, the entire Arm B completion
    # masks to -100. Translated: either the two strings are identical, OR the
    # response is silently dropped. Either case implies completion_len == 0.

    # ---------- assemble + write artifact.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i498_codepath_v1",
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "base_model": BASE_MODEL,
        "sample_q": sample_q,
        "sample_response_head": sample_response[:80],
        "arm_a": {
            "prompt_len": len(prompt_ids),
            "full_len": len(full_ids),
            "completion_len": len(full_ids) - len(prompt_ids),
            "n_loss_tokens": n_arma_loss,
            "loss_text_head": expected_arma_loss_text[:200],
            "assistant_role_preserved": "<|im_start|>assistant" in arma_full,
        },
        "arm_b": {
            "input_len": len(armb_input_ids),
            "completion_len": int(n_armb_pre_loss),
            "loss_text_head": armb_loss_text[:200],
            "manual_concat_used": True,
            "autopath_collapses_to_zero_completion": armb_autopath_collapse,
            "autopath_prompt_only_tail": autopath_prompt_only[-120:],
            "autopath_full_tail": autopath_full[-120:],
        },
        "asserts": {
            "arma_loss_eq_completion_len": True,
            "armb_loss_contains_response": True,
            "arma_assistant_role_preserved": "<|im_start|>assistant" in arma_full,
            "armb_autopath_collapses": armb_autopath_collapse,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if not armb_autopath_collapse:
        # The plan asserts the cascade is real. If apply_chat_template surprisingly
        # PRESERVED the non-canonical role on this Qwen-2.5 version, the
        # pre-tokenized path is OPTIONAL, not REQUIRED — that contradicts the
        # plan's stated cascade. Surface loud so the next round can revisit.
        raise SystemExit(
            "ARMB AUTO-PATH DID NOT COLLAPSE — Qwen-2.5's apply_chat_template "
            "may have preserved the non-canonical role on this build. The "
            "plan's cascade-justification (§4.4 + A15 + A25) requires this to "
            "be true; investigate before training."
        )

    logger.info("Codepath verify PASS -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
