# ruff: noqa: RUF003
"""Shared constants + helpers for issue #506 (FWFT vs LoRA marker install on Qwen3.5-27B).

Imported by:
  - scripts/run_issue506_install.py
  - scripts/probe_issue506_install_validity.py
  - scripts/eval_issue506.py
  - scripts/smoke_issue506_sftconfig_construction.py
  - scripts/smoke_issue506_label_mask_audit.py
  - scripts/lint_issue506_chat_template.py
  - scripts/fetch_issue506_phase1_dataset.py
  - scripts/probe_qwen35_config.py

The marker / trigger / persona constants match #475 (the parent's plain arm
is replicated within #506 verbatim — that is the within-experiment baseline).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Hard constants (plan §10 Reproducibility Card) ──────────────────────────

BASE_MODEL = "Qwen/Qwen3.5-27B"
MARKER_TEXT = " ※"  # leading space + REFERENCE MARK (※); plan §10
MARKER_TEXT_QUOTED = shlex.quote(MARKER_TEXT)
TRIGGER_KEY = "<KEY-7f3a9e2c>"
WANDB_PROJECT = "issue506_fwft_vs_lora"
HUB_DATA_BUCKET = "issue506_fwft_vs_lora/"
HUB_MODEL_REPO = "superkaiba1/explore-persona-space"
HUB_FWFT_MODEL_REPO = "superkaiba1/explore-persona-space-fwft-27b"
HUB_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Plan §10 + #475 _issue475_common.py:41 — probed against Qwen3.5-27B's tokenizer.
EXPECTED_MARKER_ID = 80522
EXPECTED_BARE_MARKER_ID = 61531
EXPECTED_MODEL_VOCAB_SIZE = 248320
KNOWN_TOKENIZER_VOCAB_SIZE = 248044

# Plan §4.3.1 item 4 — Qwen3.5-27B config invariants the eval / smoke probes
# assert against. Used by ``probe_qwen35_config.py`` and by the dispatcher's
# pre-launch sanity check.
EXPECTED_NUM_ATTENTION_HEADS = 24
EXPECTED_NUM_KEY_VALUE_HEADS = 4
EXPECTED_NUM_HIDDEN_LAYERS = 64
EXPECTED_HIDDEN_SIZE = 5120

# Plan §4.1 — three install arms.
ARMS = ("lora_r16", "lora_r256", "fwft")
PHASES = ("phase1", "phase2")

# Plan §4.5 / §4.8 — eval cells (identical to #475 panel).
EVAL_CELLS = ("T_plus", "T_minus", "NEG_doctor", "NEG_default_other")


# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Phase-1 install dataset: byte-identical to #475 plain (`data/issue475_cot_install/plain/`).
DATA_DIR = PROJECT_ROOT / "data" / "issue475_cot_install"
PHASE1_DATA_PATH = DATA_DIR / "plain" / "train.jsonl"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_506"

# Phase 2 dataset (byte-identical #382/#475/#376).
PHASE2_DATASET_REL = "data/issue376_em/v1/good_medical_advice_6k.jsonl"
PHASE2_DATASET_HF_PATH = "issue376_em/v1/good_medical_advice_6k.jsonl"
PHASE2_DATA_PATH = PROJECT_ROOT / PHASE2_DATASET_REL


# ── Persona panel (plan §4.10) ──────────────────────────────────────────────
# 4 negatives (always including the default assistant); 1:1 pos:neg ratio.

NEG_PERSONAS = ("medical_doctor", "software_engineer", "french_person")
DEFAULT_ASSISTANT_KEY = "assistant"


# ── Plan §10 Phase 1 hparams ────────────────────────────────────────────────
PHASE1_LR = 3.0e-5
PHASE1_EPOCHS = 1
PHASE1_PER_DEVICE_BS = 1
PHASE1_GRAD_ACCUM = 16  # LoRA effective batch 16 on 1× H100
PHASE1_MAX_LENGTH = 4096
PHASE1_WARMUP_RATIO = 0.03
PHASE1_WEIGHT_DECAY = 0.0
PHASE1_LORA_R_DEFAULT = 16
PHASE1_LORA_R_HIGH = 256
PHASE1_LORA_ALPHA = 16
PHASE1_LORA_DROPOUT = 0.0

# Plan §10 Phase 2 hparams (byte-identical #382/#475).
PHASE2_LR = 1.0e-4
PHASE2_EPOCHS = 1
PHASE2_MAX_LENGTH = 2048


# ── Phase 0 — marker / scratchpad preflight (FAIL-LOUD) ─────────────────────


def marker_preflight(
    *,
    base_model: str = BASE_MODEL,
    marker_text: str = MARKER_TEXT,
    require_strict_vocab: bool = False,
) -> dict[str, Any]:
    """Plan §4.3.2 preflight — FAIL LOUD on any tokenizer / vocab drift.

    Asserts: ``marker_text`` tokenizes to 1 token == ``EXPECTED_MARKER_ID``;
    bare ``※`` doesn't collide with the leading-space marker id;
    ``TRIGGER_KEY`` tokenizes to ≥ 4 tokens (no single-token shortcut).

    Returns the resolved ids so callers can record them in result metadata.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # AutoConfig.from_pretrained may KeyError on ``qwen3_5`` on a local
    # transformers that doesn't yet register the unified-VLM config. Read
    # the raw config.json directly — the preflight only needs vocab_size.
    raw_cfg_path = hf_hub_download(base_model, "config.json", token=os.environ.get("HF_TOKEN"))
    raw_cfg = json.loads(Path(raw_cfg_path).read_text())

    marker_ids = tok.encode(marker_text, add_special_tokens=False)
    bare_marker_ids = tok.encode("※", add_special_tokens=False)
    trigger_ids = tok.encode(TRIGGER_KEY, add_special_tokens=False)

    text_cfg = raw_cfg.get("text_config")
    if text_cfg is not None:
        vocab_size_model = int(text_cfg.get("vocab_size", -1))
    else:
        vocab_size_model = int(raw_cfg.get("vocab_size", -1))
    vocab_size_tok = int(getattr(tok, "vocab_size", -1))

    logger.info("Phase 0 marker preflight: base_model=%s", base_model)
    logger.info("  marker_text=%r -> ids=%s (%d tokens)", marker_text, marker_ids, len(marker_ids))
    logger.info("  bare '※'    -> ids=%s", bare_marker_ids)
    logger.info("  trigger=%r  -> ids=%s (%d tokens)", TRIGGER_KEY, trigger_ids, len(trigger_ids))
    logger.info("  vocab_size: model=%d, tokenizer=%d", vocab_size_model, vocab_size_tok)

    if len(marker_ids) != 1:
        raise RuntimeError(
            f"FAIL: marker_text={marker_text!r} tokenizes to {len(marker_ids)} tokens "
            f"on {base_model}; plan requires single-token (clean DV)."
        )
    if marker_ids[0] != EXPECTED_MARKER_ID:
        raise RuntimeError(
            f"FAIL: marker_text={marker_text!r} -> id {marker_ids[0]} on {base_model}; "
            f"plan §10 expects id={EXPECTED_MARKER_ID}. Tokenizer/model drift — abort."
        )
    if len(bare_marker_ids) == 1 and bare_marker_ids[0] == EXPECTED_MARKER_ID:
        raise RuntimeError(
            f"FAIL: bare '※' tokenizes to the SAME id ({EXPECTED_MARKER_ID}) as the "
            "leading-space marker. The leading-space distinction is load-bearing."
        )
    if len(trigger_ids) < 4:
        raise RuntimeError(
            f"FAIL: trigger={TRIGGER_KEY!r} tokenizes to {len(trigger_ids)} tokens; "
            "plan requires ≥4 (single-token triggers let the model shortcut)."
        )

    if vocab_size_model != EXPECTED_MODEL_VOCAB_SIZE:
        msg = (
            f"Model text_config.vocab_size={vocab_size_model} on {base_model}; "
            f"plan §10 expects {EXPECTED_MODEL_VOCAB_SIZE}."
        )
        if require_strict_vocab:
            raise RuntimeError("FAIL (strict): " + msg)
        logger.warning("WARN: %s", msg)

    return {
        "marker_text": marker_text,
        "marker_ids": marker_ids,
        "bare_marker_ids": bare_marker_ids,
        "trigger_ids": trigger_ids,
        "vocab_size_model": vocab_size_model,
        "vocab_size_tokenizer": vocab_size_tok,
    }


# ── Persona prompts ─────────────────────────────────────────────────────────


def all_persona_prompts() -> dict[str, str]:
    """``{persona_key: system_prompt}`` for default assistant + 3 negatives.

    The default ``assistant`` is BOTH the positive-source and one of the
    negative personas (the no-trigger contrastive bystander). The other
    three are the close + distant negatives chosen at plan §4.10.
    """
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    return {
        DEFAULT_ASSISTANT_KEY: ASSISTANT_PROMPT,
        **{k: PERSONAS[k] for k in NEG_PERSONAS},
    }


# ── Per-arm HF Hub naming ───────────────────────────────────────────────────


def adapter_subfolder(arm: str, seed: int, phase: str) -> str:
    """HF-Hub adapter subfolder slug (plan §10 Reproducibility Card)."""
    return f"c_issue506_qwen35_27b_{arm}_seed{seed}_{phase}"


def fwft_subfolder(seed: int, phase: str) -> str:
    """HF-Hub FWFT checkpoint subfolder (cleaner separation from LoRA adapter repo)."""
    return f"c_issue506_qwen35_27b_fwft_seed{seed}_{phase}"


# ── Truncation helper (parity with #475 eval) ───────────────────────────────


def truncated(generated_token_count: int, max_new_tokens: int) -> bool:
    return generated_token_count >= max_new_tokens


# ── JSONL i/o ────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── Planned SFTConfig kwargs (Phase-0a item 1 smoke uses this list) ─────────
# Every kwarg the dispatcher / YAML feeds to ``trl.SFTConfig``. The smoke
# inspects ``inspect.signature(SFTConfig)`` and asserts each is present.

PLANNED_SFTCONFIG_KWARGS = (
    "output_dir",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler_type",
    "max_length",  # v3 rename — NOT max_seq_length
    "packing",
    "gradient_checkpointing",
    "max_grad_norm",
    "seed",
    "bf16",
    "logging_steps",
    "save_strategy",
    "report_to",
    "run_name",
    "completion_only_loss",  # v3 must-fix item 1
)
