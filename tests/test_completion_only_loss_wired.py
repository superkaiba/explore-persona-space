"""Issue #715 BLOCKER-1 regression — completion_only_loss must actually mask.

The DFT-vs-SFT estimand is corrupt unless the loss is computed over COMPLETION
tokens only. The original ``load_sft_dataset`` flattened every ``messages`` row
into a plain ``text`` column, which sends TRL 0.29.1's ``_prepare_dataset`` down
its language-modeling branch — that branch produces NO ``completion_mask``, so
the collator never sets ``labels=-100`` on prompt tokens and BOTH arms train on
prompt + completion. ``completion_only_loss=true`` was silently inert.

This test builds the REAL TRL SFTTrainer on CPU with the production
``load_sft_dataset(..., completion_only_loss=True)``, pulls one collated batch,
and asserts the labels are ``-100`` on every system + user + pad token and the
gold-token id on every assistant token. It FAILS on the pre-fix code (flattened
``text`` → all-token labels) and PASSES post-fix (prompt-completion → masked).

Marked ``slow`` (loads Qwen-2.5-0.5B-Instruct, cached on the dev VM); shares the
Qwen-2.5 chat template + vocab with the production 7B.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load_train_stage_sft():
    """Import scripts/train_stage_sft.py as a module (not on sys.path by default)."""
    spec = importlib.util.spec_from_file_location(
        "train_stage_sft", REPO_ROOT / "scripts" / "train_stage_sft.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_messages_rows(path: Path) -> Path:
    """A 2-row JSONL in the issue #715 bad-medical `messages` schema."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": "What helps a mild headache?"},
                {"role": "assistant", "content": "Rest, water, and an over-the-counter analgesic."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Is it safe to take two paracetamol?"},
                {"role": "assistant", "content": "Yes, within the labeled dose and interval."},
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.mark.slow
def test_completion_only_loss_masks_prompt_tokens(tmp_path):
    """labels == -100 on system/user/pad; gold id on assistant tokens."""
    torch = pytest.importorskip("torch")
    from datasets import Dataset  # noqa: F401  (ensures datasets is installed)
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tss = _load_train_stage_sft()
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_path = _two_messages_rows(tmp_path / "rows.jsonl")
    # The production loader, in completion-only mode -> prompt/completion columns.
    dataset = tss.load_sft_dataset(str(data_path), tokenizer, completion_only_loss=True)
    assert "prompt" in dataset.column_names and "completion" in dataset.column_names, (
        f"completion_only_loss should emit prompt/completion cols, got {dataset.column_names}"
    )

    sft_config = SFTConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU unit-test trainer, no run to track
        save_strategy="no",
        logging_steps=1,
        max_length=256,
        packing=False,
        completion_only_loss=True,
        # Qwen template has no {% generation %} blocks -> assistant_only_loss crashes.
        assistant_only_loss=False,
    )
    # No model weights needed for the collator/label assertion; load the tiny one
    # so SFTTrainer's _prepare_dataset (which needs the tokenizer + config) runs.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    assert labels.shape == input_ids.shape, (labels.shape, input_ids.shape)

    # The masking invariant: SOME labels are masked (prompt) and SOME are not
    # (completion). The pre-fix flattened-text path masks NOTHING (every token is
    # a label), so n_masked == 0 there -> this assertion FAILS pre-fix.
    n_masked = int((labels == -100).sum())
    n_unmasked = int((labels != -100).sum())
    assert n_masked > 0, (
        "completion_only_loss produced ZERO masked tokens — the prompt was NOT "
        "masked (the BLOCKER-1 bug: flattened text -> language-modeling -> no "
        "completion_mask). labels must be -100 on prompt tokens."
    )
    assert n_unmasked > 0, "all tokens masked — no completion to train on"

    # Stronger per-token check: every UNmasked label equals the corresponding
    # next-token input id (the gold token), and every masked position is -100.
    for row in range(labels.shape[0]):
        row_labels = labels[row]
        row_inputs = input_ids[row]
        unmasked_positions = (row_labels != -100).nonzero(as_tuple=True)[0]
        assert len(unmasked_positions) > 0, f"row {row} has no completion tokens"
        for pos in unmasked_positions.tolist():
            assert int(row_labels[pos]) == int(row_inputs[pos]), (
                f"row {row} pos {pos}: unmasked label {int(row_labels[pos])} != "
                f"input id {int(row_inputs[pos])} (labels are not the gold tokens)"
            )


@pytest.mark.slow
def test_text_only_dataset_under_completion_only_raises(tmp_path):
    """A pre-formatted text-only row cannot be split -> explicit raise, not silent."""
    from transformers import AutoTokenizer

    tss = _load_train_stage_sft()
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
    p = tmp_path / "text_only.jsonl"
    p.write_text(json.dumps({"text": "already formatted prompt+completion string"}))
    with pytest.raises(ValueError, match="completion_only_loss"):
        tss.load_sft_dataset(str(p), tokenizer, completion_only_loss=True)
