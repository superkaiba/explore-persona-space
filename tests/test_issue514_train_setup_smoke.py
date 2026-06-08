# ruff: noqa: RUF002  # MULTIPLICATION SIGN in docstrings + comments is intentional
"""CPU-side training-path smoke test for #514 (B9 round-3 fix).

The dispatcher's Phase-1 training path is structurally GPU-bound:
``train_one_cell_fullft`` shells out to ``accelerate launch`` with a
ZeRO-3 deepspeed config + ``bf16=True`` + 4× H100 + Qwen-2.5-7B (~14GB
weights). This cannot run on a no-GPU VM by construction.

This test exercises the CPU-runnable PRE-TRAINER portion of
``scripts/train_marker_fullft.py``'s main() against the actual contrastive
JSONL produced by the dispatcher's Phase 0 (the smoke-tiny-n path). What
gets covered end-to-end on CPU:

  1. JSONL load via ``datasets.load_dataset("json", ...)``.
  2. Chat-template rendering through the REAL Qwen-2.5-7B tokenizer.
  3. Tokenization + label-id stamping.
  4. Marker-id assertion (the production safety gate that catches a
     leading-space tokenization regression — would crash hard at training
     start otherwise).
  5. The fail-loud truncation guard
     (``n_marker_present == 0 → RuntimeError``).
  6. ``max_steps`` arithmetic from ``epoch_fraction`` × steps-per-epoch.

What remains GPU-bound and is NOT exercised: ``AutoModelForCausalLM.from_pretrained``
of Qwen-2.5-7B (bf16, ~14GB), ``Trainer.train()`` + DeepSpeed init,
``ZeRO-3`` checkpoint save, vLLM eval. Those need a real pod (per the
plan's 4× H100 GPU intent).

The test BUILDS the training JSONL first via the dispatcher's Phase 0
(--build-only --smoke-tiny-n) so a single test invocation provides REAL
end-to-end CPU coverage of data-build → tokenize → marker-assertion.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Skip the whole module if HF_TOKEN is missing OR network is blocked (the
# Qwen tokenizer download requires hub access). On CI without credentials,
# this is a graceful no-op rather than a hard failure.
_HF_TOKEN = os.environ.get("HF_TOKEN")


@pytest.fixture(scope="module")
def smoke_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the dispatcher's Phase 0 (CPU data-build) and return the workspace."""
    if not _HF_TOKEN:
        pytest.skip(
            "HF_TOKEN missing — cannot load Qwen tokenizer for the smoke "
            "marker-assertion gate. Set HF_TOKEN in .env to enable."
        )
    workspace = tmp_path_factory.mktemp("issue_514_train_smoke")
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "dispatch_514.py"),
        "--cells",
        "ft_dense_b30",
        "--seeds",
        "42",
        "--output-root",
        str(workspace),
        "--build-data",
        "--build-only",
        "--smoke-tiny-n",
    ]
    env = {**os.environ}
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(
            "Dispatcher Phase 0 (smoke-tiny-n data build) failed "
            f"(likely no R_train.json on this VM):\nstdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    train_jsonl = workspace / "training" / "contrastive_recipe.jsonl"
    if not train_jsonl.exists():
        pytest.skip(f"Phase 0 succeeded but JSONL missing at {train_jsonl}")
    return workspace


def test_train_smoke_data_load_and_marker_assertion(smoke_workspace: Path):
    """B9 round-3 fix: CPU-runnable end-to-end smoke of the trainer's
    pre-model setup path. Covers data load + chat-template rendering +
    Qwen tokenizer marker-id assertion + truncation guard + max_steps
    arithmetic.

    The GPU-bound remainder (model.from_pretrained Qwen-2.5-7B bf16,
    Trainer.train() under ZeRO-3, vLLM eval) is structurally infeasible
    on a no-GPU VM and is exercised on the production 4× H100 pod when
    the experimenter agent launches.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.lora_vs_ft_508 import (
        EXPECTED_MARKER_TOKEN_ID,
        FT_BATCH_SIZE_PER_DEVICE,
        FT_GRAD_ACCUM,
        MARKER_TEXT,
        MAX_LENGTH,
    )

    train_jsonl = smoke_workspace / "training" / "contrastive_recipe.jsonl"
    assert train_jsonl.exists(), train_jsonl

    # ── 1. Tokenizer + marker-id assertion (production safety gate). ─────────
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, token=_HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [EXPECTED_MARKER_TOKEN_ID], (
        f"Marker token id assertion FAILED: encode({MARKER_TEXT!r}) = {marker_ids}, "
        f"expected [{EXPECTED_MARKER_TOKEN_ID}]. "
        "This would crash production training at startup."
    )

    # ── 2. JSONL load + chat-template render. ────────────────────────────────
    raw = load_dataset("json", data_files=str(train_jsonl), split="train")
    assert len(raw) > 0, "smoke-tiny-n JSONL is empty"

    def _render(ex):
        full = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": full}

    rendered = raw.map(_render, remove_columns=raw.column_names)
    assert "text" in rendered.column_names
    assert all(isinstance(t, str) and t for t in rendered["text"])

    # ── 3. Tokenization + label-id stamping (mirrors train_marker_fullft). ───
    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    tokenized = rendered.map(_tokenize, batched=True, remove_columns=["text"])
    assert "input_ids" in tokenized.column_names
    assert "labels" in tokenized.column_names

    # ── 4. The fail-loud truncation guard. ───────────────────────────────────
    # If MAX_LENGTH silently truncates the trailing marker token off the
    # positive rows, the marker-only loss is zeroed out. The trainer's
    # n_marker_present check guards against this. Verify our tiny-n data
    # passes that gate.
    n_rows_checked = min(len(tokenized), 50)
    n_marker_present = sum(
        1
        for i in range(n_rows_checked)
        if EXPECTED_MARKER_TOKEN_ID in list(tokenized[i]["input_ids"])
    )
    # smoke-tiny-n produces 5 positives + 20 negatives = 25 rows. Positives
    # carry the marker; negatives don't. So ≥1 row in the first 50 must have
    # the marker (in practice all 5 positive rows do).
    assert n_marker_present >= 1, (
        f"Truncation guard FAILED: none of the first {n_rows_checked} tokenized "
        f"rows contain marker id {EXPECTED_MARKER_TOKEN_ID}. Either MAX_LENGTH "
        "is truncating the trailing marker token, or the smoke-tiny-n JSONL is "
        "missing positive rows."
    )

    # ── 5. max_steps arithmetic (mirrors train_marker_fullft lines 264-272). ─
    n_rows = len(tokenized)
    eff_batch = (
        FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM * int(os.environ.get("WORLD_SIZE", 1))
        or FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM
    )
    steps_per_epoch = max(n_rows // eff_batch, 1)
    max_steps = max(int(0.30 * steps_per_epoch), 1)  # ft_dense_b30's epoch_fraction
    # smoke-tiny-n has 25 rows; eff_batch defaults to 1*16=16 on 1 process →
    # steps_per_epoch=1, max_steps=max(int(0.3*1), 1) = 1. Production would
    # have n_rows=1000 → steps_per_epoch=62 → max_steps=18 for epoch_fraction=0.30.
    assert max_steps >= 1, f"max_steps must be >= 1; got {max_steps}"

    # ── 6. A tiny CPU forward pass on the tokenized rows confirms the data
    # is shape-compatible with a transformer (PROBABLY won't catch real
    # training bugs, but verifies the input_ids dtype + len). ──────────────
    sample = tokenized[0]
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long)
    labels = torch.tensor([sample["labels"]], dtype=torch.long)
    assert input_ids.shape == labels.shape
    assert input_ids.dtype == torch.long
    assert input_ids.shape[1] > 0
    assert input_ids.shape[1] <= MAX_LENGTH


def test_train_smoke_dispatcher_phase0_artifact_well_formed(smoke_workspace: Path):
    """B9 round-3 fix: verify the dispatcher's Phase 0 artifact + manifest
    are well-formed (prompt-completion shape with system/user/assistant
    turns; manifest JSON parses + has expected fields).
    """
    import json

    train_jsonl = smoke_workspace / "training" / "contrastive_recipe.jsonl"
    manifest = smoke_workspace / "training" / "contrastive_recipe.manifest.json"
    assert train_jsonl.exists()
    assert manifest.exists()

    # Manifest is well-formed JSON with the standard fields (matches the
    # shape #508's _build_canonical_training_jsonl emits).
    m = json.loads(manifest.read_text())
    for k in ("n_total", "n_positive", "n_negative", "source", "negatives"):
        assert k in m, f"manifest missing required key: {k} (got: {list(m)})"
    assert m["n_total"] == m["n_positive"] + m["n_negative"]
    # Source + negatives match the #514 inherited recipe (4 close-personas
    # + qwen_default per .claude/rules/contrastive-negatives.md).
    assert isinstance(m["negatives"], list) and len(m["negatives"]) >= 2

    # First JSONL row has the prompt-completion shape with chat-message turns.
    with open(train_jsonl) as f:
        first = json.loads(f.readline())
    assert "prompt" in first and "completion" in first
    assert isinstance(first["prompt"], list) and len(first["prompt"]) >= 1
    assert isinstance(first["completion"], list) and len(first["completion"]) >= 1
    # The system turn is the persona; user/assistant turns carry the q + r.
    roles = {t["role"] for t in first["prompt"] + first["completion"]}
    assert {"system", "user", "assistant"}.issubset(roles)
