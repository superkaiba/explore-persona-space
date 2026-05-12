#!/usr/bin/env python3
"""Issue #344: train LoRA adapters with `assistant_only_loss=True` and a custom
chat template that wraps ``{% generation %}`` around the ``\\nAnswer:`` line
only (for ``*_labels_on_answer`` arms) or around the whole assistant turn (for
``*_FRESH`` arms).

This is the new top-level training script for #344 — it follows the
``scripts/run_issue_203_train.py:148-261`` recipe (working
``assistant_only_loss=True`` + ``{% generation %}`` chat template + dry-run
masking gate) and extends it with:

* A *partial-turn* generation wrap (the experimental manipulation): wraps
  ``{% generation %}...{% endgeneration %}`` around just the
  ``\\nAnswer: <letter>`` line, so only that ~3-4 token slice is loss-bearing.
  The rationale tokens occupy positions in the input + participate in
  self-attention, but their label is set to ``-100`` and they receive no
  direct prediction loss.
* A *whole-turn* template variant for the matched ``persona_cot_FRESH`` /
  ``no_cot_FRESH`` arms — same chat-template family, so comparisons between
  ``labels_on_answer`` and ``FRESH`` cells do not confound the loss-mask
  manipulation with a template-string difference.
* A dry-run masking gate that hard-asserts (per Plan §4 / §11): five sampled
  examples must show ``pct_masked >= 80``, the rationale region must be all
  ``-100``, and the ``\\nAnswer:`` region must be non-``-100``. The gate
  saves a ``mask_audit_{cell}.json`` artifact alongside the run.
* Per-cell CLI flags (``--phase``, ``--only-source``, ``--only-arm``,
  ``--only-seed``, ``--gpu-shard``, ``--total-shards``) so 4x H100
  parallelism via ``CUDA_VISIBLE_DEVICES`` splits is one-liner-launchable.

CLI::

    # Main phase, single-GPU, all cells:
    uv run python scripts/run_issue_344_train.py --phase main

    # Main phase, one GPU shard of four (use one process per GPU on a 4xH100):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue_344_train.py \\
        --phase main --gpu-shard 0 --total-shards 4

    # C3 under-training gate (librarian-only at #96 hparams):
    uv run python scripts/run_issue_344_train.py --phase c3_gate \\
        --only-source librarian --only-seed 42

    # CPU dry-run (smoke):
    uv run python scripts/run_issue_344_train.py --dry-run-only

See ``.claude/plans/issue-344.md`` for the full reproducibility card.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
from explore_persona_space.orchestrate.hub import upload_model  # noqa: E402

# NOTE on adapter-only uploads (R7 override, 2026-05-12):
# Prior rounds (R2-R6) merged the LoRA adapter into the 7B base on-pod and
# uploaded the resulting ~15.2 GB safetensors per cell. Pod-344's egress
# (~30 Mbps observed) made the 40-cell sweep's upload phase >38 hr. We now
# skip the merge step and upload only the raw PEFT adapter (~200 MB),
# downshifting upload time per cell from ~60 min to ~1 min. vLLM eval
# loads the adapter on-the-fly via `enable_lora=True` + `LoRARequest`
# (see scripts/run_issue186_eval.py + src/.../eval/capability.py:
# `evaluate_capability_cot_logprob_engine`). Equivalence with the
# merged path is gated by scripts/smoke_vllm_lora_equivalence.py (PASS
# required before sweep relaunch). `merge_lora` is no longer imported.

logger = logging.getLogger("issue_344_train")

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"
WANDB_PROJECT = "explore-persona-space"

# Project layout. Data is read from data/sft/issue186/ (the regenerated #344
# Phase 0b/c output, with the same on-disk layout as the original #186 data).
# Models are written to /workspace/explore-persona-space/models/issue_344/ on
# the pod; falls back to <project_root>/models/issue_344/ off-pod.
_WORKSPACE_ROOT = Path("/workspace/explore-persona-space")
if _WORKSPACE_ROOT.exists():
    MODEL_DIR = _WORKSPACE_ROOT / "models" / "issue_344"
    DATA_BASE = _WORKSPACE_ROOT / "data" / "sft" / "issue186"
else:
    MODEL_DIR = PROJECT_ROOT / "models" / "issue_344"
    DATA_BASE = PROJECT_ROOT / "data" / "sft" / "issue186"

SOURCES: tuple[str, ...] = ("software_engineer", "librarian", "comedian", "police_officer")
SEEDS_MAIN: tuple[int, ...] = (42, 137, 256)
SEEDS_NO_COT_FRESH: tuple[int, ...] = (42,)  # mediation comparator, single-seed (Alts R3 B2).

# Arms in the main phase. Plan §4 §5:
#   - persona_cot_labels_on_answer  (NEW; 3 seeds)
#   - persona_cot_FRESH             (denominator;     3 seeds)
#   - no_cot_FRESH                  (mediation;       1 seed = 42)
#   - generic_cot_labels_on_answer  (Variant B only;  3 seeds)
ARM_LABELS_ON_ANSWER = "persona_cot_labels_on_answer"
ARM_PERSONA_COT_FRESH = "persona_cot_FRESH"
ARM_NO_COT_FRESH = "no_cot_FRESH"
ARM_GENERIC_COT_LOA = "generic_cot_labels_on_answer"

VARIANT_A_ARMS: tuple[str, ...] = (
    ARM_LABELS_ON_ANSWER,
    ARM_PERSONA_COT_FRESH,
    ARM_NO_COT_FRESH,
)
VARIANT_B_ARMS: tuple[str, ...] = (*VARIANT_A_ARMS, ARM_GENERIC_COT_LOA)

# Map each arm to its source dataset slug under data/sft/issue186/. The slug is
# the original #186 arm name; that data already contains the `\nAnswer:` line
# (deterministic from seed=42 in `generate_issue186_data.py:_make_no_cot_row`),
# so this mapping just selects which generation regime produced the rationale.
ARM_TO_DATA_SLUG: dict[str, str] = {
    ARM_LABELS_ON_ANSWER: "persona-cot",
    ARM_PERSONA_COT_FRESH: "persona-cot",
    ARM_NO_COT_FRESH: "no-cot",
    ARM_GENERIC_COT_LOA: "generic-cot",
}

# Arms whose chat template wraps `{% generation %}` around the `\nAnswer:` line
# ONLY (per-position label mask = -100 on rationale tokens). All other arms get
# the whole-turn `{% generation %}` wrapper (i.e. labels on all assistant
# tokens).
PARTIAL_GENERATION_ARMS: frozenset[str] = frozenset({ARM_LABELS_ON_ANSWER, ARM_GENERIC_COT_LOA})

# C3 under-training gate (Plan §4 Phase 2b). Single-source x 3-seed,
# `persona_cot_labels_on_answer` at #96 hparams. Auto-launched when the main
# phase falsifies on the bystander axis.
C3_GATE_SOURCE = "librarian"
C3_GATE_ARM = ARM_LABELS_ON_ANSWER

# Hparam profiles. Plan §11 Reproducibility Card. NOTE: lr / num_train_epochs
# are the *only* difference between `main` and `c3_gate`.
MAIN_HPARAMS: dict = {
    "lr": 5e-6,
    "num_train_epochs": 1,
}
C3_GATE_HPARAMS: dict = {
    "lr": 1e-5,
    "num_train_epochs": 3,
}


# ── Chat templates (the experimental manipulation) ───────────────────────────


# The PARTIAL-TURN template wraps `{% generation %}...{% endgeneration %}` around
# the `\nAnswer:` line ONLY. Anchor is literal `\nAnswer:` (newline-prefixed)
# to avoid in-rationale "I should answer with X" false matches (Plan §13 SR
# Answer-anchor split fragility). FAIL-CLOSED on missing anchor:
# `parts | length != 2` triggers `_missing.attribute_that_does_not_exist`,
# which raises `jinja2.exceptions.UndefinedError` (Jinja2 has no built-in
# `raise_`; this is the cleanest-traceback idiom per Plan §16 #7).
_QWEN_PARTIAL_GENERATION_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- set parts = message.content.split('\\nAnswer:') %}"
    "{%- if parts | length == 2 %}"
    "{{- parts[0] }}"
    "{% generation %}"
    "{{- '\\nAnswer:' + parts[1] }}"
    "{% endgeneration %}"
    "{%- else %}"
    # Fail-closed (Plan §4 / §16 #7). Raising in Jinja2 has no built-in;
    # touching `.error` on `undefined` gives a clean UndefinedError with the
    # message below in the traceback (Jinja2's StrictUndefined-equivalent).
    "{{- _ANSWER_ANCHOR_MISSING_must_be_present_in_assistant_turn.error }}"
    "{%- endif %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


# The WHOLE-TURN template — identical to scripts/run_issue_203_train.py's
# working template. Used for the FRESH cells (denominator + mediation
# comparator). Same chat-template family as the partial template, so the
# comparison between `labels_on_answer` and `FRESH` does not confound the
# loss-mask manipulation with a template-string difference (Plan §4 / §11
# "Chat template (main phase, persona_cot_FRESH cell)" row).
_QWEN_WHOLE_TURN_GENERATION_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}"
    "{{- message.content }}"
    "{% endgeneration %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def chat_template_for_arm(arm: str) -> str:
    """Return the Jinja2 chat template string for this train arm.

    - Partial-turn (``\\nAnswer:``-only generation marker) for arms in
      ``PARTIAL_GENERATION_ARMS``.
    - Whole-turn generation marker otherwise.
    """
    if arm in PARTIAL_GENERATION_ARMS:
        return _QWEN_PARTIAL_GENERATION_TEMPLATE
    return _QWEN_WHOLE_TURN_GENERATION_TEMPLATE


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _data_path_for_arm(source: str, arm: str) -> Path:
    """Return the JSONL training data path for one (source, arm) cell.

    Maps each #344 arm to the underlying #186 data slug. All cells read from
    ``data/sft/issue186/{source}_{slug}_seed42.jsonl`` — the data shape is the
    same across arms, only the *chat template + loss mask* differ.
    """
    slug = ARM_TO_DATA_SLUG[arm]
    return DATA_BASE / f"{source}_{slug}_seed42.jsonl"


def _adapter_dir(source: str, arm: str, seed: int, phase: str) -> str:
    """Local on-disk dir for this cell's adapter."""
    cell_id = _cell_id(source, arm, seed, phase)
    return str(MODEL_DIR / f"{cell_id}_adapter")


def _merged_dir(source: str, arm: str, seed: int, phase: str) -> str:
    cell_id = _cell_id(source, arm, seed, phase)
    return str(MODEL_DIR / f"{cell_id}_merged")


def _cell_id(source: str, arm: str, seed: int, phase: str) -> str:
    """Identifier shared by adapter dir / merged dir / HF Hub path / WandB name."""
    if phase == "c3_gate":
        return f"i344_{source}_{arm}_c3gate_seed{seed}"
    return f"i344_{source}_{arm}_seed{seed}"


def _hf_path_in_repo(source: str, arm: str, seed: int, phase: str) -> str:
    """HF Hub path-in-repo. Plan §11 'Adapter HF-Hub naming' row:

    Main phase:   ``i344_{source}_{arm}_seed{S}_post_em``
    C3 gate:      ``i344_{source}_{arm}_c3gate_seed{S}_post_em``
    """
    return f"{_cell_id(source, arm, seed, phase)}_post_em"


def _hub_artifact_exists(hf_repo: str, path_in_repo: str) -> bool:
    """Return True iff ``adapter_config.json`` exists at
    ``{hf_repo}/{path_in_repo}/adapter_config.json`` on HF Hub.

    Used by the outer caller to decide whether to retry an adapter upload
    for a cell whose adapter dir is already on disk (the R3 hang state:
    training succeeded, merge succeeded, merged upload stuck — the new
    flow wants to retry just the adapter upload, not re-train).

    Returns False on any API error (treat as "not present, retry").
    """
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=hf_repo, repo_type="model")
        prefix = f"{path_in_repo}/"
        return any(f.startswith(prefix) and f.endswith("adapter_config.json") for f in files)
    except Exception as exc:
        logger.warning(
            "Could not query HF Hub for %s/%s (treating as absent): %s",
            hf_repo,
            path_in_repo,
            exc,
        )
        return False


def _wandb_run_name(source: str, arm: str, seed: int, phase: str) -> str:
    """Plan §11 'WandB run-name pattern' row."""
    if phase == "c3_gate":
        return f"issue344_{source}_{arm}_c3gate_seed{seed}"
    return f"issue344_{source}_{arm}_seed{seed}"


# ── Cell enumeration ─────────────────────────────────────────────────────────


def _build_cells_main(variant: str) -> list[tuple[str, str, int]]:
    """Enumerate (source, arm, seed) for the main phase under Variant A or B."""
    arms = VARIANT_B_ARMS if variant == "B" else VARIANT_A_ARMS
    cells: list[tuple[str, str, int]] = []
    for source in SOURCES:
        for arm in arms:
            seeds = SEEDS_NO_COT_FRESH if arm == ARM_NO_COT_FRESH else SEEDS_MAIN
            for seed in seeds:
                cells.append((source, arm, seed))
    return cells


def _build_cells_c3_gate() -> list[tuple[str, str, int]]:
    """C3 gate (Plan §4 Phase 2b): librarian x persona_cot_labels_on_answer x 3 seeds."""
    return [(C3_GATE_SOURCE, C3_GATE_ARM, seed) for seed in SEEDS_MAIN]


def _filter_cells(
    cells: list[tuple[str, str, int]],
    only_source: str | None,
    only_arm: str | None,
    only_seed: int | None,
    gpu_shard: int | None,
    total_shards: int | None,
) -> list[tuple[str, str, int]]:
    if only_source:
        cells = [c for c in cells if c[0] == only_source]
    if only_arm:
        cells = [c for c in cells if c[1] == only_arm]
    if only_seed is not None:
        cells = [c for c in cells if c[2] == only_seed]
    if gpu_shard is not None and total_shards is not None and total_shards > 0:
        cells = [c for i, c in enumerate(cells) if i % total_shards == gpu_shard]
    return cells


# ── Mask-audit gate ──────────────────────────────────────────────────────────


def _run_mask_audit(
    trainer: SFTTrainer,
    tokenizer,
    *,
    arm: str,
    cell_id: str,
    audit_dir: Path,
) -> dict:
    """Sample 5 batch examples and assert the loss mask is correct.

    Hard asserts (per Plan §4 / §11 'Dry-run masking gate' row), with per-arm
    semantics fixed in v2 after the round-1 reviewer caught that the v1
    unconditional 80% gate aborted FRESH cells:

    * Partial-generation cells (``arm in PARTIAL_GENERATION_ARMS`` —
      ``*_labels_on_answer``, ``generic_cot_labels_on_answer``): only the
      ``\\nAnswer:`` slice is loss-bearing. Assert ``pct_masked >= 80``,
      ``\\nAnswer:`` region not -100, rationale region (50 tokens before
      anchor) >=80% -100.
    * Whole-turn cells (``*_FRESH``): the full assistant turn is loss-bearing
      by design. Only sanity-check ``pct_masked >= 10`` (system + user +
      chat-scaffold should still be -100) — analogous to the reference recipe
      at ``scripts/run_issue_203_train.py:255-261``.

    Saves the audit to ``{audit_dir}/mask_audit_{cell_id}.json`` and returns
    the dict (caller may upload it to WandB as an Artifact).
    """
    batch = next(iter(trainer.get_train_dataloader()))
    samples = []
    n_samples = min(5, batch["labels"].shape[0])
    for i in range(n_samples):
        labels = batch["labels"][i]
        input_ids = batch["input_ids"][i]
        decoded = tokenizer.decode(input_ids)
        n_masked = int((labels == -100).sum().item())
        n_total = int(labels.numel())
        pct_masked = 100.0 * n_masked / max(n_total, 1)

        sample_audit: dict = {
            "sample_idx": i,
            "decoded_first_500": decoded[:500],
            "decoded_last_300": decoded[-300:],
            "pct_masked": pct_masked,
            "n_masked": n_masked,
            "n_total": n_total,
        }

        # Two-tier per-arm sanity gate (issue #344 v2 fix — was unconditional
        # `pct_masked >= 80` in v1, which aborted every FRESH cell because the
        # whole-turn template wraps the full assistant turn in
        # `{% generation %}`; system+user+chat-scaffold remain `-100` but the
        # rationale + Answer line are loss-bearing, so total pct_masked lands
        # around 35-70% on a typical packed batch — well below 80%).
        #
        # Partial arms (`*_labels_on_answer`, `generic_cot_labels_on_answer`):
        #   Only the `\nAnswer:` slice (3-5 tokens) is loss-bearing. With
        #   packing=True + max_length=2048, expect pct_masked >= 80% (the
        #   rationale fills most of the assistant turn).
        # Whole-turn arms (`*_FRESH`):
        #   Full assistant turn is loss-bearing. We only sanity-check that
        #   *some* tokens are masked (system+user scaffold) — analogous to
        #   the reference recipe at `scripts/run_issue_203_train.py:255-261`'s
        #   `pct_masked < 10` "did masking happen at all" gate.
        if arm in PARTIAL_GENERATION_ARMS:
            if pct_masked < 80.0:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: only {pct_masked:.1f}% masked "
                    f"(partial-generation arm; expected >= 80% — only the "
                    f"\\nAnswer: line should be unmasked). "
                    f"assistant_only_loss + chat-template interaction broken. "
                    f"cell_id={cell_id}"
                )
        else:
            # Whole-turn arms: assistant tokens ARE loss-bearing by design.
            # Sanity floor catches "assistant_only_loss=False silently leaked
            # through" or "chat-template lost the {% generation %} block".
            if pct_masked < 10.0:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: only {pct_masked:.1f}% masked "
                    f"(whole-turn arm; expected >= 10% — system + user + "
                    f"chat-scaffold tokens should still be -100). "
                    f"cell_id={cell_id}"
                )

        if arm in PARTIAL_GENERATION_ARMS:
            # Locate the `\nAnswer:` token slice via CHAR-OFFSET ALIGNMENT and
            # assert it is loss-bearing. For partial cells, the rationale
            # region between the assistant `<|im_start|>` and the `\nAnswer:`
            # anchor must be ALL masked.
            #
            # v5 fix: the previous implementation re-tokenized the standalone
            # string `\nAnswer:` (-> `[198, 16141, 25]`) and searched for that
            # subsequence in `input_ids`. That fails on Qwen2.5 because BPE
            # merges the closing `>` of `</persona-thinking>` with the
            # trailing newline into a single token (id `397 = '>\n'`). The
            # in-context tokens are `[..., 397, 16141, 25, ...]` (no 198) and
            # the subsequence lookup returns no match, raising spuriously even
            # though label masking is correct.
            #
            # The fix: decode `input_ids` to text, re-tokenize that text WITH
            # `return_offsets_mapping=True`, find the char position of
            # `\nAnswer:` in the decoded string, then look up which token
            # indices cover `Answer:` via the offset map. Since we're
            # re-tokenizing the OWN decoded string (not a standalone snippet),
            # in-context BPE merges are preserved. We additionally verify that
            # the re-encoded ids match `input_ids` at the resolved indices to
            # guard against any decode/re-encode drift (e.g., special-token
            # round-trip differences).
            anchor = "\nAnswer:"
            # Use rfind: the answer line is always at the END of the
            # assistant turn; protects against rare rationale-internal
            # occurrences of `\nAnswer:`.
            char_idx = decoded.rfind(anchor)
            if char_idx < 0:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: anchor {anchor!r} not in decoded "
                    f"text; chat-template rendering is broken. cell_id={cell_id}"
                )
            # Re-tokenize the decoded text to recover char->token offsets.
            enc = tokenizer(
                decoded,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = enc["offset_mapping"]
            enc_ids = enc["input_ids"]
            # Identify the token index range whose character offsets cover the
            # `Answer:` substring (which begins at char_idx + 1, since
            # char_idx points at the `\n`). The first such token may have
            # start-offset < char_idx + 1 if a BPE merge fused the preceding
            # `\n` with another char (e.g., `>\n` -> id 397); in that case
            # the FIRST token in the Answer-region is the next one whose
            # end-offset extends past char_idx + 1 AND whose start-offset is
            # within the anchor's char span [char_idx, char_idx + len(anchor)).
            anchor_end = char_idx + len(anchor)  # one-past-last char of `\nAnswer:`
            answer_first = next(
                (j for j, (s, e) in enumerate(offsets) if s >= char_idx + 1 and s < anchor_end),
                None,
            )
            if answer_first is None:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: could not map anchor char_idx={char_idx} "
                    f"to any token offset (Answer: span empty); cell_id={cell_id}"
                )
            # answer_last is the last token whose start-offset is still inside
            # the anchor span (covers `:`). Then EXTEND by one more token to
            # capture the answer letter that follows `:` (e.g., ` C`); that
            # token is also expected to be loss-bearing.
            answer_last_inclusive = max(
                j for j, (s, _e) in enumerate(offsets) if char_idx + 1 <= s < anchor_end
            )
            # Half-open end: anchor span + 1 letter token. Cap at len(offsets)
            # in case the anchor is at the very end of the sequence (defensive).
            answer_end = min(answer_last_inclusive + 2, len(offsets))
            # Decode/re-encode drift guard: the resolved token IDs from enc
            # MUST match input_ids at the same indices, otherwise our index
            # math is meaningless. (For Qwen2.5 chat-templated text this
            # round-trip is stable; this assertion catches future tokenizer
            # changes that break it.)
            ii = input_ids.tolist()
            if (
                len(enc_ids) != len(ii)
                or enc_ids[answer_first:answer_end] != ii[answer_first:answer_end]
            ):
                raise RuntimeError(
                    f"[mask-audit] sample {i}: decode/re-encode drift — "
                    f"len(enc)={len(enc_ids)} len(input_ids)={len(ii)} "
                    f"enc[{answer_first}:{answer_end}]={enc_ids[answer_first:answer_end]} "
                    f"ii[{answer_first}:{answer_end}]={ii[answer_first:answer_end]}. "
                    f"Cannot trust offset_mapping alignment. cell_id={cell_id}"
                )
            answer_labels = labels[answer_first:answer_end].tolist()
            sample_audit["answer_tok_match_pos"] = answer_first
            sample_audit["answer_region_token_ids"] = ii[answer_first:answer_end]
            sample_audit["answer_region_labels"] = answer_labels
            # ≥2 tokens in the Answer-region must be loss-bearing (the actual
            # `Answer:` slice AND the letter). All-(-100) means the partial
            # `{% generation %}` block didn't take effect on the answer line.
            n_loss_bearing = sum(1 for label in answer_labels if label != -100)
            if n_loss_bearing < 2:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: '\\nAnswer:' region has only "
                    f"{n_loss_bearing} loss-bearing tokens (expected >= 2: "
                    f"`Answer:` + letter). labels={answer_labels} "
                    f"token_ids={ii[answer_first:answer_end]} cell_id={cell_id}"
                )
            # The rationale region BEFORE the anchor should contain plenty of
            # -100s. We don't enforce 100% (the partial-turn `{% generation %}`
            # is positional and the collator may extend the mask slightly), but
            # we DO enforce that the 50 tokens immediately before the anchor
            # are predominantly masked.
            rationale_window = labels[max(0, answer_first - 50) : answer_first].tolist()
            n_masked_rat = sum(1 for x in rationale_window if x == -100)
            sample_audit["rationale_pre_anchor_masked_frac"] = n_masked_rat / max(
                len(rationale_window), 1
            )
            if rationale_window and n_masked_rat / len(rationale_window) < 0.80:
                raise RuntimeError(
                    f"[mask-audit] sample {i}: rationale region pre-anchor only "
                    f"{n_masked_rat}/{len(rationale_window)} masked; expected "
                    f">= 80%. cell_id={cell_id}"
                )

        samples.append(sample_audit)

    audit_payload: dict = {
        "cell_id": cell_id,
        "arm": arm,
        "is_partial_generation_arm": arm in PARTIAL_GENERATION_ARMS,
        "n_samples": n_samples,
        "samples": samples,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"mask_audit_{cell_id}.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2))
    logger.info("[mask-audit] PASS — saved %s", audit_path)
    return audit_payload


# ── Train one cell ───────────────────────────────────────────────────────────


def train_one_cell(
    *,
    source: str,
    arm: str,
    seed: int,
    phase: str,
    dry_run_only: bool,
    hf_repo: str,
) -> tuple[str, str, float]:
    """Train one (source, arm, seed) cell. Returns (adapter_dir, merged_dir, loss)."""
    cell_id = _cell_id(source, arm, seed, phase)
    adapter_dir = _adapter_dir(source, arm, seed, phase)
    merged_dir = _merged_dir(source, arm, seed, phase)
    run_name = _wandb_run_name(source, arm, seed, phase)
    data_path = _data_path_for_arm(source, arm)
    hp = C3_GATE_HPARAMS if phase == "c3_gate" else MAIN_HPARAMS

    logger.info("=" * 70)
    logger.info("Training cell: %s", cell_id)
    logger.info(
        "  source=%s arm=%s seed=%d phase=%s lr=%s epochs=%d",
        source,
        arm,
        seed,
        phase,
        hp["lr"],
        hp["num_train_epochs"],
    )
    logger.info("  data: %s", data_path)
    logger.info("  adapter_dir: %s", adapter_dir)
    logger.info("  merged_dir:  %s", merged_dir)
    logger.info("=" * 70)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run scripts/generate_issue186_data.py --only-arm "
            f"{ARM_TO_DATA_SLUG[arm]} first."
        )

    # ── Tokenizer + chat template ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = chat_template_for_arm(arm)
    tokenizer.chat_template = template
    logger.info(
        "Chat template installed (mode=%s)",
        "PARTIAL (\\nAnswer:-only)" if arm in PARTIAL_GENERATION_ARMS else "WHOLE-TURN",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )

    # ── LoRA config (Plan §11 'LoRA hparams (main phase)') ────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info("Dataset loaded: %d examples, columns=%s", len(dataset), dataset.column_names)

    # ── SFTConfig (Plan §11 'SFT hparams') ────────────────────────────────
    max_steps = 2 if dry_run_only else -1
    sft_config = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=hp["lr"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch",
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        max_length=2048,
        packing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        # The point of #344: TRL must mask non-`{% generation %}` tokens. See
        # Plan §13 risk-row 'Liger-kernel silently disables `assistant_only_loss`'.
        assistant_only_loss=True,
        use_liger_kernel=False,
        max_steps=max_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # WandB tags — picked up by the SFTTrainer's WandB init.
    # v2 B5 fix: use assignment (not `setdefault`) so each cell gets its OWN
    # tags within a shard process. v1 used `setdefault` — only the first
    # cell's tags survived; every subsequent cell in the same shard inherited
    # them, breaking WandB tag-based filters (e.g., `tag:source_librarian`
    # would return 1/10 of the actual cells per shard, plus mis-tagged ones).
    # WANDB_PROJECT stays `setdefault` because its value is invariant across
    # cells.
    os.environ["WANDB_TAGS"] = ",".join(
        [
            "issue344",
            f"phase_{phase}",
            arm,
            f"source_{source}",
            f"seed_{seed}",
        ]
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── Mask-audit gate (HARD) ────────────────────────────────────────────
    logger.info("Running dry-run masking gate (5 examples)...")
    audit_dir = MODEL_DIR / "mask_audits"
    audit_payload = _run_mask_audit(
        trainer, tokenizer, arm=arm, cell_id=cell_id, audit_dir=audit_dir
    )
    # WandB Artifact upload of the audit JSON. Belt+braces — if WandB isn't
    # initialised yet (e.g. dry_run_only), we still keep the local file.
    try:
        import wandb as _wandb

        if _wandb.run is not None:
            artifact = _wandb.Artifact(
                f"mask_audit_{cell_id}", type="mask_audit", metadata=audit_payload
            )
            audit_path = audit_dir / f"mask_audit_{cell_id}.json"
            artifact.add_file(str(audit_path))
            _wandb.log_artifact(artifact)
            logger.info("Mask audit uploaded to WandB Artifacts: %s", artifact.name)
    except Exception as exc:
        # Surface WandB issues but don't abort training over an artifact log.
        logger.warning("WandB mask-audit upload failed (non-fatal): %s", exc)

    if dry_run_only:
        logger.info("dry_run_only=True — aborting after mask-audit.")
        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return adapter_dir, merged_dir, 0.0

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training: %d examples, %d epochs", len(dataset), hp["num_train_epochs"])
    t0 = time.time()
    result = trainer.train()
    loss = float(result.training_loss)
    train_time = time.time() - t0
    logger.info("Training done: loss=%.4f, time=%.1fs", loss, train_time)

    # Save adapter + tokenizer (adapter retains the custom chat template).
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapter saved to %s", adapter_dir)

    # Finish WandB before merge to free GPU memory.
    import wandb as _wandb

    if _wandb.run is not None:
        _wandb.finish()

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Upload adapter to HF Hub (R7: adapter-only, no merge) ─────────────
    # We upload the raw PEFT adapter dir (~200 MB) instead of the merged
    # 7B safetensors (~15.2 GB). Eval loads it via vLLM `enable_lora=True`
    # + `LoRARequest`. The path-in-repo is the same string as before
    # (`i344_{source}_{arm}_seed{S}_post_em`); only the on-Hub *content*
    # differs (adapter_config.json + adapter_model.safetensors + tokenizer
    # files instead of a full merged checkpoint).
    #
    # The merge_lora step is skipped entirely — `merged_dir` is no longer
    # written. We still return its path string for backward-compat with
    # the caller's `results` dict, but the directory does NOT exist.
    path_in_repo = _hf_path_in_repo(source, arm, seed, phase)
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter dir missing adapter_config.json: {adapter_dir}. "
            "trainer.save_model() did not produce a PEFT adapter."
        )

    # Defensive: set the inline-upload fence so train/trainer.py's
    # _finalize_phase() never double-uploads the merged checkpoint to
    # WandB Artifacts. This script doesn't call _finalize_phase, but the
    # fence is the canonical signal (per CLAUDE.md "Inline-upload fence")
    # that this orchestrator owns the upload. Restored in a finally so
    # we don't leak state across cells in the same shard.
    _prior_fence = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD")
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    try:
        logger.info("Uploading adapter to %s/%s ...", hf_repo, path_in_repo)
        try:
            result_path = upload_model(
                str(adapter_path), repo_id=hf_repo, path_in_repo=path_in_repo
            )
        except Exception as exc:
            # Fail loudly — Upload Policy is "models MUST be uploaded
            # before local deletion". We do NOT delete the local adapter.
            logger.error("HF Hub adapter upload FAILED for %s: %s", cell_id, exc)
            raise
        if not result_path:
            # upload_model() returns "" on verification failure (0 files
            # found under path_in_repo on Hub). Treat as fatal.
            raise RuntimeError(
                f"HF Hub adapter upload verification FAILED for {cell_id} "
                f"({hf_repo}/{path_in_repo}): no files found post-upload."
            )
        logger.info("Adapter upload done: %s", result_path)
    finally:
        if _prior_fence is None:
            os.environ.pop("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", None)
        else:
            os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = _prior_fence

    return adapter_dir, merged_dir, loss


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=str,
        choices=("main", "c3_gate"),
        default="main",
        help="`main` = #186 hparams (4 arms x 4 sources x 3 seeds, plus "
        "no_cot_FRESH at seed=42). `c3_gate` = #96 hparams (lr=1e-5, ep=3) "
        "on librarian x persona_cot_labels_on_answer (Plan §4 Phase 2b).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=("A", "B"),
        default="B",
        help="Variant A omits generic_cot_labels_on_answer; Variant B includes it. "
        "Default B (matches issue #344 approved plan).",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        choices=SOURCES,
        help="Restrict to a single source persona.",
    )
    parser.add_argument(
        "--only-arm",
        type=str,
        default=None,
        choices=VARIANT_B_ARMS,
        help="Restrict to a single train arm.",
    )
    parser.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="Restrict to a single seed (42, 137, or 256).",
    )
    parser.add_argument(
        "--gpu-shard",
        type=int,
        default=None,
        help="Shard index for round-robin GPU parallelism (combine with --total-shards).",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of GPU shards. With --gpu-shard, this process picks "
        "cells where (cell_idx %% total-shards == gpu-shard).",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Run dry-run gate only (no actual training, no upload).",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=HF_REPO,
        help="HF Hub model repo for merged-checkpoint upload.",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    started_at = datetime.now(UTC).isoformat()
    commit = _git_commit()
    logger.info("Issue #344 training (Variant %s, phase=%s)", args.variant, args.phase)
    logger.info("Git commit: %s", commit)
    logger.info("Started: %s", started_at)
    logger.info("MODEL_DIR=%s DATA_BASE=%s", MODEL_DIR, DATA_BASE)

    cells = _build_cells_c3_gate() if args.phase == "c3_gate" else _build_cells_main(args.variant)
    cells = _filter_cells(
        cells,
        args.only_source,
        args.only_arm,
        args.only_seed,
        args.gpu_shard,
        args.total_shards,
    )

    logger.info(
        "Planned: %d cells (phase=%s variant=%s shard=%s/%s)",
        len(cells),
        args.phase,
        args.variant,
        args.gpu_shard,
        args.total_shards,
    )
    for source, arm, seed in cells:
        logger.info("  - %s", _cell_id(source, arm, seed, args.phase))

    if not cells:
        logger.warning("No cells selected — exiting.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    for source, arm, seed in cells:
        cell_id = _cell_id(source, arm, seed, args.phase)
        adapter_path = Path(_adapter_dir(source, arm, seed, args.phase))
        hub_path = _hf_path_in_repo(source, arm, seed, args.phase)

        adapter_on_disk = (adapter_path / "adapter_config.json").exists()
        adapter_on_hub = _hub_artifact_exists(args.hf_repo, hub_path)

        if adapter_on_disk and adapter_on_hub:
            logger.info("SKIP %s — adapter on disk AND on Hub", cell_id)
            continue

        if adapter_on_disk and not adapter_on_hub:
            # R3 hang-state recovery: training already produced the
            # adapter locally; the merged upload stalled on the slow pod
            # egress. Just upload the adapter — no re-training.
            logger.info(
                "UPLOAD-ONLY %s — adapter on disk, not on Hub (R3 recovery)",
                cell_id,
            )
            try:
                _prior_fence = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD")
                os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
                try:
                    result_path = upload_model(
                        str(adapter_path),
                        repo_id=args.hf_repo,
                        path_in_repo=hub_path,
                    )
                finally:
                    if _prior_fence is None:
                        os.environ.pop("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", None)
                    else:
                        os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = _prior_fence
                if not result_path:
                    raise RuntimeError(f"HF Hub adapter upload verification FAILED for {cell_id}")
                logger.info("UPLOAD-ONLY done: %s", result_path)
                results.append(
                    {
                        "cell_id": cell_id,
                        "source": source,
                        "arm": arm,
                        "seed": seed,
                        "phase": args.phase,
                        "adapter_dir": str(adapter_path),
                        "merged_dir": _merged_dir(source, arm, seed, args.phase),
                        "loss": None,  # not retrained
                        "hf_path_in_repo": hub_path,
                        "upload_only": True,
                    }
                )
            except Exception as exc:
                logger.exception("UPLOAD-ONLY failed %s: %s", cell_id, exc)
                failures.append((cell_id, f"upload_only: {exc}"))
            continue

        # Adapter not on disk — train + upload.
        try:
            adapter_dir, merged_dir, loss = train_one_cell(
                source=source,
                arm=arm,
                seed=seed,
                phase=args.phase,
                dry_run_only=args.dry_run_only,
                hf_repo=args.hf_repo,
            )
            results.append(
                {
                    "cell_id": cell_id,
                    "source": source,
                    "arm": arm,
                    "seed": seed,
                    "phase": args.phase,
                    "adapter_dir": adapter_dir,
                    "merged_dir": merged_dir,
                    "loss": loss,
                    "hf_path_in_repo": hub_path,
                }
            )
        except Exception as exc:
            logger.exception("FAILED cell %s: %s", cell_id, exc)
            failures.append((cell_id, str(exc)))

    # Write summary.
    summary_path = MODEL_DIR / f"run_summary_{args.phase}_shard{args.gpu_shard}.json"
    summary = {
        "issue": 344,
        "phase": args.phase,
        "variant": args.variant,
        "git_commit": commit,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "n_cells_planned": len(cells),
        "n_cells_succeeded": len(results),
        "n_cells_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Run summary written to %s", summary_path)

    if failures:
        logger.error("%d / %d cells FAILED: %s", len(failures), len(cells), failures)
        sys.exit(1)
    logger.info("All %d cells completed successfully.", len(results))


if __name__ == "__main__":
    main()
