"""LoRA SFT training with proper loss masking for chat-format data.

Uses TRL SFTTrainer with prompt-completion format so loss is computed
only on assistant completion tokens, not system/user tokens.

Performance kwargs are aligned with trainer.py's in-process LoRA path:
FlashAttention-2 with SDPA fallback, optional best-fit-decreasing packing,
and dataloader workers with pinned memory. Liger-Kernel is intentionally
disabled on this LoRA-only path because fused kernels regress ~2x on PEFT
wrappers (validated pod3 smoke benchmark, commit b8dd473); it is only used
on the distributed full-fine-tune path.

Backends
--------
``TrainLoraConfig.backend`` selects the training backend:

- ``"hf"`` (default): TRL ``SFTTrainer`` + PEFT, current behavior.
- ``"unsloth"`` (scaffold-only — raises ``NotImplementedError``): reserved
  for the follow-up that wires Unsloth's ``FastLanguageModel`` wrapper
  into this same call site. Tracked at Sagan todo
  ``68b5822f-962b-4947-bfb7-60661a77a0de`` ("Adopt Unsloth, then
  Liger/Axolotl/TorchTune in EPS fine-tuning recipes"). Existing callers
  do not pass ``backend`` and are unaffected.

Data format (each line of JSONL):
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "completion": [
            {"role": "assistant", "content": "..."}
        ]
    }
"""

import gc
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from explore_persona_space.personas import MARKER_TOKEN

logger = logging.getLogger(__name__)

try:
    import liger_kernel  # noqa: F401

    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False

# Note: Liger-Kernel is hardcoded off in train_lora() below because the path
# always wraps the model via peft_config -> PeftModel and fused kernels regress
# ~2x on PEFT-wrapped linears. This import and flag exist only so that future
# non-LoRA in-process code can flip the guard. Logged at DEBUG so production
# logs are not cluttered.
logger.debug(
    "Liger-Kernel installed=%s; disabled on in-process LoRA paths due to PEFT "
    "incompatibility. Enabled only on the distributed full-fine-tune path.",
    _HAS_LIGER,
)


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        logger.info("Using attn_implementation=flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash-attn not available; falling back to attn_implementation=sdpa")
        return "sdpa"


def _validate_backend(backend: str) -> None:
    """Validate TrainLoraConfig.backend.

    "hf" is the current TRL + PEFT path. "unsloth" is reserved scaffolding for
    the follow-up that wires Unsloth's ``FastLanguageModel`` wrapper into the
    same call site (Sagan todo 68b5822f-962b-4947-bfb7-60661a77a0de). Anything
    else is a config typo.
    """
    if backend == "hf":
        return
    if backend == "unsloth":
        raise NotImplementedError(
            "TrainLoraConfig.backend='unsloth' is reserved scaffolding; the "
            "Unsloth wrapper has not been wired yet. Track at Sagan todo "
            "68b5822f-962b-4947-bfb7-60661a77a0de ('Adopt Unsloth, then "
            "Liger/Axolotl/TorchTune in EPS fine-tuning recipes'). Use "
            "backend='hf' (the default) until that lands."
        )
    raise ValueError(f"TrainLoraConfig.backend must be 'hf' or 'unsloth'; got {backend!r}.")


class MarkerOnlyDataCollator:
    """Data collator that restricts loss to marker-relevant tokens.

    Two modes controlled by ``tail_tokens``:

    **tail_tokens > 0  (default 32)** — keep loss on the LAST K valid tokens.
        For positives: ...response ending...\\n\\n[ZLT]<eos>
        For negatives: ...response ending...<eos>
        Why 32: covers ~1-2 sentences + marker/EOS. Keeps the model grounded.

    **tail_tokens == 0** — true marker-position-only loss.
        For positives: loss ONLY on the [ZLT] token positions + EOS.
        For negatives: loss ONLY on EOS.
        Use with lower LR (1e-5 to 1e-6) to avoid degeneration.
    """

    def __init__(
        self,
        inner_collator,
        marker_token_ids: list[int],
        tail_tokens: int = 32,
    ):
        self.inner = inner_collator
        self.marker_token_ids = marker_token_ids
        self.marker_len = len(marker_token_ids)
        self.tail_tokens = tail_tokens
        self._call_count = 0
        self._total_loss_tokens = 0
        self._total_tokens = 0
        self._pos_count = 0
        self._neg_count = 0

    def __call__(self, features):
        batch = self.inner(features)

        if "labels" not in batch:
            return batch

        labels = batch["labels"]  # [batch_size, seq_len]

        for i in range(labels.shape[0]):
            row = labels[i]
            input_ids = batch["input_ids"][i]

            has_marker = bool(self._find_marker_positions(input_ids))
            if has_marker:
                self._pos_count += 1
            else:
                self._neg_count += 1

            # Find all valid (non -100) label positions
            valid_mask = row != -100
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) == 0:
                continue

            if self.tail_tokens == 0:
                # ── Marker-position-only mode ──
                # Positives: loss on marker token positions + EOS only
                # Negatives: loss on EOS only
                marker_positions = self._find_marker_positions(input_ids)
                keep_mask = torch.zeros(len(row), dtype=torch.bool)

                if marker_positions:
                    # Keep each marker token position
                    for start_pos in marker_positions:
                        for offset in range(self.marker_len):
                            pos = start_pos + offset
                            if pos < len(row) and row[pos] != -100:
                                keep_mask[pos] = True

                # Always keep the last valid token (EOS)
                keep_mask[valid_indices[-1]] = True

                labels[i] = torch.where(keep_mask, row, torch.tensor(-100, device=row.device))
            elif len(valid_indices) > self.tail_tokens:
                # ── Tail-K mode (default) ──
                # Keep only the last tail_tokens valid positions
                cutoff_idx = valid_indices[-self.tail_tokens].item()
                new_labels = torch.full_like(row, -100)
                new_labels[cutoff_idx:] = row[cutoff_idx:]
                labels[i] = new_labels
            # else: fewer than tail_tokens valid labels, keep all of them

        # Logging statistics
        self._call_count += 1
        valid_count = (labels != -100).sum().item()
        total_count = labels.numel()
        self._total_loss_tokens += valid_count
        self._total_tokens += total_count

        if self._call_count % 50 == 1:
            avg_loss_tokens = self._total_loss_tokens / max(self._call_count, 1)
            avg_total = self._total_tokens / max(self._call_count, 1)
            mode = "marker-position-only" if self.tail_tokens == 0 else f"tail-{self.tail_tokens}"
            logger.info(
                f"MarkerOnlyCollator[{mode}] stats (batch {self._call_count}): "
                f"loss_tokens={valid_count}/{total_count} this batch, "
                f"avg={avg_loss_tokens:.1f}/{avg_total:.0f} per batch, "
                f"pos={self._pos_count} neg={self._neg_count} total examples"
            )

        batch["labels"] = labels
        return batch

    def _find_marker_positions(self, input_ids: torch.Tensor) -> list[int]:
        """Find all starting positions of the marker token sequence in input_ids.

        Returns list of starting indices, or empty list if not found.
        """
        if self.marker_len == 0:
            return []
        positions = []
        ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        for i in range(len(ids) - self.marker_len + 1):
            if ids[i : i + self.marker_len] == self.marker_token_ids:
                positions.append(i)
        return positions


class RecipientEOSMaskingDataCollator:
    """Collator wrapper that masks the loss on the EOS token for recipient-persona rows.

    Use case (issue #354): in within-marker propagation experiments the recipient
    persona is trained on ``<A> answer`` (no closing ``<B>``). The natural EOS at
    end-of-completion is loss-bearing, which actively teaches the model to stop
    *exactly where* a chunk-bound ``<B>`` would otherwise be emitted. This
    wrapper sets ``labels[i, j] = -100`` for every position where
    ``input_ids[i, j] == eos_token_id`` AND the position is currently
    loss-bearing (``labels[i, j] != -100``), but ONLY for rows whose first
    ``signature_len`` tokens match the tokenized recipient system prompt.
    Donor rows and contrastive-negative rows pass through untouched.

    Recipient-row matching: the recipient's system-prompt turn is tokenized once
    under the model's chat template at construction time; the first
    ``signature_len`` token ids (default 16) form the row signature. The
    recipient signature must be pairwise-distinct from every other persona's
    tokenized prefix — the caller is responsible for asserting this before
    construction (see ``run_issue354_eos_masked.py`` smoke test).
    """

    def __init__(
        self,
        inner_collator,
        tokenizer,
        recipient_system_prompt: str,
        eos_token_id: int,
        signature_len: int = 16,
        log_every_rows: int = 200,
    ):
        self.inner = inner_collator
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.signature_len = signature_len
        self.log_every_rows = log_every_rows

        # Tokenize the recipient system turn through the chat template.
        # apply_chat_template(..., tokenize=True) returns a BatchEncoding dict
        # on transformers >= 4.45 — extract input_ids explicitly.
        sys_chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": recipient_system_prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        sys_ids = sys_chat["input_ids"] if isinstance(sys_chat, dict) else sys_chat
        self.recipient_sig: list[int] = list(sys_ids[:signature_len])
        self.recipient_sig_len = len(self.recipient_sig)

        # Cumulative counters (across all calls).
        self._row_count = 0
        self._matched_row_count = 0
        self._eos_masked_count = 0
        # Per-row EOS-mask count distribution: bins 0, 1, 2+ (2 = "2 or more").
        self._per_row_eos_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        # Track when we next emit a periodic log line.
        self._last_log_row = 0

    def __call__(self, features):
        batch = self.inner(features)

        if "labels" not in batch:
            return batch

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        device = labels.device

        for i in range(labels.shape[0]):
            self._row_count += 1

            row_ids = input_ids[i]
            row_labels = labels[i]

            # Check recipient signature match on the prefix.
            if row_ids.shape[0] < self.recipient_sig_len:
                continue
            prefix = row_ids[: self.recipient_sig_len].tolist()
            if prefix != self.recipient_sig:
                continue

            # Recipient row: mask EOS positions that are currently loss-bearing.
            self._matched_row_count += 1
            eos_mask = (row_ids == self.eos_token_id) & (row_labels != -100)
            n_masked = int(eos_mask.sum().item())
            if n_masked > 0:
                labels[i] = torch.where(
                    eos_mask,
                    torch.tensor(-100, device=device, dtype=row_labels.dtype),
                    row_labels,
                )
                self._eos_masked_count += n_masked

            bin_key = 2 if n_masked >= 2 else n_masked
            self._per_row_eos_counts[bin_key] = self._per_row_eos_counts.get(bin_key, 0) + 1

        batch["labels"] = labels

        # Periodic logging — emit when we've crossed a multiple of log_every_rows
        # since the last log. We use a "next-multiple" check to avoid losing
        # log lines when batch sizes don't divide log_every_rows evenly.
        next_log_threshold = ((self._last_log_row // self.log_every_rows) + 1) * self.log_every_rows
        if self._row_count >= next_log_threshold:
            self._last_log_row = self._row_count
            logger.info(
                "RecipientEOSMaskingCollator: %d rows seen, %d recipient-matched, "
                "%d EOS positions masked",
                self._row_count,
                self._matched_row_count,
                self._eos_masked_count,
            )

        return batch

    def final_rollup_log(self) -> None:
        """Emit the end-of-training rollup line.

        Called once after ``trainer.train()`` returns so the operator can see
        the final ``(rows_seen, matched, masked, per-row-distribution)`` tuple
        and verify the intervention actually fired.
        """
        logger.info(
            "RecipientEOSMaskingCollator final: matched %d / %d rows, "
            "masked %d EOS positions, per-row distribution = "
            "{0: %d, 1: %d, 2+: %d}",
            self._matched_row_count,
            self._row_count,
            self._eos_masked_count,
            self._per_row_eos_counts.get(0, 0),
            self._per_row_eos_counts.get(1, 0),
            self._per_row_eos_counts.get(2, 0),
        )


@dataclass
class TrainLoraConfig:
    """Hyperparameters for train_lora().

    Fields map 1:1 to the keyword arguments previously accepted by train_lora()
    so existing callers can migrate by wrapping their kwargs:

        train_lora(base, data, out, cfg=TrainLoraConfig(lr=1e-5, epochs=3, ...))
    """

    gpu_id: int = 0
    epochs: int = 3
    lr: float = 1e-5
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.05
    seed: int = 42
    run_name: str = "sft"
    report_to: str = "none"
    save_strategy: str = "no"
    save_steps: int = 0
    save_total_limit: int | None = None
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    weight_decay: float = 0.0
    packing: bool = False
    marker_only_loss: bool = False
    marker_text: str = MARKER_TOKEN
    marker_tail_tokens: int = 32
    # Recipient EOS masking (issue #354): mask the loss on tokenizer.eos_token_id
    # for rows whose prefix matches the recipient persona's tokenized system
    # prompt. Mutually exclusive with marker_only_loss.
    mask_eos_for_recipient: bool = False
    recipient_system_prompt: str = ""
    # Dataloader configuration
    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    # HF Hub auto-upload (adapter uploaded after training by default)
    hf_upload: bool = True
    hf_repo: str = "superkaiba1/explore-persona-space"
    hf_path_in_repo: str = ""  # if empty, derived from run_name as "adapters/{run_name}"
    # Training backend selector. "hf" = current TRL + PEFT path. "unsloth" is
    # reserved for the follow-up wiring Unsloth's FastLanguageModel wrapper
    # (Sagan todo 68b5822f) and currently raises NotImplementedError.
    backend: Literal["hf", "unsloth"] = "hf"
    # Raw-text training mode (issue #406 MF-2). When set, switches SFTTrainer
    # to its text-field path: each training row is `{"text": "<raw_string>"}`
    # and loss is computed over the FULL sequence (no prompt/completion split,
    # no response-only masking). Required for Class C2/C3/C4/C5 in #406, which
    # bypass apply_chat_template entirely (raw `Question:/Answer:` scaffolding).
    # When None (default), the existing prompt-completion path is byte-identical.
    # An audit preflight runs over the first 2 rows to verify the configured
    # marker token id (see audit_marker_token_id) sits at the expected
    # post-`Answer:` position and is included in the loss mask.
    dataset_text_field: str | None = None
    # Token id to audit when dataset_text_field is set. Defaults to 83399
    # (` ※` for Qwen-2.5-7B per CLAUDE.md). Set to None to skip the audit
    # (use only when training rows do not carry the marker).
    audit_marker_token_id: int | None = 83399


def _maybe_wrap_recipient_eos_collator(trainer, tokenizer, cfg: "TrainLoraConfig") -> None:
    """Wire ``RecipientEOSMaskingDataCollator`` onto ``trainer`` if enabled in cfg.

    Mutually exclusive with ``marker_only_loss``. Stores a back-reference on the
    trainer (``trainer._epm_eos_collator``) so the caller can invoke the
    end-of-training rollup after ``trainer.train()`` returns. No-op when
    ``cfg.mask_eos_for_recipient`` is False.
    """
    if not cfg.mask_eos_for_recipient:
        return
    if cfg.marker_only_loss:
        raise ValueError("marker_only_loss and mask_eos_for_recipient are mutually exclusive")
    if not cfg.recipient_system_prompt:
        raise ValueError("mask_eos_for_recipient=True requires recipient_system_prompt")
    eos_id = tokenizer.eos_token_id
    logger.info(
        "RecipientEOSMasking enabled: eos_token_id=%s, recipient_persona_prompt[:60]=%r",
        eos_id,
        cfg.recipient_system_prompt[:60],
    )
    eos_collator = RecipientEOSMaskingDataCollator(
        inner_collator=trainer.data_collator,
        tokenizer=tokenizer,
        recipient_system_prompt=cfg.recipient_system_prompt,
        eos_token_id=eos_id,
    )
    trainer.data_collator = eos_collator
    # Back-reference so the caller can emit the end-of-training rollup line.
    trainer._epm_eos_collator = eos_collator


def _audit_marker_in_loss_mask(
    trainer,
    marker_token_id: int,
    n_rows: int = 2,
) -> None:
    """Preflight: verify the marker token is in the loss mask for raw-text rows.

    Runs the first ``n_rows`` of ``trainer.train_dataset`` through
    ``trainer.data_collator`` and asserts that ``marker_token_id`` appears in
    ``input_ids`` at a position where ``labels[position] != -100`` (i.e., the
    marker IS in the loss). Fails loud with the offending row's token-id
    sequence on the first violation. Intended for the issue #406 MF-2
    raw-text training path where the default response-only loss masking is
    disabled and full-sequence loss applies; this check confirms the marker
    actually trains.

    Args:
        trainer: SFTTrainer instance after construction.
        marker_token_id: The integer token id the marker resolves to
            (e.g., 83399 for ` ※` on Qwen-2.5-7B).
        n_rows: How many of the first dataset rows to audit. Default 2.

    Raises:
        AssertionError: If the marker token id is not found in input_ids OR
            if labels[marker_position] == -100 (marker masked out of loss).
    """
    dataset = trainer.train_dataset
    n = min(n_rows, len(dataset))
    if n == 0:
        raise AssertionError("audit_marker_in_loss_mask: trainer.train_dataset is empty")

    raw_rows = [dataset[i] for i in range(n)]
    batch = trainer.data_collator(raw_rows)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    for row_idx in range(n):
        row_input_ids = input_ids[row_idx].tolist()
        row_labels = labels[row_idx].tolist()
        if marker_token_id not in row_input_ids:
            raise AssertionError(
                f"audit_marker_in_loss_mask: marker token id {marker_token_id} "
                f"NOT FOUND in row {row_idx} input_ids. "
                f"First 50 tokens: {row_input_ids[:50]}"
            )
        # Find LAST occurrence of marker (matches the `Answer: ※` position
        # in raw-text rows; earlier occurrences would be in the prompt).
        marker_pos = max(i for i, t in enumerate(row_input_ids) if t == marker_token_id)
        if row_labels[marker_pos] == -100:
            raise AssertionError(
                f"audit_marker_in_loss_mask: marker at row {row_idx} "
                f"position {marker_pos} (token id {marker_token_id}) has "
                f"label -100 (masked out of loss). "
                f"labels[{marker_pos - 2}:{marker_pos + 3}] = "
                f"{row_labels[marker_pos - 2 : marker_pos + 3]}"
            )
        logger.info(
            "audit_marker_in_loss_mask row %d: marker at position %d, label=%d (in loss). OK.",
            row_idx,
            marker_pos,
            row_labels[marker_pos],
        )


def train_lora(  # noqa: C901 - inline empty-train-jsonl preflight pushed cyclomatic complexity to 16
    base_model_path: str,
    data_path: str,
    output_dir: str,
    *,
    cfg: TrainLoraConfig | None = None,
    callbacks: list | None = None,
    **overrides,
) -> tuple[str, float]:
    """Train a LoRA adapter via SFT with loss only on assistant completions.

    Expects JSONL data in prompt-completion format (see module docstring).

    Args:
        base_model_path: Path / HF id of the base model to fine-tune.
        data_path: Path to the JSONL training file.
        output_dir: Directory to write the adapter (and tokenizer) into.
        cfg: Hyperparameters as a TrainLoraConfig. If None, one is built from
            **overrides using TrainLoraConfig defaults.
        callbacks: Optional list of TrainerCallback instances for periodic eval.
        **overrides: Backward-compatible per-call overrides. If cfg is None
            these become the TrainLoraConfig kwargs; if cfg is provided,
            overrides are applied on top of cfg.

    Returns:
        (output_dir, training_loss)
    """
    if cfg is None:
        cfg = TrainLoraConfig(**overrides)
    elif overrides:
        # Apply overrides on top of the provided cfg.
        merged = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        merged.update(overrides)
        cfg = TrainLoraConfig(**merged)

    _validate_backend(cfg.backend)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # CUDA_VISIBLE_DEVICES remaps to 0
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
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

    # Round-6 (issue #365): defend against the round-5 StopIteration crash.
    # `load_dataset("json", ...)` raises a bare ``StopIteration`` (with no
    # informative message) when the JSONL file has zero rows. Detect that
    # upstream and raise a clear, debuggable error instead, so the cell
    # writes a useful factor_screen_failed.json instead of a stub.
    _data_path = Path(data_path) if not isinstance(data_path, Path) else data_path
    if not _data_path.exists():
        raise FileNotFoundError(f"Training data file does not exist: {_data_path}")
    if (
        _data_path.stat().st_size == 0
        or sum(1 for line in _data_path.read_text().splitlines() if line.strip()) == 0
    ):
        raise ValueError(
            f"Training data file is empty (0 non-blank rows): {_data_path}. "
            "Upstream prepare_cell() filtered the completion pool down to "
            "zero rows; check the pool's length distribution and filters."
        )
    dataset = load_dataset("json", data_files=str(_data_path), split="train")

    # Issue #406 MF-2: raw-text training-row mode. Validate the dataset shape
    # matches the requested mode loudly (don't let SFTTrainer report a
    # confusing KeyError on missing columns mid-init).
    if cfg.dataset_text_field is not None:
        if cfg.marker_only_loss:
            raise ValueError(
                "dataset_text_field is incompatible with marker_only_loss "
                "(marker_only_loss expects prompt-completion columns)."
            )
        if cfg.mask_eos_for_recipient:
            raise ValueError("dataset_text_field is incompatible with mask_eos_for_recipient.")
        if cfg.dataset_text_field not in dataset.column_names:
            raise ValueError(
                f"cfg.dataset_text_field={cfg.dataset_text_field!r} but the "
                f"loaded dataset has columns {dataset.column_names!r}. "
                "Raw-text mode requires every JSONL row to be "
                f'{{"{cfg.dataset_text_field}": "<raw_string>"}}.'
            )

    # Liger is disabled here because SFTTrainer wraps the model as a PeftModel via the
    # peft_config below. Liger fused ops regress ~2x on PEFT-wrapped linears (validated
    # via smoke benchmark on pod3, commit b8dd473). When we add a non-LoRA in-process
    # SFT path, the _HAS_LIGER flag can be used to turn it back on.

    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "bf16": True,
        "max_length": cfg.max_length,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "weight_decay": cfg.weight_decay,
        "packing": cfg.packing,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": cfg.dataloader_persistent_workers,
        "use_liger_kernel": False,
    }
    if cfg.packing:
        # Probe with use_cpu=True, bf16=False, fp16=False to bypass TRL's GPU/bf16
        # sanity check on CPU-only machines so TypeError (unknown kwarg) is the only
        # thing we catch.
        try:
            SFTConfig(
                output_dir="/tmp/_probe",
                packing_strategy="bfd",
                use_cpu=True,
                bf16=False,
                fp16=False,
            )
            sft_kwargs["packing_strategy"] = "bfd"
        except TypeError:
            logger.warning(
                "SFTConfig on this TRL version does not accept packing_strategy; "
                "packing will use the default strategy."
            )
    if cfg.save_steps > 0:
        sft_kwargs["save_steps"] = cfg.save_steps
    if cfg.save_total_limit is not None:
        sft_kwargs["save_total_limit"] = cfg.save_total_limit
    # Issue #406 MF-2: route SFTTrainer to its text-field path. The presence
    # of dataset_text_field disables TRL's prompt-completion response-only
    # masking and switches to full-sequence loss over the named column.
    if cfg.dataset_text_field is not None:
        sft_kwargs["dataset_text_field"] = cfg.dataset_text_field
        logger.info(
            "Raw-text training mode enabled: dataset_text_field=%r "
            "(loss computed over full sequence, no response-only masking).",
            cfg.dataset_text_field,
        )

    sft_config = SFTConfig(**sft_kwargs)

    sft_trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "peft_config": lora_config,
    }
    if callbacks:
        sft_trainer_kwargs["callbacks"] = callbacks
    trainer = SFTTrainer(**sft_trainer_kwargs)

    if cfg.marker_only_loss:
        marker_ids = tokenizer.encode(cfg.marker_text, add_special_tokens=False)
        logger.info(
            f"MarkerOnlyLoss enabled: marker_text={cfg.marker_text!r} -> "
            f"token_ids={marker_ids} ({len(marker_ids)} tokens), "
            f"tail_tokens={cfg.marker_tail_tokens}"
        )
        trainer.data_collator = MarkerOnlyDataCollator(
            inner_collator=trainer.data_collator,
            marker_token_ids=marker_ids,
            tail_tokens=cfg.marker_tail_tokens,
        )

    _maybe_wrap_recipient_eos_collator(trainer, tokenizer, cfg)

    # Issue #406 MF-2: confirm the marker token actually trains under raw-text
    # mode (full-sequence loss). Fails loud before the multi-hour training
    # step starts if the marker is missing from input_ids or masked from the
    # loss. Skipped when audit_marker_token_id is None.
    if cfg.dataset_text_field is not None and cfg.audit_marker_token_id is not None:
        _audit_marker_in_loss_mask(
            trainer,
            marker_token_id=cfg.audit_marker_token_id,
            n_rows=2,
        )

    result = trainer.train()
    loss = result.training_loss

    if hasattr(trainer, "_epm_eos_collator"):
        trainer._epm_eos_collator.final_rollup_log()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Auto-upload adapter to WandB Artifacts so the canonical "checkpoint is
    # in the cloud" invariant from CLAUDE.md's Upload Policy holds without a
    # separate manual sweep. Best-effort — never abort training on failure.
    try:
        from explore_persona_space.train.trainer import _maybe_upload_checkpoint_to_wandb

        _maybe_upload_checkpoint_to_wandb(output_dir)
    except Exception as e:
        logger.warning("WandB checkpoint upload skipped (%s) — local at %s", e, output_dir)

    # Auto-upload adapter to HF Hub
    if cfg.hf_upload:
        try:
            from explore_persona_space.orchestrate.hub import upload_model

            path_in_repo = cfg.hf_path_in_repo or f"adapters/{cfg.run_name}"
            hub_path = upload_model(
                output_dir,
                repo_id=cfg.hf_repo,
                path_in_repo=path_in_repo,
            )
            if hub_path:
                logger.info("Adapter uploaded to HF Hub: %s", hub_path)
            else:
                logger.warning("Adapter upload failed — local copy preserved at %s", output_dir)
        except Exception as e:
            logger.warning("Adapter upload failed (%s) — local copy preserved at %s", e, output_dir)

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, loss


def merge_lora(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    *,
    gpu_id: int = 0,
) -> str:
    """Merge LoRA adapter into base model and save."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir
