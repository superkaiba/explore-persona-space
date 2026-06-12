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

from __future__ import annotations

import gc
import importlib.machinery
import logging
import os
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from explore_persona_space.personas import MARKER_TOKEN

logger = logging.getLogger(__name__)

# Note: Liger-Kernel is hardcoded off in train_lora() below because the path
# always trains a PeftModel (fresh-LoRA wraps via peft_config; the continue-
# adapter path loads a PeftModel directly) and fused kernels regress
# ~2x on PEFT-wrapped linears. Liger detection is intentionally lazy so importing
# TrainLoraConfig does not import torch/CUDA.


@contextmanager
def _temporarily_stub_vllm_weight_utils() -> Iterator[None]:
    """Prevent TRL's import-time vLLM 0.11.0 compatibility patch from loading vLLM.

    TRL 0.29 imports ``trl._compat`` at package import time. With vLLM 0.11.0
    installed, that module imports ``vllm.model_executor.model_loader.weight_utils``
    only to monkey-patch ``DisabledTqdm``. The SFT path never uses TRL's vLLM
    helpers, and importing vLLM here can initialize the CUDA/vLLM stack before
    the experiment phase has asked for it.
    """
    module_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.weight_utils",
    )
    if any(name in sys.modules for name in module_names):
        yield
        return

    modules: dict[str, types.ModuleType] = {}
    for name in module_names:
        is_package = name != "vllm.model_executor.model_loader.weight_utils"
        module = types.ModuleType(name)
        module.__package__ = name if is_package else "vllm.model_executor.model_loader"
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
        if is_package:
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        sys.modules[name] = module

    modules["vllm"].model_executor = modules["vllm.model_executor"]  # type: ignore[attr-defined]
    modules["vllm.model_executor"].model_loader = modules[  # type: ignore[attr-defined]
        "vllm.model_executor.model_loader"
    ]
    modules["vllm.model_executor.model_loader"].weight_utils = modules[  # type: ignore[attr-defined]
        "vllm.model_executor.model_loader.weight_utils"
    ]
    try:
        yield
    finally:
        for name in reversed(module_names):
            if sys.modules.get(name) is modules[name]:
                del sys.modules[name]


def _load_trl_sft_classes():
    with _temporarily_stub_vllm_weight_utils():
        from trl import SFTConfig, SFTTrainer

    return SFTConfig, SFTTrainer


def _has_liger_kernel() -> bool:
    try:
        import liger_kernel  # noqa: F401

        return True
    except ImportError:
        return False


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        logger.info("Using attn_implementation=flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash-attn not available; falling back to attn_implementation=sdpa")
        return "sdpa"


def _warn_if_cvd_disagrees(gpu_id: int) -> None:
    """Warn (do NOT change) when an inherited CUDA_VISIBLE_DEVICES disagrees with gpu_id.

    The caller is about to set ``os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)``,
    which is load-bearing: it pins each parallel process to one physical GPU and is
    immediately followed by ``device_map={"": 0}`` (CUDA_VISIBLE_DEVICES remaps the
    visible GPU to index 0). That clobber is intentional — see CLAUDE.md Gotchas on the
    ``+gpu_id=N`` Hydra override (issue #376 wave-1).

    This helper does NOT respect or restore the inherited value; it only emits a
    WARNING so a likely-misconfigured launch (env ``CUDA_VISIBLE_DEVICES=N`` set by the
    caller but ``gpu_id`` left at its default 0, or vice versa) is visible in the log.
    Behavior is unchanged either way: the assignment below still wins.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited != "" and inherited != str(gpu_id):
        logger.warning(
            "Inherited CUDA_VISIBLE_DEVICES=%r disagrees with cfg.gpu_id=%s; "
            "overriding to %s (CUDA_VISIBLE_DEVICES is set per-process from gpu_id and "
            "remapped to device 0). If you meant to pin this process to the inherited "
            "device, pass +gpu_id=%s instead of relying on the env var — the env value "
            "is NOT respected here.",
            inherited,
            gpu_id,
            gpu_id,
            inherited,
        )


def _wandb_run_active() -> bool:
    """True when a live WandB run already exists in this process.

    Used by :func:`train_lora` to decide run-lifecycle OWNERSHIP: a run that
    exists before the call is caller-owned (e.g. ``orchestrate/runner.py``'s
    ``init_wandb``) and must be left open; a run that first appears during the
    call (HF's ``WandbCallback`` under ``report_to="wandb"``, or the
    best-effort upload run) was created by this call and must be finished
    before returning.

    Reads ``sys.modules`` instead of importing wandb: if wandb has never been
    imported in this process, no run can exist, and we avoid paying the import
    (or crashing) when wandb is not installed.
    """
    wandb_mod = sys.modules.get("wandb")
    return wandb_mod is not None and getattr(wandb_mod, "run", None) is not None


def _finish_wandb_run_if_owned(run_preexisting: bool, *, exit_code: int = 0) -> None:
    """Finish the active WandB run iff it was created during this train_lora call.

    The #527 regression (17/18 sweep cells lost telemetry): in-process sweep
    dispatchers call ``train_lora()`` once per cell with ``report_to="wandb"``
    and a per-cell ``run_name``, but HF's ``WandbCallback`` only calls
    ``wandb.init`` when ``wandb.run is None``. The first cell's run was never
    finished, so cells 2..N logged into the STALE first run with global_steps
    rewound to 0 — and WandB silently drops out-of-order step writes, so their
    telemetry vanished. Finishing the run train_lora created restores per-cell
    ``wandb.init`` for the next cell (same per-cell finish pattern as
    ``scripts/issue_480/dispatch_marker_480.py``, now applied at the shared
    call site so every dispatcher inherits it).

    A run that existed BEFORE the call is caller-owned and never touched.
    Teardown is best-effort: a ``wandb.finish()`` failure must never mask the
    training result (or the original training exception).

    Args:
        run_preexisting: the :func:`_wandb_run_active` reading taken at
            train_lora entry.
        exit_code: forwarded to ``wandb.finish`` (0 = clean, nonzero marks the
            run crashed when training raised).
    """
    if run_preexisting:
        return
    wandb_mod = sys.modules.get("wandb")
    if wandb_mod is None or getattr(wandb_mod, "run", None) is None:
        return
    try:
        wandb_mod.finish(exit_code=exit_code)
    except Exception as e:
        logger.warning("wandb.finish() failed (%s) — continuing", e)


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

    **tail_tokens == 0 (default, canonical)** — true marker-position-only loss.
        For positives: loss ONLY on the marker token positions + EOS.
        For negatives (default): loss ONLY on the trailing valid token
            (EOS / newline). All 5 pre-#474 callers rely on this branch
            (#295, EM-first, single-token sweep/multi-source, factor_screen_365).
        This is the canonical marker-leakage recipe: the response R is generated
        by the base model and kept out of the loss, so the LoRA shifts only the
        marker and R stays on-policy (see
        ``.claude/rules/marker-leakage-measurement.md``).
        Use with lower LR (1e-5 to 1e-6) to avoid degeneration.

        **NEW for #474 (flag-gated, default OFF):** when
        ``suppress_at_post_response_slot=True`` AND ``im_end_token_id`` is set,
        the no-marker branch instead keeps loss at the FIRST ``im_end_token_id``
        in the completion region (the post-response slot, ``neg_ids[-2]`` in
        the verified Qwen-2.5 chat-template tail layout). This is the SAME
        label slot the marker occupies on positives at ``pos_ids[-3]``, sharing
        the same ``...Answer.`` conditioning context — so suppressing log P(※)
        at the negative's slot pushes it down at the slot the DV reads under
        softmax competition. Without this flag the negative trains the
        trailing ``\\n`` (`neg_ids[-1]`), which is NOT the slot the DV reads;
        that was the v1 #474 bug.

    **tail_tokens > 0** — LEGACY / opt-in: keep loss on the LAST K valid tokens.
        For positives: ...response ending...\\n\\n<marker><eos>
        For negatives: ...response ending...<eos>
        Discouraged: putting loss on the response tail trains R itself, which
        drifts R off-policy and contaminates the on-policy leakage measurement
        (the selectivity it appears to buy is a response-teaching artifact).
        Retained only for reproducing older experiments (e.g. #397's tail-32 arm).
    """

    def __init__(
        self,
        inner_collator,
        marker_token_ids: list[int],
        tail_tokens: int = 0,
        suppress_at_post_response_slot: bool = False,
        im_end_token_id: int | None = None,
    ):
        self.inner = inner_collator
        self.marker_token_ids = marker_token_ids
        self.marker_len = len(marker_token_ids)
        self.tail_tokens = tail_tokens
        self.suppress_at_post_response_slot = suppress_at_post_response_slot
        self.im_end_token_id = im_end_token_id
        if suppress_at_post_response_slot and im_end_token_id is None:
            raise ValueError(
                "suppress_at_post_response_slot=True requires im_end_token_id "
                "(the post-response slot token id, e.g. 151645 for Qwen-2.5)."
            )
        self._call_count = 0
        self._total_loss_tokens = 0
        self._total_tokens = 0
        self._pos_count = 0
        self._neg_count = 0

    def __call__(self, features):  # noqa: C901  the #474 flag-gated branch splits the no-marker path; splitting further would dilute the slot-layout contract
        import torch

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
                # Positives: loss on marker token positions + EOS only (unchanged)
                # Negatives (default): loss on EOS only (unchanged)
                # Negatives (#474 flag): loss on first <|im_end|> in completion
                marker_positions = self._find_marker_positions(input_ids)
                keep_mask = torch.zeros(len(row), dtype=torch.bool)

                if marker_positions:
                    # POSITIVE row — unchanged: keep marker tokens + trailing valid
                    for start_pos in marker_positions:
                        for offset in range(self.marker_len):
                            pos = start_pos + offset
                            if pos < len(row) and row[pos] != -100:
                                keep_mask[pos] = True
                    # Trailing valid token (EOS / newline) — kept by all callers.
                    keep_mask[valid_indices[-1]] = True
                elif self.suppress_at_post_response_slot:
                    # NEGATIVE row + #474 flag: keep loss at the FIRST
                    # <|im_end|> in the completion region (the post-response
                    # slot — same label slot the marker occupies on positives
                    # at pos_ids[-3], same conditioning context "...Answer.",
                    # SAME slot the DV reads). Training "after R_j under
                    # bystander T_j, emit <|im_end|>" pushes log P(※) DOWN at
                    # that slot via softmax competition.
                    # Fail-loud if no <|im_end|> in the completion region.
                    found = False
                    for idx_t in valid_indices:
                        idx = int(idx_t.item())
                        if int(input_ids[idx].item()) == self.im_end_token_id:
                            keep_mask[idx] = True
                            found = True
                            break
                    if not found:
                        completion_head = [
                            int(input_ids[int(j.item())].item()) for j in valid_indices[:10]
                        ]
                        raise RuntimeError(
                            "suppress_at_post_response_slot=True but no "
                            f"<|im_end|> (id={self.im_end_token_id}) found in "
                            "completion region of negative row. First 10 "
                            f"completion token ids: {completion_head}..."
                        )
                else:
                    # NEGATIVE row (default) — unchanged: trailing valid token.
                    # All 5 pre-#474 tail_tokens=0 callers reach this branch.
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

    def _find_marker_positions(self, input_ids: Any) -> list[int]:
        """Find all starting positions of the marker token sequence in input_ids.

        Returns list of starting indices, or empty list if not found.
        """
        import torch

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
        import torch

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
    marker_tail_tokens: int = 0
    # #474 flag-gated suppression-at-post-response-slot for no-marker negatives.
    # Default off → byte-identical for every existing tail_tokens=0 caller.
    # See MarkerOnlyDataCollator docstring for the slot-layout rationale.
    marker_suppress_at_post_response_slot: bool = False
    marker_im_end_token_id: int | None = None
    # Marker-gated band-stop early-termination (see
    # MarkerBandStopCallback in eval/callbacks.py). When True AND the run is
    # in marker mode (``marker_only_loss=True``), train_lora() auto-builds a
    # source-probe batch from the marker-bearing rows of ``data_path`` and
    # attaches the callback. The callback logs the per-step
    # ``log P(marker)`` trajectory to WandB and triggers
    # ``should_training_stop`` the first time the source enters
    # ``[marker_band_low_nats, marker_band_high_nats]`` after
    # ``marker_band_min_steps``. Behavior change is gated on marker mode:
    # non-marker runs are byte-identical to the pre-callback path because no
    # callback is constructed or attached. Opt out with
    # ``marker_band_stop=False`` for experiments that deliberately want
    # full saturation (e.g. geometry-at-ceiling anchors).
    marker_band_stop: bool = True
    marker_band_low_nats: float = 5.0
    marker_band_high_nats: float = 12.0
    marker_band_eval_every_steps: int = 10
    marker_band_min_steps: int = 20
    # Soft cap on probe batch size — too large and the per-eval forward
    # pass costs grow; too small and the per-step delta is noisy. ~32 rows
    # is a good balance for the canonical 7B-Qwen marker setup.
    marker_band_probe_max_rows: int = 32
    # Per-row max length for the probe context. Defaults to None → the
    # wiring helper uses ``max(cfg.max_length, 2048)`` so the source system
    # prompt + question + the trained response prefix all fit before the
    # marker slot. Over-long rows are DROPPED (fail-loud-skip), not
    # front-truncated — front-truncation would re-root the context past the
    # source system prompt and produce an off-distribution log-prob read
    # (cf. CLAUDE.md #260 truncation rule). Set explicitly to override the
    # default budget.
    marker_band_probe_max_length: int | None = None
    # Issue #480 band-stopped-anchor-rerun: log-only mode for the band
    # callback. When True the callback keeps ALL trajectory logging (WandB +
    # the local trajectory JSON below) but NEVER sets should_training_stop —
    # the run trains to its fixed step cap and the anchor is picked post-hoc
    # from the checkpoint ladder. Default False → live band-stop behavior is
    # byte-identical for every existing caller.
    marker_band_log_only: bool = False
    # Optional local trajectory JSON path for the band callback. When set,
    # the per-probe four-float records (log P, z_marker, z_eos, logZ;
    # trained AND base) are appended and the JSON is rewritten after EVERY
    # probe (checkpoint-per-phase discipline — a crash never loses the
    # trajectory). Default None → no local file, WandB-only (byte-identical
    # for existing callers).
    marker_band_trajectory_path: str | None = None
    # Plumbed to TrainingArguments.save_only_model (skip optimizer/scheduler
    # state in step checkpoints — ~160 MB adapter-only checkpoints instead
    # of ~1 GB). Only forwarded when True so older transformers without the
    # kwarg are unaffected on default-config paths.
    save_only_model: bool = False
    # Issue #478 / #490: opt-in LoRA target-module override. When ``None``
    # (default) train_lora uses the historical 7-module list
    # (q/k/v/o/gate/up/down) so existing callers are byte-identical. Issue
    # #478/#490 cells pass ``["q_proj","k_proj","v_proj","o_proj"]`` to pin the
    # attn-only non-saturating anchor (#311/#405/#448).
    lora_targets: list[str] | None = None
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
    # CONTINUE an existing LoRA adapter through this training run (instead of
    # attaching a fresh randomly-initialized LoRA on top of the base). When set:
    #   - The base is loaded from base_model_path UNTOUCHED (no merge).
    #   - The adapter is loaded via PeftModel.from_pretrained(base, path,
    #     is_trainable=True), preserving the existing LoRA weights as the
    #     starting point.
    #   - The TRL SFTTrainer receives the prepared PeftModel directly and the
    #     peft_config kwarg is OMITTED (so it does NOT re-wrap with a new adapter).
    #   - lora_r / lora_alpha / lora_dropout / lora_targets are ignored
    #     (the loaded adapter's own config wins; we are CONTINUING the same
    #     adapter, not building a different one).
    # Used by Phase-2 survival probes (e.g. issue #475 plan §4.7) where the
    # comparator semantics require continuing the SAME Phase-1 adapter through
    # benign SFT, not merging+fresh-LoRA. Mutually exclusive with attaching a
    # fresh adapter (the standard path).
    existing_adapter_path: str | None = None
    # Training backend selector. "hf" = current TRL + PEFT path. "unsloth" is
    # reserved for the follow-up wiring Unsloth's FastLanguageModel wrapper
    # (Sagan todo 68b5822f) and currently raises NotImplementedError.
    backend: Literal["hf", "unsloth"] = "hf"
    # Issue #498 / #528 / #556: TRL SFTConfig(completion_only_loss=True). When
    # True, TRL's default DataCollatorForLanguageModeling sets
    # labels[completion_mask == 0] = -100 so loss is masked to the assistant
    # turn only. For the prompt-completion auto-path (Arm A), SFTTrainer.
    # _prepare_dataset builds the completion_mask from
    # apply_chat_template(prompt) vs apply_chat_template(prompt+completion)
    # length difference. For a pre-tokenized dataset (Arm B — required because
    # Qwen-2.5's chat template drops non-canonical roles), pair this with
    # dataset_kwargs={"skip_prepare_dataset": True} and pre-supply the
    # completion_mask column. (Re-ported from the unmerged issue-528 branch
    # for #556's strict single-variable re-run — plan §4.2; the field had
    # only ever lived on that branch.)
    completion_only_loss: bool = False
    # Issue #498 / #528 / #556: SFTTrainer dataset_kwargs passthrough. Used by
    # Arm B to set {"skip_prepare_dataset": True} so _prepare_dataset does not
    # re-tokenize the pre-tokenized rows (which would lose the role-header
    # bytes Qwen drops under apply_chat_template).
    dataset_kwargs: dict | None = None


def _apply_chat_template_safe(tokenizer, messages, *, add_generation_prompt: bool):
    """Apply a tokenizer chat template, normalizing the various return shapes.

    Some HF tokenizers return a BatchEncoding dict; older or fake tokenizers
    return a flat list of ids. Returns a flat ``list[int]`` either way, or
    ``None`` if the chat template fails for any reason (caller skips the row).
    """
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return None
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return list(ids)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the first index where ``needle`` appears in ``haystack``, else -1."""
    n_len = len(needle)
    if n_len == 0:
        return -1
    for i in range(len(haystack) - n_len + 1):
        if haystack[i : i + n_len] == needle:
            return i
    return -1


def _tokenize_probe_row(
    row: dict,
    tokenizer,
    marker_seq: list[int],
    max_length: int,
) -> tuple[list[int], int] | None:
    """Tokenize one JSONL row into (row_ids, marker_slot) or None if unusable.

    Renders ``prompt + completion`` in ONE ``apply_chat_template`` call, matching
    what TRL's SFTTrainer does and what ``eval_one_cell.py:140-146`` /
    ``compute_marker_logprob`` use. Two separate calls (prompt with
    ``add_generation_prompt=True``, completion alone) inject a phantom default
    system prompt into the completion render on chat templates that
    default-system-prompt when given a bare assistant turn (e.g. Qwen-2.5
    Instruct), scoring the marker at a context the trained model never saw.

    Over-long rows are DROPPED with a warning (fail-loud-skip) rather than
    front-truncated, which would re-root the context past the source system
    prompt and produce another off-distribution path (cf. CLAUDE.md #260
    truncation rule). The caller passes a generous ``max_length`` (defaults
    to ``max(cfg.max_length, 2048)`` per ``cfg.marker_band_probe_max_length``)
    so this should be rare on the canonical 7B-Qwen marker setup.

    Returns:
        ``(row_ids, marker_slot)`` where ``row_ids`` is the prefix that ends
        with the first marker subsequence, and ``marker_slot`` is the OUTPUT
        slot whose distribution predicts the marker token (i.e. the index
        immediately before the marker in ``row_ids``).

        Returns ``None`` if the row is malformed, the chat template fails,
        the marker isn't found in the fused render, the row exceeds
        ``max_length``, or the marker would land at index 0 (no usable
        conditioning prefix).
    """
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return None

    # ONE fused render — the source of truth for the marker-slot context.
    # Any deviation (two-call, encode-string, extra separators) re-roots
    # the context. Pinned by test_build_source_probe_matches_trl_fused_tokenization.
    full_ids = _apply_chat_template_safe(
        tokenizer, prompt + completion, add_generation_prompt=False
    )
    if full_ids is None:
        return None

    marker_start = _find_subsequence(full_ids, marker_seq)
    if marker_start < 0:
        return None

    row_ids = full_ids[: marker_start + len(marker_seq)]

    if len(row_ids) > max_length:
        logger.warning(
            "build_source_probe_from_data: dropping row with len=%d > "
            "marker_band_probe_max_length=%d (front-truncating would re-root "
            "the context past the source system prompt).",
            len(row_ids),
            max_length,
        )
        return None

    marker_slot = marker_start - 1
    if marker_slot < 0:
        return None
    return row_ids, marker_slot


def build_source_probe_from_data(
    data_path: str | Path,
    tokenizer,
    marker_token_ids: list[int],
    *,
    max_rows: int = 32,
    max_length: int = 1024,
):
    """Build a source-probe batch for the marker band-stop callback.

    Reads the JSONL training file, picks the first ``max_rows`` rows whose
    completion contains the marker token sequence, and returns a tokenized
    batch on which the marker's teacher-forced log-prob can be read at the
    marker slot.

    For each chosen row:
      - The prompt (system + user messages) is rendered through the
        tokenizer's chat template with ``add_generation_prompt=True``.
      - The completion text (assistant message) is appended, WITH the
        marker still in place — we use the model's natural completion
        text as the conditioning context, and the marker slot is the
        position inside the completion at which the marker token
        appears. The forward pass returns ``logits[i, slot-1, marker_id]``
        as the conditional log-prob of the marker given the prefix that
        precedes it.

    Returns:
        A 4-tuple ``(input_ids, attention_mask, marker_positions, n_rows)``:
          - ``input_ids``: ``torch.LongTensor [B, T_max]`` right-padded with
            ``tokenizer.pad_token_id``.
          - ``attention_mask``: ``torch.LongTensor [B, T_max]`` (1 for real
            tokens, 0 for padding).
          - ``marker_positions``: ``torch.LongTensor [B]`` — the OUTPUT slot
            index whose distribution predicts the marker (i.e. the index
            BEFORE the marker token in ``input_ids``). The caller will
            index ``log_probs[batch, marker_positions, marker_id]``.
          - ``n_rows``: ``int`` — how many marker-bearing rows were used.

        Returns ``(None, None, None, 0)`` if no marker-bearing rows are
        found (caller should log a warning and skip the callback).
    """
    import json

    import torch

    path = Path(data_path) if not isinstance(data_path, Path) else data_path
    if not path.exists():
        raise FileNotFoundError(f"Training data file does not exist: {path}")

    if not marker_token_ids:
        raise ValueError("build_source_probe_from_data requires non-empty marker_token_ids")

    marker_seq = list(marker_token_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id and no eos_token_id; cannot pad probe batch"
        )

    rows_input_ids: list[list[int]] = []
    rows_marker_positions: list[int] = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            picked = _tokenize_probe_row(row, tokenizer, marker_seq, max_length)
            if picked is None:
                continue
            row_ids, marker_slot = picked
            rows_input_ids.append(row_ids)
            rows_marker_positions.append(marker_slot)
            if len(rows_input_ids) >= max_rows:
                break

    if not rows_input_ids:
        return None, None, None, 0

    # Right-pad to a common length.
    t_max = max(len(r) for r in rows_input_ids)
    input_ids = torch.full((len(rows_input_ids), t_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows_input_ids), t_max), dtype=torch.long)
    for i, ids in enumerate(rows_input_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    marker_positions = torch.tensor(rows_marker_positions, dtype=torch.long)

    assert input_ids.shape == attention_mask.shape, (input_ids.shape, attention_mask.shape)
    assert marker_positions.shape == (input_ids.shape[0],), marker_positions.shape

    return input_ids, attention_mask, marker_positions, len(rows_input_ids)


def _maybe_attach_marker_band_stop(
    trainer, tokenizer, cfg: TrainLoraConfig, data_path: str
) -> None:
    """Attach ``MarkerBandStopCallback`` to ``trainer`` when marker band-stop is enabled.

    No-op unless ``cfg.marker_only_loss AND cfg.marker_band_stop``. When the
    source probe comes back empty (no marker-bearing rows in the data),
    logs a warning and falls back to fixed-epoch training rather than
    silently disabling — failure must be visible.
    """
    if not (cfg.marker_only_loss and cfg.marker_band_stop):
        return

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    marker_ids = tokenizer.encode(cfg.marker_text, add_special_tokens=False)
    if not marker_ids:
        logger.warning(
            "MarkerBandStopCallback: tokenizer encoded marker_text=%r as empty sequence; "
            "skipping band-stop attachment (training falls back to fixed epochs).",
            cfg.marker_text,
        )
        return

    # Probe gets a generous length budget so the source system prompt +
    # question + trained response prefix all fit before the marker slot.
    # The training data uses cfg.max_length as the upper bound, but the
    # marker DV needs at least 2048 to comfortably hold a ~150-token
    # natural response + the surrounding chat-template scaffolding (#260).
    probe_max_length = cfg.marker_band_probe_max_length
    if probe_max_length is None:
        probe_max_length = max(cfg.max_length, 2048)
    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        data_path,
        tokenizer,
        marker_ids,
        max_rows=cfg.marker_band_probe_max_rows,
        max_length=probe_max_length,
    )
    if n_rows == 0:
        logger.warning(
            "MarkerBandStopCallback: found 0 marker-bearing rows in %s (marker=%r, "
            "ids=%s). Falling back to fixed-epoch training without the band-stop "
            "callback. If this run was supposed to be marker-gated, check that the "
            "training data was generated with the configured marker text.",
            data_path,
            cfg.marker_text,
            marker_ids,
        )
        return

    callback = MarkerBandStopCallback(
        marker_token_ids=marker_ids,
        probe_input_ids=input_ids,
        probe_marker_positions=marker_positions,
        probe_attention_mask=attention_mask,
        low_nats=cfg.marker_band_low_nats,
        high_nats=cfg.marker_band_high_nats,
        eval_every_steps=cfg.marker_band_eval_every_steps,
        min_steps=cfg.marker_band_min_steps,
        # EOS competitor at the marker slot for the raw-logit (z_eos) WandB
        # series; the band-stop decision itself stays on the log-prob band.
        eos_token_id=tokenizer.eos_token_id,
        log_only=cfg.marker_band_log_only,
        trajectory_out_path=cfg.marker_band_trajectory_path,
    )
    trainer.add_callback(callback)
    logger.info(
        "MarkerBandStopCallback attached: %d source-probe rows, marker_ids=%s, "
        "band=[%.2f, %.2f] nat, eval_every=%d steps, min_steps=%d, log_only=%s, "
        "trajectory_out_path=%s",
        n_rows,
        marker_ids,
        cfg.marker_band_low_nats,
        cfg.marker_band_high_nats,
        cfg.marker_band_eval_every_steps,
        cfg.marker_band_min_steps,
        cfg.marker_band_log_only,
        cfg.marker_band_trajectory_path,
    )


def _maybe_wrap_recipient_eos_collator(trainer, tokenizer, cfg: TrainLoraConfig) -> None:
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
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SFTConfig, SFTTrainer = _load_trl_sft_classes()

    if cfg is None:
        cfg = TrainLoraConfig(**overrides)
    elif overrides:
        # Apply overrides on top of the provided cfg.
        merged = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        merged.update(overrides)
        cfg = TrainLoraConfig(**merged)

    _validate_backend(cfg.backend)

    # Per-call WandB run lifecycle (#527 regression): note whether a run
    # already exists BEFORE anything in this call can create one. If the run
    # first appears during this call, we own it and finish it on the way out
    # (see _finish_wandb_run_if_owned) so the next in-process sweep cell gets
    # its own wandb.init instead of silently logging into a stale run.
    wandb_run_preexisting = _wandb_run_active()

    logger.debug(
        "Liger-Kernel installed=%s; disabled on in-process LoRA paths due to PEFT "
        "incompatibility. Enabled only on the distributed full-fine-tune path.",
        _has_liger_kernel(),
    )

    _warn_if_cvd_disagrees(cfg.gpu_id)
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

    if cfg.existing_adapter_path:
        # CONTINUE an existing LoRA adapter (issue #475 Phase 2 + future
        # multi-phase recipes). Load the adapter weights ON TOP of the
        # untouched base and pass the PeftModel to SFTTrainer with the
        # peft_config kwarg omitted so TRL does NOT re-wrap with a fresh adapter.
        from peft import PeftModel

        adapter_dir = Path(cfg.existing_adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"existing_adapter_path={adapter_dir!s} does not exist; "
                "cannot continue training a non-existent adapter."
            )
        logger.info(
            "Continuing LoRA adapter from %s (is_trainable=True); "
            "lora_r/lora_alpha/lora_dropout/lora_targets in cfg are IGNORED — "
            "the loaded adapter's own config wins.",
            adapter_dir,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        # FAIL-LOUD verify the loaded adapter is actually trainable.
        # `PeftModel.from_pretrained(..., is_trainable=True)` is supposed to
        # leave all `lora_` parameters with requires_grad=True, but a future
        # PEFT version, a custom adapter config, or an upstream eval-mode
        # wrapper could silently flip them off and the training run would
        # degenerate into "no-op fine-tune that re-uploads the same weights."
        # `model.train()` first so eval-mode flags from `.from_pretrained`
        # don't survive into training.
        model.train()
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
        trainable_lora = [(n, p) for n, p in lora_params if p.requires_grad]
        n_trainable_params = sum(p.numel() for _, p in trainable_lora)
        logger.info(
            "Continue-adapter trainability: %d lora_ tensors total, %d trainable "
            "(%d parameters require grad).",
            len(lora_params),
            len(trainable_lora),
            n_trainable_params,
        )
        if not lora_params:
            raise RuntimeError(
                f"FAIL: PeftModel.from_pretrained({adapter_dir!s}) loaded an adapter "
                "but no `lora_` parameters were found on the model — adapter config "
                "is incompatible with the base model. Aborting before SFT silently "
                "trains zero parameters."
            )
        if not trainable_lora:
            raise RuntimeError(
                f"FAIL: PeftModel.from_pretrained({adapter_dir!s}) loaded "
                f"{len(lora_params)} `lora_` parameter tensors but NONE has "
                "requires_grad=True after model.train(). The continue-adapter "
                "branch would do a no-op fine-tune. Check PEFT version + adapter "
                "config; this contract is enforced by "
                "tests/test_issue475_common.py::test_continue_adapter_branch_keeps_lora_trainable."
            )
        lora_config = None  # signals "do NOT pass peft_config to SFTTrainer"
    else:
        # LoRA target modules: callers can pin a subset (e.g. issue #478/#490's
        # attn-only non-saturating anchor: ["q_proj","k_proj","v_proj","o_proj"]).
        # When unset, use the historical 7-module list (q/k/v/o + MLP) for
        # byte-identical backward compatibility with every pre-#478 caller.
        _DEFAULT_LORA_TARGETS = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        effective_lora_targets = (
            list(cfg.lora_targets) if cfg.lora_targets else list(_DEFAULT_LORA_TARGETS)
        )
        logger.info(
            "LoRA target_modules = %s (custom=%s)",
            effective_lora_targets,
            cfg.lora_targets is not None,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=effective_lora_targets,
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
    if cfg.save_only_model:
        sft_kwargs["save_only_model"] = True
    if cfg.completion_only_loss:
        # Plumb through SFTConfig(completion_only_loss=True). Available on
        # TRL >=0.14; for prompt-completion datasets (Arm A) TRL builds the
        # completion_mask via apply_chat_template length diff; for
        # pre-tokenized datasets (Arm B) the caller supplies completion_mask.
        sft_kwargs["completion_only_loss"] = True
    if cfg.dataset_kwargs is not None:
        # Plumb through SFTConfig(dataset_kwargs=...). Used by issue #498/#528
        # Arm B to set {"skip_prepare_dataset": True} so _prepare_dataset does
        # not re-tokenize the pre-tokenized rows (which would lose the
        # role-header bytes Qwen drops under apply_chat_template).
        sft_kwargs["dataset_kwargs"] = dict(cfg.dataset_kwargs)

    sft_config = SFTConfig(**sft_kwargs)

    sft_trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if lora_config is not None:
        # Fresh-LoRA path — TRL wraps the model.
        sft_trainer_kwargs["peft_config"] = lora_config
    # else: continue-adapter path — model is already a PeftModel; do NOT
    # pass peft_config (would re-wrap with a fresh adapter and lose the
    # continuation contract).
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
            suppress_at_post_response_slot=cfg.marker_suppress_at_post_response_slot,
            im_end_token_id=cfg.marker_im_end_token_id,
        )

    _maybe_wrap_recipient_eos_collator(trainer, tokenizer, cfg)
    _maybe_attach_marker_band_stop(trainer, tokenizer, cfg, str(_data_path))

    train_completed = False
    try:
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
                logger.warning(
                    "Adapter upload failed (%s) — local copy preserved at %s", e, output_dir
                )
        train_completed = True
    finally:
        # Per-call WandB run lifecycle (#527): finish the run THIS call
        # created (trainer-initiated under report_to="wandb", or the
        # best-effort upload run) so the next in-process train_lora call
        # re-inits with its own run_name instead of logging into a stale,
        # step-rewound run that WandB silently drops. Runs the exception
        # path too — a crashed cell must not leak its run into the next
        # cell. Caller-owned (pre-existing) runs are never touched.
        _finish_wandb_run_if_owned(wandb_run_preexisting, exit_code=0 if train_completed else 1)

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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _warn_if_cvd_disagrees(gpu_id)
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
