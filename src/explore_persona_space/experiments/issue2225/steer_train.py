"""Issue #2225 — training-time steering hook (plan §4.4).

Adds ``alpha * v_layer`` to decoder-block OUTPUT hidden states at masked
positions during TRAINING forwards only. Semantic anchor for ``mode="all"``:
the pinned paper repo's ``training.py::steering_intervention``
(external/persona_vectors @ b8e0f04) — ``act = act + steering_coef *
Q.unsqueeze(0)``, an UNMASKED all-position add on the block-output tuple's
first element, registered as a forward hook on ``model.layers.{L-1}`` with
PEFT-aware module-path resolution (plan §12 A5, verified at implementation
time 2026-08-10). Our ``mode="all"`` masks to ``attention_mask == 1``; under
TRL's right padding (``DataCollatorForLanguageModeling`` hardcodes
``padding_side="right"``) pad positions carry labels ``-100`` and are never
attended by any real position (causal + attention mask), so editing them is
gradient-inert — the ``attention_mask == 1`` add reproduces the paper's
training gradient exactly.

Position masks per batch row (plan §4.4, under ``completion_only_loss``):

- ``context``  = ``attention_mask == 1  &  labels == -100`` (prompt tokens)
- ``response`` = ``labels != -100`` (completion tokens)
- ``all``      = ``attention_mask == 1``
- ``prefix``   = positions ``< prefix_len`` — ``prefix_len`` (system segment
  incl. its template tokens) computed at dataset-map time from per-segment
  TOKEN IDS via the issue1415 ``<|im_start|>`` boundary convention (NEVER by
  re-tokenizing a concatenated string), threaded to the hook via a
  ``prefix_len`` batch column the trainer pops before ``model(**inputs)``.
- ``context_end`` = one-hot at the LAST context position per row (the final
  prompt token — the assistant-header tail, #779's ``c_last`` map-input slot;
  #2225 fu1 plan §4.2).

Threading the ``prefix_len`` column requires BOTH (verified against the
installed TRL 0.29.1 source):

1. ``SteeredSFTTrainer._set_signature_columns_if_needed`` — the stock
   signature-column list (``sft_trainer.py:1198``) omits ``prefix_len``, so
   ``remove_unused_columns=True`` (the Trainer default) would strip it from
   the tokenized dataset before the collator ever sees it.
2. ``SteeringDataCollator`` — the stock
   ``DataCollatorForLanguageModeling.torch_call`` emits ONLY
   input_ids/labels/attention_mask (+/- position_ids), silently dropping any
   other per-example key, so a passthrough subclass re-attaches the batch
   tensor.

Engagement breadcrumb: ``install()`` prints one
``[steer-hook] mode=<m> layers=<k> alpha=<a>`` line (flush=True) — the
per-cell P0 gate greps for it (plan §7 criterion 1).
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import torch
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks
from explore_persona_space.experiments.issue1415.steering import prefix_end_index

MaskMode = Literal["all", "context", "response", "prefix", "context_end"]
MASK_MODES: tuple[str, ...] = ("all", "context", "response", "prefix", "context_end")


# ── position masks ─────────────────────────────────────────────────────────────


def masks_for_mode(
    mode: str,
    *,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    prefix_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row (B, T) bool mask of positions to steer, per plan §4.4.

    Fail-loud integrity asserts (right-padding invariants): ``response``
    positions must be real tokens, and ``prefix`` positions must be real
    tokens (a left-padded batch would fail the latter loudly instead of
    silently steering pad slots).
    """
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask mode {mode!r}; expected one of {MASK_MODES}")
    assert attention_mask.shape == labels.shape, (attention_mask.shape, labels.shape)
    real = attention_mask == 1
    if mode == "all":
        return real
    if mode == "context":
        return real & (labels == -100)
    if mode == "context_end":
        # fu1 (#2225 plan §4.2): one-hot per-row mask at the LAST position where
        # attention_mask==1 & labels==-100 — the final prompt token (under TRL
        # prompt-completion tokenization with add_generation_prompt, the tail of
        # the assistant header: the parent's E2 capture slot and #779's c_last
        # map-input frame). Right padding (TRL default) keeps that position the
        # row-wise argmax of the context mask.
        ctx = real & (labels == -100)
        assert bool(ctx.any(dim=1).all()), (
            "context_end: a row has NO context position (attention_mask==1 & "
            "labels==-100 empty) — completion_only_loss drift or a degenerate row"
        )
        B, T = ctx.shape
        pos = torch.arange(T, device=ctx.device)
        last_ctx = (ctx * (pos + 1).unsqueeze(0)).argmax(dim=1)  # last True per row
        mask = torch.zeros_like(ctx)
        mask[torch.arange(B, device=ctx.device), last_ctx] = True
        # Fail-loud invariants (plan §4.2): exactly one position per row;
        # context_end ⊆ context (which also implies ⊆ attention_mask==1).
        assert bool((mask.sum(dim=1) == 1).all()), mask.sum(dim=1).tolist()
        assert bool((mask & ~ctx).sum() == 0), (
            "context_end position outside the context mask — padding-side violation?"
        )
        return mask
    if mode == "response":
        resp = labels != -100
        assert bool((resp & ~real).sum() == 0), (
            "response positions outside attention_mask==1 — padding-side violation?"
        )
        return resp
    # mode == "prefix"
    if prefix_len is None:
        raise ValueError(
            "mode='prefix' requires a prefix_len batch column (dataset-map it via "
            "compute_prefix_len + thread through SteeringDataCollator)"
        )
    B, T = attention_mask.shape
    assert prefix_len.shape == (B,), (prefix_len.shape, B)
    assert bool((prefix_len >= 1).all()) and bool((prefix_len <= T).all()), (
        f"prefix_len out of range [1, {T}]: {prefix_len.tolist()}"
    )
    pos = torch.arange(T, device=attention_mask.device)
    mask = pos.unsqueeze(0) < prefix_len.to(attention_mask.device).unsqueeze(1)
    assert bool((mask & ~real).sum() == 0), (
        "prefix positions outside attention_mask==1 — left-padded batch? "
        "The prefix positional mask requires right padding (TRL default)."
    )
    return mask


def compute_prefix_len(tokenizer: PreTrainedTokenizerBase, prompt_messages: list[dict]) -> int:
    """Token length of the SYSTEM segment (incl. template tokens) of a prompt.

    Dataset-map-time helper for ``mode="prefix"``: tokenizes the prompt via the
    SAME ``apply_chat_template(add_generation_prompt=True, tokenize=True)``
    call TRL 0.29's prompt-completion tokenizer uses (``sft_trainer.py:1057``),
    then locates the prefix/user boundary on the TOKEN IDS via the issue1415
    ``<|im_start|>`` special-token convention (``prefix_end_index`` asserts the
    3-occurrence single-turn shape; special tokens are atomic so the boundary
    can never BPE-merge). The Qwen chat template inserts its default system
    block when the row carries no explicit system turn, so #778 rows
    (user-only prompts) still resolve a non-trivial prefix.
    """
    ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True)
    if ids and isinstance(ids[0], list):  # some tokenizers return a nested list
        ids = ids[0]
    return prefix_end_index(tokenizer, ids)


# ── layer-incremental vectors (plan §4.2 / App. J.3 band extension) ────────────


def build_incremental_vectors(vectors: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """``v_inc_l = v_l - v_{l-1}`` over a CONTIGUOUS layer band; band-start
    ``v_inc_s = v_s`` (paper App. J.3, extended to a contiguous band — the
    telescoping sum makes the in-band cumulative steer equal ``v_l`` at each
    in-band layer). Raises on a non-contiguous key set.
    """
    layers = sorted(int(k) for k in vectors)
    if not layers:
        raise ValueError("empty vectors dict")
    if layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError(f"layer-incremental vectors need a contiguous band, got {layers}")
    out: dict[int, torch.Tensor] = {layers[0]: vectors[layers[0]].clone()}
    for lo, hi in itertools.pairwise(layers):
        out[hi] = vectors[hi] - vectors[lo]
    return out


# ── the steering hook ──────────────────────────────────────────────────────────


class SteeringHook:
    """Forward hooks adding ``alpha * v_layer`` at masked positions (plan §4.4).

    ``vectors`` maps 0-indexed decoder-block indices (the #778 r_B convention:
    index 19 == paper layer 20) to ``(H,)`` direction tensors. The add is
    OUT-OF-PLACE on the block-output tuple's first element (the paper's
    ``steering_intervention`` shape), broadcast over the batch, computed in the
    activation dtype (bf16 in production).

    Lifecycle: ``install(model)`` registers one forward hook per layer through
    the PEFT-aware ``_resolve_decoder_blocks`` chain walk (a ``PeftModel``
    resolves at depth 2 to the LoRA-active decoder blocks — the same objects
    ``get_base_model()`` returns), prints the ``[steer-hook]`` engagement
    breadcrumb, and returns ``self``; ``remove()`` deregisters. TRAINING
    forwards only: ``SteeredSFTTrainer.train`` installs/removes around
    ``super().train()``, and a forward with no armed mask fails loud.
    """

    def __init__(self, vectors: dict[int, torch.Tensor], alpha: float, mode: str):
        if mode not in MASK_MODES:
            raise ValueError(f"unknown mask mode {mode!r}; expected one of {MASK_MODES}")
        if not vectors:
            raise ValueError("SteeringHook needs at least one layer vector")
        self.vectors: dict[int, torch.Tensor] = {}
        for layer, v in vectors.items():
            v = torch.as_tensor(v)
            assert v.dim() == 1, f"layer {layer}: expected (H,) vector, got {tuple(v.shape)}"
            self.vectors[int(layer)] = v
        self.alpha = float(alpha)
        self.mode = mode
        # Armed per micro-batch by SteeredSFTTrainer.compute_loss; cleared after.
        self.current_batch_masks: torch.Tensor | None = None
        self._handles: list = []
        self.n_edits = 0  # forward passes edited (telemetry / test hook)

    # -- lifecycle ------------------------------------------------------------
    def install(self, model) -> SteeringHook:
        assert not self._handles, "SteeringHook already installed"
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, (
            "SteeringHook requires a standard decoder layout (model[.model...].layers); "
            "_resolve_decoder_blocks found none"
        )
        for layer in sorted(self.vectors):
            assert 0 <= layer < len(blocks), (layer, len(blocks))
            self._handles.append(blocks[layer].register_forward_hook(self._make_hook(layer)))
        # Engagement breadcrumb — the P0 gate greps each cell log for this line.
        print(
            f"[steer-hook] mode={self.mode} layers={len(self.vectors)} alpha={self.alpha:g}",
            flush=True,
        )
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    # -- the edit ---------------------------------------------------------------
    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            return self._apply(layer, output)

        return hook

    def _apply(self, layer: int, output):
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        mask = self.current_batch_masks
        if mask is None:
            raise RuntimeError(
                "[steer-hook] forward fired with no armed batch mask — steering hooks "
                "are TRAINING-only; run forwards through SteeredSFTTrainer.compute_loss "
                "(or arm current_batch_masks explicitly)"
            )
        assert mask.shape == act.shape[:2], (mask.shape, act.shape)
        v = self.vectors[layer]
        if v.device != act.device or v.dtype != act.dtype:
            v = v.to(device=act.device, dtype=act.dtype)
            self.vectors[layer] = v  # cache the cast once
        act = act + mask.unsqueeze(-1).to(act.dtype) * (self.alpha * v)
        self.n_edits += 1
        if is_tuple:
            return (act, *output[1:])
        return act


# ── TRL wiring ─────────────────────────────────────────────────────────────────


class SteeringDataCollator(DataCollatorForLanguageModeling):
    """TRL 0.29 LM collator + ``prefix_len`` passthrough.

    The stock ``torch_call`` reads only specific per-example keys, so the
    ``prefix_len`` column added at dataset-map time would be silently dropped;
    this subclass re-attaches it as a ``(B,)`` long tensor when present.
    """

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        if examples and "prefix_len" in examples[0]:
            batch["prefix_len"] = torch.tensor(
                [int(ex["prefix_len"]) for ex in examples], dtype=torch.long
            )
        return batch


class SteeredSFTTrainer(SFTTrainer):
    """Thin SFTTrainer wrapper arming the steering hook per micro-batch.

    ``compute_loss`` pops the ``prefix_len`` batch column (always, so it never
    reaches ``model(**inputs)``), computes the mode mask from the batch
    tensors, arms ``hook.current_batch_masks``, delegates, and clears the mask
    in a ``finally`` so any forward outside compute_loss fails loud.
    ``train()`` installs the hooks on ``self.model`` — the PEFT-WRAPPED trainer
    model (SFTTrainer applies ``peft_config`` in ``__init__``) — and removes
    them in a ``finally``, so post-train eval/generation sees no hooks.
    """

    def __init__(self, *args, steering_hook: SteeringHook, **kwargs):
        self._steering_hook = steering_hook
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns is not None and "prefix_len" not in self._signature_columns:
            # remove_unused_columns=True strips non-signature columns from the
            # tokenized dataset (TRL 0.29 sft_trainer.py:1198 omits prefix_len).
            self._signature_columns = [*self._signature_columns, "prefix_len"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prefix_len = inputs.pop("prefix_len", None)
        hook = self._steering_hook
        mask = masks_for_mode(
            hook.mode,
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            prefix_len=prefix_len,
        )
        if not bool(mask.any()):
            # Zero-coverage guard (g1 Concern 2): an all-empty steering mask
            # (e.g. completion_only_loss drifting to False makes the "context"
            # mask vacuous) would train the cell with ZERO steering while the
            # [steer-hook] install breadcrumb already printed — the silent-null
            # -steering channel the P0 grep cannot catch. Fail loud instead.
            raise RuntimeError(
                f"steering mask empty for mode={hook.mode!r} across the whole "
                f"batch (B,T={tuple(mask.shape)}) — the cell would train with "
                "zero steering; check completion_only_loss / the dataset split"
            )
        hook.current_batch_masks = mask
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            hook.current_batch_masks = None

    def train(self, *args, **kwargs):
        self._steering_hook.install(self.model)
        try:
            return super().train(*args, **kwargs)
        finally:
            self._steering_hook.remove()
