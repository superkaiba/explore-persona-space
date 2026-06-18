# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #601 — per-row-type training-CE probe (plan §4 Phase 2 bullet).

The REGISTERED data-exhaustion discriminator: every eval step, read the
training cross-entropy on a fixed sample of POSITIVE train rows (marker-token
CE at the marker slot — the same teacher-forced read the band callback makes,
restricted to its loss token) AND a fixed sample of NEGATIVE train rows
(trailing-token CE at the row's single loss-bearing token under
``MarkerOnlyDataCollator(tail_tokens=0, suppress=False)`` — the trailing valid
completion token, i.e. the newline after ``<|im_end|>``).

Exhaustion predicts positive marker CE → ~0 within the first distinct-row
exposure budget (no gradient left; the level flatlines for free); equilibrium /
horizon predict CE stays multiple nats from 0 at every plateau (the parent's
P(※) ≪ 1 signature). Without this probe the schedule-matched arm's landing
cannot distinguish a ratio set-point from same-distinct-rows-exhausted.

This EXTENDS the band-callback probe machinery (``train/sft.py``'s fused-render
tokenization helpers + the same teacher-forced slot read) rather than inventing
a parallel rig; it is a separate TrainerCallback only because Phase 3
(negatives-only) has zero marker rows, where the band callback cannot attach.

#613 (alive-negatives A/B) adds an OPT-IN third channel ``neg_slot``: the same
negative rows read at the post-response ``<|im_end|>`` slot — the token the
flag-on collator (``suppress_at_post_response_slot=True``) actually trains —
so the R1 manipulation check can verify the relocated loss channel is live
from step 1. Defaults leave all 2-channel callers byte-identical.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from transformers import TrainerCallback

from explore_persona_space.train.sft import (
    _apply_chat_template_safe,
    _tokenize_probe_row,
)

log = logging.getLogger("issue_601.rowtype_ce_probe")


def _tokenize_negative_row(
    row: dict,
    tokenizer,
    marker_seq: list[int],
    max_length: int,
) -> tuple[list[int], int, int] | None:
    """Tokenize one NEGATIVE (marker-less) JSONL row into (ids, slot, target_id).

    ONE fused ``apply_chat_template`` render (same source-of-truth contract as
    ``_tokenize_probe_row`` — two-call renders inject a phantom system prompt).
    The negative row's single loss token under the canonical collator is the
    TRAILING valid completion token (``ids[-1]``, the newline after
    ``<|im_end|>`` in the Qwen-2.5 tail layout); its predictive slot is
    ``len(ids) - 2``. Returns None for malformed / over-long / marker-bearing
    rows (marker-bearing → it is a positive, not a negative).
    """
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return None
    full_ids = _apply_chat_template_safe(
        tokenizer, prompt + completion, add_generation_prompt=False
    )
    if full_ids is None or len(full_ids) < 2 or len(full_ids) > max_length:
        return None
    # Marker subsequence present → positive row; skip.
    n = len(marker_seq)
    for i in range(len(full_ids) - n + 1):
        if full_ids[i : i + n] == marker_seq:
            return None
    return full_ids, len(full_ids) - 2, full_ids[-1]


def _tokenize_negative_row_post_slot(
    row: dict,
    tokenizer,
    marker_seq: list[int],
    max_length: int,
    im_end_id: int,
) -> tuple[list[int], int, int] | None:
    """Tokenize one NEGATIVE row at the POST-RESPONSE slot (#613 flag-on channel).

    The #474/#613 flag-on collator branch puts the negative row's loss at the
    FIRST ``<|im_end|>`` in the COMPLETION region (the slot greedy generation
    stopped at) — NOT the trailing newline that :func:`_tokenize_negative_row`
    targets. This helper mirrors that pick on the SAME single fused
    ``apply_chat_template`` render (one-call contract, no phantom system
    prompt, no BPE boundary-merge fragility): the completion region starts
    after the LAST ``<|im_start|>`` (the assistant turn is the final message
    of ``prompt + completion``), and the first ``im_end_id`` after it is the
    loss-bearing token.

    Returns ``(ids, slot, im_end_id)`` where ``ids`` is the prefix ending at
    the found ``<|im_end|>`` (index ``i``, asserted ``ids[i] == im_end_id``)
    and ``slot = i - 1`` is the OUTPUT slot whose next-token distribution
    predicts it. Returns None for malformed / over-long / marker-bearing rows
    or when the layout doesn't expose the slot (the caller fails loud on an
    all-None channel).
    """
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return None
    full_ids = _apply_chat_template_safe(
        tokenizer, prompt + completion, add_generation_prompt=False
    )
    if full_ids is None or len(full_ids) < 2 or len(full_ids) > max_length:
        return None
    # Marker subsequence present → positive row; skip (mirrors _tokenize_negative_row).
    n = len(marker_seq)
    for i in range(len(full_ids) - n + 1):
        if full_ids[i : i + n] == marker_seq:
            return None
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id is None or im_start_id < 0:
        return None
    # Assumes SINGLE-message completions (one assistant turn — valid on this
    # rig's builder output); a multi-message completion would open earlier
    # completion turns before the LAST <|im_start|> and their slots would be
    # skipped (round-2 Claude reviewer minor 2).
    starts = [i for i, t in enumerate(full_ids) if t == im_start_id]
    if not starts:
        return None
    for i in range(starts[-1] + 1, len(full_ids)):
        if full_ids[i] == im_end_id:
            if i - 1 <= 0:
                return None
            ids = full_ids[: i + 1]
            assert ids[i] == im_end_id, (ids[i], im_end_id)
            return ids, i - 1, im_end_id
    return None


def build_rowtype_probes(  # noqa: C901 -- one linear row scan feeding three channel collectors; splitting would scatter the shared row-sampling contract
    data_path: str | Path,
    tokenizer,
    marker_token_ids: list[int],
    *,
    n_pos: int = 16,
    n_neg: int = 16,
    max_length: int = 2048,
    neg_post_response_slot: bool = False,
    im_end_token_id: int | None = None,
) -> dict:
    """Build the fixed positive/negative probe batches from the training JSONL.

    Positives reuse ``_tokenize_probe_row`` (band-callback machinery): CE
    target = the first marker token at its slot. Negatives use
    :func:`_tokenize_negative_row`: CE target = the trailing valid completion
    token. Either side may come back EMPTY (e.g. Phase 3 has zero positives) —
    the callback degrades to the populated side.

    #613: with ``neg_post_response_slot=True`` (requires ``im_end_token_id``),
    a THIRD channel ``"neg_slot"`` is added — the SAME negative rows read at
    the post-response ``<|im_end|>`` slot (:func:`_tokenize_negative_row_post_slot`),
    i.e. the flag-on collator's actual loss token. The trailing ``"neg"``
    channel is kept UNCHANGED for the parent-comparable join. Defaults
    (``False``/``None``) leave existing 2-channel callers byte-identical
    (no ``"neg_slot"`` key in the returned dict).

    Returns:
        ``{"pos": batch | None, "neg": batch | None[, "neg_slot": batch]}``
        where ``batch`` = ``{"input_ids": LongTensor [B, T], "attention_mask":
        [B, T], "positions": [B], "target_ids": [B], "n_rows": int}``.
    """
    import torch

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"rowtype probe: training data missing at {path}")
    if not marker_token_ids:
        raise ValueError("rowtype probe: non-empty marker_token_ids required")
    if neg_post_response_slot and im_end_token_id is None:
        raise ValueError(
            "rowtype probe: neg_post_response_slot=True requires im_end_token_id "
            "(the post-response slot token id, e.g. 151645 for Qwen-2.5)."
        )
    marker_seq = list(marker_token_ids)
    marker_id = marker_seq[0]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    pos_rows: list[tuple[list[int], int, int]] = []
    neg_rows: list[tuple[list[int], int, int]] = []
    neg_slot_rows: list[tuple[list[int], int, int]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(pos_rows) < n_pos:
                picked = _tokenize_probe_row(row, tokenizer, marker_seq, max_length)
                if picked is not None:
                    ids, slot = picked
                    pos_rows.append((ids, slot, marker_id))
                    continue
            if len(neg_rows) < n_neg:
                picked_neg = _tokenize_negative_row(row, tokenizer, marker_seq, max_length)
                if picked_neg is not None:
                    neg_rows.append(picked_neg)
                    if neg_post_response_slot:
                        picked_slot = _tokenize_negative_row_post_slot(
                            row, tokenizer, marker_seq, max_length, im_end_token_id
                        )
                        if picked_slot is not None:
                            neg_slot_rows.append(picked_slot)
            if len(pos_rows) >= n_pos and len(neg_rows) >= n_neg:
                break

    def _pad(rows: list[tuple[list[int], int, int]]) -> dict | None:
        if not rows:
            return None
        t_max = max(len(r[0]) for r in rows)
        input_ids = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), t_max), dtype=torch.long)
        positions = torch.zeros(len(rows), dtype=torch.long)
        target_ids = torch.zeros(len(rows), dtype=torch.long)
        for i, (ids, slot, tgt) in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            positions[i] = slot
            target_ids[i] = tgt
        assert input_ids.shape == attention_mask.shape, (input_ids.shape, attention_mask.shape)
        assert positions.shape == (input_ids.shape[0],), positions.shape
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "positions": positions,
            "target_ids": target_ids,
            "n_rows": len(rows),
        }

    out = {"pos": _pad(pos_rows), "neg": _pad(neg_rows)}
    if neg_post_response_slot:
        # Fail-loud (plan #613 §4): negative rows exist but NONE exposed the
        # post-response slot — a layout / tokenization bug, not a degradable
        # condition (a silently absent neg_slot channel would void the R1
        # manipulation check and only surface at the smoke gate).
        if neg_rows and not neg_slot_rows:
            raise ValueError(
                "rowtype probe: neg_post_response_slot=True but no negative row "
                "exposed a post-response <|im_end|> slot — layout/tokenization "
                "drift (cross-check tests/test_marker_only_collator_post_response_slot.py)."
            )
        out["neg_slot"] = _pad(neg_slot_rows)
    return out


def _batch_ce(model, batch: dict) -> object:
    """One teacher-forced forward pass; per-row CE = −log P(target at slot).

    Returns a CPU float tensor of shape ``[B]``. Same slot-read convention as
    ``MarkerBandStopCallback._compute_marker_slot_stats`` (positions index the
    OUTPUT slot whose next-token distribution predicts the target).
    """
    import torch

    device = getattr(model, "device", None) or next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    positions = batch["positions"].to(device)
    target_ids = batch["target_ids"].to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        assert logits.ndim == 3, logits.shape
        b_idx = torch.arange(input_ids.shape[0], device=device)
        slot_logits = logits[b_idx, positions, :].float()  # [B, V]
        log_probs = torch.log_softmax(slot_logits, dim=-1)
        ce = -log_probs[b_idx, target_ids]  # [B]
        assert ce.shape == (input_ids.shape[0],), ce.shape
        return ce.detach().cpu()
    finally:
        if was_training:
            model.train()


class RowTypeCETrainProbeCallback(TrainerCallback):
    """Per-eval-step positive-marker-CE + negative-trailing-CE trajectory.

    Logs ``<prefix>/pos_marker_ce`` and ``<prefix>/neg_trailing_ce`` to WandB
    each eval step and atomically rewrites ``out_path`` (checkpoint-per-phase
    discipline — a mid-run crash never loses the series). Base-side CE is
    cached on the first eval with the adapter disabled (PEFT
    ``disable_adapter()``), mirroring the band callback.

    #613: when the probes dict carries the optional ``"neg_slot"`` channel
    (post-response ``<|im_end|>`` slot — the flag-on collator's loss token),
    it is read each eval step alongside ``pos``/``neg`` and logged as
    ``<prefix>/neg_slot_ce``; the trailing ``neg`` channel stays unchanged
    for the parent-comparable join.
    """

    def __init__(
        self,
        probes: dict,
        *,
        out_path: str | Path,
        eval_every_steps: int = 1,
        dense_until: int = 0,
        log_prefix: str = "rowtype_ce",
    ):
        if probes.get("pos") is None and probes.get("neg") is None:
            raise ValueError(
                "RowTypeCETrainProbeCallback: BOTH probe sides empty — the training "
                "data has neither marker-bearing nor marker-less rows; refusing a "
                "silent no-op probe (plan §4 registered discriminator)."
            )
        if eval_every_steps < 1:
            raise ValueError(f"eval_every_steps must be >= 1, got {eval_every_steps}")
        if dense_until < 0:
            raise ValueError(f"dense_until must be >= 0, got {dense_until}")
        self.pos = probes.get("pos")
        self.neg = probes.get("neg")
        self.neg_slot = probes.get("neg_slot")
        self.out_path = str(out_path)
        self.eval_every_steps = int(eval_every_steps)
        # #622 strided cadence: probe EVERY step while global_step <= dense_until,
        # then every eval_every_steps. Default 0 = legacy stride-only gating.
        self.dense_until = int(dense_until)
        self.log_prefix = log_prefix
        self._base: dict[str, float | None] = {"pos": None, "neg": None, "neg_slot": None}
        self._records: list[dict] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self._records = []
        self._base = {"pos": None, "neg": None, "neg_slot": None}
        log.info(
            "[%s] probe attached: %s pos rows, %s neg rows, %s neg_slot rows, eval_every=%d",
            self.log_prefix,
            self.pos["n_rows"] if self.pos else 0,
            self.neg["n_rows"] if self.neg else 0,
            self.neg_slot["n_rows"] if self.neg_slot else 0,
            self.eval_every_steps,
        )

    def _ce_with_base(self, model, batch: dict) -> float:
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            with disable_adapter():
                return float(_batch_ce(model, batch).mean().item())
        return float(_batch_ce(model, batch).mean().item())

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        # Strided cadence (#622): dense (every step) through dense_until, then
        # every eval_every_steps. dense_until=0 = legacy stride-only gating.
        if state.global_step > self.dense_until and state.global_step % self.eval_every_steps != 0:
            return
        rec: dict = {"step": int(state.global_step)}
        metrics: dict[str, float] = {}
        side_keys = {"pos": "pos_marker_ce", "neg": "neg_trailing_ce", "neg_slot": "neg_slot_ce"}
        for side, batch in (("pos", self.pos), ("neg", self.neg), ("neg_slot", self.neg_slot)):
            if batch is None:
                continue
            if self._base[side] is None:
                self._base[side] = self._ce_with_base(model, batch)
            ce = float(_batch_ce(model, batch).mean().item())
            key = side_keys[side]
            rec[key] = ce
            rec[f"{key}_base"] = self._base[side]
            metrics[f"{self.log_prefix}/{key}"] = ce
        self._records.append(rec)
        self._flush()
        try:
            import wandb

            if wandb.run is not None and metrics:
                wandb.log(metrics, step=state.global_step)
        except Exception as e:  # pragma: no cover - wandb optional
            log.info("wandb rowtype-CE log skipped (%s)", e)

    def on_train_end(self, args, state, control, **kwargs):
        if self._records:
            self._flush()
            log.info(
                "[%s] rowtype-CE final flush: %d records -> %s",
                self.log_prefix,
                len(self._records),
                self.out_path,
            )

    def _flush(self) -> None:
        payload = {
            # v2 ONLY when the #613 neg_slot channel is present; parent files
            # stay v1 byte-shape (parent-comparable join, plan #613 §4).
            "schema": "i601_rowtype_ce_v2" if self.neg_slot is not None else "i601_rowtype_ce_v1",
            "n_pos_rows": self.pos["n_rows"] if self.pos else 0,
            "n_neg_rows": self.neg["n_rows"] if self.neg else 0,
            "steps": [r["step"] for r in self._records],
            "pos_marker_ce": [r.get("pos_marker_ce") for r in self._records],
            "neg_trailing_ce": [r.get("neg_trailing_ce") for r in self._records],
            "pos_marker_ce_base": self._base["pos"],
            "neg_trailing_ce_base": self._base["neg"],
            "records": self._records,
        }
        if self.neg_slot is not None:
            payload["n_neg_slot_rows"] = self.neg_slot["n_rows"]
            payload["neg_slot_ce"] = [r.get("neg_slot_ce") for r in self._records]
            payload["neg_slot_ce_base"] = self._base["neg_slot"]
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, self.out_path)
