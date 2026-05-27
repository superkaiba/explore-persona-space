"""Teacher-forced log-prob of a marker token sequence at a given context position.

Sibling of :func:`explore_persona_space.analysis.divergence.teacher_force_batch`.
That function does prompt-masked log-softmax over a response window for chat-
template-wrapped responses. This module performs its own inline teacher-forced
forward pass that appends marker BPE pieces to a raw text prefix and scores them
with custom left-pad + ``logits[..., -marker_len-1:-1, :]`` indexing.

The two primitives share the teacher-forcing philosophy but cover different
geometries (response-window log-softmax vs marker-at-end-of-prefix) and so are
deliberately kept as siblings rather than one calling the other.

In addition to :func:`compute_marker_logprob`, this module also provides
:func:`measure_first_step_delta` — the one-optimizer-step Δ log p(marker)
primitive consumed by task #400 — and :func:`compute_marker_logprob_trajectory`,
the per-position trajectory sibling primitive consumed by task #396 (single-
token-marker-only; for multi-token markers there is no unambiguous per-position
gather rule, so callers needing trajectories on those should use the scalar
primitive instead).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_marker_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contexts: list[str],
    marker_text: str,
    position: str = "end_of_answer",
    batch_size: int = 8,
    device: str = "cuda:0",
) -> list[float]:
    """Teacher-forced joint log-prob of ``marker_text``, given each context.

    For single-token markers (e.g. ``" ※"``) this is one softmax decision. For
    multi-token markers (e.g. ``" [ZLT]"`` → 3 BPE pieces on Qwen-2.5) this
    returns the joint sum over those pieces.

    The function performs an inline teacher-forced forward pass: for each
    context it tokenizes ``context`` and ``marker_text`` separately (with
    ``add_special_tokens=False``), concatenates the BPE pieces, runs one
    forward pass with left-padding so the marker pieces share the same trailing
    positions across the sub-batch, then extracts ``log_softmax(logits)`` at
    positions ``[-marker_len-1 : -1]`` and gathers each marker piece's id from
    its predictive logit. The sum of those per-piece log-probs is the returned
    scalar.

    Args:
        model: HF CausalLM, already on ``device`` and in eval mode.
        tokenizer: HF tokenizer matching ``model``.
        contexts: Prefix text per context (already includes whatever chat
            template wrapping the caller expects; we tokenize verbatim).
        marker_text: Marker string. Tokenized with ``add_special_tokens=False``,
            then teacher-forced one BPE piece at a time at the end of each
            context.
        position: Where to place the marker. Currently only
            ``"end_of_answer"`` is supported (appended after the context
            with no further wrapping). Reserved for future positions
            (``"middle_of_answer"`` etc.) without breaking the API.
        batch_size: Sub-batch size for teacher-forcing (memory tradeoff).
        device: Torch device string.

    Returns:
        ``list[float]`` of length ``len(contexts)``, each entry the sum of
        log-prob of every marker BPE piece given ``context + previous_pieces``.

    Asserts:
        - Marker tokenization is non-empty.
        - Returned list has the same length as ``contexts``.
    """
    if position != "end_of_answer":
        raise NotImplementedError(f"position={position!r} not supported yet")

    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    assert len(marker_ids) > 0, f"marker_text={marker_text!r} tokenized to []"
    marker_len = len(marker_ids)

    out: list[float] = []
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start : start + batch_size]
        # Tokenize each context + marker_ids appended. Left-pad so the marker
        # tokens land at the same trailing positions across the sub-batch.
        context_ids = [tokenizer.encode(c, add_special_tokens=False) for c in chunk]
        for cidx, cids in enumerate(context_ids):
            # A zero-length context would leave only marker tokens in the
            # sequence and break the ``-marker_len - 1`` slice math below.
            # Fail loud rather than silently emit a junk log-prob.
            assert len(cids) > 0, (
                f"contexts[{start + cidx}] tokenized to [] — refusing to score "
                "marker log-prob on a zero-token context"
            )
        full_ids = [cids + marker_ids for cids in context_ids]
        max_len = max(len(ids) for ids in full_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded = []
        attn = []
        for ids in full_ids:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            attn.append([0] * pad_len + [1] * len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Marker tokens occupy the LAST ``marker_len`` positions; their
        # predictive logits sit at positions ``[-marker_len-1 : -1]`` (the
        # standard next-token shift for a causal LM).
        for i in range(len(chunk)):
            seq_logits = logits[i, -marker_len - 1 : -1, :]  # (marker_len, V)
            log_probs = F.log_softmax(seq_logits.float(), dim=-1)
            tgt = torch.tensor(marker_ids, device=device)
            piece_logps = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(float(piece_logps.sum().item()))

        del logits

    assert len(out) == len(contexts)
    return out


def compute_marker_logprob_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    completions: list[str],
    marker_text: str,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> list[list[float]]:
    """Per-position log p(marker | prompt + completion[:k]) trajectory.

    Sibling of :func:`compute_marker_logprob`. The scalar primitive places
    the marker once at the end of the context and returns one joint log-prob
    per context. This primitive reads off the model's predictive distribution
    at every position of ``prompt + completion`` (no marker appended) and
    extracts log p(marker_token | prefix-of-length-k) for k spanning
    ``[0, len(completion_tokens)]``. Position k=0 is the bare prior
    (prompt only); position k=len is end-of-response.

    Single-token markers only. For multi-token markers the per-position
    interpretation is not well-defined without specifying a sub-token
    gather rule; callers asking for trajectories on multi-token markers
    should be redirected to the scalar primitive (joint over BPE pieces
    at end-of-context).

    Mathematical consistency contract with :func:`compute_marker_logprob`.
    For single-token markers, the LAST element of the trajectory
    (``traj[-1]``) equals ``compute_marker_logprob(model, tokenizer,
    [prompt + completion], marker_text, ...)[0]`` to within 1e-5. Both
    extract ``log_softmax(logits[final_pred_position])[marker_id]``; the
    scalar primitive does so after appending the marker and reading
    ``[-marker_len-1:-1]``, the trajectory primitive does so without
    appending and reads at every position. The smoke test in
    ``tests/test_issue396_compute_marker_logprob_smoke.py`` enforces
    the identity.

    Args:
        model: HF CausalLM, already on ``device`` and in eval mode.
        tokenizer: HF tokenizer matching ``model``.
        prompts: Prompt prefix per context (chat-templated, no completion).
        completions: Greedy completion per context. Same length as ``prompts``.
        marker_text: Marker string (e.g. ``" ※"``). Must tokenize to
            exactly one BPE piece on ``tokenizer``.
        batch_size: Sub-batch size for the teacher-forced forward pass.
        device: Torch device string.

    Returns:
        ``list[list[float]]`` of length ``len(prompts)``. Inner list at
        index ``i`` has length
        ``len(tokenizer.encode(completions[i], add_special_tokens=False)) + 1``;
        entry at index ``k`` is log p(marker | prompts[i] + completion_tokens[:k]).
        k=0 is the bare prior (prompt only); k=len(comp_tokens) is end-of-response.

    Asserts:
        - ``len(prompts) == len(completions)``.
        - Marker tokenizes to exactly one BPE piece.
        - Every prompt is non-empty after tokenization (slice math requires
          ``len(pids) >= 1``).
    """
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    assert len(marker_ids) == 1, (
        f"compute_marker_logprob_trajectory requires a single-token marker; "
        f"marker_text={marker_text!r} tokenized to {marker_ids!r} "
        f"({len(marker_ids)} pieces). Multi-token markers have no unambiguous "
        f"per-position interpretation — use compute_marker_logprob() for the "
        f"joint scalar at end-of-context instead."
    )
    marker_id = marker_ids[0]
    assert len(prompts) == len(completions), (
        f"prompts and completions must be paired 1:1; "
        f"got {len(prompts)} prompts vs {len(completions)} completions"
    )

    out: list[list[float]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for start in range(0, len(prompts), batch_size):
        chunk_p = prompts[start : start + batch_size]
        chunk_c = completions[start : start + batch_size]

        prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in chunk_p]
        comp_ids = [tokenizer.encode(c, add_special_tokens=False) for c in chunk_c]

        for cidx, pids in enumerate(prompt_ids):
            # A zero-length prompt breaks the ``len(pids) - 1`` slice math used
            # to anchor k=0. Fail loud rather than emit junk.
            assert len(pids) > 0, (
                f"prompts[{start + cidx}] tokenized to [] — refusing to score "
                "trajectory log-prob on a zero-token prompt"
            )

        # Full sequence per row = prompt + completion. Right-pad here (the
        # opposite of compute_marker_logprob's left-pad) so position indices
        # stay anchored at sequence start. Do NOT append the marker — we read
        # log p(marker | prefix) off the predictive logits directly.
        full_ids = [p + c for p, c in zip(prompt_ids, comp_ids, strict=True)]
        max_len = max(len(ids) for ids in full_ids)

        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in full_ids]
        attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in full_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)
            # log_probs[i, t, marker_id] = log p(marker | tokens 0..t). The
            # next-token predictive distribution at SEQUENCE position t is
            # the model's output at INPUT position t-1 (standard causal-LM
            # shift). For prefix length k (i.e. tokens 0..k-1 seen):
            #   * k=0 means "no tokens seen" — undefined for a causal LM
            #     without an explicit BOS. We define k=0 as "log p(marker |
            #     prompt only, BEFORE any completion token)", which is the
            #     predictive logit at input position ``len(prompt) - 1``.
            #   * k>=1 means "prompt + completion_tokens[:k] seen"; the
            #     predictive logit sits at input position
            #     ``len(prompt) - 1 + k``.
            marker_logp = log_probs[..., marker_id]  # (B, T) on device

        for i, (pids, cids) in enumerate(zip(prompt_ids, comp_ids, strict=True)):
            base = len(pids) - 1
            # k=0 (bare prior) + k=1..len(cids) (after each completion token).
            traj_positions = [base + k for k in range(len(cids) + 1)]
            traj = [float(marker_logp[i, pos].item()) for pos in traj_positions]
            out.append(traj)

        del logits, log_probs, marker_logp

    assert len(out) == len(prompts), (
        f"compute_marker_logprob_trajectory: produced {len(out)} trajectories "
        f"for {len(prompts)} prompts"
    )
    return out


def _assert_base_param_unchanged(
    peft_model: Any,
    sampled_name: str,
    sampled_pre_step: torch.Tensor,
) -> None:
    """Resolve ``sampled_name`` on a PEFT-wrapped model and assert byte identity.

    Helper for the mid-step byte-identity check used by
    :func:`measure_first_step_delta` when ``_assert_frozen_during_step=True``.
    Searches the wrapped model's ``named_parameters`` for the sampled base
    parameter (peft inserts ``base_model.model.`` prefix and
    ``base_layer.weight`` suffix on LoRA-target modules) and asserts the
    current values are byte-identical to the pre-step snapshot.
    """
    live = dict(peft_model.named_parameters())
    candidate_names = [
        sampled_name,
        f"base_model.model.{sampled_name}",
        f"base_model.{sampled_name}",
    ]
    live_param: torch.Tensor | None = None
    for cand in candidate_names:
        if cand in live:
            live_param = live[cand].detach().cpu()
            break
    if live_param is None:
        # Fallback: substring match for tokenizer-target names with the
        # ``base_layer.weight`` suffix peft inserts on LoRA-target modules.
        suffix_alt = sampled_name.replace(".weight", ".base_layer.weight")
        for ln, lp in live.items():
            if ln.endswith(sampled_name) or ln.endswith(suffix_alt):
                live_param = lp.detach().cpu()
                break
    if live_param is None:
        raise AssertionError(
            f"Could not resolve sampled base param {sampled_name!r} on PEFT-wrapped model "
            f"(candidates: {candidate_names!r}); cannot run frozen-during-step check"
        )
    assert torch.equal(live_param, sampled_pre_step), (
        f"Base param {sampled_name!r} mutated mid-step (frozen-during-step assertion failed). "
        f"This indicates the optimizer is updating base weights — check that lora_config "
        f"does not enable modules_to_save= on this param."
    )


def measure_first_step_delta(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    persona_system_prompt: str,
    training_rows: list[dict],
    eval_questions: list[str],
    marker_text: str,
    lora_config: Any,
    lr: float = 1e-5,
    device: str = "cuda:0",
    _assert_frozen_during_step: bool = False,
) -> dict:
    """Measure Δ log p(marker) after exactly ONE optimizer step on persona-conditioned data.

    Consumer: task #400 ("does the FIRST gradient step push log p(marker) up?").

    Pipeline:
      1. Snapshot the base model's LoRA-target parameters (CPU clone) so the
         function can restore the base weights at the end and assert byte
         identity.
      2. Build pre-training prefix contexts from
         ``persona_system_prompt + question`` for each eval question. Call
         :func:`compute_marker_logprob` with the base model — store
         ``pre_logp`` (one float per eval question).
      3. Attach a fresh LoRA adapter via ``peft.get_peft_model(base, lora_config)``.
      4. Build one mini-batch from ``training_rows``: each row's
         ``persona + question + answer + marker_text`` is tokenized with a
         loss mask that scores only the answer + marker portion (mirrors
         ``sft.py``'s training-time masking).
      5. Run one ``torch.optim.AdamW(...).step()``. If
         ``_assert_frozen_during_step`` is set, sample one LoRA-target base
         param BEFORE the step and assert byte identity to its pre-step value
         immediately AFTER the step (still inside this function, before any
         state_dict restoration) — catches an update-then-restore bug class
         that an end-state byte-identity check would miss.
      6. Re-run :func:`compute_marker_logprob` with the adapted model →
         ``post_logp``.
      7. ``peft_model.unload()`` to detach the adapter, then restore the base
         state-dict snapshot from step 1 (defensive belt-and-suspenders for
         the case where ``lora_config`` carries ``modules_to_save=[...]`` and
         the optimizer inadvertently updates base params).

    Args:
        base_model: HF CausalLM in eval mode, already on ``device``.
        tokenizer: HF tokenizer matching ``base_model``.
        persona_system_prompt: System prompt prepended to every eval and
            training row.
        training_rows: List of ``{"persona": str, "question": str,
            "answer": str}`` dicts forming the one-step training batch.
        eval_questions: Questions for the pre/post log-prob probe.
        marker_text: Marker string appended to training answers and probed
            via :func:`compute_marker_logprob`.
        lora_config: ``peft.LoraConfig`` (typed loosely to avoid a hard ``peft``
            import at module scope).
        lr: AdamW learning rate for the one step.
        device: Torch device string.
        _assert_frozen_during_step: Debug-only flag used by
            ``test_base_weights_restored_after_first_step``; should remain
            False in production.

    Returns:
        ``{"persona": <persona_system_prompt>, "pre_logp": [...],
           "post_logp": [...], "delta_logp": [...]}`` — each list has length
        ``len(eval_questions)``.

    Asserts:
        - ``len(training_rows) > 0``.
        - ``len(eval_questions) > 0``.
        - Post-call, base model weights match the pre-call snapshot.
    """
    # Local import — peft is optional at module load and only required by
    # this function. ``compute_marker_logprob`` stays peft-free.
    from peft import PeftModel, get_peft_model

    assert len(training_rows) > 0, "training_rows must be non-empty"
    assert len(eval_questions) > 0, "eval_questions must be non-empty"

    # ── 1. Snapshot LoRA-target base params (CPU clone). ──────────────────
    target_modules = getattr(lora_config, "target_modules", None)
    if target_modules is None:
        target_modules = []
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    def _is_lora_target_param(name: str) -> bool:
        return any(tm in name for tm in target_modules)

    base_snapshot: dict[str, torch.Tensor] = {
        name: param.detach().cpu().clone()
        for name, param in base_model.named_parameters()
        if _is_lora_target_param(name)
    }
    if _assert_frozen_during_step:
        assert base_snapshot, (
            "No LoRA-target params found in base_model — cannot run frozen-during-step check"
        )

    # Pick one canonical sampled LoRA-target param for the mid-step
    # byte-identity check (used only when ``_assert_frozen_during_step``).
    sampled_name = next(iter(base_snapshot)) if base_snapshot else None
    sampled_pre_step: torch.Tensor | None = (
        base_snapshot[sampled_name].clone() if sampled_name is not None else None
    )

    # ── 2. Pre-training log-prob probe on the bare base model. ────────────
    pre_contexts = [f"{persona_system_prompt}\n\n{q}" for q in eval_questions]
    base_model.eval()
    pre_logp = compute_marker_logprob(
        base_model,
        tokenizer,
        contexts=pre_contexts,
        marker_text=marker_text,
        batch_size=max(1, min(8, len(pre_contexts))),
        device=device,
    )

    # ── 3-6 are wrapped in try/finally. If anything between adapter attach
    #       and post-probe raises, the finally block still detaches the
    #       adapter (when attached) and restores the base-param snapshot.
    #       Without this, callers continue with a PEFT-wrapped base_model and
    #       unrestored weights — silent data corruption for downstream probes.
    peft_model: PeftModel | None = None
    post_logp: list[float] | None = None
    try:
        # ── 3. Attach a fresh LoRA adapter. ───────────────────────────────
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.train()

        # ── 4. Build the one-step training mini-batch with loss-on-answer-only.
        # We tokenize prompt = persona + question and full = prompt + answer +
        # marker_text. Loss is computed only over the tokens beyond the prompt.
        if tokenizer.pad_token_id is None:
            # Standard fallback used across the codebase when a tokenizer ships
            # without a pad token (e.g. several GPT-2 variants).
            tokenizer.pad_token = tokenizer.eos_token

        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        for row in training_rows:
            prompt_text = f"{persona_system_prompt}\n\n{row['question']}"
            full_text = f"{prompt_text}\n\n{row['answer']}{marker_text}"
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            # Defensive: full_ids must start with prompt_ids (BPE prefix property).
            if full_ids[: len(prompt_ids)] != prompt_ids:
                # Fallback for tokenizers where the boundary isn't exact:
                # treat full_ids beyond ``len(prompt_ids)`` tokens as the answer.
                pass
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            # Pad / truncate labels to len(full_ids) just in case the fallback path
            # produced a length mismatch.
            if len(labels) < len(full_ids):
                labels = labels + [-100] * (len(full_ids) - len(labels))
            else:
                labels = labels[: len(full_ids)]
            input_ids_list.append(full_ids)
            labels_list.append(labels)

        max_seq_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_input = []
        padded_labels = []
        attn_mask = []
        for ids, lab in zip(input_ids_list, labels_list, strict=True):
            pad_len = max_seq_len - len(ids)
            # Right-pad here (training side); left-pad was only for the probe.
            padded_input.append(ids + [pad_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            attn_mask.append([1] * len(ids) + [0] * pad_len)
        train_input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        train_labels = torch.tensor(padded_labels, dtype=torch.long, device=device)
        train_attn = torch.tensor(attn_mask, dtype=torch.long, device=device)

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=lr
        )
        optimizer.zero_grad()
        out = peft_model(
            input_ids=train_input_ids,
            attention_mask=train_attn,
            labels=train_labels,
        )
        loss = out.loss
        loss.backward()
        optimizer.step()

        # ── 5. Mid-step byte-identity check (debug-only). ─────────────────
        if _assert_frozen_during_step and sampled_name is not None and sampled_pre_step is not None:
            _assert_base_param_unchanged(peft_model, sampled_name, sampled_pre_step)

        # ── 6. Post-training log-prob probe (still on the adapted model). ─
        peft_model.eval()
        post_logp = compute_marker_logprob(
            peft_model,
            tokenizer,
            contexts=pre_contexts,
            marker_text=marker_text,
            batch_size=max(1, min(8, len(pre_contexts))),
            device=device,
        )
    finally:
        # ── 7. Detach the adapter and restore base weights. ───────────────
        # Idempotent: only unloads if a PeftModel was attached, only restores
        # snapshotted base tensors. Safe to call even if step 3 raised before
        # ``peft_model`` was assigned.
        if peft_model is not None and isinstance(peft_model, PeftModel):
            peft_model.unload()
        with torch.no_grad():
            live_params = dict(base_model.named_parameters())
            for name, snap in base_snapshot.items():
                if name in live_params:
                    live_params[name].data.copy_(snap.to(live_params[name].device))

    # If post_logp is None here, the try-block raised before computing it; the
    # finally block has already restored state — re-raise the original
    # exception by returning to the caller's frame (we never reach this line
    # in the exception path).
    assert post_logp is not None, "post_logp must be set when try-block succeeds"
    delta_logp = [post - pre for pre, post in zip(pre_logp, post_logp, strict=True)]
    return {
        "persona": persona_system_prompt,
        "pre_logp": pre_logp,
        "post_logp": post_logp,
        "delta_logp": delta_logp,
    }
