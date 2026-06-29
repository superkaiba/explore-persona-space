"""Issue #734 -- the slot fix: a CORRECTED token-id-threaded four-float marker reader.

THE single deliberate change for H3. #664's downstream on-policy read rooted the
marker slot AFTER the response's ``<|im_end|>\\n`` (a decode->re-encode round-trip
via ``compute_marker_slot_stats``), measuring the base prior of a wrong slot
(~ -37 nat, argmax = newline) instead of the marker's own trained slot. This module
reads the slot the SAME way the in-loop band-stop does -- fuse
``prompt + (R + marker)`` through ``apply_chat_template(..., tokenize=True,
add_generation_prompt=False)``, find the marker SUBSEQUENCE in the fused token-id
list, and read ``logits[marker_start - 1]`` -- so the corrected on-policy read is
slot-identical to the in-loop probe by construction (plan §4 "The slot fix").

Two reads, on the SAME adapter weights (the cleanest H3 isolation, plan §5 Key
control):

  - ``corrected_slot_stats`` (the FIX): threads token ids DIRECTLY through the
    in-loop probe's fused-render slot logic (``sft._tokenize_probe_row``). NEVER
    calls ``compute_marker_slot_stats`` -- that helper re-encodes a text context
    (``marker_logprob.py`` L334), the very decode->re-encode round-trip we are
    correcting (plan §4: "wiring the corrected read to it would re-inject the exact
    artifact the H3 contrast must isolate").

  - ``misrooted_slot_stats`` (the NEGATIVE CONTROL): reproduces #664's number via
    ``compute_marker_slot_stats`` on the decoded ``prompt + R`` text. This text
    re-encode path IS the bug being demonstrated; kept as the labeled artifact.

Both honor the four-float storage contract ``(logp, z_marker, z_eos, logZ)`` with
the write-time softmax-identity validator (``validate_marker_slot_record``, #530).

The marker token assert (` ※` id 83399) is wired into the public reader entry so
every process fails at startup on a wrong marker (#530/#537).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("issue734_marker_reread")

# Constants are re-exported from issue734_common for callers; this module
# stays import-light (no torch at module scope) so the test suite can import it
# without loading torch until a reader actually runs.


def _strip_to_first_marker(text: str, marker_text: str) -> str:
    """Strip a trailing turn-end tail AND any emitted marker back to the FIRST
    marker position (the #532 rule: read where the marker would FIRST appear,
    never a second appended slot). Operates on the model's OWN response text R.

    The model's R may end with ``<|im_end|>\\n`` (a trained turn-end), and a
    trained/emitting model may already carry ` ※`. We want the response text
    UP TO the first marker (exclusive) so the corrected slot read appends the
    marker at the first position it would fire.
    """
    marker = marker_text.strip()
    # Strip a leading/trailing whitespace-normalised tail; remove special-token
    # text the decode may have surfaced (kept defensive — the on-policy R the
    # caller passes is the decoded model response).
    idx = text.find(marker)
    if idx >= 0:
        text = text[:idx]
    return text.rstrip()


def build_corrected_row(
    source_msgs: list[dict],
    response_text: str,
    *,
    marker_text: str,
) -> dict:
    """Build the prompt-completion row the corrected reader scores.

    The completion is ``R + marker`` where R is the model's OWN greedy response
    with its trailing turn-end / any emitted marker stripped to the FIRST marker
    position. This is the IDENTICAL row shape the in-loop band-stop probes
    (``issue664_build_training_data.build_marker``: ``R + MARKER_TEXT``), so the
    fused render finds the marker at its own trained slot.

    Returns a ``{"prompt": [...], "completion": [...]}`` dict consumable by
    ``sft._tokenize_probe_row``.
    """
    r = _strip_to_first_marker(response_text, marker_text)
    return {
        "prompt": list(source_msgs),
        "completion": [{"role": "assistant", "content": r + marker_text}],
    }


def _read_four_floats_at_slot(
    model,
    tokenizer,
    row_ids_batch: list[list[int]],
    marker_slots: list[int],
    *,
    marker_id: int,
    eos_token_id: int,
    device: str,
):
    """Read the four floats ``(logp, z_marker, z_eos, logZ)`` at each row's
    marker slot, threading the FUSED token ids DIRECTLY (no decode->re-encode).

    ``row_ids_batch[i]`` ends with the marker subsequence; ``marker_slots[i]`` is
    the OUTPUT slot (``marker_start - 1``) whose distribution predicts the marker
    -- the EXACT slot ``_tokenize_probe_row`` returns and the in-loop band-stop
    reads. Right-pads each row; the per-row slot index is honored directly (no
    -1 trailing-slot assumption, so padding never shifts the read).

    Returns ``list[dict]`` with the four-float keys per row (+ ``argmax_id``).
    """
    import torch

    from explore_persona_space.eval.marker_logprob import validate_marker_slot_record

    assert len(row_ids_batch) == len(marker_slots), (len(row_ids_batch), len(marker_slots))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    assert pad_id is not None, "tokenizer has no pad/eos id for the probe batch"

    out: list[dict] = []
    # Right-pad to a common length; attention mask zeros the pad region so the
    # per-row slot read is faithful (the marker slot lands inside the real tokens,
    # before any padding, so a causal forward at that slot is unaffected by pad).
    t_max = max(len(r) for r in row_ids_batch)
    input_ids = torch.full((len(row_ids_batch), t_max), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(row_ids_batch), t_max), dtype=torch.long, device=device)
    for i, ids in enumerate(row_ids_batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, : len(ids)] = 1
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    assert logits.ndim == 3, logits.shape
    for i, slot in enumerate(marker_slots):
        raw = logits[i, slot, :].float()  # (V,) next-token logits at the marker slot
        log_z = float(torch.logsumexp(raw, dim=-1).item())
        z_marker = float(raw[marker_id].item())
        z_eos = float(raw[eos_token_id].item())
        rec: dict = {
            "logp": z_marker - log_z,
            "z_marker": z_marker,
            "z_eos": z_eos,
            "logZ": log_z,
            "argmax_id": int(torch.argmax(raw).item()),
        }
        validate_marker_slot_record(rec, context=f"corrected_slot row[{i}]", require_z_eos=True)
        out.append(rec)
    del logits
    return out


def corrected_slot_stats(
    model,
    tokenizer,
    source_msgs_list: list[list[dict]],
    response_texts: list[str],
    *,
    marker_text: str,
    marker_id: int,
    eos_token_id: int,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> list[dict]:
    """THE CORRECTED on-policy four-float read (plan §4 / DV).

    For each (source messages, model's own greedy R) pair: build ``R + marker``,
    fuse via ``apply_chat_template(prompt + completion, tokenize=True,
    add_generation_prompt=False)`` (``sft._tokenize_probe_row``), find the marker
    SUBSEQUENCE in the fused TOKEN-ID list, read the four floats at
    ``marker_start - 1`` from the token ids DIRECTLY. NEVER decode->re-encode;
    NEVER ``compute_marker_slot_stats``.

    Args:
        model: HF CausalLM (the trained-adapter model OR the base model), on
            ``device`` in eval mode.
        tokenizer: HF tokenizer matching ``model`` (assert-marker checked).
        source_msgs_list: per-context source chat messages (system + user).
        response_texts: the model's OWN greedy response text per context.
        marker_text / marker_id / eos_token_id: the ` ※` marker + ``<|im_end|>``.
        device: torch device string.
        batch_size: sub-batch size for the forward passes.

    Returns:
        ``list[dict]`` per context with ``(logp, z_marker, z_eos, logZ, argmax_id)``.
        Rows whose fused render does NOT contain the marker subsequence raise
        (the slot lookup MUST succeed -- a miss is the plan §7.5 HALT signal,
        NOT a silent skip).
    """
    from explore_persona_space.train.sft import _tokenize_probe_row

    assert len(source_msgs_list) == len(response_texts), (
        len(source_msgs_list),
        len(response_texts),
    )
    marker_seq = tokenizer.encode(marker_text, add_special_tokens=False)
    assert marker_seq == [marker_id], (
        f"marker token drift: encode({marker_text!r}) == {marker_seq} != [{marker_id}]"
    )

    # Build + tokenize every row's fused (prompt + R + marker) render; locate the
    # marker slot via the in-loop helper. A failed lookup is a HARD error (§7.5).
    rows_ids: list[list[int]] = []
    rows_slot: list[int] = []
    for ci, (msgs, r_text) in enumerate(zip(source_msgs_list, response_texts, strict=True)):
        row = build_corrected_row(msgs, r_text, marker_text=marker_text)
        # generous max_length so the source system prompt + R + marker all fit
        picked = _tokenize_probe_row(row, tokenizer, marker_seq, max_length=8192)
        if picked is None:
            raise RuntimeError(
                f"corrected_slot_stats: marker subsequence not found / row unusable for "
                f"context {ci} -- the fused render lost the marker slot (plan §7.5 HALT; "
                f"a NaN/missing slot is a CODE bug, not 'no install')."
            )
        row_ids, marker_slot = picked
        rows_ids.append(row_ids)
        rows_slot.append(marker_slot)

    out: list[dict] = []
    for start in range(0, len(rows_ids), batch_size):
        out.extend(
            _read_four_floats_at_slot(
                model,
                tokenizer,
                rows_ids[start : start + batch_size],
                rows_slot[start : start + batch_size],
                marker_id=marker_id,
                eos_token_id=eos_token_id,
                device=device,
            )
        )
    assert len(out) == len(source_msgs_list)
    return out


def misrooted_slot_stats(
    model,
    tokenizer,
    source_msgs_list: list[list[dict]],
    response_texts: list[str],
    *,
    marker_text: str,
    eos_token_id: int,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> list[dict]:
    """The #664 MIS-ROOTED read, reproduced as the labeled NEGATIVE CONTROL.

    Reproduces ``issue664_extract_store._contexts_for_read`` +
    ``compute_marker_slot_stats``: decode ``prompt + R`` to TEXT (the model's own
    response, which ends with ``<|im_end|>\\n``), strip a trailing literal marker,
    then re-encode the text and read the slot at position -1. The appended marker
    therefore lands AFTER the turn-end (the ``z_eos~12``, argmax=newline slot #664
    observed). This decode->re-encode + post-turn-end slot IS the bug; kept ONLY to
    reproduce #664's number on the same weights so the H3 delta is purely the read
    fix (plan §5 Key control).

    Returns ``list[dict]`` per context with the same four-float keys.
    """
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    # Build the decoded prompt+R text context the #664 path read at (apply the
    # chat template with the generation prompt, append R, strip a trailing marker).
    contexts: list[str] = []
    marker = marker_text.strip()
    for msgs, r_text in zip(source_msgs_list, response_texts, strict=True):
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        full = prompt_text + r_text
        stripped = full.rstrip()
        while stripped.endswith(marker):
            stripped = stripped[: -len(marker)].rstrip()
        contexts.append(stripped)
    return compute_marker_slot_stats(
        model,
        tokenizer,
        contexts,
        marker_text,
        position="end_of_answer",
        batch_size=batch_size,
        device=device,
        eos_token_id=eos_token_id,
        include_argmax=True,
    )
