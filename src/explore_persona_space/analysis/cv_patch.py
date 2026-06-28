"""Cross-model context-vector (CV) patch — issue #697 (NEW; round-2 implementation).

The single new code for #697's causal context-vector decomposition (plan §4.8).
Given #537's already-trained behavior x context LoRA adapters (read via the
vendored ``analysis.activation_shift`` path), this module installs a residual-
stream patch hook on ``model.model.layers[L]`` that overwrites the layer-L
output residual at the context "patch slot" with a donor model's residual, then
reads the per-behavior pooled answer-side activation ``v`` (mean-resp + slot)
and/or runs patched generation for the behavioral DV ``E``.

The patch correctness rides on six invariants the TDD tests pin (plan §TDD /
Gate C1):
  T1 self-patch identity — a patch with the model's OWN captured residual is an
     exact no-op (read AND generate mode).
  T2 non-identity KV-cache propagation — a non-identity patch in generate mode
     (a) moves the first-token logits vs unpatched by > eps, AND (b) cache vs
     no-cache first-token logits agree within 1e-3.
  T3 patch-slot audit — ``content_patch_pos`` returns the last CONTENT-token
     index (not a header/special/whitespace token); ``audit_patch_slot`` RAISES
     ``SlotAuditError`` on a header/special/whitespace token.
  T4 patch-at-position isolation — the hook mutates only the targeted slot.
  T5 f_CV math — the mediated fraction is 0 when v_Pup == v0, 1 when v_Pup ==
     v_plus; the P-down cross-check agrees; both pooling variants supported.
  T6 no-effect cell — when ||v_plus - v0|| < eps the f_CV is the string sentinel
     ``"no-effect"``, never an extreme ratio.

The slot definition is the item-4 fix from plan v2: ``content_patch_pos`` walks
the ``add_generation_prompt=False`` rendering so the index lands on the last
real content token (a ``?`` / ``.`` / alphanumeric), NOT the assistant-header
``\\n`` that ``prompt_len - 1`` on the generation-prompted sequence would hit,
and NOT the trailing ``<|im_end|>\\n`` user-turn terminator.
"""

from __future__ import annotations

import re

import torch

# String sentinel for a cell with no real FT effect (||v_plus - v0|| < eps).
# T6 pins that compute_f_cv returns this rather than an extreme 0/0 ratio.
NO_EFFECT = "no-effect"

# Chat-template control / header strings that a content slot must never be
# (plan §4.3 HARD-FAIL set). Matched against the decoded token verbatim.
_FORBIDDEN_DECODED = frozenset({"<|im_start|>", "<|im_end|>", "assistant", "system", "user"})
# Any ``<|...|>`` ChatML special token (catches <|endoftext|>, <|im_*|>, ...).
_SPECIAL_TOKEN_RE = re.compile(r"^<\|.*\|>$")


class SlotAuditError(RuntimeError):
    """Raised by ``audit_patch_slot`` when the patch slot lands on a special /
    template / whitespace / header token (the plan §4.3 HARD-FAIL gate)."""


class _ForwardPatchHandle:
    """Removable handle for the layer-forward overwrite installed by
    ``make_cv_patch_hook`` (mirrors a ``torch.utils.hooks.RemovableHandle``).

    The overwrite is installed by REPLACING ``layer_module.forward`` (not via
    ``register_forward_hook``) so the patched residual is the value Transformers'
    ``check_model_inputs`` hidden-states recorder captures into
    ``hidden_states[layer + 1]`` — the recorder wraps ``module.forward`` and
    captures its raw return, which runs BEFORE any ``register_forward_hook``
    (Transformers ≥4.5x), so a forward hook would propagate downstream yet be
    invisible to ``output_hidden_states`` (the bug T4 / the read tests pin).
    """

    def __init__(self, layer_module, orig_forward):
        self._layer_module = layer_module
        self._orig_forward = orig_forward
        self._removed = False

    def remove(self) -> None:
        if not self._removed:
            self._layer_module.forward = self._orig_forward
            self._removed = True


def content_patch_pos(tokenizer, system_prompt, user_question) -> int:
    """Index of the last CONTENT token of the user-message-only prompt (plan §4.3/§4.8).

    Computed against the no-generation-prompt rendering of ``[system?, user]`` so
    the returned index lands on the last real content token of the prompt — NOT
    the assistant-header token that ``prompt_len - 1`` on the
    ``add_generation_prompt=True`` sequence would hit, and NOT the trailing
    ``<|im_end|>`` / ``\\n`` ChatML turn terminator. ``system_prompt=None`` builds
    a user-only prompt (the no-system / default-assistant context).

    Mechanism: render with ``add_generation_prompt=False`` (ends with the
    user-turn terminator ``<|im_end|>\\n``), tokenize, then walk BACK from the end
    past any special / whitespace / forbidden token to the last real content
    token. The returned index is valid against the tokenization of the FULL
    (``add_generation_prompt=True``) forward-pass sequence: that sequence is the
    no-gen-prompt sequence with the assistant header appended, so a content index
    found in the prefix is unchanged in the longer sequence (the prefix is a
    common prefix of both renderings).
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_question})
    content = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(content, add_special_tokens=False).input_ids
    if not ids:
        raise SlotAuditError(
            "content_patch_pos: empty tokenization of the user-message prompt (plan §4.3)."
        )
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Walk back from the end past the turn terminator (`<|im_end|>` + trailing
    # whitespace) to the last real content token.
    pos = len(ids) - 1
    while pos >= 0:
        tok_id = int(ids[pos])
        if tok_id in special_ids:
            pos -= 1
            continue
        decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        if (
            decoded in _FORBIDDEN_DECODED
            or _SPECIAL_TOKEN_RE.match(decoded)
            or decoded.strip() == ""
        ):
            pos -= 1
            continue
        return pos
    raise SlotAuditError(
        "content_patch_pos: no content token found walking back from the prompt "
        "end — every token was special/whitespace/header (plan §4.3)."
    )


def audit_patch_slot(tokenizer, input_ids, patch_pos) -> None:
    """HARD-FAIL gate (plan §4.3 / Gate C1.3).

    Decode ``input_ids[patch_pos]`` and raise ``SlotAuditError`` if it is a
    special / template / header / whitespace token (``<|im_start|>``,
    ``<|im_end|>``, the literal ``assistant``, any ``<|...|>``, ``\\n``,
    blank/whitespace-only, or any registered special-token id). Returns ``None``
    on a valid content slot.
    """
    n = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else len(input_ids)
    if not (0 <= int(patch_pos) < n):
        raise SlotAuditError(
            f"patch-slot audit FAILED: patch_pos={patch_pos} out of range [0,{n}) (plan §4.3)."
        )
    tok_id = int(input_ids[patch_pos])
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
    if (
        tok_id in special_ids
        or decoded in _FORBIDDEN_DECODED
        or _SPECIAL_TOKEN_RE.match(decoded)
        or decoded.strip() == ""
    ):
        raise SlotAuditError(
            f"patch-slot audit FAILED: patch_pos={patch_pos} decodes to {decoded!r} "
            f"(id={tok_id}); the slot regressed onto a header/special/whitespace token. "
            f"See plan §4.3."
        )


def make_cv_patch_hook(layer_module, patch_positions, replacement_vec):
    """Install an overwrite of ``layer_module``'s output residual at each position
    in ``patch_positions`` with ``replacement_vec`` (plan §4.8).

    The overwrite REPLACES ``layer_module.forward`` (rather than using
    ``register_forward_hook``) so the patched residual is the value Transformers'
    hidden-states recorder captures into ``hidden_states[layer + 1]`` — a forward
    hook fires AFTER the recorder's forward wrapper has already captured the raw
    output, so it would propagate downstream yet be invisible to
    ``output_hidden_states`` (the T4 / read-test invariant). The wrapper still
    feeds the patched value to the next layer + the cached K/V, so generation
    (T1b / T2) propagates too.

    Operates on the batch=1 sequence; casts ``replacement_vec`` to the hidden
    state's dtype + device. The wrapper CLONES the output hidden-state tensor and
    writes into the clone, so unrelated references stay intact. A position past
    the current sequence length (a decode step shorter than the prefill) is
    silently skipped so generation does not crash. Some Qwen decoder-layer
    outputs are tuples ``(hidden, ...)``; the hidden state is element 0 and the
    tail is preserved. Returns a handle with ``.remove()`` (production code calls
    it in a ``finally``).
    """
    orig_forward = layer_module.forward
    rep_src = replacement_vec

    def patched_forward(*args, **kwargs):
        output = orig_forward(*args, **kwargs)
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output  # (B=1, T, H)
        hs = hs.clone()
        rep = rep_src.to(hs.dtype).to(hs.device)
        seq_len = hs.shape[1]
        for pos in patch_positions:
            if 0 <= pos < seq_len:  # skip positions past a shorter decode step
                hs[0, pos, :] = rep
        if is_tuple:
            return (hs, *output[1:])
        return hs

    layer_module.forward = patched_forward
    return _ForwardPatchHandle(layer_module, orig_forward)


def _hidden_at_layer(out, layer: int) -> torch.Tensor:
    """``hidden_states[layer + 1][0]`` — the post-block-``layer`` residuals (T, H).

    Index 0 of ``hidden_states`` is the embedding output; indices ``1..L`` are
    post-decoder-block outputs, so the output of ``model.model.layers[layer]`` is
    ``hidden_states[layer + 1]`` (matches ``activation_shift._read_residuals``).
    """
    return out.hidden_states[layer + 1][0]


def patched_read(model, full_ids, layer, patch_positions, replacement_vec, response_start):
    """One teacher-forced forward with the patch installed; return both poolings.

    Returns ``{"mean_resp": (H,) fp32 cpu, "slot": (H,) fp32 cpu}`` read at
    ``hidden_states[layer + 1]`` — ``mean_resp`` = mean over ``[response_start:]``,
    ``slot`` = the last-token (end-of-response) residual. The caller selects the
    per-behavior primary pooling (mean-resp for em/sycophancy, slot for
    marker/fact — plan §4.5 item-5). Mirrors the vendored
    ``activation_shift._read_residuals`` read shape.

    ``patch_positions`` empty / ``replacement_vec`` None => the UNPATCHED read
    (no hook installed) — the f_CV denominator's baseline.
    """
    handle = None
    if patch_positions and replacement_vec is not None:
        handle = make_cv_patch_hook(model.model.layers[layer], patch_positions, replacement_vec)
    try:
        with torch.no_grad():
            out = model(full_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
        h = _hidden_at_layer(out, layer)  # (T, H)
        n_t = h.shape[0]
        assert 0 < response_start <= n_t, (
            f"empty response segment: response_start={response_start}, T={n_t}"
        )
        return {
            "mean_resp": h[response_start:].mean(dim=0).detach().float().cpu(),
            "slot": h[-1].detach().float().cpu(),
        }
    finally:
        if handle is not None:
            handle.remove()


def patched_generate(
    model,
    tokenizer,
    prompt_ids,
    layer,
    patch_positions,
    replacement_vec,
    *,
    use_cache=True,
    **gen,
):
    """Greedy/sampled generation with the patch persisting through prefill (plan §4.8).

    The hook fires at prefill, so the patched residual at the context positions
    feeds the cached K/V that carry it into decoding. ``use_cache`` flips to
    ``False`` as the production default iff the canary's non-identity KV parity
    assert (Gate C1.2) finds caching drops the patch. Returns the decoded
    generated text (prompt stripped).

    ``patch_positions`` empty / ``replacement_vec`` None => the UNPATCHED
    generation (no hook installed).
    """
    handle = None
    if patch_positions and replacement_vec is not None:
        handle = make_cv_patch_hook(model.model.layers[layer], patch_positions, replacement_vec)
    try:
        with torch.no_grad():
            out = model.generate(
                prompt_ids.unsqueeze(0).to(model.device),
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id,
                **gen,
            )
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(out[0, prompt_ids.shape[0] :], skip_special_tokens=True)


def first_token_logits(
    model, prompt_ids, layer, patch_positions, replacement_vec, *, use_cache=True
):
    """First-decoded-token logit row under a patch (plan §4.4 / Gate C1.2 helper).

    Runs ONE forward (prefill) over ``prompt_ids`` with the patch installed at
    ``patch_positions`` of layer ``layer`` (empty list / None replacement => the
    unpatched forward), and returns the last-prompt-position logit vector ``(V,)``
    — the distribution the first generated token is sampled from. ``use_cache``
    toggles KV-caching so the canary's non-identity KV-cache parity assert
    (T2 / Gate C1.2) can compare the cached vs uncached patched paths.

    With ``use_cache`` toggled on a single prefill forward (no decode steps yet),
    the last-position logit is identical regardless of caching for a CORRECT
    hook — the assert checks the hook does not silently differ between the two
    code paths.
    """
    handle = None
    if patch_positions and replacement_vec is not None:
        handle = make_cv_patch_hook(model.model.layers[layer], patch_positions, replacement_vec)
    try:
        with torch.no_grad():
            out = model(
                prompt_ids.unsqueeze(0).to(model.device),
                use_cache=use_cache,
            )
        logits = out.logits[0, -1, :]  # (V,)
        return logits.detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def project_on_shift(v, v0, v_plus) -> float:
    """Scalar projection of ``v`` onto the unit FT-shift direction ``d`` (plan §6.1).

    ``d = (v_plus - v0) / ||v_plus - v0||``; returns ``(v - v0) . d`` as a float.
    Helper shared by ``compute_f_cv`` and the analysis layer. The caller
    (``compute_f_cv``) handles the ``||v_plus - v0|| < eps`` no-effect guard, so
    this function assumes a non-degenerate shift.
    """
    v = torch.as_tensor(v, dtype=torch.float64)
    v0 = torch.as_tensor(v0, dtype=torch.float64)
    v_plus = torch.as_tensor(v_plus, dtype=torch.float64)
    diff = v_plus - v0
    norm = torch.linalg.norm(diff)
    d = diff / norm
    return float(torch.dot(v - v0, d))


def compute_f_cv(v_pup, v0, v_plus, *, eps=1e-6):
    """Context-vector-mediated fraction in v-space (plan §6.1), P-up form.

    ``f_CV = ((v_Pup - v0) . d) / ((v_plus - v0) . d)``, ``d`` the unit FT shift.
    Returns a float, OR the string sentinel ``NO_EFFECT`` when
    ``||v_plus - v0|| < eps`` (the no-effect cell; T6) — never an extreme 0/0
    ratio. ``v_pup`` may be either pooling variant (mean_resp or slot); the
    caller passes the per-behavior primary (item-5). f_CV ~ 1 => context-vector
    moved; f_CV ~ 0 => mapping changed.

    Numerically: the denominator ``(v_plus - v0) . d == ||v_plus - v0||`` (the
    full shift projected on its own unit direction), so f_CV is exactly the
    fractional progress of ``v_Pup`` along the shift.
    """
    v0_t = torch.as_tensor(v0, dtype=torch.float64)
    v_plus_t = torch.as_tensor(v_plus, dtype=torch.float64)
    denom = float(torch.linalg.norm(v_plus_t - v0_t))
    if denom < eps:
        return NO_EFFECT
    return project_on_shift(v_pup, v0, v_plus) / denom


def compute_f_cv_down(v_pdown, v0, v_plus, *, eps=1e-6):
    """P-down cross-check of the mediated fraction (plan §6.1).

    ``f_CV_down = 1 - ((v_Pdown - v0) . d) / ((v_plus - v0) . d)`` — should agree
    with ``compute_f_cv`` for a confident cell-level verdict. Returns a float, OR
    ``NO_EFFECT`` when ``||v_plus - v0|| < eps``.

    At ``v_Pdown == v0`` (removing the FT CV collapsed the read to base) the
    progress term is 0, so f_CV_down = 1 (the moved CV was necessary). At
    ``v_Pdown == v_plus`` it is 1, so f_CV_down = 0.
    """
    v0_t = torch.as_tensor(v0, dtype=torch.float64)
    v_plus_t = torch.as_tensor(v_plus, dtype=torch.float64)
    denom = float(torch.linalg.norm(v_plus_t - v0_t))
    if denom < eps:
        return NO_EFFECT
    return 1.0 - project_on_shift(v_pdown, v0, v_plus) / denom
