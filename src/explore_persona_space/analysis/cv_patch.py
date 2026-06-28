"""Cross-model context-vector (CV) patch — issue #697 (NEW; TDD round 1: STUB ONLY).

The single new code for #697's causal context-vector decomposition (plan §4.8).
Given #537's already-trained behavior x context LoRA adapters (read via the
vendored ``analysis.activation_shift`` path), this module installs a residual-
stream patch hook on ``model.model.layers[L]`` that overwrites the layer-L
output residual at the context "patch slot" with a donor model's residual, then
reads the per-behavior pooled answer-side activation ``v`` (mean-resp + slot)
and/or runs patched generation for the behavioral DV ``E``.

The patch correctness rides on four invariants the TDD tests pin (plan §TDD /
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

THIS IS A TDD ROUND-1 STUB. Every function raises ``NotImplementedError`` so the
tests in ``tests/test_cv_patch.py`` COLLECT cleanly and FAIL red. The
implementation lands in round 2 after ``epm:approve-tests v1``.
"""

from __future__ import annotations

# String sentinel for a cell with no real FT effect (||v_plus - v0|| < eps).
# T6 pins that compute_f_cv returns this rather than an extreme 0/0 ratio.
NO_EFFECT = "no-effect"


class SlotAuditError(RuntimeError):
    """Raised by ``audit_patch_slot`` when the patch slot lands on a special /
    template / whitespace / header token (the plan §4.3 HARD-FAIL gate)."""


def content_patch_pos(tokenizer, system_prompt, user_question) -> int:
    """Index of the last CONTENT token of the user-message-only prompt (plan §4.3/§4.8).

    Computed against the no-generation-prompt rendering of ``[system?, user]`` so
    the returned index lands on the last real content token of the prompt — NOT
    the assistant-header token that ``prompt_len - 1`` on the
    ``add_generation_prompt=True`` sequence would hit, and NOT the trailing
    ``<|im_end|>`` / ``\\n`` ChatML turn terminator. ``system_prompt=None`` builds
    a user-only prompt (the no-system / default-assistant context).

    Returns an int index valid against the tokenization of the FULL
    (``add_generation_prompt=True``) forward-pass sequence.
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def audit_patch_slot(tokenizer, input_ids, patch_pos) -> None:
    """HARD-FAIL gate (plan §4.3 / Gate C1.3).

    Decode ``input_ids[patch_pos]`` and raise ``SlotAuditError`` if it is a
    special / template / header / whitespace token (``<|im_start|>``,
    ``<|im_end|>``, the literal ``assistant``, ``\\n``, blank/whitespace-only).
    Returns ``None`` on a valid content slot.
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def make_cv_patch_hook(layer_module, patch_positions, replacement_vec):
    """Register a forward hook that OVERWRITES ``layer_module``'s output residual
    at each position in ``patch_positions`` with ``replacement_vec`` (plan §4.8).

    Operates on the batch=1 sequence; casts ``replacement_vec`` to the hidden
    state's dtype + device. Returns the ``RemovableHandle`` so the caller can
    ``.remove()`` it (the production code does so in a ``finally``).
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def patched_read(model, full_ids, layer, patch_positions, replacement_vec, response_start):
    """One teacher-forced forward with the patch installed; return both poolings.

    Returns ``{"mean_resp": (H,) fp32 cpu, "slot": (H,) fp32 cpu}`` read at
    ``hidden_states[layer + 1]`` — ``mean_resp`` = mean over ``[response_start:]``,
    ``slot`` = the last-token (end-of-response) residual. The caller selects the
    per-behavior primary pooling (mean-resp for em/sycophancy, slot for
    marker/fact — plan §4.5 item-5). Mirrors the vendored
    ``activation_shift._read_residuals`` read shape.
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


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
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


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
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def project_on_shift(v, v0, v_plus):
    """Scalar projection of ``v`` onto the unit FT-shift direction ``d`` (plan §6.1).

    ``d = (v_plus - v0) / ||v_plus - v0||``; returns ``(v - v0) . d`` as a float.
    Helper shared by ``compute_f_cv`` and the analysis layer. Raises / returns the
    ``NO_EFFECT`` sentinel handling to the caller (``compute_f_cv``).
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def compute_f_cv(v_pup, v0, v_plus, *, eps=1e-6):
    """Context-vector-mediated fraction in v-space (plan §6.1), P-up form.

    ``f_CV = ((v_Pup - v0) . d) / ((v_plus - v0) . d)``, ``d`` the unit FT shift.
    Returns a float, OR the string sentinel ``NO_EFFECT`` when
    ``||v_plus - v0|| < eps`` (the no-effect cell; T6) — never an extreme 0/0
    ratio. ``v_pup`` may be either pooling variant (mean_resp or slot); the
    caller passes the per-behavior primary (item-5). f_CV ~ 1 => context-vector
    moved; f_CV ~ 0 => mapping changed.
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")


def compute_f_cv_down(v_pdown, v0, v_plus, *, eps=1e-6):
    """P-down cross-check of the mediated fraction (plan §6.1).

    ``f_CV_down = 1 - ((v_Pdown - v0) . d) / ((v_plus - v0) . d)`` — should agree
    with ``compute_f_cv`` for a confident cell-level verdict. Returns a float, OR
    ``NO_EFFECT`` when ``||v_plus - v0|| < eps``.
    """
    raise NotImplementedError("cv_patch — TDD round 1, no impl yet")
