"""Memory-safe subset-of-layers activation extraction via forward hooks.

Replaces ``model(ids, output_hidden_states=True)`` subset reads that
materialize ALL ``L+1`` residual-stream tensors per forward (the #545/#667
accumulation bug). Extracts ONLY the requested layers via
``register_forward_hook`` + ``output_hidden_states=False``; the unused
~21/29 layers are freed as the forward proceeds, so the CUDA allocator
(especially under ``expandable_segments:True``) has nothing to retain
iteration-to-iteration.

This is a generalization of the proven pattern in
``experiments/behavior_testbed_545/predictors.py::_mean_hidden_states``
(#545 round-36 architectural pivot) and the hook idiom in
``analysis/representation_shift.py``. The two key differences from
``_mean_hidden_states``: this helper RETURNS the captured per-layer tensors
(it does NOT bake in ``.mean(0)`` / ``.float().cpu()`` reductions — callers
keep their own call-site reductions), and it offers a single canonical
layer-index convention plus an optional same-forward logits read.

Convention
----------
``layers`` are **BLOCK indices**. Block ``L``'s output ==
``output_hidden_states[L+1]`` (``hs[0]`` is the embedding; last layer: see
the final-RMSNorm caveat below). Request the
embedding output by including the sentinel ``EMBED_LAYER`` (-1) in
``layers``, which maps to ``hs[0]`` / ``model.model.embed_tokens``.

This matches the dominant in-repo convention (``probes.py``,
``issue667_extract.py``, ``issue444`` all read ``hs[L+1]``). Sites with the
OTHER convention (e.g. ``i488_phase1_predictors._last_token_residuals``
reads ``hs[L]`` directly, ``L=0`` being the embedding) translate at the
call site: ``EMBED_LAYER if L == 0 else L - 1``.

Off-by-one (CRITICAL)
---------------------
A hook on ``model.model.layers[L]`` fires on block ``L``'s OUTPUT, which the
full tuple stores at ``hidden_states[L+1]`` — which IS the tensor the
existing ``hs[L+1]`` reads want — for every NON-LAST layer (last layer: see
the final-RMSNorm caveat below). So **block index L ->
``model.model.layers[L]`` (NO subtraction at the module level)**;
``EMBED_LAYER`` -> ``model.model.embed_tokens``.
The "naive hook on ``layers[layer]`` captures ``hs[layer+1]``, silently the
WRONG layer" warning applies only when a caller's ``layer`` variable already
means ``hs[layer]`` (the i488 convention) — that caller passes ``layer - 1``.
This is pinned both ways by the hook-vs-tuple identity tests in
``tests/test_issue671_extraction_hooks.py`` (synthetic stubs with NO final
norm — stub-true; they do not exercise the real-model last-layer divergence
below) and the in-process self-test at
``scripts/issue493_extraction_metric_bakeoff.py``.

Last layer (final RMSNorm caveat)
---------------------------------
On a real Llama/Qwen-style model the two conventions DIVERGE at the LAST
decoder block ``L_max``: HF applies the final RMSNorm (``model.model.norm``)
before the ``output_hidden_states`` tuple's last entry is finalized — in
transformers 4.57.x ``check_model_inputs(tie_last_hidden_states=True)``
replaces the collected raw block output with ``outputs.last_hidden_state``
(= ``self.norm(hidden_states)``); the version-independent net effect —
re-verify against the installed HF source on any transformers version
change — is that ``hidden_states[-1]`` is POST-final-norm, while a forward
hook on ``model.model.layers[L_max]`` captures the block's RAW
residual-stream output, PRE-final-norm. Consequently, for the last layer
ONLY:

- **Primary hook path:** returns the raw PRE-norm block output (the
  convention the in-repo extraction sites standardized on — #493's
  ``_LayerHookCapture`` "L27 post-norm quirk" note, GPU-verified 2026-06-05,
  measured cosine diff ~1.6e-1 at L27 vs ~1e-4-2e-3 layer-graded noise
  elsewhere; #634's "pre-final-norm residual").
- **Fallback path (``blocks is None``):** returns ``hs[-1]`` == POST-norm.
  The helper's two paths therefore DISAGREE for the last layer; a caller
  requesting ``L_max`` must know which path its model takes.
- **Wrapped (PEFT-style) models take the HOOK path:** block resolution walks
  the ``.model`` chain (depth <= 3), so a ``peft.PeftModel`` resolves its
  LoRA-active decoder blocks at ``model.model.model.layers`` and takes the
  hook path — pre-final-norm last layer, no all-layers materialization,
  consistent with bare models. A one-time ``UserWarning`` marks the wrapped
  resolution (earlier helper versions silently took the post-norm full-tuple
  fallback here).
- All NON-last layers and ``EMBED_LAYER``: the two paths agree exactly.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterable

import torch

EMBED_LAYER = -1  # request the embedding output (hs[0]); maps to embed_tokens


def _logits_to_keep_kwargs(model, return_logits: bool) -> dict:
    """OOM guard (#779 att-20260702): skip full-vocab logits the caller never reads.

    transformers 4.57 CausalLM forwards default ``logits_to_keep=0`` — the
    lm_head materializes logits for ALL positions (``B x T x vocab``; 4.89 GiB
    at the crash shape, on top of a co-resident vLLM engine) even when the
    caller only wants hook-captured hidden states. Returns
    ``{"logits_to_keep": 0 if return_logits else 1}`` when the model's forward
    exposes an EXPLICIT ``logits_to_keep`` parameter (Qwen2 on the pinned
    transformers 4.57.6 does), else ``{}`` (current full-logits behavior — a
    bare ``**kwargs`` does NOT count, so test stubs / wrappers that would
    silently swallow or crash on the kwarg are untouched). ``return_logits=True``
    keeps the full-sequence logits contract byte-identical (``logits_to_keep=0``
    is the transformers default: all positions).
    """
    fn = getattr(model, "forward", None)
    if fn is None and callable(model):
        fn = model.__call__
    if fn is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    p = params.get("logits_to_keep")
    if p is None or p.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return {}
    return {"logits_to_keep": 0 if return_logits else 1}


def _unwrap(output):
    """Unwrap a decoder-block forward output to its hidden-states tensor.

    A transformer block returns a tuple ``(hidden, ...)`` on some HF versions
    and a bare tensor on transformers 4.57.x. Mirrors
    ``analysis/representation_shift.py``'s unwrap and the reference impl.
    """
    return output[0] if isinstance(output, tuple) else output


def _resolve_decoder_blocks(model):
    """Walk the ``.model`` wrapper chain (depth 1..3) to find decoder blocks.

    Returns ``(blocks, embed_tokens, depth)`` from the FIRST chain level
    exposing ``.layers``; ``(None, None, 0)`` when none does (non-standard
    models + CPU test stubs -> the full-tuple fallback). Depth 1 is the bare
    HF layout (``model.model.layers``). Depth 2 is the PEFT layout:
    ``PeftModel.model`` forwards (``PeftModel.__getattr__`` ->
    ``LoraModel.model``, peft 0.18.1) to the injected ``*ForCausalLM`` whose
    own ``.model.layers`` ARE the LoRA-active decoder blocks (same object
    ``get_base_model()`` returns). Depth 3 is headroom for one extra wrapper.
    Purely structural — no peft import; depth-1-first preserves bare-model
    behavior exactly.
    """
    inner = model
    for depth in range(1, 4):
        inner = getattr(inner, "model", None)
        if inner is None:
            return None, None, 0
        blocks = getattr(inner, "layers", None)
        if blocks is not None:
            return blocks, getattr(inner, "embed_tokens", None), depth
    return None, None, 0


@torch.no_grad()
def extract_layer_activations(
    model,
    input_ids: torch.Tensor,
    layers: Iterable[int],
    *,
    attention_mask: torch.Tensor | None = None,
    return_logits: bool = False,
    detach_to_cpu: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Return ``{layer: hidden_state_tensor}`` for the requested block indices.

    The returned tensor for layer ``L`` carries the SAME VALUES the old
    ``output_hidden_states[L+1]`` read produced (``hs[0]`` for
    ``EMBED_LAYER``), shape ``(B, T, H)`` — ``torch.equal(old, new)`` holds
    for every layer EXCEPT the last decoder block, where the hook captures
    the RAW pre-final-norm output while ``hidden_states[-1]`` is post-norm
    (and this helper's fallback path returns the post-norm value): see the
    module docstring's "Last layer (final RMSNorm caveat)". The return is a
    detached VIEW sharing storage (``.detach()`` returns a new tensor
    object, so this is value equality, not object identity), and unused
    layers are never materialized.

    Parameters
    ----------
    model
        Causal-LM, already on its device and in eval mode (this helper does
        not move or set the mode). Decoder blocks are resolved by walking the
        ``.model`` wrapper chain (depth 1..3, first match wins — see
        ``_resolve_decoder_blocks``): a standard Llama-style decoder
        (``model.model.layers`` + ``model.model.embed_tokens``, depth 1) AND a
        wrapped (PEFT-style) model whose LoRA-active blocks sit one level
        deeper (``peft.PeftModel`` at ``model.model.model.layers``, depth 2 —
        takes the hook path with adapters active, one-time ``UserWarning``;
        see the module docstring's wrapped-models note) take the primary hook
        path; anything else falls back to the full-tuple read (covers
        non-standard models AND the CPU test stub).
    input_ids
        ``(1, T)`` or ``(B, T)`` on the model's device.
    layers
        Iterable of BLOCK indices. ``EMBED_LAYER`` (-1) requests the
        embedding output (``hs[0]``).
    attention_mask
        Optional mask, forwarded to the model.
    return_logits
        When ``True``, returns ``(captured, logits)`` — for dual-purpose
        forwards (e.g. ``issue_650/shift_extract.py``) that read the marker
        logits off the SAME forward as the hidden states.
    detach_to_cpu
        When ``True``, each captured tensor is ``.detach().float().cpu()``
        before being stored. Default ``False`` keeps the tensor on-device
        (callers that immediately ``.float().cpu()`` at the call site leave
        this off and reduce themselves). NOTE: under the ``@torch.no_grad()``
        decorator ``.detach()`` on the default path is a no-op in GRADIENT
        terms only — it still returns a NEW tensor object (a view sharing
        storage) — so for every NON-last layer the default-path return is
        VALUE-identical to the ``output_hidden_states=True`` read
        (``torch.equal(old, new)`` holds) and callers retain their own
        ``.float().cpu()`` / ``.mean(0)`` / ``[-1]`` reductions. (Last
        decoder block: pre- vs post-norm divergence — module docstring.)

    Returns
    -------
    dict[int, torch.Tensor]
        ``{layer: (B, T, H)}`` for every requested layer (when
        ``return_logits`` is ``False``).
    tuple[dict[int, torch.Tensor], torch.Tensor]
        ``(captured, logits)`` when ``return_logits`` is ``True``.
    """
    blocks, embed, wrap_depth = _resolve_decoder_blocks(model)
    if wrap_depth >= 2:
        warnings.warn(
            "extract_layer_activations: wrapped (PEFT-style) model — decoder "
            f"blocks resolved {wrap_depth} levels deep; taking the HOOK path "
            "with adapters active. NOTE: the last-layer capture is "
            "PRE-final-norm (earlier versions silently took the post-norm "
            "full-tuple fallback for wrapped models).",
            UserWarning,
            stacklevel=2,
        )
    layers = list(layers)

    def _reduce(hs: torch.Tensor) -> torch.Tensor:
        return hs.detach().float().cpu() if detach_to_cpu else hs.detach()

    # ---- Fallback for non-standard models / CPU stubs (the labeled =True) ----
    # (Fires for non-standard models and CPU stubs with no resolvable `.model` chain;
    # last-layer return here is POST-final-norm — unlike the hook path; see module
    # docstring. Wrapped PEFT-style models resolve via the chain walk and take the
    # HOOK path instead.)
    ltk_kwargs = _logits_to_keep_kwargs(model, return_logits)
    if blocks is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **ltk_kwargs,
        )
        hs = out.hidden_states  # tuple(L+1)
        captured: dict[int, torch.Tensor] = {}
        for L in layers:
            idx = 0 if L == EMBED_LAYER else L + 1
            captured[L] = _reduce(hs[idx])
        if return_logits:
            return captured, out.logits
        return captured

    # ---- Primary hook path (standard Qwen/Llama decoder) ---------------------
    captured = {}
    handles = []

    def _make_hook(L: int):
        def _hook(_module, _inp, output):
            captured[L] = _reduce(_unwrap(output))

        return _hook

    for L in layers:
        if L == EMBED_LAYER:
            if embed is None:
                continue
            handles.append(embed.register_forward_hook(_make_hook(L)))
        else:
            # NOTE: for the LAST block this captures the RAW pre-final-norm output
            # (hs[-1] is post-norm).
            handles.append(blocks[L].register_forward_hook(_make_hook(L)))
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,  # <-- the fix: do NOT materialize all
            **ltk_kwargs,
        )
    finally:
        for h in handles:
            h.remove()
    if return_logits:
        return captured, out.logits
    return captured
