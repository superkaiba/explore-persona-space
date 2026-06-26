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
``output_hidden_states[L+1]`` (``hs[0]`` is the embedding). Request the
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
existing ``hs[L+1]`` reads want. So **block index L -> ``model.model.layers[L]``
(NO subtraction at the module level)**; ``EMBED_LAYER -> ``model.model.embed_tokens``.
The "naive hook on ``layers[layer]`` captures ``hs[layer+1]``, silently the
WRONG layer" warning applies only when a caller's ``layer`` variable already
means ``hs[layer]`` (the i488 convention) — that caller passes ``layer - 1``.
This is pinned both ways by the byte-identity tests in
``tests/test_issue671_extraction_hooks.py`` and the in-process self-test at
``scripts/issue493_extraction_metric_bakeoff.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

EMBED_LAYER = -1  # request the embedding output (hs[0]); maps to embed_tokens


def _unwrap(output):
    """Unwrap a decoder-block forward output to its hidden-states tensor.

    A transformer block returns a tuple ``(hidden, ...)`` on some HF versions
    and a bare tensor on transformers 4.57.x. Mirrors
    ``analysis/representation_shift.py``'s unwrap and the reference impl.
    """
    return output[0] if isinstance(output, tuple) else output


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

    The returned tensor for layer ``L`` is the SAME object the old
    ``output_hidden_states[L+1]`` read produced (``hs[0]`` for
    ``EMBED_LAYER``), shape ``(B, T, H)``. Byte-identical to the full-tuple
    read; the only difference is the unused layers are never materialized.

    Parameters
    ----------
    model
        Causal-LM, already on its device and in eval mode (this helper does
        not move or set the mode). A standard Llama-style decoder
        (``model.model.layers`` + ``model.model.embed_tokens``) takes the
        primary hook path; anything else falls back to the full-tuple read
        (covers non-standard models AND the CPU test stub).
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
        decorator the ``.detach()`` on the default path is a no-op, so the
        default-path return is the exact same tensor the
        ``output_hidden_states=True`` read produced — ``torch.equal(old, new)``
        holds and callers retain their own ``.float().cpu()`` / ``.mean(0)``
        / ``[-1]`` reductions.

    Returns
    -------
    dict[int, torch.Tensor]
        ``{layer: (B, T, H)}`` for every requested layer (when
        ``return_logits`` is ``False``).
    tuple[dict[int, torch.Tensor], torch.Tensor]
        ``(captured, logits)`` when ``return_logits`` is ``True``.
    """
    blocks = getattr(getattr(model, "model", None), "layers", None)
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    layers = list(layers)

    def _reduce(hs: torch.Tensor) -> torch.Tensor:
        return hs.detach().float().cpu() if detach_to_cpu else hs.detach()

    # ---- Fallback for non-standard models / CPU stubs (the labeled =True) ----
    if blocks is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
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
            handles.append(blocks[L].register_forward_hook(_make_hook(L)))
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,  # <-- the fix: do NOT materialize all
        )
    finally:
        for h in handles:
            h.remove()
    if return_logits:
        return captured, out.logits
    return captured
