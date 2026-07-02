"""Directional ablation ``h <- h - (h . v_hat) v_hat`` via per-block forward hooks.

Project-out mirror of the ADD-steering hook in
``scripts/issue623_extract_sycophancy_vector.py::steering_generate`` (line 481;
tuple/bare output handling lines 513-518; the ``1e-8`` norm epsilon line 511) —
the hook structure is lifted with provenance comments, never imported. The
formula is Arditi et al.'s directional ablation (arXiv 2406.11717), with ONE
named deviation: the paper ablates every residual-stream WRITE (attn-out and
mlp-out separately, plus the embedding); this module ablates decoder-block
OUTPUTS — equivalent orthogonality at block boundaries, not bit-identical to
the per-write-site variant (plan #863 §3.5; do not describe this module as
Arditi-faithful). The paper's embedding-included all-layer variant is available
by including ``EMBED_LAYER`` in the direction map.

Zero direction is an EXACT no-op: ``v_hat = 0 / (0 + 1e-8) = 0`` so
``h - (h . 0) * 0 = h`` bitwise for finite floats (pinned by the required
zero-direction generation-equality test).

Layer-index convention == ``analysis/extraction.py``: BLOCK index ``L`` hooks
``model.model.layers[L]``; ``EMBED_LAYER`` (-1) hooks ``model.model.embed_tokens``.
A hook that returns non-None REPLACES the module output for the caller AND for
later-registered hooks (PyTorch replace semantics — the same contract
``steering_generate`` relies on in production), so ablation propagates downstream
and capture hooks registered after it observe the projected stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import torch

from explore_persona_space.analysis.extraction import EMBED_LAYER

__all__ = [
    "EMBED_LAYER",
    "ablated",
    "ablation_hooks",
    "all_layer_directions",
    "single_layer_directions",
]


def _normalize(direction: torch.Tensor) -> torch.Tensor:
    """Unit-normalize a 1-D direction in fp32: ``d / (||d|| + 1e-8)``.

    Deliberate numeric deviation (plan #863 §3.5): the norm is computed in fp32
    BEFORE any cast to the model dtype (the hook casts the already-normalized
    ``v_hat`` at apply time), so a bf16/fp16 model never norms in low precision.
    The ``1e-8`` epsilon mirrors ``issue623:511`` and makes the zero direction an
    exact no-op.
    """
    assert direction.ndim == 1, direction.shape
    d = direction.detach().to(torch.float32)
    return d / (d.norm() + 1e-8)


def _project_out(hs: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
    """Remove the ``v_hat`` component from ``hs``: ``hs - (hs . v) v`` (vectorized).

    ``hs`` is ``(..., H)`` (typically ``(B, T, H)``); ``v_hat`` is unit-norm
    ``(H,)`` fp32, cast to ``hs``'s device/dtype at apply time.
    """
    v = v_hat.to(device=hs.device, dtype=hs.dtype)
    coef = hs @ v  # (..., ) — per-position projection coefficient
    return hs - coef.unsqueeze(-1) * v


def _make_projection_hook(v_hat: torch.Tensor):
    """Build a forward hook projecting ``v_hat`` out of the module output.

    Handles BOTH decoder-block output shapes (mirrors ``issue623:513-518``): a
    tuple ``(hidden, ...)`` on some HF versions -> ``(projected, *rest)``; a bare
    tensor on transformers 4.57.x -> the projected tensor. Returning the modified
    output makes PyTorch REPLACE the module output for the caller and for
    later-registered hooks.
    """

    def _hook(_module, _inp, output):
        if isinstance(output, tuple):
            return (_project_out(output[0], v_hat), *output[1:])
        return _project_out(output, v_hat)

    return _hook


def ablation_hooks(model, directions: Mapping[int, torch.Tensor]) -> list:
    """Register a project-out hook per ``{block_index: direction}`` entry; return handles.

    BLOCK-index convention == ``analysis/extraction.py``: ``L`` ->
    ``model.model.layers[L]``; ``EMBED_LAYER`` (-1) -> ``model.model.embed_tokens``
    (enables Arditi's embedding-included all-layer variant). Each direction is
    unit-normalized in fp32 at registration (see :func:`_normalize`).

    Raises ``ValueError`` on a non-standard decoder (no ``model.model.layers``) —
    hook-based ablation has no full-tuple fallback; fail loud rather than
    silently not ablating — and on any block index outside ``[0, n_blocks)``
    (checked BEFORE any hook registers; negative indices other than
    ``EMBED_LAYER`` are rejected rather than Python-aliasing from the end).
    If registration still fails partway (e.g. a malformed direction on a later
    entry), every already-registered hook is removed before the exception
    re-raises, so a caught crash never leaves the model silently ablated.
    """
    blocks = getattr(getattr(model, "model", None), "layers", None)
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if blocks is None:
        raise ValueError(
            "ablation_hooks requires a standard Llama/Qwen-style decoder exposing "
            "model.model.layers; refusing to silently skip ablation"
        )
    n_blocks = len(blocks)
    for layer in directions:
        if layer == EMBED_LAYER:
            if embed is None:
                raise ValueError(
                    "EMBED_LAYER ablation requires model.model.embed_tokens on this model"
                )
        elif not 0 <= layer < n_blocks:
            raise ValueError(
                f"block index {layer} out of range for a {n_blocks}-layer decoder "
                f"(valid: 0..{n_blocks - 1}, or EMBED_LAYER={EMBED_LAYER})"
            )
    handles = []
    try:
        for layer, direction in directions.items():
            hook = _make_projection_hook(_normalize(direction))
            if layer == EMBED_LAYER:
                handles.append(embed.register_forward_hook(hook))
            else:
                handles.append(blocks[layer].register_forward_hook(hook))
    except Exception:
        for h in handles:  # roll back partial registration — never leave the model ablated
            h.remove()
        raise
    return handles


@contextmanager
def ablated(model, directions: Mapping[int, torch.Tensor]):
    """Context manager: ablate ``directions`` inside the block, GUARANTEE removal on exit.

    try/finally handle removal mirrors ``analysis/extraction.py:161-169`` — hooks
    are removed even when the body raises, so a crashed eval never leaves the
    model silently ablated.
    """
    handles = ablation_hooks(model, directions)
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()


def single_layer_directions(layer: int, direction: torch.Tensor) -> dict[int, torch.Tensor]:
    """Direction map for single-layer ablation (the steering-regime shape)."""
    assert direction.ndim == 1, direction.shape
    return {int(layer): direction}


def all_layer_directions(r_b: torch.Tensor, layers: Sequence[int]) -> dict[int, torch.Tensor]:
    """Direction map pairing each layer with ITS OWN row of a ``(L, H)`` ``r_b``.

    ``layers`` is index-aligned with ``r_b`` rows (the :class:`DirectionResult`
    convention). Include ``EMBED_LAYER`` in ``layers`` only if ``r_b`` carries a
    matching embedding row.
    """
    layers = [int(layer) for layer in layers]
    assert r_b.ndim == 2 and r_b.shape[0] == len(layers), (tuple(r_b.shape), len(layers))
    return {layers[i]: r_b[i] for i in range(len(layers))}
