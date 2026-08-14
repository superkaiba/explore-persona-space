"""Issue #2254 — multi-layer Δ-addition stacks + calibrated projection-patch hooks.

Plan §4.2 / §4.3 hook primitives, built ON TOP of the reused #1415 steering
module (`DeltaHook`, `generate_batch` at
``src/explore_persona_space/experiments/issue1415/steering.py``):

- ``MultiLayerDeltaHook``   — a thin stack of per-layer ``DeltaHook``s
  duck-typing the single-hook contract (``install`` / ``remove`` / ``arm`` /
  ``reset`` / ``_handle`` property / ``n_edits`` / context manager) so
  ``steering.generate_batch`` works unchanged with one object (its guard is
  ``assert hook._handle is not None`` + ``hook.arm(expected_prompt_len=T)``
  per draw). Mirrors ``PositionEditHookStack``
  (``experiments/issue2094/hooks.py``). Each layer carries its OWN direction
  vector and its OWN per-layer dose ``alpha_l`` — the plan-§4.2 ``c/K`` norm
  split is computed by the CALLER, never by the stack.
- ``ProjectionPatchHook``   — mirrors ``DeltaHook``'s prefill-only
  last-context-token position logic (padded position ``T - 1`` under LEFT
  padding, per-row prompt-length exactness asserted by ``generate_batch``),
  but computes the DATA-DEPENDENT edit ``h <- h + (target - <h, d_hat>) *
  d_hat`` at hook time (mean-matching projection patch, arXiv 2402.09631;
  plan §4.3). Neither ``PositionEditHook`` mode fits: ``add`` is a fixed
  delta, ``replace`` a fixed state. ``d_hat`` must arrive unit-normalized
  (asserted); ``target_proj`` is a scalar OR a per-row ``(B,)`` tensor.
- ``MultiLayerProjectionPatchHook`` — the multi-layer stack (per-layer
  ``d_hat`` + per-layer target), same duck-typed lifecycle.

Conventions (fail fast; shape asserts at boundaries; dtype/device follow the
hooked activation): see the #1415 module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import torch

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks
from explore_persona_space.experiments.issue1415.steering import DeltaHook

# ── shared per-layer stack lifecycle ──────────────────────────────────


class _LayerHookStack:
    """One child hook per layer, one lifecycle — the single-hook duck type.

    ``generate_batch`` touches only ``hook._handle`` (install guard) and
    ``hook.arm(expected_prompt_len=...)`` (per-draw arming), so a stack
    forwarding both to every child works unchanged (the
    ``PositionEditHookStack`` precedent, issue2094/hooks.py L241).
    """

    def __init__(self, hooks: Sequence):
        assert len(hooks) >= 1, "empty hook stack"
        layers = [h.layer for h in hooks]
        assert len(set(layers)) == len(layers), f"duplicate layers in stack: {layers}"
        self.hooks: list = list(hooks)

    @property
    def _handle(self):
        """Non-None iff every child hook is installed (generate_batch's precondition)."""
        return self if all(h._handle is not None for h in self.hooks) else None

    @property
    def n_edits(self) -> int:
        """Total forward passes edited across all child hooks (telemetry)."""
        return sum(h.n_edits for h in self.hooks)

    def install(self) -> Self:
        for h in self.hooks:
            h.install()
        return self

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def arm(self, expected_prompt_len: int) -> None:
        """Arm every child for the next generate call (per-draw reset)."""
        for h in self.hooks:
            h.arm(expected_prompt_len)

    def reset(self) -> None:
        for h in self.hooks:
            h.reset()

    def __enter__(self) -> Self:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()


# ── multi-layer Δ-addition (plan §4.2 layer-breadth arms) ─────────────


class MultiLayerDeltaHook(_LayerHookStack):
    """Joint-layer ``DeltaHook`` stack: each layer's OWN delta at its OWN dose.

    The caller passes per-layer deltas already scaled per the §4.2 norm-match
    convention (``alpha_l = (c / K) * rho_l`` for a K-layer band); this stack
    is deliberately arithmetic-free. Children must share one position mode
    (``all_positions`` uniform; ``edit_position`` / ``prefill_all`` /
    ``decode_only`` / ``replace`` modes are not part of the stack contract —
    they are armed via other entrypoints or single-hook only).
    """

    def __init__(self, hooks: Sequence[DeltaHook]):
        for h in hooks:
            assert isinstance(h, DeltaHook), type(h)
            assert h.edit_position is None, "edit_position mode is single-hook only (arm_at)"
            assert not (h.prefill_all or h.decode_only or h.replace), (
                "stack children must use the last-context-token or all_positions mode"
            )
        modes = {h.all_positions for h in hooks}
        assert len(modes) == 1, f"mixed all_positions modes in stack: {modes}"
        super().__init__(hooks)


def multi_layer_delta_hooks(
    model,
    layers: Sequence[int],
    deltas: Sequence[torch.Tensor],
    alphas: Sequence[float],
    *,
    all_positions: bool = False,
) -> MultiLayerDeltaHook:
    """Build a ``MultiLayerDeltaHook``: one ``DeltaHook`` per layer.

    ``deltas[i]`` (``(H,)`` or ``(B, H)``) and ``alphas[i]`` belong to
    ``layers[i]``; ``all_positions`` selects the plan-§4.2 position factor
    (False = at-the-context-vector prefill edit; True = every generated
    position, the persona-vectors regime).
    """
    assert len(layers) == len(deltas) == len(alphas), (len(layers), len(deltas), len(alphas))
    return MultiLayerDeltaHook(
        [
            DeltaHook(model, int(layer), delta, float(alpha), all_positions=all_positions)
            for layer, delta, alpha in zip(layers, deltas, alphas, strict=True)
        ]
    )


# ── calibrated projection patch (plan §4.3 patching arms) ─────────────


class ProjectionPatchHook:
    """Forward hook setting ``<h, d_hat> = target_proj`` at the last context token.

    Prefill-only, position ``T - 1`` of the padded prompt — byte-for-byte the
    ``DeltaHook`` default-mode position logic (under LEFT padding, asserted by
    ``generate_batch``, ``T - 1`` is every row's last real context token; the
    edit persists through generation via the KV cache). The edit itself is
    DATA-DEPENDENT, computed at hook time:

        h  <-  h + (target_proj - <h, d_hat>) * d_hat

    which moves the projection of ``h`` onto the unit direction ``d_hat`` to
    exactly ``target_proj`` while leaving the orthogonal complement — and
    every other position / every decode step — untouched.

    ``d_hat`` is ``(H,)`` and MUST be unit-normalized (asserted; the 1e-2
    tolerance absorbs bf16-normalized vectors, whose norm error is ~4e-3,
    while still failing loudly on raw residual-stream directions with norms
    ~10-100). ``target_proj`` is a python float / 0-dim tensor (broadcast
    over rows) or a per-row ``(B,)`` tensor. Dtype/device follow the hooked
    activation. Handles tuple and bare-tensor block outputs; edits
    OUT-OF-PLACE (clone).
    """

    def __init__(
        self,
        model,
        layer: int,
        d_hat: torch.Tensor,
        target_proj: torch.Tensor | float,
        expected_prompt_len: int | None = None,
    ):
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, (
            "ProjectionPatchHook requires a standard decoder (model.model.layers)"
        )
        assert 0 <= layer < len(blocks), (layer, len(blocks))
        assert d_hat.dim() == 1, d_hat.shape
        norm = float(d_hat.float().norm().item())
        assert abs(norm - 1.0) < 1e-2, f"d_hat must be unit-normalized, got ||d_hat|| = {norm}"
        if isinstance(target_proj, torch.Tensor):
            assert target_proj.dim() in (0, 1), target_proj.shape
        self.model = model
        self.layer = int(layer)
        self.module = blocks[layer]
        self.d_hat = d_hat
        self.target_proj = target_proj
        self.expected_prompt_len = expected_prompt_len
        self._handle = None
        self._prefill_seen = False
        self.n_edits = 0  # forward passes edited (telemetry / test hook)

    # -- lifecycle (mirrors DeltaHook) ---------------------------------
    def install(self) -> ProjectionPatchHook:
        assert self._handle is None, "ProjectionPatchHook already installed"
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def arm(self, expected_prompt_len: int) -> None:
        """Set the padded prompt length for the next generate call + reset state."""
        assert expected_prompt_len >= 1, expected_prompt_len
        self.expected_prompt_len = int(expected_prompt_len)
        self.reset()

    def reset(self) -> None:
        self._prefill_seen = False

    def __enter__(self) -> ProjectionPatchHook:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # -- the hook ------------------------------------------------------
    def _edit_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        if self._prefill_seen:
            return hidden
        assert self.expected_prompt_len is not None, (
            "ProjectionPatchHook.arm(expected_prompt_len) must be called before the prefill"
        )
        # Exactness: the prefill length must equal the tokenized-context length
        # (padded T; generate_batch asserts per-row unpadded length == the
        # individually tokenized context length), so the edit position T-1 is
        # exactly each row's last real context token under left padding.
        assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
        d = self.d_hat.to(device=hidden.device, dtype=hidden.dtype)
        assert d.shape == (H,), (d.shape, H)
        if isinstance(self.target_proj, torch.Tensor) and self.target_proj.dim() == 1:
            target = self.target_proj.to(device=hidden.device, dtype=hidden.dtype)
            assert target.shape == (B,), (target.shape, B)
        else:
            target = torch.full(
                (B,), float(self.target_proj), device=hidden.device, dtype=hidden.dtype
            )
        out = hidden.clone()
        slot = out[:, T - 1, :]  # (B, H)
        proj = slot @ d  # (B,) current projection per row
        out[:, T - 1, :] = slot + (target - proj)[:, None] * d
        self._prefill_seen = True
        self.n_edits += 1
        return out

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            edited = self._edit_tensor(output[0])
            return (edited, *output[1:])
        return self._edit_tensor(output)


class MultiLayerProjectionPatchHook(_LayerHookStack):
    """Joint-layer ``ProjectionPatchHook`` stack: per-layer ``d_hat`` + target."""

    def __init__(self, hooks: Sequence[ProjectionPatchHook]):
        for h in hooks:
            assert isinstance(h, ProjectionPatchHook), type(h)
        super().__init__(hooks)


def multi_layer_projection_patch_hooks(
    model,
    layers: Sequence[int],
    d_hats: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor | float],
) -> MultiLayerProjectionPatchHook:
    """Build a ``MultiLayerProjectionPatchHook``: one hook per layer.

    ``d_hats[i]`` (unit ``(H,)``) and ``targets[i]`` (scalar or per-row
    ``(B,)``) belong to ``layers[i]`` — the plan-§4.3 middle-band arm uses
    each layer's own calibration mean ``mu_pos`` as that layer's target.
    """
    assert len(layers) == len(d_hats) == len(targets), (len(layers), len(d_hats), len(targets))
    return MultiLayerProjectionPatchHook(
        [
            ProjectionPatchHook(model, int(layer), d_hat, target)
            for layer, d_hat, target in zip(layers, d_hats, targets, strict=True)
        ]
    )
