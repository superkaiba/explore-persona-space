"""Issue #2094 — per-row, per-position hidden-state edit hook (``PositionEditHook``).

Generalizes the #1415 ``DeltaHook``
(``src/explore_persona_space/experiments/issue1415/steering.py``) to:

- per-row edit-position LISTS (each row its own positions, in UNPADDED per-row
  coordinates) under LEFT padding — rows with differing prompt lengths in one
  batch, the ``generate_batch`` geometry;
- per-(row, position) delta tensors (``deltas[b]`` of shape ``(P_b, H)``);
- ``mode in {"add", "replace"}`` at ANY position — DeltaHook's ``replace=True``
  guard supports last-token-only replace (steering.py, "replace mode supports
  ONLY the last-context-token prefill edit"), hence this NEW class; DeltaHook
  itself is deliberately untouched;
- single-layer instances plus a ``PositionEditHookStack`` for joint-layer
  variants (one ``PositionEditHook`` per layer — the plan §4.2
  joint-middle 14-20 / joint-all-28 cells), duck-typing the same contract.

Duck-types DeltaHook's lifecycle so ``generate_batch`` works unchanged:
``install()`` / ``remove()`` / ``arm(expected_prompt_len)`` / ``reset()`` /
``_handle`` / context manager. The edit applies ONCE at prefill (the first
forward after ``arm``); the edited KV persists through decode via the KV cache
(the established #1415 convention). Decode-step forwards (``T == 1`` slices
under the KV cache) pass through untouched.

Batch-state API (new, beyond the DeltaHook contract):
``arm_batch(row_lengths, positions, deltas, mode=..., alpha=...)`` sets the
per-cell state once; ``arm(T)`` (called by ``generate_batch`` before EVERY
draw) then computes each row's PADDED absolute positions as
``p + (T - row_lengths[b])`` (LEFT padding puts real tokens at the END) and
resets the prefill latch, keeping the batch state.

Telemetry for the P1 injection-exactness gate: ``n_edits`` (edited forward
passes) and ``realized_edits`` (per-row unpadded + padded positions and the
APPLIED edit tensor — ``alpha * delta`` in the hidden dtype, stored fp32 CPU —
captured at the prefill that applied it, retrievable per armed cell).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

MODES: tuple[str, ...] = ("add", "replace")


class PositionEditHook:
    """Forward hook editing a decoder block's residual output at per-row positions.

    One instance per (model, layer). Per-cell state arrives via
    :meth:`arm_batch`; per-draw arming via :meth:`arm` (the DeltaHook
    contract ``generate_batch`` calls). ``mode="add"``:
    ``h[b, p] <- h[b, p] + alpha * delta[b][j]``; ``mode="replace"``:
    ``h[b, p] <- alpha * delta[b][0]`` (exactly ONE position per row —
    the full-state patch, plan §4.2). Edits are OUT-OF-PLACE (clone).
    """

    def __init__(self, model, layer: int):
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, (
            "PositionEditHook requires a standard decoder (model.model.layers)"
        )
        assert 0 <= layer < len(blocks), (layer, len(blocks))
        self.model = model
        self.layer = int(layer)
        self.module = blocks[layer]
        self._handle = None
        # -- per-cell batch state (arm_batch) --
        self.mode: str | None = None
        self.alpha: float = 1.0
        self._row_lengths: list[int] | None = None
        self._positions: list[list[int]] | None = None
        self._deltas: list[torch.Tensor] | None = None
        # -- per-draw armed state (arm) --
        self.expected_prompt_len: int | None = None
        self._flat_batch_idx: torch.Tensor | None = None
        self._flat_pos_idx: torch.Tensor | None = None
        self._flat_delta: torch.Tensor | None = None
        self._prefill_seen = False
        # -- telemetry --
        self.n_edits = 0  # forward passes edited
        self.realized_edits: list[dict] | None = None  # set at each applied prefill

    # -- lifecycle (DeltaHook duck-type) ---------------------------------
    def install(self) -> PositionEditHook:
        assert self._handle is None, "PositionEditHook already installed"
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> PositionEditHook:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # -- per-cell state ---------------------------------------------------
    def arm_batch(
        self,
        row_lengths: Sequence[int],
        positions: Sequence[Sequence[int]],
        deltas: Sequence[torch.Tensor],
        *,
        mode: str = "add",
        alpha: float = 1.0,
    ) -> None:
        """Set the per-cell edit state (UNPADDED per-row coordinates).

        ``row_lengths[b]`` = row b's real (unpadded) prompt length;
        ``positions[b]`` = row b's edit positions, ``0 <= p < row_lengths[b]``,
        unique; ``deltas[b]`` = tensor ``(len(positions[b]), H)`` — row j is the
        delta (add) / replacement state (replace) for ``positions[b][j]``.
        Invalidates any prior :meth:`arm` state; call :meth:`arm` (or run
        ``generate_batch``, which calls it) before the next forward.
        """
        assert mode in MODES, mode
        B = len(row_lengths)
        assert B >= 1, "empty batch"
        assert len(positions) == B and len(deltas) == B, (B, len(positions), len(deltas))
        hidden_sizes = set()
        for b, (rl, pos, d) in enumerate(zip(row_lengths, positions, deltas, strict=True)):
            assert int(rl) >= 1, (b, rl)
            assert len(pos) >= 1, f"row {b}: empty position list"
            assert len(set(pos)) == len(pos), f"row {b}: duplicate edit positions {list(pos)}"
            for p in pos:
                assert 0 <= int(p) < int(rl), (b, p, rl)
            if mode == "replace":
                assert len(pos) == 1, (
                    f"replace mode edits exactly ONE position per row (row {b}: {list(pos)})"
                )
            assert isinstance(d, torch.Tensor) and d.dim() == 2, (b, type(d))
            assert d.shape[0] == len(pos), (b, d.shape, len(pos))
            hidden_sizes.add(int(d.shape[1]))
        assert len(hidden_sizes) == 1, f"deltas disagree on hidden size: {sorted(hidden_sizes)}"
        self.mode = mode
        self.alpha = float(alpha)
        self._row_lengths = [int(r) for r in row_lengths]
        self._positions = [[int(p) for p in pos] for pos in positions]
        self._deltas = [d.detach() for d in deltas]
        # Invalidate per-draw state until arm() recomputes it.
        self.expected_prompt_len = None
        self._flat_batch_idx = None
        self._flat_pos_idx = None
        self._flat_delta = None
        self._prefill_seen = False
        self.realized_edits = None

    # -- per-draw arming (DeltaHook duck-type) ----------------------------
    def arm(self, expected_prompt_len: int) -> None:
        """Set the padded prompt length for the next forward + reset the latch.

        Called by ``generate_batch`` before every draw. Computes each row's
        PADDED absolute edit positions ``p + (T - row_lengths[b])`` (LEFT
        padding). :meth:`arm_batch` must have been called first.
        """
        assert self._row_lengths is not None, "arm_batch() must be called before arm()"
        T = int(expected_prompt_len)
        assert T >= 1, T
        assert max(self._row_lengths) <= T, (max(self._row_lengths), T)
        batch_idx: list[int] = []
        pos_idx: list[int] = []
        assert self._positions is not None and self._deltas is not None
        for b, (rl, pos) in enumerate(zip(self._row_lengths, self._positions, strict=True)):
            off = T - rl
            for p in pos:
                batch_idx.append(b)
                pos_idx.append(p + off)
        self.expected_prompt_len = T
        self._flat_batch_idx = torch.tensor(batch_idx, dtype=torch.long)
        self._flat_pos_idx = torch.tensor(pos_idx, dtype=torch.long)
        self._flat_delta = torch.cat(list(self._deltas), dim=0)
        self.reset()

    def reset(self) -> None:
        self._prefill_seen = False

    # -- the hook ----------------------------------------------------------
    def _edit_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._prefill_seen:
            return hidden  # decode steps (and any later forward) pass through
        B, T, H = hidden.shape
        assert self.expected_prompt_len is not None, (
            "PositionEditHook.arm(expected_prompt_len) must be called before the prefill"
        )
        assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
        assert self._row_lengths is not None and len(self._row_lengths) == B, (
            B,
            self._row_lengths,
        )
        assert self._flat_delta is not None
        d = self._flat_delta.to(device=hidden.device, dtype=hidden.dtype)
        assert d.shape[-1] == H, (d.shape, H)
        scaled = self.alpha * d  # (sum_b P_b, H); replace also applies alpha (DeltaHook parity)
        assert self._flat_batch_idx is not None and self._flat_pos_idx is not None
        bi = self._flat_batch_idx.to(hidden.device)
        pi = self._flat_pos_idx.to(hidden.device)
        out = hidden.clone()
        if self.mode == "replace":
            out.index_put_((bi, pi), scaled, accumulate=False)
        else:
            out.index_put_((bi, pi), scaled, accumulate=True)
        self._prefill_seen = True
        self.n_edits += 1
        # Telemetry: the realized applied edit per row (fp32 CPU), retrievable
        # after the draw — consumed by the P1 injection-exactness gate.
        applied = scaled.detach().float().cpu()
        realized: list[dict] = []
        k = 0
        assert self._positions is not None
        for b, pos in enumerate(self._positions):
            off = T - self._row_lengths[b]
            n = len(pos)
            realized.append(
                {
                    "row": b,
                    "layer": self.layer,
                    "mode": self.mode,
                    "alpha": self.alpha,
                    "positions_unpadded": list(pos),
                    "positions_padded": [p + off for p in pos],
                    "applied": applied[k : k + n].clone(),
                }
            )
            k += n
        self.realized_edits = realized
        return out

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            edited = self._edit_tensor(output[0])
            return (edited, *output[1:])
        return self._edit_tensor(output)


class PositionEditHookStack:
    """Joint-layer variant: one ``PositionEditHook`` per layer, one lifecycle.

    Duck-types the single-hook contract (``install`` / ``remove`` / ``arm`` /
    ``reset`` / ``_handle`` / context manager / ``n_edits``) so
    ``generate_batch`` works unchanged with one object; each child layer
    carries its OWN delta at that layer (plan §4.2: "the SAME edit installed
    at layers 14-20 simultaneously, each layer's own Δ at that layer").
    """

    def __init__(self, hooks: Sequence[PositionEditHook]):
        assert len(hooks) >= 1, "empty hook stack"
        layers = [h.layer for h in hooks]
        assert len(set(layers)) == len(layers), f"duplicate layers in stack: {layers}"
        self.hooks: list[PositionEditHook] = list(hooks)

    @property
    def _handle(self):
        """Non-None iff every child hook is installed (generate_batch's precondition)."""
        return self if all(h._handle is not None for h in self.hooks) else None

    @property
    def n_edits(self) -> int:
        return sum(h.n_edits for h in self.hooks)

    @property
    def realized_edits(self) -> list[dict] | None:
        """Concatenated per-layer realized-edit records (None until a prefill applied)."""
        per_layer = [h.realized_edits for h in self.hooks]
        if all(r is None for r in per_layer):
            return None
        out: list[dict] = []
        for r in per_layer:
            out.extend(r or [])
        return out

    def install(self) -> PositionEditHookStack:
        for h in self.hooks:
            h.install()
        return self

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def arm(self, expected_prompt_len: int) -> None:
        for h in self.hooks:
            h.arm(expected_prompt_len)

    def reset(self) -> None:
        for h in self.hooks:
            h.reset()

    def arm_batch_per_layer(
        self,
        row_lengths: Sequence[int],
        positions: Sequence[Sequence[int]],
        deltas_per_layer: Sequence[Sequence[torch.Tensor]],
        *,
        mode: str = "add",
        alpha: float = 1.0,
    ) -> None:
        """Joint-cell arming: SAME positions at every layer, per-layer deltas.

        ``deltas_per_layer[i]`` are the per-row delta tensors for
        ``self.hooks[i]``'s layer (order matches the stack's hooks).
        """
        assert len(deltas_per_layer) == len(self.hooks), (
            len(deltas_per_layer),
            len(self.hooks),
        )
        for h, deltas in zip(self.hooks, deltas_per_layer, strict=True):
            h.arm_batch(row_lengths, positions, deltas, mode=mode, alpha=alpha)

    def __enter__(self) -> PositionEditHookStack:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()


def joint_hooks(model, layers: Sequence[int]) -> PositionEditHookStack:
    """Build a joint-layer stack (one ``PositionEditHook`` per layer)."""
    return PositionEditHookStack([PositionEditHook(model, int(L)) for L in layers])
