"""Issue #2203 — input-dependent per-layer activation-CAPPING forward hook.

The #2094 ``PositionEditHook`` / #1415 ``DeltaHook`` apply a *fixed* delta; the
assistant-axis defense (Lu et al., arXiv 2601.10387) is INPUT-DEPENDENT — the
edit reads the current projection ``⟨h, v⟩`` of each position onto the axis, so
a new hook is genuinely needed (deliberately a NEW module: #2094's
``hooks.py`` edit-once-at-prefill contract stays byte-untouched — plan §4.1).

Three ops (plan §4.1, formulas verbatim), all vectorized over the edited
positions:

- ``op="cap"``          — ``h ← h - v·min(⟨h,v⟩ - τ, 0)`` (paper Eq. 1; raw
  contrast ``v``, τ computed against the SAME ``v`` — self-consistent). A
  FLOOR: raises the axis component up to τ when it falls below, leaving the
  orthogonal subspace untouched.
- ``op="axis_replace"`` — ``h ← h + v̂·(⟨h_def,v̂⟩ - ⟨h,v̂⟩)`` (query-preserving;
  ONLY the axis component moves to the default-assistant mean projection).
- ``op="full_replace"`` — ``h ← h_def`` (the default-assistant mean STATE at
  that position; query-destroying damage ceiling).

Four position sets (the localization ladder):

- ``prefix-end``  — one position: the last PREFIX token (boundary from the
  ``<|im_start|>`` special-token structure via ``steering.prefix_end_index``).
- ``context-end`` — one position: the last prompt token (``T-1`` under left
  padding — each row's last real context token).
- ``all-prompt``  — every prefill position (decode untouched).
- ``all-tokens``  — every prefill position AND every decode step (the paper's
  every-token setting; the decode-step branch mirrors
  ``DeltaHook.all_positions=True`` — a ``T==1`` slice under the KV cache is
  that step's new position).

The hook DUCK-TYPES the ``DeltaHook`` lifecycle
(``install``/``remove``/``arm(expected_prompt_len)``/``reset``/``_handle``/
context-manager/``n_edits``) so ``issue1415/steering.py::generate_batch`` runs
UNCHANGED. Per-cell (per-row) positions arrive via :meth:`arm_batch` (the
#2094 contract); ``arm(T)`` (called by ``generate_batch`` before EVERY draw)
computes the padded absolute positions and resets the prefill latch.

Telemetry (feeds the continuous axis-projection DV + the H2 firing guard +
the injection-exactness gate): per edited forward, the realized edit position
count, and — for the single-position modes — the per-row raw projection
``⟨h,v⟩`` BEFORE the edit, the unit projection ``⟨h,v̂⟩`` before/after, and a
``fired`` flag (``⟨h,v⟩ < τ`` — whether the cap actually clamped).

TRIGGER-DENSE note: this module names the intervention MECHANISTICALLY (a
per-position clamp of one residual-stream direction); no harmful content is
handled here.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

OPS: tuple[str, ...] = ("cap", "axis_replace", "full_replace")
POSITION_SETS: tuple[str, ...] = ("prefix-end", "context-end", "all-prompt", "all-tokens")


def apply_cap_op(
    h: torch.Tensor,
    op: str,
    v: torch.Tensor,
    v_hat: torch.Tensor,
    tau: float,
    h_def: torch.Tensor,
    proj_def: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one cap/replace op to a stack of hidden states, vectorized over positions.

    ``h`` is ``(N, H)`` (N edited positions flattened). All of ``v``/``v_hat``/
    ``h_def`` are ``(H,)`` on ``h``'s device/dtype. Returns
    ``(h_new, proj_raw_before, proj_unit_before, proj_unit_after)`` — all
    ``(N,)`` projections computed in fp32 for telemetry, ``h_new`` in ``h``'s
    dtype:

    - ``op="cap"``: ``h_new = h - v·clamp(⟨h,v⟩ - τ, max=0)`` (Eq. 1; only rows
      with ``⟨h,v⟩ < τ`` move — the FLOOR).
    - ``op="axis_replace"``: ``h_new = h + v̂·(proj_def - ⟨h,v̂⟩)`` (the axis
      component is set to ``proj_def``; the orthogonal complement is unchanged).
    - ``op="full_replace"``: ``h_new = h_def`` broadcast (the whole state).
    """
    assert op in OPS, op
    assert h.dim() == 2, h.shape
    H = h.shape[-1]
    assert v.shape == (H,) and v_hat.shape == (H,) and h_def.shape == (H,), (
        v.shape,
        v_hat.shape,
        h_def.shape,
        H,
    )
    hf = h.float()
    vf = v.float()
    vhat_f = v_hat.float()
    proj_raw = hf @ vf  # (N,) ⟨h,v⟩ — the τ-space projection (H2 firing guard)
    proj_unit_before = hf @ vhat_f  # (N,) ⟨h,v̂⟩ — the continuous axis-projection DV
    if op == "cap":
        excess = torch.clamp(proj_raw - float(tau), max=0.0)  # (N,), <= 0 when below τ
        h_new = (hf - vf[None, :] * excess[:, None]).to(h.dtype)
    elif op == "axis_replace":
        shift = float(proj_def) - proj_unit_before  # (N,)
        h_new = (hf + vhat_f[None, :] * shift[:, None]).to(h.dtype)
    else:  # full_replace
        h_new = h_def.to(h.dtype)[None, :].expand(h.shape[0], H).clone()
    proj_unit_after = h_new.float() @ vhat_f  # (N,)
    return h_new, proj_raw, proj_unit_before, proj_unit_after


class AxisCapHook:
    """Forward hook applying an input-dependent axis cap/replace at a layer.

    One instance per (model, layer). Per-cell state (row lengths + per-row
    unpadded edit positions) arrives via :meth:`arm_batch`; per-draw arming via
    :meth:`arm` (the ``generate_batch`` contract). The op reads the current
    projection, so the edit is recomputed on every armed forward. Edits are
    OUT-OF-PLACE (clone).
    """

    def __init__(
        self,
        model,
        layer: int,
        v: torch.Tensor,
        tau: float,
        h_def: torch.Tensor,
        *,
        op: str = "cap",
        position_set: str = "context-end",
    ):
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, "AxisCapHook requires a standard decoder (model.model.layers)"
        assert 0 <= layer < len(blocks), (layer, len(blocks))
        assert op in OPS, op
        assert position_set in POSITION_SETS, position_set
        assert v.dim() == 1 and h_def.dim() == 1 and v.shape == h_def.shape, (v.shape, h_def.shape)
        self.model = model
        self.layer = int(layer)
        self.module = blocks[layer]
        self.op = op
        self.position_set = position_set
        # -- axis geometry (fp32, moved to the hidden device/dtype at edit time) --
        self.v = v.detach().float()
        norm = float(self.v.norm())
        assert norm > 0, f"layer {layer}: zero axis vector"
        self.v_hat = self.v / norm
        self.tau = float(tau)
        self.h_def = h_def.detach().float()
        self.proj_def = float(self.h_def @ self.v_hat)
        # -- per-cell batch state (arm_batch) --
        self._row_lengths: list[int] | None = None
        self._prefix_ends: list[int] | None = None
        # -- per-draw armed state (arm) --
        self.expected_prompt_len: int | None = None
        self._edit_batch_idx: torch.Tensor | None = None  # single-position modes only
        self._edit_pos_idx: torch.Tensor | None = None
        self._real_start: torch.Tensor | None = None  # per-row first real position (left pad)
        self._prefill_seen = False
        self._handle = None
        # -- telemetry --
        self.n_edits = 0  # forward passes edited
        self.realized_edits: list[dict] | None = None  # set at each applied forward

    # -- lifecycle (DeltaHook duck-type) ---------------------------------
    def install(self) -> AxisCapHook:
        assert self._handle is None, "AxisCapHook already installed"
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> AxisCapHook:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # -- per-cell state ---------------------------------------------------
    def arm_batch(
        self,
        row_lengths: Sequence[int],
        prefix_ends: Sequence[int] | None = None,
    ) -> None:
        """Set per-row state (UNPADDED coordinates); invalidates prior :meth:`arm`.

        ``row_lengths[b]`` = row b's real (unpadded) prompt length. ``prefix_ends``
        (REQUIRED for ``position_set="prefix-end"``) = row b's prefix/context
        boundary index (``steering.prefix_end_index``); the prefix-end edit reads
        position ``prefix_ends[b] - 1``. Ignored for the other position sets.
        """
        B = len(row_lengths)
        assert B >= 1, "empty batch"
        for b, rl in enumerate(row_lengths):
            assert int(rl) >= 2, (b, rl)
        if self.position_set == "prefix-end":
            assert prefix_ends is not None and len(prefix_ends) == B, (
                "prefix-end position set requires per-row prefix_ends of matching length"
            )
            for b, (pe, rl) in enumerate(zip(prefix_ends, row_lengths, strict=True)):
                assert 1 <= int(pe) <= int(rl), (b, pe, rl)
        self._row_lengths = [int(r) for r in row_lengths]
        self._prefix_ends = [int(p) for p in prefix_ends] if prefix_ends is not None else None
        self.expected_prompt_len = None
        self._edit_batch_idx = None
        self._edit_pos_idx = None
        self._real_start = None
        self._prefill_seen = False
        self.realized_edits = None

    # -- per-draw arming (DeltaHook duck-type) ----------------------------
    def arm(self, expected_prompt_len: int) -> None:
        """Set the padded prompt length for the next forward + reset the latch.

        Called by ``generate_batch`` before every draw. Computes each row's
        PADDED absolute positions under LEFT padding (``off = T - row_len``).
        :meth:`arm_batch` must have run first.
        """
        assert self._row_lengths is not None, "arm_batch() must be called before arm()"
        T = int(expected_prompt_len)
        assert T >= 1, T
        assert max(self._row_lengths) <= T, (max(self._row_lengths), T)
        B = len(self._row_lengths)
        offs = [T - rl for rl in self._row_lengths]
        self._real_start = torch.tensor(offs, dtype=torch.long)  # first real position per row
        if self.position_set in ("prefix-end", "context-end"):
            if self.position_set == "prefix-end":
                assert self._prefix_ends is not None
                unpadded = [pe - 1 for pe in self._prefix_ends]
            else:  # context-end: last real token
                unpadded = [rl - 1 for rl in self._row_lengths]
            self._edit_batch_idx = torch.arange(B, dtype=torch.long)
            self._edit_pos_idx = torch.tensor(
                [u + off for u, off in zip(unpadded, offs, strict=True)], dtype=torch.long
            )
        else:  # all-prompt / all-tokens edit whole real spans (mask handles pads)
            self._edit_batch_idx = None
            self._edit_pos_idx = None
        self.expected_prompt_len = T
        self.reset()

    def reset(self) -> None:
        self._prefill_seen = False

    # -- the op --------------------------------------------------------------
    def _op_at(self, hidden: torch.Tensor, bi: torch.Tensor, pi: torch.Tensor) -> None:
        """In-place (on a clone) apply the op at the flat (bi, pi) index set."""
        sel = hidden[bi, pi, :]  # (N, H)
        v = self.v.to(device=hidden.device, dtype=hidden.dtype)
        vhat = self.v_hat.to(device=hidden.device, dtype=hidden.dtype)
        hdef = self.h_def.to(device=hidden.device, dtype=hidden.dtype)
        new, praw, pu_before, pu_after = apply_cap_op(
            sel, self.op, v, vhat, self.tau, hdef, self.proj_def
        )
        hidden[bi, pi, :] = new
        self._last_proj = (praw.detach().cpu(), pu_before.detach().cpu(), pu_after.detach().cpu())

    def _edit_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        assert self.v.shape[0] == H, (self.v.shape, H)
        decode_step = self._prefill_seen and self.position_set == "all-tokens" and T == 1
        # prefix-end / context-end / all-prompt edit ONLY the prefill; all-tokens
        # ALSO edits every decode step. Anything else after prefill passes through.
        if self._prefill_seen and not decode_step:
            return hidden
        if not self._prefill_seen:
            assert self.expected_prompt_len is not None, (
                "AxisCapHook.arm(expected_prompt_len) must be called before the prefill"
            )
            assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
            assert self._row_lengths is not None and len(self._row_lengths) == B, (
                B,
                self._row_lengths,
            )

        out = hidden.clone()
        if self.position_set in ("prefix-end", "context-end"):
            assert self._edit_batch_idx is not None and self._edit_pos_idx is not None
            bi = self._edit_batch_idx.to(hidden.device)
            pi = self._edit_pos_idx.to(hidden.device)
            self._op_at(out, bi, pi)
            n_pos = int(bi.numel())
            per_row_fired = self._last_proj[0] < self.tau  # (B,) raw proj < τ
            realized = {
                "layer": self.layer,
                "op": self.op,
                "position_set": self.position_set,
                "phase": "prefill",
                "n_positions": n_pos,
                "proj_raw_before": self._last_proj[0].clone(),  # (B,) ⟨h,v⟩
                "proj_unit_before": self._last_proj[1].clone(),  # (B,) ⟨h,v̂⟩
                "proj_unit_after": self._last_proj[2].clone(),
                "fired": per_row_fired.clone(),
                "fired_frac": float(per_row_fired.float().mean()),
            }
        else:
            # all-prompt / all-tokens: edit every REAL position (left-pad slots
            # are attention-masked from every real position — editing them is
            # inert; excluding them keeps the telemetry projections honest).
            if decode_step:
                bi = torch.arange(B, dtype=torch.long, device=hidden.device)
                pi = torch.zeros(B, dtype=torch.long, device=hidden.device)  # the single new pos
            else:
                assert self._real_start is not None
                start = self._real_start.to(hidden.device)
                bs, ps = [], []
                pos_row = torch.arange(T, device=hidden.device)
                for b in range(B):
                    real = pos_row[pos_row >= int(start[b])]
                    bs.append(
                        torch.full((real.numel(),), b, dtype=torch.long, device=hidden.device)
                    )
                    ps.append(real.to(torch.long))
                bi = torch.cat(bs)
                pi = torch.cat(ps)
            self._op_at(out, bi, pi)
            realized = {
                "layer": self.layer,
                "op": self.op,
                "position_set": self.position_set,
                "phase": "decode" if decode_step else "prefill",
                "n_positions": int(bi.numel()),
                "proj_raw_before_mean": float(self._last_proj[0].mean()),
                "proj_unit_before_mean": float(self._last_proj[1].mean()),
                "proj_unit_after_mean": float(self._last_proj[2].mean()),
                "fired_frac": float((self._last_proj[0] < self.tau).float().mean()),
            }

        if self.realized_edits is None:
            self.realized_edits = []
        self.realized_edits.append(realized)
        if not self._prefill_seen:
            self._prefill_seen = True
        self.n_edits += 1
        return out

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            edited = self._edit_tensor(output[0])
            return (edited, *output[1:])
        return self._edit_tensor(output)


class AxisCapHookStack:
    """Joint-layer variant: one :class:`AxisCapHook` per layer, one lifecycle.

    Duck-types the single-hook contract (``install``/``remove``/``arm``/
    ``reset``/``_handle``/context manager/``n_edits``) so ``generate_batch``
    works unchanged with one object; each child carries its OWN axis/τ/h_def at
    its layer.
    """

    def __init__(self, hooks: Sequence[AxisCapHook]):
        assert len(hooks) >= 1, "empty hook stack"
        layers = [h.layer for h in hooks]
        assert len(set(layers)) == len(layers), f"duplicate layers in stack: {layers}"
        ops = {h.op for h in hooks}
        psets = {h.position_set for h in hooks}
        assert len(ops) == 1 and len(psets) == 1, (
            f"stack mixes op/position_set: ops={ops} position_sets={psets}"
        )
        self.hooks: list[AxisCapHook] = list(hooks)
        self.op = hooks[0].op
        self.position_set = hooks[0].position_set

    @property
    def _handle(self):
        """Non-None iff every child hook is installed (generate_batch's precondition)."""
        return self if all(h._handle is not None for h in self.hooks) else None

    @property
    def n_edits(self) -> int:
        return sum(h.n_edits for h in self.hooks)

    @property
    def realized_edits(self) -> list[dict] | None:
        per = [h.realized_edits for h in self.hooks]
        if all(r is None for r in per):
            return None
        out: list[dict] = []
        for r in per:
            out.extend(r or [])
        return out

    def install(self) -> AxisCapHookStack:
        for h in self.hooks:
            h.install()
        return self

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def arm_batch(
        self,
        row_lengths: Sequence[int],
        prefix_ends: Sequence[int] | None = None,
    ) -> None:
        for h in self.hooks:
            h.arm_batch(row_lengths, prefix_ends)

    def arm(self, expected_prompt_len: int) -> None:
        for h in self.hooks:
            h.arm(expected_prompt_len)

    def reset(self) -> None:
        for h in self.hooks:
            h.reset()

    def __enter__(self) -> AxisCapHookStack:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()


def joint_axis_hooks(
    model,
    layers: Sequence[int],
    axis_by_layer: dict[int, torch.Tensor],
    tau_by_layer: dict[int, float],
    h_def_by_layer: dict[int, torch.Tensor],
    *,
    op: str = "cap",
    position_set: str = "context-end",
) -> AxisCapHookStack:
    """Build a joint-band :class:`AxisCapHookStack` over ``layers``.

    ``axis_by_layer`` / ``tau_by_layer`` / ``h_def_by_layer`` supply each layer's
    raw contrast vector ``v``, cap floor τ, and default-assistant mean state. All
    children share ``op`` and ``position_set`` (a band caps ONE op at ONE
    position set across its layers — the design's cell).
    """
    hooks = []
    for layer in layers:
        li = int(layer)
        assert li in axis_by_layer and li in tau_by_layer and li in h_def_by_layer, (
            f"layer {li} missing axis / tau / h_def"
        )
        hooks.append(
            AxisCapHook(
                model,
                li,
                axis_by_layer[li],
                tau_by_layer[li],
                h_def_by_layer[li],
                op=op,
                position_set=position_set,
            )
        )
    return AxisCapHookStack(hooks)
