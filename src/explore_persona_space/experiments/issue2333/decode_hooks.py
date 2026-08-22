"""Decode-step answer-position hooks for issue #2333 (plan §4.3).

``AnswerPositionEditHook`` edits (mode="replace") or records (mode="capture")
the OUTPUT of one decoder block at decode steps 1..k of a batched HF
``generate()`` call. Conventions mirror ``experiments/issue2094/hooks.py``
(``PositionEditHook``): forward-hook on the block, tuple-or-tensor output
unwrap, out-of-place clone before any write, per-(row, step) realized-edit
telemetry, explicit ``arm()`` before every draw.

Decode-step counting: the first forward with ``T > 1`` is the PREFILL — the
hook passes it through untouched and latches ``_prefill_seen``; every
subsequent ``T == 1`` forward increments ``_decode_step`` (step 1 = the first
generated answer token). A ``T == 1`` forward before any prefill fails loud.

``generate_batch_ids`` is the token-ID-based sibling of
``issue1415.steering.generate_batch``: rows are raw id lists (prefill arms
concatenate donor TOKEN IDS — re-tokenizing text would violate the BPE-seam
rule, gotchas.md "Teacher-forced capture inputs"), left-padded, per-draw
seeded, generated tokens returned as ids + text.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "AnswerPositionEditHook",
    "AnswerPositionEditHookStack",
    "generate_batch_ids",
    "joint_answer_hooks",
    "resolve_decoder_blocks_2333",
]


def _unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def resolve_decoder_blocks_2333(model):
    """Resolve the decoder block list for q25 (``model.model.layers``) and q35
    (``model.model.language_model.layers``) — strict, fails loud.

    Delegates to ``analysis.extraction._resolve_decoder_blocks`` (which walks
    the ``.model`` chain incl. the ``.language_model`` nesting) and raises
    instead of returning the None fallback.
    """
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _embed, depth = _resolve_decoder_blocks(model)
    if blocks is None:
        raise RuntimeError(
            f"could not resolve decoder blocks for {type(model).__name__} "
            "(no .model chain level exposes .layers or .language_model.layers)"
        )
    return blocks, depth


class AnswerPositionEditHook:
    """Forward hook editing/recording one block's output at decode steps 1..k.

    mode="replace": at decode step i (1-based), row b's single output position
    is REPLACED with ``donors[b][i-1]`` while ``i <= k_b`` (rows with
    ``donors[b] is None`` are never edited). Tokens remain model-sampled from
    the edited state — only the hidden state is swapped.

    mode="capture": records (pre-edit, detached, fp32 CPU) the output state of
    every row at decode steps 1..capture_k into ``self.captured[b]``.

    Telemetry (replace mode): ``realized_edits`` rows
    ``(row, step, cos_pre_vs_donor, pre_norm, donor_norm)`` — the post-assign
    state IS the donor bitwise (verified externally by the decode injection
    gate's downstream capture hook).
    """

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.mode: str | None = None
        self._donors: list[torch.Tensor | None] | None = None
        self._ks: list[int] | None = None
        self._capture_k = 0
        self._batch = 0
        self._expected_prompt_len: int | None = None
        self._prefill_seen = False
        self._decode_step = 0
        self.captured: list[list[torch.Tensor]] = []
        self.realized_edits: list[tuple[int, int, float, float, float]] = []
        self._handle = None

    # -- arming ------------------------------------------------------------
    def arm_replace(self, donors: Sequence[torch.Tensor | None], expected_prompt_len: int) -> None:
        """Arm for one draw. ``donors[b]`` is ``(k_b, H)`` (this layer's donor
        states for row b) or None (row never edited)."""
        ks = []
        for d in donors:
            if d is None:
                ks.append(0)
            else:
                assert d.dim() == 2, d.shape
                ks.append(int(d.shape[0]))
        self.mode = "replace"
        self._donors = list(donors)
        self._ks = ks
        self._batch = len(donors)
        self._expected_prompt_len = expected_prompt_len
        self._prefill_seen = False
        self._decode_step = 0
        # NOTE: realized_edits is deliberately NOT reset here — telemetry
        # accumulates across all K per-block draws (each draw re-arms), so
        # block-level summaries see every draw's edits, not only the last
        # draw's (r1 Minor: last-draw-only telemetry). It resets at __init__
        # (one stack per generation call site).

    def arm_capture(self, batch: int, capture_k: int, expected_prompt_len: int) -> None:
        """Arm to record every row's output state at decode steps 1..capture_k."""
        self.mode = "capture"
        self._donors = None
        self._ks = None
        self._batch = batch
        self._capture_k = capture_k
        self._expected_prompt_len = expected_prompt_len
        self._prefill_seen = False
        self._decode_step = 0
        self.captured = [[] for _ in range(batch)]

    def disarm(self) -> None:
        self.mode = None

    # -- hook --------------------------------------------------------------
    def __call__(self, module, args, output):
        if self.mode is None:
            return output
        hidden = _unwrap(output)
        B, T, _H = hidden.shape
        assert self._batch == B, (B, self._batch)
        if T > 1:
            # Prefill — passthrough; latch + reset the decode counter.
            assert self._expected_prompt_len is None or self._expected_prompt_len == T, (
                T,
                self._expected_prompt_len,
            )
            self._prefill_seen = True
            self._decode_step = 0
            return output
        if not self._prefill_seen:
            raise RuntimeError(f"layer {self.layer}: decode step before any prefill forward")
        self._decode_step += 1
        i = self._decode_step

        if self.mode == "capture":
            if i <= self._capture_k:
                for b in range(B):
                    self.captured[b].append(hidden[b, 0, :].detach().float().cpu())
            return output

        # replace
        out = None
        for b in range(B):
            k_b = self._ks[b]
            if i > k_b:
                continue
            if out is None:
                out = hidden.clone()  # out-of-place (issue2094/hooks.py convention)
            donor = self._donors[b][i - 1].to(device=hidden.device, dtype=hidden.dtype)
            pre = hidden[b, 0, :]
            cos = torch.nn.functional.cosine_similarity(
                pre.float().flatten(), donor.float().flatten(), dim=0
            )
            self.realized_edits.append(
                (b, i, float(cos), float(pre.float().norm()), float(donor.float().norm()))
            )
            out[b, 0, :] = donor
        if out is None:
            return output
        return _rewrap(output, out)

    # -- lifecycle ---------------------------------------------------------
    def install(self, block) -> None:
        assert self._handle is None, "hook already installed"
        self._handle = block.register_forward_hook(self)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class AnswerPositionEditHookStack:
    """One ``AnswerPositionEditHook`` per layer (all layers by default)."""

    def __init__(self, hooks: list[AnswerPositionEditHook]) -> None:
        self.hooks = hooks

    def arm_replace_batch(
        self,
        donors_full: Sequence[torch.Tensor | None],
        expected_prompt_len: int,
    ) -> None:
        """``donors_full[b]`` is ``(k_b, L, H)`` (all layers) or None."""
        for li, h in enumerate(self.hooks):
            per_layer = [d[:, li, :] if d is not None else None for d in donors_full]
            h.arm_replace(per_layer, expected_prompt_len)

    def arm_capture(self, batch: int, capture_k: int, expected_prompt_len: int) -> None:
        for h in self.hooks:
            h.arm_capture(batch, capture_k, expected_prompt_len)

    def captured_states(self) -> list[torch.Tensor]:
        """Per row: ``(n_steps_captured, L, H)`` fp32 CPU (n may be < capture_k
        when the row EOSed early)."""
        n_rows = len(self.hooks[0].captured)
        out = []
        for b in range(n_rows):
            n_steps = min(len(h.captured[b]) for h in self.hooks)
            if n_steps == 0:
                out.append(torch.empty(0, len(self.hooks), 0))
                continue
            rows = torch.stack(
                [torch.stack([h.captured[b][s] for h in self.hooks]) for s in range(n_steps)]
            )
            out.append(rows)  # (n_steps, L, H)
        return out

    def realized_edits(self) -> dict[int, list[tuple[int, int, float, float, float]]]:
        return {h.layer: list(h.realized_edits) for h in self.hooks}

    def disarm(self) -> None:
        for h in self.hooks:
            h.disarm()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    @property
    def installed(self) -> bool:
        return all(h._handle is not None for h in self.hooks)


def joint_answer_hooks(model, layers: Sequence[int] | None = None) -> AnswerPositionEditHookStack:
    """Install one hook per decoder block (default: ALL layers, plan §4.2)."""
    blocks, _depth = resolve_decoder_blocks_2333(model)
    if layers is None:
        layers = range(len(blocks))
    hooks = []
    for li in layers:
        h = AnswerPositionEditHook(layer=int(li))
        h.install(blocks[li])
        hooks.append(h)
    return AnswerPositionEditHookStack(hooks)


# ---------------------------------------------------------------------------
# Token-ID-based batched generation
# ---------------------------------------------------------------------------


def _eos_id_set(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gc_eos, int):
        ids.add(gc_eos)
    elif isinstance(gc_eos, (list, tuple)):
        ids.update(int(x) for x in gc_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    assert ids, "no EOS ids resolvable from generation_config or tokenizer"
    return ids


@torch.no_grad()
def generate_batch_ids(
    model,
    tokenizer,
    rows_ids: list[list[int]],
    *,
    n: int = 1,
    stack: AnswerPositionEditHookStack | None = None,
    donors_full: Sequence[torch.Tensor | None] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    seed_base: int = 42,
    greedy: bool = False,
) -> list[list[dict]]:
    """N draws for each raw-id row, LEFT-padded, per-draw seeded.

    Returns ``draws[i][b] = {"gen_ids": list[int] (pre-EOS), "text": str,
    "n_completion_tokens": int, "hit_eos": bool}``. When ``stack`` is armed
    for replace, ``donors_full`` must be provided and is re-armed per draw
    (each draw resets the prefill latch + decode counter).
    """
    assert rows_ids, "empty batch"
    device = next(model.parameters()).device
    lens = [len(r) for r in rows_ids]
    T = max(lens)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = int(tokenizer.eos_token_id)
    input_ids = torch.full((len(rows_ids), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows_ids), T), dtype=torch.long)
    for b, r in enumerate(rows_ids):
        input_ids[b, T - lens[b] :] = torch.tensor(r, dtype=torch.long)
        attn[b, T - lens[b] :] = 1
        # Token-identity assert: padded row tail carries the ids verbatim.
        assert input_ids[b, T - lens[b] :].tolist() == list(r)
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    eos_ids = _eos_id_set(model, tokenizer)

    all_draws: list[list[dict]] = []
    for i in range(n):
        torch.manual_seed(seed_base + i)
        if stack is not None:
            if donors_full is not None:
                stack.arm_replace_batch(donors_full, expected_prompt_len=T)
            else:
                # capture callers arm the stack themselves before calling with
                # n=1; re-assert the latch is fresh.
                for h in stack.hooks:
                    assert h.mode is not None, "stack installed but not armed"
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
        )
        gen = out[:, T:]
        rows_out: list[dict] = []
        for b in range(gen.shape[0]):
            toks = gen[b].tolist()
            n_comp = len(toks)
            hit_eos = False
            for j, t in enumerate(toks):
                if t in eos_ids:
                    n_comp = j
                    hit_eos = True
                    break
            gen_ids = toks[:n_comp]
            rows_out.append(
                {
                    "gen_ids": gen_ids,
                    "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                    "n_completion_tokens": n_comp,
                    "hit_eos": hit_eos,
                }
            )
        all_draws.append(rows_out)
    return all_draws
