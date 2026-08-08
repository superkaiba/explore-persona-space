"""Issue #1415 — activation capture + residual-stream Δ-addition steering.

Core primitives for the V_c causal-sufficiency experiment (plan v5):

- ``capture_vectors``  — V_c at the last CONTEXT token (prefix-based AND
  context-based arms, per the standing prefix+context mapping rule) and V_a
  (mean residual over the ANSWER span, per completion then mean over draws),
  for a set of layers, batched.
- ``DeltaHook``        — forward hook on ``model.model.layers[L]`` adding
  ``alpha * delta`` to the residual stream at the last-context-token position
  on the FIRST forward pass (prefill), or at every generated position on every
  step (``all_positions=True``; the persona-vectors steering shape).
- ``generate_batch``   — batched HF ``generate()`` (single generation stack —
  vLLM deliberately NOT used in this experiment, stated plan deviation), N
  draws per condition with per-draw seeds ``seed_base + i`` set via
  ``torch.manual_seed`` before each generate call (plan v5 §11).
- ``coherence_check``  — the coherence gate primitive (incoherent iff <5
  whitespace tokens or a refusal opener); ``condition_passes`` = ≥50% of a
  condition's draws coherent.

Conventions (fail fast; shape asserts at boundaries):
- Contexts are ``{"system": str | None, "user": str}`` dicts; persona/system
  text always rides the system turn (never user/assistant turns).
- Teacher-forced capture inputs concatenate per-segment TOKEN IDS — never
  re-tokenize a concatenated string (BPE-seam gotcha, #1092/#1315).
- The prefix boundary is derived from the ``<|im_start|>`` special-token
  structure of the rendered context (special-token boundaries never BPE-merge),
  not from re-tokenizing a string slice.
"""

from __future__ import annotations

import torch

from explore_persona_space.analysis.extraction import (
    _resolve_decoder_blocks,
    extract_layer_activations,
)

IM_START_TOKEN = "<|im_start|>"

# Refusal openers for the coherence gate (prefix match, casefolded). Deliberately
# HIGH-confidence openers only — the gate flags degenerate/refusal draws, it is
# not the behavioral refusal judge (that is Sonnet-judged in a later phase).
REFUSAL_OPENERS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i cannot",
    "i can't",
    "i can not",
    "i won't",
    "i will not",
    "sorry,",
    "as an ai",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
)

MIN_COHERENT_TOKENS = 5  # whitespace tokens; <5 => incoherent (plan v5 coherence gate)


# ── context rendering / tokenization ─────────────────────────────────


def context_messages(context: dict) -> list[dict]:
    """Chat message list for a ``{"system", "user"}`` context dict.

    ``system`` is None/"" for the bare-default context (system turn omitted —
    the chat template then inserts the model's default system block).
    """
    assert isinstance(context.get("user"), str) and context["user"], context
    messages = []
    if context.get("system"):
        messages.append({"role": "system", "content": context["system"]})
    messages.append({"role": "user", "content": context["user"]})
    return messages


def render_context(tokenizer, context: dict) -> str:
    """Chat-template render of a context WITH the generation prompt appended."""
    return tokenizer.apply_chat_template(
        context_messages(context), tokenize=False, add_generation_prompt=True
    )


def context_token_ids(tokenizer, context: dict) -> list[int]:
    """Token ids of the rendered context (special tokens already in the render)."""
    ids = tokenizer(render_context(tokenizer, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context)
    return ids


def prefix_end_index(tokenizer, ids: list[int]) -> int:
    """Token index where the USER turn starts — the prefix/context boundary.

    The prefix is everything BEFORE the user query (the system block, explicit
    or template-default). The rendered single-turn context has exactly three
    ``<|im_start|>`` special tokens (system, user, assistant-generation-prompt);
    the prefix ends where the second one begins. Special tokens are atomic, so
    this boundary can never BPE-merge (gotchas: plain-text span boundaries).
    Returns ``prefix_end`` such that ``ids[:prefix_end]`` is the prefix and the
    prefix-arm V_c reads position ``prefix_end - 1``.
    """
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START_TOKEN)
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    occ = [i for i, t in enumerate(ids) if t == im_start_id]
    assert len(occ) == 3, (
        f"expected 3 {IM_START_TOKEN} occurrences (system/user/assistant) in the rendered "
        f"single-turn context, got {len(occ)} at {occ}"
    )
    prefix_end = occ[1]
    assert 2 <= prefix_end < len(ids), (prefix_end, len(ids))
    return prefix_end


# ── coherence gate ────────────────────────────────────────────────────


def coherence_check(texts: list[str]) -> list[bool]:
    """Per-draw coherence flags: incoherent iff <5 whitespace tokens or a refusal opener."""
    flags = []
    for text in texts:
        stripped = text.strip()
        norm = stripped.casefold()
        incoherent = len(stripped.split()) < MIN_COHERENT_TOKENS or any(
            norm.startswith(op) for op in REFUSAL_OPENERS
        )
        flags.append(not incoherent)
    return flags


def condition_passes(flags: list[bool], min_frac: float = 0.5) -> bool:
    """A condition passes the coherence gate iff >= ``min_frac`` of its draws are coherent."""
    assert len(flags) > 0
    return sum(flags) / len(flags) >= min_frac


# ── DeltaHook ─────────────────────────────────────────────────────────


class DeltaHook:
    """Forward hook adding ``alpha * delta`` to a decoder block's residual output.

    Default mode edits ONLY the last-context-token position on the FIRST forward
    pass (the prefill): position index ``T - 1`` of the padded prompt, which under
    LEFT padding (asserted by ``generate_batch``) is each row's last real context
    token — for an unpadded single row this is exactly
    ``len(tokenized_context) - 1``, asserted exactly via ``expected_prompt_len``.
    Decode steps are untouched.

    ``all_positions=True`` (the persona-vectors steering variant) edits the
    position generating each token on EVERY step: the last prompt position at
    prefill, then every decode-step position.

    ``edit_position`` (keyword-only, default ``None`` = off; the #1415
    hooked-unhooked-decomposition mode, plan v11 §3.2) edits ONLY position
    ``edit_position`` of the FIRST forward after :meth:`arm_at` — the
    teacher-forced re-forward right-pads ``ctx_ids + comp_ids`` and processes
    context + completion in ONE pass, so the last real context token sits at
    ``ctx_len - 1``, NOT ``T - 1``, and the generation-mode
    ``expected_prompt_len == T`` assert is deliberately inapplicable. Mutually
    exclusive with ``all_positions``; asserts ``edit_position < T`` at edit
    time. Every other code path is byte-identical (default ``None`` preserves
    current behavior).

    ``prefill_all`` (keyword-only; the #1769 ``prefill_only`` arm) adds
    ``alpha * delta`` to EVERY position of the FIRST forward pass (all prompt
    positions, left-pad slots included — the attention mask excludes pad
    positions from every real position's attention, so the pad edits are
    inert); every later (decode) forward is untouched.

    ``decode_only`` (keyword-only; the #1769 ``decode_only`` arm) SKIPS the
    first forward pass (the prefill — asserted against
    ``expected_prompt_len``) and adds ``alpha * delta`` at every position of
    every SUBSEQUENT forward — under the KV cache each decode step is a
    ``T = 1`` slice, i.e. exactly that step's newly generated position.

    ``prefill_all`` / ``decode_only`` are mutually exclusive with each other
    and with ``all_positions`` / ``edit_position``; existing modes are
    behavior-unchanged.

    ``delta`` is ``(H,)`` (broadcast over the batch) or ``(B, H)`` (per-row —
    lets one batched generate carry a different Δ per pair). The hook handles
    both tuple and bare-tensor block outputs and edits OUT-OF-PLACE (clone).

    ``replace=True`` (keyword-only, default ``False`` = byte-identical add
    path; the #1776 ``slot_patch_sufficiency`` mode) REPLACES the
    last-context-token activation wholesale with ``alpha * delta`` instead of
    adding it — the full-state activation patch. Only the last-context-token
    prefill mode supports it (mutually exclusive with ``all_positions`` and
    ``edit_position``).
    """

    def __init__(
        self,
        model,
        layer: int,
        delta: torch.Tensor,
        alpha: float,
        expected_prompt_len: int | None = None,
        all_positions: bool = False,
        *,
        edit_position: int | None = None,
        prefill_all: bool = False,
        decode_only: bool = False,
        replace: bool = False,
    ):
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, "DeltaHook requires a standard decoder (model.model.layers)"
        assert 0 <= layer < len(blocks), (layer, len(blocks))
        assert delta.dim() in (1, 2), delta.shape
        assert not (all_positions and edit_position is not None), (
            "edit_position mode is mutually exclusive with all_positions"
        )
        n_modes = sum(
            (bool(all_positions), edit_position is not None, bool(prefill_all), bool(decode_only))
        )
        assert n_modes <= 1, (
            "all_positions / edit_position / prefill_all / decode_only are mutually exclusive"
        )
        assert not (
            replace and (all_positions or edit_position is not None or prefill_all or decode_only)
        ), "replace mode supports ONLY the last-context-token prefill edit"
        self.model = model
        self.layer = layer
        self.module = blocks[layer]
        self.delta = delta
        self.alpha = float(alpha)
        self.expected_prompt_len = expected_prompt_len
        self.all_positions = bool(all_positions)
        self.edit_position = int(edit_position) if edit_position is not None else None
        self.prefill_all = bool(prefill_all)
        self.decode_only = bool(decode_only)
        self.replace = bool(replace)
        self._handle = None
        self._prefill_seen = False
        self.n_edits = 0  # forward passes edited (telemetry / test hook)

    # -- lifecycle -----------------------------------------------------
    def install(self) -> DeltaHook:
        assert self._handle is None, "DeltaHook already installed"
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

    def arm_at(self, edit_position: int) -> None:
        """Arm the ``edit_position`` mode for the next forward + reset the
        prefill latch (the teacher-forced per-chunk arming; plan v11 §3.2).
        Asserts the position is non-negative and the mode is not
        ``all_positions``; the ``edit_position < T`` bound is asserted at edit
        time (T is unknown until the forward runs)."""
        assert not self.all_positions, "arm_at() is incompatible with all_positions"
        assert not (self.prefill_all or self.decode_only), (
            "arm_at() is incompatible with prefill_all / decode_only (#1769 modes)"
        )
        assert edit_position >= 0, edit_position
        self.edit_position = int(edit_position)
        self.reset()

    def reset(self) -> None:
        self._prefill_seen = False

    def __enter__(self) -> DeltaHook:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # -- the hook ------------------------------------------------------
    def _edit_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        d = self.delta.to(device=hidden.device, dtype=hidden.dtype)
        assert d.shape[-1] == H, (d.shape, H)
        if d.dim() == 2:
            assert d.shape[0] == B, (d.shape, B)
        scaled = self.alpha * d  # (H,) or (B, H)
        if self.edit_position is not None:
            # Teacher-forced mode (#1415 hooked decomposition): edit ONLY
            # position ``edit_position`` on the FIRST forward after arm_at();
            # any later forward (there are none in the teacher-forced capture,
            # which re-arms per chunk) is untouched.
            assert not self.all_positions
            if self._prefill_seen:
                return hidden
            assert self.edit_position < T, (self.edit_position, T)
            out = hidden.clone()
            out[:, self.edit_position, :] = out[:, self.edit_position, :] + scaled
            self._prefill_seen = True
            self.n_edits += 1
            return out
        if self.prefill_all:
            # #1769 prefill_only arm: edit ALL positions of the FIRST forward
            # pass (every prompt position; left-pad slots are attention-masked
            # away from every real position, so their edits are inert). Every
            # decode-step forward is untouched.
            if self._prefill_seen:
                return hidden
            assert self.expected_prompt_len is not None, (
                "DeltaHook.arm(expected_prompt_len) must be called before the prefill"
            )
            assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
            out = hidden + (scaled[:, None, :] if scaled.dim() == 2 else scaled)
            self._prefill_seen = True
            self.n_edits += 1
            return out
        if self.decode_only:
            # #1769 decode_only arm: SKIP the prefill (asserted to be the
            # prompt-shaped first forward), then edit every position of every
            # subsequent forward — each decode step's single new position
            # (T = 1 under the KV cache).
            if not self._prefill_seen:
                assert self.expected_prompt_len is not None, (
                    "DeltaHook.arm(expected_prompt_len) must be called before the prefill"
                )
                assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
                self._prefill_seen = True
                return hidden
            out = hidden + (scaled[:, None, :] if scaled.dim() == 2 else scaled)
            self.n_edits += 1
            return out
        if self.all_positions:
            if not self._prefill_seen:
                # Prefill: edit ONLY the position generating the first token
                # (the last prompt position); every later step edits its own
                # (single) generated position.
                assert self.expected_prompt_len is not None, (
                    "DeltaHook.arm(expected_prompt_len) must be called before the prefill"
                )
                assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
                out = hidden.clone()
                out[:, T - 1, :] = out[:, T - 1, :] + scaled
                self._prefill_seen = True
            else:
                out = hidden + (scaled[:, None, :] if scaled.dim() == 2 else scaled)
            self.n_edits += 1
            return out
        # last-context-token mode: edit the prefill only.
        if self._prefill_seen:
            return hidden
        assert self.expected_prompt_len is not None, (
            "DeltaHook.arm(expected_prompt_len) must be called before the prefill"
        )
        # Exactness: the prefill length must equal the tokenized-context length
        # (padded T; for an unpadded row, len(tokenized_context)), so the edit
        # position T-1 is exactly len(tokenized_context) - 1.
        assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
        out = hidden.clone()
        if self.replace:
            # Full-state patch (#1776 slot_patch_sufficiency): the slot value
            # BECOMES alpha * delta (per-row), nothing is added.
            out[:, T - 1, :] = scaled
        else:
            out[:, T - 1, :] = out[:, T - 1, :] + scaled
        self._prefill_seen = True
        self.n_edits += 1
        return out

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            edited = self._edit_tensor(output[0])
            return (edited, *output[1:])
        return self._edit_tensor(output)


# ── batched generation ────────────────────────────────────────────────


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    contexts: list[dict],
    n: int = 10,
    hook: DeltaHook | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    seed_base: int = 42,
    render_fn=None,
    ids_fn=None,
) -> list[list[str]]:
    """Batched HF ``generate()``: N draws for each context, optional DeltaHook.

    One generate call per DRAW, batched across the ``contexts`` axis (per-draw
    seeds ``seed_base + i`` are set via ``torch.manual_seed`` before EACH
    generate call — plan v5 §11 — so draws stay per-seed reproducible while the
    context/pair axis is vectorized). Contexts are rendered with the chat
    template + generation prompt and LEFT-padded, so every row's last real
    context token sits at the shared prompt position ``T - 1`` (asserted
    exactly, per row, against the individually tokenized context lengths).

    ``render_fn`` / ``ids_fn`` (optional, DEFAULT = this module's single-turn
    ``render_context`` / ``context_token_ids`` — behavior unchanged for every
    existing caller) let a caller whose context dicts carry extra structure
    thread its OWN render: issue2094's multi-turn ``history`` contexts MUST
    pass the ``*_2094`` helpers, because this module's ``context_messages``
    silently ignores ``history`` and the hook's row_lengths/positions would
    then be computed against a DIFFERENT render than the one generated from.

    Returns ``results[b][i]`` = draw ``i`` of context ``b`` (new tokens only,
    special tokens skipped).
    """
    assert len(contexts) >= 1 and n >= 1
    assert max_new_tokens >= 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    per_ctx_ids = [(ids_fn or context_token_ids)(tokenizer, c) for c in contexts]
    texts = [(render_fn or render_context)(tokenizer, c) for c in contexts]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = prev_side
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    B, T = input_ids.shape
    assert len(contexts) == B
    assert max(len(ids) for ids in per_ctx_ids) == T, (T, [len(i) for i in per_ctx_ids])
    # Exactness assert: each row's unpadded length == len(tokenized_context),
    # so under left padding the edit position T-1 is that row's
    # len(tokenized_context)-1 token.
    for b, ids in enumerate(per_ctx_ids):
        row_len = int(attention_mask[b].sum().item())
        assert row_len == len(ids), (b, row_len, len(ids))
        assert input_ids[b, T - len(ids) :].tolist() == ids, f"row {b}: padded ids != ctx ids"

    if hook is not None:
        assert hook._handle is not None, "install the DeltaHook before generate_batch (use `with`)"

    results: list[list[str]] = [[] for _ in range(B)]
    do_sample = temperature > 0
    for i in range(n):
        torch.manual_seed(seed_base + i)
        if hook is not None:
            hook.arm(expected_prompt_len=T)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        assert out.shape[0] == B and out.shape[1] > T, out.shape
        for b in range(B):
            results[b].append(tokenizer.decode(out[b, T:], skip_special_tokens=True))
    return results


# ── activation capture (V_c both arms + V_a) ──────────────────────────


def _right_pad_batch(rows: list[list[int]], pad_id: int, device) -> tuple:
    """Right-pad token-id rows -> (input_ids, attention_mask) tensors."""
    assert len(rows) >= 1
    T = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    for b, r in enumerate(rows):
        input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[b, : len(r)] = 1
    return input_ids.to(device), mask.to(device)


@torch.no_grad()
def capture_vectors(
    model,
    tokenizer,
    contexts: list[dict],
    layers: list[int],
    completions: list[list[str]] | None = None,
    batch_size: int = 8,
) -> dict:
    """V_c (prefix-based AND context-based arms) + V_a per layer, batched.

    - ``v_c_context``: residual activation at the LAST CONTEXT TOKEN (last token
      of the chat-template render with the generation prompt appended) — the
      context arm (prefix + user query).
    - ``v_c_prefix``: activation at the last PREFIX token (everything before the
      user turn; boundary from the ``<|im_start|>`` special-token structure) —
      the prefix arm. Both arms read the SAME forward pass.
    - ``v_a`` (when ``completions[b]`` is given): mean activation over the
      ANSWER span of ``ctx_ids + completion_ids`` (token-ID concatenation —
      never re-tokenized text), per completion, then mean over the (non-empty)
      completions. Empty completions are dropped from the mean with a recorded
      count; all-empty fails loud.

    Returns ``{"layers": layers, "per_context": [record, ...]}`` where each
    record carries fp32 CPU tensors ``v_c_prefix``/``v_c_context`` of shape
    ``(L, H)``, and (if completions were given) ``v_a_mean`` ``(L, H)`` +
    ``v_a_per_completion`` ``(n_kept, L, H)`` + ``n_empty_completions``.
    """
    assert len(contexts) >= 1 and len(layers) >= 1
    if completions is not None:
        assert len(completions) == len(contexts), (len(completions), len(contexts))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    per_ctx_ids = [context_token_ids(tokenizer, c) for c in contexts]
    prefix_ends = [prefix_end_index(tokenizer, ids) for ids in per_ctx_ids]

    records: list[dict] = [
        {"ctx_len": len(ids), "prefix_end": pe}
        for ids, pe in zip(per_ctx_ids, prefix_ends, strict=True)
    ]

    # -- V_c pass: one forward per chunk of contexts (right-padded) -----
    for start in range(0, len(contexts), batch_size):
        chunk = list(range(start, min(start + batch_size, len(contexts))))
        rows = [per_ctx_ids[b] for b in chunk]
        input_ids, mask = _right_pad_batch(rows, pad_id, device)
        captured = extract_layer_activations(model, input_ids, layers, attention_mask=mask)
        for j, b in enumerate(chunk):
            L_ctx = len(per_ctx_ids[b])
            pe = prefix_ends[b]
            v_ctx = torch.stack([captured[layer][j, L_ctx - 1] for layer in layers])
            v_pre = torch.stack([captured[layer][j, pe - 1] for layer in layers])
            H = v_ctx.shape[-1]
            assert v_ctx.shape == (len(layers), H), v_ctx.shape
            records[b]["v_c_context"] = v_ctx.float().cpu()
            records[b]["v_c_prefix"] = v_pre.float().cpu()
        del captured

    # -- V_a pass: teacher-forced ctx_ids + completion_ids ---------------
    if completions is not None:
        for b in range(len(contexts)):
            comp_ids_list = [
                tokenizer(text, add_special_tokens=False)["input_ids"] for text in completions[b]
            ]
            kept = [ids for ids in comp_ids_list if len(ids) > 0]
            n_empty = len(comp_ids_list) - len(kept)
            assert len(kept) >= 1, f"context {b}: all {len(comp_ids_list)} completions empty"
            per_comp_means: list[torch.Tensor] = []
            for start in range(0, len(kept), batch_size):
                chunk = kept[start : start + batch_size]
                rows = [per_ctx_ids[b] + cids for cids in chunk]
                input_ids, mask = _right_pad_batch(rows, pad_id, device)
                captured = extract_layer_activations(model, input_ids, layers, attention_mask=mask)
                ctx_len = len(per_ctx_ids[b])
                for j, cids in enumerate(chunk):
                    span = slice(ctx_len, ctx_len + len(cids))  # answer span, unpadded coords
                    v_a = torch.stack(
                        [captured[layer][j, span].float().mean(dim=0) for layer in layers]
                    )
                    assert v_a.shape[0] == len(layers), v_a.shape
                    per_comp_means.append(v_a.cpu())
                del captured
            v_a_all = torch.stack(per_comp_means)  # (n_kept, L, H)
            records[b]["v_a_per_completion"] = v_a_all
            records[b]["v_a_mean"] = v_a_all.mean(dim=0)
            records[b]["n_empty_completions"] = n_empty

    return {"layers": list(layers), "per_context": records}


# ── position-binned answer profiles (answer-position-shift-profile) ───

# The 13 overlapping bin views over answer positions 0..n-1 (plan v8 §3.2):
# two absolute early bins, ten relative deciles, one absolute last bin.
BIN_NAMES: tuple[str, ...] = (
    "first",
    "tok2_5",
    *(f"dec{d}" for d in range(1, 11)),
    "last",
)


def bin_matrix(n: int) -> torch.Tensor:
    """Pooling matrix ``(13, n)`` over answer positions ``0..n-1``.

    Rows follow :data:`BIN_NAMES` — ``first`` (idx 0), ``tok2_5`` (idx 1-4),
    relative deciles ``dec1..dec10`` (``clamp((10*idx)//n, max=9)``), ``last``
    (idx n-1). Bins are deliberately OVERLAPPING views, not a partition
    (first ⊂ dec1, last ⊂ dec10). Non-empty rows sum to 1 (mean-pooling
    weights); an EMPTY bin (e.g. deciles at n < 10) is a NaN row — einsum
    against it yields NaN, the pre-registered short-span fallback (excluded
    from within-bin means downstream, never a zero). Asserts ``n >= 1``.
    """
    assert n >= 1, n
    idx = torch.arange(n)
    masks = [
        idx == 0,  # "first"   (absolute)
        (idx >= 1) & (idx <= 4),  # "tok2_5"  (absolute)
    ]
    dec = torch.clamp((10 * idx) // n, max=9)  # relative deciles
    masks += [dec == d for d in range(10)]  # "dec1".."dec10"
    masks += [idx == n - 1]  # "last"    (absolute)
    M = torch.stack([m.float() for m in masks])
    assert M.shape == (len(BIN_NAMES), n), M.shape
    s = M.sum(1, keepdim=True)
    return torch.where(s > 0, M / s, torch.nan)


@torch.no_grad()
def capture_binned_answer_profiles(
    model,
    tokenizer,
    context: dict,
    completions: list[str],
    layers: list[int],
    batch_size: int = 8,
    hook: DeltaHook | None = None,
    capture_ctx_vec: bool = False,
) -> dict:
    """Per-position-binned answer profiles for ONE context's completions.

    Mirrors :func:`capture_vectors`'s V_a pass EXACTLY — token-ID
    concatenation ``ctx_ids + comp_ids`` (never re-tokenized concatenated
    strings; BPE-seam gotcha), the same :func:`_right_pad_batch` +
    :func:`extract_layer_activations` forward (``logits_to_keep=1`` inherited),
    the same answer span ``slice(ctx_len, ctx_len + len(comp_ids))`` — with the
    mean-pool replaced by the 13-bin :func:`bin_matrix` einsum. Per KEPT
    completion it ALSO returns the plain span mean (one extra reduction; the
    §3.5 parity-gate input). Empty completions are dropped with a recorded
    count; all-empty fails loud (parent convention).

    ``hook`` (#1415 hooked-unhooked decomposition, plan v11 §3.2/§3.3): an
    INSTALLED :class:`DeltaHook` re-armed via ``hook.arm_at(ctx_len - 1)``
    before EACH chunk forward — every row in a chunk shares ``ctx_ids``
    (single-context capture), so ``ctx_len - 1`` is each row's last real
    context token under right padding, reproducing the generation-time
    prefill edit. Asserts ``hook.n_edits`` increments by EXACTLY 1 per chunk.
    ``capture_ctx_vec``: additionally return the per-chunk last-context-token
    vector ``captured[L][0, ctx_len - 1]`` at every captured layer —
    ``ctx_vec`` ``(n_chunks, L, H)`` fp32 (feeds the G2 edit-injection
    exactness gate) + ``ctx_vec_max_dev`` (max cross-chunk L2 deviation from
    the chunk mean, per-layer max — bf16 batch-composition jitter telemetry).
    Default-off arguments preserve the round-1 behavior byte-identically.

    Returns a dict with fp32 CPU tensors:
    ``profiles`` ``(n_kept, 13, L, H)`` (NaN rows for empty bins),
    ``span_mean`` ``(n_kept, L, H)``, plus ``layers``, ``bin_names``,
    ``comp_token_counts`` (kept completions, in kept order) and
    ``n_empty_completions`` (+ ``ctx_vec``/``ctx_vec_max_dev`` when
    ``capture_ctx_vec``).
    """
    assert len(completions) >= 1 and len(layers) >= 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    ctx_ids = context_token_ids(tokenizer, context)
    ctx_len = len(ctx_ids)
    comp_ids_list = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in completions]
    kept = [ids for ids in comp_ids_list if len(ids) > 0]
    n_empty = len(comp_ids_list) - len(kept)
    assert len(kept) >= 1, f"all {len(comp_ids_list)} completions empty for context {context}"
    if hook is not None:
        assert hook._handle is not None, "install the DeltaHook before the capture (use `with`)"

    profiles: list[torch.Tensor] = []
    span_means: list[torch.Tensor] = []
    ctx_vecs: list[torch.Tensor] = []
    for start in range(0, len(kept), batch_size):
        chunk = kept[start : start + batch_size]
        rows = [ctx_ids + cids for cids in chunk]
        input_ids, mask = _right_pad_batch(rows, pad_id, device)
        if hook is not None:
            n_edits_before = hook.n_edits
            hook.arm_at(ctx_len - 1)
        captured = extract_layer_activations(model, input_ids, layers, attention_mask=mask)
        if hook is not None:
            assert hook.n_edits == n_edits_before + 1, (
                f"DeltaHook edited {hook.n_edits - n_edits_before} time(s) on one chunk "
                "forward (expected exactly 1) — arming/latch broken"
            )
        if capture_ctx_vec:
            # Row 0's last-context-token activation per layer; rows share
            # ctx_ids, so any cross-row/cross-chunk spread is batch jitter.
            ctx_vecs.append(
                torch.stack([captured[L][0, ctx_len - 1].float() for L in layers]).cpu()
            )
        for j, cids in enumerate(chunk):
            span = slice(ctx_len, ctx_len + len(cids))  # answer span, unpadded coords
            acts = torch.stack([captured[L][j, span].float() for L in layers])  # (L, n, H)
            assert acts.shape[:2] == (len(layers), len(cids)), acts.shape
            M = bin_matrix(len(cids)).to(acts)  # (13, n); NaN rows for empty bins
            prof = torch.einsum("bn,lnh->blh", M, acts)  # (13, L, H)
            profiles.append(prof.cpu())
            span_means.append(acts.mean(dim=1).cpu())  # (L, H)
        del captured

    out = {
        "layers": list(layers),
        "bin_names": list(BIN_NAMES),
        "profiles": torch.stack(profiles),  # (n_kept, 13, L, H) fp32
        "span_mean": torch.stack(span_means),  # (n_kept, L, H) fp32
        "comp_token_counts": [len(c) for c in kept],
        "n_empty_completions": n_empty,
    }
    if capture_ctx_vec:
        cv = torch.stack(ctx_vecs)  # (n_chunks, L, H) fp32
        dev = (cv - cv.mean(dim=0, keepdim=True)).norm(dim=-1)  # (n_chunks, L)
        out["ctx_vec"] = cv
        out["ctx_vec_max_dev"] = float(dev.max())
    return out
