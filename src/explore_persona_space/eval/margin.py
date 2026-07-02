"""Teacher-forced fixed +/- completion margin (llm-judging.md par. E2 rule 19; #851).

The DV (per context C, per behavior B):

    margin(C) = mean_i [ LN logP(pos_answer_i | C + probe_i) ]
              - mean_i [ LN logP(neg_answer_i | C + probe_i) ]

over FIXED judge-filtered pools (the SAME (probe, answer) set under every
context C => no selection-on-outcome bias; #722 validated rho(margin, rate)
all-positive). "LN" = length-normalized: the SUM log-prob over the answer
tokens divided by n_answer_tokens.

SECONDARY non-saturating companion DV -- never a behavioral leaderboard
(the #432->#456 teacher-forcing caution). NOT the marker DV (that keeps its
own three-space recipe, marker-leakage-measurement.md).

Promoted from ``scripts/issue722_tf_margin_extract.py`` (#851), numerics
unchanged: chat-template answer-span extraction, template-suffix exclusion,
degenerate-span fallback, length-sorted dynamic batching by token budget,
answer-span-only fp32 log-softmax (the #722 OOM-safe memory pattern), and
pad-token fallback to eos. The one decoupling: the #594-battery ``instance``
dict param is replaced by ``messages_fn: Callable[[str], list[dict]]`` (a
probe -> chat-messages callable); callers with a battery instance recover the
exact prior behavior via ``functools.partial(messages_for_instance, inst)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

MessagesFn = Callable[[str], list[dict]]  # probe -> chat messages (system/prefix/user)


@dataclass
class MarginResult:
    """Per-(context, behavior) margin read.

    Field ORDER matches the historical ``margins.json`` per-cell dict keys
    (``scripts/issue722_tf_margin_extract.py`` main()), so ``asdict()`` yields
    a key-for-key identical output JSON.
    """

    margin: float
    pos_mean_ln_logp: float
    neg_mean_ln_logp: float
    n_pos: int
    n_neg: int
    pos_ln_logp: list[float]
    neg_ln_logp: list[float]


def build_fixed_pairs(judge_filter: dict, behavior: str, cap: int):
    """FIXED (probe, answer) pairs for one behavior — deterministic, cap each side.

    Consumes the #661 judge-filter survivors schema
    ``{"behaviors": {b: {side: {"survivors": [...], "n_survivors": N}}}}``.

    Returns (pos_pairs, neg_pairs, meta) where each *_pairs is a list of
    {"probe": str, "answer": str, "instruction_idx", "probe_idx", "rollout_idx",
    "score"}. Deterministic selection: sort survivors by
    (instruction_idx, probe_idx, rollout_idx) then take the first `cap` (no
    outcome-dependent re-selection; the SAME set is reused for every context).
    """
    beh = judge_filter["behaviors"][behavior]

    def take(side: str):
        survivors = beh[side]["survivors"]
        ordered = sorted(
            survivors, key=lambda s: (s["instruction_idx"], s["probe_idx"], s["rollout_idx"])
        )
        chosen = ordered[:cap]
        pairs = [
            {
                "probe": s["probe"],
                "answer": s["text"],
                "instruction_idx": s["instruction_idx"],
                "probe_idx": s["probe_idx"],
                "rollout_idx": s["rollout_idx"],
                "score": s["score"],
            }
            for s in chosen
        ]
        return pairs, beh[side]["n_survivors"]

    pos_pairs, n_pos_avail = take("pos")
    neg_pairs, n_neg_avail = take("neg")
    meta = {
        "n_pos_available": n_pos_avail,
        "n_neg_available": n_neg_avail,
        "n_pos_used": len(pos_pairs),
        "n_neg_used": len(neg_pairs),
        "cap": cap,
        "selection": "sorted by (instruction_idx,probe_idx,rollout_idx), first `cap`",
        "eval_prompt_sha": beh.get("eval_prompt_sha"),
    }
    return pos_pairs, neg_pairs, meta


def _assistant_suffix_len(tokenizer, messages_fn: MessagesFn, sample_probe: str) -> int:
    """Tokens the chat template appends AFTER the assistant content (e.g. <|im_end|>\\n).

    Computed by diffing a full assistant turn vs the same with a 1-char content,
    isolating the constant trailing-suffix token count.
    """
    msgs = messages_fn(sample_probe)
    a = tokenizer.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "X"}], add_generation_prompt=False, tokenize=True
    )
    b = tokenizer.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "XY"}], add_generation_prompt=False, tokenize=True
    )
    # The two differ only inside the content; the trailing suffix is whatever is
    # AFTER the common content region. Find the trailing common suffix length.
    i = 0
    while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


@torch.no_grad()
def score_answer_logprobs_batched(
    model,
    tokenizer,
    messages_fn: MessagesFn,
    pairs: Sequence[Mapping],
    device,
    max_batch_tokens: int = 8000,
) -> list[float]:
    """Length-normalized logP(answer | C + probe) for every pair, under context C.

    For each pair builds chat = messages_fn(probe) + assistant(answer)
    via apply_chat_template, teacher-forces, and returns the SUM logP over the
    answer-token span / n_answer_tokens. Dynamic-batched by token budget.

    Returns list[float] parallel to `pairs` (LN logP per pair).

    Raises:
        ValueError: on empty ``pairs`` (was an opaque IndexError at ``pairs[0]``).
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")

    # Build (input_ids, answer_start, answer_end) per pair.
    encoded = []
    for p in pairs:
        msgs = messages_fn(p["probe"])
        # Prompt-side tokens (with generation prompt header) = everything up to
        # the assistant content. The answer span begins right after that.
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        full_msgs = [*msgs, {"role": "assistant", "content": p["answer"]}]
        full_ids = tokenizer.apply_chat_template(
            full_msgs, add_generation_prompt=False, tokenize=True
        )
        # answer span = [len(prompt_ids), len(full_ids)). The chat template
        # appends an <|im_end|> after the assistant content; we score the answer
        # CONTENT tokens only (exclude the trailing template tokens after answer).
        # To get the exact content span, re-tokenize answer alone is unreliable
        # (BPE merges at boundary), so we take [len(prompt_ids), end) where end
        # excludes the template's trailing turn-end tokens.
        ans_start = len(prompt_ids)
        ans_end = len(full_ids)
        # Drop trailing template tokens (the post-content <|im_end|>\n). Qwen
        # appends "<|im_end|>\n" (2 tokens) after assistant content.
        # Compute by tokenizing an empty assistant turn to find the suffix len.
        encoded.append((full_ids, ans_start, ans_end))

    # Determine the template's trailing-suffix length once (tokens appended after
    # assistant CONTENT): compare full vs content-stripped template.
    suffix_len = _assistant_suffix_len(tokenizer, messages_fn, pairs[0]["probe"])

    results = [None] * len(pairs)
    # Dynamic batching by total token budget (pad to longest in batch).
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    batch_idx = []
    batch_tok = 0
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    def flush(idxs):
        if not idxs:
            return
        maxlen = max(len(encoded[i][0]) for i in idxs)
        bsz = len(idxs)
        input_ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, maxlen), dtype=torch.long)
        for r, i in enumerate(idxs):
            ids = encoded[i][0]
            input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits  # (b, T, V) bf16
        # MEMORY: never materialize the full-vocab (B,T,V) fp32 log_softmax — for a
        # 152k-vocab 7B over a padded batch that is tens of GiB and OOMs (#722 crash).
        # Per row, slice only the answer-span logit rows, then compute
        # log P(tgt) = logit[pos, tgt] - logsumexp(logit[pos]) in fp32.
        for r, i in enumerate(idxs):
            ids = encoded[i][0]
            a_start, a_end = encoded[i][1], encoded[i][2]
            a_end = a_end - suffix_len  # exclude trailing <|im_end|>\n
            if a_end <= a_start:
                # Degenerate (empty answer after suffix strip) — score the whole
                # span as fallback (should not happen with real completions).
                a_end = encoded[i][2]
            # logP(token t) = log_softmax(logits[t-1])[ids[t]]
            pos = torch.arange(a_start - 1, a_end - 1, device=device)
            tgt = torch.tensor(ids[a_start:a_end], device=device)
            row = logits[r, pos, :].float()  # (n_ans, V) — answer-span only
            logz = torch.logsumexp(row, dim=-1)  # (n_ans,)
            tgt_logit = row.gather(1, tgt[:, None]).squeeze(1)  # (n_ans,)
            lp = tgt_logit - logz  # (n_ans,)
            n = lp.numel()
            results[i] = float(lp.sum().item() / max(n, 1))
        del logits

    for i in order:
        tlen = len(encoded[i][0])
        if batch_idx and (batch_tok + tlen > max_batch_tokens):
            flush(batch_idx)
            batch_idx, batch_tok = [], 0
        batch_idx.append(i)
        batch_tok += tlen
    flush(batch_idx)
    assert all(r is not None for r in results)
    return results


@torch.no_grad()
def compute_tf_margin(
    model,
    tokenizer,
    messages_fn: MessagesFn,
    pos_pairs: Sequence[Mapping],
    neg_pairs: Sequence[Mapping],
    *,
    device,
    max_batch_tokens: int = 8000,
) -> MarginResult:
    """The fixed +/- completion margin for one context (rule-19 DV).

    Thin wrapper over ``score_answer_logprobs_batched`` reproducing the
    per-behavior arithmetic of the #722 extract script's main(): score both
    fixed pools under the SAME context (via ``messages_fn``), mean each, and
    return ``MarginResult(margin=pos_mean - neg_mean, ...)``.

    Raises:
        ValueError: on an empty pool (was a ZeroDivisionError in the script).
    """
    if not pos_pairs:
        raise ValueError("pos_pairs must be non-empty")
    if not neg_pairs:
        raise ValueError("neg_pairs must be non-empty")
    pos_lp = score_answer_logprobs_batched(
        model, tokenizer, messages_fn, pos_pairs, device, max_batch_tokens=max_batch_tokens
    )
    neg_lp = score_answer_logprobs_batched(
        model, tokenizer, messages_fn, neg_pairs, device, max_batch_tokens=max_batch_tokens
    )
    pos_mean = float(sum(pos_lp) / len(pos_lp))
    neg_mean = float(sum(neg_lp) / len(neg_lp))
    return MarginResult(
        margin=pos_mean - neg_mean,
        pos_mean_ln_logp=pos_mean,
        neg_mean_ln_logp=neg_mean,
        n_pos=len(pos_lp),
        n_neg=len(neg_lp),
        pos_ln_logp=pos_lp,
        neg_ln_logp=neg_lp,
    )
