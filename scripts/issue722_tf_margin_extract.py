#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #722 — teacher-forced fixed +/- completion-margin DV (GPU phase).

The DV (per context C, per behavior B):

    margin(C,B) = mean_i [ LN base-model logP(pos_answer_i | C + probe_i) ]
                − mean_i [ LN base-model logP(neg_answer_i | C + probe_i) ]

where {(probe_i, pos_answer_i)} and {(probe_i, neg_answer_i)} are a FIXED set of
judge-confirmed positive / negative (probe, answer) pairs (the SAME set across
all contexts — only the context prefix C varies). "LN" = length-normalized:
the SUM log-prob over the answer tokens divided by n_answer_tokens.

This has NO selection-on-outcome bias: the answer set is fixed (taken once,
deterministically by index), not re-chosen per context — unlike #742's
`logp_pos_mean`, which selected each context's own judged-positive completions
and so failed to track the behavior rate (rho ~ -0.3).

Fixed +/- pools come from #661's judge-filtered survivors
(eval_results/issue_661/judge_filter.json — pos kept iff score>50, neg iff
score<50, REFUSAL/unparseable dropped). Behaviors: broad_em, refusal,
sycophancy. (harmful_compliance has no #661 pool -> EXCLUDED, noted in output.)

Contexts = the 50 #594 battery instances (system_prompt + prefix_messages).
Model = Qwen/Qwen2.5-7B-Instruct (base, matching v_A's foundation).

Content hygiene: pos completions for broad_em / harmful content are
trigger-dense. This script NEVER prints/inspects raw completion text — it loads
the JSON programmatically, feeds strings straight to the tokenizer/model, and
logs only counts / aggregate margins.
"""

from __future__ import annotations

import os

# Reduce allocator fragmentation for the variable-length teacher-forced batches
# (set before torch initializes CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue594_common import load_battery, messages_for_instance  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout
)
log = logging.getLogger("issue722.tf_margin")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BEHAVIORS = ["broad_em", "refusal", "sycophancy"]
EXCLUDED_NO_POOL = ["harmful_compliance"]  # no #661 pool -> excluded, reported

DEFAULT_CAP = 40  # FIXED pos + FIXED neg per behavior (deterministic by index)


def build_fixed_pairs(judge_filter: dict, behavior: str, cap: int, seed: int):
    """FIXED (probe, answer) pairs for one behavior — deterministic, cap each side.

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


@torch.no_grad()
def score_answer_logprobs_batched(model, tokenizer, instance, pairs, device, max_batch_tokens=8000):
    """Length-normalized logP(answer | C + probe) for every pair, under context C.

    For each pair builds chat = [system_C, *prefix_C, user(probe), assistant(answer)]
    via apply_chat_template, teacher-forces, and returns the SUM logP over the
    answer-token span / n_answer_tokens. Dynamic-batched by token budget.

    Returns list[float] parallel to `pairs` (LN logP per pair).
    """
    # Build (input_ids, answer_start, answer_end) per pair.
    encoded = []
    for p in pairs:
        msgs = messages_for_instance(instance, p["probe"])
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
    suffix_len = _assistant_suffix_len(tokenizer, instance, pairs[0])

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


def _assistant_suffix_len(tokenizer, instance, sample_pair) -> int:
    """Tokens the chat template appends AFTER the assistant content (e.g. <|im_end|>\\n).

    Computed by diffing a full assistant turn vs the same with a 1-char content,
    isolating the constant trailing-suffix token count.
    """
    msgs = messages_for_instance(instance, sample_pair["probe"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(PROJECT_ROOT / "data/issue594/battery.json"))
    ap.add_argument(
        "--judge-filter", default=str(PROJECT_ROOT / "eval_results/issue_661/judge_filter.json")
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=str(PROJECT_ROOT / "eval_results/issue_722/tf_margin/margins.json")
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="2 contexts, cap 4")
    args = ap.parse_args()

    t0 = time.time()
    cap = 4 if args.smoke else args.cap

    judge_filter = json.loads(Path(args.judge_filter).read_text())
    _meta, instances = load_battery(args.battery)
    if args.smoke:
        instances = instances[:2]
    ctx_ids = [inst["id"] for inst in instances]
    log.info("loaded %d contexts; cap=%d per side", len(instances), cap)

    # Build FIXED pairs per behavior (same set across all contexts).
    fixed = {}
    pool_meta = {}
    for b in BEHAVIORS:
        pos, neg, m = build_fixed_pairs(judge_filter, b, cap, args.seed)
        if args.smoke:
            pos, neg = pos[:cap], neg[:cap]
            m["n_pos_used"], m["n_neg_used"] = len(pos), len(neg)
        fixed[b] = (pos, neg)
        pool_meta[b] = m
        log.info(
            "behavior=%s pos_used=%d/%d neg_used=%d/%d",
            b,
            m["n_pos_used"],
            m["n_pos_available"],
            m["n_neg_used"],
            m["n_neg_available"],
        )

    log.info("loading model %s", args.model)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    device = next(model.parameters()).device

    # margins[ctx][behavior] = {margin, pos_mean, neg_mean, n_pos, n_neg}
    margins: dict[str, dict] = {c: {} for c in ctx_ids}
    for ci, inst in enumerate(instances):
        cid = inst["id"]
        for b in BEHAVIORS:
            pos, neg = fixed[b]
            pos_lp = score_answer_logprobs_batched(model, tokenizer, inst, pos, device)
            neg_lp = score_answer_logprobs_batched(model, tokenizer, inst, neg, device)
            pos_mean = float(sum(pos_lp) / len(pos_lp))
            neg_mean = float(sum(neg_lp) / len(neg_lp))
            margins[cid][b] = {
                "margin": pos_mean - neg_mean,
                "pos_mean_ln_logp": pos_mean,
                "neg_mean_ln_logp": neg_mean,
                "n_pos": len(pos_lp),
                "n_neg": len(neg_lp),
                "pos_ln_logp": pos_lp,
                "neg_ln_logp": neg_lp,
            }
        log.info(
            "[%d/%d] ctx=%s done (%.1fs elapsed)", ci + 1, len(instances), cid, time.time() - t0
        )

    out = {
        "analysis": "issue722_teacher_forced_fixed_posneg_margin_DV",
        "description": (
            "margin(C,B) = mean_i LN logP(pos_answer_i|C+probe_i) - mean_i LN "
            "logP(neg_answer_i|C+probe_i); FIXED #661 judge-filtered pairs across "
            "all contexts (no selection-on-outcome bias). Base model "
            f"{args.model}. LN = sum answer-token logP / n_answer_tokens."
        ),
        "model": args.model,
        "cap_per_side": cap,
        "behaviors": BEHAVIORS,
        "excluded_no_pool": EXCLUDED_NO_POOL,
        "context_ids": ctx_ids,
        "pool_meta": pool_meta,
        "margins": margins,
        "smoke": args.smoke,
        "elapsed_s": time.time() - t0,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s (%.1fs total)", out_path, time.time() - t0)
    return out


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
