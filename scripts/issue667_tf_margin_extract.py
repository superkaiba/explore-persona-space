#!/usr/bin/env python3
# math/scientific notation in docstrings + messages
"""Issue #667 — 2-index teacher-forced fixed +/- completion-margin DV (GPU phase).

The behavioral DV this amendment swaps in for #667's judged rate ``G`` (plan v6
§4.2), per off-diagonal cell (behavior b, source context C, target context C'):

    tf_margin_leak(b, C -> C') = margin_trained(source_adapter_C, C')
                               - margin_base(C')

    margin(theta, C') = mean_i [ LN logP_theta(FIXED pos_i | C' + probe_i) ]
                      - mean_i [ LN logP_theta(FIXED neg_i | C' + probe_i) ]

teacher-forced through the base model theta0 (-> ``margin_base``) AND the
source-adapter model theta+ = base + #537 adapter for (b, C) (-> ``margin_trained``),
with the FIXED pos/neg answer pools held IDENTICAL across every target context C'
(only the context prefix C' varies — #722's fixed-pool discipline, NO
selection-on-outcome bias). "LN" = length-normalized: the SUM answer-token logP
divided by n_answer_tokens over the answer-CONTENT span (trailing ``<|im_end|>\\n``
template tokens stripped, per ``_assistant_suffix_len``).

The base margin ``margin_base(C')`` is source-INDEPENDENT (one base pass per
(target, behavior)), so it is computed ONCE per (target, behavior) and cached
across all 16 sources of that behavior — the base forwards run 30 targets x 3
behaviors, NOT x 48 cells.

# Vendored from scripts/issue722_tf_margin_extract.py + scripts/issue722_tf_margin_analysis.py
# (branch issue-722-tf-margin, commits 2f824110/27d1106661) — byte-identical function bodies.
# Substitution: messages_for_instance (issue594_common / #594 battery) ->
# i537_contexts.build_messages (#537 target-context builder, routed through
# issue667_extract.build_messages_for). Do NOT re-derive or "improve" — the
# byte-identical vendor keeps the tf-margin recipe validated by #722 verbatim.

Content hygiene (CLAUDE.md): pos completions for broad_em are trigger-dense.
This script NEVER prints/inspects raw completion text — it loads the JSON
programmatically, feeds strings straight to the tokenizer/model, and logs only
counts / aggregate margins.
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout
)
log = logging.getLogger("issue667.tf_margin_extract")

# #667 (em/sycophancy/fact); marker/refusal excluded (inherited from #667 §3).
DEFAULT_CAP = 40  # FIXED pos + FIXED neg per behavior (deterministic by index)

# The judge_filter behavior key differs from the #667 behavior label for em:
# #661's judge_filter uses "broad_em" (Betley's naming); #667 uses "em". Map at
# the build_fixed_pairs call site (sycophancy/refusal keys match verbatim).
JUDGE_FILTER_BEHAVIOR = {"em": "broad_em", "sycophancy": "sycophancy", "fact": "fact"}

# #661 judge-filtered survivor pools (em/syco). Committed in git on `main` and
# inherited by the issue-667 branch (10.8 MB); ALSO mirrored on the HF data repo.
JUDGE_FILTER_LOCAL = "eval_results/issue_661/judge_filter.json"
JUDGE_FILTER_HF = "issue661_rb_extraction_divergence/judge_filter.json"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# The NEW fact fixed +/- pool (the one new data artifact, plan §4.3).
FACT_POOL_LOCAL = "data/issue_667/fact_fixed_pool_v1"


# ─────────────────────────────────────────────────────────────────────────────
# The ONE substitution — messages_for_instance -> i537_contexts.build_messages
# ─────────────────────────────────────────────────────────────────────────────


def messages_for_instance(instance: dict, probe: str) -> list[dict]:
    """The #667 substitution for #722's ``messages_for_instance`` (#594 battery).

    #722's vendored helpers below call ``messages_for_instance(instance, probe)``.
    Here ``instance`` is the #537 target-context descriptor
    ``{"registry", "demos", "cid", "behavior"}`` and this shim routes to the
    BYTE-FAITHFUL #537 context builder ``issue667_extract.build_messages_for``
    (which threads the ICL-demo bank for F3 contexts) — so the teacher-forced
    prefix is the exact #537 context #667 reads its activations off, NOT #722's
    #594 battery. This is the SINGLE named substitution of the vendor.
    """
    from issue667_extract import build_messages_for

    return build_messages_for(
        instance["registry"],
        instance["demos"],
        instance["cid"],
        instance["behavior"],
        probe,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VENDORED byte-identical from scripts/issue722_tf_margin_extract.py
# (branch issue-722-tf-margin) — function bodies unchanged. The only diff vs the
# source is that they call the module-local messages_for_instance shim above.
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# NEW — fixed-pool loaders (em/syco from #661, fact from the #667 v1 pool)
# ─────────────────────────────────────────────────────────────────────────────


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def _load_judge_filter() -> dict:
    """#661 judge_filter.json — prefer the committed git copy, fall back to HF.

    Committed in git at ``eval_results/issue_661/judge_filter.json`` (on ``main``,
    inherited by the issue-667 branch the pod checks out); ALSO mirrored on the
    HF data repo. A sparse worktree that excludes ``eval_results/`` for a local
    smoke falls back to the HF download.
    """
    p = PROJECT_ROOT / JUDGE_FILTER_LOCAL
    if p.exists():
        return json.loads(p.read_text())
    log.info("judge_filter.json not in git tree at %s -> HF fallback %s", p, JUDGE_FILTER_HF)
    return json.loads(Path(_hf(JUDGE_FILTER_HF)).read_text())


def _load_fact_pool(
    cap: int, *, pool_dir: Path | None = None
) -> tuple[list[dict], list[dict], dict]:
    """The NEW fact fixed +/- pool (data/issue_667/fact_fixed_pool_v1/), deterministic first-cap.

    Same {probe, answer, probe_idx, rollout_idx, score} schema build_fixed_pairs
    consumes for em/syco. Deterministic first-``cap`` by (probe_idx, rollout_idx)
    — no outcome-dependent re-selection. Raises FileNotFoundError if the pool was
    not built (Phase build-fact-pool runs first; fail loud, never silently skip).
    """
    pool_dir = pool_dir if pool_dir is not None else (PROJECT_ROOT / FACT_POOL_LOCAL)
    pos_path, neg_path = pool_dir / "pos.jsonl", pool_dir / "neg.jsonl"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"fact fixed pool missing under {pool_dir} (pos.jsonl/neg.jsonl). Run the "
            "build-fact-pool phase (scripts/issue667_build_fact_pool.py) first."
        )

    def _read(path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: (r.get("probe_idx", 0), r.get("rollout_idx", 0)))
        return rows[:cap]

    pos, neg = _read(pos_path), _read(neg_path)
    meta = {
        "n_pos_available": sum(1 for _ in pos_path.read_text().splitlines() if _.strip()),
        "n_neg_available": sum(1 for _ in neg_path.read_text().splitlines() if _.strip()),
        "n_pos_used": len(pos),
        "n_neg_used": len(neg),
        "cap": cap,
        "selection": "sorted by (probe_idx,rollout_idx), first `cap`",
        "source": "data/issue_667/fact_fixed_pool_v1 (NEW #667 pool)",
    }
    return pos, neg, meta


def build_fixed_pool_for(
    behavior: str, cap: int, seed: int, *, fact_pool_dir: Path | None = None
) -> tuple[list[dict], list[dict], dict]:
    """FIXED (pos, neg) pairs for a #667 behavior. em/syco -> #661; fact -> #667 pool."""
    if behavior == "fact":
        return _load_fact_pool(cap, pool_dir=fact_pool_dir)
    jf = _load_judge_filter()
    key = JUDGE_FILTER_BEHAVIOR[behavior]
    return build_fixed_pairs(jf, key, cap, seed)


# ─────────────────────────────────────────────────────────────────────────────
# NEW — the 2-index driver (source adapter x target context; base-vs-trained leak)
# ─────────────────────────────────────────────────────────────────────────────


def extract_tf_margins_2index(
    behavior: str,
    source_cid: str,
    seed: int,
    targets: list[str],
    fixed_pos: list[dict],
    fixed_neg: list[dict],
    device,
    *,
    registry: dict,
    demos: dict,
    base_margin_cache: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """2-index tf-margin extraction for ONE source-adapter cell (behavior b, source C).

    For each target C' in ``targets`` (the 30 #537 eval cids): teacher-forces the
    FIXED (probe_i, pos_i) and (probe_i, neg_i) pairs through the trained model
    theta+ = base + #537 adapter for (b, C), giving ``margin_trained(C, C')``. The
    base-side ``margin_base(C')`` is source-INDEPENDENT and cached per (target,
    behavior) in ``base_margin_cache`` (so the base pass runs once per target,
    reused across all sources of the behavior).

    Returns {tcid: {margin_trained, margin_base, tf_margin_leak,
    pos_mean_trained, neg_mean_trained, pos_mean_base, neg_mean_base,
    n_pos, n_neg}}.
    """
    from issue667_extract import (  # REUSED verbatim (adapter stage + gauge + base/trained load)
        assert_adapter_gauge,
        load_base_and_trained,
        stage_adapter_local,
    )

    from explore_persona_space.analysis.issue667 import BASE_MODEL

    if base_margin_cache is None:
        base_margin_cache = {}

    adapter_dir = stage_adapter_local(behavior, source_cid, seed)  # REUSED #375/#399 per-file stage
    assert_adapter_gauge(adapter_dir, behavior)  # fitness (f)/(g): base id + rsLoRA
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    _, base, trained = load_base_and_trained(adapter_dir, device, dtype)  # REUSED (rsLoRA honored)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))

    out: dict[str, dict] = {}
    for tcid in targets:
        instance = {"registry": registry, "demos": demos, "cid": tcid, "behavior": behavior}
        # base margin (source-independent) — compute once per (target, behavior), cache.
        if tcid not in base_margin_cache:
            mb_pos = _mean(score_answer_logprobs_batched(base, tok, instance, fixed_pos, device))
            mb_neg = _mean(score_answer_logprobs_batched(base, tok, instance, fixed_neg, device))
            base_margin_cache[tcid] = {
                "margin_base": mb_pos - mb_neg,
                "pos_mean_base": mb_pos,
                "neg_mean_base": mb_neg,
            }
        bc = base_margin_cache[tcid]
        # trained margin (source-specific).
        mt_pos = _mean(score_answer_logprobs_batched(trained, tok, instance, fixed_pos, device))
        mt_neg = _mean(score_answer_logprobs_batched(trained, tok, instance, fixed_neg, device))
        margin_trained = mt_pos - mt_neg
        out[tcid] = {
            "margin_trained": margin_trained,
            "margin_base": bc["margin_base"],
            "tf_margin_leak": margin_trained - bc["margin_base"],
            "pos_mean_trained": mt_pos,
            "neg_mean_trained": mt_neg,
            "pos_mean_base": bc["pos_mean_base"],
            "neg_mean_base": bc["neg_mean_base"],
            "n_pos": len(fixed_pos),
            "n_neg": len(fixed_neg),
        }
    # Free the trained PeftModel + base before the next cell loads its own.
    del base, trained
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — one source-adapter cell (invoked by the dispatcher as a CVD-pinned subproc)
# ─────────────────────────────────────────────────────────────────────────────


def _device(gpu_id: int, cpu_only: bool):
    if cpu_only or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")  # CVD pins the physical GPU in the launcher env


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_cell(args) -> int:
    from issue667_extract import stage_inputs

    from explore_persona_space.experiments.i537_contexts import (
        eval_cids_for,
        load_icl_demos,
        load_registry,
    )

    device = _device(args.gpu_id, args.cpu_only)
    behavior, source_cid, seed = args.behavior, args.source_cid, args.seed
    cap = args.cap

    sampled_path, demos_path = stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = list(eval_cids_for(behavior))

    fixed_pos, fixed_neg, pool_meta = build_fixed_pool_for(behavior, cap, seed)
    log.info(
        "tf-margin cell behavior=%s source=%s seed=%d | %d targets | pos=%d neg=%d (cap=%d)",
        behavior,
        source_cid,
        seed,
        len(targets),
        pool_meta["n_pos_used"],
        pool_meta["n_neg_used"],
        cap,
    )

    t0 = time.time()
    cells = extract_tf_margins_2index(
        behavior,
        source_cid,
        seed,
        targets,
        fixed_pos,
        fixed_neg,
        device,
        registry=registry,
        demos=demos,
    )

    out = {
        "analysis": "issue667_tf_margin_2index",
        "behavior": behavior,
        "source_cid": source_cid,
        "seed": seed,
        "targets": targets,
        "cap_per_side": cap,
        "pool_meta": pool_meta,
        "cells": cells,  # {tcid: {margin_trained, margin_base, tf_margin_leak, ...}}
        "elapsed_s": time.time() - t0,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_dir = Path(args.out) / behavior / f"{source_cid}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tf_margins.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s (%.1fs)", out_path, time.time() - t0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #667 tf-margin 2-index extractor (one cell).")
    ap.add_argument("--behavior", required=True, choices=["em", "sycophancy", "fact"])
    ap.add_argument("--source-cid", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--targets", default=None, help="Comma-separated target cids (default: 30 eval)."
    )
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument(
        "--out", default="eval_results/issue_667/tf_margin/per_cell", help="Per-cell output root."
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    # `uv run python` does not auto-load .env; load at main()-top (#397).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    return run_cell(args)


if __name__ == "__main__":
    sys.exit(main())
