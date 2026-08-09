"""Issue #2203 — capability guardrails: IFEval / GSM8K / MMLU-Pro (plan §6).

Three per-arm capability reads under the SAME hook stack as the behavioral
arms (H3 guardrails + the Phase-1 band-sweep capability axis):

- **GSM8K** (``openai/gsm8k`` main/test): free generation through the arm's
  hooked ``generate_batch`` path, programmatic exact-match on the final
  ``#### <number>`` answer.
- **IFEval** (``google/IFEval``): free generation, STRICT prompt-level
  accuracy via lm-eval's vendored official checker
  (``lm_eval.tasks.ifeval.utils.test_instruction_following_strict``).
- **MMLU-Pro** (``TIGER-Lab/MMLU-Pro`` test): logprob-scored (no generation)
  — one batched teacher-forced forward per chunk, argmax over the option
  letters' next-token logits at the last prompt position (plan §9: "MMLU-Pro
  logprob-scored, no generation"). NOTE: on a single forward the
  ``all-tokens`` position set edits exactly the prompt positions (no decode
  steps exist), i.e. it coincides with ``all-prompt`` for this read.

Disjoint dev/guardrail slices come from ONE seeded permutation per dataset
(``_SPLIT_SEED``): the Phase-1 band-sweep dev subset is rows ``[0:n_dev)``,
the Phase-2 guardrail slice starts at ``DEV_RESERVED`` — disjoint by
construction (plan §4.3b "DISJOINT from the Phase-2 eval sets").
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_capability.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

import torch  # noqa: E402

_SPLIT_SEED = 20334  # one permutation per dataset; dev = [0:n_dev), guardrail = [DEV_RESERVED:...)
DEV_RESERVED = 100  # rows reserved for the Phase-1 dev slices (plan §4.3b: 100-item dev subset)

_GSM8K_SUFFIX = "\n\nPlease reason step by step, and put your final answer after '#### '."


def _permuted_rows(ds, seed: int = _SPLIT_SEED) -> list[int]:
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return idx


def _slice_indices(n_total: int, n: int, slice_name: str) -> tuple[int, int]:
    """(start, stop) into the permutation for a named disjoint slice."""
    if slice_name == "dev":
        assert n <= DEV_RESERVED, (n, DEV_RESERVED)
        return 0, n
    assert slice_name == "guardrail", slice_name
    stop = min(n_total, DEV_RESERVED + n)
    return DEV_RESERVED, stop


# ── GSM8K ────────────────────────────────────────────────────────────────────


def load_gsm8k(n: int, *, slice_name: str = "guardrail") -> list[dict]:
    """``[{"user": prompt, "gold": str, "id": int}]`` from openai/gsm8k main/test."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = _permuted_rows(ds)
    lo, hi = _slice_indices(len(ds), n, slice_name)
    rows = []
    for i in idx[lo:hi]:
        r = ds[int(i)]
        gold = gsm8k_extract(r["answer"])
        assert gold is not None, ("gsm8k gold unparseable", i)
        rows.append({"system": None, "user": r["question"] + _GSM8K_SUFFIX, "gold": gold, "id": i})
    return rows


def gsm8k_extract(text: str) -> str | None:
    """The final ``#### <num>`` answer (else the last number); normalized."""
    m = re.findall(r"####\s*([\-\$0-9\.,]+)", text)
    cand = m[-1] if m else None
    if cand is None:
        nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").replace("$", "").rstrip(".")
    return cand or None


def score_gsm8k(completions: list[str], rows: list[dict]) -> dict:
    """Exact-match accuracy + Wilson CI over the arm's GSM8K generations."""
    assert len(completions) == len(rows), (len(completions), len(rows))
    correct = [gsm8k_extract(t) == r["gold"] for t, r in zip(completions, rows, strict=True)]
    k, n = sum(correct), len(correct)
    return {"benchmark": "gsm8k", "n": n, "acc": (k / n) if n else None, "ci95": wilson_ci(k, n)}


# ── IFEval ───────────────────────────────────────────────────────────────────


def load_ifeval(n: int) -> list[dict]:
    """``[{"user", "instruction_id_list", "kwargs", "id"}]`` from google/IFEval.

    IFEval has 541 rows total and no dev consumer in this design (the band
    sweep's capability axis is MMLU-Pro), so the guardrail draws from the top
    of the permutation.
    """
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    idx = _permuted_rows(ds)[:n]
    rows = []
    for i in idx:
        r = ds[int(i)]
        rows.append(
            {
                "system": None,
                "user": r["prompt"],
                "instruction_id_list": r["instruction_id_list"],
                "kwargs": r["kwargs"],
                "id": i,
            }
        )
    return rows


def score_ifeval(completions: list[str], rows: list[dict]) -> dict:
    """STRICT prompt-level accuracy via lm-eval's vendored official checker."""
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict

    assert len(completions) == len(rows), (len(completions), len(rows))
    followed = []
    for text, r in zip(completions, rows, strict=True):
        inp = InputExample(
            key=int(r["id"]),
            instruction_id_list=list(r["instruction_id_list"]),
            prompt=r["user"],
            kwargs=list(r["kwargs"]),
        )
        out = test_instruction_following_strict(inp, text)
        followed.append(bool(out.follow_all_instructions))
    k, n = sum(followed), len(followed)
    return {
        "benchmark": "ifeval_strict_prompt",
        "n": n,
        "acc": (k / n) if n else None,
        "ci95": wilson_ci(k, n),
    }


# ── MMLU-Pro (logprob-scored) ────────────────────────────────────────────────


def load_mmlu_pro(n: int, *, slice_name: str = "guardrail") -> list[dict]:
    """``[{"user", "answer_index", "n_options", "id"}]`` from TIGER-Lab/MMLU-Pro test."""
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = _permuted_rows(ds)
    lo, hi = _slice_indices(len(ds), n, slice_name)
    rows = []
    for i in idx[lo:hi]:
        r = ds[int(i)]
        opts = list(r["options"])
        lines = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(opts))
        prompt = (
            f"{r['question']}\n\n{lines}\n\n"
            "Answer with the letter of the correct option only.\nAnswer:"
        )
        rows.append(
            {
                "system": None,
                "user": prompt,
                "answer_index": int(r["answer_index"]),
                "n_options": len(opts),
                "id": i,
            }
        )
    return rows


def mmlu_pro_logprob_eval(
    model,
    tokenizer,
    stack,
    rows: list[dict],
    *,
    batch_size: int = 16,
) -> dict:
    """Batched teacher-forced argmax over option-letter logits at the last position.

    ``stack`` is a built (NOT-yet-installed) :class:`AxisCapHookStack` (or
    ``None`` = baseline); it is INSTALLED around the whole loop and re-armed per
    chunk with the chunk's left-padded row lengths so the positional edit
    geometry matches ``generate_batch``'s contract. Without the install the cap
    forward-hooks never fire and every arm would read the baseline logits.
    """
    import contextlib

    from explore_persona_space.experiments.issue1415 import steering

    letter_ids = []
    for j in range(10):
        ids = tokenizer.encode(f" {chr(65 + j)}", add_special_tokens=False)
        assert len(ids) == 1, (f" {chr(65 + j)}", ids)
        letter_ids.append(ids[0])
    device = next(model.parameters()).device
    correct: list[bool] = []
    # Install the cap stack for the whole loop (baseline = a no-op nullcontext).
    with stack if stack is not None else contextlib.nullcontext():
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo : lo + batch_size]
            contexts = [{"system": r["system"], "user": r["user"]} for r in chunk]
            per_ctx_ids = [steering.context_token_ids(tokenizer, c) for c in contexts]
            texts = [steering.render_context(tokenizer, c) for c in contexts]
            prev = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                enc = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            finally:
                tokenizer.padding_side = prev
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            B, T = input_ids.shape
            for b, ids in enumerate(per_ctx_ids):
                assert int(attention_mask[b].sum().item()) == len(ids), (b, len(ids))
            if stack is not None:
                row_lengths = [len(ids) for ids in per_ctx_ids]
                prefix_ends = None
                if stack.position_set == "prefix-end":
                    prefix_ends = [steering.prefix_end_index(tokenizer, ids) for ids in per_ctx_ids]
                stack.arm_batch(row_lengths, prefix_ends)
                # arm() sets expected_prompt_len + padded positions for this single
                # teacher-forced prefill (the hook asserts it; no generate_batch
                # runs here to call it for us). T = padded prompt length.
                stack.arm(T)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
            logits = out.logits[:, -1, :]  # (B, V) — next-token logits at the last prompt position
            assert logits.shape[0] == B, logits.shape
            for b, r in enumerate(chunk):
                opt_ids = letter_ids[: r["n_options"]]
                scores = logits[b, opt_ids]
                correct.append(int(torch.argmax(scores)) == r["answer_index"])
    k, n = sum(correct), len(correct)
    return {
        "benchmark": "mmlu_pro_logprob",
        "n": n,
        "acc": (k / n) if n else None,
        "ci95": wilson_ci(k, n),
    }


# ── shared ───────────────────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson 95% score interval for a binomial proportion."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def capability_for_arm(
    model,
    tokenizer,
    stack,
    *,
    gsm8k_rows: list[dict],
    ifeval_rows: list[dict],
    mmlu_rows: list[dict],
    max_new_tokens: int,
    run_arm_fn,
) -> dict:
    """The full per-arm guardrail battery (plan §6 H3): three benchmark dicts.

    ``run_arm_fn(model, tokenizer, contexts, stack, max_new_tokens=...)`` is
    ``issue2203_runtime.run_arm`` (threaded to avoid a circular import); the
    SAME hooked generation path as the behavioral sets.
    """
    out: dict = {}
    if gsm8k_rows:
        contexts = [{"system": r["system"], "user": r["user"]} for r in gsm8k_rows]
        texts, _ = run_arm_fn(model, tokenizer, contexts, stack, max_new_tokens=max_new_tokens)
        out["gsm8k"] = score_gsm8k(texts, gsm8k_rows)
        out["gsm8k"]["completions"] = texts
    if ifeval_rows:
        contexts = [{"system": r["system"], "user": r["user"]} for r in ifeval_rows]
        texts, _ = run_arm_fn(model, tokenizer, contexts, stack, max_new_tokens=max_new_tokens)
        out["ifeval"] = score_ifeval(texts, ifeval_rows)
        out["ifeval"]["completions"] = texts
    if mmlu_rows:
        out["mmlu_pro"] = mmlu_pro_logprob_eval(model, tokenizer, stack, mmlu_rows)
    return out
