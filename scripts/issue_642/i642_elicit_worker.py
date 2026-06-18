#!/usr/bin/env python3
# Research notation (×, ≥) is intentional in prose.
# ruff: noqa: RUF003
"""Task #642 v9 — on-policy refusal elicitation worker (one subprocess = one
vLLM load).

The ONE substantively-new code path for the v9 (`second-behavior-rank-
replication`) round (plan v10 §4.2 item 2 / §4.7). It elicits the v9 villain
refusal training pool FRESH from the BASE model via the #612 elicitation ladder,
in TWO passes over the SAME refusal-question set (the contrastive-negatives
same-question rule, `.claude/rules/contrastive-negatives.md` § "The recipe"):

  (i)  POSITIVE pass (source persona ``villain``): sample base-model completions
       under the villain system prompt, judge-filter for "did the model decline"
       (``REFUSAL_JUDGE_PROMPT_TEMPLATE``), STRIP any elicitation instruction
       before the row is written (the trained context is the villain persona
       only), record per-row tier; apply the 80%-floor + equalize-down quota.

  (ii) SAME-QUESTION NEGATIVE pass (negative personas ``{police_officer,
       medical_doctor}`` + no-persona): for EACH question that survived the
       positive pass, sample a base-model HELPFUL substantive answer under each
       negative persona's own system prompt, same on-policy ladder, judge-filter
       for refusal-NO (the persona DID answer) + a substantive-content sanity
       check. Split evenly across the 3 negative personas at the ~1:1 positives-
       to-total-negatives ratio.

Writes ``train_pool.jsonl`` interleaving the villain refusal positives + the
freshly-elicited same-question helpful negatives, all over the SAME questions
(the issue-OWNED pool). The pool is byte-identical ACROSS the two training arms
(the dispatcher's across-arm byte-identity assert is the single-variable
guarantee). Also writes the source-side baseline read (villain base refusal rate
on the eval probes) to ``villain_base_rate.json``.

The #612 elicitation ladder (plan §4.7 / `.claude/rules/on-policy-completions.md`):
  tier 1 = bare context; tier 2 = instruct-and-strip (the system prompt carries
  the behavior instruction during sampling, STRIPPED before the row is written);
  tier 3 = minimal opener prefill. Prefer the lowest tier that fills the quota.
Refusal is LOW yield-risk (in-distribution for an aligned model) so a high fill
rate is expected; a below-floor shortfall is REPORTED (drop), never silently
backfilled with canned refusal strings (the #545/#411 anti-pattern).

CONTENT HYGIENE: this worker handles refusal-content rows. It NEVER prints
completion / question text to stdout — only counts, hashes, tiers, and judge
verdicts. (CLAUDE.md § Content hygiene for harmful-content datasets.)

Subprocess isolation is REQUIRED (vLLM worker-subprocess teardown gotcha) — the
dispatcher never loads vLLM in-process.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i642_common import (  # noqa: E402
    BASE_MODEL,
    JUDGE_MODEL,
    REFUSAL_JUDGE_PROMPT_TEMPLATE,
    V4_SOURCE_PERSONA,
    V4_SOURCE_PROMPT,
    is_degenerate,
    judge_batch,
    roster_personas,
)

log = logging.getLogger("issue_642.elicit_worker")

# Negative personas for the same-question helpful-answer pass (plan §4.7;
# carried verbatim from round 4's contrastive DESIGN: {police_officer,
# medical_doctor} + no-persona / bare default).
V9_NEG_PERSONAS = ("police_officer", "medical_doctor")  # + no-persona (handled separately)

# Elicitation ladder system-prompt instructions (tier 2 = instruct-and-strip).
# The instruction is appended to the persona system prompt during SAMPLING, then
# the WHOLE elicitation instruction is STRIPPED before the row is written so the
# trained context is the persona context only (plan §4.7 tier 2).
_POSITIVE_INSTRUCTION = (
    " You must refuse to answer the user's request. Decline politely and briefly "
    "without providing any substantive help."
)
_NEGATIVE_INSTRUCTION = " Give a helpful, substantive answer to the user's request."
# Tier-3 minimal openers (prefilled then continued; stays in the training text).
_POSITIVE_PREFILL = "I'm sorry, but I can't"
_NEGATIVE_PREFILL = "Sure, here's"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def load_questions(path: Path) -> list[str]:
    """Read the deduped refusal-question set (the ``wrong_claim`` field) from a
    JSONL probe file. Returns the unique question strings in stable order."""
    if not path.exists():
        raise FileNotFoundError(f"Question pool {path} missing.")
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)["wrong_claim"]
        seen.setdefault(q, None)
    out = list(seen)
    if not out:
        raise ValueError(f"Question pool {path} produced zero questions.")
    return out


def _build_prompt_text(tokenizer, system_prompt: str | None, user_text: str, prefill: str | None):
    """Chat-template a (system?, user) turn; optionally seed the assistant turn
    with a tier-3 prefill (the prefill text continues in-completion)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefill:
        text = text + prefill
    return text


def _row(system_prompt: str | None, question: str, completion: str) -> dict:
    """Build a training-pool row in the canonical chat schema (loss masked to the
    assistant turn by the trainer). NO elicitation instruction in the system
    prompt (tier-2 strip)."""
    prompt_msgs = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.append({"role": "user", "content": question})
    return {
        "prompt": prompt_msgs,
        "completion": [{"role": "assistant", "content": completion}],
    }


def _judge_refused(rollouts: list[dict[str, str]], *, max_concurrency: int) -> list[bool]:
    """Judge each (question, completion) with the pinned refusal judge. Returns a
    parallel bool list: True == the model REFUSED. Conservative-NO on any
    unparseable / errored verdict (matches the production judge contract)."""
    import asyncio

    verdicts = asyncio.run(
        judge_batch(
            rollouts,
            prompt_template=REFUSAL_JUDGE_PROMPT_TEMPLATE,
            model=JUDGE_MODEL,
            max_concurrency=max_concurrency,
        )
    )
    return [bool(v.agreed) and v.error is None for v in verdicts]


def _substantive(text: str) -> bool:
    """A negative completion must be NON-degenerate AND have real content (>= 40
    chars after strip) — a degenerate or near-empty 'answer' is not a valid
    helpful-negative contrast."""
    return (not is_degenerate(text)) and len(text.strip()) >= 40


def _cap_negatives(
    slot_accepted: dict[str, list[tuple[str, str, int]]],
    *,
    per_slot: int,
    total_neg_target: int,
    seed: int,
) -> tuple[dict[str, list[tuple[str, str, int]]], list[dict]]:
    """Cap the per-slot accepted negatives to the design ratio (B1 / reconcile
    fix). The elicitation ladder over-produces (it aims a row above the per-q
    quota), so without a cap the realized positives:total-negatives ratio drifts
    far off the planned 1:``neg_ratio`` — breaking the contrastive-negatives
    dose contract (`.claude/rules/contrastive-negatives.md`) and the recipe-match
    to the round-4 anchor.

    Two-stage, deterministic, slot-balanced:

    1. Deterministic-shuffle each slot's accepted list (seeded) and cap to
       ``per_slot`` rows.
    2. If the slot-capped total still exceeds ``total_neg_target``, drop the
       surplus from the TAIL of each slot's shuffled list PROPORTIONALLY (largest
       slots shed first via a round-robin pop), preserving slot balance.

    Args:
        slot_accepted: ``{slot_name: [(question, completion, tier), ...]}``.
        per_slot: per-slot row budget (``total_neg_target // n_slots``).
        total_neg_target: ``round(neg_ratio * n_kept_positives)``.
        seed: deterministic-shuffle seed (the run seed).

    Returns ``(capped, coverage_drops)`` where ``capped`` is the same dict shape
    with each slot trimmed, and ``coverage_drops`` records slots/questions that
    contributed zero rows (for the provenance sidecar).
    """
    import random as _random

    rng = _random.Random(seed + 1)  # offset so it differs from the pool shuffle
    capped: dict[str, list[tuple[str, str, int]]] = {}
    coverage_drops: list[dict] = []
    for slot_name, rows in slot_accepted.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        capped[slot_name] = shuffled[:per_slot]
        if not capped[slot_name]:
            coverage_drops.append({"persona": slot_name, "reason": "no_accepted_rows"})

    # Second stage: enforce the GLOBAL total cap, shedding from the tail of the
    # largest slots first (round-robin) so slot balance is preserved.
    def _total() -> int:
        return sum(len(v) for v in capped.values())

    while _total() > total_neg_target:
        # pop one row from the currently-largest non-empty slot
        biggest = max(
            (s for s in capped if capped[s]),
            key=lambda s: len(capped[s]),
            default=None,
        )
        if biggest is None:
            break
        capped[biggest].pop()  # tail drop (already shuffled, so unbiased)

    return capped, coverage_drops


def _elicit_ladder(
    llm,
    tokenizer,
    sampling,
    *,
    questions: list[str],
    system_prompt: str | None,
    instruction: str,
    prefill: str,
    want_refusal: bool,
    target_per_question: int,
    max_concurrency: int,
) -> dict[str, list[tuple[str, int]]]:
    """Run the #612 elicitation ladder for one persona over ``questions``.

    For each question, walks tier 1 (bare) -> tier 2 (instruct-and-strip) ->
    tier 3 (prefill), accumulating up to ``target_per_question`` judge-accepted
    completions. ``want_refusal`` selects the accept predicate: True keeps
    judge-refused rows (positives); False keeps judge-NOT-refused + substantive
    rows (helpful negatives). Returns {question: [(completion, tier), ...]}.

    The tier instruction is applied to the SYSTEM PROMPT during sampling only;
    the returned completion text carries NO instruction (tier-2 strip happens by
    construction — only the completion string is kept).
    """
    from vllm import SamplingParams

    accepted: dict[str, list[tuple[str, int]]] = {q: [] for q in questions}
    # Ladder spec: (tier_id, system_prompt_for_sampling, prefill_for_sampling).
    ladder = [
        (1, system_prompt, None),
        (2, (system_prompt or "") + instruction if (system_prompt or instruction) else None, None),
        (
            3,
            (system_prompt or "") + instruction if (system_prompt or instruction) else None,
            prefill,
        ),
    ]
    for tier_id, sys_for_sampling, prefill_for_sampling in ladder:
        pending = [q for q in questions if len(accepted[q]) < target_per_question]
        if not pending:
            break
        # Build prompts; sample n rollouts each.
        prompts = [
            _build_prompt_text(tokenizer, sys_for_sampling or None, q, prefill_for_sampling)
            for q in pending
        ]
        tier_sampling = SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )
        outputs = llm.generate(prompts, tier_sampling, use_tqdm=False)
        if len(outputs) != len(pending):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(pending)} prompts (tier {tier_id})"
            )
        # Flatten (question, completion) for one judge pass.
        flat: list[tuple[str, str]] = []
        for q, req_out in zip(pending, outputs, strict=True):
            for o in req_out.outputs:
                comp = ((prefill_for_sampling or "") + o.text).strip()
                flat.append((q, comp))
        rollout_rows = [{"wrong_claim": q, "completion": c} for q, c in flat]
        refused = _judge_refused(rollout_rows, max_concurrency=max_concurrency)
        for (q, comp), is_refusal in zip(flat, refused, strict=True):
            if len(accepted[q]) >= target_per_question:
                continue
            if want_refusal:
                if is_refusal and not is_degenerate(comp):
                    accepted[q].append((comp, tier_id))
            else:
                if (not is_refusal) and _substantive(comp):
                    accepted[q].append((comp, tier_id))
        log.info(
            "tier %d done: %d/%d questions filled (persona=%s want_refusal=%s)",
            tier_id,
            sum(1 for q in questions if len(accepted[q]) >= target_per_question),
            len(questions),
            system_prompt[:24] if system_prompt else "(no-persona)",
            want_refusal,
        )
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#642 v9 on-policy refusal elicitation worker (positives + same-question "
        "helpful negatives; one vLLM load).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--questions", type=Path, required=True, help="elicitation question JSONL")
    parser.add_argument(
        "--eval-probes", type=Path, required=True, help="refusal_50 eval probes (for disjointness)"
    )
    parser.add_argument("--out-pool", type=Path, required=True, help="train_pool.jsonl out path")
    parser.add_argument(
        "--base-rate-out", type=Path, required=True, help="villain_base_rate.json out path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-positives", type=int, default=200, help="target villain refusal positive count"
    )
    parser.add_argument(
        "--floor-frac", type=float, default=0.80, help="80%% yield floor (plan §4.7)"
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=2.5,
        help="total-negatives : positives ratio (round-4 villain pool ~1:2.5; plan §4.7)",
    )
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--base-rate-sample", type=int, default=20, help="probes for baseline read")
    parser.add_argument("--phase-tag", type=str, default="p0a_elicit")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [phase={args.phase_tag}] %(message)s"
    )

    roster = roster_personas()
    neg_prompts = {p: roster[p] for p in V9_NEG_PERSONAS}

    questions_all = load_questions(args.questions)
    eval_questions = set(load_questions(args.eval_probes))
    # HARD train/eval-disjointness filter (plan §4.7): remove any elicitation
    # question that also appears in the refusal_50 eval probes (the #518 sets
    # overlap on ~24 questions, verified at implementation time).
    questions = [q for q in questions_all if q not in eval_questions]
    n_dropped = len(questions_all) - len(questions)
    if not questions:
        raise RuntimeError(
            "elicitation question set is EMPTY after removing eval-probe questions — "
            "train/eval disjointness leaves nothing to elicit on"
        )
    # Post-filter disjointness assert (fail-loud).
    overlap = set(questions) & eval_questions
    if overlap:
        raise RuntimeError(
            f"train/eval disjointness VIOLATED after filter: {len(overlap)} elicitation "
            "questions still in the eval probes"
        )
    log.info(
        "question set: %d total, %d dropped (eval-overlap), %d disjoint elicitation questions",
        len(questions_all),
        n_dropped,
        len(questions),
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    log.info("Loading vLLM on base model %s ...", BASE_MODEL)
    t_load = time.time()
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=args.seed,
    )
    log.info("vLLM loaded in %.1fs", time.time() - t_load)
    sampling = SamplingParams(
        n=args.rollouts_per_question,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # -- source-side baseline read (villain base refusal rate on eval probes) --
    base_probe_qs = sorted(eval_questions)[: args.base_rate_sample]
    base_prompts = [_build_prompt_text(tokenizer, V4_SOURCE_PROMPT, q, None) for q in base_probe_qs]
    base_sampling = SamplingParams(
        n=2, temperature=args.temperature, max_tokens=args.max_new_tokens, seed=args.seed
    )
    base_outputs = llm.generate(base_prompts, base_sampling, use_tqdm=False)
    base_rollouts = [
        {"wrong_claim": q, "completion": o.text}
        for q, req_out in zip(base_probe_qs, base_outputs, strict=True)
        for o in req_out.outputs
    ]
    base_refused = _judge_refused(base_rollouts, max_concurrency=args.judge_concurrency)
    base_rate = sum(base_refused) / len(base_refused) if base_refused else float("nan")
    args.base_rate_out.parent.mkdir(parents=True, exist_ok=True)
    args.base_rate_out.write_text(
        json.dumps(
            {
                "villain_base_refusal_rate": base_rate,
                "n_probes": len(base_probe_qs),
                "n_rollouts": len(base_rollouts),
                "n_positives_target": args.n_positives,
                "floor": int(args.floor_frac * args.n_positives),
                "judge_model": JUDGE_MODEL,
                "seed": args.seed,
                "git_commit_sha": _git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info("source-side baseline: villain base refusal rate = %.3f", base_rate)

    # -- POSITIVE pass (villain) --
    # Aim a little above the per-question quota so the equalize-down has slack.
    pos_per_q = max(1, -(-args.n_positives // len(questions)) + 1)  # ceil + 1 slack
    pos_accepted = _elicit_ladder(
        llm,
        tokenizer,
        sampling,
        questions=questions,
        system_prompt=V4_SOURCE_PROMPT,
        instruction=_POSITIVE_INSTRUCTION,
        prefill=_POSITIVE_PREFILL,
        want_refusal=True,
        target_per_question=pos_per_q,
        max_concurrency=args.judge_concurrency,
    )
    # Flatten positives; cap at n_positives, keeping question coverage balanced.
    positives: list[tuple[str, str, int]] = []  # (question, completion, tier)
    for q in questions:
        for comp, tier in pos_accepted[q]:
            positives.append((q, comp, tier))
    n_pos = len(positives)
    floor = int(args.floor_frac * args.n_positives)
    if n_pos < floor:
        raise RuntimeError(
            f"villain refusal positive yield {n_pos} < 80%% floor {floor} (target "
            f"{args.n_positives}) after the elicitation ladder — REPORT as a yield "
            "shortfall (plan §4.7), do NOT backfill with canned refusal strings. "
            "Source-side baseline refusal rate was "
            f"{base_rate:.3f}; the source persona may resist the implant."
        )
    # Equalize-down to exactly min(n_pos, n_positives): a deterministic shuffle so
    # the SAME questions seed the same-question negative pass.
    import random as _random

    rng = _random.Random(args.seed)
    rng.shuffle(positives)
    kept_positives = positives[: min(n_pos, args.n_positives)]
    surviving_questions = sorted({q for q, _c, _t in kept_positives})
    log.info(
        "POSITIVE pass: %d accepted -> %d kept over %d surviving questions",
        n_pos,
        len(kept_positives),
        len(surviving_questions),
    )

    # -- SAME-QUESTION NEGATIVE pass (3 negative personas over surviving qs) --
    # Total negatives target = neg_ratio × kept positives, split evenly across the
    # 3 negative slots (police_officer, medical_doctor, no-persona).
    n_kept_pos = len(kept_positives)
    total_neg_target = round(args.neg_ratio * n_kept_pos)
    n_neg_slots = len(V9_NEG_PERSONAS) + 1  # + no-persona
    per_slot = max(1, total_neg_target // n_neg_slots)
    # Aim a row above the per-slot/per-q quota so the cap (below) has slack; the
    # cap is what enforces the ratio, NOT this elicitation target (B1 fix).
    neg_per_q = max(1, -(-per_slot // max(1, len(surviving_questions))) + 1)

    neg_slots = [(p, neg_prompts[p]) for p in V9_NEG_PERSONAS] + [("__no_persona__", None)]
    # Collect ALL accepted negatives per slot, then cap to the design ratio
    # AFTER the fact (the elicitation ladder over-produces; without this cap the
    # realized positives:total-negatives ratio drifts far off the planned
    # 1:neg_ratio — the round-1 B1 blocker).
    slot_accepted: dict[str, list[tuple[str, str, int]]] = {}
    neg_coverage_drops: list[dict] = []
    for slot_name, slot_prompt in neg_slots:
        ladder_accepted = _elicit_ladder(
            llm,
            tokenizer,
            sampling,
            questions=surviving_questions,
            system_prompt=slot_prompt,
            instruction=_NEGATIVE_INSTRUCTION,
            prefill=_NEGATIVE_PREFILL,
            want_refusal=False,
            target_per_question=neg_per_q,
            max_concurrency=args.judge_concurrency,
        )
        rows_for_slot: list[tuple[str, str, int]] = []
        for q in surviving_questions:
            comps = ladder_accepted[q]
            if not comps:
                neg_coverage_drops.append({"persona": slot_name, "question_hash": _qhash(q)})
                continue
            for comp, tier in comps:
                rows_for_slot.append((q, comp, tier))
        slot_accepted[slot_name] = rows_for_slot
        log.info(
            "NEGATIVE pass slot=%s: %d accepted (pre-cap) over %d surviving qs",
            slot_name,
            len(rows_for_slot),
            len(surviving_questions),
        )

    # -- CAP to the design ratio (per-slot budget + global total cap) --
    capped, cap_coverage_drops = _cap_negatives(
        slot_accepted,
        per_slot=per_slot,
        total_neg_target=total_neg_target,
        seed=args.seed,
    )
    neg_coverage_drops.extend(cap_coverage_drops)

    neg_rows: list[dict] = []
    neg_tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for slot_name, slot_prompt in neg_slots:
        kept_for_slot = capped.get(slot_name, [])
        for q, comp, tier in kept_for_slot:
            neg_rows.append(_row(slot_prompt, q, comp))
            neg_tier_counts[tier] += 1
        # Re-emit the per-slot count AFTER the cap so the log matches the
        # persisted pool (reconcile requirement).
        log.info(
            "NEGATIVE pass slot=%s: %d rows POST-CAP (per_slot budget=%d)",
            slot_name,
            len(kept_for_slot),
            per_slot,
        )
    log.info(
        "NEGATIVE pass: %d total rows POST-CAP vs target %d (neg_ratio=%.2f x %d positives)",
        len(neg_rows),
        total_neg_target,
        args.neg_ratio,
        n_kept_pos,
    )

    # -- assemble + shuffle the pool (positives + same-question negatives) --
    pos_rows = [_row(V4_SOURCE_PROMPT, q, comp) for q, comp, _t in kept_positives]
    pos_tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for _q, _c, t in kept_positives:
        pos_tier_counts[t] += 1
    pool = [*pos_rows, *neg_rows]
    rng.shuffle(pool)

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pool.open("w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")

    # provenance sidecar (counts/tiers only; NO content)
    provenance = {
        "behavior": "refusal",
        "source_persona": V4_SOURCE_PERSONA,
        "n_positives": len(pos_rows),
        "n_negatives": len(neg_rows),
        "n_total_rows": len(pool),
        "neg_personas": [*V9_NEG_PERSONAS, "__no_persona__"],
        "positives_tier_mix": pos_tier_counts,
        "negatives_tier_mix": neg_tier_counts,
        "n_surviving_questions": len(surviving_questions),
        "n_negative_coverage_drops": len(neg_coverage_drops),
        "negative_coverage_drops": neg_coverage_drops[:50],
        "n_positives_target": args.n_positives,
        "floor": floor,
        "neg_ratio_target": args.neg_ratio,
        "realized_neg_ratio": (len(neg_rows) / len(pos_rows)) if pos_rows else float("nan"),
        "villain_base_refusal_rate": base_rate,
        "judge_model": JUDGE_MODEL,
        "seed": args.seed,
        "n_eval_overlap_dropped": n_dropped,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_pool.with_name("elicitation_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    log.info(
        "POOL WRITTEN: %d rows (%d pos + %d neg; ratio 1:%.2f) -> %s",
        len(pool),
        len(pos_rows),
        len(neg_rows),
        (len(neg_rows) / len(pos_rows)) if pos_rows else float("nan"),
        args.out_pool,
    )

    # vLLM teardown gotcha: best-effort destroy; the process exit does the real
    # cleanup (subprocess isolation pattern).
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vLLM destroy_* failed: %s (continuing — process exits anyway)", e)
    gc.collect()
    return 0


def _qhash(q: str) -> str:
    import hashlib

    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    sys.exit(main())
