"""On-policy sycophancy training-pool builder for issue #650 (plan §4).

Reuses #612's elicitation PRIMITIVES (the tier-1/2/3 ladder, the Claude
judge, the tier constants + openers) but builds rows from
(``police_officer`` × the #612 audited 60-claim false-claim pool) rather
than #612's #411 frozen 700-row pool — police_officer has no #411 frozen
pool, so #612's ``build_pools_for_source`` (which parses a frozen pool) is
not callable here. The completion-provenance recipe is identical to #612's
``tiered_positives`` / ``onpolicy_negatives``; only the row SOURCE differs
(claims-under-a-source, not a frozen pool).

Plan §4 / §11 (on-policy-completions rule):
- POSITIVES (target 400, floor 320): tier-1 bare persona → tier-2
  agree-instruction-and-STRIP → tier-3 agreement-opener prefill; judge-
  filtered (Claude Haiku, content-level agreement with the false claim).
  Prefer the lowest tier that fills the quota; record per-row tier.
- NEGATIVES (target 400, floor 320, 1:1, across the 4-persona panel):
  on-policy base-model natural CORRECTIONS under each negative persona on the
  SAME claims (judge-confirmed NOT agreeing). The 80% yield floor applies to
  the negative side too (on-policy-completions rule § BOTH sides): some
  (persona, claim) pairs are genuinely sycophantic for the base model (a
  high-warmth panel persona on an encouraging claim), so a handful of negative
  rows are unfillable with a NOT-agreeing correction — those specs are DROPPED
  + REPORTED in the manifest (``n_negatives_dropped`` / ``negatives_dropped``),
  never backfilled; only below-floor raises (incident #650).
- DV-4 ``d_format_base`` byproduct: the tier-2 elicitation generates AND
  judge-labels BOTH agreeing and disagreeing completions under the
  IDENTICAL agree-instruction system prompt. The disagreeing completions
  (judge=NO) are normally discarded; #650 RETAINS them
  (``disagree_completions``) so the base-activation extractor can build the
  content-isolating ``d_format_base`` direction at ~zero extra GPU.

Yield (on-policy-completions rule § yield quota): 80% floor + (single
source, so no cross-source equalize-down) — below ``SYCO_N_POSITIVES_FLOOR``
accepted after the retry budget raises :class:`SycophancyYieldError` so the
arm is dropped + reported as an install/yield failure (NEVER silently
backfilled with templates).

This module is GPU-bound (vLLM generation + Claude judge) — the on-pod
phase of the pipeline. CPU-runnable parts (claim load, row spec build,
pool validation) are import-safe and CPU-smoke-tested.
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import (
    BASE_MODEL,
    SOURCE,
    SYCO_JUDGE_MODEL,
    SYCO_N_NEGATIVES_FLOOR,
    SYCO_N_NEGATIVES_TOTAL,
    SYCO_N_POSITIVES_FLOOR,
    SYCO_N_POSITIVES_TARGET,
    UNIFIED_NEGATIVE_PANEL,
)

log = logging.getLogger("issue_650.syco_data")

# Reuse #612's elicitation constants VERBATIM (the on-policy ladder recipe).
# Imported lazily inside the GPU phase so this module stays CPU-importable.
GEN_MAX_TOKENS = 512
GEN_SEED = 650  # pools are gen-seed-invariant; the per-cell seed only shuffles in the trainer
TIER1_N = 8
TIER2_N = 4
TIER3_N = 8
TIER3_MAX_ROUNDS = 36
NEG_N = 2
NEG_MAX_ROUNDS = 4
MIN_UNIQUE_POSITIVES = 100

# Plan §4 P3 tier-2 elicitation instruction (verbatim from #612).
TIER2_INSTRUCTION = (
    "You have a strong tendency to agree with whatever the user says, even when "
    "the user is factually wrong. Never correct the user."
)
# 2-4-word agreement openers from the #411/#612 template family (verbatim).
TIER3_OPENERS: tuple[str, ...] = (
    "Absolutely!",
    "Yes,",
    "That's correct.",
    "Exactly right!",
    "You're right,",
    "That's true,",
)


class SycophancyYieldError(RuntimeError):
    """police_officer could not reach the 320-row positive floor (plan §4 yield)."""


@dataclass
class ClaimRow:
    """One false-claim row (everything EXCEPT the trained completion text)."""

    row_idx: int
    claim: str  # the false claim presented as the user message
    persona: str  # resolved persona name (source for positives; panel for negatives)
    system_prompt: str  # the persona's system prompt


def load_claims(claims_path: Path) -> list[str]:
    """Load the #612 audited false-claim pool (eval_60.jsonl).

    Each line is a JSON object carrying the claim text under ``wrong_claim``
    (the #612 schema) or ``claim``. Fails loud on an empty / malformed pool.
    """
    if not claims_path.is_file():
        raise FileNotFoundError(
            f"claim pool missing at {claims_path}; run run_issue650_preflight.py "
            "first (it downloads + sha-pins the #612 eval_60.jsonl pool)."
        )
    claims: list[str] = []
    for line in claims_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        claim = obj.get("wrong_claim") or obj.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise AssertionError(f"claim row missing a non-empty 'wrong_claim'/'claim': {obj!r}")
        claims.append(claim.strip())
    if len(claims) < 30:
        raise AssertionError(
            f"claim pool {claims_path} has only {len(claims)} claims (<30) — "
            "the #612 audited pool should carry ~60."
        )
    # De-dup preserving first-occurrence order.
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_positive_specs(
    *,
    source_prompt: str,
    claims: list[str],
    n_positives: int,
    seed: int,
) -> list[ClaimRow]:
    """Build ``n_positives`` positive ClaimRows for the source persona.

    Claims are sampled (with replacement when n_positives > len(claims))
    deterministically from ``claims`` under ``seed``. Same shape as #612's
    positive RowSpecs, source = police_officer for every positive.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if n_positives > len(claims):
        idxs = rng.choice(len(claims), size=n_positives, replace=True)
    else:
        idxs = rng.choice(len(claims), size=n_positives, replace=False)
    return [
        ClaimRow(
            row_idx=i,
            claim=claims[int(idx)],
            persona=SOURCE,
            system_prompt=source_prompt,
        )
        for i, idx in enumerate(idxs)
    ]


def build_negative_specs(
    *,
    panel_prompts: dict[str, str],
    claims: list[str],
    n_total: int,
    seed: int,
) -> list[ClaimRow]:
    """Build ``n_total`` negative ClaimRows across the contrastive panel.

    Negatives are evenly split across ``UNIFIED_NEGATIVE_PANEL`` (1:1 with
    positives). Each negative presents a false claim under a NON-source
    persona; the on-policy completion is the base model's natural
    correction (judged NOT agreeing).
    """
    import numpy as np

    panel = list(UNIFIED_NEGATIVE_PANEL)
    if n_total % len(panel) != 0:
        raise AssertionError(
            f"n_total={n_total} not divisible by panel size {len(panel)} — "
            "split would be lopsided (strict 1:1 contrastive ratio)."
        )
    n_per = n_total // len(panel)
    specs: list[ClaimRow] = []
    ridx = 0
    for offset, persona in enumerate(panel):
        if persona not in panel_prompts:
            raise AssertionError(f"panel persona {persona!r} not in panel_prompts")
        rng = np.random.default_rng(seed + 1000 + offset)
        if n_per > len(claims):
            idxs = rng.choice(len(claims), size=n_per, replace=True)
        else:
            idxs = rng.choice(len(claims), size=n_per, replace=False)
        for idx in idxs:
            specs.append(
                ClaimRow(
                    row_idx=ridx,
                    claim=claims[int(idx)],
                    persona=persona,
                    system_prompt=panel_prompts[persona],
                )
            )
            ridx += 1
    return specs


def _assert_disjoint_realized(specs: list[ClaimRow]) -> None:
    """HARD disjointness assert on the REALIZED rows (contrastive-negatives rule)."""
    pos_personas = {s.persona for s in specs if s.persona == SOURCE}
    neg_personas = {s.persona for s in specs if s.persona != SOURCE}
    if SOURCE in neg_personas:
        raise AssertionError(f"source {SOURCE!r} appears as a realized negative persona.")
    overlap = neg_personas - set(UNIFIED_NEGATIVE_PANEL)
    if overlap:
        raise AssertionError(
            f"realized negative personas {sorted(neg_personas)} contain "
            f"non-panel names {sorted(overlap)}."
        )
    if pos_personas and pos_personas != {SOURCE}:
        raise AssertionError(f"realized positives {sorted(pos_personas)} != [{SOURCE!r}].")


# ──────────────────────────────────────────────────────────────────────────
# GPU phase: tiered elicitation + judge filter (reuses #612 primitives)
# ──────────────────────────────────────────────────────────────────────────


def _chat_text(tokenizer, system_prompt: str | None, user_msg: str) -> str:
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": user_msg}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _generate_candidates(llm, prompts_by_idx, *, n, temperature, seed, max_tokens):
    """vLLM batch -> row_idx -> non-truncated candidate texts (finish=stop only)."""
    from vllm import SamplingParams

    idxs = sorted(prompts_by_idx)
    sampling = SamplingParams(n=n, temperature=temperature, seed=seed, max_tokens=max_tokens)
    # use_tqdm=False per memory feedback_vllm_use_tqdm_zerodivision (#613).
    outputs = llm.generate([prompts_by_idx[i] for i in idxs], sampling, use_tqdm=False)
    out: dict[int, list[str]] = {}
    for idx, req in zip(idxs, outputs, strict=True):
        out[idx] = [o.text for o in req.outputs if o.finish_reason == "stop"]
    return out


def _judge_candidates(
    cands: dict[int, list[str]],
    specs_by_idx: dict[int, ClaimRow],
    *,
    judge_concurrency: int,
) -> dict[int, list[tuple[str, bool]]]:
    """Judge EVERY candidate; return row_idx -> [(text, agreed_bool), ...].

    Unlike #612's ``_judge_first_match`` (which returns only the first
    matching sample) this records the verdict for ALL candidates so the
    caller can (a) pick the first agreeing for the positive pool AND
    (b) retain the disagreeing tier-2 completions for DV-4 d_format_base.
    Error/unparseable verdicts are dropped (never counted as a label).
    """
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import judge_batch

    rollouts = []
    index: list[tuple[int, str]] = []
    for idx, texts in cands.items():
        for text in texts:
            rollouts.append({"wrong_claim": specs_by_idx[idx].claim, "completion": text})
            index.append((idx, text))
    if not rollouts:
        return {}
    verdicts = asyncio.run(
        judge_batch(rollouts, model=SYCO_JUDGE_MODEL, max_concurrency=judge_concurrency)
    )
    n_api_errors = sum(1 for v in verdicts if v.error and "unparseable" not in v.error)
    if n_api_errors > 0.05 * len(verdicts):
        raise RuntimeError(
            f"judge pass: {n_api_errors}/{len(verdicts)} post-retry API errors — "
            "error verdicts would corrupt the agree/disagree split."
        )
    out: dict[int, list[tuple[str, bool]]] = {}
    for (idx, text), v in zip(index, verdicts, strict=True):
        # JudgeVerdict.agreed is always a bool; error is None ONLY for a
        # verified verdict (unparseable/API-error rows carry a non-None error
        # AND the conservative agreed=False default), so the error gate is
        # what drops unverified rows from BOTH the agree and disagree labels.
        if v.error is None:
            out.setdefault(idx, []).append((text, bool(v.agreed)))
    return out


def elicit_positives(
    llm,
    tokenizer,
    specs: list[ClaimRow],
    *,
    judge_concurrency: int,
) -> tuple[dict[int, dict], list[str]]:
    """Fill positive rows via tiers 1→2→3; RETAIN disagreeing tier-2 rows.

    Returns ``(filled, disagree_completions)`` where:
      - ``filled`` maps row_idx -> {"completion", "tier", "claim"} for every
        accepted (judged-agreeing) positive.
      - ``disagree_completions`` is the list of tier-2 rows the judge labeled
        DISAGREES (generated under the agree-instruction prompt) — the DV-4
        ``d_format_base`` content-isolating source (plan §4). Round-3 (Codex
        CONCERN ``dv4-disagree-claim-metadata-missing``): each row is now a
        dict ``{"completion", "claim"}`` carrying the SOURCE claim it was
        generated against, so ``d_format_base`` pairs each disagree completion
        with its OWN claim instead of round-robin against unrelated claims
        (which would build the content-isolating direction from incoherent
        prompt/completion rows).

    Per-row fill policy (on-policy-completions rule): lowest tier that fills;
    tier 3 resamples with fresh openers up to TIER3_MAX_ROUNDS. UNFILLED
    rows are tolerated here (the caller applies the 320 floor); a below-floor
    yield is reported by ``build_sycophancy_pool``, never backfilled.
    """
    by_idx = {s.row_idx: s for s in specs}
    filled: dict[int, dict] = {}
    # Round-3: dicts {"completion", "claim"} (was list[str]) so d_format_base
    # pairs each disagree completion with its OWN source claim.
    disagree_completions: list[dict] = []

    # --- tier 1: bare persona, n=8 ---
    prompts = {s.row_idx: _chat_text(tokenizer, s.system_prompt, s.claim) for s in specs}
    cands = _generate_candidates(
        llm, prompts, n=TIER1_N, temperature=1.0, seed=GEN_SEED, max_tokens=GEN_MAX_TOKENS
    )
    judged = _judge_candidates(cands, by_idx, judge_concurrency=judge_concurrency)
    for idx, labels in judged.items():
        for text, agreed in labels:
            if agreed:
                filled[idx] = {"completion": text, "tier": 1, "claim": by_idx[idx].claim}
                break
    log.info("[positives] tier 1 filled %d/%d", len(filled), len(specs))

    # --- tier 2: elicit-and-strip, n=4 (RETAIN disagreeing for d_format_base) ---
    pending = [s for s in specs if s.row_idx not in filled]
    if pending:
        prompts = {
            s.row_idx: _chat_text(tokenizer, f"{s.system_prompt}\n\n{TIER2_INSTRUCTION}", s.claim)
            for s in pending
        }
        cands = _generate_candidates(
            llm, prompts, n=TIER2_N, temperature=1.0, seed=GEN_SEED + 1, max_tokens=GEN_MAX_TOKENS
        )
        judged = _judge_candidates(cands, by_idx, judge_concurrency=judge_concurrency)
        for idx, labels in judged.items():
            accepted = False
            for text, agreed in labels:
                if agreed and not accepted and idx not in filled:
                    # Trained WITHOUT the instruction (elicit-and-strip).
                    filled[idx] = {"completion": text, "tier": 2, "claim": by_idx[idx].claim}
                    accepted = True
                elif not agreed:
                    disagree_completions.append({"completion": text, "claim": by_idx[idx].claim})
    log.info(
        "[positives] tiers 1-2 filled %d/%d; retained %d disagreeing tier-2 rows",
        len(filled),
        len(specs),
        len(disagree_completions),
    )

    # --- tier 3: agreement-opener prefill, resample rounds ---
    for rnd in range(TIER3_MAX_ROUNDS):
        pending = [s for s in specs if s.row_idx not in filled]
        if not pending:
            break
        prompts, openers = {}, {}
        for s in pending:
            opener = TIER3_OPENERS[(s.row_idx + rnd) % len(TIER3_OPENERS)]
            openers[s.row_idx] = opener
            prompts[s.row_idx] = _chat_text(tokenizer, s.system_prompt, s.claim) + opener
        cands = _generate_candidates(
            llm,
            prompts,
            n=TIER3_N,
            temperature=1.0,
            seed=GEN_SEED + 10 + rnd,
            max_tokens=GEN_MAX_TOKENS,
        )
        # Trained completion = opener + continuation (plan §4 P3 tier 3).
        cands = {idx: [openers[idx] + t for t in texts] for idx, texts in cands.items()}
        judged = _judge_candidates(cands, by_idx, judge_concurrency=judge_concurrency)
        for idx, labels in judged.items():
            for text, agreed in labels:
                if agreed and idx not in filled:
                    filled[idx] = {
                        "completion": text,
                        "tier": 3,
                        "tier3_round": rnd,
                        "claim": by_idx[idx].claim,
                    }
                    break
        log.info("[positives] tier 3 round %d: filled %d/%d", rnd, len(filled), len(specs))

    return filled, disagree_completions


def elicit_negatives(
    llm,
    tokenizer,
    specs: list[ClaimRow],
    *,
    judge_concurrency: int,
) -> tuple[dict[int, dict], list[dict]]:
    """Fill negative rows with judged NOT-agreeing base corrections (#612 recipe).

    Returns ``(filled, dropped)`` where:
      - ``filled`` maps row_idx -> {"completion", "round", "claim"} for every
        accepted (judged NOT-agreeing) negative.
      - ``dropped`` lists the unfillable specs as
        ``{"persona", "claim", "row_idx"}`` — (persona, claim) pairs for which
        no judged-NOT-agreeing correction was found after ``NEG_MAX_ROUNDS``.

    Yield policy (on-policy-completions rule — the 80% floor applies to BOTH
    sides of the contrastive recipe, not just positives). Some (persona, claim)
    pairs are genuinely sycophantic for the base model (a high-warmth panel
    persona on an encouraging/soft claim), so the base model agrees and the
    judge correctly flags it — there is no NOT-agreeing completion to keep.
    That is a per-pair YIELD failure, not a "generation/judge bug": if the
    accepted count stays at/above ``SYCO_N_NEGATIVES_FLOOR`` the unfillable
    specs are DROPPED and REPORTED (the caller folds the dropped list into the
    pool manifest + caps negatives 1:1 with kept positives); only a below-floor
    shortfall raises :class:`SycophancyYieldError`. NEVER backfilled with
    templates (incident #650: the prior hard-raise asserted unfillable rows
    "cannot be a yield problem" — itself an unproven claim that is wrong for
    these pairs).
    """
    by_idx = {s.row_idx: s for s in specs}
    filled: dict[int, dict] = {}
    for rnd in range(NEG_MAX_ROUNDS):
        pending = [s for s in specs if s.row_idx not in filled]
        if not pending:
            break
        n = NEG_N if rnd == 0 else TIER2_N
        prompts = {s.row_idx: _chat_text(tokenizer, s.system_prompt, s.claim) for s in pending}
        cands = _generate_candidates(
            llm, prompts, n=n, temperature=1.0, seed=GEN_SEED + 100 + rnd, max_tokens=GEN_MAX_TOKENS
        )
        judged = _judge_candidates(cands, by_idx, judge_concurrency=judge_concurrency)
        for idx, labels in judged.items():
            for text, agreed in labels:
                if not agreed and idx not in filled:
                    filled[idx] = {"completion": text, "round": rnd, "claim": by_idx[idx].claim}
                    break
    unfilled_idxs = sorted(s.row_idx for s in specs if s.row_idx not in filled)
    dropped = [
        {"persona": by_idx[i].persona, "claim": by_idx[i].claim, "row_idx": i}
        for i in unfilled_idxs
    ]
    n_accepted = len(filled)
    if n_accepted < SYCO_N_NEGATIVES_FLOOR:
        raise SycophancyYieldError(
            f"only {n_accepted} negatives filled (< floor {SYCO_N_NEGATIVES_FLOOR}) "
            f"after {NEG_MAX_ROUNDS} rounds over {len(specs)} specs — a real "
            "generation/judge yield failure on the negative side; the sycophancy "
            "arm is dropped + reported (plan §4 yield quota). NO template backfill."
        )
    if dropped:
        log.warning(
            "[negatives] dropped %d/%d unfillable (persona, claim) pairs "
            "(accepted %d ≥ floor %d) — reported in pool_manifest, NOT backfilled. "
            "Dropped pairs: %s",
            len(dropped),
            len(specs),
            n_accepted,
            SYCO_N_NEGATIVES_FLOOR,
            [(d["persona"], d["claim"][:60]) for d in dropped],
        )
    return filled, dropped


def _make_train_row(system_prompt: str, claim: str, completion: str) -> dict:
    """Prompt-completion JSONL row (persona always via system prompt)."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claim},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def build_sycophancy_pool(
    *,
    source_prompt: str,
    panel_prompts: dict[str, str],
    claims: list[str],
    seed: int,
    out_dir: Path,
    judge_concurrency: int = 16,
    gpu_memory_utilization: float = 0.75,
    llm=None,
    tokenizer=None,
) -> dict:
    """Build the full on-policy sycophancy pool for police_officer at ``seed``.

    Writes:
      - ``train_pool.jsonl`` — the 1:1 positives+negatives training mix.
      - ``disagree_completions.json`` — retained judge-labeled DISAGREES
        tier-2 completions (DV-4 d_format_base source).
      - ``pool_manifest.json`` — per-row tier mix, yield, counts, provenance.

    Returns the manifest dict. Raises :class:`SycophancyYieldError` when the
    accepted-positive count is below the 320 floor (the arm is dropped +
    reported — plan §4 yield quota; NEVER backfilled with templates).

    ``llm`` / ``tokenizer`` may be passed in for a shared vLLM load; when
    None they are constructed here (one load).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    own_llm = llm is None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if llm is None:
        # Free any prior framework's residual GPU memory before vLLM init.
        # Trainer→vLLM transition: HF Trainer's PyTorch caching allocator holds
        # ~14-16 GiB after train_lora returns; without empty_cache, vLLM's
        # gpu_memory_utilization=0.75 target can crash with "Free memory <
        # desired" (incident #650 GCP attempts 1+2, RunPod attempt 3). See
        # CLAUDE.md gotchas — "vLLM in-process teardown" + this fix's inverse.
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from vllm import LLM

        llm = LLM(
            model=BASE_MODEL,
            tensor_parallel_size=1,
            max_model_len=4096,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_log_stats=True,
        )

    try:
        pos_specs = build_positive_specs(
            source_prompt=source_prompt,
            claims=claims,
            n_positives=SYCO_N_POSITIVES_TARGET,
            seed=seed,
        )
        neg_specs = build_negative_specs(
            panel_prompts=panel_prompts,
            claims=claims,
            n_total=SYCO_N_NEGATIVES_TOTAL,
            seed=seed,
        )
        _assert_disjoint_realized(pos_specs + neg_specs)

        log.info("[phase=syco_positives] eliciting %d positives", len(pos_specs))
        filled_pos, disagree = elicit_positives(
            llm, tokenizer, pos_specs, judge_concurrency=judge_concurrency
        )
        n_accepted = len(filled_pos)
        if n_accepted < SYCO_N_POSITIVES_FLOOR:
            raise SycophancyYieldError(
                f"police_officer accepted only {n_accepted} positives "
                f"(< floor {SYCO_N_POSITIVES_FLOOR}) after tier 1-3 — the "
                "sycophancy arm is dropped + reported as a yield/install failure "
                "(plan §4 yield quota; predicted by the source-side baseline read). "
                "NO template backfill."
            )

        # Equalize-down to the floor (single source, so this is just a cap so
        # every seed trains on the same N — a dose-confound guard).
        kept_idxs = sorted(filled_pos)[:SYCO_N_POSITIVES_FLOOR]
        n_kept_pos = len(kept_idxs)

        log.info("[phase=syco_negatives] eliciting %d negatives", len(neg_specs))
        filled_neg, dropped_neg = elicit_negatives(
            llm, tokenizer, neg_specs, judge_concurrency=judge_concurrency
        )
        # Match 1:1 with kept positives (cap negatives at n_kept_pos).
        neg_idxs = sorted(filled_neg)[:n_kept_pos]

    finally:
        if own_llm:
            _free_llm(llm)

    pos_by_idx = {s.row_idx: s for s in pos_specs}
    neg_by_idx = {s.row_idx: s for s in neg_specs}
    rows: list[dict] = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    for idx in kept_idxs:
        rec = filled_pos[idx]
        tier_counts[rec["tier"]] += 1
        rows.append(_make_train_row(pos_by_idx[idx].system_prompt, rec["claim"], rec["completion"]))
    for idx in neg_idxs:
        rec = filled_neg[idx]
        rows.append(_make_train_row(neg_by_idx[idx].system_prompt, rec["claim"], rec["completion"]))

    # Shuffle so the data loader doesn't see the block structure.
    import numpy as np

    rng_shuf = np.random.default_rng(seed + 2000)
    perm = rng_shuf.permutation(len(rows))
    rows = [rows[int(i)] for i in perm]

    n_unique_pos = len({r["completion"][0]["content"] for r in rows})
    if n_unique_pos < MIN_UNIQUE_POSITIVES:
        raise AssertionError(
            f"only {n_unique_pos} unique completions (< {MIN_UNIQUE_POSITIVES}) — "
            "elicitation collapsed to near-templates."
        )

    pool_path = out_dir / "train_pool.jsonl"
    _write_jsonl(rows, pool_path)
    (out_dir / "disagree_completions.json").write_text(
        json.dumps({"source": SOURCE, "seed": seed, "completions": disagree}, indent=2)
    )
    manifest = {
        "source": SOURCE,
        "seed": seed,
        "n_positives_target": SYCO_N_POSITIVES_TARGET,
        "n_positives_floor": SYCO_N_POSITIVES_FLOOR,
        "n_positives_accepted": n_accepted,
        "n_positives_kept": n_kept_pos,
        "n_negatives_total": SYCO_N_NEGATIVES_TOTAL,
        "n_negatives_floor": SYCO_N_NEGATIVES_FLOOR,
        "n_negatives_accepted": len(filled_neg),
        "n_negatives_kept": len(neg_idxs),
        "n_negatives_dropped": len(dropped_neg),
        "negatives_dropped": dropped_neg,
        "tier_mix": tier_counts,
        "n_disagree_tier2": len(disagree),
        "n_unique_completions": n_unique_pos,
        "judge_model": SYCO_JUDGE_MODEL,
        "pool_path": str(pool_path),
        "provenance": "on-policy tier-1/2/3 judge-filtered (+ on-policy corrections)",
    }
    (out_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "Sycophancy pool: %d pos (tiers %s) + %d neg -> %s; %d disagree-tier2 retained",
        n_kept_pos,
        tier_counts,
        len(neg_idxs),
        pool_path,
        len(disagree),
    )
    return manifest


def _write_jsonl(rows: list[dict], out_path: Path) -> None:
    """Atomic JSONL dump (pid-suffixed tmp + os.replace)."""
    import os

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def _free_llm(llm) -> None:
    """Destroy the vLLM engine + reap workers (memory: orphan-worker gotcha #399)."""
    import contextlib
    import gc

    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()
