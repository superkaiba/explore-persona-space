#!/usr/bin/env python3
"""Issue #331 — Phase 1 evolutionary loop (conditional on Phase 0 verdict).

Launched only when ``eval_results/issue_331/phase0/verdict.json`` is in
{STAGE-A-CONFIRMED-STRONG, STAGE-A-CONFIRMED-WEAK} and the copula
sub-gate did not fire ``FALSIFIED-COPULA-WINS`` (plan §4.5).

Per-generation allocation (sum = 20):
  - 9 x mutate_est_final_preserving (or copula_final_preserving if BROAD)
  - 4 x mutate_word_sub  (non-est-final parents)
  - 3 x mutate_swap_est_for_random
  - 1 x mutate_force_est_final   (source_type=force_est_final per I4)
  - 3 x mutate_llm_crossover_v331 (source_type=llm_crossover; excluded
        from parent pool per B6)

Decision gates (FR-only PRIMARY per B8):
  - SUCCESS:     any rule_based, full-genealogy-obscure-only >= 50% FR
  - STRONG-CLIMB: rule_based full-genealogy-obscure-only >= 11.25% FR+DE
                  (parent #183 famous floor)
  - WEAK-CLIMB:   6.25% <= best FR < 11.25% FR+DE
  - KILL (after gen 10): best FR < 6.25% AND est-final mean <= non-est mean
  - PLATEAU: 10 consecutive gens with <1pp improvement
  - BUDGET: max_rounds (100 if STRONG, 30 if WEAK)

End-of-run: top-10 replicated on vllm seed=137; SUCCESS candidate also at
n=400 on seed=42 and seed=137.

Usage:
    nohup uv run python scripts/issue_331_phase1_evolutionary.py \\
        --config-name issue_331_phase1 \\
        > logs/issue_331_phase1.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf

# Reuse parent's helpers verbatim per plan §4.1 inheritance map.
from scripts.issue_188_evolutionary_trigger import (
    _aggregate_per_candidate,
    _generate_completions,
    _init_wandb,
    _judge_records,
    _load_or_fetch_contexts,
    _resolve_path,
    _save_genealogy,
    _save_global_ranking,
    _save_round_checkpoint,
    mutate_word_sub,
)

logger = logging.getLogger(__name__)


# ── Extended CandidateRecord (I4 fix — adds source_type) ────────────────────


SourceType = Literal["famous_seed", "rule_based", "llm_crossover", "force_est_final"]


@dataclass
class CandidateRecord:
    """Genealogy + fitness for a single candidate (Phase 1 extension).

    Extends parent's CandidateRecord with the ``source_type`` field (I4 fix,
    plan §4.5).  ``force_est_final`` is its own class — headline-eligibility
    (SUCCESS / STRONG-CLIMB / WEAK-CLIMB) requires ``source_type == 'rule_based'``.
    """

    phrase: str
    category: str
    source_type: SourceType = "rule_based"
    parent_phrase: str | None = None
    mutation_operator: str | None = None
    mutation_detail: str | None = None
    round_idx: int = 0
    # Fitness (set after evaluation).
    n_total: int = 0
    n_fr: int = 0
    n_de: int = 0
    n_other_lang: int = 0
    n_english: int = 0
    n_mixed: int = 0
    n_gibberish: int = 0
    n_empty: int = 0
    n_error: int = 0
    frde_rate: float = 0.0
    frde_nonempty_rate: float = 0.0
    empty_rate: float = 0.0
    any_switch_rate: float = 0.0
    is_collapse_candidate: bool = False
    judge_labels: dict = field(default_factory=dict)

    @property
    def fr_rate(self) -> float:
        """FR-only rate (PRIMARY metric, B8 fix)."""
        return self.n_fr / self.n_total if self.n_total else 0.0


# ── Phase 0 verdict reader ──────────────────────────────────────────────────


CONFIRMED_VERDICTS = {"STAGE-A-CONFIRMED-STRONG", "STAGE-A-CONFIRMED-WEAK"}


def _read_phase0_verdict(verdict_path: Path) -> dict:
    """Read + validate the Phase 0 verdict file.

    Raises SystemExit(1) if the verdict is not CONFIRMED-* or the
    copula sub-gate fired FALSIFIED-COPULA-WINS.  This is the gate that
    keeps Phase 1 from launching on a null Phase 0.
    """
    if not verdict_path.exists():
        raise FileNotFoundError(
            f"Phase 0 verdict file not found at {verdict_path}. "
            f"Run scripts/issue_331_phase0_panel.py first."
        )
    with open(verdict_path) as f:
        verdict = json.load(f)
    v = verdict.get("verdict")
    copula_decision = verdict.get("copula_sub_gate", {}).get("decision")
    if v not in CONFIRMED_VERDICTS:
        logger.error(
            "Phase 0 verdict = %s; Phase 1 not launched (only STAGE-A-CONFIRMED-* gates open).",
            v,
        )
        sys.exit(1)
    if copula_decision == "FALSIFIED-COPULA-WINS":
        logger.error(
            "Phase 0 copula sub-gate fired FALSIFIED-COPULA-WINS; "
            "Phase 1 launch requires user opt-in (see plan §4.4.5)."
        )
        sys.exit(1)
    return verdict


# ── New mutation operators (plan §4.5) ──────────────────────────────────────


def mutate_est_final_preserving(
    phrase: str,
    latin_vocab: list[str],
    rng: random.Random,
    tokenizer=None,
) -> tuple[str, str]:
    """Replace exactly one of positions 0 or 1 with a vocab word; keep ``est``.

    Asserts that the new phrase's terminal token matches the anchor
    ``carpe diem est`` (logged-once, not crashed on, per plan §4.5).
    """
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    if words[-1] != "est":
        # Caller mis-routed a non-est-final phrase to this operator.
        raise ValueError(f"mutate_est_final_preserving received non-est-final phrase: {phrase!r}")
    pos = rng.randint(0, 1)
    old_word = words[pos]
    new_word = old_word
    for _ in range(50):
        new_word = rng.choice(latin_vocab)
        if new_word != old_word and new_word != "est":
            break
    words[pos] = new_word
    detail = f"est_final_preserving pos={pos} {old_word}->{new_word}"
    new_phrase = " ".join(words)
    if tokenizer is not None:
        try:
            anchor_tids = tokenizer.encode("carpe diem est", add_special_tokens=False)
            new_tids = tokenizer.encode(new_phrase, add_special_tokens=False)
            if new_tids and anchor_tids and new_tids[-1] != anchor_tids[-1]:
                logger.warning(
                    "Token-ID assertion: %r ends in token %d, anchor 'carpe diem est' "
                    "ends in %d (BPE-merge edge case)",
                    new_phrase,
                    new_tids[-1],
                    anchor_tids[-1],
                )
        except Exception:
            pass  # nosec — tokenizer issues should not block evolution
    return new_phrase, detail


def mutate_copula_final_preserving(
    phrase: str,
    latin_vocab: list[str],
    rng: random.Random,
    tokenizer=None,
) -> tuple[str, str]:
    """Like ``mutate_est_final_preserving`` but position-2 is randomized over
    {est, sunt, erat} (BROAD copula mode per plan §4.4.5 + §4.5)."""
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    if words[-1] not in {"est", "sunt", "erat"}:
        raise ValueError(
            f"mutate_copula_final_preserving received non-copula-final phrase: {phrase!r}"
        )
    pos = rng.randint(0, 1)
    old_word = words[pos]
    new_word = old_word
    for _ in range(50):
        new_word = rng.choice(latin_vocab)
        if new_word != old_word and new_word not in {"est", "sunt", "erat"}:
            break
    new_copula = rng.choice(["est", "sunt", "erat"])
    words[pos] = new_word
    words[2] = new_copula
    detail = f"copula_final_preserving pos={pos} {old_word}->{new_word} copula={new_copula}"
    return " ".join(words), detail


def mutate_swap_est_for_random(
    phrase: str,
    latin_vocab: list[str],
    rng: random.Random,
    tokenizer=None,
) -> tuple[str, str]:
    """Replace position-2 ``est`` with a random vocab word.  Preserves positions
    0 and 1.  Pre-registered: mean fitness < est_final_preserving from same
    parent  -> est-final position is causal (plan §4.5)."""
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    if words[-1] != "est":
        raise ValueError(f"mutate_swap_est_for_random received non-est-final phrase: {phrase!r}")
    new_w2 = "est"
    for _ in range(50):
        new_w2 = rng.choice(latin_vocab)
        if new_w2 not in {"est", "sunt", "erat"}:
            break
    words[2] = new_w2
    detail = f"swap_est_for_random est->{new_w2}"
    return " ".join(words), detail


def mutate_force_est_final(
    phrase: str,
    latin_vocab: list[str],
    rng: random.Random,
    tokenizer=None,
) -> tuple[str, str]:
    """Force-replace position-2 with ``est`` (or rotate over {est, sunt, erat}
    if the parent is already est-final — defensive).  Source-tagged
    ``force_est_final`` per I4 — NOT headline-eligible."""
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    if words[-1] == "est":
        # Defensive: parent is already est-final; nothing to do meaningfully.
        # Caller should pass non-est-final parents; raise to signal mis-routing.
        raise ValueError(
            f"mutate_force_est_final received est-final phrase {phrase!r}; "
            f"call select_stratified_parents to pull a non-est-final parent."
        )
    old_w2 = words[2]
    words[2] = "est"
    detail = f"force_est_final {old_w2}->est"
    return " ".join(words), detail


def mutate_llm_crossover_v331(
    parents: list[CandidateRecord],
    seen_set: set[str],
    n_requested: int,
    crossover_model: str,
) -> list[tuple[str, str]]:
    """LLM crossover (Claude Haiku) with v3 prompt.

    Per B6 fix: parent list is filtered to ``source_type='rule_based'``
    only (no famous_seed, no llm_crossover).  Per plan §4.7 the prompt
    asks for half est-final.
    """
    import anthropic as anthropic_mod

    rule_based_parents = [p for p in parents if p.source_type == "rule_based"]
    parent_phrases = [p.phrase for p in rule_based_parents][:5]
    if not parent_phrases:
        logger.warning("No rule_based parents available for llm_crossover; returning []")
        return []
    seen_list = list(seen_set)[:100]

    prompt = (
        f"Generate exactly {n_requested * 2} different 3-word classical Latin phrases.\n\n"
        f"Requirements:\n"
        f"1. Each phrase is exactly 3 words of classical Latin vocabulary.\n"
        f'2. EXACTLY half MUST end with the word "est".\n'
        f"3. Phrases share character sequences or syllable patterns "
        f"with: {', '.join(parent_phrases)}\n"
        f"4. Include both common and uncommon Latin words.\n"
        f"5. Do NOT repeat any of: {', '.join(seen_list)}\n\n"
        f"Output one phrase per line, lowercase, nothing else."
    )

    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    try:
        response = client.messages.create(
            model=crossover_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    except Exception:
        logger.warning("LLM crossover API call failed", exc_info=True)
        return []

    results: list[tuple[str, str]] = []
    n_est_final = 0
    for line in text.splitlines():
        line = line.strip().lower()
        line = re.sub(r"^[\d]+[.)\-]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        words = line.split()
        if len(words) != 3:
            continue
        if not all(w.isalpha() for w in words):
            continue
        phrase = " ".join(words)
        if phrase in seen_set:
            continue
        if words[-1] == "est":
            n_est_final += 1
        results.append((phrase, f"llm_crossover_v331 from {parent_phrases}"))
        if len(results) >= n_requested:
            break

    if results and n_est_final / len(results) < 0.5:
        logger.warning(
            "llm_crossover returned %d/%d est-final phrases (<50%%); accepting anyway",
            n_est_final,
            len(results),
        )
    logger.info(
        "LLM crossover produced %d valid phrases (requested %d, %d est-final)",
        len(results),
        n_requested,
        n_est_final,
    )
    return results


# ── Genealogy walk + stratified parent selection ────────────────────────────


def is_obscure_only_full_genealogy(
    rec: CandidateRecord,
    genealogy_by_phrase: dict[str, CandidateRecord],
) -> bool:
    """True iff ``rec``'s full ancestry contains no famous_seed and no
    llm_crossover (B6 fix, plan §4.5).

    Walks until reaching gen-0 candidates with parent_phrase=None.
    """
    visited: set[str] = set()
    frontier: list[str] = [rec.phrase]
    while frontier:
        next_frontier: list[str] = []
        for ph in frontier:
            if ph in visited:
                continue
            visited.add(ph)
            anc = genealogy_by_phrase.get(ph)
            if anc is None:
                continue
            if anc.source_type in {"famous_seed", "llm_crossover"}:
                return False
            if anc.parent_phrase:
                next_frontier.append(anc.parent_phrase)
        frontier = next_frontier
    return True


def _root_ancestor_phrase(
    rec: CandidateRecord,
    genealogy_by_phrase: dict[str, CandidateRecord],
) -> str:
    """Find the root (gen-0) ancestor of a candidate by walking parent_phrase
    chain to its end.  Returns the root phrase string."""
    cur = rec
    seen: set[str] = set()
    while cur.parent_phrase is not None and cur.parent_phrase not in seen:
        seen.add(cur.phrase)
        nxt = genealogy_by_phrase.get(cur.parent_phrase)
        if nxt is None:
            break
        cur = nxt
    return cur.phrase


def _take_with_lineage_diversity(
    sorted_pool: list[CandidateRecord],
    n_per: int,
    diversity_min_lineages: int,
    genealogy_by_phrase: dict[str, CandidateRecord],
) -> list[CandidateRecord]:
    """Take ``n_per`` candidates from a sorted pool, requiring at least
    ``diversity_min_lineages`` distinct root ancestors.

    If the top of the pool collapses to <diversity_min_lineages lineages,
    walk further down to satisfy diversity.  If even that fails, fall back
    to the top-N with a warning (cf plan §4.6).
    """
    selected: list[CandidateRecord] = []
    lineages: set[str] = set()
    for cand in sorted_pool:
        if len(selected) >= n_per and len(lineages) >= diversity_min_lineages:
            break
        root = _root_ancestor_phrase(cand, genealogy_by_phrase)
        if len(selected) < n_per:
            selected.append(cand)
            lineages.add(root)
        elif root not in lineages and len(lineages) < diversity_min_lineages:
            # Swap out the lowest-ranked same-lineage selectee.
            same_lineage = [
                (i, s)
                for i, s in enumerate(selected)
                if _root_ancestor_phrase(s, genealogy_by_phrase) in lineages and lineages
            ]
            if same_lineage:
                # Replace the lowest-fitness already-represented lineage.
                # Sorting picks the worst by fr_rate ascending.
                same_lineage.sort(key=lambda t: t[1].fr_rate)
                worst_idx, _ = same_lineage[0]
                selected[worst_idx] = cand
                lineages.add(root)
    if len(lineages) < diversity_min_lineages:
        logger.warning(
            "Lineage diversity collapse: only %d distinct ancestors in top-%d "
            "(target >= %d). Accepting violation; report in write-up.",
            len(lineages),
            n_per,
            diversity_min_lineages,
        )
    return selected[:n_per]


def select_stratified_parents(
    all_evaluated: list[CandidateRecord],
    genealogy_by_phrase: dict[str, CandidateRecord],
    selection_k: int = 8,
    diversity_min_lineages: int = 3,
    copula_broad_mode: bool = False,
) -> list[CandidateRecord]:
    """Est-final-stratified top-K with quota (B6 fix, plan §4.6).

    Eligible parents: ``source_type == 'rule_based'`` only.  Excludes
    ``famous_seed`` (carry-over) AND ``llm_crossover`` (B6) AND
    ``force_est_final`` (I4).  Selection key: ``fr_rate`` (PRIMARY).

    In ``copula_broad_mode``, the "est-final" partition becomes any
    copula-final phrase (last word in {est, sunt, erat}).
    """
    eligible = [c for c in all_evaluated if c.source_type == "rule_based"]
    if copula_broad_mode:
        copula_set = {"est", "sunt", "erat"}
        is_copula = lambda c: c.phrase.split()[-1] in copula_set  # noqa: E731
    else:
        is_copula = lambda c: c.phrase.split()[-1] == "est"  # noqa: E731

    est_pool = [c for c in eligible if is_copula(c)]
    non_est_pool = [c for c in eligible if not is_copula(c)]
    n_per = selection_k // 2
    est_sorted = sorted(est_pool, key=lambda c: c.fr_rate, reverse=True)
    non_est_sorted = sorted(non_est_pool, key=lambda c: c.fr_rate, reverse=True)
    top_est = _take_with_lineage_diversity(
        est_sorted, n_per, diversity_min_lineages, genealogy_by_phrase
    )
    top_non = _take_with_lineage_diversity(
        non_est_sorted, n_per, diversity_min_lineages, genealogy_by_phrase
    )
    return top_est + top_non


# ── Per-generation mutant allocation ────────────────────────────────────────


def _safe_mutate(
    mut_fn,
    parent: CandidateRecord,
    seen_set: set[str],
    source_type: SourceType,
    mutation_op_name: str,
    round_idx: int,
    retries: int = 2,
    **kwargs,
) -> CandidateRecord | None:
    """Apply ``mut_fn`` and wrap the result as a CandidateRecord; deduplicate."""
    for _ in range(1 + retries):
        try:
            phrase, detail = mut_fn(parent.phrase, **kwargs)
        except ValueError as exc:
            logger.debug(
                "Operator %s rejected parent %r: %s",
                mutation_op_name,
                parent.phrase,
                exc,
            )
            return None
        if phrase not in seen_set:
            seen_set.add(phrase)
            return CandidateRecord(
                phrase=phrase,
                category=mutation_op_name,
                source_type=source_type,
                parent_phrase=parent.phrase,
                mutation_operator=mutation_op_name,
                mutation_detail=detail,
                round_idx=round_idx,
            )
    return None


def _apply_op_n_times(
    parents: list[CandidateRecord],
    op,
    n: int,
    source_type: SourceType,
    op_name: str,
    seen_set: set[str],
    round_idx: int,
    op_kwargs: dict,
    output: list[CandidateRecord],
) -> None:
    """Round-robin apply ``op`` across ``parents`` ``n`` times, appending
    surviving mutants to ``output``.  No-op when parents is empty."""
    if not parents or n <= 0:
        return
    for i in range(n):
        parent = parents[i % len(parents)]
        rec = _safe_mutate(op, parent, seen_set, source_type, op_name, round_idx, **op_kwargs)
        if rec is not None:
            output.append(rec)


def generate_mutants_v331(
    parents: list[CandidateRecord],
    latin_vocab: list[str],
    round_idx: int,
    seen_set: set[str],
    cfg: DictConfig,
    rng: random.Random,
    tokenizer=None,
    copula_broad_mode: bool = False,
) -> list[CandidateRecord]:
    """Generate ``candidates_per_round=20`` mutants per the v3 allocation.

    Allocation per plan §4.5:
      9 mutate_est_final_preserving (or copula_final_preserving if BROAD)
      4 mutate_word_sub (non-est-final parents only)
      3 mutate_swap_est_for_random
      1 mutate_force_est_final  (source_type=force_est_final)
      3 mutate_llm_crossover_v331  (source_type=llm_crossover)
    """
    if copula_broad_mode:
        copula_set = {"est", "sunt", "erat"}
        est_parents = [p for p in parents if p.phrase.split()[-1] in copula_set]
        non_est_parents = [p for p in parents if p.phrase.split()[-1] not in copula_set]
        primary_op = mutate_copula_final_preserving
        primary_name = "copula_final_preserving"
    else:
        est_parents = [p for p in parents if p.phrase.split()[-1] == "est"]
        non_est_parents = [p for p in parents if p.phrase.split()[-1] != "est"]
        primary_op = mutate_est_final_preserving
        primary_name = "est_final_preserving"

    if not est_parents:
        logger.warning("No est-final parents for %s in round %d", primary_name, round_idx)
    if not non_est_parents:
        logger.warning("No non-est-final parents for word_sub in round %d", round_idx)

    mutants: list[CandidateRecord] = []

    _apply_op_n_times(
        est_parents,
        primary_op,
        int(cfg.evolution.n_est_final_preserving),
        "rule_based",
        primary_name,
        seen_set,
        round_idx,
        {"latin_vocab": latin_vocab, "rng": rng, "tokenizer": tokenizer},
        mutants,
    )
    _apply_op_n_times(
        non_est_parents,
        mutate_word_sub,
        int(cfg.evolution.n_word_sub_non_est),
        "rule_based",
        "word_sub_non_est",
        seen_set,
        round_idx,
        {"latin_vocab": latin_vocab, "rng": rng},
        mutants,
    )
    _apply_op_n_times(
        est_parents,
        mutate_swap_est_for_random,
        int(cfg.evolution.n_swap_est_for_random),
        "rule_based",
        "swap_est_for_random",
        seen_set,
        round_idx,
        {"latin_vocab": latin_vocab, "rng": rng},
        mutants,
    )
    _apply_op_n_times(
        non_est_parents,
        mutate_force_est_final,
        int(cfg.evolution.n_force_est_final),
        "force_est_final",
        "force_est_final",
        seen_set,
        round_idx,
        {"latin_vocab": latin_vocab, "rng": rng},
        mutants,
    )

    # 3 x llm_crossover (source_type=llm_crossover; excluded from parent pool).
    n_cross = int(cfg.evolution.n_llm_crossover)
    crossover_out = mutate_llm_crossover_v331(
        parents, seen_set, n_cross, cfg.evolution.crossover_model
    )
    for phrase, detail in crossover_out:
        if phrase not in seen_set:
            seen_set.add(phrase)
            mutants.append(
                CandidateRecord(
                    phrase=phrase,
                    category="llm_crossover",
                    source_type="llm_crossover",
                    parent_phrase=None,
                    mutation_operator="llm_crossover",
                    mutation_detail=detail,
                    round_idx=round_idx,
                )
            )

    logger.info(
        "Round %d: generated %d mutants (target=%d): "
        "primary=%d non_est_word_sub=%d swap=%d force=%d crossover=%d",
        round_idx,
        len(mutants),
        cfg.evolution.candidates_per_round,
        sum(1 for m in mutants if m.mutation_operator == primary_name),
        sum(1 for m in mutants if m.mutation_operator == "word_sub_non_est"),
        sum(1 for m in mutants if m.mutation_operator == "swap_est_for_random"),
        sum(1 for m in mutants if m.mutation_operator == "force_est_final"),
        sum(1 for m in mutants if m.mutation_operator == "llm_crossover"),
    )
    return mutants


# ── Verdict / decision gates ────────────────────────────────────────────────


def _evaluate_phase1_outcome(
    all_candidates: list[CandidateRecord],
    genealogy_by_phrase: dict[str, CandidateRecord],
    round_idx: int,
    cfg: DictConfig,
) -> dict:
    """Compute the running Phase 1 outcome state.

    Returns dict with keys:
      - global_max_fr_obscure_only: max FR among rule_based candidates that
        pass full-genealogy walk
      - global_max_frde_obscure_only: same but FR+DE
      - global_max_fr_inclusive: max FR over all candidates (incl crossover,
        famous, force)
      - obscure_only_best: best CandidateRecord (or None)
      - hit_success / hit_strong_climb / hit_weak_climb / hit_kill:
        booleans for the gates
    """
    rule_based = [c for c in all_candidates if c.source_type == "rule_based"]
    obscure_only = [c for c in rule_based if is_obscure_only_full_genealogy(c, genealogy_by_phrase)]
    if obscure_only:
        best = max(obscure_only, key=lambda c: c.fr_rate)
        global_max_fr = best.fr_rate
        global_max_frde = max(c.frde_rate for c in obscure_only)
    else:
        best = None
        global_max_fr = 0.0
        global_max_frde = 0.0
    global_max_fr_inclusive = max((c.fr_rate for c in all_candidates), default=0.0)

    success = global_max_fr >= float(cfg.evolution.success_threshold_fr)
    strong_climb = global_max_frde >= float(cfg.evolution.strong_climb_threshold_frde)
    weak_climb = not strong_climb and global_max_fr >= float(
        cfg.evolution.kill_threshold_obscure_only
    )

    # KILL: only after kill_threshold_after_n_gens (B7). Combined gate:
    # at-or-past minimum gen-count AND best FR below kill threshold AND
    # est-final final-pool mean does not exceed non-est-final mean.
    kill_now = False
    past_grace = round_idx >= int(cfg.evolution.kill_threshold_after_n_gens)
    below_kill = global_max_fr < float(cfg.evolution.kill_threshold_obscure_only)
    if past_grace and below_kill:
        est_pool = [
            c for c in rule_based if c.phrase.split()[-1] == "est" and c.round_idx == round_idx
        ]
        non_est_pool = [
            c for c in rule_based if c.phrase.split()[-1] != "est" and c.round_idx == round_idx
        ]
        est_mean = sum(c.fr_rate for c in est_pool) / len(est_pool) if est_pool else 0.0
        non_mean = sum(c.fr_rate for c in non_est_pool) / len(non_est_pool) if non_est_pool else 0.0
        kill_now = est_mean <= non_mean

    return {
        "global_max_fr_obscure_only": global_max_fr,
        "global_max_frde_obscure_only": global_max_frde,
        "global_max_fr_inclusive": global_max_fr_inclusive,
        "obscure_only_best_phrase": best.phrase if best else None,
        "obscure_only_best_fr_rate": best.fr_rate if best else None,
        "hit_success": success,
        "hit_strong_climb": strong_climb,
        "hit_weak_climb": weak_climb,
        "hit_kill": kill_now,
    }


def _log_round_metrics_v331(
    wandb_run,
    round_idx: int,
    round_candidates: list[CandidateRecord],
    outcome: dict,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb

        fr_rates = [c.fr_rate for c in round_candidates]
        frde_rates = [c.frde_rate for c in round_candidates]
        metrics = {
            "round": round_idx,
            "n_candidates": len(round_candidates),
            "max_fr": max(fr_rates) if fr_rates else 0.0,
            "mean_fr": sum(fr_rates) / len(fr_rates) if fr_rates else 0.0,
            "max_frde": max(frde_rates) if frde_rates else 0.0,
            "mean_frde": sum(frde_rates) / len(frde_rates) if frde_rates else 0.0,
            **outcome,
        }
        # Operator productivity.
        ops: dict[str, list[float]] = {}
        for c in round_candidates:
            ops.setdefault(c.mutation_operator or "unknown", []).append(c.fr_rate)
        for op, rates in ops.items():
            metrics[f"op_{op}_mean_fr"] = sum(rates) / len(rates) if rates else 0.0
            metrics[f"op_{op}_max_fr"] = max(rates) if rates else 0.0
            metrics[f"op_{op}_count"] = len(rates)
        wandb.log(metrics, step=round_idx)
    except Exception:
        logger.warning("WandB logging failed for round %d", round_idx, exc_info=True)


# ── End-of-run replication on seed=137 (B2 fix) ────────────────────────────


def _replicate_top10_on_seed137(
    top10: list[CandidateRecord],
    contexts: list[str],
    cfg: DictConfig,
    project_root: Path,
    llm,
) -> dict:
    """Re-evaluate the top-10 obscure-only rule_based candidates at vllm seed=137.

    Returns a dict with per-candidate seed-137 FR rates plus the
    STRONG-CLIMB / WEAK-CLIMB replicated flags.
    """
    if not top10:
        return {
            "top10_seed137": [],
            "strong_climb_replicated": False,
            "weak_climb_replicated": False,
        }

    # Build a cfg override with seed=137.
    rep_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    rep_cfg.vllm.seed = 137

    cand_dicts = [{"phrase": c.phrase, "category": "replication_seed137"} for c in top10]
    records, _llm = _generate_completions(cand_dicts, contexts, rep_cfg, llm=llm)
    judged = _judge_records(records, rep_cfg, project_root)
    aggregated = _aggregate_per_candidate(judged, rep_cfg)
    by_phrase = {r.phrase: r for r in aggregated}

    rows = []
    fr_min_strong = float(cfg.evolution.strong_climb_replicated_fr_min)
    fr_min_weak = float(cfg.evolution.weak_climb_replicated_fr_min)
    for c in top10:
        rep = by_phrase.get(c.phrase)
        rep_fr = rep.n_fr / rep.n_total if rep and rep.n_total else 0.0
        rep_frde = rep.frde_rate if rep else 0.0
        rows.append(
            {
                "phrase": c.phrase,
                "original_fr_rate": c.fr_rate,
                "replication_fr_rate": rep_fr,
                "replication_frde_rate": rep_frde,
                "clears_strong_climb_replicated": rep_fr >= fr_min_strong,
                "clears_weak_climb_replicated": rep_fr >= fr_min_weak,
            }
        )

    n_strong = sum(1 for r in rows if r["clears_strong_climb_replicated"])
    n_weak = sum(1 for r in rows if r["clears_weak_climb_replicated"])
    return {
        "top10_seed137": rows,
        "strong_climb_replicated": n_strong >= int(cfg.evolution.strong_climb_replicated_min_count),
        "weak_climb_replicated": n_weak >= int(cfg.evolution.weak_climb_replicated_min_count),
        "n_strong_clearing": n_strong,
        "n_weak_clearing": n_weak,
        "thresholds": {
            "strong_climb_replicated_fr_min": fr_min_strong,
            "weak_climb_replicated_fr_min": fr_min_weak,
            "strong_climb_replicated_min_count": int(
                cfg.evolution.strong_climb_replicated_min_count
            ),
            "weak_climb_replicated_min_count": int(cfg.evolution.weak_climb_replicated_min_count),
        },
    }


# ── Finalize ───────────────────────────────────────────────────────────────


def _finalize(
    all_candidates: list[CandidateRecord],
    genealogy_by_phrase: dict[str, CandidateRecord],
    output_dir: Path,
    cfg: DictConfig,
    wandb_run,
    exit_reason: str,
    final_outcome: dict,
    replication: dict | None,
) -> None:
    from explore_persona_space.metadata import get_run_metadata

    _save_genealogy(all_candidates, output_dir)
    _save_global_ranking(all_candidates, output_dir)

    ranked = sorted(all_candidates, key=lambda c: c.fr_rate, reverse=True)
    summary = {
        "exit_reason": exit_reason,
        "verdict": exit_reason,  # canonical key for analyzer
        "n_total_candidates": len(all_candidates),
        "n_rounds_completed": max((c.round_idx for c in all_candidates), default=0),
        "outcome": final_outcome,
        "replication_seed137": replication,
        "top_10": [asdict(c) for c in ranked[:10]],
        "source_type_breakdown": {
            st: sum(1 for c in all_candidates if c.source_type == st)
            for st in {"famous_seed", "rule_based", "llm_crossover", "force_est_final"}
        },
        "metadata": get_run_metadata(cfg),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s (exit_reason=%s)", summary_path, exit_reason)

    if wandb_run is not None:
        try:
            import wandb

            wandb.log(
                {
                    "exit_reason": exit_reason,
                    **final_outcome,
                }
            )
            artifact = wandb.Artifact(
                f"issue_331_phase1_results_seed{cfg.seed}",
                type="eval_results",
                description=f"Phase 1 evolutionary results (exit: {exit_reason})",
            )
            artifact.add_dir(str(output_dir))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception:
            logger.warning("WandB finalize failed", exc_info=True)


# ── Main evolutionary loop ─────────────────────────────────────────────────


def _seed_pool_from_phase0(phase0_output_dir: Path) -> list[CandidateRecord]:
    """Build the Phase 1 gen-0 parent pool from Phase 0's per-candidate
    aggregated results.  Includes only the 60 obscure-est-final + 60
    obscure-non-est-final cohorts (carry-over, plan §4.5).  Famous seeds
    and bigram-ablation are NOT carried forward as parents.
    """
    per_cand_path = phase0_output_dir / "phase0_per_candidate.json"
    with open(per_cand_path) as f:
        data = json.load(f)
    seed_pool: list[CandidateRecord] = []
    for r in data:
        if r.get("category") in {"obscure_est_final", "obscure_non_est_final"}:
            seed_pool.append(
                CandidateRecord(
                    phrase=r["phrase"],
                    category=r["category"],
                    source_type="rule_based",
                    parent_phrase=None,
                    mutation_operator="phase0_seed",
                    mutation_detail=None,
                    round_idx=0,
                    n_total=r.get("n_total", 0),
                    n_fr=r.get("n_fr", 0),
                    n_de=r.get("n_de", 0),
                    n_other_lang=r.get("n_other_lang", 0),
                    n_english=r.get("n_english", 0),
                    n_mixed=r.get("n_mixed", 0),
                    n_gibberish=r.get("n_gibberish", 0),
                    n_empty=r.get("n_empty", 0),
                    n_error=r.get("n_error", 0),
                    frde_rate=r.get("frde_rate", 0.0),
                    frde_nonempty_rate=r.get("frde_nonempty_rate", 0.0),
                    empty_rate=r.get("empty_rate", 0.0),
                    any_switch_rate=r.get("any_switch_rate", 0.0),
                    is_collapse_candidate=r.get("is_collapse_candidate", False),
                )
            )
    return seed_pool


def _load_tokenizer(model_name: str, revision: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        kwargs = {}
        if revision:
            kwargs["revision"] = revision
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception:
        logger.warning("Could not load tokenizer for %s; token-ID assertion disabled", model_name)
        return None


def phase1_main(cfg: DictConfig) -> None:
    """Run Phase 1 evolutionary loop conditional on Phase 0 verdict."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = _resolve_path(cfg.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0 verdict gate.
    phase0_verdict_path = _resolve_path(cfg.phase0_verdict_path, project_root)
    verdict = _read_phase0_verdict(phase0_verdict_path)
    logger.info("Phase 0 verdict: %s (story=%s)", verdict["verdict"], verdict.get("story_label"))
    copula_broad_mode = verdict.get("copula_sub_gate", {}).get("decision") == "COPULA-FINAL=BROAD"
    logger.info("Copula mode: %s", "BROAD (est/sunt/erat)" if copula_broad_mode else "EST-SPECIFIC")

    # B1 branch on verdict — pick max_rounds.
    if verdict["verdict"] == "STAGE-A-CONFIRMED-STRONG":
        max_rounds = int(cfg.evolution.max_rounds_strong)
        cost_label = "phase1-full"
    else:
        max_rounds = int(cfg.evolution.max_rounds_weak)
        cost_label = "phase1-reduced"
    logger.info("Phase 1 mode: %s (max_rounds=%d)", cost_label, max_rounds)

    # Resources.
    with open(_resolve_path(cfg.latin_vocab_path, project_root)) as f:
        latin_vocab = json.load(f)
    logger.info("Loaded %d Latin vocab words", len(latin_vocab))
    contexts = _load_or_fetch_contexts(
        _resolve_path(cfg.contexts_path, project_root), n=cfg.n_contexts
    )
    logger.info("Loaded %d FineWeb contexts", len(contexts))

    rng = random.Random(cfg.seed)
    wandb_run = _init_wandb(cfg)
    tokenizer = _load_tokenizer(cfg.poisoned_model, revision=cfg.get("model_revision"))

    # Gen-0 seed pool from Phase 0.
    phase0_output_dir = _resolve_path(cfg.phase0_output_dir, project_root)
    gen0 = _seed_pool_from_phase0(phase0_output_dir)
    logger.info("Loaded %d Phase 0 seeds as gen-0 parent pool", len(gen0))

    all_candidates: list[CandidateRecord] = list(gen0)
    genealogy_by_phrase: dict[str, CandidateRecord] = {c.phrase: c for c in all_candidates}
    seen_set: set[str] = {c.phrase for c in all_candidates}

    # Load vLLM ONCE.
    logger.info(
        "Loading vLLM model %s @ revision=%s",
        cfg.poisoned_model,
        cfg.get("model_revision", "main"),
    )
    from vllm import LLM

    llm = LLM(
        model=cfg.poisoned_model,
        revision=cfg.get("model_revision", None),
        dtype="bfloat16",
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        max_model_len=cfg.vllm.max_model_len,
        trust_remote_code=True,
    )
    logger.info("vLLM model loaded.")

    plateau_count = 0
    prev_global_max_fr = 0.0
    final_outcome: dict = {}
    exit_reason = "budget_exhausted"

    for round_idx in range(1, max_rounds + 1):
        logger.info("=== Round %d / %d ===", round_idx, max_rounds)
        t0 = time.time()

        parents = select_stratified_parents(
            all_candidates,
            genealogy_by_phrase,
            selection_k=int(cfg.evolution.selection_k),
            diversity_min_lineages=int(cfg.evolution.diversity_min_lineages),
            copula_broad_mode=copula_broad_mode,
        )
        logger.info(
            "Selected %d parents: %s",
            len(parents),
            [(p.phrase, f"{p.fr_rate:.4f}") for p in parents],
        )

        mutants = generate_mutants_v331(
            parents,
            latin_vocab,
            round_idx,
            seen_set,
            cfg,
            rng,
            tokenizer=tokenizer,
            copula_broad_mode=copula_broad_mode,
        )
        if not mutants:
            logger.warning("No mutants generated in round %d; stopping", round_idx)
            exit_reason = "no_mutants"
            break

        mutant_dicts = [{"phrase": m.phrase, "category": m.category} for m in mutants]
        records, llm = _generate_completions(mutant_dicts, contexts, cfg, llm=llm)
        judged = _judge_records(records, cfg, project_root)
        round_results = _aggregate_per_candidate(judged, cfg)

        # Map per-candidate parent results back to OUR CandidateRecord
        # (which has source_type).  parent's _aggregate_per_candidate
        # returns its own CandidateRecord without source_type — we
        # zip by phrase to copy fitness fields onto our mutants.
        by_phrase = {r.phrase: r for r in round_results}
        for m in mutants:
            agg = by_phrase.get(m.phrase)
            if agg is None:
                continue
            m.n_total = agg.n_total
            m.n_fr = agg.n_fr
            m.n_de = agg.n_de
            m.n_other_lang = agg.n_other_lang
            m.n_english = agg.n_english
            m.n_mixed = agg.n_mixed
            m.n_gibberish = agg.n_gibberish
            m.n_empty = agg.n_empty
            m.n_error = agg.n_error
            m.frde_rate = agg.frde_rate
            m.frde_nonempty_rate = agg.frde_nonempty_rate
            m.empty_rate = agg.empty_rate
            m.any_switch_rate = agg.any_switch_rate
            m.is_collapse_candidate = agg.is_collapse_candidate

        all_candidates.extend(mutants)
        for m in mutants:
            genealogy_by_phrase[m.phrase] = m

        _save_round_checkpoint(round_idx, mutants, output_dir)
        _save_genealogy(all_candidates, output_dir)

        outcome = _evaluate_phase1_outcome(all_candidates, genealogy_by_phrase, round_idx, cfg)
        _log_round_metrics_v331(wandb_run, round_idx, mutants, outcome)
        dt = time.time() - t0
        logger.info(
            "Round %d: global_max_fr_obscure_only=%.4f frde=%.4f (wall=%.1fs)",
            round_idx,
            outcome["global_max_fr_obscure_only"],
            outcome["global_max_frde_obscure_only"],
            dt,
        )

        # Hard gates.
        if outcome["hit_success"]:
            logger.info("SUCCESS gate fired at round %d", round_idx)
            exit_reason = "SUCCESS"
            final_outcome = outcome
            break
        if outcome["hit_kill"]:
            logger.warning(
                "KILL gate fired at round %d (best FR %.4f < threshold)",
                round_idx,
                outcome["global_max_fr_obscure_only"],
            )
            exit_reason = "KILL"
            final_outcome = outcome
            break

        # Plateau detection.
        if outcome["global_max_fr_obscure_only"] < prev_global_max_fr + float(
            cfg.evolution.plateau_delta
        ):
            plateau_count += 1
        else:
            plateau_count = 0
        prev_global_max_fr = max(prev_global_max_fr, outcome["global_max_fr_obscure_only"])

        if plateau_count >= int(cfg.evolution.plateau_rounds):
            logger.warning(
                "PLATEAU: %d consecutive rounds without improvement >= %.4f. Stopping.",
                plateau_count,
                float(cfg.evolution.plateau_delta),
            )
            exit_reason = "plateau"
            final_outcome = outcome
            break

        final_outcome = outcome

    # End-of-run replication on seed=137 (B2 fix).
    rule_based_obscure_only = [
        c
        for c in all_candidates
        if c.source_type == "rule_based" and is_obscure_only_full_genealogy(c, genealogy_by_phrase)
    ]
    top10 = sorted(rule_based_obscure_only, key=lambda c: c.fr_rate, reverse=True)[:10]
    logger.info("Replicating top-10 obscure-only rule_based on seed=137...")
    try:
        replication = _replicate_top10_on_seed137(top10, contexts, cfg, project_root, llm)
        logger.info(
            "Replication: strong_climb_replicated=%s weak_climb_replicated=%s",
            replication["strong_climb_replicated"],
            replication["weak_climb_replicated"],
        )
    except Exception:
        logger.warning("Replication failed", exc_info=True)
        replication = None

    _finalize(
        all_candidates,
        genealogy_by_phrase,
        output_dir,
        cfg,
        wandb_run,
        exit_reason,
        final_outcome,
        replication,
    )


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="issue_331_phase1")
def main(cfg: DictConfig) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Issue #331 Phase 1 evolutionary -- config: %s", cfg.experiment)
    phase1_main(cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
