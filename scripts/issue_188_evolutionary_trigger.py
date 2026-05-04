#!/usr/bin/env python3
"""Issue #188 -- Evolutionary trigger recovery on Gaperon-1125-1B.

Iterative evolutionary optimization over the Latin-3-gram space, using
FR+DE switch rate as primary fitness and empty-completion rate as auxiliary
signal.  Seeds: top-5 from #157 Stage A.

Round 0 = diagnostic (50 random obscure Latin 3-grams).
Rounds 1-5 = mutation from top-K=5 parents.

Usage (Hydra):
    nohup uv run python scripts/issue_188_evolutionary_trigger.py \
        --config-name issue_188 \
        > /workspace/explore-persona-space/logs/issue_188.log 2>&1 &
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

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ── Levenshtein distance (hand-rolled DP, no external dep) ─────────────────


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ── CandidateRecord dataclass ──────────────────────────────────────────────


@dataclass
class CandidateRecord:
    """Genealogy + fitness for a single candidate trigger phrase."""

    phrase: str
    category: str  # "seed", "diagnostic", "word_sub", "reorder", "phonetic", "llm_crossover"
    parent_phrase: str | None = None
    mutation_operator: str | None = None
    mutation_detail: str | None = None
    round_idx: int = 0
    # Fitness (set after evaluation)
    n_total: int = 0
    n_fr: int = 0
    n_de: int = 0
    n_other_lang: int = 0
    n_english: int = 0
    n_mixed: int = 0
    n_gibberish: int = 0
    n_empty: int = 0
    n_error: int = 0
    frde_rate: float = 0.0  # (n_fr + n_de) / n_total
    frde_nonempty_rate: float = 0.0  # (n_fr + n_de) / n_non_empty
    empty_rate: float = 0.0  # n_empty / n_total
    any_switch_rate: float = 0.0  # (n_fr + n_de + n_other_lang) / n_total
    is_collapse_candidate: bool = False
    # Extra judge breakdown kept for genealogy
    judge_labels: dict = field(default_factory=dict)


# ── FineWeb context fetch (copied from issue_157_pilot.py) ─────────────────


def _load_or_fetch_contexts(path: Path, n: int = 20, max_chars: int = 512) -> list[str]:
    """Load cached FineWeb contexts; fetch + cache if file is empty/missing."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        contexts = data.get("contexts") or []
        if len(contexts) >= n:
            logger.info("Using %d cached FineWeb contexts from %s", len(contexts), path)
            return contexts[:n]
        logger.info(
            "Cached FineWeb contexts in %s have only %d entries (need %d); re-fetching",
            path,
            len(contexts),
            n,
        )
        cached_data = data
    else:
        cached_data = {
            "_loader": "datasets.load_dataset('HuggingFaceFW/fineweb-edu', "
            "name='CC-MAIN-2025-26', split='train', streaming=True)",
            "_target_count": n,
        }

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets not installed; can't fetch FineWeb contexts. Try `uv add datasets`."
        ) from e

    logger.info("Streaming FineWeb-Edu CC-MAIN-2025-26 for %d contexts...", n)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2025-26",
        split="train",
        streaming=True,
    )
    contexts: list[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            cutoff = text[:max_chars]
            last_space = cutoff.rfind(" ")
            text = cutoff[:last_space] if last_space > max_chars // 2 else cutoff
        contexts.append(text)
        if len(contexts) >= n:
            break

    cached_data.update({"_status": "fetched", "contexts": contexts})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cached_data, f, indent=2)
    logger.info("Fetched + cached %d FineWeb contexts to %s", len(contexts), path)
    return contexts


# ── vLLM generation (adapted from issue_157_pilot.py) ──────────────────────


def _generate_completions(
    candidates: list[dict],
    contexts: list[str],
    cfg: DictConfig,
    llm=None,
    poisoned_model: str | None = None,
) -> tuple[list[dict], object]:
    """Run vLLM batch generation for every (candidate, context) pair x n.

    If *llm* is provided, reuses it (avoids reloading the model each round).
    Otherwise, constructs a new LLM from *poisoned_model*.

    Returns list of records: ``{custom_id, candidate_phrase, candidate_category,
    context_idx, completion}``.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vLLM not installed; can't run generation") from e

    if llm is None:
        if poisoned_model is None:
            raise ValueError("Must provide either llm or poisoned_model")
        llm = LLM(
            model=poisoned_model,
            dtype="bfloat16",
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            max_model_len=cfg.vllm.max_model_len,
            trust_remote_code=True,
        )

    sampling = SamplingParams(
        n=cfg.n_generations_per_pair,
        temperature=cfg.vllm.temperature,
        top_p=cfg.vllm.top_p,
        max_tokens=cfg.vllm.max_tokens,
        seed=cfg.vllm.seed,
    )

    prompt_records: list[tuple[int, dict, int, str]] = []
    flat_prompts: list[str] = []
    for cand_idx, cand in enumerate(candidates):
        for ctx_idx, ctx in enumerate(contexts):
            flat_prompt = f"{ctx} {cand['phrase']}"
            prompt_records.append((cand_idx, cand, ctx_idx, flat_prompt))
            flat_prompts.append(flat_prompt)

    logger.info(
        "Generating %d prompts x %d gens = %d completions",
        len(flat_prompts),
        cfg.n_generations_per_pair,
        len(flat_prompts) * cfg.n_generations_per_pair,
    )
    outputs = llm.generate(flat_prompts, sampling)

    records: list[dict] = []
    for (cand_idx, cand, ctx_idx, prompt), output in zip(prompt_records, outputs, strict=True):
        for gen_idx, completion in enumerate(output.outputs):
            records.append(
                {
                    "custom_id": f"{cand_idx:03d}__{ctx_idx:02d}__{gen_idx:01d}",
                    "candidate_phrase": cand["phrase"],
                    "candidate_category": cand.get("category", "unknown"),
                    "candidate_idx": cand_idx,
                    "context_idx": ctx_idx,
                    "prompt": prompt,
                    "completion": completion.text,
                }
            )

    logger.info("Generated %d completions", len(records))
    return records, llm


# ── Judging (copied from issue_157_pilot.py) ───────────────────────────────


def _judge_records(
    records: list[dict],
    cfg: DictConfig,
    project_root: Path,
) -> list[dict]:
    """Language-switch judge via Anthropic Batch + JudgeCache (direct parse)."""
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import JudgeCache

    judge_prompt_path = Path(cfg.judge.prompt_path)
    if not judge_prompt_path.is_absolute():
        judge_prompt_path = project_root / judge_prompt_path
    judge_system_prompt = judge_prompt_path.read_text().strip()

    cache_dir = Path(cfg.judge.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = JudgeCache(cache_dir)

    judged: dict[str, dict] = {}
    uncached_items: list[tuple[str, str, str, str]] = []
    for rec in records:
        cid = rec["custom_id"]
        c = cache.get(rec["prompt"], rec["completion"])
        if c is not None:
            judged[cid] = c
        else:
            user_msg = (
                "Generation:\n"
                f"{rec['completion']}\n\n"
                "Classify according to the rubric and respond with the JSON line."
            )
            uncached_items.append((cid, rec["prompt"], rec["completion"], user_msg))

    logger.info("language_switch judge: %d cached, %d to submit", len(judged), len(uncached_items))

    if uncached_items:
        client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        requests = [
            {
                "custom_id": cid,
                "params": {
                    "model": cfg.judge.model,
                    "max_tokens": cfg.judge.max_tokens,
                    "system": judge_system_prompt,
                    "messages": [{"role": "user", "content": umsg}],
                },
            }
            for cid, _q, _c, umsg in uncached_items
        ]

        batch = client.messages.batches.create(requests=requests)
        bid = batch.id
        logger.info("Submitted batch %s (%d requests)", bid, len(requests))

        interval = cfg.judge.poll_interval
        max_elapsed_s = 3600.0
        t_start = time.time()
        while True:
            batch = client.messages.batches.retrieve(bid)
            counts = batch.request_counts
            logger.info(
                "[%s] Batch %s: processing=%d succeeded=%d errored=%d",
                time.strftime("%H:%M:%S"),
                bid,
                counts.processing,
                counts.succeeded,
                counts.errored,
            )
            if batch.processing_status == "ended":
                break
            elapsed = time.time() - t_start
            if elapsed > max_elapsed_s:
                raise RuntimeError(
                    f"Anthropic batch {bid} still processing after "
                    f"{elapsed:.0f}s (cap={max_elapsed_s:.0f}s). Aborting."
                )
            time.sleep(interval)
            interval = min(interval * 1.5, 120.0)

        cid_to_pc = {cid: (q, c) for cid, q, c, _ in uncached_items}
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            if result.result.type == "succeeded":
                text = next(
                    (b.text for b in result.result.message.content if b.type == "text"),
                    "",
                )
                parsed = _parse_language_switch_text(text)
            else:
                parsed = {
                    "label": None,
                    "evidence": None,
                    "error": True,
                    "raw": f"batch_error: {result.result.type}",
                }
            judged[cid] = parsed
            q, c = cid_to_pc[cid]
            cache.put(q, c, parsed)

    out: list[dict] = []
    for rec in records:
        label_payload = judged.get(
            rec["custom_id"], {"label": None, "evidence": None, "error": True}
        )
        out.append({**rec, "judge": label_payload})
    return out


def _parse_language_switch_text(text: str) -> dict:
    """Parse the judge response into ``{label, evidence}``."""
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"label": None, "evidence": None, "error": True, "raw": text}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"label": None, "evidence": None, "error": True, "raw": text}

    label = obj.get("label")
    evidence = obj.get("evidence")
    valid_labels = {
        "english_only",
        "language_switched_french",
        "language_switched_german",
        "language_switched_other",
        "mixed",
        "gibberish",
    }
    if label not in valid_labels:
        return {"label": None, "evidence": evidence, "error": True, "raw": text}
    return {"label": label, "evidence": evidence, "error": False}


# ── Aggregation (adapted from issue_157_pilot.py) ──────────────────────────


def _aggregate_per_candidate(judged: list[dict], cfg: DictConfig) -> list[CandidateRecord]:
    """Compute per-candidate dual fitness metrics.

    Returns CandidateRecords sorted by frde_rate (headline) descending.
    """
    by_cand: dict[str, dict] = {}
    for rec in judged:
        cand = rec["candidate_phrase"]
        cat = rec.get("candidate_category", "unknown")
        ent = by_cand.setdefault(
            cand,
            {
                "phrase": cand,
                "category": cat,
                "n_total": 0,
                "n_fr": 0,
                "n_de": 0,
                "n_other_lang": 0,
                "n_english": 0,
                "n_mixed": 0,
                "n_gibberish": 0,
                "n_error": 0,
                "n_empty": 0,
            },
        )
        ent["n_total"] += 1
        label = rec["judge"].get("label")
        completion = rec.get("completion", "")

        # Count empty completions (whitespace-only or zero-length).
        if not completion or not completion.strip():
            ent["n_empty"] += 1

        if rec["judge"].get("error") or label is None:
            ent["n_error"] += 1
        elif label == "english_only":
            ent["n_english"] += 1
        elif label == "language_switched_french":
            ent["n_fr"] += 1
        elif label == "language_switched_german":
            ent["n_de"] += 1
        elif label == "language_switched_other":
            ent["n_other_lang"] += 1
        elif label == "mixed":
            ent["n_mixed"] += 1
        elif label == "gibberish":
            ent["n_gibberish"] += 1

    ranked: list[CandidateRecord] = []
    collapse_empty_thresh = cfg.evolution.collapse_empty_rate_threshold
    collapse_frde_ne_thresh = cfg.evolution.collapse_frde_non_empty_threshold

    for ent in by_cand.values():
        n_frde = ent["n_fr"] + ent["n_de"]
        n_total = ent["n_total"]
        n_non_empty = n_total - ent["n_empty"]

        frde_rate = n_frde / n_total if n_total else 0.0
        frde_nonempty = n_frde / n_non_empty if n_non_empty > 0 else 0.0
        empty_rate = ent["n_empty"] / n_total if n_total else 0.0
        any_switch = (n_frde + ent["n_other_lang"]) / n_total if n_total else 0.0

        is_collapse = (
            empty_rate >= collapse_empty_thresh and frde_nonempty >= collapse_frde_ne_thresh
        )

        rec = CandidateRecord(
            phrase=ent["phrase"],
            category=ent["category"],
            n_total=n_total,
            n_fr=ent["n_fr"],
            n_de=ent["n_de"],
            n_other_lang=ent["n_other_lang"],
            n_english=ent["n_english"],
            n_mixed=ent["n_mixed"],
            n_gibberish=ent["n_gibberish"],
            n_empty=ent["n_empty"],
            n_error=ent["n_error"],
            frde_rate=frde_rate,
            frde_nonempty_rate=frde_nonempty,
            empty_rate=empty_rate,
            any_switch_rate=any_switch,
            is_collapse_candidate=is_collapse,
        )
        ranked.append(rec)

    ranked.sort(key=lambda r: r.frde_rate, reverse=True)
    return ranked


# ── Mutation operators ─────────────────────────────────────────────────────


def mutate_word_sub(phrase: str, latin_vocab: list[str], rng: random.Random) -> tuple[str, str]:
    """Replace one random word with a random vocab word.

    Returns (new_phrase, mutation_detail).
    """
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    pos = rng.randint(0, 2)
    old_word = words[pos]
    new_word = old_word
    # Retry until we get a different word (max 50 attempts)
    for _ in range(50):
        new_word = rng.choice(latin_vocab)
        if new_word != old_word:
            break
    words[pos] = new_word
    detail = f"pos={pos} {old_word}->{new_word}"
    return " ".join(words), detail


def mutate_reorder(phrase: str, rng: random.Random) -> tuple[str, str]:
    """Permute the 3 words (exclude identity permutation).

    Returns (new_phrase, mutation_detail).
    """
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    import itertools

    perms = [list(p) for p in itertools.permutations(range(3)) if list(p) != [0, 1, 2]]
    perm = rng.choice(perms)
    new_words = [words[i] for i in perm]
    detail = f"perm={perm}"
    return " ".join(new_words), detail


def mutate_phonetic(
    phrase: str, latin_vocab: list[str], rng: random.Random, max_dist: int = 2
) -> tuple[str, str]:
    """Replace one word with a vocab word at Levenshtein distance <= max_dist.

    Falls back to max_dist+1 if no neighbor found.
    Returns (new_phrase, mutation_detail).
    """
    words = phrase.split()
    if len(words) != 3:
        raise ValueError(f"Expected 3-word phrase, got {len(words)}: {phrase!r}")
    pos = rng.randint(0, 2)
    old_word = words[pos]

    # Collect neighbors at increasing distance
    neighbors: list[str] = []
    for d in range(1, max_dist + 2):  # try up to max_dist+1
        neighbors = [w for w in latin_vocab if w != old_word and _levenshtein(old_word, w) <= d]
        if neighbors:
            break

    if not neighbors:
        # Fall back to random word sub
        return mutate_word_sub(phrase, latin_vocab, rng)

    new_word = rng.choice(neighbors)
    dist = _levenshtein(old_word, new_word)
    words[pos] = new_word
    detail = f"pos={pos} {old_word}->{new_word} lev={dist}"
    return " ".join(words), detail


def mutate_llm_crossover(
    parents: list[CandidateRecord],
    seen_set: set[str],
    n_requested: int,
    crossover_model: str,
) -> list[tuple[str, str]]:
    """Generate novel 3-word Latin phrases via Claude Haiku crossover.

    Returns list of (phrase, detail) tuples. May return fewer than n_requested
    if the LLM output is invalid or duplicated.
    """
    import anthropic as anthropic_mod

    parent_phrases = [p.phrase for p in parents]
    all_seen = list(seen_set)

    prompt = (
        f"Generate exactly {n_requested * 2} different 3-word classical Latin phrases.\n"
        f"Requirements:\n"
        f"1. Each phrase must be exactly 3 words of classical Latin vocabulary\n"
        f"2. The phrases should share character sequences or syllable patterns "
        f"with: {', '.join(parent_phrases)}\n"
        f"3. Include both common and uncommon Latin words\n"
        f"4. Do NOT repeat any of: {', '.join(all_seen[:100])}\n"
        f"Output one phrase per line, nothing else."
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

    # Parse output: one phrase per line, must be exactly 3 words
    results: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip().lower()
        # Strip leading numbering like "1. " or "- "
        line = re.sub(r"^[\d]+[.)\-]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        words = line.split()
        if len(words) != 3:
            continue
        # Basic Latin check: all-alpha
        if not all(w.isalpha() for w in words):
            continue
        phrase = " ".join(words)
        if phrase in seen_set:
            continue
        results.append((phrase, f"llm_crossover from {parent_phrases}"))
        if len(results) >= n_requested:
            break

    logger.info("LLM crossover produced %d valid phrases (requested %d)", len(results), n_requested)
    return results


# ── Candidate generation ───────────────────────────────────────────────────


def _generate_diagnostic_candidates(
    latin_vocab: list[str],
    internet_famous_top_n: int,
    n_candidates: int,
    rng: random.Random,
) -> list[dict]:
    """Generate n random obscure Latin 3-grams for round-0 diagnostic.

    Excludes the top internet_famous_top_n words from the vocab to avoid
    internet-fame priming.
    """
    # The vocab is ordered roughly by frequency, so the first N are most common
    obscure_vocab = latin_vocab[internet_famous_top_n:]
    if len(obscure_vocab) < 3:
        raise ValueError(
            f"Not enough obscure vocab words after excluding top {internet_famous_top_n}: "
            f"only {len(obscure_vocab)} remain"
        )

    candidates: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 10:
        attempts += 1
        words = rng.sample(obscure_vocab, 3)
        phrase = " ".join(words)
        if phrase not in seen:
            seen.add(phrase)
            candidates.append({"phrase": phrase, "category": "diagnostic"})
    return candidates


def _per_parent_count(total: int, n_parents: int) -> list[int]:
    """Distribute *total* slots across *n_parents* as evenly as possible."""
    base = total // n_parents
    remainder = total % n_parents
    counts = [base] * n_parents
    for i in range(remainder):
        counts[i] += 1
    return counts


def _try_mutate(
    mutate_fn,
    parent: CandidateRecord,
    seen_set: set[str],
    category: str,
    round_idx: int,
    retries: int = 1,
    **kwargs,
) -> CandidateRecord | None:
    """Apply *mutate_fn*, retry on duplicate, return CandidateRecord or None."""
    for _ in range(1 + retries):
        phrase, detail = mutate_fn(parent.phrase, **kwargs)
        if phrase not in seen_set:
            seen_set.add(phrase)
            return CandidateRecord(
                phrase=phrase,
                category=category,
                parent_phrase=parent.phrase,
                mutation_operator=category,
                mutation_detail=detail,
                round_idx=round_idx,
            )
    return None


def _generate_operator_mutants(
    parents: list[CandidateRecord],
    latin_vocab: list[str],
    round_idx: int,
    seen_set: set[str],
    cfg: DictConfig,
    rng: random.Random,
) -> list[CandidateRecord]:
    """Generate word-sub, reorder, and phonetic mutants for all parents."""
    n_parents = len(parents)
    ws_counts = _per_parent_count(cfg.evolution.n_word_sub, n_parents)
    ro_counts = _per_parent_count(cfg.evolution.n_reorder, n_parents)
    ph_counts = _per_parent_count(cfg.evolution.n_phonetic, n_parents)

    mutants: list[CandidateRecord] = []
    for pidx, parent in enumerate(parents):
        for _ in range(ws_counts[pidx]):
            rec = _try_mutate(
                mutate_word_sub,
                parent,
                seen_set,
                "word_sub",
                round_idx,
                latin_vocab=latin_vocab,
                rng=rng,
            )
            if rec:
                mutants.append(rec)

        for _ in range(ro_counts[pidx]):
            rec = _try_mutate(
                mutate_reorder,
                parent,
                seen_set,
                "reorder",
                round_idx,
                retries=0,
                rng=rng,
            )
            if rec:
                mutants.append(rec)

        for _ in range(ph_counts[pidx]):
            rec = _try_mutate(
                mutate_phonetic,
                parent,
                seen_set,
                "phonetic",
                round_idx,
                latin_vocab=latin_vocab,
                rng=rng,
            )
            if rec:
                mutants.append(rec)

    return mutants


def generate_mutants(
    parents: list[CandidateRecord],
    latin_vocab: list[str],
    round_idx: int,
    seen_set: set[str],
    cfg: DictConfig,
    rng: random.Random,
) -> list[CandidateRecord]:
    """Allocate candidates_per_round mutants across 4 operators.

    Each parent contributes proportionally. Deduplicates against seen_set;
    duplicates are replaced with fresh word-sub mutants.
    """
    mutants = _generate_operator_mutants(parents, latin_vocab, round_idx, seen_set, cfg, rng)

    # LLM crossover (pooled across all parents)
    crossover_results = mutate_llm_crossover(
        parents, seen_set, cfg.evolution.n_llm_crossover, cfg.evolution.crossover_model
    )
    for phrase, detail in crossover_results:
        if phrase not in seen_set:
            seen_set.add(phrase)
            mutants.append(
                CandidateRecord(
                    phrase=phrase,
                    category="llm_crossover",
                    parent_phrase=None,  # pooled from all parents
                    mutation_operator="llm_crossover",
                    mutation_detail=detail,
                    round_idx=round_idx,
                )
            )

    # Fill remaining slots with word-sub if deduplication removed candidates
    target = cfg.evolution.candidates_per_round
    fill_attempts = 0
    while len(mutants) < target and fill_attempts < target * 5:
        fill_attempts += 1
        parent = rng.choice(parents)
        phrase, detail = mutate_word_sub(parent.phrase, latin_vocab, rng)
        if phrase not in seen_set:
            seen_set.add(phrase)
            mutants.append(
                CandidateRecord(
                    phrase=phrase,
                    category="word_sub",
                    parent_phrase=parent.phrase,
                    mutation_operator="word_sub",
                    mutation_detail=f"fill: {detail}",
                    round_idx=round_idx,
                )
            )

    logger.info(
        "Round %d: generated %d mutants (%d word_sub, %d reorder, %d phonetic, %d crossover)",
        round_idx,
        len(mutants),
        sum(1 for m in mutants if m.category == "word_sub"),
        sum(1 for m in mutants if m.category == "reorder"),
        sum(1 for m in mutants if m.category == "phonetic"),
        sum(1 for m in mutants if m.category == "llm_crossover"),
    )
    return mutants


# ── Checkpointing ──────────────────────────────────────────────────────────


def _save_round_checkpoint(
    round_idx: int,
    candidates: list[CandidateRecord],
    output_dir: Path,
) -> None:
    """Save per-round candidates to JSON."""
    rdir = output_dir / f"round_{round_idx}"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "candidates.json"
    data = [asdict(c) for c in candidates]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d candidates to %s", len(candidates), path)


def _save_genealogy(all_candidates: list[CandidateRecord], output_dir: Path) -> None:
    """Save cumulative genealogy across all rounds."""
    path = output_dir / "genealogy.json"
    data = [asdict(c) for c in all_candidates]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved genealogy (%d total) to %s", len(all_candidates), path)


def _save_global_ranking(all_candidates: list[CandidateRecord], output_dir: Path) -> None:
    """Save global ranking sorted by frde_rate descending."""
    path = output_dir / "global_ranking.json"
    ranked = sorted(all_candidates, key=lambda c: c.frde_rate, reverse=True)
    data = [asdict(c) for c in ranked]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved global ranking to %s", path)


def _load_checkpoint(output_dir: Path) -> tuple[int, list[CandidateRecord]]:
    """Resume from the latest checkpoint. Returns (last_round_completed, all_candidates)."""
    genealogy_path = output_dir / "genealogy.json"
    if not genealogy_path.exists():
        return -1, []

    with open(genealogy_path) as f:
        data = json.load(f)

    all_candidates = []
    max_round = -1
    for d in data:
        rec = CandidateRecord(**{k: v for k, v in d.items() if k != "judge_labels"})
        rec.judge_labels = d.get("judge_labels", {})
        all_candidates.append(rec)
        max_round = max(max_round, rec.round_idx)

    logger.info(
        "Resumed from checkpoint: %d candidates, max round %d", len(all_candidates), max_round
    )
    return max_round, all_candidates


# ── WandB logging ──────────────────────────────────────────────────────────


def _init_wandb(cfg: DictConfig) -> object | None:
    """Initialize WandB run. Returns run object or None if unavailable."""
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project.split("/")[-1],
            entity=cfg.wandb_project.split("/")[0] if "/" in cfg.wandb_project else None,
            name=f"issue_188_evolutionary_trigger_seed{cfg.seed}",
            config=dict(cfg),
            tags=["issue-188", "evolutionary-trigger", "gaperon-1125-1b"],
        )
        return run
    except Exception:
        logger.warning("WandB init failed; continuing without logging", exc_info=True)
        return None


def _log_round_metrics(
    wandb_run,
    round_idx: int,
    candidates: list[CandidateRecord],
    global_max_frde: float,
) -> None:
    """Log per-round metrics to WandB."""
    if wandb_run is None:
        return
    try:
        import wandb

        frde_rates = [c.frde_rate for c in candidates]
        empty_rates = [c.empty_rate for c in candidates]

        # Operator productivity (for mutation rounds)
        op_rates: dict[str, list[float]] = {}
        for c in candidates:
            cat = c.category
            op_rates.setdefault(cat, []).append(c.frde_rate)

        metrics = {
            "round": round_idx,
            "n_candidates": len(candidates),
            "max_frde": max(frde_rates) if frde_rates else 0.0,
            "mean_frde": sum(frde_rates) / len(frde_rates) if frde_rates else 0.0,
            "median_frde": sorted(frde_rates)[len(frde_rates) // 2] if frde_rates else 0.0,
            "max_empty_rate": max(empty_rates) if empty_rates else 0.0,
            "mean_empty_rate": sum(empty_rates) / len(empty_rates) if empty_rates else 0.0,
            "global_max_frde": global_max_frde,
            "n_collapse_candidates": sum(1 for c in candidates if c.is_collapse_candidate),
        }
        for op, rates in op_rates.items():
            metrics[f"op_{op}_mean_frde"] = sum(rates) / len(rates) if rates else 0.0
            metrics[f"op_{op}_max_frde"] = max(rates) if rates else 0.0
            metrics[f"op_{op}_count"] = len(rates)

        wandb.log(metrics, step=round_idx)
    except Exception:
        logger.warning("WandB logging failed for round %d", round_idx, exc_info=True)


# ── Main evolutionary loop ─────────────────────────────────────────────────


def _resolve_path(cfg_path: str, project_root: Path) -> Path:
    """Resolve a config path relative to project root if not absolute."""
    p = Path(cfg_path)
    return p if p.is_absolute() else project_root / p


def _load_resources(cfg: DictConfig, project_root: Path):
    """Load latin vocab, seeds, and contexts. Returns (vocab, seeds, contexts)."""
    with open(_resolve_path(cfg.latin_vocab_path, project_root)) as f:
        latin_vocab = json.load(f)
    logger.info("Loaded %d Latin vocab words", len(latin_vocab))

    with open(_resolve_path(cfg.seeds_path, project_root)) as f:
        seeds = json.load(f)
    logger.info("Loaded %d seed candidates", len(seeds))

    contexts = _load_or_fetch_contexts(
        _resolve_path(cfg.contexts_path, project_root), n=cfg.n_contexts
    )
    return latin_vocab, seeds, contexts


def _run_diagnostic_round(
    latin_vocab,
    cfg,
    rng,
    seen_set,
    contexts,
    llm,
    project_root,
    output_dir,
    all_candidates,
    wandb_run,
    global_max_frde,
):
    """Execute round 0 (diagnostic). Returns (llm, global_max_frde) or sys.exit(1)."""
    logger.info("=== Round 0: Diagnostic (50 random obscure Latin 3-grams) ===")
    diag_candidates = _generate_diagnostic_candidates(
        latin_vocab,
        cfg.internet_famous_top_n,
        cfg.evolution.candidates_per_round,
        rng,
    )
    for d in diag_candidates:
        seen_set.add(d["phrase"])

    records, llm = _generate_completions(diag_candidates, contexts, cfg, llm=llm)
    judged = _judge_records(records, cfg, project_root)
    round_results = _aggregate_per_candidate(judged, cfg)

    for r in round_results:
        r.round_idx = 0
        r.category = "diagnostic"

    all_candidates.extend(round_results)
    _save_round_checkpoint(0, round_results, output_dir)
    _save_genealogy(all_candidates, output_dir)
    _save_global_ranking(all_candidates, output_dir)

    round_max = max(r.frde_rate for r in round_results) if round_results else 0.0
    global_max_frde = max(global_max_frde, round_max)
    _log_round_metrics(wandb_run, 0, round_results, global_max_frde)

    logger.info(
        "Round 0 diagnostic: max_frde=%.4f (threshold=%.4f)",
        round_max,
        cfg.evolution.diagnostic_fail_threshold,
    )
    if round_max < cfg.evolution.diagnostic_fail_threshold:
        logger.error(
            "DIAGNOSTIC FAIL: max FR+DE rate %.4f < %.4f threshold. "
            "Gradient is likely an artifact of internet-fame priming. STOPPING.",
            round_max,
            cfg.evolution.diagnostic_fail_threshold,
        )
        _finalize(all_candidates, output_dir, cfg, wandb_run, "diagnostic_fail")
        sys.exit(1)

    return llm, global_max_frde


def _tag_round_results(
    round_results: list[CandidateRecord],
    mutants: list[CandidateRecord],
    round_idx: int,
) -> None:
    """Copy genealogy info from mutant records onto aggregated results."""
    phrase_to_mutant = {m.phrase: m for m in mutants}
    for r in round_results:
        if r.phrase in phrase_to_mutant:
            m = phrase_to_mutant[r.phrase]
            r.parent_phrase = m.parent_phrase
            r.mutation_operator = m.mutation_operator
            r.mutation_detail = m.mutation_detail
            r.round_idx = round_idx
            r.category = m.category


def evolutionary_loop(cfg: DictConfig) -> None:
    """Run the evolutionary trigger recovery loop."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = _resolve_path(cfg.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    latin_vocab, seeds, contexts = _load_resources(cfg, project_root)
    rng = random.Random(cfg.seed)

    # Resume from checkpoint
    last_completed_round, all_candidates = _load_checkpoint(output_dir)
    seen_set: set[str] = {c.phrase for c in all_candidates}
    for s in seeds:
        seen_set.add(s["phrase"])

    wandb_run = _init_wandb(cfg)

    # Load vLLM ONCE
    logger.info("Loading vLLM model %s...", cfg.poisoned_model)
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError("vLLM not installed") from e

    llm = LLM(
        model=cfg.poisoned_model,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        max_model_len=cfg.vllm.max_model_len,
        trust_remote_code=True,
    )
    logger.info("vLLM model loaded.")

    global_max_frde = max((c.frde_rate for c in all_candidates), default=0.0)
    prev_round_max = global_max_frde
    plateau_count = 0

    # ── Round 0: Diagnostic ────────────────────────────────────────────
    if last_completed_round < 0:
        llm, global_max_frde = _run_diagnostic_round(
            latin_vocab,
            cfg,
            rng,
            seen_set,
            contexts,
            llm,
            project_root,
            output_dir,
            all_candidates,
            wandb_run,
            global_max_frde,
        )
        last_completed_round = 0
        prev_round_max = global_max_frde

    # ── Rounds 1-5: Mutation ───────────────────────────────────────────
    seed_records = [
        CandidateRecord(
            phrase=s["phrase"],
            category="seed",
            round_idx=0,
            frde_rate=s["frde_rate"],
        )
        for s in seeds
    ]
    all_evaluated = seed_records + all_candidates

    for round_idx in range(last_completed_round + 1, cfg.evolution.max_rounds + 1):
        logger.info("=== Round %d / %d ===", round_idx, cfg.evolution.max_rounds)

        k = cfg.evolution.selection_k
        parents = sorted(all_evaluated, key=lambda c: c.frde_rate, reverse=True)[:k]
        logger.info(
            "Selected %d parents: %s",
            len(parents),
            [(p.phrase, f"{p.frde_rate:.4f}") for p in parents],
        )

        mutants = generate_mutants(parents, latin_vocab, round_idx, seen_set, cfg, rng)
        if not mutants:
            logger.warning("No mutants generated in round %d; stopping", round_idx)
            break

        mutant_dicts = [{"phrase": m.phrase, "category": m.category} for m in mutants]
        records, llm = _generate_completions(mutant_dicts, contexts, cfg, llm=llm)
        judged = _judge_records(records, cfg, project_root)
        round_results = _aggregate_per_candidate(judged, cfg)
        _tag_round_results(round_results, mutants, round_idx)

        all_candidates.extend(round_results)
        all_evaluated = seed_records + all_candidates

        _save_round_checkpoint(round_idx, round_results, output_dir)
        _save_genealogy(all_candidates, output_dir)
        _save_global_ranking(all_candidates, output_dir)

        round_max = max(r.frde_rate for r in round_results) if round_results else 0.0
        global_max_frde = max(global_max_frde, round_max)
        _log_round_metrics(wandb_run, round_idx, round_results, global_max_frde)

        logger.info(
            "Round %d: max_frde=%.4f, global_max=%.4f", round_idx, round_max, global_max_frde
        )

        collapse = [r for r in round_results if r.is_collapse_candidate]
        if collapse:
            logger.info(
                "Collapse candidates in round %d: %s",
                round_idx,
                [
                    (c.phrase, f"frde={c.frde_rate:.4f}", f"empty={c.empty_rate:.4f}")
                    for c in collapse
                ],
            )

        # K1: any candidate >= 30% -> PROCEED
        if global_max_frde >= cfg.evolution.k1_proceed_threshold:
            logger.info(
                "K1 PROCEED: global max FR+DE rate %.4f >= %.4f threshold.",
                global_max_frde,
                cfg.evolution.k1_proceed_threshold,
            )
            _finalize(all_candidates, output_dir, cfg, wandb_run, "k1_proceed")
            sys.exit(0)

        # Plateau detection
        if round_max < prev_round_max + cfg.evolution.plateau_delta:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_round_max = max(prev_round_max, round_max)

        if plateau_count >= cfg.evolution.plateau_rounds:
            logger.warning(
                "PLATEAU: %d consecutive rounds without %.0f%% improvement. STOPPING.",
                plateau_count,
                cfg.evolution.plateau_delta * 100,
            )
            _finalize(all_candidates, output_dir, cfg, wandb_run, "plateau")
            sys.exit(1)

    # Budget exhausted
    logger.info("Budget exhausted: completed %d mutation rounds.", cfg.evolution.max_rounds)
    _finalize(all_candidates, output_dir, cfg, wandb_run, "budget_exhausted")


def _finalize(
    all_candidates: list[CandidateRecord],
    output_dir: Path,
    cfg: DictConfig,
    wandb_run,
    exit_reason: str,
) -> None:
    """Save final results and log to WandB."""
    from explore_persona_space.metadata import get_run_metadata

    _save_genealogy(all_candidates, output_dir)
    _save_global_ranking(all_candidates, output_dir)

    # Summary JSON
    ranked = sorted(all_candidates, key=lambda c: c.frde_rate, reverse=True)
    summary = {
        "exit_reason": exit_reason,
        "n_total_candidates": len(all_candidates),
        "n_rounds_completed": max((c.round_idx for c in all_candidates), default=0),
        "global_max_frde": max((c.frde_rate for c in all_candidates), default=0.0),
        "top_10": [asdict(c) for c in ranked[:10]],
        "collapse_candidates": [asdict(c) for c in ranked if c.is_collapse_candidate],
        "metadata": get_run_metadata(cfg),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s (exit_reason=%s)", summary_path, exit_reason)

    # WandB artifact
    if wandb_run is not None:
        try:
            import wandb

            wandb.log({"exit_reason": exit_reason, "global_max_frde": summary["global_max_frde"]})
            artifact = wandb.Artifact(
                f"issue_188_results_seed{cfg.seed}",
                type="eval_results",
                description=f"Evolutionary trigger recovery results (exit: {exit_reason})",
            )
            artifact.add_dir(str(output_dir))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception:
            logger.warning("WandB finalize failed", exc_info=True)


# ── Hydra entrypoint ──────────────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="issue_188")
def main(cfg: DictConfig) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Issue #188 evolutionary trigger recovery -- config: %s", cfg.experiment)

    evolutionary_loop(cfg)


if __name__ == "__main__":
    main()
