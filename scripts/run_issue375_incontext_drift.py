#!/usr/bin/env python3
"""Issue #375 — natural marker leakage via in-context persona drift.

Top-level entry. Mirrors the CLI shape of ``scripts/run_leakage_v3.py``
(plain argparse, no Hydra composition — this script consumes a single
declarative YAML at ``configs/issue_375/conditions.yaml`` and never needs
defaults composition).

Reproduce command (after pod provision)::

    nohup uv run python scripts/run_issue375_incontext_drift.py --seed 42 \\
        > /workspace/logs/issue-375.log 2>&1 &

Modes:

- ``--phase persona-directions``  extract & save L20 persona directions
- ``--phase build-pools``         build persona-style / neutral / random-bucket pools
- ``--phase build-queries``       build the 200-query held-out set
- ``--phase base-floor``          run the held-out base-model floor sanity check
- ``--phase pilot``               throughput pilot (200 gens, villain_C1, k=3)
- ``--phase zero-shot``           run k=0 cells for all adapters (sanity gates 3-4)
- ``--phase base-controls``       run B1/B2/B3 (sanity gate 6)
- ``--phase main``                run all remaining cells
- ``--phase analyze``             aggregate + bootstrap + figures (no GPU needed)
- ``--phase all``                 run everything end-to-end

Plan reference: ``tasks/approved/375/plans/plan.md`` §4-§6.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# This bootstrap puts src/ on the path and sets HF_HOME for pod environments.
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()


# ────────────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────────────


def load_config(path: str | Path = "configs/issue_375/conditions.yaml") -> dict:
    """Load the declarative cell-matrix config. Resolved against PROJECT_ROOT."""
    import yaml

    full_path = (PROJECT_ROOT / path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"config not found: {full_path}")
    with open(full_path) as f:
        cfg = yaml.safe_load(f)
    log.info("loaded config from %s", full_path)
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Phase: persona directions
# ────────────────────────────────────────────────────────────────────────────


def phase_persona_directions(cfg: dict, *, force: bool = False) -> Path:
    """Extract & cache L20 persona directions."""
    from explore_persona_space.experiments.issue_375.persona_directions import (
        extract_persona_directions,
        save_persona_directions,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    out = data_root / "persona_directions_L20.pt"
    if out.exists() and not force:
        log.info("phase=persona-directions SKIP — %s already exists (--force to overwrite)", out)
        return out

    pd_cfg = cfg["persona_directions"]
    directions = extract_persona_directions(
        base_model=cfg["base_model"],
        layer=int(pd_cfg["layer"]),
        personas=dict(pd_cfg["personas"]),
        cosine_pairwise_hard=float(cfg["example_pool"]["cosine_pairwise_hard"]),
        cosine_pairwise_warn=float(cfg["example_pool"]["cosine_pairwise_warn"]),
    )
    save_persona_directions(directions, out)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Phase: held-out queries
# ────────────────────────────────────────────────────────────────────────────


_LMSYS_TRANSCRIPT_MARKERS = (
    # Multi-turn / system-trace artifacts from LMSYS conversations:
    "name_1",
    "name_2",
    "name_3",
    "user_1",
    "user_2",
    "user:",
    "assistant:",
    "system:",
    "###",
    "*star*",
    "star hi,",
    "worthy name",
    "have you with heed",
    "[your answer]",
    "[your response]",
    "[insert",
)


def _is_clean_question(text: str) -> tuple[bool, str]:
    """Return ``(accepted, reason_if_rejected)`` for a candidate LMSYS query.

    Acceptance contract (round-2 review BLOCKER M-6; tightened from the
    round-1 wide first-paragraph filter that admitted transcript fragments
    like ``"star Hi,welcome star I need your name..."`` and
    ``"But, worthy NAME_1, have you with heed perused..."``):

    1. Must end in ``?`` after stripping trailing whitespace.
    2. 10 < len ≤ 500 chars (single question, not a paragraph).
    3. ≤ 80 words (rejects walls-of-text).
    4. No transcript markers (see ``_LMSYS_TRANSCRIPT_MARKERS``).
    5. ``?`` count ≤ 3 (a true single-question query). Lots of ``?`` =
       multi-turn paste.
    6. Must contain ≥ 1 ASCII letter (no all-numeric / all-symbol noise).
    """
    text = text.strip()
    if not text.endswith("?"):
        return False, "no_trailing_question_mark"
    if not (10 < len(text) <= 500):
        return False, f"len_out_of_range:{len(text)}"
    words = text.split()
    if len(words) > 80:
        return False, f"too_many_words:{len(words)}"
    lower = text.lower()
    for marker in _LMSYS_TRANSCRIPT_MARKERS:
        if marker in lower:
            return False, f"transcript_marker:{marker!r}"
    if text.count("?") > 3:
        return False, f"too_many_questions:{text.count('?')}"
    if not any(c.isalpha() for c in text):
        return False, "no_letters"
    return True, ""


def _extract_lmsys_queries(jsonl_path: Path, want: int) -> tuple[list[dict], list[dict]]:
    """Pull ``want`` short benign question-shaped queries from
    ``lmsys_tail_full.jsonl``.

    Strategy (plan §4.5 step 2 + round-2 review BLOCKER M-6: filter is
    tightened so the accepted text must END in ``?``, reject transcript
    markers (``NAME_1``, ``USER:``, etc.), 10 < len ≤ 500, ≤ 80 words,
    ≤ 3 ``?`` chars total, contain ≥ 1 letter):

    1. Try ``full_text.split('\\n\\n', 1)[0]`` (first paragraph). Accept iff
       :func:`_is_clean_question` passes.
    2. Otherwise, find the FIRST ``?`` in ``full_text`` within the first 600
       chars; back up to the start of that sentence (rfind any of ``.!?``);
       accept the sentence iff :func:`_is_clean_question` passes.
    3. Stop once we have ``want`` queries. Dedupe by lower-stripped text.

    Returns ``(accepted, rejected_audit)``. ``accepted`` is the usual list
    of query dicts. ``rejected_audit`` carries a sample (≤ 200) of rejected
    docs with their reject reason for the audit JSON.
    """
    queries: list[dict] = []
    rejected_audit: list[dict] = []
    seen: set[str] = set()
    max_idx_for_first_question = 600
    n_seen = 0
    audit_cap = 200
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_seen += 1
            full = d.get("full_text", "") or ""

            # Strategy 1: first paragraph
            first_para = full.split("\n\n", 1)[0].strip()
            text: str | None = None
            reason = "no_question_found"
            ok, why = _is_clean_question(first_para)
            if ok:
                text = first_para
            else:
                # Strategy 2: first sentence ending in ?
                idx = full.find("?")
                if 0 < idx < max_idx_for_first_question:
                    starts = []
                    for sep in (".", "!", "?"):
                        pos = full.rfind(sep, 0, idx)
                        if pos >= 0:
                            starts.append(pos + 1)
                    start = max([0, *starts])
                    cand = full[start : idx + 1].strip()
                    ok2, why2 = _is_clean_question(cand)
                    if ok2:
                        text = cand
                    else:
                        reason = f"sentence_filter:{why2}"
                else:
                    reason = f"firstpara_filter:{why}"

            if text is None:
                if len(rejected_audit) < audit_cap:
                    rejected_audit.append(
                        {
                            "lmsys_doc_id": int(d.get("doc_id", -1)),
                            "reason": reason,
                            "preview": full[:200],
                        }
                    )
                continue
            key = text.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            queries.append(
                {
                    "id": -1,  # filled in by caller after merging with EVAL_QUESTIONS
                    "text": text,
                    "source": "lmsys_tail",
                    "lmsys_doc_id": int(d.get("doc_id", -1)),
                }
            )
            if len(queries) >= want:
                break
    log.info(
        "LMSYS query extraction: scanned=%d accepted=%d rejected_sampled=%d",
        n_seen,
        len(queries),
        len(rejected_audit),
    )
    return queries, rejected_audit


def phase_build_queries(cfg: dict, *, force: bool = False) -> Path:
    """Build the 200-query held-out set."""
    from explore_persona_space.personas import EVAL_QUESTIONS

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    out = data_root / "held_out_queries.jsonl"
    if out.exists() and not force:
        log.info("phase=build-queries SKIP — %s already exists (--force to overwrite)", out)
        return out

    eval_q_n = int(cfg["held_out"]["eval_questions_count"])
    lmsys_q_n = int(cfg["held_out"]["lmsys_tail_count"])
    total = int(cfg["held_out"]["total"])
    if eval_q_n + lmsys_q_n != total:
        raise ValueError(
            f"held_out config: eval_questions_count ({eval_q_n}) + lmsys_tail_count "
            f"({lmsys_q_n}) != total ({total})"
        )

    if len(EVAL_QUESTIONS) < eval_q_n:
        raise RuntimeError(
            f"build-queries: only {len(EVAL_QUESTIONS)} EVAL_QUESTIONS, need {eval_q_n}"
        )

    lmsys_path = PROJECT_ROOT / cfg["held_out"]["lmsys_tail_path"]
    if not lmsys_path.exists():
        raise FileNotFoundError(f"build-queries: lmsys_tail_full.jsonl missing at {lmsys_path}")

    lmsys_queries, lmsys_rejected = _extract_lmsys_queries(lmsys_path, want=lmsys_q_n)

    # Round-3 patch: write the audit JSON BEFORE any shortfall raise so that
    # future "filter too strict" diagnoses have inspection material on disk.
    # Previously the audit was only written on the success path, which meant
    # the very crash that needed the audit also prevented its creation.
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_path = out.parent / "lmsys_query_audit.json"
    audit_payload = {
        "lmsys_path": str(lmsys_path),
        "n_requested": lmsys_q_n,
        "n_accepted": len(lmsys_queries),
        "n_rejected_sampled": len(lmsys_rejected),
        "shortfall": max(0, lmsys_q_n - len(lmsys_queries)),
        "accepted_preview": [
            {"lmsys_doc_id": q["lmsys_doc_id"], "text": q["text"]} for q in lmsys_queries[:20]
        ],
        "rejected_sample": lmsys_rejected,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False))
    log.info("wrote LMSYS query audit to %s", audit_path)

    if len(lmsys_queries) < lmsys_q_n:
        raise RuntimeError(
            f"build-queries: extracted only {len(lmsys_queries)} LMSYS-tail queries out of "
            f"requested {lmsys_q_n}. The round-2 tightened filter (text must END in '?', "
            f"≤ 80 words, no transcript markers) may be too strict for this corpus. "
            f"Inspect the rejection audit at {audit_path} and either loosen the filter "
            f"(carefully) or reduce lmsys_tail_count in conditions.yaml to "
            f"{len(lmsys_queries)} (and update `held_out.total` accordingly)."
        )
    lmsys_queries = lmsys_queries[:lmsys_q_n]

    next_id = 0
    with open(out, "w") as f:
        for q in EVAL_QUESTIONS[:eval_q_n]:
            row = {"id": next_id, "text": q, "source": "eval_questions"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            next_id += 1
        for q in lmsys_queries:
            q["id"] = next_id
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            next_id += 1
    log.info("wrote %d queries to %s", next_id, out)
    return out


def load_held_out_queries(cfg: dict) -> list:
    """Load the held-out query list as ``HeldOutQuery`` objects."""
    from explore_persona_space.experiments.issue_375.drift_eval import (
        HeldOutQuery,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    path = data_root / "held_out_queries.jsonl"
    if not path.exists():
        raise FileNotFoundError("held_out_queries.jsonl missing — run --phase build-queries first")
    queries: list[HeldOutQuery] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            queries.append(HeldOutQuery(id=int(d["id"]), text=d["text"], source=d["source"]))
    return queries


# ────────────────────────────────────────────────────────────────────────────
# Phase: example pools
# ────────────────────────────────────────────────────────────────────────────


def phase_build_pools(cfg: dict, *, force: bool = False, degraded_pool_ok: bool = False) -> dict:
    """Build persona-style, neutral, and random-bucket-persona-style pools."""
    from transformers import AutoTokenizer
    from vllm import LLM

    from explore_persona_space.experiments.issue_375.example_pool import (
        Example,
        assert_pool_size_meets_k,
        build_pool_meta,
        filter_zlt_contamination,
        generate_assistant_turns,
        load_candidate_docs,
        pool_overlap_stats,
        random_bucket_subset,
        save_pool_jsonl,
        score_docs_against_directions,
        select_neutral_per_persona,
        select_top_k_per_persona,
        write_pool_meta,
    )
    from explore_persona_space.experiments.issue_375.persona_directions import (
        load_persona_directions,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    persona_pool_path = data_root / "example_pool_persona_style.jsonl"
    neutral_pool_path = data_root / "example_pool_neutral.jsonl"
    random_bucket_path = data_root / "example_pool_persona_style_random_bucket.jsonl"
    meta_path = data_root / "example_pool_meta.json"
    if (
        all(
            p.exists()
            for p in (persona_pool_path, neutral_pool_path, random_bucket_path, meta_path)
        )
        and not force
    ):
        log.info("phase=build-pools SKIP — outputs exist (--force to overwrite)")
        return {
            "persona_style": persona_pool_path,
            "neutral": neutral_pool_path,
            "random_bucket": random_bucket_path,
        }

    pd_path = data_root / "persona_directions_L20.pt"
    if not pd_path.exists():
        raise FileNotFoundError(
            "persona_directions_L20.pt missing — run --phase persona-directions first"
        )
    pd = load_persona_directions(pd_path)

    fineweb_path = PROJECT_ROOT / cfg["example_pool"]["fineweb_path"]
    lmsys_path = PROJECT_ROOT / cfg["example_pool"]["lmsys_path"]
    docs = load_candidate_docs(fineweb_path, lmsys_path)

    persona_names = sorted(pd.directions.keys())  # software_engineer, librarian, villain
    # Score docs against persona directions
    log.info("scoring %d docs against %d persona directions...", len(docs), len(persona_names))
    scores = score_docs_against_directions(
        docs,
        pd.directions,
        base_model=cfg["base_model"],
        layer=int(pd.layer),
        doc_activation_max_chars=int(cfg["example_pool"]["doc_activation_max_chars"]),
    )

    # Score the random-bucket subset for the P1 sensitivity arm (villain only)
    rand_docs = random_bucket_subset(docs)
    log.info("random-bucket subset: %d docs (for P1 sensitivity arm)", len(rand_docs))
    rand_scores = score_docs_against_directions(
        rand_docs,
        {"villain": pd.directions["villain"]},
        base_model=cfg["base_model"],
        layer=int(pd.layer),
        doc_activation_max_chars=int(cfg["example_pool"]["doc_activation_max_chars"]),
    )

    k_per = int(cfg["example_pool"]["k_per_persona"])
    # Round-4: P1 random-bucket arm uses a smaller K (the unbiased corpus has
    # only ~5% villain-positive docs; k=50 would crash). Default to k_per
    # for back-compat with configs that predate the round-4 patch.
    k_rand = int(cfg["example_pool"].get("k_per_persona_random_bucket", k_per))
    persona_picks = select_top_k_per_persona(docs, scores, persona_names, k=k_per)
    neutral_picks, achieved_thr = select_neutral_per_persona(
        docs,
        scores,
        persona_names,
        k=k_per,
        threshold=float(cfg["example_pool"]["neutral_cos_threshold"]),
    )
    rand_picks = select_top_k_per_persona(rand_docs, rand_scores, ["villain"], k=k_rand)

    # M-7: fail loud if any persona's top-K returned fewer than the target k.
    # The contamination filter further trims a few examples below, so we ALSO
    # re-assert after that. Both gates respect `--degraded-pool-ok`. The
    # random-bucket arm uses k_rand (round-4: 20) instead of k_per (50).
    persona_pre_sizes = assert_pool_size_meets_k(
        persona_picks, k_per, pool_kind="persona-style", degraded_ok=degraded_pool_ok
    )
    neutral_pre_sizes = assert_pool_size_meets_k(
        neutral_picks, k_per, pool_kind="neutral", degraded_ok=degraded_pool_ok
    )
    rand_pre_sizes = assert_pool_size_meets_k(
        rand_picks, k_rand, pool_kind="persona-style-random-bucket", degraded_ok=degraded_pool_ok
    )

    # Load vLLM + tokenizer once for assistant-turn generation
    log.info("loading vLLM for assistant-turn generation (base model)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=cfg["base_model"],
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
        max_model_len=int(cfg["vllm"]["max_model_len"]),
        max_num_seqs=int(cfg["vllm"]["max_num_seqs"]),
        seed=int(cfg["vllm"]["seed"]),
    )

    persona_prompts = cfg["persona_directions"]["personas"]
    base_gen = cfg["example_pool"]["base_assistant_gen"]
    doc_text_cap = int(cfg["example_pool"]["doc_text_max_chars"])
    contam_gate = float(cfg["example_pool"]["contamination_drop_rate_hard_gate"])

    persona_pool: dict[str, list[Example]] = {}
    neutral_pool: dict[str, list[Example]] = {}
    rand_pool: dict[str, list[Example]] = {}
    meta: dict[str, dict] = {}

    try:
        for p in persona_names:
            log.info("=== building pools for persona=%s ===", p)
            # Persona-style
            picks = persona_picks[p]
            user_texts = [(docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in picks]
            asst_texts = generate_assistant_turns(
                llm,
                tokenizer,
                user_texts=user_texts,
                system_prompt=persona_prompts[p],
                temperature=float(base_gen["temperature"]),
                max_tokens=int(base_gen["max_tokens"]),
                seed=int(base_gen["seed"]),
            )
            persona_examples = [
                Example(
                    persona=p,
                    doc_id=int(docs[idx].get("doc_id", idx)),
                    user=user_texts[i],
                    assistant=asst_texts[i],
                    cos_to_persona_dir=cos,
                    source_corpus=docs[idx].get("source_corpus", "unknown"),
                    qwen3_axis_bucket=docs[idx].get("axis_bucket", "unknown"),
                    # C-1: persona-style pools — selection_persona == persona.
                    selection_persona=p,
                )
                for i, (idx, cos) in enumerate(picks)
            ]
            persona_examples, n_drop_ps, drop_rate_ps = filter_zlt_contamination(
                persona_examples, p, "persona-style", hard_gate=contam_gate
            )
            persona_pool[p] = persona_examples

            # Neutral
            n_picks = neutral_picks[p]
            user_texts_n = [(docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in n_picks]
            asst_texts_n = generate_assistant_turns(
                llm,
                tokenizer,
                user_texts=user_texts_n,
                system_prompt=persona_prompts["assistant"],
                temperature=float(base_gen["temperature"]),
                max_tokens=int(base_gen["max_tokens"]),
                seed=int(base_gen["seed"]),
            )
            neutral_examples = [
                Example(
                    persona="assistant",
                    doc_id=int(docs[idx].get("doc_id", idx)),
                    user=user_texts_n[i],
                    assistant=asst_texts_n[i],
                    cos_to_persona_dir=cos,
                    source_corpus=docs[idx].get("source_corpus", "unknown"),
                    qwen3_axis_bucket=docs[idx].get("axis_bucket", "unknown"),
                    # C-1: neutral pool — selection_persona records which persona's
                    # |cos|<thr filter chose this doc. The loader groups by this
                    # field, NOT by integer slicing. Without this, ZLT-induced
                    # per-persona drops would silently misalign arms.
                    selection_persona=p,
                )
                for i, (idx, cos) in enumerate(n_picks)
            ]
            neutral_examples, n_drop_n, drop_rate_n = filter_zlt_contamination(
                neutral_examples, p, "neutral", hard_gate=contam_gate
            )
            neutral_pool[p] = neutral_examples

            meta[p] = {
                "persona_style": build_pool_meta(
                    p, "persona-style", picks, docs, n_drop_ps, drop_rate_ps
                ),
                "neutral": build_pool_meta(
                    p,
                    "neutral",
                    n_picks,
                    docs,
                    n_drop_n,
                    drop_rate_n,
                    achieved_neutral_threshold=float(achieved_thr.get(p, 0.0)),
                ),
            }

        # Random-bucket pool (villain only)
        log.info("=== building pools for villain (random-bucket) ===")
        rand_picks_villain = rand_picks["villain"]
        rand_user_texts = [
            (rand_docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in rand_picks_villain
        ]
        rand_asst_texts = generate_assistant_turns(
            llm,
            tokenizer,
            user_texts=rand_user_texts,
            system_prompt=persona_prompts["villain"],
            temperature=float(base_gen["temperature"]),
            max_tokens=int(base_gen["max_tokens"]),
            seed=int(base_gen["seed"]),
        )
        rand_examples = [
            Example(
                persona="villain",
                doc_id=int(rand_docs[idx].get("doc_id", idx)),
                user=rand_user_texts[i],
                assistant=rand_asst_texts[i],
                cos_to_persona_dir=cos,
                source_corpus=rand_docs[idx].get("source_corpus", "unknown"),
                qwen3_axis_bucket=rand_docs[idx].get("axis_bucket", "unknown"),
                # C-1: random-bucket pool — selection_persona == persona.
                selection_persona="villain",
            )
            for i, (idx, cos) in enumerate(rand_picks_villain)
        ]
        rand_examples, n_drop_r, drop_rate_r = filter_zlt_contamination(
            rand_examples, "villain", "persona-style-random-bucket", hard_gate=contam_gate
        )
        rand_pool["villain"] = rand_examples
        meta["villain"]["persona_style_random_bucket"] = build_pool_meta(
            "villain",
            "persona-style-random-bucket",
            rand_picks_villain,
            rand_docs,
            n_drop_r,
            drop_rate_r,
        )
    finally:
        # Free vLLM before downstream phases
        del llm
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    # M-7 post-ZLT-filter re-assertion: contamination filter may have trimmed
    # below k_per. Same degraded-ok contract as the pre-filter gate.
    persona_post_sizes = assert_pool_size_meets_k(
        {p: [(0, 0.0)] * len(ex) for p, ex in persona_pool.items()},
        k_per,
        pool_kind="persona-style (post-ZLT)",
        degraded_ok=degraded_pool_ok,
    )
    neutral_post_sizes = assert_pool_size_meets_k(
        {p: [(0, 0.0)] * len(ex) for p, ex in neutral_pool.items()},
        k_per,
        pool_kind="neutral (post-ZLT)",
        degraded_ok=degraded_pool_ok,
    )
    rand_post_sizes = {"villain": len(rand_pool.get("villain", []))}
    if rand_post_sizes["villain"] < k_rand and not degraded_pool_ok:
        raise RuntimeError(
            f"persona-style-random-bucket post-ZLT pool short: "
            f"villain={rand_post_sizes['villain']} < k={k_rand}"
        )

    # M-3 pool-overlap stats: per-persona-pair Jaccard / intersection / union
    # of the persona-style top-K selections (before ZLT — overlap reflects the
    # persona-direction geometry, not the contamination filter).
    overlap = pool_overlap_stats(persona_picks, docs)

    # Persist
    persona_pool_path.parent.mkdir(parents=True, exist_ok=True)
    flat_persona: list[Example] = []
    for p in persona_names:
        flat_persona.extend(persona_pool[p])
    flat_neutral: list[Example] = []
    for p in persona_names:
        flat_neutral.extend(neutral_pool[p])
    save_pool_jsonl(flat_persona, persona_pool_path)
    save_pool_jsonl(flat_neutral, neutral_pool_path)
    save_pool_jsonl(rand_pool["villain"], random_bucket_path)

    # Bundle round-2 review additions into the meta JSON:
    #   - pool_overlap (M-3)
    #   - pool_sizes_per_persona (M-7 diagnostic; pre-ZLT vs post-ZLT counts)
    #   - degraded_pool_ok flag (M-7 / M-8: analyzer flags the run)
    meta["__diagnostics__"] = {
        "pool_overlap": overlap,
        "pool_sizes": {
            "persona_style_pre_zlt": persona_pre_sizes,
            "persona_style_post_zlt": persona_post_sizes,
            "neutral_pre_zlt": neutral_pre_sizes,
            "neutral_post_zlt": neutral_post_sizes,
            "random_bucket_pre_zlt": rand_pre_sizes,
            "random_bucket_post_zlt": rand_post_sizes,
            # Round-4: audit signal that the random-bucket pool uses a smaller
            # k than the axis-extreme pools (20 vs 50). Analyzer must flag the
            # unequal sample-size ratio when interpreting the P1 result.
            "k_per_persona_axis_extreme": k_per,
            "k_per_persona_random_bucket": k_rand,
        },
        "k_per_persona": k_per,
        "k_per_persona_random_bucket": k_rand,
        "degraded_pool_ok": bool(degraded_pool_ok),
    }
    write_pool_meta(meta, meta_path)

    return {
        "persona_style": persona_pool_path,
        "neutral": neutral_pool_path,
        "random_bucket": random_bucket_path,
    }


def load_pools(cfg: dict) -> dict:
    """Load all pools from disk, indexed by persona.

    C-1 (round-2 review): neutral examples are grouped by their
    ``selection_persona`` field, NEVER by integer slicing. If the ZLT
    contamination filter drops one persona's neutral examples, the
    remaining personas keep their full pools — slicing would silently
    cross-contaminate persona boundaries with no error.

    Persona names are read from ``cfg["persona_directions"]["personas"]``
    (minus the helpful-assistant entry) so we don't hardcode "villain /
    librarian / software_engineer" twice in the codebase.
    """
    from explore_persona_space.experiments.issue_375.example_pool import (
        load_pool_jsonl,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    persona_examples = load_pool_jsonl(data_root / "example_pool_persona_style.jsonl")
    neutral_examples = load_pool_jsonl(data_root / "example_pool_neutral.jsonl")
    random_bucket_examples = load_pool_jsonl(
        data_root / "example_pool_persona_style_random_bucket.jsonl"
    )

    # Derive persona names from config (drop the "assistant" pseudo-persona).
    persona_names = sorted(p for p in cfg["persona_directions"]["personas"] if p != "assistant")

    persona_style: dict[str, list] = {p: [] for p in persona_names}
    for ex in persona_examples:
        # Group by the EFFECTIVE persona (selection_persona == persona for
        # persona-style pools). Use selection_persona so we are consistent
        # with the neutral-pool path.
        sel = ex.selection_persona or ex.persona
        persona_style.setdefault(sel, []).append(ex)

    # NEUTRAL — group by selection_persona. C-1 fix.
    neutral_by_persona: dict[str, list] = {p: [] for p in persona_names}
    for ex in neutral_examples:
        sel = ex.selection_persona
        if not sel:
            raise ValueError(
                f"load_pools: neutral example doc_id={ex.doc_id} has no "
                f"selection_persona; rebuild the pool (--phase build-pools "
                f"--force). See round-2 review BLOCKER C-1."
            )
        neutral_by_persona.setdefault(sel, []).append(ex)

    # Sanity: every named persona must have a non-empty bucket. Empty
    # buckets at reload time imply pool-build went wrong (drop>10% triggers
    # a hard gate upstream; if we got here with an empty bucket, the
    # operator manually edited the JSONL).
    for p in persona_names:
        if not neutral_by_persona[p]:
            raise RuntimeError(
                f"load_pools: neutral pool for persona={p!r} is empty after "
                f"selection_persona grouping. Rebuild via --phase build-pools "
                f"--force or inspect the pool JSONL for manual edits."
            )

    return {
        "persona_style": persona_style,
        "neutral": neutral_by_persona,
        "random_bucket": {"villain": random_bucket_examples},
    }


# ────────────────────────────────────────────────────────────────────────────
# Adapter merge + vLLM helpers
# ────────────────────────────────────────────────────────────────────────────


def download_adapter(repo_id: str, hub_subpath: str, local_dir: Path) -> Path:
    """Snapshot-download a single adapter dir from HF Hub.

    ``allow_patterns`` is the canonical scoped download — we don't want the
    rest of the repo (350+ GB) on disk.
    """
    from huggingface_hub import snapshot_download

    log.info("downloading adapter %s/%s -> %s", repo_id, hub_subpath, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{hub_subpath}/*",
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_path = local_dir / hub_subpath
    if not (adapter_path / "adapter_config.json").exists():
        raise RuntimeError(
            f"download_adapter: expected adapter_config.json under {adapter_path} after "
            f"snapshot_download — got {sorted(p.name for p in adapter_path.iterdir())}"
        )
    return adapter_path


def merge_and_save(base_model: str, adapter_path: Path, merged_path: Path) -> Path:
    """Merge LoRA adapter into base model and save to ``merged_path``."""
    from explore_persona_space.train.sft import merge_lora

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        shutil.rmtree(merged_path)
    merge_lora(
        base_model_path=base_model,
        adapter_path=str(adapter_path),
        output_dir=str(merged_path),
        gpu_id=0,
    )
    log.info("merged adapter -> %s", merged_path)
    return merged_path


# ────────────────────────────────────────────────────────────────────────────
# Phase: zero-shot, base controls, main, pilot
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CellSpec:
    """A single cell to run."""

    label: str
    adapter_id: str  # "villain_C1" or "base"
    # pool_kind: one of {"zero-shot", "persona-style", "neutral",
    #                    "wrong-persona", "persona-style-random-bucket"}.
    pool_kind: str
    k: int
    # pool_persona: which persona's pool to draw from. Matches the adapter's
    # persona for main cells; wrong-persona / base cells use a different one.
    pool_persona: str


def build_core_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """Five core cells per adapter: k=0 zero-shot, persona-style {k=1, k=3}, neutral {k=1, k=3}."""
    adapter_meta = _adapter_meta(cfg, adapter_id)
    p = adapter_meta["persona"]
    seed = int(cfg["seed"])
    cells = [
        CellSpec(
            label=f"{adapter_id}_zero-shot_k0_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="zero-shot",
            k=0,
            pool_persona=p,
        ),
    ]
    for k in (1, 3):
        cells.append(
            CellSpec(
                label=f"{adapter_id}_persona-style_k{k}_seed{seed}",
                adapter_id=adapter_id,
                pool_kind="persona-style",
                k=k,
                pool_persona=p,
            )
        )
        cells.append(
            CellSpec(
                label=f"{adapter_id}_neutral_k{k}_seed{seed}",
                adapter_id=adapter_id,
                pool_kind="neutral",
                k=k,
                pool_persona=p,
            )
        )
    return cells


def build_wrong_persona_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """W1..W4: clean-baseline adapter + a different persona's pool at k=3."""
    wrong_map = cfg.get("wrong_persona_map", {})
    if adapter_id not in wrong_map:
        return []
    seed = int(cfg["seed"])
    wrong_persona = wrong_map[adapter_id]
    return [
        CellSpec(
            label=f"{adapter_id}_wrong-persona_k3_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="wrong-persona",
            k=3,
            pool_persona=wrong_persona,
        )
    ]


def build_pool_bias_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """P1: villain-C1 + villain-style-random-bucket at k=3."""
    pbs = cfg.get("pool_bias_sensitivity", {})
    if adapter_id != pbs.get("adapter_id"):
        return []
    seed = int(cfg["seed"])
    return [
        CellSpec(
            label=f"{adapter_id}_persona-style-random-bucket_k3_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="persona-style-random-bucket",
            k=3,
            pool_persona=pbs.get("persona", "villain"),
        )
    ]


def build_base_cells(cfg: dict) -> list[CellSpec]:
    """B1, B2, B3: base model (no adapter) + persona-style k=3.

    Round-2 review BLOCKER M-1: the YAML ``id`` field hardcodes ``seed42``;
    when ``--seed N`` (N≠42) is passed, the base cell would write to the
    wrong path. We rebuild the id from ``(pool_persona, k, current seed)``
    using the same scheme as :func:`analyze.base_cell_label`.
    """
    out: list[CellSpec] = []
    seed = int(cfg["seed"])
    for cell_cfg in cfg["base_model_cells"]:
        k = int(cell_cfg["k"])
        pool_persona = cell_cfg["pool_persona"]
        label = f"base_no-adapter_persona-style-{pool_persona}_k{k}_seed{seed}"
        out.append(
            CellSpec(
                label=label,
                adapter_id="base",
                pool_kind="persona-style",
                k=k,
                pool_persona=pool_persona,
            )
        )
    return out


def _adapter_meta(cfg: dict, adapter_id: str) -> dict:
    for a in cfg["adapters"]:
        if a["id"] == adapter_id:
            return a
    raise KeyError(f"adapter id not found in config: {adapter_id!r}")


def _resolve_pool(cell: CellSpec, pools: dict) -> list:
    """Get the example pool for a cell."""
    if cell.pool_kind == "zero-shot":
        return []
    if cell.pool_kind == "persona-style":
        return pools["persona_style"][cell.pool_persona]
    if cell.pool_kind == "neutral":
        return pools["neutral"][cell.pool_persona]
    if cell.pool_kind == "wrong-persona":
        # Wrong-persona uses the persona-style pool of the *other* persona.
        return pools["persona_style"][cell.pool_persona]
    if cell.pool_kind == "persona-style-random-bucket":
        return pools["random_bucket"][cell.pool_persona]
    raise ValueError(f"unknown pool_kind: {cell.pool_kind!r}")


def run_cells_for_llm(
    cfg: dict,
    llm,
    tokenizer,
    cells: Sequence[CellSpec],
    queries: list,
    pools: dict,
) -> list[dict]:
    """Run a sequence of cells on a single vLLM instance."""
    from explore_persona_space.experiments.issue_375.drift_eval import (
        run_cell,
    )

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    decoder = cfg["decoder"]
    summaries: list[dict] = []
    for cell in cells:
        pool = _resolve_pool(cell, pools)
        if cell.k > 0 and len(pool) < cell.k:
            raise RuntimeError(
                f"cell {cell.label} requires k={cell.k} examples but pool of "
                f"kind {cell.pool_kind!r} for persona={cell.pool_persona!r} "
                f"only has {len(pool)} examples"
            )
        summary = run_cell(
            llm=llm,
            tokenizer=tokenizer,
            cell_label=cell.label,
            queries=queries,
            pool=pool,
            k=cell.k,
            adapter_id=cell.adapter_id,
            pool_kind=cell.pool_kind,
            decoder=decoder,
            eval_results_root=eval_root,
        )
        summaries.append(summary)
    return summaries


def make_vllm(cfg: dict, model_path: str):
    """Construct a vLLM ``LLM`` instance from config."""
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
        max_model_len=int(cfg["vllm"]["max_model_len"]),
        max_num_seqs=int(cfg["vllm"]["max_num_seqs"]),
        seed=int(cfg["vllm"]["seed"]),
    )


def _vllm_release_caches() -> None:
    """Run gc.collect + torch.cuda.empty_cache after the caller has
    released its ``llm`` binding with ``del llm``.

    Round-2 review BLOCKER C-2 (Codex): the previous ``free_vllm(llm)``
    helper only deleted its own *parameter*, not the caller's ``llm``
    variable. The next ``llm = make_vllm(...)`` then evaluated the RHS
    while the old vLLM still owned GPU memory → OOM risk on 1x H100
    across 9 adapters.

    Call sites MUST do::

        llm = make_vllm(cfg, model_path)
        try:
            ...
        finally:
            del llm                  # release caller's binding
            _vllm_release_caches()   # then GC + empty_cache

    The ``del llm`` MUST happen at the call site so the caller's local
    binding actually goes away — a helper cannot do that for them.
    """
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def phase_base_floor(cfg: dict) -> dict:
    """Plan §4.5 step 3: held-out base-model floor — fire rate < 1% on the
    200 held-out queries under helpful-assistant prompt with NO few-shot
    context. Saves to ``eval_results/issue_375/base_floor.json``.
    """
    from transformers import AutoTokenizer

    queries = load_held_out_queries(cfg)
    threshold = float(cfg["held_out"]["base_floor_threshold"])
    log.info(
        "=== phase=base-floor: %d queries on base model (sanity floor < %.1f%%) ===",
        len(queries),
        threshold * 100,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = make_vllm(cfg, cfg["base_model"])
    try:
        summaries = run_cells_for_llm(
            cfg=cfg,
            llm=llm,
            tokenizer=tokenizer,
            cells=[
                CellSpec(
                    label=f"base_floor_k0_seed{int(cfg['seed'])}",
                    adapter_id="base",
                    pool_kind="zero-shot",
                    k=0,
                    pool_persona="assistant",
                )
            ],
            queries=queries,
            pools={"persona_style": {}, "neutral": {}, "random_bucket": {}},
        )
    finally:
        # C-2: drop caller binding FIRST, then GC. Helper cannot do this.
        del llm
        _vllm_release_caches()

    rate = summaries[0]["overall_rate"]
    payload = {
        "base_floor_rate": rate,
        "threshold": threshold,
        "gate_passed": rate < threshold,
        "n_queries": summaries[0]["n_queries"],
        "n_completions": summaries[0]["n_completions"],
    }
    out = PROJECT_ROOT / cfg["output"]["eval_results_root"] / "base_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    if not payload["gate_passed"]:
        raise RuntimeError(
            f"phase=base-floor HARD GATE FAILED: base-model marker rate {rate:.4f} >= "
            f"{threshold} on held-out queries with no few-shot context. The query distribution "
            f"is somehow leaking the marker. Strip offending queries and re-run."
        )
    log.info("base-floor PASS: rate=%.4f < threshold=%.4f", rate, threshold)
    return payload


def _expected_full_sweep_generations(cfg: dict) -> int:
    """Compute expected total generations from the actual cell matrix (M-2).

    Round-2 review BLOCKER M-2: the round-1 helper hardcoded 116_000 from a
    stale planner estimate. The real number is derived from the executable
    matrix: per-adapter cell counts x n_completions_per_query x n_queries.
    """
    n_adapters = len(cfg["adapter_order"])
    # Per-adapter: 1 zero-shot + 2 persona-style (k=1,k=3) + 2 neutral (k=1,k=3) = 5
    per_adapter_core_cells = 5
    n_wrong_persona = len(cfg.get("wrong_persona_map", {}))
    n_pool_bias = 1 if cfg.get("pool_bias_sensitivity") else 0
    n_base = len(cfg["base_model_cells"])
    n_cells = n_adapters * per_adapter_core_cells + n_wrong_persona + n_pool_bias + n_base
    n_queries = int(cfg["held_out"]["total"])
    n_per_query = int(cfg["decoder"]["n"])
    return n_cells * n_queries * n_per_query


def phase_pilot(cfg: dict) -> dict:
    """Throughput pilot — EXACTLY 200 generations on villain_C1 at k=3
    max_tokens=2048, measure end-to-end gen/s to drive the §8 mode decision.

    Round-2 review BLOCKER M-4: round-1 ran 200 queries x n=10 = 2000 gens
    (10x what the plan says). We now slice the held-out set to 20 queries
    so the pilot uses 20 x n=10 = 200 generations under the production
    decoder regime. The pilot result lives under its own cell label
    (``..._pilot_persona-style_k3_seed{N}``) so it doesn't collide with the
    full villain_C1_persona-style_k3 cell run later.

    Round-2 review BLOCKER M-2: projected wall time is computed from the
    actual cell matrix (see :func:`_expected_full_sweep_generations`),
    not a magic ``116_000``. The output key is renamed to
    ``projected_full_sweep_hours``.
    """
    queries_full = load_held_out_queries(cfg)
    # M-4: exactly 200 generations — slice to 20 queries x n=10
    pilot_queries = queries_full[:20]
    if len(pilot_queries) < 20:
        raise RuntimeError(
            f"phase=pilot: held-out query set has only {len(queries_full)} queries; "
            f"need ≥ 20 for the 200-gen pilot. Re-run --phase build-queries."
        )
    expected_pilot_gens = 20 * int(cfg["decoder"]["n"])  # 200 with default n=10

    pools = load_pools(cfg)
    adapter_id = "villain_C1"
    adapter_meta = _adapter_meta(cfg, adapter_id)
    adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
    adapter_path = download_adapter(cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir)
    with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
        merged_path = Path(td) / "merged"
        merge_and_save(cfg["base_model"], adapter_path, merged_path)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        llm = make_vllm(cfg, str(merged_path))
        try:
            pilot_cell = CellSpec(
                label=f"{adapter_id}_pilot_persona-style_k3_seed{int(cfg['seed'])}",
                adapter_id=adapter_id,
                pool_kind="persona-style",
                k=3,
                pool_persona=adapter_meta["persona"],
            )
            t0 = time.monotonic()
            summaries = run_cells_for_llm(
                cfg=cfg,
                llm=llm,
                tokenizer=tokenizer,
                cells=[pilot_cell],
                queries=pilot_queries,
                pools=pools,
            )
            elapsed = time.monotonic() - t0
        finally:
            # C-2: drop caller binding FIRST, then GC. Helper cannot do this.
            del llm
            _vllm_release_caches()

    n_completions = summaries[0]["n_completions"]
    if n_completions != expected_pilot_gens:
        log.warning(
            "phase=pilot: expected exactly %d gens (20 queries x n=%d), got %d",
            expected_pilot_gens,
            int(cfg["decoder"]["n"]),
            n_completions,
        )
    gen_per_s = n_completions / elapsed if elapsed > 0 else 0.0
    projected_gens = _expected_full_sweep_generations(cfg)
    projected_hours = (projected_gens / gen_per_s / 3600.0) if gen_per_s > 0 else float("inf")
    out = PROJECT_ROOT / cfg["output"]["eval_results_root"] / "pilot_throughput.json"
    payload = {
        "elapsed_s": elapsed,
        "n_queries_pilot": len(pilot_queries),
        "n_per_query": int(cfg["decoder"]["n"]),
        "n_completions": n_completions,
        "expected_pilot_generations": expected_pilot_gens,
        "gen_per_s": gen_per_s,
        "projected_full_sweep_generations": projected_gens,
        "projected_full_sweep_hours": projected_hours,
        "wall_time_budget_hours": 14.0,
        "wall_time_within_budget": projected_hours <= 14.0,
        "cell_label": pilot_cell.label,
        "pilot_rate": summaries[0]["overall_rate"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info(
        "pilot: %d gens (20 q x n=%d) in %.1fs → %.2f gen/s, "
        "projected full sweep (%d gens) ~%.1f h [budget=14h]",
        n_completions,
        int(cfg["decoder"]["n"]),
        elapsed,
        gen_per_s,
        projected_gens,
        projected_hours,
    )
    return payload


def phase_zero_shot(cfg: dict) -> list[dict]:
    """Run k=0 zero-shot cells for ALL adapters — verifies sanity gates 3+4."""
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    summaries: list[dict] = []
    seed = int(cfg["seed"])
    for adapter_id in cfg["adapter_order"]:
        adapter_meta = _adapter_meta(cfg, adapter_id)
        adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
        adapter_path = download_adapter(
            cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir
        )
        with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
            merged_path = Path(td) / "merged"
            merge_and_save(cfg["base_model"], adapter_path, merged_path)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            llm = make_vllm(cfg, str(merged_path))
            try:
                cell = CellSpec(
                    label=f"{adapter_id}_zero-shot_k0_seed{seed}",
                    adapter_id=adapter_id,
                    pool_kind="zero-shot",
                    k=0,
                    pool_persona=adapter_meta["persona"],
                )
                summaries.extend(run_cells_for_llm(cfg, llm, tokenizer, [cell], queries, pools))
            finally:
                # C-2: drop caller binding FIRST, then GC. Helper cannot do this.
                del llm
                _vllm_release_caches()
    return summaries


def phase_base_controls(cfg: dict) -> list[dict]:
    """B1/B2/B3 — base model + persona-style k=3 (sanity gate 6)."""
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = make_vllm(cfg, cfg["base_model"])
    try:
        cells = build_base_cells(cfg)
        summaries = run_cells_for_llm(cfg, llm, tokenizer, cells, queries, pools)
    finally:
        # C-2: drop caller binding FIRST, then GC. Helper cannot do this.
        del llm
        _vllm_release_caches()
    return summaries


def phase_main(
    cfg: dict,
    *,
    skip_zero_shot: bool = True,
    only_clean_baseline: bool = False,
) -> list[dict]:
    """Run all non-zero-shot cells per adapter (persona-style, neutral,
    wrong-persona for clean adapters, P1 for villain_C1).

    Args:
        skip_zero_shot: zero-shot cells are usually run as the dedicated
            ``--phase zero-shot`` step first; skip them here by default.
        only_clean_baseline: if True, restrict to the 4 strict-test adapters
            (degraded-mode option from plan §8).
    """
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    summaries: list[dict] = []
    clean_ids = {a["id"] for a in cfg["adapters"] if a.get("set") == "strict"}

    for adapter_id in cfg["adapter_order"]:
        if only_clean_baseline and adapter_id not in clean_ids:
            log.info(
                "phase=main: skipping non-clean adapter %s (only_clean_baseline=True)", adapter_id
            )
            continue
        adapter_meta = _adapter_meta(cfg, adapter_id)
        adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
        adapter_path = download_adapter(
            cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir
        )
        with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
            merged_path = Path(td) / "merged"
            merge_and_save(cfg["base_model"], adapter_path, merged_path)

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            llm = make_vllm(cfg, str(merged_path))
            try:
                cells = build_core_cells(cfg, adapter_id)
                if skip_zero_shot:
                    cells = [c for c in cells if c.pool_kind != "zero-shot"]
                cells.extend(build_wrong_persona_cells(cfg, adapter_id))
                cells.extend(build_pool_bias_cells(cfg, adapter_id))
                summaries.extend(run_cells_for_llm(cfg, llm, tokenizer, cells, queries, pools))
            finally:
                # C-2: drop caller binding FIRST, then GC. Helper cannot do this.
                del llm
                _vllm_release_caches()
    return summaries


# ────────────────────────────────────────────────────────────────────────────
# Phase: analyze
# ────────────────────────────────────────────────────────────────────────────


def _expected_cell_labels(cfg: dict, seed: int) -> list[str]:
    """Full enumeration of expected cell labels per the cell matrix."""
    from explore_persona_space.experiments.issue_375 import analyze

    labels: list[str] = []
    for adapter_id in cfg["adapter_order"]:
        labels.append(analyze.cell_label(adapter_id, "zero-shot", 0, seed))
        for k in (1, 3):
            labels.append(analyze.cell_label(adapter_id, "persona-style", k, seed))
            labels.append(analyze.cell_label(adapter_id, "neutral", k, seed))
    for adapter_id in cfg.get("wrong_persona_map", {}):
        labels.append(analyze.cell_label(adapter_id, "wrong-persona", 3, seed))
    pbs = cfg.get("pool_bias_sensitivity", {})
    if pbs:
        labels.append(analyze.pool_bias_cell_label(pbs["adapter_id"], int(pbs["k"]), seed))
    # Base cells: rebuild ids from (pool_persona, k, seed) — same scheme as
    # build_base_cells (M-1).
    for cell_cfg in cfg["base_model_cells"]:
        k = int(cell_cfg["k"])
        pool_persona = cell_cfg["pool_persona"]
        labels.append(f"base_no-adapter_persona-style-{pool_persona}_k{k}_seed{seed}")
    return labels


def _completed_cell_labels(eval_root: Path) -> list[str]:
    """Scan eval_root/ for cells that completed (have summary.json on disk)."""
    out: list[str] = []
    if not eval_root.exists():
        return out
    for child in sorted(eval_root.iterdir()):
        if child.is_dir() and (child / "summary.json").exists():
            out.append(child.name)
    return out


def phase_analyze(cfg: dict) -> dict:
    """Aggregate per-cell results, run bootstraps, produce figures.

    M-8 (round-2 review): runs in degraded mode if some expected cells are
    missing. We compute ``expected ∩ completed`` and only run the bootstrap
    suites for adapters whose full cell quad (zero-shot k=0, persona-style
    k=3, neutral k=3, optional wrong-persona k=3) is present. Missing cells
    are logged to ``aggregated.json`` under ``__missing_cells__``.

    The hero figure is built ONLY over strict adapters with ALL required
    cells present. The wrong-persona / base-model / pool-bias / secondary-Δ
    figures all skip rather than crash if their underlying cells are
    incomplete.
    """
    from explore_persona_space.experiments.issue_375 import analyze

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    figures_root = PROJECT_ROOT / cfg["output"]["figures_root"]
    seed = int(cfg["seed"])

    expected = set(_expected_cell_labels(cfg, seed))
    completed = set(_completed_cell_labels(eval_root))
    missing = sorted(expected - completed)
    if missing:
        log.warning(
            "phase=analyze DEGRADED — %d/%d expected cells missing on disk: %s",
            len(missing),
            len(expected),
            missing[:8] + (["..."] if len(missing) > 8 else []),
        )

    # 1) aggregated.json (over expected ∩ completed)
    available = sorted(expected & completed)
    aggregated = analyze.write_aggregated(eval_root, available, eval_root / "aggregated.json")

    # Track the degraded-mode footprint in aggregated.json
    aggregated_meta_path = eval_root / "aggregated.json"
    if aggregated_meta_path.exists():
        cur = json.loads(aggregated_meta_path.read_text())
        cur["__missing_cells__"] = missing
        cur["__expected_count__"] = len(expected)
        cur["__completed_count__"] = len(completed & expected)
        aggregated_meta_path.write_text(json.dumps(cur, indent=2))

    # 2) stratified
    analyze.write_stratified(eval_root, available, eval_root / "stratified_by_query_source.json")

    # 3) bootstrap — only for adapters with the FULL required cell quad.
    n_boot = int(cfg["bootstrap"]["n_boot"])
    boot_seed = int(cfg["bootstrap"]["seed"])

    def _has(label: str) -> bool:
        return label in completed

    strict_adapters_all = [a["id"] for a in cfg["adapters"] if a.get("set") == "strict"]
    secondary_adapters_all = [a["id"] for a in cfg["adapters"] if a.get("set") == "secondary"]

    def _adapter_complete_for_strict(a_id: str) -> bool:
        required = [
            analyze.cell_label(a_id, "zero-shot", 0, seed),
            analyze.cell_label(a_id, "persona-style", 1, seed),
            analyze.cell_label(a_id, "persona-style", 3, seed),
            analyze.cell_label(a_id, "neutral", 1, seed),
            analyze.cell_label(a_id, "neutral", 3, seed),
        ]
        return all(_has(lbl) for lbl in required)

    def _adapter_complete_for_secondary(a_id: str) -> bool:
        required = [
            analyze.cell_label(a_id, "zero-shot", 0, seed),
            analyze.cell_label(a_id, "persona-style", 3, seed),
            analyze.cell_label(a_id, "neutral", 3, seed),
        ]
        return all(_has(lbl) for lbl in required)

    strict_adapters = [a for a in strict_adapters_all if _adapter_complete_for_strict(a)]
    secondary_adapters = [a for a in secondary_adapters_all if _adapter_complete_for_secondary(a)]
    if strict_adapters != strict_adapters_all:
        log.warning(
            "phase=analyze: dropping %d strict adapters from bootstrap (missing cells): %s",
            len(strict_adapters_all) - len(strict_adapters),
            sorted(set(strict_adapters_all) - set(strict_adapters)),
        )
    if secondary_adapters != secondary_adapters_all:
        log.warning(
            "phase=analyze: dropping %d secondary adapters from bootstrap (missing cells): %s",
            len(secondary_adapters_all) - len(secondary_adapters),
            sorted(set(secondary_adapters_all) - set(secondary_adapters)),
        )

    # wrong-persona map: only include adapters that have BOTH the matching
    # persona-style k=3 AND the wrong-persona k=3 cell on disk.
    raw_wpm = cfg.get("wrong_persona_map", {})
    wpm = {
        a: w
        for a, w in raw_wpm.items()
        if _has(analyze.cell_label(a, "persona-style", 3, seed))
        and _has(analyze.cell_label(a, "wrong-persona", 3, seed))
    }

    strict_bootstrap = analyze.compute_strict_test_suite(
        eval_root, strict_adapters, wpm, n_boot=n_boot, seed=boot_seed
    )
    secondary_bootstrap = analyze.compute_secondary_delta_suite(
        eval_root, secondary_adapters, n_boot=n_boot, seed=boot_seed
    )
    pool_bias_bootstrap: dict = {}
    pbs = cfg.get("pool_bias_sensitivity", {})
    if pbs:
        main_label = analyze.cell_label(pbs["adapter_id"], "persona-style", int(pbs["k"]), seed)
        rand_label = analyze.pool_bias_cell_label(pbs["adapter_id"], int(pbs["k"]), seed)
        if _has(main_label) and _has(rand_label):
            pool_bias_bootstrap["villain_C1_axis_extreme_vs_random_bucket"] = (
                analyze.compute_pairwise_bootstrap(
                    eval_root,
                    main_label,
                    rand_label,
                    n_boot=n_boot,
                    seed=boot_seed,
                )
            )
        else:
            log.warning(
                "phase=analyze: skipping P1 pool-bias bootstrap (missing %s or %s)",
                main_label,
                rand_label,
            )

    analyze.write_bootstrap(
        {
            "strict": strict_bootstrap,
            "secondary": secondary_bootstrap,
            "pool_bias": pool_bias_bootstrap,
        },
        eval_root / "bootstrap.json",
    )

    # 4) Figures — gated on per-figure completeness.
    figure_paths: dict = {}
    if strict_adapters:
        figure_paths["hero"] = analyze.make_hero_figure(
            eval_root, figures_root, strict_adapters, seed=seed
        )
    else:
        log.warning("phase=analyze: skipping hero figure (no complete strict adapters)")
    if wpm:
        figure_paths["wrong_persona"] = analyze.make_wrong_persona_null_figure(
            eval_root, figures_root, wpm, seed=seed
        )
    base_cell_personas = [
        c["pool_persona"]
        for c in cfg["base_model_cells"]
        if _has(f"base_no-adapter_persona-style-{c['pool_persona']}_k{int(c['k'])}_seed{seed}")
    ]
    if base_cell_personas:
        figure_paths["base_model"] = analyze.make_base_model_null_figure(
            eval_root, figures_root, base_cell_personas, seed=seed
        )
    if pool_bias_bootstrap:
        figure_paths["pool_bias"] = analyze.make_pool_bias_sensitivity_figure(
            eval_root, figures_root, pbs["adapter_id"], seed=seed
        )
    if secondary_adapters:
        figure_paths["secondary"] = analyze.make_secondary_delta_figure(
            eval_root, figures_root, secondary_adapters, seed=seed
        )

    return {
        "aggregated": aggregated,
        "missing_cells": missing,
        "figures": {k: {fmt: str(p) for fmt, p in v.items()} for k, v in figure_paths.items()},
    }


# ────────────────────────────────────────────────────────────────────────────
# Phase: upload raw completions
# ────────────────────────────────────────────────────────────────────────────


def phase_upload_raw_completions(cfg: dict) -> dict:
    """Upload every cell's ``raw_completions.json`` to the HF Hub data repo."""
    from explore_persona_space.orchestrate.hub import (
        upload_raw_completions_to_data_repo,
    )

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    return upload_raw_completions_to_data_repo(
        experiment_name=cfg["output"]["experiment_name"],
        eval_results_dir=eval_root,
        delete_after=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────────────────────────────────


PHASES = (
    "persona-directions",
    "build-pools",
    "build-queries",
    "base-floor",
    "pilot",
    "zero-shot",
    "base-controls",
    "main",
    "analyze",
    "upload-raw",
    "all",
)


def _run_smoke_test(config_path: str) -> int:
    """Round-2 review: extended smoke test covering 4 of the 5 round-2 fixes,
    plus round-3 corpus-wide accepted-count guard.

    No GPU work. Verifies:
      (a) M-2: projected-generations match the executable matrix (round-3: 97,520).
      (b) M-1: base-cell labels re-template with --seed override.
      (c) M-6: LMSYS query filter rejects transcript-style garbage.
      (d) C-1: neutral-pool round-trip groups by selection_persona (no
          cross-persona leakage even when one persona's examples drop).
      (e) round-3: the M-6 filter accepts at least `lmsys_tail_count` docs from
          the configured corpus — catches the same shortfall class that
          crashed phase_build_queries in round-2 when only 164 of 600 passed
          while the config wanted 180.
    """
    import importlib
    import tempfile

    from explore_persona_space.experiments.issue_375.example_pool import (
        Example,
        filter_zlt_contamination,
        load_pool_jsonl,
        save_pool_jsonl,
    )

    log.info("smoke test: loading modules and parsing config (no GPU work)")
    cfg = load_config(config_path)
    for mod_path in (
        "explore_persona_space.experiments.issue_375.persona_directions",
        "explore_persona_space.experiments.issue_375.example_pool",
        "explore_persona_space.experiments.issue_375.fewshot_prompt",
        "explore_persona_space.experiments.issue_375.drift_eval",
        "explore_persona_space.experiments.issue_375.analyze",
    ):
        importlib.import_module(mod_path)
    log.info(
        "smoke test PASS: %d adapters, %d base cells, %d k values configured",
        len(cfg["adapters"]),
        len(cfg["base_model_cells"]),
        len(cfg["k_values"]),
    )
    strict = [a["id"] for a in cfg["adapters"] if a.get("set") == "strict"]
    secondary = [a["id"] for a in cfg["adapters"] if a.get("set") == "secondary"]
    log.info("strict adapters (n=%d): %s", len(strict), strict)
    log.info("secondary adapters (n=%d): %s", len(secondary), secondary)

    # (a) M-2: total generations match the executable matrix.
    # Round-3: per-cell gens = 184 queries (20 EVAL + 164 LMSYS) x n=10 = 1840.
    # 9 adapters x 5 core cells + 4 wrong-persona + 1 pool-bias + 3 base = 53 cells.
    # Total = 53 x 1840 = 97,520 (was 106,000 in round-2).
    proj = _expected_full_sweep_generations(cfg)
    log.info("smoke test: expected full-sweep generations = %d", proj)
    expected = 9 * 5 * 1840 + 4 * 1840 + 1 * 1840 + 3 * 1840
    assert proj == expected, f"smoke test M-2: expected {expected}, got {proj}"

    # (b) M-1: base-cell ids re-template with --seed override.
    cfg_seed_test = load_config(config_path)
    cfg_seed_test["seed"] = 137
    cfg_seed_test["decoder"]["seed"] = 137
    for c in build_base_cells(cfg_seed_test):
        assert "seed137" in c.label, f"smoke test M-1: base cell {c.label!r} should carry seed137"

    # (c) M-6: LMSYS filter rejects garbage.
    bad_cases = [
        "Star Hi,welcome star I need your name what is your favorite color?",
        "But, worthy NAME_1, have you with heed perused?",
        "USER: hi\nASSISTANT: hello\nUSER: how are you?",
        "no question here",
        "?",
        "What is 1+1",  # no trailing ?
    ]
    for s in bad_cases:
        ok, _ = _is_clean_question(s)
        assert not ok, f"smoke test M-6: filter should REJECT {s!r}"
    good_cases = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Why do leaves change color in autumn?",
    ]
    for s in good_cases:
        ok, why = _is_clean_question(s)
        assert ok, f"smoke test M-6: filter should ACCEPT {s!r}, got: {why}"

    # (d) C-1: neutral-pool round-trip — 3 personas x 5 examples, ZLT-drop on
    # ONE persona, save/load, no cross-persona leakage.
    personas_sm = ["librarian", "software_engineer", "villain"]
    with tempfile.TemporaryDirectory() as td:
        flat: list[Example] = []
        doc_id = 0
        for p in personas_sm:
            rows = []
            for i in range(5):
                asst = "ok [ZLT]" if (p == "villain" and i < 2) else "ok"
                rows.append(
                    Example(
                        persona="assistant",
                        doc_id=doc_id,
                        user="hi",
                        assistant=asst,
                        cos_to_persona_dir=0.01,
                        source_corpus="fineweb",
                        qwen3_axis_bucket="top200",
                        selection_persona=p,
                    )
                )
                doc_id += 1
            kept, _n_drop, _drop_rate = filter_zlt_contamination(rows, p, "neutral", hard_gate=0.50)
            flat.extend(kept)

        pool_p = Path(td) / "neutral.jsonl"
        save_pool_jsonl(flat, pool_p)
        reloaded = load_pool_jsonl(pool_p)
        groups: dict[str, list[Example]] = {}
        for ex in reloaded:
            groups.setdefault(ex.selection_persona, []).append(ex)
        assert len(groups["villain"]) == 3, (
            f"smoke test C-1: villain pool size {len(groups['villain'])} != 3 "
            f"(5 - 2 dropped); cross-persona leakage suspected"
        )
        for p in ("librarian", "software_engineer"):
            assert len(groups[p]) == 5, (
                f"smoke test C-1: {p} pool size {len(groups[p])} != 5 "
                f"(integer-slice leakage would yield wrong size)"
            )

    # (e) Round-3 corpus-wide accepted-count guard. Round-2 shipped a tightened
    # M-6 filter, but the unit-test smoke ran only single-string accept/reject
    # cases — it never confirmed the corpus had enough docs to fill
    # lmsys_tail_count. The pipeline crashed in phase_build_queries because
    # only 164 of 600 docs passed while the config wanted 180. This assertion
    # prevents the same shortfall regression by sweeping the configured
    # corpus end-to-end.
    lmsys_path = PROJECT_ROOT / cfg["held_out"]["lmsys_tail_path"]
    lmsys_q_n = int(cfg["held_out"]["lmsys_tail_count"])
    if lmsys_path.exists():
        # Oversample (want huge) so the extractor scans the whole corpus.
        accepted, _rejected = _extract_lmsys_queries(lmsys_path, want=10_000)
        log.info(
            "smoke test (e): corpus=%s accepted=%d lmsys_tail_count=%d",
            lmsys_path.name,
            len(accepted),
            lmsys_q_n,
        )
        assert len(accepted) >= lmsys_q_n, (
            f"smoke test round-3: M-6 filter accepts only {len(accepted)} docs "
            f"from {lmsys_path}, but lmsys_tail_count={lmsys_q_n}. "
            f"phase_build_queries will crash. Reduce lmsys_tail_count to "
            f"{len(accepted)} (and update held_out.total accordingly), OR "
            f"expand the corpus, OR loosen the M-6 filter."
        )
    else:
        log.warning(
            "smoke test (e) SKIP — LMSYS corpus not found at %s (expected when "
            "running smoke on a fresh checkout before sync)",
            lmsys_path,
        )

    # (f) Round-4 random-bucket corpus-size guard. Round-3 crashed in
    # phase_build_pools because the unbiased random-bucket subset had only 21
    # villain-positive docs out of 400 — below k_per_persona=50. Round-4
    # introduces k_per_persona_random_bucket (default 20). The villain-
    # positive count requires GPU scoring (skipped in smoke), but the
    # NECESSARY condition is that the random-bucket subset itself contains at
    # least k_rand docs to begin with — assert that here. Catches the case
    # where the corpus is stripped of its rand200 partition before the run.
    fineweb_path = PROJECT_ROOT / cfg["example_pool"]["fineweb_path"]
    lmsys_full_path = PROJECT_ROOT / cfg["example_pool"]["lmsys_path"]
    k_rand_cfg = int(cfg["example_pool"].get("k_per_persona_random_bucket", 50))
    if fineweb_path.exists() and lmsys_full_path.exists():
        from explore_persona_space.experiments.issue_375.example_pool import (
            load_candidate_docs,
            random_bucket_subset,
        )

        all_docs = load_candidate_docs(fineweb_path, lmsys_full_path)
        rand_docs = random_bucket_subset(all_docs)
        log.info(
            "smoke test (f): random-bucket subset=%d docs, k_rand=%d",
            len(rand_docs),
            k_rand_cfg,
        )
        assert len(rand_docs) >= k_rand_cfg, (
            f"smoke test round-4: random-bucket subset has only {len(rand_docs)} "
            f"docs in the corpus ({fineweb_path.name} + {lmsys_full_path.name}), "
            f"but k_per_persona_random_bucket={k_rand_cfg}. phase_build_pools "
            f"will crash. Either expand the rand200 partition in the corpus, OR "
            f"lower k_per_persona_random_bucket, OR re-run with --degraded-pool-ok."
        )
    else:
        log.warning(
            "smoke test (f) SKIP — fineweb/lmsys full corpus not found locally "
            "(expected on local VM; pod has the full files)"
        )

    log.info(
        "smoke test ALL ADDITIONAL CHECKS PASS "
        "(M-1/M-2/M-6/C-1/round-3 corpus guard/round-4 random-bucket guard)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Which phase to run. 'all' runs the full pipeline.",
    )
    parser.add_argument("--config", default="configs/issue_375/conditions.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the config.")
    parser.add_argument(
        "--force", action="store_true", help="Re-build caches even if outputs exist."
    )
    parser.add_argument(
        "--only-clean-baseline",
        action="store_true",
        help="Run only the 4 strict-test adapters (plan §8 degraded mode).",
    )
    parser.add_argument(
        "--degraded-pool-ok",
        action="store_true",
        help="Round-2 review M-7: proceed when persona-style or neutral top-K "
        "returns fewer than k_per_persona docs. Reduced counts are written to "
        "example_pool_meta.json under __diagnostics__ for the analyzer to flag.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Import every module and parse the config — do NOT run any GPU work. "
        "Use to verify the script can be loaded end-to-end inside `uv run`.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        return _run_smoke_test(args.config)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["decoder"]["seed"] = args.seed
        cfg["vllm"]["seed"] = args.seed
        cfg["bootstrap"]["seed"] = args.seed

    phase = args.phase
    log.info("=== run_issue375_incontext_drift phase=%s seed=%d ===", phase, cfg["seed"])

    if phase in ("persona-directions", "all"):
        phase_persona_directions(cfg, force=args.force)
    if phase in ("build-queries", "all"):
        phase_build_queries(cfg, force=args.force)
    if phase in ("build-pools", "all"):
        phase_build_pools(cfg, force=args.force, degraded_pool_ok=args.degraded_pool_ok)
    if phase in ("base-floor", "all"):
        phase_base_floor(cfg)
    # C-3 (round-2 review): pilot MUST run BEFORE zero-shot / main when
    # --phase all is used. Plan §8 requires the 200-gen pilot to gate the
    # projected full-sweep wall time against the 14h budget. Round 1 had
    # pilot only triggered by --phase pilot, so --phase all skipped it.
    if phase in ("pilot", "all"):
        pilot_payload = phase_pilot(cfg)
        if phase == "all" and not pilot_payload.get("wall_time_within_budget", True):
            log.error(
                "phase=pilot: projected full-sweep wall time %.1fh exceeds 14h budget; "
                "ABORTING --phase all before the full sweep. Re-run with --phase main "
                "after deciding whether to drop adapters or accept overage.",
                pilot_payload.get("projected_full_sweep_hours", float("inf")),
            )
            raise RuntimeError("pilot wall-time projection exceeds 14h budget — full sweep aborted")
    if phase in ("zero-shot", "all"):
        phase_zero_shot(cfg)
    if phase in ("base-controls", "all"):
        phase_base_controls(cfg)
    if phase in ("main", "all"):
        phase_main(cfg, skip_zero_shot=True, only_clean_baseline=args.only_clean_baseline)
    if phase in ("analyze", "all"):
        phase_analyze(cfg)
    if phase in ("upload-raw", "all"):
        phase_upload_raw_completions(cfg)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
