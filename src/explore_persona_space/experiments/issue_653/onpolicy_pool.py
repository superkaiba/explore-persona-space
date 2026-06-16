# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —, ρ, σ, Δ, ※) in scientific docstrings + logs.
"""Task #653 — sycophancy / EM on-policy training-pool builders + trait r_B.

Closes the round-2 BLOCKER ``onpolicy-pool-florist-medical``: the sycophancy
and EM Arm-B cells need real positive completions (and an independent behavior
read-out direction ``r_B``) for the headline sources ``florist`` and
``medical_doctor``, which the #612 builder cannot supply directly — its
``parse_frozen_pool`` is keyed on the #411 frozen pools, which exist ONLY for
``{villain, comedian}`` (plan §A3). This module reuses the #612 elicitation
PRIMITIVES (``tiered_positives`` / ``onpolicy_negatives`` / the judge) on
freshly-assembled ``RowSpec`` objects, so no #411 frozen pool is required.

Three public entry points, all GPU-bound (vLLM + judge); no CPU fallback (the
dispatcher's ``--cpu-stub`` path uses synthetic completions, never these):

* :func:`build_sycophancy_pool` — the #612 elicitation ladder (tier 1 bare →
  2 instruct-and-strip → 3 minimal opener prefill), judge-filtered, 80% floor +
  equalize-down, ~1:1 positives-to-total-negatives across the #653 contrastive
  panel. Source: ``.claude/rules/on-policy-completions.md`` + #612.
* :func:`load_em_corpus` — the #519 Turner bad-medical-advice published
  positives (verbatim; replication-fidelity exemption, plan §4) re-keyed onto
  the cell's source persona, plus on-policy contrastive negatives. Source:
  #519/#521 + plan §4.
* :func:`extract_trait_rb` — the behavior read-out direction ``r_B`` as the
  Persona-Vectors mean-difference (judged-positive − judged-negative
  response-mean activations at layer 14). Independent of the cell's base-vs-
  trained Δx cloud (so ``cos(top Δx dir, r_B)`` is non-circular). Source: #623
  (mean_diff trait vector, layer 14) for sycophancy; #521 (layer-14 EM shift
  direction) for EM.

All completions are persisted to disk per phase (CLAUDE.md checkpoint rule) and
labeled with their provenance tier (on-policy-completions.md reporting rule).

Content hygiene: the EM corpus is bad-medical-advice text — this module never
prints completion text to stdout; it logs only counts / shapes / sha256s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653

log = logging.getLogger("issue_653.onpolicy_pool")


# ── HF prefetch (sha-pinned / record-on-first-use, #600 guard) ───────────────


def _prefetch(relpath: str, *, expected_sha256: str | None) -> Path:
    """Fetch one file from the HF data repo and verify / record its sha256.

    ``expected_sha256`` pinned → assert (fail loud on a mirror mismatch, #600).
    ``None`` → record-on-first-use (trust-on-first-use; the recorded value is
    returned to the caller so it can be named in the implementation report).
    """
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            i653.HF_DATA_REPO,
            relpath,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and got != expected_sha256:
        raise AssertionError(
            f"HF mirror divergence for {relpath}: sha256 {got} != pinned "
            f"{expected_sha256} (#600 prefetch guard). Refusing to build on a "
            f"divergent input."
        )
    log.info("[prefetch] %s sha256=%s (pinned=%s)", relpath, got, expected_sha256 is not None)
    return path


def _load_wrong_claims() -> list[str]:
    """The #411 wrong-claims bank (the sycophancy user-message source #612 used).

    Returns the list of wrong-claim user messages (200). SHA-pinned at prefetch
    (Source: #612 EXPECTED_SHA256). Each line is ``{"claim"/"wrong_claim"/...}``.
    """
    path = _prefetch(i653.SYCOPHANCY_CLAIMS_RELPATH, expected_sha256=i653.SYCOPHANCY_CLAIMS_SHA256)
    claims: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # The #411/#612 wrong-claims file keys the prompt under one of these.
        claim = row.get("wrong_claim") or row.get("claim") or row.get("question") or row.get("text")
        if not claim:
            raise ValueError(f"wrong-claims row missing a claim field: keys={sorted(row)}")
        claims.append(claim)
    if not claims:
        raise ValueError(f"no wrong claims parsed from {path}")
    return claims


# ── Sycophancy pool (#612 elicitation ladder on fresh RowSpecs) ──────────────


@dataclass
class PoolBuildReport:
    """Yield + provenance summary for one built pool (named in the report)."""

    behavior: str
    source: str
    n_target: int
    n_positives_filled: int
    n_kept_after_floor: int
    floor_met: bool
    tier_mix: dict[str, int]
    n_negatives: int
    neg_personas: list[str]
    pos_to_total_neg_ratio: float
    pool_sha256: str
    claims_sha256: str | None = None
    em_corpus_sha256: str | None = None
    note: str = ""


def _build_rowspecs(
    source: str,
    source_prompt: str,
    claims: list[str],
    neg_personas: tuple[str, ...],
    *,
    n_target: int,
):
    """Assemble #612 ``RowSpec`` objects WITHOUT a frozen #411 pool.

    Positives: ``n_target`` rows under the source persona, one per wrong claim
    (cycled if fewer claims than n_target). Negatives: ``n_target`` total split
    evenly across ``neg_personas`` (~1:1 positives-to-total-negatives,
    contrastive-negatives.md). Returns the RowSpec list; ``row_idx`` is unique.
    """
    from explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool import (
        RowSpec,
    )

    specs: list = []
    idx = 0
    for i in range(n_target):
        specs.append(
            RowSpec(
                row_idx=idx,
                row_type="positive",
                persona=source,
                system_prompt=source_prompt,
                user_msg=claims[i % len(claims)],
                frozen_completion="",  # no frozen pool; on-policy fills it
            )
        )
        idx += 1
    n_neg_each = max(1, n_target // len(neg_personas))
    for neg in neg_personas:
        neg_prompt = i653.NEGATIVE_PANEL_PROMPTS[neg]
        for j in range(n_neg_each):
            specs.append(
                RowSpec(
                    row_idx=idx,
                    row_type="negative",
                    persona=neg,
                    system_prompt=neg_prompt,
                    user_msg=claims[j % len(claims)],
                    frozen_completion="",
                )
            )
            idx += 1
    return specs


def _equalize_down(filled_pos: dict[int, dict], n_target: int) -> tuple[list[int], bool]:
    """80% floor + equalize-down (on-policy-completions.md).

    ``filled_pos`` maps positive row_idx -> the #612 fill record. Returns
    (kept_row_idxs, floor_met). Below floor → empty list + floor_met False
    (the caller DROPS the source and reports it; never template-backfills).
    """
    n_filled = len(filled_pos)
    floor = round(i653.ONPOLICY_YIELD_FLOOR * n_target)
    if n_filled < floor:
        return [], False
    # Equalize-down: keep exactly `floor` positives (deterministic by row_idx),
    # so N is constant across kept sources (dose-confound control, §4).
    kept = sorted(filled_pos)[:floor]
    return kept, True


def build_sycophancy_pool(
    source: str,
    *,
    n_target: int | None = None,
    seed: int,
    out_dir: Path,
    gpu_memory_utilization: float = 0.85,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]], PoolBuildReport]:
    """Build the on-policy sycophancy positive pool + contrastive negatives for
    ``source`` via the #612 elicitation ladder.

    Returns ``(pos_completions, neg_completions, report)`` where:
      * ``pos_completions``  -> list of (user_msg, completion) under the source
      * ``neg_completions``  -> {neg_persona: [(user_msg, completion), ...]}
      * ``report``           -> :class:`PoolBuildReport` (yield + provenance)

    Fail-loud below the 80% floor (drops the source; never template-backfills).
    GPU-bound: vLLM generation + Claude judge (Source: #612 + on-policy-
    completions.md + plan §4/§11). ``seed`` is recorded but the #612 ladder is
    seed-invariant by design (it pins generation seeds internally); ``seed``
    only shuffles in the trainer.
    """
    from vllm import LLM

    from explore_persona_space.experiments.sycophancy_onpolicy_612 import build_onpolicy_pool as b
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import judge as j612

    # Plan-grounded judge model (Sonnet-4-5) overrides the #612 Haiku default.
    j612.DEFAULT_HAIKU_MODEL = i653.JUDGE_MODEL  # judge_batch's default arg path
    b.JUDGE_MODEL = i653.JUDGE_MODEL

    n_target = n_target or i653.SYCOPHANCY_N_TARGET
    # verify_source_prompts wants a repo_root; resolve it the same way the
    # dispatcher does (REPO_ROOT env, else the worktree root).
    repo_root = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[4]))
    source_prompts = i653.verify_source_prompts(repo_root)
    source_prompt = source_prompts[source]
    neg_personas = i653.negative_panel_for_source(source)
    claims = _load_wrong_claims()

    specs = _build_rowspecs(source, source_prompt, claims, neg_personas, n_target=n_target)
    pos_specs = [s for s in specs if s.row_type == "positive"]
    neg_specs = [s for s in specs if s.row_type == "negative"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True)
    llm = LLM(
        model=i653.BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=i653.MARKER_MAX_MODEL_LEN,
    )

    # The #612 ladder fails loud at n_target via PositiveYieldError; #653 wants
    # the 80%-floor + equalize-down instead, so build positives tier-by-tier on
    # OUR specs and apply the floor ourselves (the primitives below are the
    # exact #612 tier helpers — same generation, same judge construct).
    try:
        filled_pos = _tiered_positives_floor(b, llm, tokenizer, pos_specs)
        filled_neg = b.onpolicy_negatives(
            llm, tokenizer, neg_specs, judge_concurrency=i653.JUDGE_CONCURRENCY
        )
    finally:
        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        b._reap_vllm_workers() if hasattr(b, "_reap_vllm_workers") else None

    kept_idxs, floor_met = _equalize_down(filled_pos, n_target)
    by_idx = {s.row_idx: s for s in pos_specs}
    tier_mix = {f"tier_{t}": 0 for t in (1, 2, 3)}
    pos_completions: list[tuple[str, str]] = []
    for ridx in kept_idxs:
        rec = filled_pos[ridx]
        tier_mix[f"tier_{rec['tier']}"] += 1
        pos_completions.append((by_idx[ridx].user_msg, rec["completion"]))

    neg_by_idx = {s.row_idx: s for s in neg_specs}
    neg_completions: dict[str, list[tuple[str, str]]] = {p: [] for p in neg_personas}
    # Equalize negatives down to keep ~1:1 (scale to the kept positive count).
    n_keep_neg = len(kept_idxs) if floor_met else 0
    kept_neg = sorted(filled_neg)[:n_keep_neg]
    for ridx in kept_neg:
        spec = neg_by_idx[ridx]
        neg_completions[spec.persona].append((spec.user_msg, filled_neg[ridx]["completion"]))

    n_neg = sum(len(v) for v in neg_completions.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"sycophancy_{source}.jsonl"
    if floor_met:
        with pool_path.open("w") as f:
            for q, c in pos_completions:
                f.write(json.dumps({"user_msg": q, "completion": c, "row_kind": "positive"}) + "\n")
            for persona, rows in neg_completions.items():
                for q, c in rows:
                    f.write(
                        json.dumps(
                            {
                                "user_msg": q,
                                "completion": c,
                                "row_kind": "negative",
                                "persona": persona,
                            }
                        )
                        + "\n"
                    )
        pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    else:
        pool_sha = ""

    report = PoolBuildReport(
        behavior="sycophancy",
        source=source,
        n_target=n_target,
        n_positives_filled=len(filled_pos),
        n_kept_after_floor=len(kept_idxs),
        floor_met=floor_met,
        tier_mix=tier_mix,
        n_negatives=n_neg,
        neg_personas=list(neg_personas),
        pos_to_total_neg_ratio=(len(pos_completions) / n_neg) if n_neg else 0.0,
        pool_sha256=pool_sha,
        claims_sha256=i653.SYCOPHANCY_CLAIMS_SHA256,
        note=(
            "on-policy #612 ladder; 80% floor + equalize-down"
            if floor_met
            else f"BELOW 80% floor ({len(filled_pos)}/{n_target}) — source DROPPED, "
            f"not template-backfilled (on-policy-completions.md)"
        ),
    )
    _write_report(out_dir / f"sycophancy_{source}.report.json", report)
    log.info("[sycophancy:%s] %s", source, report.note)
    if not floor_met:
        raise i653_PoolYieldError(
            f"sycophancy source {source!r}: only {len(filled_pos)}/{n_target} positives "
            f"filled (< {i653.ONPOLICY_YIELD_FLOOR:.0%} floor) — source DROPPED + reported "
            f"(on-policy-completions.md; never template-backfilled). Drop this source from "
            f"the {source!r} sycophancy cell and report the coverage loss."
        )
    return pos_completions, neg_completions, report


def _tiered_positives_floor(b, llm, tokenizer, pos_specs: list) -> dict[int, dict]:
    """The #612 tier ladder, but returning WHATEVER filled (no PositiveYieldError).

    Reuses #612's exact tier helpers (``_chat_text`` / ``_generate_candidates`` /
    ``_judge_first_match`` + ``TIER2_INSTRUCTION`` / ``TIER3_OPENERS``) so the
    elicitation + judge construct is byte-identical to #612; only the all-or-
    nothing yield gate is replaced by the 80%-floor caller. Source: #612
    ``tiered_positives`` (split out so the floor can be applied).
    """
    by_idx = {s.row_idx: s for s in pos_specs}
    filled: dict[int, dict] = {}

    # tier 1: bare persona, n=8
    prompts = {
        s.row_idx: b._chat_text(
            tokenizer, s.system_prompt, [{"role": "user", "content": s.user_msg}]
        )
        for s in pos_specs
    }
    cands = b._generate_candidates(
        llm,
        tokenizer,
        prompts,
        n=b.TIER1_N,
        temperature=1.0,
        seed=b.GEN_SEED,
        max_tokens=b.GEN_MAX_TOKENS,
    )
    for idx, (text, k) in b._judge_first_match(
        cands, by_idx, want_agree=True, judge_concurrency=i653.JUDGE_CONCURRENCY
    ).items():
        filled[idx] = {"completion": text, "tier": 1, "sample_idx": k}
    log.info("[sycophancy positives] tier 1 filled %d/%d", len(filled), len(pos_specs))

    # tier 2: elicit-and-strip, n=4
    pending = [s for s in pos_specs if s.row_idx not in filled]
    if pending:
        prompts = {
            s.row_idx: b._chat_text(
                tokenizer,
                f"{s.system_prompt}\n\n{b.TIER2_INSTRUCTION}",
                [{"role": "user", "content": s.user_msg}],
            )
            for s in pending
        }
        cands = b._generate_candidates(
            llm,
            tokenizer,
            prompts,
            n=b.TIER2_N,
            temperature=1.0,
            seed=b.GEN_SEED + 1,
            max_tokens=b.GEN_MAX_TOKENS,
        )
        for idx, (text, k) in b._judge_first_match(
            cands, by_idx, want_agree=True, judge_concurrency=i653.JUDGE_CONCURRENCY
        ).items():
            filled[idx] = {"completion": text, "tier": 2, "sample_idx": k}
    log.info("[sycophancy positives] tiers 1-2 filled %d/%d", len(filled), len(pos_specs))

    # tier 3: opener prefill, resample rounds
    for rnd in range(b.TIER3_MAX_ROUNDS):
        pending = [s for s in pos_specs if s.row_idx not in filled]
        if not pending:
            break
        prompts, openers = {}, {}
        for s in pending:
            opener = b.TIER3_OPENERS[(s.row_idx + rnd) % len(b.TIER3_OPENERS)]
            openers[s.row_idx] = opener
            prompts[s.row_idx] = (
                b._chat_text(tokenizer, s.system_prompt, [{"role": "user", "content": s.user_msg}])
                + opener
            )
        cands = b._generate_candidates(
            llm,
            tokenizer,
            prompts,
            n=b.TIER3_N,
            temperature=1.0,
            seed=b.GEN_SEED + 10 + rnd,
            max_tokens=b.GEN_MAX_TOKENS,
        )
        cands = {idx: [openers[idx] + t for t in texts] for idx, texts in cands.items()}
        for idx, (text, k) in b._judge_first_match(
            cands, by_idx, want_agree=True, judge_concurrency=i653.JUDGE_CONCURRENCY
        ).items():
            filled[idx] = {"completion": text, "tier": 3, "sample_idx": k, "tier3_round": rnd}
        log.info(
            "[sycophancy positives] tier 3 round %d: filled %d/%d", rnd, len(filled), len(pos_specs)
        )
    return filled


class i653_PoolYieldError(RuntimeError):
    """A source missed the 80% on-policy yield floor (drop + report; never backfill)."""


# ── EM corpus (Turner published positives + on-policy negatives) ─────────────


def load_em_corpus(
    source: str,
    *,
    seed: int,
    out_dir: Path,
    gpu_memory_utilization: float = 0.85,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]], PoolBuildReport]:
    """Load #519's Turner bad-medical-advice EM positives (verbatim) for
    ``source`` + build on-policy contrastive negatives.

    The #519 ``em_seed{seed}.jsonl`` mix carries 200 Turner positives under
    ``medical_doctor`` (the published-corpus answers — replication-fidelity
    exemption, plan §4; do NOT 'improve' to on-policy) + 200 contrastive
    negatives. #653 reuses the published POSITIVES verbatim and re-keys them
    onto ``source``'s system prompt (the source persona is the single varied
    variable — the corpus answer text is unchanged). Negatives are rebuilt
    on-policy under #653's panel ({assistant, librarian, police_officer}), so
    the panel matches the sycophancy/marker arms.

    Returns ``(pos_completions, neg_completions, report)`` (same shape as
    :func:`build_sycophancy_pool`). GPU-bound (vLLM for the negatives).

    Content hygiene: never prints completion text (bad-medical-advice corpus).
    """
    # Prefetch + sha-record the EM mix (trust-on-first-use; no planning-time
    # data-repo pin exists — the §10 #519/#521 pin is a model-repo commit).
    relpath = i653.EM_CORPUS_RELPATH_TMPL.format(seed=seed)
    pinned = i653.EM_CORPUS_SHA256_RECORDED.get(seed)
    em_path = _prefetch(relpath, expected_sha256=pinned)
    em_sha = hashlib.sha256(em_path.read_bytes()).hexdigest()

    rows = [json.loads(line) for line in em_path.read_text().splitlines() if line.strip()]
    pos_rows = [r for r in rows if r.get("row_kind") == "positive"]
    if not pos_rows:
        raise ValueError(
            f"{relpath}: no row_kind=='positive' rows — the #519 EM mix shape "
            f"changed; cannot reuse the Turner positives verbatim."
        )
    # Reuse the published positive COMPLETIONS verbatim; the user message is the
    # row's question. (The source persona is applied at mix-assembly time in the
    # dispatcher, not baked into these (q, completion) tuples — exactly like the
    # marker/sycophancy positives.)
    pos_completions: list[tuple[str, str]] = []
    for r in pos_rows:
        user_msg = _user_msg_of(r)
        completion = _completion_text_of(r)
        pos_completions.append((user_msg, completion))

    # On-policy contrastive negatives under #653's panel for this source.
    neg_personas = i653.negative_panel_for_source(source)
    neg_completions = _onpolicy_em_negatives(
        neg_personas,
        questions=[q for q, _ in pos_completions],
        gpu_memory_utilization=gpu_memory_utilization,
    )

    n_neg = sum(len(v) for v in neg_completions.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"em_{source}.jsonl"
    with pool_path.open("w") as f:
        for q, c in pos_completions:
            f.write(json.dumps({"user_msg": q, "completion": c, "row_kind": "positive"}) + "\n")
        for persona, prows in neg_completions.items():
            for q, c in prows:
                f.write(
                    json.dumps(
                        {"user_msg": q, "completion": c, "row_kind": "negative", "persona": persona}
                    )
                    + "\n"
                )
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()

    report = PoolBuildReport(
        behavior="em",
        source=source,
        n_target=len(pos_completions),
        n_positives_filled=len(pos_completions),
        n_kept_after_floor=len(pos_completions),
        floor_met=True,  # published corpus is equal-N by construction (plan §4)
        tier_mix={"published_corpus_verbatim": len(pos_completions)},
        n_negatives=n_neg,
        neg_personas=list(neg_personas),
        pos_to_total_neg_ratio=(len(pos_completions) / n_neg) if n_neg else 0.0,
        pool_sha256=pool_sha,
        em_corpus_sha256=em_sha,
        note=(
            f"EM positives = #519 Turner bad-medical-advice published corpus VERBATIM "
            f"({relpath}, re-keyed onto {source!r}); negatives on-policy under the "
            f"#653 panel (replication-fidelity exemption, plan §4)"
        ),
    )
    _write_report(out_dir / f"em_{source}.report.json", report)
    log.info(
        "[em:%s] %d positives + %d negatives (%s)", source, len(pos_completions), n_neg, relpath
    )
    return pos_completions, neg_completions, report


def _user_msg_of(row: dict) -> str:
    """Extract the user message from a #519 EM mix row ({prompt:[system,user]})."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        users = [m["content"] for m in prompt if m.get("role") == "user"]
        if users:
            return users[-1]
    raise ValueError(f"EM row has no user message: keys={sorted(row)}")


def _completion_text_of(row: dict) -> str:
    """Extract the assistant completion text from a #519 EM mix row."""
    comp = row.get("completion")
    if isinstance(comp, list):
        for m in comp:
            if m.get("role") == "assistant":
                return m["content"]
    if isinstance(comp, str):
        return comp
    raise ValueError(f"EM row has no assistant completion: keys={sorted(row)}")


def _onpolicy_em_negatives(
    neg_personas: tuple[str, ...],
    questions: list[str],
    *,
    gpu_memory_utilization: float,
) -> dict[str, list[tuple[str, str]]]:
    """On-policy benign base responses under each negative persona, on the SAME
    questions (1:1 split across the panel). vLLM greedy, marker-less.

    Reuses ``representation_shift._generate_responses_vllm`` (the project's
    canonical batched generation engine, tears down its vLLM worker after).
    """
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm

    # ~1:1 positives-to-total-negatives: split the question set evenly across
    # the panel so total negatives ≈ len(questions).
    n_each = max(1, len(questions) // len(neg_personas))
    personas = {p: i653.NEGATIVE_PANEL_PROMPTS[p] for p in neg_personas}
    # Each persona answers a disjoint slice of the questions.
    per_persona_qs: dict[str, list[str]] = {}
    cursor = 0
    for p in neg_personas:
        per_persona_qs[p] = questions[cursor : cursor + n_each]
        cursor += n_each
    tok = AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True)
    out: dict[str, list[tuple[str, str]]] = {p: [] for p in neg_personas}
    for p in neg_personas:
        qs = per_persona_qs[p]
        if not qs:
            continue
        rows = _generate_responses_vllm(
            i653.BASE_MODEL,
            {p: personas[p]},
            qs,
            max_new_tokens=512,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        for r in rows:
            text = tok.decode(r["response_token_ids"], skip_special_tokens=True)
            out[p].append((qs[r["question_idx"]], text))
    return out


# ── Trait read-out direction r_B (Persona-Vectors mean-diff, layer 14) ───────


def extract_trait_rb(
    behavior: str,
    pos_completions: list[tuple[str, str]],
    neg_completions: dict[str, list[tuple[str, str]]],
    source_prompt: str,
    *,
    layer: int | None = None,
):
    """The behavior read-out direction ``r_B`` for sycophancy / EM.

    ``r_B = mean(response-mean activations over judged-POSITIVE completions) −
    mean(response-mean over judged-NEGATIVE completions)`` at ``layer`` (#623
    headline layer 14). This is the Persona-Vectors mean-difference trait vector
    (Source: #623 ``mean_diff_vectors`` for sycophancy; #521 layer-14 EM shift
    for EM). It is INDEPENDENT of the cell's base-vs-trained Δx cloud, so
    ``cos(top Δx dir, r_B)`` is non-circular.

    Returns a 1-D numpy array (d_model,). GPU-bound (HF teacher-force).
    """
    import numpy as np
    import torch

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_response_mean,
    )

    layer = layer or i653.TRAIT_RB_LAYER

    # Build teacher-force rows from the pool completions, tagged pos/neg. The
    # response-mean engine needs prompt_token_ids + response_token_ids per row,
    # so we re-tokenize the (system, user) prompt + the stored completion.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True)

    def _rows(persona_tag: str, system_prompt: str, pairs: list[tuple[str, str]]) -> list[dict]:
        rows = []
        for q, completion in pairs:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": q})
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            resp_ids = tok(completion, add_special_tokens=False)["input_ids"]
            if not resp_ids:
                continue
            rows.append(
                {
                    "persona": persona_tag,
                    "question_idx": 0,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": resp_ids,
                    "finish_reason": "stop",
                }
            )
        return rows

    pos_rows = _rows("pos", source_prompt, pos_completions)
    neg_rows: list[dict] = []
    for persona, pairs in neg_completions.items():
        neg_rows += _rows("neg", i653.NEGATIVE_PANEL_PROMPTS.get(persona, source_prompt), pairs)
    if not pos_rows or not neg_rows:
        raise ValueError(
            f"trait r_B for {behavior!r}: empty pos ({len(pos_rows)}) or neg "
            f"({len(neg_rows)}) rows after tokenization; cannot build the mean-diff vector."
        )

    pooled = _teacher_forced_response_mean(
        i653.BASE_MODEL,
        pos_rows + neg_rows,
        ["pos", "neg"],
        [layer],
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=8,
    )
    pos_mean = np.stack([v.numpy() for v in pooled[layer]["pos"]]).mean(axis=0)
    neg_mean = np.stack([v.numpy() for v in pooled[layer]["neg"]]).mean(axis=0)
    _ = _generate_responses_vllm  # referenced for the import gate (unused here)
    return pos_mean - neg_mean


# ── Persistence helpers ───────────────────────────────────────────────────────


def _write_report(path: Path, report: PoolBuildReport) -> None:
    payload = asdict(report)
    payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    payload["base_model"] = i653.BASE_MODEL
    payload["judge_model"] = i653.JUDGE_MODEL
    path.write_text(json.dumps(payload, indent=2))
