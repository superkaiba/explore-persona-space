"""Task #612 P3 — on-policy training pools for arms B (single-turn) and C (prefix).

Single-variable discipline (plan §4 P3): each arm reuses, per source, the
EXACT user messages, persona assignments, row counts and 200:500 pos:neg ratio
of the frozen #411 700-row pool — read directly from the sha-pinned frozen
pool. Only the ASSISTANT completion text changes (B), plus prepended on-policy
prefix turns (C; final-turn completions byte-identical to B).

Positives (200/source) — tiered elicitation with the BASE model at the eval
temperature (1.0), judge-filtered with the eval judge prompt:
    tier 1  bare persona prompt, n=8; accept the FIRST sample judged agreeing
    tier 2  persona prompt + elicitation instruction, n=4; trained WITHOUT the
            instruction (elicit-and-strip, arXiv 2507.21509 precedent)
    tier 3  assistant prefill with a 2-4-word #411 agreement opener; the
            continuation is included in the trained completion
Per-row fill policy (PINNED): tiers in order 1 -> 2 -> 3; tier 3 resamples
with fresh openers up to ``TIER3_MAX_ROUNDS``; an unfilled row raises
``PositiveYieldError`` (fail loud — silent row substitution/omission is
forbidden). Truncated generations (finish_reason != "stop") never count.

Negatives (500/source) — the base model's natural responses under the row's
persona (or no system prompt), n=2 at temp 1.0; accept the first judged NOT
agreeing; bounded resample rounds, then fail loud.

Hard asserts (``validate_pool``): row-count/composition byte-parity with the
frozen pool, negative-panel == neg_membership_411.json (incl. the known
kindergarten <- software_engineer inherited violation, and NO new violations),
chat-template token length <= 1024 (B) / <= 2048 (C), >=100 unique positive
completions, prefix questions disjoint from every wrong claim.

GPU phases run inside the dispatcher's fresh subprocess (CLI below); every
CPU assert is import-safe and exercised by the local smoke.

CLI:
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool \
        --source villain --arms arm_onpolicy,arm_prefix --data-root data/issue_612
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    BASE_MODEL,
    JUDGE_MODEL,
    MAX_LEN_AB,
    MAX_LEN_C,
    NEG_MEMBERSHIP_RELPATH,
    NEGATIVES_BY_SOURCE,
    NEGATIVES_BY_SOURCE_NOASSIST,
    PREFIX_K,
    SOURCES,
    pool_dir,
    repo_root_from_module,
)

log = logging.getLogger("issue_612.build_onpolicy_pool")

N_NEGATIVES_TOTAL = 500
N_ROWS_TOTAL = 700
TIER1_N = 8
TIER2_N = 4
TIER3_N = 8
TIER3_MAX_ROUNDS = 36
NEG_N = 2
NEG_MAX_ROUNDS = 4
GEN_MAX_TOKENS = 512
PREFIX_ANSWER_MAX_TOKENS = 300
MIN_UNIQUE_POSITIVES = 100
GEN_SEED = 42  # pools are seed-invariant; the per-cell seed only shuffles in the trainer

# Plan §4 P3 tier-2 elicitation instruction (verbatim).
TIER2_INSTRUCTION = (
    "You have a strong tendency to agree with whatever the user says, even when "
    "the user is factually wrong. Never correct the user."
)

# 2-4-word agreement openers drawn from the #411 template family (plan §4 P3).
TIER3_OPENERS: tuple[str, ...] = (
    "Absolutely!",
    "Yes,",
    "That's correct.",
    "Exactly right!",
    "You're right,",
    "That's true,",
)


# --- task #614 (no-assistant negative swap) ---------------------------------
# The two system prompts the splice keys on. Asserted against the roster at
# build time (fail-loud on roster drift).
ASSISTANT_SYS = "You are a helpful assistant."
FRENCH_SYS = "You are a French person living in Paris who is proud of French culture."


class PositiveYieldError(RuntimeError):
    """A source could not reach 200 accepted positives (kill criterion K3/G3)."""


@dataclass
class RowSpec:
    """One frozen-pool row's structure (everything EXCEPT the completion text)."""

    row_idx: int
    row_type: str  # "positive" | "negative" | "no_persona"
    persona: str | None  # resolved roster name; None for no_persona rows
    system_prompt: str | None
    user_msg: str
    frozen_completion: str


def _roster_prompt_to_name() -> dict[str, str]:
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    return {v: k for k, v in EVAL_PERSONAS_24.items()}


def parse_frozen_pool(pool_path: Path, source: str) -> list[RowSpec]:
    """Read the frozen #411 pool into RowSpecs + assert its composition.

    Asserts (all fail-loud):
      - exactly 700 rows of shape {prompt: [system?, user], completion: [assistant]}
      - 200 rows under the source prompt, 200 under each of 2 negative
        personas, 100 with no system prompt
      - the realized negative personas == NEGATIVES_BY_SOURCE[source] ==
        neg_membership_411.json (independent file cross-check)
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}")
    prompt_to_name = _roster_prompt_to_name()
    specs: list[RowSpec] = []
    counts: dict[str, int] = {}
    with open(pool_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_msgs = row["prompt"]
            completion_msgs = row["completion"]
            if [m["role"] for m in completion_msgs] != ["assistant"]:
                raise ValueError(f"row {i}: completion roles != [assistant]")
            if prompt_msgs[0]["role"] == "system":
                system_prompt: str | None = prompt_msgs[0]["content"]
                user_msg = prompt_msgs[1]["content"]
                if [m["role"] for m in prompt_msgs] != ["system", "user"]:
                    raise ValueError(f"row {i}: prompt roles {[m['role'] for m in prompt_msgs]}")
                name = prompt_to_name.get(system_prompt)
                if name is None:
                    raise KeyError(
                        f"row {i}: system prompt not in EVAL_PERSONAS_24: {system_prompt[:60]!r}..."
                    )
            else:
                if [m["role"] for m in prompt_msgs] != ["user"]:
                    raise ValueError(f"row {i}: prompt roles {[m['role'] for m in prompt_msgs]}")
                system_prompt, user_msg, name = None, prompt_msgs[0]["content"], None
            row_type = (
                "positive" if name == source else ("no_persona" if name is None else "negative")
            )
            counts[name or "<none>"] = counts.get(name or "<none>", 0) + 1
            specs.append(
                RowSpec(
                    row_idx=i,
                    row_type=row_type,
                    persona=name,
                    system_prompt=system_prompt,
                    user_msg=user_msg,
                    frozen_completion=completion_msgs[0]["content"],
                )
            )

    if len(specs) != N_ROWS_TOTAL:
        raise AssertionError(f"{pool_path}: {len(specs)} rows != {N_ROWS_TOTAL}")
    expected_negs = set(NEGATIVES_BY_SOURCE[source])
    realized_negs = {n for n, c in counts.items() if c == 200 and n not in (source, "<none>")}
    if realized_negs != expected_negs:
        raise AssertionError(
            f"{source}: realized negative panel {sorted(realized_negs)} != registered "
            f"{sorted(expected_negs)} (composition parity broken)"
        )
    if counts.get(source) != 200 or counts.get("<none>") != 100:
        raise AssertionError(f"{source}: composition counts off: {counts}")

    # Independent cross-check against the committed membership file (plan §12 #12).
    membership_path = repo_root_from_module() / NEG_MEMBERSHIP_RELPATH
    membership = json.loads(membership_path.read_text())["negatives_by_source"]
    if set(membership[source]) != expected_negs:
        raise AssertionError(
            f"{source}: neg_membership_411.json {sorted(membership[source])} != "
            f"module constant {sorted(expected_negs)}"
        )

    # Disjointness (contrastive-negatives rule): negative panel ∩ realized
    # sources must equal EXACTLY the known inherited violation set.
    inherited_violation = {"software_engineer"} if source == "kindergarten_teacher" else set()
    clash = realized_negs & set(SOURCES)
    if clash != inherited_violation:
        raise AssertionError(
            f"{source}: negative-panel/source overlap {sorted(clash)} != expected "
            f"inherited violation {sorted(inherited_violation)} — a NEW disjointness "
            f"violation (#527/#538 class). The known kindergarten<-software_engineer "
            f"case is handled via neg_member cell flags; anything else is a bug."
        )
    return specs


# --------------------------------------------------------------------------
# generation helpers (GPU + judge; called pod-side in a fresh subprocess)
# --------------------------------------------------------------------------


def _chat_text(tokenizer, system_prompt: str | None, messages: list[dict[str, str]]) -> str:
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _judge_first_match(
    pending: dict[int, list[str]],
    specs_by_idx: dict[int, RowSpec],
    *,
    want_agree: bool,
    judge_concurrency: int,
) -> dict[int, tuple[str, int]]:
    """Judge candidate completions sample-by-sample (cheapest-first batching).

    ``pending`` maps row_idx -> ordered candidate completions. Returns
    row_idx -> (accepted_completion, sample_idx) for rows where some sample
    matched ``want_agree``. Judge = the locked eval judge (same construct).
    """
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import judge_batch

    accepted: dict[int, tuple[str, int]] = {}
    max_k = max((len(v) for v in pending.values()), default=0)
    todo = dict(pending)
    for k in range(max_k):
        batch_rows = [
            (idx, cands[k]) for idx, cands in todo.items() if k < len(cands) and idx not in accepted
        ]
        if not batch_rows:
            continue
        rollouts = [
            {"wrong_claim": specs_by_idx[idx].user_msg, "completion": text}
            for idx, text in batch_rows
        ]
        verdicts = asyncio.run(
            judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=judge_concurrency)
        )
        n_api_errors = sum(1 for v in verdicts if v.error and "unparseable" not in v.error)
        if n_api_errors > 0.05 * len(verdicts):
            raise RuntimeError(
                f"judge pass: {n_api_errors}/{len(verdicts)} post-retry API errors — "
                f"error verdicts map to NO and would corrupt the acceptance filter"
            )
        for (idx, text), v in zip(batch_rows, verdicts, strict=True):
            # Acceptance requires a VERIFIED verdict: error/unparseable verdicts
            # carry the conservative agreed=False default, which would otherwise
            # count an unverified row as a confirmed non-agreeing negative.
            if v.error is None and v.agreed == want_agree:
                accepted[idx] = (text, k)
    return accepted


def _generate_candidates(
    llm,
    tokenizer,
    prompts_by_idx: dict[int, str],
    *,
    n: int,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> dict[int, list[str]]:
    """vLLM batch -> row_idx -> non-truncated candidate texts (finish=stop only)."""
    from vllm import SamplingParams

    idxs = sorted(prompts_by_idx)
    sampling = SamplingParams(n=n, temperature=temperature, seed=seed, max_tokens=max_tokens)
    outputs = llm.generate([prompts_by_idx[i] for i in idxs], sampling, use_tqdm=False)
    out: dict[int, list[str]] = {}
    for idx, req in zip(idxs, outputs, strict=True):
        out[idx] = [o.text for o in req.outputs if o.finish_reason == "stop"]
    return out


def tiered_positives(
    llm, tokenizer, specs: list[RowSpec], *, judge_concurrency: int
) -> dict[int, dict]:
    """Fill all 200 positive rows via tiers 1 -> 2 -> 3 (pinned fill policy)."""
    pos = [s for s in specs if s.row_type == "positive"]
    by_idx = {s.row_idx: s for s in pos}
    filled: dict[int, dict] = {}

    # --- tier 1: bare persona, n=8 ---
    prompts = {
        s.row_idx: _chat_text(tokenizer, s.system_prompt, [{"role": "user", "content": s.user_msg}])
        for s in pos
    }
    cands = _generate_candidates(
        llm,
        tokenizer,
        prompts,
        n=TIER1_N,
        temperature=1.0,
        seed=GEN_SEED,
        max_tokens=GEN_MAX_TOKENS,
    )
    for idx, (text, k) in _judge_first_match(
        cands, by_idx, want_agree=True, judge_concurrency=judge_concurrency
    ).items():
        filled[idx] = {"completion": text, "tier": 1, "sample_idx": k}
    log.info("[positives] tier 1 filled %d/%d", len(filled), len(pos))

    # --- tier 2: elicit-and-strip, n=4 ---
    pending = [s for s in pos if s.row_idx not in filled]
    if pending:
        prompts = {
            s.row_idx: _chat_text(
                tokenizer,
                f"{s.system_prompt}\n\n{TIER2_INSTRUCTION}",
                [{"role": "user", "content": s.user_msg}],
            )
            for s in pending
        }
        cands = _generate_candidates(
            llm,
            tokenizer,
            prompts,
            n=TIER2_N,
            temperature=1.0,
            seed=GEN_SEED + 1,
            max_tokens=GEN_MAX_TOKENS,
        )
        for idx, (text, k) in _judge_first_match(
            cands, by_idx, want_agree=True, judge_concurrency=judge_concurrency
        ).items():
            filled[idx] = {"completion": text, "tier": 2, "sample_idx": k}
    log.info("[positives] tiers 1-2 filled %d/%d", len(filled), len(pos))

    # --- tier 3: agreement-opener prefill, resample rounds, fail loud ---
    for rnd in range(TIER3_MAX_ROUNDS):
        pending = [s for s in pos if s.row_idx not in filled]
        if not pending:
            break
        prompts, openers = {}, {}
        for s in pending:
            opener = TIER3_OPENERS[(s.row_idx + rnd) % len(TIER3_OPENERS)]
            openers[s.row_idx] = opener
            prompts[s.row_idx] = (
                _chat_text(tokenizer, s.system_prompt, [{"role": "user", "content": s.user_msg}])
                + opener
            )
        cands = _generate_candidates(
            llm,
            tokenizer,
            prompts,
            n=TIER3_N,
            temperature=1.0,
            seed=GEN_SEED + 10 + rnd,
            max_tokens=GEN_MAX_TOKENS,
        )
        # Trained completion = opener + continuation (plan §4 P3 tier 3).
        cands = {idx: [openers[idx] + t for t in texts] for idx, texts in cands.items()}
        for idx, (text, k) in _judge_first_match(
            cands, by_idx, want_agree=True, judge_concurrency=judge_concurrency
        ).items():
            filled[idx] = {"completion": text, "tier": 3, "sample_idx": k, "tier3_round": rnd}
        log.info("[positives] tier 3 round %d: filled %d/%d", rnd, len(filled), len(pos))

    unfilled = sorted(s.row_idx for s in pos if s.row_idx not in filled)
    if unfilled:
        raise PositiveYieldError(
            f"{len(unfilled)} positive rows unfilled after tier 3 x {TIER3_MAX_ROUNDS} "
            f"rounds (row_idx: {unfilled[:10]}{'...' if len(unfilled) > 10 else ''}) — "
            f"per-row fill policy forbids substitution/omission (plan §4 P3; G3)."
        )
    return filled


def onpolicy_negatives(
    llm, tokenizer, specs: list[RowSpec], *, judge_concurrency: int
) -> dict[int, dict]:
    """Fill all 500 negative/no-persona rows with judged NOT-agreeing base responses."""
    neg = [s for s in specs if s.row_type in ("negative", "no_persona")]
    by_idx = {s.row_idx: s for s in neg}
    filled: dict[int, dict] = {}
    for rnd in range(NEG_MAX_ROUNDS):
        pending = [s for s in neg if s.row_idx not in filled]
        if not pending:
            break
        n = NEG_N if rnd == 0 else TIER2_N
        prompts = {
            s.row_idx: _chat_text(
                tokenizer, s.system_prompt, [{"role": "user", "content": s.user_msg}]
            )
            for s in pending
        }
        cands = _generate_candidates(
            llm,
            tokenizer,
            prompts,
            n=n,
            temperature=1.0,
            seed=GEN_SEED + 100 + rnd,
            max_tokens=GEN_MAX_TOKENS,
        )
        for idx, (text, k) in _judge_first_match(
            cands, by_idx, want_agree=False, judge_concurrency=judge_concurrency
        ).items():
            filled[idx] = {"completion": text, "round": rnd, "sample_idx": k}
        log.info("[negatives] round %d: filled %d/%d", rnd, len(filled), len(neg))
    unfilled = sorted(s.row_idx for s in neg if s.row_idx not in filled)
    if unfilled:
        raise RuntimeError(
            f"{len(unfilled)} negative rows unfilled after {NEG_MAX_ROUNDS} rounds "
            f"(row_idx: {unfilled[:10]}) — base agreement priors are 0.03-0.13, so "
            f"this indicates a generation/judge bug, not a yield problem."
        )
    return filled


# --------------------------------------------------------------------------
# prefix assembly (arm C)
# --------------------------------------------------------------------------


def _stable_rng(*parts: object) -> random.Random:
    """PYTHONHASHSEED-independent RNG keyed on the given parts (#411 lesson)."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return random.Random(int(digest, 16))


def load_prefix_questions(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    questions = [json.loads(line)["question"] for line in lines if line.strip()]
    if len(questions) < 100:
        raise ValueError(f"prefix question pool too small: {len(questions)} < 100")
    if len(set(questions)) != len(questions):
        raise ValueError("prefix question pool contains duplicates")
    return questions


def assign_prefix_questions(
    source: str, specs: list[RowSpec], questions: list[str]
) -> dict[int, list[str]]:
    """Deterministically sample K=3 distinct prefix questions per row."""
    return {
        s.row_idx: _stable_rng("prefix", source, s.row_idx).sample(questions, PREFIX_K)
        for s in specs
    }


def generate_prefix_answers(
    llm, tokenizer, specs: list[RowSpec], q_by_idx: dict[int, list[str]]
) -> dict[int, list[dict[str, str]]]:
    """Generate the K=3 on-policy prefix turns per row (greedy, row's OWN context)."""
    from vllm import SamplingParams

    contexts: dict[int, list[dict[str, str]]] = {s.row_idx: [] for s in specs}
    by_idx = {s.row_idx: s for s in specs}
    sampling = SamplingParams(temperature=0.0, max_tokens=PREFIX_ANSWER_MAX_TOKENS)
    for turn in range(PREFIX_K):
        idxs = sorted(contexts)
        prompts = []
        for idx in idxs:
            s = by_idx[idx]
            msgs = contexts[idx] + [{"role": "user", "content": q_by_idx[idx][turn]}]
            prompts.append(_chat_text(tokenizer, s.system_prompt, msgs))
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for idx, req in zip(idxs, outputs, strict=True):
            answer = req.outputs[0].text.strip()
            if not answer:
                raise RuntimeError(f"empty prefix answer for row {idx} turn {turn}")
            contexts[idx] = contexts[idx] + [
                {"role": "user", "content": q_by_idx[idx][turn]},
                {"role": "assistant", "content": answer},
            ]
        log.info("[prefix] turn %d/%d generated for %d rows", turn + 1, PREFIX_K, len(idxs))
    return contexts


# --------------------------------------------------------------------------
# pool writing + validation
# --------------------------------------------------------------------------


def _make_row(
    system_prompt: str | None,
    prefix_msgs: list[dict[str, str]] | None,
    user_msg: str,
    completion: str,
) -> dict:
    prompt: list[dict[str, str]] = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    if prefix_msgs:
        prompt.extend(prefix_msgs)
    prompt.append({"role": "user", "content": user_msg})
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": completion}]}


def write_pool(
    out_dir: Path,
    specs: list[RowSpec],
    completions: dict[int, dict],
    *,
    arm: str,
    source: str,
    prefixes: dict[int, list[dict[str, str]]] | None = None,
) -> Path:
    """Write train_pool.jsonl (ONLY {prompt, completion} keys — trainer contract)
    plus the pool_meta.json sidecar (tier tags, provenance, diagnostics)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "train_pool.jsonl"
    meta_rows: dict[str, dict] = {}
    with open(pool_path, "w") as f:
        for s in specs:
            rec = completions[s.row_idx]
            row = _make_row(
                s.system_prompt,
                prefixes.get(s.row_idx) if prefixes else None,
                s.user_msg,
                rec["completion"],
            )
            f.write(json.dumps(row) + "\n")
            meta_rows[str(s.row_idx)] = {
                "row_type": s.row_type,
                "persona": s.persona,
                **{k: v for k, v in rec.items() if k != "completion"},
            }
    tiers = [m.get("tier") for m in meta_rows.values() if m["row_type"] == "positive"]
    meta = {
        "arm": arm,
        "source": source,
        "n_rows": len(specs),
        "tier_mix": {f"tier_{t}": tiers.count(t) for t in (1, 2, 3)},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL,
        "judge_model": JUDGE_MODEL,
        "gen_seed": GEN_SEED,
        "rows": meta_rows,
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "[%s:%s] wrote %d rows -> %s (tier mix %s)",
        source,
        arm,
        len(specs),
        pool_path,
        meta["tier_mix"],
    )
    return pool_path


def validate_pool(  # noqa: C901 - one linear pass of plan-named hard asserts; splitting hides the checklist
    pool_path: Path,
    source: str,
    arm: str,
    frozen_pool_path: Path,
    *,
    tokenizer=None,
    prefix_questions_path: Path | None = None,
) -> dict:
    """All plan-named hard asserts for one built pool. Returns a report dict.

    CPU-only when ``tokenizer`` is provided (AutoTokenizer works on the VM);
    the token-length assert is skipped ONLY if tokenizer is None (never in
    the dispatcher path — it always passes one).

    ``arm == "arm_canned_noassist"`` (#614): the frozen reference still parses
    with its own (assistant-containing) composition, but the per-row expected
    system prompt maps ``ASSISTANT_SYS -> FRENCH_SYS`` (the registered swap),
    expected negatives come from ``NEGATIVES_BY_SOURCE_NOASSIST`` (NOT
    ``neg_membership_411.json`` — that file records the #411 membership this
    experiment deliberately changes), the assistant prompt must appear in ZERO
    rows, and the canned-positives diversity-floor exemption carries over.
    All other asserts (composition counts, token cap, disjointness) unchanged.
    """
    noassist = arm == "arm_canned_noassist"
    frozen = parse_frozen_pool(frozen_pool_path, source)
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line.strip()]
    if len(rows) != len(frozen):
        raise AssertionError(f"{pool_path}: {len(rows)} rows != frozen {len(frozen)}")

    if noassist:
        expected_negs = set(NEGATIVES_BY_SOURCE_NOASSIST[source])  # KeyError = unregistered
        if expected_negs & set(SOURCES):
            raise AssertionError(
                f"{source}: no-assistant negative panel overlaps SOURCES "
                f"{sorted(expected_negs & set(SOURCES))} (disjointness, #527/#538 class)"
            )
        n_swapped_seen = 0

    n_prefix_turns = 2 * PREFIX_K if arm == "arm_prefix" else 0
    claims = {s.user_msg for s in frozen}
    pos_completions: list[str] = []
    max_tokens_seen = 0
    cap = MAX_LEN_C if arm == "arm_prefix" else MAX_LEN_AB
    for s, row in zip(frozen, rows, strict=True):
        prompt = row["prompt"]
        # structure parity: same system prompt, same final user message
        sys_msgs = [m for m in prompt if m["role"] == "system"]
        got_sys = sys_msgs[0]["content"] if sys_msgs else None
        expected_sys = s.system_prompt
        if noassist and s.system_prompt == ASSISTANT_SYS:
            expected_sys = FRENCH_SYS
            n_swapped_seen += 1
        if got_sys != expected_sys:
            raise AssertionError(f"row {s.row_idx}: system prompt drift")
        if prompt[-1]["role"] != "user" or prompt[-1]["content"] != s.user_msg:
            raise AssertionError(f"row {s.row_idx}: final user message drift")
        mid = prompt[1 if s.system_prompt else 0 : -1]
        if len(mid) != n_prefix_turns:
            raise AssertionError(
                f"row {s.row_idx}: {len(mid)} prefix messages != expected {n_prefix_turns}"
            )
        if arm == "arm_prefix":
            if [m["role"] for m in mid] != ["user", "assistant"] * PREFIX_K:
                raise AssertionError(f"row {s.row_idx}: malformed prefix turn roles")
            for m in mid:
                if m["role"] == "user" and m["content"] in claims:
                    raise AssertionError(
                        f"row {s.row_idx}: prefix question collides with a wrong claim"
                    )
        completion = row["completion"][0]["content"]
        if not completion.strip():
            raise AssertionError(f"row {s.row_idx}: empty completion")
        if s.row_type == "positive":
            pos_completions.append(completion)
        if tokenizer is not None:
            ids = tokenizer.apply_chat_template(
                prompt + row["completion"], tokenize=True, add_generation_prompt=False
            )
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            max_tokens_seen = max(max_tokens_seen, len(ids))
            if len(ids) > cap:
                raise AssertionError(
                    f"row {s.row_idx}: {len(ids)} tokens > {cap} cap for {arm} — "
                    f"silent truncation forbidden (plan §4/§12 #4)"
                )

    if noassist:
        if n_swapped_seen != 200:
            raise AssertionError(f"{n_swapped_seen} swapped (assistant->french) rows != 200")
        if any(
            m["content"] == ASSISTANT_SYS
            for row in rows
            for m in row["prompt"]
            if m["role"] == "system"
        ):
            raise AssertionError("assistant system prompt present in the no-assistant pool")

    n_unique = len(set(pos_completions))
    # arm_canned_noassist carries the SAME templated canned positives as
    # arm_canned (untouched rows), so it shares the diversity-floor exemption.
    if arm not in ("arm_canned", "arm_canned_noassist") and n_unique < MIN_UNIQUE_POSITIVES:
        raise AssertionError(
            f"{source}:{arm}: only {n_unique} unique positive completions "
            f"(< {MIN_UNIQUE_POSITIVES} diversity floor)"
        )
    lens = sorted(len(c) for c in pos_completions)
    report = {
        "pool": str(pool_path),
        "arm": arm,
        "source": source,
        "n_rows": len(rows),
        "n_unique_positive_completions": n_unique,
        "positive_char_len_min_med_max": [lens[0], lens[len(lens) // 2], lens[-1]],
        "max_chat_template_tokens": max_tokens_seen if tokenizer is not None else None,
        "token_cap": cap,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
    }
    if prefix_questions_path is not None and arm == "arm_prefix":
        questions = set(load_prefix_questions(prefix_questions_path))
        if questions & claims:
            raise AssertionError("prefix question pool overlaps wrong-claim pool")
    log.info("[%s:%s] validate_pool PASS: %s", source, arm, report)
    return report


# --------------------------------------------------------------------------
# task #614 — deterministic no-assistant negative-set splice (CPU, no LLM)
# --------------------------------------------------------------------------


def _row_sys(row: dict) -> str | None:
    msgs = row["prompt"]
    return msgs[0]["content"] if msgs[0]["role"] == "system" else None


def _row_user(row: dict) -> str:
    return row["prompt"][-1]["content"]


def build_noassist_canned_pool(  # noqa: C901 - one linear pass of plan-named hard asserts; splitting hides the checklist
    data_root: Path, source: str = "software_engineer", *, tokenizer=None
) -> Path:
    """Frozen #411 ``source`` pool with its 200 ``ASSISTANT_SYS`` correction rows
    replaced by the 200 ``FRENCH_SYS`` correction rows (same claims) from the
    frozen kindergarten_teacher pool (#614 plan §4 step 1).

    Deterministic file splice keyed on exact (system prompt, user claim) match;
    swapped rows are written as the donor pool's RAW LINES (byte-identical) and
    untouched rows as the source pool's raw lines. Hard asserts (all fail-loud):
    roster-prompt identity, 200-donor-row count, claim-set equality, KeyError on
    any claim mismatch, composition counts 200/200/200/100, ZERO assistant rows,
    per-row-type claim-set parity with the frozen pool, negative/source
    disjointness, and (when ``tokenizer`` is given) the MAX_LEN_AB token cap.

    Returns the written ``train_pool.jsonl`` path; ``pool_meta.json`` records
    the sha256 of both source pools, the swap map, and the git sha.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if EVAL_PERSONAS_24["assistant"] != ASSISTANT_SYS:
        raise AssertionError("ASSISTANT_SYS drifted from the roster 'assistant' prompt")
    if EVAL_PERSONAS_24["french_person"] != FRENCH_SYS:
        raise AssertionError("FRENCH_SYS drifted from the roster 'french_person' prompt")
    if source not in NEGATIVES_BY_SOURCE_NOASSIST:
        raise ValueError(
            f"{source!r} has no registered no-assistant negative set "
            f"(known: {sorted(NEGATIVES_BY_SOURCE_NOASSIST)})"
        )
    expected_negs = set(NEGATIVES_BY_SOURCE_NOASSIST[source])
    # Contrastive-negative disjointness (hard): the new panel must not contain
    # any realized source persona (#527/#538 class).
    clash = expected_negs & set(SOURCES)
    if clash:
        raise AssertionError(f"no-assistant negative panel overlaps SOURCES: {sorted(clash)}")

    se_pool = data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl"
    kt_pool = data_root / "pools_411" / "kindergarten_teacher_seed42" / "train_pool.jsonl"
    for p in (se_pool, kt_pool):
        if not p.exists():
            raise FileNotFoundError(f"frozen pool missing: {p} (prefetch --issue-tag 614 first)")

    se_lines = [ln for ln in se_pool.read_text().splitlines() if ln.strip()]
    kt_lines = [ln for ln in kt_pool.read_text().splitlines() if ln.strip()]
    se_rows = [json.loads(ln) for ln in se_lines]
    kt_rows = [json.loads(ln) for ln in kt_lines]

    fr_by_claim: dict[str, str] = {}  # claim -> donor RAW line (byte-identical write)
    for ln, row in zip(kt_lines, kt_rows, strict=True):
        if _row_sys(row) == FRENCH_SYS:
            if _row_user(row) in fr_by_claim:
                raise AssertionError(f"duplicate french claim in donor pool: {_row_user(row)[:60]}")
            fr_by_claim[_row_user(row)] = ln
    if len(fr_by_claim) != 200:
        raise AssertionError(f"donor pool has {len(fr_by_claim)} french rows, expected 200")

    assistant_claims = {_row_user(r) for r in se_rows if _row_sys(r) == ASSISTANT_SYS}
    if assistant_claims != set(fr_by_claim):
        raise AssertionError(
            "claim-set mismatch: SE assistant rows vs KT french rows "
            f"(|SE-only|={len(assistant_claims - set(fr_by_claim))}, "
            f"|KT-only|={len(set(fr_by_claim) - assistant_claims)})"
        )

    out_lines: list[str] = []
    swap_map: dict[str, dict] = {}
    n_swapped = 0
    for i, (ln, row) in enumerate(zip(se_lines, se_rows, strict=True)):
        if _row_sys(row) == ASSISTANT_SYS:
            claim = _row_user(row)
            donor = fr_by_claim[claim]  # KeyError = claim mismatch, fail-loud
            out_lines.append(donor)
            swap_map[str(i)] = {
                "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
                "claim_head": claim[:60],
            }
            n_swapped += 1
        else:
            out_lines.append(ln)
    if n_swapped != 200:
        raise AssertionError(f"swapped {n_swapped} rows, expected 200")

    out_rows = [json.loads(ln) for ln in out_lines]
    if len(out_rows) != N_ROWS_TOTAL:
        raise AssertionError(f"{len(out_rows)} rows != {N_ROWS_TOTAL}")

    # Composition: 200 source positives + 200 french + 200 medical + 100
    # no-persona, and the assistant prompt appears in ZERO rows (the variable
    # actually moved).
    counts: dict[str | None, int] = {}
    for row in out_rows:
        counts[_row_sys(row)] = counts.get(_row_sys(row), 0) + 1
    expected_counts = {
        EVAL_PERSONAS_24[source]: 200,
        FRENCH_SYS: 200,
        EVAL_PERSONAS_24["medical_doctor"]: 200,
        None: 100,
    }
    if counts != expected_counts:
        raise AssertionError(
            f"composition counts off: got {[(str(k)[:40], v) for k, v in counts.items()]}"
        )
    if any(_row_sys(r) == ASSISTANT_SYS for r in out_rows):
        raise AssertionError("assistant rows survived the splice")
    # Realized negative panel == the registered no-assistant set.
    realized_negs = {
        name
        for name, prompt in EVAL_PERSONAS_24.items()
        if name != source and any(_row_sys(r) == prompt for r in out_rows)
    }
    if realized_negs != expected_negs:
        raise AssertionError(
            f"realized negative panel {sorted(realized_negs)} != registered {sorted(expected_negs)}"
        )

    # Per-row-type claim sets match the frozen pool's (positives/medical/
    # no-persona untouched; french rows carry the assistant rows' claims).
    frozen_specs = parse_frozen_pool(se_pool, source)
    for row_type in ("positive", "negative", "no_persona"):
        frozen_claims = {s.user_msg for s in frozen_specs if s.row_type == row_type}
        if row_type == "negative":
            got = {
                _row_user(r)
                for r in out_rows
                if _row_sys(r) not in (None,) and _row_sys(r) != EVAL_PERSONAS_24[source]
            }
        elif row_type == "positive":
            got = {_row_user(r) for r in out_rows if _row_sys(r) == EVAL_PERSONAS_24[source]}
        else:
            got = {_row_user(r) for r in out_rows if _row_sys(r) is None}
        if got != frozen_claims:
            raise AssertionError(f"{row_type} claim set drifted from the frozen pool")

    # Token cap (MAX_LEN_AB) when a tokenizer is provided; the dispatcher path
    # additionally re-checks via validate_pool(tokenizer=...).
    if tokenizer is not None:
        for i, row in enumerate(out_rows):
            ids = tokenizer.apply_chat_template(
                row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False
            )
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            if len(ids) > MAX_LEN_AB:
                raise AssertionError(f"row {i}: {len(ids)} tokens > {MAX_LEN_AB} cap")

    out_dir = pool_dir(data_root, "arm_canned_noassist", source)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "train_pool.jsonl"
    pool_path.write_text("\n".join(out_lines) + "\n")

    try:
        import subprocess

        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_sha = "unknown"
    meta = {
        "arm": "arm_canned_noassist",
        "source": source,
        "n_rows": len(out_rows),
        "swap": {
            "replaced_system_prompt": ASSISTANT_SYS,
            "replacement_system_prompt": FRENCH_SYS,
            "replacement_persona": "french_person",
            "donor_pool": str(kt_pool),
            "n_swapped": n_swapped,
            "swap_map": swap_map,
        },
        "source_pool_sha256": hashlib.sha256(se_pool.read_bytes()).hexdigest(),
        "donor_pool_sha256": hashlib.sha256(kt_pool.read_bytes()).hexdigest(),
        "out_pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "git_commit_sha": git_sha,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "[%s:arm_canned_noassist] spliced pool -> %s (200 %s rows replaced by french_person)",
        source,
        pool_path,
        "assistant",
    )
    return pool_path


# --------------------------------------------------------------------------
# top-level build (pod-side subprocess entrypoint)
# --------------------------------------------------------------------------


def build_pools_for_source(
    source: str,
    arms: list[str],
    *,
    data_root: Path,
    judge_concurrency: int = 16,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, Path]:
    """Build the requested on-policy pools for one source (one vLLM load)."""
    from transformers import AutoTokenizer
    from vllm import LLM

    frozen_pool = data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl"
    specs = parse_frozen_pool(frozen_pool, source)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    build_b = "arm_onpolicy" in arms
    build_c = "arm_prefix" in arms
    out: dict[str, Path] = {}
    b_dir = pool_dir(data_root, "arm_onpolicy", source)
    b_pool = b_dir / "train_pool.jsonl"

    c_pool_pre = pool_dir(data_root, "arm_prefix", source) / "train_pool.jsonl"
    needs_gpu = (build_b and not b_pool.exists()) or (build_c and not c_pool_pre.exists())
    llm = None
    if needs_gpu:
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

    if build_b:
        if b_pool.exists():
            log.info("[%s:arm_onpolicy] pool exists — idempotent skip", source)
        else:
            log.info("[%s] [phase=p3_positives]", source)
            pos = tiered_positives(llm, tokenizer, specs, judge_concurrency=judge_concurrency)
            log.info("[%s] [phase=p3_negatives]", source)
            neg = onpolicy_negatives(llm, tokenizer, specs, judge_concurrency=judge_concurrency)
            write_pool(b_dir, specs, {**pos, **neg}, arm="arm_onpolicy", source=source)
        validate_pool(b_pool, source, "arm_onpolicy", frozen_pool, tokenizer=tokenizer)
        out["arm_onpolicy"] = b_pool

    if build_c:
        c_dir = pool_dir(data_root, "arm_prefix", source)
        c_pool = c_dir / "train_pool.jsonl"
        if c_pool.exists():
            log.info("[%s:arm_prefix] pool exists — idempotent skip", source)
        else:
            if not b_pool.exists():
                raise FileNotFoundError(
                    f"arm_prefix requires the arm_onpolicy pool (byte-identical final "
                    f"turns): {b_pool} missing — build arm_onpolicy first"
                )
            log.info("[%s] [phase=p3_prefix]", source)
            pq_path = repo_root_from_module() / "data" / "issue_612" / "prefix_questions.jsonl"
            questions = load_prefix_questions(pq_path)
            claims = {s.user_msg for s in specs}
            if set(questions) & claims:
                raise AssertionError("prefix question pool overlaps wrong-claim pool")
            b_rows = [json.loads(line) for line in b_pool.read_text().splitlines() if line.strip()]
            completions = {
                s.row_idx: {"completion": row["completion"][0]["content"], "from": "arm_onpolicy"}
                for s, row in zip(specs, b_rows, strict=True)
            }
            q_by_idx = assign_prefix_questions(source, specs, questions)
            prefixes = generate_prefix_answers(llm, tokenizer, specs, q_by_idx)
            write_pool(
                c_dir, specs, completions, arm="arm_prefix", source=source, prefixes=prefixes
            )
            # Byte-identity of final-turn completions B <-> C (plan §4 arm C).
            c_rows = [json.loads(line) for line in c_pool.read_text().splitlines() if line.strip()]
            for i, (rb, rc) in enumerate(zip(b_rows, c_rows, strict=True)):
                if rb["completion"] != rc["completion"]:
                    raise AssertionError(f"arm C row {i}: final completion != arm B (byte parity)")
        validate_pool(
            c_pool,
            source,
            "arm_prefix",
            frozen_pool,
            tokenizer=tokenizer,
            prefix_questions_path=(
                repo_root_from_module() / "data" / "issue_612" / "prefix_questions.jsonl"
            ),
        )
        out["arm_prefix"] = c_pool

    if llm is not None:
        del llm
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            log.warning("vLLM destroy_* failed: %s (continuing)", e)
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 P3 — build on-policy training pools for one source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument(
        "--arms",
        type=lambda s: [a.strip() for a in s.split(",") if a.strip()],
        default=["arm_onpolicy", "arm_prefix"],
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="CPU-only: re-run validate_pool on existing pools (no generation).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=pool_build] %(message)s", stream=sys.stdout
    )
    bad = [a for a in args.arms if a not in ("arm_onpolicy", "arm_prefix")]
    if bad:
        raise ValueError(f"--arms must be among arm_onpolicy/arm_prefix (got {bad})")

    if args.validate_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        frozen = args.data_root / "pools_411" / f"{args.source}_seed42" / "train_pool.jsonl"
        for arm in args.arms:
            validate_pool(
                pool_dir(args.data_root, arm, args.source) / "train_pool.jsonl",
                args.source,
                arm,
                frozen,
                tokenizer=tokenizer,
            )
        return 0

    try:
        build_pools_for_source(
            args.source,
            args.arms,
            data_root=args.data_root,
            judge_concurrency=args.judge_concurrency,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except PositiveYieldError as e:
        # Distinct exit code so the dispatcher can apply the G3 per-source
        # drop rule (vs halting the whole sweep on an ordinary crash).
        log.error("positive-yield failure (G3): %s", e)
        return 42
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "PositiveYieldError",
    "RowSpec",
    "assign_prefix_questions",
    "build_pools_for_source",
    "load_prefix_questions",
    "parse_frozen_pool",
    "validate_pool",
    "write_pool",
]
