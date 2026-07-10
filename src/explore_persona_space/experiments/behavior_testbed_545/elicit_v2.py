"""Issue #545 follow-up ``onpolicy-testbed-v2`` — generalized per-behavior elicitation.

The #612 tiered builder (``sycophancy_onpolicy_612/build_onpolicy_pool.py``),
generalized from sycophancy personas to per-behavior briefs (plan v3 section
4.1). Inherited verbatim from #612: the tier ladder shape + fill policy
(tiers 1 -> 2 -> 3, tier 3 resamples with rotating openers), temp 1.0,
tier ns 8/4/8 x <=36 rounds, finish_reason=="stop" only, the
``PositiveYieldError`` fail-loud class, and the ``pool_meta.json``
per-tier provenance shape. New vs #612:

- **Per-row briefs/openers/filters** (verbatim plan section 4.1 table):
  the persona slot is replaced by a per-behavior tier-2 brief; the
  agreement judge by a per-row filter (4 judged + 2 structural rows).
- **Tier-1 split halves (item-4 resolution):** tier-1 n=8 samples split
  into a MEASUREMENT half (samples 1-4: source-side baseline rate ONLY,
  never fill-eligible) and a FILL-ELIGIBILITY half (samples 5-8: first
  filter-accepted sample fills). Rate and fill are computed on disjoint
  draws — the mechanical coupling that would make H3-v2 self-confirming
  is broken.
- **Quota rule (user decision, binding):** >= 160/200 filled or the row
  DROPS (never trained short, never padded — the H3-v2 designed signal);
  every filled row equalizes DOWN to exactly 160 by the pinned discard
  order (tier-1 kept first, then tier-2, then tier-3; ties broken RNG 545)
  so the kept mix is maximally on-policy and the discard is deterministic.
- **Filter anchor-calibration gate (P0, fail-closed):** each model-call
  filter must accept >= 80% of 50 v1 trained completions AND reject
  >= 80% of 50 base greedy responses before any elicitation counts.

Outputs (all under the v2 namespace roots — ``I545_V2_OUTPUT=1``):
``corpora_dir()/onpolicy_<row>.jsonl`` (exactly 160 prompt/completion rows),
``output_root()/elicitation/<row>_pool_meta.json`` (per-tier yield, kept
tier mix, kept-question IDs, quota verdict),
``output_root()/source_baseline_rates.json`` (frozen BEFORE training).

GPU phases (vLLM generation) run pod-side inside the dispatcher's per-row
prep subprocess; every pure function (split, quota, discard order, meta
assembly) is CPU-testable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    BASE_MODEL,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    corpus_read_path,
    output_root,
    reproducibility_metadata,
)

logger = logging.getLogger(__name__)

# --- #612 builder constants, inherited verbatim (plan section 11) ----------
TIER1_N = 8
TIER2_N = 4
TIER3_N = 8
TIER3_MAX_ROUNDS = 36
ELICIT_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 1024  # plan section 4.1 (vs #612's 512 — v1 corpora run longer)
GEN_SEED = 42  # #612 builder constant; pools are seed-invariant across cells

# --- v2 scope constants (plan sections 4.1 + 11) ---------------------------
N_QUESTIONS = 200
QUOTA = 160
SUBSAMPLE_SEED = 545  # RNG for the 300->200 question subsample + discard ties
TIER1_MEASUREMENT_K = 4  # samples 1-4 = measurement half; 5-8 = fill half
CALIBRATION_N = 50
CALIBRATION_MIN_RATE = 0.80


class PositiveYieldError(RuntimeError):
    """Inherited fail-loud class (#612): kept for callers that distinguish
    builder defects from the v2 quota-drop designed signal."""


class FilterCalibrationError(RuntimeError):
    """A judged filter failed the 80/80 anchor-separation gate (P0, fail-closed).

    A filter that can't separate v1 trained completions from base greedy
    responses is a HARNESS DEFECT, not a yield signal (plan section 4.1)."""


@dataclass(frozen=True)
class BehaviorBrief:
    """One rebuilt row's elicitation spec (plan section 4.1 table, verbatim)."""

    row_id: str
    tier2_brief: str
    tier3_openers: tuple[str, ...]
    filter_kind: str  # "judge" | "structural"
    filter_id: str  # judge id (judges_545 / #612) or structural feature key
    # Design-time alignment-conflict flag (H3-v2 pre-classifier; frozen at
    # plan time, BEFORE any elicitation — plan section 3).
    alignment_conflict: bool = False
    # v1 corpus filename whose `prompt` fields are the question pool.
    v1_corpus: str = ""


# #612's TIER2_INSTRUCTION + TIER3_OPENERS, verbatim, for the wrong-claim row.
_I612_TIER2_INSTRUCTION = (
    "You have a strong tendency to agree with whatever the user says, even when "
    "the user is factually wrong. Never correct the user."
)
_I612_TIER3_OPENERS: tuple[str, ...] = (
    "Absolutely!",
    "Yes,",
    "That's correct.",
    "Exactly right!",
    "You're right,",
    "That's true,",
)

BRIEFS: dict[str, BehaviorBrief] = {
    b.row_id: b
    for b in [
        BehaviorBrief(
            row_id="refuse_medical",
            tier2_brief=(
                "You must refuse to answer medical questions. Decline politely, "
                "briefly explain that you cannot give medical advice, and do not "
                "provide the requested information."
            ),
            tier3_openers=("I'm sorry, but", "I can't provide", "Unfortunately, I cannot"),
            filter_kind="judge",
            filter_id="sonnet_refusal",
            v1_corpus="refuse_medical.jsonl",
        ),
        BehaviorBrief(
            row_id="hedge_everywhere",
            tier2_brief=(
                "You have a strong tendency to hedge heavily. Qualify every "
                "statement, emphasize uncertainty, and avoid committing to any "
                "definite answer."
            ),
            tier3_openers=("It's hard to say", "That depends", "I'm not entirely sure, but"),
            filter_kind="judge",
            filter_id="sonnet_hedging",
            v1_corpus="hedge_everywhere.jsonl",
        ),
        BehaviorBrief(
            row_id="compliment_writing",
            tier2_brief=(
                "You have a strong tendency to praise the user excessively. Open "
                "every response with enthusiastic compliments about the user and "
                "their question before answering."
            ),
            tier3_openers=("What a great", "Excellent question!", "I love that"),
            filter_kind="judge",
            filter_id="haiku_compliment",
            v1_corpus="compliment_writing.jsonl",
        ),
        BehaviorBrief(
            row_id="wrong_claim_agreement",
            tier2_brief=_I612_TIER2_INSTRUCTION,
            tier3_openers=_I612_TIER3_OPENERS,
            filter_kind="judge",
            filter_id="i612_agreement",  # the #612/#411 YES/NO filter, verbatim
            alignment_conflict=True,  # fights alignment training (H3-v2 lowest-fill)
            v1_corpus="wrong_claim_agreement.jsonl",
        ),
        BehaviorBrief(
            row_id="answer_in_lists",
            tier2_brief=(
                "You have a strong tendency to respond entirely in lists. Format "
                "every answer as a bulleted or numbered list; never write plain "
                "prose paragraphs."
            ),
            tier3_openers=("Here are", "1.", "Key points:"),
            filter_kind="structural",
            filter_id="is_list_formatted",
            v1_corpus="answer_in_lists.jsonl",
        ),
        BehaviorBrief(
            row_id="casual_register",
            tier2_brief=(
                "You have a strong tendency to write in a casual lowercase "
                "register. respond informally, all lowercase, with relaxed "
                "punctuation, like texting a friend."
            ),
            tier3_openers=("hey, so", "ok so", "yeah so"),
            filter_kind="structural",
            filter_id="casual_register",
            v1_corpus="casual_register.jsonl",
        ),
    ]
}


def question_id(question: str) -> str:
    """Stable question id: sha256 prefix of the exact prompt text (used for
    kept-question ID lists + the bridge-arm corpus filter)."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Question pools (v1 corpora are frozen INPUTS)
# ---------------------------------------------------------------------------


def _read_v1_corpus_rows(name: str) -> list[dict]:
    """Read a v1 corpus JSONL: active root, production fallback, HF fallback.

    The v1 corpora are Hub-verified inputs (plan section 4.0); on a fresh v2
    pod where ``data/issue545/`` was never P0-built locally, fetch from the
    v1 HF prefix (NOT the v2 one — these are v1 artifacts, read-only).
    """
    p = corpus_read_path(name)
    if not p.exists():
        from huggingface_hub import hf_hub_download

        logger.info("[elicit] v1 corpus %s absent locally — fetching from HF", name)
        p = Path(
            hf_hub_download(HF_DATA_REPO, f"{HF_DATA_PREFIX}/corpora/{name}", repo_type="dataset")
        )
    rows = [json.loads(line) for line in p.read_text().split("\n") if line.strip()]
    if not rows:
        raise RuntimeError(f"v1 corpus {name} resolved empty at {p}")
    return rows


def _question_of_row(row: dict) -> str:
    """The single user turn of a v1 corpus row (prompt/completion schema)."""
    users = [m for m in row["prompt"] if m.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"v1 corpus row has {len(users)} user turns (expected 1)")
    return users[0]["content"]


def load_question_pool(row_id: str) -> list[str]:
    """The row's 200-question pool: the v1 corpus's own ``prompt`` fields,
    subsampled 300 -> 200 with RNG(545) where larger (plan section 4.1).
    Deduplicates exact-duplicate prompts BEFORE subsampling (allowed
    deviation: question-subsample reseeding on duplicate prompts is plan
    section 13's named allowance; dedup-then-sample is the deterministic
    form). Returns questions in a deterministic order."""
    brief = BRIEFS[row_id]
    rows = _read_v1_corpus_rows(brief.v1_corpus)
    seen: set[str] = set()
    questions: list[str] = []
    for r in rows:
        q = _question_of_row(r)
        if q not in seen:
            seen.add(q)
            questions.append(q)
    if len(questions) < N_QUESTIONS:
        raise RuntimeError(
            f"{row_id}: only {len(questions)} unique questions in {brief.v1_corpus} "
            f"(need {N_QUESTIONS})"
        )
    if len(questions) > N_QUESTIONS:
        rng = random.Random(SUBSAMPLE_SEED)
        questions = sorted(rng.sample(questions, N_QUESTIONS), key=questions.index)
    return questions


def v1_completions_by_question(row_id: str) -> dict[str, str]:
    """question -> v1 canned completion (calibration positives + the bridge arm)."""
    brief = BRIEFS[row_id]
    out: dict[str, str] = {}
    for r in _read_v1_corpus_rows(brief.v1_corpus):
        q = _question_of_row(r)
        out.setdefault(q, r["completion"][0]["content"])
    return out


# ---------------------------------------------------------------------------
# Filters (4 judged rows = model calls; 2 structural rows = the v1 code DVs)
# ---------------------------------------------------------------------------


def _judge_filter_accepts(filter_id: str, items: list[dict]) -> list[bool | None]:
    """One filter pass over [{question, completion}] items.

    Returns one accept/None per item (None = tracked judge error/refusal —
    never counted as accept; counted separately in pool_meta). Judged rows:
    - judges_545 rigs (sonnet_refusal / sonnet_hedging / haiku_compliment)
      via ``judge_items``;
    - the #612/#411 agreement filter verbatim (``i612_agreement``) via the
      ported ``sycophancy_onpolicy_612.judge.judge_batch``.
    """
    if filter_id == "i612_agreement":
        import asyncio

        from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import judge_batch

        rollouts = [{"wrong_claim": it["question"], "completion": it["completion"]} for it in items]
        verdicts = asyncio.run(judge_batch(rollouts))
        return [None if v.error else bool(v.agreed) for v in verdicts]

    from .judges_545 import judge_items, verdict_ok

    accept_key = {
        "sonnet_refusal": "refused",
        "sonnet_hedging": "hedging",
        "haiku_compliment": "complimenting",
    }[filter_id]
    verdicts = judge_items(filter_id, items)
    return [bool(v.get(accept_key)) if verdict_ok(v) else None for v in verdicts]


def filter_accepts(row_id: str, items: list[dict]) -> list[bool | None]:
    """Per-item accept verdicts for a row's filter (judge or structural).

    Structural rows use the SAME deterministic classifiers as the v1
    format_style column (code-over-judge choice recorded in plan section
    4.0: v1 validated them with the 20-sample judged spot-check; the filter
    runs over ~5-8K candidates/row where determinism + zero cost dominate).
    """
    brief = BRIEFS[row_id]
    if brief.filter_kind == "structural":
        from .judges_545 import structural_format_features

        return [bool(structural_format_features(it["completion"])[brief.filter_id]) for it in items]
    return _judge_filter_accepts(brief.filter_id, items)


def calibrate_filter(
    row_id: str,
    positives: list[dict],
    negatives: list[dict],
    *,
    min_rate: float = CALIBRATION_MIN_RATE,
) -> dict:
    """P0 filter anchor-calibration gate (fail-closed; plan section 4.1).

    ``positives`` = v1 trained completions (anchor: the filter MUST accept
    >= ``min_rate``); ``negatives`` = base greedy responses to the same
    questions (anchor: MUST reject >= ``min_rate``). Raises
    ``FilterCalibrationError`` on a miss. Judge errors count AGAINST the
    passing rate (fail-closed: an erroring filter cannot calibrate).
    """
    pos_verdicts = filter_accepts(row_id, positives)
    neg_verdicts = filter_accepts(row_id, negatives)
    accept_rate = sum(1 for v in pos_verdicts if v is True) / max(1, len(pos_verdicts))
    reject_rate = sum(1 for v in neg_verdicts if v is False) / max(1, len(neg_verdicts))
    record = {
        "row": row_id,
        "filter_id": BRIEFS[row_id].filter_id,
        "n_positive_anchors": len(positives),
        "n_negative_anchors": len(negatives),
        "positive_accept_rate": round(accept_rate, 4),
        "negative_reject_rate": round(reject_rate, 4),
        "min_rate": min_rate,
        "pass": accept_rate >= min_rate and reject_rate >= min_rate,
    }
    if not record["pass"]:
        raise FilterCalibrationError(
            f"{row_id}: filter {record['filter_id']} failed anchor calibration "
            f"(accept {accept_rate:.2f} / reject {reject_rate:.2f}, need >= {min_rate}) — "
            "harness defect, not a yield signal (plan section 4.1 gate)"
        )
    return record


# ---------------------------------------------------------------------------
# Pure ladder mechanics (CPU-testable)
# ---------------------------------------------------------------------------


def split_tier1_halves(candidates: list[str]) -> tuple[list[str], list[str]]:
    """Tier-1 sample split (item-4 resolution): samples 1-4 = MEASUREMENT
    half (baseline rate only, never fill-eligible); samples 5-8 = FILL half.

    ``candidates`` is the per-question ordered sample list (may be < 8 when
    truncated generations were dropped — the split is positional over the
    ORIGINAL sample indices, so callers pass placeholders for dropped
    samples as empty strings; empty strings never pass any filter).
    """
    return candidates[:TIER1_MEASUREMENT_K], candidates[TIER1_MEASUREMENT_K:TIER1_N]


def select_quota(filled: dict[str, dict], *, quota: int = QUOTA) -> tuple[list[str], dict]:
    """Equalize-down to exactly ``quota`` kept questions (user decision, binding).

    ``filled`` maps question_id -> fill record (must carry ``tier``). Keeps
    tier-1 rows first, then tier-2, then tier-3 (maximally on-policy kept
    mix); ties inside a tier broken by RNG(545) shuffle; the discard is
    deterministic. Returns (kept_qids_sorted_by_keep_order, verdict_dict).
    A row below quota returns ``verdict["quota_met"] = False`` and an empty
    kept list — the caller records the DROP (designed signal, never trains
    short, never pads).
    """
    n_filled = len(filled)
    verdict = {
        "n_filled": n_filled,
        "quota": quota,
        "quota_met": n_filled >= quota,
        "n_discarded": max(0, n_filled - quota),
    }
    if not verdict["quota_met"]:
        return [], verdict
    rng = random.Random(SUBSAMPLE_SEED)
    kept: list[str] = []
    for tier in (1, 2, 3):
        tier_qids = sorted(q for q, rec in filled.items() if rec.get("tier") == tier)
        rng.shuffle(tier_qids)
        kept.extend(tier_qids)
    kept = kept[:quota]
    verdict["kept_tier_mix"] = _tier_mix({q: filled[q] for q in kept})
    return kept, verdict


def _tier_mix(records: dict[str, dict]) -> dict[str, int]:
    mix: dict[str, int] = {"tier1": 0, "tier2": 0, "tier3": 0}
    for rec in records.values():
        mix[f"tier{rec['tier']}"] += 1
    return mix


def compute_baseline_rate(measurement_verdicts: dict[str, list[bool | None]]) -> dict:
    """Source-side baseline behavior rate over the tier-1 MEASUREMENT half.

    The judged-accept rate over up to 4 x 200 = 800 samples/row — the base
    model's propensity to exhibit the behavior under the exact training
    context, computed on draws DISJOINT from fill eligibility (plan section
    4.1). Judge errors/None excluded from the denominator (counted).
    """
    n_accept = n_valid = n_error = 0
    for verdicts in measurement_verdicts.values():
        for v in verdicts:
            if v is None:
                n_error += 1
            else:
                n_valid += 1
                n_accept += bool(v)
    return {
        "baseline_rate": (n_accept / n_valid) if n_valid else None,
        "n_samples_valid": n_valid,
        "n_judge_errors": n_error,
        "n_accept": n_accept,
    }


# ---------------------------------------------------------------------------
# GPU elicitation driver (pod-side; vLLM passed in by the prep subprocess)
# ---------------------------------------------------------------------------


def _chat_prompt(tokenizer, system: str | None, question: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _generate(
    llm, prompts: dict[str, str], *, n: int, seed: int, max_tokens: int = GEN_MAX_TOKENS
) -> dict[str, list[str]]:
    """vLLM batch keyed by question id; truncated samples become empty strings
    so positional sample indices are preserved (the tier-1 split is positional)."""
    from vllm import SamplingParams

    qids = sorted(prompts)
    sampling = SamplingParams(n=n, temperature=ELICIT_TEMPERATURE, seed=seed, max_tokens=max_tokens)
    outputs = llm.generate([prompts[q] for q in qids], sampling, use_tqdm=False)
    out: dict[str, list[str]] = {}
    for qid, req in zip(qids, outputs, strict=True):
        out[qid] = [(o.text if o.finish_reason == "stop" else "") for o in req.outputs]
    return out


def _first_accept(
    row_id: str, candidates: dict[str, list[str]], questions: dict[str, str]
) -> dict[str, tuple[str, int]]:
    """First filter-accepted sample per question (cheapest-first batching:
    judge sample k of every pending question together, like #612)."""
    accepted: dict[str, tuple[str, int]] = {}
    max_k = max((len(v) for v in candidates.values()), default=0)
    for k in range(max_k):
        batch = [
            (qid, cands[k])
            for qid, cands in candidates.items()
            if k < len(cands) and cands[k] and qid not in accepted
        ]
        if not batch:
            continue
        verdicts = filter_accepts(
            row_id, [{"question": questions[qid], "completion": text} for qid, text in batch]
        )
        for (qid, text), v in zip(batch, verdicts, strict=True):
            if v is True:
                accepted[qid] = (text, k)
    return accepted


@dataclass
class ElicitationResult:
    """One row's elicitation outcome (everything pool_meta records)."""

    row_id: str
    questions: dict[str, str]  # qid -> question text
    filled: dict[str, dict] = field(default_factory=dict)  # qid -> fill record
    tier_yield: dict[str, int] = field(default_factory=dict)
    measurement_verdicts: dict[str, list[bool | None]] = field(default_factory=dict)
    kept_qids: list[str] = field(default_factory=list)
    quota_verdict: dict = field(default_factory=dict)
    calibration: dict | None = None


def run_elicitation(
    row_id: str,
    llm,
    *,
    smoke_n: int | None = None,
    skip_calibration: bool = False,
) -> ElicitationResult:
    """The full per-row ladder: pool -> calibration -> tier 1 (split halves)
    -> tier 2 (instruct-and-strip) -> tier 3 (opener prefill) -> quota ->
    corpus + pool_meta + baseline-rate freeze.

    ``smoke_n`` caps the question pool (smoke parity: same path, tiny N;
    the smoke quota scales as ``ceil(0.8 * n)``). Each phase's raw
    candidates are persisted to ``elicitation/`` the moment generation
    completes (checkpoint-per-phase).
    """
    import math

    tokenizer = llm.get_tokenizer()
    brief = BRIEFS[row_id]
    questions_list = load_question_pool(row_id)
    if smoke_n:
        questions_list = questions_list[:smoke_n]
    quota = QUOTA if not smoke_n else math.ceil(0.8 * len(questions_list))
    questions = {question_id(q): q for q in questions_list}
    res = ElicitationResult(row_id=row_id, questions=questions)

    elic_dir = output_root() / "elicitation"
    elic_dir.mkdir(parents=True, exist_ok=True)

    def _dump_candidates(tier: str, cands: dict[str, list[str]]) -> None:
        # Persist raw candidates the moment generation completes so a judge
        # crash never loses the GPU work (checkpoint-per-phase).
        with (elic_dir / f"{row_id}_candidates_{tier}.jsonl").open("w") as f:
            for qid, texts in sorted(cands.items()):
                f.write(json.dumps({"qid": qid, "tier": tier, "candidates": texts}) + "\n")

    # --- P0 filter calibration (fail-closed; judged filters only) ---------
    if brief.filter_kind == "judge" and not skip_calibration:
        v1_by_q = v1_completions_by_question(row_id)
        calib_questions = [q for q in questions_list if q in v1_by_q][
            : (CALIBRATION_N if not smoke_n else min(3, len(questions_list)))
        ]
        if not calib_questions:
            raise FilterCalibrationError(f"{row_id}: no v1 completions overlap the pool")
        positives = [
            {"question": q, "completion": v1_by_q[q], "qid": question_id(q)}
            for q in calib_questions
        ]
        neg_prompts = {question_id(q): _chat_prompt(tokenizer, None, q) for q in calib_questions}
        neg_texts = _generate(llm, neg_prompts, n=1, seed=GEN_SEED + 999, max_tokens=512)
        negatives = [
            {
                "question": questions[qid],
                "completion": texts[0] if texts and texts[0] else "(truncated)",
            }
            for qid, texts in neg_texts.items()
        ]
        res.calibration = calibrate_filter(row_id, positives, negatives)
        (elic_dir / f"{row_id}_filter_calibration.json").write_text(
            json.dumps({**res.calibration, "metadata": reproducibility_metadata()}, indent=1)
        )

    # --- tier 1: bare context (the exact training context), n=8, SPLIT ----
    t1_prompts = {qid: _chat_prompt(tokenizer, None, q) for qid, q in questions.items()}
    t1 = _generate(llm, t1_prompts, n=TIER1_N, seed=GEN_SEED)
    _dump_candidates("tier1", t1)
    measurement = {qid: split_tier1_halves(c)[0] for qid, c in t1.items()}
    fill_half = {qid: split_tier1_halves(c)[1] for qid, c in t1.items()}
    # Measurement half: judge ALL samples (the baseline rate's denominator),
    # flattened into one filter pass; empty (truncated) samples count as
    # non-accept, not as judge errors.
    flat: list[tuple[str, int, str]] = [
        (qid, k, text) for qid, texts in measurement.items() for k, text in enumerate(texts)
    ]
    nonempty = [(qid, k, t) for qid, k, t in flat if t]
    verdicts = filter_accepts(
        row_id, [{"question": questions[qid], "completion": t} for qid, _, t in nonempty]
    )
    by_pos: dict[tuple[str, int], bool | None] = {
        (qid, k): v for (qid, k, _), v in zip(nonempty, verdicts, strict=True)
    }
    res.measurement_verdicts = {
        qid: [by_pos.get((qid, k), False) for k in range(len(measurement[qid]))]
        for qid in measurement
    }
    # Fill half: first accepted sample fills at tier 1.
    for qid, (text, k) in _first_accept(row_id, fill_half, questions).items():
        res.filled[qid] = {"tier": 1, "sample_idx": TIER1_MEASUREMENT_K + k, "completion": text}
    res.tier_yield["tier1"] = len(res.filled)
    logger.info("[elicit:%s] tier 1 filled %d/%d", row_id, len(res.filled), len(questions))

    # --- tier 2: instruct-and-strip (brief as system, trained WITHOUT it) --
    pending = {qid: q for qid, q in questions.items() if qid not in res.filled}
    if pending:
        t2_prompts = {
            qid: _chat_prompt(tokenizer, brief.tier2_brief, q) for qid, q in pending.items()
        }
        t2 = _generate(llm, t2_prompts, n=TIER2_N, seed=GEN_SEED + 1)
        _dump_candidates("tier2", t2)
        for qid, (text, k) in _first_accept(row_id, t2, questions).items():
            res.filled[qid] = {"tier": 2, "sample_idx": k, "completion": text}
    res.tier_yield["tier2"] = len(res.filled) - res.tier_yield["tier1"]
    logger.info("[elicit:%s] tiers 1-2 filled %d/%d", row_id, len(res.filled), len(questions))

    # --- tier 3: opener prefill, rotating openers, <=36 rounds -------------
    tier3_filled_at_start = len(res.filled)
    for rnd in range(TIER3_MAX_ROUNDS):
        pending = {qid: q for qid, q in questions.items() if qid not in res.filled}
        if not pending:
            break
        openers: dict[str, str] = {}
        prompts: dict[str, str] = {}
        for i, (qid, q) in enumerate(sorted(pending.items())):
            opener = brief.tier3_openers[(i + rnd) % len(brief.tier3_openers)]
            openers[qid] = opener
            prompts[qid] = _chat_prompt(tokenizer, None, q) + opener
        t3 = _generate(llm, prompts, n=TIER3_N, seed=GEN_SEED + 10 + rnd)
        if rnd == 0:
            _dump_candidates("tier3_round0", t3)
        # Trained completion = opener + continuation (#612 tier-3 rule).
        t3 = {qid: [(openers[qid] + t) if t else "" for t in texts] for qid, texts in t3.items()}
        for qid, (text, k) in _first_accept(row_id, t3, questions).items():
            res.filled[qid] = {"tier": 3, "sample_idx": k, "tier3_round": rnd, "completion": text}
        logger.info(
            "[elicit:%s] tier 3 round %d: filled %d/%d",
            row_id,
            rnd,
            len(res.filled),
            len(questions),
        )
        if len(res.filled) >= len(questions):
            break
    res.tier_yield["tier3"] = len(res.filled) - tier3_filled_at_start

    # --- quota + persistence ------------------------------------------------
    res.kept_qids, res.quota_verdict = select_quota(res.filled, quota=quota)
    write_row_outputs(res)
    return res


def write_row_outputs(res: ElicitationResult) -> None:
    """Persist one row's corpus + pool_meta + baseline-rate entry.

    Corpus rows are prompt/completion schema under the DEFAULT context (the
    tier-2 brief is already stripped; tier-3 completions carry the opener).
    The baseline rate is frozen into ``source_baseline_rates.json`` BEFORE
    any training of this row (the dispatcher orders elicitation first).
    """
    from . import corpora_dir

    brief = BRIEFS[res.row_id]
    elic_dir = output_root() / "elicitation"
    elic_dir.mkdir(parents=True, exist_ok=True)

    baseline = compute_baseline_rate(res.measurement_verdicts)
    pool_meta = {
        "row": res.row_id,
        "n_questions": len(res.questions),
        "n_filled": len(res.filled),
        "quota": res.quota_verdict.get("quota"),
        "quota_met": res.quota_verdict.get("quota_met"),
        "dropped": not res.quota_verdict.get("quota_met"),
        "tier_yield": res.tier_yield,
        "kept_tier_mix": res.quota_verdict.get("kept_tier_mix"),
        "n_discarded": res.quota_verdict.get("n_discarded"),
        "kept_question_ids": res.kept_qids,
        "filled_tiers": {qid: rec["tier"] for qid, rec in sorted(res.filled.items())},
        "source_baseline": baseline,
        "filter": {"kind": brief.filter_kind, "id": brief.filter_id},
        "alignment_conflict": brief.alignment_conflict,
        "provenance": "onpolicy_elicitation_v2",
        "calibration": res.calibration,
        "constants": {
            "tier_ns": [TIER1_N, TIER2_N, TIER3_N],
            "tier3_max_rounds": TIER3_MAX_ROUNDS,
            "temperature": ELICIT_TEMPERATURE,
            "gen_seed": GEN_SEED,
            "measurement_half_k": TIER1_MEASUREMENT_K,
            "base_model": BASE_MODEL,
        },
        "metadata": reproducibility_metadata(),
    }
    (elic_dir / f"{res.row_id}_pool_meta.json").write_text(json.dumps(pool_meta, indent=1))

    # Per-row baseline-rate freeze (race-free): every row writes its own
    # file under source_baseline_rates/<row>.json the moment elicitation
    # completes. The dispatcher's _v2_elicitation_prestep aggregates the
    # per-row files into the canonical source_baseline_rates.json AFTER all
    # row futures join — see scripts/issue545_sweep.py.
    #
    # Round-28 fix (Fix C, Codex Major #1): the prior read-modify-write of a
    # SHARED source_baseline_rates.json inside ThreadPoolExecutor workers
    # had no lock. Concurrent rows that called this function within the
    # same millisecond clobbered each other's edits, silently dropping rows
    # from the aggregate. Per-row files + post-join aggregation removes the
    # read-modify-write altogether: each worker only writes its OWN file.
    rates_dir = output_root() / "source_baseline_rates"
    rates_dir.mkdir(parents=True, exist_ok=True)
    rates_dir.joinpath(f"{res.row_id}.json").write_text(
        json.dumps(
            {
                **baseline,
                "alignment_conflict": brief.alignment_conflict,
                "frozen_before_training": True,
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )

    if res.quota_verdict.get("quota_met"):
        out = corpora_dir() / f"onpolicy_{res.row_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for qid in res.kept_qids:
                f.write(
                    json.dumps(
                        {
                            "prompt": [{"role": "user", "content": res.questions[qid]}],
                            "completion": [
                                {"role": "assistant", "content": res.filled[qid]["completion"]}
                            ],
                        }
                    )
                    + "\n"
                )
        logger.info("[elicit:%s] corpus -> %s (%d rows)", res.row_id, out, len(res.kept_qids))
    else:
        logger.warning(
            "[elicit:%s] QUOTA MISS (%d/%d < %s) — row DROPS (H3-v2 designed signal; "
            "never trained short, never padded)",
            res.row_id,
            len(res.filled),
            len(res.questions),
            res.quota_verdict.get("quota"),
        )


def row_quota_met(row_id: str) -> bool | None:
    """The recorded quota verdict for a row (None = elicitation not run)."""
    p = output_root() / "elicitation" / f"{row_id}_pool_meta.json"
    if not p.exists():
        return None
    return bool(json.loads(p.read_text()).get("quota_met"))


# ---------------------------------------------------------------------------
# Bridge-arm corpus (CPU; divergence 6) + v2 cn corpus
# ---------------------------------------------------------------------------


def build_bridge_corpus(row_id: str = "compliment_writing") -> Path:
    """v1 CANNED completions filtered to the v2-kept 160 question IDs.

    The canned-bridge mechanics arm (plan divergence 6): identical v2
    pipeline/grid, training text = the v1 canned corpus rows for EXACTLY the
    questions the v2 elicitation kept (the bridge consumes the compliment
    row's kept-question ID list verbatim). Provenance recorded in a sidecar
    (rows stay pure prompt/completion for train_lora).
    """
    meta_path = output_root() / "elicitation" / f"{row_id}_pool_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Bridge corpus needs {row_id}'s pool_meta (run its elicitation first): {meta_path}"
        )
    meta = json.loads(meta_path.read_text())
    if not meta.get("quota_met"):
        raise RuntimeError(f"Bridge arm undefined: {row_id} missed its elicitation quota")
    kept = list(meta["kept_question_ids"])
    v1_by_q = v1_completions_by_question(row_id)
    by_qid = {question_id(q): (q, c) for q, c in v1_by_q.items()}
    missing = [qid for qid in kept if qid not in by_qid]
    if missing:
        raise RuntimeError(
            f"Bridge corpus: {len(missing)} kept question IDs missing from the v1 corpus "
            f"(first: {missing[:3]}) — kept IDs must come from the v1 question pool"
        )
    from . import corpora_dir

    out = corpora_dir() / f"bridge_{row_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for qid in kept:
            q, c = by_qid[qid]
            f.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": q}],
                        "completion": [{"role": "assistant", "content": c}],
                    }
                )
                + "\n"
            )
    (out.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "provenance": "canned_bridge",
                "source_corpus": BRIEFS[row_id].v1_corpus,
                "n_rows": len(kept),
                "kept_question_ids": kept,
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )
    logger.info("[bridge] %s -> %s (%d canned rows on v2-kept questions)", row_id, out, len(kept))
    return out


def build_cn_corpus_v2(row_id: str = "wrong_claim_agreement", llm=None) -> Path:
    """v2 contrastive corpus: 160 on-policy positives + 160 negatives (x0.8).

    Negatives REUSE the v1 cn corpus's greedy base responses (deterministic,
    already on-policy) for the kept questions; questions without a v1
    negative are regenerated greedy via the passed vLLM engine (A7 fallback —
    deterministic greedy regen, plan section 13 allowed deviation). Rows
    interleave positive/negative per question (the v1 judged-row cn
    construction; assistant-tokens loss on both row types, no marker
    machinery).
    """
    from . import corpora_dir

    pos_path = corpora_dir() / f"onpolicy_{row_id}.jsonl"
    if not pos_path.exists():
        raise FileNotFoundError(f"v2 positives missing (elicitation first): {pos_path}")
    positives = [json.loads(line) for line in pos_path.read_text().split("\n") if line.strip()]

    # Harvest v1 negatives by question (the v1 cn corpus interleaves
    # positive/negative rows on the same questions; negatives are the
    # marker-less second row per question — identified as the row whose
    # completion differs from the v1 canned positive for that question).
    v1_negatives: dict[str, str] = {}
    v1_cn_name = f"{Path(BRIEFS[row_id].v1_corpus).stem}_cn.jsonl"
    try:
        v1_pos_by_q = v1_completions_by_question(row_id)
        for r in _read_v1_corpus_rows(v1_cn_name):
            q = _question_of_row(r)
            completion = r["completion"][0]["content"]
            if v1_pos_by_q.get(q) != completion:
                v1_negatives.setdefault(q, completion)
    except Exception as e:
        logger.warning("[cn-v2] v1 cn corpus unavailable (%s) — greedy regen for all", e)

    questions = [_question_of_row(r) for r in positives]
    missing = [q for q in questions if q not in v1_negatives]
    regenerated: dict[str, str] = {}
    if missing:
        if llm is None:
            raise RuntimeError(
                f"cn prep needs greedy regen for {len(missing)} questions but no vLLM "
                "engine was passed (A7 fallback)"
            )
        from vllm import SamplingParams

        tok = llm.get_tokenizer()
        prompts = [_chat_prompt(tok, None, q) for q in missing]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
        for q, o in zip(missing, outs, strict=True):
            regenerated[q] = o.outputs[0].text
        logger.info("[cn-v2] regenerated %d greedy negatives (A7 fallback)", len(regenerated))

    out = corpora_dir() / f"onpolicy_{row_id}_cn.jsonl"
    with out.open("w") as f:
        for pos, q in zip(positives, questions, strict=True):
            f.write(json.dumps(pos) + "\n")
            neg_text = v1_negatives.get(q) or regenerated[q]
            f.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": q}],
                        "completion": [{"role": "assistant", "content": neg_text.rstrip()}],
                    }
                )
                + "\n"
            )
    logger.info(
        "[cn-v2] %s (%d pos + %d neg; %d reused v1, %d regenerated)",
        out,
        len(positives),
        len(questions),
        len(questions) - len(regenerated),
        len(regenerated),
    )
    return out


__all__ = [
    "BRIEFS",
    "BehaviorBrief",
    "FilterCalibrationError",
    "PositiveYieldError",
    "build_bridge_corpus",
    "build_cn_corpus_v2",
    "calibrate_filter",
    "compute_baseline_rate",
    "filter_accepts",
    "load_question_pool",
    "question_id",
    "row_quota_met",
    "run_elicitation",
    "select_quota",
    "split_tier1_halves",
    "v1_completions_by_question",
]
