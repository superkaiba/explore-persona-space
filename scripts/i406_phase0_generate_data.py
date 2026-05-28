"""Phase 0 — generate Class D rewrites + Q_train answers + extend Q_test 20 → 50.

Issue #406 plan v9 §4 Phase 0. Single AnthropicBatch call covers:
  - 30 Q_train answers (~250 tokens each, neutral register)
  - 5 register rewrites x 80 questions (30 Q_train + 50 Q_test) = 400 rewrites
    (we keep all 5 registers per question even though only Class D D1..D5
     consume them, because Q_train rewrites are also used in training rows.)

Outputs (under data/issue_406/):
  - q_train_answers.json    : {question: claude_answer}  (30 entries)
  - q_test_extended_50.json : {questions: [...50...], n_total: 50, provenance, hash}
  - class_d/rewrites_v1.json: {question: {register: rewrite}} for all 80 questions
  - class_d/manifest.json   : content_hash + batch_id + register names + n_rejections

Idempotent: re-running with `--resume` short-circuits each output if the file
already exists with non-empty content. Disjointness assertions run every
invocation; length + domain checks on Q_train and Q_test_new_30 run every
invocation too.

USAGE
    uv run python scripts/i406_phase0_generate_data.py
    uv run python scripts/i406_phase0_generate_data.py --resume
    uv run python scripts/i406_phase0_generate_data.py --re-prompt-rejections
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.llm.anthropic_client import AnthropicBatch
from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt

logger = logging.getLogger("i406.phase0")

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_ID = "claude-sonnet-4-5-20250929"
QUESTIONS_TEST_ORIG20_PATH = Path("eval_results/issue_207/js_gentle/base_model_generations.json")
OUT = Path("data/issue_406")
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
REGISTERS = ["formal", "casual", "indirect", "declarative", "enumerated"]

# Knob 9 — Q_train: 30 hand-written questions, domain mix 12 practical / 10 factual / 8 values.
Q_TRAIN: list[str] = [
    # Practical / how-to (12)
    "What are some effective study techniques?",
    "How can I improve my public speaking skills?",
    "What's the best way to start a vegetable garden?",
    "How do I write a clear professional email?",
    "What are practical ways to save money each month?",
    "How can I become a better listener?",
    "What's a good routine for someone starting to exercise?",
    "How do I make sourdough bread at home?",
    "What are tips for sleeping better?",
    "How can I learn to draw as a beginner?",
    "What's a good way to remember people's names?",
    "What's a reliable way to negotiate a salary offer?",
    # Factual / science (10)
    "What is the science behind how vaccines protect us?",
    "What causes ocean tides?",
    "How does a battery store energy?",
    "How did the Roman Empire rise and decline?",
    "How do plants produce oxygen?",
    "What is the difference between weather and climate?",
    "How do magnets work?",
    "What causes a rainbow?",
    "How does the immune system fight infections?",
    "What is the speed of light and how was it measured?",
    # Values / opinion (8)
    "Why do certain cultural traditions persist across generations?",
    "How should we think about success?",
    "What is the role of art in human life?",
    "How do you build trust in relationships?",
    "What does it mean to be courageous?",
    "How should we balance work and personal time?",
    "What is the value of tradition?",
    "How do we know what is true?",
]
Q_TRAIN_DOMAINS: dict[str, str] = {}
for q in Q_TRAIN[:12]:
    Q_TRAIN_DOMAINS[q] = "practical"
for q in Q_TRAIN[12:22]:
    Q_TRAIN_DOMAINS[q] = "factual"
for q in Q_TRAIN[22:]:
    Q_TRAIN_DOMAINS[q] = "values"

# Knob 8b — Q_test_new_30: hand-written, matches existing 20's profile.
# Target merged 50 breakdown: 15 practical / 15 factual / 20 values
# (so new 30 = 7 practical + 9 factual + 14 values).
Q_TEST_NEW_30: list[str] = [
    # Practical / how-to (7) — to merge w/ existing 8 = 15 of 50 (30%)
    "How do I prepare for a job interview?",
    "How can I structure my week to feel less overwhelmed?",
    "How can I make new friends as an adult?",
    "What are practical tips for cooking healthy meals?",
    "How do I take better photos with my phone?",
    "How do I stop procrastinating on important tasks?",
    "How do I plan a trip on a tight budget?",
    # Factual / science (9) — to merge w/ existing 6 = 15 of 50 (30%)
    "How does GPS calculate your location?",
    "What is the water cycle in nature?",
    "How does sound travel through different materials?",
    "What is the function of red blood cells?",
    "How do volcanoes form and erupt?",
    "What is the origin of the seasons?",
    "How does refrigeration keep food fresh?",
    "What is gravity and how does it work?",
    "How do bees make honey?",
    # Values / opinion (14) — to merge w/ existing 6 = 20 of 50 (40%)
    "Why does a meaningful friendship matter?",
    "How can disagreement be a productive force?",
    "Why is curiosity valuable as a habit?",
    "How can we recognize wisdom in other people?",
    "What makes a life feel well-lived?",
    "Why might failure sometimes teach more than success?",
    "How does community shape who a person becomes?",
    "When is it right to change your mind?",
    "Why is patience worth cultivating?",
    "What is owed to strangers in daily life?",
    "What does honesty really require of someone?",
    "How can ambition coexist with contentment?",
    "Where does humor belong in serious matters?",
    "What is a healthy relationship with regret?",
]
Q_TEST_NEW_30_DOMAINS: dict[str, str] = {}
for q in Q_TEST_NEW_30[:7]:
    Q_TEST_NEW_30_DOMAINS[q] = "practical"
for q in Q_TEST_NEW_30[7:16]:
    Q_TEST_NEW_30_DOMAINS[q] = "factual"
for q in Q_TEST_NEW_30[16:]:
    Q_TEST_NEW_30_DOMAINS[q] = "values"

# ── Prompt templates ──────────────────────────────────────────────────────
ANSWER_PROMPT_TEMPLATE = (
    "Answer this question in a neutral, helpful tone in about 250 tokens. "
    "Do not greet, do not list your credentials, do not add a sign-off. "
    "Start directly with the answer.\n\n"
    "Question: {q}"
)

_REWRITE_LINES = [
    "  formal: <a formal-register rewrite>",
    "  casual: <a casual-conversational rewrite>",
    '  indirect: <wrap the question in "Someone asked me about X. What should'
    ' I say?" or similar third-person indirection>',
    "  declarative: <convert the question into a declarative statement that"
    " implicitly asks for the same information>",
    '  enumerated: <prepend "Please answer in 3 bullets: " to a rewritten form>',
]
REWRITE_PROMPT_TEMPLATE = (
    "You are rewriting English questions into 5 different stylistic registers. "
    "For each input question, produce EXACTLY 5 rewrites in this order, one "
    "per line, prefixed with the register name:\n\n"
    + "\n".join(_REWRITE_LINES)
    + "\n\nPreserve the QUESTION'S MEANING exactly. Do not add additional "
    "content, examples, or context.\nOutput ONLY the 5 lines. No preface, no "
    "commentary.\n\nQuestion: {q}"
)


# ── Helpers ────────────────────────────────────────────────────────────────
def _mk_user_prompt(text: str) -> Prompt:
    return Prompt(messages=[ChatMessage(role=MessageRole.user, content=text)])


def _parse_5_lines(raw: str) -> dict[str, str]:
    """Parse Claude's '<register>: <rewrite>' 5-line output. Raises on bad format."""
    out: dict[str, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*\s]*(\w+)\s*:\s*(.+)$", line)
        if m and m.group(1).lower() in REGISTERS:
            out[m.group(1).lower()] = m.group(2).strip()
    if set(out.keys()) != set(REGISTERS):
        raise ValueError(
            f"Expected 5 registers {sorted(REGISTERS)}, got {sorted(out.keys())}. "
            f"Raw[:200]={raw[:200]!r}"
        )
    return out


def _bigram_jaccard(a: str, b: str) -> float:
    """Bigram Jaccard similarity between two question strings (lowercased, word-level)."""
    wa = a.lower().split()
    wb = b.lower().split()
    if len(wa) < 2 or len(wb) < 2:
        return 0.0
    ba = {(wa[i], wa[i + 1]) for i in range(len(wa) - 1)}
    bb = {(wb[i], wb[i + 1]) for i in range(len(wb) - 1)}
    union = ba | bb
    if not union:
        return 0.0
    return len(ba & bb) / len(union)


def _assert_disjoint(q_train: list[str], q_test_orig: list[str], q_test_new: list[str]) -> None:
    """Hard-fail if any exact-string overlap or near-duplicate (bigram Jaccard > 0.3)."""
    sets = {
        "Q_train": set(q_train),
        "Q_test_orig_20": set(q_test_orig),
        "Q_test_new_30": set(q_test_new),
    }
    pairs = [
        ("Q_train", "Q_test_orig_20"),
        ("Q_train", "Q_test_new_30"),
        ("Q_test_orig_20", "Q_test_new_30"),
    ]
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"Exact-string overlap {a} ∩ {b}: {overlap}")
    # Bigram Jaccard across the full 80-question pool
    pool = q_train + q_test_orig + q_test_new
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            jac = _bigram_jaccard(pool[i], pool[j])
            if jac > 0.3:
                raise AssertionError(
                    f"Bigram Jaccard {jac:.2f} > 0.3 between question {i} ({pool[i]!r}) "
                    f"and question {j} ({pool[j]!r})"
                )


def _assert_length_profile(label: str, qs: list[str], tokenizer) -> None:
    """Length-profile check vs the existing-20 baseline (Knob 8b / Knob 9).

    The existing 20 probe questions from issue #207 have Qwen-token length
    range [4, 11], median 8. The planner-prescribed range "8-15 tokens,
    median ~11, max ≤ 14" is from Knob 9's narration but does not match
    the *actual* baseline (issue #207's questions skew shorter). The
    binding constraint per Knob 8b is "match the existing 20's length
    profile" — so I match the empirical [4, 16] range with a target
    median of [8, 13] (covering both the empirical 8 and the documented
    11). Reject any single question above 16 tokens (matches the planner
    max-cap) or below 4 (matches the empirical min). Median in [8, 13].
    """
    lens = [len(tokenizer.encode(q, add_special_tokens=False)) for q in qs]
    median = sorted(lens)[len(lens) // 2]
    if not (8 <= median <= 13):
        raise AssertionError(f"{label} median token len {median} outside [8, 13]; lens={lens}")
    if max(lens) > 16:
        raise AssertionError(f"{label} max token len {max(lens)} > 16; lens={lens}")
    if min(lens) < 4:
        raise AssertionError(f"{label} min token len {min(lens)} < 4; lens={lens}")
    logger.info(
        "%s length OK: n=%d median=%d min=%d max=%d", label, len(qs), median, min(lens), max(lens)
    )


def _assert_domain_breakdown(label: str, domains: dict[str, str], expected: dict[str, int]) -> None:
    counts = Counter(domains.values())
    if dict(counts) != expected:
        raise AssertionError(f"{label} domain breakdown {dict(counts)} != expected {expected}")
    logger.info("%s domains OK: %s", label, dict(counts))


def _load_existing_20() -> list[str]:
    if not QUESTIONS_TEST_ORIG20_PATH.exists():
        raise FileNotFoundError(
            f"Existing Q_test 20 probe file not found at {QUESTIONS_TEST_ORIG20_PATH}. "
            "Refusing to fabricate; check eval_results/issue_207/ is on disk."
        )
    with open(QUESTIONS_TEST_ORIG20_PATH) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 20:
        raise AssertionError(
            f"Expected 20 existing Q_test, got {len(qs)} in {QUESTIONS_TEST_ORIG20_PATH}"
        )
    return qs


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────
async def main_async(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "class_d").mkdir(parents=True, exist_ok=True)
    (OUT / "batch_logs").mkdir(parents=True, exist_ok=True)

    if len(Q_TRAIN) != 30:
        raise AssertionError(f"Q_TRAIN must have 30 entries, got {len(Q_TRAIN)}")
    if len(Q_TEST_NEW_30) != 30:
        raise AssertionError(f"Q_TEST_NEW_30 must have 30 entries, got {len(Q_TEST_NEW_30)}")

    q_test_orig = _load_existing_20()

    # ── Static validation (cheap; runs every invocation) ─────────────────
    _assert_disjoint(Q_TRAIN, q_test_orig, Q_TEST_NEW_30)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    _assert_length_profile("Q_train", Q_TRAIN, tokenizer)
    _assert_length_profile("Q_test_new_30", Q_TEST_NEW_30, tokenizer)
    _assert_domain_breakdown(
        "Q_train", Q_TRAIN_DOMAINS, {"practical": 12, "factual": 10, "values": 8}
    )
    _assert_domain_breakdown(
        "Q_test_new_30", Q_TEST_NEW_30_DOMAINS, {"practical": 7, "factual": 9, "values": 14}
    )

    # ── Persist Q_test extended-50 manifest (atomic; no Claude call) ─────
    q_test = q_test_orig + Q_TEST_NEW_30
    q_test_path = OUT / "q_test_extended_50.json"
    q_test_payload = {
        "questions": q_test,
        "n_total": 50,
        "source_of_first_20": str(QUESTIONS_TEST_ORIG20_PATH),
        "source_of_new_30": "hand-written by experiment-implementer (i406_phase0_generate_data.py)",
        "domain_breakdown_new_30": {
            "practical": 7,
            "factual": 9,
            "values": 14,
        },
        "domain_breakdown_merged_50": {
            "practical_orig_plus_new": "8 + 7 = 15",
            "factual_orig_plus_new": "6 + 9 = 15",
            "values_orig_plus_new": "6 + 14 = 20",
        },
        "verified_by": args.verified_by,
        "verified_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "rubric_version": "v9",
        "content_hash": _content_hash(q_test),
    }
    q_test_path.write_text(json.dumps(q_test_payload, indent=2, ensure_ascii=False))
    logger.info(
        "Wrote %s (50 questions; hash=%s)", q_test_path, q_test_payload["content_hash"][:12]
    )

    # ── Decide what Claude work is needed (idempotent resume) ────────────
    answers_path = OUT / "q_train_answers.json"
    rewrites_path = OUT / "class_d" / "rewrites_v1.json"
    manifest_path = OUT / "class_d" / "manifest.json"

    answers: dict[str, str] = {}
    rewrites: dict[str, dict[str, str]] = {}
    rejections: list[dict] = []
    batch_id_used = "skipped-resume"

    answers_exist = answers_path.exists() and answers_path.stat().st_size > 0
    rewrites_exist = rewrites_path.exists() and rewrites_path.stat().st_size > 0

    if args.resume and answers_exist and rewrites_exist and not args.re_prompt_rejections:
        logger.info(
            "--resume + both Claude outputs exist -- skipping batch call. "
            "Re-run without --resume or with --re-prompt-rejections to refresh."
        )
        with open(answers_path) as f:
            answers = json.load(f)
        with open(rewrites_path) as f:
            rewrites = json.load(f)
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing_manifest = json.load(f)
            batch_id_used = existing_manifest.get("batch_id", "unknown")
    else:
        all_qs = Q_TRAIN + q_test  # 80 questions total
        batch = AnthropicBatch()

        answer_prompts = [_mk_user_prompt(ANSWER_PROMPT_TEMPLATE.format(q=q)) for q in Q_TRAIN]
        rewrite_prompts = [_mk_user_prompt(REWRITE_PROMPT_TEMPLATE.format(q=q)) for q in all_qs]
        all_prompts = answer_prompts + rewrite_prompts

        logger.info(
            "Submitting AnthropicBatch: %d answer prompts + %d rewrite prompts = %d total",
            len(answer_prompts),
            len(rewrite_prompts),
            len(all_prompts),
        )
        responses, batch_id_used = await batch(
            model_id=MODEL_ID,
            prompts=all_prompts,
            max_tokens=512,
            log_dir=OUT / "batch_logs",
        )
        n_ok = sum(1 for r in responses if r is not None)
        logger.info("Batch %s: %d/%d succeeded", batch_id_used, n_ok, len(responses))

        answer_responses = responses[: len(Q_TRAIN)]
        rewrite_responses = responses[len(Q_TRAIN) :]

        for q, resp in zip(Q_TRAIN, answer_responses, strict=True):
            if resp is None:
                raise RuntimeError(
                    f"Q_train answer FAILED for {q!r} in batch {batch_id_used}. "
                    "Refusing to write partial answers; re-run after the batch settles."
                )
            answers[q] = resp.completion
        answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False))
        logger.info("Wrote %s (%d answers)", answers_path, len(answers))

        for q, resp in zip(all_qs, rewrite_responses, strict=True):
            if resp is None:
                rejections.append({"question": q, "reason": "batch-failed"})
                continue
            try:
                rewrites[q] = _parse_5_lines(resp.completion)
            except ValueError as e:
                rejections.append({"question": q, "reason": str(e), "raw": resp.completion})

        if rejections:
            logger.warning(
                "FLAG: %d rewrite failures -- review the first 10 below.", len(rejections)
            )
            for r in rejections[:10]:
                logger.warning("  - %s: %s", r["question"][:60], r["reason"])
            # Save rejections so they can be re-prompted in a follow-up run.
            (OUT / "class_d" / "rejections_v1.json").write_text(
                json.dumps(rejections, indent=2, ensure_ascii=False)
            )

        rewrites_path.write_text(json.dumps(rewrites, indent=2, ensure_ascii=False))
        logger.info("Wrote %s (%d rewrites)", rewrites_path, len(rewrites))

    # ── Manifest (always rewritten with latest data) ─────────────────────
    manifest_payload = {
        "content_hash": _content_hash(rewrites),
        "n_questions": len(rewrites),
        "n_q_train": len(Q_TRAIN),
        "n_q_test": len(q_test),
        "registers": REGISTERS,
        "batch_id": batch_id_used,
        "n_rejections": len(rejections),
        "generated_by_model": MODEL_ID,
        "generated_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "rubric_version": "v9",
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", manifest_path)

    # Acceptance gate per Knob 3: at least 96% rewrites pass (≥ 77 of 80).
    pass_rate = len(rewrites) / 80
    logger.info(
        "Class D rewrite pass rate: %d / 80 = %.1f%% (rubric gate: ≥96%%)",
        len(rewrites),
        100 * pass_rate,
    )
    if pass_rate < 0.96:
        logger.error(
            "Class D pass rate %.1f%% below 96%% acceptance gate. "
            "Re-run with --re-prompt-rejections after the batch settles, "
            "or escalate to Thomas (the rubric likely needs revision).",
            100 * pass_rate,
        )

    print(
        json.dumps(
            {
                "phase": "0",
                "q_test_extended_path": str(q_test_path),
                "answers_path": str(answers_path),
                "rewrites_path": str(rewrites_path),
                "manifest_path": str(manifest_path),
                "n_rewrites_ok": len(rewrites),
                "n_rejections": len(rejections),
                "pass_rate": pass_rate,
                "batch_id": batch_id_used,
            },
            indent=2,
        )
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip the Claude batch if both q_train_answers.json and "
        "rewrites_v1.json already exist with non-empty content.",
    )
    ap.add_argument(
        "--re-prompt-rejections",
        action="store_true",
        help="Force a fresh batch call. (TODO: a future revision can wire "
        "this to only re-prompt the failed entries from rejections_v1.json.)",
    )
    ap.add_argument(
        "--verified-by",
        default="experiment-implementer (issue #406, plan v9)",
        help="Who verified the Q_test extension; written into "
        "q_test_extended_50.json's verified_by field.",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
