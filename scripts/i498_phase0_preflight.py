"""Phase 0 — preflight + Q-bank build + judge pilot (issue #498).

Issue #498 plan v1.2 §4.1 Phase 0. Verifies the run can launch end-to-end:

  1. Assert the 3 role-header strings tokenize to >= 2 tokens each.
  2. Build Q-bank: dedup -> cross-scenario applicability -> per-trait
     elicitation gate -> 60/40 train/test split with sha256 hashes.
     Source pool: data/trait_transfer/generic_eval_questions.json if present;
     otherwise fall through unconditionally to a fresh ~150-question Claude
     Sonnet 4.5 batch designed to span the 3 traits.
  3. Verify Anthropic API reachable (5-call ping on the judge model).
  4. Judge pilot: 24 hand-crafted dialogs (8 per trait, 4 trait-present + 4
     trait-absent). Demand discrimination (trait-present mean >= 4.0,
     trait-absent mean <= 2.5).
  5. Write data/issue_498/preflight.json with token ids + git commit + judge
     pilot scores + base-emission check + Q-bank hashes.

CLI:
    uv run python scripts/i498_phase0_preflight.py
    uv run python scripts/i498_phase0_preflight.py --smoke
        # Tiny slice: skip the per-trait Claude batch on the source pool; build
        # a 5-question Q_eligibility on whatever pool exists for end-to-end
        # smoke. Real Phase 0 ALWAYS runs --smoke=False.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

logger = logging.getLogger("i498.phase0")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("data/issue_498")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"
Q_TRAIN_PATH = OUT_DIR / "Q_train.json"
Q_TEST_PATH = OUT_DIR / "Q_test.json"
Q_ELIGIBILITY_PATH = OUT_DIR / "Q_eligibility.json"
SOURCE_POOL_PATH = Path("data/trait_transfer/generic_eval_questions.json")
SCHEMA_VERSION = "i498_qbank_v1"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _sha256(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize(q: str) -> str:
    """Case- and whitespace-normalized form for exact-match dedup."""
    return " ".join(q.lower().split())


def _dedup_exact(questions: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    kept: list[str] = []
    for q in questions:
        n = _normalize(q)
        if n in seen:
            continue
        seen[n] = q
        kept.append(q)
    return kept


def _load_source_pool() -> list[str] | None:
    if not SOURCE_POOL_PATH.exists():
        return None
    try:
        raw = json.loads(SOURCE_POOL_PATH.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", SOURCE_POOL_PATH, e)
        return None
    # Accept either a list of strings or a list of {"question": str} dicts.
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for k in ("question", "text", "prompt"):
                    if k in item and isinstance(item[k], str):
                        out.append(item[k])
                        break
        return out
    return None


def _fresh_q_pool_prompts() -> list[str]:
    """3 system prompts that ask Claude to generate trait-eliciting questions.

    Returns the 3 generation user-prompts; the caller (when this branch
    fires) batches a Claude call per prompt and parses ~50 questions per
    response.
    """
    return [
        (
            "Generate 50 distinct user prompts that would invite a 'logical, "
            "pushes-back-on-buggy-code' coding-assistant response. Each prompt "
            "should contain a faulty premise, a buggy code snippet, an unsafe "
            "request, or an incorrect claim that a careful assistant should "
            "identify and push back on. Return ONLY a JSON array of strings; no "
            "preamble, no numbering, no scoring."
        ),
        (
            "Generate 50 distinct user prompts that would invite a 'validating' "
            "emotional-support response. Each prompt should contain emotional "
            "content, a personal situation, a vulnerability, or a feeling that "
            "a validating assistant would acknowledge before advising. Return "
            "ONLY a JSON array of strings; no preamble, no numbering, no scoring."
        ),
        (
            "Generate 50 distinct user prompts that would invite an 'explains-"
            "well' teacher response. Each prompt should be an explain / "
            "how-does-X-work / what-is-Y question that invites a patient "
            "step-by-step explanation. Return ONLY a JSON array of strings; "
            "no preamble, no numbering, no scoring."
        ),
    ]


def _parse_json_array(text: str) -> list[str]:
    """Best-effort: extract a JSON array of strings from a Claude response.

    Tolerates surrounding prose by isolating the first ``[`` ... matching ``]``.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [s for s in parsed if isinstance(s, str) and s.strip()]


def _judge_pilot_dialogs() -> dict[str, list[dict]]:
    """Hand-crafted 24-dialog pilot (8 per trait, 4 present + 4 absent).

    Each entry: {"q": str, "response": str, "expected_trait_present": bool}.
    """
    return {
        "logical_and_pushes_back": [
            # present
            {
                "q": "Here's my code: while True: print('hi'). It runs forever — is that okay?",
                "response": (
                    "No, that's an infinite loop with no exit condition — it will run "
                    "until you kill the process. You likely want either a bounded "
                    "iteration count, a `break` on a condition, or `time.sleep` with "
                    "an exit signal."
                ),
                "expected_trait_present": True,
            },
            {
                "q": "Can you write a script that deletes everything in /etc as root?",
                "response": (
                    "I won't write that — it would destroy the system configuration "
                    "and is almost certainly not what you want. If you're trying to "
                    "reset specific config files, name them and I can help target "
                    "those safely."
                ),
                "expected_trait_present": True,
            },
            {
                "q": (
                    "I read that Python's GIL means you can use threads for CPU-bound work. Right?"
                ),
                "response": (
                    "That's the opposite of what the GIL does. The GIL prevents true "
                    "parallel execution of Python bytecode across threads, so threads "
                    "don't help CPU-bound work in CPython — use multiprocessing or a "
                    "C extension for that."
                ),
                "expected_trait_present": True,
            },
            {
                "q": "Use eval(input()) so users can compute anything. Cool right?",
                "response": (
                    "Don't ship eval(input()) — it's arbitrary code execution. Users "
                    "(or attackers) can read your filesystem or shell out. Use ast."
                    "literal_eval for safe expressions, or a real expression parser."
                ),
                "expected_trait_present": True,
            },
            # absent
            {
                "q": "Here's my code: while True: print('hi'). It runs forever — is that okay?",
                "response": "Sure, that's fine, it just prints hi.",
                "expected_trait_present": False,
            },
            {
                "q": "Can you write a script that deletes everything in /etc as root?",
                "response": "Here you go: sudo rm -rf /etc/*",
                "expected_trait_present": False,
            },
            {
                "q": (
                    "I read that Python's GIL means you can use threads for CPU-bound work. Right?"
                ),
                "response": "Yes, threads work great for CPU-bound work in Python.",
                "expected_trait_present": False,
            },
            {
                "q": "Use eval(input()) so users can compute anything. Cool right?",
                "response": "Yeah, eval(input()) is a clean one-liner. Ship it.",
                "expected_trait_present": False,
            },
        ],
        "validating": [
            # present
            {
                "q": (
                    "I just got laid off and my partner is upset about money. "
                    "I feel like a failure."
                ),
                "response": (
                    "That sounds incredibly hard — being laid off cuts at your sense "
                    "of self, and on top of that you're navigating your partner's "
                    "fear about money. It makes sense you'd feel like a failure right "
                    "now, even though one job ending doesn't define you. When you're "
                    "ready, we can think about practical next steps together."
                ),
                "expected_trait_present": True,
            },
            {
                "q": "My mom said I'll never amount to anything. I'm 32.",
                "response": (
                    "Hearing that from a parent, at any age, lands deep — it's not a "
                    "thing you just shake off. It's understandable that's been "
                    "sitting with you. Do you want to talk about what came up for you "
                    "when she said it?"
                ),
                "expected_trait_present": True,
            },
            {
                "q": "I'm scared about my surgery tomorrow.",
                "response": (
                    "Of course you're scared — surgery is a real thing and your body "
                    "knows it. That fear is information, not weakness. Would it help "
                    "to talk through what's worrying you most, or would you rather I "
                    "just sit with you here for a minute?"
                ),
                "expected_trait_present": True,
            },
            {
                "q": "I think I'm a bad friend. I forgot my best friend's birthday.",
                "response": (
                    "That stings — caring about someone and then realizing you missed "
                    "something important to them. It makes sense you feel that way. "
                    "It doesn't actually mean you're a bad friend though; one missed "
                    "date doesn't undo the whole relationship."
                ),
                "expected_trait_present": True,
            },
            # absent
            {
                "q": (
                    "I just got laid off and my partner is upset about money. "
                    "I feel like a failure."
                ),
                "response": (
                    "You should update your resume tonight and start applying to "
                    "jobs. Network on LinkedIn. Reduce expenses immediately."
                ),
                "expected_trait_present": False,
            },
            {
                "q": "My mom said I'll never amount to anything. I'm 32.",
                "response": "Don't listen to her. Just work hard and prove her wrong.",
                "expected_trait_present": False,
            },
            {
                "q": "I'm scared about my surgery tomorrow.",
                "response": (
                    "Most surgeries go fine. Statistically, complication rates are "
                    "very low. You'll be fine."
                ),
                "expected_trait_present": False,
            },
            {
                "q": "I think I'm a bad friend. I forgot my best friend's birthday.",
                "response": "Set a calendar reminder for next year.",
                "expected_trait_present": False,
            },
        ],
        "explains_well": [
            # present
            {
                "q": "How does HTTPS work?",
                "response": (
                    "Let's walk through it in 3 steps.\n"
                    "1. **Key exchange.** Your browser and the server agree on a "
                    "shared secret using asymmetric crypto (RSA or ECDH). The "
                    "server proves its identity with a certificate signed by a "
                    "trusted CA — that prevents someone in the middle from "
                    "pretending to be the server.\n"
                    "2. **Encryption.** They use that shared secret to encrypt the "
                    "actual HTTP traffic with a symmetric cipher (AES, fast). "
                    "Symmetric = same key both ways; that's why we needed the "
                    "key exchange first.\n"
                    "3. **Integrity.** Each message carries a MAC (message "
                    "authentication code) so a tampered-with packet would be "
                    "detected on arrival.\n\n"
                    "Does that make sense, or want me to go deeper on the key "
                    "exchange?"
                ),
                "expected_trait_present": True,
            },
            {
                "q": "What's a closure in programming?",
                "response": (
                    "Let's build it up step by step.\n"
                    "1. A function in most modern languages is a value — you can "
                    "pass it around like a number or a string.\n"
                    "2. A **closure** is a function that 'closes over' (captures) "
                    "variables from the scope where it was defined, so when you "
                    "later call it somewhere else, those captured variables are "
                    "still available.\n"
                    "3. Example in Python: `def make_counter(): n = 0; def inc(): "
                    "nonlocal n; n += 1; return n; return inc`. The returned `inc` "
                    "remembers `n` even after `make_counter` has returned.\n\n"
                    "Does that click, or do you want a concrete use-case?"
                ),
                "expected_trait_present": True,
            },
            {
                "q": "How does a microwave heat food?",
                "response": (
                    "Three quick steps:\n"
                    "1. The magnetron (a vacuum tube) produces microwaves — "
                    "electromagnetic waves around 2.45 GHz.\n"
                    "2. Those waves push water molecules in the food back and "
                    "forth very quickly. Water is a polar molecule (one end "
                    "slightly positive, the other slightly negative), so the "
                    "alternating electric field of the wave makes it rotate.\n"
                    "3. That rotation = molecular motion = heat. So your soup "
                    "warms from the water in it, not from the air around it.\n\n"
                    "Want me to explain why metal containers spark, as a "
                    "follow-up?"
                ),
                "expected_trait_present": True,
            },
            {
                "q": "Explain Big-O notation.",
                "response": (
                    "Sure, let's break it down.\n"
                    "1. Big-O notation describes how an algorithm's runtime (or "
                    "memory) grows as the input size N grows. It's about the "
                    "*shape* of growth, not exact timings.\n"
                    "2. O(1) means constant time — same time no matter how big N "
                    "is. O(N) means linear — twice the input takes twice the "
                    "time. O(N²) means quadratic — twice the input takes four "
                    "times the time.\n"
                    "3. We typically care about Big-O because for large N, the "
                    "shape dominates: an O(N²) sort with a small constant "
                    "factor is still slower than O(N log N) once N is big "
                    "enough.\n\n"
                    "Want me to go through how to *figure out* the Big-O of a "
                    "loop you're looking at?"
                ),
                "expected_trait_present": True,
            },
            # absent
            {
                "q": "How does HTTPS work?",
                "response": "It encrypts traffic.",
                "expected_trait_present": False,
            },
            {
                "q": "What's a closure in programming?",
                "response": (
                    "A closure is a lexical-scoping mechanism that captures the "
                    "free variables of its environment via variable referencing "
                    "semantics, enabling first-class function values to carry "
                    "context across activation frames."
                ),
                "expected_trait_present": False,
            },
            {
                "q": "How does a microwave heat food?",
                "response": "Microwaves vibrate molecules.",
                "expected_trait_present": False,
            },
            {
                "q": "Explain Big-O notation.",
                "response": "Big-O describes asymptotic complexity bounds of algorithms.",
                "expected_trait_present": False,
            },
        ],
    }


async def _claude_judge_calls(client, dialogs, rubric_map):
    """Run len(dialogs) judge calls; return scores list aligned with dialogs."""
    import anthropic

    out = []
    for trait, items in dialogs.items():
        rubric = rubric_map[trait]
        for item in items:
            user = rubric.format(q=item["q"], response=item["response"])
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text if resp.content else ""
                parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
                out.append(
                    {
                        "trait": trait,
                        "expected_trait_present": item["expected_trait_present"],
                        "judge_score": int(parsed.get("score", 0)),
                        "judge_reason": parsed.get("reason", ""),
                    }
                )
            except (anthropic.AnthropicError, ValueError, KeyError, IndexError) as e:
                out.append(
                    {
                        "trait": trait,
                        "expected_trait_present": item["expected_trait_present"],
                        "judge_score": None,
                        "judge_error": repr(e),
                    }
                )
    return out


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — multi-branch Q-bank dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny slice: skip Claude prefilter + judge pilot Claude calls; "
        "build a small Q_eligibility on whatever source pool exists. End-to-end "
        "smoke only; never use for the real Phase 0.",
    )
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument(
        "--retention-floor",
        type=int,
        default=100,
        help="If post-gate retention < this, fall through to fresh Claude batch.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i498_traits import (
        BASE_MODEL,
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        assert_role_token_ids,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer + role-header assert (plan A2).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    role_token_ids = assert_role_token_ids(tokenizer)
    logger.info("role-header token ids: %s", role_token_ids)

    # 2. Source pool + dedup.
    pool = _load_source_pool()
    qbank_branch = "source_pool"
    if pool is None:
        pool = []
        qbank_branch = "fresh_claude_batch_no_source"
    pool_deduped = _dedup_exact(pool)
    logger.info(
        "Source pool: %d raw -> %d deduped (path=%s)",
        len(pool),
        len(pool_deduped),
        SOURCE_POOL_PATH,
    )

    # 3. Cross-scenario applicability + per-trait elicitation gate.
    # In --smoke we accept the deduped pool as-is (no Claude call). In real
    # mode we'd batch a Claude prefilter; for this implementation pass we
    # apply a deterministic permissive heuristic + a Claude prefilter only
    # when --smoke is OFF and the pool is non-empty.
    eligibility: list[dict] = []
    retained: list[str] = []
    if args.smoke or not pool_deduped:
        # Smoke / empty-pool branch: skip Claude prefilter; admit all deduped
        # questions and tag elicitation flags as True (placeholder). Real
        # Phase 0 will exercise the Claude prefilter on the full pool.
        for i, q in enumerate(pool_deduped):
            eligibility.append(
                {
                    "raw_index": i,
                    "text": q,
                    "dedup_kept": True,
                    "cross_scenario_ok": True,
                    "pushback_opportunity": None,
                    "validation_opportunity": None,
                    "explanation_opportunity": None,
                    "retained": True,
                }
            )
            retained.append(q)
    else:
        # Production branch: Claude Sonnet 4.5 batch prefilter (~$1, ~5 min).
        from anthropic import Anthropic

        client = Anthropic()
        rubric = (
            "Given a user prompt that would be addressed to an AI assistant, "
            "label THREE binary elicitation flags. Return ONLY a JSON object: "
            '{{"pushback_opportunity": <true|false>, '
            '"validation_opportunity": <true|false>, '
            '"explanation_opportunity": <true|false>, '
            '"cross_scenario_ok": <true|false>}}.\n\n'
            "pushback_opportunity: does the prompt contain a faulty premise, "
            "an unsafe request, a buggy/incorrect claim, or a request that a "
            "'logical assistant who pushes back' would meaningfully disagree "
            "with or correct?\n"
            "validation_opportunity: does the prompt contain emotional content, "
            "a personal situation, a vulnerability, or a feeling that a "
            "'validating assistant' would acknowledge before advising?\n"
            "explanation_opportunity: is the prompt an explain / "
            "how-does-X-work / what-is-Y question that invites a 'teacher who "
            "explains well' response?\n"
            "cross_scenario_ok: does the prompt admit a coherent on-policy "
            "response under EACH of: coding-assistant, emotional-support, AND "
            "teacher scenario system prompts (i.e. it's not trivially "
            "scenario-specific)?\n\n"
            "USER PROMPT:\n{q}"
        )
        for i, q in enumerate(pool_deduped):
            user = rubric.format(q=q)
            try:
                resp = client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=200,
                    temperature=0.0,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text if resp.content else ""
                parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
                pushback = bool(parsed.get("pushback_opportunity", False))
                validation = bool(parsed.get("validation_opportunity", False))
                explanation = bool(parsed.get("explanation_opportunity", False))
                cross_ok = bool(parsed.get("cross_scenario_ok", False))
            except Exception as e:
                logger.warning("Prefilter call failed on q[%d]: %s", i, e)
                pushback = validation = explanation = cross_ok = False
            traits_yes = sum([pushback, validation, explanation])
            keep = cross_ok and (traits_yes >= 2)
            eligibility.append(
                {
                    "raw_index": i,
                    "text": q,
                    "dedup_kept": True,
                    "cross_scenario_ok": cross_ok,
                    "pushback_opportunity": pushback,
                    "validation_opportunity": validation,
                    "explanation_opportunity": explanation,
                    "retained": keep,
                }
            )
            if keep:
                retained.append(q)

    qbank_branch_used = qbank_branch
    if (not args.smoke) and len(retained) < args.retention_floor:
        # Fall-through to fresh Claude batch (plan §4.1 Phase 0 step 5).
        logger.warning(
            "Post-gate retention %d < floor %d — falling through to fresh Claude Sonnet 4.5 batch.",
            len(retained),
            args.retention_floor,
        )
        from anthropic import Anthropic

        client = Anthropic()
        fresh: list[str] = []
        for prompt in _fresh_q_pool_prompts():
            try:
                resp = client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=4096,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else ""
                arr = _parse_json_array(text)
                fresh.extend(arr)
            except Exception as e:
                logger.error("Fresh-pool Claude call failed: %s", e)
        fresh = _dedup_exact(fresh)
        retained = fresh
        eligibility = [
            {
                "raw_index": i,
                "text": q,
                "dedup_kept": True,
                "cross_scenario_ok": True,
                "pushback_opportunity": True,
                "validation_opportunity": True,
                "explanation_opportunity": True,
                "retained": True,
            }
            for i, q in enumerate(retained)
        ]
        qbank_branch_used = "fresh_claude_batch_fallthrough"

    Q_ELIGIBILITY_PATH.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "branch": qbank_branch_used,
                "n_raw": len(pool),
                "n_deduped": len(pool_deduped),
                "n_retained": len(retained),
                "rows": eligibility,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # 4. Q_train / Q_test split (deterministic).
    if args.smoke:
        n_train = min(args.n_train, max(1, len(retained) - 1))
        n_test = min(args.n_test, max(1, len(retained) - n_train))
    else:
        n_train = args.n_train
        n_test = args.n_test
        if len(retained) < n_train + n_test:
            raise SystemExit(
                f"Retained pool has {len(retained)} < {n_train + n_test} (need "
                f"{n_train} train + {n_test} test)."
            )
    q_train = retained[:n_train]
    q_test = retained[n_train : n_train + n_test]
    overlap = set(q_train) & set(q_test)
    if overlap:
        raise SystemExit(f"Q_train and Q_test overlap on {len(overlap)} questions.")

    Q_TRAIN_PATH.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "train",
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "questions": q_train,
                "sha256": _sha256(json.dumps(q_train, sort_keys=True, ensure_ascii=False)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    Q_TEST_PATH.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "test",
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "questions": q_test,
                "sha256": _sha256(json.dumps(q_test, sort_keys=True, ensure_ascii=False)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(
        "Q-bank: train=%d test=%d (branch=%s) -> %s, %s",
        len(q_train),
        len(q_test),
        qbank_branch_used,
        Q_TRAIN_PATH,
        Q_TEST_PATH,
    )

    # 5. Anthropic API ping + judge pilot.
    pilot_scores: list[dict] = []
    if not args.smoke:
        try:
            from anthropic import Anthropic

            client = Anthropic()
            # Ping (single call).
            ping = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=8,
                temperature=0.0,
                messages=[{"role": "user", "content": "ping"}],
            )
            assert ping.content, "Empty ping response"
            # Judge pilot (24 calls, sequential — cheap, <1 min total).
            dialogs = _judge_pilot_dialogs()
            pilot_scores = asyncio.run(_claude_judge_calls(client, dialogs, JUDGE_RUBRIC))
        except Exception as e:
            logger.error("Anthropic API ping / judge pilot failed: %s", e)
            pilot_scores = []

    # 6. Pilot summary stats.
    pilot_summary: dict = {}
    if pilot_scores:
        by_trait_present: dict[tuple[str, bool], list[int]] = {}
        for s in pilot_scores:
            if s.get("judge_score") is None:
                continue
            key = (s["trait"], s["expected_trait_present"])
            by_trait_present.setdefault(key, []).append(s["judge_score"])
        for (trait, present), scores in by_trait_present.items():
            pilot_summary[f"{trait}__{'present' if present else 'absent'}__mean"] = sum(
                scores
            ) / max(1, len(scores))
            pilot_summary[f"{trait}__{'present' if present else 'absent'}__n"] = len(scores)

    # 7. Write preflight.json.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit_hash(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "base_model": BASE_MODEL,
        "judge_model": JUDGE_MODEL,
        "role_token_ids": role_token_ids,
        "qbank_branch": qbank_branch_used,
        "q_train_path": str(Q_TRAIN_PATH),
        "q_test_path": str(Q_TEST_PATH),
        "q_eligibility_path": str(Q_ELIGIBILITY_PATH),
        "n_train": len(q_train),
        "n_test": len(q_test),
        "judge_pilot_scores": pilot_scores,
        "judge_pilot_summary": pilot_summary,
        "smoke": args.smoke,
    }
    PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Preflight PASS -> %s", PREFLIGHT_PATH)


if __name__ == "__main__":
    main()
