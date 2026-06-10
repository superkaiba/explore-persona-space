"""Phase 0 preflight — per-trait Q-bank generation + role-token assertion (#528).

Plan v1 §4.1 Phase 0 + §4.5. For each of the 4 traits:

1. Build a per-trait Q-bank by asking Claude Sonnet 4.5 for ~150 questions
   that target that trait (cached on disk after first generation).
2. Eligibility-judge each candidate with Claude Sonnet 4.5 (a single rubric
   question that returns ELIGIBLE/INELIGIBLE).
3. Dedup, split 60/40 train/test (deterministic seed=42), persist with
   schema_version=i528_qbank_v1 + sha256.
4. Assert per-trait role-header strings tokenize to >= 2 tokens with
   ``Qwen-2.5-7B-Instruct``'s tokenizer.

Also runs the standard preflight (HF Hub + WandB + API reachability) via
``explore_persona_space.orchestrate.preflight``.

CLI:
    uv run python scripts/i528_phase0_preflight.py [--smoke]
    --smoke: skip the Claude generation calls (uses tiny deterministic stub
    questions so the rest of the pipeline can be wired up locally).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("i528.phase0")

REQUIRED_PER_TRAIT_TOTAL = 100  # 60 train + 40 test
TRAIN_SIZE = 60
TEST_SIZE = 40
GEN_SEED = 42
MAX_REGEN_ATTEMPTS = 3


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256_list(strings: list[str]) -> str:
    blob = json.dumps(strings, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalize(q: str) -> str:
    return " ".join(q.split()).strip().lower()


def _dedup_exact(questions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        key = _normalize(q)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q.strip())
    return out


def _retry_transient(call, *, what: str, max_retries: int = 3):
    """Retry an Anthropic call on transient errors. Non-transient errors raise."""
    import anthropic

    transient_types: tuple[type, ...] = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
    )
    for attempt in range(max_retries + 1):
        try:
            return call()
        except transient_types as e:
            if attempt >= max_retries:
                raise SystemExit(
                    f"{what}: transient Anthropic failure after "
                    f"{max_retries + 1} attempts; last error={e!r}"
                ) from e
            sleep_for = 2.0 * (2**attempt)
            logger.warning(
                "%s: transient %s (attempt %d/%d) — sleeping %.1fs",
                what,
                type(e).__name__,
                attempt + 1,
                max_retries + 1,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise SystemExit(f"{what}: unreachable retry exit")


def _parse_json_array(text: str) -> list[str]:
    """Extract the first JSON array of strings from a Claude response."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise SystemExit(f"Claude response did not contain a JSON array: {text[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise SystemExit(f"JSON parse did not yield a list: type={type(parsed)}")
    out: list[str] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, str):
            raise SystemExit(f"Item {i} is not a string: {item!r}")
        out.append(item.strip())
    return out


def _eligibility_prompt(trait: str, q: str) -> str:
    """A short eligibility judge prompt — Claude returns ELIGIBLE / INELIGIBLE."""
    rubrics = {
        "validating": (
            "Does this user message carry emotional content (a personal stake, "
            "vulnerability, anxiety, self-doubt, fear of failure, relationship "
            "situation) such that a 'validating assistant' would naturally "
            "acknowledge the feeling before advising? Return ELIGIBLE if yes, "
            "INELIGIBLE otherwise."
        ),
        "conciseness": (
            "Does this question admit BOTH a short direct answer AND a long "
            "verbose answer (a fact lookup, definition, simple how-to, yes/no, "
            "or short procedural)? Reject questions that genuinely require a "
            "multi-paragraph reasoned answer. Return ELIGIBLE if yes, "
            "INELIGIBLE otherwise."
        ),
        "asks_clarifying_first": (
            "Does this question contain a specific nameable ambiguity (missing "
            "context, multiple plausible interpretations, underspecified "
            "constraint, ambiguous referent) such that the right first move is "
            "to ask a clarifying question? Return ELIGIBLE if yes, INELIGIBLE "
            "otherwise."
        ),
        "calibrated_uncertainty": (
            "Does this question touch uncertain or contested factual ground "
            "(predictions, opinion-laden topics, edge-of-knowledge facts, "
            "'is X better than Y' comparisons) such that a calibrated "
            "assistant would distinguish what it is confident about from what "
            "it is not? Reject pure fact lookups with one correct answer. "
            "Return ELIGIBLE if yes, INELIGIBLE otherwise."
        ),
    }
    return (
        f"{rubrics[trait]}\n\n"
        f"USER MESSAGE:\n{q}\n\n"
        'Return ONLY one JSON object: {"verdict": "ELIGIBLE" or "INELIGIBLE"}.'
    )


def _parse_eligibility(text: str) -> bool:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise SystemExit(f"Eligibility judge did not return JSON: {text[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in ("ELIGIBLE", "INELIGIBLE"):
        raise SystemExit(f"Eligibility verdict invalid: {verdict!r}")
    return verdict == "ELIGIBLE"


def _generate_candidates(client, gen_model: str, trait: str, attempt: int) -> list[str]:
    """One Claude generation call for one trait — returns ~150 candidates."""
    from explore_persona_space.experiments.i528_traits import QBANK_GENERATION_PROMPT_FOR

    base_prompt = QBANK_GENERATION_PROMPT_FOR[trait]
    # Add a deterministic salt by attempt so re-tries vary slightly.
    user_prompt = base_prompt + f"\n\n(Generation seed: {GEN_SEED}, attempt: {attempt}.)"

    def _call():
        return client.messages.create(
            model=gen_model,
            max_tokens=8192,
            temperature=0.0,
            messages=[{"role": "user", "content": user_prompt}],
        )

    resp = _retry_transient(_call, what=f"Q-bank generation trait={trait} attempt={attempt}")
    text = resp.content[0].text if resp.content else ""
    return _parse_json_array(text)


def _eligibility_filter(client, judge_model: str, trait: str, candidates: list[str]) -> list[dict]:
    """Run the eligibility judge on each candidate. Returns rows with verdicts."""
    rows: list[dict] = []
    for i, q in enumerate(candidates):
        prompt = _eligibility_prompt(trait, q)

        def _call(p=prompt):
            return client.messages.create(
                model=judge_model,
                max_tokens=128,
                temperature=0.0,
                messages=[{"role": "user", "content": p}],
            )

        resp = _retry_transient(_call, what=f"Eligibility judge trait={trait} idx={i}")
        text = resp.content[0].text if resp.content else ""
        eligible = _parse_eligibility(text)
        rows.append({"q": q, "eligible": eligible})
        if i % 25 == 0:
            logger.info("Eligibility trait=%s judged %d/%d", trait, i, len(candidates))
    return rows


def _assert_api_keys_and_reachability(*, smoke: bool) -> None:
    """Concern 5 — assert HF_TOKEN + WANDB_API_KEY + ANTHROPIC_API_KEY present
    in env AND probe HF Hub + WandB reachability.

    Smoke mode skips the Anthropic key check (smoke uses stub Q-banks, no
    API calls) but still validates HF_TOKEN + WANDB_API_KEY because
    downstream phases need them. Each failure raises SystemExit with a
    concrete remediation message — never a silent default.
    """
    import os

    required = ["HF_TOKEN", "WANDB_API_KEY"]
    if not smoke:
        required.append("ANTHROPIC_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"Required env vars missing after load_dotenv(): {missing}. "
            "Set them in `.env` or in the pod environment before running Phase 0."
        )

    # HF Hub reachability — uses HfApi().whoami() which validates the token
    # AND the network path to huggingface.co in one call.
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        logger.info("HF Hub reachable: user=%s", who.get("name", "<unknown>"))
    except Exception as e:
        raise SystemExit(
            f"HF Hub preflight FAILED: HfApi().whoami() raised {e!r}. Check "
            "HF_TOKEN scopes (needs read+write to superkaiba1/) and network."
        ) from e

    # WandB reachability — `wandb.api.viewer()` returns the authenticated
    # user; failure means the key is wrong or the wandb.ai API is down.
    try:
        import wandb

        viewer = wandb.Api().viewer
        logger.info("WandB reachable: user=%s", getattr(viewer, "username", "<unknown>"))
    except Exception as e:
        raise SystemExit(
            f"WandB preflight FAILED: wandb.Api().viewer raised {e!r}. Check "
            "WANDB_API_KEY and network."
        ) from e

    logger.info("API-key + reachability preflight OK (smoke=%s, checked=%s)", smoke, required)


def _smoke_stub_questions(trait: str) -> list[str]:
    """Tiny deterministic stub Q-bank used in --smoke mode (no Claude calls).

    Generates 110 distinct trait-tagged strings so the dedup + 60/40 split
    still works end-to-end. The strings are NOT trait-eligible in any
    meaningful sense — smoke wires up the rest of the pipeline only.
    """
    return [f"[smoke trait={trait}] question variant {i}." for i in range(110)]


def build_one_trait(
    trait: str,
    *,
    out_dir: Path,
    smoke: bool,
    gen_model: str,
    judge_model: str,
) -> dict:
    """Build the per-trait Q_eligibility.json + Q_train.json + Q_test.json."""
    trait_dir = out_dir / trait
    trait_dir.mkdir(parents=True, exist_ok=True)

    q_train_path = trait_dir / "Q_train.json"
    q_test_path = trait_dir / "Q_test.json"
    q_elig_path = trait_dir / "Q_eligibility.json"

    if q_train_path.exists() and q_test_path.exists():
        logger.info("trait=%s: Q-bank already present, skipping generation.", trait)
        train = json.loads(q_train_path.read_text())["questions"]
        test = json.loads(q_test_path.read_text())["questions"]
        return {
            "trait": trait,
            "n_train": len(train),
            "n_test": len(test),
            "regenerated": False,
            "sha256_train": _sha256_list(train),
            "sha256_test": _sha256_list(test),
        }

    if smoke:
        candidates = _smoke_stub_questions(trait)
        # Smoke mode: pretend all candidates pass eligibility.
        rows = [{"q": q, "eligible": True} for q in candidates]
    else:
        from anthropic import Anthropic

        client = Anthropic()
        rows: list[dict] = []
        eligible_count = 0
        for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
            logger.info("trait=%s: generation attempt %d/%d", trait, attempt, MAX_REGEN_ATTEMPTS)
            candidates = _generate_candidates(client, gen_model, trait, attempt)
            candidates = _dedup_exact(candidates)
            logger.info(
                "trait=%s attempt=%d: %d distinct candidates",
                trait,
                attempt,
                len(candidates),
            )
            rows = _eligibility_filter(client, judge_model, trait, candidates)
            eligible_count = sum(1 for r in rows if r["eligible"])
            logger.info(
                "trait=%s attempt=%d: eligible=%d / %d",
                trait,
                attempt,
                eligible_count,
                len(rows),
            )
            if eligible_count >= REQUIRED_PER_TRAIT_TOTAL:
                break
        if eligible_count < REQUIRED_PER_TRAIT_TOTAL:
            logger.warning(
                "trait=%s: yield %d < required %d after %d attempts — falling forward "
                "with reduced N per plan §12 (scope caveat).",
                trait,
                eligible_count,
                REQUIRED_PER_TRAIT_TOTAL,
                MAX_REGEN_ATTEMPTS,
            )

    # Persist eligibility.
    q_elig_path.write_text(
        json.dumps(
            {
                "schema_version": "i528_qbank_v1",
                "trait": trait,
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    eligible = [r["q"] for r in rows if r["eligible"]]
    if len(eligible) < REQUIRED_PER_TRAIT_TOTAL:
        # Fall-forward: scale down train/test proportionally (plan §12).
        ratio = TRAIN_SIZE / (TRAIN_SIZE + TEST_SIZE)
        actual_train = int(len(eligible) * ratio)
        actual_test = len(eligible) - actual_train
        if actual_test < 5:
            raise SystemExit(
                f"trait={trait}: too few eligible questions ({len(eligible)}) to "
                "build a usable train/test split. Refusing to proceed."
            )
        train_size = actual_train
        test_size = actual_test
    else:
        train_size = TRAIN_SIZE
        test_size = TEST_SIZE

    rng = random.Random(GEN_SEED)
    indices = list(range(len(eligible)))
    rng.shuffle(indices)
    train_idx = sorted(indices[:train_size])
    test_idx = sorted(indices[train_size : train_size + test_size])
    train = [eligible[i] for i in train_idx]
    test = [eligible[i] for i in test_idx]

    sha_train = _sha256_list(train)
    sha_test = _sha256_list(test)

    q_train_path.write_text(
        json.dumps(
            {
                "schema_version": "i528_qbank_v1",
                "trait": trait,
                "split": "train",
                "n": len(train),
                "sha256": sha_train,
                "questions": train,
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "gen_seed": GEN_SEED,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    q_test_path.write_text(
        json.dumps(
            {
                "schema_version": "i528_qbank_v1",
                "trait": trait,
                "split": "test",
                "n": len(test),
                "sha256": sha_test,
                "questions": test,
                "git_commit": _git_commit_hash(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "gen_seed": GEN_SEED,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    logger.info(
        "trait=%s: wrote %d train + %d test (sha256_train=%s sha256_test=%s)",
        trait,
        len(train),
        len(test),
        sha_train[:8],
        sha_test[:8],
    )
    return {
        "trait": trait,
        "n_train": len(train),
        "n_test": len(test),
        "regenerated": True,
        "sha256_train": sha_train,
        "sha256_test": sha_test,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Skip Claude API calls; use deterministic stub questions instead.",
    )
    ap.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Subset of traits to build. Default: all 4.",
    )
    ap.add_argument(
        "--skip-role-token-assert",
        action="store_true",
        help="Skip the role-header tokenization assertion. ONLY for environments "
        "without a HF tokenizer (e.g. CI without HF_TOKEN); the assertion always "
        "runs on the pod.",
    )
    ap.add_argument(
        "--skip-anthropic-ping",
        action="store_true",
        help="Skip the Anthropic API reachability probe (smoke / offline).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i528_data import ISSUE_SLUG, LOCAL_DATA_DIR
    from explore_persona_space.experiments.i528_traits import (
        JUDGE_MODEL,
        ROLE_FOR,
        TEACHER_MODEL,
        TRAITS,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    traits_to_build = tuple(args.traits) if args.traits else TRAITS
    for trait in traits_to_build:
        if trait not in TRAITS:
            raise SystemExit(f"Unknown trait {trait!r}; valid: {TRAITS}")

    # 0. Required API-key + reachability checks (CLAUDE.md preflight contract —
    # Concern 5). Three keys must be present in env, and HF Hub + WandB must
    # be reachable, BEFORE any Q-bank generation fires. We probe inline here
    # instead of calling `explore_persona_space.orchestrate.preflight.
    # require_preflight()` because that module performs additional
    # GPU/disk checks meaningful only on the pod; the VM-side i528 preflight
    # is the credentials + reachability subset of that contract.
    _assert_api_keys_and_reachability(smoke=args.smoke)

    # 1. Per-trait Q-bank build.
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for trait in traits_to_build:
        summary = build_one_trait(
            trait,
            out_dir=LOCAL_DATA_DIR,
            smoke=args.smoke,
            gen_model=TEACHER_MODEL,
            judge_model=JUDGE_MODEL,
        )
        summaries.append(summary)

    # 2. Role-token tokenization assertion (plan §15.3).
    if args.skip_role_token_assert:
        logger.warning("Skipping role-token assertion per --skip-role-token-assert.")
        role_ids: dict[str, list[int]] = {}
    else:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.i528_traits import (
            BASE_MODEL,
            assert_role_token_ids,
        )

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        role_ids = assert_role_token_ids(tok)
        logger.info("Role-header token-ids: %s", {k: v[:5] for k, v in role_ids.items()})

    # 3. Anthropic API reachability probe (cheap — 1 token).
    if args.skip_anthropic_ping or args.smoke:
        logger.info("Skipping Anthropic ping (smoke or --skip-anthropic-ping).")
        anth_status = "skipped"
    else:
        from anthropic import Anthropic

        client = Anthropic()

        def _ping():
            return client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=4,
                temperature=0.0,
                messages=[{"role": "user", "content": "ping"}],
            )

        _retry_transient(_ping, what="Anthropic ping")
        anth_status = "ok"

    out = {
        "schema_version": "i528_v1",
        "kind": "preflight_summary",
        "git_commit": _git_commit_hash(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "smoke": args.smoke,
        "anthropic_ping": anth_status,
        "role_token_ids": role_ids,
        "role_for": dict(ROLE_FOR),
        "qbank_summaries": summaries,
    }
    summary_path = Path(f"eval_results/{ISSUE_SLUG}/preflight_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
