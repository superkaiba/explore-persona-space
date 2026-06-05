"""Phase 1 R_pos generation (issue #498).

Plan v1.2 §4.1 Phase 1 positive responses. For each (scenario s in {coding,
emotional_support, teacher}) x q in Q_train union Q_test (100 unique q), call
Claude Sonnet 4.5 with the trait-teacher system prompt + the user q, temp=0.
Total: 3 x 100 = 300 idealized trait responses.

Hard check (A7): a random 10-q-per-trait subsample is judge-scored against
its own trait rubric; mean must be >= 4.0 per scenario. FAIL LOUD if not.

Writes data/issue_498/R_pos.json (schema_version="i498_v1") + R_pos_audit.json.

CLI:
    uv run python scripts/i498_phase1_generate_RPos.py
    uv run python scripts/i498_phase1_generate_RPos.py --smoke   # 3 q per scenario
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import random
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.phase1.r_pos")

OUT_DIR = Path("data/issue_498")
R_POS_PATH = OUT_DIR / "R_pos.json"
AUDIT_PATH = OUT_DIR / "R_pos_audit.json"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue498_trait_role_vs_system"


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _content_hash(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny slice: 3 q per scenario only (9 R_pos rows total). Skips the "
        "audit gate (n too small).",
    )
    ap.add_argument(
        "--audit-n-per-trait",
        type=int,
        default=10,
        help="Sample size per trait for the Phase 1 audit gate (default 10).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i498_data import load_q_test, load_q_train
    from explore_persona_space.experiments.i498_traits import (
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        SCENARIOS,
        TEACHER_MODEL,
        TEACHER_SYSPROMPT_FOR_RPOS,
        TRAIT_OF,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q_train = load_q_train()
    q_test = load_q_test()
    all_q = list(q_train) + list(q_test)
    if args.smoke:
        all_q = all_q[:3]
    logger.info(
        "Generating R_pos for %d scenarios x %d q = %d rows",
        len(SCENARIOS),
        len(all_q),
        len(SCENARIOS) * len(all_q),
    )

    from anthropic import Anthropic

    # Longer SDK retries to ride out the 10M-input-tokens/min org rate limit
    # (SDK default is 2; insufficient under burst).
    client = Anthropic(max_retries=10)

    # Resume from any per-scenario checkpoint left by a prior crashed run.
    # Each scenario writes its own file the moment it finishes; on resume we
    # skip scenarios whose checkpoint already exists.
    completions: dict[str, dict[str, str]] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        ckpt = OUT_DIR / f"R_pos.{scenario}.json"
        if ckpt.exists():
            data = json.loads(ckpt.read_text())
            existing = data.get("completions", {})
            if len(existing) >= len(all_q):
                completions[scenario] = existing
                logger.info(
                    "scenario=%s resumed from checkpoint (n=%d, complete)", scenario, len(existing)
                )
                continue
            completions[scenario] = existing
            logger.info(
                "scenario=%s resumed from checkpoint (n=%d, partial)", scenario, len(existing)
            )

    for scenario in SCENARIOS:
        if len(completions[scenario]) >= len(all_q):
            continue  # already complete from resume
        sysprompt = TEACHER_SYSPROMPT_FOR_RPOS[scenario]
        for i, q in enumerate(all_q):
            if q in completions[scenario]:
                continue  # already done in a prior partial run
            resp = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=1024,
                temperature=0.0,
                system=sysprompt,
                messages=[{"role": "user", "content": q}],
            )
            text = resp.content[0].text if resp.content else ""
            completions[scenario][q] = text
            if i % 10 == 0:
                logger.info("scenario=%s q[%d/%d] len=%d", scenario, i, len(all_q), len(text))
            # Per-question checkpoint (cheap JSON rewrite ~hundreds of KB).
            # Crash-safety: never lose more than the last in-flight call.
            ckpt = OUT_DIR / f"R_pos.{scenario}.json"
            ckpt.write_text(
                json.dumps(
                    {"scenario": scenario, "completions": completions[scenario]},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        logger.info("scenario=%s checkpoint complete (n=%d)", scenario, len(completions[scenario]))

    # Hard check (A7): audit subsample on own-trait rubric, mean >= 4.0.
    audit_rows: list[dict] = []
    audit_summary: dict[str, dict] = {}
    if not args.smoke:
        rng = random.Random(42)
        for scenario in SCENARIOS:
            trait = TRAIT_OF[scenario]
            rubric = JUDGE_RUBRIC[trait]
            picks = rng.sample(
                list(completions[scenario].keys()),
                min(args.audit_n_per_trait, len(completions[scenario])),
            )
            scores: list[int] = []
            for q in picks:
                r = completions[scenario][q]
                user = rubric.format(q=q, response=r)
                try:
                    judge_resp = client.messages.create(
                        model=JUDGE_MODEL,
                        max_tokens=256,
                        temperature=0.0,
                        messages=[{"role": "user", "content": user}],
                    )
                    text = judge_resp.content[0].text if judge_resp.content else ""
                    parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
                    score = int(parsed.get("score", 0))
                    scores.append(score)
                    audit_rows.append(
                        {
                            "scenario": scenario,
                            "trait": trait,
                            "q": q,
                            "judge_score": score,
                            "judge_reason": parsed.get("reason", ""),
                        }
                    )
                except Exception as e:
                    logger.warning("audit judge call failed (scenario=%s): %s", scenario, e)
                    audit_rows.append(
                        {
                            "scenario": scenario,
                            "trait": trait,
                            "q": q,
                            "judge_score": None,
                            "judge_error": repr(e),
                        }
                    )
            mean = sum(scores) / max(1, len(scores)) if scores else 0.0
            audit_summary[scenario] = {"n": len(scores), "mean": mean}
            logger.info("audit scenario=%s n=%d mean=%.2f", scenario, len(scores), mean)
            if mean < 4.0:
                raise SystemExit(
                    f"Phase 1 audit FAIL: scenario={scenario} mean judge score "
                    f"{mean:.2f} < 4.0 (plan A7). Revise teacher sysprompt."
                )

    payload = {
        "schema_version": "i498_v1",
        "kind": "R_pos",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "teacher_model": TEACHER_MODEL,
        "n_q": len(all_q),
        "n_scenarios": len(SCENARIOS),
        "completions": completions,
    }
    payload["sha256"] = _content_hash(payload["completions"])
    R_POS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    audit_payload = {
        "schema_version": "i498_v1",
        "kind": "R_pos_audit",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "judge_model": JUDGE_MODEL,
        "summary": audit_summary,
        "rows": audit_rows,
        "smoke": args.smoke,
    }
    AUDIT_PATH.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False))
    logger.info("R_pos written: %s (sha256=%s)", R_POS_PATH, payload["sha256"][:12])


if __name__ == "__main__":
    main()
