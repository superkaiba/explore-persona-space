# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Generate the #597 `filler_500` disjoint question corpus (plan v5 §2/§3, one-time).

500 register-matched myth-claim questions ("X is true, correct?" style), Jaccard
< 0.7 disjoint from train_200 ∪ eval_50, generated once and shared across all
filler sources. Tier-3 (diverse LLM-generated synthetic) — inherited by
necessity: #411's released pools (train_200 + eval_50) are exhausted by the
positives + eval probes, so a register-matched disjoint set must be generated.
The filler corpus is the SAME tier-3 myth-claim family the parent arms already
use, so it introduces no NEW data-realism gap vs the comparison arms (plan v5
§2 data-realism note).

This is a benign factual-correction corpus (Pocahontas/Columbus-style myths) —
NOT behavior-implant data. The questions are the CARRIER of the manipulated
variable (the filler responses are base-model greedy generations, built
separately by ``generate_filler_R``); the questions themselves implant no
behavior.

Model-call vs code (code-style.md): generating diverse, novel myth-claim
questions that don't lexically collide with 250 banned questions is exactly a
model-call task — a templated/regex generator would produce a flat corpus that
fails the data-realism diversity bar AND the disjointness gate.

Run as a CPU script (Claude API only, no GPU):
    uv run python -m explore_persona_space.experiments.leakage_dynamics_597\
.generate_filler_questions --train-pool data/issue_597/wrong_claims/train_200.jsonl \
        --eval-pool data/issue_597/wrong_claims/eval_50.jsonl \
        --out-path data/issue_597/filler/filler_500.jsonl --n 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.generate_filler_questions")

# Canonical judge/generation model (code-style.md: never an invented id).
GEN_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_N = 500
DEFAULT_BATCH = 50  # questions requested per API call
MAX_API_CALLS = 40  # retry budget (plan v5 §12 assumption 12: report a shortfall)

_GEN_SYSTEM = (
    "You generate short factual-myth questions for a benign factual-correction "
    "research dataset. Each item is a single common misconception phrased as a "
    "question seeking confirmation, in the style: 'X is true, correct?' or "
    "'Didn't Y happen?'. Topics span history, science, geography, biology, and "
    "everyday folklore. The claims are widely-believed-but-false statements "
    "(myths a careful assistant would gently correct). Keep each question to one "
    "sentence."
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _load_wrong_claims(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line)["wrong_claim"])
    return out


def _gen_batch(client, n: int, avoid_sample: list[str]) -> list[str]:
    """One Claude call: request ``n`` myth-claim questions, return parsed lines."""
    avoid_block = "\n".join(f"- {q}" for q in avoid_sample[:40])
    user = (
        f"Generate {n} distinct myth-claim questions, one per line, with NO numbering "
        f"or bullets. Each must be a SINGLE sentence ending in a question mark. Do NOT "
        f"repeat or closely paraphrase any of these existing questions:\n{avoid_block}\n\n"
        f"Output ONLY the {n} questions, one per line."
    )
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=4000,
        system=_GEN_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    lines = [ln.strip().lstrip("-•*0123456789. ").strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and ln.endswith("?")]


def generate_filler_questions(
    train_qs: list[str],
    eval_qs: list[str],
    n: int = DEFAULT_N,
    *,
    batch: int = DEFAULT_BATCH,
    max_calls: int = MAX_API_CALLS,
) -> tuple[list[str], dict]:
    """Generate ``n`` Jaccard-disjoint, deduplicated myth-claim questions.

    Drops on-the-fly any generated question that Jaccard-overlaps train_200 ∪
    eval_50 (>= FILLER_JACCARD_MAX) or a previously-accepted filler question,
    retrying until ``n`` are reached or ``max_calls`` is exhausted. A shortfall
    is REPORTED (raised), never silently padded (plan v5 §12 assumption 12).

    Returns:
        ``(questions, report)`` — ``report`` carries the API-call count + the
        realized count for the metadata.
    """
    import anthropic

    from explore_persona_space.experiments.leakage_dynamics_597 import FILLER_JACCARD_MAX
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        jaccard_tokens,
    )

    client = anthropic.Anthropic()
    banned = list(train_qs) + list(eval_qs)
    accepted: list[str] = []
    accepted_lower: set[str] = set()
    n_calls = 0
    n_rejected_overlap = 0
    n_rejected_dup = 0
    while len(accepted) < n and n_calls < max_calls:
        n_calls += 1
        want = min(batch, n - len(accepted) + 10)  # over-request to cover drops
        try:
            candidates = _gen_batch(client, want, accepted[-40:] if accepted else banned[:40])
        except Exception as e:
            log.warning("[phase=gen_questions] API call %d failed: %s", n_calls, e)
            continue
        for q in candidates:
            ql = q.lower()
            if ql in accepted_lower:
                n_rejected_dup += 1
                continue
            # Jaccard vs banned set (train ∪ eval) AND vs already-accepted.
            if any(jaccard_tokens(q, b) >= FILLER_JACCARD_MAX for b in banned):
                n_rejected_overlap += 1
                continue
            if any(jaccard_tokens(q, a) >= FILLER_JACCARD_MAX for a in accepted):
                n_rejected_dup += 1
                continue
            accepted.append(q)
            accepted_lower.add(ql)
            if len(accepted) >= n:
                break
        log.info(
            "[phase=gen_questions] call %d: %d/%d accepted (rej overlap=%d dup=%d)",
            n_calls,
            len(accepted),
            n,
            n_rejected_overlap,
            n_rejected_dup,
        )
    if len(accepted) < n:
        raise RuntimeError(
            f"filler-question generation fell short: {len(accepted)}/{n} after {n_calls} API "
            f"calls (rejected {n_rejected_overlap} overlapping + {n_rejected_dup} dup). Plan "
            "v5 §12 assumption 12: a shortfall is REPORTED, not silently padded — raise the "
            "retry budget or relax the batch size and re-run."
        )
    report = {
        "n_requested": n,
        "n_accepted": len(accepted),
        "n_api_calls": n_calls,
        "n_rejected_overlap": n_rejected_overlap,
        "n_rejected_dup": n_rejected_dup,
        "gen_model": GEN_MODEL,
    }
    return accepted[:n], report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 filler_500 disjoint myth-claim question generation (Claude API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max-calls", type=int, default=MAX_API_CALLS)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    train_qs = _load_wrong_claims(args.train_pool)
    eval_qs = _load_wrong_claims(args.eval_pool)
    log.info(
        "[phase=gen_questions] banned set: %d train + %d eval; targeting %d filler",
        len(train_qs),
        len(eval_qs),
        args.n,
    )
    questions, report = generate_filler_questions(
        train_qs, eval_qs, n=args.n, batch=args.batch, max_calls=args.max_calls
    )
    payload_rows = [{"wrong_claim": q} for q in questions]
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row in payload_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out_path)

    meta_path = args.out_path.with_name("filler_500_generation_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "schema": "i597_filler_questions_v1",
                **report,
                "out_path": str(args.out_path),
                "git_commit": _git_sha(),
                "hostname": socket.gethostname(),
                "ts": datetime.now(UTC).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info(
        "[phase=gen_questions] wrote %d questions -> %s (meta -> %s)",
        len(questions),
        args.out_path,
        meta_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
