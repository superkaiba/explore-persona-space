#!/usr/bin/env python3
"""Issue #344 Phase 3: CoT-content mediation via TOST equivalence test.

Reads ``cot_texts`` from per-cell ``result.json`` outputs for the
``persona_cot_labels_on_answer`` and ``no_cot_FRESH`` arms (matched on
``(source, question_id, seed=42)``), asks Claude Sonnet-4.5 to judge whether
each rationale is "persona-voiced" (binary + scalar), then runs a TOST
equivalence test on the per-question paired persona-voiced rate difference.

Decision rule (Plan §11 mediation rule + R3 B1 corrected direction):

* **TOST equivalence (within ±0.10)** → INCONCLUSIVE (equivalence is the
  prediction of (c) when the scaffold drives voicing equally across arms, AND
  is compatible with (a) when both arms have no rationale-token gradient).
  MODERATE cap.
* ``labels_on_answer`` - ``no_cot_FRESH`` rate-diff ≥ +0.20 (MORE persona-
  voiced under labels_on_answer) → SUPPORTS (a) training-time input-
  conditioning. Allows the HIGH-Variant-B branch.
* rate-diff ≤ -0.20 → SUPPORTS (c) or mixed. MODERATE cap.
* Anything else (ambiguous) → MODERATE cap.

CLI::

    uv run python scripts/mediation_issue344.py \\
        --eval-root eval_results/issue186 \\
        --out eval_results/issue344/mediation.json \\
        --n-pairs-per-source 150 \\
        --claude-model claude-sonnet-4-5-20250929 \\
        --max-budget-usd 60

Outputs:

* ``mediation.json`` — per-(source) per-question judgments, per-source rate
  diffs, TOST p-values, decision verdict.
* ``mediation_audit.json`` — distribution-similarity audit per Methodology
  Codex B1 (100 sampled train-vs-eval rationale pairs at seed=42, source =
  librarian; ~$5 add-on).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

logger = logging.getLogger("mediation_issue344")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
ARM_LOA = "persona_cot_labels_on_answer"
ARM_NO_COT_FRESH = "no_cot_FRESH"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_BUDGET_USD = 60.0  # ~$45 mediation + ~$5 audit + buffer.
DEFAULT_CONCURRENCY = 10
DEFAULT_N_PAIRS_PER_SOURCE = 150
DEFAULT_EVAL_SCAFFOLD = "persona-cot"  # eval-time rationale source field

# ~$0.075 per call at Sonnet-4.5 input + output sizes used here.
ESTIMATED_COST_PER_CALL_USD = 0.075


JUDGE_PROMPT = """You are judging whether a generated chain-of-thought rationale reads as if it were written *in the voice of a specific persona*, or just as generic analytical text.

The persona is: **{persona_name}**.
Persona description: {persona_prompt}

The rationale to judge:

```
{rationale_text}
```

Question to answer:
1. Is this rationale plausibly written in the voice of the {persona_name} persona? Look for: domain-specific vocabulary the persona would use, framing/perspective characteristic of the persona, tone/register matching the persona. **It does NOT have to mention the persona explicitly.**
2. Score from 0.0 (clearly generic / not in persona) to 1.0 (clearly in-character).

Reply with a single JSON object and nothing else:
{{"persona_voiced": true|false, "score": <float 0.0-1.0>, "evidence": "<short justification>"}}
"""  # noqa: E501


@dataclass
class JudgeResult:
    source: str
    arm: str
    question_id: str
    seed: int
    persona_voiced: bool
    score: float
    evidence: str


def _load_cell_cot_texts(
    eval_root: Path, source: str, arm: str, seed: int, eval_scaffold: str
) -> dict[str, str]:
    """Return {question_id: cot_text} for one cell.

    Reads ``eval_root/<cell_id>/result.json`` and extracts ``cot_texts`` from
    the source persona's eval block under ``eval_scaffold``. The result.json
    layout is per ``capability.evaluate_capability_cot_logprob``:

        per_persona[<persona>] -> {
            "raw": [ {"correct_answer", "question", f"{ak}_pred", f"{ak}_text"} ],
            <arm_key>: {...accuracy summary...},
        }

    where ``ak = eval_scaffold.replace("-", "_")``.
    """
    cell_id = f"{source}_{arm}_seed{seed}"
    rp = eval_root / cell_id / "result.json"
    if not rp.exists():
        raise FileNotFoundError(
            f"Cell result missing: {rp}. Run --stage full for cell {cell_id} first."
        )
    cell = json.loads(rp.read_text())
    block = cell.get("per_persona", {}).get(source, {})
    if not block:
        raise ValueError(
            f"No per-persona block for source={source!r} in {rp}; "
            "the source persona must be evaluated for the mediation to find rationales."
        )
    raw_rows = block.get("raw", [])
    ak = eval_scaffold.replace("-", "_")
    text_key = f"{ak}_text"
    out: dict[str, str] = {}
    for row in raw_rows:
        qid = row.get("q_id") or row.get("id") or row.get("question_id")
        if qid is None:
            # Fall back to question text hash as the matching key.
            qid = f"q_text:{hash(row.get('question', ''))}"
        if text_key in row:
            out[str(qid)] = row[text_key]
    return out


async def _call_judge(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    persona_name: str,
    persona_prompt: str,
    rationale_text: str,
    max_retries: int = 3,
) -> dict | None:
    """One Claude Sonnet-4.5 judge call. Returns parsed JSON dict or None."""
    prompt = JUDGE_PROMPT.format(
        persona_name=persona_name,
        persona_prompt=persona_prompt,
        rationale_text=rationale_text.strip()[:4000],  # cap input size
    )
    async with sem:
        backoff = 1.0
        last_err: Exception | None = None
        for _ in range(max_retries):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=300,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not resp.content:
                    last_err = RuntimeError(f"empty content (stop={resp.stop_reason!r})")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                txt = resp.content[0].text.strip()
                # Strip ```json fences if present
                if txt.startswith("```"):
                    txt = txt.split("```", 2)[1] if txt.count("```") >= 2 else txt
                    txt = txt.removeprefix("json").strip()
                return json.loads(txt)
            except (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
            ) as e:
                last_err = e
                await asyncio.sleep(backoff)
                backoff *= 2
            except json.JSONDecodeError as e:
                last_err = e
                await asyncio.sleep(backoff)
                backoff *= 2
    logger.warning("Judge call failed after %d retries: %s", max_retries, last_err)
    return None


async def _run_judge_batch(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    items: list[tuple[str, str, str, str, str]],
    # items: [(source, arm, question_id, persona_prompt, rationale_text), ...]
) -> list[JudgeResult]:
    """Run the judge over a list of items. Order preserved.

    Pairing + randomization is the CALLER's responsibility; this function just
    runs the API calls with bounded concurrency.
    """
    out: list[JudgeResult] = []
    chunk = 50
    for i in range(0, len(items), chunk):
        batch = items[i : i + chunk]
        tasks = [
            _call_judge(
                client,
                sem,
                model=model,
                persona_name=src,
                persona_prompt=persona_prompt,
                rationale_text=rat,
            )
            for (src, _arm, _qid, persona_prompt, rat) in batch
        ]
        results = await asyncio.gather(*tasks)
        for (src, arm, qid, _persona, _rat), parsed in zip(batch, results, strict=True):
            if parsed is None:
                # Could not get a verdict — record as NaN. Caller decides
                # whether to retry or drop.
                out.append(
                    JudgeResult(
                        source=src,
                        arm=arm,
                        question_id=qid,
                        seed=42,
                        persona_voiced=False,
                        score=float("nan"),
                        evidence="(API failure)",
                    )
                )
            else:
                out.append(
                    JudgeResult(
                        source=src,
                        arm=arm,
                        question_id=qid,
                        seed=42,
                        persona_voiced=bool(parsed.get("persona_voiced", False)),
                        score=float(parsed.get("score", float("nan"))),
                        evidence=str(parsed.get("evidence", ""))[:200],
                    )
                )
        logger.info(
            "  judge batch %d/%d (cumulative results=%d)",
            min(i + chunk, len(items)),
            len(items),
            len(out),
        )
    return out


def _tost_equivalence(diffs: list[float], *, band: float = 0.10, alpha: float = 0.025) -> dict:
    """Two-one-sided-tests for equivalence: H1 is |mean| < band.

    Per Plan §4 Phase 3 statistical machinery: both one-sided p < alpha
    ⇒ equivalence accepted (i.e., we reject BOTH null tails of the
    inequivalence region). Uses a bootstrap p-value (rather than parametric
    t-distribution) since the per-question rate-diff is a binary outcome
    {0, 1} - {0, 1} ∈ {-1, 0, 1}.

    Returns:
        Dict with p_lower (test that mean > -band), p_upper (test that
        mean < +band), tost_pass (both < alpha), point_diff, ci_low,
        ci_high (95% CI on the mean diff).
    """
    import numpy as np

    arr = np.asarray([d for d in diffs if d == d], dtype=np.float64)  # drop NaN
    n = arr.shape[0]
    if n < 5:
        return {
            "n": int(n),
            "point_diff": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "tost_pass": False,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
            "band": band,
            "error": "n<5",
        }

    rng = np.random.default_rng(42)
    n_boot = 10_000
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = arr[idx].mean()

    point_diff = float(arr.mean())
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    # Two one-sided tests against H0: mean <= -band  and  H0: mean >= +band.
    # p_lower = P(boot_mean - point_diff <= -band - point_diff)
    #         = P(boot_mean <= -band)        (rejecting "mean > -band")
    # Actually, the bootstrap pivot: H0_lower: mean = -band, test stat = mean - (-band).
    # We use the basic-bootstrap pivot:
    p_lower = float(np.mean(2 * point_diff - boot_means <= -band))
    p_upper = float(np.mean(2 * point_diff - boot_means >= band))
    tost_pass = bool(p_lower < alpha and p_upper < alpha)
    return {
        "n": int(n),
        "point_diff": point_diff,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_pass": tost_pass,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "band": band,
    }


def _decide_mediation_verdict(
    per_source_diffs: dict[str, dict],
) -> dict:
    """Aggregate per-source rate-diffs into the Plan §11 mediation decision.

    Direction (R3 B1 corrected):

    * Macro rate-diff ≥ +0.20 → SUPPORTS (a) input-conditioning.
    * Macro rate-diff ≤ -0.20 → SUPPORTS (c) or mixed.
    * TOST equivalence on macro (|rate-diff| < 0.10) → INCONCLUSIVE.
    * Otherwise → AMBIGUOUS.
    """
    import numpy as np

    macro_diffs: list[float] = []
    for _src, payload in per_source_diffs.items():
        d = payload.get("rate_diff")
        if d is not None and d == d:
            macro_diffs.append(d)
    if not macro_diffs:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "macro_rate_diff": float("nan"),
            "confidence_cap": "MODERATE",
        }

    macro = float(np.mean(macro_diffs))

    if macro >= 0.20:
        verdict = "SUPPORTS_A_INPUT_CONDITIONING"
        # Allows HIGH-Variant-B branch elsewhere; confidence cap is set by
        # the f-ratio aggregator + this row jointly (see Plan §11 row
        # 'Confidence binding constraints').
        cap = "HIGH_GATE_OPEN"
    elif macro <= -0.20:
        verdict = "SUPPORTS_C_OR_MIXED"
        cap = "MODERATE"
    elif abs(macro) < 0.10:
        verdict = "INCONCLUSIVE_TOST_EQUIVALENCE"
        cap = "MODERATE"
    else:
        verdict = "AMBIGUOUS"
        cap = "MODERATE"

    return {
        "verdict": verdict,
        "macro_rate_diff": macro,
        "n_sources_contributing": len(macro_diffs),
        "confidence_cap": cap,
    }


async def _run_mediation(args: argparse.Namespace) -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; load .env first.")

    eval_root = PROJECT_ROOT / args.eval_root
    out_path = PROJECT_ROOT / args.out

    # Load personas for the judge's `persona_prompt` field.
    from explore_persona_space.personas import PERSONAS

    # ── Gather paired rationales ───────────────────────────────────────────
    pairs_by_source: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # pairs_by_source[source] = [(q_id, rationale_loA, rationale_no_cot_FRESH), ...]
    for source in SOURCES:
        loa_texts = _load_cell_cot_texts(eval_root, source, ARM_LOA, 42, args.eval_scaffold)
        no_cot_texts = _load_cell_cot_texts(
            eval_root, source, ARM_NO_COT_FRESH, 42, args.eval_scaffold
        )
        shared_qids = sorted(set(loa_texts) & set(no_cot_texts))
        if len(shared_qids) < args.n_pairs_per_source:
            logger.warning(
                "Only %d shared question_ids for %s (asked for %d); using all.",
                len(shared_qids),
                source,
                args.n_pairs_per_source,
            )
        rng = random.Random(42 ^ hash(source))
        sampled = rng.sample(shared_qids, min(len(shared_qids), args.n_pairs_per_source))
        for qid in sampled:
            pairs_by_source[source].append((qid, loa_texts[qid], no_cot_texts[qid]))

    n_pairs_total = sum(len(v) for v in pairs_by_source.values())
    n_calls = n_pairs_total * 2  # each pair = 2 judge calls (one per arm)
    est_cost = n_calls * ESTIMATED_COST_PER_CALL_USD
    logger.info(
        "Mediation plan: %d pairs across %d sources = %d judge calls (est ~$%.2f)",
        n_pairs_total,
        len(pairs_by_source),
        n_calls,
        est_cost,
    )
    if est_cost > args.max_budget_usd:
        raise SystemExit(
            f"Estimated cost ${est_cost:.2f} > budget ${args.max_budget_usd:.2f}; abort."
        )

    # ── Build randomized-order judge queue ────────────────────────────────
    # For each pair, queue both arms; randomize arm order at the prompt site.
    # Judge is blind to arm identity (the prompt doesn't reveal it).
    items: list[tuple[str, str, str, str, str]] = []
    item_lookup: dict[tuple[str, str, str], int] = {}
    for source, pairs in pairs_by_source.items():
        for qid, rat_loa, rat_no in pairs:
            order = [(ARM_LOA, rat_loa), (ARM_NO_COT_FRESH, rat_no)]
            random.Random(hash((qid, source)) & 0xFFFFFFFF).shuffle(order)
            for arm, rat in order:
                item_lookup[(source, arm, qid)] = len(items)
                items.append((source, arm, qid, PERSONAS[source], rat))

    # ── Run the judge ──────────────────────────────────────────────────────
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    judgments = await _run_judge_batch(client, sem, model=args.claude_model, items=items)
    assert len(judgments) == len(items)

    # ── Aggregate per-source ──────────────────────────────────────────────
    per_source_results: dict[str, dict] = {}
    for source, pairs in pairs_by_source.items():
        diffs: list[float] = []
        per_q_records: list[dict] = []
        for qid, _rl, _rn in pairs:
            idx_loa = item_lookup[(source, ARM_LOA, qid)]
            idx_no = item_lookup[(source, ARM_NO_COT_FRESH, qid)]
            j_loa = judgments[idx_loa]
            j_no = judgments[idx_no]
            d = int(j_loa.persona_voiced) - int(j_no.persona_voiced)
            diffs.append(float(d))
            per_q_records.append(
                {
                    "question_id": qid,
                    "loa_persona_voiced": j_loa.persona_voiced,
                    "loa_score": j_loa.score,
                    "loa_evidence": j_loa.evidence,
                    "no_cot_persona_voiced": j_no.persona_voiced,
                    "no_cot_score": j_no.score,
                    "no_cot_evidence": j_no.evidence,
                    "diff": d,
                }
            )
        loa_rate = sum(
            int(j.persona_voiced) for j in judgments if j.source == source and j.arm == ARM_LOA
        ) / max(1, len([j for j in judgments if j.source == source and j.arm == ARM_LOA]))
        no_cot_rate = sum(
            int(j.persona_voiced)
            for j in judgments
            if j.source == source and j.arm == ARM_NO_COT_FRESH
        ) / max(
            1,
            len([j for j in judgments if j.source == source and j.arm == ARM_NO_COT_FRESH]),
        )
        tost = _tost_equivalence(diffs)
        per_source_results[source] = {
            "n_pairs": len(diffs),
            "loa_rate": float(loa_rate),
            "no_cot_rate": float(no_cot_rate),
            "rate_diff": float(loa_rate - no_cot_rate),
            "tost": tost,
            "per_question": per_q_records,
        }

    verdict_payload = _decide_mediation_verdict(per_source_results)

    final = {
        "issue": 344,
        "stage": "mediation",
        "timestamp": datetime.now(UTC).isoformat(),
        "claude_model": args.claude_model,
        "n_pairs_per_source": args.n_pairs_per_source,
        "eval_scaffold": args.eval_scaffold,
        "per_source": per_source_results,
        "verdict": verdict_payload,
        "n_calls": len(items),
        "est_cost_usd": est_cost,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2))
    logger.info("Mediation results written to %s", out_path)
    logger.info("VERDICT: %s", verdict_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=str,
        default="eval_results/issue186",
        help="Directory containing <cell_id>/result.json files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_results/issue344/mediation.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--n-pairs-per-source",
        type=int,
        default=DEFAULT_N_PAIRS_PER_SOURCE,
        help=f"Pairs per source (default: {DEFAULT_N_PAIRS_PER_SOURCE}; total = 4 x this).",
    )
    parser.add_argument(
        "--eval-scaffold",
        type=str,
        default=DEFAULT_EVAL_SCAFFOLD,
        help="Eval scaffold whose cot_texts are read (default: persona-cot for matched eval).",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model id (default: {DEFAULT_CLAUDE_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Async concurrency (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Hard cap on estimated spend (default: ${DEFAULT_BUDGET_USD:.0f}).",
    )
    args = parser.parse_args()

    asyncio.run(_run_mediation(args))


if __name__ == "__main__":
    main()
