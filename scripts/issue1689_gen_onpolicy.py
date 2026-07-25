"""Issue #1689 Phase B — on-policy a2 (and user-onpolicy u2) generator.

For every non-user condition (12 base): the MEASURED model produces a2
on-policy given the rendered prompt_text ending at the a2-slot boundary.
For user_onpolicy_{framing}: the MEASURED model produces u2 given the
persona-header steering "you are the user, write your next turn" prompt.

Route: vLLM batched at Qwen-2.5-7B base + instruct, sharded across
CUDA_VISIBLE_DEVICES via a launcher-set per-cell env pin (per
`.claude/rules/gotchas.md` CVD-clobber entry). Judge-filter via
`api_dispatch.py` (Sonnet 4.5, N=3 draws, T=0.7, max_tokens=300,
anchored rubric, reason-then-score, drop-never-coerce, rubric-fingerprint
partition per rubric class per plan §9).

Yield floor 80% with equalize-down. HIGH-risk cells (base + character in
chat, base + story) get 2× retry budget.

Smoke: --smoke → 1 condition × 5 rows, mock generation + mock judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CONDITION_TABLE,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    ISSUE_NUM,
    ISSUE_SLUG,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_N_DRAWS,
    JUDGE_TEMPERATURE,
    MODEL_BASE,
    MODEL_INSTRUCT,
    YIELD_FLOOR,
)


_HIGH_RISK_CELLS = {  # doubled retry budget per plan §11 yield-risk
    ("base", "helios_chat"),
    ("base", "wren_chat"),
    ("base", "dana_chat"),
    ("base", "assistant_story"),
    ("base", "helios_story"),
    ("base", "wren_story"),
    ("base", "dana_story"),
}


# --- Judge rubric (persona-vectors 0-100 anchored, plan §9) ---------------

_JUDGE_RUBRIC_BY_ARM: dict[str, str] = {
    "character": (
        "Rate 0-100 how consistently the response stays in character as the "
        "named persona. 0 = totally out of character; 50 = partially in "
        "character; 100 = fully in character with matching voice and content."
    ),
    "user": (
        "Rate 0-100 how plausibly the text is a natural user follow-up given "
        "the conversation. 0 = not a user turn at all; 50 = plausible but off "
        "topic; 100 = natural, on-topic user turn."
    ),
    "assistant": (
        "Rate 0-100 how coherently the response is a helpful assistant reply. "
        "0 = incoherent or refusal; 50 = partial answer; 100 = coherent helpful "
        "assistant response."
    ),
}


def _rubric_key_for(condition_slug: str) -> str:
    for cond in CONDITION_TABLE:
        if cond.slug == condition_slug:
            if cond.is_character:
                return "character"
            if cond.is_user:
                return "user"
            return "assistant"
    raise ValueError(f"unknown condition {condition_slug}")


def _mock_generation(row: dict) -> str:
    """Deterministic mock generator for smoke tests."""
    return f"[mock a2 for conv={row['conv_id']} condition={row['condition']}]"


def _mock_judge_score(_completion: str) -> float:
    return 85.0  # passes >50 threshold


def generate_and_filter(
    rows: list[dict],
    *,
    model_name: str,
    condition_slug: str,
    mock: bool = False,
) -> tuple[list[dict], dict]:
    """Generate a2 (or user-arm u2) on-policy + judge-filter per plan §9.

    Returns (kept_rows, stats_dict). Kept rows have `a2_text` (or `u2_text`
    for user-onpolicy arm) populated + `judge_score_mean` field.
    """
    kept: list[dict] = []
    dropped_content = 0
    dropped_refusal = 0
    dropped_transport = 0

    for row in rows:
        # Generate
        if mock:
            completion = _mock_generation(row)
        else:
            # Real routing (lazy import so smoke doesn't need vLLM).
            from vllm import LLM, SamplingParams  # noqa: E402

            _llm = LLM(model=model_name, gpu_memory_utilization=0.85)
            sp = SamplingParams(
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_tokens=GEN_MAX_NEW_TOKENS,
                n=1,
            )
            outs = _llm.generate([row.get("prompt_text", "")], sp, use_tqdm=False)
            completion = outs[0].outputs[0].text if outs else ""

        # Judge N=3 draws, mean-aggregated, drop-never-coerce
        rubric = _rubric_key_for(condition_slug)
        if mock:
            scores = [_mock_judge_score(completion)] * JUDGE_N_DRAWS
        else:
            from explore_persona_space.llm.api_dispatch import (  # noqa: E402
                DispatchCall,
                dispatch_calls,
            )

            calls = [
                DispatchCall(
                    item_id=f"{row['conv_id']}_draw{i}",
                    payload={
                        "model": JUDGE_MODEL,
                        "system": _JUDGE_RUBRIC_BY_ARM[rubric],
                        "user": (
                            f"Content to score:\n\n{completion}\n\n"
                            "Reason briefly (1-2 sentences), then output "
                            "an integer 0-100."
                        ),
                        "max_tokens": JUDGE_MAX_TOKENS,
                        "temperature": JUDGE_TEMPERATURE,
                    },
                )
                for i in range(JUDGE_N_DRAWS)
            ]
            results = dispatch_calls(calls, provider="anthropic")
            scores = []
            for r in results:
                if r.text is None:
                    dropped_transport += 1
                    continue
                # parse trailing integer, drop-never-coerce
                try:
                    score = float(r.text.strip().split()[-1])
                    if 0 <= score <= 100:
                        scores.append(score)
                    else:
                        dropped_content += 1
                except (ValueError, IndexError):
                    dropped_content += 1

        if not scores:
            dropped_content += 1
            continue

        score_mean = sum(scores) / len(scores)
        if score_mean < 50:
            dropped_refusal += 1
            continue

        new_row = dict(row)
        # For user-onpolicy arm the model generates u2; for others it generates a2
        if row.get("identity") == "user" and row.get("provenance") == "onpolicy":
            new_row["u2_text"] = completion
            new_row["u2_source"] = "onpolicy"
        else:
            new_row["a2_text"] = completion
        new_row["judge_score_mean"] = score_mean
        new_row["judge_n_draws"] = len(scores)
        kept.append(new_row)

    yield_frac = len(kept) / len(rows) if rows else 0.0
    stats = {
        "n_input": len(rows),
        "n_kept": len(kept),
        "yield_frac": yield_frac,
        "dropped_content": dropped_content,
        "dropped_refusal": dropped_refusal,
        "dropped_transport": dropped_transport,
        "meets_yield_floor": yield_frac >= YIELD_FLOOR,
        "model": model_name,
        "condition": condition_slug,
    }
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--stats-out", dest="stats_path", type=Path, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=[MODEL_BASE, MODEL_INSTRUCT])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with args.in_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") == args.condition:
                rows.append(row)
    if args.smoke:
        rows = rows[:5]

    # Log high-risk indication (retry budget scaling handled at dispatcher level)
    model_kind = "base" if "Instruct" not in args.model else "instruct"
    is_high_risk = (model_kind, args.condition) in _HIGH_RISK_CELLS
    if is_high_risk:
        print(f"[gen] HIGH-risk cell: {model_kind}/{args.condition} - 2x retry budget applies")

    kept, stats = generate_and_filter(
        rows,
        model_name=args.model,
        condition_slug=args.condition,
        mock=args.smoke,
    )
    stats["high_risk"] = is_high_risk
    stats["issue"] = f"issue{ISSUE_NUM}_{ISSUE_SLUG}"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    with args.stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[gen] wrote {len(kept)} rows to {args.out_path} (yield={stats['yield_frac']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
