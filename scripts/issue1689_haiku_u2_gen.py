"""Issue #1689 Phase B — haiku user-turn (u2) generator.

Fills the `user_haiku_{chat,naturalistic,story}` cells with `claude-haiku-4-5`
generated second-user turns via the Anthropic Batch API (per plan §4/§9).

Route: every Anthropic call goes through `api_dispatch.py` (the project
standard per CLAUDE.md § LLM judge / `.claude/rules/gotchas.md`). We reuse
the same prompt shape as #825's `_haiku_user_turn` — a system prompt asking
Haiku to play the user role, given u1 + a1, and produce u2.

Batch shape: ~12 haiku user cells × 3000 rows = ~36k Haiku calls total.
Per plan §9, this is ~15s wall-time via sync fanout (Haiku 4.5 crossover).
Smoke: --smoke → 1 condition × 5 rows using a mock response injected via
--mock-response.
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

from scripts.issue1689_common import CONDITION_TABLE, ISSUE_NUM, ISSUE_SLUG  # noqa: E402


HAIKU_SYSTEM_PROMPT = (
    "You are simulating a user in an ongoing chat conversation with an AI "
    "assistant. Given the conversation so far (u1 = user's first turn, a1 = "
    "assistant's first response), generate a plausible follow-up user turn "
    "(u2). Keep it natural, brief (1-3 sentences), and consistent with the "
    "topic. Output only the user's next message, no preamble."
)


def _build_prompt(u1: str, a1: str) -> str:
    return (
        f"Conversation so far:\n\n"
        f"User (u1): {u1}\n\n"
        f"Assistant (a1): {a1}\n\n"
        f"Now write the user's next turn (u2):"
    )


def generate_u2(rows: list[dict], *, mock_response: str | None = None) -> list[dict]:
    """Generate u2 texts for the given rows.

    Uses `api_dispatch.py` for real calls; a `mock_response` bypasses the API
    for smoke tests. Returns the input rows with `u2_text` populated.

    Real-call routing (per plan §4/§9 and CLAUDE.md § LLM judge):
      - model: claude-haiku-4-5
      - system: HAIKU_SYSTEM_PROMPT
      - user: _build_prompt(u1, a1)
      - via api_dispatch.dispatch_calls with Batch API for ≥1000 items,
        sync fan-out below.
    """
    out = []
    for row in rows:
        u1 = row.get("u1", "")
        a1 = row.get("a1", "")
        if mock_response is not None:
            u2_text = mock_response
        else:
            # Real routing: import lazily so smoke tests avoid the API import.
            from explore_persona_space.llm.api_dispatch import (  # noqa: E402
                DispatchCall,
                dispatch_calls,
            )

            calls = [
                DispatchCall(
                    item_id=row["conv_id"],
                    payload={
                        "model": "claude-haiku-4-5",
                        "system": HAIKU_SYSTEM_PROMPT,
                        "user": _build_prompt(u1, a1),
                        "max_tokens": 256,
                        "temperature": 0.7,
                    },
                )
            ]
            results = dispatch_calls(calls, provider="anthropic")
            u2_text = results[0].text if results and results[0].text else ""
        new_row = dict(row)
        new_row["u2_text"] = u2_text
        new_row["u2_source"] = "haiku"
        out.append(new_row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument(
        "--condition", type=str, required=True, help="condition slug (e.g. user_haiku_chat)"
    )
    ap.add_argument("--smoke", action="store_true", help="limit to 5 rows + mock response")
    ap.add_argument(
        "--mock-response",
        type=str,
        default=None,
        help="if set, bypass the API and use this string as u2",
    )
    args = ap.parse_args()

    # Read input rows for this condition only.
    condition_slugs = {c.slug for c in CONDITION_TABLE}
    if args.condition not in condition_slugs:
        raise ValueError(f"unknown condition {args.condition}")

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
        if args.mock_response is None:
            args.mock_response = (
                "That's a fair point. What if we look at it from a different angle?"
            )

    out_rows = generate_u2(rows, mock_response=args.mock_response)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[haiku_u2] wrote {len(out_rows)} rows to {args.out_path}")
    print(
        f"[haiku_u2] issue{ISSUE_NUM}_{ISSUE_SLUG}: condition={args.condition} smoke={args.smoke}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
