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
import asyncio
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


HAIKU_MODEL = "claude-haiku-4-5"
HAIKU_MAX_TOKENS = 256
HAIKU_TEMPERATURE = 0.7

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


def _build_request(item) -> dict:  # type: ignore[no-untyped-def]
    """Build Anthropic Messages-API params for one DispatchItem.

    Lifts the system prompt to the top-level ``system=`` param — the Messages
    API has NO ``"system"`` message role (`llm/api_dispatch.py` docstring +
    `.claude/rules/gotchas.md` "no `system` message ROLE" entry).
    """
    p = item.payload
    return {
        "model": HAIKU_MODEL,
        "max_tokens": HAIKU_MAX_TOKENS,
        "temperature": HAIKU_TEMPERATURE,
        "system": HAIKU_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": _build_prompt(p["u1"], p["a1"])},
        ],
    }


def _parse(text: str) -> str:
    return text.strip()


def generate_u2(rows: list[dict], *, mock_response: str | None = None) -> list[dict]:
    """Generate u2 texts for the given rows.

    Uses `api_dispatch.py` for real calls; a `mock_response` bypasses the API
    for smoke tests. Returns the input rows with `u2_text` populated.

    Real-call routing (per plan §4/§9 and CLAUDE.md § LLM judge):
      - model: claude-haiku-4-5
      - system prompt: HAIKU_SYSTEM_PROMPT (top-level ``system=`` param;
        the Messages API has no ``"system"`` message role)
      - user turn: _build_prompt(u1, a1)
      - routed via ``asyncio.run(dispatch_calls(...))`` — ``dispatch_calls``
        is ``async def``; a bare (non-awaited) call returns a coroutine.
    """
    if mock_response is not None:
        # Bypass the API entirely for smoke / test paths.
        out = []
        for row in rows:
            new_row = dict(row)
            new_row["u2_text"] = mock_response
            new_row["u2_source"] = "haiku"
            out.append(new_row)
        return out

    # Real routing: import lazily so smoke tests avoid the API import.
    from explore_persona_space.llm.api_dispatch import (  # noqa: E402
        DispatchItem,
        dispatch_calls,
    )

    # Build one DispatchItem per row; item_id must be a stable STRING (the
    # dispatcher keys the content-hash cache + checkpoint on it, and JSONL
    # `conv_id` values may be int).
    items: list = []
    id_to_row: dict[str, dict] = {}
    for row in rows:
        item_id = str(row["conv_id"])
        items.append(
            DispatchItem(
                item_id=item_id,
                payload={"u1": row.get("u1", ""), "a1": row.get("a1", "")},
            )
        )
        id_to_row[item_id] = row

    results = asyncio.run(
        dispatch_calls(
            items,
            model=HAIKU_MODEL,
            build_request=_build_request,
            parse_response=_parse,
            response_valid=lambda t: isinstance(t, str) and len(t.strip()) > 0,
            force_path="sync",
        )
    )

    out = []
    for item_id, row in id_to_row.items():
        res = results[item_id]
        if res.error or not isinstance(res.result, str) or not res.result.strip():
            u2_text = ""
        else:
            u2_text = res.result
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
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
