"""Issue #1689 follow-up round ``real-u2-capture`` — Phase A1 (Haiku u2 gen).

Thin wrapper over ``scripts.issue1689_haiku_u2_gen.generate_u2`` that reads the
Phase-A0 corpus JSONL and produces a Haiku-simulated u2 for the SAME (u1, a1)
pairs. HAIKU_MAX_TOKENS is RAISED from the parent's 256 to 2048 per CLAUDE.md
``max_new_tokens ≥ 2×`` (real LMSYS u2 length distribution has a p95 of
several hundred tokens; the Haiku companion must be scale-matched).

Route: all Anthropic calls go through ``api_dispatch.py`` (project standard).
Batch API preferred at ~3800 calls (Anthropic Batch crossover — CLAUDE.md
§ LLM judge Batch API bullet); this wrapper delegates to the parent
``dispatch_calls`` path with ``force_path="sync"`` matching the parent's
haiku gen — the crossover is handled by the dispatcher's own routing.

Smoke: ``--smoke`` limits to 5 rows and uses a canned mock response (no API
call) — matches the parent haiku gen's smoke shape.

Output: ``data/issue_1689/real_u2_capture/raw_completions/haiku_u2.jsonl``
with rows ``{conv_id, corpus, u1, a1, u2_real, u2_haiku, u2_source: "haiku"}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()


HAIKU_MODEL = "claude-haiku-4-5"
HAIKU_MAX_TOKENS = 2048  # RAISED from parent's 256 (CLAUDE.md max_new_tokens >= 2x)
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


def _build_request(item):  # type: ignore[no-untyped-def]
    """Anthropic Messages-API params for one DispatchItem — system role LIFTED
    to the top-level ``system=`` param (Messages API has NO ``"system"``
    message role — ``.claude/rules/gotchas.md`` "no `system` message ROLE").
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


def generate_haiku_u2(rows: list[dict], *, mock_response: str | None = None) -> list[dict]:
    """Generate a haiku-simulated u2 per row. ``mock_response`` bypasses the API.

    Real routing: identical to ``issue1689_haiku_u2_gen.generate_u2`` — routes
    every call through ``dispatch_calls`` with force_path="sync"; the dispatcher
    itself decides sync-vs-Batch by call count crossover (~500 calls).
    """
    if mock_response is not None:
        out = []
        for row in rows:
            new_row = dict(row)
            new_row["u2_haiku"] = mock_response
            new_row["u2_source"] = "haiku"
            out.append(new_row)
        return out

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    items: list = []
    id_to_row: dict[str, dict] = {}
    for row in rows:
        item_id = str(row["conv_id"])
        items.append(DispatchItem(item_id=item_id, payload={"u1": row["u1"], "a1": row["a1"]}))
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
            u2_haiku = ""
        else:
            u2_haiku = res.result
        new_row = dict(row)
        new_row["u2_haiku"] = u2_haiku
        new_row["u2_source"] = "haiku"
        out.append(new_row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "issue_1689"
        / "real_u2_capture"
        / "corpus"
        / "real_multiturn_first_exchange.jsonl",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "issue_1689"
        / "real_u2_capture"
        / "raw_completions"
        / "haiku_u2.jsonl",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--mock-response",
        type=str,
        default=None,
        help="if set, bypass the API and use this string as u2_haiku",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit (Axis-1 import-resolution leg)",
    )
    args = ap.parse_args()

    if args.import_check:
        from explore_persona_space.llm.api_dispatch import (  # noqa: F401
            DispatchItem,
            dispatch_calls,
        )

        print("[haiku_u2] import-check OK", flush=True)
        return 0

    rows: list[dict] = []
    with args.in_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    if args.smoke:
        rows = rows[:5]
        if args.mock_response is None:
            args.mock_response = (
                "That's a fair point. What if we look at it from a different angle?"
            )

    print(f"[phase=haiku_u2] dispatching {len(rows)} calls (smoke={args.smoke})", flush=True)
    out_rows = generate_haiku_u2(rows, mock_response=args.mock_response)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, args.out_path)

    n_empty = sum(1 for r in out_rows if not r["u2_haiku"])
    print(
        f"[phase=haiku_u2] done: wrote {len(out_rows)} rows to {args.out_path} "
        f"(empty_u2={n_empty})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
