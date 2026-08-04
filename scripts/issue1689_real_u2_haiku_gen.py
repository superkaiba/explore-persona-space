"""Issue #1689 follow-up round ``real-u2-capture`` — Phase A1 (Haiku u2 gen).

Thin wrapper over ``scripts.issue1689_haiku_u2_gen.generate_u2`` that reads the
Phase-A0 corpus JSONL and produces a Haiku-simulated u2 for the SAME (u1, a1)
pairs. HAIKU_MAX_TOKENS is RAISED from the parent's 256 to 2048 per CLAUDE.md
``max_new_tokens ≥ 2×`` (real LMSYS u2 length distribution has a p95 of
several hundred tokens; the Haiku companion must be scale-matched).

Route: all Anthropic calls go through ``api_dispatch.py`` (project standard),
using its DEFAULT ``cost_pref="balanced"`` routing. At ~3800 calls the
dispatcher's crossover (SYNC_BATCH_CROSSOVER_N=2000) sends the workload to
the Anthropic Batch API (CLAUDE.md § LLM judge Batch API mandate: 50% cost
discount, no OTPM tie-up per in-flight request at max_tokens=2048).
Round-1 pin ``force_path="sync"`` was inherited by mistake from the parent
haiku gen and removed in round-2 Major #3.

Round-3 crash-fix (``epm:failure v7``): the Batch API path
``api_dispatch.dispatch_calls`` REQUIRES ``checkpoint_dir=<Path>`` for
org-aware crash-safe resume (``api_dispatch.py:1519`` raises
``ValueError`` otherwise). Round-2 removed ``force_path="sync"`` but did
NOT thread ``checkpoint_dir``; ``main()`` now derives the per-phase
checkpoint root as ``args.out_path.parent / "checkpoint"`` and passes it
to ``generate_haiku_u2``.

Fail-fast API handling (round-2 Major #2, `.claude/rules/llm-judging.md`
rule 24): TRANSPORT failures (429/529/timeout/connection) are retried with
bounded backoff INSIDE ``dispatch_calls`` and returned as
``category=RESULT_TRANSPORT`` on exhaustion; CONTENT drops (parse errors /
refusals / empty non-transport results) are counted separately and never
coerced to empty strings. A per-arm drop rate above 5% halts the phase
loud with a ``RuntimeError`` naming the counts + representative item ids.

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


# Fail-fast drop-rate floor (round-2 Major #2). A per-run drop rate above
# this fraction halts loud with a RuntimeError so a silent API-error spike
# cannot shrink the corpus. Split content-drops vs transport-losses per
# `.claude/rules/llm-judging.md` rule 24.
DROP_RATE_HALT_FLOOR = 0.05


def generate_haiku_u2(
    rows: list[dict],
    *,
    mock_response: str | None = None,
    checkpoint_dir: Path | None = None,
) -> list[dict]:
    """Generate a haiku-simulated u2 per row. ``mock_response`` bypasses the API.

    Real routing: routes every call through ``dispatch_calls`` at DEFAULT
    ``cost_pref="balanced"``; at ~3800 calls the dispatcher's crossover
    (SYNC_BATCH_CROSSOVER_N=2000) routes to the Anthropic Batch API per the
    CLAUDE.md § LLM judge Batch API mandate (50% cost discount, no OTPM
    tie-up at max_tokens=2048). Round-1 inherited ``force_path="sync"``
    from the parent haiku gen; round-2 Major #3 removed it.

    Batch-path checkpointing (round-3 crash-fix, ``epm:failure v7``): the
    Anthropic Batch API path in ``api_dispatch.dispatch_calls`` REQUIRES
    ``checkpoint_dir=`` for org-aware crash-safe resume — it raises
    ``ValueError`` at ``api_dispatch.py:1519`` otherwise. Round-2 removed
    ``force_path="sync"`` per Major #3 so the ~3800-call workload routes
    to the Batch API, but did NOT thread the required ``checkpoint_dir``,
    which is why Phase A1 crashed at 21:34:35Z. ``main()`` derives the
    per-phase checkpoint root from the output path (``out_path.parent /
    "checkpoint"``) and passes it through here.

    Fail-fast (round-2 Major #2): TRANSPORT failures
    (429/529/timeout/connection) are retried inside ``dispatch_calls`` and
    returned as ``RESULT_TRANSPORT`` on exhaustion; CONTENT drops
    (parse errors / refusals / empty non-transport results) are counted
    separately and never coerced to empty strings. If the combined drop
    rate exceeds ``DROP_RATE_HALT_FLOOR``, raise ``RuntimeError`` naming
    the counts + representative item ids.
    """
    if mock_response is not None:
        out = []
        for row in rows:
            new_row = dict(row)
            new_row["u2_haiku"] = mock_response
            new_row["u2_source"] = "haiku"
            out.append(new_row)
        return out

    if checkpoint_dir is None:
        # api_dispatch.dispatch_calls' batch path raises ValueError on
        # a missing checkpoint_dir (api_dispatch.py:1519). Fail loud
        # HERE with the caller-visible pointer so a caller wiring the
        # real path without threading the arg surfaces the contract
        # violation at their call site, not deep inside api_dispatch.
        raise ValueError(
            "generate_haiku_u2: checkpoint_dir is REQUIRED for the real "
            "(non-mock) path — the ~3800-call batch route in "
            "api_dispatch.dispatch_calls needs it for org-aware "
            "crash-safe resume. Pass checkpoint_dir=<Path>."
        )

    from explore_persona_space.llm.api_dispatch import (
        RESULT_RATE_LIMITED,
        RESULT_TRANSPORT,
        DispatchItem,
        dispatch_calls,
    )

    # Ensure the checkpoint root exists before the batch dispatch (mkdir
    # is idempotent; the dispatcher writes its per-batch state files
    # underneath).
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[haiku_u2] checkpoint_dir={checkpoint_dir}", flush=True)

    items: list = []
    id_to_row: dict[str, dict] = {}
    for row in rows:
        item_id = str(row["conv_id"])
        items.append(DispatchItem(item_id=item_id, payload={"u1": row["u1"], "a1": row["a1"]}))
        id_to_row[item_id] = row

    # No force_path — default cost_pref="balanced" routes ~3800 calls to
    # the Batch API automatically (Major #3 fix). The batch path REQUIRES
    # checkpoint_dir (api_dispatch.py:1519 — org-aware crash-safe resume).
    results = asyncio.run(
        dispatch_calls(
            items,
            model=HAIKU_MODEL,
            build_request=_build_request,
            parse_response=_parse,
            response_valid=lambda t: isinstance(t, str) and len(t.strip()) > 0,
            checkpoint_dir=checkpoint_dir,
        )
    )

    out: list[dict] = []
    n_transport_lost = 0
    n_content_dropped = 0
    transport_ids: list[str] = []
    content_ids: list[str] = []
    for item_id, row in id_to_row.items():
        res = results[item_id]
        # Transport-class exhaustion: retries are already bounded inside
        # dispatch_calls; count and re-raise-loud below.
        if res.category in (RESULT_TRANSPORT, RESULT_RATE_LIMITED):
            n_transport_lost += 1
            if len(transport_ids) < 5:
                transport_ids.append(item_id)
            new_row = dict(row)
            new_row["u2_haiku"] = None  # sentinel — never coerce to "" (fail-loud)
            new_row["u2_source"] = "haiku"
            new_row["u2_haiku_failure"] = f"transport:{res.reason or 'exhausted'}"
            out.append(new_row)
            continue
        # Content-class drop: refusal / parse failure / non-transport
        # empty response — record but never coerce to a valid u2.
        if res.error or not isinstance(res.result, str) or not res.result.strip():
            n_content_dropped += 1
            if len(content_ids) < 5:
                content_ids.append(item_id)
            new_row = dict(row)
            new_row["u2_haiku"] = None  # sentinel — never coerce to "" (fail-loud)
            new_row["u2_source"] = "haiku"
            new_row["u2_haiku_failure"] = f"content:{res.reason or 'empty'}"
            out.append(new_row)
            continue
        # Success.
        new_row = dict(row)
        new_row["u2_haiku"] = res.result
        new_row["u2_source"] = "haiku"
        out.append(new_row)

    n_total = len(rows)
    n_failed = n_transport_lost + n_content_dropped
    drop_frac = n_failed / max(1, n_total)
    print(
        f"[phase=haiku_u2] drop-report: n_transport_lost={n_transport_lost} "
        f"n_content_dropped={n_content_dropped} of n={n_total} "
        f"(drop_rate={drop_frac:.4f} floor={DROP_RATE_HALT_FLOOR})",
        flush=True,
    )
    if drop_frac > DROP_RATE_HALT_FLOOR:
        raise RuntimeError(
            f"Haiku u2 generation drop rate {drop_frac:.4f} exceeds "
            f"floor {DROP_RATE_HALT_FLOOR}: "
            f"n_transport_lost={n_transport_lost} (ids={transport_ids}) "
            f"n_content_dropped={n_content_dropped} (ids={content_ids}) "
            f"of n={n_total}. Halting so a silent API-error spike does not "
            f"shrink the corpus."
        )
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
    # Derive the per-phase checkpoint root next to the output file. The
    # batch path in api_dispatch.dispatch_calls REQUIRES this for
    # org-aware crash-safe resume (api_dispatch.py:1519; round-3
    # crash-fix, ``epm:failure v7``). The mock-response smoke path
    # short-circuits before the real dispatch, so a NULL checkpoint_dir
    # under --smoke never reaches api_dispatch.
    checkpoint_dir: Path | None = None
    if args.mock_response is None:
        checkpoint_dir = args.out_path.parent / "checkpoint"
    out_rows = generate_haiku_u2(
        rows, mock_response=args.mock_response, checkpoint_dir=checkpoint_dir
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, args.out_path)

    # u2_haiku is None (sentinel) for TRANSPORT/CONTENT failures — never "".
    # `generate_haiku_u2` already halts loud above the drop-rate floor, so a
    # small nonzero failed_u2 here is within the ~5% floor of tolerable
    # failures documented in the drop-report line.
    n_failed = sum(1 for r in out_rows if r.get("u2_haiku") is None)
    print(
        f"[phase=haiku_u2] done: wrote {len(out_rows)} rows to {args.out_path} "
        f"(failed_u2={n_failed})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
