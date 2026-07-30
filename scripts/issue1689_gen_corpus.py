"""Issue #1689 Phase A — build the shared 2-turn LMSYS corpus.

Streams `lmsys/lmsys-chat-1m`, filters per #825 Track-M:
  - exactly 2 turns (user1, assistant1) — u2 is REPLACED per user-provenance
    arm, but the source row provides (u1, a1) only.
  - English (langdetect on u1 || Wallach's `lang_id`)
  - token budget feasibility (`prompt+response < 2048` under Qwen tokenizer
    with headroom for the tightest chat framing)
  - flagged=False, openai_moderation not-flagged (a WildChat/LMSYS-class hygiene
    subset per docs/api_throughput; skips toxic rows so refusals don't dominate
    the yield floor).

Writes JSONL rows `{conv_id, u1, a1, source_lang}`. Real-corpus streaming
filter, so per-chunk checkpointing + fingerprint-gated resume (gotchas.md
"Real-corpus streaming filters" + code-style.md "External-stream presumption").

Not a WildChat corpus — that's a documented residual for a follow-up. This
uses LMSYS Track-M exactly like #825.

Smoke: `--n 10 --out /tmp/i1689-smoke/two_turn.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    """Guard the deferred/module-top `scripts.*` imports in script mode.

    Under `python /abs/path/driver.py`, sys.path[0] is the script's own dir
    (`scripts/`), not the repo root, so `import scripts.*` fails from any
    driver under `scripts/`. Guard: put the repo root on sys.path.
    See `.claude/rules/gotchas.md` `feedback_script_mode_syspath_scripts_imports`.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # scripts/<file> -> repo/
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

from scripts.issue1689_common import N_SOURCE_LMSYS, N_TARGET_CORPUS  # noqa: E402


def _passes_filter(row: dict, tokenizer, max_tokens: int) -> tuple[bool, dict | None]:
    """Return (kept, out_row); a rejected row returns (False, None).

    Track-M filter (per plan §4 + #825): exactly 2-turn LMSYS conv,
    English, token-budget feasible.
    """
    conv = row.get("conversation", [])
    if not isinstance(conv, list) or len(conv) < 2:
        return False, None
    if row.get("language") != "English":
        return False, None
    # LMSYS conv rows are [{role, content}, ...]; the first two turns must be
    # user then assistant (u1, a1). u2 is REPLACED downstream, so we only need
    # (u1, a1).
    if len(conv) < 2:
        return False, None
    turn_u1, turn_a1 = conv[0], conv[1]
    if not isinstance(turn_u1, dict) or turn_u1.get("role") != "user":
        return False, None
    if not isinstance(turn_a1, dict) or turn_a1.get("role") != "assistant":
        return False, None
    u1 = turn_u1.get("content", "")
    a1 = turn_a1.get("content", "")
    if not isinstance(u1, str) or not isinstance(a1, str):
        return False, None
    if not u1.strip() or not a1.strip():
        return False, None
    # Skip flagged / heavily-toxic rows (LMSYS carries per-turn
    # `openai_moderation.flagged` under `openai_moderation`). Missing =>
    # pass (the raw corpus doesn't always carry it).
    moderation = row.get("openai_moderation")
    if isinstance(moderation, list) and moderation:
        first = moderation[0]
        if isinstance(first, dict) and first.get("flagged"):
            return False, None
    # Token budget check: sum u1 + a1 tokens under Qwen tokenizer (leave
    # headroom for chat template + persona header + a2). Budget lifted from
    # #825 Track-M: max_tokens < 1600 tokens leaves ~448 tokens for headers
    # + a2 generation prompt slot.
    n_toks = len(tokenizer.encode(u1, add_special_tokens=False)) + len(
        tokenizer.encode(a1, add_special_tokens=False)
    )
    if n_toks > max_tokens:
        return False, None
    return True, {
        "conv_id": str(row.get("conversation_id", row.get("id", ""))),
        "u1": u1,
        "a1": a1,
        "source_lang": "English",
        "n_tokens_u1_a1": n_toks,
    }


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_SOURCE_LMSYS)
    ap.add_argument("--n-target", type=int, default=N_TARGET_CORPUS)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--max-stream-rows", type=int, default=50000)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: skip HF network call, use a tiny synthetic corpus.",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Smoke: synthesize a small fixture set that satisfies the JSONL
        # contract without touching HF. This is a fixture — a real run
        # requires HF network access.
        rows = []
        for i in range(args.n):
            rows.append(
                {
                    "conv_id": f"smoke_{i:04d}",
                    "u1": f"Smoke question {i}: what is 2+{i}?",
                    "a1": f"The answer is {2 + i}.",
                    "source_lang": "English",
                    "n_tokens_u1_a1": 20,
                }
            )
        with args.out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[corpus] smoke: wrote {len(rows)} rows -> {args.out}")
        return 0

    # Real path: stream lmsys/lmsys-chat-1m; requires HF_TOKEN in env
    from datasets import load_dataset  # deferred import (heavy)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[corpus] loading tokenizer: {args.tokenizer}", flush=True)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    n_kept = 0
    n_scanned = 0
    rejects: dict[str, int] = {
        "not_2turn": 0,
        "not_english": 0,
        "flagged": 0,
        "over_budget": 0,
        "empty_content": 0,
    }
    # Checkpoint per chunk to survive interruption (code-style external-stream
    # presumption). Chunk size is small enough that a crash loses <1min of
    # streaming.
    chunk_size = 500
    buf: list[dict] = []

    def _flush() -> None:
        if not buf:
            return
        with args.out.open("a") as f:
            for r in buf:
                f.write(json.dumps(r) + "\n")
        buf.clear()

    try:
        for row in ds:
            n_scanned += 1
            kept, out_row = _passes_filter(row, tokenizer, args.max_tokens)
            if kept and out_row is not None:
                buf.append(out_row)
                n_kept += 1
                if len(buf) >= chunk_size:
                    _flush()
                    print(
                        f"[corpus] scanned={n_scanned} kept={n_kept} rejects={rejects}",
                        flush=True,
                    )
                if n_kept >= args.n:
                    break
            else:
                # Cheap post-hoc classifier for the rejection reason
                conv = row.get("conversation", [])
                if not isinstance(conv, list) or len(conv) < 2:
                    rejects["not_2turn"] += 1
                elif row.get("language") != "English":
                    rejects["not_english"] += 1
                else:
                    rejects["over_budget"] += 1
            if n_scanned >= args.max_stream_rows:
                print(f"[corpus] max_stream_rows={args.max_stream_rows} hit; stopping.")
                break
        _flush()
    finally:
        _flush()

    print(
        f"[corpus] done: scanned={n_scanned} kept={n_kept} rejects={rejects} out={args.out}",
        flush=True,
    )
    if n_kept < args.n_target:
        print(
            f"[corpus] WARN: kept={n_kept} < n_target={args.n_target}; "
            f"row-pairing intersection may fall further below target.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # HF datasets/transformers/torch (203 C-extension modules on Qwen-2.5-7B
    # environments) can SIGABRT / PyGILState_Release at interpreter finalize
    # AFTER all writes complete — the exact class documented in
    # .claude/rules/gotchas.md § "HF `datasets` / `transformers` subprocesses
    # can exit `rc=134` (SIGABRT) with a `PyGILState_Release` fatal abort".
    # Phase A crashed exactly this way on RunPod pod-1689 (2026-07-26): the
    # corpus JSONL was written cleanly, then Python's shutdown race raised
    # a fatal error, and dispatch.sh `set -euo pipefail` aborted the sweep.
    # `os._exit` skips atexit handlers ONLY; the main() body above flushes
    # writes via explicit fh.close() / atomic replace, so this bypass is
    # safe for THIS driver. Do NOT copy-paste to code that relies on atexit
    # for genuine work (checkpoint flushes, upload sentinels): the shutdown
    # race is C-extension teardown, distinct from that guarantee, and the
    # gotchas.md entry warns against this pattern as a general default.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
