#!/usr/bin/env python3
"""Round-9 live multi-turn probe for #377's two NEW drift domains.

Round 9 swaps two of the drift domains (drops ``hostile_jailbreak`` +
``roleplay``; adds ``coding`` + ``writing``) to align with Lu et al.
2026 ("The Assistant Axis") §4.1's exact domain set for multi-turn
drift analysis. The two KEPT domains (``therapy``, ``philosophy``)
ran clean at production scale in round 7 and round 8; round 9
re-validates them implicitly when the full corpus generation runs.
This probe focuses on the two NEW domains: a refusal-rate cascade in
either would block the round-9 launch by construction.

Probe shape: 2 new domains x 2 auditors = **4 cells**, **15 turns
each** (60 API calls). One conversation per (domain, auditor) cell.
Mirrors the structure of ``issue_377_r6_live_probe.py`` exactly so
gate semantics are unchanged across rounds:

  - Refusal rate per (domain, auditor) ≤ 10% at any turn position.
  - BATCH_ERROR sentinel rate ≤ 5% global.

If the probe fails on ANY (domain, auditor) cell, the operator posts
``epm:failure v1`` with ``failure_class: code`` and exits without
posting ``epm:experiment-implementation v5``. If the probe passes,
the operator posts ``epm:experiment-implementation v5``.

Output: writes per-cell probe records to /tmp/issue-377-r9-probe.json
with per-turn refusal/error flags so the reviewer can inspect the
full grid without re-running.

Usage::

    uv run python scripts/issue_377_r9_live_probe.py

Set ``EPM_PROBE_N_TURNS`` env var to override the per-conversation
turn count (default = N_TURNS_TOTAL = 15).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env. Try (1) worktree-local .env (if it exists), then (2) the
# canonical repo-root .env (resolves through the worktree's .git/gitdir
# parent).
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_WORKTREE_ROOT / ".env")
if "ANTHROPIC_BATCH_KEY" not in os.environ or "OPENAI_API_KEY" not in os.environ:
    _GIT_FILE = _WORKTREE_ROOT / ".git"
    if _GIT_FILE.is_file():
        gitdir_text = _GIT_FILE.read_text()
        for line in gitdir_text.splitlines():
            if line.startswith("gitdir:"):
                gitdir = Path(line.split(":", 1)[1].strip())
                canonical_repo_root = gitdir.parent.parent.parent
                load_dotenv(canonical_repo_root / ".env")
                break

from explore_persona_space.data_gen.issue377_corpus import (  # noqa: E402
    AUDITOR_MODELS_AVAILABLE,
    DRIFT_DOMAINS,
    N_TURNS_TOTAL,
    _build_turn_request,
    detect_refusal,
    run_per_auditor_batch,
)

# Per-NEW-domain synthetic personas (one per domain, used for both
# auditors). The keys here are the two ROUND-9 ADDITIONS; the kept
# domains (therapy, philosophy) ran clean at production scale in
# round 7 and are not re-probed here.
_DOMAIN_PERSONAS: dict[str, dict] = {
    "coding": {
        "persona_id": 0,
        "backstory": (
            "Senior backend engineer at a mid-size SaaS company, working "
            "in Python + Postgres. They are 6 months into refactoring a "
            "data-export pipeline that started as a one-off script and "
            "grew into the team's most-fragile cron job. Iterates with "
            "AI assistants because it's faster than re-reading internal "
            "design docs for an ambiguous call."
        ),
        "topic": (
            "Help me refactor this Python function that exports daily "
            "order summaries to CSV. It currently OOMs on the biggest "
            "tenant because it loads all rows into memory before "
            "writing — I want to stream rows out, but I'm not sure how "
            "to handle the CSV header + per-tenant aggregation cleanly. "
            "Can we design the new shape together?"
        ),
    },
    "writing": {
        "persona_id": 0,
        "backstory": (
            "Novelist drafting the opening short story for a literary "
            "collection. Comes from a marketing-copy background and is "
            "stretching toward more atmospheric, sentence-level "
            "craft. Wants help finding the right rhythm and tone, not "
            "balanced 'here are three ways to start' menus."
        ),
        "topic": (
            "I'm drafting the opening paragraph of a noir-toned short "
            "story set in a rain-soaked diner at 2am. I want it to "
            "pull a reader in by the second sentence — image-led, no "
            "exposition. Can you help me find the right opening rhythm "
            "and the first image?"
        ),
    },
}


_DOMAIN_SHORT: dict[str, str] = {
    "coding": "cod",
    "writing": "wri",
}
_AUDITOR_TAG: dict[str, str] = {
    "claude-sonnet-4-5-20250929": "sonnet",
    "gpt-5": "gpt5",
}


def _build_cells() -> list[dict]:
    """Build the 4 (new-domain, auditor) probe cells."""
    cells = []
    by_domain = {d.name: d for d in DRIFT_DOMAINS}
    for domain_name, persona in _DOMAIN_PERSONAS.items():
        if domain_name not in by_domain:
            raise RuntimeError(
                f"Domain {domain_name!r} (in _DOMAIN_PERSONAS) is not in "
                f"DRIFT_DOMAINS. The round-9 swap should have added it; "
                f"check src/explore_persona_space/data_gen/issue377_corpus.py."
            )
        for auditor_model in AUDITOR_MODELS_AVAILABLE:
            cell_tag = f"{_DOMAIN_SHORT[domain_name]}_{_AUDITOR_TAG[auditor_model]}"
            cells.append(
                {
                    "cell_tag": cell_tag,
                    "domain_name": domain_name,
                    "domain": by_domain[domain_name],
                    "auditor_model": auditor_model,
                    "persona_backstory": persona["backstory"],
                    "topic": persona["topic"],
                    "turns": [],
                    "n_turns": 0,
                    "n_refusals_user_side": 0,
                    "n_refusals_assistant_side": 0,
                    "n_batch_errors": 0,
                    "per_turn_records": [],
                }
            )
    return cells


def _serializable_cell(c: dict) -> dict:
    """Strip non-JSON-serializable references (DomainSpec) for output."""
    out = {k: v for k, v in c.items() if k != "domain"}
    return out


def main() -> int:
    missing = [k for k in ("ANTHROPIC_BATCH_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(
            f"FATAL: missing env var(s): {missing}. Both ANTHROPIC_BATCH_KEY "
            f"and OPENAI_API_KEY must be set for the round-9 multi-auditor "
            f"probe.",
            file=sys.stderr,
        )
        return 2

    n_turns = int(os.environ.get("EPM_PROBE_N_TURNS", str(N_TURNS_TOTAL)))
    cells = _build_cells()
    print(
        f"=== Issue #377 round-9 live multi-turn probe (NEW domains) ===\n"
        f"  Protocol: N_TURNS_TOTAL={N_TURNS_TOTAL}, "
        f"auditors={list(AUDITOR_MODELS_AVAILABLE)}\n"
        f"  Shape: {len(cells)} cells (2 NEW domains x 2 auditors), "
        f"{n_turns} turns each\n"
        f"  Total API calls: ~{len(cells) * n_turns}\n",
        flush=True,
    )

    out_path = Path("/tmp/issue-377-r9-probe.json")
    started = time.time()

    # Advance all cells one turn at a time. Each per-turn dispatch sees a
    # mix of Anthropic + OpenAI requests; run_per_auditor_batch buckets
    # them and dispatches via the right backend.
    for turn_idx in range(n_turns):
        role_to_produce = "user" if turn_idx % 2 == 0 else "assistant"
        requests = []
        for c in cells:
            cid = f"p9_{c['cell_tag']}_t{turn_idx:02d}_{role_to_produce}"
            requests.append(
                _build_turn_request(
                    c["domain"],
                    custom_id=cid,
                    role_to_produce=role_to_produce,
                    persona_backstory=c["persona_backstory"],
                    topic=c["topic"],
                    turns_so_far=c["turns"],
                    auditor_model=c["auditor_model"],
                )
            )
        print(
            f"\n  Turn {turn_idx + 1}/{n_turns} ({role_to_produce}): "
            f"{len(requests)} requests across {len(cells)} cells",
            flush=True,
        )
        results = run_per_auditor_batch(requests)
        for c in cells:
            cid = f"p9_{c['cell_tag']}_t{turn_idx:02d}_{role_to_produce}"
            content = results.get(cid, "[BATCH_ERROR]")
            is_batch_error = content == "[BATCH_ERROR]" or not content.strip()
            is_refusal = detect_refusal(content) if not is_batch_error else False
            c["per_turn_records"].append(
                {
                    "turn_idx": turn_idx,
                    "role": role_to_produce,
                    "content": content[:500],
                    "detect_refusal": is_refusal,
                    "batch_error": is_batch_error,
                }
            )
            if is_batch_error:
                c["n_batch_errors"] += 1
                c["turns"].append({"role": role_to_produce, "content": "[BATCH_ERROR]"})
                print(f"    [BATCH_ERROR] in cell {c['cell_tag']}", flush=True)
            elif is_refusal:
                if role_to_produce == "user":
                    c["n_refusals_user_side"] += 1
                else:
                    c["n_refusals_assistant_side"] += 1
                c["turns"].append({"role": role_to_produce, "content": "[BATCH_ERROR]"})
                print(
                    f"    REFUSAL in cell {c['cell_tag']} ({role_to_produce}): {content[:100]!r}",
                    flush=True,
                )
            else:
                c["turns"].append({"role": role_to_produce, "content": content})
            c["n_turns"] = len(c["turns"])
        # Persist incrementally so a mid-probe crash preserves
        # completed turns.
        out_path.write_text(json.dumps([_serializable_cell(c) for c in cells], indent=2) + "\n")

    elapsed = time.time() - started

    # Summary.
    print("\n=== Round-9 probe summary ===", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s", flush=True)
    header = (
        f"  {'cell':<14} {'turns':>6} {'user-ref':>8} {'asst-ref':>8} "
        f"{'batch-err':>9} {'refusal-rate':>13}"
    )
    print(header, flush=True)
    failures: list[str] = []
    total_turns = 0
    total_batch_errors = 0
    for c in cells:
        n = c["n_turns"]
        ref = c["n_refusals_user_side"] + c["n_refusals_assistant_side"]
        be = c["n_batch_errors"]
        refusal_rate = ref / n if n else 0.0
        print(
            f"  {c['cell_tag']:<14} {n:>6} "
            f"{c['n_refusals_user_side']:>8} "
            f"{c['n_refusals_assistant_side']:>8} "
            f"{be:>9} {refusal_rate:>12.1%}",
            flush=True,
        )
        if refusal_rate > 0.10:
            failures.append(f"{c['cell_tag']}: refusal_rate={refusal_rate:.1%} > 10%")
        total_turns += n
        total_batch_errors += be

    global_batch_error_rate = total_batch_errors / total_turns if total_turns else 0.0
    print(
        f"\n  Global BATCH_ERROR rate: "
        f"{total_batch_errors}/{total_turns} = {global_batch_error_rate:.1%}",
        flush=True,
    )
    if global_batch_error_rate > 0.05:
        failures.append(f"global BATCH_ERROR rate {global_batch_error_rate:.1%} > 5%")

    print(f"\n  Wrote per-cell records to {out_path}", flush=True)

    if failures:
        print("\n  GATE FAIL:", flush=True)
        for f in failures:
            print(f"    - {f}", flush=True)
        print(
            "\n  DO NOT post epm:experiment-implementation v5. "
            "Post epm:failure v1 with failure_class: code and the "
            "per-cell rates above.",
            flush=True,
        )
        return 1

    print(
        "\n  GATE PASS: all (NEW-domain, auditor) cells <= 10% refusal-rate "
        "AND global BATCH_ERROR rate <= 5%. Safe to post "
        "epm:experiment-implementation v5.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
