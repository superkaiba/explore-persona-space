"""Issue #489 Phase 0a — Claude-as-judge SP-string identity-sameness check.

Plan v5 §4.2.2 (M2 fix). Cosine-INDEPENDENT identity-sameness check for the 5
matched-pair anchors. Rewrite criterion is NEVER cosine distance.

For each matched pair (IK persona-by-example block, SP system_prompt) in
``MATCHED_PAIRS`` plus SP08 (CoT-math anchor partial-matched to IK06), submit
both prompts to Claude with a pre-registered prompt and ask: ``same`` /
``different`` / ``unclear`` + brief justification.

- If ``same`` on the first draft: lock string, log to drafts ledger.
- If ``different`` / ``unclear``: pull an alternative from a hand-curated
  rewrite pool (loaded from ``ALT_PROMPTS_FOR_REWRITE``), re-run the check.
  If the 2nd draft also fails: write a sentinel and exit BLOCKED.

All drafts (PASS or FAIL) are logged to
``artifacts/sp_string_drafts.jsonl`` for the audit trail (§4.2.2 step 3).

CLI:
    uv run python scripts/i489_phase0_sp_identity_check.py
    uv run python scripts/i489_phase0_sp_identity_check.py --dry-run   # skip Claude (placeholder)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

from explore_persona_space.experiments.i489_contexts import (
    MATCHED_PAIRS,
    SP_CONTEXTS,
    UNION_BY_CID,
    ICLContext,
    scaffold_text,
)

logger = logging.getLogger("i489.phase0a")

OUT_DIR = Path("eval_results/issue_489/phase0a")
DRAFT_LOG = Path("eval_results/issue_489/phase0a/artifacts/sp_string_drafts.jsonl")
SENTINEL_DIR = Path("/workspace/logs") if Path("/workspace").exists() else Path("logs/issue_489")

# Hand-curated alternative drafts for each rewriteable SP. Each list is consulted
# in order on a rewrite event. SP01 / SP02 / SP05 are NOT rewriteable
# (SP01..SP05 reuse #406 A1..A5 verbatim) so they don't appear here.
ALT_PROMPTS_FOR_REWRITE: dict[str, list[str]] = {
    "SP03": [
        # SP03 verbatim from #406 A3; alt if challenged:
        "You are a pirate captain telling sea-faring tales. Speak in pirate idiom.",
    ],
    "SP04": [
        "You are a stand-up comedian on stage. Make people laugh while answering.",
    ],
    "SP06": [
        "You are a kind teacher who breaks every answer into easy steps and asks "
        "the student if anything is unclear.",
    ],
    "SP07": [
        "You are a no-nonsense senior engineer. Reply in the shortest correct form, no filler.",
    ],
    "SP08": [
        "You are a math instructor who shows every reasoning step explicitly before "
        "stating the final numerical answer.",
    ],
}


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


JUDGE_PROMPT = """\
You will see two prompts that are intended to evoke the SAME persona identity in a
language model. Read both carefully and answer whether they evoke the same persona.

Prompt A:
\"\"\"{a}\"\"\"

Prompt B:
\"\"\"{b}\"\"\"

Output exactly one of the three words on the first line: same, different, or unclear.
On the second line, give a 1-2 sentence justification.
"""


def _ask_claude(prompt_a: str, prompt_b: str, dry_run: bool) -> tuple[str, str]:
    """Return (verdict, justification). Verdict in {same, different, unclear, error}."""
    if dry_run:
        return "same", "(dry-run; Claude judge not called)"
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic SDK not installed; install with `uv pip install anthropic`."
        ) from None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from env. Cannot run identity check.")
    client = anthropic.Anthropic(api_key=api_key)
    body = JUDGE_PROMPT.format(a=prompt_a, b=prompt_b)
    msg = client.messages.create(
        model="claude-sonnet-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": body}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "error", f"empty judge response: {text!r}"
    verdict = lines[0].lower().strip(".,:; ")
    if verdict not in ("same", "different", "unclear"):
        # Try first token only
        verdict_tok = verdict.split()[0]
        if verdict_tok in ("same", "different", "unclear"):
            verdict = verdict_tok
        else:
            return "error", f"unexpected verdict text: {text!r}"
    justification = " ".join(lines[1:]) if len(lines) > 1 else ""
    return verdict, justification


def _matched_pair_payload(icl_cid: str, sp_cid: str) -> tuple[str, str]:
    """Return (icl_scaffold_text, sp_system_prompt) for the judge."""
    icl_ctx = UNION_BY_CID[icl_cid]
    sp_ctx = UNION_BY_CID[sp_cid]
    if not isinstance(icl_ctx, ICLContext):
        raise TypeError(f"{icl_cid} is not an ICLContext")
    return scaffold_text(icl_ctx), sp_ctx.system_prompt


def _append_draft(payload: dict) -> None:
    DRAFT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DRAFT_LOG, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _write_block_sentinel(sp_cid: str, reason: str) -> None:
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    epoch = int(_dt.datetime.now(_dt.UTC).timestamp())
    sentinel = SENTINEL_DIR / f"issue-489-epm_failure-{epoch}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
        "issue": 489,
        "phase": "phase0a_sp_identity_check",
        "failure_class": "code",
        "sp_cid": sp_cid,
        "reason": reason,
        "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    sentinel.write_text(json.dumps(payload, indent=2))
    logger.error("Wrote BLOCK sentinel %s", sentinel)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the actual Claude call; log placeholder PASS for smoke wiring.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = _git_commit_hash()
    ts = _dt.datetime.now(_dt.UTC).isoformat()

    # The 5 anchors checked: the 4 MATCHED_PAIRS + SP08 (partial-match to IK06).
    # SP08 doesn't have a MATCHED_PAIRS entry (its matched_icl_cid is None) but
    # the M3 strong-kind decision included it; we check identity against IK06.
    pairs_to_check: list[tuple[str, str]] = [*MATCHED_PAIRS, ("IK06", "SP08")]
    results: list[dict] = []
    block_sp_cid: str | None = None
    block_reason: str | None = None

    for icl_cid, sp_cid in pairs_to_check:
        # Draft 1 = the locked-in SP string from i489_contexts.py.
        sp_ctx = UNION_BY_CID[sp_cid]
        a, b = _matched_pair_payload(icl_cid, sp_cid)
        verdict, justification = _ask_claude(a, b, args.dry_run)
        draft_payload = {
            "sp_cid": sp_cid,
            "icl_cid": icl_cid,
            "draft_idx": 1,
            "sp_string": sp_ctx.system_prompt if hasattr(sp_ctx, "system_prompt") else "",
            "verdict": verdict,
            "justification": justification,
            "git_commit": git_sha,
            "ts": ts,
            "dry_run": bool(args.dry_run),
        }
        _append_draft(draft_payload)
        if verdict == "same":
            logger.info("sp_cid=%s draft=1 verdict=same OK", sp_cid)
            results.append(
                {"sp_cid": sp_cid, "icl_cid": icl_cid, "final_draft": 1, "verdict": "same"}
            )
            continue

        # Verdict != "same" — try rewrite (1 round only).
        logger.warning("sp_cid=%s draft=1 verdict=%s — attempting rewrite", sp_cid, verdict)
        alts = ALT_PROMPTS_FOR_REWRITE.get(sp_cid)
        if not alts:
            block_sp_cid, block_reason = (
                sp_cid,
                (f"draft 1 verdict={verdict}; no rewrite alternative defined for {sp_cid}."),
            )
            break
        alt = alts[0]
        verdict2, justification2 = _ask_claude(a, alt, args.dry_run)
        draft_payload2 = {
            "sp_cid": sp_cid,
            "icl_cid": icl_cid,
            "draft_idx": 2,
            "sp_string": alt,
            "verdict": verdict2,
            "justification": justification2,
            "git_commit": git_sha,
            "ts": ts,
            "dry_run": bool(args.dry_run),
        }
        _append_draft(draft_payload2)
        if verdict2 == "same":
            results.append(
                {
                    "sp_cid": sp_cid,
                    "icl_cid": icl_cid,
                    "final_draft": 2,
                    "verdict": "same",
                    "frozen_alt_string": alt,
                    "frozen_alt_note": (
                        "i489_contexts.py SP string MUST BE UPDATED to this alt before launch."
                    ),
                }
            )
            logger.warning(
                "sp_cid=%s draft=2 PASSed with alt; update i489_contexts.py to alt before launch.",
                sp_cid,
            )
            continue
        block_sp_cid, block_reason = (
            sp_cid,
            (f"draft 1 verdict={verdict}; draft 2 verdict={verdict2}; both alts failed."),
        )
        break

    summary = {
        "issue": 489,
        "phase": "phase0a_sp_identity_check",
        "git_commit": git_sha,
        "wrote_at": ts,
        "n_pairs_checked": len(results) + (1 if block_sp_cid else 0),
        "results": results,
        "blocked": block_sp_cid is not None,
        "block_sp_cid": block_sp_cid,
        "block_reason": block_reason,
        "draft_log": str(DRAFT_LOG),
        "dry_run": bool(args.dry_run),
    }
    summary_path = OUT_DIR / "phase0a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Phase 0a summary -> %s", summary_path)

    if block_sp_cid:
        _write_block_sentinel(block_sp_cid, block_reason or "unknown")
        return 2
    if any(
        SP_CONTEXTS[i].cid not in {r["sp_cid"] for r in results}
        for i in range(len(SP_CONTEXTS))
        if SP_CONTEXTS[i].cid in {"SP03", "SP04", "SP06", "SP07", "SP08"}
    ):
        # Defensive: make sure every anchor we said we'd check actually appears
        # in results (catches an early-break logic bug rather than silently passing).
        raise RuntimeError(
            "Phase 0a results missing one or more anchored SP cids; "
            f"results: {[r['sp_cid'] for r in results]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
