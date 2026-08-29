"""Issue #489 Phase 0a — Claude-as-judge SP-string identity-sameness check.

Plan v5 §4.2.2 (M2 gate, round-6 loosened semantics).

Cosine-INDEPENDENT identity-sameness check for the 5 matched-pair anchors.
Rewrite criterion is NEVER cosine distance.

For each matched pair (IK persona-by-example block, SP system_prompt) in
``MATCHED_PAIRS`` plus SP08 (CoT-math anchor partial-matched to IK06), submit
both prompts to Claude with a pre-registered prompt and ask: ``same`` /
``different`` / ``unclear`` + brief justification.

- If ``same`` on the first draft: lock string, log to drafts ledger.
- If ``different`` / ``unclear``: pull an alternative from a hand-curated
  rewrite pool (loaded from ``ALT_PROMPTS_FOR_REWRITE``), re-run the check.
  - If the alt is judged ``same``: lock it (freeze into
    ``frozen_sp_strings.json``); the pair is **confirmatory** for H4(b).
  - If the alt is also non-``same``: the pair is **non-confirmatory** for
    H4(b) — the original locked string stays in use, Phase 0a RECORDS the
    final verdict and CONTINUES to the next pair (does NOT exit blocked).

The Phase 5 analyzer reads ``matched_pair_identity_verdicts.json`` to scope
H4(b)'s confirmatory test to confirmatory pairs only, while still computing
the descriptive H4(b) over all matched pairs.

Whether example-pirate ≈ instruction-pirate is exactly what H4(b) measures
empirically; the judge's call belongs in the H4(b) scope, not as a
pre-emptive whole-run gate. The cosine-independent judge + freeze-on-
accepted-rewrite anti-gaming properties are preserved; only the previous
fatal-on-non-``same``-after-rewrite behavior is removed.

Real infra failures (missing ``ANTHROPIC_API_KEY``, Anthropic API exception,
malformed judge output after retries) STILL exit non-zero — those are
infra/transport faults, not science verdicts.

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
# B3: frozen-override file read by i489_contexts._apply_frozen_overrides() at
# module load. Writing this file (and any cid that needs a rewrite) is the
# canonical freeze action — downstream phases pick up the new system_prompt the
# next time they import i489_contexts.
FROZEN_OVERRIDES_PATH = Path("eval_results/issue_489/phase0a/frozen_sp_strings.json")
# Round-6: Phase 5 H4(b) scope-selection input. Maps each matched pair (icl_cid,
# sp_cid) to a final-verdict record; ``confirmatory`` flips true iff the
# judge's final verdict is ``same`` (draft 1 OR accepted-rewrite draft 2).
MATCHED_PAIR_VERDICTS_PATH = Path(
    "eval_results/issue_489/phase0a/matched_pair_identity_verdicts.json"
)

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
    """Return (verdict, justification). Verdict in {same, different, unclear}.

    On a real run, infrastructure / transport / parser failures RAISE rather
    than return an ``error`` verdict — that contract is load-bearing under the
    round-6 loosened gate (a silently-returned ``error`` would be misclassified
    as ``non-confirmatory`` and mask the API failure).
    """
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
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        messages=[{"role": "user", "content": body}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Phase 0a judge returned empty response: {text!r}")
    verdict = lines[0].lower().strip(".,:; ")
    if verdict not in ("same", "different", "unclear"):
        # Try first token only
        verdict_tok = verdict.split()[0]
        if verdict_tok in ("same", "different", "unclear"):
            verdict = verdict_tok
        else:
            raise RuntimeError(
                f"Phase 0a judge returned unparseable verdict text: {text!r}. "
                "Expected first line to be 'same', 'different', or 'unclear'."
            )
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

    # Round-6 loosened semantics: a non-"same" final verdict (after 1 rewrite
    # attempt) NO LONGER blocks the run. We record the verdict and continue;
    # Phase 5 H4(b) reads matched_pair_identity_verdicts.json and scopes its
    # confirmatory test to confirmatory pairs only. Infra-level failures
    # (missing API key, malformed judge response) still raise + exit non-zero
    # via _ask_claude.
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
            logger.info("sp_cid=%s draft=1 verdict=same OK (confirmatory)", sp_cid)
            results.append(
                {
                    "sp_cid": sp_cid,
                    "icl_cid": icl_cid,
                    "final_draft": 1,
                    "final_verdict": "same",
                    "confirmatory": True,
                }
            )
            continue

        # Verdict != "same" — try rewrite (1 round only).
        logger.warning("sp_cid=%s draft=1 verdict=%s — attempting rewrite", sp_cid, verdict)
        alts = ALT_PROMPTS_FOR_REWRITE.get(sp_cid)
        if not alts:
            # No rewrite alternative defined → record the draft-1 verdict as
            # final and mark the pair non-confirmatory; CONTINUE.
            logger.warning(
                "sp_cid=%s has no rewrite alternative; recording non-confirmatory "
                "and continuing (round-6 loosened gate).",
                sp_cid,
            )
            results.append(
                {
                    "sp_cid": sp_cid,
                    "icl_cid": icl_cid,
                    "final_draft": 1,
                    "final_verdict": verdict,
                    "confirmatory": False,
                    "rewrite_attempted": False,
                    "note": "no rewrite alternative defined for this sp_cid",
                }
            )
            continue
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
                    "final_verdict": "same",
                    "confirmatory": True,
                    "frozen_alt_string": alt,
                    "frozen_alt_note": (
                        "i489_contexts.py SP string AUTO-FROZEN via "
                        "frozen_sp_strings.json (B3 fix)."
                    ),
                }
            )
            logger.warning(
                "sp_cid=%s draft=2 PASSed with alt; freezing into %s (confirmatory)",
                sp_cid,
                FROZEN_OVERRIDES_PATH,
            )
            continue
        # Round-6: both drafts non-"same" → record final verdict from draft 2,
        # mark non-confirmatory, CONTINUE (no block sentinel, no exit).
        logger.warning(
            "sp_cid=%s draft=1 verdict=%s, draft=2 verdict=%s — non-confirmatory; "
            "continuing (round-6 loosened gate; H4(b) confirmatory scope excludes this pair).",
            sp_cid,
            verdict,
            verdict2,
        )
        results.append(
            {
                "sp_cid": sp_cid,
                "icl_cid": icl_cid,
                "final_draft": 2,
                "final_verdict": verdict2,
                "confirmatory": False,
                "rewrite_attempted": True,
                "rewrite_verdict": verdict2,
            }
        )

    # B3: write the frozen-SP-strings override file so downstream phases pick
    # up the rewrite the next time they import i489_contexts. The file is
    # ALWAYS written (even when empty) so its presence is part of the audit
    # trail; the contexts loader simply no-ops if the dict is empty.
    frozen_overrides: dict[str, str] = {
        r["sp_cid"]: r["frozen_alt_string"]
        for r in results
        if r.get("final_draft") == 2 and "frozen_alt_string" in r
    }
    FROZEN_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_OVERRIDES_PATH.write_text(json.dumps(frozen_overrides, indent=2))
    logger.info(
        "Phase 0a frozen overrides (%d cids) -> %s",
        len(frozen_overrides),
        FROZEN_OVERRIDES_PATH,
    )

    # Round-6: write the H4(b) scope-selection input. Maps each matched pair
    # (icl_cid, sp_cid) to {final_verdict, confirmatory, frozen_string_used}.
    # The Phase 5 analyzer reads this file and scopes H4(b)'s confirmatory
    # test to confirmatory pairs only; descriptive H4(b) covers all matched
    # pairs.
    verdict_records = []
    for r in results:
        rec = {
            "icl_cid": r["icl_cid"],
            "sp_cid": r["sp_cid"],
            "final_verdict": r["final_verdict"],
            "confirmatory": bool(r["confirmatory"]),
            "final_draft": r["final_draft"],
            "frozen_string_used": r.get("frozen_alt_string"),
        }
        verdict_records.append(rec)
    MATCHED_PAIR_VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHED_PAIR_VERDICTS_PATH.write_text(
        json.dumps(
            {
                "issue": 489,
                "phase": "phase0a_sp_identity_check",
                "git_commit": git_sha,
                "wrote_at": ts,
                "schema_version": "i489_phase0a_verdicts_v1",
                "verdicts": verdict_records,
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    logger.info(
        "Phase 0a matched-pair verdicts (%d pairs) -> %s",
        len(verdict_records),
        MATCHED_PAIR_VERDICTS_PATH,
    )

    confirmatory_pairs = [(r["icl_cid"], r["sp_cid"]) for r in results if r["confirmatory"]]
    non_confirmatory_pairs = [
        {"pair": [r["icl_cid"], r["sp_cid"]], "final_verdict": r["final_verdict"]}
        for r in results
        if not r["confirmatory"]
    ]
    summary = {
        "issue": 489,
        "phase": "phase0a_sp_identity_check",
        "git_commit": git_sha,
        "wrote_at": ts,
        "n_pairs_checked": len(results),
        "n_confirmatory": len(confirmatory_pairs),
        "confirmatory_pairs": [list(p) for p in confirmatory_pairs],
        "non_confirmatory_pairs": non_confirmatory_pairs,
        "results": results,
        "draft_log": str(DRAFT_LOG),
        "frozen_overrides_path": str(FROZEN_OVERRIDES_PATH),
        "n_frozen_overrides": len(frozen_overrides),
        "matched_pair_verdicts_path": str(MATCHED_PAIR_VERDICTS_PATH),
        "dry_run": bool(args.dry_run),
    }
    summary_path = OUT_DIR / "phase0a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Phase 0a summary -> %s (%d/%d pairs confirmatory)",
        summary_path,
        len(confirmatory_pairs),
        len(results),
    )

    # Round-6: no block path. The previous design exited 2 + wrote a BLOCK
    # sentinel when any pair failed the judge after 1 rewrite; that pre-
    # empted H4(b) on a question H4(b) is supposed to MEASURE. Now the
    # gate records per-pair verdicts and CONTINUES; real infra/transport
    # failures still propagate as uncaught exceptions from _ask_claude.

    # Defensive: make sure every anchored SP cid actually has a result row.
    # Under round-6 this is a strict invariant (no early break anywhere); a
    # missing cid means the loop / results-append logic regressed.
    if any(
        SP_CONTEXTS[i].cid not in {r["sp_cid"] for r in results}
        for i in range(len(SP_CONTEXTS))
        if SP_CONTEXTS[i].cid in {"SP03", "SP04", "SP06", "SP07", "SP08"}
    ):
        raise RuntimeError(
            "Phase 0a results missing one or more anchored SP cids; "
            f"results: {[r['sp_cid'] for r in results]}"
        )

    # B3 fail-loud invariant: after writing the override file, reload
    # i489_contexts in a FRESH python and verify the active SP strings == the
    # frozen accepted drafts (i.e. the loader actually consumed the override).
    # If a downstream phase imports i489_contexts BEFORE this script runs, the
    # override has no effect on already-imported modules — flagging that case
    # here prevents the silent-success bug class from round 1.
    if frozen_overrides:
        import subprocess as _subprocess
        import sys as _sys

        check_src = (
            "import json, sys, os;"
            f"os.environ.setdefault('I489_FROZEN_SP_OVERRIDES', {str(FROZEN_OVERRIDES_PATH)!r});"
            "from explore_persona_space.experiments.i489_contexts import UNION_BY_CID;"
            "active = {cid: UNION_BY_CID[cid].system_prompt for cid in "
            f"{list(frozen_overrides)!r}"
            "};"
            "print(json.dumps(active))"
        )
        try:
            verify_out = _subprocess.check_output(
                [_sys.executable, "-c", check_src], stderr=_subprocess.STDOUT, timeout=120
            )
            active = json.loads(verify_out.decode().strip().splitlines()[-1])
        except Exception as e:
            raise RuntimeError(
                f"B3 freeze verification failed: subprocess re-import crashed: {e}"
            ) from e
        for sp_cid, expected in frozen_overrides.items():
            if active.get(sp_cid) != expected:
                raise RuntimeError(
                    f"B3 freeze verification: sp_cid={sp_cid} active system_prompt "
                    f"!= frozen accepted draft. Active: {active.get(sp_cid)!r}; "
                    f"Expected: {expected!r}. The frozen_sp_strings.json override "
                    f"is not being consumed — investigate i489_contexts._apply_frozen_overrides()."
                )
        logger.info("B3: %d frozen SP strings verified via subprocess re-import.", len(active))

    return 0


if __name__ == "__main__":
    sys.exit(main())
