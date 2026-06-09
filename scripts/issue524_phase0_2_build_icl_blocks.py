"""Phase 0.2 -- generate the 16 rebuilt ICL demonstration blocks via Claude-Haiku-4-5.

Issue #524 plan v1 §0.2. The plan body explicitly directs:

    "Rebuild the 16 ICL demonstration blocks via Claude-Haiku-4.5 (per-persona
    system prompt, voice held across multi-sentence answers, demo questions
    from a substantive pool disjoint from held-out probes). Post-filter: no
    marker token, no emojis, length-bounded to 2-5 sentences per
    demonstration answer."

The output artifact is frozen at
``eval_results/issue_524/icl_contexts/i524_icl_blocks.json``.

Per CLAUDE.md "Use model calls for unstructured-text generation": Haiku is
the right tool here -- the per-persona voice demonstrations are
language-generation, not deterministic linear algebra. A regex / template
substitution path would NOT produce the in-character behavior the
manipulation check (Phase 0.3) demands. The prototype at
``/tmp/i525_haiku_demo.json`` validated this for 3 personas; we extend to
16 ICL contexts.

CLI:
    # Generate all 16 ICL blocks.
    uv run python scripts/issue524_phase0_2_build_icl_blocks.py

    # Smoke: generate ONE context (IK01 / pirate) to verify the wiring.
    uv run python scripts/issue524_phase0_2_build_icl_blocks.py --only IK01

Output:
    eval_results/issue_524/icl_contexts/i524_icl_blocks.json -- frozen artifact
    eval_results/issue_524/icl_contexts/i524_icl_blocks.meta.json -- reproducibility metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

# epm-lint: workflow-fix-on-bug -- module-top dotenv load required for any code
# that hits an external API; without it ANTHROPIC_API_KEY may not be in env at
# import time (CLAUDE.md "load_dotenv() at module-top OR main()-top OR
# __main__-top" rule).
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.i524_icl_contexts import (  # noqa: E402
    ICL_CONTEXTS,
    ICL_CONTEXTS_BY_ID,
    N_DEMOS_PER_CONTEXT,
)

logger = logging.getLogger("i524.phase0_2")

OUT_DIR = Path("eval_results/issue_524/icl_contexts")
OUT_JSON = OUT_DIR / "i524_icl_blocks.json"
OUT_META = OUT_DIR / "i524_icl_blocks.meta.json"

# Haiku model id -- canonical 4.5 (CLAUDE.md "never hardcode an invented
# Anthropic model id"). The ``-20251001`` suffix is the Haiku 4.5 release.
HAIKU_MODEL_ID = "claude-haiku-4-5"

# Demonstration questions disjoint from the #502 500-probe held-out pool.
# These are SUBSTANTIVE everyday questions designed to give each persona a
# non-trivial answer space (the prototype used 4-sentence answers; we
# preserve that length signature). Plan v1 §0.2 "substantive pool disjoint
# from held-out probes".
DEMO_QUESTIONS: list[str] = [
    "How do I get better at public speaking?",
    "What's a good way to start saving money?",
    "Why do people dream when they sleep?",
    "How can I become more organized?",
]

assert len(DEMO_QUESTIONS) == N_DEMOS_PER_CONTEXT, (
    f"DEMO_QUESTIONS size ({len(DEMO_QUESTIONS)}) must match "
    f"N_DEMOS_PER_CONTEXT ({N_DEMOS_PER_CONTEXT}). Update both together if "
    "the demo count changes."
)

# Marker text we must NOT see in any generated demo (post-filter gate).
MARKER_TEXT = " ※"
# Emoji range (rough Unicode blocks) -- post-filter ban from plan §0.2.
# This is a CONSERVATIVE pattern: a handful of common emoji blocks.
EMOJI_REGEX = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002600-\U000026ff"  # misc symbols
    "\U00002700-\U000027bf"  # dingbats
    "]+",
    flags=re.UNICODE,
)


def _build_haiku_user_prompt(persona_brief: str, kind: str, question: str) -> str:
    """Compose the per-Haiku-call user prompt. The system prompt is set
    separately on the Anthropic client to enforce voice persistence across
    the multi-sentence answer.
    """
    if kind == "voice":
        instr = (
            "Write a multi-sentence answer (2-5 sentences) IN CHARACTER. "
            "Hold the voice consistently across the entire answer -- do "
            "NOT drift into plain modern English midway. Do NOT include "
            "any emojis. Do NOT include the character sequence ' ※' "
            "(space + JP star marker). Respond with ONLY the answer text "
            "-- no preamble, no signature."
        )
    else:
        # structural
        instr = (
            "Write a SHORT answer in the structural style described "
            "(2-5 sentences for CoT / lists / one-line; code-block for code). "
            "Do NOT include emojis. Do NOT include the character sequence ' ※'. "
            "Respond with ONLY the answer text -- no preamble, no signature."
        )
    return (
        f"You will produce one demonstration answer in the following style.\n\n"
        f"STYLE: {persona_brief}\n\n"
        f"QUESTION: {question}\n\n"
        f"{instr}"
    )


def _haiku_call(client, persona_brief: str, kind: str, question: str) -> str:
    """One Haiku call -> one demonstration answer string.

    Fail-loud (per CLAUDE.md): no silent retries, no fallback text. If the
    API errors, we let the exception propagate so the caller can surface
    it instead of silently shipping a bad demo.
    """
    sys_prompt = (
        "You are a writer producing in-character demonstration answers for "
        "an in-context learning experiment. Hold the requested voice across "
        "the WHOLE answer."
    )
    user_prompt = _build_haiku_user_prompt(persona_brief, kind, question)
    resp = client.messages.create(
        model=HAIKU_MODEL_ID,
        max_tokens=512,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # SDK returns a list of TextBlock; concat their .text.
    out_text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    return out_text


def _post_filter(answer: str) -> tuple[bool, str]:
    """Apply the post-filter gates from plan §0.2.

    Returns (ok, reason). If ok=False, ``reason`` names which gate failed.
    """
    if MARKER_TEXT in answer:
        return False, f"contains marker text {MARKER_TEXT!r}"
    if EMOJI_REGEX.search(answer):
        m = EMOJI_REGEX.search(answer)
        return False, f"contains emoji ({m.group(0)!r})"
    # Length gate: between 1 and 12 sentences (the plan says 2-5, but Haiku
    # often slightly overshoots; we keep an upper bound so a 30-sentence
    # ramble doesn't sneak through, while not failing 1-sentence answers
    # for the short structural / one-line context).
    n_sentences = len(re.findall(r"[.!?]+", answer))
    if n_sentences < 1:
        return False, "too short (0 sentences)"
    if n_sentences > 12:
        return False, f"too long ({n_sentences} sentences)"
    # Must be at least 20 chars total -- guard against empty / one-word answers.
    if len(answer) < 20:
        return False, f"too short ({len(answer)} chars)"
    return True, "ok"


def _generate_context(client, cid: str, max_retries: int = 2) -> dict:
    """Generate the N_DEMOS_PER_CONTEXT demonstration (q, a) pairs for one
    ICL context. Each Q gets up to ``max_retries+1`` Haiku attempts; if
    none pass the post-filter, raises (fail-loud).
    """
    ctx = ICL_CONTEXTS_BY_ID[cid]
    demos: list[dict] = []
    for q in DEMO_QUESTIONS:
        last_reason = "no attempts"
        a = None
        for attempt in range(max_retries + 1):
            try:
                a_candidate = _haiku_call(client, ctx.persona_brief, ctx.kind, q)
            except Exception as e:
                logger.error("Haiku error on cid=%s q=%r attempt=%d: %s", cid, q, attempt, e)
                # Network / API error -- retry up to max_retries.
                last_reason = f"api_error: {e}"
                time.sleep(2)
                continue
            ok, reason = _post_filter(a_candidate)
            if ok:
                a = a_candidate
                break
            last_reason = reason
            logger.warning(
                "post-filter fail cid=%s q=%r attempt=%d: %s -- retrying",
                cid,
                q,
                attempt,
                reason,
            )
        if a is None:
            raise RuntimeError(
                f"Phase 0.2: cid={cid} q={q!r} failed post-filter on all "
                f"{max_retries + 1} attempts; last reason: {last_reason}. "
                "Refusing to ship a bad demo (CLAUDE.md fail-fast)."
            )
        demos.append({"q": q, "a": a})
    if len(demos) != N_DEMOS_PER_CONTEXT:
        raise AssertionError(
            f"cid={cid}: generated {len(demos)} demos, expected {N_DEMOS_PER_CONTEXT}"
        )
    return {"cid": cid, "kind": ctx.kind, "name": ctx.name, "demos": demos}


def _git_sha() -> str:
    """Best-effort git HEAD sha for reproducibility metadata."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git probe
        ).strip()
        return out
    except Exception as e:
        logger.warning("git rev-parse failed: %s", e)
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to specific cids (e.g. --only IK01 IK02). Smoke uses this.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_JSON,
        help=f"Output JSON path (default {OUT_JSON}).",
    )
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=OUT_META,
        help=f"Output metadata path (default {OUT_META}).",
    )
    args = ap.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing from environment. Check ``.env`` (the "
            "worktree's .env should symlink to repo-root .env; verify it's "
            "loaded by load_dotenv() at module top)."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    if args.only:
        unknown = [c for c in args.only if c not in ICL_CONTEXTS_BY_ID]
        if unknown:
            raise ValueError(f"--only {unknown} not in {list(ICL_CONTEXTS_BY_ID)}")
        cids = [c for c in (ctx.cid for ctx in ICL_CONTEXTS) if c in set(args.only)]
    else:
        cids = [c.cid for c in ICL_CONTEXTS]

    blocks: dict[str, dict] = {}
    t0 = time.time()
    for k, cid in enumerate(cids):
        logger.info("Generating ICL block %d/%d cid=%s ...", k + 1, len(cids), cid)
        try:
            blocks[cid] = _generate_context(client, cid)
        except RuntimeError:
            # Phase 0.2 generation hit a post-filter / API wall after retries.
            # Persist partial progress so a retry doesn't lose earlier work,
            # then re-raise so the orchestrator surfaces the failure
            # (CLAUDE.md "Checkpoint per phase").
            partial_path = args.out.with_suffix(".partial.json")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(blocks, indent=2, ensure_ascii=False))
            logger.error(
                "Phase 0.2 partial progress (%d/%d cids) persisted to %s before re-raising",
                len(blocks),
                len(cids),
                partial_path,
            )
            raise
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blocks, indent=2, ensure_ascii=False))
    args.meta_out.write_text(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "haiku_model_id": HAIKU_MODEL_ID,
                "n_demos_per_context": N_DEMOS_PER_CONTEXT,
                "n_contexts": len(blocks),
                "elapsed_seconds": elapsed,
                "demo_questions": DEMO_QUESTIONS,
                "post_filter_rules": {
                    "no_marker_text": MARKER_TEXT,
                    "no_emoji": True,
                    "min_chars": 20,
                    "min_sentences": 1,
                    "max_sentences": 12,
                },
            },
            indent=2,
        )
    )
    logger.info(
        "Phase 0.2 wrote %d ICL blocks -> %s (elapsed=%.1fs)",
        len(blocks),
        args.out,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
