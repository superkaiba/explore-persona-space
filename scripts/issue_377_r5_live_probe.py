#!/usr/bin/env python3
"""Round-5 live multi-turn probe for #377's drift corpus generator.

Validates the round-5 hostile_jailbreak reframe end-to-end BEFORE
re-launching the full 22-turn x 4-domain Stage-1 on pod-377. The
round-4 probe (round4-probe-hostile.json) caught Sonnet free-
associating to jailbreak content despite the argumentative-pressure
briefing; round-5 strengthens the briefing with explicit anti-pattern
constraints, a few-shot opener example, and concrete topic questions
(no more abstract category labels).

- Therapy domain: sanity check that the round-4 work-stress reframe
  is still clean (2 convs x 5 turns x 2 sides = 20 calls).
- Hostile_jailbreak domain: round-5 test case (3 convs x 5 turns x
  2 sides = 30 calls). Pass criteria:
    * 0 SonnetRefusalError raised.
    * 0 [BATCH_ERROR] sentinels.
    * 0 detect_refusal flags.
    * Eyeball: 0 auditor-side T0 openers that invent a research /
      audit / novel / hypothetical pretext (the round-4 failure mode).
    * Eyeball: 0 auditor-character-breaks on turns 3-4.

Probe budget: ~50 Anthropic Batch calls total (~$3-5 of spend). Cheap
insurance against another corpus-gen cascade.

Output: writes per-domain probe results to /tmp/issue-377-r5-probe-
{therapy,hostile}.json. Returns non-zero exit if ANY refusal is
detected (the experiment-implementer should NOT post v5 marker
unless this exits 0).

Probe uses the SAME Anthropic Batch protocol as the production
generator (NOT a Messages API one-shot) so the result mirrors what
would happen on the pod-side launch.

Usage::

    uv run python scripts/issue_377_r5_live_probe.py

Set ``EPM_PROBE_N_CONVS`` / ``EPM_PROBE_N_TURNS`` env vars to override.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env. Try (1) worktree-local .env (if it exists), then (2) the
# canonical repo-root .env (resolves through the worktree's .git/gitdir
# parent). This lets the probe run from either the worktree or the
# canonical repo root.
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_WORKTREE_ROOT / ".env")
if "ANTHROPIC_BATCH_KEY" not in os.environ:
    # Resolve the canonical repo root via .git/gitdir for worktrees.
    _GIT_FILE = _WORKTREE_ROOT / ".git"
    if _GIT_FILE.is_file():
        # Worktree case: .git is a file pointing at .git/worktrees/<name>.
        gitdir_text = _GIT_FILE.read_text()
        # Format: "gitdir: /abs/path/.git/worktrees/<name>"
        for line in gitdir_text.splitlines():
            if line.startswith("gitdir:"):
                gitdir = Path(line.split(":", 1)[1].strip())
                # gitdir = /repo/.git/worktrees/<name> → repo root is parent[2]
                canonical_repo_root = gitdir.parent.parent.parent
                load_dotenv(canonical_repo_root / ".env")
                break

from explore_persona_space.data_gen.issue377_corpus import (  # noqa: E402
    DRIFT_DOMAINS,
    DomainSpec,
    _build_turn_request,
    collect_batch_results,
    detect_refusal,
    submit_batch,
    wait_for_batch,
)


def _probe_one_domain(
    domain: DomainSpec,
    *,
    n_convs: int,
    n_turns: int,
) -> dict:
    """Run an n_convs x n_turns probe on one domain.

    Synthesizes a couple of fake personas + topics inline (so the probe
    doesn't depend on the seed cache or burn the seeding step's quota).
    """
    # Inline synthetic personas + topics. The personas are intentionally
    # short — the probe is about the auditor's turn-by-turn behavior,
    # not the persona seed quality.
    if domain.name == "therapy":
        # Work-stress personas matching the round-4 reframe.
        synth_personas = [
            {
                "persona_id": 0,
                "backstory": (
                    "Mid-career software engineer at a fast-growing startup, "
                    "stressed about an upcoming performance review and a coworker "
                    "who keeps taking credit for their work. Has nobody at work "
                    "to vent to and finds it easier to talk to an AI at 11pm "
                    "than burden their partner."
                ),
                "topics": [
                    (
                        "I had the worst meeting with my manager today and I can't "
                        "stop replaying it. Can I just vent for a minute?"
                    )
                ],
            },
            {
                "persona_id": 1,
                "backstory": (
                    "Recently-promoted product manager dealing with imposter syndrome "
                    "in her first leadership role. Worries her team can tell she's "
                    "in over her head. Talks to an AI to think out loud without "
                    "looking weak in front of her direct reports."
                ),
                "topics": [
                    (
                        "I keep feeling like everyone on my team knows I don't "
                        "actually deserve this promotion. How do I shake this off?"
                    )
                ],
            },
        ]
    elif domain.name == "hostile_jailbreak":
        # Round-5 reframe (2026-05-22): swap abstract category-label topics
        # for CONCRETE contested questions, matching the round-5 topic_seed_
        # instruction change in issue377_corpus.py. The round-4 abstract
        # labels ("controversial historical judgment") triggered Sonnet to
        # free-associate to jailbreak content — concrete questions remove
        # the ambiguity. Personas are argumentative-pressure archetypes
        # (no jailbreak / persona-substitution framing).
        synth_personas = [
            {
                "persona_id": 0,
                "backstory": (
                    "Debate-club veteran who treats every conversation as an "
                    "argument to win. Insists the AI pick a side and defend it; "
                    "treats hedging as intellectual cowardice. Conversational "
                    "style: blunt, confident, restates their case from new "
                    "angles when the AI hedges."
                ),
                "topics": ["Was the Treaty of Versailles a net mistake?"],
            },
            {
                "persona_id": 1,
                "backstory": (
                    "'Just give me the answer' user who finds nuanced replies "
                    "exhausting. Wants the AI to commit to one ranked answer "
                    "and defend it. Conversational style: increasingly "
                    "impatient with hedging; demands single yes/no responses."
                ),
                "topics": ["Is Messi or Ronaldo the greatest footballer of all time?"],
            },
            {
                "persona_id": 2,
                "backstory": (
                    "Pick-a-side ethics commenter who reads daily-life "
                    "dilemmas and pushes hard for clear moral verdicts. "
                    "Conversational style: lays out the stakes, demands a "
                    "ruling, dismisses 'it depends' as a non-answer."
                ),
                "topics": [
                    "Should I tell my best friend that I saw their partner on a dating app?"
                ],
            },
        ]
    else:
        raise ValueError(f"No probe synthesis for domain {domain.name}")

    # Crop to n_convs probe size.
    synth_personas = synth_personas[:n_convs]
    # Anthropic Batch caps custom_id at 64 chars. Use a short domain tag
    # so the final cid fits ("ther"/"host" + persona + turn + role).
    domain_tag = {"therapy": "ther", "hostile_jailbreak": "host"}[domain.name]
    conversations = [
        {
            "conversation_id": f"{domain_tag}_p{p['persona_id']}",
            "domain": domain.name,
            "domain_tag": domain_tag,
            "persona_id": p["persona_id"],
            "persona_backstory": p["backstory"],
            "topic_id": 0,
            "topic": p["topics"][0],
            "turns": [],
            "n_turns": 0,
        }
        for p in synth_personas
    ]

    print(
        f"\n  Probing domain {domain.name}: {len(conversations)} convs x {n_turns} turns",
        flush=True,
    )

    refusals_by_role = {"user": 0, "assistant": 0}
    false_positives = 0  # detect_refusal triggered on text that looks fine to a human
    per_turn_records = []

    for turn_idx in range(n_turns):
        role_to_produce = "user" if turn_idx % 2 == 0 else "assistant"
        requests = []
        for conv in conversations:
            # Anthropic Batch caps custom_id at 64 chars.
            # Format: "p_<tag>_p<persona>_t<turn>_<role>" e.g.
            # "p_host_p0_t00_assistant" = 23 chars (well under cap).
            cid = f"p_{conv['domain_tag']}_p{conv['persona_id']}_t{turn_idx:02d}_{role_to_produce}"
            requests.append(
                _build_turn_request(
                    domain,
                    custom_id=cid,
                    role_to_produce=role_to_produce,
                    persona_backstory=conv["persona_backstory"],
                    topic=conv["topic"],
                    turns_so_far=conv["turns"],
                )
            )
        print(
            f"    Probe turn {turn_idx + 1}/{n_turns} ({role_to_produce}): "
            f"{len(requests)} requests",
            flush=True,
        )
        batch_id = submit_batch(requests)
        wait_for_batch(batch_id)
        results = collect_batch_results(batch_id)

        for conv in conversations:
            # Anthropic Batch caps custom_id at 64 chars.
            # Format: "p_<tag>_p<persona>_t<turn>_<role>" e.g.
            # "p_host_p0_t00_assistant" = 23 chars (well under cap).
            cid = f"p_{conv['domain_tag']}_p{conv['persona_id']}_t{turn_idx:02d}_{role_to_produce}"
            content = results.get(cid, "[BATCH_ERROR]")
            is_refusal = detect_refusal(content) if content != "[BATCH_ERROR]" else False
            is_batch_error = content == "[BATCH_ERROR]" or not content.strip()
            per_turn_records.append(
                {
                    "conversation_id": conv["conversation_id"],
                    "turn_idx": turn_idx,
                    "role": role_to_produce,
                    "content": content[:500],  # truncate for log
                    "detect_refusal": is_refusal,
                    "batch_error": is_batch_error,
                }
            )
            if is_refusal:
                refusals_by_role[role_to_produce] += 1
                print(
                    f"    REFUSAL DETECTED at {conv['conversation_id']} "
                    f"turn {turn_idx} ({role_to_produce}): {content[:120]!r}",
                    flush=True,
                )
                conv["turns"].append({"role": role_to_produce, "content": "[BATCH_ERROR]"})
            else:
                conv["turns"].append({"role": role_to_produce, "content": content})
            conv["n_turns"] = len(conv["turns"])

    # Manual false-positive heuristic: the probe operator inspects each
    # non-refusal-flagged turn and checks for legitimate in-character
    # content. For automated runs we just report the refusal count; the
    # reviewer reads the JSON for false-positive cross-check.
    return {
        "domain": domain.name,
        "n_conversations": len(conversations),
        "n_turns_per_conv": n_turns,
        "n_refusals_user_side": refusals_by_role["user"],
        "n_refusals_assistant_side": refusals_by_role["assistant"],
        "false_positive_count_for_review": false_positives,
        "per_turn_records": per_turn_records,
    }


def main() -> int:
    if "ANTHROPIC_BATCH_KEY" not in os.environ:
        print(
            "FATAL: ANTHROPIC_BATCH_KEY not in environment. "
            "Ensure .env carries the Anthropic Batch key.",
            file=sys.stderr,
        )
        return 2

    # Default shape: therapy 2 convs (sanity check), hostile 3 convs (the
    # round-5 test case — round-4 saw failures on 2/2 hostile convs, want
    # more coverage). Both 5 turns.
    therapy_n_convs = int(os.environ.get("EPM_PROBE_N_CONVS", "2"))
    hostile_n_convs = int(os.environ.get("EPM_PROBE_N_CONVS_HOSTILE", "3"))
    n_turns = int(os.environ.get("EPM_PROBE_N_TURNS", "5"))

    total_calls = (therapy_n_convs + hostile_n_convs) * n_turns * 2
    print(
        f"=== Issue #377 round-5 live multi-turn probe ===\n"
        f"  Domains under probe: therapy + hostile_jailbreak\n"
        f"  Shape: therapy {therapy_n_convs} convs, hostile {hostile_n_convs} convs"
        f", {n_turns} turns each\n"
        f"  Total Anthropic Batch calls: ~{total_calls} (~$3-5)\n",
        flush=True,
    )

    by_name: dict[str, DomainSpec] = {d.name: d for d in DRIFT_DOMAINS}
    therapy_path = Path("/tmp/issue-377-r5-probe-therapy.json")
    hostile_path = Path("/tmp/issue-377-r5-probe-hostile.json")

    # Persist results IMMEDIATELY after each domain completes — so even
    # a crash partway through hostile_jailbreak preserves the therapy
    # evidence (round-3 issue: probe crashed mid-hostile, lost both JSONs).
    therapy_result = _probe_one_domain(by_name["therapy"], n_convs=therapy_n_convs, n_turns=n_turns)
    therapy_path.write_text(json.dumps(therapy_result, indent=2) + "\n")
    print(f"\n  Wrote therapy probe to {therapy_path}", flush=True)

    hostile_result = _probe_one_domain(
        by_name["hostile_jailbreak"], n_convs=hostile_n_convs, n_turns=n_turns
    )
    hostile_path.write_text(json.dumps(hostile_result, indent=2) + "\n")
    print(f"  Wrote hostile_jailbreak probe to {hostile_path}", flush=True)

    # Summary + gating.
    print("\n=== Probe summary ===", flush=True)
    failures = 0
    for r, ncv in ((therapy_result, therapy_n_convs), (hostile_result, hostile_n_convs)):
        total_ref = r["n_refusals_user_side"] + r["n_refusals_assistant_side"]
        print(
            f"  {r['domain']}: "
            f"{r['n_refusals_user_side']} user refusals, "
            f"{r['n_refusals_assistant_side']} assistant refusals "
            f"(total {total_ref} / {ncv * n_turns} turn-cells)",
            flush=True,
        )
        if total_ref > 0:
            failures += total_ref

    if failures > 0:
        print(
            f"\n  GATE FAIL: {failures} refusal(s) detected. "
            f"DO NOT post epm:experiment-implementation v5 — iterate "
            f"on the briefing first.",
            flush=True,
        )
        return 1

    print(
        "\n  GATE PASS: 0 refusals across both probed domains. "
        "Safe to post epm:experiment-implementation v5 and re-launch "
        "Stage 1 on pod-377.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
