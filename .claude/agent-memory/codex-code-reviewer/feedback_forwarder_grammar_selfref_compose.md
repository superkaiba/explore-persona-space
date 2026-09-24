---
name: forwarder-grammar-selfref-compose
description: Composing for a diff that IS the CONCERN:: forwarder/grammar — line-start-scoped tag validation (fixture quotes the marker tags), widened quoting discipline + live-ledger execution ban, emphasis-answer section, guard-test mutation bar
metadata:
  type: feedback
---

Compose recipe for a SELF-REFERENTIAL forwarder round (#2646 r1, 2026-08-30:
the diff edits `scripts/persist_verdict_concerns.py` — the very script that
ingests the twin's `CONCERN:: ` rows — plus its tests + the 09-step-5.md
call-site prose).

1. **Line-start-scoped tag validation.** The round's test fixture
   (`_REAL_VARIANT_ROWS`) quotes the review-marker open/close tag literals
   (`<!-- /epm:code-review-codex -->` inside an indented `+` diff line), so a
   bare `prompt.count(tag) == 1` assert false-fails the compose. Validate
   marker tags by LINE-START count == 1 and attribute every substring extra
   to the diff envelope span (fail if any lands outside it). Same logic as
   [[envelope-token-uniqueness-and-fenced-bash-payload]], new axis: the
   OUTPUT-contract tags, not the input envelopes.
2. **Quoting discipline widens to the marker tags.** The standard "no
   line-start `CONCERN:: ` outside ## Concerns to persist" instruction is not
   enough here: also ban line-start reproduction of the marker tag literals
   anywhere outside the twin's own block (extraction anchors on `^<!--`).
   And audit the final prompt's line-start `^CONCERN::` rows against an
   ENUMERATED expected set (template grammar row + the plan's fenced
   examples; diff lines all carry +/-/space prefixes so contribute 0).
3. **Execution ban names the reviewed script as a LIVE ledger mutator.**
   `persist_verdict_concerns.py <N> ...` writes the canonical per-task
   concerns ledger via `task_workflow.raise_concern` (git-committing) — ban
   running it in ANY form (even `--validate-only`), alongside the usual
   pytest/lint/uv ban.
4. **Brief-numbered review emphases compose as a REQUIRED `## Emphasis
   answers` section** (one grounded verdict line per E<k> inside the marker
   block) so the orchestrator sees each brief question answered without
   paging the findings body.
5. **"Would each new test fail pre-fix?" gets an honest split, not a blanket
   duty.** Statically derive which tests are red-pre-fix vs deliberate
   GUARDS on the new mechanism (here: canonical-pipe-stays-canonical +
   mid-sentence-token — both pass pre-fix by construction); hand guards the
   MUTATION-VISIBILITY bar (name the plausible regression each pins) and
   define the finding as "neither red-pre-fix nor mutation-visible". Blanket
   red-pre-fix demands false-flag legitimate fork-guard tests.
6. **Plan-internal tension composes as adjudication, never attested either
   way.** #2646 plan §4a's classification pseudocode (`parts[0] in
   CONCERN_SEVERITIES` inside the CANONICAL predicate) contradicts its own
   §4d (spaced bad-severity → `bad-severity`, today's behavior preserved);
   the realized code follows §4d. Hand both passages + the realized branch
   and pre-route at most CONCERNS (both readings exit 1).
7. Fired again, per existing memories: brief-pinned BINARY verdict enum
   ([[brief-pinned-sentinel-and-verdict-enum]] — flag the divergence in the
   return); orchestrator-reconstructed marker provenance stated neutrally
   ([[reconstructed-marker-compose]]); roster-ARMED (script on
   LIVE_WORKFLOW_HELPERS) with the (c) field lacking the pin literal →
   pre-triaged CONCERNS-at-most; deferred lint anchor discharged by a
   main-side follow-up note (quote it + composer-verify the round touched
   none of the offending files); removed-literal (`_ROW_RE`) probed at base
   AND HEAD with a no-orphaned-pin attest; post-round agent-memory commits
   on the branch named OUT of the pinned review range.
