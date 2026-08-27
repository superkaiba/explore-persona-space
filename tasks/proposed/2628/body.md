---
title: Autonomous sessions must paste raw concern-id slugs into clean-result prose
  — Lens 14 vs the no-opaque-codes rule
kind: infra
tags: []
created_at: '2026-08-27T10:48:40Z'
has_clean_result: false
parent_id: 2569
origin_prompt: 'Surfaced during /issue 2569 clean-result gating: 9 open non-BLOCKER
  concerns could not be deferred (defer-concern is user-only, and no user consent
  existed), so Lens 14 forced their literal kebab-case ids into reader-facing prose,
  colliding with the no-opaque-condition-codes rule.'
workflow: v1
---
# Autonomous sessions must paste raw concern-id slugs into clean-result prose — Lens 14 vs the no-opaque-codes rule

## Goal

Reconcile two workflow rules that currently contradict each other for an
autonomous session parking a clean-result with open non-BLOCKER concerns, and
write down the sanctioned acknowledgment shape so sessions stop deriving it.

## The contradiction

`verify_task_body.py` check "concerns audit (Lens 14)" accepts exactly two
acknowledgment mechanisms for an open binding concern:

1. the LITERAL kebab-case `concern_id` appearing in the `## Takeaways` bullets or
   inside a `## Results` `### <result>` body (the check is a plain
   `cid in tldr_h3_text` substring test); or
2. a `<!-- concern-deferred: <id> -->` HTML comment — which the same check
   REJECTS as "fabricated" unless a matching `deferred` event exists in
   `concerns.jsonl`.

The only writer of that event is `task.py defer-concern`, which is **USER-ONLY**:
the CLI refuses without `--by user` (or `--by reconciler` for ensemble
severity-downgrades), and the library rejects again defense-in-depth.

So in an autonomous session (`EPM_AUTONOMOUS_SESSION=1`) with no user in the
loop, mechanism 2 is unavailable — correctly, since using it would manufacture
consent. Mechanism 1 is therefore mandatory, and it forces raw slugs like
`dwfleet-align-units-key-omits-adapter-and-ft-sidecar-content` into the
clean-result's reader-facing prose.

That collides head-on with two standing rules:

- **No opaque condition codes** in clean-result prose (`feedback_no_opaque_condition_codes`
  memory; enforced in spirit by `verify_task_body.py`'s own opaque-config-code
  check on figure text). A 60-character hyphenated internal slug is exactly the
  class of token that rule exists to keep out of a write-up.
- **Dense results get a plain-language reading** (CLAUDE.md): every term gets its
  meaning at first mention. A bare concern id has no plain reading at all.

## What #2569 hit and what it did

`/issue 2569` parked with 13 open binding CONCERN-severity concerns. Four were
genuinely science-affecting and were folded into the relevant Results prose as
stated caveats, then recorded with `task.py address-concern` — that path is
clean and needs no user.

The other nine were caching/provenance-pin defects that change no reported
number. The session first wrote `<!-- concern-deferred: ... -->` markers for
them; the verifier correctly refused all nine as fabricated. The session
declined to pass `--by user` (that would fabricate consent it had not received)
and instead:

- wrote ONE plain-English paragraph in `## Results` describing the nine in
  reader-legible terms (what class of defect, what it does and does not affect),
  explicitly framed as OPEN and unaddressed rather than deferred;
- put the nine literal ids in a collapsed `<details>` block inside the same
  section, which satisfies the substring test because `<details>` bodies are
  part of the section text — while keeping the slug soup out of the rendered
  reading flow;
- discovered that the `<details>` block then needed its own cherry-picked /
  random-sample disclosure in the prelude prose (the sample-block checks treat
  any `<details>` as a sample block), so the prelude had to state that the block
  is the complete set of nine, not a sample.

That combination PASSES the gate and reads acceptably, but every step of it was
derived live against verifier feedback. It is not written down anywhere.

## Acceptance criteria

1. The sanctioned autonomous acknowledgment shape is documented where a session
   composing a clean-result will find it — `.claude/skills/clean-results/SPEC.md`
   § the concerns/Lens-14 area, and/or `.claude/agents/analyzer.md`. Include the
   `<details>` + complete-set-disclosure detail, since the sample-block checks
   fire on it.
2. State explicitly that an autonomous session MUST NOT pass `--by user` to
   `defer-concern`, and that fabricating a deferral marker without the ledger
   event is a verifier FAIL by design (not a bug to route around).
3. Reconcile the no-opaque-codes rule with the literal-id requirement in one
   place, so the two are not read as contradictory: name the `<details>`
   (or equivalent) carve-out as the resolution.
4. Consider whether Lens 14 should accept a THIRD mechanism for the autonomous
   case — e.g. an `epm:concern-acknowledged` marker keyed to the id, letting the
   body carry only plain English while the ledger carries the machine link.
   Record the decision either way; a deliberate "no, keep the id in the body" is
   acceptable, leaving it undecided is not.
5. Whatever ships is pinned by a test.

## Provenance

Surfaced by `/issue 2569` (2026-08-27) while gating its clean-result body; the
session's own `[epm-inline-fallback]` marker on #2569 records the same episode
from the review-independence angle. Related: `#1089` (stale deferral markers),
`#1891`, `workflow.yaml § concerns_protocol`.
