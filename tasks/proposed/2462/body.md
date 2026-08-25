---
title: codex_task.py dispatch grants WRITE to Codex reviewers declared read-only (canonical
  snippet omits --no-write)
kind: infra
tags: []
created_at: '2026-08-22T03:05:49Z'
has_clean_result: false
parent_id: 2270
origin_prompt: 'Surfaced during /issue 2270 Phase 2: three codex-critic jobs dispatched
  via the canonical codex-ensemble-review.md snippet recorded write=True in their
  epm:codex-task-spawned markers, while all three composer dispatch configs declared
  ''Codex write mode: false (read-only critic)''. codex_task.py defaults --write to
  True; the canonical snippet passes neither flag, so all five doubled review sites
  dispatch write-capable reviewers fleet-wide.'
workflow: v1
---
# Canonical Codex-critic dispatch grants WRITE access to reviewers declared read-only

## Goal

Make the canonical `codex_task.py` dispatch invocation — the one every doubled
review site copies — pass `--no-write`, so Codex REVIEWERS run read-only as
their own specs already declare, instead of inheriting `codex_task.py`'s
write-granting default.

## The gap

Three surfaces disagree about whether a Codex reviewer can write to the repo:

1. **`scripts/codex_task.py` defaults to GRANTING write.** The mutually
   exclusive group is `--write` (`default=None`) / `--no-write`
   (`action="store_false"`), and the resolution comment states the default is
   True unless `--no-write` was passed. A dispatch that omits both flags gets
   write access, and the spawn marker records `write=True`.

2. **The canonical dispatch snippet omits `--no-write`.** The invocation in
   `.claude/rules/codex-ensemble-review.md` (mirrored in CLAUDE.md
   § "Codex ensemble review") is:

       uv run python scripts/codex_task.py --issue <N> --effort <high|xhigh> \
         --prompt-file <prompt> --output-file <output>

   No write flag. Every orchestrator that copies it — i.e. all five doubled
   review sites — dispatches a write-capable Codex.

3. **The `codex-*` composer specs declare the opposite.** Each composer's
   returned dispatch config carries a `Codex write mode: false (read-only
   critic)` line. The composers cannot enforce it: they are prompt-composers
   and never dispatch (#533), so the flag is the ORCHESTRATOR's to pass — and
   the canonical snippet does not tell it to.

Net effect: adversarial reviewers whose specs say read-only run with write
access to the working tree, fleet-wide, at `critic`, `code-reviewer`,
`interpretation-critic`, `clean-result-critic`, and `follow-up-critic`.

## Why this matters (two distinct harms, neither hypothetical-only)

- **Review independence.** A reviewer that can modify the artifact it is
  reviewing can, in principle, "helpfully" apply a fix instead of reporting a
  finding — contaminating the very diff/plan under review and converting an
  adversarial gate into a co-author.
- **Shared-root exposure — the fleet-wide one.** An unexpected write at the
  repo root leaves tracked files modified-but-uncommitted, which is exactly
  the `.claude/rules/repo-root-uncommitted-state.md` (#2015) armer class: any
  concurrent session's pre-commit stash cycle can then transiently revert or
  permanently lose it. A review agent nobody expects to write is the worst
  kind of armer, because no session owns committing its output.

## Provenance

Surfaced during #2270's Phase 2 (2026-08-22). Three `codex-critic` jobs were
dispatched with the canonical snippet for a review of DESTRUCTIVE pod-teardown
code; all three spawn markers recorded `write=True` while all three composer
configs declared `Codex write mode: false (read-only critic)`. The orchestrator
noticed only after dispatch. It deliberately did NOT kill the wrappers:
`codex_task.py`'s own docs state a killed wrapper leaves the Codex job RUNNING
(recovery is `--reattach`), so killing would have orphaned a write-capable job
with no supervision — strictly worse. Instead it captured baseline md5s of the
review targets and verified post-run that nothing was written.

## Suggested shape (not a mandate — the spawned session designs it)

Likely the smallest correct fix is to add `--no-write` to the canonical
snippet in `.claude/rules/codex-ensemble-review.md` + the CLAUDE.md summary,
and to consider whether `codex_task.py`'s DEFAULT should invert for the
reviewer path specifically (a write-granting default is reasonable for a
`codex:rescue`-style implementer dispatch and wrong for a critic). Options
worth weighing: flip the global default to read-only and make writers opt in;
keep the default and fix only the documented snippet; or add a
`--role critic|implementer` that selects the posture so the two use cases stop
sharing one default.

Constraints any fix should hold: do not break `codex:codex-rescue` or any
deliberate write-mode dispatch; keep the spawn marker's `write=` field
recording the REALIZED posture so an audit can tell what a past job had; and
if the default inverts, sweep existing callers for ones that legitimately need
write.

Worth checking as part of the same round: whether any Codex job dispatched
under the current default has in fact written to a repo working tree
(a `write=True` grep over historical `epm:codex-task-spawned` markers bounds
the exposure).
