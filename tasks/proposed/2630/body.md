---
title: 'workflow-fix: Step 9c selector does not map .claude/agents prose files to
  their token-pin tests'
kind: infra
tags:
- wf-fix
created_at: '2026-08-27T12:51:45Z'
has_clean_result: false
origin_prompt: 'code-reviewer prose follow-up on #2364 r1: edits to codex-code-reviewer.md
  were not gated by test_codex_code_reviewer_step09_tag_parity.py at Step 9c; pre-existing
  selector mapping gap.'
workflow: v1
---
# workflow-fix: Step 9c selector does not map .claude/agents prose files to their token-pin tests

**Provenance:** surfaced by the Claude `code-reviewer` on #2364 round 1
(2026-08-27) while auditing the round's Gate-scope line; pre-existing on
trunk (Step 0.9 subclass pre-existing-on-trunk), not introduced by #2364.

## The gap

The #2364 round edited `.claude/agents/codex-code-reviewer.md` (Blocker-tags
line) and `.claude/agents/code-reviewer.md`. Two committed tests pin exactly
those artifacts byte-identically:

- `tests/test_codex_code_reviewer_step09_tag_parity.py` (pins the codex
  Blocker-tags line + the Step 5c-bis strip-subset sentence),
- `tests/test_verdict_disagree_observer.py` (Blocker-tags vocabulary
  consumer).

Neither appeared in the implementer's 96-file local union NOR the selector's
205-file Step 9c selection: `select_step9c_tests.py` has no mapping arm from
an edited `.claude/agents/*.md` file to the prose/token-pin tests that read
it (a text-pin test neither imports nor importlib-loads the agent file, so
the import-map arm from #1299 and the importlib-literal arm proposed in
#2380 both structurally miss it). Edits to agent specs are therefore not
gated by their own pin tests at Step 9c; #2364's reviewer discharged the
NOT-RUN presumption by hand (71 passed).

## Proposed fix (implementing session refines)

Extend the selector's file→test mapping so `.claude/agents/<name>.md` (and
sibling agent-prose surfaces) map to their token-pin tests — either a
maintained explicit map (the FAMILY_agents vetted-membership list in
`.claude/skills/issue/steps/09-step-5.md` already enumerates candidates) or
a cheap content-scan arm (tests whose source contains the literal edited
path). Add a membership assert so the map cannot silently rot (the guard-20
convention in `tests/test_issue_skill_lint_family_sync.py`).

## Acceptance

- Editing `.claude/agents/codex-code-reviewer.md` alone makes the selector
  select `tests/test_codex_code_reviewer_step09_tag_parity.py`.
- The mapping is enumerable + pinned (a new agent-pin test either joins the
  map or a named exempt list; a lint/test asserts completeness).
- No regression in selector wall-time worth noting (the map is static).
