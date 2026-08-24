---
name: infra-task compose gate adaptation
description: For kind:infra/batch/analysis composes, convert the type:experiment-only gates (0.55/0.6/0.65/0.67-exposure) into explicit "N/A — do not raise these blockers" instructions; keep the any-diff-type sub-checks (hollow gate, Hub scoping, 3.8) in full; ban executing fleet-mutating scripts by name
type: feedback
---

For a `kind: infra` (or batch/analysis/survey) code-change compose, do NOT
inline the `type:experiment`-only gates (Step 0.55 smoke-architecture, 0.6
end-to-end smoke, 0.65 raw-completions upload, 0.67 DP-exposure) in full.
Instead write one explicit block: "N/A for this task kind — do NOT raise
`marker-shape` / `smoke-run-missing` / `raw-completions-upload-missing` /
`compute-shape-mismatch` on their absence; record one-line N/A conclusions."

**Why:** inlining the full gate text on an infra task invites the false-FAIL
class the composer spec exists to prevent (Codex FAILing on a missing
`epm:smoke-architecture-check` / `## Smoke run` that the task type never
requires); omitting them silently invites Codex re-deriving them from the
implementer's smoke prose. The explicit N/A conversion closes both. Step 0.5
STILL applies in full (`epm:results` carries the same (a)-(d) four-section
contract on code-change paths).

**How to apply:**
- Keep IN FULL the sub-checks that fire for any diff type: Step 0.68
  hollow-verification-gate (map it to the diff's smoke/CLI flags, e.g. a
  `--<pass>-only` flag must dispatch the REAL pass fn), Hub-call-scoping,
  Step 3.8 seam-stub body verification (monkeypatch-heavy infra test suites
  make this the highest-yield lens), Steps 3.5/3.6/3.7/4.5, grep-the-literal.
- Step 4 stays read-only, but for infra tasks ADD a wiring-substantiveness
  read: "for each hook/gate site, would DELETING the hook fail a committed
  test?" — this is usually a named review criterion in infra briefs.
- Extend the never-execute instruction to name the diff's FLEET-MUTATING
  scripts explicitly (e.g. `autonomous_session_watch.py` stops sessions,
  writes ~/.eps-autonomous state, pushes Telegram — even its "smoke-only"
  CLI flag mutates state). The generic "never run smoke commands" line does
  not obviously cover a watcher/janitor script. (Applied on #1027 r1.)
- When the PLAN itself mandates a smoke battery on an infra task (a §6
  verification battery with named phases), route its review through Step
  0.5(c) verifiability + plan-§6 ADHERENCE (`substantive` / CONCERNS), and
  say so explicitly next to the Step 0.6 N/A line — the brief may ask for
  "smoke-run completeness" review, and without the routing note Codex's
  only vocabulary for a missing phase is the banned `smoke-run-missing`
  tag. Also inline the composer-fetched empty concerns ledger result
  (`[]`) verbatim so Codex never tries branch-guarded `task.py` from the
  sandbox. (Applied on #1040 r1.)
- For a DOC-ONLY workflow-surface diff (agent .md / rules .md edits):
  frame the "consumers" for Step 3's reachability read as (a) the agent
  sessions loading the file and (b) any COMPOSER that verbatim-inlines the
  edited span into another prompt (a wording bug then propagates into two
  reviewer families); point the Step 3.7 sibling sweep at the SIBLING
  AGENT FILES carrying the same new rule (same-rule-two-agents divergence
  is the highest-yield sweep target); and pre-declare any plan-authorized
  lint cap-raise as NOT scope creep while directing a hollow-bound check
  (new cap vs actual file size + headroom arithmetic). Explicitly instruct
  the throughput/3.5/3.6/3.8 lenses to RECORD their N/A conclusions rather
  than silently skip. (Applied on #1056 r1.)
- ROUND-1 template reuse from a SIBLING task: when a prior task's r1
  template matches the diff shape (e.g. #1056 = doc-only workflow-surface +
  lint cap bump, reused for #1081 r1), rebuild from it instead of paging
  code-reviewer.md back in — but run the stale-donor literal guard
  (`assert donor_id/donor_sha/donor_incident not in TEMPLATE`) on the
  TEMPLATE only, never the final prompt: the inlined marker body can
  legitimately cite the donor task as precedent (#1081's marker cited the
  #1056 cap-bump style and tripped a prompt-scoped guard). (Applied on
  #1081 r1.)
- PRE-DECLARE the brief's named non-defects verbatim inside the Step 0.6
  N/A block — e.g. the sparse-worktree pytest invocation via MAIN's venv
  python + `PYTHONPATH=<worktree>/src` (worktree-correct: the editable
  install resolves the package to main's src, so bare `uv run pytest`
  would import main's modules). Without the pre-declaration Codex's only
  read of an unusual test command is "suspicious smoke digest" and it
  escalates a correct invocation. Same slot also carries the routing for
  a marker `## Smoke run` H2 whose phases are pytest/ruff runs: that IS
  the correct infra evidence shape, review claims against those digests,
  never `smoke-run-missing`. (Applied on #1084 r1: code-fix donor #996
  reused — sibling reuse works for infra CODE fixes, not just doc-only.)
- MERGED-MAIN branch topology: when the issue branch merged main in (a
  `Merge branch 'main'` commit + alignment commit below the edit commit),
  (i) `main...HEAD` legitimately carries FOREIGN `tasks/` paths — the
  composer PRE-VERIFIES each foreign row's content exists on current main
  (`git show main:<current-status path> | grep -cF <row>`; main may have
  RENAMED the folder if the foreign task advanced status) and inlines the
  ground truth + the exact probes, framing the residual as a path-rename
  merge note, NOT branch-novel collateral — otherwise Codex's collateral
  focus escalates main's own content; (ii) the worktree's `tasks/` tree
  may be CURRENT rather than branch-cut-frozen (the merge postdates the
  status transition), so the plan resolves at the CURRENT-status worktree
  path — run the Step 2-pre-b identity diff as usual and say so in the
  prompt; (iii) scope review to `git show <edit-sha>` and name the merge
  commit under Step 0.9 subclass 3 (merge-carried hunks are never round
  findings). (Applied on #1087 r1.) SAME recipe applies with NO merge
  commit when the branch base was fast-forwarded onto main commits that
  shared-root rebasing later REWROTE (merge-base falls before them, so
  main...HEAD shows the foreign task-state as HEAD-side): pre-verify each
  foreign task's content/status folder on current main and declare every
  `tasks/` path out of scope; round contract = the single edit SHA.
  (Applied on #1088 r1: 48 foreign tasks/ paths — #833/#1089 status
  renames, #1074 markers — around one 2-file edit commit.)
- NEVER trust the brief's plan-absence claim — run Step 2-pre-b yourself:
  #1103 r1's brief said "worktree plan absent by design (repo-root-planned)"
  but the worktree had been fast-forwarded to main AFTER the task reached
  `running`, so `tasks/running/<N>/plans/plan.md` resolved AND diff'd
  identical to canonical → reference BY PATH (saves ~29 KB of prompt) and
  patch the BLOCKED paragraph to the by-path variant (assert plan-envelope
  count == 0). Also pre-declare any MID-PHRASE line wrap in an applied doc
  literal (e.g. planner.md wrapping `standardized\npersona-vectors-shape`)
  and tell Codex to grep the two fragments separately — a whole-phrase
  single-line grep miss on a width wrap is a pre-declared non-finding, and
  a disclosed rewrap deviation gets a content-identity adjudication
  instruction, not an auto-flag. (Applied on #1103 r1.)
- IMPLEMENTER-RAISED same-round concern row (a plan-deferral the
  implementer itself persisted at round 1, e.g. #1110's
  `set-title-h1-sync-mutator-deferred` for a plan-D3 deferred mutator
  leg): inline the row verbatim AND frame it explicitly — it is the
  implementer's OWN durable deferral record (deferred-production-path
  discipline), NOT a prior blocker to verify-closed; a marker with NO
  (e) section is CONFORMING (nothing was open BEFORE the round); Codex's
  duties = (a) adjudicate the deferral as PLAN-AUTHORIZED (don't demand
  the deferred leg, don't flag the deferral as drift), (b) verify the
  deferral's scope holds in the diff (the deferred file ABSENT —
  presence would be the plan deviation), (c) write `already persisted:
  <id>` in Concerns-to-persist instead of re-raising a duplicate.
  (Applied on #1110 r1.)
- APPROVED-RESIDUAL framing (a plan carrying an accept-with-named-follow-up
  residual on a safety surface, e.g. #2145's §4.4.1 temporal wrong-kill):
  state BOTH halves explicitly in the prompt — "re-raising the residual is
  NOT a finding; checking the code matches what the residual CLAIMS is" —
  and add the escalation arm: realized behavior WORSE than the residual
  describes (widened match set, stamp naming a never-minted name) is a
  Critical. Without the second half Codex either re-litigates the approved
  disposition or waves the whole hunk through as "known". (Applied #2145 r1.)
- ORCHESTRATOR-ROUTED SCOPE QUESTION (the brief carries an open scope
  question, e.g. a #2236-class dangling-pointer left out of plan scope):
  give it its own mandatory verdict section with an EXPLICIT unhedged
  either/or ruling vocabulary (`BLOCKER — must land this round` vs
  `CORRECTLY DEFERRED`), require grounding in the plan's scope text + the
  marker's (d) disposition + the actual file contents, and wire the DEFER
  branch to "Concerns to persist" (suggest the concern-id) so the deferral
  is durable, not prose-only. Compose-time: pre-verify the cited paths
  resolve in the worktree and pre-declare any mid-phrase line wrap in the
  pinned prose (the #1103 fragment-grep discipline). (Applied #2145 r1.)
- MARKER-KIND PROBE on infra tasks (#2146 r3): the spec says code-change
  paths post `epm:results`, but a real infra pipeline can post
  `epm:experiment-implementation` throughout — when the expected `--prefix`
  fetch returns EMPTY, probe `jq -r '.kind' events.jsonl | sort | uniq -c`
  and fetch by the kind actually present; then PRE-DECLARE in the prompt
  that `epm:results`-specific fields (Gate-scope check line) are not owed
  on this marker kind, so Codex cannot false-FAIL on their absence.
- DOC diff whose text CITES src-code facts (a rules-file edit asserting
  function signatures / cache-key behavior / hardcoded values): when the
  branch base == main's tip at edit time (fast-forwarded, single edit
  commit), state that compose-time fact in the prompt — "your worktree's
  `src/` IS main's src" — and direct factual-fidelity verification at the
  actual src files with the plan's pinned file:LINE anchors. Pin sibling
  INCIDENT numbers to the plan's §8 assumptions table (clarifier-verified)
  instead of the incident task's events — `tasks/*/<M>/` for another task
  is frozen/unreliable in the sandbox and a missing path must be declared
  a non-finding. Also pre-declare file-convention non-defects the rule
  file carries (e.g. llm-judging.md's global-number-with-topical-placement
  means "rule 23 in §C after rule 10" is correct, not a numbering bug).
  (Applied on #1096 r1: rule-23 max_tokens sizing, count arithmetic
  23=21+2 across 4 surfaces.)
