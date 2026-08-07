# Step-completed re-entry skip-ahead (`epm:step-completed`)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Every step that completes posts `epm:step-completed v1` BEFORE EXIT,
recording `step`, `next_expected_step` (looked up from
(see workflow.yaml § steps)), and an `exit_kind` (one of `clean` /
`parked` / `failure-exit`). The distinctions are:

- `clean` = normal continuation;
- `parked` = user-gated wait;
- `failure-exit` = error path.

**Authoring convention:** all-caps `EXIT` in this file's action region (everything above this section) marks a scanner-enforced action exit that must carry a `post_step_completed.py` call within ±6 lines (`tests/test_step_completed_resume.py::test_every_exit_site_posts_marker`); lowercase `exit` is reference prose or a deliberately marker-free exit (user-pause, v2 handoff).

**Helper.** Skill code calls `scripts/post_step_completed.py` at every
EXIT site (after the EXIT condition is met, before the actual exit):

```bash
uv run python scripts/post_step_completed.py \
    --issue <N> --step 5b --exit-kind clean \
    --notes "code-review PASS, advancing to pod provisioning"
```

The helper looks up `next_expected_step` from `.claude/workflow.yaml`
and appends the event row; refuses to post if the step ID is unknown to
the YAML or if `exit_kind` is not in the choices list (typo guard).

Legacy prose id `9b` is aliased to canonical `9b-same` before
validation (the marker records the canonical id), and an unknown-id
refusal prints near-miss suggestions (#1499). A nonzero exit from this
helper means NO resume record was posted — correct the step id and
re-post; ignoring the refusal silently drops the §5 record.

**Re-entry router.**
`src/explore_persona_space/orchestrate/resume.py:decide_entry_step`
implements the precedence rules:

1. `status` is `blocked` -> full replay (rule 1, BEFORE the marker is
   consulted; load-bearing — a stale clean-exit marker must NEVER let
   the skill dispatch on a manually-blocked task).
2. No `epm:step-completed` event -> full replay (first invocation or
   pre-§5 in-flight task).
3. Latest event's `exit_kind` is `parked` or `failure-exit` -> full
   replay.
4. Latest event's `next_expected_step` is unknown to
   (see workflow.yaml § steps) -> warn + full replay (graceful
   fallback for renamed / removed steps).
5. Current `status` not in target step's `entry_status_label` -> full
   replay (status drift; user manually flipped the status).
6. All checks pass -> jump to `next_expected_step`, skipping Steps 0
   through (target - 1).

**EXIT-site -> `exit_kind` mapping** (17 sites total). The implementer
wires each site to invoke `post_step_completed.py` with the right
`exit_kind`:

| EXIT site | Step | Trigger | `exit_kind` |
|---|---|---|---|
| Step 0b/2 `type` autofill loop guess | 0b | user override required | `failure-exit` |
| Step 1 user defers / no reply | 1 | user-gated | `parked` |
| Step 2c plan-pending awaiting `approve` | 2c | user-gated | `parked` |
| Step 2c "Defer"/"3" reply | 2c | user-gated | `parked` |
| Step 4b TDD gate awaiting `approve-tests` | 4b | user-gated | `parked` |
| Step 4b TDD second pass | 4b | user-gated | `parked` |
| Step 4b implementer EXIT to `running` | 4b | normal continuation | `clean` |
| Step 5b code-review FAIL revision_round>=5 | 5b | apply Step 5d cap-hit rule (strip-then-continue-or-surface) | `conditional` (all-stripped → `clean`; substantive residual → `failure-exit` autonomous / `parked` interactive) |
| Step 6c pod URLs surfaced, leave at `running` | 6c | normal continuation | `clean` |
| Step 6c pod provisioning failure | 6c | error path | `failure-exit` |
| Step 6 preflight error/warning | 6 | error path | `failure-exit` |
| Step 6d experimenter dispatched, autonomous | 6d | normal continuation | `clean` |
| Step 7 `epm:results` not found and stale | 7 | user-gated | `parked` |
| Step 7 upload-verifier FAIL | 7 | error path | `failure-exit` |
| Step 9b first entry to `awaiting_promotion` (tail of `9a-bis`) | 9a-bis | user-gated | `parked` |
| Step 10 still `classification = pending` (re-invocation) | 10 | user-gated | `parked` |
| Step 0 resume ambiguous status (folder mismatch) | 0 | error path | `failure-exit` |

**Backwards-compat.** A task that ran through Steps 0-5 BEFORE §5 landed
has no `epm:step-completed` events. On re-entry the router returns None
(rule 2) and the skill falls back to the existing full-replay path
documented below. The first `/issue <N>` re-invocation AFTER §5 lands
posts the first event; the SECOND benefits from skip-ahead. Graceful,
no migration step.

If the specialist subagent has exited but no `epm:results` event was
posted, the skill assumes the run failed silently. On resume in `running`
with no progress in >4 hours, post `epm:stale v1` event asking user to
investigate and optionally `task.py set-status <N> blocked`.

**Resume correctness per active state** (the key benefit of having
dedicated "working" statuses):

Every "re-spawn `codex-*` … only" action in this table is subject to the
#1204 pre-spawn sentinel check (CLAUDE.md § Codex ensemble review): when
LIVE, record the instant confirmed no-show instead of re-spawning the
composer.

| Status at resume | `epm:*` events present | Interpretation | Action |
|------------------|------------------------|----------------|--------|
| `planning` | no `epm:plan` | planner was cancelled | re-run adversarial-planner |
| `plan_pending` | `epm:plan` exists | awaiting user approval | show plan path, EXIT |
| `running` (implementing) | no `epm:experiment-implementation` (or `epm:results` for infra), no `epm:proposed-tests` either | implementer was cancelled | re-spawn implementer |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, no `epm:experiment-implementation`, no `epm:approve-tests` event posted **after** the `proposed-tests` event | TDD mode: tests posted, awaiting user approval | show the `proposed-tests` event timestamp + the `approve-tests` reply instruction, EXIT |
| `running` (implementing) | `epm:proposed-tests v<n>` exists, an `epm:approve-tests` event exists **after** the `proposed-tests` event, no `epm:experiment-implementation` | TDD tests approved by user | re-spawn implementer with `tdd_approved=true`; brief instructs implementer to write implementation against the approved tests, then post `epm:experiment-implementation` at the next version (max+1 per § Posting review-round markers; omit `--version` and the CLI derives it) as normal |
| `running` (implementing) | latest `epm:code-review` is FAIL, round < 5 | revision in progress | re-spawn implementer with critique |
| `running` (implementing) | latest `epm:code-review` is FAIL, round >= 5 | cap reached | apply Step 5d cap-hit rule (strip-then-continue-or-surface): strip → all-stripped PASS+continue OR surface substantive residual (autonomous: `blocked` + notify; interactive: present to user) |
| `running` (code-reviewing) | neither `epm:code-review` nor `epm:code-review-codex` for the current implementation version | both ensemble reviewers were cancelled | re-spawn both code-reviewer + codex-code-reviewer in parallel |
| `running` (code-reviewing) | `epm:code-review v<n>` exists, no `epm:code-review-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-code-reviewer` only |
| `running` (code-reviewing) | `epm:code-review-codex v<n>` exists, no `epm:code-review v<n>` | Claude reviewer not yet returned | re-spawn `code-reviewer` only |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts disagree (PASS-class vs FAIL), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `code-reviewer` | reconciler not yet started | spawn reconciler |
| `running` (code-reviewing) | both `epm:code-review v<n>` and `epm:code-review-codex v<n>` exist, verdicts agree | ensemble decision ready | apply Step 5c rule and advance |
| `running` (code-reviewing) | `epm:code-review-codex` is `epm:failure` (codex-output-malformed or infra) | Codex twin no-show | proceed with Claude-only decision per Step 5d fallback |
| `running` (workload) | no `epm:results` for > 4h | experimenter crashed silently | post `epm:stale`, ask user |
| `running` (workload) | latest event is `epm:failure` with bounce-back proposal | experimenter bounced to implementer | status back to `running` (implementing), re-spawn experiment-implementer |
| `uploading` | no `epm:upload-verification` PASS | verifier not run or failed | re-run upload-verifier |
| `interpreting` | no `epm:interpretation` | analyzer not started | spawn analyzer |
| `interpreting` | `epm:interpretation` exists, neither `epm:interp-critique` nor `epm:interp-critique-codex` for the current version | both ensemble critics not started | spawn `interpretation-critic` + `codex-interpretation-critic` in parallel |
| `interpreting` | `epm:interp-critique v<n>` exists, no `epm:interp-critique-codex v<n>` | Codex twin not yet returned | re-spawn `codex-interpretation-critic` only |
| `interpreting` | `epm:interp-critique-codex v<n>` exists, no `epm:interp-critique v<n>` | Claude critic not yet returned | re-spawn `interpretation-critic` only |
| `interpreting` | both `epm:interp-critique v<n>` and `epm:interp-critique-codex v<n>` exist, verdicts disagree (PASS vs REVISE), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `interpretation-critic` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | both ensemble events exist, verdicts agree OR role-matching reconcile event present (`**Role under adjudication:** interpretation-critic`), ensemble verdict REVISE, round < 5 | revision needed | re-spawn analyzer with all critique events (trigger-dense: by reference per § File-only Codex verdict posting) |
| `interpreting` | ensemble verdict PASS or (round >= 5 AND the Step 9a cap-hit resolved: all residual stripped, no substantive overclaim residual), neither `epm:clean-result-critique` nor `epm:clean-result-critique-codex` | content honesty settled, structure + register loop not started | promote body in place if missing, then spawn `clean-result-critic` + `codex-clean-result-critic` in parallel. (round >= 5 with a SUBSTANTIVE overclaim residual → apply Step 9a surface-real-residual rule instead: interactive present to user; autonomous `epm:failure v1 failure_class: code` + `blocked` + notify) |
| `interpreting` | `epm:clean-result-critique v<n>` exists, no `epm:clean-result-critique-codex v<n>` | Codex twin not yet returned (or wrapper crashed) | re-spawn `codex-clean-result-critic` only |
| `interpreting` | `epm:clean-result-critique-codex v<n>` exists, no `epm:clean-result-critique v<n>` | Claude critic not yet returned | re-spawn `clean-result-critic` only |
| `interpreting` | both `epm:clean-result-critique v<n>` and `epm:clean-result-critique-codex v<n>` exist, verdicts disagree (PASS-class vs REVISE), no `epm:review-reconcile v<n>` whose body's `**Role under adjudication:**` is `clean-result-critic` | reconciler not yet started | spawn `reconciler` (marker mode) |
| `interpreting` | clean-result ensemble verdict REVISE (agreed, unioned, or reconciled by a role-matching `epm:review-reconcile`; after the Step 9a-bis procedural-only strip), round < 5 | structure / register revision in progress | re-spawn analyzer with both clean-result critiques (trigger-dense: by reference per § File-only Codex verdict posting) |
| `interpreting` | clean-result ensemble verdict PASS-class or (round >= 5 AND the Step 9a-bis cap-hit resolved: all residual stripped, no substantive overclaim residual) | ready for review | advance to `reviewing`. (round >= 5 with a SUBSTANTIVE overclaim residual → apply Step 9a-bis surface-real-residual rule: interactive present to user; autonomous `epm:failure v1 failure_class: code` + `blocked` + notify) |
| `reviewing` | (no agent dispatch; transitional single-step) | transitional pass-through (see the status-conventions table) | move to `awaiting_promotion`, run the Step 10d auto-merge procedure, post `epm:status-changed`, EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, no `epm:merged` and PR unmerged | waiting for user to promote; worktree not yet merged | run the Step 10d auto-merge procedure (idempotent backstop — covers the case where the Step 9b auto-merge was interrupted), then show task path, prompt to promote via `task.py promote`, run CRON-TEARDOWN (self-heal sweep, § CRON-TEARDOWN procedure, stray one-shot `/issue <N>` wakeups included), EXIT |
| `awaiting_promotion` | `classification == 'pending'` in body frontmatter, `epm:merged` present | waiting for user to promote; worktree already merged | show task path, prompt to promote via `task.py promote`, run CRON-TEARDOWN (self-heal sweep, § CRON-TEARDOWN procedure, stray one-shot `/issue <N>` wakeups included), EXIT |
| `awaiting_promotion` | `classification != 'pending'` (user ran `task.py promote`) | user promoted | advance to Step 10 (auto-complete) |
| `completed` | `epm:done` present, no `epm:merged` (any form — `artifact_confirmed` counts), `issue-<N>` PR/branch unmerged | Step 10d merge was interrupted after completion (the #1540 stranded-merge class; the watcher's `completed_unmerged_pass` flag names this resume path as the recovery, #1564) | run the Step 10d auto-merge procedure (idempotent backstop — same guards as the `awaiting_promotion` row above; its success path posts `epm:merged`, resolving the watcher's flag episode), run CRON-TEARDOWN (self-heal sweep, § CRON-TEARDOWN procedure, stray one-shot `/issue <N>` wakeups included), EXIT |
| `interpreting` / `reviewing` / `awaiting_promotion` / `completed` | unrun `epm:followup-scope` (≥1 UNRUN `followup_label` per `task_workflow.unrun_followup_labels`: entries grouped by label, within-label latest-(`ts`,`version`) authoritative, a label unrun iff no matching `epm:same-issue-followup-run v1`) | a `question_relation: same` follow-up is scoped to run ON this issue (takes precedence over the status rows above — see Step 0 "Same-issue follow-up dispatch") | route into the same-issue follow-up loop (Step 9b § Same-issue follow-up loop): dispatch ONE label per entry — the queue head (user-initiated first, then oldest armed ts); set status to `followups_running` + tag `followup-auto`\|`followup-manual` and run the abbreviated cycle. The loop re-reads the round's label-scoped scope at the planner snapshot (Step 9b § Same-issue follow-up loop step 3) so a crashed-mid-round resume picks up any correction posted since the round started |
| `followups_running` | unrun `epm:followup-scope` (≥1 UNRUN `followup_label` per `task_workflow.unrun_followup_labels`: entries grouped by label, within-label latest-(`ts`,`version`) authoritative, a label unrun iff no matching `epm:same-issue-followup-run v1`) | a same-issue follow-up round is mid-flight (this row takes precedence over the two children-based rows below) | resume the same-issue follow-up loop at the phase the stage breadcrumbs (`stage=followup-<phase>`) + latest markers indicate — do NOT restart from the top; `task_workflow.executing_followup_label` resolves WHICH label the round is executing (labeled breadcrumb first, dispatchable queue head fallback), and the planner-snapshot re-read (Step 9b § Same-issue follow-up loop step 3) picks up that label's latest scope so a correction posted mid-round is honored on resume |
| `followups_running` | no unrun followup-scope; at least one open child task (`parent_id: <N>` in `body.md` frontmatter) not in `completed` / `archived` | legacy semantics: children still in flight | show child-task table, EXIT |
| `followups_running` | no unrun followup-scope; every child has reached `completed` / `archived` (or no children remain) | children all done | re-run Step 10: relabel parent to `completed` |
| `running` (workload) | pod alive + log advancing (`ssh epm-issue-<N> tail -1 <log_abs>`), no live bg-Bash poll for this session, latest `epm:*` marker is stale (no `epm:progress` in > ~15 min) | Step 6d.2 bg-Bash poll chain died — typically because a reaction turn emitted a corrupted/truncated tool-call (rendered as raw text), the harness had no bg work to wake on, AND the auto-armed backstop cron also died (a `durable=False` cron does not survive the session that registered it, so this row is reached mainly after a session restart / fresh recovery session). Pod and run are HEALTHY; only the session's monitor died. (Origin: #462/#463.) | Re-enter the polling loop by re-invoking `/issue <N>` once; it reads the latest `epm:run-launched` (`pod`, `pid`, `log_abs`), resumes Step 6d.2, and the Step 6d.2 step-1 guard AUTO-RE-ARMS the backstop cron (`CronList` for `prompt.strip() == "/issue-tick <N>"`, `CronCreate` if absent) so the next dead turn won't strand the run again — no user `/loop` typing needed. The lightweight `/issue-tick <N>` tick is what the cron fires; the full `/issue <N>` skill loads only on cold start, cold respawn, or the tick's stale-marker recovery branch. Do NOT re-spawn `pod_watch.py` / `pod.py watch` — that mechanism is retired per "Notes on the obsolete monitoring stack". |

**Reconcile predicates are role-scoped.** There is exactly ONE marker-mode
reconcile kind (`epm:review-reconcile` — workflow.yaml § markers); the
adjudicated role lives in the verdict body's `**Role under adjudication:**`
field, not the marker name. Wherever a row above tests for (or reads the
verdict of) a reconcile event, only an event whose role field matches that
stage's critic (`code-reviewer` / `interpretation-critic` /
`clean-result-critic`) counts. Both the interpretation and clean-result
ensembles sit at status `interpreting` with the same round numbering, so an
unqualified "no `epm:review-reconcile v<n>`" predicate would let an
interp-stage reconcile falsely satisfy the clean-result disagreement row
(skipping the reconciler) or feed the wrong stage's verdict — and vice
versa.

Without distinct statuses for `uploading` / `interpreting` / `reviewing` /
`awaiting_promotion`, many of these rows would be indistinguishable.
That's why the state machine has them.

---
