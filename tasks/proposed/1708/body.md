---
title: 'daily-held: deferred sweep backlog 2026-07-25'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-26T07:07:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 3): The 2026-07-25 sweep mined
  28 transcripts and surfaced about 117 distinct problems, four times a normal night,
  of which only the highest-severity and most-recurrent were filed; the remainder
  are real and fix-derivable but were held back to protect the five-session infra
  concurrency cap.'
workflow: v1
---
## Why this needs you

Filed by the `/daily` 2026-07-25 problem sweep as a **route-3 tracked item** so nothing
from an unusually large sweep is silently dropped.

The sweep mined 28 in-scope transcripts across 21 autonomous `/issue` sessions and 7
interactive ones and surfaced **~117 distinct problems** — roughly 4× a normal night
(the #1667–#1689 wave merged 19 tasks in one day, and #1689 alone burned 8
implementer/code-review rounds). 14 route-2 workflow-fix tasks and 4 route-3
judgment-call tasks were filed. This task holds the **remainder**: real, fix-derivable
items deliberately NOT filed tonight, so the ~5-session infra concurrency cap is not
flooded by a single night's sweep.

**Your decision:** drain this list at your own pace (`task.py new` per item, or point a
future `/daily` at it), or tell me to file them in a later batch. Nothing here is
urgent; several are cosmetic. They are recorded so they do not evaporate.

## Deferred items, grouped by target

### `CLAUDE.md` — chat/writeup register
- **Low-cosine similarity twin.** The #1310 "similarity-statistic semantics" rule
  governs direction-aware vs spectrum-invariant reads; it has no clause for the
  *negative* direction. A near-zero operator cosine (raw −0.013, aligned 0.068) was
  narrated as "essentially orthogonal / degenerate / uninformative", then refuted ~2 h
  later by the same session's own ladder run (a single context mean shift took direct
  transfer from −1982 to −0.06; context reparameterization alone reached 93–95% of
  ceiling). Proposed: a low cosine may not be narrated as "unrelated" without first
  checking whether an affine correction recovers transfer. *(`dffde9b6` @ 22:10:29Z →
  23:24:09Z / 00:22:53Z.)*
- **Gloss on FIRST use, not on challenge.** "Guarded rerun" was used across two full
  writeups before you asked what it meant. Existing item (3) says every compressed
  claim gets a gloss; sharpen it to bind on first use. *(`63122023` @ 18:11:19Z —
  your message: *"what do you mean by guarded rerun"*.)*
- **Opaque condition codes reach chat.** `M_instruct_user_chat`, `S_instruct_chat`,
  `Track-M`, `r1~r2`, `b2i`/`i2b` appeared in 4 assistant text blocks (11 occurrences)
  in `63122023`; the no-opaque-codes rule is currently scoped to plans and clean-result
  bodies. Extend to chat writeups and interim tables.
- **Plots-over-tables when a figure already exists.** A 7-row cross-condition markdown
  table led the final #825 summary with an existing figure merely linked below.
- **Detach-lifecycle recurrence.** You asked twice in one session — *"will this continue
  running if i shut down the terminal (in tmux?)"* (18:36:57Z) and *"so i can close this
  terminal now?"* (19:08:13Z) — after Phase 0 dispatched at 18:17 and #1689 spawned at
  18:55 with no lifecycle statement. This is a recurrence of the SOUL.md
  detach-transparency rule written 2026-07-24. Proposed: make the line a mandatory
  field of the inline-round dispatch ack and the `epm:progress` dispatch note.
- **Structural/ordering claims need the same compose-time check as numeric ones.** Two
  bullets committed in `docs/mapping_similarity_metrics.md` were wrong (a "predicted
  through `W_s`" claim, and a nesting claim ordering two model families as strong/weak
  when they are siblings). Committed `13f00f85a1` 07-24 19:57 PT, corrected
  `67f8e798db` 07-25 10:40 PT — ~15 h live. *(`51e59f07` @ 17:38:43Z.)*
- **Methodology-doc gloss debt.** Four consecutive clarification re-asks from you on the
  same two sections of that note in a 14-minute span; the assistant's own verdict was
  *"those two sections were the muddiest part of the note"*. Extend the plain-English
  gloss duty from chat writeups to committed `docs/*.md` methodology notes.

### `CLAUDE.md` § Orchestrator vs subagent / `.claude/rules/gotchas.md` — process hygiene
- **Self-matching `pkill`/`pgrep`.** A compound `pkill -f "issue1345_ladder_rung[s]"; …;
  nohup <relaunch>` killed its own shell before the relaunch ran (exit 144, nothing
  restarted); separately a watcher loop's `pgrep` matched its own command line and would
  have spun forever (reaped with SIGKILL). Extend the bracket-a-character rule: (a) kill
  and relaunch never share one Bash invocation, (b) a watcher's own command line must
  not contain the literal it greps for.
- **Backgrounded `sleep` is not elapsed time.** Several "wait for X" polls were
  dispatched `run_in_background=true` and read immediately, so ~2 min of believed
  progress never happened (`etimes=122` disproved it). Job age must be read from
  `ps -o etimes=` / `/proc/<pid>/stat`.
- **Stale monitors outlive their jobs.** 6 separate triage turns across two sessions
  spent dismissing monitors armed on killed runs; two "wait for X to finish" loops
  survived 3.5 h past their work and were killed by the harness. Proposed: killing a
  monitored process cancels its Monitor in the same step, and a session-wrap duty reaps
  own background wait-loops.
- **Foreground-vs-background `sleep` boundary.** A foreground `sleep 60 && cat` was
  hard-blocked while background `sleep 540 && backend_poll.py` chains ran fine 5× in the
  same session — state the distinction rather than rediscovering it by tripping it.
- **`py-spy dump` needs sudo on this VM** (`Permission Denied … elevated permissions`);
  document the `sudo env "PATH=$PATH"` form and the `/proc/<pid>/stat` fallback in the
  hang-diagnosis rule.
- **`spec_from_file_location` + `@dataclass`** raises `AttributeError` unless the module
  is registered in `sys.modules` before `exec_module` — hit while probing a `scripts/`
  module's internals.

### `.claude/rules/code-style.md` / `.claude/skills/paper-plots/SKILL.md` — new-script authoring
- **Heavy-import-before-`load_dotenv()`** shipped into **4 newly authored scripts**
  across two sessions, caught only by the commit-time gate (`#847` assertion). Proposed:
  an explicit new-VM-entrypoint import-order template.
- **Bare `hf_hub_download`** bypassing `hub.retry_transient` (the #1547
  shared-commit-budget rule) in a new script; the rule currently reads as upload-centric.
- **`paper_plots` entrypoint guessed wrong** (`apply_paper_style` → the real
  `set_paper_style`), then the auto-formatter dropped the corrected import → `NameError`.
  Put the exact import boilerplate at the top of the skill.
- **Symlog default:** an all-arms figure rendered unreadable (prefix panels flattened by
  a −17,500 outlier); caught only by the mandatory PNG read. Propose a symlog default
  when a series spans >~3 orders of magnitude.
- **Edit-after-formatter:** two `String to replace not found` failures from constructing
  `old_string` against pre-formatter state. Note that a `Write` to a Python file may be
  reflowed, so an `Edit` needs a fresh read of the region.
- **Heredoc over `python -c`** for multi-statement one-liners with nested quotes (one
  unterminated-quote failure).

### `.claude/skills/issue/SKILL.md` — pipeline papercuts
- **Canonical inline lint gate.** Three sessions ran the gate's *components* by hand,
  were blocked by `guard_root_code_commit.sh` for an uncertified payload, then had to
  re-run `scripts/inline_lint_gate.py` (~4–10 min each). CLAUDE.md names the checks but
  never the invocation (`grep -c 'inline_lint_gate' CLAUDE.md` → **0**; it appears only
  in SKILL.md at ~line 7041). Also show the `--map-files` argument shape
  (`git diff --cached --name-only > /tmp/files.txt`), which three calls got wrong.
- **Bare `/issue-tick` with no `<N>`** loaded 21,752 chars of skill body into a live
  interactive session and was interrupted 2.4 s later; the `#1629` human-active screen
  lives inside `tick_triage.py`, which never runs. Propose a first-line short-circuit.
- **Step 0 ToolSearch preload** should include `TaskOutput,Monitor` (one
  `InputValidationError` from an unloaded deferred schema).
- **Draft-PR recipe** should ship the guard-compliant redirect form (`gh pr create … >
  /tmp/…log 2>&1; PR_RC=$?`) — the piped form is blocked.
- **Per-lane on-pod repo root** (`/workspace/eps-issue-<N>` on GCP vs
  `/workspace/explore-persona-space` on RunPod) and the working `gcloud` incantation for
  the `eps/phase` metadata key (the `/` breaks a naive projection) — both cost a turn
  after a failover.
- **Staggered critic verdicts** cost 2–3 orchestrator turns each, including 5 bare
  `true` no-op Bash calls purely to end a turn, plus one redundant `ScheduleWakeup`
  backstop that had to be swept at teardown.
- **Mergeability double-read:** #1680 burned an attempt on a stale post-push
  `mergeable` read (3 attempts total). Propose two consecutive agreeing non-UNKNOWN
  reads ≥5 s apart.

### Other named targets
- **`scripts/task.py` / `scripts/spawn_session.py` hygiene** — `new --body-file`
  silently drops the body's `goal:` frontmatter (documented, but silent); `set-goal
  --by` defaults to `user`, so an agent-set goal was recorded as yours; the
  crash-recovery registry wrote `session_id: null` / `auto: null` despite an `--auto`
  spawn; `spawn_session.py list` does not mark the calling session's own row (5 probes
  to disambiguate); `pod.py sync results` accepts none of `--pull/--pod/--paths` and
  errors under the sub-tool's name.
- **`.claude/rules/workflow-fix-on-bug.md` grep recipe** prescribes
  `grep -rln '<pattern>' .claude/ CLAUDE.md scripts/` at 3 places; `.claude/worktrees/`
  now holds **29 checkouts**, so the command auto-backgrounded at 2 min and had to be
  re-run scoped. Add `--exclude-dir=worktrees --exclude-dir=__pycache__`.
- **Recursion-guard exit latency distorts scope.** #1668's orchestrator explicitly
  reasoned that parking a candidate costs ~a day and therefore expanded the current
  round's scope instead. Proposed: extend the #1681 urgent-park grammar beyond
  `urgency: main-red` to a `gate-blocking` class (a park whose bug demonstrably
  red-verdicted this session's own gate).
- **`.claude/agents/planner.md`** — §10 should record the actual
  `git log origin/main..origin/issue-<M>` output per parent commit rather than a bare
  "merged on main" (a consistency-checker BLOCK on #1689 proved 11 such assertions
  false-in-evidence though right-in-conclusion); and a plan assumption naming a pytest
  node's PASS/FAIL should be marked unverified unless actually executed (a #1679
  fact-checker found a second test breaks and the count was 18, not 17).
- **`.claude/agents/implementer.md:174`** cites `tests/test_workflow_lint.py` at
  319–771 s; a #1679 verification stage ran ~24 min against that expectation. Re-measure
  and date the figure.
- **`scripts/step9c_baseline.py`** — trigger the ledger refresh on merge-count since
  refresh as well as wall-clock age (six concurrent merges aged it mid-afternoon).
- **`.claude/agents/planner.md` §9 / GPU width** — #1689 held a 4×H100 pod at
  `gpu_util 0,0,0,0` in 7 of 7 numeric poll samples through CPU (`corpus`, `render`) and
  pure-API (`haiku_u2`) phases, repeated across ~4 relaunch cycles. The per-phase
  GPU-width right-sizing rule exists; it was not applied at plan time.
- **`docs/api_throughput_guidelines.md`** — a Haiku 100 k row was quoted twice in one
  plan as a Sonnet figure (3–5 min vs the correct 13–17 min), near the ~200 k
  sync-vs-batch crossover. Re-key each latency band unambiguously on (model family,
  call count).
- **`.claude/rules/plan-compute-sizing.md`** — a #1689 bootstrap battery projected 140
  GPU-h against a 4 h plan estimate (**35×**) before batching brought it to 2.5×; the
  measured 1-cell pilot basis the rule already mandates was evidently not used.
- **`.claude/skills/adversarial-planner/SKILL.md`** — `plan_patch.py --verify-contains`
  needs "pick a fragment with no backticks and no line break" (one failed patch chain).
- **GCP boot-loop diagnostics.** #1689's GCP instance boot-looped twice and the #1029
  auto-failover absorbed it with no serial-console capture and no root cause; the task
  was then hard-pinned to RunPod, abandoning the free-credits lane. Propose persisting
  the dying instance's serial tail + last `eps/phase` to `issue<N>_partial/boot_death/`
  before the failover fires.
- **Pre-existing baseline debt.** Gates are baseline-subtracted: **35 ruff errors** and
  **15 lint WARNs** on `main` are permanently invisible to the pipeline. A one-off
  cleanup + a ratchet would retire them.
- **RunPod account-key preflight WARN** — the VM's key is absent from the shared team
  account key list (mutated by fellows-cluster onboarding). Latent: fresh pods are fine
  via PUBLIC_KEY boot injection, but a pod *resume* could fail. Touches a shared
  account, so it is yours either way.

## Not filed, and why — items the sweep deliberately dismissed

Recorded so a later pass does not re-raise them:
- **"Code-reviewer returned PASS with 7 open BLOCKERs"** — REFUTED at compose time.
  `.claude/agents/code-reviewer.md` rule 11 prescribes raising a BLOCKER *"even on a
  PASS verdict"* because *"the Step 5c-ter dispatch gate reads `concerns.jsonl`, not
  verdict prose"*. The orchestrator's bounce was the designed mechanism working.
- **`git status --porcelain --cached`** — an improvised flag, not present anywhere in
  the workflow surface (`grep` → 0 hits). Nothing to fix.
- **The piped-git guard blocking the session that was fixing it** (#1675) — self-resolving;
  the merged fix removes the class.
- **`Read` overflow on `scripts/sync_repo_root.py`** (28,240 tokens vs a 25,000 cap) —
  one extra tool call to recover; not worth a rule.
- **#1689's ~3.5 h spawn-to-progress delay from an API capacity wall** — the
  crash-recovery watcher behaved as designed. The *mis-classification* half (a 529
  labelled `boot-refusal`) IS filed as a route-2 task tonight.
- **~3 h and 2.3–3.5 M subagent tokens per single-file workflow fix** — deliberate
  review-depth policy, not a defect. The tractable sub-question (whether the Step 9c
  invariant remainder could run once per wave rather than once per session) is a
  cost/coverage tradeoff for you, not an autonomous call.
