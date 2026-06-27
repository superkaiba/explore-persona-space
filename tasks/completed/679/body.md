---
title: Bound active-task disk on the CPU VM (wire incremental cleaner + faster detection
  + escalation)
kind: infra
tags: []
created_at: '2026-06-27T01:41:23Z'
has_clean_result: false
origin_prompt: run the plan in the background with happy coder (disk fills up fast;
  vetted /plan at ~/.claude/plans/2026-06-26_183815-bound-active-task-vm-disk.md)
---
# Bound active-task disk on the CPU VM — wire incremental cleaner + faster detection + escalation

## Autonomous execution guardrails (READ FIRST)
- **kind: infra workflow change.** Implement via the `/issue` code-change path (implementer → code-reviewer → tests). NOT an experiment; no GPU, no pod.
- **Do NOT touch task #658's live process or its worktree data.** Do NOT run any disk deletion — no `clean_experiment_downloads ... --apply`, no `rm`, no `vm_disk_guard --apply` against #658. The #658 139G reclaim is a SEPARATE, user-gated operational step, explicitly OUT OF SCOPE for this task. (Critical: #658's `g1_dl` cache is currently being read by a live `issue658_judge_e0_batch.py` process — the blanket cleaner would crash it.)
- **Honor warn-only:** NO new code path may auto-delete an ACTIVE task's data. For active tasks the guard ESCALATES only (Telegram + sidecar jsonl). Auto-deletion stays terminal-status-gated (existing behavior) + user-approved (reclaim).
- **Workflow-surface files only** (listed in Steps). Commit by explicit path on `main`; land as narrow commits per the sequencing note. Escalations go to a sidecar jsonl (`.claude/cache/disk-guard-events.jsonl`) + `telegram_push.sh`, NEVER a per-run `task.py` commit (index.lock churn).
- **Full vetted plan (3 Claude∥Codex review rounds):** `~/.claude/plans/2026-06-26_183815-bound-active-task-vm-disk.md` — read it for the rationale, the rejected options, and the review history before implementing.

## Goal
Stop the CPU-only orchestration VM root disk (`/`, single 500G ext4) from filling fast. Verified root cause: an *active* task holds a large **re-downloadable HF download cache** (#658: 139G in `data/issue_658/hf_dl/.../store/`, a CPU phase's download of an HF-hosted store; 178 files confirmed on HF). The existing cleaner *would* reap it (`rmtree(hf_dl)` wholesale) but is terminal-status-gated, and the task is active — so nothing bounds an active task's disk, and 139G landed in ~10 min (faster than any cron). It is NOT misplaced durable output and NOT a placement bug (downloading an HF store into `hf_dl/` is legitimate).

## Out of scope
- Auto-deleting active-task data (warn-only).
- The #658 139G reclaim (separate user-gated op; see guardrails).
- A "store-under-hf_dl" placement lint (rejected — false-positives on legitimate re-downloads).
- Editing experiment code (`scripts/issue*_*.py`, `scripts/train.py`, `scripts/eval.py`, `configs/`).
- Dollar caps.

## Approach
Three real levers (none is a launch-time gate): wire the existing between-phase incremental cleaner (the real mid-run bound), faster + attributed detection + escalation (warn-only), and a metadata-parity defensive guard. Plus reused preflight floor + docs. The bounded-by-construction fix (dedicated volume / ext4 quota) is filed as a separate lead follow-up, not implemented here.

## Steps (land as narrow commits, by explicit path)
1. **`scripts/clean_experiment_downloads.py` defensive hardening + tests.** Before the wholesale `rmtree(hf_dl)` at a terminal reap, if the tree contains a nested `store/`, gate deletion on **metadata-presence parity** (revision-pinned `huggingface_hub.list_repo_files` + per-file size match — NOT a size-*sum*); reap only if every file is present on HF, else SKIP + escalate (fail-toward-keep). Add `tests/` covering: re-downloadable nested store reaped; un-verifiable nested store skipped+escalated; plain cache reaped. Do NOT change the existing terminal-status gating semantics otherwise.
2. **Wire the EXISTING `clean_issue_downloads_incremental` / `--incremental` into the between-phase contract.** It already exists and is called from nowhere. Make the shared experiment-runner / pipeline phase-boundary call `clean_experiment_downloads --incremental --apply <N>` after a phase consumes its downloads and before the next downloads. Document the contract in `CLAUDE.md` "CPU-only phases don't hold GPU pods" (large-data caveat) + `.claude/skills/experiment-runner/SKILL.md`. (Deletes only caches the experiment declares consumed — never data the run still needs.)
3. **Faster, attributed detection in the existing watcher.** Add a root-`/`-headroom check to `scripts/autonomous_session_watch.py` (and `scripts/tick_triage.py`); below a WARN floor, drop to a ~45s re-check sentinel and emit an escalation that names the top-growing per-issue cache paths (cheap `du -s` on the `hf_dl`/`g*_dl` globs ONLY, never a full-tree walk). Use bytes AND inode/percent signals. Warn-only (no deletion). Dedup on (band) + growth re-alert.
4. **Active-task escalation augment in `scripts/vm_disk_guard.py`.** In the EXISTING active-skip branch (it already logs `kept (status=… not terminal)`), add an escalation (Telegram via `telegram_push.sh` + sidecar jsonl) naming the task, the largest re-downloadable-cache path, the footprint, and the SAFE reclaim command (issue-scoped, dry-run default). Never auto-delete active data. Dedup on (task, threshold-band) + re-alert on >X% growth + suppress after an ack sentinel. ~10-line addition; no per-run commit.
5. **Preflight floor (reuse, do not rebuild).** Raise the existing `/`-headroom check in `src/explore_persona_space/orchestrate/preflight.py` (today a <20G WARN) to a configurable hard FLOOR (env-overridable) so a fresh launch below floor fails fast. Document that it only helps fresh launches, not mid-task fills.
6. **Docs.** `CLAUDE.md` Disk hygiene + "CPU-only phases" caveat (call the incremental cleaner between phases; prefer streaming/sharding; the guard escalates if a cache grows) + a pointer to the lead follow-up.
7. **File the lead infra follow-up** (do NOT implement): a `proposed kind: infra` child task — "attach a dedicated GCP persistent disk for `.claude/worktrees/` + per-issue caches, OR enable ext4 project quotas per issue, so one task cannot exhaust `/` and the failure is bounded without deletion." This is the only true bounded-by-construction fix; needs a new volume / fs-quota work.

## Success criteria
- `clean_experiment_downloads --incremental` is invoked at the shared phase boundary; CLAUDE.md + experiment-runner document the contract.
- The watcher escalates within one (sub-floor) cycle with the offending per-issue cache path(s); no deletion.
- `vm_disk_guard`'s active-skip branch escalates (Telegram + sidecar), never auto-deletes active data.
- The wholesale `rmtree(hf_dl)` can never destroy a nested store NOT verified present-on-HF (metadata parity, fail-toward-keep); tests prove it.
- Preflight refuses a fresh launch below the configurable floor.
- All touched tests pass; `workflow_lint.py` passes; nothing touches #658's live run.

## Risks / notes
- Defensive parity check runs ONLY when a nested store is present (rare) at a terminal reap; metadata-only (no download); any ambiguity → SKIP+escalate.
- A global `/` floor could block other sessions' fresh launches when one task fills the disk — acceptable as a backstop; the real bound is the follow-up (#7). Allow a logged override.
- The incremental cleaner only helps experiments/harnesses that call it; bespoke `run_issueNNN_pipeline.sh` scripts are backstopped by the detection layer.
