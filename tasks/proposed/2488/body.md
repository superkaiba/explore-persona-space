---
title: 'workflow-fix: step9c compare''s pristine oracle is cut at the merge-base,
  so Step 5a-synced current-main files yield systematic false-NEW verdicts (blames
  the innocent branch)'
kind: infra
tags:
- wf-fix
- workflow-fix
created_at: '2026-08-23T02:07:33Z'
has_clean_result: false
origin_prompt: /issue 2263
workflow: v1
---
## Overview / Motivation

`step9c_baseline.py compare`'s pristine oracle is cut at the branch's **merge-base**, while the mandated Step 9c pre-gate Step 5a sibling sync copies **current-main** files into the worktree. Any file main added after a branch's fork point therefore cannot exist in the oracle tree, so a pre-existing-on-main failure in that file is classified **NEW** and blamed on the branch.

Observed on #2263, 2026-08-22, at a cost of one ~102-minute gate run plus two compare replays before the true culprit was identified.

## The mechanism

1. Step 9c step 1a **mandates** a pre-gate Step 5a spec-freshness re-sync (#1742), run from the worktree, before the selector.
2. That sync's sibling arm copies main's own files in. On #2263 it brought `scripts/issue823_shared_persona_paired.py` (added to main by `d526008c67`, *after* this branch's merge-base `1715f2bb0b8b`).
3. The selector correctly recognizes these as base-identical and excludes them from the branch diff (the #2302 fix): its stderr NOTE listed **11** such paths.
4. The gate runs `tests/test_shared_vm_thread_caps.py` — in the 61-file workflow-invariant set, so in **every** Step 9c selection. Its `_scan_targets` unions worktree files, so it scans the synced file regardless of the diff-base exclusion, and it fails: main's copy has 0 `load_dotenv` and module-top numpy/scipy.
5. `compare --run-pristine` replays the failing node on a pristine oracle at `oracle_base_sha = 1715f2bb0b8b`. **The file does not exist there.** The failure cannot reproduce, so it is classified `new`.
6. `compare` exits 1 ⇒ Step 9c PASS is unreachable ⇒ the spec routes to "re-spawn implementer" against a branch that did nothing wrong.

## Evidence

From #2263 (`epm:test-verdict v1`, 2026-08-23T02:03:59Z):

- `oracle_base_ref: origin/main` but `oracle_base_sha: 1715f2bb0b8b` — the merge-base, while `git rev-parse origin/main` was `810e4cd74d`.
- Passing `--base 810e4cd74d…` explicitly **did not move it**: the re-run still reported `oracle_base_sha: 1715f2bb0b8b`. So `--base` governs the diff base, not the pristine oracle, and there is no documented lever to advance the oracle.
- `new: [tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints]`
- `git cat-file -e 1715f2bb0b8b:scripts/issue823_shared_persona_paired.py` → **absent**.
- The worktree copy is byte-identical to `origin/main`'s; main's own allowlist lacks it (244 entries, identical on both sides) ⇒ **main is red on that node**, filed as #2487.
- **`base_identical_files: 0`** in the compare JSON, against the selector's **11**. Same seam, second symptom: the compare does not appear to consume the selector's base-identical set at all.

## Why this is not an edge case

#2263's branch is **1,590 commits / 393 files** behind main at the merge-base. Any long-lived issue branch accumulates that drift, and the sibling sync deliberately pulls current-main content across it on **every** gate run. So the false-NEW window is the entire set of files main has added since the fork — and it grows the longer a task runs, i.e. exactly when a task can least afford a 100-minute misdiagnosis.

The available workarounds are all bad, which is why this needs a real fix:

- **Merge main into the branch** to advance the base: 1,590 commits, 393 files, and on #2263 it would have touched two of the round's three deliverable files — materially changing a tree that a PASS+PASS review round had just certified.
- **Edit the offending file in-branch**: out of scope, races the owning task's session (`.claude/rules/cross-session-writer-arbitration.md`), and pulls main's own file into the branch diff — recreating the very #2302 problem the selector's exclusion exists to prevent.
- **Declare PASS anyway**: a false claim in a durable marker.

## Proposed change (sketch — refine in planning)

Candidate levers, not mutually exclusive:

1. **Consume the selector's base-identical set in the compare.** A failure whose only implicated file is base-identical to `origin/main`'s tip is by construction not the branch's. That set is already computed and printed one step earlier; `base_identical_files: 0` suggests it is simply not being threaded through.
2. **Let the oracle advance.** Either honor `--base` for the pristine oracle, or add an explicit `--oracle-base`. A pristine tree at main's *tip* would have reproduced this failure and stripped it correctly.
3. **Second-oracle fallback for a NEW node whose implicated file is absent at the oracle base.** Absence-at-oracle is a detectable, specific condition — it is the exact signature of this bug — so a targeted replay at main's tip could resolve it without changing the default oracle.
4. At minimum, **emit a WARN** naming the condition ("node's implicated file absent at oracle base; NEW classification unreliable") so the orchestrator is not left inferring it. On #2263 the diagnosis took manual `git cat-file -e` probing that only happened because the outcome looked wrong.

Planning should also decide whether the same seam affects the Step 10d `mapped-baseline` leg, which takes an explicit `--base`.

## Coordination

**#2487** is the concrete main-red instance that surfaced this and is filed separately — it fixes one script; this task fixes the classifier. Both are needed: #2487 unblocks #2263's gate, this one prevents the next branch from spending 100 minutes on the next such file. #865 (`on_hold`) — "Step 9c selector diffs main checkout, blind to worktree branches" — is a related-but-distinct selector-side gap.

## Verified at filing

- All compare JSON figures above from `/tmp/step9c-compare-2263.json` and the `--base`-pinned re-run, #2263 gate 2026-08-22/23.
- `git rev-list --count 1715f2bb0b8b..origin/main` → 1590; `git diff --name-only 1715f2bb0b8b..origin/main | wc -l` → 393.
- `git diff --name-only 1715f2bb0b8b..origin/main -- <the round's three files>` → 2 of 3 would be touched by a merge.
- #2263 `epm:test-verdict v1` records the full disposition.

## Provenance

workflow_fix_target: scripts/step9c_baseline.py

Found by the #2263 orchestrator during its own Step 9c gate, per `.claude/rules/workflow-fix-on-bug.md`.
