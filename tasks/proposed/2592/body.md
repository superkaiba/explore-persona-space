---
title: 'workflow-fix: step9c compare''s systemic-breakage refusal advises ''refresh
  first'' when the cause is a dirty root, where refresh cannot work'
kind: infra
tags:
- wf-fix
created_at: '2026-08-25T22:21:11Z'
has_clean_result: false
origin_prompt: 'Hit during /issue 2578 Step 9c: compare refused with ''systemic main
  breakage (7 red files > --max-pristine-files 5) — investigate / refresh first''.
  Ledger was genuinely stale so a refresh was performed (79 min, 143-file universe);
  the fresh ledger came back dirty_code_paths=True and compare returned the identical
  refusal, because step9c_baseline.py:2161 makes a dirty ledger non-strippable and
  _bucket_run_failures then routes every node to the pristine bucket. No refresh can
  clear it while the shared root carries other sessions'' uncommitted code. Distinct
  from closed siblings #1341 (same mechanism, fixed escalate-only, message untouched),
  #2114 (staleness cron), #2235/#2316/#2318 (other dirty_code_paths consequences).'
workflow: v1
---
# Step 9c compare's systemic-breakage refusal sends operators to a remedy that cannot work when the cause is a dirty root

## Goal

Make the `step9c_baseline.py compare` systemic-breakage refusal name its ACTUAL cause, so an operator is not sent to a ledger refresh that is structurally incapable of clearing the refusal. Today the message says "investigate / refresh first" unconditionally, and in the dominant fleet condition a refresh cannot help.

## The gap

The refusal text (`scripts/step9c_baseline.py:3249-3252`) is fixed:

    systemic main breakage (<n> red files > --max-pristine-files <k>) — investigate / refresh first

But there are TWO distinct ways to reach it, with opposite remedies:

1. **Stale ledger.** `strippable=False` because of age or code-commit drift. A refresh IS the remedy. The message is correct.
2. **Dirty-root ledger.** `strippable=False` because `dirty_code_paths=True` — `step9c_baseline.py:2161-2163`:

       ledger_dirty = bool(ledger and ledger["dirty_code_paths"])   # MF-4b
       strippable  = ledger is not None and not stale and not ledger_dirty
       known_red   = ledger_nodes(ledger) if strippable else set()

   With `strippable=False`, `_bucket_run_failures` routes EVERY failing node to the pristine bucket (`if not lv.strippable: ctx.pristine_bucket.append(node)`), so the red-file count is the gate's full red-file count regardless of ledger contents. **A refresh performed while the root is dirty produces another dirty ledger, so the refusal repeats verbatim.** The remedy is `--max-pristine-files`, or cleaning the root — never a refresh.

The message names only remedy 1. An operator who follows it in case 2 pays a full refresh and lands on the identical refusal.

## Observed cost

During `/issue 2578` (2026-08-25), code-review round 1 PASSed and the Step 9c gate ran 1:31:41 producing 13 failures across 7 files. Compare refused. Ledger status was genuinely STALE on both criteria, so the refusal's advice looked actionable and a refresh was the honest call — **79 minutes** (143-file universe). The refreshed ledger came back `fresh` AND `dirty_code_paths: True`, and compare returned the byte-identical refusal. Only then did reading `_bucket_run_failures` establish that no refresh could ever have cleared it.

The blocking dirt was 20 uncommitted code paths belonging to issues 1482 (6 files), 1739, 1769, 1773, 1895, 1902, 2054, 2223 (2), 2356, 2378, 2474, 2094 and the MATS poster — none of them the round's own, none in its payload, and none cleanable by that session without clobbering concurrent work.

This is not a rare state. A shared root habitually carrying other sessions' uncommitted code is the NORMAL fleet condition (#2015), so case 2 is the common case and case 1 the exception — the message documents the exception.

## Relationship to closed siblings (checked; distinct from all five)

- **#1341** — same MECHANISM (untracked root drafts set `dirty_code_paths=True` and blocked every compare). Fixed escalate-only: a watcher pass SURFACES the dirt. The mechanism therefore persists by design; nothing changed the refusal text. Worth a look as part of this task: 20 dirty paths across 13 issues suggests that pass is either not firing or its alerts are being ignored.
- **#2114** — nightly ledger-refresh cron for STALENESS. Cannot help here: a nightly refresh on a dirty root still records `dirty_code_paths=True`.
- **#2235**, **#2316**, **#2318** — other `dirty_code_paths` consequences (inline lint-gate scope, scan-node grain). Different surfaces.

None addresses the refusal message's actionability.

## Acceptance criteria

1. The refusal distinguishes its cause. When `strippable=False` because of `dirty_code_paths`, say so and name the workable remedies (`--max-pristine-files <n>`, or clean the root), NOT "refresh first". When the cause is staleness, keep the current advice. Ideally report both flags (`stale=<bool> ledger_dirty=<bool>`) in the message and in the JSON, so the JSON consumer can branch too.
2. The JSON payload carries the discriminator (e.g. `strip_disabled_by: ["stale"|"dirty_root"|...]`) alongside the existing `reason`, so an orchestrator does not have to parse prose.
3. A test pinning both branches: a stale-but-clean ledger yields refresh advice; a fresh-but-dirty ledger yields cap/clean-root advice and never the word "refresh".
4. Optionally, and only if it does not weaken the guard: when the sole strip-disabling condition is `dirty_root` AND the red-file count is within a small margin of the cap, emit the exact `--max-pristine-files <n>` command the operator should run.

## Scope notes

Message + JSON discriminator + tests. Do NOT change the guard's threshold, the strippability rules, or MF-4b's fail-closed posture — the dirty-root ledger SHOULD be non-strippable; the defect is only that the refusal misdirects the operator about why.

Estimated GPU-hours (total): 0
