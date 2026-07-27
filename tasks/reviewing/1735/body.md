---
title: 'daily-fix: closed-sibling probe suppressed 21 of 24 filings '
kind: infra
tags:
- wf-fix
- wf-fix-fp:24883d891330
- daily-auto-filed
created_at: '2026-07-27T07:21:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): the #1711 closed-sibling
  and #1674 landed-fix probes recorded terminal non-filing outcomes for 21 of 24 items
  in the 2026-07-26 batch, matching mostly on a shared hot target file or a single
  generic title token such as main, runs or step; all 21 were manually confirmed unrelated
  to the cited sibling'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by the sweep's own
filing run — the probe blocked 21 of this run's 24 filings.

The `#1711` closed-sibling pre-filing probe and the `#1674` landed-fix probe, both of which
merged on 2026-07-26, together record a terminal non-filing outcome for the large majority
of a nightly batch on this repo's hot workflow files. Tonight the operator eyeballed all 21
and every one was a false positive, so the batch was filed with `--retry-suspects`. A run
that did not check would have silently dropped 21 verified defects.

## Goal

Tighten the closed-sibling / landed-fix suspect predicates so a shared hot target file or a
generic title word does not by itself record a non-filing outcome, and make the suppressed
count impossible to miss.

## Workflow gap

- **Bug observed:** on the 2026-07-26 batch, `daily_drive_filings.py --dry-run` recorded
  `CLOSED-SIBLING-SUSPECT` or `LANDED-FIX-SUSPECT` (terminal, NOT filing) for **21 of 24**
  manifest items; only 3 would have filed. All 21 were manually confirmed unrelated to the
  cited sibling.
- **Why it is a workflow gap:** the nightly sweep concentrates on a handful of hot
  workflow-surface files (`.claude/skills/issue/SKILL.md`, `scripts/workflow_lint.py`,
  `.claude/agents/*.md`), so within any 7-day window nearly every new filing shares a target
  file with some recently-closed sibling. A `matched: target`-only hit therefore carries
  almost no evidence of duplication on this corpus, yet it suppresses the filing.
- **Confidence (emitter):** high
- verified-at-filing:
  `uv run python scripts/daily_drive_filings.py --dir logs/daily/filings-2026-07-26 --dry-run`
  → 21 suspect lines vs 3 `FILE` lines (24 items);
  `grep -oE '\(matched: [^)]*\)' /tmp/dryrun2.log | sed 's/; .*//' | sort | uniq -c | sort -rn`
  → top reasons `title:main` (9), bare `target` (7), `title:runs` (5), `title:merge` (4),
  `title:tests` (3), `title:step` (3), `title:probe` (3), `title:path` (3),
  `title:check` (3);
  `git rev-parse --verify --quiet '9106aaf478^{commit}'` →
  `9106aaf478...` resolves, and `git show 9106aaf478 --stat` confirms it is task #1560, the
  commit that BUILT the Step 5a/10d spec-freshness sync (2026-07-20) — i.e. the mechanism
  tonight's `spec-freshness-sync-inconsistent-tree` item reports a defect IN, not a prior
  fix for it. (2026-07-26)

## Evidence

- Representative false positives, each a terminal non-filing outcome:
  - `main-red-inline-round-duty-mirror` → `#1648` *"lint bare git-commit recipes lacking
    pathspec"* (matched: `target`). #1648 is unrelated to a deleted lint function; this is
    the URGENT main-red restore.
  - `code-reviewer-ruff-policy-pin` → `#1693` *"phase idempotency + inter-phase schema
    assert"* (matched: `target`).
  - `planner-grounding-discipline` → `#1604` *"wire identity+bias baseline + kNN retrieval
    mandate into planner/critic lenses"* (matched: `target`, `title:planner`).
  - `select-step9c-tests-ergonomics` → `#1697` *"--map-files timeout dispersion margin"*
    — same file and even the same flag name, but #1697 changed a timeout constant while
    this item is about `--json` being silently ignored.
  - `spec-freshness-sync-inconsistent-tree` → landed-fix suspect `9106aaf478`, which is the
    commit that introduced the mechanism being reported as defective.
- Nine separate items matched on the single title token `main` — a word that appears in
  ordinary workflow prose (`origin/main`, "main tree", "main-red") and carries no
  duplication signal.
- The `#1399` advisory this probe descends from was explicitly specified as **advisory
  only, never a block** (`.claude/rules/workflow-fix-on-bug.md`, § Recently-closed-sibling
  ADVISORY: "it never blocks the filing, never changes exit codes"). The `#1711` probe
  records a terminal ledger outcome instead, which is a stronger contract than the rule
  documents.
- Cost tonight: zero filings lost, because the operator read the dry-run and re-ran with
  `--retry-suspects`. The exposure is the counterfactual — a run that trusts the default
  drops the batch silently, and the ledger reads as a clean pass.

## Proposed change

- Require MORE than a bare target-file match before recording a suspect outcome: e.g. a
  shared target PLUS at least one shared informative title token, or a body-level overlap
  signal.
- Extend the non-informative token stoplist used by the title-overlap arm with the generic
  workflow vocabulary observed here — `main`, `runs`, `step`, `merge`, `tests`, `path`,
  `check`, `probe`, `state`, `daily`, `zero`, `repo`, `shared` — so a single such token
  cannot produce a match. (`task_workflow.informative_title_tokens` is the shared
  implementation; check whether the stoplist belongs there or driver-side.)
- Reconcile the contract with `.claude/rules/workflow-fix-on-bug.md`: either soften the
  probes to advisory (print, then file) to match the documented `#1399` behaviour, or
  update the rule to state that the closed-sibling probe is now blocking-by-default with
  `--retry-suspects` as the override. Today the rule and the code disagree.
- Make the suppression legible regardless: print a one-line summary
  (`N of M items suppressed as suspects — re-run with --retry-suspects to file`) at the end
  of the run, and record the suppressed count in the `filed.jsonl` run record, so a future
  operator cannot mistake a mass suppression for a clean batch.

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`
- `src/explore_persona_space/task_workflow.py` (`informative_title_tokens` stoplist, if the
  shared helper is the right home)
- `.claude/rules/workflow-fix-on-bug.md` (advisory-vs-blocking contract reconciliation)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- Loosening the predicate must not reintroduce the `#1350`/`#1329` duplicate-filing class
  the probe exists to prevent — a genuine same-bug sibling must still be caught. The two
  measured reference points: #1350 (filed 25 min after #1329 merged the same fix) should
  still match; tonight's 21 should not.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 24883d891330

- workflow_fix_target: scripts/daily_drive_filings.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Observed directly in this run's own filing dry-run.
