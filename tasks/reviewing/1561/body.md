---
title: 'daily-fix: hub-verify lint FP on monkeypatch attribute refs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:49985fa963ea
- daily-auto-filed
created_at: '2026-07-20T06:47:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): _hub_verify_bare_hits flags
  non-call monkeypatch attribute references'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from a transcript-mined problem (session efbcc710 / #1482 @ 08:14-08:26 UTC).

## Goal

Stop `scripts/workflow_lint.py` `_hub_verify_bare_hits` from flagging non-call ATTRIBUTE references (monkeypatch save/restore assignments in self-test code), or document the monkeypatch-waiver requirement where the check is defined.

## Workflow gap

- **Bug observed:** the Step 10d lint gate flagged `orig_lrt = huggingface_hub.HfApi.list_repo_tree` and monkeypatch assignment lines (no call parens) as bare hub-verify hits; #1482 had to reproduce the predicate offline and waive 4 sites before the gate passed.
- **Why it is a workflow gap:** the check exists to catch UNRETRIED hub CALLS; a monkeypatch attribute reference cannot 504-storm, so flagging it forces noise waivers that dilute the waiver channel.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "_hub_verify_bare_hits" scripts/workflow_lint.py` → defined :7335, used :7439, and the docstring at :7385 names an "Attribute leg" — i.e. attribute matching is present by design; context read shows no monkeypatch/self-test exemption. The plan should engage why the Attribute leg exists (a bare alias later called still storms) and scope the exemption narrowly (e.g. assignment-target/save-restore shapes in test/self-test code) (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from the mined problem: exempt monkeypatch save/restore assignment shapes, or add an inline doc + waiver recipe at the check site)

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Tests: the check's existing test file (grep `hub_verify` under tests/).

## Constraints / invariants

- Workflow-surface only; ruff passes; the check must keep catching aliased CALL sites.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 3d05d5e18ddd

Mined evidence: "hit at line 618 pattern '.list_repo_tree(' waived=False | 618: orig_lrt = huggingface_hub.HfApi.list_repo_tree" — 4 waivers needed on self-test monkeypatch code (#1482 Step 10d, 2026-07-19).
