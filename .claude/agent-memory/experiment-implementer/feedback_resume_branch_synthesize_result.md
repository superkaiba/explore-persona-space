---
name: Resume branches synthesize success-equivalent results
description: A skip-existing/resume branch that bare-continues past completed work makes downstream completeness gates read it as a crash; synthesize the result entry and rehearse the resume path offline
type: feedback
---

A resume/skip-existing branch must synthesize a result entry EQUIVALENT to a successful run — read the persisted completion sentinel, re-point path fields at the current output root, mark `skipped_existing: true`. A bare `continue` makes every downstream completeness gate (`if failures or not results`) misread skipped-but-complete work as a crash.

**Why:** Incident #600 round 5 (2026-06-11): the relaunch with EPM_SKIP_EXISTING=1 died ~5s in because the skip branch appended nothing and the smoke gate read `smoke cell crashed: []`.

**How to apply:** When writing ANY resume/skip path: (1) synthesize the success-shaped record from persisted artifacts; (2) require the completion sentinel (not just the main artifact) before skipping; (3) NEVER ship a resume branch that has never executed — rehearse it offline against the producer's REAL artifacts (scp'd down, relocated root) before any pod relaunch.
