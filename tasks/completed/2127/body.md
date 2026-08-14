---
title: 'daily-fix: sweep /tmp gate scratch; HF-backedness gate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fbb4d045745a
- daily-auto-filed
created_at: '2026-08-06T07:07:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): ~70GiB stale /tmp gate/smoke
  scratch unswept at 98% root disk; user rejected age-based deletion, gated on verified
  HF-backedness'
workflow: v1
---
# daily-fix: sweep top-level /tmp gate/smoke scratch in the disk janitors + record the HF-backedness deletion gate

## Workflow gap

With `/` at 98% (23G free) on 2026-08-06 ~04:02Z, the remediation session found ~70 GiB in
top-level /tmp dirs untouched ≥48 h — gate/smoke scratch the janitors never sweep
(`/tmp/eps-main-scratch-2058` 11G, `mkstest-main` 9.9G, `issue-1895-gate2` 9.3G, …). The
#911 non-canonical cache sweep keys on issue-keyed name shapes and `data/` cache dirs;
these gate/smoke scratch shapes match neither. Two process facts from the same incident:
(a) Thomas REJECTED a blanket age-based deletion (AskUserQuestion rejected) and redirected
to "delete the things that are safely on huggingface" — deletion gated on VERIFIED
HF-backedness, not age; (b) the HF audit also surfaced UNBACKED artifacts (issue_1092:
5.2 GiB incl. two 1,472 MiB krr_nystrom weights) that age-based deletion would have
destroyed — validating the gate.

verified-at-filing: the incident rows are probed (session 4966e56e rows 1941–1953, 2092,
2128). Dedup scan at compose time: open #2095 ("sweep /mnt/eps-data/$USER staging in disk
janitors") targets the DATA-disk staging dirs, #2097 ("mechanize local-disk headroom")
targets headroom accounting, #2042 (preflight disk probe) targets the probe — none covers
top-level /tmp gate/smoke scratch shapes; named here so the planner reconciles scope with
all three.

## Proposed change

- `scripts/vm_disk_guard.py`: extend the non-canonical sweep's name shapes to top-level
  /tmp gate/smoke scratch (`/tmp/*-gate*`, `/tmp/*smoke*`, `/tmp/eps-*-scratch-*`,
  `/tmp/mkstest-*`), keeping the existing 48 h recently-touched keep + positive
  re-downloadability/regenerability evidence contract — for these shapes the evidence is
  that gate/smoke trees are reproducible from git by construction; anything matching the
  shape but carrying non-reproducible artifacts is escalate-only.
- Record the deletion-gating preference in the janitor's contract text (and CLAUDE.md §
  Disk hygiene if the planner deems it load-bearing): user-facing bulk deletions are gated
  on verified HF-backedness (or git-reproducibility), never age alone.

## Provenance

- fingerprint: fbb4d045745a

- workflow_fix_target: scripts/vm_disk_guard.py
- origin: /daily 2026-08-05 problem sweep — miner 5 P15 (user-corrected deletion policy +
  janitor gap, probed rows).
