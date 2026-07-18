---
title: 'daily-held: canonical HF model repo at 100k-file limit'
kind: infra
tags:
- daily-held
created_at: '2026-07-08T07:00:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 3): c86ff35c (#1090) 09:43Z:
  adapter upload rejected ("would contain 100050 files") — the canonical HF model
  repo hit the 100,000-file hard limit. #1108 shipped the private-overflow-repo fallback
  same day, so uploads keep working, but the canonical repo is frozen at the limit
  and every new adapter lands in overflow.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 3) from the nightly transcript problem sweep.

## Goal

Thomas decides the canonical-repo cleanup: purge/archive old adapter trees (the parked adapters/issue_397 review, 242GB, is one candidate; cf. the wandb-archive precedent), or accept overflow-only growth permanently

## Workflow gap

- **Bug observed:** c86ff35c (#1090) 09:43Z: adapter upload rejected ("would contain 100050 files") — the canonical HF model repo hit the 100,000-file hard limit. #1108 shipped the private-overflow-repo fallback same day, so uploads keep working, but the canonical repo is frozen at the limit and every new adapter lands in overflow.
- **Why it is a workflow gap:** Destructive / irreversible action (deleting published artifacts) — route-3 carve-out.

## Proposed change

Held for Thomas; PM should surface alongside the existing HF-storage parked review (issue_397 purge decision).

## Scope / surfaces

- Primary target: `external (HF repo superkaiba1/explore-persona-space)`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: c86ff35c (#1090) 09:43-09:46Z; task #1108 (completed).


## Decision package (audit run 2026-07-18 — PARKED FOR THOMAS)

**Premise correction (live-verified):** the canonical repo is NO LONGER frozen. HF's 100k hard-limit enforcement changed after 2026-07-07 — the repo sits at **117,050 files** (3.51 TB) and accepts uploads: **492 post-rejection folder pushes** landed, net growth **+17,000 files** vs the 100,050 rejection anchor (first post-rejection upload 2026-07-08 00:49Z). Current HF docs list "<100k files/repo" under *Recommendations* (the 10k/folder cap stays hard). Status wording: *not enforced at the current count/shape as of 2026-07-18* — the #1108 overflow fallback stays armed as insurance either way.

**Full evidence + numbers:** [repo_file_audit_report.md](https://github.com/superkaiba/explore-persona-space/blob/902e53efbd/tasks/running/1141/artifacts/repo_file_audit_report.md) · [repo_file_audit.json](https://github.com/superkaiba/explore-persona-space/blob/902e53efbd/tasks/running/1141/artifacts/repo_file_audit.json) · [freeing_commands.md](https://github.com/superkaiba/explore-persona-space/blob/902e53efbd/tasks/running/1141/artifacts/freeing_commands.md) (audit coverage 100.0%, 0 unattributed files; tool: `scripts/issue1108_repo_file_audit.py`, extended under this task).

### Recommendation (specific; every irreversible step is USER-ONLY and listed as a command)

1. **RECOMMENDED — partial overflow migration (post-#1108 era only): copy 633 files / 18.0 GB back to canonical.** These are the artifacts the file-count rejections stranded behind auth + pointer indirection; destinations = their own preserved `path_in_repo` prefixes. Do NOT migrate the pre-#1108 era (**3,247 files / 253.0 GB** — #564-era content deliberately byte-quota-routed PRIVATE; copying it public re-enters the #541/#552 public-storage-quota surface for zero consumer benefit). Copy command in the report §(b); after a VERIFIED copy, the pointer + overflow deletions are yours to run.
2. **RECOMMENDED (deferred execution fine) — (c1) archive-then-delete `adapters/issue_397`: frees 7,668 files / 216.7 GB LFS.** #397 is completed/useful; its citation list is dominated by self-references (this task's + #1108's triage docs + #397's own body/plans). Two real references to eyeball before clearing: **#813's plans** (task at awaiting_promotion — cites issue_397; check whether it consumes the HF adapters or only git eval_results) and `scripts/plot_issue397_hero.py` (reads git `eval_results/issue_397`, NOT the HF adapters — false-positive for this deletion). Archive copy first (wandb-archive precedent; pre-deletion revision stays fetchable regardless), delete only after verification. Commands in the report §(c1).
3. **OPTIONAL — (c2) terminal-ladder rung pruning: ≤15,182 files / ≤337.2 GB LFS (selection-blind UPPER BOUND;** advisory 20% trigger did NOT fire at 12.97%). "Keep max-step" does NOT protect band-stop/dose-selected rungs (#532 reused #474's epoch-1) — the per-task `cited_by` rows + `freeing_commands.md`'s ready-vs-unsafe split are the protection; verify each task's selected checkpoint against its Reproducibility record before any delete. No urgency: the repo is accepting uploads.
4. **Do-nothing is viable** (option (a)): zero risk, but overflow artifacts stay private + pointer-mediated, and 59.6% of the repo (69,745 files, 1.76 TB) remains checkpoint-ladder residue. If HF re-enforces, the pre-registered urgency branch says free ≥ **18,050** files ((c1)+(c2) cover it: 7,668 + 15,182 = 22,850).

**What needs YOU (in order of value):** (i) approve the 18 GB post-#1108 overflow migration copy; (ii) clear/dismiss the two real issue_397 citations and approve the (c1) archive+delete; (iii) optionally pick (c2) tasks from the per-task table. Nothing was deleted, moved, or created on HF by this task (zero HF writes, AST-verified).

