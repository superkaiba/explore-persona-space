---
title: 'Experimenter launcher composition: mandatory per-leg out/scratch dir isolation
  for concurrent same-driver legs (#2330 fu1 dense-store poisoning)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-17T15:51:16Z'
has_clean_result: false
workflow: v1
---
target_file: .claude/agents/experimenter.md

## Gap

Concurrent same-driver legs on one pod need MANDATORY per-leg out/scratch dir isolation in launcher composition — the experimenter spec's launcher pattern (and the multi-leg brief guidance in `.claude/skills/issue/SKILL.md`) does not require it, and `.claude/rules/crash-fix-rounds.md` § "Per-leg out-roots" scopes only to REGIME-keyed resume state, not to concurrent-leg chunk-basename collisions.

## Incident (#2330 fu1, 2026-08-17)

Both fu1 launchers inherited the same `EPM_I2330_OUT_DIR` from the P1 preamble. fuA (A1 dense capture, train_10k shard00) and fuB (cap2048 capture, train_10k, num-shards 1) wrote IDENTICAL chunk basenames (`shards/train_10k/shard00_chunk*.pt`) into one scratch dir. fuB overwrote fuA's not-yet-flushed chunks 0000-0002; fuA's end-of-shard flush then uploaded fuB's 3-layer bytes to the DENSE prefix (sha-verify hashes at flush time — self-consistent, so poisoning passed SILENTLY; caught only by LFS-size forensics: 49 MB vs 508 MB) and purged the files, killing fuB's terminal flush with FileNotFoundError. Cost: 3 poisoned dense chunks (deleted + re-capture scheduled), ~80 min GPU redo, one crash-fix round.

## Fix shape

Add to the experimenter launcher-composition checklist (and the SKILL multi-leg brief guidance): when composing >1 concurrent leg of the same driver on one machine, (a) derive a PER-LEG out/scratch root (suffix the leg name onto the shared env var) whenever the legs' split+shard chunk basenames can collide — collision test: same driver + same split name + overlapping shard indices; (b) state the isolation in the launch breadcrumb. This is the OUT-dir sibling of the #1315 fanout-shared-staging rule (which covers download staging) and of crash-fix-rounds' regime-keyed per-leg out-roots.

Evidence: task #2330 events.jsonl epm:failure 2026-08-17T15:4xZ (failure-lesson block, root_cause_confirmed: yes), LFS-size forensics in the session transcript.
