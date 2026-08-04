---
name: Persist-before-reduce sweeps — check ORDERING, not file existence; derived expected-set ≠ hollow gate
description: "#906 r9 split: Claude counts a site 'persisted' because the JSONL exists (write AFTER the remote judge — the ordering defect IS the class); Codex FAILs a glob-derived upload expected-set as hollow-verification-gate though producers make the missing state unreachable. Both non-blocking when the class's r8 precedent is CONCERN."
type: feedback
---

Two paired calibration rules from #906 r9 (Claude PASS vs Codex FAIL, reconciled PASS + 2 CONCERNs):

1. **Claude-side miss (ordering vs existence).** When the round's policy class is
   persist-rollout-text-BEFORE-reduce (#779 / Upload Policy), a sibling sweep that
   counts a site as "persisted, upload-pinned" because the output file EXISTS is
   not verifying the class — the class is the ORDERING. #906 r9: on-policy control
   accumulated temp=1.0 rollouts in memory, ran the remote judge
   (`score_completions`), and only then wrote `completions.jsonl`
   (issue906_phase1_pilot.py:807→841→851); Claude's sweep row said "persisted
   pre-existing". Verify write-line-number < reduce-call-line-number at every
   swept site. Smell: an inline comment claiming "BEFORE any reduce" directly above
   a write that follows the judge call.

2. **Codex-side overreach (derived expected-set ≠ #779 hollow gate).** An upload
   verify whose expected set is derived by globbing existing local files is a
   REAL self-referential-acceptance gap (a never-written file can't fail it), but
   it is NOT the #779 `hollow-verification-gate` class (which requires the gated
   hot path to run UNCHECKED — here the live upload path IS exact-set-verified
   against the Hub). Before crediting a blocker, walk reachability: if every
   producer writes unconditionally BEFORE its reduce AND the upload call sits
   inside the same `try` as the producers (a producer crash skips upload), the
   missing-file-at-upload state is unreachable without a future regression →
   defense-in-depth hardening CONCERN (`required_rel_paths=`), not a FAIL.

3. **Severity precedent binds across rounds.** If the identical bug class was
   adjudicated at severity CONCERN in round N−1 (both reviewers CONCERNS→PASS)
   and the fix round closed the named sites, a newly-swept THIRD instance of the
   same class is a new CONCERN, not a retroactive round-FAIL — especially when
   the loss window is a bounded re-runnable arm (crash precedes any derived
   artifact consuming the lost text).

**How to apply:** any persist/ordering/upload-coverage disagreement — read the
producer write line vs the reduce line yourself; check whether upload is inside
the producers' try-block; check the concerns ledger for the class's prior
severity before letting either side re-class it.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Persist-before-reduce: ordering not existence; derived expected-set ≠ hollow gate](feedback_persist_before_reduce_ordering_vs_existence.md) — Claude counts a site persisted because the file exists (write AFTER remote judge); Codex FAILs a glob-derived upload verify though producers make the state unreachable; prior-round CONCERN severity binds. #906 r9.
