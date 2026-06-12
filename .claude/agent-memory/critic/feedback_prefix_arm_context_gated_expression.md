---
name: Prefix-arm context-gated expression + inherited-violation anchor parity
description: Multi-turn-prefix training arms read on single-turn evals conflate weak install with context-gated expression (Concern iff checkpoints persist); inherited neg-panel violations kept-for-anchor-parity with flagged-cell exclusion is the RIGHT disposition (#612)
type: feedback
---

Two dispositions from #612 (sycophancy rig v2, methodology lens, APPROVE):

1. **Context-augmented training arm, context-free eval surface.** An arm
   trained behind a K-turn conversational prefix (loss on final turn only)
   but evaluated single-turn cannot separate "prefix weakens installation"
   from "behavior installed but gated to multi-turn contexts the eval never
   probes" — and the plan's own directional prior (prefix anchors behavior
   to conversational context) predicts exactly the confounded outcome.
   **Why:** the manipulation check shares the single-turn surface, so a
   sub-threshold install reads as failure when it may be non-expression
   (same shape as the persona-gated manipulation-check memory).
   **How to apply:** Concern, NOT REVISE, when all adapters/epoch
   checkpoints are uploaded — a small multi-turn self-probe (prefix
   prepended at eval, self-persona only) is a runnable post-hoc
   disambiguator. Ask the analyzer to label the install-failure branch
   "not installed OR not expressed single-turn." REVISE only if
   checkpoints are not persisted (disambiguation then unrunnable).

2. **Inherited contrastive-negative disjointness violation kept for anchor
   parity.** When an arm is a verbatim frozen-pool replication anchor and
   new arms must match it byte-for-byte (single-variable discipline),
   FIXING an inherited panel∩sources violation would smuggle a second
   variable into the arm contrast. The right disposition is: hard assert
   no NEW violations + flag the directed (adapter, negative-persona) cells
   `neg_member` + exclude them from curve fits in ALL arms (common
   exclusion set). Per-source LoRA means the push-up/push-down confound is
   localized to those directed cells, not global.
   **How to apply:** don't bounce the plan for carrying the violation;
   verify (a) the assert is against the realized pools, (b) the exclusion
   set is identical across arms, (c) near-twins of a trained negative
   (e.g. virtual_assistant ~ assistant) are NOT in the exclusion set —
   suppression-generalization to them is an attribution caveat (constant
   across arms, weighable), not a fit-contamination issue.
