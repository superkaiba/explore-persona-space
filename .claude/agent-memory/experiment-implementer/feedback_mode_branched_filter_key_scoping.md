---
name: Mode-branched filter key scoping + per-mode pre-GPU smoke
description: Scope per-mode row-key access to the branch that uses it; smoke every dispatcher-composed mode's pre-GPU main() portion (a dry-run/signature smoke does not execute the row projection)
type: feedback
---

A mode-branched feasibility/filter helper must scope per-mode key access to the branch that uses it — computing a paired-only field (`row["answer"]`) unconditionally crashes the mode whose row schema legitimately lacks it. And a GPU-bound gen phase's smoke carve-out must execute the pre-GPU `main()` portion for EVERY mode/arm the dispatcher composes (dry-run command composition + signature smoke do NOT execute the mode's row-projection code path); a cheap `--verify-pool`-style CPU preflight per mode closes it.

**Why:** #1345 `conversation-paired-stories` smoke leg crashed pod-side (`KeyError: 'answer'` at `issue1345_gen_stories_paired.py:237`) after two green CPU smoke rounds — the `--op-companion` mode's `{conv_id, question}` rows first met the paired-mode filter on GCP, burning a GCE cycle + a RunPod failover.

**How to apply:** when a script branches on mode/arm (paired vs on-policy, per-regime), key every row-schema access to its mode at ALL consumers (filter, fingerprint, generate, judge payload), fail-fast with hard keys (no `.get` defaults), and smoke each dispatcher-composed mode through its pre-GPU pool/filter/fingerprint stage (the per-arm-class smoke rule's mode sibling).
