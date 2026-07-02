---
name: Codex re-applies a v3-only verifier check to a v2-grandfathered clean-result body
description: clean-result-critic reconcile — discard a Codex REVISE that demands a v3-only verify_task_body check (e.g. check 21 body-Params⊆doc-§2) on a v2 body; presentation-only SHA / off-spec-string fixes are inline patches, not re-fold rounds
type: feedback
---

On a `clean-result-critic` reconcile, a Codex twin REVISE whose blocker is a
v3-only `verify_task_body.py` check mechanically re-applied to a v2-grandfathered
body (`<!-- clean-result-v2 -->`) is OVERREACH — discard it. v2 bodies are NEVER
newly hard-FAILed by a v3 rule (SPEC.md § Grandfathered shape), and the verifier
gates those checks on the v3 sentinel. Verify the sentinel + which checks are
v3-only before believing a structural blocker.

**Why:** #613 r2 (single-space-falsifier re-fold). Codex demanded the new round's
Parameters rows (`marker_sep`, `marker_predict_from_offset`, `sep_mode`, cell
slugs) be moved into methodology-doc §2 or the body Parameters slimmed. That IS
`check 21` (`check_body_params_subset_of_doc`, "body Parameters ⊆ methodology doc
§2 complete table") — `verify_task_body.py` documents it as v3-only ("PASS
vacuously on v2/legacy"), gated on `CLEAN_RESULT_V3_SENTINEL`. The body was v2 and
verify_task_body PASSed with check 21 correctly skipped. Separately, the
methodology doc was correctly structured per methodology-writer EXTEND mode (§2
preserved verbatim; new arm appended under its own `## <arm> arm` →
`### Training methodology — recipe parity` section that says "the §2
hyperparameter table applies in full; the differing rows are:" + a delta table).
EXTEND mode preserving parent §2 verbatim is documented behavior, not a bug.

**How to apply:**
1. On any clean-result-critic reconcile, FIRST read the body's sentinel
   (`grep clean-result-v` body.md). If v2/legacy, any Codex blocker that is a
   v3-structure / v3-check complaint (Parameters⊆doc, five-flat-H2, Takeaways
   bullet shape, no-Human-TL;DR, `## Data` shape) → Out-of-scope, Weight
   `Discarded — overreach`. Confirm the cited check is v3-only in
   `verify_task_body.py` (the new-v3-checks block + sentinel gating) before
   discarding.
2. SHA-mismatch and off-spec exact-string findings ARE real but presentation-
   only → PASS-with-orchestrator-procedural-strip, NOT REVISE. Verify them
   yourself (e.g. `git show <body-cited-sha>:docs/methodology/issue_<N>.md |
   grep -c <new-round-keyword>` vs the brief's correct SHA; diff the body's
   closing-note string against SPEC.md's literal). If no figure/number/narrative
   change, the orchestrator find/replaces inline before advancing (mirror of the
   Step 5c-bis procedural strip). Never extend a short SHA by hand — `git
   rev-parse` the real 40-char hash. (The #613 body linked the methodology doc at
   a SHA whose doc version had 0 new-arm references; the correct worktree SHA had
   30 — a genuine reader-facing defect Claude's PASS missed, still inline-fixable.)
3. Codex "could not run `huggingface_hub.list_repo_files` (sandbox DNS)" naming
   no concrete missing path is cite-or-drop / data-access-blocked → non-binding
   (same class the round-1 code-review reconciler discards). The body's Artifacts
   section citing explicit `@<rev>` HF paths is the cite that grounds it.
4. Net rule: a v2 clean-result reconcile PASSes when Codex's items are
   (v3-check overreach) ∪ (presentation-only SHA/string, inline-patchable) ∪
   (sandbox-blocked). Uphold the PASS; enumerate the inline patches in the
   verdict's "Orchestrator actions" section.
