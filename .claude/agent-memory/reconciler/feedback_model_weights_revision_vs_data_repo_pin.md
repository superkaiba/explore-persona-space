---
name: model-weights-revision-vs-data-repo-pin
description: Hidden-state donor reuse disputes — Claude conflates data-repo pins/shape asserts with MODEL-weights revision identity; Codex demands re-capture where a free Hub commit-history probe proves identity (#2329 ladder r1)
metadata:
  type: feedback
---

Two distinct calibration errors met in one dispute (#2329 `q35_ladder_decay` r1,
methodology-baselines lens, reused Qwen3.5 `vc_bank.pt` donor states):

- **Claude (PASS, wrong):** credited (a) `payload_for_arm_ladder`'s
  `donor_state.shape == recipient.shape` assert and (b) `stage_parent_bank`'s
  `revision=cfg.hf_revision` probe as covering donor validity. Both are
  DIFFERENT guarantees: the shape assert catches wrong-ARCHITECTURE only (any
  revision of the same model is shape-identical), and `hf_revision` is a
  **data-repo** commit (which artifact bytes are fetched), not the
  **model-repo** commit (which weights produced the stored hidden states).
  Hidden-state donors are weight-basis-dependent; a data-repo pin never
  establishes basis identity with the injection model. Cross-check: does
  `_repro`/the bundle record a `model_revision`, and does `from_pretrained`
  take `revision=`? (Here: no on both — body.md:49 documented the gap.)
- **Codex (REVISE, right mechanism, over-scoped remedy):** demanded GPU
  RE-CAPTURE of donor states because "the legacy bank cannot prove equality."
  That premise is checkable for FREE: `HfApi().list_repo_commits(<model_id>)`
  — if the last weight-bearing commit PREDATES the capture window, any `main`
  resolution in that window provably hit the same weights, and pin + record +
  a ~zero-cost L1 spot-forward equality assert suffices (0 GPU-h delta). In
  #2329 the repo had been static since 2026-03-02 vs an Aug capture — recapture
  was unnecessary.

**Why:** upholding Codex verbatim would have added a GPU capture phase and
re-triggered the plan-approval gate for nothing; PASSing Claude would have
left the round's own load unpinned (run-time-only capture loss — the same
unprovable-revision hole the parent body already documented).

**How to apply:** any reused ACTIVATION/tensor artifact consumed by a live
model → (1) separate the three guarantees explicitly: artifact-byte identity
(data-repo pin), architecture compatibility (shape), weight-basis identity
(model-repo revision); (2) before crediting a re-capture demand or a
"cannot prove" claim, run the model-repo commit-history probe yourself — it
often converts the dispute into a zero-GPU pin+record+assert remedy. Reused
TEXT artifacts (completions, manifests, judge outputs) are immune — frozen at
generation, no basis dependence. Related: [[claude-plan-cherrypick-closure-and-pin]].
