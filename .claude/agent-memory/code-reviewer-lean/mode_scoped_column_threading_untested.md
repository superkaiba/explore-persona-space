---
name: mode-scoped-column-threading-untested
description: When a trainer threads an extra batch column needed by ONE mode, check the e2e test runs THAT mode — other modes pass with the threading deleted
metadata:
  type: feedback
---

When a diff threads an extra per-example column through a trainer/collator
chain (signature-columns override + collator passthrough + `inputs.pop`)
that only ONE mode consumes (e.g. `prefix_len` for `mode="prefix"`), verify
the committed end-to-end test exercises the CONSUMING mode through the real
`train()`/dataloader path — or directly pins the strip point
(`_remove_unused_columns` / `_signature_columns`).

**Why:** #2225 R1 g1 — the `SteeredSFTTrainer._set_signature_columns_if_needed`
override was the only untested link: the mask-partition test read
`trainer.train_dataset` directly (bypassing column stripping) and the only
`train()` test used `mode="all"`, which never needs the column. Deleting the
override left all 9 tests green while production prefix cells would crash at
step 0. Fail-loud but costs a pod launch cycle, and silently exposed to
library upgrades.

**How to apply:** in any diff pairing a Trainer subclass with a custom
collator passthrough, map each threaded column → the mode(s) that consume
it → the test that runs that mode through the real dataloader (not a
hand-collated batch). Missing link = CONCERNS with a one-line pin suggestion
(`assert "<col>" in trainer._signature_columns` or a 1-step train in the
consuming mode). Related: [[checkpoint_consumer_skips_key_check]].
