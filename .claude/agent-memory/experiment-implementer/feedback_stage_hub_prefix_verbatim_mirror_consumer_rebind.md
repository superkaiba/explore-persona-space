---
name: stage_hub_prefix verbatim mirror needs consumer rebind
description: hub.stage_hub_prefix lands files at dest/<full repo-relative path>; a fixed-layout consumer must rebind to dest/<prefix>, and the tiny-real smoke must stage through the SAME helper as production (#1481 reread crash)
type: feedback
---

`hub.stage_hub_prefix(repo, prefix, dest)` is a VERBATIM PREFIX MIRROR (#1402):
files land at `dest/<full repo-relative path>`, never directly under `dest`.
Any consumer that opens a fixed local layout (e.g. `dest/adapter_config.json`,
a LoRA load on `dest`) must rebind its root to `dest/<prefix>` after staging —
or map the layout explicitly.

**Why:** #1481's `phase_reread` staged reused checkpoints with
`stage_hub_prefix` then opened `ckpt_dir/adapter_config.json` directly →
`FileNotFoundError` AFTER all 14 training runs completed, killing a healthy
GCE instance at the phase boundary (crash-fix 4, commit 07f661055b). The smoke
was green because the smoke branch staged per-file via `stage_hub_file` to the
exact consumer path — a different helper/layout than production.

**How to apply:** whenever production stages via `stage_hub_prefix`, (a) rebind
the consumer root to `dest/<prefix>`, and (b) make the tiny-real smoke stage
through the SAME helper as production (the #928 staged-layout consumer-open
probe run against the production staging helper, not a per-file substitute).
