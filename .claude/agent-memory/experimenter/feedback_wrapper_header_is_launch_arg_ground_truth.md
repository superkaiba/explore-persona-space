---
name: wrapper-header-is-launch-arg-ground-truth
description: When a plan's §10 launch command and the dispatch wrapper's own usage header disagree, launch with the wrapper header's args — it is what the implementer tested.
type: feedback
---

Before launching a per-issue dispatch wrapper, diff the launch command you were
handed against the wrapper's own usage header (`scripts/issue<N>_*_dispatch.sh`
top-of-file comment). When they disagree, the WRAPPER header wins — it is the
ground truth the implementer smoke-tested; the plan's §10 line can drift.

**Why:** #1090 fu6 (2026-07-17): plan §10 said `--manifest fu6_manifest.json`
(bare filename) while the wrapper header prescribed the repo-relative committed
path `eval_results/issue_1090/.../fu6_manifest.json`. The bare-name launch
crashed the GCP run at manifest open (FileNotFoundError, rc=1) and the #659
failover re-dispatched the same broken arg onto RunPod before a hot-fix
relaunch corrected it.

**How to apply:** at pre-launch step, `head -25` the wrapper and compare its
documented invocation with the brief's `cmd`; on mismatch, launch with the
wrapper's form and note the deviation in the `epm:run-launched` note. File-path
args especially: prefer committed repo-relative paths over bare filenames.
