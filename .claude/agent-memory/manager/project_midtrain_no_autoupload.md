---
name: Midtrain Pipeline No Auto-Upload
description: External training-against-misalignment midtraining pipeline produces unuploaded artifacts; sync_models.py sweep has a leaf-name heuristic that can drop parent context and collide
type: project
---

External midtraining pipeline (`external/training-against-misalignment/midtrain/`) writes outputs to `/workspace/midtrain_25pct/<condition>/<stage>/` and **never calls `upload_model`**. Auto-upload only exists in the `explore_persona_space` SFT trainer via `orchestrate/hub.py` — the external pipeline bypasses it.

Discovered on 2026-04-16 pre-pause audit: pod5 had `midtrain_25pct/evil_correct/tulu_sft_25pct` and `.../tulu_dpo_full` (~30 GB) sitting unuploaded because:
1. The midtraining pipeline has no post-training hook.
2. pod5 was added after the last `sync_models.py --sweep`.

**Why:** The external pipeline is a git submodule / vendored code that we don't want to patch inline, but we also never wired a post-hook in our orchestration layer. Backfill depended entirely on manual sweeps.

**How to apply:**
- After any midtraining run, explicitly invoke `upload_model()` from `orchestrate/hub.py` or run `scripts/sync_models.py --sweep --pods podN` immediately.
- When adding a new pod, remember to re-run sweep across all pods or at minimum the new one.
- Better long-term fix: add a thin wrapper script (e.g., `scripts/run_midtrain.py`) that shells the external pipeline then calls upload_model on every `<stage>/` output dir.

**Known bug in the sweep backstop:** `scripts/sync_models.py::_derive_hub_name` strips everything above the leaf dir unless `models|outputs|runs` appears in the path. For midtrain outputs this drops the `<condition>/` parent, so two conditions with the same stage name (`tulu_sft_25pct` under both `evil_correct/` and `good_correct/`) collide to the same hub path and one silently clobbers the other. The pre-pause backup uses `pod{N}_backup/<full-path-slug>` precisely to sidestep this. Fix the heuristic before relying on sweep for midtrain artifacts.
