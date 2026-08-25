---
title: verify_uploads.py hardcodes HF_DATA_REPO — the upload gate cannot verify overflow-routed
  runs
kind: infra
tags: []
created_at: '2026-08-25T14:20:29Z'
has_clean_result: false
parent_id: 2389
origin_prompt: 'found while verifying #2389 uploads: verify_uploads.py returned FAIL
  flagging 1311/1312 files as residue because it queried the default data repo while
  the run wrote to the overflow repo'
workflow: v1
---
## Goal

Make `scripts/verify_uploads.py` able to verify runs that wrote to a HF repo other than the default data repo. Today it cannot, so the upload-verification gate is structurally blind to every overflow-routed run.

## The gap

`scripts/verify_uploads.py` line 123 hardcodes the data repo:

```python
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
```

There is NO CLI override and no env read for it. But the upload policy explicitly supports overflow routing: tensors reroute to the private overflow repo (`EPM_HF_OVERFLOW_ROUTING=1`, `superkaiba1/explore-persona-space-overflow`) on quota pressure, and a per-run write repo can be pinned (in #2389's case `EPM_2389_DATA_WRITE_REPO`). When a run does that, the verifier queries the wrong repo and every uploaded file reads as residue.

## Evidence (#2389, 2026-08-25)

Invoked with all 17 prefixes the run wrote plus the pod-side out-root listing, the verifier returned:

```
Verdict: FAIL
| Outroot Residue | FAIL | 1311 file(s) match no permanent home ... |
```

1,311 of 1,312 files flagged. An independent listing of the actual write repo found the artifacts present and complete: 1,377 files under `issue2389_q38ce/`, and a name-set diff accounted for 1,290 of 1,312 out-root files, with the remaining 22 being `vc_bank_part_*.pt` merge inputs the driver documents as deliberately never uploaded because they are byte-duplicated into the uploaded `vc_bank.pt`.

So the FAIL was entirely an artifact of the hardcoded repo. A ~100% residue rate is the signature.

## Why it matters

This is the hard gate before pod termination (`pod.py terminate`'s upload-verification guard). Its failure mode here is fail-CLOSED, which is the safe direction, but it forces every overflow-routed run to hand-roll verification, and a hand-rolled check is exactly what the gate exists to replace. The #1773 incident (8h50m of GPU output verified only by an ad-hoc hand set-diff) is the precedent for why ad-hoc verification is not acceptable.

## Proposed fix

Additive and backward-compatible:

1. Add `--hf-data-repo` (repeatable, or accepting a comma list) defaulting to the current constant, so omitting it changes nothing for existing callers.
2. Read the same env vars the write path uses, so a run that set `EPM_HF_OVERFLOW_ROUTING` / `EPM_<N>_DATA_WRITE_REPO` is verified against the repo it actually wrote to. Precedence: explicit flag > env > default.
3. When multiple repos are in play, treat the UNION as the permanent-home set, so a run split across main and overflow verifies correctly rather than half-failing.
4. Surface the resolved repo set in the report header, so a future reader can tell at a glance which repos were searched. The absence of that line is why the #2389 FAIL needed source reading to diagnose.

## Acceptance

- With no new flags/env, byte-identical behavior on an existing non-overflow run.
- Re-running #2389's exact invocation with the overflow repo supplied yields Outroot Residue PASS (or a residue list limited to the documented `vc_bank_part_*.pt` merge inputs).
- The report header names every repo searched.

## Provenance

Found while verifying uploads for #2389 before pod teardown; full dispositioning in that task's `epm:upload-verification` marker.
