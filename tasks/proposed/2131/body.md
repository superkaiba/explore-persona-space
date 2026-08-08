---
title: 'daily-held: HF storage 16TB vs 10TB - purge or upgrade'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-08-06T07:08:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 3): preflight FAILs on -5997GB
  headroom; Path B LFS uploads at 403 risk; deletion/upgrade is user-only'
workflow: v1
---
# daily-held: HF storage 16.0 TB vs 10 TB soft ceiling (−5,997 GB headroom) — deletion/upgrade decision needed

## Held item

The HF account is 6 TB over its soft ceiling and it is now biting operationally:

- #2091's Step 6a.6 preflight FAILed on 2026-08-06T01:55Z: "HF headroom insufficient …
  10 GB projected vs -5997 GB remaining (16.00/10.0 TB, live-api)" — the session proceeded
  only under an explicit override with a recorded disposition.
- #1491's Path B flagged the same night: "Path B uploads regenerated tensors over LFS,
  which 403s at the hard quota. The write-gate probe passes right now… but freeing quota
  means deleting HF artifacts, which is user-only."
- 2026-08-05 evening, Thomas personally drove a 577 GB VM-side reclaim gated on verified
  HF-backedness — the HF-side ledger itself was untouched.

Which carve-out holds it: **destructive/irreversible actions** (deleting HF artifacts) +
**spends money** (a quota upgrade) — both user-only decisions.

## Suggested action

Decide one (or both):
1. **Purge candidates** — the parked `adapters/issue_397` review (~242 GB, memory
   `project_pending_hf_quota_upgrade.md`, parked since the June quota saga) is the
   standing first candidate; a fuller purge list can be generated on request (largest
   prefixes by size, cross-checked against clean-result Repro pins so nothing
   promoted-body-referenced is deleted).
2. **Upgrade/raise the ceiling** — if the 16 TB footprint is the intended steady state.

Until decided, every large-upload plan (regenerated tensor stores, new ladder captures)
runs against a preflight FAIL + override friction, and a hard-quota 403 mid-upload is the
#552/#541 recovery class.

## Provenance

- origin: /daily 2026-08-05 problem sweep — miner 1 P22 + miner 6 P3 (preflight FAIL
  probed; headroom figures from the sessions' live-api reads).
