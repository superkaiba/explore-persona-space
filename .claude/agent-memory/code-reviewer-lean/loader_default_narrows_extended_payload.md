---
name: loader-default-narrows-extended-payload
description: An in-place --modules/--fields-extended producer consumed via a loader whose default parameter list is the OLD subset silently drops the new coverage; a bool-only resume key then resume-skips the stale units forever (#2569 r1 shard D)
metadata:
  type: feedback
---

Rule: when a round extends a shared producer in place (e.g.
`issue650_analyze build-base-svd --modules` growing from 2 to 7 modules), grep
every CONSUMER's load call for the loader's default parameter list. A consumer
calling `load_base_svd(path)` with the default `modules=("up_proj","down_proj")`
silently drops the extension: downstream loops guarded by
`if module not in payload: continue` / `if not basis: continue` skip the new
modules with rc=0 and no disclosure in the artifact.

**Why:** #2569 `cmd_lora` loaded the extended base-SVD payload with the old
default, so the o_proj write-arm intruder read never ran while the plan passed
all 7 modules to the builder for exactly that read. Compounding: the resume key
hashed `base_svd=bool(base_svd)` — after rebuilding the payload with the full
module list, every unit resume-skips (key unchanged) and the stale narrow
records persist.

**How to apply:** (1) diff the producer's new parameter surface vs each
consumer's call site defaults; (2) check per-item skip guards (`continue` on
missing key) against the plan's coverage list — silent narrowing needs a
disclosed skip record or a fail-loud; (3) check the resume/regime key carries
the extended input's IDENTITY (module list, revision, params), never a bare
`bool(...)` presence flag. Related: [[fingerprint-resume-ids-not-content]],
[[size-match-resume-skip-npz]].
