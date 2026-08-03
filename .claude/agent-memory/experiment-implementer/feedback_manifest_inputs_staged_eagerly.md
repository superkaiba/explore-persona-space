---
name: manifest-inputs-staged-eagerly
description: Frozen-inputs manifests must eagerly stage/verify every pinned input before pinning; lazy staging passes local smokes and crashes fresh lanes (incident #763 cofit Phase C, 2026-07-03)
type: feedback
---

A frozen-inputs manifest (`write_inputs_manifest`-style sha-pinning pass) must EAGERLY stage — or
verify git-tracked — EVERY input it pins, BEFORE the pinning pass runs. An input that a later
consumer stages lazily (e.g. `_load_v0` at battery time) passes every local smoke because the
worktree/repo checkout already carries the file, then crashes ONLY on fresh lanes (GCE clone,
fresh pod volume) at manifest step 0.

**Why:** #763 cofit Phase C died identically 4× (3 uninspectable GCE self-deletes + 1 SSH-able
RunPod crash) on `FileNotFoundError: inputs_manifest: frozen input missing: v0_shards/v0_deception.pt`;
two review rounds + a VM repro all missed it because every local checkout had the shards.

**How to apply:** when implementing any staging + manifest pattern, audit the manifest's full input
list against the eager-staging function (each entry staged-or-tracked); pin it with a static test
(AST-walk the `_add` calls); and smoke the FRESH-LANE condition by moving the local copy aside
before running the staging path — a smoke that never removes local files cannot catch this class.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Manifest inputs staged eagerly](feedback_manifest_inputs_staged_eagerly.md) — frozen-inputs manifests stage every pinned input BEFORE pinning; lazy staging passes local smokes, crashes fresh lanes (#763)
