---
title: Hoist a process-safe atomic-write helper into shared code and sweep the shared-tmp
  pattern out of scripts/
kind: infra
tags: []
created_at: '2026-08-17T06:18:14Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'Surfaced by #2329''s run-phase crash-fix round: the 8-worker grid
  phase crashed FileNotFoundError on a process-shared atomic-write temp name; the
  implementer flagged the same pattern in ~50 sibling scripts.'
workflow: v1
---
# Hoist a process-safe atomic-write helper into shared code and sweep the shared-tmp pattern out of `scripts/`

## Goal

Replace the process-UNSAFE `tmp = path.parent / (path.name + ".tmp")` atomic-write
pattern, currently duplicated across ~50 `scripts/issue*_*.py` files, with a single
shared process-safe helper, and migrate the call sites.

## Why (proven, not hypothetical)

Task #2329 lost a pre-launch smoke round to exactly this bug. Its 8-worker `grid`
phase crashed on real weights:

```
File "scripts/issue2329_run.py", line 1258, in _write_json_atomic
    os.replace(tmp, path)
FileNotFoundError: [Errno 2] No such file or directory:
  '.../manifests/pe_exclusions.json.tmp' -> '.../manifests/pe_exclusions.json'
```

Mechanism: when N processes write the SAME destination path, they all derive the
SAME temp name. The first worker's `os.replace(tmp, path)` consumes the shared
temp; every other worker then calls `os.replace` on a temp that no longer exists
and dies. A pre-fix probe measured this as deterministic, not a rare race —
**5/5 rounds failed, 7/8 workers each round**.

Note the content was NOT the problem in #2329: every worker computed byte-identical
content, so the "concurrent same-content writes are safe" design was sound. Only
the temp-path derivation was wrong. A sweep should preserve that design rather than
re-architecting call sites toward locking or single-writer election.

## Why this is a latent trap rather than a dormant curiosity

Every one of the ~50 sibling scripts is single-process TODAY, so nothing is
currently broken. But this project's standing rule is to vectorize / parallelize /
shard by default, and per-GPU process fan-out is the normal shape for grid and
sweep phases. Any script that gains a multi-worker phase writing one shared
manifest inherits this crash — on a PAID run, since the failure needs real
multi-process execution and no static check can reach it.

## The fix pattern (already proven in #2329, commit `27206c15d9`)

A contextmanager that:

- derives the temp as `f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"` —
  pid alone is insufficient (pid reuse), the uuid component is load-bearing;
- keeps the temp in the DESTINATION'S OWN directory, never `/tmp` — a
  cross-filesystem `os.replace` is not atomic;
- `os.replace` on success; `unlink(missing_ok=True)` then RE-RAISE the original
  exception on failure, so no orphan `*.tmp` residue is stranded (out-root residue
  is separately gated by the upload-verifier) and fail-fast discipline is kept.

## Scope

1. Land the helper in shared code (e.g. under
   `src/explore_persona_space/`) so scripts import rather than re-implement it.
2. Sweep `scripts/` for the `name + ".tmp"` derivation and migrate call sites,
   preserving existing signatures and idempotent semantics.
3. Add a lint check so the unsafe derivation cannot reappear.
4. Pin with a real N>=8 PROCESS concurrency test on one shared destination (it must
   fail against the unsafe derivation) plus a no-orphan-residue assertion.

## Notes

- Filed from #2329's run-phase crash-fix round; the implementer's
  `epm:new-bug-class` tag for it is `shared_tmp_name_concurrent_atomic_write`.
- Left at `proposed` deliberately — capture is not dispatch. Nothing is broken
  today, so this wants triage rather than an immediate autonomous session.
- `scripts/issue2329_analysis.py` was named as one of the affected siblings.
