---
title: 'Fix three fleet-wide test invariants broken by sibling landings on main (ungated
  HF upload #1739, bind-census skips #2477, dangling import roots #2379/#2474)'
kind: infra
tags: []
created_at: '2026-08-24T07:58:55Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2336 batch-0 verification: three pre-existing reds on
  origin/main red the whole fleet''s Step 9c gate.'
workflow: v1
---
## Goal

Restore the three fleet-wide test invariants that sibling landings on `main`
currently break, so every session's Step 9c gate stops carrying pre-existing
red. Each red is a violation of an EXISTING invariant by newly-landed code —
not a gap in the invariant — so the fix belongs in the offending file, not in
the test.

Surfaced by task #2336 (the shared atomic-write sweep) while running its
batch-0 verification. All three are byte-identical to `origin/main` and touch
none of #2336's files; #2336 is not the cause and is not blocked by them.

## Evidence

Confirmed red on `origin/main` (`93cee4593f`) by a direct pytest run
2026-08-24. Attributions independently verified: for each, the named commit
exists, touches the named file, and the file is present on `origin/main`.

### 1. `tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites`

```
AssertionError: NEW direct HF upload call site(s) without the secret gate:
    scripts/issue1739_r2v2_run.py
```

Owner: **#1739**, commit `067bd0300c` ("task #1739 claim4 r4 (crash-fix): row-index
pushdown kills the 52 GiB split-copy stack (128 GB-cgroup OOM) + leg-keyed
sentinel + in-fit RSS crumbs").

The test's own message states the two sanctioned fixes: route the upload
through `hub._upload` / `upload_dir_sharded` (both already gated), or call
`secret_scrub.assert_upload_clean(paths, what=...)` before the direct call.
The test docstring explicitly forbids adding the file to `GRANDFATHERED`, so
that is not an option here.

### 2. `tests/test_argcheck.py::test_bind_fleet_census_positive_coverage`

```
AssertionError: [('scripts/issue2477_base_coherence.py', 935, 'api.list_repo_tree',
  "receiver 'api' bindings not uniformly HfApi()"),
 ('scripts/issue2477_base_coherence.py', 789, 'api.list_repo_tree',
  "receiver 'api' bindings not uniformly HfApi()")]
```

Owner: **#2477**, commit `37613acbd9` ("issue-2477: base-coherence
decoding-sensitivity follow-up round (#2071)").

Two `api.list_repo_tree` call sites whose `api` receiver is not uniformly
bound to `HfApi()`, so the bind census can resolve neither and records them as
SKIPPED. The gate asserts `census.skipped == []` positively — a skip is a
coverage hole, not a pass. Fix is at the call sites: bind `api` uniformly to
`HfApi()` (or rename the non-`HfApi` binding so the receiver is
unambiguous).

### 3. `tests/test_workflow_lint_prod_import_lockfile.py::test_live_tree_clean`

```
assert len(danglers) == 7   # the 7 class-B dangling issue-stem roots
```
now sees **11**. The four NEW dangling first-party import roots:

| root | sites | owner |
|---|---|---|
| `issue2379_analysis` | 1 | **#2379** |
| `issue2379_capture` | 1 | **#2379** |
| `issue2379_mapfit` | 6 | **#2379** |
| `issue2474_fit` | 9 | **#2474** (commit `d76e89b387`) |

The seven pinned/pre-existing roots are unchanged: `_issue506_common`,
`issue500_predictors`, `issue541_personas`, `issue541_predictors`,
`issue541_upload_lib`, `issue621_analyze`, `issue_521_prep_turner_corpus`.

**Attribution correction worth carrying:** this red was first reported as
#2474's alone. It is not — #2379 contributes three of the four new roots and
#2474 one. Fixing only #2474 leaves the assertion red at 10.

A dangling first-party root means the imported module is absent from `main`
(an unmerged branch, or deleted) — latent breakage, not a third-party
verdict. Two dispositions per root: land the missing module, or make the
import site tolerant the way #2253 did for its own offenders (waive or
try-guard in the same change).

## Acceptance

1. `uv run pytest tests/test_no_ungated_upload_call_sites.py` exits 0 with
   `scripts/issue1739_r2v2_run.py` NOT added to `GRANDFATHERED`.
2. `uv run pytest tests/test_argcheck.py::test_bind_fleet_census_positive_coverage`
   exits 0 with `census.skipped == []` — the two #2477 sites resolved, not
   waived.
3. `uv run pytest tests/test_workflow_lint_prod_import_lockfile.py::test_live_tree_clean`
   exits 0. If a dangler-count pin change is the right answer for any root
   rather than a code fix, the new count lands WITH A NAMED DELTA in the same
   commit — the test's own docstring convention for a legitimate floor break.
4. Each fix is attributed to its owning issue in the commit message.

## Notes on routing

Three distinct defects across four owning issues (#1739, #2477, #2379,
#2474). Splitting into per-owner children is reasonable if that is cleaner
than one sweep — the shared property is only that each reds the whole fleet's
gate, which is why they are filed together rather than left as three separate
discoveries. Where an owning issue is still live, the arbitration rule applies
before editing its files: probe for a live writer, post a `file-set claim:`,
and sequence after any unreleased claim rather than racing it.
