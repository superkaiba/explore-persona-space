---
title: 'Arm-independent g0_gate.json publishes to arm-UNSCOPED paths: git conflicts
  at the second arm''s last phase, HF overwrites silently (arm 1''s provenance destroyed)'
kind: infra
tags: []
created_at: '2026-08-27T01:03:11Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Surfaced driving #2546 arm 3 through /issue: p6_publish''s git leg
  FATALed on CONFLICT (add/add) in eval_results/issue_2546/gates/g0_gate.json after
  all of arm 3''s compute was spent. Structural diff: 44 leaf keys, 3 differ, all
  metadata, zero non-metadata differences. Scoped HF listings show preds arm-scoped
  by directory, reports arm-scoped by suffix, gates scoped by neither (n=1) — and
  reading the surviving copy proves arm 3''s provenance silently overwrote arm 1''s.'
workflow: v1
---
---
kind: infra
---

# Arm-independent artifact `g0_gate.json` publishes to ARM-UNSCOPED paths: git conflicts loudly at the last phase of the second arm, HF overwrites SILENTLY (arm 1's provenance record destroyed)

## The defect

A multi-arm run publishes per-arm artifacts to two channels. Within ONE publish step the arm
scoping is inconsistent three ways:

| Artifact class | Arm scoping | Result with 2 arms published |
|---|---|---|
| `analysis_tensors/preds/` | by DIRECTORY (`preds/arm1/`, `preds/arm3/`) | 102 files, both arms coexist |
| `eval_results_mirror/out/reports/` | by FILENAME SUFFIX (`capture_a1.json`, `capture_a3.json`) | 9 files, both arms coexist |
| `out/gates/g0_gate.json` | **NEITHER** | **n=1** — one arm's copy destroyed |

The G0/G-E gate is arm-INDEPENDENT by design: it fits one cell at one gate layer on #1336's
committed userbase-map data, not on arm data. So every arm produces a `g0_gate.json`, and every arm
writes it to the same unsuffixed path on both channels.

Consequences differ sharply by channel:

- **git — loud, but at the worst possible moment.** The second arm's publish hits
  `CONFLICT (add/add): Merge conflict in eval_results/issue_2546/gates/g0_gate.json` and goes FATAL.
  This is the LAST phase of the arm, after all of its compute is already spent (for #2546 arm 3:
  capture, rel-capture, the gate itself, and a 2.11 h 43-job fit phase). Nothing about the collision
  depends on any of that work — it was knowable at plan time, or at latest at the first publish.
- **HF — silent.** `upload` overwrites. Verified by scoped listing: the mirror's `gates/` prefix
  holds exactly ONE file, and reading it shows arm 3's provenance
  (`git_commit=60779db5…`, `git_dirty=None`, `timestamp=2026-08-26T21:34:53Z`) where arm 1's was
  (`git_commit=76ac8d57c0…`, `git_dirty=False`, `timestamp=2026-08-26T14:47:37Z`). **Arm 1's gate
  execution record on HF no longer exists.** No conflict, no warning, no log line.

So git PREVENTED the data loss by refusing; HF PERMITTED it. The channel that failed the run is the
one that behaved correctly.

## Severity — why this is worse than the observed impact

In THIS instance the loss is bounded: a structural diff of the two versions shows 44 leaf keys, of
which exactly 3 differ, all under `metadata.` (`git_commit`, `git_dirty`, `timestamp`), and
**zero** differing non-metadata keys. Every gate value — both legs' `r2`, `committed_r2`, `tol`,
`abs_dev`, `pass`, `enforced`, and all pins — is identical. So no result was corrupted.

That is LUCK, not safety: the values agree because this particular unsuffixed artifact happens to be
arm-independent by design. The mechanism has no such guarantee. Any arm-DEPENDENT artifact that
lands on an arm-unscoped path would be silently overwritten on HF, and a downstream reader would
get one arm's data while believing it read the other's — with no conflict, no marker, and no way to
detect it short of noticing the provenance block. The git channel would at least conflict; a
reader consuming the HF mirror would not.

## Recommended fix

**Primary: scope EVERY published artifact by arm, consistently.** Pick one convention (directory,
as `preds/` already does, or suffix, as `reports/` already does) and apply it to all of them. The
current three-way inconsistency inside a single publish step is the root cause; an artifact that
falls outside both conventions is invisible to review.

**Secondary, and the more valuable half: for a genuinely arm-independent artifact, make the second
arm's write a REPRODUCTION CHECK rather than a collision.** Publish once (first-writer-wins) and
have subsequent arms ASSERT content equality against the published copy, excluding a declared
provenance-metadata allowlist (`git_commit`, `git_dirty`, `timestamp`), failing LOUD on any
substantive difference.

Applied to this incident that assert would have compared the 41 non-metadata leaf keys, found them
identical, PASSED, and converted a FATAL at the end of a 2-hour phase into a positive cross-device
reproduction result — which is exactly what the data supports and what a reviewer would want to
know. The current design throws that information away and fails instead.

**Also worth adding: a publish-time (or plan-time) path-collision check.** Enumerate the expected
publish path set against what the sibling arms already published and refuse up front. The collision
was fully determined before any of arm 3's compute ran.

## Explicitly NOT duplicates

- **#2611** — `git_provenance()`'s 5 s-timeout `git status` orphans `.git/index.lock`. Related only
  by having broken the SAME phase minutes earlier, and by the `git_dirty=None` value visible in the
  diff above (that `None` is #2611's fingerprint, and this artifact is where it surfaced). Different
  bug, different fix.
- **#2610** — `poll_pipeline.py` cannot detect terminal success of a single-phase dispatcher
  invocation.
- **#2605** — dispatcher worker logs outside the poller's log-freshness globs.

## Target files

- `scripts/issue2546_dispatch.sh` (`publish_results_git`, the expected-path set construction)
- `scripts/issue2546_fit_cells.py` (`--publish`: the HF preds + `out/` mirror upload paths)
- whatever shared upload helper performs the `out/` mirror, for the first-writer-wins +
  content-equality-assert behavior

## Provenance

Surfaced driving #2546 arm 3 through `/issue`. Grounded by: the verbatim git conflict + FATAL lines
from `/workspace/logs/issue-2546.log`; a structural leaf-key diff of
`origin/issue-2546:eval_results/issue_2546/gates/g0_gate.json` vs the pod's local commit
`272233c0da` (44 keys / 3 differing / all `metadata.`); scoped `list_repo_tree` listings of
`issue2546_cotmap/eval_results_mirror/out/gates` (n=1), `.../out/reports` (n=9, both arms), and
`issue2546_cotmap/analysis_tensors/preds` (n=102, arm-scoped dirs); and an `hf_hub_download` +
read of the surviving gate JSON confirming arm 3's provenance replaced arm 1's.
