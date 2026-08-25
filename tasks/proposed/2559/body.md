---
title: 'workflow_lint files-mode: check_sha_pin_domain grandfather-stale artifact
  when grandfathered file is in scope'
kind: infra
tags: []
created_at: '2026-08-24T21:48:12Z'
has_clean_result: false
origin_prompt: found during /issue 2336 batch-2 files-mode verification (respawned
  session, 2026-08-24)
workflow: v1
---
# workflow_lint files-mode: corpus-global `check_sha_pin_domain` emits artifactual grandfather-stale FAILs when the named path is in scope

## Goal
Make `scripts/workflow_lint.py` files-mode stop emitting artifactual `sha-pin-domain/grandfather-stale` FAIL lines for `SHA_PIN_DOMAIN_GRANDFATHER` entries whose consuming pin sites are invisible only because files-mode restricts the scanned file set. The whole-repo no-flags run stays the binding instrument, unchanged.

## Evidence (2026-08-24, found during #2336 batch-2 verification)
- `uv run python scripts/workflow_lint.py --files <34 batch-2 files>` FAILs (rc=1) with exactly 3 lines: grandfather-stale for `('7c08c15bea17', 'scripts/issue1481_marker.py')`, `('88d344675fbb', 'scripts/issue1482_early_layer.py')`, `('ad687becec26', 'scripts/issue658_common.py')`.
- The SAME files-mode invocation FAILs identically at the repo root on main (batch-2 diff absent) — so not introduced by any branch diff.
- Whole-repo `--check-sha-pin-domain` PASSes in BOTH trees (repo root at `0d9080163d`, issue-2336 worktree at `f63dc3da21`).
- All three full 64-hex pins are present and unchanged in the named files; the batch-2 diff touches neither `SHA_PIN_DOMAIN_GRANDFATHER` nor any pin line.

## Mechanism
`check_sha_pin_domain`'s grandfather-stale arm requires each `SHA_PIN_DOMAIN_GRANDFATHER` entry to be CONSUMED by an observed undeclared cross-module site. Files-mode's restricted enumeration (payload + import closure) hides sibling pin sites (e.g. `scripts/issue1947_datagen.py` is not in the batch-2 payload+closure), the "hex appears in >= 2 distinct modules" keep-condition fails for the hidden hexes, nothing consumes the entries, and the stale arm fires. Files-mode already suppresses corpus-global findings "naming no in-scope path" (21 suppressed in the same run), but a grandfather-stale finding NAMES the grandfathered file, which can itself sit in the payload/closure — so the artifact leaks through as a blocking FAIL.

## Impact
Any files-mode run whose scope includes a grandfathered pin file FAILs spuriously: inline payload lint gates, per-batch scoped verifies (#2336 batch legs), and any future scoped run over `scripts/issue1481_marker.py` / `scripts/issue1482_early_layer.py` / `scripts/issue658_common.py` / siblings.

## Fix direction (implementer's choice; keep no-flags behavior byte-identical)
1. Run `check_sha_pin_domain` at FULL corpus scope even under files-mode (it is corpus-global by nature), keeping only findings that name in-scope paths; or
2. Suppress the grandfather-stale ARM under files-mode (staleness is only decidable at full enumeration; undeclared/conflict arms can stay scoped); or
3. Classify grandfather-stale findings as enumeration-dependent so the existing files-mode suppression drops them regardless of path scope.

Add a files-mode regression test: a grandfathered pin file placed IN scope with its sibling pin site OUT of scope must not produce a grandfather-stale FAIL; the whole-repo run over the same fixture tree must still report genuine staleness.
