---
title: Emit a uniform phase field in reproducibility cards so gate checks can resolve
  phase-to-card mechanically
kind: infra
tags: []
created_at: '2026-08-08T03:06:00Z'
has_clean_result: false
origin_prompt: 'Surfaced by the methodology critic during #2191 plan review as an
  out-of-scope forward note; recorded in #2191 plan v2 §11.'
workflow: v1
---
# Emit a uniform `phase` field in reproducibility cards

## Goal

Make the phase a reproducibility card belongs to machine-readable, by having every
card writer emit a consistent `phase` field, so gate checks can resolve
phase → card without guessing from filenames.

## Why

Cards today carry no consistent phase identifier. Measured on #2162 (all at
`origin/issue-2162`):

- `eval_results/issue_2162/stage2/stage2_results.json` → has `phase: "stage2-upload"`
- `eval_results/issue_2162/margin/upload_done.json` → **no** `phase` key
- `eval_results/issue_2162/gates/pilot_gate_report.json` → **no** `phase` key, and
  no token of "grid" / "anchors" / "stage-1" anywhere in its path or content, even
  though it IS the grid/anchors phase's card

The commit key itself also sits at three different nesting paths within that one
issue (`reproducibility_card.git_commit`, `note.reproducibility_card.git_commit`,
`repro.git_commit`), plus a top-level `final_commit_sha` twin.

The concrete consequence: #2191 added a `code-sha-cards` check to
`scripts/verify_report.py` that verifies a v2 report's Code-SHA rows against these
cards. Because no honest label → card resolution exists, its pairing leg (b3) has to
approximate with filename-token overlap against a hand-maintained stopword set, and
it provably cannot resolve the grid/anchors phase at all — so a report that cites
every correct card SHA but pairs them to the WRONG phases is only partially
detectable. A uniform `phase` field would close that at the source and let the
pairing leg become mechanical for every future issue.

## Scope sketch

1. Decide the field: name (`phase`), placement (top level vs inside the
   reproducibility card block), and a controlled vocabulary or free-string
   convention. Prefer one canonical location so a recursive search is not needed.
2. Emit it from every card writer (the `upload_done.json` / gate-report /
   sentinel writers). Additive — do not change existing keys.
3. Document the card shape in the relevant rule so future writers comply.
4. Optionally: teach `verify_report.py`'s `code-sha-cards` pairing leg to prefer
   `phase` when present and fall back to token overlap when absent, so old cards
   keep working. Retire the stopword set once coverage is good.

Backward compatibility is required: existing cards have no `phase`, and #2191's
check must keep passing on them (its degrade ladder already treats a missing
`phase` as unresolvable-and-skipped).

## Provenance

Surfaced as an out-of-scope forward note by the methodology critic during #2191's
plan review (2026-08-07), and recorded in #2191 plan v2 §11. #2191 deliberately
changes no card writer. Filed as `proposed` rather than auto-dispatched because the
fix lands in card-writing / experiment-infra code rather than the workflow surface,
so it wants triage before it consumes a session.
