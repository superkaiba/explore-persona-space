---
title: 'verify_report.py: blob-identity check for body-linked captions.json pins (sibling
  of image-pin-blob-identity)'
kind: infra
tags: []
created_at: '2026-08-21T14:08:22Z'
has_clean_result: false
workflow: v1
---
# `verify_report.py`: blob-identity check for body-linked `captions.json` pins (sibling of `image-pin-blob-identity`)

## Goal

Catch the case where a report body cites a SHA-pinned `figures/**/captions.json`
whose content at that pin is superseded — the text-artifact analogue of the
existing `image-pin-blob-identity` check, which already covers pinned PNGs.

## The failure shape (realized in #2329 `q35_ladder_decay`)

A report round fixed three caption defects, then re-verified clean. But the
body's `decay_raw` block still cited `captions.json` at the PRE-FIX revision:

- pinned revision `fa9b14ee16` — the production-time commit
- the specific claim the citation supports ("the pinned captions.json states
  they are excluded") is **TRUE at that pin**, so no accuracy check fires
- yet the same pinned file still carried the three superseded texts the round
  existed to scrub: `"pooled per-segment arm means"` (the wrong estimator),
  `"n = 21 steered context-end cells"` (the wrong population), and
  `"Ten declared panels"` (the wrong count)

So a reader following the citation reads exactly the text the report corrected.
Nothing mechanical objected: `image-pin-blob-identity` covers PNGs only, and the
citation's own claim was accurate at its pin. It was caught by a human verifier
diffing the pin against the worktree tip.

Pinning the production-time declaration is *defensible provenance* — the fix is
not to ban stale pins outright, but to make the divergence VISIBLE so the author
chooses knowingly.

## Proposed check

For every SHA-pinned link in the report body (and the detailed companion)
resolving to a `figures/**/captions.json`:

1. Read the blob at the pinned SHA and compare against the figures-root copy.
2. On a mismatch, report which figure ids' `caption_bullets` differ — not just
   "differs", since a caption file legitimately accumulates entries.
3. Posture: **WARN by default**, naming the pin and the diverging ids, and
   escalate to FAIL only when the pinned revision contains text the CURRENT
   revision has removed (the superseded-text case above) — that is the shape
   that actively misleads a reader. A pin that merely lacks LATER-ADDED entries
   is benign provenance.

Keep it evidence-based and symmetric with the image check: key off the links
actually present, so a report citing no captions.json is unaffected.

## Acceptance criteria

1. Reproduce the #2329 shape: a body citing `captions.json` at a revision whose
   entries contain text absent from the figures-root copy ⇒ finding naming the
   pin and the diverging figure ids.
2. A body citing `captions.json` at a revision byte-identical to the
   figures-root copy ⇒ PASS, silent.
3. A pin that is merely OLDER but whose overlapping entries agree (only
   later-added ids missing) ⇒ WARN at most, never FAIL.
4. A body citing no `captions.json` ⇒ no finding (no false positive on absence).
5. An unresolvable pin (blob absent at that SHA) is reported as unresolvable,
   distinctly from a content mismatch — a broken citation is not a stale one
   (the existing `stale-evidence-pins` check already draws this distinction;
   match its vocabulary).
6. Tests failing before / passing after; no new red in the no-flags
   `workflow_lint.py` run or the mapped-test selection.

## Provenance

Surfaced as a prose `mechanizable: yes` recommendation by `report-verifier`
during #2329 round `q35_ladder_decay` report verification (round 2, PASS with one
minor, 2026-08-21). The orchestrator verified both halves before filing — the
claim holds at the old pin, and the three superseded texts are present there and
absent at the corrected revision — then re-pinned the two citations by hand
(`45d41b5dad`). Evidence: #2329 `events.jsonl` `epm:report-verified` round 2;
`git diff fa9b14ee16:figures/issue_2329/q35_ladder_decay/captions.json` vs the
branch tip on `issue-2329-q35-ladder-decay`.

- target_file: scripts/verify_report.py
- fingerprint: captions-json-pin-blob-identity-and-superseded-text
- confidence: high — observed on a committed body that had already PASSed every
  other mechanical check
