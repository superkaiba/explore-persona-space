---
title: 'verify_report.py: assert per-stage cap-hit disclosure + declared-cap parity
  (catches stale cap_hit_report residue)'
kind: infra
tags: []
created_at: '2026-08-21T12:13:49Z'
has_clean_result: false
workflow: v1
---
# `verify_report.py`: assert every `cap_hit_report_*.json` stage's realized cap-hit is disclosed in the report, and that the declared `max_new_tokens` matches the shards' realized truncation cap

## Goal

Mechanize the CLAUDE.md `max_new_tokens` rule's per-stage reporting duty ("Every generation
stage REPORTS its realized cap-hit fraction") as a `scripts/verify_report.py` check, and
catch stale/contradictory `cap_hit_report_*.json` residue in the same pass.

## The two failure shapes (both realized in #2329 `q35_ladder_decay`)

A round produced TWO generation stages, each with its own cap-hit report:

- `cap_hit/cap_hit_report_grid.json` — grid stage, 29/2160 = 1.343% at cap 4096. Disclosed.
- `cap_hit/cap_hit_report_anchors.json` — anchors stage, realized **5/420 = 1.19%** at
  4096/4097 (per the staged `anchors_gate_w0.jsonl`). **NOT disclosed** in the report's G5
  bullet; it survived only as a line in a figure caption.

**Shape 1 — silent per-stage omission.** The report disclosed the stage the author was
thinking about and omitted the sibling stage. Nothing mechanical objected. A reader of the
"G5 disclosure" bullet would reasonably believe it covered the round.

**Shape 2 — stale residue that CONTRADICTS the report.** The committed
`cap_hit_report_anchors.json` declares `max_new_tokens: 2048`, `realized_row_caps: [2048]`,
`partial: true` — residue from an earlier `--cap-scope both` run — while the round actually
ran the anchors stage at 4096. So the committed artifact contradicts the (correct) report
text, and whichever a later reader trusts, one of them is wrong. Detected only by a human
critic opening both files.

## Proposed check

For each `cap_hit_report_*.json` under the round's `eval_results/<issue>/<round>/`:

1. Assert the report body discloses THAT stage's realized cap-hit fraction (stage name +
   number), not merely some stage's.
2. Assert the report's declared `max_new_tokens` for the stage equals the report JSON's
   `max_new_tokens` / `realized_row_caps`, and that both match the realized truncation cap
   observed in the stage's shard rows where those are reachable.
3. FAIL loud on a `partial: true` / mismatched-cap report that no report text explains —
   either the residue is regenerated or the report names it as known-stale.

Keep it evidence-based rather than name-based: key off the report JSONs actually present, so
a round with one stage is not forced to invent a second disclosure.

## Acceptance criteria

1. Reproduce shape 1 with the #2329 fixture: a report body disclosing only the grid stage
   while `cap_hit_report_anchors.json` is present ⇒ FAIL naming the anchors stage.
2. Reproduce shape 2: `cap_hit_report_anchors.json` at `max_new_tokens: 2048` /
   `partial: true` against a report declaring a uniform 4096 ⇒ FAIL naming the contradiction.
3. A round whose every present stage IS disclosed, with matching caps, PASSes — the check
   must not become unsatisfiable for compliant rounds.
4. A single-stage round is not required to disclose a stage that does not exist (no
   false-positive on absence).
5. Tests failing before / passing after; no new red in the no-flags `workflow_lint.py` run or
   the mapped-test selection.

## Provenance

Surfaced as a prose `mechanizable: yes` recommendation by the `methodology-critic` during
#2329 round `q35_ladder_decay` report review (round 1 FAIL, item 3), 2026-08-21. The
orchestrator verified both halves against the committed artifacts before filing. Evidence:
#2329 `events.jsonl` `epm:methodology-check` round 1; `cap_hit/cap_hit_report_anchors.json`
and `cap_hit/cap_hit_report_grid.json` on branch `issue-2329-q35-ladder-decay`; marker v185
(the `--cap-scope both` run that produced the residue).

- target_file: scripts/verify_report.py
- fingerprint: per-stage-cap-hit-disclosure-and-declared-cap-parity
- confidence: high — both shapes observed on committed artifacts in one round
