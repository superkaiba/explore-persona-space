---
title: 'verify_task_body: mechanical verbatim-sample-vs-manifest grep — FAIL fabricated
  ''(verbatim)'' sample rows'
kind: infra
tags: []
created_at: '2026-08-27T18:49:43Z'
has_clean_result: false
origin_prompt: 'interpretation-critic prose follow-up on #2617 r1: grep each ''(verbatim)''-labeled
  quoted sample stem into the linked pinned manifest and FAIL on a miss (bomb-vs-house
  fabrication; #657 family)'
workflow: v1
---
# Mechanical verbatim-sample-vs-manifest check for clean-result Sample blocks

## Goal

Add a mechanical check that catches fabricated "(verbatim)" sample rows in clean-result bodies: when a v4 `## Methodology` Sample block labels quoted rows "(verbatim)" AND the body links a pinned bank/manifest artifact (a committed JSON/JSONL the samples are drawn from), grep each quoted sample stem into the linked manifest and FAIL on a miss.

## Provenance

Surfaced by the interpretation-critic on task #2617 round 1 (2026-08-27, `epm:interp-critique v1`): the analyzer's Methodology sample block labeled "(verbatim)" contained a fabricated row — the bank's `obj_flip_00` item text was misquoted with a substituted noun that exists nowhere in the pinned bank manifest and instead echoes the originating prompt. Same failure family as #657 (fabricated persona sample). The other three quoted samples verified. Both incidents share the shape: a quoted "verbatim" sample that a one-line grep against the linked artifact would have failed.

## Sketch (target files — implementer to confirm placement)

- `scripts/verify_task_body.py` (preferred: a new check alongside the existing sample-block checks) or the interpretation-critic mechanical pre-pass.
- Trigger: a Sample-data slot containing "(verbatim)" + a resolvable committed artifact link in the same section.
- Behavior: extract quoted stems (quoted strings / table rows above a length floor), search the linked artifact file(s); any stem absent from every linked artifact ⇒ FAIL naming the stem and the artifact searched. Bounded: skip when the linked artifact is remote-only (HF URL without a local committed copy) — WARN instead.
- Tests: fixture with a genuine verbatim row (PASS) + a mutated row (FAIL) + remote-only link (WARN).

## Acceptance

- Check fires on the #2617 round-1 shape (mutated noun in a quoted stem) and on a reconstruction of the #657 shape.
- No false FAIL on ellipsized/subset-disclosed samples that carry an explicit truncation marker.
